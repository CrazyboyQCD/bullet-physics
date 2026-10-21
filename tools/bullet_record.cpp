// bullet_record.cpp - C++ scene recorder for the frame-by-frame parity harness.
//
// Reads the scene tables generated from tools/scenes.json (bullet_scenes.h),
// simulates one scene one fixed step at a time, and writes one CSV per
// recorded body: <scene>_<body>.csv with one line per frame
//
//     t px py pz qx qy qz qw
//
// Files are written to the current working directory (tools/parity.py runs
// the recorder with cwd = target/). The Rust counterpart is
// crates/bullet-physics/examples/record_scenes.rs; both consume the same generated
// scene tables, so scene parameters have a single source of truth.
//
// Usage: bullet_record.exe [scene_name]

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "btBulletDynamicsCommon.h"
#include "BulletCollision/CollisionDispatch/btInternalEdgeUtility.h"
#include "BulletCollision/CollisionDispatch/btManifoldResult.h"
#include "BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h"
#include "BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h"
#include "BulletCollision/CollisionShapes/btUniformScalingShape.h"
#include "bullet_scenes.h"

extern ContactAddedCallback gContactAddedCallback;

// The contact-added callback the internal-edge utility rides on (Bullet's
// btManifoldResult hook; only bodies carrying CF_CUSTOM_MATERIAL_CALLBACK
// reach it). All real upstream callers hand the triangle side as colObj0:
// the manifold's body0 is the convex, so the triangle is colObj1 and the
// (partId, index) that index the triangle info map are partId1/index1.
bool internal_edge_contact_callback(btManifoldPoint& cp,
                                    const btCollisionObjectWrapper* col_obj0_wrap,
                                    int /*part_id0*/, int /*index0*/,
                                    const btCollisionObjectWrapper* col_obj1_wrap,
                                    int part_id1, int index1) {
  btAdjustInternalEdgeContacts(cp, col_obj1_wrap, col_obj0_wrap, part_id1,
                               index1);
  return true;
}

namespace {

// Broadphase pair filter mirroring the crate's set_pair_filter: the group &
// mask gate first, then the excluded pair is rejected. Bullet's own
// btOverlapFilterCallback REPLACES the default test entirely
// (btHashedOverlappingPairCache::needsBroadphaseCollision), so the gate is
// re-done here - matching the one-layer consumer API on the Rust side (the
// dispatcher's needsCollision still tests group & mask a second time).
class PairExcludeFilter : public btOverlapFilterCallback {
 public:
  PairExcludeFilter() : exclude_a(-1), exclude_b(-1) {}
  int exclude_a, exclude_b;  // body indices of the excluded pair; -1 = none
  bool needBroadphaseCollision(btBroadphaseProxy* p0, btBroadphaseProxy* p1) const override {
    bool collides = (p0->m_collisionFilterGroup & p1->m_collisionFilterMask) != 0;
    collides = collides && (p1->m_collisionFilterGroup & p0->m_collisionFilterMask) != 0;
    if (!collides) return false;
    if (exclude_a >= 0) {
      const int i0 = static_cast<const btCollisionObject*>(p0->m_clientObject)->getUserIndex();
      const int i1 = static_cast<const btCollisionObject*>(p1->m_clientObject)->getUserIndex();
      const int lo = i0 < i1 ? i0 : i1;
      const int hi = i0 < i1 ? i1 : i0;
      if (lo == exclude_a && hi == exclude_b) return false;
    }
    return true;
  }
};

btCollisionShape* make_shape(
    const BodySpec& b, std::vector<std::unique_ptr<btTriangleMesh>>& meshes,
    std::vector<std::unique_ptr<btTriangleIndexVertexArray>>& mesh_arrays,
    std::vector<std::unique_ptr<btBvhTriangleMeshShape>>& bvh_storage,
    std::vector<std::unique_ptr<btConvexShape>>& child_storage) {
  const btScalar* p = b.params;
  switch (b.shape) {
    case SHAPE_BOX:
      return new btBoxShape(btVector3(p[0], p[1], p[2]));
    case SHAPE_SPHERE:
      return new btSphereShape(p[0]);
    case SHAPE_CAPSULE:
      // btCapsuleShape(radius, height) with height = 2 * half_height, axis Y
      // (matches this crate's capsule(radius, half_height)).
      return new btCapsuleShape(p[0], btScalar(2.0) * p[1]);
    case SHAPE_CYLINDER:
      // btCylinderShape takes half extents; axis Y.
      return new btCylinderShape(btVector3(p[0], p[1], p[0]));
    case SHAPE_CONE:
      // btConeShape(radius, height) with height = 2 * half_height, apex +Y.
      return new btConeShape(p[0], btScalar(2.0) * p[1]);
    case SHAPE_PLANE:
      return new btStaticPlaneShape(btVector3(p[0], p[1], p[2]), p[3]);
    case SHAPE_MESH: {
      // Same vertex/index data as this crate's TriangleMesh (generated from
      // the shared scenes.json); btBvhTriangleMeshShape mirrors our BVH path.
      btTriangleMesh* tm = new btTriangleMesh();
      const btScalar* v = b.mesh->vertices;
      const int* idx = b.mesh->indices;
      for (int i = 0; i < b.mesh->num_triangles; ++i) {
        const btScalar* a = v + 3 * idx[3 * i + 0];
        const btScalar* c = v + 3 * idx[3 * i + 1];
        const btScalar* e = v + 3 * idx[3 * i + 2];
        tm->addTriangle(btVector3(a[0], a[1], a[2]),
                        btVector3(c[0], c[1], c[2]),
                        btVector3(e[0], e[1], e[2]));
      }
      meshes.push_back(std::unique_ptr<btTriangleMesh>(tm));
      return new btBvhTriangleMeshShape(tm, true);
    }
    case SHAPE_MESH_PARTS: {
      // btTriangleIndexVertexArray over the independent IndexedMesh parts -
      // the multi-part API the crate mirrors with TriangleMesh::from_parts
      // (the BVH spans every part; the vertex/index arrays live in the
      // generated header, so only the mesh interface needs owning here).
      btTriangleIndexVertexArray* iv = new btTriangleIndexVertexArray();
      for (int pi = 0; pi < b.num_mesh_parts; ++pi) {
        const MeshSpec& part = b.mesh_parts[pi];
        btIndexedMesh im;
        im.m_numVertices = part.num_vertices;
        im.m_vertexBase = reinterpret_cast<const unsigned char*>(part.vertices);
        im.m_vertexStride = sizeof(btScalar) * 3;
        im.m_numTriangles = part.num_triangles;
        im.m_triangleIndexBase = reinterpret_cast<const unsigned char*>(part.indices);
        im.m_triangleIndexStride = sizeof(int) * 3;
        iv->addIndexedMesh(im, PHY_INTEGER);
      }
      mesh_arrays.push_back(std::unique_ptr<btTriangleIndexVertexArray>(iv));
    }
    case SHAPE_UNIFORM_SCALING: {
      // btUniformScalingShape over a convex child (kind 0 = sphere with
      // p2 = radius, kind 1 = box with p2..p4 = half extents); the child
      // is owned here, the wrapper references it.
      btConvexShape* child;
      if (static_cast<int>(p[1]) == 0) {
        child = new btSphereShape(p[2]);
      } else {
        child = new btBoxShape(btVector3(p[2], p[3], p[4]));
      }
      child_storage.push_back(std::unique_ptr<btConvexShape>(child));
      return new btUniformScalingShape(child, p[0]);
    }
    case SHAPE_SCALED_MESH: {
      // Same child BVH as SHAPE_MESH, wrapped in
      // btScaledBvhTriangleMeshShape with the per-axis scaling (p0..p2);
      // p3..p5 stay zero. The wrapper does not own the child BVH.
      btTriangleMesh* tm = new btTriangleMesh();
      const btScalar* v = b.mesh->vertices;
      const int* idx = b.mesh->indices;
      for (int i = 0; i < b.mesh->num_triangles; ++i) {
        const btScalar* a = v + 3 * idx[3 * i + 0];
        const btScalar* c = v + 3 * idx[3 * i + 1];
        const btScalar* e = v + 3 * idx[3 * i + 2];
        tm->addTriangle(btVector3(a[0], a[1], a[2]),
                        btVector3(c[0], c[1], c[2]),
                        btVector3(e[0], e[1], e[2]));
      }
      meshes.push_back(std::unique_ptr<btTriangleMesh>(tm));
      btBvhTriangleMeshShape* child_bvh = new btBvhTriangleMeshShape(tm, true);
      bvh_storage.push_back(std::unique_ptr<btBvhTriangleMeshShape>(child_bvh));
      return new btScaledBvhTriangleMeshShape(child_bvh,
                                              btVector3(p[0], p[1], p[2]));
    }
    case SHAPE_HEIGHTFIELD: {
      // btHeightfieldTerrainShape over the shared grid: float data, the
      // min/max the generator computed (both engines center height on the
      // same midpoint), the flip flags, and the cell size via localScaling
      // (upstream's unit-cell grid scaled post-centering, height axis
      // unscaled - identical to this crate's materialized vertices). The
      // heights array is static in the generated header, so the shape can
      // reference it without ownership.
      const HeightfieldSpec& hf = *b.heightfield;
      btHeightfieldTerrainShape* terrain = new btHeightfieldTerrainShape(
          hf.width, hf.length, hf.heights, btScalar(1.0), hf.min_height,
          hf.max_height, hf.up_axis, PHY_FLOAT, hf.flip_quad_edges != 0);
      terrain->setUseDiamondSubdivision(hf.use_diamond_subdivision != 0);
      terrain->setUseZigzagSubdivision(hf.use_zigzag_subdivision != 0);
      terrain->setFlipTriangleWinding(hf.flip_triangle_winding != 0);
      terrain->setLocalScaling(btVector3(hf.cell_x, btScalar(1.0), hf.cell_z));
      return terrain;
    }
  }
  return nullptr;
}

const SceneSpec* find_scene(const char* name) {
  for (int i = 0; i < int(sizeof(kScenes) / sizeof(kScenes[0])); ++i) {
    if (std::strcmp(kScenes[i]->name, name) == 0) return kScenes[i];
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  const char* scene_name = argc > 1 ? argv[1] : "falling_box";
  const char* suffix = argc > 2 ? argv[2] : "bullet";
  const SceneSpec* spec = find_scene(scene_name);
  if (!spec) {
    std::fprintf(stderr, "bullet_record: unknown scene '%s'\n", scene_name);
    return 2;
  }

  // --- World (mirrors PhysicsWorld::new) ---
  btDefaultCollisionConfiguration config;
  btCollisionDispatcher dispatcher(&config);
  btDbvtBroadphase broadphase;
  btSequentialImpulseConstraintSolver solver;
  btDiscreteDynamicsWorld world(&dispatcher, &broadphase, &solver, &config);
  world.setGravity(btVector3(spec->gravity[0], spec->gravity[1], spec->gravity[2]));

  // Pair filter (optional): resolve the excluded pair's names to body
  // indices. Lives on the stack and outlives the simulation loop.
  PairExcludeFilter pair_filter;
  if (spec->pair_filter) {
    int ia = -1, ib = -1;
    for (int i = 0; i < spec->num_bodies; ++i) {
      if (std::strcmp(spec->bodies[i].name, spec->pair_filter[0]) == 0) ia = i;
      if (std::strcmp(spec->bodies[i].name, spec->pair_filter[1]) == 0) ib = i;
    }
    if (ia < 0 || ib < 0) {
      std::fprintf(stderr, "bullet_record: pair_filter body not found\n");
      return 2;
    }
    pair_filter.exclude_a = ia < ib ? ia : ib;
    pair_filter.exclude_b = ia < ib ? ib : ia;
    broadphase.getOverlappingPairCache()->setOverlapFilterCallback(&pair_filter);
  }

  // Reproducibility precondition (docs/parity-harness.md).
  btContactSolverInfo& info = world.getSolverInfo();
  info.m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
  info.m_numIterations = spec->iterations;
  // Scene solver-info overrides (btContactSolverInfo knobs; the generated
  // table fills every field, so both engines read the same 12 values).
  if (spec->solver_info) {
    const SolverInfoSpec* si = spec->solver_info;
    info.m_erp = si->erp;
    info.m_erp2 = si->erp2;
    info.m_linearSlop = si->linear_slop;
    info.m_restitutionVelocityThreshold = si->restitution_velocity_threshold;
    info.m_splitImpulse = si->split_impulse != 0;
    info.m_splitImpulsePenetrationThreshold = si->split_impulse_penetration_threshold;
    info.m_splitImpulseTurnErp = si->split_impulse_turn_erp;
    info.m_warmstartingFactor = si->warmstarting_factor;
    info.m_frictionERP = si->friction_erp;
    info.m_globalCfm = si->global_cfm;
    info.m_sor = si->sor;
    info.m_damping = si->damping;
  }
  // The internal-edge utility hook: inert unless a body carries
  // CF_CUSTOM_MATERIAL_CALLBACK (only the internal_edge scene's mesh does).
  gContactAddedCallback = internal_edge_contact_callback;

  const btScalar dt = btScalar(spec->dt_num) / btScalar(spec->dt_den);

  // --- Build bodies (owned here; world references them until it is destroyed
  // first, which the declaration order guarantees) ---
  std::vector<std::unique_ptr<btCollisionShape>> shapes;
  std::vector<std::unique_ptr<btTriangleMesh>> mesh_storage;
  std::vector<std::unique_ptr<btTriangleIndexVertexArray>> mesh_array_storage;
  std::vector<std::unique_ptr<btBvhTriangleMeshShape>> bvh_storage;
  std::vector<std::unique_ptr<btConvexShape>> child_shape_storage;
  std::vector<std::unique_ptr<btMotionState>> motions;
  std::vector<std::unique_ptr<btRigidBody>> bodies;
  for (int i = 0; i < spec->num_bodies; ++i) {
    const BodySpec& b = spec->bodies[i];
    shapes.push_back(std::unique_ptr<btCollisionShape>(
        make_shape(b, mesh_storage, mesh_array_storage,
                   bvh_storage, child_shape_storage)));
    btCollisionShape* shape = shapes.back().get();
    motions.push_back(std::unique_ptr<btMotionState>(
        new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1),
                                             btVector3(b.pos[0], b.pos[1], b.pos[2])))));
    btVector3 local_inertia(0, 0, 0);
    if (!b.is_static) shape->calculateLocalInertia(b.mass, local_inertia);
    btRigidBody::btRigidBodyConstructionInfo ci(
        b.is_static ? btScalar(0.0) : b.mass, motions.back().get(), shape, local_inertia);
    // Construction-info-only opt-in (Bullet exposes no post-construction
    // setter): the ragdoll-stack "additional damping" block of applyDamping.
    ci.m_additionalDamping = b.additional_damping != 0;
    bodies.push_back(std::unique_ptr<btRigidBody>(new btRigidBody(ci)));
    btRigidBody* body = bodies.back().get();
    body->setFriction(b.friction);
    body->setRestitution(b.restitution);
    body->setRollingFriction(b.rolling_friction);
    body->setSpinningFriction(b.spinning_friction);
    body->setDamping(b.linear_damping, b.angular_damping);
    body->setAnisotropicFriction(btVector3(b.anisotropic_friction[0],
                                           b.anisotropic_friction[1],
                                           b.anisotropic_friction[2]));
    if (b.contact_stiffness_set) {
      body->setContactStiffnessAndDamping(b.contact_stiffness, b.contact_damping);
    }
    if (b.no_contact_response) {
      body->setCollisionFlags(body->getCollisionFlags() |
                              btCollisionObject::CF_NO_CONTACT_RESPONSE);
    }
    if (b.internal_edges) {
      // Mesh-opt-in: build the triangle info map (btGenerateInternalEdgeInfo
      // also stores it on the shape) and arm the callback gate flag.
      if (b.shape == SHAPE_MESH) {
        btGenerateInternalEdgeInfo(
            static_cast<btBvhTriangleMeshShape*>(shape), new btTriangleInfoMap());
      }
      body->setCollisionFlags(body->getCollisionFlags() |
                              btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
    }
    // Per-body gravity: detach from world propagation first (the addRigidBody
    // sync and world setGravity respect the flag), then set the body's own
    // acceleration - every step still applies it (world applyGravity has no
    // flag check, it just uses each body's own m_gravity_acceleration).
    if (b.disable_world_gravity) {
      body->setFlags(body->getFlags() | BT_DISABLE_WORLD_GRAVITY);
      body->setGravity(
          btVector3(b.body_gravity[0], b.body_gravity[1], b.body_gravity[2]));
    }
    if (!b.is_static) {
      body->setLinearVelocity(btVector3(b.velocity[0], b.velocity[1], b.velocity[2]));
      body->setAngularVelocity(
          btVector3(b.angular_velocity[0], b.angular_velocity[1], b.angular_velocity[2]));
      if (b.ccd_on) {
        body->setCcdMotionThreshold(b.ccd_motion_threshold);
        body->setCcdSweptSphereRadius(b.ccd_swept_sphere_radius);
      }
    }
    body->setUserIndex(i);
    world.addRigidBody(body, b.group, b.mask);
  }

  // Map recorded body names to indices.
  std::vector<int> record_idx;
  for (int r = 0; r < spec->num_record; ++r) {
    int idx = -1;
    for (int i = 0; i < spec->num_bodies; ++i) {
      if (std::strcmp(spec->bodies[i].name, spec->record[r]) == 0) {
        idx = i;
        break;
      }
    }
    if (idx < 0) {
      std::fprintf(stderr, "bullet_record: record body '%s' not found\n", spec->record[r]);
      return 2;
    }
    record_idx.push_back(idx);
  }

  // --- Record one line per fixed step into <scene>_<body>_<suffix>.csv ---
  std::vector<FILE*> out;
  for (int r = 0; r < spec->num_record; ++r) {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "%s_%s_%s.csv", scene_name, spec->record[r], suffix);
    FILE* f = std::fopen(fname, "w");
    if (!f) {
      std::fprintf(stderr, "bullet_record: cannot open %s\n", fname);
      return 2;
    }
    out.push_back(f);
  }

  for (int frame = 1; frame <= spec->frames; ++frame) {
    // A scene-level substeps spec drives the fixed-timestep accumulator
    // (stepSimulation(dt, maxSubSteps, fixedTimeStep)); the default is
    // exactly one dt-sized substep.
    if (spec->substeps) {
      world.stepSimulation(dt, spec->substeps->max_sub_steps,
                           spec->substeps->fixed_dt);
    } else {
      world.stepSimulation(dt, 1, dt);
    }
    for (int r = 0; r < spec->num_record; ++r) {
      const btTransform& t = bodies[record_idx[r]]->getCenterOfMassTransform();
      const btVector3& p = t.getOrigin();
      const btQuaternion& q = t.getRotation();
      std::fprintf(out[r], "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   btScalar(frame) * dt, p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
    }
  }
  for (FILE* f : out) std::fclose(f);

  // --- Rays (probed once after the last frame) ---
  // One line per hit in <scene>_rays_<suffix>.csv: ray_idx body_idx fraction.
  // The all-hits callback collects in broadphase pair order, so the hits are
  // sorted by fraction to match the Rust recorder's nearest-first order.
  if (spec->num_rays > 0) {
    char rname[256];
    std::snprintf(rname, sizeof(rname), "%s_rays_%s.csv", scene_name, suffix);
    FILE* rf = std::fopen(rname, "w");
    if (!rf) {
      std::fprintf(stderr, "bullet_record: cannot open %s\n", rname);
      return 2;
    }
    for (int i = 0; i < spec->num_rays; ++i) {
      const RaySpec& ray = spec->rays[i];
      const btVector3 from(ray.from[0], ray.from[1], ray.from[2]);
      const btVector3 to(ray.to[0], ray.to[1], ray.to[2]);
      if (ray.all_hits) {
        btCollisionWorld::AllHitsRayResultCallback cb(from, to);
        cb.m_collisionFilterGroup = ray.group;
        cb.m_collisionFilterMask = ray.mask;
        world.rayTest(from, to, cb);
        std::vector<std::pair<btScalar, int>> hits;
        for (int h = 0; h < cb.m_collisionObjects.size(); ++h) {
          hits.push_back({cb.m_hitFractions[h], cb.m_collisionObjects[h]->getUserIndex()});
        }
        std::sort(hits.begin(), hits.end());
        for (const auto& hit : hits) {
          std::fprintf(rf, "%d %d %.6f\n", i, hit.second, hit.first);
        }
      } else {
        btCollisionWorld::ClosestRayResultCallback cb(from, to);
        cb.m_collisionFilterGroup = ray.group;
        cb.m_collisionFilterMask = ray.mask;
        world.rayTest(from, to, cb);
        if (cb.hasHit()) {
          std::fprintf(rf, "%d %d %.6f\n", i, cb.m_collisionObject->getUserIndex(),
                       cb.m_closestHitFraction);
        }
      }
    }
    std::fclose(rf);
  }
  return 0;
}
