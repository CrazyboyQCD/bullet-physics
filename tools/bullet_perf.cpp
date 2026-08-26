// bullet_perf.cpp - C++ performance runner for the Rust vs Bullet wall-clock
// comparison (docs/bullet-perf.md).
//
// Reads the perf scene tables generated from tools/perf_scenes.json
// (bullet_perf_scenes.h), builds one scene, runs `warmup` untimed frames then
// times `frames` fixed steps, and repeats the whole run (fresh world each
// time) printing one line per repeat. The output protocol matches the Rust
// runner (crates/bullet/examples/perf_scenes.rs) so tools/perf.py can parse
// both:
//
//     perf scene <name> bodies <n> frames <f> warmup <w>
//     repeat <i> total_ms <x> last300_ms <y>
//
// Fairness notes (documented in docs/bullet-perf.md):
//   - the broad phase is btAxisSweep3 (the same 3-axis sweep-and-prune
//     algorithm as this crate's AxisSweep3; btDbvtBroadphase would compare a
//     different algorithm), scoped with the scene's broadphase_aabb hint;
//   - SOLVER_RANDMIZE_ORDER is cleared (same reproducibility precondition as
//     the parity recorder);
//   - multibody scenes use btMultiBodyDynamicsWorld with link colliders in
//     the DefaultFilter group (required for the island solver) and a manual
//     forwardKinematics before attaching them.
//
// Usage: bullet_perf.exe <scene_name> [repeats]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "btBulletDynamicsCommon.h"
#include <BulletCollision/BroadphaseCollision/btAxisSweep3.h>

#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>

#include "bullet_perf_scenes.h"

namespace {

btCollisionShape* make_shape(const PerfBodySpec& b,
                             std::vector<std::unique_ptr<btTriangleMesh>>& meshes) {
  const btScalar* p = b.params;
  switch (b.shape) {
    case PERF_SHAPE_BOX:
      return new btBoxShape(btVector3(p[0], p[1], p[2]));
    case PERF_SHAPE_SPHERE:
      return new btSphereShape(p[0]);
    case PERF_SHAPE_CAPSULE:
      // btCapsuleShape(radius, height) with height = 2 * half_height, axis Y
      // (matches this crate's capsule(radius, half_height)).
      return new btCapsuleShape(p[0], btScalar(2.0) * p[1]);
    case PERF_SHAPE_MESH: {
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
  }
  return nullptr;
}

// Owns everything the world references; destroyed (and so unregistered)
// before the world member goes away.
struct SceneRun {
  btDefaultCollisionConfiguration config;
  btCollisionDispatcher dispatcher;
  btAxisSweep3 broadphase;
  btMultiBodyConstraintSolver solver;   // plain scenes: solves rigid rows too
  std::vector<std::unique_ptr<btCollisionShape>> shapes;
  std::vector<std::unique_ptr<btTriangleMesh>> mesh_storage;
  std::vector<std::unique_ptr<btMotionState>> motions;
  std::vector<std::unique_ptr<btRigidBody>> bodies;
  std::vector<std::unique_ptr<btMultiBody>> multibodies;
  // Declared last so it is destroyed first, while the bodies and multibodies
  // it references are still alive (unregistration happens in its destructor).
  btMultiBodyDynamicsWorld world;       // plain scenes: behaves like the discrete world

  SceneRun(const PerfSceneSpec& spec)
      : config(),
        dispatcher(&config),
        broadphase(btVector3(spec.bp_min[0], spec.bp_min[1], spec.bp_min[2]),
                   btVector3(spec.bp_max[0], spec.bp_max[1], spec.bp_max[2])),
        solver(),
        world(&dispatcher, &broadphase, &solver, &config) {
    world.setGravity(btVector3(spec.gravity[0], spec.gravity[1], spec.gravity[2]));
    // Reproducibility precondition (docs/parity-harness.md).
    btContactSolverInfo& info = world.getSolverInfo();
    info.m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
    info.m_numIterations = spec.iterations;

    for (int i = 0; i < spec.num_bodies; ++i) {
      const PerfBodySpec& b = spec.bodies[i];
      shapes.push_back(std::unique_ptr<btCollisionShape>(make_shape(b, mesh_storage)));
      btCollisionShape* shape = shapes.back().get();
      motions.push_back(std::unique_ptr<btMotionState>(
          new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1),
                                               btVector3(b.pos[0], b.pos[1], b.pos[2])))));
      btVector3 local_inertia(0, 0, 0);
      const bool is_static = b.mass == btScalar(0);
      if (!is_static) shape->calculateLocalInertia(b.mass, local_inertia);
      btRigidBody::btRigidBodyConstructionInfo ci(
          is_static ? btScalar(0.0) : b.mass, motions.back().get(), shape, local_inertia);
      bodies.push_back(std::unique_ptr<btRigidBody>(new btRigidBody(ci)));
      btRigidBody* body = bodies.back().get();
      body->setFriction(b.friction);
      body->setRestitution(b.restitution);
      world.addRigidBody(body);
    }

    if (spec.chains.count > 0) add_chains(spec.chains);
  }

  // A grid of fixed-base revolute pendulums, matching the Rust runner's
  // add_chains (base positions, joint offsets and initial angles must be
  // identical on both sides).
  void add_chains(const PerfChainSpec& c) {
    const int rows = (c.count + c.cols - 1) / c.cols;
    const btVector3 length(c.link_length, 0, 0);
    for (int k = 0; k < c.count; ++k) {
      const int col = k % c.cols;
      const int row = k / c.cols;
      const btScalar x = (btScalar(col) - (btScalar(c.cols) - 1) / 2) * c.spacing;
      const btScalar z = (btScalar(row) - (btScalar(rows) - 1) / 2) * c.spacing;
      btMultiBody* body =
          new btMultiBody(c.links, c.base_mass, btVector3(1, 1, 1), true, true);
      for (int i = 0; i < c.links; ++i) {
        body->setupRevolute(i, c.link_mass, btVector3(1, 1, 1), i == 0 ? -1 : i - 1,
                            btQuaternion(0, 0, 0, 1), btVector3(0, 0, 1), length, length);
      }
      body->finalizeMultiDof();
      body->setBasePos(btVector3(x, c.base_height, z));
      for (int i = 0; i < c.links; ++i) body->setJointPosMultiDof(i, &c.initial_angle);
      world.addMultiBody(body);
      multibodies.push_back(std::unique_ptr<btMultiBody>(body));

      // Colliders must exist (island solver precondition) and be registered
      // in the DefaultFilter group with the DefaultFilter|StaticFilter mask
      // (the official demo recipe; a plain addCollisionObject would use the
      // StaticFilter group and never pair with static geometry).
      {
        btAlignedObjectArray<btQuaternion> world_to_local;
        btAlignedObjectArray<btVector3> local_origin;
        body->forwardKinematics(world_to_local, local_origin);
      }
      btSphereShape* s = new btSphereShape(c.collider_radius);
      shapes.push_back(std::unique_ptr<btCollisionShape>(s));
      btMultiBodyLinkCollider* baseCol = new btMultiBodyLinkCollider(body, -1);
      baseCol->setCollisionShape(s);
      baseCol->setWorldTransform(body->getBaseWorldTransform());
      world.addCollisionObject(baseCol, btBroadphaseProxy::DefaultFilter,
                               btBroadphaseProxy::DefaultFilter | btBroadphaseProxy::StaticFilter);
      body->setBaseCollider(baseCol);
      for (int i = 0; i < c.links; ++i) {
        btMultiBodyLinkCollider* col = new btMultiBodyLinkCollider(body, i);
        col->setCollisionShape(s);
        col->setWorldTransform(body->getLink(i).m_cachedWorldTransform);
        world.addCollisionObject(col, btBroadphaseProxy::DefaultFilter,
                                 btBroadphaseProxy::DefaultFilter | btBroadphaseProxy::StaticFilter);
        body->getLink(i).m_collider = col;
      }
    }
  }
};

const PerfSceneSpec* find_scene(const char* name) {
  for (int i = 0; i < int(sizeof(kPerfScenes) / sizeof(kPerfScenes[0])); ++i) {
    if (std::strcmp(kPerfScenes[i]->name, name) == 0) return kPerfScenes[i];
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bullet_perf.exe <scene_name> [repeats]\n");
    return 2;
  }
  const char* scene_name = argv[1];
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
  const PerfSceneSpec* spec = find_scene(scene_name);
  if (!spec) {
    std::fprintf(stderr, "bullet_perf: unknown scene '%s'\n", scene_name);
    return 2;
  }
  const btScalar dt = btScalar(spec->dt_num) / btScalar(spec->dt_den);
  const int bodies = spec->num_bodies + (spec->chains.count > 0 ? spec->chains.count : 0);
  std::printf("perf scene %s bodies %d frames %d warmup %d\n", spec->name, bodies,
              spec->frames, spec->warmup);

  using clock = std::chrono::steady_clock;
  for (int r = 0; r < repeats; ++r) {
    SceneRun run(*spec);
    for (int f = 0; f < spec->warmup; ++f) run.world.stepSimulation(dt, 1, dt);
    const auto start = clock::now();
    clock::time_point last300_start = start;
    for (int f = 0; f < spec->frames; ++f) {
      if (f + 300 == spec->frames) last300_start = clock::now();
      run.world.stepSimulation(dt, 1, dt);
    }
    const auto total = clock::now() - start;
    const auto last300 = clock::now() - last300_start;
    std::printf("repeat %d total_ms %.3f last300_ms %.3f\n", r,
                std::chrono::duration<double, std::milli>(total).count(),
                std::chrono::duration<double, std::milli>(last300).count());
  }
  return 0;
}
