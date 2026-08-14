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
// crates/bullet/examples/record_scenes.rs; both consume the same generated
// scene tables, so scene parameters have a single source of truth.
//
// Usage: bullet_record.exe [scene_name]

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "btBulletDynamicsCommon.h"
#include "bullet_scenes.h"

namespace {

btCollisionShape* make_shape(const BodySpec& b) {
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

  // Reproducibility precondition (docs/parity-harness.md).
  btContactSolverInfo& info = world.getSolverInfo();
  info.m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
  info.m_numIterations = spec->iterations;

  const btScalar dt = btScalar(spec->dt_num) / btScalar(spec->dt_den);

  // --- Build bodies (owned here; world references them until it is destroyed
  // first, which the declaration order guarantees) ---
  std::vector<std::unique_ptr<btCollisionShape>> shapes;
  std::vector<std::unique_ptr<btMotionState>> motions;
  std::vector<std::unique_ptr<btRigidBody>> bodies;
  for (int i = 0; i < spec->num_bodies; ++i) {
    const BodySpec& b = spec->bodies[i];
    shapes.push_back(std::unique_ptr<btCollisionShape>(make_shape(b)));
    btCollisionShape* shape = shapes.back().get();
    motions.push_back(std::unique_ptr<btMotionState>(
        new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1),
                                             btVector3(b.pos[0], b.pos[1], b.pos[2])))));
    btVector3 local_inertia(0, 0, 0);
    if (!b.is_static) shape->calculateLocalInertia(b.mass, local_inertia);
    btRigidBody::btRigidBodyConstructionInfo ci(
        b.is_static ? btScalar(0.0) : b.mass, motions.back().get(), shape, local_inertia);
    bodies.push_back(std::unique_ptr<btRigidBody>(new btRigidBody(ci)));
    btRigidBody* body = bodies.back().get();
    body->setFriction(b.friction);
    body->setRestitution(b.restitution);
    world.addRigidBody(body);
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
    world.stepSimulation(dt, 1, dt);
    for (int r = 0; r < spec->num_record; ++r) {
      const btTransform& t = bodies[record_idx[r]]->getCenterOfMassTransform();
      const btVector3& p = t.getOrigin();
      const btQuaternion& q = t.getRotation();
      std::fprintf(out[r], "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   btScalar(frame) * dt, p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
    }
  }
  for (FILE* f : out) std::fclose(f);
  return 0;
}
