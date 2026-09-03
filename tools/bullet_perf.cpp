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
// `--dump` instead steps one run from frame 0 (including warmup) and prints
// aggregate stack health every 5 frames (matching the Rust runner's
// run_dump protocol, for behavioral comparison of settling scenes).
//
// Fairness notes (documented in docs/bullet-perf.md):
//   - the broad phase follows the scene spec: btAxisSweep3 (matching this
//     crate's default AxisSweep3, scoped with the broadphase_aabb hint) or
//     btDbvtBroadphase (matching DbvtBroadPhase) for the dbvt_* scenes;
//   - SOLVER_RANDMIZE_ORDER is cleared (same reproducibility precondition as
//     the parity recorder);
//   - multibody scenes use btMultiBodyDynamicsWorld with link colliders in
//     the DefaultFilter group (required for the island solver) and a manual
//     forwardKinematics before attaching them;
//   - dynamic mesh bodies (mass > 0) use btGImpactMeshShape with the GImpact
//     collision algorithm registered on the dispatcher (without the
//     registration every gimpact pair silently misses - the numbers would
//     look fast and be meaningless), matching this crate's dynamic
//     TriangleMesh path; static meshes stay btBvhTriangleMeshShape.

// Usage: bullet_perf.exe <scene_name> [repeats]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "btBulletDynamicsCommon.h"
#include <BulletCollision/BroadphaseCollision/btAxisSweep3.h>
#include <BulletCollision/BroadphaseCollision/btDbvtBroadphase.h>
#include <BulletCollision/CollisionShapes/btStaticPlaneShape.h>
#include <BulletCollision/Gimpact/btGImpactShape.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>

#include <BulletSoftBody/btSoftBody.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>

#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>

#include "bullet_perf_scenes.h"

namespace {

btCollisionShape* make_shape(const PerfBodySpec& b, bool is_static,
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
    case PERF_SHAPE_PLANE:
      return new btStaticPlaneShape(btVector3(p[0], p[1], p[2]), p[3]);
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
      if (is_static) return new btBvhTriangleMeshShape(tm, true);
      // Dynamic concave mesh: the GImpact route (updateBound is mandatory -
      // without it the shape keeps an empty AABB and never pairs).
      btGImpactMeshShape* g = new btGImpactMeshShape(tm);
      g->updateBound();
      return g;
    }
  }
  return nullptr;
}

// Owns everything the world references; destroyed (and so unregistered)
// before the world member goes away.
struct SceneRun {
  // Registers the soft-vs-rigid collision algorithms (the default
  // configuration would silently drop every soft pair - the nodes fall
  // straight through the ground); a superset of the default config, so
  // non-soft scenes dispatch exactly as before.
  btSoftBodyRigidBodyCollisionConfiguration config;
  btCollisionDispatcher dispatcher;
  std::unique_ptr<btBroadphaseInterface> broadphase;  // per-scene: AxisSweep3 or Dbvt
  btMultiBodyConstraintSolver solver;   // plain scenes: solves rigid rows too
  std::vector<std::unique_ptr<btCollisionShape>> shapes;
  std::vector<std::unique_ptr<btTriangleMesh>> mesh_storage;
  std::vector<std::unique_ptr<btMotionState>> motions;
  std::vector<std::unique_ptr<btRigidBody>> bodies;
  std::vector<std::unique_ptr<btMultiBody>> multibodies;
  // Soft scenes run btSoftRigidDynamicsWorld (soft + rigid stages); every
  // other scene keeps btMultiBodyDynamicsWorld (its mb stages are no-ops
  // without multibodies). Both derive from btDiscreteDynamicsWorld; the
  // scene kinds never mix (the generator validates softs/chains as
  // mutually exclusive).
  std::unique_ptr<btDiscreteDynamicsWorld> world;
  std::vector<btSoftBody*> softs;       // borrowed (owned by the world)
  long frame = 0;                       // total stepped frames (warmup included)
  const bool arena;                     // per-frame arena tick (launcher + dozer)
  // Staggered-rain scenes: (spawn_frame, bodies index) pairs built but not
  // registered; the step loop drops each layer when its frame comes due
  // (table order keeps pending sorted). spawned[i] gates the dump stats.
  std::vector<std::pair<int, size_t>> pending;
  std::vector<char> spawned;

  SceneRun(const PerfSceneSpec& spec)
      : config(),
        dispatcher(&config),
        broadphase(spec.bp_kind ? static_cast<btBroadphaseInterface*>(new btDbvtBroadphase())
                                : static_cast<btBroadphaseInterface*>(new btAxisSweep3(
                                      btVector3(spec.bp_min[0], spec.bp_min[1], spec.bp_min[2]),
                                      btVector3(spec.bp_max[0], spec.bp_max[1], spec.bp_max[2]),
                                      // The default handle cap is 16384; the big sweep
                                      // tiers (32k bodies) need room for 2x that, and
                                      // unsigned short caps the parameter at 65535.
                                      static_cast<unsigned short>(std::min<long>(
                                          65535, std::max<long>(16384, spec.num_bodies * 2L)))))),
        solver(),
        arena(spec.script == 1) {
    const bool soft_scene = spec.softs.count > 0;
    if (soft_scene) {
      world = std::make_unique<btSoftRigidDynamicsWorld>(&dispatcher, broadphase.get(), &solver, &config);
    } else {
      world = std::make_unique<btMultiBodyDynamicsWorld>(&dispatcher, broadphase.get(), &solver, &config);
    }
    world->setGravity(btVector3(spec.gravity[0], spec.gravity[1], spec.gravity[2]));
    // The soft world info carries its own gravity default and setGravity
    // does not sync it (predictMotion reads worldInfo); sync explicitly.
    if (soft_scene) {
      static_cast<btSoftRigidDynamicsWorld*>(world.get())->getWorldInfo().m_gravity =
          btVector3(spec.gravity[0], spec.gravity[1], spec.gravity[2]);
    }
    // Reproducibility precondition (docs/parity-harness.md).
    btContactSolverInfo& info = world->getSolverInfo();
    info.m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
    info.m_numIterations = spec.iterations;
    // Registers on this run's dispatcher (the table lives per dispatcher
    // instance, so every SceneRun needs it; harmless for gimpact-free scenes).
    btGImpactCollisionAlgorithm::registerAlgorithm(&dispatcher);

    for (int i = 0; i < spec.num_bodies; ++i) {
      const PerfBodySpec& b = spec.bodies[i];
      const bool is_static = b.mass == btScalar(0);
      shapes.push_back(std::unique_ptr<btCollisionShape>(
          make_shape(b, is_static, mesh_storage)));
      btCollisionShape* shape = shapes.back().get();
      motions.push_back(std::unique_ptr<btMotionState>(
          new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1),
                                               btVector3(b.pos[0], b.pos[1], b.pos[2])))));
      btVector3 local_inertia(0, 0, 0);
      if (!is_static) shape->calculateLocalInertia(b.mass, local_inertia);
      btRigidBody::btRigidBodyConstructionInfo ci(
          is_static ? btScalar(0.0) : b.mass, motions.back().get(), shape, local_inertia);
      bodies.push_back(std::unique_ptr<btRigidBody>(new btRigidBody(ci)));
      btRigidBody* body = bodies.back().get();
      body->setFriction(b.friction);
      body->setRestitution(b.restitution);
      if (b.spawn_frame > 0 && !is_static) {
        pending.push_back({b.spawn_frame, bodies.size() - 1});
        spawned.push_back(0);
      } else {
        world->addRigidBody(body);
        spawned.push_back(1);
      }
    }

    if (spec.chains.count > 0) add_chains(spec.chains);
    if (spec.softs.count > 0) add_softs(spec.softs);
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
      static_cast<btMultiBodyDynamicsWorld*>(world.get())->addMultiBody(body);
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
      world->addCollisionObject(baseCol, btBroadphaseProxy::DefaultFilter,
                               btBroadphaseProxy::DefaultFilter | btBroadphaseProxy::StaticFilter);
      body->setBaseCollider(baseCol);
      for (int i = 0; i < c.links; ++i) {
        btMultiBodyLinkCollider* col = new btMultiBodyLinkCollider(body, i);
        col->setCollisionShape(s);
        col->setWorldTransform(body->getLink(i).m_cachedWorldTransform);
        world->addCollisionObject(col, btBroadphaseProxy::DefaultFilter,
                                  btBroadphaseProxy::DefaultFilter | btBroadphaseProxy::StaticFilter);
        body->getLink(i).m_collider = col;
      }
    }
  }

  // A grid of cloth patches laid out like the Rust runner's add_softs (patch
  // corners on a deterministic grid, identical material/iteration settings);
  // each patch is built by btSoftBodyHelpers::CreatePatch - the same helper
  // the Rust create_patch was ported from (byte-exact, perturbation 0).
  void add_softs(const PerfSoftSpec& c) {
    const int rows = (c.count + c.cols - 1) / c.cols;
    for (int k = 0; k < c.count; ++k) {
      const int col = k % c.cols;
      const int row = k / c.cols;
      const btScalar x0 = c.origin[0] + btScalar(col) * (c.size_x + c.gap);
      const btScalar z0 = c.origin[2] + btScalar(row) * (c.size_z + c.gap);
      const btScalar y = c.origin[1];
      btSoftBody* psb = btSoftBodyHelpers::CreatePatch(
          static_cast<btSoftRigidDynamicsWorld*>(world.get())->getWorldInfo(),
          btVector3(x0, y, z0),
          btVector3(x0 + c.size_x, y, z0),
          btVector3(x0, y, z0 + c.size_z),
          btVector3(x0 + c.size_x, y, z0 + c.size_z),
          c.resx, c.resy, c.fixeds, true);
      psb->m_cfg.piterations = c.piterations;
      psb->m_cfg.kDF = c.k_df;
      psb->m_materials[0]->m_kLST = c.k_lst;
      psb->m_cfg.aeromodel = c.aero == 1 ? btSoftBody::eAeroModel::V_TwoSided
                                         : btSoftBody::eAeroModel::V_Point;
      psb->setWindVelocity(btVector3(c.wind[0], c.wind[1], c.wind[2]));
      static_cast<btSoftRigidDynamicsWorld*>(world.get())->addSoftBody(psb);
      softs.push_back(psb);
    }
  }

  // One scripted step (plain scenes: just the solver step). The arena tick
  // mirrors the Rust runner's arena_tick formula for formula so the two
  // sides stay workload-comparable; it runs before the solver sees the frame.
  void step(btScalar dt) {
    if (arena) {
      // Launcher: a small ball every 30 frames (21 total), aimed at a
      // deterministic spread across the box tower.
      if (frame % 30 == 0 && frame < 610) {
        const long k = frame / 30;
        const btVector3 from(0, btScalar(2.5), -8);
        const btVector3 target(btScalar((k % 5) - 2) * btScalar(0.6), btScalar(0.7), 0);
        const btVector3 dir = (target - from).normalize();
        btSphereShape* s = new btSphereShape(btScalar(0.25));
        shapes.push_back(std::unique_ptr<btCollisionShape>(s));
        btVector3 inertia(0, 0, 0);
        s->calculateLocalInertia(btScalar(1), inertia);
        btDefaultMotionState* ms = new btDefaultMotionState(
            btTransform(btQuaternion(0, 0, 0, 1), from));
        motions.push_back(std::unique_ptr<btMotionState>(ms));
        btRigidBody::btRigidBodyConstructionInfo ci(btScalar(1), ms, s, inertia);
        btRigidBody* b = new btRigidBody(ci);
        b->setFriction(btScalar(0.5));
        b->setLinearVelocity(dir * btScalar(12));
        world->addRigidBody(b);
        bodies.push_back(std::unique_ptr<btRigidBody>(b));
      }
      // Dozer (bodies[1], mass 50): velocity-driven along
      // x = 4*sin(2*pi*t/240); only x is overridden so y/z still solve
      // naturally. activate() first - setting a velocity on a sleeping body
      // would not wake it (and the wake propagates through contacts).
      btRigidBody* dozer = bodies[1].get();
      dozer->activate();
      btVector3 v = dozer->getLinearVelocity();
      v.setX(btScalar(4) * (btScalar(2) * SIMD_PI / btScalar(240)) *
             std::cos(btScalar(2) * SIMD_PI * frame / btScalar(240)));
      dozer->setLinearVelocity(v);
    }
    // Staggered rain: register every layer whose spawn frame has come due
    // (table order keeps pending sorted, so a prefix erase suffices).
    {
      size_t k = 0;
      while (k < pending.size() && pending[k].first <= frame) ++k;
      for (size_t j = 0; j < k; ++j) {
        world->addRigidBody(bodies[pending[j].second].get());
        spawned[pending[j].second] = 1;
      }
      if (k > 0) pending.erase(pending.begin(), pending.begin() + static_cast<long>(k));
    }
    world->stepSimulation(dt, 1, dt);
    ++frame;
  }
};

const PerfSceneSpec* find_scene(const char* name) {
  for (int i = 0; i < int(sizeof(kPerfScenes) / sizeof(kPerfScenes[0])); ++i) {
    if (std::strcmp(kPerfScenes[i]->name, name) == 0) return kPerfScenes[i];
  }
  return nullptr;
}

// Debug probe matching the Rust runner's run_dump: aggregate stack health
// every 5 frames (sum of dynamic-body y, sum/max speed, awake count).
void run_dump(const PerfSceneSpec& spec, btScalar dt) {
  SceneRun run(spec);
  const int frames = spec.warmup + spec.frames;
  for (int f = 0; f < frames; ++f) {
    run.step(dt);
    if (f % 5 == 0 || f == frames - 1) {
      double sum_y = 0.0, sum_v = 0.0, max_v = 0.0;
      int awake = 0, count = 0;
      for (int i = 0; i < int(run.bodies.size()); ++i) {
        const btRigidBody* b = run.bodies[i].get();
        if (b->getInvMass() == btScalar(0)) continue;
        if (!run.spawned.empty() && !run.spawned[i]) continue;  // not dropped yet
        ++count;
        sum_y += double(b->getWorldTransform().getOrigin().getY());
        const double v = double(b->getLinearVelocity().length());
        sum_v += v;
        if (v > max_v) max_v = v;
        if (b->isActive()) ++awake;
      }
      // Soft-body aggregate: node count, lowest node, fastest node (the
      // settle/activity health of the cloth grid).
      int soft_nodes = 0;
      double soft_min_y = 0.0, soft_max_v = 0.0;
      for (int i = 0; i < int(run.softs.size()); ++i) {
        const btSoftBody* psb = run.softs[i];
        for (int j = 0; j < psb->m_nodes.size(); ++j) {
          const btSoftBody::Node& n = psb->m_nodes[j];
          if (soft_nodes == 0 || double(n.m_x.getY()) < soft_min_y)
            soft_min_y = double(n.m_x.getY());
          const double v = double(n.m_v.length());
          if (v > soft_max_v) soft_max_v = v;
          ++soft_nodes;
        }
      }
      std::printf("dump frame %d bodies %d sum_y %.4f sum_v %.5f max_v %.5f awake %d "
                  "soft_nodes %d soft_min_y %.4f soft_max_v %.5f\n",
                  f, count, sum_y, sum_v, max_v, awake,
                  soft_nodes, soft_min_y, soft_max_v);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bullet_perf.exe <scene_name> [repeats] [--dump]\n");
    return 2;
  }
  const char* scene_name = argv[1];
  bool dump = false;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump") == 0) dump = true;
  }
  const PerfSceneSpec* spec = find_scene(scene_name);
  if (!spec) {
    std::fprintf(stderr, "bullet_perf: unknown scene '%s'\n", scene_name);
    return 2;
  }
  const btScalar dt = btScalar(spec->dt_num) / btScalar(spec->dt_den);
  if (dump) {
    run_dump(*spec, dt);
    return 0;
  }
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
  const int bodies = spec->num_bodies + (spec->chains.count > 0 ? spec->chains.count : 0)
                     + spec->softs.count;
  std::printf("perf scene %s bodies %d frames %d warmup %d\n", spec->name, bodies,
              spec->frames, spec->warmup);

  using clock = std::chrono::steady_clock;
  for (int r = 0; r < repeats; ++r) {
    SceneRun run(*spec);
    for (int f = 0; f < spec->warmup; ++f) run.step(dt);
    const auto start = clock::now();
    clock::time_point last300_start = start;
    for (int f = 0; f < spec->frames; ++f) {
      if (f + 300 == spec->frames) last300_start = clock::now();
      run.step(dt);
    }
    const auto total = clock::now() - start;
    const auto last300 = clock::now() - last300_start;
    std::printf("repeat %d total_ms %.3f last300_ms %.3f\n", r,
                std::chrono::duration<double, std::milli>(total).count(),
                std::chrono::duration<double, std::milli>(last300).count());
  }
  return 0;
}
