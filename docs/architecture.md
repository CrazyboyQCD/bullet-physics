# Architecture

This document describes the design of `bullet_physics`, a pure Rust
reimplementation of the core of Bullet Physics.

## 1. Workspace crates

The project is a Cargo workspace split along Bullet's official library
boundaries. Dependency direction is strictly layered (no cycles):

```
bullet-math  ──►  bullet-collision  ──►  bullet-dynamics  ──►  bullet (facade)
   │ (glam)                                          │
   └────────────────────────────────────────────────┘
```

| Crate             | Contents                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| `bullet-math`     | `Real` scalar + `Vector3`/`Quaternion`/`Matrix3`/`Transform` aliases, `Aabb`, inertia helpers |
| `bullet-collision`| `shapes` (CollisionShape enum), `broadphase`, `narrowphase` (GJK/EPA/SAT), `manifold` |
| `bullet-dynamics` | `body` (RigidBody), `world` (PhysicsWorld), `solver` (sequential impulse) |
| `bullet`          | Re-exports the crates plus a `prelude` for convenience                     |

Each crate is independently versioned and publishable; the `bullet` facade is
the recommended entry point for downstream users.

## 2. Precision (f32 / f64)

glam provides separate `f32` and `f64` type families, so precision is selected
with cargo features plus `cfg` switching:

```rust
#[cfg(not(feature = "f64"))] pub type Real = f32;
#[cfg(feature = "f64")]     pub type Real = f64;
```

Every crate defines `f32` (default) and `f64` features; `f64` is forwarded to
the upstream crates so a single `--features f64` on the facade switches the
whole workspace. All code uses only the aliases, never a concrete precision.

## 3. Math layer

- `Vector3 = glam::Vec3` / `DVec3`, `Matrix3 = Mat3` / `DMat3`,
  `Quaternion = Quat` / `DQuat`, `Transform = Affine3A` / `DAffine3`.
- `Aabb { min, max }` with `overlaps` / `merged` / `center` / `extents`.
- `inertia_tensor_world` builds `R * diag(inv_I) * R^T` for solver use.

## 4. Collision pipeline

1. **Broad phase** (`BroadPhase` trait + `BruteForceBroadPhase`): coarse
   AABB-based pair filtering. A `DbvtBroadPhase` is planned for performance.
2. **Narrow phase**: exact convex-convex tests, dispatched on the
   `CollisionShape` enum (no `dyn` in the hot path):
   - `sphere-sphere`, `sphere-box` closed forms;
   - `box-box` via the ODE `dBoxBox2` SAT (15 axes + face clipping);
   - generic convex pairs via **GJK** (Minkowski difference + simplex solve)
     seeded into **EPA** for penetration depth.
3. **Manifold**: `ContactPoint { position, normal, penetration, impulses }`
   collected into a `ContactManifold`.

## 5. Dynamics & simulation loop

`PhysicsWorld::step_simulation` mirrors `btDiscreteDynamicsWorld`:

1. integrate velocities (gravity + accumulated forces, semi-implicit Euler);
2. predict positions for contact detection;
3. update broad-phase AABBs;
4. compute broad-phase pairs;
5. narrow phase → contact manifolds → constraints;
6. solve constraints (correct velocities);
7. integrate positions with the corrected velocities;
8. clear accumulated forces.

## 6. Solver

`sequential_impulse` implements a projected Gauss-Seidel contact solver:
normal impulse (non-negative, with restitution and Baumgarte stabilization) and
two tangent friction directions clamped by Coulomb's law.

## 7. Memory management

- Arenas (`Vec<RigidBody>`) owned by `PhysicsWorld`; bodies are addressed by
  index.
- No per-step heap allocation on the hot path (broad-phase pairs, manifolds,
  and constraints are collected into reusable buffers).
- `CollisionShape` is a value enum, cheap to clone and shareable across bodies
  (equivalent to Bullet sharing one `btCollisionShape`).

## 8. Roadmap

- **Phase 1 (done)**: MVP end-to-end rigid-body pipeline — sphere/box shapes,
  brute-force broad phase, GJK/EPA + specialized narrow phase, sequential
  impulse solver, falling-box demo.
- **Phase 2**: capsule/cylinder/cone/convex-hull/compound/plane shapes, raycast,
  `AxisSweep3`, triangle mesh (BVH) and heightfield.
- **Phase 3**: sleeping/activation, CCD, character controller, vehicle.
- **Phase 4**: constraints (Point2Point, Hinge, Generic6Dof, ...).
- **Phase 5**: Featherstone multibody.
- Serialization is deferred until the core is complete.

## 9. Bullet → Rust mapping

| Bullet 3                                | Rust module / type                          |
| ---------------------------------------- | ------------------------------------------- |
| `btScalar`                               | `bullet_math::Real`                         |
| `btVector3` / `btQuaternion` / `btMatrix3x3` / `btTransform` | `Vector3` / `Quaternion` / `Matrix3` / `Transform` |
| `btAlignedObjectArray<T>`                | `Vec<T>`                                    |
| `btCollisionShape`                       | `bullet_collision::shapes::CollisionShape`  |
| `btBoxShape` / `btSphereShape`           | `BoxShape` / `Sphere`                       |
| `btDbvtBroadphase` / `btAxisSweep3`      | `broadphase::DbvtBroadPhase` / `AxisSweep3` |
| `btGjkPairDetector` + `btVoronoiSimplexSolver` | `narrowphase::gjk`                    |
| `btGjkEpa2`                              | `narrowphase::epa`                          |
| `btPersistentManifold`                   | `manifold::ContactManifold`                 |
| `btRigidBody` / `btCollisionObject`      | `dynamics::RigidBody`                       |
| `btDiscreteDynamicsWorld`                | `dynamics::PhysicsWorld`                    |
| `btSequentialImpulseConstraintSolver`    | `solver::sequential_impulse`                |
| `btTypedConstraint` subclasses           | `solver::joints` (planned)                  |
| Featherstone `btMultiBody*`              | `multibody` (planned)                       |
