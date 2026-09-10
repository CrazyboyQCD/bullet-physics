# Architecture

This document describes the design of `bullet_physics`, a pure Rust
reimplementation of the core of Bullet Physics. It reflects the code as of
commit `64ab184` (2026-09-07): the full rigid-body pipeline, Featherstone
multibodies with collision integration, the classic SoftBody system, and the
consumer-facing API layer. Behavior-level differences from Bullet are tracked
in `bullet-deviations.md`, not here.

## 1. Workspace crates

The project is a Cargo workspace split along Bullet's official library
boundaries. Dependency direction is strictly layered (no cycles):

```
bullet-math  ──►  bullet-collision  ──►  bullet-dynamics  ──►  bullet (facade)
   │ (glam)                                          │
   └────────────────────────────────────────────────┘
```

| Crate             | Contents                                                                 |
| ------------------ | ------------------------------------------------------------------------ |
| `bullet-math`     | `Real` scalar + `Vector3`/`Quaternion`/`Matrix3`/`Transform` aliases, `Aabb`, inertia helpers, spatial algebra (`btSpatialAlgebra.h` port) |
| `bullet-collision`| `shapes` (10 shape kinds), `broadphase`, `narrowphase` (GJK/EPA/SAT/convex cast), `manifold`, `raycast` |
| `bullet-dynamics` | `body` (RigidBody), `world` (PhysicsWorld), `solver`, `constraint` (joints), `character`, `vehicle`, `multibody` |
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
Precision-sensitive constants (EPA tolerances, twist-axis normalization,
spherical-limit skip thresholds) follow Bullet's `BT_USE_DOUBLE_PRECISION`
behavior per feature.

## 3. Math layer

- `Vector3 = glam::Vec3` / `DVec3`, `Matrix3 = Mat3` / `DMat3`,
  `Quaternion = Quat` / `DQuat`, `Transform = Affine3A` / `DAffine3`.
- `Aabb { min, max }` with `overlaps` / `merged` / `center` / `extents`.
- `inertia_tensor_world` builds `R * diag(inv_I) * R^T` for solver use.
- Precision-independent constants (`PI`, `FRAC_PI_2`, `FRAC_1_SQRT_2`) are
  re-exported here because the toolchain lacks the associated float constants.
- `spatial` ports `btSpatialAlgebra.h` (spatial motion/force vectors,
  symmetric spatial dyads, spatial transform matrices) for the Featherstone
  subsystem.
- `#![no_std]` builds: every engine-side transcendental routes through
  `bullet_math::math` (std branch is a bit-identical pass-through; `libm`
  feature is the scalar backend), selected per crate via the `std` (default)
  / `libm` feature pair.

## 4. Shapes

`CollisionShape` is a value enum (cheap to clone, shareable across bodies —
the equivalent of sharing one `btCollisionShape`): sphere, box, capsule,
cylinder, cone (all `radius + half_height` along Y), convex hull, compound,
plane, triangle mesh (internal BVH), heightfield (materialized as a triangle
mesh). A mesh on a static body is Bullet's `btBvhTriangleMeshShape`; on a
dynamic body it becomes the GImpact case (vertex point-mass inertia,
mesh-vs-mesh and plane-vs-mesh contact paths — the local-frame BVH is carried
by the pose and never rebuilt). Heightfields stay static-only. Inertia: exact
analytic formulas for the primitives, AABB-box approximation for
hull/compound, vertex point masses for meshes (matching Bullet).

## 5. Collision pipeline

1. **Broad phase**: selectable via `BroadPhaseBackend` (an enum dispatch, no
   `dyn` on the hot path; `PhysicsWorld::new()` defaults to `AxisSweep3`,
   `with_broadphase` opts into `Dbvt`).
   - `AxisSweep3` — three-axis sweep-and-prune over incrementally maintained
     sorted edge lists (`O(n + k)` per step; static bodies with unchanged
     AABBs cost nothing); the cached pair set is kept sorted so
     `compute_pairs` is a single exact-filtered walk.
     `raycast_candidates` / `aabb_candidates` prune via the slab method
     (equivalent to `btAxisSweep3::rayTest`).
   - `Dbvt` — two dynamic AABB trees (moving / resting-for-two-steps) with
     lazily maintained fat leaf volumes (Bullet's `btDbvt` +
     `btDbvtBroadphase`): a proxy whose exact AABB stays contained in its
     fat volume touches the tree not at all; insertion descends greedily
     by center proximity and a per-step rotation pass counteracts
     degradation. `compute_pairs` doubles as the per-step tick (rotation,
     stage rotation, dynamic→fixed migration, separated-pair pruning). No
     object-count ceiling, no world-extent frame — the structure for very
     large dynamic scenes.
2. **Narrow phase**, dispatched on the `CollisionShape` enum (no `dyn` in the
   hot path):
   - `sphere-sphere`, `sphere-box`, `plane-convex` closed forms;
   - `box-box` via the ODE `dBoxBox2` SAT (15 axes + face clipping);
   - generic convex pairs via **GJK** (Minkowski difference + simplex solve)
     seeded into **EPA** for penetration depth — a line-by-line port of
     `btGjkEpa2` (same tolerances, winding rules, degeneracy aborts);
   - `convex-mesh`: convex AABB → mesh local space → BVH candidate triangles →
     per-triangle GJK+EPA, merged into one manifold and reduced to 4 points
     (deepest point + greedy most-spread selection, Bullet's manifold
     reduction). Works for moving meshes too (the query accepts any mesh pose).
   - `mesh-mesh` (the GImpact case): tree-vs-tree traversal mapping one tree's
     node boxes into the other's local frame (band-inflated conservative
     overlap) → per-triangle-pair GJK with margin-shell contacts
     (`penetration = band − separation`, Bullet's GImpact triangle-margin
     semantics); heightfields join through their internal mesh.
   - `plane-mesh`: per-vertex plane distances, single-sided like Bullet's
     GImpact-plane path, reduced to 4 points.
   - compound shapes recurse into children.
3. **Manifold**: `ContactPoint { position, normal, penetration, impulses }`
   collected into a `ContactManifold`, then merged into a per-pair
   `PersistentManifold` with Bullet's `btPersistentManifold` semantics
   (matched points replace and inherit their normal impulse, new points
   append, undetected points survive until the re-projection refresh prunes
   them; capacity 4). Fresh detections seed the merge; slow pairs are served
   from the re-projected anchors without re-running the narrow phase (see
   the refresh guards in `world.rs`).

## 6. Queries

- **Raycast** (`raycast.rs`): analytic intersections for the primitive shapes,
  GJK-distance bisection for convex hulls, BVH + Möller–Trumbore (double
  sided) for meshes/heightfields. Bullet `rayTest` semantics: `fraction` is a
  segment parameter, the normal points toward the ray origin; interior starts
  report `fraction = 0` (a documented deviation). `PhysicsWorld` queries
  (`raycast_closest`, `raycast_closest_excluding`) prune with the broad phase
  first.
- **Convex sweeps** (`narrowphase/convex_sweep.rs`): Minkowski GJK distance
  walk for convex targets, analytic planes, BVH-pruned per-triangle casts for
  meshes — the backend of `PhysicsWorld::convex_sweep_closest`, which feeds
  the character controller (equivalent of `btCollisionWorld::convexSweep` for
  a closest-hit query).
- **Contact test**: `PhysicsWorld::contact_test` returns the overlap manifold
  of a body pair without adding contacts to the solver.

## 7. Dynamics & simulation loop

`PhysicsWorld::step_simulation` mirrors `btDiscreteDynamicsWorld` (with
`btMultiBodyDynamicsWorld`/`btSoftRigidDynamicsWorld` stages fused inline —
rigid steps below; multibody `1m/3m` and soft-body `1s/3s/5s/7s/8s` inserts
are documented in `multibody-collision.md` D1 and `softbody.md` §4):

1. integrate velocities (gravity + accumulated forces, semi-implicit Euler);
2. predict positions (`position + v·dt`) for contact detection;
3. update broad-phase AABBs (sleeping bodies skip — they do not move);
4. compute broad-phase pairs;
5. simulation-island pass (union-find over broad-phase pairs, joints and
   multibody constraints): an island with any awake member wakes its
   sleeping members, an all-resting island sleeps as a unit — the island
   pass runs *before* the narrow phase so later stages filter on the final
   activation states;
6. narrow phase → contact manifolds → warm-start matching (cached normal
   impulses inherited for points within `gContactBreakingThreshold` with a
   consistent normal; friction impulses are never inherited — a documented
   deviation that prevents tangent-basis rotation oscillation);
7. joint rows flattened from the current poses (`convertJoints` equivalent);
8. solve constraints (correct velocities);
9. write solved impulses back to the warm-start cache;
10. deactivation check (`updateDeactivation` / `goToSleep` semantics);
11. integrate positions with the corrected velocities — CCD fast-movers are
    clamped at the first swept-sphere TOI, and split-impulse push velocities
    are applied to the pose here;
12. synchronize motion states: opted-in `MotionState`s of *active,
    non-static/kinematic* bodies receive the step's final pose (Bullet's
    `synchronizeMotionStates` point);
13. clear accumulated forces.

**Sleeping**: `ActivationState` mirrors Bullet's five states; thresholds
0.8 linear / 1.0 angular, `DEACTIVATION_TIME = 2 s`; static bodies start
asleep; `apply_force` auto-activates.

**CCD**: bodies with a motion threshold exceeding their per-step displacement
are swept as a sphere of `sweptSphereRadius` against convex shapes (GJK on the
inflated shape), planes (analytic) and meshes (per-triangle analytic casts via
the BVH); the position is clamped at the first TOI and the response happens on
the next step's narrow phase (Bullet's `integrateTransforms` behavior).

## 8. Solver

`sequential_impulse` implements a projected Gauss-Seidel contact solver:
normal impulses (non-negative, with restitution above a velocity threshold and
Baumgarte stabilization past a 5 mm slop), two tangent friction directions
clamped by Coulomb's law, friction/restitution combined as products (matching
`btManifoldResult`), and a **split-impulse** pass: penetrations deeper than
0.04 m drop the Baumgarte bias from the velocity solve and are corrected by a
post-iteration pseudo-impulse accumulated on each body's push velocity,
applied to the pose after position integration (never touching the real
velocities). Joint rows share the PGS loop with contact rows.

## 9. Joints (rigid-body constraints)

`Constraint` (`dynamics/constraint.rs`, Bullet's `ConstraintSolver` module)
with a `ConstraintKind` enum mirroring the six Bullet joints:

| Kind           | Bullet class              | Formulation notes |
|----------------|---------------------------|-------------------|
| `Point2Point`  | `btPoint2PointConstraint` | three world-axis bilateral rows, ERP bias, optional impulse clamp |
| `Hinge`        | `btHingeConstraint`       | classic `getInfo2Internal`: pivot rows + two axis-alignment rows + optional limit/motor row (`btAngularLimit` math) |
| `Fixed`        | `btFixedConstraint`       | pivot rows + three angular rows driving the relative Euler angles to zero |
| `ConeTwist`    | `btConeTwistConstraint`   | elliptical swing limit + one-sided twist limit + motor with in-cone target clamping |
| `Slider`       | `btSliderConstraint`      | frame-offset formulation: blended axis, two angular + two linear ortho rows, limit/motor rows |
| `Generic6Dof`  | `btGeneric6DofConstraint` | classic recipe: Euler-angle angular rows along `m_calculatedAxis`, frame-column linear rows, per-axis limit motors |

Rows are explicit Jacobians (`ConstraintRow`) with one-sided/bilateral clamp
windows; joints are not warm-started (Bullet default); pairs linked by a joint
skip narrow-phase collision (`disableCollisionsBetweenLinkedBodies`).

## 10. Featherstone multibody

`bullet_dynamics::multibody` is a self-contained port of
`BulletDynamics/Featherstone`:

- `MultiBody` (`btMultiBody`): the five joint types (revolute, prismatic,
  spherical, planar, fixed), forward kinematics, `compTreeLinkVelocities`,
  the articulated-body algorithm (up-down-up passes, per-link `D⁻¹` scatter
  layout, gyro/Coriolis terms, damping), 6×6 spatial-inertia solve for the
  base, constraint-impulse response reusing the ABA factorization, and the
  exponential-map position integrator.
- `MultiBodyConstraint`: point-to-point, fixed, slider, joint motor, gear,
  joint limit, spherical joint limit, spherical joint motor — the
  `fillMultiBodyConstraint` recipe (unit-impulse response, `jacDiagABInv`,
  split-penetration RHS suppression like Bullet's defaults).
- Joint reaction feedback (`btMultiBodyJointFeedback`, phase E): an optional
  per-link slot (`enable_link_joint_feedback`) written by the articulated-
  body algorithm's second upward loop — `Î·â + ẑ` at the link COM, stored
  (`=`) by the external-force pass and augmented (`+=`) by a post-solve
  constraint pass that reruns the algorithm on the accumulated solver forces
  (contact + constraint rows), with optional world-space / joint-frame
  reporting. `top` = linear force, `bottom` = angular torque.
- `PhysicsWorld` steps the multibodies inline (the `btMultiBodyDynamicsWorld`
  fusion, phase A of `docs/design/multibody-collision.md`): gravity → ABA velocity
  integration → row conversion + PGS (10×1 iterations, alternating
  direction) → writeback with `±maxCoordinateVelocity` clamping → position
  integration → sleeping check, interleaved with the rigid-body stages.
  Collision integration (`btMultiBodyLinkCollider`) landed with phases B–D
  of `docs/design/multibody-collision.md`: link/base colliders register in the
  shared broad phase, detect contacts with rigid bodies and each other, and
  solve in the same PGS loop, inside shared simulation islands.

## 11. Higher-level features

- **Kinematic character controller** (`character.rs`): port of
  `btKinematicCharacterController` (stepUp / stepForwardAndStrafe / stepDown
  double-sweep, Quake-2 anti-oscillation, penetration recovery). The character
  is a static solid body moved through world queries (`convex_sweep_closest`,
  `contact_test`).
- **Raycast vehicle** (`vehicle.rs`): port of `btRaycastVehicle` — wheel rays,
  spring-damper suspension scaled by chassis mass, steering/drive/brake
  impulses, lateral friction with the friction-cone sliding clamp and
  `ROLLING_INFLUENCE_FIX`. Runs after `step_simulation` like Bullet's actions;
  wheel rays explicitly exclude the chassis.
- **Consumer-facing collision surface** (`docs/api-gaps.md`): collision
  groups/masks with the symmetric `group & mask` pair test (every narrow
  phase, the CCD sweep, the soft dispatch, ray filters); trigger volumes via
  `CollisionFlags::NO_CONTACT_RESPONSE`; Rapier-style collision events
  (`world.contact_events()`, per-body opt-in, sides reported as the
  `CollisionObject` enum for rigid bodies, multibody link colliders and soft
  bodies); the persistent touch listing `world.overlapping(handle)`;
  kinematic rigid bodies (`CollisionFlags::KINEMATIC`, velocity-driven pose);
  motion states (`MotionState` trait); optional `PairFilter` predicate.

## 12. Memory management

- Arenas (`Vec<RigidBody>`) owned by `PhysicsWorld`; bodies are addressed by
  typed handles (`BodyHandle`/`MultiBodyHandle`/`SoftBodyHandle` — newtypes
  over `NonZeroU32`, the analog of Bullet's raw pointers; rigid slots are
  stable and never reused, soft-body slots renumber on removal).
- Body removal is supported: `remove_body` destroys the proxy and cascades
  the constraints and cached manifolds that referenced the body (a tombstoned
  slot panics on later access — see deviations §1/§3).
- No per-step heap allocation on the hot path beyond reusable buffers.
- `CollisionShape` is a value enum, cheap to clone and shareable across bodies
  (equivalent to Bullet sharing one `btCollisionShape`).

## 13. Verification strategy

- **Unit tests**: 350 tests across the crates, all run in both precisions
  (`--features f64`); `python tools/check.py` runs them plus the 14 demo
  assertions.
- **Parity harness** (`docs/parity-harness.md`): frame-by-frame trajectory
  comparison against the real Bullet static library (C++ and Rust recorders
  driven from one `tools/scenes.json`), 8 scenes × f32/f64; the difference
  ledger `docs/bullet-compare.md` classifies every exceedance as a known
  trade-off or an implementation problem.
- **Multibody probes**: `tools/multibody_probe.cpp` vs
  `crates/bullet/examples/multibody_probe.rs`, 24 scenes × f32/f64 against
  the real Bullet library (13 multibody-dynamics scenes, the 6 collider
  scenes of `docs/design/multibody-collision.md` §7, the joint-feedback pair and
  the three multibody-vs-rigidbody constraint scenes, compared with
  `tools/mbdiff.py`; build via `tools/build_multibody_probe.py`).
- **Soft-body / GImpact probes**: `tools/softbody_probe.cpp` +
  `examples/softbody_probe.rs` (classic soft-body scenes against
  `BulletSoftBody`) and `tools/gimpact_probe.cpp` +
  `examples/gimpact_probe.rs` (dynamic mesh bodies), same ledger.
- **no_std**: `cargo check --no-default-features --features libm` per crate
  (f32 + f64) plus a `wasm32-unknown-unknown` cross-check.

## 14. Status

- **Done**: the end-to-end rigid-body pipeline (all shape classes, both
  broad phases, GJK/EPA narrow phase, persistent manifolds, sequential-
  impulse solver, CCD, sleeping/islands, six joints, kinematic bodies,
  rolling friction, motion states); Featherstone multibodies with collision
  integration and the full constraint set; the classic SoftBody system
  phases A–F (rope/cloth/volume/clusters/joints/cut/refine, soft↔multibody
  coupling); raycast vehicle + kinematic character controller; the consumer
  API layer (typed handles, collision groups/masks, pair filter, all-hits
  rays, per-body gravity, triggers + collision events for all
  collision-object kinds, ghost overlap list, compound child transforms);
  `#![no_std]` builds (`libm` backend).
- **Next**: minimal debug draw (`world.debug_primitives`,
  `docs/api-gaps.md` P3), then correctness hardening / independent
  subsystem audits (`docs/audit/`). Later candidates: serialization.

## 15. Bullet → Rust mapping

| Bullet 3                                | Rust module / type                          |
| ---------------------------------------- | ------------------------------------------- |
| `btScalar`                               | `bullet_math::Real`                         |
| `btVector3` / `btQuaternion` / `btMatrix3x3` / `btTransform` | `Vector3` / `Quaternion` / `Matrix3` / `Transform` |
| `btSpatialAlgebra.h`                     | `bullet_math::spatial`                      |
| `btAlignedObjectArray<T>`                | `Vec<T>`                                    |
| `btCollisionShape`                       | `bullet_collision::shapes::CollisionShape`  |
| `btBoxShape` / `btSphereShape` / …       | `CollisionShape::Box` / `::Sphere` / …      |
| `btBvhTriangleMeshShape` / `btHeightfieldTerrainShape` | `CollisionShape::TriangleMesh` / `::Heightfield` |
| `btGImpactMeshShape` (dynamic concave mesh) | `CollisionShape::TriangleMesh` on a dynamic body (`RigidBody::new_dynamic`) |
| `btAxisSweep3`                           | `broadphase::AxisSweep3` (default)          |
| `btDbvtBroadphase`                       | `broadphase::DbvtBroadPhase` (opt-in via `BroadPhaseBackend`) |
| `btGjkPairDetector` + `btVoronoiSimplexSolver` | `narrowphase::gjk`                    |
| `btGjkEpa2`                              | `narrowphase::epa`                          |
| `btCollisionWorld::rayTest` / `convexSweep` | `raycast::raycast`, `PhysicsWorld::raycast_closest` / `narrowphase::convex_cast` |
| `btPersistentManifold`                   | `manifold::ContactManifold` (+ impulse cache) |
| `btRigidBody` / `btCollisionObject`      | `dynamics::RigidBody`                       |
| `btDiscreteDynamicsWorld`                | `dynamics::PhysicsWorld`                    |
| `btSequentialImpulseConstraintSolver`    | `solver::sequential_impulse`                |
| `btTypedConstraint` subclasses           | `constraint::{Constraint, ConstraintKind}`   |
| `btKinematicCharacterController`         | `dynamics::KinematicCharacterController`    |
| `btRaycastVehicle`                       | `dynamics::RaycastVehicle`                  |
| `btMultiBody` / `btMultiBodyConstraint` / `btMultiBodyDynamicsWorld` | `multibody::{MultiBody, MultiBodyConstraint}`, stepped by `dynamics::PhysicsWorld` |
| `btMultiBodyLinkCollider` | `multibody::MultiBody` link/base collider slots registered in the shared broad phase (phases B–D of `docs/design/multibody-collision.md`) |
| `btRigidBody::setRollingFriction` | `RigidBody.rolling_friction` (torsional rows per contact) |
| `btMotionState` | `dynamics::MotionState` trait (`initial_pose` / `synchronize`, builder `motion_state(..)`) |
| `btGhostObject` overlap listing | sensor flag (`CollisionFlags::NO_CONTACT_RESPONSE`) + `PhysicsWorld::overlapping(handle)` |
| `btCollisionObject` (collision-object kinds in the events API) | `CollisionObject` enum: `Body` / `Link { body, link }` / `Soft` |
