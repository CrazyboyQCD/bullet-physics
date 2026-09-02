# API usability notes — differences from Bullet's C++ API

This document captures *usage feel*: where the Rust API reads better or
worse than Bullet's C++ API, and the prioritized backlog to close the
gap. Behavior semantics are out of scope here — see
`bullet-deviations.md` (the canonical "does X match Bullet?" reference).

## 1. Already better than Bullet

- **World setup**: Bullet needs four heap objects assembled by hand
  (`btDefaultCollisionConfiguration`, `btCollisionDispatcher`,
  `btBroadphase`, `btSequentialImpulseConstraintSolver`) before the
  `btDiscreteDynamicsWorld` constructor. Here: `PhysicsWorld::new()`.
- **Resource lifetime**: shapes and bodies are values; nothing to
  `delete`. Bullet's ownership (who frees the shape? the motion state?)
  is a classic leak/UB source — its own demos leak.
- **Stable body indices**: `add_body` returns an index that survives
  other removals; `remove_body` cascades constraints and panics on
  double removal. Bullet hands out raw pointers and leaves constraint
  cleanup to the caller.
- **Precision**: f64 is a cargo feature (`--features f64`), not a
  library rebuild; tests and demos run in both modes from one source.
- **Soft bodies**: `create_patch` → `add_soft_body` without the
  `btSoftBodyWorldInfo` pitfalls (Bullet's world-info gravity does not
  follow `setGravity` — every Bullet soft-body demo must set it twice).
- **Determinism**: bit-exact parity against stock Bullet is the
  project's core selling point, exercised per commit by the parity
  harness (`tools/parity.py`).
- **Doc comments**: module docs carry usage patterns and the known
  pitfalls (e.g. `vehicle.rs` explains the action-after-step contract
  and the suspension-tower placement rule).

## 2. Friction points (prioritized backlog)

- **P1 — constraint setter double dispatch** (done 2026-09-02):
  `Constraint` now has delegating `point2point_mut() / hinge_mut() /
  fixed_mut() / cone_twist_mut() / slider_mut() / generic_6dof_mut()`
  accessors (one dispatch; panics name the found kind on a mismatch), so
  tuning code never touches `kind_mut()`. Immutable reads stay on
  `kind().hinge()` etc. — each kind's constructor owns that name on
  `Constraint`.
- **P2 — `RigidBody` damping** (done 2026-09-02):
  `body.set_damping(linear, angular)` (both factors clamped to `[0, 1]`,
  like Bullet's `setDamping`; defaults 0). The velocity-integration stage
  applies `v *= pow(1 - damping, dt)` before gravity/forces, matching
  `btRigidBody::applyDamping` in `predictUnconstraintMotion`. The opt-in
  "additional damping" block is not ported (`bullet-deviations.md` §3).
- **P3 — minimal debug draw** (medium): a debug-primitives output —
  `world.debug_primitives(&mut impl FnMut(DebugPrim))` — emitting AABBs,
  contact points, manifold normals and wheel rays, with a category mask
  mirroring Bullet's `DBG_*` flags. This is the *producer* role of
  Bullet's `btIDebugDraw`; the consumer stays caller-side, so a Bevy
  user bridges it to `Gizmos` (the immediate-mode line buffer) in a few
  lines while any other renderer implements the same closure. Bullet's
  OpenGL debug visualization is a big part of its tuning workflow;
  today we only have dumps and prints.
- **P4 — callbacks** (core done 2026-09-02): the Rapier event model,
  not Bullet's process-global function pointers — `world.contact_events()`
  drains `CollisionEvent::Started/Stopped { a, b, sensor }` per step
  (pair transitions; game code wants enter/exit, not a per-contact-point
  callback storm), with **per-body opt-in**
  (`body.collision_events(true)`) so quiet scenes pay nothing, and
  **trigger volumes** via `body.no_contact_response(true)` (Bullet's
  `CF_NO_CONTACT_RESPONSE`): detected and reported, never solved. Bullet's
  `setInternalTickCallback` needs no counterpart: callers own the step
  loop. Still open: a contact-force event with a magnitude threshold
  (post-release); per-pair solver hooks map onto the existing
  `no_collision_pairs` / filter system.
- **P5 — typed handles** (breaking; schedule before 1.0):
  `add_body` / `add_multibody` / `add_soft_body` all return `usize`, so
  a confused index compiles fine and fails at runtime (or silently).
  Newtype handles (`BodyHandle`, `MultiBodyHandle`, `SoftBodyHandle`)
  would catch the mistake at compile time.
- **P6 — the §3 feature surface** (partially done 2026-09-02): collision
  groups/masks (`body.collision_group` / `.collision_mask`, symmetric
  `group & mask` pair test on every pair + CCD sweep) and trigger
  volumes are in. Still open: all-hits ray tests, ghost-object overlap
  listings (the sensor flag + events + `contact_test` cover the main
  use case), serialization, motion states... see `bullet-deviations.md`
  §3 for the verified, categorized list.

## 3. Migration cheat-sheet (Bullet → this crate)

| Bullet | This crate | Note |
|---|---|---|
| `btDiscreteDynamicsWorld` + 4 helper allocations | `PhysicsWorld::new()` | |
| `world->setGravity(v)` | `world.set_gravity(v)` | |
| `new btRigidBody(info)` | `RigidBody::new_dynamic(shape, pos, mass)` | inertia computed (exact analytic for most shapes) |
| static body | `RigidBody::new_static(shape, pos)` | starts asleep, like Bullet |
| `addRigidBody` | `world.add_body(b)` | returns the stable body index |
| `stepSimulation(dt)` | `world.step_simulation(dt)` | |
| `setLinearVelocity(v)` | `world.body_mut(i).linear_velocity = v` | or the `.velocity(v)` builder before adding |
| `applyCentralImpulse(j)` | `body.apply_impulse(j, Vector3::ZERO)` | |
| `applyImpulse(j, rel)` | `body.apply_impulse(j, rel)` | |
| `setWorldTransform(...)` | `world.set_body_transform(i, pos, rot)` | |
| `removeRigidBody` | `world.remove_body(i)` | cascades constraints (§1 deviation) |
| `addConstraint` / `removeConstraint` | `world.add_constraint(c)` / `remove_constraint(i)` | |
| `p2p->setMaxImpulseClamp(x)` | `c.point2point_mut().set_impulse_clamp(x)` | one dispatch per kind |
| `body->setDamping(l, a)` | `body.set_damping(l, a)` | factors clamped to `[0, 1]` |
| collision groups / masks | `body.collision_group` / `.collision_mask` | symmetric `group & mask` pair test |
| trigger volume / sensor | `body.no_contact_response(true)` + `.collision_events(true)` | `world.contact_events()` drain |
| `rayTest` (all hits) | `world.raycast_closest(...)` | closest hit only (§3) |
| `contactTest` | `world.contact_test(...)` | returns manifolds, not a callback |
| `btRaycastVehicle` | `RaycastVehicle` | `update_vehicle` after `step_simulation` |
| `btKinematicCharacterController` | `KinematicCharacterController` | no ghost object needed |
| `btSoftBodyHelpers::CreatePatch` | `softbody::helpers::create_patch` | |
| `btMultiBody` | `MultiBody` | colliders attach to the shared world |

Re-verified against the codebase 2026-09-01.
