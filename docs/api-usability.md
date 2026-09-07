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
- **Typed handles**: `add_body` / `add_multibody` / `add_soft_body` return
  `BodyHandle` / `MultiBodyHandle` / `SoftBodyHandle` (2026-09-03), so the
  three index spaces cannot be mixed up at the call site; `remove_body`
  cascades constraints and panics on double removal. Bullet hands out raw
  pointers and leaves constraint cleanup to the caller.
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
- **P4 — callbacks** (core done 2026-09-02; sides generalized 2026-09-07):
  the Rapier event model, not Bullet's process-global function pointers —
  `world.contact_events()` drains `CollisionEvent::Started/Stopped { a, b,
  sensor }` per step (pair transitions; game code wants enter/exit, not a
  per-contact-point callback storm), with **per-body opt-in**
  (`body.collision_events(true)`) so quiet scenes pay nothing, and
  **trigger volumes** via `body.no_contact_response(true)` (Bullet's
  `CF_NO_CONTACT_RESPONSE`): detected and reported, never solved. Since
  2026-09-07 the `a`/`b` sides are the `CollisionObject` enum (`Body` /
  `Link { body, link }` / `Soft`) — a breaking change covering multibody
  colliders and soft bodies as event sources. Bullet's
  `setInternalTickCallback` needs no counterpart: callers own the step
  loop. Still open: a contact-force event with a magnitude threshold
  (post-release); per-pair solver hooks map onto the existing
  `no_collision_pairs` / filter system.
- **P5 — typed handles** (done 2026-09-03, breaking):
  `add_body` / `add_multibody` / `add_soft_body` return the newtypes
  `BodyHandle` / `MultiBodyHandle` / `SoftBodyHandle` (`NonZeroU32`), so a
  confused index no longer compiles; constraint endpoints, events, filters,
  ray/sweep returns and soft anchors are typed too. Migration is mechanical:
  `handle.index()` at any remaining raw-slot site.
- **P6 — the §3 feature surface** (mostly done; last batch 2026-09-07):
  collision groups/masks, trigger volumes, all-hits rays + ray filters,
  per-body gravity, the pair filter, typed handles, ghost overlap listing,
  kinematic rigid bodies (2026-09-04), rolling friction, compound child
  transforms, motion states and multibody/soft-body collision-event sources
  (2026-09-07) are in. Still open: serialization and the remaining property
  surface (anisotropic friction, contact stiffness/damping) — see
  `bullet-deviations.md` §3 for the verified, categorized list.

## 3. Migration cheat-sheet (Bullet → this crate)

| Bullet | This crate | Note |
|---|---|---|
| `btDiscreteDynamicsWorld` + 4 helper allocations | `PhysicsWorld::new()` | |
| `world->setGravity(v)` | `world.set_gravity(v)` | |
| `new btRigidBody(info)` | `RigidBody::new_dynamic(shape, pos, mass)` | inertia computed (exact analytic for most shapes) |
| static body | `RigidBody::new_static(shape, pos)` | starts asleep, like Bullet |
| `addRigidBody` | `world.add_body(b)` | returns a typed `BodyHandle` |
| `stepSimulation(dt)` | `world.step_simulation(dt)` | |
| `setLinearVelocity(v)` | `world.body_mut(h).linear_velocity = v` | or the `.velocity(v)` builder before adding |
| `applyCentralImpulse(j)` | `body.apply_impulse(j, Vector3::ZERO)` | |
| `applyImpulse(j, rel)` | `body.apply_impulse(j, rel)` | |
| `setWorldTransform(...)` | `world.set_body_transform(h, pos, rot)` | |
| `removeRigidBody` | `world.remove_body(h)` | cascades constraints (§1 deviation) |
| `addConstraint` / `removeConstraint` | `world.add_constraint(c)` / `remove_constraint(i)` | |
| `p2p->setMaxImpulseClamp(x)` | `c.point2point_mut().set_impulse_clamp(x)` | one dispatch per kind |
| `body->setDamping(l, a)` | `body.set_damping(l, a)` | factors clamped to `[0, 1]` |
| `body->setMotionState(ms)` | `body.motion_state(ms)` (builder) / `set_motion_state(ms)` | `MotionState::initial_pose` read at `add_body`, `synchronize` after each step |
| collision groups / masks | `body.collision_group` / `.collision_mask` | symmetric `group & mask` pair test |
| trigger volume / sensor | `body.no_contact_response(true)` + `.collision_events(true)` | `world.contact_events()` drain |
| `btGhostObject` overlap list | `world.overlapping(handle)` | persistent touch list; narrow-phase level (Bullet's is AABB level) |
| `body->setRollingFriction(x)` | `body.rolling_friction = x` | torsional rows per contact; combined `rf_a·μ_b + rf_b·μ_a` (Bullet quirk) |
| `rayTest` (closest) | `world.raycast_closest(...)` | |
| `rayTest` (all hits) | `world.raycast_all(...)` / `raycast_all_filtered(...)` | filter = the ray's own group/mask |
| `RayResultCallback` filter | `RayFilter { group, mask }` | symmetric pair test, default group 1 / mask all |
| `body->setGravity(v)` | `body.set_gravity(v)` | add-time sync + awake-only propagation; `DISABLE_WORLD_GRAVITY` opts out |
| `btOverlapFilterCallback` | `world.set_pair_filter(...)` | rigid pairs + CCD sweep, on top of group/mask |
| `contactTest` | `world.contact_test(...)` | returns manifolds, not a callback |
| `btRaycastVehicle` | `RaycastVehicle` | `update_vehicle` after `step_simulation` |
| `btKinematicCharacterController` | `KinematicCharacterController` | no ghost object needed |
| `btSoftBodyHelpers::CreatePatch` | `softbody::helpers::create_patch` | |
| `btMultiBody` | `MultiBody` | colliders attach to the shared world |

Re-verified against the codebase 2026-09-07 (typed-handle migration
reflected throughout).
