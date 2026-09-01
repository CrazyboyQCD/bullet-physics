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

- **P1 — constraint setter double dispatch** (small fix):
  `joint.kind_mut().point2point_mut().set_impulse_clamp(0.001)` where
  Bullet writes `p2p->setMaxImpulseClamp(...)`. Add delegating
  `Constraint::point2point() / point2point_mut() / hinge() / ...`
  accessors (panic with a clear message on the wrong kind) so user code
  never touches `kind_mut()`.
- **P2 — `RigidBody` damping** (small fix): linear/angular damping is a
  common Bullet API (`setDamping`, defaults 0). Port `applyDamping`
  (`v *= pow(1 - damping, dt)` inside `btRigidBody::applyDamping`) into
  the velocity-integration stage. Missing entirely today
  (`bullet-deviations.md` §3, rigid-body property surface).
- **P3 — minimal debug draw** (medium): a debug-primitives output —
  `world.debug_primitives(&mut impl FnMut(DebugPrim))` — emitting AABBs,
  contact points, manifold normals and wheel rays, with a category mask
  mirroring Bullet's `DBG_*` flags. This is the *producer* role of
  Bullet's `btIDebugDraw`; the consumer stays caller-side, so a Bevy
  user bridges it to `Gizmos` (the immediate-mode line buffer) in a few
  lines while any other renderer implements the same closure. Bullet's
  OpenGL debug visualization is a big part of its tuning workflow;
  today we only have dumps and prints.
- **P4 — callbacks** (larger, post-release): contact-added,
  contact-processed and tick callbacks are how Bullet games wire
  damage/sound/particles. Design note: prefer an explicit post-step
  event drain (`contact_events() -> &[...]`) or a user-supplied trait
  object over Bullet's process-global function pointers.
- **P5 — typed handles** (breaking; schedule before 1.0):
  `add_body` / `add_multibody` / `add_soft_body` all return `usize`, so
  a confused index compiles fine and fails at runtime (or silently).
  Newtype handles (`BodyHandle`, `MultiBodyHandle`, `SoftBodyHandle`)
  would catch the mistake at compile time.
- **P6 — the §3 feature surface**: collision groups/masks, all-hits ray
  tests, ghost objects, serialization, motion states... see
  `bullet-deviations.md` §3 for the verified, categorized list.

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
| `p2p->setMaxImpulseClamp(x)` | `c.kind_mut().point2point_mut().set_impulse_clamp(x)` | backlog P1 |
| `body->setDamping(l, a)` | — MISSING | backlog P2 |
| collision groups / masks | — MISSING | §3 |
| `rayTest` (all hits) | `world.raycast_closest(...)` | closest hit only (§3) |
| `contactTest` | `world.contact_test(...)` | returns manifolds, not a callback |
| `btRaycastVehicle` | `RaycastVehicle` | `update_vehicle` after `step_simulation` |
| `btKinematicCharacterController` | `KinematicCharacterController` | no ghost object needed |
| `btSoftBodyHelpers::CreatePatch` | `softbody::helpers::create_patch` | |
| `btMultiBody` | `MultiBody` | colliders attach to the shared world |

Re-verified against the codebase 2026-09-01.
