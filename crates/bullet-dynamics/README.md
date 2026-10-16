# bullet-dynamics

Rigid body dynamics for
[bullet-physics](https://github.com/CrazyboyQCD/bullet-physics), a pure Rust
reimplementation of Bullet Physics (zlib).

## What's inside

- `RigidBody` and `PhysicsWorld`: forces, damping, sleeping with simulation
  islands, kinematic bodies, CCD, collision groups, sensors and contact events.
- Sequential-impulse solver with warm starting, split impulse, rolling and
  spinning friction, contact stiffness/damping and per-pair solver tuning.
- All six joint kinds plus the generic 6DOF spring-2 constraint.
- The Featherstone multibody system, the classic soft-body system (ropes, cloth,
  volumes, clusters), a raycast vehicle and a kinematic character controller.

## Features

- `f32` (default) or `f64`.
- `std` (default) or `libm` for `no_std` targets.

The engine's public API, the parity ledger against upstream Bullet and the
performance ledgers live in the
[workspace README](https://github.com/CrazyboyQCD/bullet-physics#readme) and in
`docs/`.
