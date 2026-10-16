# bullet-physics

A pure Rust reimplementation of
[Bullet Physics](https://github.com/bulletphysics/bullet3) (zlib): rigid bodies,
collision detection, constraints, multibody dynamics and soft bodies, built on
`glam` with SIMD. No C/C++ toolchain, no FFI.

This is the facade crate: it re-exports `bullet-math`, `bullet-collision` and
`bullet-dynamics`, plus a `prelude` for the common imports.

```rust
use bullet_physics::dynamics::{PhysicsWorld, RigidBody};
use bullet_physics::math::Vector3;
use bullet_physics::shapes::CollisionShape;

let mut world = PhysicsWorld::new();
world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

// Static ground plane, then a 1 m cube dropped from 5 m.
world.add_body(RigidBody::new_static(
    CollisionShape::box_shape(Vector3::new(25.0, 0.5, 25.0)),
    Vector3::new(0.0, -0.5, 0.0),
));
world.add_body(RigidBody::new_dynamic(
    CollisionShape::box_shape(Vector3::splat(0.5)),
    Vector3::new(0.0, 5.0, 0.0),
    1.0,
));

for _ in 0..300 {
    world.step_simulation(1.0 / 60.0);
}
```

## Features

- `f32` (default) or `f64` — matching Bullet's `BT_USE_DOUBLE_PRECISION` build.
- `std` (default) or `libm` for `no_std` targets.

## Evidence

The port is validated against the real C++ library rather than only against its
own tests: frame-by-frame parity, wall-clock and memory ledgers, and
per-subsystem adversarial audits — all linked from the
[workspace README](https://github.com/CrazyboyQCD/bullet-physics#readme).
