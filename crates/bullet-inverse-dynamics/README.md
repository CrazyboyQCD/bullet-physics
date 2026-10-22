# bullet-inverse-dynamics

Inverse dynamics for tree-structured multibody systems — the Rust port of
Bullet's `BulletInverseDynamics` (`btInverseDynamics::MultiBodyTree`), part of
the [bullet-physics](https://github.com/CrazyboyQCD/bullet-physics) workspace
(a pure Rust reimplementation of Bullet Physics, zlib).

Given `(q, u, u̇)` it computes the generalized joint forces that realize them,
plus the joint-space mass matrix (composite rigid body algorithm), forward
kinematics at three levels and per-body 3×N world-frame jacobians — for trees
of fixed/revolute/prismatic/floating/spherical joints.

## Features

- `f32` (default) or `f64` — the same API at either precision.
- `std` (default) or `libm` — `libm` routes transcendentals through the
  pure-Rust libm crate for `no_std` targets.
- Zero heap allocation in any `calculate_*`/getter after `finalize`.
- Depends only on
  [`bullet-math`](https://crates.io/crates/bullet-math); the
  [`bullet-physics`](https://crates.io/crates/bullet-physics) facade
  re-exports the crate behind its optional `inverse-dynamics` feature (off by
  default).

## Usage

```rust
use bullet_inverse_dynamics::builder::{JointType, TreeBuilder};
use bullet_inverse_dynamics::{Matrix3, Vector3};

// Fixed root + one Z-revolute link whose joint sits one unit along +x.
let mut b = TreeBuilder::new();
b.add_body(
    0, -1, JointType::Fixed, Vector3::ZERO, Matrix3::IDENTITY,
    Vector3::ZERO, 0.0, Vector3::ZERO, Matrix3::IDENTITY,
)
.unwrap();
b.add_body(
    1, 0, JointType::Revolute, Vector3::new(1.0, 0.0, 0.0), Matrix3::IDENTITY,
    Vector3::Z, 1.0, Vector3::new(0.5, 0.0, 0.0), Matrix3::IDENTITY,
)
.unwrap();
let mut tree = b.finalize().unwrap();
tree.set_gravity_in_world_frame(Vector3::new(0.0, -9.8, 0.0));

// Generalized forces holding the arm horizontal at rest: the joint carries
// the gravity moment of the link in front of it.
let q = [0.0];
let mut tau = [0.0; 1];
tree.calculate_inverse_dynamics(&q, &[0.0], &[0.0], &mut tau)
    .unwrap();

// Joint-space mass matrix at the same pose (1x1, row-major).
let mut mm = [0.0; 1];
tree.calculate_mass_matrix(&q, &mut mm).unwrap();
```

## Scope

The crate is verified against the real C++ library, not only against its own
tests; the parity ledger, the bench/allocation numbers and the recorded
deviations live in the
[workspace repo](https://github.com/CrazyboyQCD/bullet-physics)
(`docs/bullet-compare.md`, `docs/perf/inverse-dynamics.md`,
`docs/bullet-deviations.md`).
