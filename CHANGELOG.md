# Changelog

All notable changes to the `bullet-*` crates are documented in this file.
The crates share a single version line, in the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Full development
history lives in `docs/`.

## [0.2.0] - 2026-10-22

### Added

- New crate `bullet-inverse-dynamics`: tree-structured multibody inverse dynamics (builder, kinematics, Newton-Euler forces, CRBA mass matrix, jacobians), probed byte-level against the C++ library.
- Optional `inverse-dynamics` feature on `bullet-physics`, re-exporting the crate as `inverse_dynamics` (off by default).
- `bullet_math::CrtRand`, the portable MSVC `rand()` replica, is now public.
- Heightfield option surface: `up_axis`, `flip_quad_edges`, diamond/zigzag subdivision, `flip_triangle_winding`.

### Changed

- Heightfield (breaking): grids are centered on the local origin, matching upstream.

### Fixed

- quickhull: flat point clouds could fold into non-closed hulls; hulls are now always closed and contain every input point.

## [0.1.1] - 2026-10-19

### Changed

- Exact exponential-map orientation integration (was first-order quaternion step).
- Implicit gyroscopic impulse applied by default, matching upstream.
- Parity and performance ledgers re-recorded.

### Added

- MSRV 1.88 declared (`rust-version`), enforced by cargo and shown on crates.io.

## [0.1.0] - 2026-10-16

### Added

- Initial public release: pure Rust + glam reimplementation of the Bullet Physics core.
- Shapes: sphere, box, capsule, cylinder, cone, convex hull, compound, plane, triangle mesh, heightfield, GImpact.
- Narrow phase: GJK + EPA, persistent manifolds, filtered rays/sweeps.
- Solver: sequential impulse, warm/split impulse, islands + sleeping, CCD.
- Constraints: point2point, hinge, fixed, cone twist, slider, 6DOF, 6DOF spring-2.
- Featherstone multibody; soft bodies (ropes, cloth, clusters, cutting).
- Raycast vehicle and kinematic character controller.
- Sensors, collision/force events, kinematic bodies, motion states, debug draw, step accumulator.
- `f32`/`f64` precision; `no_std` via `libm`.
