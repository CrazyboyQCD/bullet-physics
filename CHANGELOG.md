# Changelog

All notable changes to the `bullet-*` crates are documented in this file.
The four crates share a single version line, in the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Full development
history lives in `docs/`.

## [Unreleased]

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
