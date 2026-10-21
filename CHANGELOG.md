# Changelog

All notable changes to the `bullet-*` crates are documented in this file.
The four crates share a single version line, in the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Full development
history lives in `docs/`.

## [Unreleased]

### Added

- Heightfield: the `btHeightfieldTerrainShape` option surface —
  `HeightfieldOptions` with `up_axis` (X/Y/Z), `flip_quad_edges`,
  `use_diamond_subdivision`, `use_zigzag_subdivision` and
  `flip_triangle_winding`; parity scenes `heightfield` / `heightfield_x`.

### Changed

- Heightfield (breaking): the grid is centered on the local origin like
  upstream — in-plane coordinates shift by `-(n-1)/2 * cell` and heights
  are offset by the min/max midpoint, so the terrain AABB centers on the
  body transform.

### Fixed

- quickhull: flat point clouds could produce a hull with missing faces (a
  vertex could drive repeated expansions and figure-eight horizons could
  fold fans onto one edge); the output is now always a closed surface
  containing every input point.

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
