# Changelog

All notable changes to the `bullet-*` crates are documented in this file.
The four crates share a single version line. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The full development history (per-scenario parity ledgers, audit notes,
performance analyses) lives in `docs/`; this file only records what a
consumer of the published crates should know.

## [Unreleased]

## [0.1.1] - 2026-10-19

### Changed

- Rigid body orientation integration now uses the exact exponential-map
  update (mirroring `btTransformUtil::integrateTransform`) instead of a
  first-order quaternion step. Tumbling bodies conserve angular momentum
  noticeably better; collapse transients are physically heavier while
  sleeping steady states are unchanged or cheaper.
- An implicit gyroscopic impulse — upstream's default
  (`BT_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_BODY`) — is now applied after
  velocity integration, so freely spinning bodies no longer gain/lose
  angular momentum from the old Euler integration.
- The parity and performance ledgers were re-recorded for the above
  (`docs/bullet-compare.md`, `docs/perf/speed.md`). Non-rotating scenarios
  are bit-identical; rotating scenarios stay within their previously
  classified divergence categories.

### Added

- Declared MSRV **1.88** via `rust-version` in every crate manifest. Cargo
  now enforces it (older toolchains get a clear resolution error) and
  crates.io surfaces it. The code base uses let-chains, which edition 2024
  stabilized in Rust 1.88.

## [0.1.0] - 2026-10-16

### Added

- Initial public release: a pure Rust, `glam`-SIMD-based reimplementation of
  the Bullet Physics core, ported against the upstream C++ source with a
  bit-level parity harness (`refs/bullet3` record/replay, 18 scenes, f32+f64
  ledgers) and performance ledgers against MSVC Bullet.
- Rigid body pipeline: 10 analytic-inertia convex shapes (sphere, box,
  capsule, cylinder, cone, convex hull, compound, plane) plus triangle mesh
  (static BVH), heightfield, and GImpact dynamic concave meshes; collision
  groups/masks, sensors and collision events, per-body gravity, kinematic
  bodies, motion states with interpolation.
- Narrow phase: GJK distance walking with margin-band architecture, EPA
  (`btGjkEpa2`-faithful), analytic/SAT fast paths, persistent manifolds with
  warm starting, raycasting and swept queries with filters.
- Solver: sequential impulse with warm starting and split impulse,
  Baumgarte/slop stabilization, island-based sleeping with union-find,
  CCD sphere sweeps, solver info (ERP/CFM/SOR/iterations), rolling and
  spinning friction, anisotropic friction, contact stiffness/damping,
  internal-edge correction utility.
- Constraints: point2point, hinge, fixed, cone twist, slider, generic 6DOF,
  and generic 6DOF spring-2 (the PMX/MMD joint constraint), plus servo and
  motor configurations.
- Multibody: full Featherstone ABA dynamics with the constraint set
  (point2point, fixed, slider, motors, gear, limits), rigid-multibody and
  softbody coupling.
- Soft bodies: the classic A-F feature set — ropes, cloth, tetrahedra,
  K-mean clusters with quickhull convex hulls, CL_RS/CL_SS/CL_SELF
  coupling, joints, cutting and refinement.
- Vehicles (`btRaycastVehicle` formulas) and kinematic character controller.
- World API: typed handles, accumulator-based `step_simulation_fixed` with
  motion-state interpolation, contact/force events, ghost overlap lists,
  pair filters, minimal read-only debug draw.
- Precision and platform: `f32` (default) and `f64` features; `no_std`
  support with a pure-Rust `libm` math backend behind the `libm` feature.
