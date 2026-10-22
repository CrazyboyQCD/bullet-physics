# bullet-physics

A pure Rust physics engine reimplementing the core of
[Bullet Physics](https://github.com/bulletphysics/bullet3).

Requires Rust **1.88** or newer (declared as `rust-version` in `Cargo.toml`;
the code uses let-chains, which edition 2024 stabilized in 1.88). See
[CHANGELOG.md](CHANGELOG.md) for release notes.

## Workspace layout

The project is a Cargo workspace of focused crates, mirroring Bullet's official
library boundaries:

| Crate             | Description                                                        | Bullet equivalent |
| ----------------- | ------------------------------------------------------------------ | ----------------- |
| `bullet-math`     | SIMD-accelerated linear math, precision-swappable glam aliases     | `LinearMath`      |
| `bullet-collision`| Shapes, broad phase, narrow phase, contact manifolds               | `BulletCollision` |
| `bullet-dynamics` | Rigid bodies, simulation world, constraint solvers                 | `BulletDynamics`  |
| `bullet-physics`  | Facade crate re-exporting the full API                             | —                 |
| `bullet-inverse-dynamics` | Tree-structured multibody inverse dynamics (optional feature) | `BulletInverseDynamics` |

See [docs/design/architecture.md](docs/design/architecture.md) for the detailed design.

## Precision features

- `f32` (default): single-precision scalar and SIMD math types.
- `f64`: double-precision math types (matches Bullet's default
  `BT_USE_DOUBLE_PRECISION`).

```toml
bullet-physics = { path = "crates/bullet-physics", features = ["f64"] }
```

## no_std

The workspace compiles without the standard library: enable the `libm`
feature (routes scalar transcendentals and glam's internal math to the
pure-Rust `libm` crate) and drop the default features:

```toml
[dependencies]
bullet-physics = { path = "crates/bullet-physics", default-features = false, features = ["libm"] }
```

- Default/std builds are bit-identical to always; only `libm` builds may
  drift from the platform CRT at the last ulp of `sin`/`cos`/`powf`-style
  calls (recorded in `docs/bullet-deviations.md`).
- `StepTimings` reads zero unless a monotonic clock is injected via
  `PhysicsWorld::set_step_clock` (under `std` the standard monotonic
  `Instant` is used automatically).
- SIMD is target-feature driven as usual (SSE2/NEON); on `wasm32` add
  `-C target-feature=+simd128` for `Vec3A` SIMD lanes.

## Quick start

```bash
# Build the whole workspace
cargo build

# Run the tests (both precision modes)
cargo test
cargo test --features f64

# Run the falling-box demo
cargo run -p bullet-physics --example falling_box
```

## Examples

Worked snippets for every subsystem — rigid bodies, kinematic platforms,
joints, Featherstone multibodies, soft bodies, raycast vehicles, the
character controller, heightfields and inverse dynamics — live in
[docs/examples.md](docs/examples.md). Runnable demos are under
`crates/bullet-physics/examples/`:

```bash
cargo run -p bullet-physics --example falling_box
cargo run -p bullet-physics --example joints
```

## Correctness and performance evidence

The port is checked against the real C++ library, not only against its own tests:

- **Frame-by-frame parity.** This crate and upstream Bullet build the same scenes
  from the same generated tables and are diffed frame by frame in both `f32` and
  `f64`. The numbers, and the classification of every exceedance, are in
  [`docs/bullet-compare.md`](docs/bullet-compare.md); each intentional divergence
  from upstream is recorded in [`docs/bullet-deviations.md`](docs/bullet-deviations.md).
- **Wall-clock and memory ledgers.** The same reference binary is timed scene by
  scene in [`docs/perf/speed.md`](docs/perf/speed.md), with per-scene allocator
  curves in [`docs/perf/memory.md`](docs/perf/memory.md).
- **Subsystem audits.** Every numerically sensitive area — manifold, GJK, EPA,
  convex-convex, the axis-sweep and DBVT broad phases, CCD, and an integrated
  multi-system pass — has an adversarial audit record under
  [`docs/audit/`](docs/audit/README.md).
- **Reproducing it.** [`docs/parity-harness.md`](docs/parity-harness.md)
  documents the C++ reference build (CMake + MSVC + a pinned `refs/bullet3`
  clone); `python tools/check.py` runs the Rust-only gate (test matrix plus demo
  assertions) with no C++ toolchain.

## License

Licensed under the [zlib license](LICENSE). This project is a reimplementation
of Bullet Physics, which is also zlib licensed.
