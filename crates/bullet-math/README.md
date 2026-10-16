# bullet-math

The linear-math layer of
[bullet-physics](https://github.com/CrazyboyQCD/bullet-physics), a pure Rust
reimplementation of Bullet Physics (zlib).

Precision-swappable `glam` aliases (`Real`, `Vector3`, `Quaternion`, `Matrix3`,
`Transform`), an `Aabb`, inertia helpers, a small incremental quickhull, and the
scalar-math routing that keeps the workspace `no_std`-clean.

## Features

- `f32` (default) or `f64` — the same API at either precision.
- `std` (default) or `libm` — `libm` routes transcendentals through the pure-Rust
  crate for `no_std` targets.

## Scope

This crate is an implementation detail of the workspace rather than a
standalone math library: the engine's public API, the frame-by-frame parity
ledger against upstream Bullet and the performance ledgers live in the
[workspace README](https://github.com/CrazyboyQCD/bullet-physics#readme) and in
`docs/`.
