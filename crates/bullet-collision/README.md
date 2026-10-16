# bullet-collision

Collision detection for
[bullet-physics](https://github.com/CrazyboyQCD/bullet-physics), a pure Rust
reimplementation of Bullet Physics (zlib).

## What's inside

- Shapes: sphere, box, capsule, cylinder, cone, convex hull, compound, static
  triangle mesh, heightfield, and GImpact for dynamic concave meshes.
- Broad phase: three backends — brute force, incremental axis sweep, and a DBVT.
- Narrow phase: GJK distance walking with a margin band, EPA penetration, plus
  analytic and SAT paths; persistent contact manifolds with warm-start data.
- Queries: ray casts (closest and all-hits), convex sweeps, and contact tests.

## Features

- `f32` (default) or `f64`.
- `std` (default) or `libm` for `no_std` targets.

The engine's public API, the parity ledger against upstream Bullet and the
performance ledgers live in the
[workspace README](https://github.com/CrazyboyQCD/bullet-physics#readme) and in
`docs/`.
