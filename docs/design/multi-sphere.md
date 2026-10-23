# Multi-sphere shape (btMultiSphereShape)

Status: approved for implementation (2026-10-23, freeze-review row
"`btMultiSphereShape`").

## Upstream inventory

`BulletCollision/CollisionShapes/btMultiSphereShape.{h,cpp}` — the convex
hull of a sphere collection (support of the union = support of the hull):

- Support (`localGetSupportingVertexWithoutMargin`): normalize `vec` (zero
  direction falls back to `+x` when `len² < SIMD_EPSILON²`), then per sphere
  `temp_k = pos_k·s + vec·s·r_k − vec·margin`, pick `argmax vec·temp_k`.
  The margin is subtracted **inside** the without-margin function, so the
  with-margin support (= core + `vec·margin`, added by the caller) is
  exactly `pos_k + vec·s·r_k` — the margin cancels and the observable
  surface is the pure sphere union. The core (what GJK walks) is the
  spheres shrunk by the 0.04 `CONVEX_DISTANCE_MARGIN`.
- Tie-break: `btVector3::maxDot` scalar path uses strict `>` → the first
  maximal sphere wins (btVector3.h:1018).
- `calculateLocalInertia`: **box approximation** from the cached local AABB
  (`mass/12·(ly²+lz²)` triple — the api-priorities row's "analytic inertia"
  was wrong; the source is explicit: "as an approximation, take the inertia
  of the box that bounds the spheres").
- Cached local AABB (identity transform, `getAabbSlow`): per axis,
  with-margin support (the pure sphere bound after cancellation) then
  `± margin` again → the box is the sphere bounds expanded by 0.04 per
  side. That same box feeds the inertia.
- No specialized narrowphase/raycast anywhere — every query goes through
  the generic convex paths (upstream same). `recalcLocalAabb` after
  mutation is the caller's job (we have no mutation API — same shape-value
  semantics as every other shape).

## Port shape

- `shapes/multi_sphere.rs`:
  `MultiSphere { spheres: Vec<(Vector3, Real)> }`,
  `MultiSphere::new(&[(Vector3, Real)])` (panics on empty input — upstream
  would produce a garbage support; an empty sphere set has no surface),
  `CollisionShape::multi_sphere(..)` constructor.
- `margin()` = `0.04` (`CONVEX_DISTANCE_MARGIN`, the convex default).
- `support_local(dir)`: normalize + zero-dir fallback, then the first-max
  sphere with `pos + dir·r` (the cancellation folded — bit-identical to
  upstream's core+margin round trip for any `|dir|`, and it keeps the
  hull-style support contract of returning the surface point).
- `core_support_local(dir)`: same selection on `pos + dir·(r − margin)`
  (upstream's without-margin verbatim). Note this is the *opposite*
  margin stance from our hulls (margin-free, recorded deviation) — here
  the upstream core genuinely shrinks, and replicating it keeps the
  GJK band behavior exact instead of opening a new deviation.
- `local_aabb()`: per-axis `max_k(±dir·pos_k + r_k)` expanded by `margin`
  (upstream's cached box). `local_inertia`: the upstream box triple on
  that AABB's half extents.
- `is_convex()` = true; `ShapeType::MultiSphere`; dispatch arms in
  `support_local` / `core_support_local` / `local_aabb` / `local_inertia` /
  `margin` / `is_convex` / `shape_type`. Raycast, GJK/EPA contacts and
  sweeps pick the shape up through the generic convex arms (verify the
  raycast dispatch table during implementation; add the arm if it is
  type-switched).

## Reference decisions covered

| Upstream decision | Port |
|---|---|
| support = max over spheres, first-max tie (strict `>`) | same iteration order, strict `>` |
| zero direction → `+x` fallback (`SIMD_EPSILON²` gate) | same gate |
| margin subtracted inside without-margin support | core = `r − margin` per sphere |
| with-margin support = pure sphere surface (cancellation) | `support_local` returns `pos + dir·r` |
| inertia = box of the cached AABB (explicit approximation) | same formula, same box |
| cached AABB = sphere bounds ± margin | `local_aabb` per-axis + margin |
| no specialized narrowphase/raycast/sweep | generic convex arms only |
| caller-owned `recalcLocalAabb` after mutation | immutable shape value (blanket) |
| serialization | blanket deviation |

## Tests

- Support: two spheres `±x` — `+x` direction hits the far sphere's
  `pos + r·x̂`; diagonal direction picks the correct sphere (hand-computed).
- Tie: two spheres equidistant along `y` → first wins (binary-exact
  spacing per the f32 tie discipline).
- Zero direction → `+x` fallback point.
- Margin: core support is `margin` short of the surface along the ray
  through a sphere center; full support lands exactly on the sphere.
- Inertia: matches `mass/12` box formula on the expanded AABB (hand calc);
  AABB includes the margin.
- Contacts smoke: multi-sphere body dropped on a plane settles; GJK
  penetration vs a hull resolves (generic-arm integration).
- `shape_type` / `is_convex` / accessors.

## Verification

- Full gate + parity anchors bit-identical (additive shape, no existing
  dispatch touched).
- Probe: not planned — no upstream specialization exists to compare
  against beyond the support math the unit tests pin; the generic convex
  pipeline is already probed end-to-end by the existing rigs.
