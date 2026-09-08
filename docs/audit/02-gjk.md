# Audit — GJK (distance walk + contact walk) (02)

- **Date**: 2026-09-08 (conclusions fixed across the 2026-09-07/08 hardening
  rounds; this record consolidates them — no re-audit performed)
- **HEAD at audit**: `9aebf38`
- **Scope**: `crates/bullet-collision/src/narrowphase/gjk.rs` —
  `support_minkowski`/`core_support_minkowski`, `gjk_intersect` /
  `gjk_intersect_seeded`, `gjk_contact` (the `GjkOutcome::Band`/`Overlap`
  router), the simplex state machine (`reduce_*`, `solve_*`,
  `enclosing_walk`, `shrink_or_normal`, `perpendicular`).
- **Reference**: `refs/bullet3/src/BulletCollision/NarrowPhaseCollision/
  btGjkEpa2.cpp` (GJK section: constants, termination, `Inside` gate),
  `btGjkPairDetector` (consumer contract).
- **Verdict**: ✅ **Audit-complete.** Two reproduced bugs found and fixed by
  the hardening rounds (scale guards `d876cfa`, strict-walk direction
  invariant `9aebf38`), both pinned by adversarial tests. Remaining
  absolute thresholds are classified as tolerance policy.

## 1. Reference decisions covered

| Decision point | Bullet reference | This crate | Match |
|---|---|---|---|
| Iteration cap | `GJK_MAX_ITERATIONS 128` | `MAX_ITERATIONS = 64` | ⚠️ half of Bullet's; the enclosing-walk router is a different formulation (strict walk with per-step support), the budget is tolerance policy — divergence exhausted at 64 steps is covered by the Band/Overlap tests |
| Termination accuracy | `GJK_ACCURACY` = 1e-12 (f64) / 1e-4 (f32) | projected-improvement bound `delta = v·w` vs `|v|²`, first-iteration `prev = ∞` with an `is_finite` guard | ✅ same family (relative bounds), our bound is the closer formulation |
| Touching/inside gate | `rl < GJK_MIN_DISTANCE` → `eStatus::Inside` (enters EPA) | `GjkOutcome::Overlap` (enters EPA); gate scaled by the first support's squared magnitude, floor 1 | ✅ after `d876cfa` (the unscaled gate was the reproduced bug §2.2) |
| Support transform | shape support with margins applied per shape | `support_minkowski` / `core_support_minkowski` (margin-core architecture, see `04-convex-convex.md`) | ✅ (margin handling audited in 04) |
| Simplex reduction | Voronoi region case split per dimension | `reduce_segment` / `reduce_triangle` / `reduce_tetrahedron`; degenerate determinant (< 1e-12) falls back to the best face and never reports `Inside` | ✅ + stricter: degenerate simplices cannot claim enclosure |
| Direction robustness | none explicit (relies on geometry) | loop-head invariant "direction finite and non-zero" in `enclosing_walk`; `perpendicular` is always-safe (zero/NaN deterministic fallback to `Z`) | ⚠️ stricter than Bullet (fix `9aebf38`, see §2.4) |
| Seeded re-walk | (Bullet has no seeded re-entry) | `gjk_intersect_seeded` — same walk from a warm simplex | ⚠️ extension (consumed by CCD/sweeps); equivalence pinned by the seeded-vs-unseeded tests |
| Constants kept as policy | `TOUCH_TOLERANCE2 = 1e-12`, `REL_ERROR2 = 1e-6`, coplanar gate 1e-12, degenerate-determinant gate 1e-24 | same values | ✅ tolerance policy — do not retune without a reproduced failure |

## 2. Reproduced bugs — found and fixed by the hardening rounds

1. **Strict-walk plane loop (`d876cfa`, fix 1)**: the origin-escape gate used
   an absolute `1e-24` on a squared quantity — under f32 cancellation noise
   (~1e-14 · span²) it never fired, so deeply-overlapping boxes (any seed)
   got stuck cycling a planar slice of the Minkowski difference until the
   iteration budget exhausted and reported `Separated`. Fix: gate
   relativized to `1e-12 × span²` (span = largest simplex vertex
   magnitude²).
2. **Large-scale touch gate (`d876cfa`, fix 2)**: the absolute `1e-12`
   touch tolerance stalled the walk at ~1.5e-5 residual for coordinates
   ~100, reporting `Band(−0.08)` for a depth-70 overlap instead of
   `Overlap` → EPA. Fix: gate multiplied by the first support's squared
   magnitude (floor 1 — unit scale is bit-identical).
3. **Raycast/convex-sweep exhausted-loop exit (`d876cfa`, fix 3)**: the
   budget-exhausted exit computed `v / v.length()` → 0/0 NaN; the exit now
   reuses the loop-head `1e-10` criterion (bit-identical when the criterion
   is false).
4. **Strict-walk direction invariant (`9aebf38`)**: simplex reductions can
   produce zero directions (`solve_segment`'s `dir = ao` when `a` collapses
   onto the origin; collinear-triangle face normals); the loop head now
   restarts from the newest support's perpendicular. `perpendicular`
   became an always-safe primitive (axis-retry + deterministic zero/NaN
   fallback to `Z` — the old helper produced NaN for short axis-aligned
   inputs). EPA seeding exit status fixed to `DegenerateExpansion`
   (Bullet's `InvalidHull` analogue) instead of the accidental
   `IterationLimit`.

## 3. Pinned degeneracy limits (tests; not bugs)

- Coincident point-based shapes (spheres/capsules): the contact walk
  reports `Separated` — the Minkowski difference degenerates to the origin
  point and no enclosing simplex can form. Unreachable in the world
  pipeline (sphere-sphere takes the analytic path); pinned by tests.
- Box/hull exact-touch: `gjk_intersect` returns `None` (origin on a face
  center + discrete corner supports cannot assemble an enclosing tetrahedron
  within budget); the pipeline's Band semantics covers touching
  (distance 0 < threshold).
- Small scales are margin-dominated: the box's fixed margin 0.04 dominates
  the band classification (`distance = centers − 0.08`, not `centers − s`).
  Bullet's `setSafeMargin` construction-time clamp is the related
  undocumented upstream behavior; our tests assert against the margin model,
  semantics unchanged.

## 4. Tests (adversarial suite, `d876cfa` + `9aebf38`)

17 tests: degenerate closest-point units, exact-touch and ±1e-5 boundary
routing, scale sweep 1e-3..1e3, seeded vs unseeded equivalence, degenerate
line-segment hulls, slender rods, perpendicular total-input-domain
finiteness/orthogonality, reduction-layer finite directions, end-to-end
loop-head invariant, box/hull duplicate-support classification.

## 5. Verdict

**Audit-complete.** The walk is a strict-enclosure formulation rather than
Bullet's exact `btGjkEpa2` GJK loop; the observable contract (Separated /
Band / Overlap routing into EPA, touching semantics, degeneracy limits) is
pinned by tests and the parity/probe corpus. Two reproduced bugs are fixed;
the remaining absolute thresholds are tolerance policy (§1) — do not retune
without a reproduced failure. The GJK→EPA contract (4-point simplex,
non-degenerate, origin enclosed) is asserted in `convex_convex` and detailed
in [`04-convex-convex.md`](04-convex-convex.md).
