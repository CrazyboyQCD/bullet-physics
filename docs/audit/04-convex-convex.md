# Audit — Convex-Convex (generic GJK/EPA pair path) (04)

- **Date**: 2026-09-08 (conclusions fixed in the 2026-09-07 hardening round;
  this record consolidates them — no re-audit performed)
- **HEAD at audit**: `9aebf38`
- **Scope**: `crates/bullet-collision/src/narrowphase/convex_convex.rs` —
  the generic convex-vs-convex pipeline (`convex_convex`): margin-band
  routing, the GJK→EPA contract, per-status EPA result acceptance and the
  contact witness construction. The analytic fast paths (sphere, box-box
  SAT) live in their own modules and their own audit slots.
- **Reference**: `refs/bullet3/src/BulletCollision/NarrowPhaseCollision/
  btGjkEpa2.cpp` (`btGjkEpaSolver2::Penetration`, :973-1004 — witness
  construction and status acceptance),
  `btGjkPairDetector` (margin/band conventions).
- **Verdict**: ✅ **Audit-complete.** The GJK Overlap contract is proven and
  `debug_assert`-pinned; EPA acceptance is mirrored from the reference
  source; the witness construction is a documented behavioral deviation
  (`docs/bullet-deviations.md` §1 "Convex-convex contact witness"): our
  single-point A-witness is structurally equivalent to Bullet's barycentric
  interpolation but not bit-equal, and the B-side witness is not exactly
  Bullet's for arbitrary generic convex pairs.

## 1. Reference decisions covered

| Decision point | Bullet reference | This crate | Match |
|---|---|---|---|
| Margin architecture | shapes carry margins inside their support (`getSupport` with margin) | two-tier support: `core_support_minkowski` (margin-core) and `support_minkowski` (with margin); the band router decides which tier feeds GJK | ✅ equivalent contract (audited in `02-gjk.md`) |
| Separated routing | GJK distance > margin ⇒ no contact | `GjkOutcome::Separated` ⇒ no contact | ✅ |
| Touching band | distance within [0, margin band] ⇒ contact with distance | `GjkOutcome::Band(distance)` ⇒ contact carrying the band distance | ✅ |
| Overlap routing | GJK `Inside` ⇒ EPA | `GjkOutcome::Overlap` ⇒ EPA (`penetration_with_status`) | ✅ |
| EPA status acceptance | `Penetration` (:973-1004): every non-`Failed` status is **accepted** (iteration exhaustion keeps `Valid`; `InvalidHull`, `OutOfVertices` still yield the best-face result); only the seed `FallBack` yields zero depth; `Failed` unreachable | all non-fatal statuses accepted; status does **not** filter contacts (filtering would deviate from Bullet); zero depth filtered by `core_depth <= 0 → None` | ✅ |
| GJK→EPA contract | implicit (Bullet trusts its simplex) | proven from the code paths and pinned by `debug_assert` in `convex_convex`: exactly 4 points (`ReduceStep::Inside` hardcodes `len: 4` + the degenerate gate's explicit `len() == 4`), structurally non-degenerate (`reduce_tetrahedron`'s < 1e-12 determinant falls back to the best face and never reports `Inside`), origin strictly enclosed (`Inside` / `solve_tetrahedron` = exact 4-face sign tests including boundary touching; `solve_segment`/`solve_triangle` never return true) | ✅ proven + asserted |
| Witness on B | `witnesses[1] = wtrs0 * (w0 - normal*depth)`, `w0` = barycentric interpolation of the **A-side** supports over the best-face's three vertices | `wa = supportA(n) - n*depth` (single A-support point), then the world-transform composition; both land on B's penetrating support plane (`n·position == n·supportB(-n)` — depth is carried by this) and inherit A's tangential coordinate only through the point choice | ⚠️ deviation (§3): same construction family, different A-point choice |
| Normal convention | Bullet's `m_normalWorldOnB` is B→A (`results.normal = -epa.m_normal`) | contact normal A→B; row value `(B−A)·dir`; bias takes the negated `m_constraintError` equivalent | ✅ sign-translated, documented convention |
| Result filtering | `Penetration` returns false only for GJK-`Separated`-style failures | `core_depth <= 0` filters the zero-depth seed fallback | ✅ |

## 2. The proven + pinned GJK Overlap contract

`gjk_contact`'s `Overlap` outcome hands EPA a simplex that is, provably:

1. **Exactly 4 points** — the `ReduceStep::Inside` branch hardcodes
   `len: 4`, and the degenerate gate explicitly checks `len() == 4`
   before claiming enclosure.
2. **Structurally non-degenerate** — `reduce_tetrahedron` classifies
   coplanar/duplicate configurations by determinant (< 1e-12) and falls
   back to the best face; `Inside` is unreachable for them.
3. **Origin-enclosing** — `Inside` and `solve_tetrahedron`'s `true` are
   exact 4-face sign tests (boundary = touching counts); the lower-dim
   solvers always return false.

This is asserted in `convex_convex` with `debug_assert`s at the boundary,
so any future change to `gjk.rs` that breaks the contract fails immediately
at the consumer instead of corrupting EPA input.

## 3. Deviation — witness construction (`docs/bullet-deviations.md` §1)

Bullet's `witnesses[1] = w0 - normal*depth` uses `w0`, the barycentric
interpolation of the three best-face **A-side** supports. Ours uses the
single `supportA(n)` point. Both constructions:

- land on B's penetrating support plane (`n·position == n·supportB(-n)`),
  which is what carries the depth, and
- inherit A's tangential coordinate only through the A-point choice.

The difference is therefore a bounded tangential offset of the witness
point within the contact patch, **not** a depth/normal difference — but it
is real, observable through contact position, and **not exactly Bullet's
B-side witness for arbitrary generic convex pairs**. Accepted (parity is
bit-pinned with the current construction; changing it would require a
parity re-record) and pinned by three witness-support-plane tests.

## 4. Tests (adversarial suite, contract-pinning round)

8 tests: the three contract pins, per-status acceptance propagation,
witness support-plane validation ×3 (convex hull / box / mixed pair),
unequal margins, a 78-step Band→EPA continuity sweep, exact touching,
micro-penetration and micro-separation routing.

## 5. Verdict

**Audit-complete.** The routing contract (Separated / Band / Overlap), the
EPA interface (status acceptance mirrored from `:973-1004`), the margin
architecture and the proven-and-pinned GJK→EPA contract are stable. The
witness construction remains the one documented behavioral deviation; it is
characterized (support-plane property, bounded tangential offset), recorded
in deviations §1 and pinned by tests. Changing it toward Bullet's
barycentric witness is a parity-re-record-level decision, deferred to
post-1.0.
