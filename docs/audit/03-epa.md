# Audit — EPA (expanding polytope algorithm) (03)

- **Date**: 2026-09-08 (conclusions fixed in the 2026-09-07 hardening round
  and re-verified against the reference source; this record consolidates
  them — no re-audit performed)
- **HEAD at audit**: `9aebf38` (fix round `1286304`, revised by `d876cfa`)
- **Scope**: `crates/bullet-collision/src/narrowphase/epa.rs` —
  `penetration` / `penetration_with_status`, `EpaScratch`, `expand`
  (two-phase horizon), `face_visible`, `find_best`, `build_initial_faces`.
- **Reference**: `refs/bullet3/src/BulletCollision/NarrowPhaseCollision/
  btGjkEpa2.cpp` EPA section (constants, `Evaluate`, `expand`,
  `bestface`/horizon commit, status set, `btGjkEpaSolver2::Penetration`
  acceptance).
- **Verdict**: ✅ **Audit-complete.** One reproduced bug fixed (expand
  atomicity, `1286304`); status semantics matched to Bullet's acceptance
  behavior; the queued `INSIDE_EPS` / `FALLBACK` risk is **resolved** —
  both are dead constants in the reference (defined and undefined, never
  used).

## 1. Reference decisions covered

| Decision point | Bullet reference (`btGjkEpa2.cpp`) | This crate | Match |
|---|---|---|---|
| Vertex budget | `EPA_MAX_VERTICES 128` | `MAX_VERTICES = 128` | ✅ |
| Iteration budget | `EPA_MAX_ITERATIONS 255` | `MAX_ITERATIONS = 255` | ✅ |
| Convergence tolerance | `EPA_ACCURACY` = 1e-12 (f64) / 1e-4 (f32) | `EPSILON` = 1e-12 / 1e-4 | ✅ |
| Plane epsilon | none (face normal from cross products) | `PLANE_EPS` = 1e-5 / 1e-14 (degenerate-face guard) | ⚠️ stricter |
| Face budget | `EPA_MAX_FACES = EPA_MAX_VERTICES * 2` = 256 | fixed-capacity face pool of the same size | ✅ |
| Seed | the 4-point simplex from GJK; `expand` failure falls back through a six-axis seeding loop | same seed + six-axis seeding loop | ✅ |
| Horizon commit atomicity | Bullet's `expand` is transactional: `valid && horizon.nf >= 3` gates the remove/append commit (`btGjkEpa2.cpp:716` branch) | two-phase `expand`: phase 1 collects the visible-face horizon and validates it fully (degenerate-edge check), phase 2 commits — a validation failure leaves the hull byte-identical | ✅ same guarantee, stricter structure |
| Visibility test | sign of `dot(n, w - a)` | `face_visible` reused by both phases (same inputs, same float order — IEEE-deterministic) | ✅ |
| Termination | `AccuraryReached` on face-gain < accuracy; iteration/vertex exhaustion keeps status `Valid` | `Converged` / `IterationLimit` / `VertexLimit` | ✅ |
| Degenerate classification | `InvalidHull` when the horizon cannot commit | `DegenerateExpansion`; the six-axis seeding failure reports `DegenerateExpansion` (was mislabeled `IterationLimit` before `9aebf38`) | ✅ after `9aebf38` |
| All-degenerate seed (collinear tetrahedron, 0 valid faces) | not specifically handled | early return `(DegenerateExpansion, ZERO, 0)` before `find_best` — removes the empty-hull `unwrap` panic path | ⚠️ stricter |
| Acceptance semantics | `btGjkEpaSolver2::Penetration` (:973-1004): `epa_status != Failed` ⇒ accept the best-face result (`witnesses[1] = w0 - normal*depth`, `w0` = barycentric interpolation of the A-side supports); only the seed `FallBack` produces zero depth; `Failed` is unreachable inside EPA | `penetration_with_status`: all non-fatal statuses accepted; `convex_convex` filters zero-depth (`core_depth <= 0 → None`); `Failed` unreachable | ✅ (details in `04-convex-convex.md`) |
| `INSIDE_EPS` / `EPA_FALLBACK` | **defined and `#undef`'d, never used** — dead constants, not behavior | not ported | ✅ nothing to port (resolves the queued risk, see §3) |
| `EPA_FALLBACK`-named status | `eStatus::FallBack` is set at :756 when the seed loop exhausts: `normal = -guess`, `depth = 0` | seed failure early-returns `DegenerateExpansion` with zero depth; consumers filter it | ✅ equivalent outcome |

## 2. Reproduced bug — found and fixed (`1286304`)

**Non-atomic `expand`.** The old implementation mutated the hull
(remove/append) incrementally and could bail mid-way, leaving a hull with
holes; a later `find_best` then read a corrupted polytope. Bullet's expand
is transactional (`valid && horizon.nf >= 3` before commit). Fix: two-phase
expand — collect + validate the full horizon first (degenerate-edge check),
then commit; `face_visible` extracted so both phases run the identical
float-order test. Success-path output is bit-identical (parity 5 scenes
re-run bit-exact).

## 3. Resolved queued risk: `INSIDE_EPS` / `FALLBACK`

`docs/bullet-deviations.md` §2 queued "INSIDE_EPS / FALLBACK not ported"
for verification. Source check of `btGjkEpa2.cpp`: `EPA_INSIDE_EPS`
(:65/:69) and `EPA_FALLBACK` (:72) are `#define`d and `#undef`'d with zero
uses in between — dead constants inherited from the original `btGjkEpa`
prototype. The `eStatus::FallBack` *status* is real (seed exhaustion,
:756) and is covered by our seed-failure early return; its zero-depth
result is filtered exactly like Bullet's (the penetration result only
feeds the pipeline when depth > 0). No adversarial input can observe the
difference; the nine degeneracy tests (identical points, collinear/coplanar
simplexes, near-degenerate seeds, point-set supports with no new vertex,
thin-plate horizons, micro-penetration, 1e-3..1e3 scale sweep) cover the
paths that would have exercised these constants.

## 4. Tests (adversarial suite, `1286304`)

9 tests: identical-point, collinear, near-degenerate and coplanar simplexes,
point-set supports with no new vertex, thin-plate degenerate horizons,
micro-penetration, 1e-3..1e3 scale sweep, plus status-propagation asserts
through `penetration_with_status`.

## 5. Verdict

**Audit-complete.** Constants, seed, horizon transaction, termination
statuses and consumer acceptance semantics match the reference (with two
stricter guards of ours). The queued `INSIDE_EPS`/`FALLBACK` risk is closed
as dead reference code. EPA is consumed only through `convex_convex`
(see [`04-convex-convex.md`](04-convex-convex.md)), which pins the witness
construction and the per-status acceptance behavior.
