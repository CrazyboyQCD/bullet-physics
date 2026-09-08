# Audit — Persistent Manifold (01)

- **Date**: 2026-09-08
- **HEAD at audit**: `9aebf38`
- **Scope**: `crates/bullet-collision/src/manifold.rs` (`PersistentManifold`,
  `ContactManifold`, `PersistentPoint`) plus its consumption in
  `crates/bullet-dynamics/src/world/step.rs` (refresh/redetect paths, capture,
  impulse writeback) and `crates/bullet-collision/src/shapes/mod.rs`
  (`contact_breaking_threshold`).
- **Reference**: `refs/bullet3/src/BulletCollision/NarrowPhaseCollision/
  btPersistentManifold.{h,cpp}`, `btManifoldPoint.h`,
  `BulletCollision/CollisionDispatch/btManifoldResult.{h,cpp}`,
  `btCollisionDispatcher.cpp::getNewManifold`, `btCollisionShape.cpp`
  (`getContactBreakingThreshold` / `getAngularMotionDisc`),
  `btContactSolverInfo.h` (warm-start factor).
- **Verdict**: ✅ **Audit-complete with documented behavioral deviations.**
  No correctness or solver-instability candidate reproduced. Three accepted
  deviation rows in `docs/bullet-deviations.md` §1 (population policy,
  matching policy, rotation-drift gate detail); two stale doc comments fixed
  in `manifold.rs` (comment-only, zero behavior). Deviations D1–D3 are
  pinned by the adversarial validation round (`ba3cd9a`, 2026-09-08, 11
  focused tests) — see §6 for the final per-deviation verdicts.

> Reading guide: "audit-complete" means every observable decision point has
> been compared and the behavioral differences are precisely measured and
> pinned by tests. It does **not** mean the subsystem is 100 % Bullet-
> equivalent — D1 in particular is a real behavioral deviation (accepted,
> see §6).

## 1. Reference decisions covered

Mechanical comparison of every observable decision point:

| Decision point | Bullet reference | This crate | Match |
|---|---|---|---|
| Global breaking threshold | `gContactBreakingThreshold = 0.02` | `CONTACT_BREAKING_THRESHOLD = 0.02` | ✅ |
| Cache capacity | `MANIFOLD_CACHE_SIZE = 4` | `MANIFOLD_CACHE_SIZE = 4` | ✅ |
| Per-pair threshold | `CD_USE_RELATIVE_CONTACT_BREAKING_THRESHOLD` is on **by default** (`btCollisionDispatcher` ctor, `:32`); threshold = `min(shapeA.getContactBreakingThreshold(g), shapeB...)` = `min(discA, discB) · 0.02`, where `disc = bounding-sphere radius + center length` (`btCollisionShape.cpp:44-55`) | `a.shape.contact_breaking_threshold().min(b...)` (`step.rs:556-559`); disc = local-AABB half-diagonal + center length (`shapes/mod.rs:287-291`) | ✅ formula; ⚠️ our AABB carries no margin, so the disc reads a few percent low for default 0.04 margins (documented at the function; well inside the band it feeds) |
| Detection-point entry gate | `btManifoldResult::addContactPoint` drops `depth > manifold threshold` (`:108-110`) | detection output is not gated before `capture`; the same pair threshold then prunes in the immediate `refresh` (`step.rs:561`), same inequality (sign-translated: our `penetration = −m_distance1`) | ✅ equivalent (different route, one re-projection of float noise, same as Bullet's own detect-vs-refresh double computation) |
| Witness construction | `pointA = pointInWorld + normalOnB·depth`; `localA/localB = invXform` into each body (`btManifoldResult.cpp:115-129`) | `wa = wb + n·pen`; `local = inv·(w − pos)` (`manifold.rs:328-332`). With `n = −normalOnB`, `pen = −depth` these are symbol-identical | ✅ |
| Point matching radius | `getCacheEntry`: nearest `localPointA` within `threshold²`, nearest-wins (distance keeps shrinking, `:194-208`) | `capture` find: **first** cached point with `Δlocal_a² < threshold²` (`manifold.rs:337-339`) | ⚠️ first-match vs nearest-match (deviation D2) |
| Point matching extra test | none — normal not consulted | additional `normal·normal ≥ 0.9` requirement | ⚠️ stricter (deviation D2) |
| Matched replacement | `replaceContactPoint` keeps `m_appliedImpulse`, `m_appliedImpulseLateral1/2`, `m_prevRHS`, `m_lifeTime` (friction-cone check only under `BT_CONTACT_FLAG_FRICTION_ANCHOR`, which requires an opt-in collision flag — default path replaces unconditionally) | replace in place, inherit `normal_impulse` only | ✅ for the normal impulse; friction/lateral inheritance is the standing "no friction warm starting" deviation (§1); `m_prevRHS`/`m_lifeTime` have no consumer here (see below) |
| Overflow policy | `addManifoldPoint` at capacity: `sortCachedPoints(newPt)` picks the slot whose replacement **maximizes the 3-point hull area** of the surviving four (`gContactCalcArea3Points = true` default), deepest point exempt (`KEEP_DEEPEST_POINT`); the **new point always enters**, evicting a measured-to-be-redundant old one; successive new points see each other's replacements | after merging the whole detection batch: sort all cached points by penetration descending, truncate to 4 (`manifold.rs:349-353`) | ⚠️ sort-truncate vs area-maximizing replacement (deviation D1): a shallow new point enters Bullet's cache but is dropped here; the evicted point is chosen by depth, not by area contribution |
| Point removal order | `removeContactPoint` swaps the last point into the hole (cache order **not** stable, `btPersistentManifold.h:164-190`); refresh iterates backwards to stay swap-safe | `refresh` uses `retain_mut` (order stable) | ⚠️ order semantics differ; only observable through D1's area computation and through point indexing in solver rows (no behavior depends on Bullet's scrambled order here) |
| Refresh gate 1 (separation) | `validContactDistance`: `m_distance1 ≤ threshold` (`m_distance1 > threshold` ⇒ drop) | `penetration < −threshold ⇒ drop` | ✅ sign-identical |
| Refresh gate 2 (lateral drift) | `posB − (posA − n·d1)` projected squared length > `threshold²` ⇒ drop (`:284-287`) | `(d − n·(d·n))² > threshold²` (`:255-257`) — same quantity, negated | ✅ |
| Refresh lifetime | `m_lifeTime++` per point per refresh (consumed by `FRICTION_ANCHOR` and debug paths only) | no lifetime field; `FRICTION_ANCHOR` not ported (default flag absent in Bullet too ⇒ no observable difference) | ✅ default-equivalent |
| Impulse writeback | solver writes `m_appliedImpulse` + lateral impulses (+ push impulse) back per point (`writebackContacts`) | writeback stores `normal_impulse` only (`step.rs:688-711`) | ✅ for the normal impulse; lateral is the standing friction deviation; **push impulse is not warm-started** (our split push starts from zero each step) — folded into D1's row notes |
| Warm-start fade | `m_warmstartingFactor = 0.85` (`btContactSolverInfo.h:99`) | `WARM_STARTING_FACTOR = 0.85` (`world.rs:53`) | ✅ |
| Dead-pair cleanup | dispatcher `releaseManifold` destroys the manifold when the broad-phase pair dies | lifecycle prune drops entries not `paired_at(stamp)` (`step.rs:712-716`), with the sleeping-island carry-over (§1 deviations) | ✅ equivalent + documented sleeping extension |
| Processing threshold | `m_contactProcessingThreshold = min(...)` defaults to `BT_LARGE_FLOAT`, unused in the default solver path | no field (equivalent to +inf) | ✅ default-equivalent |
| Reduction of multi-point detection sets | none as a separate stage — each algorithm reports its points and the persistent cache itself is the 4-point reducer | `ContactManifold::reduce(4)` greedy deepest-first + max-min-distance on the three many-point paths (plane / mesh-mesh / convex-mesh) before `capture` | ⚠️ architectural: our detection layer pre-reduces; same goal (well-spread 4 points), greedy spread mirrors the area-maximization intent (part of deviation D1) |

## 2. Queued risk questions — resolutions

1. **Rotation-drift semantics** (the visible symptom class in the `rolling`
   ledger): our rotation gate (`REDRIFT_ANGLE = 0.01` rad, summed per body,
   `manifold.rs:285-292`) is part of the self-authored refresh-guard
   mechanism — Bullet has no counterpart (it re-runs detection every step for
   every active pair). The `rolling` ledger's f64 friction-row noise class is
   solver-side tangent-phase accumulation, not manifold rotation drift; the
   guards only bound *when* detection reruns. Already documented in §1
   "Persistent manifold cache: refresh guards"; this audit added the rotation
   gate to that row.
2. **Quaternion q / −q double cover**: no correctness issue. Bullet's
   per-pair threshold uses `getAngularMotionDisc()` (bounding sphere —
   quaternion-free); its refresh composes transforms (q and −q yield the same
   matrix). Our refresh composition is likewise cover-invariant. The only
   cover-sensitive spot is `rotation_angle(q) = 2·acos(w)`, which reads
   `2π − θ` for −q: a sign-flipped quaternion would trigger one spurious
   re-detection (harmless — same detection result, tiny cost). Quaternion
   integration in glam keeps the sign continuous; only an explicit user
   `set_orientation(−q)` can flip it.
3. **Contact-point matching robustness under fast rotation**: the matching
   radius equals Bullet's (`threshold²` on the A-side local anchor). The two
   policy differences (nearest vs first; the extra 0.9 normal-agreement
   check) are recorded as D2. The added normal check makes us *more*
   conservative: a fast-flipping normal never inherits an impulse across the
   flip (Bullet would), at the cost of an occasional warm-start reset. The
   speed/drift guards already re-detect fast pairs, bounding anchor staleness.
4. **Manifold reduction beyond tested scenes**: the structural difference
   (two-layer pre-reduce + sort-truncate cache vs Bullet's single-layer
   area-maximizing replacement) is deviation D1. The resting-flat-box growth
   path (merge accumulation across frames, `capture` keeping undetected
   points) matches Bullet's mechanism and is what the parity scenes and the
   flat-box probe already exercise; adversarial reduction inputs belong to
   the dedicated adversarial-testing phase.

## 3. Deviations (rows in `docs/bullet-deviations.md` §1)

- **D1 — Population policy on overflow**: sort-by-depth truncate vs
  area-maximizing slot replacement (`sortCachedPoints`); shallow new points
  never enter our cache, evicted point chosen by depth not by area; split
  push impulse is not warm-started.
- **D2 — Matching policy**: first-match within `threshold²` plus a
  `normal·normal ≥ 0.9` agreement test, vs Bullet's nearest-match-only.
- **D3 — Rotation-drift gate** (folded into the existing refresh-guards
  row): the `0.01` rad per-body rotation gate and its q/−q asymmetry
  (spurious re-detection only).

## 4. Test gaps → closed by the adversarial validation round

The three gaps below at audit time are all closed by `ba3cd9a` (11 tests,
`manifold.rs` `#[cfg(test)]`):

- ~~No unit test drives the overflow path~~ → clustered 6-point batch,
  spread-cache + near-duplicate divergence, planar/triangular patch
  evolution under repeated overflow (capacity, deep-anchor stability,
  identity freeze, spread retention), each against a minimal reference of
  Bullet's `sortCachedPoints`.
- ~~No test distinguishes first-match from nearest-match~~ → two cached
  points inside one matching radius: the reference model would pick the
  other point; inheritance follows cache order deterministically; the
  0.9 normal-agreement guard appends instead of inheriting.
- ~~No test covers the rotation gate's q/−q asymmetry~~ → gate boundaries
  (strict `>`, either body alone, opposite rotations add up),
  near-threshold self-consistency, q/−q (one spurious re-detection,
  identical refreshed contact geometry), and 20-frame resting-contact
  churn (anchors bit-stable, impulse riding, separation kills the point).

## 5. Candidates (none reproduced — remain candidates)

- `ContactManifold::reduce(0)` returns the single deepest point instead of
  an empty manifold (the `while kept.len() < max` loop never runs but the
  deepest point is pushed unconditionally). Unreachable in the codebase:
  every call site passes the constant `4`. Recorded, not fixed (no repro,
  and the doc contract says "at most `max`" — 1 > 0 violates it only in this
  dead configuration).
- Doc-comment drift (fixed in this audit, comment-only): `capture` claimed
  the overflow path was "Bullet's `sortCachedPoints`" and "the deepest four
  survive" — neither is what the code does (see D1); the wording now
  describes the actual policy and names the difference.

## 6. Final verdicts (post adversarial validation, `ba3cd9a`)

- **D1 — Accepted deviation (behavioral).** The cache
  population/reduction policy differs materially from Bullet's
  area-maximizing replacement: a clustered 6-point batch keeps
  `{deep + 3 clustered}` where the reference keeps `{deep + 2 far + 1
  clustered}` (spread 0.04 vs 1.41), and under growing contact patches the
  cache identity freezes after the fourth frame — newly arriving points are
  absorbed later than Bullet would absorb them (contact-patch coverage /
  manifold-adaptation deviation). Mitigations that contain it in practice:
  the depth-order policy self-heals (a tilting contact deepens its point
  into the top-4), the lateral-drift refresh frees slots, and the detection
  layer pre-reduces to a spread set. The divergence is bidirectional — on
  spread inputs Bullet admits a shallow near-duplicate and evicts a support
  point, which ours refuses. No correctness or solver instability was
  reproduced in the parity/probe/test corpus (8 parity scenes, 24 mb
  probes, the adversarial evolution tests). Future convergence toward
  Bullet should implement area-maximizing replacement and requires a
  parity re-record.
- **D2 — Accepted deviation (intentional engineering choice).** First-match
  + normal agreement is deterministic (follows cache order, tests pin both
  orders), the inherited impulse always comes from the point actually
  matched, and a flipped normal never inherits across the flip (Bullet
  would). The extra `dot ≥ 0.9` gate is considered the right call for this
  codebase: it trades an occasional warm-start reset for never inheriting
  an impulse across a contact-normal flip.
- **D3 — Accepted deviation (intentional refresh policy).** The
  rotation-drift gate is part of the self-authored refresh-guard mechanism
  (deviations §1 "refresh guards"); q/−q double cover can cause one
  redundant re-detection only — the refreshed contact geometry is
  identical either way.

Subsystems not covered by this audit: narrow-phase band semantics
(`04-convex-convex`), detection-side `reduce` targets, and solver consumption
of the emitted rows — those live in their own audit slots.
