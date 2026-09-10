# Audit 08 — CCD

- **Date**: 2026-09-09
- **Baseline**: `da6d3f9` (HEAD, = `origin/main`); this round's fix and tests
  are uncommitted working-tree changes at the time of writing
- **Scope**: the CCD swept-sphere clamping path only — trigger conditions,
  swept-AABB pretest, candidate filtering, the exact sweep/TOI, target
  selection, position correction, and the lifecycle/numerical edges of that
  path. BroadPhase / Manifold / Solver / Islands are **not** re-audited;
  they appear only as interface contracts the CCD path depends on.
- **Reference**: `refs/bullet3/src/BulletDynamics/Dynamics/
  btDiscreteDynamicsWorld.cpp` (`integrateTransformsInternal` :951-1050,
  `btClosestNotMeConvexResultCallback` :756-844),
  `BulletDynamics/Dynamics/btRigidBody.cpp:100-103`
  (`predictIntegratedTransform`), `BulletCollision/CollisionDispatch/
  btCollisionObject.h:306-309` (`isActive`),
  `BulletCollision/BroadphaseCollision/btDispatcher.h:42-62`
  (`m_useContinuous`, `m_allowedCcdPenetration`), and the already-audited
  sweep core `crates/bullet-collision/src/narrowphase/convex_sweep.rs`.
- **Status: COMPLETE**
- **Verdict**: ✅ **Audit-complete.** One new confirmed bug found and fixed
  (C-1: sensor targets were not excluded from the CCD sweep). All other
  trigger/filter/TOI/clamp decision points match the Bullet reference or
  fall under existing documented deviations; the three known CCD
  deviations were re-verified against the source and remain accurate.
  Nine regression tests joined the suite (436 total, f32/f64 green);
  the `ccd` parity scene reproduces its ledger numbers at both precisions.

## 1. Scope

Audited: the CCD path in `crates/bullet-dynamics/src/world/step.rs`
(motion-clamp computation, target loop, clamp application), the sweep core
`bullet-collision/src/narrowphase/convex_sweep.rs` (already
audit-covered by 02/04 for its GJK internals; re-checked here only for the
fraction/None contract the CCD consumes), the CCD fields and gates on
`RigidBody` (`body.rs`), and the `is_convex` shape gate
(`bullet-collision/src/shapes/mod.rs`).

Not audited (covered elsewhere): AxisSweep3/DBVT internals (05/06),
manifold lifecycle (01), solver (—), island membership (07). The broadphase
`proxy_aabb` pretest and `no_collision_pairs` appear only as CCD-side
interface consumers.

## 2. Baseline / Git State

```
git log:  da6d3f9 (HEAD -> main, origin/main) docs: record the external-audit fix round
          aafdb08 fix(dynamics): close four external-audit findings
git status: docs/audit/07-broadphase-integration.md (new, uncommitted) + README table
Baseline tests: 427 (collision 222 / dynamics 199 / math 6), f32+f64 green
```

Existing CCD regression coverage before this round: `ccd_clamps_fast_bodies_at_the_impact`,
`ccd_ignores_slow_motion`, `ccd_skips_collision_suppressed_pairs`,
`ccd_respects_collision_filter` (`world/step.rs`),
`removed_body_does_not_block_ccd` (`world.rs`), plus the
`swept_sphere_cast` unit tests in `convex_sweep.rs`.

## 3. CCD Pipeline

The actual flow as implemented (`step.rs`, stage 7), not inferred from
names:

```
solve / deactivation (stages 1-6)
    ↓
CCD eligibility, per body  — `is_active()`, `!is_static_or_kinematic()`,
    |                        `ccd_motion_threshold() > 0`,
    |                        `ccd_swept_sphere_radius() > 0`, `shape.is_convex()`
    ↓
swept segment  — from = current position (pre-integration snapshot);
    |            to = from + linear_velocity * dt (post-solve velocity)
    |            skip unless threshold² < square_motion
    ↓
swept AABB  — Aabb(from.min(to), from.max(to)).expanded(radius)
    ↓
target loop over all rigid bodies j ≠ i:
    removed → skip; sensor (NO_CONTACT_RESPONSE) → skip   (C-1 fix, this round)
    no_collision_pairs → skip; group/mask (both ways) → skip
    PairFilter → skip; proxy AABB ∩ swept AABB → else skip
    ↓
exact sweep  — narrowphase::swept_sphere_cast(from, to, radius,
    |          target shape @ target position/orientation)
    ↓
minimum fraction across targets  (best starts at 1.0)
    ↓
integrate_position + split-impulse push (all bodies)
    ↓
clamp: position = (position − v·dt) + (position − (position − v·dt)) · f
    ↓
normal simulation continues (collision response next step)
```

Both sweep endpoints and every target pose are read from the
pre-integration snapshot (`ccd_clamps` is computed in a first pass, the
integration and clamping happen in a second): the sweep is an internally
consistent snapshot.

## 4. Trigger Conditions

Decision table, Rust vs Bullet (`integrateTransformsInternal`,
btDiscreteDynamicsWorld.cpp:951-1050):

| Condition | CCD expected | Rust actual | Bullet | Status |
|---|---|---|---|---|
| below threshold | no sweep | `square_motion <= thr²` → skip | `ccdSquareMotionThreshold() < squareMotion` false → skip (`:965`) | match |
| exactly at threshold | no sweep | skip (`<=`) | skip (strict `<`) | match |
| above threshold | sweep | sweep | sweep | match |
| static target | hit | allowed | allowed (`needsCollision` does not exclude static; `USE_STATIC_ONLY` variant is off by default) | match |
| kinematic target | hit | allowed | allowed | match |
| dynamic target | hit | allowed | allowed | match |
| group/mask-filtered target | no hit | skip (both directions) | skip (`ClosestConvexResultCallback::needsCollision`, both directions) | match |
| PairFilter-rejected target | no hit | skip | skip (`needsBroadphaseCollision` overlap filter, `:825-832`) | match |
| sensor (`NO_CONTACT_RESPONSE`) target | no hit | **was hit → now skipped (C-1)** | skip (`hasContactResponse`, `:778-780`) | **fixed this round** |
| suppression pair (`no_collision_pairs`) | no hit | skip | skip (`dispatcher->needsCollision` → `checkCollideWithOverride`, `:833-836`) | match |
| self | never | `i == j` skip | `m_me` check (`:771`, `:820`) | match |
| removed target | never | `is_removed()` skip | n/a — removal destroys the proxy, the sweep never sees it | equivalent |
| inactive caster | no sweep | `is_active()` gate | `isActive()` (`:957`; semantics per `btCollisionObject.h:306-309`) | match |
| static/kinematic caster | no sweep | `is_static_or_kinematic()` gate | `isStaticOrKinematicObject()` (`:957`) | match |
| non-convex caster | no sweep | `is_convex()` gate (compound = all children convex) | `isConvex()` enum-range check (`:969`; compound never convex) | deviation 3 (below) |
| threshold ≤ 0 or radius ≤ 0 | no sweep | both gate the sweep | threshold gate only; a zero-radius cast would still run | implementation difference (degenerate config) |
| world-level `m_useContinuous` | n/a | no world switch | `m_useContinuous` default true (`btDispatcher.h:42`) | implementation difference (API) |

The threshold comparison semantics are pinned by
`ccd_threshold_boundary_is_exclusive`.

## 5. Swept AABB

`Aabb::new(from.min(to), from.max(to)).expanded(radius)`:

- Contains both endpoints and the sphere radius expansion in all
  directions — velocity sign plays no role (min/max).
- The pretest compares against the target's stored proxy AABB, so it can
  only over-select (false positives are filtered by the exact cast); a
  false negative would require the stored proxy AABB to not contain the
  target, which the broadphase contract (Audit 07 §2) forbids.
- No NaN/Inf production: `min`/`max` propagate input NaN into the overlap
  test, which then returns false — a conservative skip. All finite-body
  paths stay finite (verified by the numerical tests below, including
  1e4-scale coordinates and 1000x speeds).
- Small/large dt: the swept box scales with `v·dt`; dt only affects its
  size, never the containment invariant.

Pinned by: `ccd_threshold_boundary_is_exclusive` (a 1.5-unit step clears
the thin wall entirely — the clamp is only possible if the swept box
covered the full path) and `ccd_selects_nearest_of_multiple_targets` (the
near wall sits outside the endpoint AABB, mid-flight).

## 6. Candidate Filtering

The target loop applies, in order: self → removed → **sensor** (C-1 fix) →
`no_collision_pairs` → group/mask → PairFilter → proxy-AABB swept pretest
→ exact cast. This matches the necessary subset of Bullet's CCD filter
chain (`:771-836`): self, group/mask, overlap filter,
`checkCollideWithOverride`, `hasContactResponse`. The Bullet direction
guard (`:788-790`) is the documented allowed-penetration deviation
(deviation 1), not a filter.

One deliberate ordering note: the kinematic wake in the island pass (07)
and the CCD target loop are independent consumers of eligibility; the CCD
loop reads the body fields directly and is not affected by the island
`proxy_flags` fast path.

## 7. Exact Sweep / TOI

`swept_sphere_cast` / `sweep_with_dist` (`convex_sweep.rs:155-209`):

- The returned fraction comes from a binary search bounded in `[0, s_min]`
  with `s_min ≤ 1` — it cannot leave `[0, 1]`. The plane path additionally
  checks `(0.0..=1.0).contains(&fraction)` explicitly (`:287`), and the
  already-touching case returns fraction 0 (`:273-279`).
- No-hit is `None` (`d_min² > 1e-8` at the closest approach); negative
  fractions, fractions > 1, NaN and Inf have no production path: the
  distance functions return squared magnitudes (non-negative), the search
  parameters are convex combinations of `lo`/`hi` starting at `[0,1]`, and
  divisions are guarded (`length <= EPSILON → None`, normal fallback
  `-dir`).
- Starting inside the target: `point_convex_distance` returns `(0, ZERO)`
  → binary search collapses to fraction 0 → the clamp holds the body in
  place (no backward teleport, no explosive correction).
- Grazing/tangent approach: the convex distance along the segment is
  unimodal, so the ternary search locates the closest approach; pinned by
  the sloped-triangle test in `convex_sweep.rs` and the clamp tests here.
- Multiple targets: the caller keeps `best = min(fractions)`; see §11.

## 8. High-Speed / Tunneling Tests

Deterministic scene: 0.3-radius sphere, thin 0.1-thick static wall,
`step_simulation` one step.

- **Test A (ordinary high speed, CCD on/off)**: covered by the existing
  `ccd_clamps_fast_bodies_at_the_impact` (120 u/s: CCD clamps at the wall's
  inflated top 0.9, the CCD-less control crosses the whole wall).
- **Test B (threshold boundary)**: `ccd_threshold_boundary_is_exclusive` —
  motion exactly 1.5 with threshold 1.5 passes through; motion 1.6 clamps
  at 3.4. (The geometry deliberately exceeds what the predicted-pose
  narrow phase can catch, so the CCD gate is the only decider.)
- **Test C (speed sweep 1x/10x/100x/1000x)**:
  `ccd_speed_sweep_stays_finite` — 120 … 120 000 u/s, all clamped at 0.9,
  position/velocity/orientation finite at every multiple, at 1e4-scale
  world coordinates too.

No NaN/Inf/invalid fraction was produced at any speed. Note the clamp is
not claimed to be zero-penetration: the body is left touching the
inflated sweep sphere and the contact resolves on the next step's narrow
phase (Bullet behaves the same way; collision response runs next step).

## 9. High Angular Velocity

Confirmed: the clamp rewinds **position only** — `ccd_clamps` scales
`(position − from)`; the orientation integrates the full step
(`step.rs`, `body.integrate_position`). Bullet re-predicts the *whole*
transform by `dt · hitFraction` (`:1005-1006`,
`predictIntegratedTransform` = full `integrateTransform`,
btRigidBody.cpp:100-103). This is **existing deviation 2** (below) —
re-verified against the source, still accurate, no new correctness
failure. `ccd_high_spin_stays_finite` (3000 rad/s spin + 120 u/s linear)
pins: finite quaternion, normalized (|q|² = 1), finite position, clamp
still lands at 0.9.

## 10. Starting-Inside / Penetration

`ccd_starting_inside_is_finite`: a body starting with its center inside
the wall clamps at fraction 0 and stays put (finite, no backward
teleport, no explosive correction). The direction-guard semantics (a body
starting inside and moving away still clamps at 0, because there is no
`hitNormal · v >= -allowedPenetration` reject) are **existing deviation
1** — re-verified, still accurate. The sweep's allowed-penetration slack
(`m_allowedCcdPenetration = 0.04`, `btDispatcher.h:47`, applied at
`:994`) is likewise not ported — same deviation entry.

## 11. Target Selection

`ccd_selects_nearest_of_multiple_targets`: two parallel walls, the near
wall's hit (sphere-center entry 1.65 = face 1.95 − radius 0.3) wins over
the far wall's (5.65). Second scenario: the near wall is group/mask
excluded → the far wall's hit is selected (5.65). The minimum is taken
over *valid* hits only — filtering runs before the cast, so an excluded
near target cannot shadow the far one.

## 12. Lifecycle / Filter Tests

- **Remove target**: existing `removed_body_does_not_block_ccd` (the
  sweep ignores the removed wall, the ball flies on).
- **Proxy recreate**: `removed_ccd_target_proxy_recreate_is_not_stale` —
  remove the wall, add a new wall at a different height (the freed proxy
  slot is recycled): the clamp lands on the *new* wall's inflated top
  (0.7), not the stale AABB (0.9). No stale proxy/handle/ContactKey
  effect is reachable from the CCD path: the target loop resolves bodies
  by slot index each step and the proxy pretest reads the live proxy.
- **Filter mutation**: `ccd_suppression_flip_gates_the_sweep` — pair
  enabled → clamp; `set_disable_collisions(true)` → next step crosses;
  re-enabled → next step clamps on the second wall. Group/mask is covered
  by `ccd_respects_collision_filter` and the excluded-near scenario of
  §11; sensors by the C-1 regression below.

## 13. Numerical Robustness

All assertions measured, f32 and f64:

- dt = 1e-4 (motion below threshold, plain integration) and dt = 0.5
  (motion 60, clamped): `ccd_extreme_timestep_stays_finite` — finite
  positions, exact plain-integration landing for the small step.
- Very small velocity: `ccd_ignores_slow_motion` (existing) — no sweep.
- Small object / large coordinates: the 1e4-offset scene in
  `ccd_speed_sweep_stays_finite`.
- High angular velocity: §9.
- No failure (NaN/Inf/negative fraction/backward teleport) was observed in
  any scenario; nothing needed a reduced repro.

## 14. Bullet Parity

| Behavior | Rust | Bullet | Difference | Classification |
|---|---|---|---|---|
| Trigger gates | active, non-static/kinematic, `thr² < motion²`, convex, radius > 0 | active, non-static/kinematic, `m_useContinuous` + `thr² < motion²` (`:965`), `isConvex()` (`:969`) | compound gate; zero-radius; no world switch | deviations 3 + implementation differences |
| Cast shape | sphere of `ccd_swept_sphere_radius` (position-only segment) | `btSphereShape` over `from → modifiedPredictedTrans` (origin predicted, basis current, `:998-1000`) | none | match |
| Target filter chain | self/removed/sensor/suppression/group-mask/pair-filter/pretest | self/group-mask/overlap-filter/`checkCollideWithOverride`/`hasContactResponse`/direction guard (`:771-836`) | sensor was missing → **C-1 fixed**; direction guard = deviation 1 | match after fix |
| Clamp | position-only retraction by fraction | full-transform re-prediction (`:1005-1006`) + `proceedToTransform` | rotation not rewound | deviation 2 |
| Target selection | min fraction over valid hits | closest-hit callback | none | match |
| Sweep targets | rigid bodies only | all collision objects (incl. multibody colliders, soft-body objects) | link/soft targets not swept | implementation difference (link-side already documented in `docs/design/multibody-collision.md`; target side recorded here) |
| Detection snapshot | consistent pre-integration snapshot for all targets | in-loop `proceedToTransform` writes back per body (order-dependent poses) | only observable with two fast bodies sweeping each other | implementation difference |
| Contact anticipation | predicted-pose narrow phase (deviations §1 "Contact detection pose") | `createPredictiveContacts` for CCD movers (`:856-905`) | different mechanism, same purpose | already documented |

No numeric difference was automatically treated as a bug; each row above
carries its classification.

## 15. Findings

### C-1 — CCD sweep hits sensor (no-contact-response) targets

- **ID**: C-1
- **Severity**: P2 (edge-case correctness)
- **Status**: CONFIRMED → RESOLVED (this round)
- **Trigger**: a CCD-enabled fast body whose swept path crosses a
  `NO_CONTACT_RESPONSE` (sensor) body.
- **Observed**: the ball was clamped at the sensor's surface (velocity
  killed by the resulting contact) — a sensor stops a fast body dead.
  Reproduced by `ccd_ignores_no_contact_response_targets` before the fix
  (position 0.9, expected < −0.5).
- **Expected**: the sensor is ignored by the sweep; the body flies
  through.
- **Root cause**: the CCD target loop never consulted
  `collision_flags.contains(NO_CONTACT_RESPONSE)`, while Bullet's CCD
  callback rejects objects without contact response
  (`btClosestNotMeConvexResultCallback::addSingleResult`,
  btDiscreteDynamicsWorld.cpp:778-780).
- **Evidence**: the repro above (failed before, passes after); the
  ordinary collision pipeline already skips sensor *rows* but still
  detects them, which is what made the discrepancy visible only on the
  CCD path.
- **Fix**: one gate in the target loop (step.rs): sensor targets are
  skipped, with a comment citing the Bullet callback line.
- **Regression test**: `ccd_ignores_no_contact_response_targets`.

No other new correctness findings. In particular, the investigation of
the threshold-boundary scenario surfaced only the already-documented
predicted-pose detection architecture (deviations §1 "Contact detection
pose"), not a CCD bug.

## 16. Regression Tests

Nine tests joined the suite this round (names as in the tree):

| Test | Pins |
|---|---|
| `ccd_ignores_no_contact_response_targets` | C-1: sensor walls never clamp |
| `ccd_threshold_boundary_is_exclusive` | threshold comparison is strict; exactly-at does not sweep |
| `ccd_selects_nearest_of_multiple_targets` | min-fraction selection; near-excluded → far target |
| `ccd_speed_sweep_stays_finite` | 1x–1000x speeds clamp correctly, all finite, 1e4 coordinates |
| `ccd_high_spin_stays_finite` | spin + clamp: finite normalized orientation, position-only rewind is benign |
| `ccd_starting_inside_is_finite` | fraction-0 clamp: stays put, finite, no teleport |
| `ccd_extreme_timestep_stays_finite` | tiny/huge dt: plain integration vs clamp, finite |
| `ccd_suppression_flip_gates_the_sweep` | runtime suppression flips gate the sweep from the next step, both directions |
| `removed_ccd_target_proxy_recreate_is_not_stale` | proxy slot reuse: the sweep resolves the new owner/AABB |

All are deterministic (fixed dt, zero gravity or analytic expectations),
binary-exact where threshold equality matters, f32/f64-clean, independent
of test order, and use only public world APIs.

Current suite totals (measured): **436 tests** — collision 222,
dynamics 208, math 6 — green on **f32** and **f64**. Toolchain gates:
clippy clean on both features; no_std check (`--no-default-features
--features libm,f32|f64`) zero errors/warnings; `python tools/check.py`
green (tests + 14 demos); parity scene `ccd` reproduces the ledger at
both precisions (pos max 0.000075 @ frame 0, exceed=0 — the established
clamp/rest class).

## 17. Existing Deviations

Re-verified against the current source this round:

| Existing deviation | Still accurate? | New correctness risk? | Evidence |
|---|---|---|---|
| CCD sweep: allowed penetration & direction guard (deviations §1) | yes — no `m_allowedCcdPenetration` slack, no `hitNormal·v ≥ −allowed` reject; starting-inside-moving-away clamps at 0 | no | source: `convex_sweep.rs` (no reject path), Bullet `:788-790`, `:994`; `ccd_starting_inside_is_finite` |
| CCD clamp: which motion is rewound — position only (deviations §1) | yes — orientation integrates the full step | no | source: clamp block scales position only; Bullet `:1005-1006`; `ccd_high_spin_stays_finite` |
| CCD sweep: compound bodies — swept when all children convex, targets via aggregate support (deviations §1) | yes — `is_convex` recurses into compounds (`shapes/mod.rs:146-157`), compound targets hit via `support_world` | no (conservative: sweeps more often than Bullet) | source as cited; unchanged this round |

No deviation document changes were needed; no deviation was re-filed as a
finding.

## 18. Conclusion

**08 CCD — COMPLETE.** The CCD implementation matches Bullet's
swept-sphere clamping on every observable decision point except the four
entries classified above: three documented deviations (penetration slack,
position-only rewind, compound gate) and a small set of implementation
differences (target range, detection snapshot, degenerate-config and
world-switch gates) that are conservative or unobservable in the recorded
scenarios. One real bug (sensor targets not excluded) was found by the
adversarial round and fixed; the full trigger/filter/TOI/clamp surface is
now pinned by thirteen CCD-domain tests plus the sweep unit tests.
Recommendation for Audit 09: per the project roadmap
(`docs/audit/00-project-status.md`), the integrated adversarial pass is
next; CCD-specific ground is done and should not be re-opened there
beyond its interface contracts.
