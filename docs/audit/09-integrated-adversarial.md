# Audit 09 — Full Integrated Adversarial Pass

- **Date**: 2026-09-09
- **Baseline**: `e43c6d1` (HEAD, = `origin/main`; `9dd63ce` CCD sensor fix and
  `093f021` the Audit 07 record are already on the remote). This round's
  fix, tests, and records are uncommitted working-tree changes at the time
  of writing.
- **Scope**: the interaction contracts between the audited subsystems
  (01 manifold, 02 GJK, 03 EPA, 04 convex-convex, 05 AxisSweep3, 06 DBVT,
  07 broadphase integration, 08 CCD) as they run together inside
  `PhysicsWorld::step_simulation` — cross-frame state consistency,
  lifecycle consistency, numerical stability, and determinism. No
  subsystem-internal algorithm is re-audited.
- **Status: COMPLETE**

## 1. Scope

Ten adversarial scenario groups plus the deterministic state machine
(§15), each attacking a module boundary: deep penetration × solver ×
sleep, extreme mass ratios, CCD → normal contact handoff, sleep/wake ×
events, filter mutation × island × manifold × events, remove/recreate ×
entire state, broad-phase backend consistency, multibody × rigid,
softbody × rigid, teleport/large motion. Subsystem internals were **not**
re-opened; where a scenario touched a documented deviation, the deviation
absorbed the observation (none was re-filed as a finding).

## 2. Baseline / Git State

```
git status: clean at start
git log:    e43c6d1 (HEAD -> main, origin/main) docs(audit): CCD record (08)
            093f021 docs(audit): broadphase integration record (07)
            9dd63ce fix(dynamics): exclude sensor targets from the CCD sweep
rev-list HEAD...origin/main: 0 0 (pushed)
Baseline tests: 436 (collision 222 / dynamics 208 / math 6), f32+f64 green
```

## 3. Existing Audit Contracts

The integration surface leans on these previously verified contracts:

- **Pair contract** (07 §2): `compute_pairs` returns stored-proxy-AABB
  overlap candidates; false positives allowed, false negatives forbidden.
- **Island eligibility** (07 §5.1): unions consume collision-eligible
  pairs only (group/mask, pair filter, sensor); suppression is not an
  eligibility condition.
- **Removal cascades** (07 §5.3): removal-time event metadata resolves
  before list shifts; soft joints renumber.
- **CCD target filter chain** (08 §6): self/removed/sensor/suppression/
  group-mask/pair-filter/pretest/exact-cast.
- **Predicted-pose detection** (deviations §1): the narrow phase detects
  contacts at the integrated pose; contact response lands one step early
  relative to Bullet's discrete detection.

## 4. Cross-System Invariants

Checked in every scenario below:

- **Identity**: a removed object ≠ a recreated object, regardless of slot
  or proxy reuse.
- **Filter coherence**: disabled collision → no solver contact, no new
  event, no island effect; re-enabled → contact rebuilds normally.
- **Lifecycle coherence**: remove → no stale proxy, manifold, island
  member, or event identity.
- **Numerical coherence**: `finite(position/rotation/velocities)` every
  frame.
- **Determinism**: same initial state + same scripted operations →
  identical results (bit-level within one binary).

## 5. Deep Penetration

`deep_penetration_settles_without_nonfinite_state` — boxes spawned inside
the floor at 0.05 / 0.25 / 0.45 (single-sided plane, within its
recoverable depth) and 1.0 / 2.0 (thick box floor), 300 steps each:

- Every frame finite (position, linear and angular velocity).
- Recovery is upward and bounded (never above the surface + overshoot);
  peak speed bounded.
- All depths settle to the surface and reach `IslandSleeping`.

**Behavioral boundary recorded (not a bug)**: a box whose center is fully
past a single-sided plane stops colliding and falls freely — exactly
Bullet's `btStaticPlaneShape` semantics (first observed at depth 0.5 =
half-extent: the body free-falls to −123 m over 300 steps, finite and
unbounded-in-space but collision-correct). "Deep penetration recovery"
has a physical depth limit on one-sided geometry in both engines.

## 6. Extreme Mass Ratio

`extreme_mass_ratio_remains_bounded` — 1e-6 : 1 pairs at restitution 0,
three scenarios:

- Heavy moving into a resting light body: the pair merges onto the
  heavy body's speed (perfectly inelastic); the light body stays under
  the 2× exchange bound; the heavy body never accelerates beyond its
  incoming speed.
- Light firing into a resting heavy body: the heavy body moves by a
  sliver (< 1e-3), the light body stays bounded.
- Light-heavy-static crush chain: finite and bounded for 240 steps, the
  heavy body never squeezed into the wall.

No NaN, no velocity explosion, no sleep/wake oscillation observed.

## 7. CCD → Normal Contact

`ccd_clamp_then_normal_contact_settles_and_sleeps` — a ball dropped from
y = 100 (impact speed ≈ 44 u/s, 0.74 u/step, beyond the predicted-pose
catch window):

- The CCD clamp lands the ball at the surface; no tunneling below the
  floor in any later frame; no position jump after the clamp.
- Contact then resolves through the normal pipeline; a few
  Started/Stopped micro-bounce flips during the settle are physically
  expected (solver push-out and re-detect); once asleep there are no
  further events.
- The ball reaches `IslandSleeping` — the CCD → manifold → solver →
  sleep chain is consistent end to end.

**No additional reproducible correctness finding was found in the CCD →
normal-contact transition.**

## 8. Sleep / Wake

`sleep_wake_preserves_contact_lifecycle` — a box and a ball settle and
sleep; the ball is woken, activated, and rolled into the box:

- The hit wakes the box exactly as needed (`IslandSleeping` → active)
  and produces exactly one Started for the hit.
- Teleporting the ball away emits exactly one Stopped and leaves no
  stale touching (`overlapping` empty for the pair).
- The box settles back to `IslandSleeping`.
- Sleeping pairs emit nothing while asleep (covered further by the CCD
  settle scenario's empty final buffer).

No spurious wakes from unrelated pairs, no duplicate begins, no stale
touching state.

## 9. Filter Mutation

`filter_flip_clears_and_recreates_contact_state` — a ball settling onto
a static sphere:

- Mask off: the very next step tears down the touching state with
  exactly one Stopped; the ball then falls through the sphere with no
  contact response (gravity unchecked).
- Mask on (after sinking into the sphere): the deep overlap
  re-establishes contact with a fresh Started; `overlapping` repopulates.

The four layers (broad-phase candidate, island membership, manifold,
events) were each observed consistent with the filter state. Runtime
constraint-suppression flips are already pinned by Audit 08's
`ccd_suppression_flip_gates_the_sweep`; group/mask, PairFilter and the
CCD sweep were pinned there and in 07.

## 10. Remove / Recreate

`remove_recreate_does_not_retain_simulation_state` — a touching chain
A-B-C, with the middle body removed and a *different* shape spawned
while the proxy slot recycles:

- The removal emits exactly the old pair's Stopped events (in the same
  buffer, per the Rapier-style drain: the next step clears it) and
  leaves A and C with no touching.
- The new body (new rigid index — removed rigid indices stay retired,
  only the broad-phase proxy slot recycles, as pinned by
  `remove_then_add_recycles_the_proxy_slot`) floats alone with zero
  phantom events and no inherited touching for ten steps.
- Dropped onto A, it collides normally with fresh events.

Identity invariant holds: no path leaks the old body's contact, event,
or island state into the new one.

## 11. Backend Consistency

`backend_choice_preserves_collision_lifecycle` — the identical scripted
scenario (fall, settle, get hit by a rolling ball, settle) on the
default AxisSweep3 world and on `PhysicsWorld::with_broadphase(Dbvt)`:

- Identical Started and Stopped counts.
- Final poses within 1e-3 tolerance.

`state_machine_matches_across_backends` — the **full** deterministic
state machine (§15's entire scripted lifecycle) also runs on both
backends: identical Started and Stopped counts, identical activation
states, and final poses within tolerance. The two backends — each
individually audited (05, 06) — present the same candidate set to the
world (07 §3), and the full simulation state downstream of the pair
list agrees across the entire lifecycle, not just the short scripted
scenario.

## 12. Multibody × Rigid

`multibody_rigid_lifecycle_is_consistent` — a rigid box settles onto a
fixed-base multibody's base collider:

- Contact establishes with the canonical event identity
  (`Body` ↔ `Link`); removing the box emits exactly one Stopped in the
  same buffer; ten clean steps follow; a recreated box re-establishes
  contact normally.
- Proxy ownership, collider AABB and island participation were covered
  by the multibody-pass suite (`link_collider_registers_and_updates_aabb`
  and friends); this round verified the lifecycle on top of them.

## 13. Softbody × Rigid

`soft_rigid_remove_recreate_is_consistent` — a rope drops onto a static
box:

- The soft-rigid pair reports Started with the canonical identity
  (`Body` ↔ `Soft`).
- Removing the rigid partner ends the pair cleanly (one Stopped in the
  same buffer, then an empty buffer), with no stale soft-side state.
- A recreated box under the rope re-establishes contact with a fresh
  Started.

Audit 07's soft-removal fixes (metadata-before-shift, joint renumbering)
were exercised through this scenario without regression.

## 14. Teleport / Large Motion

`teleport_does_not_leave_stale_contact` — a resting box teleported 50 m
away and back:

- Away: exactly one Stopped, empty touching, no stale horizontal impulse
  (velocity stays a clean gravity-only free fall).
- Back into the resting pose: contact re-establishes with a fresh
  Started.
- The old manifold does not survive the discontinuity; `set_body_transform`
  refreshes the proxy immediately (07), so the pair simply ends.

## 15. Stateful Deterministic Testing

`deterministic_state_machine_is_repeatable` — the full scripted
lifecycle (240-step settle → sleep → wake-by-hit → 120 steps → filter
off → 30 steps → filter on + teleport → 60 steps → remove → recreate →
360 steps to re-sleep) run **twice** from the same initial state:

- Identical Started and Stopped counts.
- Identical final per-body snapshots (activation state, position,
  linear velocity — bit-level within the binary).

No hidden order dependence and no nondeterministic state surfaced.

## 16. Findings

### M-1 — Zero-link fixed-base multibody panics on the step path

- **ID**: M-1
- **Severity**: P3 (edge-case correctness; degenerate API configuration)
- **Status**: CONFIRMED → RESOLVED (this round)
- **Scenario**: `MultiBody::new(0, ..., fixed_base = true)` with a base
  collider, stepped in a world (a legitimate "static ball" configuration
  a consumer can plausibly write).
- **Trigger**: any step that reads base velocities (forward kinematics /
  integration) for the fixed-base body.
- **Observed**: index-out-of-bounds panic in `base_vel()`
  (`multibody/body.rs:553-560`) — the generalized-velocity vector is
  empty for a zero-link fixed base, and the getters indexed `[0..3]`
  unconditionally.
- **Expected**: a fixed base has no base degrees of freedom; its base
  velocity is always zero.
- **Root cause**: the getters assumed base components always exist.
- **Evidence**: the `multibody_rigid_lifecycle_is_consistent` scenario
  panicked at `body.rs:555` before the fix and passes after.
- **Fix**: `base_vel()` / `base_omega()` return `Vector3::ZERO` when
  `fixed_base` is set.
- **Regression test**: covered inside
  `multibody_rigid_lifecycle_is_consistent` (the scenario panics without
  the fix).

**No other integrated correctness findings.** Observations that could be
mistaken for bugs were each traced to a documented deviation (predicted-
pose detection), a documented API semantic (the event-buffer drain on
remove), or correct engine behavior outside the scenario's assumptions
(single-sided plane pass-through; micro-bounce event flips; the sphere
margin band equaling the diameter, so a filter-off separation step is
enough to leave the band).

## 17. Regression Tests

Eleven tests joined the suite this round (all in `world.rs` unless
noted):

| Test | Group |
|---|---|
| `deep_penetration_settles_without_nonfinite_state` | §5 |
| `extreme_mass_ratio_remains_bounded` | §6 |
| `ccd_clamp_then_normal_contact_settles_and_sleeps` | §7 |
| `sleep_wake_preserves_contact_lifecycle` | §8 |
| `filter_flip_clears_and_recreates_contact_state` | §9 |
| `remove_recreate_does_not_retain_simulation_state` | §10 |
| `backend_choice_preserves_collision_lifecycle` | §11 |
| `multibody_rigid_lifecycle_is_consistent` | §12 (+ M-1 fix) |
| `soft_rigid_remove_recreate_is_consistent` | §13 |
| `teleport_does_not_leave_stale_contact` | §14 |
| `deterministic_state_machine_is_repeatable` | §15 |
| `state_machine_matches_across_backends` | §15 × §11 (full machine on both backends) |

All deterministic (fixed dt, scripted operations, no wall clock, no
inter-test dependencies), f32/f64-clean, public-API only, each pinning a
named invariant.

Current suite totals (measured): **448 tests** — collision 222,
dynamics 220, math 6 — green on **f32** and **f64**. Gates: clippy clean
on both features; no_std check clean (`--no-default-features --features
libm,f32|f64`); `python tools/check.py` green (tests + 14 demos).

## 18. Parity

All eight ledger scenes were re-run after the changes and reproduce the
ledger numbers **bit-for-bit** at f32 (ccd additionally at f64):

falling_box (0.155333 @ 56, 10 exceed), falling_capsule (0.110625 @ 49 /
0.165274 @ 62), stacking (0.003114/0.008280/0.011092/0.017019, 548+575
exceed), mesh (0.155324 @ 81), ccd (0.000075 @ 0, exceed 0 both
precisions), sleeping (0.023872 … 0.392783 @ 270), composite_consumer
(all zeros), rolling (ball_rf 0.000030, ball_plain/ball_spin in their
classified classes). Every exceedance falls inside the ledger's existing
classification table. This is **zero regression against the ledger**,
which is distinct from — and does not claim — "Rust == Bullet".

## 19. Performance

Not a performance audit. No benchmark matrix was re-run; the only
production-code change is the M-1 branch (two constant-time flag checks
off the hot path's data-dependent flow, only reachable for fixed-base
multibodies). No performance anomaly was observed during the rounds.

## 20. Remaining Risks

- The deep-penetration recovery bound is geometry-dependent (one-sided
  planes stop colliding past full pass-through, like Bullet); consumers
  spawning bodies deeply inside one-sided geometry have no recovery
  guarantee in either engine.
- Link-collider and soft-body CCD (as sweep *targets*) remain
  unimplemented — the documented implementation difference (08 §14);
  an integrated scenario with a fast body tunneling a soft body would
  rely on the normal pipeline only.

## 21. Conclusion

**Audit 09 — Full Integrated Adversarial Pass: COMPLETE.** The audited
subsystems hold their contracts when run together: filter flips tear
down and rebuild state coherently across all four layers, removals and
recreations preserve identity, CCD hands off cleanly to the normal
pipeline, both broad phases drive identical simulations, deep
penetration and extreme mass ratios stay finite and bounded, and the
full scripted lifecycle is bit-repeatable. One real edge-case panic
(zero-link fixed-base multibody) was found and fixed. All eleven new
adversarial tests are in the suite (447 total, f32/f64 green), parity is
at the ledger, and the audit phase's integrated pass is done — the
remaining 1.0 items are the minimal debug draw and the API freeze
review.
