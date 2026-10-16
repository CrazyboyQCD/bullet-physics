# Audit 07 — BroadPhase Integration

- **Date**: audited & fixed 2026-09-08; record archived 2026-09-09
- **HEAD at fix round**: `aafdb08`; record written at `da6d3f9`
- **Scope**: the seam between the broad phases and the simulation pipeline —
  backend dispatch, proxy lifecycle, `compute_pairs` consumption,
  simulation islands, collision filtering, the narrowphase/manifold/contact
  lifecycle, solver inputs, collision events, constraint collision
  suppression, `DisableSimulation`, and proxy reuse.
- **References**: `refs/bullet3/src/BulletCollision/BroadphaseCollision/`
  (`btOverlappingPairCache.{h,cpp}`, `btAxisSweep3Internal.h`,
  `btDbvtBroadphase.{h,cpp}`),
  `BulletCollision/CollisionDispatch/` (`btCollisionDispatcher.{h,cpp}`,
  `btCollisionObject.{h,cpp}`, `btSimulationIslandManager.cpp`),
  `BulletDynamics/Dynamics/btDiscreteDynamicsWorld.cpp`.
- **Status: COMPLETE**
- **Verdict**: ✅ **Audit-complete.** The backend-internal audits (05
  AxisSweep3, 06 DBVT, 01 Manifold) stand on their own; this round audited
  the *integration* contracts and closed the four findings raised by the
  external review round — I-01/I-03 correctness (P1), I-04 correctness (P2),
  and I-02, a parity deviation (P3) resolved by alignment. All fixes are in
  `aafdb08`, pinned by ten in-domain regression tests. Parity 8/8 scenes are
  bit-identical to the pre-fix ledger; performance returned to baseline
  after a mid-round filter-cost excursion.

> Reading guide: this record is the final state of the round, not a process
> log. "Zero regression" below means *bit-identical to the existing parity
> ledger*, not "Rust equals Bullet" — the project's documented deviations
> (`docs/bullet-deviations.md`) remain the authoritative difference list and
> are not re-opened here.

## 1. Scope

This round covered the integration semantics of the broad phases inside
`PhysicsWorld`:

- **AxisSweep3** and **DBVT** — audited standalone in `05-axis-sweep3.md`
  and `06-dbvt.md`; their internal algorithms are **not** re-analyzed here.
  This round only checks the contract each backend presents to the world
  when dispatched through `BroadPhaseBackend`.
- **BroadPhase trait / backend dispatch** — `BroadPhaseBackend`
  (`AxisSweep3` / `Dbvt`) enum dispatch at the world level.
- **`PhysicsWorld` proxy ownership** — the unified proxy namespace over
  `Body` / `Collider` (multibody link/base) / `SoftBody` proxies.
- **body / collider / soft proxy lifecycle** — create, owner assignment,
  `set_aabb`, destroy, free-slot recycling, owner reassignment.
- **`step_simulation` pair consumption** — how the raw broad-phase pair
  list feeds islands, the narrow phase, and the persistent-manifold cache.
- **Simulation islands** — union-find membership, sleep/wake propagation,
  the moving-kinematic wake.
- **Collision filters** — group/mask, the custom pair filter, sensors
  (`NO_CONTACT_RESPONSE`), constraint suppression.
- **Persistent manifold lifecycle** — touch/untouch as observed through the
  pair pass (manifold internals audited in `01-manifold.md`).
- **Multibody collider dispatch** and **soft-body dispatch**.
- **Collision events** — tracked-pair diffing and `Stopped` identity.
- **Constraint collision suppression** — the `no_collision_pairs` set.
- **`DisableSimulation`** island membership.
- **Proxy reuse** after body removal.

## 2. The BroadPhase integration contract

The invariant adopted for this audit:

```
compute_pairs()
→ returns stored proxy AABB overlap candidates
```

- **False positives are allowed** — the broad phase may report an AABB
  overlap that the shapes themselves reject; the narrow phase filters it.
- **False negatives are forbidden** — a stored-proxy-AABB overlap that
  `compute_pairs` fails to report is a correctness bug.

The single most important conclusion of this round:

> **This pair contract is not, by itself, simulation-island eligibility.**
> Island union must consume only *collision-eligible* pairs — the raw
> broad-phase overlap set contains pairs (group/mask-excluded, pair-filter
> rejected, sensor) that overlap in space but must not merge islands. The
> old code fed raw pairs straight into `step_islands`, which is finding
> I-01 below.

## 3. Backend consistency

The two backends now present the same final semantics to the world:

| Backend | Pair cache | Retention | Output filter |
|---|---|---|---|
| AxisSweep3 | exact stored-AABB overlap pairs | pair removed when the stored AABBs separate | exact stored-AABB |
| DBVT | fat-volume pairs | **fat-overlap retention** (pair survives while the leaf volumes overlap) | exact stored-AABB |

Both therefore deliver, through `compute_pairs()`, the same thing to the
world: **the current stored-proxy-AABB overlaps**. DBVT's fat-overlap
retention criterion was fixed in its standalone audit (`ef25fa9`, exact
AABB → fat leaf volume, matching Bullet's cleanup pass); that fix is *not*
re-listed here — this round only verified the world observes identical
candidate semantics from either backend, with the exact filtering
happening downstream.

## 4. Proxy lifecycle

All collision proxies live in one slot namespace resolved through
`ProxyOwner`:

```
ProxyOwner::Body(usize)      — rigid body
ProxyOwner::Collider(usize)  — multibody base/link collider
ProxyOwner::Soft(usize)      — whole soft body (one proxy over its node hull)
ProxyOwner::Free             — destroyed slot awaiting reuse
```

Verified lifecycle: `create_proxy` → owner assignment → `set_aabb` updates
→ `destroy` → the slot returns to `Free` → a later `create_proxy` recycles
the slot with a reassigned owner. Body removal cascades proxies, joints,
multibody constraints, and tracked event pairs, and keeps remaining indices
stable (tombstone-free slot renumbering for soft bodies, see I-03).

Existing regression coverage (test names as they exist in the tree):

- removed body: `removed_body_stops_colliding_and_falls`,
  `removed_body_is_invisible_to_queries`,
  `removed_body_keeps_remaining_indices_stable`,
  `removed_body_does_not_block_ccd`,
  `remove_body_cascades_joints`,
  `remove_body_cascades_multibody_constraints` (`world.rs`)
- proxy slot reuse: `remove_then_add_recycles_the_proxy_slot` (`world.rs`)
- multibody collider registration/update:
  `link_collider_registers_and_updates_aabb`,
  `sleeping_multibody_keeps_collider_aabb` (`world/multibody_pass.rs`)
- soft-body proxy lifecycle: `soft_body_respects_collision_mask`
  (`world/softbody_pass.rs`) plus the soft-removal tests of §5.3.

This is the coverage that exists today; multibody *link* registration has
a dedicated test, soft-body proxies are covered through the mask and
removal tests — no broader claim is made.

## 5. Findings from the external review round

The external review of the island/removal/constraint paths raised four
findings. All four were closed in `aafdb08` and each is pinned by
in-domain regression tests.

### I-01 — Filtered pair incorrectly consumed by island union

**CONFIRMED · P1 · RESOLVED**

*Original problem.* The old flow fed **raw** broad-phase pairs into island
union:

```
raw BroadPhase pairs → step_islands() → collision filtering (too late)
```

so a pair excluded by group/mask or the pair filter — or touching a sensor
— still united islands. Consequences: a static scene could be pinned awake
forever by an excluded neighbor's overlap; wrong sleep/island membership;
wrong moving-kinematic wake propagation.

*Bullet reference.* Bullet does **not** re-check `needsCollision` inside
`findUnions()`. The correct chain is:

```
addOverlappingPair()
→ needsBroadphaseCollision()      (btOverlappingPairCache.cpp:473, :490;
                                   btOverlappingPairCache.h:108)
→ collision group/mask + overlap filter
→ a rejected pair never enters the overlapping pair cache at all

findUnions()                       (btSimulationIslandManager.cpp:40)
→ consumes the already-filtered pair cache
→ additionally gates on mergesSimulationIslands() (:55-56), which
   excludes static/kinematic and NO_CONTACT_RESPONSE objects
```

`no_collision_pairs` — the constraint-suppression set — is **not** part of
this broadphase-level filtering: suppression lives in the pair cache (the
pairs exist) and `findUnions` never consults
`checkCollideWithOverride`, so constraint-suppressed pairs deliberately
still merge islands.

*Rust fix.* Island union (and the moving-kinematic wake) now consume a
single `pair_eligible` test — group/mask, the custom pair filter, and the
`mergesSimulationIslands` sensor exclusion — before uniting; constraint
suppression is deliberately *not* an eligibility condition. A per-proxy
eligibility byte (pooled in `IslandScratch`) keeps the all-default common
case at one byte test per side, with the full walk only for proxies that
deviate from the defaults; the kinematic pretest runs before the
eligibility check.

*Tests* (`world/islands.rs`): `group_mask_excluded_pair_does_not_pin_island_awake`,
`sensor_body_does_not_pin_island_awake`,
`moving_kinematic_does_not_wake_group_excluded_body`,
`disable_simulation_body_bridges_islands` (control for the I-02
alignment). The custom pair filter rides the same eligibility closure on
the shared code path; the round's dedicated island regressions cover the
group/mask, sensor, kinematic-wake and bridging dimensions. Sleep/wake
behavior remains covered by the pre-existing island suite. Residual
per-body-vs-per-island wake differences are recorded in
`bullet-deviations.md` §1 "Islands: gating & kinematic wake" — documented
deviation, not a new finding.

### I-02 — DisableSimulation island membership

**NOT A BUG · P3 PARITY DEVIATION · RESOLVED BY ALIGNMENT**

*External review finding.* Bullet's actual behavior: a
`DISABLE_SIMULATION` body has `isActive() == false` and gets **no position
integration**, but it can still take part in manifolds and the solver, and
its object **still merges simulation islands** — the island tag pass only
skips static/kinematic (`btSimulationIslandManager` unions through any
non-static object), so a `DisableSimulation` body can bridge its
neighbors' islands.

The Rust `participates_in_islands()` used to exclude `DisableSimulation`
bodies, so they could not serve as island bridges. This is a *parity
deviation*, not a correctness bug in either direction — but the aligned
behavior is the reference behavior.

*Rust fix.* Island membership now includes `DisableSimulation` bodies
(they bridge), while `go_to_sleep` gained the `setActivationState`-style
guard so the island pass never rewrites the body's own activation state —
a `DisableSimulation`/`DisableDeactivation` body is never woken or put to
sleep by the pass.

**Explicitly recorded:** the solver writing impulses into a
`DisableSimulation` body's velocity while its position is never
integrated is **Bullet's real behavior**, not a Rust bug. It is pinned as
such by `disable_simulation_body_takes_solver_impulses_without_integrating`
(`world/step.rs`).

### I-03 — remove_soft_body stale ContactKey / index lifecycle

**CONFIRMED · P1 · RESOLVED**

*Original problem.* `remove_soft_body` resolved the ended pairs' event
metadata *after* `soft_bodies.remove(index)`:

```
removed = ContactKey::soft(index)
soft_bodies.remove(index)
… ended-pair handling calls pair_opted_in() / pair_sensor()
  indexing the post-removal list with the pre-removal key
```

After a middle removal every subsequent soft index shifts left, so the
stale key either read the wrong body's metadata or fell off the end:
last-slot removal **panicked** (index out of bounds); middle removals
silently **dropped or mis-attributed** `Stopped` events. The same class
was latent in cross-soft joints — a `SoftJoint`'s `Side::Cluster { soft }`
indices would point at the wrong body after renumbering.

*Rust fix.* The ended pairs' opted-in/sensor metadata is resolved
**before** the list removal; then `remove`, then joint/`ContactKey`
renumbering, then event emission. Cross-soft joints renumber with the
removal; joints anchored to the removed body die with it.

*Tests* (`world.rs`): `remove_last_soft_body_with_tracked_touch_emits_stopped`
(last-soft removal: the panic is gone, `Stopped` still emitted with
correct identity),
`remove_middle_soft_body_emits_stopped` (middle removal: `Stopped`
identity is correct, no drop / no mis-attribution),
`remove_soft_body_renumbers_cross_soft_joint_sides` (cross-soft joint
indices are correct after renumbering).

### I-04 — Collision suppression cache incoherent with the Constraint setter

**CONFIRMED · P2 · RESOLVED**

*Original problem.* `Constraint::set_disable_collisions()` can flip
suppression at runtime, but `no_collision_pairs` was a separate cache
maintained incrementally at `add_constraint` / `remove_constraint` /
`remove_body` — and **not** by the setter. So `true → false` left stale
suppression behind, and `false → true` never took effect.

*Bullet reference.* Bullet's runtime semantics are not a live-read of a
`m_disableCollisionsBetweenLinkedBodies` field across the pipeline:
`setIgnoreCollisionCheck` (`btCollisionObject.h:238`) records pairs on
each rigid body's ignore list, and the collision check queries that list
live. The actual Rust bug is therefore the mismatch between the API's
promised mutable-setter semantics and the derived `no_collision_pairs`
cache.

*Rust fix.* `no_collision_pairs` is re-derived from the constraint list at
the start of every step (`rebuild_no_collision_pairs`, called from the
step prologue) — the cached equivalent of Bullet resolving suppression
live at pair processing — replacing the three incremental maintenance
points. Runtime flips now take effect on the next step in both directions.
Additionally, the `Constraint::set_enabled` documentation was corrected:
disabled constraints still suppress collisions between their bodies
(Bullet's ignore list is independent of `enabled`); the old doc text
claimed otherwise.

*Tests* (`world/step.rs`): `constraint_suppression_flips_take_effect_next_step`
(both flip directions, next-step effect),
`disabled_constraint_still_suppresses_collisions` (suppression
independent of `enabled`).

## 6. Regression suite

Ten regression tests joined the workspace suite in this round — 4 islands,
3 step, 3 world, each in its domain file's `#[cfg(test)]` module:

| Domain | Tests |
|---|---|
| `world/islands.rs` (4) | `group_mask_excluded_pair_does_not_pin_island_awake`, `sensor_body_does_not_pin_island_awake`, `moving_kinematic_does_not_wake_group_excluded_body`, `disable_simulation_body_bridges_islands` |
| `world/step.rs` (3) | `disable_simulation_body_takes_solver_impulses_without_integrating`, `constraint_suppression_flips_take_effect_next_step`, `disabled_constraint_still_suppresses_collisions` |
| `world.rs` (3) | `remove_last_soft_body_with_tracked_touch_emits_stopped`, `remove_middle_soft_body_emits_stopped`, `remove_soft_body_renumbers_cross_soft_joint_sides` |

Suite totals, measured at `da6d3f9` (the round's ten included):

```
collision 222 + dynamics 199 + math 6 = 427 tests
full test suite: PASS   (f32: PASS, f64: PASS)
```

(Pre-round baseline: 417.) Toolchain gates at the fix round: `cargo clippy`
clean on both features, `cargo check -p bullet-physics --no-default-features
--features libm` clean for f32/f64 (no_std), `python tools/check.py` green
(demos + assertions), parity fresh.

## 7. Parity validation

After the fixes, all **8/8 parity scenes** reproduce the existing ledger
numbers **bit-for bit** (`docs/bullet-compare.md`).

"Zero regression" means precisely: the recorded per-frame numbers and
exceedance counts match the pre-fix ledger exactly for both f32 and f64.
It does **not** mean Rust == Bullet — the ledger's known deviation classes
(e.g. the predicted-pose contact transient, the persistent-manifold
rest-height trade-off, recorded in `docs/bullet-deviations.md` §1/§2) are
pre-existing, classified, and unchanged by this round. They are not
re-opened here.

## 8. Performance validation

`stack_1000`, the scene that exercises the island pass hardest (1000
bodies, most sleeping):

- The first filtering implementation ran the full eligibility walk per raw
  pair per step: the islands stage grew **9.2% → 24%** of the frame
  (flamegraph-attributed) and the Rust median moved **0.892 → 1.074
  ms/frame** (~20% Rust-side slowdown).
- The fix: per-proxy eligibility bytes pooled in `IslandScratch` (the
  all-default case is one byte test per side, with the full walk only for
  proxies deviating from defaults) and the kinematic pretest moved ahead
  of the eligibility check.
- Final state: islands back to **12.1%** of the frame, Rust median
  **0.891 ms/frame** — equal to the ledger row (0.892 ms).

Paired measurements (default repeats, medians) around the fix: **0.66x /
0.67x / 0.65x** vs the Bullet binary. The Bullet-side denominator drifted
between runs (1.33–1.45 ms, environment); attribution follows the ledger
discipline — Rust-side absolute medians against the ledger row — and that
denominator drift is **not** a Rust regression. Conclusion: **no lasting
performance regression** from the round.

## 9. Finding table

| ID | Finding | Severity | Status |
|---|---|---|---|
| I-01 | Filtered broad-phase pair contaminates island unions (raw pairs fed to `step_islands`) | P1 | RESOLVED |
| I-02 | `DisableSimulation` island membership differs from Bullet | P3 parity | RESOLVED (aligned) |
| I-03 | `remove_soft_body` stale `ContactKey` / soft-index lifecycle (panic / lost `Stopped`) | P1 | RESOLVED |
| I-04 | Collision-suppression cache incoherent with the `Constraint` setter | P2 | RESOLVED |

## 10. Conclusion

**07 BroadPhase Integration — COMPLETE.**

- AxisSweep3 / DBVT backend contracts verified through the world-level
  dispatch: both deliver the same stored-proxy-AABB overlap candidates,
  with exact filtering downstream.
- The proxy lifecycle (unified `ProxyOwner` namespace, slot recycling,
  removal cascades) is covered by the existing regression suite.
- The filtering → islands → narrowphase chain is closed: island union and
  the moving-kinematic wake now consume collision-eligible pairs only
  (I-01), matching Bullet's pair-creation-time filtering.
- The soft-body removal lifecycle bug (stale index after list shift) is
  fixed with metadata resolved pre-removal plus joint renumbering (I-03).
- Collision suppression is re-derived per step and coherent with the
  runtime setter; the `set_enabled` doc contradiction is corrected (I-04).
- `DisableSimulation` island membership is aligned with Bullet, with its
  genuine upstream quirk (solver impulses without integration) pinned as
  behavior, not fixed as a bug (I-02).
- Ten regression tests joined the suite (427 total, f32/f64 green);
  parity 8/8 bit-identical to the ledger; performance back at baseline.

Residual documented differences live in `docs/bullet-deviations.md`
("Islands: gating & kinematic wake") and are accepted deviations, not open
findings.

## 11. Git references

- `aafdb08` — `fix(dynamics): close four external-audit findings`
- `da6d3f9` — `docs: record the external-audit fix round`

Both commits are on `origin/main` at the time of this record
(`da6d3f9 == origin/main`).
