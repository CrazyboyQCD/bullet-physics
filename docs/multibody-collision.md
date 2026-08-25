# Multibody Collision Integration — Design

Design for porting `btMultiBodyLinkCollider` and the multibody collision
pipeline: Featherstone multibodies colliding with rigid bodies and with each
other, solved together with rigid-body contacts in one world (the
`btMultiBodyDynamicsWorld` architecture). Written against `refs/bullet3`
sources on 2026-08-24; implementation lands in phases (§6).

## 1. Goal and scope

**In scope**
- Link/base colliders on a `MultiBody` entering the existing
  broadphase/narrowphase pipeline.
- Contact rows between (multibody link ↔ rigid body) and (multibody link ↔
  multibody link), solved in the same PGS loop as rigid contacts and joints.
- A fused world: `PhysicsWorld` owns multibodies and multibody constraints;
  `MultiBodyWorld` is retired.
- Simulation islands (union-find) covering rigid bodies, link colliders and
  multibody constraints, replacing the simplified wake-up propagation.

**Out of scope (later or never)**
- Articulated warm starting (`SOLVER_USE_ARTICULATED_WARMSTARTING`, off by
  default in Bullet) and `btMultiBodyJointFeedback` — follow-ups.
- Split impulse for multibody contacts — Bullet does not implement it either
  (the branch is commented out in `setupMultiBodyContactConstraint`).
- MLCP solver, RK4, kinematic links — unchanged roadmap items.

## 2. Bullet reference pipeline

`btMultiBodyDynamicsWorld` (inherits `btDiscreteDynamicsWorld`) overrides the
step as: `solveConstraints` = `solveExternalForces` (forward kinematics →
clear constraint forces → gravity per base/link → ABA) → `buildIslands` →
`solveInternalConstraints` (the island callback's `processConstraints` runs
`btMultiBodyConstraintSolver::solveMultiBodyGroup`, then a joint-feedback ABA
pass (not needed for us) and `processDeltaVeeMultiDof2`); `integrateTransforms`
adds `integrateMultiBodyTransforms` (`addSplitV` → `stepPositionsMultiDof` →
`updateCollisionObjectWorldTransforms` writes the link colliders' world
transforms → `substractSplitV`; sleeping bodies get `clearVelocities`).

Key source files:
- `Featherstone/btMultiBodyLinkCollider.h` — a `btCollisionObject` plus
  `{ m_multiBody, m_link }` (`m_link = -1` for the base) and
  `checkCollideWithOverride` (self-collision filtering, §4.6).
- `Featherstone/btMultiBodyDynamicsWorld.cpp` — `calculateSimulationIslands`
  (island merging, §4.8), `solveExternalForces`, `integrateMultiBodyTransforms`.
- `BulletCollision/CollisionDispatch/btSimulationIslandManager.cpp` +
  `btUnionFind.{h,cpp}` — the union-find, the island sleep/wake pass
  (`buildIslands`) and the per-island batching (`processIslands`) that phase D
  ports (detailed design §9).
- `Featherstone/btMultiBodyInplaceSolverIslandCallback.h` — per-island
  batching into `solveMultiBodyGroup`.
- `Featherstone/btMultiBodyConstraintSolver.cpp` — `convertMultiBodyContact`,
  `setupMultiBodyContactConstraint`, `resolveSingleConstraintRowGeneric`,
  `writeBackSolverBodyToMultiBody`.

## 3. Contact rows (the `btMultiBodySolverConstraint` recipe)

Per contact point, one normal row plus two friction rows (we use two friction
directions engine-wide; recorded deviation from Bullet's default one):

1. **Jacobian**: for each multibody side, `fillContactJacobianMultiDof` at the
   contact position along the normal (already ported as
   `MultiBody::fill_constraint_jacobian`); for a rigid side, the classic
   `{ normal, rel_pos × normal }` row. Sign convention: the B-side jacobian is
   filled along `-normal`.
2. **Unit-impulse response**: `calcAccelerationDeltasMultiDof` (already
   ported) into `deltaVelocitiesUnitImpulse`; rigid side uses
   `invMass + n·(I⁻¹(r×n) × r)`.
3. **Effective mass**: `jacDiagABInv = sor / (Σᵢ jacᵢ·deltaᵢ + cfm)` with
   `cfm = m_globalCfm/dt = 0`, `sor = 1`; singular rows get 0 (disabled).
4. **RHS** (`erp = m_erp2 = 0.2`, same as our contact ERP): with
   `distance = penetration + slop` (our 0.005 slop, engine-wide deviation),
   `velocityError = restitutionCurve(rel_vel) − rel_vel`;
   `distance > 0 → velocityError −= distance/dt` else
   `positionalError = −distance·erp/dt`; `rhs = (pos+vel)·jacDiagABInv`.
   `rel_vel` sums `vᵢ·jacᵢ` over the multibody velocity vector
   (`velocities[0..6+dofs]`, the same buffer ABA produces) and `J·v` on rigid
   sides. **No split impulse**: position and velocity terms are always merged
   (Bullet's split branch is disabled for multibodies). Normal row limits
   `[0, 1e10]`.
5. **Friction rows**: same structure along `btPlaneSpace1(normal)`; limits
   `±μ·normalImpulse` refreshed from the paired normal row each iteration
   (`m_friction` from the combined friction, product combination like rigid
   contacts); no positional term (no friction anchors).
6. **Iteration** (`resolveSingleConstraintRowGeneric`):
   `Δλ = rhs − λ·cfm − (Δv_A·jac_A + Δv_B·jac_B)·jacDiagABInv`, clamp to
   `[lower, upper]`, then apply the impulse. Multibody sides accumulate into
   the shared `deltaVelocities` block keyed by the body's `companion_id`
   (`applyDeltaVee` helper); rigid sides apply to the body velocity directly
   (our solver mutates real velocities, which is what Bullet's
   `internalApplyImpulse` does on solver bodies). The **multibody real
   velocities are not touched during iterations** (Bullet compiles without
   `DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS`).
7. **Writeback** (`writeBackSolverBodyToMultiBody`): once after the loop, each
   row applies `deltaVelocitiesUnitImpulse · appliedImpulse` via
   `applyDeltaVeeMultiDof` (with the ±maxCoordinateVelocity clamp — our
   `apply_delta_vee`), records `appliedImpulse` on the owning constraint (for
   `MultiBodyConstraint` rows), and accumulates `normal·λ/dt` /
   `(r×n)·λ/dt` into the per-link/base constraint forces (the input of
   Bullet's joint-feedback pass; we keep the force bookkeeping so the later
   feedback feature has its input, but no pass consumes it yet).
8. **Warm starting**: off by default (`m_appliedImpulse = 0`), like our
   multibody constraints. Contact impulses are written back into the manifold
   in Bullet for the persistent-manifold warm start; we keep our per-step
   normal-impulse cache for rigid-rigid only and start multibody contact rows
   from zero (deviation §D5).

Solver iteration order per PGS sweep (matching `solveSingleIteration`):
rigid contact + joint rows first, then multibody normal rows, then multibody
friction rows (limits depend on the same-sweep normal impulses), then the
multibody constraint (non-contact) rows.

## 4. Design decisions

**D1 — Fused world.** `PhysicsWorld` gains `multibodies: Vec<MultiBody>`,
multibody constraints, and the multibody stepping stages inline; the
multibody-only `MultiBodyWorld` is deleted after migration (demo, probe and
tests move over). Step order (merging our parity-verified rigid order with
Bullet's multibody inserts):

1. rigid: integrate velocities; multibody: gravity + FK + clear constraint
   forces + ABA (`solveExternalForces`)
2. rigid: predicted positions
3. broadphase AABB refresh — rigid from predicted poses (existing deviation),
   link colliders from the FK world transforms (Bullet detects on current
   poses; no predicted multibody pose)
4. pairs → narrowphase → manifolds (incl. link colliders) with self-collision
   filtering
5. row assembly: rigid contacts (warm start) + rigid joints + multibody
   contacts + multibody constraints
6. wake-up propagation over contact and constraint pairs (Phase A–C keeps the
   simplified one-hop rule; Phase D replaces it with islands)
7. PGS loop (§3.8)
8. writeback: rigid impulse cache, multibody `apply_delta_vee` +
   `process_delta_vee_2`, applied impulses
9. rigid deactivation check; multibody motion/sleep check
10. position integration: rigid (with CCD clamp + split push), multibody
    `step_positions` + collider world transforms
11. clear forces

**D2 — Link colliders.** `MultiBody` grows per-link and base collider slots
(`Option<CollisionShape>` + friction/restitution + an `ActivationState`
mirroring the rigid-body enum). The world maintains a flat
`link_collider_proxies: Vec<(mb_index, link_index)>` registered in the same
`AxisSweep3` as rigid bodies (proxy ids share one index space:
`proxy < bodies.len()` = rigid body, else link collider). Narrowphase needs
only `(shape, world_pos, world_rot)`; link poses come from the FK cache. No
`dyn`, no separate collision-object type.

**D3 — Contact storage.** Multibody contact points reuse `ContactManifold`
and the same reduction; a new `MultiBodyContactRow` (normal + 2 friction)
lives beside `ContactConstraint`, with either side being
`enum ContactSide { Rigid(usize), Link { mb: usize, link: i32 } }`.

**D4 — Velocity accumulation.** Reuse the existing
`SolverData { jacobians, delta_velocities, delta_velocities_unit_impulse }`
pool and `companion_id` scheme from `MultiBodyWorld` verbatim (it already
mirrors Bullet's `m_data`). Rigid sides keep direct velocity mutation
(equivalent to Bullet's solver-body deltas).

**D5 — Detection pose.** Multibody link colliders detect at the FK (current)
pose like Bullet; rigid bodies keep our predicted-pose detection. The
mixed-pair asymmetry is the same one-step phase difference already recorded
for rigid contacts.

**D6 — Self-collision.** `hasSelfCollision` defaults to **true**
(`btMultiBody` ctor), but link colliders only exist if the user attaches
them, so nothing collides until then. The `checkCollideWithOverride` filter
is ported as a pair filter in the narrowphase loop: same multibody without
self-collision → skip; `DISABLE_PARENT_COLLISION` /
`DISABLE_ALL_PARENT_COLLISION` link flags → skip the flagged ancestor pairs.
Adjacent-parent pairs are **not** skipped by default (matches Bullet).
Colliders on a fixed-base multibody's base are still dynamic colliders (the
base is immobile but its flags decide, as in Bullet).

**D7 — Sleeping.** A multibody sleeps as a unit: `checkMotionAndSleepIfRequired`
on the generalized velocities (already implemented). With islands (D8) the
island sleep state drives the colliders' activation states, and a sleeping
island skips ABA + row assembly entirely. Until then, the one-hop contact
wake-up extends to link colliders: any contact linking an awake island
member wakes the whole multibody.

**D8 — Islands (Phase D).** Port `calculateSimulationIslands` as a union-find
over: broad-phase AABB pairs between two non-static colliders (conservative,
like `findUnions` on the pair cache), rigid joints between two dynamic
bodies, and multibody constraints. Each multibody is one node (its collider
chain merges trivially). The island sleep/wake pass replaces the one-hop
wake-up, and rows of sleeping islands are skipped entirely. The solve itself
stays one global batch (Bullet's `m_minimumSolverBatchSize = 128`
accumulates small islands into a single group anyway — §9.6). Detailed
design in §9.

**D9 — Retired equivalences.** Bullet's `getOrInitSolverBody` pool maps to
our direct body mutation; `setPosUpdated` / interpolation transforms /
predictive contacts for multibodies are not ported (no CCD for link colliders
in the first landing; recorded as not-implemented).

## 5. Bullet → Rust mapping (new pieces)

| Bullet | Rust |
|---|---|
| `btMultiBodyLinkCollider` | `MultiBody` collider slots + `PhysicsWorld` proxy registration |
| `btMultiBodySolverConstraint` | `multibody::contact::MultiBodyContactRow` |
| `convertMultiBodyContact` / `setupMultiBodyContactConstraint` | row assembly in the fused `step_simulation` |
| `resolveSingleConstraintRowGeneric` | shared `resolve_row` extension to `ContactSide` |
| `writeBackSolverBodyToMultiBody` | post-loop `apply_delta_vee` + constraint-force bookkeeping |
| `MultiBodyInplaceSolverIslandCallback` | island batch (Phase D) |
| `calculateSimulationIslands` link/constraint merges | union-find stages (Phase D) |

## 6. Phased landing

- **Phase A — fused world (no behavior change).** Move multibodies +
  constraints into `PhysicsWorld`, delete `MultiBodyWorld`, migrate
  demo/probe/tests. Acceptance: all 176 tests + 8 demos green; multibody
  probe bit-identical to the current output.
- **Phase B — link colliders detect.** Collider slots, proxy registration,
  AABB refresh from FK, narrowphase integration, self-collision filter. No
  solver response yet (manifolds computed and dropped). Acceptance: new unit
  tests for pair filtering + AABB updates; existing scenes unchanged.
- **Phase C — contact response.** `MultiBodyContactRow` (normal + friction),
  mixed PGS loop, writeback, sleeping interactions. New probe scenes (§7).
  Acceptance: probe parity within the tolerances of §7; new unit tests.
- **Phase D — islands.** Union-find, island-driven sleeping and row skipping
  for both rigid bodies and multibodies (detailed design §9; the solve stays
  one batch). Acceptance: sleeping scene parity unchanged; wake-up
  propagation tests updated to island semantics.
- **Phase E — optional follow-ups.** Articulated warm starting,
  `btMultiBodyJointFeedback` (consumes the constraint-force bookkeeping),
  CCD for link colliders.

## 7. Validation plan

Probe scenes (extend `tools/multibody_probe.cpp` + the Rust twin, f32/f64):
1. `link_ground` — pendulum-style multibody with a link sphere collider
   swinging into a static plane box: impulse timing and settle height.
2. `base_ground` — free-base multibody with a box base dropped onto the
   ground: rest height and impulse trace.
3. `mb_pushes_rb` — a motorized link pushes a dynamic rigid box: box
   trajectory + link trajectory.
4. `rb_hits_mb` — a falling dynamic sphere hits a link collider of an
   otherwise still multibody: wake-up + impulse exchange.
5. `mb_mb` — two single-link multibodies colliding (link ↔ link).
6. `self_collision` — two-link chain with self-collision on/off.

Expected agreement classes (per the existing ledger conventions): impulse
transients one frame apart (D5), rest offsets within the slop trade-off, and
chaotic scattering scenes validated qualitatively (first ~15 frames exact).
Plus `python tools/check.py --parity` and the rigid parity ledger must stay
green through every phase (fused-world refactor included).

## 8. Deviations to record when landing

- D5: mixed detection pose asymmetry (rigid predicted vs multibody current).
- No CCD for link colliders (initially), no interpolation transforms.
- Two friction directions on multibody contacts (engine-wide choice).
- No articulated warm starting for contacts or constraints (Bullet default).
- Phase A–C keep the one-hop wake-up simplification (until Phase D).
- Engine-wide slop 0.005 applies to multibody contact rows (Bullet default 0).
- Phase D additions (§9.8): one union-find node per multibody, single solve
  batch, sleeping-island pairs skip the narrow phase, two-phase rigid
  deactivation timing, mb gravity/ABA gating on the `awake` flag.

## 9. Phase D — simulation islands (detailed design)

Port of `btSimulationIslandManager` + `calculateSimulationIslands`:
union-find islands over rigid bodies and multibodies, an island-level
sleep/wake decision, and row skipping for sleeping islands — replacing the
Phase A–C one-hop wake-up propagation. The PGS solve itself stays one
global batch (§9.6).

### 9.1 Node space

- Every **non-static rigid body** is a node. Static bodies are never island
  members (Bullet: no island tag) — their contacts merge nothing.
- Every **multibody** is exactly **one node**. Bullet instead creates one
  node per link/base collider and merges the adjacent-collider chain; the
  chain is connected by construction, so the partitions are equivalent and
  we skip the per-collider bookkeeping.
- Bodies with `DisableSimulation` are excluded from the node set (they are
  not simulated and must not be forced into `IslandSleeping`).

Node ids are transient per step: `0..num_dynamic_bodies` for rigid bodies
(mapped through a per-step `body → node` table since static bodies leave
holes), then one node per multibody.

### 9.2 Merge rules (per step)

1. **Broad-phase AABB pairs** (`findUnions` on the pair cache): any proxy
   pair whose two owners are both dynamic (non-static body, or a collider of
   any multibody) unites their nodes. This is conservative — overlapping
   AABBs merge islands before actual contact — and exactly what Bullet does;
   pairs suppressed by `no_collision_pairs` still merge (they are joint
   partners anyway).
2. **Rigid joints**: an enabled joint between two non-static bodies unites
   them (Bullet unites only if both sides are non-static/kinematic; a joint
   to a static anchor merges nothing).
3. **Multibody constraints**: always unite the two multibody nodes.
   Deviation: Bullet requires `tagA/tagB ≥ 0` (each side has ≥1 registered
   collider) — our collider-less wake keeps the existing tested semantics
   (`body_sleeps_after_rest_and_wakes_via_constraint`).

### 9.3 Island activation pass (replaces stage 5b)

For each island (union-find root), an **awake contributor** is a rigid body
in state `Active` or `DisableDeactivation`, or a multibody with `awake =
true` (Bullet's `buildIslands` pass: anything but `ACTIVE_TAG`/
`DISABLE_DEACTIVATION` counts as sleep-eligible).

- **Any awake member** → every sleeping member wakes: rigid bodies go
  `IslandSleeping → WantsDeactivation` with `deactivation_time = 0` (new
  `RigidBody::island_wake`), multibodies `wake_up()`. The body simulates
  immediately but re-qualifies for sleeping from a zero timer, so a
  briefly-poked island can re-sleep without another full 2 s window.
- **No awake member** → every member sleeps: rigid `go_to_sleep()` (zeroes
  velocities, like Bullet's end-of-step zeroing), multibody `go_to_sleep()`
  (velocities cleared at the integrate stage as today).

The pass runs right after `compute_pairs` (new stage 4b), before the narrow
phase — Bullet's equivalent runs inside `solveConstraints` after collision
detection, and its union rules likewise consume only broad-phase pairs, so
the ordering is equivalent while letting stages 5+ filter on the final
activation states.

### 9.4 Row filtering after the pass

With the decision made, "sleeping island" reduces to per-owner checks:

- **Rigid contact pair**: skip manifold computation and row assembly when
  both bodies are `!is_active()` (post-pass). Bullet solves no manifolds of
  sleeping islands; we additionally skip the narrow phase itself (deviation,
  §9.8).
- **Multibody pending contact**: skip when both sides are inactive — the
  existing Phase C check, now exact because the wake decision already
  happened.
- **Rigid joint rows**: skip when both bodies are `!is_active()` (covers a
  joint to a static anchor whose dynamic side sleeps — its island is the
  single body). Enabled joints of awake islands are flattened as today.
- **Multibody constraint rows**: skip when both multibodies are `!awake`.

Stage 5b (one-hop wake propagation over contacts/joints/mb contacts and
`mb_wake_constraint_linked`) is deleted — subsumed by 4b.

### 9.5 Two-phase rigid deactivation timing

Today stage 6c calls `go_to_sleep()` immediately when the 2 s timer expires.
Phase D splits it like Bullet:

- **End of step N** (6c): `update_deactivation` only — timer expiry sets
  `WantsDeactivation`; the body stays fully simulated for the rest of that
  step (`is_active()` is true for `WantsDeactivation`).
- **Step N+1** (4b): the island pass decides — whole-island rest freezes
  everyone (`IslandSleeping`, rows and integration skipped from this step),
  otherwise `WantsDeactivation` members keep simulating.

This matches Bullet's `updateActivationState` / `buildIslands` split and
removes the current glitch where a body slept at the end of step N, was
re-woken at N+1's stage 5b and then solved with zeroed stale velocities. The
one extra gravity tick before the freeze is Bullet-identical
(`predictUnconstraintMotion` of step N+1 runs before the island pass).

Two verified details (against the real library, `tools/island_probe.cpp`):

- `updateDeactivation` resets a moving body to `ACTIVE_TAG` right where it
  zeroes the timer (`setActivationState(0)` in Bullet) — without this, a
  knocked-away `WANTS_DEACTIVATION` body whose island split would be frozen
  mid-air by the next island pass.
- The island wake itself (`ISLAND_SLEEPING → WANTS_DEACTIVATION`) fires as
  soon as the broad-phase AABBs overlap — one to two frames before the
  actual impact, exactly like Bullet's `findUnions` conservative merge.

### 9.6 The solve stays one batch

Bullet's island callback accumulates islands and only calls the solver when
the accumulated row count exceeds `m_minimumSolverBatchSize = 128`; scenes
below that threshold solve **all islands as one group**, exactly like a
global solve. We therefore keep the single global PGS loop and the current
row order (pairs in broad-phase order, joints in add order):

- currently-green scenes stay bit-identical — islands only change *which*
  rows exist, not their order or math;
- large scenes (>128 rows) would differ from Bullet's batch flushing in row
  order only; splitting batches is a later optional perf item.

### 9.7 Warm-start and cache implications

The contact cache is rebuilt from solved rows only, so pairs of a sleeping
island lose their cached impulses and cold-start on wake (Bullet's
persistent manifolds keep warm-starting through sleep). Observable as a
one-frame impulse transient on wake; acceptable and recorded.

### 9.8 Phase D deviations

- **One node per multibody** (Bullet: per-collider nodes + chain merge) —
  equivalent partitions; enables collider-less constraint wake (tested).
- **Single solve batch** — no per-island batching/sorting; matches Bullet
  below the 128-row batch threshold.
- **Sleeping-island pairs skip the narrow phase** — Bullet still runs
  collision detection for sleeping pairs (only solving is skipped); with the
  per-step impulse cache this only affects warm-start freshness after wake.
- **Two-phase rigid deactivation** (§9.5) — matches Bullet's structure; the
  previous immediate-sleep behavior was the deviation.
- **mb gravity/ABA gating stays on `awake`** — Bullet gates on collider
  `ISLAND_SLEEPING` synced one step later in `updateActivationState`; the
  probe-verified behavior is kept.

### 9.9 Tests and acceptance

- Unit (new): island merging via AABB pair / joint / mb constraint; an
  island sleeps as a unit (resting stack) and stays positionally stable with
  rows skipped; a dynamic body falling onto a sleeping island wakes it;
  `DisableDeactivation` member keeps its island permanently awake.
- Unit (updated): the wake-propagation tests move to island semantics
  (contact wake now island-wide, joint wake unchanged in effect).
- All 185 existing tests + 8 demos green, f32 and f64.
- Parity: all 6 rigid scenes × f32/f64 re-run (`python tools/check.py
  --parity`); `sleeping` is the sensitive scene — expected to stay within its
  recorded tolerances (the impulse-kick path deviations are unchanged). The
  13 existing multibody probe scenes keep every object awake over their
  short runs (no sleeping interactions), so they must stay bit-identical.
  The §7 collider probe scenes landed on 2026-08-25 as six additional probe
  scenes (ledger: `docs/bullet-compare.md` § "Multibody collider probes");
  the recording surfaced and fixed one real bug — the rigid side of a mixed
  contact row double-counted the assembly-time relative velocity (impulse
  ~2.4x) — and classified the remaining first-impact divergences as the
  known margin/detection-threshold trade-offs.
