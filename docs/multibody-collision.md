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
over: contact manifolds between two non-static colliders, rigid joints,
adjacent non-static link colliders of one multibody, and multibody
constraints (tagA/tagB). Per-island row pools replace the global solve; this
also upgrades the rigid-body side to true island semantics, closing the
simplified wake-up deviation for both worlds. Solved per island in island
order (Bullet sorts for determinism); sleeping islands are skipped.

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
- **Phase D — islands.** Union-find, per-island solving, island-driven
  sleeping for both rigid bodies and multibodies. Acceptance: sleeping scene
  parity unchanged; wake-up propagation tests updated to island semantics.
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
