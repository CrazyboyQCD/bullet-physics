# Project Status

> The single source of truth for the current state of the project. Update
> this file whenever the state changes. Fine-grained evidence lives in the
> ledgers (`docs/bullet-compare.md`, `docs/perf/speed.md`,
> `docs/perf/memory.md`) and the audit
> records (`docs/audit/`).

## Current HEAD

`67e46f3` — *docs: re-record the speed ledger after the soft-contact fix*
(2026-09-10, pushed). The working tree carries the uncommitted on-demand
third batch (additional damping, multi-part meshes, parity scenes
`damped_stack`/`mesh_parts`) plus the 2026-09-11 cluster-joint
drift-split fix (`terminate` parameter order — the split channel had been
silently dead since the first port; `cluster_joints` probe now bit-identical
in f64, regression test pinned).

## Current Phase

**1.0 API freeze (pre-freeze review).** The functional core is complete and
audited: the rigid-body pipeline, Featherstone multibodies with collision
integration, the classic SoftBody system, and the consumer-facing API layer
are all implemented and covered by the test/parity/probe suites. The
subsystem audits (01–09) are complete, the minimal debug draw shipped, and
the consumer-API backlog (`docs/api-priorities.md`) is fully dispatched —
every row is Done or explicitly Won't-do/deferred. The remaining 1.0 item
is the API freeze review (naming / docs / breaking surface).

## Completed

Feature status at this HEAD (evidence for *tested* cases only):

- **Rigid bodies**: 10 convex shapes with analytic inertia + TriangleMesh
  (static BVH) / Heightfield + GImpact (dynamic concave mesh); collision
  groups/masks; kinematic bodies; rolling friction; motion states; body
  removal with constraint/manifold cascade.
- **Broad phase**: AxisSweep3 (default), DbvtBroadPhase, BruteForce —
  `BroadPhaseBackend` enum dispatch.
- **Narrow phase**: GJK distance walk (margin band + margin-core), EPA,
  analytic/SAT fast paths, per-pair persistent manifolds with merge-on-
  capture + anchored refresh (2026-09-01).
- **Solver**: sequential impulse + warm starting + split impulse +
  union-find islands (sleeping/wake propagation) + CCD swept-sphere clamping
  with broadphase candidate pruning.
- **Joints**: Point2Point / Hinge / Fixed / ConeTwist / Slider / Generic6Dof.
- **Multibody**: Featherstone ABA, all constraint classes, link/base
  colliders in the shared pipeline, joint feedback, shared islands.
- **SoftBody**: the complete classic system (rope/cloth/face/tetra/cluster,
  CL_RS/CL_SS/CL_SELF, L/A/C joints, transform API, cut/refine, soft↔mb
  coupling).
- **Vehicle + Character**: `btRaycastVehicle` and
  `btKinematicCharacterController` ports.
- **Consumer API (batch 1 + 2)**: typed handles (breaking), all-hits rays +
  ray filters, per-body gravity, pair filter, triggers +
  `CollisionEvent` (sides generalized to the `CollisionObject` enum:
  Body / Link / Soft), ghost overlap list, compound child transforms,
  motion states, multibody/soft-body event sources (2026-09-07).
- **Consumer API (on-demand rounds, 2026-09-09/10)**: persistent-manifold
  extras (`ContactPoint::lifetime`, `HAS_FRICTION_ANCHOR`,
  `set_contact_point_events`), `btShapeHull` (42-direction table;
  quickhull moved to `bullet-math`), character directional jump,
  anisotropic friction, per-pair contact stiffness/damping (parity scenes
  `anisotropic_friction` / `contact_soft`), additional damping
  (`RigidBody::additional_damping` + builder, parity scene `damped_stack`),
  multi-part triangle meshes (`TriangleMeshPart` / `from_parts`, parity
  scene `mesh_parts`). Cluster core verified against real Bullet
  (`cluster_joints` probe f64 bit-identical after the drift-split fix).
- **Debug draw** (2026-09-09): `debug_primitives(flags, emit)` — five
  read-only domains, `DebugPrim` `#[non_exhaustive]`, invariance contract
  test-pinned.
- **no_std**: whole workspace behind `--no-default-features --features libm`
  (std default bit-identical; libm scalar backend; injectable step clock).
- **Tooling**: parity harness (10 scenes × f32/f64, ledger current),
  multibody/softbody/GImpact probes (24 mb scenes × f32/f64), perf ledger
  (2026-09-11 re-record after the third batch + drift-split fix: wall clock
  in band, memory +0.031 MB attributed), memory-trend harness, flamegraph
  workflow.
- **Quality gates**: 475 unit tests × f32/f64 green, 14 demos, clippy clean
  on both features.

## In Progress

Independent source audit of the correctness-critical subsystems. All six
subsystem records are complete (2026-09-08): Manifold
([`01`](01-manifold.md)), GJK ([`02`](02-gjk.md)), EPA ([`03`](03-epa.md)),
Convex-Convex ([`04`](04-convex-convex.md)), AxisSweep3
([`05`](05-axis-sweep3.md)), DBVT ([`06`](06-dbvt.md)). The records for
02–05 consolidate conclusions that were already fixed by the hardening and
adversarial rounds; no re-audit was performed. GJK/EPA hardening alone
collected 3 reproduced bugs (fixed), the DBVT round 1 (pair-cache cleanup
criterion), all pinned by adversarial suites.

**External-audit fix round (2026-09-08, `aafdb08`)**: an external review
confirmed four findings, all fixed and pinned by ten in-domain regression
tests — (1) island unions consumed unfiltered broad-phase pairs (a
group/mask-excluded or sensor overlap could pin a resting island awake
forever); unions now gate on collision eligibility (pair-cache +
`mergesSimulationIslands` semantics) and the kinematic wake rides the same
gate, with a per-proxy eligibility byte keeping the all-default common case
cost-free; (2) `DisableSimulation` bodies merge islands again (Bullet's tag
pass only skips static/kinematic), bridging neighbors, with their own state
guarded in `go_to_sleep`; (3) `remove_soft_body` resolved its ended-pair
event metadata against the post-removal list — panic on last-slot removal,
silent Stopped loss on middle removals — now resolved pre-shift, with
cross-body soft joints renumbering; (4) `no_collision_pairs` is re-derived
per step instead of incrementally maintained, so runtime
`set_disable_collisions` flips take effect (the `set_enabled` doc claimed
disabled constraints stop suppressing — they do not, matching Bullet).
Verified: tests ×2, clippy ×2, check.py, parity 8/8 bit-identical, no_std,
perf in band (the first island-gating cut cost +20% on stack_1000; fixed by
a per-proxy eligibility fast path, `docs/perf/speed.md`). Residual
semantic note recorded as deviations §1 "Islands: gating & kinematic wake".

**Stage: audits 01–09 complete; on-demand API rounds dispatched.** The
broadphase integration audit landed as
[`07-broadphase-integration.md`](07-broadphase-integration.md), the CCD audit
as [`08-ccd.md`](08-ccd.md) (one confirmed sensor-target bug fixed), and the
full integrated adversarial pass as
[`09-integrated-adversarial.md`](09-integrated-adversarial.md) (one P3
zero-link fixed-base multibody panic fixed; 11 cross-system adversarial
tests; deterministic state machine bit-repeatable). The minimal debug
draw shipped on 2026-09-09 per `docs/design/debug-draw.md`. The on-demand
consumer-API rounds landed 2026-09-09/10 and the third batch (additional
damping, multi-part meshes, cluster-core verification) on 2026-09-10/11 —
the cluster-core round reproduced and fixed the long-standing dead
cluster-joint split channel (`terminate` parameter order; f64 bit-identical
probe, regression test pinned). The fourth batch is in progress (user call,
2026-09-11): contact-force events landed 2026-09-11 (Rapier-style
`ContactForceEvent` drain, no Bullet counterpart — deviations §3);
spinning friction landed 2026-09-11 (the third torsional row around the
contact normal, inside the rolling-friction branch — upstream gate kept
verbatim); only the scaled-shape wrappers remain. After the batch: the
API freeze review. Known residual: the `cluster_pair` CL_SS chaos probe diverges
qualitatively (pre-existing, tracked for a separate round).

## Known Deviations

All recorded in `docs/bullet-deviations.md` (§1 intentional, §2 numerical
defaults). Headlines: predicted-pose contact detection, penetration slop
dead zone, restitution threshold 1.0 vs 0.2, no friction warm starting,
no gyroscopic force on rigid bodies, soft-body exact distance vs
`btSparseSdf` (~85 mm settle offset vs Bullet's stacked approximations),
CCD `m_allowedCcdPenetration` slack not ported, kinematic pose driving,
motion-state interpolation blend, kinematic-driven motion states. These are
**accepted trade-offs**, not defects; each cites its parity classification.

## Known Risks

Items that **need independent verification** (none is currently a confirmed
bug — see the audit policy in [`README.md`](README.md)):

- none — the last queued item (EPA `INSIDE_EPS`/`FALLBACK`) was resolved by
  the EPA audit: both are dead constants in the reference source, defined
  and `#undef`'d without a single use.

Resolved by the audits (2026-09-08): EPA degenerate-polytope paths
(`INSIDE_EPS`/`FALLBACK` not ported — dead constants in the reference;
[`03-epa.md`](03-epa.md) §3); persistent-manifold rotation-drift semantics
(self-authored refresh-guard mechanism, now fully documented incl. the
0.01 rad rotation gate), quaternion q/−q double cover (no correctness
issue; at most one spurious re-detection), contact-point matching
robustness (two policy differences recorded as the point-matching
deviation row), manifold reduction beyond tested scenes (population-policy
deviation row recorded; adversarial inputs deferred to the dedicated
adversarial-testing phase); external-audit round (`aafdb08`): island-union
gating, soft-body removal event indexing, and constraint-suppression cache
coherence (all fixed and test-pinned, see In Progress above).

## 1.0 Required

- Independent source audits of the correctness-critical subsystems —
  **done** for manifold, GJK, EPA, convex-convex, AxisSweep3, DBVT
  (records 01–06), broadphase integration (07) and CCD (08).
- Adversarial correctness testing (deep penetration, extreme mass ratios,
  high spin, degenerate shapes) — **done**: per-subsystem adversarial
  suites from the audit rounds plus the full integrated pass (09).
- Minimal debug draw (`docs/api-gaps.md` P3) — **done** (2026-09-09,
  `docs/design/debug-draw.md`).
- API freeze review: naming, docs, breaking-change surface settled
  (typed handles were the last planned breaking change).

## Post-1.0

- Serialization (`.bullet`-style; standing deferral, format design in
  `docs/api-gaps.md` P6).
- IslandWork / parallel island solver (bit-identity argument already
  validated; no rayon — new-crate cost, on-demand only).
- Further performance items: SipHash residue in `manifold_cache` /
  `no_collision_pairs` (FxHash), quaternion rotation tax, BVH
  `triangles_in_aabb` pooling (all validated numerically neutral, no new
  dependencies).
- Optional semantic parity items (each triggers a parity re-record):
  `m_allowedCcdPenetration = 0.04`, implicit gyroscopic impulse, CCD
  rotation rewind / compound gate alignment (deviations §1 candidates).
- Per-pair solver hooks (on-demand, `docs/api-gaps.md`); contact-force
  events and spinning friction landed 2026-09-11 (`docs/api-gaps.md`).

## Explicitly Out of Scope

Standing decisions — do not re-add these to the plan because an API gap
exists:

- `btDeformable*` system (NeoHookean / backward-Euler / Krylov; deformable
  contact projection) — `docs/design/softbody.md` §11.
- `BulletReducedDeformableBody` (reduced deformable).
- VHACD / HACD / ConvexDecomposition runtime integration (offline tooling
  at most, separate crates if ever).
- `btSimpleDynamicsWorld` / `btContinuousDynamicsWorld` (toy world variants).
- `btGhostObject` as a dedicated type (sensor flag + events + `overlapping()`
  cover the use cases; the `btPairCachingGhostObject` manifold-cache override
  is not ported).
- MLCP solver, RK4 integrator, kinematic multibody links, link CCD
  (upstream has none either).
- Multithreaded solver via `btThreadSupportInterface` (the parallel-island
  work is a perf item, not a parity item, and is on-demand).

## Current Priority

1. Documentation alignment (done, 2026-09-07).
2. Independent source audits 01–09 (done 2026-09-09).
3. Adversarial correctness testing (done).
4. Minimal debug draw (done, 2026-09-09).
5. On-demand consumer-API rounds (done 2026-09-09/10; third batch +
   cluster-core verification done 2026-09-10/11; fourth batch in progress —
   contact-force events + spinning friction done 2026-09-11, scaled-shape
   wrappers next).
6. 1.0 API freeze review (after the fourth batch).
7. IslandWork / parallel island solver.
8. Further performance optimization.

## Do Not Work On Yet

- **No physics algorithm changes without an audit finding that requires
  one** — no GJK/EPA/manifold/solver "improvements", no SIMD/SoA rewrites,
  no Rayon. Correctness work must be driven by reproduced findings, not
  taste.
- **No new dependencies** without explicit approval (the rayon decision
  stands: complexity/compile-time/binary-size cost, on demand only).
- **No API redesign** before the audits complete; the typed-handle
  migration was the last planned breaking change.
- **No serialization** (standing deferral until after the 1.0 freeze).
- **No out-of-scope features** (see the list above), even though their API
  gaps are documented.
- **No parity re-records without cause**: re-record when a semantic change
  lands or a scene is added; do not churn the ledgers.
