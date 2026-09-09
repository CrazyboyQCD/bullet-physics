# Project Status

> The single source of truth for the current state of the project. Update
> this file whenever the state changes. Fine-grained evidence lives in the
> ledgers (`docs/bullet-compare.md`, `docs/bullet-perf.md`) and the audit
> records (`docs/audit/`).

## Current HEAD

`7a20273` — *docs(design): debug draw design specification* (2026-09-09),
followed by the debug-draw implementation commit (same day).
`origin/main` is behind by 2 commits (push pending).

## Current Phase

**Correctness hardening / independent audit.** The functional core is
complete: the rigid-body pipeline, Featherstone multibodies with collision
integration, the classic SoftBody system, and the consumer-facing API layer
are all implemented and covered by the test/parity/probe suites. Development
now shifts from adding features to verifying what exists — subsystem-by-
subsystem independent source review, adversarial testing, and only then the
remaining API items and the 1.0 API freeze.

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
- **no_std**: whole workspace behind `--no-default-features --features libm`
  (std default bit-identical; libm scalar backend; injectable step clock).
- **Tooling**: parity harness (8 scenes × f32/f64, ledger current),
  multibody/softbody/GImpact probes (24 mb scenes × f32/f64), perf ledger
  (2026-09-04 rerun, rust ahead 0.15–0.62× on tracked scenes), memory-trend
  harness, flamegraph workflow.
- **Quality gates**: 350 unit tests × f32/f64 green, 14 demos, clippy clean
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
a per-proxy eligibility fast path, `docs/bullet-perf.md`). Residual
semantic note recorded as deviations §1 "Islands: gating & kinematic wake".

**Stage: audits 01–09 complete.** The broadphase integration audit landed as
[`07-broadphase-integration.md`](07-broadphase-integration.md), the CCD audit
as [`08-ccd.md`](08-ccd.md) (one confirmed sensor-target bug fixed), and the
full integrated adversarial pass as
[`09-integrated-adversarial.md`](09-integrated-adversarial.md) (one P3
zero-link fixed-base multibody panic fixed; 11 cross-system adversarial
tests; deterministic state machine bit-repeatable). The minimal debug
draw — the last feature item — shipped on 2026-09-09 per
`docs/design/debug-draw.md` (five read-only domains, invariance contract
test-pinned). The remaining 1.0 item is the API freeze review.

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
- Contact-force events, per-pair solver hooks, spinning friction,
  btShapeHull, character directional jump (all on-demand, `docs/api-gaps.md`).

## Explicitly Out of Scope

Standing decisions — do not re-add these to the plan because an API gap
exists:

- `btDeformable*` system (NeoHookean / backward-Euler / Krylov; deformable
  contact projection) — `docs/softbody.md` §11.
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
2. Independent source audit: manifold (`01-manifold.md` — done 2026-09-08).
3. Independent source audit: GJK (next).
4. Independent source audit: EPA.
5. Independent source audit: convex-convex.
6. Independent source audit: broadphase (AxisSweep3 + DBVT).
7. Adversarial correctness testing (done).
8. Minimal debug draw (done, 2026-09-09).
9. 1.0 API freeze (next).
10. IslandWork / parallel island solver.
11. Further performance optimization.

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
