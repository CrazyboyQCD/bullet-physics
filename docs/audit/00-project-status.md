# Project Status

> The single source of truth for the current state of the project. Update
> this file whenever the state changes. Fine-grained evidence lives in the
> ledgers (`docs/bullet-compare.md`, `docs/bullet-perf.md`) and the audit
> records (`docs/audit/`).

## Current HEAD

`64ab184` — *feat: motion states and multibody/soft-body collision-event
sources* (2026-09-07). `origin/main` is at `259aa9c`; this commit is local,
pending push.

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

- Documentation alignment (this round): docs updated to HEAD `64ab184`,
  audit framework created.

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

- Persistent-manifold rotation-drift semantics — needs independent
  verification (the f64 friction-row noise class in the `rolling` ledger is
  the visible symptom class).
- Quaternion `q` / `-q` double-cover handling across joints and manifolds —
  needs independent verification.
- Contact-point matching robustness under fast rotation (matching radius,
  refresh guards) — needs independent verification.
- Manifold reduction behavior vs Bullet beyond the tested scenes — needs
  independent verification.
- EPA degenerate-polytope paths (`INSIDE_EPS` / `FALLBACK` not ported,
  deviations §2) — needs independent verification on adversarial inputs.

## 1.0 Required

- Independent source audits of the correctness-critical subsystems
  (manifold, GJK, EPA, convex-convex, AxisSweep3, DBVT) — see
  [`README.md`](README.md).
- Adversarial correctness testing (deep penetration, extreme mass ratios,
  high spin, degenerate shapes).
- Minimal debug draw (`docs/api-gaps.md` P3).
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

1. Documentation alignment (this round — done).
2. Independent source audit: manifold (`01-manifold.md`).
3. Independent source audit: GJK.
4. Independent source audit: EPA.
5. Independent source audit: convex-convex.
6. Independent source audit: broadphase (AxisSweep3 + DBVT).
7. Adversarial correctness testing.
8. Minimal debug draw.
9. 1.0 API freeze.
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
