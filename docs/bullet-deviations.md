# Deviations from Bullet

This document tracks every place where this crate deliberately differs from
Bullet's behavior, is numerically different, or has not yet been implemented.
It is the canonical reference for "does X match Bullet?" — checked against
`refs/bullet3` source on 2026-08-13.

## 1. Intentional deviations (kept)

| Area | This crate | Bullet | Rationale |
|---|---|---|---|
| Capsule/Cylinder/Cone local inertia | Exact analytic formulas | AABB-box approximation | More accurate, cheap to compute |
| Ray cast from inside a shape | Reports `fraction = 0` with `normal = ray direction` | Misses (the `n · r` check rejects interior starts) | Mainstream-engine semantics; documented in `raycast.rs` |
| Concave-concave and plane-concave pairs | Return `None` (no contacts) | Concave-concave unsupported too; plane-vs-concave handled via dedicated algorithms | Out of scope, documented in `narrowphase/mod.rs` |
| Friction warm starting | Not inherited across steps (rebuilds from zero each step) | Inherited (`m_appliedImpulseLateral1/2`), `m_warmstartingFactor = 0.85` | Our manifolds are rebuilt per step; the tangent basis rotates with the point set on triangle meshes, and stale friction kicks made resting bodies oscillate (`falling_mesh`). Normal-impulse warm starting IS implemented |
| Contact manifold persistence | Impulse cache only (normal impulse, point matched by distance + normal) | Full `btPersistentManifold` with point lifetime, friction anchors, `refreshContactPoints` | Impulse cache delivers the warm-starting benefit with a fraction of the machinery |
| `activate()` on disabled bodies | Skips `DisableDeactivation` and `DisableSimulation` bodies (they keep their state) | Forces `ACTIVE_TAG` — only the static/kinematic collision flags are checked | Preserving `DISABLE_DEACTIVATION`/`DISABLE_SIMULATION` semantics is safer for the common "never sleep / never simulate" usage |

## 2. Numerical differences from Bullet defaults

| Parameter | This crate | Bullet default | Impact |
|---|---|---|---|
| Restitution velocity threshold | `1.0` m/s | `m_restitutionVelocityThreshold = 0.2` | Low-speed bounces are suppressed (more stable, slightly different feel) |
| Penetration slop | `0.005` m | `m_linearSlop = 0.0` (Baumgarte acts on full penetration) | Resting bodies settle ~5 mm deeper; more stable |
| EPA `INSIDE_EPS` / `FALLBACK` | Not implemented (degenerate → abort with current best) | `INSIDE_EPS = 0.01` (f32), `FALLBACK = 10 × EPA_ACCURACY` | Edge-case polytope handling may differ in extreme configurations |
| Warm-starting factor | `0.85` | `m_warmstartingFactor = 0.85` | Aligned |

## 3. Not yet implemented

- **Split impulse** (Bullet `m_splitImpulse = true` by default, threshold `-0.04`): we rely on Baumgarte alone for penetration recovery.
- **Simulation islands** (union-find island manager + per-island solving): we solve all constraints globally. Activation propagation uses a simplified per-step contact-graph pass instead (a sleeping body touching an active one wakes up one hop per step); a sleeping body is never woken by another sleeping body, so stable stacks stay asleep.
- **CCD** (continuous collision detection).
- **CharacterController**, **Vehicle**.
- **Joints**: Point2Point, Hinge, Generic6Dof, ConeTwist, Slider, Fixed.
- **Featherstone multibody**.
- **Full persistent-manifold point lifecycle** (lifetime tracking, friction anchors, contact-processed callbacks) — see §1.

## 4. Aligned with Bullet (reference)

- EPA: `MAX_VERTICES = 128`, `EPA_ACCURACY = 1e-4`, `EPA_PLANE_EPS = 1e-5`, `MAX_ITERATIONS = 255` (see `narrowphase/epa.rs`); f64 builds use precision-dependent `1e-12` / `1e-14`, matching `btGjkEpa2` under `BT_USE_DOUBLE_PRECISION`.
- Narrow-phase dispatch: dedicated sphere-sphere / sphere-box / box-box (SAT) / plane-convex paths, generic GJK+EPA otherwise; compound recursion; mesh only vs. convex.
- AxisSweep3: three-axis sweep-and-prune with incremental sorted lists and ray-test pruning (equivalent to `btAxisSweep3`, including `updateHandle` semantics).
- Solver structure: sequential impulse, two-tangent friction cone (clamped by `μ · normal_impulse`), Baumgarte `erp = 0.2`, `iterations = 10`, friction `√(f₁·f₂)` and restitution `max` combination.
- Sleeping / activation: `ActivationState` mirrors `btCollisionObject::ActivationState` (`ACTIVE_TAG`, `ISLAND_SLEEPING`, `WANTS_DEACTIVATION`, `DISABLE_DEACTIVATION`, `DISABLE_SIMULATION`); sleeping thresholds default `0.8`/`1.0` (Bullet `m_linearSleepingThreshold`/`m_angularSleepingThreshold`), `DEACTIVATION_TIME = 2.0` (Bullet's hard-coded check), `updateDeactivation`/`goToSleep` semantics match (including `DISABLE_DEACTIVATION` early return), static bodies start asleep. See §1 for the `activate()` difference and §3 for the island-propagated wake-up simplification.
- Ray cast semantics: `fraction ∈ [0, 1]`, normal pointing toward the ray origin (see §1 for the interior-start exception).
- Broad-phase ray test: slab-method `segment_hits_aabb`, `gContactBreakingThreshold = 0.02` point matching for warm starting.
