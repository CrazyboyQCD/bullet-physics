# Consumer API priority assessment

> Unified evaluation of the remaining consumer-API gaps (`docs/api-gaps.md`
> backlog + deviations §3), scoring usage frequency and implementation size.
> Decided with the maintainer on 2026-09-07; the order below drives the
> priorities in `HANDOFF.md` §5. Update this document when items ship or
> verdicts change.

## Unified scoring table

| Item | Frequency | Size | Verdict | Rationale |
|---|---|---|---|---|
| **Kinematic rigid bodies** | ★★★★★ | M | **Done** | Daily necessity for moving platforms / elevators / doors / vehicles; no correct workaround (hacks propagate no velocity). Shipped 2026-09-04. |
| **Compound child dynamic transforms** | ★★★★ | S–M | **Done** | Kinematic compounds are the standard moving-vehicle setup; implemented together with kinematics. Shipped 2026-09-07. |
| **Ghost persistent overlap list** | ★★★☆ | S | **Done** | Level-query alternative to hand-rolled event bookkeeping; the pair pass already tracks touch state — pure exposure. Shipped 2026-09-07 (`world.overlapping`). |
| **Rolling friction** | ★★★☆ | S–M | **Done** | Physical-correctness gap for resting balls/cylinders (pool / wheel scenes); damping hacks are unphysical. Shipped 2026-09-07 (`body.rolling_friction`, parity scene `rolling` — rf rows exceed-0 exact). |
| **Motion states** | ★★★ | S–M | **Done** | Render interpolation is an advanced need; revisit depending on whether the kinematic driving API wants them early. Shipped 2026-09-07 (`MotionState` trait, `add_body` pose read + step-tail synchronize; the interpolation blend and the kinematic drive stay unported — deviations). |
| **mb/soft event sources** | ★★☆ | S–M | **Done** | Robot-arm perception and similar composite scenes exist; extend the event-source dispatch. Shipped 2026-09-07 (`CollisionObject` sides on `CollisionEvent`, `LinkCollider`/`SoftBody` opt-in, pair-level soft events). |
| **Persistent-manifold extras** | ★★☆ | S | **Done** | Shipped 2026-09-10: `ContactPoint::lifetime` (+ `PersistentPoint`), `HAS_FRICTION_ANCHOR` friction-anchor servo (rigid solver, Coulomb-cone bounded), `set_contact_point_events` point drain. Rigid paths only (deviations §1); memory ledger addendum +0.125 MB. |
| **btShapeHull** | ★★☆ | S | **Done** | Shipped 2026-09-10 (`ShapeHull::build`, lowres 42-direction table); quickhull moved to `bullet-math` (LinearMath mapping). Highres/penetration extras stay unported (deviations §1). |
| **Character directional jump** | ★★ | S | **Done** | Shipped 2026-09-10: `jump_direction(v)` + `set_jump_axis` with the `m_SetjumpSpeed` overwrite/restore semantics. `setInterpolateUp` / `setMaxJumpHeight` stay unexposed (deviations §3). |
| **Serialize** | ★★ | L | Defer | Savegames work with handwritten pose snapshots (bodies are all plain data); bit-exact round-trip has no consumer ecosystem — post-release. |
| **Anisotropic friction** | ★★ | M | **Done** | Shipped 2026-09-10: `RigidBody::anisotropic_friction` (pub field, Bullet's unity gate) applied to both pair sides per tangent axis in `ContactConstraint::prepare`; parity scene `anisotropic_friction` — box_z/box_iso exceed = 0, box_x classifies the pre-existing fixed-basis vs velocity-dependent friction-direction gap (deviations §1). Multibody contacts ignore the field (same scope cut as the friction anchor). |
| **Contact stiffness/damping** | ★★ | M | **Done** | Shipped 2026-09-10: `set_contact_stiffness_and_damping` + `CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING` (0x80); the pair's normal row swaps its ERP and adds the compliance CFM (combined damping sums, stiffness in series — Bullet verbatim). Parity scene `contact_soft` — ball_soft/ball_hard exceed = 0, ball_soft_heavy classifies the deterministic 2.4 cm equilibrium drift (deviations §1). Multibody contacts ignore the parameters. |
| **Additional damping** | ★☆ | S | **Done** | A humanoid-stack tuning safety valve; scenario-specific. Shipped 2026-09-10: `RigidBody::additional_damping` + `additional_damping_params` builder (Bullet exposes the block only through constructor info); `apply_damping` block verbatim including the dead `m_additionalAngularDampingFactor` quirk (deviations §3). Parity scene `damped_stack`. |
| **Multi-part triangle meshes** | ★★ | M | **Done** | Vertex/index concatenation is a workaround with no runtime benefit. Shipped 2026-09-10: `TriangleMeshPart` + `TriangleMesh::from_parts` (constructor-time concatenation, single triangle stream at runtime; `CollisionShape::triangle_mesh_parts`). Parity scene `mesh_parts`. |
| **Soft cluster core (CLinks)** | ★☆ | S | **Done** | Clusters are niche; completeness-only. Shipped/verified 2026-09-10/11: the `PSolve_CLinks`/`VSolve_CLinks` half of this row was a phantom entry (no such symbols in bullet3 master); the real cluster core (static `solveClusters` pass + `applyClusters`/`dampClusters` + LJoint/AJoint/CJoint with the drift split) was already ported and now reproduces the `cluster_joints` probe bit-for-bit in f64 after the `terminate` parameter-order fix. The `cluster_pair` CL_SS residual was root-caused 2026-09-18 (no port bug — blow-up regime + Bullet's tunneled triangulated plane at explosion speed + hull-order scatter; the `cluster_pair_soft` light-touch variant agrees qualitatively — see `bullet-compare.md` Phase D). |
| **Extras decomposition toolbox** | ★★★ | **L+** | Won't do | VHACD is library-scale work; industry practice is offline decomposition with runtime consumption of ConvexHull + Compound (already supported) — point the docs at it. |

## Suggested order

~~kinematic → compound child transforms (same round) → ghost list → rolling
friction~~ — all four shipped (2026-09-04 / 2026-09-07). ~~motion states and
mb/soft event sources (batch 2)~~ shipped 2026-09-07. ~~Minimal debug draw
(P3)~~ — shipped 2026-09-09 per `docs/design/debug-draw.md`. ~~Persistent-
manifold extras, btShapeHull, character directional jump~~ — shipped
2026-09-10 (the whole on-demand batch closed out together). ~~Anisotropic
friction, contact stiffness/damping~~ — shipped 2026-09-10 (second on-demand
batch, parity scenes `anisotropic_friction` / `contact_soft`). Remaining
rows are a Defer or Won't-do verdict; the project milestone is the **1.0 API
freeze review**.

## Key observation

The highest-frequency items are all **physical-behavior correctness** gaps
(kinematic / child transforms / rolling friction), while the two L-size items
(serialize, decomposition toolbox) are **ecosystem/workflow** items — for a
library, consumers building their own is the better deal.
