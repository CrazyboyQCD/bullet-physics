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

## Full-surface audit additions (2026-09-18)

Rows below come from the first full feature-surface audit against
`refs/bullet3/src` (all missing functionality not already carrying a
verdict above). All rows are **Open** — the verdicts are the maintainer's
call during the 1.0 API freeze review; the rationale states the ported
replacement, if any.

| Item | Frequency | Size | Verdict | Rationale |
|---|---|---|---|---|
| **Internal edge utility** (`btInternalEdgeUtility` + `btTriangleInfoMap`) | ★★★ | M | **Done** | Shipped 2026-09-18 (the audit row promoted to work): `TriangleMesh::generate_internal_edge_info()` builds the direct-indexed edge-angle table (a `Vec` replaces Bullet's `btHashMap` — dense triangle-index keys, the hash-collection-removal ruling) and convex-vs-mesh contacts on shared edges get the adjacent-triangle-aware normal (verbatim `SolveContact`/`btClampNormal` flow, normal directions source-verified via the upstream callers' `colObj1` argument order). Parity scene `internal_edge` (sphere dropped on a coplanar seam): x/z exactly 0 through impact and settle in both engines — the seam side-kick artifact is gone on both; residual = the predicted-pose impact transient + the 2.7 mm mesh band-settle class (f32 = f64). Scope cuts recorded (deviations §1): the three `normalAdjustFlags` modes, the contact-position re-projection, and GImpact/dynamic meshes. |
| **`btContactSolverInfo` parameter surface** | ★★★ | M | **Open** | The solver runs hardcoded constants today (`erp 0.2`, `slop 0.005`, split impulse always on with `erp2 0.2` / `turnErp 0.1`); upstream exposes ~20 knobs (globalCfm, frictionERP/CFM, warmstartingFactor, solverMode flags, the splitImpulse switch — default **off** upstream, we run always-on, deviations §4). A config struct with Bullet defaults keeps parity. |
| **`stepSimulation(dt, maxSubSteps, fixedTimeStep)` accumulator** | ★★★ | M | **Open** | The fixed-timestep accumulator + interpolation mode; only single-step `step_simulation(dt)` exists (the motion-state blend deviation is the same gap, deviations §3). |
| **Rigid `btGearConstraint`** | ★☆ | S | **Open** | Rigid-body gear coupling between two rotation axes; the multibody variant (`MultiBodyGearConstraint`) is already ported. |
| **`btGeneric6DofSpring2Constraint`** | ★☆ | M | **Open** | Spring axes / soft limits (the game-standard rewrite); the elastic limit behavior is not reproducible with plain 6DOF rows. |
| **`btMultiSphereShape`** | ★☆ | S–M | **Open** | Multi-sphere compound convex (metaballs, simplified proxies); GJK support = farthest center + radius; upstream has analytic inertia. No hull blow-up alternative. |
| **Heightfield up-axis / flip options** | ★☆ | S | **Open** | Fixed Y-up today; upstream supports X/Z up, `flipQuadEdges` and the right/left-handed winding toggle. |
| **Per-triangle materials** (`btMaterial` + `btMultimaterialTriangleMeshShape` + `btTriangleIndexVertexMaterialArray`) | ★ | M | **Open** | Per-triangle friction/restitution on triangle meshes. `TriangleMesh` has parts but no per-triangle material dispatch. |
| **Soft helpers leftovers** (`write_vtk`, `CreateFromVtk`, `CreateFromTetGenData`, `pointNeighbors`) | ★ | S | **Open** | VTK debug export is the most-used helper still missing; the rest are file-format conveniences. |
| **`btHinge2Constraint` / `btUniversalConstraint`** | ★ | S | **Open** | Convenience specializations of 6DOF (wheel/universal joints); reproducible by hand today — API-surface gap, not behavior. |
| **`btConvexHullComputer` / `btGeometryUtil`** | ★☆ | S | **Open** | Adjacency-carrying hull (decomposition tooling input) and plane-equation utilities; LinearMath conveniences, the ported quickhull covers runtime needs. |
| **`BulletInverseDynamics` (MultiBodyTree)** | ★ | L | **Open** | Inverse dynamics (q/qd/qdd → generalized forces; robot feed-forward control). Forward ABA is ported; the entire inverse tree is untouched. Propose Out-of-Scope unless a robotics consumer appears. |
| **Hull/shadow shape variants** (`btConvexPointCloudShape`, `btTetrahedronShape`, `btConvexTriangleMeshShape`) | ☆ | S | **Open (propose Won't do)** | Functionally covered by `ConvexHull` constructors (API convenience only — Bullet validates nothing either); propose documenting the equivalence instead of porting. |
| **`btMPRPenetration`** | ☆ | S | **Open (propose Won't do)** | An alternative penetration-depth solver; EPA covers the pipeline. Only reaches consumers who disable EPA. |
| **`btMinkowskiSumShape` / `btEmptyShape`** | ☆ | S | **Open (propose Won't do)** | Niche: compound ≈ Minkowski sum for physics purposes; an empty `Compound` fills the empty-shape slot. |
| **2D shapes & algorithms** (`btBox2dShape` + `btBox2dBox2dCollisionAlgorithm`) | ☆ | M | **Open (propose Won't do)** | `convex_2d` is already a forwarding marker; true 2D is achievable by plane-constraining a 3D scene (Rapier ships a separate 2D crate — out of proportion here). |
| **`btSdfCollisionShape` / MiniSDF** | ☆ | M | **Open (propose Won't do)** | Grid-SDF collision shapes; heightfield + meshes cover terrain uses. The deformable system's dependency does not apply (out of scope). |
| **`Bullet 3.x` GPU pipeline** (`Bullet3OpenCL` / `Bullet3Collision` / `Bullet3Dynamics` / `Bullet3Serialize` / clew) | — | — | **Open (propose Won't do)** | Independent GPU product line (the `b3*` classes, OpenCL/CUDA); not a port target for a CPU-only Rust rewrite. Propose recording it in `00-project-status.md` Out of Scope explicitly. |
| **Rigid API minutiae** (`setContactProcessingThreshold`, `setAnisotropicRollingFrictionDirection`) | ☆ | S | **Open** | Manifold-point admission threshold and rolling-friction anisotropy direction; tiny surface additions riding the body flags. |

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
freeze review**. The audit-additions rows below the table stay **Open**
until the maintainer rules on them in that review (proposals included
per row); rows ruled Won't-do there move to Out of Scope, rows accepted
join this table as planned work.

## Key observation

The highest-frequency items are all **physical-behavior correctness** gaps
(kinematic / child transforms / rolling friction), while the two L-size items
(serialize, decomposition toolbox) are **ecosystem/workflow** items — for a
library, consumers building their own is the better deal.
