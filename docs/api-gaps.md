# Consumer API gaps — the user-facing feature backlog

A consolidated view of what a *consumer* of the crate cannot yet do,
ordered by expected usage frequency within each tier. Planning view
only: the authoritative per-item behavior notes live in
`bullet-deviations.md` §3 (verified, categorized) and the usage-feel
context in `api-usability.md` §2. Each entry carries an implementation
sketch (verified against the codebase 2026-09-02); sizes are S ≤ one
session, M a couple, L a dedicated round.

Verification note (2026-09-03): the landed consumer-API tier is covered
by the `composite_consumer` parity scene against real Bullet (trigger +
per-body gravity + pair filter + CCD + filtered all-hits/closest rays in
one scene; impact transients classified under the predicted-pose
trade-off, rest states and ray probes exact) — see
`docs/bullet-compare.md` and `docs/parity-harness.md`.

Legend: **Impl** = core logic · **Files** = touched modules · **Risk** =
performance traps / API debates / upstream-semantics gaps · **Size** =
S/M/L · **See** = existing notes.

## Near term (non-breaking)

- **P3 — minimal debug draw** ✅ (2026-09-09): `world.debug_primitives(...)`
  emitting kind-tagged physics facts with a category mask. Producer side
  only; the consumer stays caller-side (e.g. a Bevy `Gizmos` bridge).
  Design baseline: `docs/design/debug-draw.md` (Bullet coverage checklist,
  Rapier producer/consumer split, Avian-style domains — none of their APIs).
  - Shipped: `DebugDrawFlags` (`SHAPES` / `AABBS` / `CONTACTS` /
    `CONSTRAINTS` / `SLEEPING`; empty mask early-outs) and a `#[non_exhaustive]`
    `DebugPrim<'a>` enum (`Shape` borrowing the body's `CollisionShape`,
    `Aabb` from `BroadPhase::proxy_aabb` for every live proxy incl. mb
    link colliders and soft bodies, `Contact` mirroring the persistent
    manifold's cached points with the B-side re-projected anchor, cached
    A→B normal and last-solve normal impulse, `Constraint` carrying both
    world-projected joint anchors plus a `ConstraintTag`, `Activation`
    reporting `activation_state()`), all pushed through one callback:
    `PhysicsWorld::debug_primitives(flags, &mut impl FnMut(DebugPrim<'_>))`
    (`&self` only; no new world state; zero allocation; the emission is
    order-unspecified).
  - Deviations from the original sketch (design-doc decisions): wheel
    rays stay consumer-side (vehicles are caller-owned; `WheelInfo` is
    public), normals ride on the `Contact` prim instead of a separate
    flag, and islands/CCD/mb-soft contacts are deferred because their
    state is step-transient.
  - Files: `bullet-dynamics/src/world/debug.rs` (new; types + emit pass
    + 8 in-domain tests incl. the draw-vs-no-draw bit-identical
    invariance contract), `world.rs` (mod wiring), facade prelude.
  - Risk resolved: a "contact point" is a cached persistent-manifold
    point (what the solver warm-starts from), not a solver row; per-
    domain gating keeps quiet masks free.
  - Size: S. See: `api-usability.md` §2 (P3 rationale).

- **P4 — callbacks**: `Started/Stopped` events, per-body opt-in and
  trigger volumes are in. Still open:
  - *Contact-force event*: after the impulse writeback (points carry
    `point_index` into the manifold cache), sum normal impulses per
    touching pair and emit when above a threshold — the events pipeline
    (touch buffers, per-body opt-in, drain API) is reused as-is.
    Files: `world.rs` writeback + `CollisionEvent` enum. Risk: threshold
    semantics — accumulated impulse vs force (= impulse/dt); no Bullet
    counterpart, so the choice must be documented in deviations §3.
    Size: S–M.
  - *Per-pair solver hooks*: map onto the existing
    `no_collision_pairs` / filter system (a pair allow/deny
    predicate at pair-acceptance time — same shape as the P6
    overlap-filter callback below).
    Size: S. See: `api-usability.md` §2 (P4).

- **P5 — typed handles** ✅ (2026-09-03, breaking): `add_body` /
  `add_multibody` / `add_soft_body` used to return `usize`.
  - Shipped: newtypes `BodyHandle` / `MultiBodyHandle` / `SoftBodyHandle`
    over `NonZeroU32` (`handles.rs`, `Option<H>` same size as `H`,
    `const fn from_index`/`index()` for literal migration and interop —
    no generation counters, so stale handles behave exactly like the old
    stale indices: inert data for removed rigid bodies, renumbered
    lookups for soft bodies). The world keeps `usize` slots internally
    and converts at the API boundary; constraint endpoints
    (`Constraint::body_a`/`body_b`, the mb-vs-rigid `rigid_body_b`, the
    `mb-p2p/fixed/slider` constructors), `Side::Rigid`/`Cluster.soft`,
    soft anchors, the vehicle chassis, the character body,
    `CollisionEvent` fields, `PairFilter` args, ray/sweep/contact-test
    returns and `RaycastInfo::ground_body` are all typed. Constraint /
    multibody-constraint *list* slots stay `usize` (no other usize index
    space is user-visible anymore, so the confusion class is closed).
    The `*_to_rigid` mb constructors pass `body_a` as the placeholder
    side-B handle (Bullet's `m_multiBodyB == 0` null branch; every
    side-B read is gated by `rigid_body_b`).
  - Files: `bullet-dynamics` `handles.rs` (new), `world.rs`,
    `constraint.rs`, `multibody/constraint.rs`, `softbody/{body,collide,
    joints,solver}.rs`, `vehicle.rs`, `character.rs`, facade prelude,
    all examples/demos.
  - Risk: verified output-neutral — parity `falling_box` /
    `composite_consumer` (f64) reproduce the ledger-classified numbers
    exactly (balloon bit-identical, rays body mismatches = 0, impact
    transients under the predicted-pose class); 327 tests x f32/f64,
    14 demos, clippy x2 clean.
  - Size: M (mechanical but wide). See: `api-usability.md` §2 (P5).

## P6 — feature surface, by expected usage frequency

- **all-hits ray** ✅ (2026-09-02): `raycast_closest` only; Bullet's `rayTest` also
  offers all-hits via `RayResultCallback`.
  - Impl: the closest-hit walk (`raycast_closest_excluding`) already
    iterates every candidate from `BroadPhase::raycast_candidates`;
    add `raycast_all(from, to) -> Vec<(usize, RayHit)>` collecting
    instead of minimizing (sort by fraction).
  - Files: `world.rs` query section only (per-shape ray tests in
    `bullet-collision` are unchanged).
  - Risk: none semantic; document ordering guarantee.
  - Size: S.
- **ray filter masks** ✅ (2026-09-02): rays consume no group/mask yet (the group/mask
  system itself is ported).
  - Impl: optional `(group, mask)` parameters (or a filter closure) on
    the ray entry points; the per-candidate acceptance test applies the
    same symmetric `group & mask` rule the pair pass uses (Bullet's
    `RayResultCallback::needsCollision`).
  - Files: `world.rs` ray queries.
  - Risk: keep the existing `ignore: Option<usize>` parameter working
    (parameter-position churn on a small API is fine).
  - Size: S.
- **ghost persistent overlap list** ✅ (2026-09-07): sensor flag + events +
  `contact_test` cover the main use case; Bullet's `btGhostObject`
  listing is the full version.
  - Shipped: `PhysicsWorld::overlapping(handle) -> Vec<BodyHandle>` (the
    `getNumOverlappingObjects` / `getOverlappingObject` surface) reads the
    pair pass's persistent touch state — no second bookkeeping structure.
    The `touching` buffers now track sensor pairs (either side
    `NO_CONTACT_RESPONSE`) in addition to the opted-in pairs; the event
    diff and the `remove_body` cascade filter on the opt-in flags, so
    events stay per-body opt-in and pure-sensor pairs only feed the list.
    Entries are narrow-phase touches (Bullet's listing is broad-phase
    AABB overlaps — deviation recorded in deviations §1), survive island
    sleep via the same carry-over the events use, and clear on removal
    or separation. 4 scene tests (pass-through + empty after exit,
    sleep survival, removal cascade, multi-body ascending order).
  - Size: S.
- **serialization**: deliberately deferred (standing decision).
  - Impl: versioned binary format hand-written over the public fields
    (shape enum + parameters, transform, velocities, material, flags,
    constraint kinds, world settings); every shape is plain data today,
    so no reflection is needed. Round-trip must be bit-exact (the
    determinism selling point makes this a test, not a hope).
  - Files: new `serialize.rs` in `bullet-dynamics` + facade re-export;
    `bullet-collision` shapes expose no new code (fields suffice).
  - Risk: format versioning as the shape enum grows; f32/f64
    cross-compat (reject or widen on load).
  - Size: L. See: `api-usability.md` §2 (P6).
- **motion states** ✅ (2026-09-07): Bullet's pose-callback sync
  (`btMotionState`).
  - Shipped: a `MotionState` trait (`initial_pose()` =
    `getWorldTransform`, `synchronize(&Transform)` = `setWorldTransform`)
    stored as `Option<Box<dyn MotionState>>` on `RigidBody`
    (builder `motion_state(..)` + `set_motion_state(..)`), re-exported
    through the facade prelude. `add_body` places the body at the state's
    reported pose (Bullet's constructor read); the step hands the final
    pose back after position integration (Bullet's `synchronizeMotionStates`
    point), gated exactly like `synchronizeSingleMotionState`: *active,
    non-static/kinematic* bodies only.
  - Reference decisions covered: construction-time pose read ✅ (Bullet
    `setupRigidBody`); active-only traversal ✅ (default
    `synchronizeMotionStates`, `m_synchronizeAllMotionStates` not ported);
    static/kinematic exclusion ✅; every-step call even for sleeping
    bodies inside the active window ✅ (a woken island resyncs on wake).
    NOT ported: the interpolation blend (`synchronizeSingleMotionState`
    feeds the *interpolated* transform — single-step worlds have no
    remainder fraction; recorded in deviations) and the kinematic drive
    (`saveKinematicState` reads the pose from the motion state and derives
    velocities — our kinematics are user-velocity-driven, deviations §3).
  - Files: `body.rs` (trait + field + `pose_transform`/`split_pose`
    f32-`Vec3A`↔`Vec3` shims), `world.rs` (`add_body` read),
    `world/step.rs` (7m sync pass).
  - Risk resolved: the per-step consumer callback stays off the solver
    path (once per step, gated on a world-level "any motion state" check —
    zero cost without opt-in). Ownership: `Box` per body only when opted in.
  - Tests: 2 (place + one-sync-per-step with semi-implicit pose check,
    static/kinematic exclusion).
  - Size: S–M (the design debate was the cost, not the code).
- **kinematic rigid body** ✅ (2026-09-04): the full kinematic type surface
  (`CF_KINEMATIC_OBJECT` + velocity propagation).
  - Shipped: `CollisionFlags::KINEMATIC` (bit 0x0004; `DISABLE_WORLD_GRAVITY`
    had already taken 0x0002) + `is_kinematic()` / `is_static_or_kinematic()`;
    `RigidBody::new_kinematic(shape, position)` (infinite mass via the static
    inertia path, `DisableDeactivation` - a kinematic body is the activation
    source its resting contacts rely on) and a `kinematic(on)` builder.
  - Gates (the `is_static_or_kinematic()` review pass): velocity integration
    (gravity/damping/forces never touch the user-set velocities), `add_body`
    gravity inheritance, split-impulse pushes, CCD clamping, the pair pass
    (both-sides-immovable pairs skipped, mirroring Bullet's broadphase
    `StaticFilter` group exclusion), `participates_in_islands` (kinematics
    sit outside the island node space like statics) and the softbody cluster
    contact branches (`kSKHR_CL` / anchored-cluster skip).
  - Wake propagation: a *moving* kinematic (nonzero velocities) wakes every
    island it pairs with, decided in the island pass (which runs before the
    narrow phase); an idle one keeps its resting contacts sleeping, like
    Bullet's contact-point activation on a resting manifold.
  - Solver needed zero changes: `inv_mass == 0` degrades every row
    mathematically, and the kinematic's velocity still feeds the relative
    velocity (velocity propagation).
  - Deviations recorded (bullet-deviations §3): pose integration from the
    user-set velocity (Bullet drives the pose from the user/motion state and
    only uses the velocity for contacts; here a user `set_translation` every
    frame composes with the velocity integration the same way), and
    `new_kinematic` starts `DisableDeactivation` (Bullet kinematics walk the
    sleep state machine; without it the velocity-driven pose would stall).
  - Tests: 5 scene tests (push, force/gravity/damping immunity + pose
    integration, wake propagation, idle-platform sleeping, inert
    kinematic-kinematic/static pairs).
  - Size: M.
- **per-body gravity** ✅ (2026-09-02): `btRigidBody::setGravity` — only world gravity
  exists here.
  - Impl: `RigidBody.gravity: Vector3` filled from the world at
    `add_body` (skip static/kinematic); `world.set_gravity` propagates
    to active bodies lacking a `DISABLE_WORLD_GRAVITY` flag (next
    `CollisionFlags` bit); `integrate_velocities` consumes the per-body
    value (default == world value, so the default path is
    bit-identical — zero parity risk).
  - Files: `body.rs`, `world.rs` (`add_body`, `set_gravity`, step 1m).
  - Risk: propagating `set_gravity` changes an existing API's behavior
    (today it is a plain field write); Bullet updates *active* bodies
    only (sleeping bodies keep the stale value until woken — copy the
    quirk verbatim and record it); rigid-only (mb/soft keep the world
    value; `btMultiBody` has no per-body gravity either).
  - Size: S. See: `refs/bullet3/src/BulletDynamics/Dynamics/
    btDiscreteDynamicsWorld.cpp:322` (`applyGravity`), `:499`
    (propagating `setGravity`), `:538` (`addRigidBody` sync);
    `btRigidBody.cpp:205` (`applyGravity` = cached-force application).
- **overlap-filter callback** ✅ (2026-09-02): `btOverlapFilterCallback` — the
  group/mask pair test is in, custom predicates are not.
  - Impl: `world.set_pair_filter(Option<Box<dyn FnMut(usize, usize) ->
    bool>>)` consulted at pair acceptance (where the group & mask test
    runs today) and in the CCD sweep; clearing resets to default
    semantics.
  - Files: `world.rs` pair pass (+ CCD gate).
  - Risk: a `dyn` call once per broad-phase pair (~2-3k/frame on
    stack_1000) is acceptable and outside the solver inner loop;
    interaction order with group/mask must mirror Bullet (both must
    pass).
  - Size: S.
- **compound child transforms** ✅ (2026-09-07): `set_child_transform` /
  `remove_child_shape` at the `Compound`, `RigidBody` and
  `PhysicsWorld` layers (Bullet's `updateChildTransform` +
  `removeChildShape` + `updateSingleAabb`).
  - Impl: `Compound` keeps no derived caches (AABB/support/inertia are
    computed on demand), so the shape-layer mutation is plain data;
    `RigidBody::set_child_transform` recomputes `inv_inertia_local` for
    dynamic bodies (static/kinematic keep zero) via the constructor
    formula; `PhysicsWorld::set_child_transform` additionally refreshes
    the broad-phase proxy from the current pose immediately — required
    because the per-step AABB pass only covers *active* bodies, so
    sleeping/static compounds would keep stale bounds (verified by a
    ray-walk test through the proxy). Kinematic compounds (moving
    vehicles with animated doors) refresh naturally every step.
  - Files: `bullet-collision/src/shapes/compound.rs`,
    `bullet-dynamics/src/body.rs` (+ `inv_inertia_local()` accessor),
    `bullet-dynamics/src/world.rs`.
  - Risk resolved: mutation must happen between steps (solver rows hold
    no child references, so inter-step is safe) — documented on the
    world API. Numerically inert for existing scenes (no default-path
    change; parity untouched).
  - Size: M (as estimated; the missing cached tree made it S–M).

## Farther (on demand — see deviations §3 for details)

- **rolling friction** ✅ (2026-09-07): the ball/cylinder
  rolling-forever gap (Bullet's `m_rollingFriction`).
  - Shipped: `RigidBody.rolling_friction` (default 0) + two
    pure-angular torsional rows per contact point (Bullet's
    `setupTorsionalFrictionConstraint` pair on the `btPlaneSpace1`
    basis), driven toward zero relative angular velocity and clamped
    per iteration to `min(rf · normal_impulse, rf)` — rolling bodies
    only decelerate under contact load. Combined coefficient is
    Bullet's quirk verbatim: `rf_a·μ_b + rf_b·μ_a` clamped ±10 (not a
    product). Rows live in a world-owned pool (`rolling_rows`) indexed
    from the contact row, so the hot-path row size is unchanged and
    scenes that never set the field keep bit-identical behavior
    (baseline parity re-checked digit-for-digit).
  - Files: `body.rs`, `solver/sequential_impulse.rs` (`RollingFriction`,
    `plane_space`, `attach_rolling`), `world.rs` (pool + attach at both
    row-assembly sites + solve plumbing), recorder tables
    (`generate_scenes.py` + both recorders gained `rolling_friction`
    and `angular_velocity` body fields).
  - Risk resolved: parity scene `rolling` added (see
    `bullet-compare.md`) — `ball_rf` is exceed-0 exact in f32 and f64;
    the never-resting no-rf balls expose the pre-existing friction-row
    noise class (classified, not a regression). Spinning friction
    (`m_spinningFriction`, the one row around the contact normal that
    Bullet installs in the same block) is not ported — same shape, on
    demand.
  - Size: S–M (the parity scene was the M half).
- **Rigid-body property surface (remaining)**: anisotropic friction,
  per-object contact stiffness/damping
  (`CF_HAS_CONTACT_STIFFNESS_DAMPING`), the opt-in additional-damping
  block, spinning friction.
  - Impl: plain fields on `RigidBody` consumed in
    `ContactConstraint::prepare` (anisotropic scaling inside the
    effective-mass/friction setup; spinning friction adds one row
    around the contact normal — the same torsional-row machinery the
    rolling friction shipped); contact stiffness/damping replaces the
    default erp/cfm route in row assembly.
  - Files: `body.rs`, `solver/sequential_impulse.rs`.
  - Risk: each property is a solver-semantic change → parity re-record
    for scenes using it (default-off keeps the rest bit-identical).
  - Size: M per property.
- **Scaled-shape wrappers**: `btScaledBvhTriangleMeshShape`,
  `btUniformScalingShape`, `btConvex2DShape`.
  - Impl: new `CollisionShape` variants wrapping an inner shape + scale,
    delegating support/AABB/inertia through the scaling (margin-core
    scaling needs care: non-uniform scale breaks the exact ball-Minkowski
    shortcut the GJK margin band relies on).
  - Files: `bullet-collision/src/shapes/`.
  - Risk: non-uniform scaling interacts with the margin-core
    architecture (deviations §1) — sphere/box cores stop being exact
    under scale.
  - Size: M.
- **Multi-part triangle meshes**: `btTriangleIndexVertexArray` parts
  (one vertex/index pair here).
  - Impl: `TriangleMesh` accepts a part list; the BVH unions the parts'
    triangles (the `tri_order` slice layout is unchanged — parts just
    append).
  - Files: `bullet-collision/src/shapes/triangle_mesh.rs`.
  - Risk: low — construction-time only; no runtime path changes.
  - Size: M.
- **`btShapeHull`** ✅ (2026-09-10): hull decimation / debug hull tool.
  - Impl: sample the convex shape's supports over Bullet's lowres
    42-direction unit-sphere table (transcribed verbatim), dedupe, and
    run the quickhull — which moved from `softbody/quickhull.rs` to
    `bullet-math/src/quickhull.rs` (Bullet's hull library lives in
    LinearMath; collision may now share it). `ShapeHull::to_convex_hull`
    round-trips a primitive into a `ConvexHull`.
  - Files: `bullet-collision/src/shapes/shape_hull.rs`, module move
    `softbody/quickhull.rs` → `bullet-math/src/quickhull.rs`.
  - Not ported: the highres 256-direction variant and the
    preferred-penetration direction extras (no shape implements them
    here) — deviations §1.
  - Tests: 4 (sphere keeps all 42 samples on the surface, box collapses
    to 8 corners × 12 triangles, capsule volume vs analytic, round-trip
    support identity).
  - Size: S.
- **Extras toolkits**: `ConvexDecomposition`, `HACD`/`VHACD` — convex
  decomposition is expected to run offline, like Bullet's own demos
  suggest for stacks.
  - Impl: separate offline crates (or accepted-format input: import the
    decomposed convex hulls as a `Compound`); no engine changes.
  - Files: new workspace crates, if ever.
  - Risk: L-scale algorithm ports with little parity surface.
  - Size: L each.
- **Character controller API surface**: directional `jump(v)`,
  `setInterpolateUp(false)`, `setMaxJumpHeight` (its only consumer is
  commented-out dead code upstream).
  - Impl: `jump_along(dir)` generalizes the existing `jump()` (which
    already records `jump_axis`); the two toggles are fields on the
    controller with the documented clamps.
  - Files: `character.rs` (`jump` at L154 is the template).
  - Risk: none; upstream defaults already verified (deviations §3).
  - Size: S.
- **Soft-body cluster core** ✅ (2026-09-10/11): the full cluster-core
  lifecycle is ported and verified against real Bullet — the static
  `solveClusters(bodies)` pass (joints Prepare → `citerations` ×
  Solve(sor = 1) → Terminate + delete), `applyClusters(false/true)`,
  `dampClusters`, and the LJoint / AJoint / CJoint trio with the drift
  split (specs defaults erp = cfm = split = 1, master = `clusters[0]`,
  refs = full inverse-frame · pin). The cluster-link kernels
  `PSolve_CLinks` / `VSolve_CLinks` listed here earlier are a **phantom
  entry**: a full bullet3-master refs sweep found no such symbols (the
  ePSolver set has 4 kernels, the eVSolver set 1 — all ported). The
  `cluster_joints` probe (two 7×7 patches + LJoint@midpoint + AJoint∥Y,
  patch B at +2 m/s) reproduces Bullet bit-for-bit in f64 across 20
  frames after the 2026-09-11 fix of the `terminate` drift-impulse
  parameter order (the split channel had been silently dead since the
  first port); the regression test
  `linear_joint_drift_split_reaches_nodes` pins it.
  Known residual: the `cluster_pair` CL_SS chaos probe still diverges
  qualitatively from Bullet (pre-existing, unrelated to the split
  channel — first frame already off; tracked for a separate round).
  Files: `softbody/cluster.rs`, `softbody/joints.rs`, `softbody/solver.rs`.
- **Collision-event sources** ✅ (2026-09-07): multibody-collider and
  soft-body pairs joined the rigid pairs as event sources.
  - Shipped: `CollisionEvent::Started/Stopped` sides widened from
    `BodyHandle` to a `CollisionObject` enum (`Body` / `Link { body,
    link }` with `link < 0` = base / `Soft`) — Bullet's collision-object
    kinds; pair-level events, not node-level (soft nodes have no stable
    identity). Per-side opt-in extended: `LinkCollider.collision_events`
    and `SoftBody.collision_events` join `RigidBody.collision_events`
    (plain pub fields, like the group/mask pair). Internally the tracked
    pairs moved from `[u32; 2]` body slots to canonically ordered
    `ContactKey` pairs (kind-first: bodies < links < soft), so the sorted
    two-pointer Started/Stopped diff, the sleeping carry-over and
    `overlapping()` (still rigid-only) keep their shape.
  - Touch sources: the mixed collider pass tracks at the sleeping-pair,
    detection-none and detection-hit paths (freeze/none/touched); the
    soft dispatches (`soft_vs_rigid`/`soft_vs_mb`/`soft_vs_soft`) now
    carry a touched signal from `collide_rigid`/`collide_multibody`/
    `collide_cluster_rigid`/`collide_cluster_soft`/`collide_vf_ss` (all
    return bool; contact-count deltas, zero behavior change).
    `remove_soft_body` cascades Stopped like `remove_body`, renumbering
    the surviving soft slots inside the tracked keys.
  - Reference decisions covered: pair-level granularity (soft nodes
    churn); the unsupported `CL_RS`-against-multibody and non-matching
    soft dispatch modes report a stopped touch instead of silently
    keeping stale state; group/mask-filtered pairs stay untracked
    (Stopped falls out of the diff), matching the rigid semantics.
  - Files: `world.rs` (`CollisionObject`, `ContactKey`, diff, removal
    cascades, `pair_opted_in`/`track_pair`), `world/step.rs`
    (mixed-pass collection), `world/softbody_pass.rs` (dispatch
    tracking), `softbody/collide.rs` (bool returns),
    `multibody/body.rs` + `softbody/body.rs` (opt-in fields), facade
    prelude re-export.
  - Risk resolved: opt-in-gated collection keeps quiet scenes at zero
    cost (empty touching sets); rigid-rigid event order and payload
    semantics are unchanged (same diff, new key encoding).
  - Tests: 3 (link-collider Started/Link payload + removal Stopped, soft
    rope pair events, soft-removal Stopped with slot renumbering).
  - Size: M.
- **Persistent-manifold extras** ✅ (2026-09-10): point lifetime
  counters, friction anchors, contact-processed callbacks.
  - Impl: `PersistentPoint.lifetime` + `ContactPoint.lifetime` (capture
    starts at 0, refresh increments survivors — Bullet's `m_lifeTime`
    with the `replaceContactPoint` reset-on-redetect semantics). The
    friction anchor is `CollisionFlags::HAS_FRICTION_ANCHOR` (0x0200):
    the assembly sites project the cached A-side anchor and hand the
    anchor separation to the row (Bullet's `setupFrictionConstraint`
    `positionalError` branch, `m_frictionERP = 0.2` — sign flipped on
    our rows because our friction `rel` is the negation of Bullet's).
    The processed callback became a world-level opt-in drain
    (`set_contact_point_events`, `ContactPointEvent { a, b, point }`),
    cleared each step like `contact_events`.
  - Files: `bullet-collision/src/manifold.rs`, `solver/sequential_impulse.rs`,
    `body.rs`, `world.rs`, `world/step.rs`, facade prelude.
  - Boundaries: the anchor respects the Coulomb cone (clamped by
    `max_friction` — no super-friction; the scene test pins both sides);
    multibody contacts ignore the anchor flag and report no point
    events (no persistent manifold on that path — deviations §1);
    sensor pairs report point events but solve no rows.
  - Tests: 2 manifold (lifetime grow / reset-on-replace), 1 solver
    (servo direction + plain-row no-op), 2 world (Coulomb cone guard,
    point-event drain with growing lifetimes).
  - Size: M.
- **CCD candidate pruning** ✅ (2026-09-02): the sweep tests every body linearly;
  Bullet prunes through a broad-phase AABB query first (cost only).
  - Impl: inside the CCD pass (the swept-sphere closure in `world.rs`,
    `integrate_transforms` stage), wrap the per-target test with
    `BroadPhase::aabb_candidates(swept_aabb)` — output-neutral because
    the exact filter bounds the candidate set.
  - Files: `world.rs` CCD section only.
  - Risk: none numeric; pure culling.
  - Size: S.
- **Toy world variants**: `btSimpleDynamicsWorld` /
  `btContinuousDynamicsWorld` — deliberately not ported. No action.
- **Deformable system**: `btDeformable*` (NeoHookean, reduced volumes)
  — explicitly out of scope, tracked in `design/softbody.md` §11. No action.
