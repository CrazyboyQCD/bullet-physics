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

- **P3 — minimal debug draw** (next up): `world.debug_primitives(...)`
  emitting AABBs, contact points, manifold normals and wheel rays with a
  `DBG_*`-style category mask. Producer side only; the consumer stays
  caller-side (e.g. a Bevy `Gizmos` bridge).
  - Impl: `DebugPrim` enum (`Aabb`, `ContactPoint`, `Normal`,
    `Ray`) + a flags bitmask; `PhysicsWorld::debug_primitives(mask, &mut
    impl FnMut(DebugPrim))` walks live bodies' world AABBs
    (`BroadPhase::proxy_aabb` exists), the pair manifold cache's cached
    points/normals, and `RaycastVehicle` wheel rays.
  - Files: `bullet-dynamics` (new `debug.rs` or world.rs methods),
    facade re-export, `vehicle.rs` (wheel-ray source).
  - Risk: decide what a "contact point" is — solver rows or cached
    manifold points (the two diverge between steps); per-category gating
    so quiet masks cost nothing.
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
- **ghost persistent overlap list**: sensor flag + events +
  `contact_test` cover the main use case; Bullet's `btGhostObject`
  listing is the full version.
  - Impl: the sensor touch state already exists in the pair pass (it
    feeds `sensor = true` events); expose a per-body query
    `world.overlapping(sensor) -> Vec<usize>` reading that state, or
    mark sensor bodies and collect their pairs during the pair pass.
  - Files: `world.rs` pair pass + a small query API.
  - Risk: must respect sleeping-pair carry-over (events already solve
    this — reuse their state, don't re-derive).
  - Size: S–M.
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
- **motion states**: Bullet's interpolation / callback pose sync
  (`btMotionState`).
  - Impl: a `MotionState` trait (`synchronize(&Transform)`, optional
    `initial_pose()`), stored as `Option<Box<dyn MotionState>>` on
    `RigidBody`, driven from the step tail where transforms are final
    (Bullet's `synchronizeMotionStates` point).
  - Files: `body.rs` (trait + field), `world.rs` (step tail).
  - Risk: the project API rule keeps hot paths `dyn`-free — a
    per-step consumer callback is exactly the sanctioned exception, but
    it must stay off the solver path (call once per step, not per
    contact). Ownership: `Box` per body only when opted in.
  - Size: S–M (the design debate is the cost, not the code).
- **kinematic rigid body**: the full kinematic type surface
  (`CF_KINEMATIC_OBJECT` + velocity propagation).
  - Impl: `CollisionFlags::KINEMATIC` (next free bit after
    `NO_CONTACT_RESPONSE` = 0x0001) + `is_kinematic()`; integration
    moves the pose from velocity but applies no gravity/damping/forces;
    solver and island paths treat it as infinite mass (its `inv_mass`
    is already 0 — mostly gating, not new math).
  - Files: `body.rs` (flag + accessors), `world.rs` — every
    `is_static()` gate in integrate / islands / solver prepare / CCD
    needs an `is_static_or_kinematic()` review pass (~20 sites).
  - Risk: the widest blast radius in this tier — a missed gate lets a
    kinematic body gain impulses or join islands; kinematic bodies must
    wake dynamic contacts they move into (Bullet's contact generation
    does this for free since pairs are detected).
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
- **compound child transforms**: fixed at construction; no
  `removeChildShape` / dynamic child transforms.
  - Impl: `Compound` owns `Vec<CompoundChild>`; add
    `remove_child(i)` / `set_child_transform(i, ...)`, recompute the
    compound's local AABB (+ cached tree, if touched), and refresh the
    owning body's broad-phase AABB (it re-expands on the next
    `update_aabbs` pass anyway for awake bodies).
  - Files: `bullet-collision/src/shapes/compound.rs`,
    `bullet-dynamics/src/body.rs` (inertia invalidation).
  - Risk: the body's inertia was computed at construction from the old
    children — either recompute on mutation (Bullet recomputes via
    `calculateLocalInertia` on demand) or document stale inertia;
    mutation must happen between steps (solver rows hold no child
    references, so inter-step is safe).
  - Size: M.

## Farther (on demand — see deviations §3 for details)

- **Rigid-body property surface**: anisotropic friction, rolling
  friction, per-object contact stiffness/damping
  (`CF_HAS_CONTACT_STIFFNESS_DAMPING`), the opt-in additional-damping
  block.
  - Impl: plain fields on `RigidBody` consumed in
    `ContactConstraint::prepare` (anisotropic scaling inside the
    effective-mass/friction setup; rolling friction adds an angular
    row, Bullet's `rollingFriction` path); contact stiffness/damping
    replaces the default erp/cfm route in row assembly.
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
- **`btShapeHull`**: hull decimation / debug hull tool.
  - Impl: sample the convex shape's supports over a direction set and
    run the existing self-implemented quickhull (already in the softbody
    cluster path) to produce a `ConvexHull`.
  - Files: `bullet-collision/src/shapes/` (utility fn).
  - Risk: none; pure utility.
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
- **Soft-body cluster kernels**: `PSolve_CLinks` / `VSolve_CLinks`
  (Bullet's default presets never install them; custom solver sequences
  would need them).
  - Impl: the softbody solver sequence gains the two cluster-link
    kernels; only reachable when a user installs a custom sequence.
  - Files: `softbody/solver.rs`.
  - Risk: none for default behavior (presets unchanged).
  - Size: S. See: deviations §3.
- **Collision-event sources**: multibody-collider and soft-body pairs
  are not event sources yet (events are rigid-rigid only).
  - Impl: the mixed passes grow the same touch-state tracking the rigid
    pass has (pair id = collider/link/node index), emitting the existing
    `CollisionEvent` variants; per-body opt-in extends to
    `LinkCollider`/`SoftBody`.
  - Files: `world.rs` mb/soft pair passes, `body.rs` flags.
  - Risk: pair identity for soft nodes churns (nodes have no stable ids
    beyond slot) — pair-level events, not node-level.
  - Size: M.
- **Persistent-manifold extras**: point lifetime counters, friction
  anchors, contact-processed callbacks.
  - Impl: lifetime counters are a small field on `ContactPoint` +
    capture/refresh updates; the callback maps onto a world-level drain
    (zero-dyn rule), not a per-pair function pointer.
  - Files: `bullet-collision/src/manifold.rs`, `world.rs` refresh path.
  - Risk: fields must stay out of the numeric path (warm start reads
    impulses only) or parity breaks.
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
  — explicitly out of scope, tracked in `softbody.md` §11. No action.
