# SoftBody Port Design (Classic btSoftBody)

> Design doc for the `bullet-dynamics::softbody` submodule, porting the classic
> `refs/bullet3/src/BulletSoftBody/` system.
> User rulings (2026-08-31): (1) phase one implements the **complete classic system**
> (cloth/rope/volumetric cluster soft bodies/anchors/soft joints/soft-soft);
> `btDeformable*` (NeoHookean/backward-Euler/Krylov) and `BulletReducedDeformableBody`
> are recorded as backlog, not implemented; (2) node-vs-rigid contact distance
> uses **exact distance** (GJK point distance / closed form) instead of Bullet's
> `btSparseSdf` voxel-interpolation cache; the difference is recorded in deviations.

## 0. Bullet-side Facts (design basis, verified against the source)

- **Stepping order** (`btSoftRigidDynamicsWorld::internalSingleStepSimulation`):
  1. `predictUnconstraintMotion`: rigid velocity integration (parent class) → per soft body
     `predictMotion(dt)` (gravity, external forces, nodes integrated into predicted `x`,
     cluster update, bounds, ndbvt/fdbvt update, pose matching, **clears rcontacts/scontacts**)
  2. `collisionDetection`: dispatcher per broadphase pair; `(soft, rigid)` →
     `btSoftRigidCollisionAlgorithm` → `defaultCollisionHandler(colObj)` fills `m_rcontacts`
     (SDF_RS) or cluster joints (CL_RS); `(soft, soft)` → `btSoftSoftCollisionAlgorithm` →
     `defaultCollisionHandler(psb)` fills `m_scontacts` (VF_SS).
     **soft uses the predicted x, rigid uses the pre-solve pose**
  3. islands: a soft body is a regular collision object and joins union-find
     (soft is non-static → merges islands with touched rigid bodies)
  4. rigid solve (PGS per island) → `integrateTransforms` (rigid position integration)
  5. `solveSoftBodiesConstraints`: `solveClusters` (global cluster-joint iterations) →
     per body `solveConstraints()` (PBD position iterations; contact rows read rigid velocities
     **after** solve and the applied impulses **affect rigid next step** — one-step lagged
     coupling, aligned as-is, do not "improve")
  6. self-collision loop: per body `defaultCollisionHandler(psb)` (CL_SELF only)
  7. `updateSoftBodies` → per body `integrateMotion()` (= `updateNormals`, end-of-step
     normal refresh)
- **PBD core**: `q=x; v+=g·dt; x+=v·dt` → position iterations fix `x` → `v=(x−q)·isdt·(1−kDP)`.
- **Default solve sequence** (`setSolver(Positions)`, what initDefaults uses):
  `psequence=[Anchors, RContacts, SContacts, Linear]`, `vsequence/dsequence=[]`;
  the `Velocities` preset: `vsequence=[Linear(v)]`, `psequence=[Anchors, RContacts, SContacts]`,
  `dsequence=[Linear]`.
- **Default config**: piterations=1 (demos often set 4), viterations=0, diterations=0,
  citerations=4; kDF=0.2, kCHR=1, kKHR=0.1, kSHR=1, kAHR=0.7; kVCF=1, kDP/kDG/kLF/kPR/kVC/kMT=0;
  soft shape **margin=0.25** (contact-band semantics, huge); maxDisplacement=1000
  (anti-explosion per-component clamp); sleepingThreshold=0.04 (but the pipeline never puts
  soft bodies to sleep — see §7).
- **Island semantics**: `updateActivationState` only iterates `m_nonStaticRigidBodies`; soft
  bodies stay ACTIVE_TAG forever → **soft bodies pin the rigid islands they touch awake and
  never sleep themselves** (actual Bullet behavior).
- **Concave route**: the soft body's whole AABB transformed into mesh local space → mesh BVH
  callback per triangle → each triangle gets an extruded hull
  (`BT_SOFTBODY_TRIANGLE_EXTRUSION=0.04`) → the soft-vs-convex algorithm runs
  (SDF cache key = per-triangle hull).
- **`checkContact`**: `dst = SDF interpolation(wtr.invXform(x)) − margin`; contact iff `dst<0`;
  `cti.offset = −dot(n, x − n·dst)`; PSolve_RContacts uses `dp=min(dot(x,n)+offset, mrg)`.

## 1. Scope and Non-Goals

**In scope**: Node/Link/Face/Tetra/Material/Anchor/RContact/SContact/Pose/Cluster data
structures; the predictMotion/solveConstraints/staticSolve solve loops;
PSolve_{Links,RContacts,SContacts,Anchors} + VSolve_Links kernels; applyForces
(7 aero models/Pressure/Volume); K-mean clusters + CL_RS/CL_SS/CL_SELF; LJoint/AJoint/CJoint;
helpers (Rope/Patch/Ellipsoid/FromTriMesh/FromConvexHull + quickhull); anchors (vs rigid);
soft-body raycast; the `{translate,rotate,scale,transform,transform_to,set_total_mass,...}`
transform/mass API; `randomizeConstraints` (built-in deterministic LCG — §8).

**Out of scope (backlog + deviations)**: the `btDeformable*` system
(NeoHookean/Corotated/LinearElasticity force models, backward Euler, CG/CR/preconditioning,
deformable contact projection); `BulletReducedDeformableBody`; SparseSdf (replaced by exact
distance); the **deformable form** of soft↔multibody coupling (`SDF_MDF` /
`btSoftMultiBodyDynamicsWorld` — the classic `CO_FEATHERSTONE_LINK` branch landed in Phase F);
Serialize (btSoftBodyData); GPU solver interface; debug draw; Note/RenderNode
(render mesh refinement `interpolateRenderMesh`); windVelocity stored in worldInfo
(merged into body fields, same effective semantics as Bullet).

## 2. Data Structure Mapping (C++ → Rust)

Bullet uses `Node*` pointer webs plus runtime `Body{Cluster*|btRigidBody*}` wrappers; Rust
replaces everything with indices (Bullet's own serialization path
`pointersToIndices/indicesToPointers` proves index isomorphism works):

| C++ | Rust (`softbody/body.rs`) | Notes |
|---|---|---|
| `Material` | `Material { k_lst, k_ast, k_vst, flags }` | Element::m_tag not ported |
| `Node` | `Node { x, q, v, f, n, im, area, leaf, battach, index }` | `m_vn/m_effectiveMass*` are deformable-only, not ported; same for `m_constrained` |
| `Link` | `Link { n: [u32;2], mat: u32, rl, c0, c1, c2, c3, bending }` | c0=(im0+im1)/kLST (division, source semantics), c1=rl², c2=1/(grad²·c0) at runtime |
| `Face` | `Face { n: [u32;3], mat: u32, normal, ra, leaf }` | m_pcontact/m_n0/m_n1 are deformable-only, not ported |
| `Tetra` | `Tetra { n: [u32;4], mat: u32, rv, leaf, c0:[Vector3;4], c1, c2 }` | `m_Dm_inverse/m_F/P_inv` are deformable force-model fields; the classic kVST only uses c0/c1/c2 |
| `Anchor` | `Anchor { node, body: BodyHandle, local, influence, c0, c1, c2 }` | body = typed handle of the pinned-to rigid body |
| `RContact` | `RContact { cti: Cti, node, c0: Matrix3, c1, c2, c3, c4 }` | the mb-side jacobian is computed at solve time via `MbSolveContext` (Phase F) |
| `Cti` | `Cti { body: ContactOwner, normal, offset }` | `ContactOwner::{Rigid(usize), Link{mb, link}}` (Phase F; index-based stand-in for the `m_colObj` pointer) |
| `SContact` | `SContact { node, face, weights, normal, margin, friction, cfm: [Real;2] }` | |
| `Pose` | `Pose { bvolume, bframe, volume, pos, wgh, com, rot, scl, aqq }` | |
| `Cluster` | `Cluster { masses, nodes, framerefs, framexform, idmass, imass, locii, invwi, com, vimpulses, dimpulses, lv, av, leaf, ndamping, ldamping, adamping, matching, contains_anchor, collide, index }` | btTransform framexform → (Vector3, Quaternion) |
| `Joint/LJoint/AJoint/CJoint` | `enum SoftJoint { Linear(..), Angular(..), Contact(..) }` + `Prepare/Solve/Terminate` dispatched on the enum | the three virtual states of Bullet's Joint |
| `Body{soft,rigid}` | `enum Side { Cluster { soft: SoftBodyHandle, cluster }, Rigid(BodyHandle) }` | impulse dispatch helper (§6 joints); handle-typed since the typed-handles migration |
| `Config` | `Config { ...same-named fields, collisions: SoftCollision, psequence: Vec<PSolverKind>, ... }` | fCollision/fMaterial as bitflags (project convention: bitflags for stackable flag combinations, enums for exclusive states) |
| `SolverState` | `SolverState { sdt, isdt, velmrg, radmrg, updmrg }` | |
| `btSoftBody` | `SoftBody { cfg, sst, pose, nodes, links, faces, tetras, anchors, rcontacts, scontacts, joints, materials, clusters, ndbvt, fdbvt, cdbvt, bounds, b_update_rt_cst, timeacc, max_speed_squared, ... }` | the world owns `Vec<SoftBody>`; `SoftBodyWorldInfo` not ported (gravity/air merged into world and Config) |
| `Impulse{velocity,drift}` | `struct Impulse { velocity, drift, as_velocity, as_drift }` | used by clusters |

**dbvt**: `m_ndbvt` (node tree) / `m_fdbvt` (face tree) / `m_cdbvt` (cluster tree) are plain
`btDbvt` (no pair cache). The internal `Tree` in `broadphase/dbvt.rs` is extracted as
`bullet-collision::dbvt::DbvtTree` (payload-generic over u32/usize), plus `collide_tt`
(tree-tree, used by VF_SS/CL_SS) and leaf `update` (the existing `update_fat` semantics =
Bullet `update`'s fat-volume path). **This refactor touches the broadphase hot path;
acceptance = parity 6 scenes bit-exact + all existing dbvt unit tests.**

## 3. Module Layout

```
crates/bullet-dynamics/src/softbody/
├── mod.rs        // SoftBody aggregate + public API (addForce/setMass/transform/raycast/...)
├── body.rs       // Node/Link/Face/Tetra/Anchor/Cluster/Pose/Material + append* + mass/inertia setters
├── config.rs     // Config/SolverState/PSolverKind/VSolverKind + SoftCollision bitflags + initDefaults
├── solver.rs     // predict_motion/solve_constraints/static_solve + PSolve_*/VSolve_* kernels
├── forces.rs     // apply_forces (aero node/face + pressure + volume)
├── collide.rs    // check_contact (exact distance) / SDF_RS / VF_SS / CL_RS / CL_SS / CL_SELF
├── cluster.rs    // generate_clusters (K-mean) / initialize/update/solve/apply/damp clusters
├── joints.rs     // SoftJoint tri-state Prepare/Solve/Terminate + Side impulse dispatch
├── helpers.rs    // create_rope/create_patch/create_ellipsoid/from_tri_mesh/from_convex_hull
├── quickhull.rs  // minimal quickhull (face generation for CreateEllipsoid/FromConvexHull)
├── rng.rs        // C runtime rand() replay, portable (patch perturbation only)
└── tests.rs      // unit tests (analytic cross-checks)
```

The facade `bullet-physics/src/lib.rs` re-exports; `PhysicsWorld` gains `soft_bodies: Vec<SoftBody>` +
`add_soft_body/soft_body(_mut)/remove_soft_body` (proxy cascades, same pattern as body removal).

## 4. World Stepping Integration (stage insertion points)

Aligned with the §0 order, inserted into our 8-stage `step_simulation` pipeline:

| New stage | Position | Content | Bullet counterpart |
|---|---|---|---|
| 1s | after stage 1 | `soft_predict_motion(dt)`: gravity/external forces/node integration (maxDisplacement clamp)/clusters/bounds/ndbvt/fdbvt update/pose matching/clear contacts | `predictMotion` |
| 3s | inside stage 3 | soft proxy AABB ← `m_bounds` | `updateSingleAabb` (soft as collision object) |
| 4b | islands | one soft body = one island node (unions with touched/anchoring rigid bodies); soft always awake → pins the island awake | `findUnions` + soft always ACTIVE |
| 5s | dispatch inside stage 5 | broadphase pair contains soft owner → skip regular narrowphase, run soft collisions (§5) | dispatcher soft algorithms |
| 7s | after stage 7 (before 8) | `soft_solve_constraints()`: solveClusters (global) → per body PBD (contact impulses hit rigid here — already solved and integrated, affecting next step) → CL_SELF loop | `solveSoftBodiesConstraints` + self loop |
| 8s | beside stage 8 | `update_normals()` (=integrateMotion) | `updateSoftBodies` |

Note: anchored rigid bodies (Anchor) must already be awake before stage 1s (aligns with
Bullet's `a.m_body->activate()` inside `solveConstraints`; we `activate` at PSolve_Anchors
assembly time — needs a mutable world borrow, see the §6 borrow design).

## 5. Collision Design (exact-distance route)

`check_contact(body_idx, node_world_x, margin) -> Option<Cti>`:
- **Convex rigid** (Sphere/Box/Capsule/Cylinder/Cone/ConvexHull/Compound → recursive children):
  point-vs-convex GJK distance (reuses the `gjk.rs` distance walk — the zero shape side is the
  point; Sphere has an analytic fast path). `dst = signed distance − margin`; `dst<0` →
  contact; `normal` = surface→point direction (world space, rigid rotation applied).
- **Plane**: analytic `dot(n,x)+c`, same on both sides as Bullet (its SDF is also exact here).
- **Static TriangleMesh/Heightfield**: node-local AABB → BVH candidate triangles →
  `ClosestPtPointTriangle` (the CCD path already has this) for closest-point distance/normal.
  **Deviation**: Bullet uses extruded-hull + SDF interpolation; we use exact point-triangle
  distance + margin band (pre-registered deviations §11-①②).
- `cti.offset = −dot(n, x − n·dst)` (verbatim; PSolve uses `dp=min(dot(x,n)+offset, mrg)`).

**SDF_RS** (`(soft, rigid)` pair): `ndbvt.collide_tv(root, rigid_world_aabb + margin)` → per
hit node `check_contact` → `RContact`: `ra = x − COM_rigid`, `va = rigid.velocity_at_local_point(ra)·sdt`,
`vb = x − q`, `vr = vb − va`, `dn = dot(vr,n)`, `fv = vr − n·dn`, `fc = kDF·friction_rigid`,
`c3 = |fv|² < (dn·fc)² ? 0 : 1−fc`, `c4 = static ? kKHR : kCHR`,
`c0 = ImpulseMatrix(sdt, ima, imb, invI_rigid, ra)`, `c2 = ima·sdt`. Anchored nodes (battach)
skip; dynamic bodies add the `timemargin` (positional step amount, verbatim
`dynmargin/stamargin`).
`(soft, soft)` pairs → **VF_SS**: `collide_tt`(ndbvt_a, fdbvt_b) both directions → per
(node, face) point-triangle with margin penetration and barycentric weights → `SContact`
(cfm via the `kSHR` formula; cross-check `CollideVF_SS::Process` at implementation time).
**CL_RS/CL_SS/CL_SELF** (cluster mode, Phase D): cluster convex hull (quickhull over cluster
nodes, a temporary `ConvexHull` support walks the GJK band distance) → `CJoint`
(massmatrix/erp/split verbatim from `ClusterBase::SolveContact`); CL_SS skips same-body
adjacent clusters via the cluster connectivity matrix.

## 6. Solve Kernels (formula list; implement each against btSoftBody.cpp line numbers)

| Kernel | Bullet lines | Formula / notes |
|---|---|---|
| predict_motion | 2099 | `sdt=dt·timescale; velmrg=sdt·3; radmrg=margin; updmrg=radmrg/4`; `v += g·sdt` then `f·im·sdt` (per-component clamp ±maxDisp/sdt); `x += v·sdt`; `f=0`; ndbvt leaf=FromCR(x, radmrg)+velocity extrapolation, fdbvt leaf=VolumeOf(face)+pad |
| solve_constraints | 2208 | assembly (links c3=qb−qa, c2=1/(c3²·c0); anchors c0/c1/c2) → viterations sequence (then `x=q+v·sdt`) → piterations (`ti=isolve/piters` sequence) → `v=(x−q)·isdt·(1−kDP)` → diterations (q=x, sequence, `v += (x−q)·kVCF·isdt`) → dampClusters+applyClusters(true) |
| PSolve_Links | 4004 | `k=(c1−len)/(c0·(c1+len))·kst`; `xa−=del·k·ima; xb+=del·k·imb` (len=\|del\|²) |
| VSolve_Links | 4027 | `j=−dot(c3, v0−v1)·c2·kst` |
| PSolve_Anchors | 3873 | `vr=(va−vb)+(wa−x)·kAHR`; `impulse=c0·vr·influence`; `x+=impulse·c2`; `rigid.apply_impulse(−impulse, c1)` |
| PSolve_RContacts | 3894 | `vr=vb−va; dn=dot(vr,n)`; when `dn≤EPS`: `dp=min(dot(x,n)+offset, mrg)`; `impulse=c0·((vr−fv·c3+n·dp·c4)·kst)`; `x−=impulse·c2`; `rigid.apply_impulse(impulse, c1)` (**note the sign: opposite to anchors; the source is like that**) |
| PSolve_SContacts | 3969 | `vr=(x−q)_node−(p−q)_face`; when `dot(vr,nr)<0`: `j=margin−(dot(nr,x)−dot(nr,p))`, `corr+=n·j`; `corr−=ProjectOnPlane(vr,nr)·friction`; node `x+=corr·cfm0`, face three points `x−=corr·cfm1·w_i` |
| apply_forces | 3750 | pressure: `f += n·(area·(1/\|volume\|·kPR))`; volume: `f += n·(area·(pose.volume−volume)·kVC)`; aero see forces.rs |
| ImpulseMatrix×3 | internals 1004-1032 | `Diag(1/dt)·(Diag(ima)+MassMatrix(imb,iwi,r))⁻¹` etc., MassMatrix=iwi−iwi·r×·iwi·r×ᵀ (verbatim) |
| clusters | cluster.rs | K-mean (generateClusters k/maxiter), idmass/imass/locii (point-mass inertia), solveClusters (citerations·SOR=1), applyClusters(drift), dampClusters(ndamping/ldamping/adamping), matching (pose matching kMT) |
| joints L/A/C | joints.rs | LJoint: position anchoring mass matrix; AJoint: axis alignment (IControl→trait callback); CJoint: contact friction joint (life/maxlife, split drift). Formulas vs btSoftBody.cpp 3390-3747 |

**Borrow design** (the key difficulty of world integration): PSolve_RContacts/Anchors need
"read soft nodes + read-write rigid velocities". Approach:
`solve_constraints(&mut self.soft_bodies, &mut self.bodies, ...)` uses field-level split
borrows (soft_bodies and bodies are distinct PhysicsWorld fields, naturally `&mut`
simultaneously); rigid impulses are written directly through an `apply_impulse`-equivalent
(bypassing the solver delta pool — Bullet also writes directly via
`btRigidBody::applyImpulse` here).

## 7. Sleep / Activation

- Soft bodies are **always awake** (aligned with Bullet: `updateActivationState` only iterates
  the rigid list); island merging pins touched rigid bodies awake.
- `update_deactivation/wants_sleeping/set_zero_velocity` are not wired into the pipeline
  (same as Bullet's actual pipeline); recorded in deviations.
- max_speed_squared/maxDisplacement clamps verbatim (anti-explosion).

## 8. Determinism (rand replication)

`randomizeConstraints` uses Bullet's **built-in deterministic LCG** (local seed=243703,
`seed = (1664525·seed + 1013904223) & 0xffffffff`, shuffles links/faces by `seed % len`) —
already bit-exact replicated in Rust, no rand dependency.
`CreatePatch`'s perturbation uses C `rand()` — both probe sides compile with MSVC, so Phase C
replicates MSVC's sequence (`state = state·214013 + 2531011; (state>>16)&0x7fff`, `srand(0)`
means state=0) in plain integer arithmetic, hence portably: the same replay runs on every
target and needs no MSVC toolchain. Patch builds default to `perturbation=0`. Bullet's rand
differs under clang — probes use MSVC (the parity convention).

## 9. quickhull (`quickhull.rs`)

`CreateEllipsoid` (Hammersley point set, deterministic) and `CreateFromConvexHull` (user
vertices) need hull face generation. Self-implemented incremental quickhull (a reduced
version of Stan Melax's HullLibrary semantics: initial tetrahedron → farthest-point
expansion → coplanar tolerance), emitting vertex indices + triangle faces → links
(deduped under `idx0<idx1`) + faces. **Known difference**: the hull vertex order/winding may
differ from Bullet's HullLibrary → different link generation order → different PBD solve
order → settle-class scatter (deviations §11-⑤).

## 10. Phase Plan (each phase lands independently + check.py all green + clippy clean on both features)

- **Phase A: skeleton + rope dynamics** ✅ done (2026-08-31)
  body/config/solver(PSolve_Links/VSolve_Links)/helpers(create_rope); world stage 1s/7s
  skeleton (no collisions); facade re-export; demo `softbody_rope` (pendulum/catenary/free
  fall); 8 unit tests: analytic free-fall mirror (per-node semi-implicit Euler cross-check +
  f32 elimination tolerance note), swing past bottom/energy retention, catenary sag, mass API,
  link constant formulas, randomize determinism (LCG replication), stretch recovery,
  world-vs-manual-loop bit-exactness.
  Acceptance: check.py all green (f32/f64 + 10 demos), clippy clean on both features,
  parity 6 scenes match the ledger (zero regression).
  **Implementation corrections**: (1) `randomizeConstraints` is the built-in LCG, not rand()
  (§8 corrected; no MSVC rand replication needed); (2) `updateLinkConstants`'s
  `c0 = (im0+im1) / kLST` is a division (§6 table corrected); (3) `setTotalMass` scales
  `im *= total/mass`.
- **Phase B: SDF_RS collisions + anchors + broadphase/island integration** ✅ done
  (2026-08-31 late night + probe same day)
  DbvtTree generalized out of dbvt.rs (Tree→pub DbvtTree, insert/remove/update_fat/
  for_each_overlapping/optimize_incremental/clear made public — pure visibility refactor,
  broadphase logic unchanged); check_contact (plane analytic / convex GJK point-sphere /
  static mesh·heightfield point-triangle closest); SDF_RS→RContact; PSolve_RContacts/
  PSolve_Anchors (solve_constraints takes &mut bodies, field-level split borrow);
  add_soft_body proxy/ProxyOwner::Soft dispatch/4b island soft-pins-awake/5s collisions/
  3s bounds proxy refresh/remove_soft_body + remove_body anchor cascade.
  Demo: `softbody_rope_ground` (rope settles in band on plane 0.149 / rope drapes over sphere
  without tunneling 1.23 / dual-anchor cargo swing retention).
  **Phase B second half done (2026-08-31)**: probes — `tools/softbody_probe.cpp` +
  `examples/softbody_probe.rs` + `tools/build_softbody_probe.py` (links `BulletSoftBody.lib`;
  cmake-build that target first), scenes rope_pendulum/rope_ground/cloth_plane ×f32/f64,
  ledger `docs/bullet-compare.md` "SoftBody probes".
  Readings: pure PBD (rope_pendulum) aligns within print precision (f32 287/300 byte-exact) —
  the A/B solve kernels are confirmed; contact scenes share the first-touch frame (57) and
  differ by 85–86mm at settle (f32=f64 → structural) — Bullet's soft-vs-plane route stacks
  three approximations (concave triangulation + GJK + voxel interpolation;
  `isConvex(plane)`=false, plane never enters the GJK branch), lifting the effective contact
  band above the geometric margin; our analytic distance settles at margin+3mm.
  deviations §1 has two entries (exact-distance quantification, soft never sleeps).
  Soft-body raycast moves to Phase C (needs fdbvt faces, see below).
  **Implementation notes/corrections**: (1) Bullet's `timemargin` is dead code (wtr and ctr
  are the same object) → dynmargin=stamargin; (2) with `dst = distance − margin` semantics,
  `offset = dst − dot(n,x)` and PSolve's dp is the detection-time dst; (3) **tunneling bound**:
  the margin band must be ≥ the per-step displacement (v·dt) — Bullet's 0.25 default exists
  for this (~15 m/s cap); test margin 0.05 with 7.7 m/s impact must tunnel (the same physical
  limit in Bullet, not a bug); (4) PBD chain under-convergence stretch (hanging 31-node chain
  ~60% elongation at 4 iterations) and 33% anchor-swing stretch are Bullet's own signatures —
  assertions follow the morphological signatures; (5) **found a pre-existing rigid bug (not
  introduced this round)**: a wide flat box (e.g. 1×0.1×1) landing on a plane contacts on one
  corner → self-spin → sinks/tunnels; reproduced at HEAD (cube boxes fine, parity scenes never
  covered flat boxes); ignored test + backlog entry recorded.
  Acceptance: check.py all green (f32/f64 + 11 demos), clippy clean on both features,
  parity 6 scenes digit-identical, dbvt 13 tests green (mb path diff empty, argued skip).
- **Phase C: Faces + cloth phenomena** ✅ done (2026-08-31)
  faces/update_normals/update_area (folded into update_constants tail; fdbvt rebuilt on
  topology change); all 7 aero models (forces.rs, V_*/F_* families + lift/drag +
  ApplyClampedForce reversal clamp); pressure/volume (getVolume divergence theorem,
  setPose/updatePose — PolarDecompose deferred to Phase D; kVC uses setPose(true,false));
  PSolve_SContacts + VF_SS (collide_tt tree-tree for_each_overlapping_pair with Bullet's push
  order; solve_constraints becomes an associated function
  `SoftBody::solve_constraints(idx, &mut [SoftBody], bodies)` with field-level split — the
  soft contact kernel writes nodes on both bodies); create_patch (full fixeds bitmap/gendiags/
  perturbation + CrtRand replication — rand consumes 2 draws per node and only affects
  geometry when perturbation≠0); soft-body raycast (SoftBody::ray_test full face scan =
  Bullet's bcountonly path + PhysicsWorld::raycast_soft_closest; fdbvt pruning recorded as a
  perf backlog). Demo: `softbody_flag` (12th check.py demo: pinned top corner wind bulge 0.745
  + two-panel stitched pillow kPR inflation 19× bounded). Probes: patch_catenary (ulp-level
  first divergence → 1.6cm settle scatter, the 385-link operation-order class),
  soft_soft_vf (**first touch same frame 43, both sides converge to the margin sum 0.504/0.505
  — VF_SS band semantics aligned**; the absolute height offset inherits the lower panel's
  plane settle deviation). Tests +6 (patch topology/area + closed-box volume/VF_SS in-band
  retention/raycast hit+fraction/aero downwind drift/pillow inflation) on both features.
  **Implementation notes/corrections**: (1) **fCollision::Default here = SDF_RS alone**
  (no VF_SS — old memory error corrected); soft-soft needs explicit opt-in, and the handler's
  switch matches VF_SS with **exact match** on `cf & SVSmask` (combination bits fall through
  to no-op, verbatim); (2) Bullet's fdbvt is cleared + initializeFaceTree at predictMotion's
  head on topology change (only when the VF_SS bit is on) — appending faces does not build the
  tree; (3) CollideVF_SS's "skip faces containing the node" loop is dead code in Bullet
  (continue bound to the wrong scope), ported verbatim unfixed; (4) collideTV/collideTT's
  Bullet stack order pops **childs[1] first** — DbvtTree::walk_overlapping aligned (softbody
  rcontacts order), collideTT added with the same push order; (5) CreatePatch's rand
  consumption is ungated (perturbation is only a coefficient); geometry is deterministic when
  perturbation=0 ✓.
- **Phase D: Tetra + cluster volumetric soft bodies** ✅ done (2026-08-31)
  tetras (`append_tetra`/`set_volume_mass`/`set_volume_density` — **the classic system has no
  kVST solve kernel**; tetras only drive the mass distribution and `generate_clusters(0)`'s
  per-tetra fallback; the deformation-gradient fields are NeoHookean-only, not ported);
  `set_total_mass_from_faces` (`setTotalMass(mass, true)` area-weighted — required for
  volumetric soft bodies); pose completion (`update_pose`'s PolarDecompose +
  `scl = aqq·rᵀ·Apq` + maxvolume clamp — kMT matching takes effect); K-mean
  `generate_clusters` ((i·29873)%k seeding / w=slope relaxation / face merge / master swap to
  index 0 / fallback has no master — verbatim) + cluster inertia/solve/damping/apply;
  quickhull (self-implemented: initial simplex → farthest-point expansion → horizon rebuild;
  vertices = input indices in ascending order — the vertex-order difference vs HullLibrary is
  recorded in deviations); CL_RS/CL_SS/CL_SELF (cluster implicit hull = node point-set
  support, no quickhull needed; GJK Band/EPA vs rigid convexes; planes use the extreme-vertex
  analytic route = exact equivalence); CJoint (Prepare/Solve/Terminate verbatim, the solve
  split into `compute` (read-only) + side_apply (write per side) to satisfy borrows — joint
  sides may live on any soft body); `generate_bending_constraints` (distance==2 adjacency
  specialization + general Floyd).
  Demo: `softbody_ellipsoid` (13th check.py demo: volumetric sphere lands, flattens, rebounds,
  min_y 0.18 at settle / shape half-height 0.95; CL_SS side-by-side push-apart — **CJoint
  friction applying only when rvac<0 is Bullet's verbatim behavior**, sliding after separation
  without decelerating). Probes: ellipsoid_ground (first 21 frames within 1e-4, settle differs
  122mm = Bullet's plane-triangulated band vs the analytic band — the cluster-scale
  rope_ground class; f32=f64 structural), cluster_pair (**Bullet's own CL_SS numerically
  explodes on large-overlap first contact, tunneling to −223m**; we push apart steadily and
  settle — the gimpact_stack class of "Bullet's own weakness"; root-caused 2026-09-18 as
  blow-up regime + Bullet's tunneled triangulated plane at explosion speed + hull-order
  scatter, no port bug — the `cluster_pair_soft` light-touch variant agrees qualitatively,
  see `bullet-compare.md` Phase D; the behavioral divergence is
  recorded in deviations). Tests +6 (quickhull 3 + ellipsoid topology/volume, face-area/tetra
  volume mass, bending count, cluster master/connectivity, per-face fallback, polar rotation
  recovery, pose rotation tracking, CL_RS settle, CL_SS separation, hull interior-point
  rejection) on both features.
  **Implementation notes/corrections (most important this round)**: (1) **the glam
  column-major outer-product trap** — transcribing Bullet's row assembly
  `Apq[row] += a[row]*b` into glam stores the **transpose** (symmetric matrices don't expose
  it; `update_pose`'s Apq/cluster covariance got the inverse rotation — caught by the pose
  test); the correct form is `m += Matrix3::from_cols(a*b.x, a*b.y, a*b.z)`;
  (2) K-mean empty clusters get deleted by Terminate (cluster count ≤ k+1); the fallback
  (k=0) **installs no master** (Bullet's indentation, verbatim); (3) the cluster impulse
  chain: joint→cluster_v_impulse (immediate lv/av + vimpulses accumulation) →
  `apply_clusters(false)` at solve_constraints' head spreads to node positions; (4) co-axial
  cluster stacking is unstable (numerically asymmetric horizontal slide — demo scene 2 uses
  side-by-side push-apart).
- **Phase E: soft joints + API completion** ✅ done (2026-08-31)
  The `SoftJoint` enum (Linear/Angular/Contact dispatching Prepare/Solve/Terminate on the
  enum — the three virtual states of Bullet's Joint; after CJoint merged in, `solve_clusters`
  unifies prepare (mem::replace borrow pattern)/compute (read-only half-impulses)/apply/
  terminate); LJoint/AJoint verbatim (3587/3630: maxdrift=4/π·16 clamps,
  ImpulseMatrix/AngularImpulseMatrix, dual drift-split channels); `trait AngularControl`
  (IControl's prepare(joint)/speed(joint,current), keeping SteerControl's ref mutation —
  prepare uses a take pattern to satisfy borrows; `clone_box` replaces Bullet's shared static
  singleton) + `DefaultAngularControl`; Side extension (xform/inv_mass/inv_world_inertia/
  angular_velocity — cluster side framepos/framerot/invwi/av, rigid side world transform/
  invInertiaTensorWorld/ω) + side_apply angular (cluster_va/da_impulse verbatim 1321/1330,
  new rigid `apply_torque_impulse` = Bullet applyTorqueImpulse); append_linear/angular_joint
  (refs = both sides' xform-inverse-localized; the world convenience
  `add_soft_linear/angular_joint` uses master cluster = Bullet's single-arg overload);
  transform API: translate/rotate/scale/transform/transform_to/get_rigid_transform (transform
  re-inserts ndbvt leaves exactly = new `DbvtTree::update_exact`; updateConstants recomputes
  rl → scaled link lengths follow geometry; transform_to's Kabsch rotation reuses the cluster
  polar decomposition R=Wᵀ (= V·Uᵀ at full rank, recorded in deviations); rest_positions is a
  constructor-time snapshot of m_X, never refreshed — Bullet quirk verbatim); cut_link
  (MatchEdge undirected matching, link/face edge splits, !done node rollback); refine
  (ImplicitFn trait, btSymMatrix symmetric triangle indexing, ImplicitSolve secant iterations
  256 steps, mass-redistribution f formula, face reprocessing loop (--i;break), cut duplication
  + signed redirection + orphan cleanup — the `todetach==0` sentinel quirk verbatim).
  Demo `softbody_joints` (14th check.py demo: LJoints drag the cloth follow err 0.69 / an
  AJoint motor holds exactly 1.500 rad/s); tests +8 on both features (transform
  translate/rotate/scale, transform_to fit recovery, cut_link topology/velocity inheritance,
  refine interpolation/mass conservation, cut sign semantics + chain clearing (Bullet
  verbatim), LJoint anchoring against gravity, AJoint axis-aligned convergence, motor constant
  speed, torque impulse). **Implementation corrections**: (1) Bullet's `m_X` is snapshotted
  once in the constructor (append/cut/refine never refresh it) — rest_positions verbatim;
  transform_to's fit has no samples for newly appended nodes; (2) refine's node duplication
  condition is `|eval|<accuracy` (narrow band) while link redirection uses `eval<accuracy`
  (negative half-space) — two different predicates, both verbatim; (3) the cluster frame's
  xform takes effect **one step late** after rotate/transform (update_clusters runs inside
  predict_motion).
- **Phase F: classic soft↔multibody coupling** ✅ done (2026-09-01)
  The classic `CollideSDF_RS` `CO_FEATHERSTONE_LINK` branch (not the deformable system's
  SDF_MDF): the `ContactOwner` enum (Rigid/Link{mb,link}) replaces `Cti.body: usize`;
  `collide_rigid` refactors out the shared kernel `collide_body` (DoNode parameterization:
  shape/pose/imb/inv_inertia/velocity_at/friction/is_static); `collide_multibody` assembles
  kinematically (the `m_rigidBody` upcast-failure semantics: imb=0, iwiStatic, va=0 —
  btSoftBodyInternals.h 1613-1653 verbatim); the solve side gets `MbSolveContext`
  (multibodies + scratch + jac/delta pools) passed into `solve_constraints`; the RContacts
  position kernel's Link branch: `fill_constraint_jacobian` → `vel·jac` for va → after the
  position fix, `apply_delta_vee(delta, −impulse.length()·0.5)` writeback (PSolve_RContacts
  3894-3966 verbatim, including the `impulse.length()` quirk); world 5s pair dispatch
  (Soft, Collider) → `soft_vs_mb` (sleeping mb skipped + RVS_MASK exact match with SDF_RS +
  FK poses); CL_RS unsupported against mb (Bullet's cluster route degrades to a massless
  one-sided push for link colliders — recorded in deviations). Tests +4 on both features
  (rope resting on a swinging ball's writeback, pendulum sweeping a rope, sleeping mb woken by
  soft contact, CL_RS ignoring mb). Probe scene `rope_ball_tunnel` (both sides: an in-band
  offset long rope pressing a ball + a fast-impact tunneling contrast — **a two-engine shared
  limit**, quantified in the deviations "SDF_RS band capacity" ledger entry).
  **Key measurement**: a short rope under a 3.9 m/s falling ball tunnels in stock Bullet too
  (−186.8 vs −181.0 @400 frames, the first 23 frames byte-identical) — a capacity limit of the
  classic per-frame partial-correction system, not a porting deviation; the draped variant is
  qualitatively identical on both sides (holds ~30 frames → slides off, ~3-frame phase offset).
- perf: `cloth_wind`/`softbody_rain` perf scenes added after C/D (runner extensions on both
  sides); `softbody_mesh` (the SDF home-turf decomposition variant) added later.

## 11. Pre-registered Deviations (deviations draft; merged into docs/bullet-deviations.md as they land)

1. **No SparseSdf**: node-vs-rigid distance is exact; Bullet's 0.25-voxel trilinear
   interpolation error disappears → settle-position differences within the contact band
   (probe-quantified; expected in the same class as the "margin shell 44mm hover").
   **Quantified (2026-08-31)**: plane scenes differ 85–86mm at settle (Bullet's three stacked
   approximations lift the effective band above the geometric margin; we settle at margin+3mm)
   — landed as deviations §1 "Soft body: node-vs-rigid distance".
2. **Static meshes skip the extruded hull**: exact point-triangle distance + margin band
   (Bullet: triangle ±0.04 extrusion + SDF).
3. **Soft bodies never sleep** (aligned with Bullet's actual pipeline, not our simplification).
4. **mb coupling** (landed 2026-09-01, Phase F: the classic `CO_FEATHERSTONE_LINK` branch +
   jacobian writeback; CL_RS unsupported against mb — Bullet likewise degrades to a massless
   one-sided push, recorded in deviations).
5. **quickhull vertex-order differences** → link order → settle-class scatter.
6. **rand replication is only guaranteed MSVC-toolchain-probe-identical**.
7. Soft-solve impulses on rigid lag one step (aligned with Bullet's order; do not "improve"
   into synchronous).

## 12. Risks and Mitigations

- **PBD order sensitivity**: prefer deterministic probe construction (fixeds /
  perturbation=0 / randomize off); shuffled scenes fall into the chaos-class tolerance.
- **margin 0.25's huge contact band**: our margin semantics (Bullet's `dst<margin` contact
  test + `dp=min(...,mrg)` clamp) copied formula-for-formula; probe rope_ground/cloth_plane
  are the first quantification points.
- **The DbvtTree refactor touches broadphase**: Phase B's first cut; acceptance = parity 6
  scenes bit-exact + 13 dbvt unit tests + spot-checked mb probes.
- **Borrow conflicts**: world field-level splits (soft_bodies/bodies borrowed `&mut`
  separately); joint `Side` impulses dispatch inside a `PhysicsWorld` method (no cross-object
  `&mut` retention).
- **Soft-body chaos is intrinsic**: probe metrics use "first N frames bit-exact/high-precision
  + settle quantities (COM/sag/node height distribution/awake)", per the mesh/two_link class
  experience; do not chase long-horizon bit-exactness.
- **Precision-dependent numeric constants** (SIMD_EPSILON branches; run f32/f64 both): the
  twist lesson (§6 joint round) applies — `btFuzzyZero/SIMD_EPSILON` thresholds branch on
  precision.

## 13. Port Cross-Reference Index

| Ours | Bullet |
|---|---|
| softbody/{mod,body,config}.rs | btSoftBody.h + btSoftBody.cpp (structure/append/mass/transform/raycast) |
| softbody/solver.rs | btSoftBody.cpp 2099-2352 (predict/solve/static) + 3873-4072 (PSolve/VSolve/getSolver) |
| softbody/forces.rs | btSoftBody.cpp 3750-3815 + addAeroForceToNode/Face |
| softbody/collide.rs | btSoftBodyInternals.h 1440-2000 (CollideSDF_RS/VF_SS/CL_*/ClusterBase) + checkContact 2748 |
| softbody/cluster.rs | btSoftBody.cpp generateClusters/initializeClusters/updateClusters/solveClusters/applyClusters/dampClusters |
| softbody/joints.rs | btSoftBody.cpp 3390-3747 (Joint::Prepare/Solve/Terminate) + Body 515-620 |
| softbody/helpers.rs | btSoftBodyHelpers.cpp 691-1120 (Rope/Patch/Ellipsoid/TriMesh/ConvexHull) |
| softbody/quickhull.rs | Extras/ConvexDecomposition/… HullLibrary semantics, reduced (new implementation) |
| world.rs stages 1s/3s/4b/5s/7s/8s (now `world/step.rs` + `world/softbody_pass.rs` + `world/islands.rs`) | btSoftRigidDynamicsWorld.cpp 69-118 + btDiscreteDynamicsWorld stepping |
