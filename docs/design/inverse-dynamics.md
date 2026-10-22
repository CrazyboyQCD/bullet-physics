# Design: `bullet-inverse-dynamics` (port of `BulletInverseDynamics`)

> Draft for maintainer review (2026-10-21). Upstream reference:
> `refs/bullet3/src/BulletInverseDynamics` (17 files, 4281 lines total; the
> algorithm core is `details/MultiBodyTreeImpl.cpp`, 1286 lines). The
> `Extras/InverseDynamics` helpers are in the license-excluded region and are
> neither copied nor included (§10).

## 1. Goals

- Port the standalone inverse-dynamics tree library
  (`btInverseDynamics::MultiBodyTree`) as a **fifth workspace crate**
  depending only on `bullet-math`.
- **Bit-level numeric parity** with the upstream library across the whole
  calculated surface (inverse dynamics, mass matrix, kinematics, jacobians),
  pinned by a C++ probe + Rust twin pair.
- **Perf & memory ≥ upstream**: zero heap allocation per `calculate_*` call
  after `finalize`, SoA state layout, f32/f64 generics, no `dyn` in hot paths.
- `no_std` like the rest of the workspace (`std` default / `libm` fallback
  through `bullet-math::math`).
- Occupy the ecosystem slot upstream fills: PyBullet's `calcInverseDynamics`
  command is the flagship consumer
  (`examples/SharedMemory/PhysicsServerCommandProcessor.cpp`); the example
  browser ships an `InverseDynamicsExample`.

Non-goals (recorded, not ported):

- The Eigen backend (`IDConfigEigen.hpp`) — upstream optional config; the
  default build uses the LinearMath shim (`IDConfig.hpp:104-105`).
- `Extras/InverseDynamics` tree creators and the `btMultiBody` bridge
  (`btMultiBodyTreeCreator`, `invdyn_bullet_comparison`): license-excluded
  region. Upstream's `MultiBodyTree` shares no code with `btMultiBody`
  either; consumers sync q/u themselves. We stay symmetric: no bridge.
- Force elements (springs/dampers/friction): an upstream TODO
  (`MultiBodyTree.hpp:55`), not implemented upstream either.
- Inverse kinematics: **not part of upstream BID** — `MultiBodyTree.hpp` has
  no IK entry point (IK lives in example-layer code upstream).

Quality discipline (house standard): **parity first**. Any semantic
improvement over upstream (e.g. the missing triangle-inequality branch,
§7-Q1) is a recorded deviation and opt-in, never silently mixed in.

## 2. Upstream surface inventory

| Upstream file | Lines | Role |
|---|---|---|
| `MultiBodyTree.hpp/.cpp` | 367/548 | Public façade: `addBody`/`finalize` + calculate/getter dispatch |
| `details/MultiBodyTreeImpl.hpp/.cpp` | 288/1286 | `RigidBody` struct + all algorithms |
| `details/MultiBodyTreeInitCache.hpp/.cpp` | 116/131 | Builder-time storage (`InertiaData`/`JointData`) |
| `IDMath.hpp/.cpp` | 99/510 | Transform helpers + validation gates + tolerances |
| `details/IDLinearMathInterface.hpp` | 203 | Default math backend (btVector3/btMatrix3x3/btMatrixX shims) |
| `IDConfig.hpp` | 108 | Backend selection; `BT_ID_WITH_JACOBIANS` + `BT_ID_HAVE_MAT3X` both on by default (`:11`) |

Joint types (`MultiBodyTree.hpp:10-22`): `FIXED` (0 dof), `REVOLUTE` (1),
`PRISMATIC` (1), `FLOATING` (6), `SPHERICAL` (3).

q/u conventions (`MultiBodyTree.hpp:27-54`):

- `REVOLUTE`/`PRISMATIC`: q = angle/displacement, u = its time derivative.
- `FLOATING`: q = Euler X-Y-Z angles then displacement, u = **angular
  velocity** (not Euler rates) then linear velocity. Joint-force layout:
  `[moment xyz, force xyz]`.
- `SPHERICAL`: q = Euler X-Y-Z angles, u = angular velocity.

Algorithms:

- `calculateKinematics` at three levels (POSITION_ONLY / POSITION_VELOCITY /
  POSITION_VELOCITY_ACCELERATION): per-joint-type relative updates, then an
  absolute recursion over bodies 1..n (`MultiBodyTreeImpl.cpp:386-585`).
  Gravity is folded into `m_body_acc` at the root and propagates
  (`:541-542`).
- `calculateInverseDynamics` = kinematics (full level) + per-body
  Newton-Euler EOM LHS (`:298-310`) + backward parent-joint force recursion
  (`:321-342`) + per-joint-type projection into the generalized-force vector
  (`:344-382`). This is RNEA written out as explicit Newton-Euler terms in
  body frames (not Featherstone's articulated-body formulation).
- `calculateMassMatrix` — composite rigid body algorithm ("method 3",
  Walker & Orin 1982, `:777-780`) with Huygens-Steiner subtree shifts
  (`:883-895`) and ancestor-chain column fill (`:949-984`); upper triangle
  only, optional mirror (`:989-998`).
- `calculateJacobians` — per-body 3×N translational/rotational jacobians +
  velocity-dependent `dot(J)·u` terms (`:589-678`), gated on
  `BT_ID_WITH_JACOBIANS` (on by default).

Upstream consumers: `btInverseDynamics::MultiBodyTree` is self-contained;
`MultiBodyImpl` keeps one `RigidBody` struct per body (AOS,
`MultiBodyTreeImpl.hpp:16-127`) including CRBA scratch (`:107-113`) and two
3×N jacobian arrays (`:115-126`).

## 3. Crate placement

New crate `crates/bullet-inverse-dynamics`, the fifth workspace member
(`members` in the root `Cargo.toml` is an explicit list — add one line).
Dependency: `bullet-math` only. Features follow the house pattern
(`f32` default / `f64`; `std` default / `libm`), forwarded one level.

Mirror of upstream's standalone directory: zero interface with
`bullet-dynamics`' `MultiBody`. The `btMultiBody ↔ MultiBodyTree` bridge
lives upstream in license-excluded Extras and stays out of scope; a
`bullet-dynamics` consumer would write the q/u sync themselves (same as
upstream).

Publishing: a pure leaf above `bullet-math`; insert anywhere after
`bullet-math` in the leaf-first publish order. Name availability on
crates.io verified before first publish (naming precedent). Crate gets its
own `README.md` + metadata (publish-hygiene precedent).

## 4. Math layer mapping

| Upstream type | Port |
|---|---|
| `vec3` (btVector3 shim) | `bullet_math::Vector3` (glam `Vec3A`/`DVec3`) |
| `mat33` (btMatrix3x3 shim) | `bullet_math::Matrix3` (glam `Mat3A`/`DMat3`) |
| `vecx` (btVectorX) | `&[Real]` in, `&mut [Real]` out |
| `matxx` / `mat3x` (btMatrixX, **row-major** — `btMatrixX.h` `m_storage[col + row*cols]`) | flat row-major `&mut [Real]` / owned `Vec<Real>` scratch |

Transcendentals route through `bullet_math::math` (no_std). The ID library
uses matrices only — no quaternions anywhere — which removes the
quat-half-space test class entirely.

**Parity risk to pin at M1**: glam is column-major, `btMatrix3x3` row-major.
Every transcribed product (`mat * mat`, `mat * vec`, transpose-mul) must
match the upstream element-sum order; where glam's association could differ,
write explicit component arithmetic in a private `mat3` helper module.
Precedent: the glam column-major outer-product trap (two incidents in the
dynamics port). M1 gate: canonical-tree probe bit-exact in f64 **and** f32;
if any op drifts, replace that op with the explicit form.

**M1 outcome (2026-10-22)**: glam ops are bit-identical to the BID builtin
math (`IDMatVec.hpp` uses the same `((t0+t1)+t2)` element expressions glam
compiles to) — no explicit-ops fallback needed; the floating-joint scene
(no trig) matches byte-for-byte. The one drift source was the *reference*:
bullet3's CMakeLists defaults `USE_MSVC_FAST_FLOATINGPOINT=ON`, and under
`/fp:fast` MSVC swaps in approximate `sin`/`cos` that diverge from ucrt by
1 ulp at some arguments. Resolution: the BID probe links an IEEE-precise
BID build (`target/bullet3-build-id[-double]`, `-DUSE_MSVC_FAST_FLOATINGPOINT=OFF`):
`btCos/btSin` are plain `cos/sin` under `BT_USE_DOUBLE_PRECISION`
(`btScalar.h:442-443`), and precise MSVC trig equals ucrt equals Rust std.
The main `bullet3-build[-double]` parity/perf rigs stay on `/fp:fast` with
their recorded binaries untouched.

Per-body jacobian storage: two flat `Vec<Real>` of length `3 * num_dofs` per
body, indexed `[dof_column][component]` (3 floats per column — matches the
column-write access pattern of `addRelativeJacobianComponent` and the
`mat3x * vecx` products). Probe compares values, not layout.

## 5. Data layout

Two plain-data groups, all sized at `finalize` (zero allocation afterwards):

- **Static config** (immutable after finalize): `joint_type`, `parent`,
  `q_index` per body; `parent_pos_parent_body_ref: Vec<Vector3>`,
  `body_T_parent_ref: Vec<Matrix3>`, `jac_jr`/`jac_jt: Vec<Vector3>`;
  mass properties (`mass`, `mass_com`, `inertia`).
- **Per-call state** (preallocated scratch): relative kinematics
  (`body_T_parent`, `parent_pos_parent_body`, relative vel/acc), absolute
  kinematics (`body_T_world`, `body_pos/vel/acc`, `ang_vel/ang_acc`), EOM
  LHS pair, force/moment at joint, user force/moment, CRBA subtree triple,
  dot-jacobian terms, and the per-body jacobian arrays (zeroed at finalize,
  exactly like `calculateStaticData`, `MultiBodyTreeImpl.cpp:262-274`).

SoA instead of upstream's AOS `RigidBody`: each pass touches a subset of the
fields; SoA keeps the hot arrays dense (expected ≤ upstream footprint; the
ledger pins actual numbers). Layout is internal — the public surface is
index-based like upstream.

## 6. Public API sketch

```rust
pub enum JointType { Fixed, Revolute, Prismatic, Floating, Spherical }

pub struct TreeBuilder { /* InitCache equivalent */ }
impl TreeBuilder {
    pub fn accept_invalid_mass_parameters(&mut self, flag: bool) -> &mut Self;
    pub fn add_body(&mut self, body: usize, parent: isize, joint: JointType,
        parent_pos_parent_body_ref: Vector3, body_t_parent_ref: Matrix3,
        axis_of_motion: Vector3, mass: Real, body_com: Vector3,
        body_inertia: Matrix3) -> Result<(), AddBodyError>;
    pub fn finalize(self) -> Result<MultiBodyTree, BuildError>;
}

pub struct MultiBodyTree { /* config + state */ }
impl MultiBodyTree {
    pub fn num_bodies(&self) -> usize;
    pub fn num_dofs(&self) -> usize;
    pub fn set_gravity_in_world_frame(&mut self, gravity: Vector3);

    pub fn calculate_inverse_dynamics(&mut self, q: &[Real], u: &[Real],
        dot_u: &[Real], joint_forces: &mut [Real]) -> Result<(), TreeError>;
    pub fn calculate_mass_matrix(&mut self, q: &[Real],
        mass_matrix: &mut [Real]) -> Result<(), TreeError>;
    pub fn calculate_mass_matrix_ex(&mut self, q: &[Real],
        update_kinematics: bool, initialize_matrix: bool,
        set_lower_triangular: bool, mass_matrix: &mut [Real])
        -> Result<(), TreeError>;
    pub fn calculate_kinematics(&mut self, q: &[Real], u: &[Real],
        dot_u: &[Real]) -> Result<(), TreeError>;
    pub fn calculate_position_kinematics(&mut self, q: &[Real]) -> Result<(), TreeError>;
    pub fn calculate_position_and_velocity_kinematics(&mut self, q: &[Real],
        u: &[Real]) -> Result<(), TreeError>;
    pub fn calculate_jacobians(&mut self, q: &[Real], u: &[Real]) -> Result<(), TreeError>;
    pub fn calculate_position_jacobians(&mut self, q: &[Real]) -> Result<(), TreeError>;

    // Getters mirror the upstream get* surface; Option on bad index.
    pub fn body_origin(&self, body: usize) -> Option<Vector3>;
    pub fn body_com(&self, body: usize) -> Option<Vector3>;
    pub fn body_transform(&self, body: usize) -> Option<Matrix3>;
    pub fn body_angular_velocity(&self, body: usize) -> Option<Vector3>;
    pub fn body_linear_velocity(&self, body: usize) -> Option<Vector3>;
    pub fn body_linear_velocity_com(&self, body: usize) -> Option<Vector3>;
    pub fn body_angular_acceleration(&self, body: usize) -> Option<Vector3>;
    pub fn body_linear_acceleration(&self, body: usize) -> Option<Vector3>;
    pub fn parent_index(&self, body: usize) -> Option<isize>;
    pub fn joint_type(&self, body: usize) -> Option<JointType>;
    pub fn dof_offset(&self, body: usize) -> Option<usize>;
    pub fn body_jacobian_trans(&self, body: usize, out: &mut [Real]) -> Result<(), TreeError>;
    pub fn body_jacobian_rot(&self, body: usize, out: &mut [Real]) -> Result<(), TreeError>;
    pub fn body_dot_jacobian_trans_u(&self, body: usize) -> Option<Vector3>;
    pub fn body_dot_jacobian_rot_u(&self, body: usize) -> Option<Vector3>;

    // Runtime mass-property mutation + external wrenches.
    pub fn set_body_mass(&mut self, body: usize, mass: Real) -> Result<(), TreeError>;
    pub fn set_body_first_mass_moment(&mut self, body: usize, m: Vector3) -> Result<(), TreeError>;
    pub fn set_body_second_mass_moment(&mut self, body: usize, i: Matrix3) -> Result<(), TreeError>;
    pub fn add_user_force(&mut self, body: usize, force: Vector3) -> Result<(), TreeError>;
    pub fn add_user_moment(&mut self, body: usize, moment: Vector3) -> Result<(), TreeError>;
    pub fn clear_all_user_forces_and_moments(&mut self);
}
```

Naming follows upstream 1:1 in snake case so the probe twin reads as a
transcript. Mass-matrix output is row-major `num_dofs²`, mirroring
`btMatrixX` storage; `body_jacobian_trans/rot` fill row-major `3 × num_dofs`.

Dropped (deviations): `user_int`/`user_ptr` (Rust consumers keep their own
maps; upstream uses the pointer in the PyBullet bridge, which we do not
port), `printTree`/`printTreeData` (upstream debug stdout; `Debug` impl
instead), `getJointTypeStr` (derive `Debug`/`Display` on `JointType`).

Error mapping: upstream returns `int` (−1) and `abort()`s in one unreachable
path (`MultiBodyTreeImpl.cpp:768`). Port: `Result` with small typed error
enums per phase (`AddBodyError`, `BuildError`, `TreeError`); no logging in
the library (consumers get typed errors; upstream's `bt_id_error_message`
prints to stderr — deviation note).

## 7. Verbatim decisions to preserve

Excerpt of the "Reference decisions covered" list (the full list ships with
the implementing commits; every row cites upstream file:line):

- **Q1** Inertia validation repeats the `I00 + I11 < I22` branch twice and
  never checks `I00 + I22 < I11` (`IDMath.cpp:385-396`) — preserve verbatim,
  record as upstream quirk.
- **Q2** Mass-property gates: fuzzy PSD on leading principal minors with
  `kIsZero = 5ε` slack (`IDMath.cpp:8, 321-339`); triangle inequality skipped
  for bodies joined by a fixed joint (`:347-371`); non-negative diagonal;
  symmetry within `kIsZero` (`:410-441`). `kAxisLengthEpsilon = 10·kIsZero =
  50ε` (`:11`) gates rotation-matrix validity: unit columns, orthogonal
  columns, `determinant > 0` strict (`:445-494`). Both constants are
  `Real::EPSILON`-relative per precision.
- **Q3** Axis handling: non-unit revolute/prismatic axes are normalized with
  a warning; error only when length < `sqrt(REAL_MIN)`
  (`MultiBodyTree.cpp:266-277`).
- **Q4** `calculatePositionKinematics` calls `calculateKinematics(q, q, q,
  POSITION_VELOCITY)` — velocities are evaluated at u = q
  (`MultiBodyTree.cpp:167`); `calculatePositionAndVelocityKinematics` passes
  `(q, u, u)` where the third argument is unused at that level
  (`:183`). Preserve both call shapes.
- **Q5** The public `calculateKinematics` zeroes gravity for the call and
  restores it afterwards, so accelerations come out gravity-free
  (`MultiBodyTree.cpp:140-156`); `calculateInverseDynamics` keeps gravity in
  the acceleration terms.
- **Q6** SPHERICAL velocity block rotates the previous
  `parent_vel_rel`/`parent_acc_rel` by the new `body_T_parentᵀ`
  (`MultiBodyTreeImpl.cpp:500, 507`) — a no-op given the zero init from
  `calculateStaticData` (`:253-258`). Preserve as written.
- **Q7** FLOATING/SPHERICAL finalize stores zero `Jac_JR`/`Jac_JT` (upstream
  "not really correct" TODO, `MultiBodyTree.cpp:442-461`); per-dof columns
  are supplied dynamically by the jacobian tables
  (`setThreeDoFJacobians`/`setSixDoFJacobians`, `MultiBodyTreeImpl.cpp:680-749`)
  in the mass-matrix and jacobian paths only.
- **Q8** FLOATING kinematics: rotation `T = Rz(q2)·Ry(q1)·Rx(q0)`
  (`:443-445`); displacement taken in the child-at-q=0 frame and rotated by
  `body_T_parent` (`:446-449`); translational u is rotated by
  `body_T_parentᵀ` (`:461`).
- **Q9** EOM LHS subtracts user force/moment (`:302-309`); backward
  recursion transforms child joint wrenches by `child.body_T_parentᵀ` and
  takes moments about `parent_pos_parent_body` (`:321-342`); projection
  iterates per-joint-type index lists, floating output ordered
  `[moment, force]` (`:344-382`).
- **Q10** CRBA: subtree composition backward over bodies; parallel-axis
  shift written as the `tilde(r_com)² − tilde(r_body_com)²` difference
  (`:883-895`); column fill loops `col` descending then rows within the
  joint, then the ancestor chain rotates the EOM pair by
  `parent_T_child = child.body_T_parentᵀ` and shifts by r×f
  (`:899-987`); upper triangle only, mirror behind flag (`:989-998`).
- **Q11** Jacobian recursion exact op order: child `Jac_R = T·parent Jac_R`,
  child `Jac_T = parent Jac_T − tilde(parent_pos)·parent Jac_R`, then the
  relative component, then left-multiply by `T`
  (`:652-675`); `dot(J)·u` terms per `:666-674`.
- **Q12** World getters: `world = body_T_worldᵀ · body` (`:1047-1131`);
  `body_origin` uses `m_body_pos`, CoM divides by mass with a zero-mass
  fallback (`:1059-1068`).
- **Q13** Index sets: `parent == −1` only valid for body 0; the parent bound
  check is `parent < num_bodies − 1` (`:174-199`); body 0 is the single root
  and typically carries `FLOATING` for free-base trees. `finalize`'s
  `num_dofs < 0` guard is dead upstream (commented-out return,
  `MultiBodyTree.cpp:352-356`) — we return `BuildError` instead (recorded
  deviation, strictly a validation gain).
- **Q14** `bodyTParentFromAxisAngle` uses `s = −sin(angle)` (transposed
  Rodrigues, `IDMath.cpp:258-279`); `transformX/Y/Z` are transposed-form
  elementary rotations (`:150-217`); `tildeOperator` is the standard
  cross-product matrix (`:219-232`). Do not "fix" any of these.
- **Q15** Gravity default `(0, 0, −9.8)` (`MultiBodyTreeImpl.cpp:21-23`);
  `getBodyLinearAcceleration` includes it unless gravity is zeroed (doc note
  on the getter).

## 8. Verification plan

- **Probe pair**: `tools/inverse_dynamics_probe.cpp` +
  `tools/build_inverse_dynamics_probe.py` (f32/f64, `build_multibody_probe.py`
  pattern); Rust twin `crates/bullet-physics/examples/id_probe.rs`
  (`required-features = ["inverse-dynamics"]`). Output diffed with
  `tools/mbdiff.py --strip-trailing-cr`.
- **Trees**:
  1. *Canonical set* — hand-pinned: 2R planar arm; prismatic+revolute+fixed
     mix; spherical chain; floating-base branched tree; single fixed-joint
     body. Parameters from a generated table (single source like
     `tools/scenes.json`) so both sides consume identical bytes.
  2. *Seeded random trees* — a generator written natively in the probe
     (license-clean; the upstream `IDRandomUtil` is Extras and only serves
     as the idea): identical draw order both sides; C++ side
     `srand(seed)/rand()`, Rust side the existing portable `CrtRand`
     replica. **`CrtRand` moves from `bullet-dynamics/softbody::rng`
     (private) to a public `bullet-math` module** — upstream's own random
     home is `LinearMath/btRandom.h`, so `bullet-math` is the natural
     LinearMath-mapped location; `softbody` consumes it from there and the
     softbody probes must stay byte-identical (M0 gate).
- **Quantities compared** per tree × state: generalized forces (ID), full
  mass matrix, world-frame kinematics getters for every body, jacobian
  columns + `dot(J)·u` terms, and the add-time validation gates (accept /
  reject parity on near-limit inertia tensors, non-orthonormal transforms,
  degenerate axes).
- **Targets**: f64 bit-exact; f32 bit-exact (same op order both sides). Any
  drift: classify in the parity ledger or fix op order — never tolerance
  away silently.
- **Ledger**: new "Inverse dynamics" section in `docs/bullet-compare.md`
  (single-ledger house pattern; separate file only if it outgrows it).

## 9. Perf & memory plan

- **Zero-alloc contract**: no heap allocation in any `calculate_*`/getter
  after `finalize`. Enforced by a unit test running under a counting global
  allocator (perf-runner pattern).
- **Bench**: probe `--bench` mode — ns/call, interleaved repeats (perf
  discipline: multi-round interleaving, two-window recheck, drift anchor),
  scenes `chain_16 / chain_64 / chain_256 / random_64` ×
  {ID, mass matrix, kinematics, jacobians}. C++ side identical harness.
- **Ledger**: `docs/perf/inverse-dynamics.md` (speed + allocation counts +
  live bytes per tree; `tools/perf.py` stays PhysicsWorld-scoped and is not
  extended).
- **Allocation counting C++ side**: global `operator new/delete` hook
  (memory-ledger pattern).
- Expectation, not promise: f32 glam SIMD wins in the mat-vec dense
  sections; the O(bodies × dofs) jacobian storage is inherent to the
  upstream formulation and recorded as absolute numbers, not hidden.

## 10. Milestones

Each milestone: full gate (dual-precision tests, clippy ×2, `check.py`,
no_std ×2) + one atomic Conventional Commit.

- **M0** ✅ (2026-10-21) — Crate skeleton (manifest, workspace member,
  features, MSRV) + `CrtRand` lift to `bullet-math` (softbody probes
  byte-identical) + math helper module with hand-computed unit tests
  (transforms, tilde, axis-angle).
- **M1** ✅ (2026-10-22) — Builder + validation gates + finalize + kinematics
  (3 levels) + inverse dynamics. Canonical-tree probe bit-exact in f64 and
  f32 (4 scenes × 3 states × ID + kinematics groups, `cmp`-level after
  CRLF strip): glam-vs-btMatrix3x3 product-order risk retired — glam ops
  pinned bit-identical; the sole drift source was the `/fp:fast` reference
  trig (see §4 outcome). Probe rig:
  `tools/inverse_dynamics_probe.cpp` + `tools/build_inverse_dynamics_probe.py`
  (IEEE-precise BID in `target/bullet3-build-id[-double]`) + twin
  `crates/bullet-physics/examples/id_probe.rs` (since moved to the facade
  examples under the `inverse-dynamics` feature).
- **M2** ✅ (2026-10-22) — CRBA mass matrix (`calculate_mass_matrix[_ex]` +
  runtime mass setters), jacobians (position/velocity levels, world getters,
  dot-J terms, `KinUpdate::PositionOnly`). Facade integration ruling: the
  facade re-exports the crate as `inverse_dynamics` behind an optional
  `inverse-dynamics` feature, **off by default** (the leaf stays out of the
  default dependency tree); the probe twin lives in
  `crates/bullet-physics/examples/id_probe.rs` under `required-features`,
  like every other probe twin. Probe extended with the M/J groups and three
  seeded random trees (CrtRand ↔ MSVC `rand`): f64 byte-identical 84/84
  lines; f32 residual confined to one branched-tree scene, exactly 1 ulp per
  differing entry — classified as rounding noise in `docs/bullet-compare.md`
  ("Inverse-dynamics probes"; parity is the floor, not the goal). Rig-fixed
  port deviations: btVector3 reciprocal division (3 sites), CRBA child-first
  centroid associativity, `btPow(x, 2)` axis normalization.
- **M3** — Bench + allocation audit + `docs/perf/inverse-dynamics.md`;
  deviation rows in `docs/bullet-deviations.md`; api-priorities row → Done.
  Candidate from the M2 access-pattern sweep (not applied blind): flatten the
  per-body jacobian storage `Vec<Vec<Real>>` into one `Vec<Real>` with
  `as_chunks` windows — bench-arbitrated.
- **M4** — Crate README + AGENTS.md project-map line + CHANGELOG Unreleased
  + crates.io name check; publish ruling with the next release train.

## 11. Decisions needing maintainer ruling (block M0)

1. **Fifth crate** vs module inside `bullet-dynamics` — recommend fifth
   crate (mirrors upstream separation, zero intrusion, own ecosystem slot).
2. **glam aliases** vs hand-rolled fixed 3×3 math — recommend glam with the
   M1 parity pin and an explicit-ops fallback module.
3. **Output style**: `&mut [Real]` slices (caller-owned, zero-alloc,
   mirrors upstream out-parameters) — recommend yes.
4. **`user_int`/`user_ptr`**: drop with a deviation row — recommend drop.
5. **Ledger placement**: section in `bullet-compare.md` — recommend yes.
6. **Publish**: join the shared workspace version, ship with the next
   release train — recommend yes; exact leaf order decided at M4.
