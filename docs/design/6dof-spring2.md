# Design: `btGeneric6DofSpring2Constraint` → `Generic6DofSpring2Constraint`

> Status: **design baseline** — the implementation must stay inside this
> scope (no silent expansion). Upstream reference:
> `refs/bullet3/src/BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.{h,cpp}`
> (2014 rewrite by Gabor Puhr & Tamas Umenhoffer). Motivating consumer:
> MMD/PMX-style model physics — PMX joints are exactly this constraint
> (per-axis translation/rotation limits + position springs pulling toward
> the bind pose). The elastic-limit behavior is *not* reproducible with the
> classic 6DOF rows: the classic limit row is a hard one-sided stop, while
> spring2's limited axis is a **soft, force-limited two-row window** that
> produces zero force inside the range, an ERP pull-back at the boundary,
> accurate bounce, and optional springs/servo motors per axis.

## 1. Scope

- New additive constraint kind `ConstraintKind::Generic6DofSpring2` in
  `crates/bullet-dynamics/src/constraint/generic6dof_spring2.rs`; the
  classic [`Generic6DofConstraint`] (`generic6dof.rs`) is untouched.
- Ported: limits (free / locked / limited per axis), bounce, motors,
  servo motors, springs (with the stiffness/damping auto-clamps), the six
  Euler rotate orders, per-axis ERP/CFM overrides (`setParam`), the
  `rotAllowed`/`factA`/`factB` static-body jacobian scaling, and the
  reversed-limit convenience setters.
- Not ported (upstream-dead or out of scope — see §9): the
  `BT_6DOF_FLAGS_USE_INFINITE_ERROR` flag (no setter exists on this
  class in this version), serialization, `setAxis` (needs world access at
  construction; PMX/MMD loaders build frames directly), the no-arg
  `setEquilibriumPoint()` variants (needs world access; §8 gives the
  Rust-level substitute).
- Constraint rows still carry **no warm starting** (our joint pipeline
  rebuilds rows every step; upstream joints are also not warm-started).

## 2. Data model

```rust
/// Which Euler order decomposes the relative rotation (Bullet's
/// `RotateOrder`; default `RO_XYZ`).
pub enum RotateOrder { Xyz, Xzy, Yxz, Yzx, Zxy, Zyx }

/// Per-angular-axis limit/motor/spring parameters (Bullet's
/// `btRotationalLimitMotor2`). Defaults verbatim.
pub struct RotationalLimitMotor2 {
    // upper < lower: free; ==: locked; >: limited
    pub lower_limit: Real,          // 1.0
    pub upper_limit: Real,          // -1.0
    pub bounce: Real,               // 0.0
    pub stop_erp: Real,             // 0.2 (used only when the ERP_STOP flag is set)
    pub stop_cfm: Real,             // 0.0 (only when CFM_STOP flag set)
    pub motor_erp: Real,            // 0.9 (only when ERP_MOTO flag set)
    pub motor_cfm: Real,            // 0.0 (only when CFM_MOTO flag set)
    pub enable_motor: bool,         // false
    pub target_velocity: Real,      // 0.0
    pub max_motor_force: Real,      // 6.0
    pub servo_motor: bool,          // false
    pub servo_target: Real,         // 0.0
    pub enable_spring: bool,        // false
    pub spring_stiffness: Real,     // 0.0
    pub spring_stiffness_limited: bool, // false
    pub spring_damping: Real,       // 0.0
    pub spring_damping_limited: bool,   // false
    pub equilibrium_point: Real,    // 0.0
    // per-step state
    pub current_limit: i32,         // 0 free / 3 locked / 4 limited
    pub current_limit_error: Real,  // pos - lower
    pub current_limit_error_hi: Real, // pos - upper (limit 4 only)
    pub current_position: Real,     // the adjusted angle
}

/// Linear triplet: same parameter set as the rotational motor, with the
/// vector fields as `Vector3` and bools as `[bool; 3]` (Bullet's
/// `btTranslationalLimitMotor2`). Defaults: lower=upper=ZERO (all three
/// axes locked — same behavior as the classic 6DOF), stop_erp=0.2,
/// motor_erp=0.9, max_motor_force=ZERO (upstream differs from the angular
/// 6.0 default here), everything else zero/false.
pub struct TranslationalLimitMotor2 { ... }

pub struct Generic6DofSpring2Constraint {
    pub frame_a: JointFrame,
    pub frame_b: JointFrame,
    pub rotate_order: RotateOrder,          // Xyz
    pub linear: TranslationalLimitMotor2,
    pub angular: [RotationalLimitMotor2; 3],
    // per-step state (upstream m_calculated*), public like the classic 6DOF
    pub current_linear_diff: Vector3,       // m_calculatedLinearDiff
    pub euler_angles: Vector3,              // m_calculatedAxisAngleDiff
    pub angular_axes: [Vector3; 3],         // m_calculatedAxis
    pub fact_a: Real,                       // m_factA (miB / (miA+miB), 0.5 fallback)
    pub has_static_body: bool,              // invMass < Real::EPSILON either side
    // per-axis ERP/CFM override flags (Bullet's m_flags, 4 bits per axis,
    // linear = axes 0..2, angular = 3..5); private, driven by the
    // set_stop_erp/set_stop_cfm/set_motor_erp/set_motor_cfm methods.
    flags: u32,
}
```

`has_static_body` and `fact_a`/`fact_b` are recomputed each prepare
(upstream `calculateTransforms`: `factA = miB/(miA+miB)`, `0.5` when
`miS == 0`; `factB = 1 − factA`; static test `< Real::EPSILON`
(`SIMD_EPSILON` — precision-dependent constant, keep `Real::EPSILON`).

## 3. Euler machinery (six orders)

- Element access: Bullet `btGetMatrixElem(mat, i) = mat[i%3][i/3]`
  (row-major) maps to glam `m.col(i/3)[i%3]` — same mapping the shared
  `euler_xyz` already documents.
- `RO_XYZ` **reuses `shared::euler_xyz`** (bit-identical, already shared
  with the multibody constraints — do not duplicate).
- The five other decompositions (`matrixToEulerXZY/YXZ/YZX/ZXY/ZYX`) are
  private fns in the new module, ported verbatim (gimbal fallback values
  included). Upstream returns a `bool` "exact" flag that
  `calculateAngleInfo` **ignores** — the fallback angles always feed the
  state; our port returns the `Vector3` only.
- `calculate_angle_info`: relative basis `trA⁻¹ · trB` → angles by
  rotate order, then the per-order **constraint axes** from explicit
  cross products (all six cases ported verbatim, e.g. XYZ:
  `ax[1] = trA.col(2)×trB.col(0)`, `ax[0] = ax[1]×trA.col(2)`,
  `ax[2] = trB.col(0)×ax[1]`; each order picks its A/B columns per the
  upstream switch), normalized.
- Row emission order for the angular axes follows the upstream `cIdx`
  permutation (XYZ: 0,1,2; XZY: 0,2,1; YXZ: 1,0,2; YZX: 1,2,0;
  ZXY: 2,0,1; ZYX: 2,1,0) — solve order is load-bearing for parity.

## 4. Limit model (the semantic core)

`test_limit_value` (both motor classes, ported verbatim) has **no
"in-range" state** — unlike the classic 6DOF:

| condition | `current_limit` | errors |
|---|---|---|
| `lo > hi` | 0 (free) | cleared |
| `lo == hi` | 3 (locked) | `err = pos − lo` (signed, bilateral) |
| `lo < hi` | 4 (limited) | `err = pos − lo`, `err_hi = pos − hi` |

A limited axis *always* reports 4; the one-sidedness comes from the
**impulse windows** of its two rows, which can only push back into the
range (so a body resting inside the range gets zero force, and the
boundary behaves like a soft, force-limited wall). Linear limits test
`current_linear_diff[i]` directly (upstream `calculateLinearInfo`);
angular limits first run the angle through `adjust_angle_to_limits`
(`btAdjustAngleToLimits`, already in `shared.rs`) and **do not
re-normalize** (upstream comment: normalizing would lose the sign that
selects which side the wrap lands on).

## 5. Row assembly (`get_limit_motor_info2` port)

Per axis, rows are emitted in this order (verbatim):

1. **limit == 4 → two one-sided rows**
   - Row 1 (lower side): `err = inv_dt · stop_erp · limit_err · (rot ? −1 : +1)`;
     Bullet window rotational `[0, +inf]` / linear `[−inf, 0]`.
   - Row 2 (upper side): same with `limit_err_hi`; window mirrored
     (rotational `[−inf, 0]` / linear `[0, +inf]`).
   - Bounce per row, four verbatim branches — the condition is
     `err − vel·stop_erp` (does the bias already pull the right way
     given the incoming velocity), then a `max`/`min` against
     `−bounce·vel`. `vel` here is the **plain** projection
     `linVel·ax1 − linVel·ax1` (no lever-arm terms — asymmetry with the
     spring row's velocity is upstream-verbatim).
2. **limit == 3 → one bilateral row** (`[−inf, +inf]`),
   `err = inv_dt · stop_erp · limit_err · (rot ? −1 : +1)`.
3. **motor** (`enable_motor && !servo_motor`):
   `tag_vel = rot ? target : −target`;
   `mot_fact = get_motor_factor(pos, lo, hi, tag_vel, inv_dt · motor_erp)`
   (shared.rs); `err = mot_fact · target_velocity` (no rot sign flip —
   upstream asymmetry, keep); window `±max_motor_force / inv_dt`.
4. **servo** (`enable_motor && servo_motor`): `error = pos − servo_target`
   (rotational: ±2π wrap on both `error` and `cur_servo_target`); dynamic
   limit window around the servo target (verbatim branch cascade);
   `targetvelocity = error < 0 ? −target : target`; `tag_vel = −targetvelocity`;
   `mot_fact` via `get_motor_factor` against the dynamic window (0 when
   `error == 0`); `err = mot_fact · targetvelocity · (rot ? −1 : +1)`
   (sign flip **is** applied here, unlike case 3); window `±max/fps`.
5. **spring** (§6).

Window mirroring through our row convention (`λ_ours = −λ_bullet`) uses
`mirrored_row_limits` — but note rows 1/2 and the spring row produce
**asymmetric finite-or-infinite windows**, which is exactly
`RowLimits::Window(lo, hi)` (already exists, cone-twist motor precedent).
No solver-core row-limit changes needed.

### ERP/CFM resolution (the flag mechanism)

Upstream defaults every axis to the **global** solver values and only
uses the struct fields when the per-axis flag is set:

```
stop_erp_used  = flag(ERP_STOP, axis)  ? field : info.erp      // global
stop_cfm_used  = flag(CFM_STOP, axis)  ? field : global_cfm
motor_erp_used = flag(ERP_MOTO, axis)  ? field : info.erp
motor_cfm_used = flag(CFM_MOTO, axis)  ? field : global_cfm
```

`prepare_rows` already receives the global `erp`; it additionally needs
`global_cfm` (see §7). Numeric defaults coincide (fields 0.2/0.0 vs
global 0.2/0.0), so a fresh joint behaves identically until either the
global solver values or the per-axis overrides are tuned.

### Jacobians and the static-body scaling

`calculate_jacobi` port: rotational rows `J = (0, ax1 | 0, −ax1)`;
linear rows `J = (ax1, r_a×ax1 | −ax1, −r_b×ax1)` where
`r_a/r_b` = each frame's world origin minus its body COM (upstream
`m_calculatedTransformA/B.origin − trans.origin`) — this coincides with
our per-side lever-arm convention, so **no classic-6DOF-style deviation
here**. When `has_static_body && !rot_allowed`, the angular jacobians
scale: `r_a×ax1 *= fact_a`, `r_b×ax1 *= fact_b`.

`rot_allowed` (linear axes only): 0 when **both** orthogonal angular
axes are violated. Upstream branch structure references `current_limit`
values 1/2 that spring2 never produces (copy-paste from the classic
constraint — see §9); the live conditions are limit 3 with
`|err| > 1e-3` and limit 4 with `err < −1e-3 || err_hi > 1e-3`
(`D6_LIMIT_ERROR_THRESHOLD_FOR_ROTATION = 1e-3`). Port the branch
structure verbatim, dead arms included.

Emission order: angular rows first (cIdx order), then linear 0..2 —
matching upstream `getInfo2`. The linear `rotAllowed` test reads the
angular limit states, so the angular test must run first (it does:
linear info → angle info → angular rows → linear rows, one fused
`prepare_rows` — upstream's `getInfo1`+`getInfo2` run back-to-back in
`convertJoints`, no integration between).

## 6. Spring row math

`error = current_position − equilibrium_point` (linear:
`current_linear_diff[i]`; angular: adjusted angle).

Velocity (upstream-verbatim asymmetry): rotational
`vel = (ω_a − ω_b)·ax1`; **linear includes the lever-arm terms**:
`vel = (v_a + ω_a×r_a − v_b − ω_b×r_b)·ax1`.

Effective masses:

```
mA = 1/inv_mass_a; mB = 1/inv_mass_b          // +inf for static bodies
rotational && inv_mass != 0: mA = mA·|r_a|² + 1/|invI_a·ax1|   (same for B)
m = inv_a==0 ? mB : inv_b==0 ? mA : mA·mB/(mA+mB)
```

Clamps (gated by the `*_limited` flags, both verbatim):

```
angular_freq = sqrt(ks / m)
if spring_stiffness_limited && 0.25 < angular_freq·dt:
    ks = m / (16·dt²)      // never sample the spring faster than ω/4
if spring_damping_limited && kd·dt > m:
    kd = m / dt            // never overshoot critically
```

Force and row:

```
fs = ks·error·dt
fd = −kd·vel·(rot ? −1 : +1)·dt
f  = fs + fd
err = vel + f/m·(rot ? −1 : +1)      // see upstream comment: any velocity
                                     // ≥ the "proper" one works; the window
                                     // is what bounds the force
window (Bullet convention): linear  [min(f,fd) if <0 else 0, max(f,fd) if >0 else 0]
                            rotational [−max(f,fd) if >0 else 0, −min(f,fd) if <0 else 0]
cfm = 0 (always)
```

Mirror the windows through `λ_ours = −λ_bullet` and push as
`RowLimits::Window`.

## 7. Solver integration

- **`ConstraintRow` gains a `cfm: Real` field** (joint rows currently
  have none). Upstream joint-row consumption
  (`btSequentialImpulseConstraintSolver`): `m_cfm = info->cfm[srow]`
  (raw — joints do **not** fold cfm into the effective mass, unlike
  contacts' `cfm·jacDiagABInv` pre-scaling at setup), and the solve is
  `delta = rhs − applied·cfm − (J·v_delta)·invK`. In our row terms:

  ```
  delta = (bias − row_vel·damping_gate − cfm·applied) · mass
  ```

  with the existing `solve_joint_row` shape; gate the extra term behind
  `cfm != 0.0` (hot-path lesson from the contact-cfm round: a constant
  zero field must not change the default instruction sequence).
  `ConstraintRow::prepare` keeps its signature; add a
  `prepare_with_cfm(...)` (or builder) used only by spring2 rows.
- **`prepare_rows` signature**: append a `global_cfm: Real` parameter
  (call site: the joint-prepare site in `step.rs`, passing
  `solver_info.global_cfm`). All other kinds ignore it.
- Default-path zero impact: no existing row sets `cfm != 0`, no existing
  code path changes numerically. `ConstraintKind` already carries
  `#[allow(clippy::large_enum_variant)]`.
- Not applied (pre-existing gap, record in deviations): upstream feeds
  `infoGlobal.m_globalCfm` into **every** joint kind's rows as the
  default cfm; our other joint kinds keep ignoring it. With
  `global_cfm = 0` (default) both are bit-identical; the divergence only
  exists for tuned global cfm on non-spring2 joints.

## 8. API surface

Factory + accessors mirroring the established pattern:

```rust
Constraint::generic_6dof_spring2(body_a, body_b, frame_a, frame_b,
                                 rotate_order: RotateOrder) -> Constraint
Constraint::generic_6dof_spring2_mut()  // + kind()/kind_name() arms
```

Setters on `Generic6DofSpring2Constraint` (upstream 1:1):

- `set_linear_limits` / `set_angular_limits` / `set_limit(axis, lo, hi)`
  (angular normalized via `normalize_angle`) and the **reversed**
  variants `set_limit_reversed` / `set_angular_*_limit_reversed`
  (negate + swap — cheap, upstream-verbatim; some MMD bridges need the
  flipped convention).
- `set_rotation_order` / `set_bounce(index, v)`
- `enable_motor(index, on)` / `set_target_velocity(index, v)` /
  `set_max_motor_force(index, v)`
- `set_servo(index, on)` / `set_servo_target(index, target)` — angular
  target wrapped to [−π, π] with the upstream branch cascade; the
  upstream `floor`-based positive-mod is numerically reproduced by
  `target % 2π` + the existing `m < 0` / `m ≥ 2π` branches (result
  identical for all inputs; no new `bullet_math::math` routing needed —
  note the ulp caveat for very large inputs, servo targets are O(π)).
- `enable_spring(index, on)` / `set_stiffness(index, k, limit_needed = true)`
  / `set_damping(index, d, limit_needed = true)` (Rust: explicit bool
  arg; no default args).
- `set_stop_erp/set_stop_cfm/set_motor_erp/set_motor_cfm(axis 0..5, v)`
  — the `setParam` mapping, setting `flags` internally (this is why
  `stop_erp` etc. are pub fields but *loads* only happen through these).
- `set_equilibrium_point(index, value)` (value variant only — the
  no-arg/current variants need world access; the MMD/PMX flow builds
  frames at the bind pose where equilibrium 0 is already correct, and a
  consumer can read `get_angle`/`get_relative_pivot_position` after a
  step and pass values explicitly. Rationale recorded; revisit on demand).
- Readbacks: `get_angle(axis)`, `get_relative_pivot_position(axis)`,
  `get_axis(i)` (from `angular_axes`).

Facade: re-export `Generic6DofSpring2Constraint`,
`RotationalLimitMotor2`, `TranslationalLimitMotor2`, `RotateOrder` from
the `bullet` facade alongside the classic 6DOF names.

## 9. Upstream-dead / deviation notes (draft §1 entries for
`bullet-deviations.md`)

1. **`BT_6DOF_FLAGS_USE_INFINITE_ERROR`** (`1<<16`): read in the spring
   branch, but this version exposes no setter for `m_flags` — dead.
   Skipped; our spring rows always use the `vel + f/m` form.
2. **`rotAllowed` dead arms**: `current_limit == 1/2` are unreachable
   (spring2's testLimitValue only yields 0/3/4). Ported as written.
3. **`matrixToEuler*` bool return ignored** by upstream; fallback angles
   always consumed. Our fns return angles only.
4. **`getParam` assert semantics** (`btAssertConstrParams(flag set)`) not
   reproduced: our reads are plain field reads (assert-on-read has no
   Rust analogue in release; flags state is caller-managed).
5. **`getBreakingImpulseThreshold` clamp** in `convertJoint` finalize:
   not applied (default threshold is +inf — inert; the engine has no
   constraint-breaking feature at all).
6. **global_cfm on other joint kinds**: upstream default-feeds it into
   all joint rows; ours only into spring2 (§7). Default-zero identical.
7. **Bounce position accuracy**: upstream header note — bounce computes
   the correct velocity but not position; inherent, documented upstream.

## 10. Testing plan

Unit tests (inline `#[cfg(test)]` in the new module, colocated per
project rule):

1. Euler: each of the six orders against hand-composed rotation matrices
   (identity → zero; ±90° axes; one gimbal case per order hitting the
   fallback arm), column-storage mapping spot-checked.
2. `test_limit_value` table: free/locked/limited × error signs.
3. Locked axis → one bilateral row; free axis → no rows.
4. Limited axis, position inside range → windows clamp the impulse to 0
   (no force); outside → restoring bias `inv_dt·erp·err` and one-sided
   window direction per row.
5. Bounce: `bounce = 0` kills the reflection; `bounce = 1` mirrors the
   incoming velocity (check the four branch shapes).
6. Motor: `tag_vel` asymmetry (linear vs rotational sign), fade near
   limits via `get_motor_factor`, `±max_force/inv_dt` window.
7. Servo: convergence to `servo_target`; ±2π wrap of the angular target;
   dynamic window vs free-axis branch.
8. Spring: single-axis analytic — small-oscillation period
   `T = 2π·sqrt(m/ks)` within tolerance; equilibrium pull-back; both
   clamps (`*_limited = true` with absurd ks/kd → stable, no blow-up);
   `spring_cfm == 0`.
9. Static-body scaling: one static + one dynamic, both orthogonal angular
   limits violated → `fact_a`/`fact_b` scaled jacobians (hand-computed
   row).
10. Row order: locked/limited/motor/spring mixed → emitted order matches
    upstream (angular cIdx first, then linear 0..2).

Demo: extend `tools/check.py`'s demo set with a small spring2 chain
(three boxes, two spring2 joints with angular springs) asserting
bounded drift — keeps the joint under the demo gate like the others.

Validation gates: `cargo test` × f32/f64, clippy ×2, `check.py`,
no_std check both precisions; perf quick-pass (speed + mem) — the
`ConstraintRow` +8 B cfm field only inflates the joint-row pool
(stack_1000 joint count is small; expect within noise, verify).

## 11. "Reference decisions covered" pre-draft (for the commit message)

- Limit state machine (0/3/4) × ported verbatim (§4); no in-range state.
- Two-row elastic limit windows + four bounce branches × verbatim (§5.1).
- Locked bilateral row, error signs rot/linear × verbatim (§5.2).
- Motor `tag_vel` asymmetry + fade + window × verbatim (§5.3).
- Servo wrap, dynamic windows, `mot_fact = 0` on zero error, sign flip ×
  verbatim (§5.4).
- Spring effective mass, both clamps, window shapes, cfm = 0 × verbatim
  (§6); `USE_INFINITE_ERROR` arm skipped (upstream-dead, §9.1).
- ERP/CFM flag resolution incl. global fallback × verbatim (§5).
- Jacobian + `rotAllowed`/`factA/factB` × verbatim incl. dead arms (§5).
- Row emission order (cIdx, limit4→3→motor→servo→spring, angular→linear)
  × verbatim (§5).
- Defaults table × verbatim (§2), incl. linear `max_motor_force = 0`.
- `setServoTarget` wrap × reproduced (§8, ulp note recorded).
- Dead in upstream: `USE_INFINITE_ERROR` setter, `getParam` asserts,
  breaking-threshold effect (§9).
