//! Cone-twist joint (Bullet's `btConeTwistConstraint`): elliptical swing limit, one-sided twist limit and motor.

use super::*;

/// A cone-and-twist (ragdoll) joint (Bullet's `btConeTwistConstraint`): the
/// pivots stay coincident (three linear rows, like the point-to-point joint)
/// and the relative orientation is limited to an elliptical cone around the
/// frame X axes plus a one-sided twist limit around the cone axis.
///
/// `swing_span1` bounds the swing in the frame's Y direction, `swing_span2`
/// in the Z direction; `twist_span` bounds the rotation around the X axis in
/// one direction only (Bullet only tests the positive side). A span of 0
/// locks that degree of freedom, a negative span frees it, and the default
/// `setLimit(0, 0, 0)` style call with `BT_LARGE_FLOAT` spans frees
/// everything. Softness 1.0 (default) makes the limits hard; the correction
/// factor `bias_factor` (0.3) scales the pull-back bias and
/// `relaxation_factor` (1.0) scales the row jacobian.
#[derive(Debug, Clone, PartialEq)]
pub struct ConeTwistConstraint {
    /// Constraint frame on body A (pivot + basis in A's COM frame).
    pub frame_a: JointFrame,
    /// Constraint frame on body B (pivot + basis in B's COM frame).
    pub frame_b: JointFrame,
    /// Swing limit around the frame Y axis, radians.
    pub swing_span1: Real,
    /// Swing limit around the frame Z axis, radians.
    pub swing_span2: Real,
    /// Twist limit around the frame X axis, radians (one-sided).
    pub twist_span: Real,
    /// Limit softness (Bullet's `m_limitSoftness`, default 1.0).
    pub limit_softness: Real,
    /// Position-correction factor on the limit bias (Bullet's
    /// `m_biasFactor`, default 0.3).
    pub bias_factor: Real,
    /// Row-jacobian scale (Bullet's `m_relaxationFactor`, default 1.0).
    pub relaxation_factor: Real,
    /// Optional motor driving the joint to a target orientation (Bullet's
    /// `setMotorTarget` + `m_maxMotorImpulse`).
    pub motor: Option<ConeTwistMotor>,
    /// Per-step state (Bullet's `m_swingAxis` etc.), updated by the last
    /// solve step.
    swing_correction: Real,
    swing_axis: Vector3,
    solve_swing_limit: bool,
    twist_correction: Real,
    twist_axis: Vector3,
    twist_angle: Real,
    solve_twist_limit: bool,
    twist_axis_a: Vector3,
    k_swing: Real,
    k_twist: Real,
}

/// The threshold below which the swing spans are treated as locked/hinge
/// configurations (Bullet's `CONETWIST_DEF_FIX_THRESH`).
const CONE_TWIST_FIX_THRESH: Real = 0.05;

/// Bullet's `SIMD_EPSILON` (precision-dependent): near-zero tolerance used
/// by the cone/twist angle tests (`btFuzzyZero` and the
/// `twistAngle > SIMD_EPSILON` normalization guard). Normalizing a
/// near-zero rotation axis is numerically unstable (the row's effective
/// mass explodes as `1/|axis|²`), so the guard must be at Bullet's exact
/// epsilon or the double-precision build re-excites the twist.
const SIMD_EPSILON: Real = Real::EPSILON;

/// The cone axis in constraint space (Bullet's `vTwist`).
const CONE_TWIST_AXIS: Vector3 = Vector3::X;

/// A cone-twist motor: rotate the joint toward `target` (a quaternion in
/// constraint space, see [`ConeTwistConstraint::set_motor_target`]),
/// limited to `max_impulse` per step (Bullet's `enableMotor` /
/// `setMotorTarget`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConeTwistMotor {
    /// Target relative orientation in constraint space (Bullet's
    /// `m_qTarget`, clamped to the limits at set time).
    pub target: Quaternion,
    /// Maximum motor impulse per step (Bullet's `m_maxMotorImpulse`).
    pub max_impulse: Real,
}

impl ConeTwistConstraint {
    /// A cone-and-twist joint with the given spans (radians) and the default
    /// softness 1.0 / bias factor 0.3 / relaxation factor 1.0 (Bullet's
    /// `setLimit(swing1, swing2, twist)`).
    pub fn new(
        frame_a: JointFrame,
        frame_b: JointFrame,
        swing_span1: Real,
        swing_span2: Real,
        twist_span: Real,
    ) -> Self {
        Self {
            frame_a,
            frame_b,
            swing_span1,
            swing_span2,
            twist_span,
            limit_softness: 1.0,
            bias_factor: 0.3,
            relaxation_factor: 1.0,
            motor: None,
            swing_correction: 0.0,
            swing_axis: Vector3::ZERO,
            solve_swing_limit: false,
            twist_correction: 0.0,
            twist_axis: Vector3::ZERO,
            twist_angle: 0.0,
            solve_twist_limit: false,
            twist_axis_a: Vector3::ZERO,
            k_swing: 0.0,
            k_twist: 0.0,
        }
    }

    /// Set the spans (radians), softness, bias factor and relaxation factor
    /// (Bullet's `setLimit`).
    pub fn set_limit(
        &mut self,
        swing_span1: Real,
        swing_span2: Real,
        twist_span: Real,
        softness: Real,
        bias_factor: Real,
        relaxation_factor: Real,
    ) {
        self.swing_span1 = swing_span1;
        self.swing_span2 = swing_span2;
        self.twist_span = twist_span;
        self.limit_softness = softness;
        self.bias_factor = bias_factor;
        self.relaxation_factor = relaxation_factor;
    }

    /// Enable the motor toward a world-space target orientation `q` (Bullet's
    /// `setMotorTarget`): `q` is the desired relative orientation of body B
    /// w.r.t. body A, converted to constraint space with the joint frames.
    pub fn set_motor_target(&mut self, q: Quaternion) {
        let q_constraint = self.frame_b.basis_quat().inverse() * q * self.frame_a.basis_quat();
        self.set_motor_target_in_constraint_space(q_constraint);
    }

    /// Enable the motor toward a target given directly in constraint space
    /// (Bullet's `setMotorTargetInConstraintSpace`); the target is clamped to
    /// the cone/twist limits.
    pub fn set_motor_target_in_constraint_space(&mut self, q: Quaternion) {
        let target = q;
        // Clamp the motor target to within the limits, like Bullet.
        let v_twisted = target * CONE_TWIST_AXIS;
        let mut q_target_cone = quat_from_rotation_arc(CONE_TWIST_AXIS, v_twisted);
        q_target_cone = q_target_cone.normalize();
        let mut q_target_twist = q_target_cone.inverse() * target;
        q_target_twist = q_target_twist.normalize();

        if self.swing_span1 >= CONE_TWIST_FIX_THRESH && self.swing_span2 >= CONE_TWIST_FIX_THRESH {
            let (mut swing_angle, swing_axis, swing_limit) =
                self.compute_cone_limit_info(&q_target_cone);
            let softness = 1.0;
            if swing_angle.abs() > SIMD_EPSILON {
                if swing_angle > swing_limit * softness {
                    swing_angle = swing_limit * softness;
                } else if swing_angle < -swing_limit * softness {
                    swing_angle = -swing_limit * softness;
                }
                q_target_cone = quat_from_axis_angle(swing_axis, swing_angle);
            }
        }

        if self.twist_span >= CONE_TWIST_FIX_THRESH {
            let (mut twist_angle, twist_axis) = self.compute_twist_limit_info(&q_target_twist);
            if twist_angle.abs() > SIMD_EPSILON {
                if twist_angle > self.twist_span * 1.0 {
                    twist_angle = self.twist_span * 1.0;
                } else if twist_angle < -self.twist_span * 1.0 {
                    twist_angle = -self.twist_span * 1.0;
                }
                q_target_twist = quat_from_axis_angle(twist_axis, twist_angle);
            }
        }

        let max_impulse = self.motor.map(|m| m.max_impulse).unwrap_or(-1.0);
        self.motor = Some(ConeTwistMotor {
            target: q_target_cone * q_target_twist,
            max_impulse,
        });
    }

    /// Whether a motor is enabled.
    pub fn motor_enabled(&self) -> bool {
        self.motor.is_some()
    }

    /// The last swing correction (radians), updated by the last solve step.
    pub fn swing_correction(&self) -> Real {
        self.swing_correction
    }

    /// The last twist correction (radians), updated by the last solve step.
    pub fn twist_correction(&self) -> Real {
        self.twist_correction
    }

    /// The current twist angle in radians (Bullet's `getTwistAngle`),
    /// updated by the last solve step.
    pub fn twist_angle(&self) -> Real {
        self.twist_angle
    }

    /// Whether the swing limit was active in the last solve step.
    pub fn solve_swing_limit(&self) -> bool {
        self.solve_swing_limit
    }

    /// Whether the twist limit was active in the last solve step.
    pub fn solve_twist_limit(&self) -> bool {
        self.solve_twist_limit
    }

    /// Flatten this cone-twist joint into its solver rows (Bullet's
    /// `getInfo2NonVirtual`): three linear pivot rows plus the swing and
    /// twist limit rows.
    pub(crate) fn prepare_rows(
        &mut self,
        body_a: usize,
        body_b: usize,
        bodies: &[RigidBody],
        inv_dt: Real,
        erp: Real,
        out: &mut Vec<ConstraintRow>,
    ) {
        let a = &bodies[body_a];
        let b = &bodies[body_b];
        let tr_a = Matrix3::from_quat(a.orientation) * self.frame_a.basis;
        let tr_b = Matrix3::from_quat(b.orientation) * self.frame_b.basis;
        let pivot_a = a.position + a.orientation * self.frame_a.origin;
        let pivot_b = b.position + b.orientation * self.frame_b.origin;
        let r_a = pivot_a - a.position;
        let r_b = pivot_b - b.position;
        let k = inv_dt * erp;

        // Three linear pivot rows along the world axes (Bullet writes
        // `J1 = e_j` with the skew of `−a1`, exactly the point-to-point
        // recipe; `m_constraintError = k·(pivotB − pivotA)`, mirrored to our
        // `k·(pivotA − pivotB)` bias).
        for axis in 0..3 {
            let mut normal = Vector3::ZERO;
            normal[axis] = 1.0;
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                normal,
                r_a.cross(normal),
                normal,
                r_b.cross(normal),
                a,
                b,
                k * (pivot_a - pivot_b)[axis],
                RowLimits::Unbounded,
            ));
        }

        self.calc_angle_info(&tr_a, &tr_b, &a.inv_inertia_world(), &b.inv_inertia_world());

        // Swing limit rows.
        if self.solve_swing_limit {
            if self.swing_span1 < CONE_TWIST_FIX_THRESH && self.swing_span2 < CONE_TWIST_FIX_THRESH {
                // Both spans locked: two rows keeping the X axes parallel
                // (Bullet's fixed branch, rows along trA's Y and Z).
                let p = tr_a.col(1);
                let q = tr_a.col(2);
                let fact = inv_dt * self.relaxation_factor;
                for dir in [p, q] {
                    out.push(ConstraintRow::prepare(
                        body_a,
                        body_b,
                        Vector3::ZERO,
                        dir,
                        Vector3::ZERO,
                        dir,
                        a,
                        b,
                        -fact * self.swing_axis.dot(dir),
                        RowLimits::Unbounded,
                    ));
                }
            } else {
                // Elliptical cone: one row pulling back along the swing axis.
                let ax1 = self.swing_axis * (self.relaxation_factor * self.relaxation_factor);
                let limits = match self.motor {
                    Some(m) if m.max_impulse >= 0.0 => {
                        // Bullet's `[0, maxImpulse]`, mirrored through
                        // λ_ours = −λ_bullet.
                        RowLimits::Window(-m.max_impulse, 0.0)
                    }
                    _ => RowLimits::UpperBounded(0.0),
                };
                out.push(ConstraintRow::prepare(
                    body_a,
                    body_b,
                    Vector3::ZERO,
                    ax1,
                    Vector3::ZERO,
                    ax1,
                    a,
                    b,
                    -inv_dt * self.bias_factor * self.swing_correction,
                    limits,
                ));
            }
        }

        // Twist limit row.
        if self.solve_twist_limit {
            let ax1 = self.twist_axis * (self.relaxation_factor * self.relaxation_factor);
            let limits = if self.twist_span > 0.0 {
                if self.twist_correction > 0.0 {
                    // Bullet's `[0, +inf]`, mirrored.
                    RowLimits::UpperBounded(0.0)
                } else {
                    RowLimits::LowerBounded(0.0)
                }
            } else {
                // Locked twist (`twist_span == 0`): a plain equality row.
                RowLimits::Unbounded
            };
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                Vector3::ZERO,
                ax1,
                Vector3::ZERO,
                ax1,
                a,
                b,
                -inv_dt * self.bias_factor * self.twist_correction,
                limits,
            ));
        }
    }

    /// Port of Bullet's `calcAngleInfo2`: split the relative rotation into
    /// cone and twist, test the limits, and update the per-step swing/twist
    /// state consumed by [`prepare_rows`](Self::prepare_rows).
    fn calc_angle_info(
        &mut self,
        tr_a: &Matrix3,
        tr_b: &Matrix3,
        inv_inertia_a: &Matrix3,
        inv_inertia_b: &Matrix3,
    ) {
        self.swing_correction = 0.0;
        self.solve_twist_limit = false;
        self.solve_swing_limit = false;

        if let Some(motor) = self.motor {
            // Motor branch: the swing axis/correction come from the delta
            // between the current relative pose and the target pose (Bullet
            // assumes `setMotorTarget` was called and the target is within
            // the limits).
            let r_delta = *tr_b * Matrix3::from_quat(motor.target) * tr_a.transpose();
            let mut q_delta = quat_from_mat3(&r_delta);
            q_delta = q_delta.normalize();
            let swing_axis = quat_xyz(q_delta);
            let len2 = swing_axis.length_squared();
            if len2 > SIMD_EPSILON {
                self.swing_axis = swing_axis / math::sqrt(len2);
                self.swing_correction = quaternion_angle(&q_delta);
                if self.swing_correction.abs() > SIMD_EPSILON {
                    self.solve_swing_limit = true;
                }
            }
            return;
        }

        // Non-motor branch: decompose the relative rotation into cone (no
        // twist) and pure twist, from B's perspective.
        let q_a = quat_from_mat3(tr_a);
        let q_b = quat_from_mat3(tr_b);
        let mut q_ab = q_b.inverse() * q_a;
        q_ab = q_ab.normalize();
        let mut v_cone_no_twist = q_ab * CONE_TWIST_AXIS;
        v_cone_no_twist = v_cone_no_twist.normalize();
        let mut q_ab_cone = quat_from_rotation_arc(CONE_TWIST_AXIS, v_cone_no_twist);
        q_ab_cone = q_ab_cone.normalize();
        let mut q_ab_twist = q_ab_cone.inverse() * q_ab;
        q_ab_twist = q_ab_twist.normalize();

        if self.swing_span1 >= CONE_TWIST_FIX_THRESH && self.swing_span2 >= CONE_TWIST_FIX_THRESH {
            let (swing_angle, mut swing_axis, swing_limit) =
                self.compute_cone_limit_info(&q_ab_cone);
            if swing_angle > swing_limit * self.limit_softness {
                self.solve_swing_limit = true;
                self.swing_correction = swing_angle - swing_limit * self.limit_softness;
                self.adjust_swing_axis_to_use_ellipse_normal(&mut swing_axis);
                self.swing_axis = q_b * (-swing_axis);
                self.k_swing = 1.0
                    / (angular_impulse_denominator(self.swing_axis, inv_inertia_a)
                        + angular_impulse_denominator(self.swing_axis, inv_inertia_b));
            }
        } else {
            // One or both spans below the fix threshold: hinge/fixed
            // configurations (Bullet's `else` branch of the cone test).
            let iv_a = tr_a.col(0);
            let jv_a = tr_a.col(1);
            let kv_a = tr_a.col(2);
            let iv_b = tr_b.col(0);
            let mut x = iv_b.dot(iv_a);
            let mut y = iv_b.dot(jv_a);
            let mut z = iv_b.dot(kv_a);
            if self.swing_span1 < CONE_TWIST_FIX_THRESH && self.swing_span2 < CONE_TWIST_FIX_THRESH {
                // Fixed: one row keeps the X axes parallel.
                if y.abs() > SIMD_EPSILON || z.abs() > SIMD_EPSILON {
                    self.solve_swing_limit = true;
                    self.swing_axis = -iv_b.cross(iv_a);
                }
            } else {
                if self.swing_span1 < CONE_TWIST_FIX_THRESH {
                    // Hinge around the Y axis.
                    if x.abs() > SIMD_EPSILON || z.abs() > SIMD_EPSILON {
                        self.solve_swing_limit = true;
                        if self.swing_span2 >= CONE_TWIST_FIX_THRESH {
                            y = 0.0;
                            let span2 = math::atan2(z, x);
                            if span2 > self.swing_span2 {
                                x = math::cos(self.swing_span2);
                                z = math::sin(self.swing_span2);
                            } else if span2 < -self.swing_span2 {
                                x = math::cos(self.swing_span2);
                                z = -math::sin(self.swing_span2);
                            }
                        }
                    }
                } else {
                    // Hinge around the Z axis.
                    if x.abs() > SIMD_EPSILON || y.abs() > SIMD_EPSILON {
                        self.solve_swing_limit = true;
                        if self.swing_span1 >= CONE_TWIST_FIX_THRESH {
                            z = 0.0;
                            let span1 = math::atan2(y, x);
                            if span1 > self.swing_span1 {
                                x = math::cos(self.swing_span1);
                                y = math::sin(self.swing_span1);
                            } else if span1 < -self.swing_span1 {
                                x = math::cos(self.swing_span1);
                                y = -math::sin(self.swing_span1);
                            }
                        }
                    }
                }
                let target = (x * iv_a + y * jv_a + z * kv_a).normalize();
                self.swing_axis = -iv_b.cross(target);
                self.swing_correction = self.swing_axis.length();
                if self.swing_correction.abs() > SIMD_EPSILON {
                    self.swing_axis = self.swing_axis.normalize();
                }
            }
        }

        if self.twist_span >= 0.0 {
            let (twist_angle, twist_axis) = self.compute_twist_limit_info(&q_ab_twist);
            self.twist_angle = twist_angle;
            if twist_angle > self.twist_span * self.limit_softness {
                self.solve_twist_limit = true;
                self.twist_correction = twist_angle - self.twist_span * self.limit_softness;
                self.twist_axis = q_b * (-twist_axis);
                self.k_twist = 1.0
                    / (angular_impulse_denominator(self.twist_axis, inv_inertia_a)
                        + angular_impulse_denominator(self.twist_axis, inv_inertia_b));
            }
            if self.solve_swing_limit {
                self.twist_axis_a = q_a * (-twist_axis);
            }
        } else {
            self.twist_angle = 0.0;
        }
    }

    /// Port of `computeConeLimitInfo`: the swing angle/axis of a cone
    /// rotation (twist removed) and the elliptical cone limit for that
    /// axis.
    fn compute_cone_limit_info(&self, q_cone: &Quaternion) -> (Real, Vector3, Real) {
        let swing_angle = quaternion_angle(q_cone);
        if swing_angle > SIMD_EPSILON {
            let mut swing_axis = quat_xyz(*q_cone);
            swing_axis = swing_axis.normalize();
            let x_ellipse = swing_axis.y;
            let y_ellipse = -swing_axis.z;
            let mut swing_limit = self.swing_span1;
            if x_ellipse.abs() > SIMD_EPSILON {
                let surface_slope2 = (y_ellipse * y_ellipse) / (x_ellipse * x_ellipse);
                let mut norm = 1.0 / (self.swing_span2 * self.swing_span2);
                norm += surface_slope2 / (self.swing_span1 * self.swing_span1);
                let swing_limit2 = (1.0 + surface_slope2) / norm;
                swing_limit = math::sqrt(swing_limit2);
            }
            (swing_angle, swing_axis, swing_limit)
        } else {
            (swing_angle, Vector3::ZERO, self.swing_span1)
        }
    }

    /// Port of `computeTwistLimitInfo`: the angle and axis of a pure twist
    /// rotation, taking the short way around.
    fn compute_twist_limit_info(&self, q_twist: &Quaternion) -> (Real, Vector3) {
        let mut q_min_twist = *q_twist;
        let mut twist_angle = quaternion_angle(q_twist);
        if twist_angle > bullet_math::PI {
            q_min_twist = -*q_twist;
            twist_angle = quaternion_angle(&q_min_twist);
        }
        let mut twist_axis = quat_xyz(q_min_twist);
        if twist_angle > SIMD_EPSILON {
            twist_axis = twist_axis.normalize();
        }
        (twist_angle, twist_axis)
    }

    /// Port of `adjustSwingAxisToUseEllipseNormal`: for an elliptical cone
    /// limit, aim the swing correction along the ellipse normal at the
    /// current swing point instead of straight back to the center.
    fn adjust_swing_axis_to_use_ellipse_normal(&self, swing_axis: &mut Vector3) {
        let y = -swing_axis.z;
        let z = swing_axis.y;
        if z.abs() > SIMD_EPSILON {
            let mut grad = y / z;
            grad *= self.swing_span2 / self.swing_span1;
            let y = if y > 0.0 {
                (grad * z).abs()
            } else {
                -(grad * z).abs()
            };
            swing_axis.z = -y;
            swing_axis.y = z;
            *swing_axis = swing_axis.normalize();
        }
    }
}

/// `axis · (inv_inertia_world · axis)`, Bullet's
/// `computeAngularImpulseDenominator`.
fn angular_impulse_denominator(axis: Vector3, inv_inertia_world: &Matrix3) -> Real {
    axis.dot(*inv_inertia_world * axis)
}

/// A quaternion's rotation angle in `[0, 2π]` (Bullet's `getAngle`).
pub(crate) fn quaternion_angle(q: &Quaternion) -> Real {
    2.0 * math::acos(q.w.clamp(-1.0, 1.0))
}