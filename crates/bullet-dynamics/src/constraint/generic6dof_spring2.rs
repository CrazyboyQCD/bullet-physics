//! Generic six-degree-of-freedom joint with springs (Bullet's
//! `btGeneric6DofSpring2Constraint`, the 2014 rewrite by Gabor Puhr & Tamas
//! Umenhoffer). Design baseline: `docs/design/6dof-spring2.md`.
//!
//! Unlike the classic [`super::Generic6DofConstraint`], a limited axis here
//! has no "in range" limit state: `testLimitValue` yields free (0), locked
//! (3) or limited (4) only, and a limited axis always emits *two* one-sided
//! rows whose impulse windows can only push back into the range. That is
//! what makes the limits elastic (zero force inside the range, a soft
//! force-limited wall at the boundary, accurate bounce) and what the
//! classic 6DOF rows cannot reproduce. On top of the limits each axis can
//! carry a motor, a servo motor (position servo built on the motor) and a
//! damped position spring toward an equilibrium point, with automatic
//! stiffness/damping clamps. The Euler rotation order of the angular
//! decomposition is configurable (six orders).

use super::*;
use super::shared::{adjust_angle_to_limits, get_motor_factor, normalize_angle};

/// Which Euler order decomposes the relative rotation (Bullet's
/// `RotateOrder`; the default `Xyz` matches the classic 6DOF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RotateOrder {
    #[default]
    Xyz,
    Xzy,
    Yxz,
    Yzx,
    Zxy,
    Zyx,
}

/// Per-axis flag bits (Bullet's `BT_6DOF_FLAGS_*2`): when set for an axis,
/// the axis uses its struct ERP/CFM fields; when clear, it falls back to the
/// *global* solver values (`info->erp` / `info->cfm[0]`). Four bits per
/// axis, linear = axes 0-2, angular = 3-5 (`BT_6DOF_FLAGS_AXIS_SHIFT2`).
const CFM_STOP_FLAG: u32 = 1;
const ERP_STOP_FLAG: u32 = 2;
const CFM_MOTO_FLAG: u32 = 4;
const ERP_MOTO_FLAG: u32 = 8;
const AXIS_SHIFT: u32 = 4;

/// The angular limit/motor/spring parameters of one rotation axis (Bullet's
/// `btRotationalLimitMotor2`). `upper < lower` means free, `==` locked,
/// `>` limited.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationalLimitMotor2 {
    /// Lower limit, radians (Bullet's `m_loLimit`; default 1.0 = free).
    pub lower_limit: Real,
    /// Upper limit, radians (Bullet's `m_hiLimit`; default −1.0 = free).
    pub upper_limit: Real,
    /// Bounce restitution at the limit (Bullet's `m_bounce`).
    pub bounce: Real,
    /// Positional-correction factor at the limit, used when the ERP_STOP
    /// flag is set (Bullet's `m_stopERP`; otherwise the global ERP).
    pub stop_erp: Real,
    /// Limit compliance, used when the CFM_STOP flag is set (Bullet's
    /// `m_stopCFM`; otherwise the global CFM).
    pub stop_cfm: Real,
    /// Motor bias factor, used when the ERP_MOTO flag is set (Bullet's
    /// `m_motorERP`; otherwise the global ERP).
    pub motor_erp: Real,
    /// Motor compliance, used when the CFM_MOTO flag is set (Bullet's
    /// `m_motorCFM`; otherwise the global CFM).
    pub motor_cfm: Real,
    /// Whether the motor is enabled (Bullet's `m_enableMotor`).
    pub enable_motor: bool,
    /// Motor target velocity, rad/s (Bullet's `m_targetVelocity`).
    pub target_velocity: Real,
    /// Motor max force, N·m (Bullet's `m_maxMotorForce`).
    pub max_motor_force: Real,
    /// Whether the motor acts as a position servo toward `servo_target`
    /// (Bullet's `m_servoMotor`; requires `enable_motor`).
    pub servo_motor: bool,
    /// Servo target angle, radians (Bullet's `m_servoTarget`).
    pub servo_target: Real,
    /// Whether the position spring is enabled (Bullet's `m_enableSpring`).
    pub enable_spring: bool,
    /// Spring stiffness (Bullet's `m_springStiffness`).
    pub spring_stiffness: Real,
    /// Whether to auto-clamp the stiffness for stability (Bullet's
    /// `m_springStiffnessLimited`).
    pub spring_stiffness_limited: bool,
    /// Spring damping (Bullet's `m_springDamping`).
    pub spring_damping: Real,
    /// Whether to auto-clamp the damping for stability (Bullet's
    /// `m_springDampingLimited`).
    pub spring_damping_limited: bool,
    /// Spring rest position (Bullet's `m_equilibriumPoint`).
    pub equilibrium_point: Real,
    /// Per-step limit state (0 = free, 3 = locked, 4 = limited) and errors.
    pub current_limit: i32,
    /// `current_position − lower_limit` (Bullet's `m_currentLimitError`).
    pub current_limit_error: Real,
    /// `current_position − upper_limit` (Bullet's `m_currentLimitErrorHi`).
    pub current_limit_error_hi: Real,
    /// The current (adjusted) angle (Bullet's `m_currentPosition`).
    pub current_position: Real,
}

impl Default for RotationalLimitMotor2 {
    fn default() -> Self {
        Self {
            lower_limit: 1.0,
            upper_limit: -1.0,
            bounce: 0.0,
            stop_erp: 0.2,
            stop_cfm: 0.0,
            motor_erp: 0.9,
            motor_cfm: 0.0,
            enable_motor: false,
            target_velocity: 0.0,
            max_motor_force: 6.0,
            servo_motor: false,
            servo_target: 0.0,
            enable_spring: false,
            spring_stiffness: 0.0,
            spring_stiffness_limited: false,
            spring_damping: 0.0,
            spring_damping_limited: false,
            equilibrium_point: 0.0,
            current_limit: 0,
            current_limit_error: 0.0,
            current_limit_error_hi: 0.0,
            current_position: 0.0,
        }
    }
}

impl RotationalLimitMotor2 {
    /// Test the angle against the limits (Bullet's
    /// `btRotationalLimitMotor2::testLimitValue`). The test value is *not*
    /// normalized (upstream: normalizing would lose the sign that selects
    /// which side of the wrap the angle sits on).
    pub(crate) fn test_limit_value(&mut self, test_value: Real) {
        if self.lower_limit > self.upper_limit {
            self.current_limit = 0;
            self.current_limit_error = 0.0;
        } else if self.lower_limit == self.upper_limit {
            self.current_limit_error = test_value - self.lower_limit;
            self.current_limit = 3;
        } else {
            self.current_limit_error = test_value - self.lower_limit;
            self.current_limit_error_hi = test_value - self.upper_limit;
            self.current_limit = 4;
        }
    }
}

/// The linear limit/motor/spring parameters (Bullet's
/// `btTranslationalLimitMotor2`): the rotational motor's parameter set
/// tripled. Defaults: all three axes locked (`lower == upper == 0`), and
/// unlike the angular motors `max_motor_force` defaults to zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationalLimitMotor2 {
    /// Lower limits along each frame axis (Bullet's `m_lowerLimit`).
    pub lower_limit: Vector3,
    /// Upper limits along each frame axis (Bullet's `m_upperLimit`).
    pub upper_limit: Vector3,
    /// Per-axis bounce at the limit (Bullet's `m_bounce`).
    pub bounce: Vector3,
    /// Per-axis limit ERP, used when the ERP_STOP flag is set (Bullet's
    /// `m_stopERP`; otherwise the global ERP).
    pub stop_erp: Vector3,
    /// Per-axis limit CFM, used when the CFM_STOP flag is set (Bullet's
    /// `m_stopCFM`; otherwise the global CFM).
    pub stop_cfm: Vector3,
    /// Per-axis motor ERP, used when the ERP_MOTO flag is set (Bullet's
    /// `m_motorERP`; otherwise the global ERP).
    pub motor_erp: Vector3,
    /// Per-axis motor CFM, used when the CFM_MOTO flag is set (Bullet's
    /// `m_motorCFM`; otherwise the global CFM).
    pub motor_cfm: Vector3,
    /// Per-axis motor switches (Bullet's `m_enableMotor`).
    pub enable_motor: [bool; 3],
    /// Per-axis servo switches (Bullet's `m_servoMotor`).
    pub servo_motor: [bool; 3],
    /// Per-axis spring switches (Bullet's `m_enableSpring`).
    pub enable_spring: [bool; 3],
    /// Servo targets along each axis (Bullet's `m_servoTarget`).
    pub servo_target: Vector3,
    /// Spring stiffnesses (Bullet's `m_springStiffness`).
    pub spring_stiffness: Vector3,
    /// Per-axis stiffness auto-clamp switches (Bullet's
    /// `m_springStiffnessLimited`).
    pub spring_stiffness_limited: [bool; 3],
    /// Spring dampings (Bullet's `m_springDamping`).
    pub spring_damping: Vector3,
    /// Per-axis damping auto-clamp switches (Bullet's
    /// `m_springDampingLimited`).
    pub spring_damping_limited: [bool; 3],
    /// Spring rest positions (Bullet's `m_equilibriumPoint`).
    pub equilibrium_point: Vector3,
    /// Motor target velocities (Bullet's `m_targetVelocity`).
    pub target_velocity: Vector3,
    /// Motor max forces, N (Bullet's `m_maxMotorForce`; default zero).
    pub max_motor_force: Vector3,
    /// Per-step limit states (0 = free, 3 = locked, 4 = limited).
    pub current_limit: [i32; 3],
    /// `pos − lower` per axis (Bullet's `m_currentLimitError`).
    pub current_limit_error: Vector3,
    /// `pos − upper` per axis (Bullet's `m_currentLimitErrorHi`).
    pub current_limit_error_hi: Vector3,
}

impl Default for TranslationalLimitMotor2 {
    fn default() -> Self {
        Self {
            lower_limit: Vector3::ZERO,
            upper_limit: Vector3::ZERO,
            bounce: Vector3::ZERO,
            stop_erp: Vector3::splat(0.2),
            stop_cfm: Vector3::ZERO,
            motor_erp: Vector3::splat(0.9),
            motor_cfm: Vector3::ZERO,
            enable_motor: [false; 3],
            servo_motor: [false; 3],
            enable_spring: [false; 3],
            servo_target: Vector3::ZERO,
            spring_stiffness: Vector3::ZERO,
            spring_stiffness_limited: [false; 3],
            spring_damping: Vector3::ZERO,
            spring_damping_limited: [false; 3],
            equilibrium_point: Vector3::ZERO,
            target_velocity: Vector3::ZERO,
            max_motor_force: Vector3::ZERO,
            current_limit: [0; 3],
            current_limit_error: Vector3::ZERO,
            current_limit_error_hi: Vector3::ZERO,
        }
    }
}

impl TranslationalLimitMotor2 {
    /// Test one linear axis against its limits (Bullet's
    /// `btTranslationalLimitMotor2::testLimitValue`).
    pub(crate) fn test_limit_value(&mut self, index: usize, test_value: Real) {
        let lo = self.lower_limit[index];
        let hi = self.upper_limit[index];
        if lo > hi {
            self.current_limit_error[index] = 0.0;
            self.current_limit[index] = 0;
        } else if lo == hi {
            self.current_limit_error[index] = test_value - lo;
            self.current_limit[index] = 3;
        } else {
            self.current_limit_error[index] = test_value - lo;
            self.current_limit_error_hi[index] = test_value - hi;
            self.current_limit[index] = 4;
        }
    }
}

/// A generic six-degree-of-freedom joint with springs (Bullet's
/// `btGeneric6DofSpring2Constraint`). Bullet's defaults are kept: the three
/// linear axes are locked at zero and the three angular axes are free, so a
/// fresh joint behaves like a ball joint.
#[derive(Debug, Clone, PartialEq)]
pub struct Generic6DofSpring2Constraint {
    /// Constraint frame on body A (pivot + basis in A's COM frame).
    pub frame_a: JointFrame,
    /// Constraint frame on body B (pivot + basis in B's COM frame).
    pub frame_b: JointFrame,
    /// The Euler order of the angular decomposition (Bullet's
    /// `m_rotateOrder`).
    pub rotate_order: RotateOrder,
    /// The three linear limit/motor/spring axes (Bullet's
    /// `m_linearLimits`).
    pub linear: TranslationalLimitMotor2,
    /// The three angular limit/motor/spring axes (Bullet's
    /// `m_angularLimits`).
    pub angular: [RotationalLimitMotor2; 3],
    /// Per-step state: the pivot offset in A's frame (Bullet's
    /// `m_calculatedLinearDiff`).
    pub current_linear_diff: Vector3,
    /// Per-step state: the relative Euler angles (Bullet's
    /// `m_calculatedAxisAngleDiff`).
    pub euler_angles: Vector3,
    /// Per-step state: the world angular constraint axes (Bullet's
    /// `m_calculatedAxis`).
    pub angular_axes: [Vector3; 3],
    /// Per-step state: the static-body mass split (Bullet's `m_factA` /
    /// `m_factB` = `miB/(miA+miB)` and `1 − fact_a`, `0.5` fallback).
    pub fact_a: Real,
    pub fact_b: Real,
    /// Per-step state: whether either body is static (Bullet's
    /// `m_hasStaticBody`).
    pub has_static_body: bool,
    /// Per-axis ERP/CFM override flags (Bullet's `m_flags`); driven by the
    /// `set_stop_erp`/`set_stop_cfm`/`set_motor_erp`/`set_motor_cfm`
    /// methods.
    flags: u32,
}

impl Generic6DofSpring2Constraint {
    /// A spring-2 6DOF joint with Bullet's defaults (linear locked, angular
    /// free).
    pub fn new(frame_a: JointFrame, frame_b: JointFrame, rotate_order: RotateOrder) -> Self {
        Self {
            frame_a,
            frame_b,
            rotate_order,
            linear: TranslationalLimitMotor2::default(),
            angular: Default::default(),
            current_linear_diff: Vector3::ZERO,
            euler_angles: Vector3::ZERO,
            angular_axes: [Vector3::ZERO; 3],
            fact_a: 0.5,
            fact_b: 0.5,
            has_static_body: false,
            flags: 0,
        }
    }

    // -- limits ----------------------------------------------------------

    /// Set the linear lower/upper limits along the frame axes (Bullet's
    /// `setLinearLowerLimit`/`setLinearUpperLimit`).
    pub fn set_linear_limits(&mut self, lower: Vector3, upper: Vector3) {
        self.linear.lower_limit = lower;
        self.linear.upper_limit = upper;
    }

    /// Set the angular lower/upper limits around the relative-rotation axes
    /// (Bullet's `setAngularLowerLimit`/`setAngularUpperLimit`; the angles
    /// are normalized like Bullet's `btNormalizeAngle`).
    pub fn set_angular_limits(&mut self, lower: Vector3, upper: Vector3) {
        for i in 0..3 {
            self.angular[i].lower_limit = normalize_angle(lower[i]);
            self.angular[i].upper_limit = normalize_angle(upper[i]);
        }
    }

    /// Set one axis's limits; axes 0-2 are linear, 3-5 angular (Bullet's
    /// `setLimit(axis, lo, hi)`).
    pub fn set_limit(&mut self, axis: usize, lo: Real, hi: Real) {
        if axis < 3 {
            self.linear.lower_limit[axis] = lo;
            self.linear.upper_limit[axis] = hi;
        } else {
            self.angular[axis - 3].lower_limit = normalize_angle(lo);
            self.angular[axis - 3].upper_limit = normalize_angle(hi);
        }
    }

    /// Reversed-convention variants (Bullet's `setLimitReversed` /
    /// `setAngularLowerLimitReversed` / `setAngularUpperLimitReversed`): the
    /// angular values are negated and swapped, matching tools that specify
    /// the angular limits with the opposite sign.
    pub fn set_limit_reversed(&mut self, axis: usize, lo: Real, hi: Real) {
        if axis < 3 {
            self.linear.lower_limit[axis] = lo;
            self.linear.upper_limit[axis] = hi;
        } else {
            self.angular[axis - 3].upper_limit = normalize_angle(-lo);
            self.angular[axis - 3].lower_limit = normalize_angle(-hi);
        }
    }

    /// Set the angular lower limits given in the reversed convention
    /// (Bullet's `setAngularLowerLimitReversed`).
    pub fn set_angular_lower_limit_reversed(&mut self, lower: Vector3) {
        for i in 0..3 {
            self.angular[i].upper_limit = normalize_angle(-lower[i]);
        }
    }

    /// Set the angular upper limits given in the reversed convention
    /// (Bullet's `setAngularUpperLimitReversed`).
    pub fn set_angular_upper_limit_reversed(&mut self, upper: Vector3) {
        for i in 0..3 {
            self.angular[i].lower_limit = normalize_angle(-upper[i]);
        }
    }

    /// Set the Euler rotation order (Bullet's `setRotationOrder`).
    pub fn set_rotation_order(&mut self, order: RotateOrder) {
        self.rotate_order = order;
    }

    // -- motors / servos / springs ----------------------------------------

    /// Set the bounce on `axis` (0-2 linear, 3-5 angular; Bullet's
    /// `setBounce`).
    pub fn set_bounce(&mut self, index: usize, bounce: Real) {
        if index < 3 {
            self.linear.bounce[index] = bounce;
        } else {
            self.angular[index - 3].bounce = bounce;
        }
    }

    /// Enable/disable the motor on `axis` (Bullet's `enableMotor`).
    pub fn enable_motor(&mut self, index: usize, on_off: bool) {
        if index < 3 {
            self.linear.enable_motor[index] = on_off;
        } else {
            self.angular[index - 3].enable_motor = on_off;
        }
    }

    /// Set whether the motor acts as a position servo (Bullet's
    /// `setServo`; requires `enable_motor` too).
    pub fn set_servo(&mut self, index: usize, on_off: bool) {
        if index < 3 {
            self.linear.servo_motor[index] = on_off;
        } else {
            self.angular[index - 3].servo_motor = on_off;
        }
    }

    /// Set the motor target velocity on `axis` (Bullet's
    /// `setTargetVelocity`).
    pub fn set_target_velocity(&mut self, index: usize, velocity: Real) {
        if index < 3 {
            self.linear.target_velocity[index] = velocity;
        } else {
            self.angular[index - 3].target_velocity = velocity;
        }
    }

    /// Set the servo target on `axis` (Bullet's `setServoTarget`; angular
    /// targets are wrapped to [−π, π]).
    pub fn set_servo_target(&mut self, index: usize, target_org: Real) {
        if index < 3 {
            self.linear.servo_target[index] = target_org;
        } else {
            // Wrap between −π and π (Bullet's floor-based positive mod;
            // reproduced with truncating `%` plus the original `m < 0` /
            // `m >= 2π` branches — identical results, no floor needed).
            let two_pi = 2.0 * bullet_math::PI;
            let mut target = target_org + bullet_math::PI;
            let m = target % two_pi;
            if m >= two_pi {
                target = 0.0;
            } else if m < 0.0 {
                if two_pi + m == two_pi {
                    target = 0.0;
                } else {
                    target = two_pi + m;
                }
            } else {
                target = m;
            }
            target -= bullet_math::PI;
            self.angular[index - 3].servo_target = target;
        }
    }

    /// Set the motor max force on `axis` (Bullet's `setMaxMotorForce`).
    pub fn set_max_motor_force(&mut self, index: usize, force: Real) {
        if index < 3 {
            self.linear.max_motor_force[index] = force;
        } else {
            self.angular[index - 3].max_motor_force = force;
        }
    }

    /// Enable/disable the position spring on `axis` (Bullet's
    /// `enableSpring`).
    pub fn enable_spring(&mut self, index: usize, on_off: bool) {
        if index < 3 {
            self.linear.enable_spring[index] = on_off;
        } else {
            self.angular[index - 3].enable_spring = on_off;
        }
    }

    /// Set the spring stiffness on `axis` (Bullet's `setStiffness`); with
    /// `limit_needed` the solver auto-clamps the stiffness where it would
    /// otherwise oscillate unrealistically.
    pub fn set_stiffness(&mut self, index: usize, stiffness: Real, limit_needed: bool) {
        if index < 3 {
            self.linear.spring_stiffness[index] = stiffness;
            self.linear.spring_stiffness_limited[index] = limit_needed;
        } else {
            self.angular[index - 3].spring_stiffness = stiffness;
            self.angular[index - 3].spring_stiffness_limited = limit_needed;
        }
    }

    /// Set the spring damping on `axis` (Bullet's `setDamping`); with
    /// `limit_needed` the solver auto-clamps the damping where it would
    /// otherwise blow up.
    pub fn set_damping(&mut self, index: usize, damping: Real, limit_needed: bool) {
        if index < 3 {
            self.linear.spring_damping[index] = damping;
            self.linear.spring_damping_limited[index] = limit_needed;
        } else {
            self.angular[index - 3].spring_damping = damping;
            self.angular[index - 3].spring_damping_limited = limit_needed;
        }
    }

    /// Set the spring rest value of `axis` explicitly (Bullet's
    /// `setEquilibriumPoint(index, val)`). The no-argument/current-pose
    /// upstream variants need world access; pass measured values
    /// ([`Self::get_angle`] / [`Self::get_relative_pivot_position`]) or
    /// build the frames at the rest pose so the default zero equilibrium is
    /// correct.
    pub fn set_equilibrium_point(&mut self, index: usize, value: Real) {
        if index < 3 {
            self.linear.equilibrium_point[index] = value;
        } else {
            self.angular[index - 3].equilibrium_point = value;
        }
    }

    // -- per-axis ERP/CFM overrides (Bullet's `setParam`) ------------------

    /// Override the limit ERP of `axis` (Bullet's
    /// `setParam(BT_CONSTRAINT_STOP_ERP, v, axis)`); without this override
    /// the axis uses the global solver ERP.
    pub fn set_stop_erp(&mut self, axis: usize, value: Real) {
        if axis < 3 {
            self.linear.stop_erp[axis] = value;
        } else {
            self.angular[axis - 3].stop_erp = value;
        }
        self.flags |= ERP_STOP_FLAG << (axis as u32 * AXIS_SHIFT);
    }

    /// Override the limit CFM of `axis` (Bullet's
    /// `setParam(BT_CONSTRAINT_STOP_CFM, v, axis)`); without this override
    /// the axis uses the global solver CFM.
    pub fn set_stop_cfm(&mut self, axis: usize, value: Real) {
        if axis < 3 {
            self.linear.stop_cfm[axis] = value;
        } else {
            self.angular[axis - 3].stop_cfm = value;
        }
        self.flags |= CFM_STOP_FLAG << (axis as u32 * AXIS_SHIFT);
    }

    /// Override the motor ERP of `axis` (Bullet's
    /// `setParam(BT_CONSTRAINT_ERP, v, axis)`); without this override the
    /// axis uses the global solver ERP.
    pub fn set_motor_erp(&mut self, axis: usize, value: Real) {
        if axis < 3 {
            self.linear.motor_erp[axis] = value;
        } else {
            self.angular[axis - 3].motor_erp = value;
        }
        self.flags |= ERP_MOTO_FLAG << (axis as u32 * AXIS_SHIFT);
    }

    /// Override the motor CFM of `axis` (Bullet's
    /// `setParam(BT_CONSTRAINT_CFM, v, axis)`); without this override the
    /// axis uses the global solver CFM.
    pub fn set_motor_cfm(&mut self, axis: usize, value: Real) {
        if axis < 3 {
            self.linear.motor_cfm[axis] = value;
        } else {
            self.angular[axis - 3].motor_cfm = value;
        }
        self.flags |= CFM_MOTO_FLAG << (axis as u32 * AXIS_SHIFT);
    }

    // -- readbacks ---------------------------------------------------------

    /// The current relative Euler angle of `axis` (0-2; Bullet's `getAngle`),
    /// updated by the last solve step.
    pub fn get_angle(&self, axis: usize) -> Real {
        self.euler_angles[axis]
    }

    /// The current pivot offset along `axis` (0-2, in A's frame; Bullet's
    /// `getRelativePivotPosition`), updated by the last solve step.
    pub fn get_relative_pivot_position(&self, axis: usize) -> Real {
        self.current_linear_diff[axis]
    }

    /// The current angular constraint axis in world coordinates (Bullet's
    /// `getAxis`), updated by the last solve step.
    pub fn get_axis(&self, axis: usize) -> Vector3 {
        self.angular_axes[axis]
    }

    // -- row assembly ------------------------------------------------------

    /// Flatten this joint into its solver rows (Bullet's `getInfo1` +
    /// `getInfo2` fused, as in the other joint kinds). `erp`/`global_cfm`
    /// are the global solver values the per-axis flags fall back to.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_rows(
        &mut self,
        body_a: usize,
        body_b: usize,
        bodies: &[RigidBody],
        inv_dt: Real,
        erp: Real,
        global_cfm: Real,
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

        // Linear info (Bullet's `calculateLinearInfo`): the pivot offset in
        // A's frame, tested per axis.
        let diff = tr_a.transpose() * (pivot_b - pivot_a);
        self.current_linear_diff = diff;
        for i in 0..3 {
            self.linear.test_limit_value(i, diff[i]);
        }

        // Angle info + angular limit tests (Bullet's `calculateAngleInfo`
        // and `getInfo1`'s `testAngularLimitMotor` loop).
        let (euler, axes) = calculate_angle_info(&tr_a, &tr_b, self.rotate_order);
        self.euler_angles = euler;
        self.angular_axes = axes;
        for i in 0..3 {
            self.test_angular_limit_motor(i);
        }

        // The static-body mass split (Bullet's `calculateTransforms` tail).
        let mi_a = a.inv_mass();
        let mi_b = b.inv_mass();
        self.has_static_body = mi_a < Real::EPSILON || mi_b < Real::EPSILON;
        let mi_s = mi_a + mi_b;
        self.fact_a = if mi_s > 0.0 { mi_b / mi_s } else { 0.5 };
        self.fact_b = 1.0 - self.fact_a;

        // Angular rows first (Bullet's `setAngularLimits`), in the
        // rotate-order solve permutation. Upstream resolves the global
        // ERP/CFM fallback *into the struct* for the angular axes when the
        // per-axis flag is unset (so a field written without the flag is
        // overwritten by the global value every step) — verbatim.
        for i in 0..3 {
            if self.angular_produces_rows(i) {
                let shift = (i as u32 + 3) * AXIS_SHIFT;
                if self.flags & (CFM_STOP_FLAG << shift) == 0 {
                    self.angular[i].stop_cfm = global_cfm;
                }
                if self.flags & (ERP_STOP_FLAG << shift) == 0 {
                    self.angular[i].stop_erp = erp;
                }
                if self.flags & (CFM_MOTO_FLAG << shift) == 0 {
                    self.angular[i].motor_cfm = global_cfm;
                }
                if self.flags & (ERP_MOTO_FLAG << shift) == 0 {
                    self.angular[i].motor_erp = erp;
                }
            }
        }
        for i in 0..3 {
            if self.angular_produces_rows(i) {
                let axis = self.angular_axes[i];
                self.push_limit_motor_rows(
                    body_a,
                    body_b,
                    a,
                    b,
                    inv_dt,
                    axis,
                    true,
                    true,
                    Vector3::ZERO,
                    Vector3::ZERO,
                    &self.angular[i],
                    out,
                );
            }
        }

        // Linear rows (Bullet's `setLinearLimits`): the axis is frame A's
        // column i; the per-axis parameters are merged into a rotational
        // motor value with the global ERP/CFM fallback (non-destructive,
        // like upstream's local `limot`).
        for i in 0..3 {
            if self.linear.current_limit[i] != 0
                || self.linear.enable_motor[i]
                || self.linear.enable_spring[i]
            {
                let shift = i as u32 * AXIS_SHIFT;
                let flag = |f: u32| self.flags & (f << shift) != 0;
                let limot = RotationalLimitMotor2 {
                    bounce: self.linear.bounce[i],
                    current_limit: self.linear.current_limit[i],
                    current_position: diff[i],
                    current_limit_error: self.linear.current_limit_error[i],
                    current_limit_error_hi: self.linear.current_limit_error_hi[i],
                    enable_motor: self.linear.enable_motor[i],
                    servo_motor: self.linear.servo_motor[i],
                    servo_target: self.linear.servo_target[i],
                    enable_spring: self.linear.enable_spring[i],
                    spring_stiffness: self.linear.spring_stiffness[i],
                    spring_stiffness_limited: self.linear.spring_stiffness_limited[i],
                    spring_damping: self.linear.spring_damping[i],
                    spring_damping_limited: self.linear.spring_damping_limited[i],
                    equilibrium_point: self.linear.equilibrium_point[i],
                    upper_limit: self.linear.upper_limit[i],
                    lower_limit: self.linear.lower_limit[i],
                    max_motor_force: self.linear.max_motor_force[i],
                    target_velocity: self.linear.target_velocity[i],
                    stop_cfm: if flag(CFM_STOP_FLAG) {
                        self.linear.stop_cfm[i]
                    } else {
                        global_cfm
                    },
                    stop_erp: if flag(ERP_STOP_FLAG) {
                        self.linear.stop_erp[i]
                    } else {
                        erp
                    },
                    motor_cfm: if flag(CFM_MOTO_FLAG) {
                        self.linear.motor_cfm[i]
                    } else {
                        global_cfm
                    },
                    motor_erp: if flag(ERP_MOTO_FLAG) {
                        self.linear.motor_erp[i]
                    } else {
                        erp
                    },
                };
                let axis = tr_a.col(i);
                // `rotAllowed` (Bullet's static-body angular jacobian
                // scaling gate): off when both orthogonal angular axes are
                // violated. The `current_limit == 1/2` arms are upstream
                // copy-paste from the classic constraint (spring-2 states
                // are 0/3/4 only) — ported verbatim.
                let indx1 = (i + 1) % 3;
                let indx2 = (i + 2) % 3;
                let mut rot_allowed = true;
                if angular_axis_violated(&self.angular[indx1])
                    && angular_axis_violated(&self.angular[indx2])
                {
                    rot_allowed = false;
                }
                self.push_limit_motor_rows(
                    body_a,
                    body_b,
                    a,
                    b,
                    inv_dt,
                    axis,
                    false,
                    rot_allowed,
                    r_a,
                    r_b,
                    &limot,
                    out,
                );
            }
        }
    }

    /// Whether angular axis `i` produces rows this step (Bullet's
    /// `setAngularLimits` gate; the global ERP/CFM fallback resolution uses
    /// the same gate).
    fn angular_produces_rows(&self, i: usize) -> bool {
        self.angular[i].current_limit != 0
            || self.angular[i].enable_motor
            || self.angular[i].enable_spring
    }

    /// Test the angular limit on `axis` (Bullet's
    /// `testAngularLimitMotor`): adjust the angle to the limit window, store
    /// it and run the limit state machine.
    fn test_angular_limit_motor(&mut self, axis: usize) {
        let angle = self.euler_angles[axis];
        let angle = adjust_angle_to_limits(
            angle,
            self.angular[axis].lower_limit,
            self.angular[axis].upper_limit,
        );
        self.angular[axis].current_position = angle;
        self.angular[axis].test_limit_value(angle);
    }

    /// All row shapes of one axis (Bullet's `get_limit_motor_info2`): the
    /// limit-4 pair, the locked row, the motor row, the servo row and the
    /// spring row, in upstream emission order.
    #[allow(clippy::too_many_arguments)]
    fn push_limit_motor_rows(
        &self,
        body_a: usize,
        body_b: usize,
        a: &RigidBody,
        b: &RigidBody,
        inv_dt: Real,
        ax1: Vector3,
        rotational: bool,
        rot_allowed: bool,
        r_a: Vector3,
        r_b: Vector3,
        limot: &RotationalLimitMotor2,
        out: &mut Vec<ConstraintRow>,
    ) {
        let sign: Real = if rotational { -1.0 } else { 1.0 };

        if limot.current_limit == 4 {
            // Two one-sided rows whose windows can only push back into the
            // range — the elastic limit. The bounce branches compare the
            // bias against the incoming velocity first (accurate bounce).
            let vel = if rotational {
                a.angular_velocity.dot(ax1) - b.angular_velocity.dot(ax1)
            } else {
                a.linear_velocity.dot(ax1) - b.linear_velocity.dot(ax1)
            };

            // Row 1: the lower side.
            let mut row_error = inv_dt * limot.stop_erp * limot.current_limit_error * sign;
            if rotational {
                if row_error - vel * limot.stop_erp > 0.0 {
                    let bounce_error = -limot.bounce * vel;
                    if bounce_error > row_error {
                        row_error = bounce_error;
                    }
                }
            } else if row_error - vel * limot.stop_erp < 0.0 {
                let bounce_error = -limot.bounce * vel;
                if bounce_error < row_error {
                    row_error = bounce_error;
                }
            }
            let (lo, hi) = if rotational {
                (0.0, Real::INFINITY)
            } else {
                (Real::NEG_INFINITY, 0.0)
            };
            self.push_row(
                body_a, body_b, a, b, ax1, rotational, rot_allowed, r_a, r_b, row_error,
                limot.stop_cfm,
                mirrored_window(lo, hi),
                out,
            );

            // Row 2: the upper side.
            let mut row_error = inv_dt * limot.stop_erp * limot.current_limit_error_hi * sign;
            if rotational {
                if row_error - vel * limot.stop_erp < 0.0 {
                    let bounce_error = -limot.bounce * vel;
                    if bounce_error < row_error {
                        row_error = bounce_error;
                    }
                }
            } else if row_error - vel * limot.stop_erp > 0.0 {
                let bounce_error = -limot.bounce * vel;
                if bounce_error > row_error {
                    row_error = bounce_error;
                }
            }
            let (lo, hi) = if rotational {
                (Real::NEG_INFINITY, 0.0)
            } else {
                (0.0, Real::INFINITY)
            };
            self.push_row(
                body_a, body_b, a, b, ax1, rotational, rot_allowed, r_a, r_b, row_error,
                limot.stop_cfm,
                mirrored_window(lo, hi),
                out,
            );
        }

        if limot.current_limit == 3 {
            // Locked axis: a bilateral equality row pulling to the locked
            // position.
            let row_error = inv_dt * limot.stop_erp * limot.current_limit_error * sign;
            self.push_row(
                body_a, body_b, a, b, ax1, rotational, rot_allowed, r_a, r_b, row_error,
                limot.stop_cfm,
                RowLimits::Unbounded,
                out,
            );
        }

        if limot.enable_motor && !limot.servo_motor {
            // Plain velocity motor. The tag-velocity sign convention flips
            // for the linear axes; the error carries no rotational sign
            // flip (upstream asymmetry, kept).
            let tag_vel = if rotational {
                limot.target_velocity
            } else {
                -limot.target_velocity
            };
            let mot_fact = get_motor_factor(
                limot.current_position,
                limot.lower_limit,
                limot.upper_limit,
                tag_vel,
                inv_dt * limot.motor_erp,
            );
            let row_error = mot_fact * limot.target_velocity;
            let f = limot.max_motor_force / inv_dt;
            self.push_row(
                body_a, body_b, a, b, ax1, rotational, rot_allowed, r_a, r_b, row_error,
                limot.motor_cfm,
                mirrored_window(-f, f),
                out,
            );
        }

        if limot.enable_motor && limot.servo_motor {
            // Position servo: a velocity motor whose target and fade window
            // follow the servo target.
            let mut error = limot.current_position - limot.servo_target;
            let mut cur_servo_target = limot.servo_target;
            if rotational {
                if error > bullet_math::PI {
                    error -= 2.0 * bullet_math::PI;
                    cur_servo_target += 2.0 * bullet_math::PI;
                }
                if error < -bullet_math::PI {
                    error += 2.0 * bullet_math::PI;
                    cur_servo_target -= 2.0 * bullet_math::PI;
                }
            }
            let targetvelocity = if error < 0.0 {
                -limot.target_velocity
            } else {
                limot.target_velocity
            };
            let tag_vel = -targetvelocity;
            let mot_fact = if error != 0.0 {
                let (low_limit, hi_limit) = if limot.lower_limit > limot.upper_limit {
                    (
                        if error > 0.0 {
                            cur_servo_target
                        } else {
                            Real::NEG_INFINITY
                        },
                        if error < 0.0 {
                            cur_servo_target
                        } else {
                            Real::INFINITY
                        },
                    )
                } else {
                    (
                        if error > 0.0 && cur_servo_target > limot.lower_limit {
                            cur_servo_target
                        } else {
                            limot.lower_limit
                        },
                        if error < 0.0 && cur_servo_target < limot.upper_limit {
                            cur_servo_target
                        } else {
                            limot.upper_limit
                        },
                    )
                };
                get_motor_factor(
                    limot.current_position,
                    low_limit,
                    hi_limit,
                    tag_vel,
                    inv_dt * limot.motor_erp,
                )
            } else {
                0.0
            };
            let row_error = mot_fact * targetvelocity * sign;
            let f = limot.max_motor_force / inv_dt;
            self.push_row(
                body_a, body_b, a, b, ax1, rotational, rot_allowed, r_a, r_b, row_error,
                limot.motor_cfm,
                mirrored_window(-f, f),
                out,
            );
        }

        if limot.enable_spring {
            // Damped position spring (upstream in-block comment: the
            // requested velocity only needs to be at least the "proper"
            // one, because the impulse window bounds the actual force).
            let error = limot.current_position - limot.equilibrium_point;
            let vel = if rotational {
                a.angular_velocity.dot(ax1) - b.angular_velocity.dot(ax1)
            } else {
                let tan_vel_a = a.angular_velocity.cross(r_a);
                let tan_vel_b = b.angular_velocity.cross(r_b);
                (a.linear_velocity + tan_vel_a).dot(ax1)
                    - (b.linear_velocity + tan_vel_b).dot(ax1)
            };
            let mut m_a = 1.0 / a.inv_mass();
            let mut m_b = 1.0 / b.inv_mass();
            if rotational {
                if a.inv_mass() != 0.0 {
                    let rr_a = r_a.length_squared();
                    m_a = m_a * rr_a + 1.0 / (a.inv_inertia_world() * ax1).length();
                }
                if b.inv_mass() != 0.0 {
                    let rr_b = r_b.length_squared();
                    m_b = m_b * rr_b + 1.0 / (b.inv_inertia_world() * ax1).length();
                }
            }
            let m = if a.inv_mass() == 0.0 {
                m_b
            } else if b.inv_mass() == 0.0 {
                m_a
            } else {
                m_a * m_b / (m_a + m_b)
            };
            let dt = 1.0 / inv_dt;
            let mut kd = limot.spring_damping;
            let mut ks = limot.spring_stiffness;
            let angularfreq = math::sqrt(ks / m);
            // Limit stiffness: the spring must not be sampled faster than a
            // quarter of its angular frequency.
            if limot.spring_stiffness_limited && 0.25 < angularfreq * dt {
                ks = 1.0 / dt / dt / 16.0 * m;
            }
            // Avoid damping that would blow up the spring.
            if limot.spring_damping_limited && kd * dt > m {
                kd = m / dt;
            }
            let fs = ks * error * dt;
            let fd = -kd * vel * sign * dt;
            let f = fs + fd;
            let row_error = vel + f / m * sign;
            let minf = if f < fd { f } else { fd };
            let maxf = if f < fd { fd } else { f };
            let (lo, hi) = if !rotational {
                (
                    if minf > 0.0 { 0.0 } else { minf },
                    if maxf < 0.0 { 0.0 } else { maxf },
                )
            } else {
                (
                    if -maxf > 0.0 { 0.0 } else { -maxf },
                    if -minf < 0.0 { 0.0 } else { -minf },
                )
            };
            // Spring rows never carry compliance (`cfm = BT_ZERO` upstream).
            self.push_row(
                body_a, body_b, a, b, ax1, rotational, rot_allowed, r_a, r_b, row_error, 0.0,
                mirrored_window(lo, hi),
                out,
            );
        }
    }

    /// One solver row (Bullet's `calculateJacobi` + row write). Bullet's
    /// jacobians are `J1 = +ax1`, `J2 = −ax1` with the linear angular parts
    /// `tmpA = r_a×ax1` / `tmpB = r_b×ax1` scaled by `fact_a`/`fact_b` for a
    /// static body when the orthogonal rotations are both violated; our row
    /// convention mirrors λ, so the B-side jacobians enter with a flipped
    /// sign (see [`ConstraintRow`]).
    #[allow(clippy::too_many_arguments)]
    fn push_row(
        &self,
        body_a: usize,
        body_b: usize,
        a: &RigidBody,
        b: &RigidBody,
        ax1: Vector3,
        rotational: bool,
        rot_allowed: bool,
        r_a: Vector3,
        r_b: Vector3,
        bullet_error: Real,
        cfm: Real,
        limits: RowLimits,
        out: &mut Vec<ConstraintRow>,
    ) {
        let (lin_a, ang_a, lin_b, ang_b) = if rotational {
            (Vector3::ZERO, ax1, Vector3::ZERO, ax1)
        } else {
            let mut ang_a = r_a.cross(ax1);
            let mut ang_b = r_b.cross(ax1);
            if self.has_static_body && !rot_allowed {
                ang_a *= self.fact_a;
                ang_b *= self.fact_b;
            }
            (ax1, ang_a, ax1, ang_b)
        };
        if cfm != 0.0 {
            out.push(ConstraintRow::prepare_with_cfm(
                body_a, body_b, lin_a, ang_a, lin_b, ang_b, a, b, -bullet_error, limits, cfm,
            ));
        } else {
            out.push(ConstraintRow::prepare(
                body_a, body_b, lin_a, ang_a, lin_b, ang_b, a, b, -bullet_error, limits,
            ));
        }
    }
}

/// The linear-axis violation gate for `rot_allowed` (Bullet's
/// `indx1Violated`/`indx2Violated`).
fn angular_axis_violated(m: &RotationalLimitMotor2) -> bool {
    const THRESHOLD: Real = 1.0e-3; // D6_LIMIT_ERROR_THRESHOLD_FOR_ROTATION
    m.current_limit == 1
        || m.current_limit == 2
        || (m.current_limit == 3
            && (m.current_limit_error < -THRESHOLD || m.current_limit_error > THRESHOLD))
        || (m.current_limit == 4
            && (m.current_limit_error < -THRESHOLD || m.current_limit_error_hi > THRESHOLD))
}

/// The relative-orientation info for the configured order (Bullet's
/// `calculateAngleInfo`): the Euler angles of `trA⁻¹ · trB` and the three
/// world angular constraint axes built from explicit cross products per
/// order, normalized.
fn calculate_angle_info(
    tr_a: &Matrix3,
    tr_b: &Matrix3,
    order: RotateOrder,
) -> (Vector3, [Vector3; 3]) {
    let relative = tr_a.transpose() * *tr_b;
    let euler = match order {
        RotateOrder::Xyz => euler_xyz(&relative),
        RotateOrder::Xzy => matrix_to_euler_xzy(&relative),
        RotateOrder::Yxz => matrix_to_euler_yxz(&relative),
        RotateOrder::Yzx => matrix_to_euler_yzx(&relative),
        RotateOrder::Zxy => matrix_to_euler_zxy(&relative),
        RotateOrder::Zyx => matrix_to_euler_zyx(&relative),
    };
    let mut axes = [Vector3::ZERO; 3];
    match order {
        RotateOrder::Xyz => {
            let axis0 = tr_b.col(0);
            let axis2 = tr_a.col(2);
            axes[1] = axis2.cross(axis0);
            axes[0] = axes[1].cross(axis2);
            axes[2] = axis0.cross(axes[1]);
        }
        RotateOrder::Xzy => {
            let axis0 = tr_b.col(0);
            let axis1 = tr_a.col(1);
            axes[2] = axis0.cross(axis1);
            axes[0] = axis1.cross(axes[2]);
            axes[1] = axes[2].cross(axis0);
        }
        RotateOrder::Yxz => {
            let axis1 = tr_b.col(1);
            let axis2 = tr_a.col(2);
            axes[0] = axis1.cross(axis2);
            axes[1] = axis2.cross(axes[0]);
            axes[2] = axes[0].cross(axis1);
        }
        RotateOrder::Yzx => {
            let axis0 = tr_a.col(0);
            let axis1 = tr_b.col(1);
            axes[2] = axis0.cross(axis1);
            axes[0] = axis1.cross(axes[2]);
            axes[1] = axes[2].cross(axis0);
        }
        RotateOrder::Zxy => {
            let axis1 = tr_a.col(1);
            let axis2 = tr_b.col(2);
            axes[0] = axis1.cross(axis2);
            axes[1] = axis2.cross(axes[0]);
            axes[2] = axes[0].cross(axis1);
        }
        RotateOrder::Zyx => {
            let axis0 = tr_a.col(0);
            let axis2 = tr_b.col(2);
            axes[1] = axis2.cross(axis0);
            axes[0] = axes[1].cross(axis2);
            axes[2] = axis0.cross(axes[1]);
        }
    }
    for axis in axes.iter_mut() {
        *axis = axis.normalize();
    }
    (euler, axes)
}

/// Bullet's `btGetMatrixElem`: `mat[index % 3][index / 3]` (row-major) =
/// glam's `m.col(index / 3)[index % 3]`.
#[inline]
fn elem(m: &Matrix3, index: usize) -> Real {
    m.col(index / 3)[index % 3]
}

/// Bullet's `matrixToEulerXZY` (the `bool` "exact" return is ignored
/// upstream — the gimbal fallback angles always feed the state; our port
/// returns the angles only).
fn matrix_to_euler_xzy(m: &Matrix3) -> Vector3 {
    let fi = elem(m, 1);
    if fi < 1.0 {
        if fi > -1.0 {
            Vector3::new(
                math::atan2(elem(m, 7), elem(m, 4)),
                math::atan2(elem(m, 2), elem(m, 0)),
                math::asin(-elem(m, 1)),
            )
        } else {
            Vector3::new(
                -math::atan2(-elem(m, 6), elem(m, 8)),
                0.0,
                bullet_math::FRAC_PI_2,
            )
        }
    } else {
        Vector3::new(
            math::atan2(-elem(m, 6), elem(m, 8)),
            0.0,
            -bullet_math::FRAC_PI_2,
        )
    }
}

/// Bullet's `matrixToEulerYXZ`.
fn matrix_to_euler_yxz(m: &Matrix3) -> Vector3 {
    let fi = elem(m, 5);
    if fi < 1.0 {
        if fi > -1.0 {
            Vector3::new(
                math::asin(-elem(m, 5)),
                math::atan2(elem(m, 2), elem(m, 8)),
                math::atan2(elem(m, 3), elem(m, 4)),
            )
        } else {
            Vector3::new(
                bullet_math::FRAC_PI_2,
                -math::atan2(-elem(m, 1), elem(m, 0)),
                0.0,
            )
        }
    } else {
        Vector3::new(
            -bullet_math::FRAC_PI_2,
            math::atan2(-elem(m, 1), elem(m, 0)),
            0.0,
        )
    }
}

/// Bullet's `matrixToEulerYZX`.
fn matrix_to_euler_yzx(m: &Matrix3) -> Vector3 {
    let fi = elem(m, 3);
    if fi < 1.0 {
        if fi > -1.0 {
            Vector3::new(
                math::atan2(-elem(m, 5), elem(m, 4)),
                math::atan2(-elem(m, 6), elem(m, 0)),
                math::asin(elem(m, 3)),
            )
        } else {
            Vector3::new(
                0.0,
                -math::atan2(elem(m, 7), elem(m, 8)),
                -bullet_math::FRAC_PI_2,
            )
        }
    } else {
        Vector3::new(
            0.0,
            math::atan2(elem(m, 7), elem(m, 8)),
            bullet_math::FRAC_PI_2,
        )
    }
}

/// Bullet's `matrixToEulerZXY`.
fn matrix_to_euler_zxy(m: &Matrix3) -> Vector3 {
    let fi = elem(m, 7);
    if fi < 1.0 {
        if fi > -1.0 {
            Vector3::new(
                math::asin(elem(m, 7)),
                math::atan2(-elem(m, 6), elem(m, 8)),
                math::atan2(-elem(m, 1), elem(m, 4)),
            )
        } else {
            Vector3::new(
                -bullet_math::FRAC_PI_2,
                0.0,
                -math::atan2(elem(m, 2), elem(m, 0)),
            )
        }
    } else {
        Vector3::new(
            bullet_math::FRAC_PI_2,
            0.0,
            math::atan2(elem(m, 2), elem(m, 0)),
        )
    }
}

/// Bullet's `matrixToEulerZYX`.
fn matrix_to_euler_zyx(m: &Matrix3) -> Vector3 {
    let fi = elem(m, 6);
    if fi < 1.0 {
        if fi > -1.0 {
            Vector3::new(
                math::atan2(elem(m, 7), elem(m, 8)),
                math::asin(-elem(m, 6)),
                math::atan2(elem(m, 3), elem(m, 0)),
            )
        } else {
            Vector3::new(
                0.0,
                bullet_math::FRAC_PI_2,
                -math::atan2(-elem(m, 1), -elem(m, 2)),
            )
        }
    } else {
        Vector3::new(
            0.0,
            -bullet_math::FRAC_PI_2,
            math::atan2(-elem(m, 1), -elem(m, 2)),
        )
    }
}

/// Mirror a Bullet impulse window `[lower, upper]` through our row
/// convention (`λ_ours = −λ_bullet` → `[−upper, −lower]`), collapsing to the
/// dedicated [`RowLimits`] shapes where they apply.
fn mirrored_window(lower: Real, upper: Real) -> RowLimits {
    match (lower.is_finite(), upper.is_finite()) {
        (true, true) if lower == -upper => RowLimits::Bilateral(upper),
        (true, true) => RowLimits::Window(-upper, -lower),
        (false, true) => RowLimits::LowerBounded(-upper),
        (true, false) => RowLimits::UpperBounded(-lower),
        (false, false) => RowLimits::Unbounded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::PhysicsWorld;
    use bullet_collision::shapes::CollisionShape;
    use bullet_math::{FRAC_PI_2, PI};

    const DT: Real = 1.0 / 60.0;

    // -- limit state machine ------------------------------------------------

    /// `testLimitValue` has no "in range" state: free only when `lo > hi`,
    /// locked (3) when equal, limited (4) otherwise — with both signed
    /// errors.
    #[test]
    fn limit_state_machine_has_no_in_range_state() {
        let mut m = RotationalLimitMotor2 {
            lower_limit: 1.0,
            upper_limit: -1.0,
            ..Default::default()
        };
        m.test_limit_value(0.3);
        assert_eq!(m.current_limit, 0);
        assert_eq!(m.current_limit_error, 0.0);

        m.lower_limit = 0.5;
        m.upper_limit = 0.5;
        m.test_limit_value(0.7);
        assert_eq!(m.current_limit, 3);
        assert!((m.current_limit_error - 0.2).abs() < 1e-6);

        m.lower_limit = -1.0;
        m.upper_limit = 1.0;
        m.test_limit_value(0.3);
        assert_eq!(m.current_limit, 4);
        assert!((m.current_limit_error - 1.3).abs() < 1e-6);
        assert!((m.current_limit_error_hi + 0.7).abs() < 1e-6);
        m.test_limit_value(-1.6);
        assert_eq!(m.current_limit, 4);
        assert!((m.current_limit_error + 0.6).abs() < 1e-6);
        assert!((m.current_limit_error_hi + 2.6).abs() < 1e-6);

        let mut t = TranslationalLimitMotor2::default();
        t.test_limit_value(0, -0.2);
        assert_eq!(t.current_limit[0], 3);
        assert_eq!(t.current_limit_error[0], -0.2);
    }

    /// Defaults are Bullet's: angular axes free (`lo 1.0 / hi −1.0`),
    /// linear axes locked at zero with a zero max motor force, and the
    /// motor/stop factors 0.2 / 0.9.
    #[test]
    fn defaults_match_bullet() {
        let m = RotationalLimitMotor2::default();
        assert_eq!(m.lower_limit, 1.0);
        assert_eq!(m.upper_limit, -1.0);
        assert_eq!(m.stop_erp, 0.2);
        assert_eq!(m.motor_erp, 0.9);
        assert_eq!(m.max_motor_force, 6.0);
        assert!(!m.enable_motor && !m.servo_motor && !m.enable_spring);

        let t = TranslationalLimitMotor2::default();
        assert_eq!(t.lower_limit, Vector3::ZERO);
        assert_eq!(t.upper_limit, Vector3::ZERO);
        assert_eq!(t.max_motor_force, Vector3::ZERO);
        assert_eq!(t.stop_erp, Vector3::splat(0.2));
        assert_eq!(t.motor_erp, Vector3::splat(0.9));
    }

    // -- euler machinery ----------------------------------------------------

    /// Identity decomposes to zero angles under all six orders.
    #[test]
    fn euler_identity_is_zero_for_all_orders() {
        let identity = Matrix3::IDENTITY;
        for order in [
            RotateOrder::Xyz,
            RotateOrder::Xzy,
            RotateOrder::Yxz,
            RotateOrder::Yzx,
            RotateOrder::Zxy,
            RotateOrder::Zyx,
        ] {
            let (euler, axes) = calculate_angle_info(&identity, &identity, order);
            assert!(euler[0].abs() < 1e-6 && euler[1].abs() < 1e-6 && euler[2].abs() < 1e-6, "{order:?} → {euler}");
            for axis in axes {
                assert!((axis.length() - 1.0).abs() < 1e-5);
            }
        }
    }

    /// Angle pins derived by hand from the extraction formulas (the sign
    /// conventions are Bullet's own — do not "fix" them): under XYZ a
    /// rotation of B about Y reads as a *negative* Y angle, under XZY a
    /// rotation about X reads as a negative X angle.
    #[test]
    fn euler_sign_pins_for_xyz_and_xzy() {
        let theta = 0.4;
        let ry = Matrix3::from_quat(Quaternion::from_rotation_y(theta));
        let (euler, _) = calculate_angle_info(&Matrix3::IDENTITY, &ry, RotateOrder::Xyz);
        assert!(euler[0].abs() < 1e-5 && euler[2].abs() < 1e-5);
        assert!((euler[1] + theta).abs() < 1e-5, "y = {}", euler[1]);

        let rx = Matrix3::from_quat(Quaternion::from_rotation_x(theta));
        let (euler, _) = calculate_angle_info(&Matrix3::IDENTITY, &rx, RotateOrder::Xzy);
        assert!(euler[1].abs() < 1e-5 && euler[2].abs() < 1e-5);
        assert!((euler[0] + theta).abs() < 1e-5, "x = {}", euler[0]);
    }

    /// The gimbal fallback arm feeds the state (upstream ignores the
    /// `matrixToEuler*` bool): Ry(π/2) lands in the `fi ≤ −1` branch with
    /// `y = −π/2` under XYZ.
    #[test]
    fn gimbal_fallback_feeds_the_state() {
        let ry = Matrix3::from_quat(Quaternion::from_rotation_y(FRAC_PI_2));
        let (euler, _) = calculate_angle_info(&Matrix3::IDENTITY, &ry, RotateOrder::Xyz);
        // The quaternion→matrix rounding can land just inside the main
        // branch; the fallback value and the asin agree to ~1e-3 here.
        assert!((euler[1] + FRAC_PI_2).abs() < 1e-3, "y = {}", euler[1]);
    }

    // -- small helpers --------------------------------------------------------

    #[test]
    fn mirrored_window_shapes() {
        assert!(matches!(
            mirrored_window(0.0, Real::INFINITY),
            RowLimits::UpperBounded(_)
        ));
        assert!(matches!(
            mirrored_window(Real::NEG_INFINITY, 0.0),
            RowLimits::LowerBounded(_)
        ));
        assert!(matches!(
            mirrored_window(Real::NEG_INFINITY, Real::INFINITY),
            RowLimits::Unbounded
        ));
        assert!(matches!(
            mirrored_window(-4.0, 4.0),
            RowLimits::Bilateral(f) if f == 4.0
        ));
        assert!(matches!(
            mirrored_window(-3.0, 2.0),
            RowLimits::Window(lo, hi) if lo == -2.0 && hi == 3.0
        ));
    }

    #[test]
    fn angular_axis_violation_gate() {
        let mut m = RotationalLimitMotor2 {
            lower_limit: 0.5,
            upper_limit: 1.0,
            ..Default::default()
        };
        m.test_limit_value(0.0); // violated
        assert!(angular_axis_violated(&m));
        m.test_limit_value(0.7); // inside
        assert!(!angular_axis_violated(&m));
        m.lower_limit = 0.0;
        m.upper_limit = 0.0;
        m.test_limit_value(1e-4); // locked, error below the 1e-3 threshold
        assert!(!angular_axis_violated(&m));
        m.test_limit_value(1e-2); // locked, error above the threshold
        assert!(angular_axis_violated(&m));
    }

    // -- setters ---------------------------------------------------------------

    #[test]
    fn servo_target_wraps_into_pi_range() {
        let mut c = test_constraint(RotateOrder::Xyz);
        c.set_servo_target(3, 7.0);
        assert!((c.angular[0].servo_target - (7.0 - 2.0 * PI)).abs() < 1e-5);
        c.set_servo_target(3, -7.0);
        assert!((c.angular[0].servo_target + (7.0 - 2.0 * PI)).abs() < 1e-5);
        c.set_servo_target(3, 0.3);
        assert!((c.angular[0].servo_target - 0.3).abs() < 1e-6);
        c.set_servo_target(0, 12.0); // linear targets are stored verbatim
        assert_eq!(c.linear.servo_target[0], 12.0);
    }

    #[test]
    fn reversed_limit_setters_mirror_the_convention() {
        let mut c = test_constraint(RotateOrder::Xyz);
        c.set_angular_lower_limit_reversed(Vector3::new(0.5, 0.0, 0.0));
        assert_eq!(c.angular[0].upper_limit, -0.5);
        c.set_angular_upper_limit_reversed(Vector3::new(-0.2, 0.0, 0.0));
        assert_eq!(c.angular[0].lower_limit, 0.2);
        c.set_limit_reversed(4, 0.3, 0.7);
        assert_eq!(c.angular[1].upper_limit, -0.3);
        assert_eq!(c.angular[1].lower_limit, -0.7);
        // Normalization still applies to the reversed values:
        // `normalize(−(π+0.1)) = π−0.1`.
        c.set_limit_reversed(3, PI + 0.1, PI + 0.2);
        assert!((c.angular[0].upper_limit - (PI - 0.1)).abs() < 1e-5);
    }

    // -- world behavior ---------------------------------------------------------

    /// Two 1 kg spheres (radius 0.5) at x = ∓1 m, joined by a spring-2 6DOF
    /// whose pivots coincide at the origin (`frame_a.origin = +X`,
    /// `frame_b.origin = −X`): the linear diff starts at zero. Gravity off,
    /// so the joint is the only actor. `b_quat` pre-rotates body B — its
    /// frame origin is counter-rotated so the pivots still coincide (a
    /// constraint-satisfying start, no Baumgarte kick).
    fn spring2_world(order: RotateOrder) -> (PhysicsWorld, usize, BodyHandle, BodyHandle) {
        spring2_world_rotated(order, Quaternion::IDENTITY)
    }

    fn spring2_world_rotated(
        order: RotateOrder,
        b_quat: Quaternion,
    ) -> (PhysicsWorld, usize, BodyHandle, BodyHandle) {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(
            RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(-1.0, 0.0, 0.0),
                1.0,
            )
            .disable_deactivation(),
        );
        let b = world.add_body(
            RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(1.0, 0.0, 0.0),
                1.0,
            )
            .disable_deactivation(),
        );
        let mut c = Constraint::generic_6dof_spring2(
            a,
            b,
            JointFrame::new(Vector3::new(1.0, 0.0, 0.0), Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(-1.0, 0.0, 0.0), Matrix3::IDENTITY),
            order,
        );
        c.kind_mut()
            .generic_6dof_spring2_mut()
            .frame_b
            .origin = b_quat.inverse() * Vector3::new(-1.0, 0.0, 0.0);
        world.body_mut(b).orientation = b_quat;
        let index = world.add_constraint(c);
        (world, index, a, b)
    }

    fn test_constraint(order: RotateOrder) -> Generic6DofSpring2Constraint {
        Generic6DofSpring2Constraint::new(
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            order,
        )
    }

    /// Bullet's defaults make the joint a ball joint: the locked linear
    /// axes hold the pivots coincident against a velocity kick while the
    /// free angular axes let the relative rotation drift.
    #[test]
    fn default_joint_is_a_ball_joint() {
        let (mut world, _idx, _a, b) = spring2_world(RotateOrder::Xyz);
        world.body_mut(b).linear_velocity = Vector3::new(0.0, 2.0, 0.0);
        world.body_mut(b).angular_velocity = Vector3::new(0.0, 0.0, 3.0);
        for _ in 0..120 {
            world.step_simulation(DT);
        }
        let pa = world.body(_a).position;
        let pb = world.body(b).position;
        let ra = world.body(_a).orientation;
        let rb = world.body(b).orientation;
        // The pivots (frame origins ±X rotated with the bodies) stay coincident.
        let pivot_a = pa + ra * Vector3::new(1.0, 0.0, 0.0);
        let pivot_b = pb + rb * Vector3::new(-1.0, 0.0, 0.0);
        let pivot_err = pivot_b - pivot_a;
        assert!(pivot_err.length() < 0.01, "pivot drift {pivot_err}");
        // The free angular axes let B rotate away from its initial pose.
        let spin = world.body(b).orientation;
        assert!(spin.w < 0.999, "orientation frozen: {spin:?}");
    }

    /// The elastic limit produces no force inside the range and a restoring
    /// pull outside.
    #[test]
    fn limited_axis_is_free_inside_and_restores_outside() {
        // Inside the range: the angle stays put.
        let (mut world, idx, _a, _b) =
            spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_x(0.2)); // angle −0.2
        world
            .constraint_mut(idx)
            .kind_mut()
            .generic_6dof_spring2_mut()
            .set_limit(3, -1.0, 1.0);
        for _ in 0..120 {
            world.step_simulation(DT);
        }
        let c = world.constraint(idx).kind().generic_6dof_spring2();
        assert!(
            (c.get_angle(0) + 0.2).abs() < 0.02,
            "angle inside the range drifted: {}",
            c.get_angle(0)
        );

        // Outside the range: the limit pulls the angle back.
        let (mut world, idx, _a, _b) =
            spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_x(1.5)); // angle −1.5
        world
            .constraint_mut(idx)
            .kind_mut()
            .generic_6dof_spring2_mut()
            .set_limit(3, -1.0, 1.0);
        for _ in 0..60 {
            world.step_simulation(DT);
        }
        let c = world.constraint(idx).kind().generic_6dof_spring2();
        assert!(
            c.get_angle(0) > -1.3,
            "angle outside the range did not restore: {}",
            c.get_angle(0)
        );
    }

    /// A locked angular axis (lo == hi) pulls the relative angle to the
    /// locked value.
    #[test]
    fn locked_axis_pulls_to_the_lock_angle() {
        let (mut world, idx, _a, _b) =
            spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_x(-0.5)); // angle +0.5
        world
            .constraint_mut(idx)
            .kind_mut()
            .generic_6dof_spring2_mut()
            .set_limit(3, 0.3, 0.3);
        for _ in 0..180 {
            world.step_simulation(DT);
        }
        let c = world.constraint(idx).kind().generic_6dof_spring2();
        assert!(
            (c.get_angle(0) - 0.3).abs() < 0.02,
            "locked angle not held: {}",
            c.get_angle(0)
        );
    }

    /// The angular spring oscillates toward the equilibrium and decays.
    #[test]
    fn angular_spring_oscillates_and_decays() {
        let (mut world, idx, _a, _b) =
            spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_y(0.3)); // angle −0.3
        {
            let c = world.constraint_mut(idx).kind_mut().generic_6dof_spring2_mut();
            c.enable_spring(4, true); // angular axis 1 (Y)
            c.set_stiffness(4, 40.0, true);
            c.set_damping(4, 0.5, true);
        }
        let mut crossings = 0;
        let mut prev = world
            .constraint(idx)
            .kind()
            .generic_6dof_spring2()
            .get_angle(1);
        let mut max_angle = prev.abs();
        for step in 0..300 {
            world.step_simulation(DT);
            let angle = world
                .constraint(idx)
                .kind()
                .generic_6dof_spring2()
                .get_angle(1);
            if step % 5 == 0 {
                println!(
                    "f{step}: angle {angle} wb {:?} active {}",
                    world.body(_b).angular_velocity,
                    world.body(_b).is_active()
                );
            }
            if prev.signum() != angle.signum() {
                crossings += 1;
            }
            max_angle = max_angle.max(angle.abs());
            prev = angle;
        }
        assert!(crossings >= 2, "no oscillation (crossings {crossings})");
        assert!(max_angle < 0.35, "spring blew up: {max_angle}");
    }

    /// An absurd stiffness with the auto-clamp stays stable (the spring is
    /// never sampled faster than a quarter of its angular frequency).
    #[test]
    fn stiffness_clamp_keeps_a_huge_spring_stable() {
        let (mut world, idx, _a, _b) =
            spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_y(0.3));
        {
            let c = world.constraint_mut(idx).kind_mut().generic_6dof_spring2_mut();
            c.enable_spring(4, true); // angular axis 1 (Y)
            c.set_stiffness(4, 1.0e6, true);
        }
        let mut max_angle: Real = 0.35;
        for _ in 0..240 {
            world.step_simulation(DT);
            let angle = world
                .constraint(idx)
                .kind()
                .generic_6dof_spring2()
                .get_angle(1);
            assert!(angle.is_finite(), "spring diverged: {angle}");
            max_angle = max_angle.max(angle.abs());
        }
        assert!(max_angle < 0.6, "clamped spring diverged: {max_angle}");
    }

    /// An absurd damping with the auto-clamp is overdamped: the angle
    /// decays to the equilibrium without a single overshoot.
    #[test]
    fn damping_clamp_overdamps_without_overshoot() {
        let (mut world, idx, _a, _b) =
            spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_y(0.3));
        {
            let c = world.constraint_mut(idx).kind_mut().generic_6dof_spring2_mut();
            c.enable_spring(4, true); // angular axis 1 (Y)
            c.set_stiffness(4, 40.0, true);
            c.set_damping(4, 100.0, true); // kd·dt = 1.67 > m → clamped
        }
        let mut crossed = false;
        for _ in 0..300 {
            world.step_simulation(DT);
            let angle = world
                .constraint(idx)
                .kind()
                .generic_6dof_spring2()
                .get_angle(1);
            assert!(angle.is_finite(), "spring diverged: {angle}");
            if angle > 0.0 {
                crossed = true;
            }
        }
        assert!(!crossed, "overdamped spring overshot the equilibrium");
    }

    /// The linear spring drives the pivot offset to the equilibrium point.
    #[test]
    fn linear_spring_pulls_to_the_equilibrium() {
        let (mut world, idx, _a, _b) = spring2_world(RotateOrder::Xyz);
        let c = world.constraint_mut(idx).kind_mut().generic_6dof_spring2_mut();
        // A spring axis needs free limits (a locked axis emits an
        // unlimited equality row that would fight the spring, exactly like
        // upstream).
        c.set_linear_limits(Vector3::ONE, -Vector3::ONE);
        c.enable_spring(0, true);
        c.set_stiffness(0, 5.0, true);
        c.set_damping(0, 0.5, true);
        c.set_equilibrium_point(0, 0.5); // pivot offset of +0.5 along X
        for _ in 0..360 {
            world.step_simulation(DT);
        }
        let c = world.constraint(idx).kind().generic_6dof_spring2();
        assert!(
            (c.get_relative_pivot_position(0) - 0.5).abs() < 0.05,
            "linear spring did not settle: {}",
            c.get_relative_pivot_position(0)
        );
    }

    /// The servo motor drives the relative angle to the servo target and
    /// holds it.
    #[test]
    fn servo_drives_to_the_target() {
        let (mut world, idx, _a, _b) = spring2_world(RotateOrder::Xyz);
        let c = world.constraint_mut(idx).kind_mut().generic_6dof_spring2_mut();
        c.enable_motor(3, true);
        c.set_servo(3, true);
        c.set_servo_target(3, 0.5);
        c.set_target_velocity(3, 2.0);
        c.set_max_motor_force(3, 10.0);
        for _ in 0..300 {
            world.step_simulation(DT);
        }
        let c = world.constraint(idx).kind().generic_6dof_spring2();
        assert!(
            (c.get_angle(0) - 0.5).abs() < 0.05,
            "servo did not reach the target: {}",
            c.get_angle(0)
        );
    }

    /// The global ERP/CFM fallback: an angular axis without the override
    /// flag gets its stop fields *overwritten* with the global solver values
    /// every step (upstream `setAngularLimits` writes them into the struct),
    /// while a `set_stop_erp` override survives.
    #[test]
    fn global_erp_fallback_quirk() {
        {
            let (mut world, idx, _a, _b) =
                spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_x(1.5));
            {
                let c = world.constraint_mut(idx).kind_mut().generic_6dof_spring2_mut();
                c.set_limit(3, -1.0, 1.0); // limited axis produces rows
                c.set_stop_erp(3, 0.3); // flagged axis keeps its value
            }
            world.solver_info_mut().erp = 0.5;
            world.step_simulation(DT);
            let c = world.constraint(idx).kind().generic_6dof_spring2();
            assert_eq!(c.angular[0].stop_erp, 0.3, "override flag lost");
        }
        // Unflagged axis: the global ERP is written into the struct.
        let (mut world, idx, _a, _b) =
            spring2_world_rotated(RotateOrder::Xyz, Quaternion::from_rotation_y(1.5));
        world
            .constraint_mut(idx)
            .kind_mut()
            .generic_6dof_spring2_mut()
            .set_limit(4, -1.0, 1.0);
        world.solver_info_mut().erp = 0.5;
        world.step_simulation(DT);
        let c = world.constraint(idx).kind().generic_6dof_spring2();
        assert_eq!(c.angular[1].stop_erp, 0.5, "global fallback not applied");
    }

    /// The static-body mass split (Bullet's `calculateTransforms` tail):
    /// `fact_a + fact_b == 1`, the static side takes the other side's share
    /// and `has_static_body` is flagged from `invMass < SIMD_EPSILON`.
    #[test]
    fn static_body_mass_split() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(RigidBody::new_static(
            CollisionShape::sphere(0.5),
            Vector3::ZERO,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        ));
        let c = Constraint::generic_6dof_spring2(
            a,
            b,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(1.0, 0.0, 0.0), Matrix3::IDENTITY),
            RotateOrder::Xyz,
        );
        let idx = world.add_constraint(c);
        // Both orthogonal angular axes violated → `rot_allowed` is off for
        // the linear rows, arming the fact scaling.
        {
            let c = world.constraint_mut(idx).kind_mut().generic_6dof_spring2_mut();
            c.set_limit(4, 0.5, 1.0); // axis 1 violated at angle 0
            c.set_limit(5, -1.0, -0.5); // axis 2 violated at angle 0
        }
        world.step_simulation(DT);
        let c = world.constraint(idx).kind().generic_6dof_spring2();
        assert!(c.has_static_body);
        assert!((c.fact_a - 1.0).abs() < 1e-6, "fact_a = {}", c.fact_a);
        assert!((c.fact_b - 0.0).abs() < 1e-6, "fact_b = {}", c.fact_b);
    }
}
