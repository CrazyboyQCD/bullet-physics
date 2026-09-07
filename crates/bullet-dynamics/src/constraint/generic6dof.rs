//! Generic six-degree-of-freedom joint (Bullet's `btGeneric6DofConstraint`, classic `getInfo1`/`getInfo2` recipe).

use super::*;
use super::shared::{adjust_angle_to_limits, get_motor_factor, mirrored_row_limits, normalize_angle, relative_rotation_info};

/// A generic six-degree-of-freedom joint (Bullet's
/// `btGeneric6DofConstraint`): three linear axes (the frame A X/Y/Z
/// directions) and three angular axes (the relative-rotation axes of
/// [`relative_rotation_info`]), each with independent limits and an
/// optional motor.
///
/// Bullet's defaults are kept: all three linear limits are locked at zero
/// and all three angular limits are free (`1.0 > −1.0`), so a fresh joint
/// behaves like a ball joint (the frame origins stay coincident, rotation
/// is free). A locked linear/angular axis produces an equality row
/// (`lo == hi`); a free axis (`lo > hi`) produces nothing. The angular
/// rows use the classic Euler-angle recipe (`calculateAngleInfo` /
/// `get_limit_motor_info2`); the linear rows use the point-to-point-style
/// lever arms instead of Bullet's frame-offset formulation (see
/// `docs/bullet-deviations.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct Generic6DofConstraint {
    /// Constraint frame on body A (pivot + basis in A's COM frame).
    pub frame_a: JointFrame,
    /// Constraint frame on body B (pivot + basis in B's COM frame).
    pub frame_b: JointFrame,
    /// The three linear limit/motor axes (Bullet's
    /// `btTranslationalLimitMotor`).
    pub linear: TranslationalLimitMotor,
    /// The three angular limit/motor axes (Bullet's
    /// `btRotationalLimitMotor` array).
    pub angular: [RotationalLimitMotor; 3],
    /// Per-step state: the pivot offset in A's frame (Bullet's
    /// `m_calculatedLinearDiff`), the relative Euler angles (Bullet's
    /// `m_calculatedAxisAngleDiff`) and the world angular constraint axes.
    pub current_linear_diff: Vector3,
    pub euler_angles: Vector3,
    pub angular_axes: [Vector3; 3],
}

/// The linear limit/motor parameters of one axis triplet (Bullet's
/// `btTranslationalLimitMotor`). The parameters used by the solver rows are
/// kept; the obsolete-solver fields (`limitSoftness`, `damping`,
/// `restitution`, CFM) are omitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranslationalLimitMotor {
    /// Lower limits along each axis (Bullet's `m_lowerLimit`).
    pub lower_limit: Vector3,
    /// Upper limits along each axis (Bullet's `m_upperLimit`).
    pub upper_limit: Vector3,
    /// Motor target velocities along each axis (Bullet's
    /// `m_targetVelocity`).
    pub target_velocity: Vector3,
    /// Motor max forces along each axis, N (Bullet's `m_maxMotorForce`).
    pub max_motor_force: Vector3,
    /// Whether the motor on each axis is enabled (Bullet's
    /// `m_enableMotor`).
    pub enable_motor: [bool; 3],
    /// Per-axis positional-correction factor at the limit (Bullet's
    /// `m_stopERP`, default 0.2).
    pub stop_erp: Vector3,
    /// Per-step limit state (Bullet's `m_currentLimit`:
    /// 0 = free, 1 = upper limit, 2 = lower limit) and the signed
    /// violation (Bullet's `m_currentLimitError`).
    pub current_limit: [i32; 3],
    pub current_limit_error: Vector3,
}

impl Default for TranslationalLimitMotor {
    fn default() -> Self {
        Self {
            lower_limit: Vector3::ZERO,
            upper_limit: Vector3::ZERO,
            target_velocity: Vector3::ZERO,
            max_motor_force: Vector3::ZERO,
            enable_motor: [false; 3],
            stop_erp: Vector3::splat(0.2),
            current_limit: [0; 3],
            current_limit_error: Vector3::ZERO,
        }
    }
}

/// The angular limit/motor parameters of one rotation axis (Bullet's
/// `btRotationalLimitMotor`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationalLimitMotor {
    /// Lower limit, radians (Bullet's `m_loLimit`; default 1.0 = free).
    pub lower_limit: Real,
    /// Upper limit, radians (Bullet's `m_hiLimit`; default −1.0 = free).
    pub upper_limit: Real,
    /// Motor target velocity, rad/s (Bullet's `m_targetVelocity`).
    pub target_velocity: Real,
    /// Motor max force, N·m (Bullet's `m_maxMotorForce`, default 6.0).
    pub max_motor_force: Real,
    /// Whether the motor is enabled (Bullet's `m_enableMotor`).
    pub enable_motor: bool,
    /// Positional-correction factor at the limit (Bullet's `m_stopERP`,
    /// default 0.2).
    pub stop_erp: Real,
    /// Bounce restitution at the limit (Bullet's `m_bounce`, default 0).
    pub bounce: Real,
    /// Per-step limit state (Bullet's `m_currentLimit`: 0 = free,
    /// 1 = lower limit, 2 = upper limit), the signed violation, and the
    /// current (adjusted) angle.
    pub current_limit: i32,
    pub current_limit_error: Real,
    pub current_position: Real,
}

impl Default for RotationalLimitMotor {
    fn default() -> Self {
        Self {
            lower_limit: 1.0,
            upper_limit: -1.0,
            target_velocity: 0.0,
            max_motor_force: 6.0,
            enable_motor: false,
            stop_erp: 0.2,
            bounce: 0.0,
            current_limit: 0,
            current_limit_error: 0.0,
            current_position: 0.0,
        }
    }
}

impl Generic6DofConstraint {
    /// A generic 6DOF joint with Bullet's defaults (linear locked, angular
    /// free).
    pub fn new(frame_a: JointFrame, frame_b: JointFrame) -> Self {
        Self {
            frame_a,
            frame_b,
            linear: TranslationalLimitMotor::default(),
            angular: [
                RotationalLimitMotor::default(),
                RotationalLimitMotor::default(),
                RotationalLimitMotor::default(),
            ],
            current_linear_diff: Vector3::ZERO,
            euler_angles: Vector3::ZERO,
            angular_axes: [Vector3::ZERO; 3],
        }
    }

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
            let i = axis - 3;
            self.angular[i].lower_limit = normalize_angle(lo);
            self.angular[i].upper_limit = normalize_angle(hi);
        }
    }

    /// Enable the motor on `axis` (0-2 linear, 3-5 angular) at
    /// `target_velocity` with `max_force` (Bullet's `getLimitMotor` +
    /// `m_enableMotor`).
    pub fn enable_motor(&mut self, axis: usize, target_velocity: Real, max_force: Real) {
        if axis < 3 {
            self.linear.enable_motor[axis] = true;
            self.linear.target_velocity[axis] = target_velocity;
            self.linear.max_motor_force[axis] = max_force;
        } else {
            let i = axis - 3;
            self.angular[i].enable_motor = true;
            self.angular[i].target_velocity = target_velocity;
            self.angular[i].max_motor_force = max_force;
        }
    }

    /// Disable the motor on `axis`.
    pub fn disable_motor(&mut self, axis: usize) {
        if axis < 3 {
            self.linear.enable_motor[axis] = false;
        } else {
            self.angular[axis - 3].enable_motor = false;
        }
    }

    /// The current relative Euler angle of `axis` (0-2; Bullet's
    /// `getAngle`), updated by the last solve step.
    pub fn get_angle(&self, axis: usize) -> Real {
        self.euler_angles[axis]
    }

    /// The current pivot offset along `axis` (0-2, in A's frame; Bullet's
    /// `getRelativePivotPosition`), updated by the last solve step.
    pub fn get_relative_pivot_position(&self, axis: usize) -> Real {
        self.current_linear_diff[axis]
    }

    /// Test the linear limit on `axis` (Bullet's
    /// `btTranslationalLimitMotor::testLimitValue`): 0 = free, 2 = below
    /// the lower limit, 1 = above the upper limit.
    pub fn test_linear_limit(&mut self, axis: usize, test_value: Real) -> i32 {
        let lo = self.linear.lower_limit[axis];
        let hi = self.linear.upper_limit[axis];
        if lo > hi {
            self.linear.current_limit[axis] = 0;
            self.linear.current_limit_error[axis] = 0.0;
            return 0;
        }
        if test_value < lo {
            self.linear.current_limit[axis] = 2;
            self.linear.current_limit_error[axis] = test_value - lo;
            return 2;
        }
        if test_value > hi {
            self.linear.current_limit[axis] = 1;
            self.linear.current_limit_error[axis] = test_value - hi;
            return 1;
        }
        self.linear.current_limit[axis] = 0;
        self.linear.current_limit_error[axis] = 0.0;
        0
    }

    /// Test the rotational limit on `axis` (Bullet's
    /// `testAngularLimitMotor`): updates the current position/error and
    /// returns whether a row is needed.
    pub fn test_angular_limit_motor(&mut self, axis: usize) -> bool {
        let angle = self.euler_angles[axis];
        let angle = adjust_angle_to_limits(
            angle,
            self.angular[axis].lower_limit,
            self.angular[axis].upper_limit,
        );
        self.angular[axis].current_position = angle;
        let m = &mut self.angular[axis];
        if m.lower_limit > m.upper_limit {
            m.current_limit = 0;
            return m.enable_motor;
        }
        if angle < m.lower_limit {
            m.current_limit = 1;
            m.current_limit_error = angle - m.lower_limit;
            if m.current_limit_error > bullet_math::PI {
                m.current_limit_error -= 2.0 * bullet_math::PI;
            } else if m.current_limit_error < -bullet_math::PI {
                m.current_limit_error += 2.0 * bullet_math::PI;
            }
        } else if angle > m.upper_limit {
            m.current_limit = 2;
            m.current_limit_error = angle - m.upper_limit;
            if m.current_limit_error > bullet_math::PI {
                m.current_limit_error -= 2.0 * bullet_math::PI;
            } else if m.current_limit_error < -bullet_math::PI {
                m.current_limit_error += 2.0 * bullet_math::PI;
            }
        } else {
            m.current_limit = 0;
        }
        m.current_limit != 0 || m.enable_motor
    }

    /// Flatten this 6DOF joint into its solver rows: the angular rows
    /// first, then the linear rows (Bullet solves angular limits first
    /// when the frame-offset formulation is on, for stability).
    pub(crate) fn prepare_rows(
        &mut self,
        body_a: usize,
        body_b: usize,
        bodies: &[RigidBody],
        inv_dt: Real,
        erp: Real,
        out: &mut Vec<ConstraintRow>,
    ) {
        // The global ERP is unused: the 6DOF rows use the per-axis
        // `stop_erp` (0.2 by default, like Bullet's `m_stopERP`).
        let _ = erp;
        let a = &bodies[body_a];
        let b = &bodies[body_b];
        let tr_a = Matrix3::from_quat(a.orientation) * self.frame_a.basis;
        let tr_b = Matrix3::from_quat(b.orientation) * self.frame_b.basis;
        let pivot_a = a.position + a.orientation * self.frame_a.origin;
        let pivot_b = b.position + b.orientation * self.frame_b.origin;
        let r_a = pivot_a - a.position;
        let r_b = pivot_b - b.position;

        // Linear info: the pivot offset expressed in A's frame (Bullet's
        // `calculateLinearInfo`), tested against the limits.
        let diff = tr_a.transpose() * (pivot_b - pivot_a);
        self.current_linear_diff = diff;
        for i in 0..3 {
            self.test_linear_limit(i, diff[i]);
        }

        // Angle info: the relative Euler angles and the angular constraint
        // axes (Bullet's `calculateAngleInfo`).
        let (euler, axes) = relative_rotation_info(&tr_a, &tr_b);
        self.euler_angles = euler;
        self.angular_axes = axes;

        // Angular rows (Bullet's `setAngularLimits`, rotational = 1).
        for (i, &axis) in axes.iter().enumerate() {
            if self.test_angular_limit_motor(i) {
                let m = self.angular[i];
                // Bullet's bounce velocity: `angVelA·ax1 − angVelB·ax1`.
                let row_vel = a.angular_velocity.dot(axis) - b.angular_velocity.dot(axis);
                self.push_limit_motor_row(
                    body_a,
                    body_b,
                    a,
                    b,
                    inv_dt,
                    axis,
                    false,
                    Vector3::ZERO,
                    Vector3::ZERO,
                    m.current_limit,
                    m.current_limit_error,
                    m.lower_limit,
                    m.upper_limit,
                    m.stop_erp,
                    m.enable_motor,
                    m.target_velocity,
                    m.max_motor_force,
                    m.current_position,
                    m.bounce,
                    row_vel,
                    out,
                );
            }
        }

        // Linear rows (Bullet's `setLinearLimits`, rotational = 0). The
        // axis is frame A's column i; the angular jacobians use the
        // point-to-point lever arms (deviation: Bullet's frame-offset
        // formulation is not used).
        for i in 0..3 {
            if self.linear.current_limit[i] != 0 || self.linear.enable_motor[i] {
                let ax1 = tr_a.col(i);
                // Bullet's bounce velocity: `linVelA·ax1 − linVelB·ax1`.
                let row_vel = a.linear_velocity.dot(ax1) - b.linear_velocity.dot(ax1);
                self.push_limit_motor_row(
                    body_a,
                    body_b,
                    a,
                    b,
                    inv_dt,
                    ax1,
                    true,
                    r_a,
                    r_b,
                    self.linear.current_limit[i],
                    self.linear.current_limit_error[i],
                    self.linear.lower_limit[i],
                    self.linear.upper_limit[i],
                    self.linear.stop_erp[i],
                    self.linear.enable_motor[i],
                    self.linear.target_velocity[i],
                    self.linear.max_motor_force[i],
                    self.current_linear_diff[i],
                    0.0,
                    row_vel,
                    out,
                );
            }
        }
    }

    /// One limit/motor row (Bullet's `get_limit_motor_info2`): builds the
    /// constraint error (Bullet's `m_constraintError`), the impulse window
    /// and the bounce, then pushes the mirrored solver row. `linear`
    /// selects the linear vs rotational error signs and the tag-velocity
    /// flip; `row_vel` is Bullet's `J·v = axis·(velA − velB)` used by the
    /// bounce test.
    #[allow(clippy::too_many_arguments)]
    fn push_limit_motor_row(
        &self,
        body_a: usize,
        body_b: usize,
        a: &RigidBody,
        b: &RigidBody,
        inv_dt: Real,
        ax1: Vector3,
        linear: bool,
        lin_r_a: Vector3,
        lin_r_b: Vector3,
        limit: i32,
        limit_err: Real,
        lo: Real,
        hi: Real,
        stop_erp: Real,
        powered: bool,
        target_velocity: Real,
        max_motor_force: Real,
        current_pos: Real,
        bounce: Real,
        row_vel: Real,
        out: &mut Vec<ConstraintRow>,
    ) {
        let mut powered = powered;
        // Locked axes make the motor ineffective (Bullet's `if (limit &&
        // (loLimit == hiLimit)) powered = false`).
        if limit != 0 && lo == hi {
            powered = false;
        }
        let mut row_error: Real = 0.0;
        if powered && limit == 0 {
            // The motor fades near the limits via `getMotorFactor`; the
            // tag velocity flips sign for the linear axes (Bullet's
            // `tag_vel = rotational ? target : -target`).
            let tag_vel = if linear { -target_velocity } else { target_velocity };
            let mot_fact = get_motor_factor(current_pos, lo, hi, tag_vel, inv_dt * stop_erp);
            row_error += mot_fact * target_velocity;
        }
        if limit != 0 {
            let k = inv_dt * stop_erp;
            // Linear rows add `+k·err`, rotational rows `−k·err` (Bullet's
            // `get_limit_motor_info2` branches).
            row_error += if linear { k * limit_err } else { -k * limit_err };
            // Bounce: only for incoming velocities, replacing the error
            // when it exceeds what we already have.
            if bounce > 0.0 {
                if limit == 1 {
                    if row_vel < 0.0 {
                        let newc = -bounce * row_vel;
                        if newc > row_error {
                            row_error = newc;
                        }
                    }
                } else if row_vel > 0.0 {
                    let newc = -bounce * row_vel;
                    if newc < row_error {
                        row_error = newc;
                    }
                }
            }
        }
        // The Bullet-side impulse window, mirrored through our row
        // convention (our λ = −Bullet's λ).
        let (lower, upper) = if powered && limit == 0 {
            (-max_motor_force / inv_dt, max_motor_force / inv_dt)
        } else if limit != 0 {
            if lo == hi {
                (-Real::INFINITY, Real::INFINITY)
            } else if limit == 1 {
                // Rotational low / linear high: `[0, +inf]` in Bullet.
                (0.0, Real::INFINITY)
            } else {
                (-Real::INFINITY, 0.0)
            }
        } else {
            (0.0, 0.0)
        };
        let limits = mirrored_row_limits(lower, upper);
        if linear {
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                ax1,
                lin_r_a.cross(ax1),
                ax1,
                lin_r_b.cross(ax1),
                a,
                b,
                -row_error,
                limits,
            ));
        } else {
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                Vector3::ZERO,
                ax1,
                Vector3::ZERO,
                ax1,
                a,
                b,
                -row_error,
                limits,
            ));
        }
    }
}

