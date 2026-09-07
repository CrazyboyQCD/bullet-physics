//! Hinge (revolute) joint (Bullet's `btHingeConstraint`, classic `getInfo2Internal` formulation).

use super::*;
use super::shared::{get_hinge_angle, get_motor_factor};

/// A hinge motor: rotates the joint at `target_velocity` rad/s around the
/// axis, limited to `max_impulse` per step (Bullet's `enableAngularMotor`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HingeMotor {
    /// Target angular velocity around the hinge axis, rad/s.
    pub target_velocity: Real,
    /// Maximum motor impulse per step (Bullet's `m_maxMotorImpulse`).
    pub max_impulse: Real,
}

/// A revolute (hinge) joint (Bullet's `btHingeConstraint`, using the classic
/// `getInfo2Internal` formulation): three linear pivot rows, two angular rows
/// keeping the hinge axes parallel, and an optional limit/motor row around
/// the axis.
///
/// Deviations from Bullet (see `docs/bullet-deviations.md`): the frames are
/// built with a deterministic perpendicular basis around the given axes (the
/// zero-angle reference is not aligned to body A's initial X axis), and the
/// modern frame-offset formulation of the linear rows is not used.
#[derive(Debug, Clone, PartialEq)]
pub struct HingeConstraint {
    /// Hinge frame on body A (pivot + basis in A's COM frame).
    pub frame_a: JointFrame,
    /// Hinge frame on body B (pivot + basis in B's COM frame).
    pub frame_b: JointFrame,
    /// Skip the three linear pivot rows, keeping only the axis alignment
    /// (Bullet's `setAngularOnly`).
    pub angular_only: bool,
    /// Angular limits around the hinge axis (Bullet's `setLimit`).
    pub limit: Option<AngularLimit>,
    /// Hinge motor (Bullet's `enableAngularMotor`).
    pub motor: Option<HingeMotor>,
    /// Per-step state: the current hinge angle (Bullet's `m_hingeAngle`).
    pub hinge_angle: Real,
    limit_correction: Real,
    limit_sign: Real,
    solve_limit: bool,
}

impl HingeConstraint {
    /// Build the constraint frames from pivots and axes in each body's frame
    /// (Bullet's pivot/axis constructor). The perpendicular basis is chosen
    /// deterministically: `t1` perpendicular to `axis_a`, `t2 = axis_a × t1`,
    /// and frame B's basis is frame A's rotated by the shortest arc from
    /// `axis_a` to `axis_b`.
    pub fn new(
        pivot_a: Vector3,
        axis_a: Vector3,
        pivot_b: Vector3,
        axis_b: Vector3,
    ) -> Self {
        let axis_a = axis_a.normalize();
        let axis_b = axis_b.normalize();
        let t1 = axis_a
            .cross(if axis_a.x.abs() < 0.9 { Vector3::X } else { Vector3::Y })
            .normalize();
        let t2 = axis_a.cross(t1);
        let basis_a = Matrix3::from_cols(t1, t2, axis_a);
        // Rotate the basis so its Z aligns with axis_b (Bullet's
        // `shortestArcQuat`); X follows, Y is rebuilt orthogonal.
        let rotation = quat_from_rotation_arc(axis_a, axis_b);
        let t1_b = rotation * t1;
        let t2_b = axis_b.cross(t1_b);
        let basis_b = Matrix3::from_cols(t1_b, t2_b, axis_b);
        Self {
            frame_a: JointFrame::new(pivot_a, basis_a),
            frame_b: JointFrame::new(pivot_b, basis_b),
            angular_only: false,
            limit: None,
            motor: None,
            hinge_angle: 0.0,
            limit_correction: 0.0,
            limit_sign: 0.0,
            solve_limit: false,
        }
    }

    /// The current hinge angle in radians (Bullet's `getHingeAngle`), updated
    /// by the last solve step.
    pub fn hinge_angle(&self) -> Real {
        self.hinge_angle
    }

    /// Set angular limits around the axis (Bullet's `setLimit`); `low > high`
    /// deactivates them.
    pub fn set_limit(&mut self, low: Real, high: Real) {
        self.limit = Some(AngularLimit::new(low, high));
    }

    /// Enable the hinge motor (Bullet's `enableAngularMotor`).
    pub fn enable_motor(&mut self, target_velocity: Real, max_impulse: Real) {
        self.motor = Some(HingeMotor {
            target_velocity,
            max_impulse,
        });
    }

    /// Disable the hinge motor.
    pub fn disable_motor(&mut self) {
        self.motor = None;
    }

    /// Flatten this hinge into its solver rows (Bullet's `getInfo2Internal`).
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
        // World-space frames (Bullet's `trA = transA · m_rbAFrame`).
        let tr_a = Matrix3::from_quat(a.orientation) * self.frame_a.basis;
        let tr_b = Matrix3::from_quat(b.orientation) * self.frame_b.basis;
        let pivot_a = a.position + a.orientation * self.frame_a.origin;
        let pivot_b = b.position + b.orientation * self.frame_b.origin;
        let r_a = pivot_a - a.position;
        let r_b = pivot_b - b.position;
        // Hinge axis (Z) and its perpendiculars (X, Y) of frame A.
        let ax1 = tr_a.col(2);
        let p = tr_a.col(0);
        let q = tr_a.col(1);
        let ax2 = tr_b.col(2);
        let k = inv_dt * erp;

        // Linear pivot rows (Bullet's `m_J1linearAxis` = ±e_j), unless the
        // joint is angular-only.
        if !self.angular_only {
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
        }

        // Two angular rows keeping the hinge axes parallel: drive the
        // relative spin around p and q to zero, with a bias proportional to
        // the misalignment u = ax1 × ax2 (Bullet's rows 3-4).
        let u = ax1.cross(ax2);
        out.push(ConstraintRow::prepare(
            body_a,
            body_b,
            Vector3::ZERO,
            p,
            Vector3::ZERO,
            p,
            a,
            b,
            -k * u.dot(p),
            RowLimits::Unbounded,
        ));
        out.push(ConstraintRow::prepare(
            body_a,
            body_b,
            Vector3::ZERO,
            q,
            Vector3::ZERO,
            q,
            a,
            b,
            -k * u.dot(q),
            RowLimits::Unbounded,
        ));

        // The limit/motor row around the hinge axis (Bullet's row 5).
        self.hinge_angle = get_hinge_angle(&tr_a, &tr_b);
        let (correction, sign, solve) = self
            .limit
            .as_ref()
            .map(|l| l.test(self.hinge_angle))
            .unwrap_or((0.0, 0.0, false));
        self.limit_correction = correction;
        self.limit_sign = sign;
        self.solve_limit = solve;

        let powered = self.motor.is_some();
        if solve || powered {
            // Bullet's row error (in its `v = (ωA −ωB)·axis` convention);
            // our mirrored row negates it.
            let mut row_error: Real = 0.0;
            let lostop = self.limit.map(|l| l.low).unwrap_or(1.0);
            let histop = self.limit.map(|l| l.high).unwrap_or(-1.0);
            let mut motor_ineffective = false;
            if solve && lostop == histop {
                motor_ineffective = true;
            }
            if powered && !motor_ineffective {
                let motor = self.motor.as_ref().unwrap();
                let mot_fact = get_motor_factor(
                    self.hinge_angle,
                    lostop,
                    histop,
                    motor.target_velocity,
                    inv_dt * erp,
                );
                row_error += mot_fact * motor.target_velocity;
            }
            if solve {
                row_error += k * correction;
                // Bounce: only for incoming velocities, and only if it
                // exceeds the current error (Bullet's bounce block).
                let bounce = self.limit.map(|l| l.relaxation_factor).unwrap_or(1.0);
                if bounce > 0.0 {
                    let vel = a.angular_velocity.dot(ax1) - b.angular_velocity.dot(ax1);
                    if sign > 0.0 {
                        if vel < 0.0 {
                            let newc = -bounce * vel;
                            if newc > row_error {
                                row_error = newc;
                            }
                        }
                    } else if vel > 0.0 {
                        let newc = -bounce * vel;
                        if newc < row_error {
                            row_error = newc;
                        }
                    }
                }
                row_error *= self.limit.map(|l| l.bias_factor).unwrap_or(1.0);
            }

            let limits = if solve {
                if lostop == histop {
                    RowLimits::Unbounded
                } else if sign > 0.0 {
                    // Low limit: mirrored clamp of Bullet's [0, +inf].
                    RowLimits::UpperBounded(0.0)
                } else {
                    RowLimits::LowerBounded(0.0)
                }
            } else {
                RowLimits::Bilateral(self.motor.map(|m| m.max_impulse).unwrap_or(0.0))
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
                -row_error,
                limits,
            ));
        }
    }
}

