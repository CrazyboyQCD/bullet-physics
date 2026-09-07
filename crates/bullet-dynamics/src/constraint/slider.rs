//! Slider (prismatic) joint (Bullet's `btSliderConstraint`, frame-offset formulation).

use super::*;
use super::shared::{adjust_angle_to_limits, get_motor_factor, mirrored_row_limits};

/// A slider (prismatic) joint (Bullet's `btSliderConstraint`): the bodies
/// slide along the frame X axes, with two angular rows keeping the axes
/// parallel, two linear rows keeping the pivots aligned on the axis, and
/// optional linear/angular limits and motors.
///
/// Bullet's defaults are kept: the linear limits are free
/// (`lower 1.0 > upper −1.0`) and the angular limits are locked at zero, so
/// a freshly created slider only allows translation along the axis. The
/// frame-offset formulation (`USE_OFFSET_FOR_CONSTANT_FRAME`) is ported:
/// the slider axis is the mass-weighted blend of the two bodies' X axes and
/// the linear rows use the offset-corrected lever arms. CFM is not
/// supported (Bullet's defaults are all zero).
#[derive(Debug, Clone, PartialEq)]
pub struct SliderConstraint {
    /// Constraint frame on body A (pivot + basis in A's COM frame).
    pub frame_a: JointFrame,
    /// Constraint frame on body B (pivot + basis in B's COM frame).
    pub frame_b: JointFrame,
    /// Linear lower limit along the slider axis (Bullet's
    /// `m_lowerLinLimit`; default 1.0 = free).
    pub lower_lin_limit: Real,
    /// Linear upper limit along the slider axis (Bullet's
    /// `m_upperLinLimit`; default −1.0 = free).
    pub upper_lin_limit: Real,
    /// Angular lower limit around the slider axis (Bullet's
    /// `m_lowerAngLimit`; default 0.0 = locked).
    pub lower_ang_limit: Real,
    /// Angular upper limit around the slider axis (Bullet's
    /// `m_upperAngLimit`; default 0.0 = locked).
    pub upper_ang_limit: Real,
    /// Linear limit softness (Bullet's `m_softnessLimLin`, default 1.0).
    pub softness_lim_lin: Real,
    /// Angular limit softness (Bullet's `m_softnessLimAng`, default 1.0).
    pub softness_lim_ang: Real,
    /// Linear limit damping, used only for the bounce coefficient
    /// `|1 − damping|` (Bullet's `m_dampingLimLin`, default 1.0).
    pub damping_lim_lin: Real,
    /// Angular limit damping, used only for the bounce coefficient
    /// `|1 − damping|` (Bullet's `m_dampingLimAng`, default 1.0).
    pub damping_lim_ang: Real,
    /// Orthogonal-linear ERP scale (Bullet's `m_softnessOrthoLin`; the row
    /// ERP is `softness_ortho_lin · erp`, default 1.0).
    pub softness_ortho_lin: Real,
    /// Orthogonal-angular ERP scale (Bullet's `m_softnessOrthoAng`).
    pub softness_ortho_ang: Real,
    /// Linear motor: target velocity and max force (Bullet's
    /// `setPoweredLinMotor`).
    pub lin_motor: Option<SliderMotor>,
    /// Angular motor: target velocity and max force (Bullet's
    /// `setPoweredAngMotor`).
    pub ang_motor: Option<SliderMotor>,
    /// Whether the linear reference frame is A's (Bullet's
    /// `m_useLinearReferenceFrameA`; default true).
    pub use_linear_reference_frame_a: bool,
    /// Per-step state: the linear position along the axis (Bullet's
    /// `getLinearPos`), the linear depth past the limit, and the angular
    /// position/depth (Bullet's `getAngularPos`/`getAngDepth`).
    pub lin_pos: Real,
    pub ang_pos: Real,
    pub lin_depth: Real,
    pub ang_depth: Real,
    /// Whether the linear/angular limits were active in the last solve step
    /// (Bullet's `getSolveLinLimit`/`getSolveAngLimit`).
    pub solve_lin_limit: bool,
    pub solve_ang_limit: bool,
}

/// A slider motor: drives the joint at `target_velocity` along/around the
/// axis, limited to `max_force` (Bullet's `setPoweredLinMotor` /
/// `setPoweredAngMotor`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliderMotor {
    /// Target velocity (m/s for the linear motor, rad/s for the angular).
    pub target_velocity: Real,
    /// Maximum force (N for the linear motor, N·m for the angular).
    pub max_force: Real,
}

impl SliderConstraint {
    /// A slider joint with Bullet's defaults: free linear travel, locked
    /// rotation.
    pub fn new(frame_a: JointFrame, frame_b: JointFrame) -> Self {
        Self {
            frame_a,
            frame_b,
            lower_lin_limit: 1.0,
            upper_lin_limit: -1.0,
            lower_ang_limit: 0.0,
            upper_ang_limit: 0.0,
            softness_lim_lin: 1.0,
            softness_lim_ang: 1.0,
            damping_lim_lin: 1.0,
            damping_lim_ang: 1.0,
            softness_ortho_lin: 1.0,
            softness_ortho_ang: 1.0,
            lin_motor: None,
            ang_motor: None,
            use_linear_reference_frame_a: true,
            lin_pos: 0.0,
            ang_pos: 0.0,
            lin_depth: 0.0,
            ang_depth: 0.0,
            solve_lin_limit: false,
            solve_ang_limit: false,
        }
    }

    /// Set the linear limits along the slider axis (Bullet's
    /// `setLowerLinLimit`/`setUpperLinLimit`); `lower > upper` frees the
    /// axis.
    pub fn set_lin_limits(&mut self, lower: Real, upper: Real) {
        self.lower_lin_limit = lower;
        self.upper_lin_limit = upper;
    }

    /// Set the angular limits around the slider axis (Bullet's
    /// `setLowerAngLimit`/`setUpperAngLimit`); `lower > upper` frees the
    /// axis, `lower == upper` locks it.
    pub fn set_ang_limits(&mut self, lower: Real, upper: Real) {
        self.lower_ang_limit = lower;
        self.upper_ang_limit = upper;
    }

    /// Enable the linear motor along the slider axis (Bullet's
    /// `setPoweredLinMotor`).
    pub fn enable_lin_motor(&mut self, target_velocity: Real, max_force: Real) {
        self.lin_motor = Some(SliderMotor {
            target_velocity,
            max_force,
        });
    }

    /// Disable the linear motor.
    pub fn disable_lin_motor(&mut self) {
        self.lin_motor = None;
    }

    /// Enable the angular motor around the slider axis (Bullet's
    /// `setPoweredAngMotor`).
    pub fn enable_ang_motor(&mut self, target_velocity: Real, max_force: Real) {
        self.ang_motor = Some(SliderMotor {
            target_velocity,
            max_force,
        });
    }

    /// Disable the angular motor.
    pub fn disable_ang_motor(&mut self) {
        self.ang_motor = None;
    }

    /// Flatten this slider into its solver rows (Bullet's
    /// `getInfo2NonVirtual` with the frame-offset formulation).
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
        // World frames and pivots (Bullet's `calculateTransforms`).
        let tr_a = Matrix3::from_quat(a.orientation) * self.frame_a.basis;
        let tr_b = Matrix3::from_quat(b.orientation) * self.frame_b.basis;
        let pivot_a = a.position + a.orientation * self.frame_a.origin;
        let pivot_b = b.position + b.orientation * self.frame_b.origin;
        let delta = pivot_b - pivot_a;
        let ax1_a = tr_a.col(0);
        // Per-axis offsets (m_depth) and limit tests.
        let mut depth: [Real; 3] = core::array::from_fn(|i| delta.dot(tr_a.col(i)));
        self.lin_pos = depth[0];
        self.solve_lin_limit = false;
        if self.lower_lin_limit <= self.upper_lin_limit {
            if depth[0] > self.upper_lin_limit {
                depth[0] -= self.upper_lin_limit;
                self.solve_lin_limit = true;
            } else if depth[0] < self.lower_lin_limit {
                depth[0] -= self.lower_lin_limit;
                self.solve_lin_limit = true;
            } else {
                depth[0] = 0.0;
            }
        } else {
            depth[0] = 0.0;
        }
        self.lin_depth = depth[0];

        self.solve_ang_limit = false;
        self.ang_depth = 0.0;
        if self.lower_ang_limit <= self.upper_ang_limit {
            let axis_a0 = tr_a.col(1);
            let axis_a1 = tr_a.col(2);
            let axis_b0 = tr_b.col(1);
            let rot = adjust_angle_to_limits(
                math::atan2(axis_b0.dot(axis_a1), axis_b0.dot(axis_a0)),
                self.lower_ang_limit,
                self.upper_ang_limit,
            );
            self.ang_pos = rot;
            if rot < self.lower_ang_limit {
                self.ang_depth = rot - self.lower_ang_limit;
                self.solve_ang_limit = true;
            } else if rot > self.upper_ang_limit {
                self.ang_depth = rot - self.upper_ang_limit;
                self.solve_ang_limit = true;
            }
        }

        let sign_fact = if self.use_linear_reference_frame_a { 1.0 } else { -1.0 };
        let ofs = pivot_b - pivot_a;
        let inv_mass_a = a.inv_mass();
        let inv_mass_b = b.inv_mass();
        let has_static = inv_mass_a < Real::EPSILON || inv_mass_b < Real::EPSILON;
        let inv_mass_sum = inv_mass_a + inv_mass_b;
        let fact_a = if inv_mass_sum > 0.0 {
            inv_mass_b / inv_mass_sum
        } else {
            0.5
        };
        let fact_b = 1.0 - fact_a;

        // The slider axis: mass-weighted blend of the two X axes.
        let ax1 = (ax1_a * fact_a + tr_b.col(0) * fact_b).normalize();

        // Rows 0-1: angular orthos keeping the axes parallel. The bias
        // rotates both bodies along u = ax1A x ax1B so the axes realign
        // (Bullet's rows 0-1, ERP scaled by `softnessOrthoAng · erp`).
        let curr_erp_ang = self.softness_ortho_ang * erp;
        let k = inv_dt * curr_erp_ang;
        let u = ax1_a.cross(tr_b.col(0));
        // Offset-corrected lever arms and the average orthos to the slider
        // axis (Bullet's frame-offset block).
        let ortho_a = {
            let rel = pivot_a - a.position;
            rel - ax1 * rel.dot(ax1)
        };
        let ortho_b = {
            let rel = pivot_b - b.position;
            rel - ax1 * rel.dot(ax1)
        };
        let slider_offs = self.lin_pos - depth[0];
        let total_dist = ax1 * (ax1.dot(pivot_a - a.position) + slider_offs)
            - ax1 * ax1.dot(pivot_b - b.position);
        let rel_a = ortho_a + total_dist * fact_a;
        let rel_b = ortho_b - total_dist * fact_b;
        // The average ortho to the slider axis; q completes the basis.
        let p = {
            let avg = ortho_b * fact_a + ortho_a * fact_b;
            if avg.length_squared() > Real::EPSILON {
                avg.normalize()
            } else {
                tr_a.col(1)
            }
        };
        let q = ax1.cross(p);

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
                -k * u.dot(dir),
                RowLimits::Unbounded,
            ));
        }

        // Rows 2-3: linear orthos keeping the pivots aligned on the axis
        // (Bullet's rows 2-3 with the offset-corrected lever arms).
        let curr_erp_lin = self.softness_ortho_lin * erp;
        let k_lin = inv_dt * curr_erp_lin;
        for (i, dir) in [p, q].into_iter().enumerate() {
            let (tmp_a, tmp_b) = if i == 0 {
                (rel_a.cross(p), rel_b.cross(p))
            } else {
                let mut tmp_a = rel_a.cross(q);
                let mut tmp_b = rel_b.cross(q);
                if has_static && self.solve_ang_limit {
                    // Make the constraint between static and dynamic
                    // objects more rigid when an angular limit is hit.
                    tmp_a *= fact_a;
                    tmp_b *= fact_b;
                }
                (tmp_a, tmp_b)
            };
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                dir,
                tmp_a,
                dir,
                tmp_b,
                a,
                b,
                -k_lin * dir.dot(ofs),
                RowLimits::Unbounded,
            ));
        }

        // Linear limit/motor row along the slider axis.
        let mut limit_err: Real = 0.0;
        let mut limit: i32 = 0;
        if self.solve_lin_limit {
            limit_err = self.lin_depth * sign_fact;
            limit = if limit_err > 0.0 { 2 } else { 1 };
        }
        let mut powered = self.lin_motor.is_some();
        if limit != 0 && self.lower_lin_limit == self.upper_lin_limit {
            powered = false;
        }
        if limit != 0 || powered {
            let curr_erp = erp; // Bullet's `m_softnessLimLin` flag path only
            let mut row_error: Real = 0.0;
            let mut lower: Real = 0.0;
            let mut upper: Real = 0.0;
            if powered {
                let motor = self.lin_motor.unwrap();
                let mot_fact = get_motor_factor(
                    self.lin_pos,
                    self.lower_lin_limit,
                    self.upper_lin_limit,
                    motor.target_velocity,
                    inv_dt * curr_erp,
                );
                row_error -= sign_fact * mot_fact * motor.target_velocity;
                lower += -motor.max_force / inv_dt;
                upper += motor.max_force / inv_dt;
            }
            if limit != 0 {
                row_error += inv_dt * curr_erp * limit_err;
                if self.lower_lin_limit == self.upper_lin_limit {
                    lower = -Real::INFINITY;
                    upper = Real::INFINITY;
                } else if limit == 1 {
                    // Low limit.
                    lower = -Real::INFINITY;
                    upper = 0.0;
                } else {
                    // High limit.
                    lower = 0.0;
                    upper = Real::INFINITY;
                }
                // Bounce (Bullet uses `|1 − dampingLimLin|`).
                let bounce = (1.0 - self.damping_lim_lin).abs();
                if bounce > 0.0 {
                    let mut vel = a.linear_velocity.dot(ax1) - b.linear_velocity.dot(ax1);
                    vel *= sign_fact;
                    if limit == 1 {
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
                row_error *= self.softness_lim_lin;
            }
            let limits = mirrored_row_limits(lower, upper);
            let (ang_a, ang_b) = if has_static {
                (Vector3::ZERO, Vector3::ZERO)
            } else {
                (rel_a.cross(ax1), rel_b.cross(ax1))
            };
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                ax1,
                ang_a,
                ax1,
                ang_b,
                a,
                b,
                -row_error,
                limits,
            ));
        }

        // Angular limit/motor row around the slider axis.
        limit_err = 0.0;
        limit = 0;
        if self.solve_ang_limit {
            limit_err = self.ang_depth;
            limit = if limit_err > 0.0 { 1 } else { 2 };
        }
        powered = self.ang_motor.is_some();
        if limit != 0 && self.lower_ang_limit == self.upper_ang_limit {
            powered = false;
        }
        if limit != 0 || powered {
            let curr_erp = erp; // Bullet's `m_softnessLimAng` flag path only
            let mut row_error: Real = 0.0;
            let mut lower: Real = 0.0;
            let mut upper: Real = 0.0;
            if powered {
                let motor = self.ang_motor.unwrap();
                let mot_fact = get_motor_factor(
                    self.ang_pos,
                    self.lower_ang_limit,
                    self.upper_ang_limit,
                    motor.target_velocity,
                    inv_dt * curr_erp,
                );
                row_error = mot_fact * motor.target_velocity;
                lower = -motor.max_force / inv_dt;
                upper = motor.max_force / inv_dt;
            }
            if limit != 0 {
                row_error += inv_dt * curr_erp * limit_err;
                if self.lower_ang_limit == self.upper_ang_limit {
                    lower = -Real::INFINITY;
                    upper = Real::INFINITY;
                } else if limit == 1 {
                    // High limit (Bullet's `limit_err > 0` = `rot > upper`).
                    lower = 0.0;
                    upper = Real::INFINITY;
                } else {
                    // Low limit.
                    lower = -Real::INFINITY;
                    upper = 0.0;
                }
                let bounce = (1.0 - self.damping_lim_ang).abs();
                if bounce > 0.0 {
                    let vel =
                        a.angular_velocity.dot(ax1) - b.angular_velocity.dot(ax1);
                    if limit == 1 {
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
                row_error *= self.softness_lim_ang;
            }
            let limits = mirrored_row_limits(lower, upper);
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

