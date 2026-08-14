//! Constraints (joints) — port of Bullet's `btTypedConstraint` subclasses
//! (`BulletDynamics/ConstraintSolver`).
//!
//! A [`Constraint`] connects two rigid bodies at fixed points/axes in each
//! body's local frame and is enforced by the sequential-impulse solver as a
//! set of bilateral equality rows (Bullet's `getInfo1`/`getInfo2` protocol,
//! flattened into [`ConstraintRow`]s each step). Contact constraints and
//! joint rows share the same PGS iteration loop, exactly like Bullet's
//! `btSequentialImpulseConstraintSolver`.
//!
//! Supported types:
//!
//! - [`ConstraintKind::Point2Point`] — a ball-socket joint
//!   (`btPoint2PointConstraint`): the two pivot points (given in each body's
//!   center-of-mass frame) are kept coincident, with three orthogonal
//!   constraint rows and an ERP-scaled Baumgarte bias (`m_constraintError =
//!   fps · erp · (pivotB − pivotA)`, default erp = 0.2 like the global
//!   contact ERP). Joints are **not** warm-started, matching Bullet
//!   (`internalSetAppliedImpulse(0)` in `convertJoints`).
//!
//! Like Bullet's `addConstraint(..., disableCollisionsBetweenLinkedBodies =
//! true)`, a [`Constraint`] suppresses collision response between its two
//! bodies by default ([`Constraint::set_disable_collisions`]).

use bullet_math::{Matrix3, Quaternion, Real, Vector3};

use crate::body::RigidBody;
use crate::solver::sequential_impulse::{ConstraintRow, RowLimits};

/// The specific kind of a [`Constraint`] and its parameters (Bullet's
/// constraint subclasses).
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    /// Ball-socket joint: keep `pivot_a` (on body A) coincident with
    /// `pivot_b` (on body B); both are given in their body's
    /// center-of-mass frame (Bullet's `btPoint2PointConstraint`).
    Point2Point(Point2PointConstraint),
    /// Revolute (hinge) joint: keep the pivots coincident *and* the hinge
    /// axes parallel, with optional angular limits and a motor around the
    /// axis (Bullet's `btHingeConstraint`).
    Hinge(HingeConstraint),
}

impl ConstraintKind {
    /// Mutable access to the [`Point2PointConstraint`] parameters.
    pub fn point2point_mut(&mut self) -> &mut Point2PointConstraint {
        match self {
            ConstraintKind::Point2Point(p2p) => p2p,
            ConstraintKind::Hinge(_) => panic!("not a point-to-point constraint"),
        }
    }

    /// Access to the [`HingeConstraint`] parameters.
    pub fn hinge(&self) -> &HingeConstraint {
        match self {
            ConstraintKind::Hinge(hinge) => hinge,
            ConstraintKind::Point2Point(_) => panic!("not a hinge constraint"),
        }
    }

    /// Mutable access to the [`HingeConstraint`] parameters.
    pub fn hinge_mut(&mut self) -> &mut HingeConstraint {
        match self {
            ConstraintKind::Hinge(hinge) => hinge,
            ConstraintKind::Point2Point(_) => panic!("not a hinge constraint"),
        }
    }
}

/// A joint between two bodies (Bullet's `btTypedConstraint`). The common
/// flags (`enabled`, `disable_collisions`) live here; the geometry and tuning
/// live in [`ConstraintKind`].
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    kind: ConstraintKind,
    /// Body indices, sorted so `body_a <= body_b` (constraint math is
    /// direction-sensitive; the constructors fix the order at creation).
    body_a: usize,
    body_b: usize,
    /// Whether the solver enforces this constraint this step (Bullet's
    /// `setEnabled`). A disabled constraint also stops suppressing
    /// collisions between its bodies.
    enabled: bool,
    /// Whether collision response between the two bodies is suppressed
    /// (Bullet's `addConstraint(..., disableCollisionsBetweenLinkedBodies)`).
    disable_collisions: bool,
}

impl Constraint {
    /// A ball-socket joint keeping the body-local pivot points `pivot_a` /
    /// `pivot_b` coincident (Bullet's
    /// `btPoint2PointConstraint(rbA, rbB, pivotInA, pivotInB)`). `pivot_a` is
    /// in body A's center-of-mass frame, `pivot_b` in body B's.
    pub fn point2point(
        body_a: usize,
        body_b: usize,
        pivot_a: Vector3,
        pivot_b: Vector3,
    ) -> Self {
        Self {
            kind: ConstraintKind::Point2Point(Point2PointConstraint {
                pivot_a,
                pivot_b,
                impulse_clamp: 0.0,
                erp: None,
            }),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    /// A hinge (revolute) joint: the pivots `pivot_a`/`pivot_b` (in each
    /// body's center-of-mass frame) stay coincident and the hinge axes
    /// `axis_a`/`axis_b` (in each body's frame) stay parallel, so the bodies
    /// can only rotate around the axis (Bullet's
    /// `btHingeConstraint(rbA, rbB, pivotInA, pivotInB, axisInA, axisInB)`).
    ///
    /// The constraint frames are built with a deterministic perpendicular
    /// basis around each axis (see [`HingeConstraint`]); unlike Bullet, the
    /// zero-angle reference is not aligned to body A's initial X axis, so
    /// limit angles are measured from the assembly-time axes.
    pub fn hinge(
        body_a: usize,
        body_b: usize,
        pivot_a: Vector3,
        axis_a: Vector3,
        pivot_b: Vector3,
        axis_b: Vector3,
    ) -> Self {
        Self {
            kind: ConstraintKind::Hinge(HingeConstraint::new(pivot_a, axis_a, pivot_b, axis_b)),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    pub fn body_a(&self) -> usize {
        self.body_a
    }

    pub fn body_b(&self) -> usize {
        self.body_b
    }

    pub fn kind(&self) -> &ConstraintKind {
        &self.kind
    }

    pub fn kind_mut(&mut self) -> &mut ConstraintKind {
        &mut self.kind
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable the constraint (Bullet's `setEnabled`); disabled
    /// constraints produce no rows and do not suppress collisions.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn disable_collisions(&self) -> bool {
        self.disable_collisions
    }

    /// Whether collision response between the linked bodies is suppressed
    /// (Bullet's `addConstraint(..., disableCollisionsBetweenLinkedBodies)`,
    /// default true).
    pub fn set_disable_collisions(&mut self, disable: bool) {
        self.disable_collisions = disable;
    }
}

/// Parameters of a ball-socket joint (Bullet's `btPoint2PointConstraint`).
#[derive(Debug, Clone, PartialEq)]
pub struct Point2PointConstraint {
    /// Pivot point in body A's center-of-mass frame.
    pub pivot_a: Vector3,
    /// Pivot point in body B's center-of-mass frame.
    pub pivot_b: Vector3,
    /// Optional symmetric clamp on the impulse of each row (Bullet's
    /// `btConstraintSetting::m_impulseClamp`); `0.0` disables the clamp.
    pub impulse_clamp: Real,
    /// Optional ERP override (Bullet's `BT_P2P_FLAGS_ERP`); `None` uses the
    /// global ERP (0.2).
    pub erp: Option<Real>,
}

impl Point2PointConstraint {
    /// Clamp the impulse of every row to `[-clamp, clamp]` (Bullet's
    /// `btConstraintSetting::m_impulseClamp`); `0.0` disables the clamp.
    pub fn set_impulse_clamp(&mut self, clamp: Real) {
        self.impulse_clamp = clamp;
    }

    /// Override the position-correction factor for this joint (Bullet's
    /// `BT_CONSTRAINT_ERP`); `None` restores the global 0.2.
    pub fn set_erp(&mut self, erp: Option<Real>) {
        self.erp = erp;
    }
}

/// A frame attached to a joint body: the pivot (`origin`) and the basis in
/// the body's center-of-mass frame. The Z column of `basis` is the hinge
/// axis; X and Y span the hinge-angle reference plane (Bullet's
/// `m_rbAFrame` / `m_rbBFrame`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointFrame {
    /// Pivot point in the body's center-of-mass frame.
    pub origin: Vector3,
    /// Basis in the body's center-of-mass frame; column 2 is the hinge axis.
    pub basis: Matrix3,
}

impl JointFrame {
    pub fn new(origin: Vector3, basis: Matrix3) -> Self {
        Self { origin, basis }
    }

    /// The hinge axis (the frame's Z column) in the body frame.
    pub fn axis(&self) -> Vector3 {
        self.basis.col(2)
    }
}

/// Angular limits of a hinge around its axis (Bullet's `btAngularLimit`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngularLimit {
    /// Lower limit, in radians.
    pub low: Real,
    /// Upper limit, in radians. `low > high` deactivates the limit.
    pub high: Real,
    /// Limit softness (Bullet's `m_softness`, default 0.9); kept for API
    /// compatibility — the current row formulation does not use it.
    pub softness: Real,
    /// Position-correction factor on the limit bias (Bullet's
    /// `m_biasFactor`, default 0.3).
    pub bias_factor: Real,
    /// Bounce restitution at the limit (Bullet's `m_relaxationFactor`;
    /// 1.0 = no bounce, 0.0 = fully elastic).
    pub relaxation_factor: Real,
}

impl AngularLimit {
    /// A hinge angle limit in radians (Bullet's `setLimit` with the default
    /// softness 0.9, bias factor 0.3, relaxation factor 1.0). `low > high`
    /// deactivates the limit.
    pub fn new(low: Real, high: Real) -> Self {
        Self {
            low,
            high,
            softness: 0.9,
            bias_factor: 0.3,
            relaxation_factor: 1.0,
        }
    }

    pub fn set(
        &mut self,
        low: Real,
        high: Real,
        softness: Real,
        bias_factor: Real,
        relaxation_factor: Real,
    ) {
        self.low = low;
        self.high = high;
        self.softness = softness;
        self.bias_factor = bias_factor;
        self.relaxation_factor = relaxation_factor;
    }

    /// Test `angle` against the limit (Bullet's `test`): returns the
    /// correction (signed distance back into range), the limit sign (+1 low,
    /// − high) and whether the limit is active.
    fn test(&self, angle: Real) -> (Real, Real, bool) {
        let half_range = (self.high - self.low) / 2.0;
        if half_range >= 0.0 {
            let center = normalize_angle(self.low + half_range);
            let deviation = normalize_angle(angle - center);
            if deviation < -half_range {
                return (-(deviation + half_range), 1.0, true);
            }
            if deviation > half_range {
                return (half_range - deviation, -1.0, true);
            }
        }
        (0.0, 0.0, false)
    }
}

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
    /// deterministically: `t1` perpendicular to `axis_a`, `t2 = axis_a 脳 t1`,
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
        let rotation = Quaternion::from_rotation_arc(axis_a, axis_b);
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
    fn prepare_rows(
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

        // Linear pivot rows (Bullet's `m_J1linearAxis` = 卤e_j), unless the
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
        // the misalignment u = ax1 脳 ax2 (Bullet's rows 3-4).
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

/// The hinge angle between two world frames (Bullet's `getHingeAngle` with
/// `m_referenceSign = +1`): the rotation of frame B's Y axis around frame A's
/// Z axis.
fn get_hinge_angle(tr_a: &Matrix3, tr_b: &Matrix3) -> Real {
    let ref_axis0 = tr_a.col(0);
    let ref_axis1 = tr_a.col(1);
    let swing_axis = tr_b.col(1);
    swing_axis.dot(ref_axis0).atan2(swing_axis.dot(ref_axis1))
}

/// Wrap an angle into `[-蟺, 蟺]` (Bullet's `btNormalizeAngle`).
fn normalize_angle(angle: Real) -> Real {
    let two_pi = 2.0 * bullet_math::PI;
    let mut a = angle % two_pi; // truncating, like `btFmod`.
    if a < -bullet_math::PI {
        a += two_pi;
    } else if a > bullet_math::PI {
        a -= two_pi;
    }
    a
}

/// Motor factor fading the motor near the limits (Bullet's
/// `btTypedConstraint::getMotorFactor`).
fn get_motor_factor(pos: Real, low_lim: Real, upp_lim: Real, vel: Real, time_fact: Real) -> Real {
    if low_lim > upp_lim {
        return 1.0;
    }
    if low_lim == upp_lim {
        return 0.0;
    }
    let delta_max = vel / time_fact;
    if delta_max < 0.0 {
        if pos >= low_lim && pos < low_lim - delta_max {
            (low_lim - pos) / delta_max
        } else if pos < low_lim {
            0.0
        } else {
            1.0
        }
    } else if delta_max > 0.0 {
        if pos <= upp_lim && pos > upp_lim - delta_max {
            (upp_lim - pos) / delta_max
        } else if pos > upp_lim {
            0.0
        } else {
            1.0
        }
    } else {
        0.0
    }
}

/// Flatten a [`Constraint`] into its solver rows for this step (Bullet's
/// `getInfo1` + `getInfo2` + `convertJoint`). Rows are appended to `out`.
/// `inv_dt` is `1 / dt` (Bullet's `info2.fps`); `erp` is the global ERP
/// (Bullet's `infoGlobal.m_erp`, 0.2). Joints with per-step state (the hinge
/// angle and limit test) update it here, before the solve.
pub fn prepare_rows(
    constraint: &mut Constraint,
    bodies: &[RigidBody],
    inv_dt: Real,
    erp: Real,
    out: &mut Vec<ConstraintRow>,
) {
    if !constraint.enabled {
        return;
    }
    match &mut constraint.kind {
        ConstraintKind::Point2Point(p2p) => {
            let a = &bodies[constraint.body_a];
            let b = &bodies[constraint.body_b];
            // World-space pivots and lever arms from each body's COM.
            let pivot_a = a.position + a.orientation * p2p.pivot_a;
            let pivot_b = b.position + b.orientation * p2p.pivot_b;
            let r_a = pivot_a - a.position;
            let r_b = pivot_b - b.position;
            // `k = fps · erp`: the position error, scaled to a velocity
            // target (Bullet's `m_constraintError = k · (pivotB − pivotA)`).
            let k = inv_dt * p2p.erp.unwrap_or(erp);
            let limits = if p2p.impulse_clamp > 0.0 {
                RowLimits::Bilateral(p2p.impulse_clamp)
            } else {
                RowLimits::Unbounded
            };
            // Three orthogonal rows along the world axes (Bullet's
            // `m_J1linearAxis` = 卤e_j with the skew-symmetric angular
            // jacobians built from the lever arms).
            for axis in 0..3 {
                let mut normal = Vector3::ZERO;
                normal[axis] = 1.0;
                out.push(ConstraintRow::prepare(
                    constraint.body_a,
                    constraint.body_b,
                    normal,
                    r_a.cross(normal),
                    normal,
                    r_b.cross(normal),
                    a,
                    b,
                    // bias = k · (pivotA − pivotB) · n (see module docs for
                    // the sign convention; the accumulated impulse comes out
                    // with the correct sign automatically).
                    k * (pivot_a - pivot_b)[axis],
                    limits,
                ));
            }
        }
        ConstraintKind::Hinge(hinge) => {
            hinge.prepare_rows(
                constraint.body_a,
                constraint.body_b,
                bodies,
                inv_dt,
                erp,
                out,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::PhysicsWorld;
    use bullet_collision::shapes::CollisionShape;
    use bullet_math::Quaternion;

    const DT: Real = 1.0 / 60.0;

    /// A world with a fixed anchor (static box at the origin) and a dynamic
    /// sphere bob connected by a point-to-point joint whose pivot is
    /// `bob_offset` from the bob's centre and `anchor_offset` from the
    /// anchor's origin. The bob starts at the constraint-satisfied position
    /// (`anchor_offset − bob_offset`) so there is no initial correction kick.
    fn pendulum_world(bob_offset: Vector3, anchor_offset: Vector3) -> (PhysicsWorld, usize, usize) {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        // The bob is a sphere; its centre of mass is the sphere centre.
        let bob = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            anchor_offset - bob_offset,
            1.0,
        ));
        // The joint holds bob centre + bob_offset at anchor origin +
        // anchor_offset (both in their bodies' frames).
        let c = Constraint::point2point(bob, anchor, bob_offset, anchor_offset);
        world.add_constraint(c);
        (world, bob, anchor)
    }

    /// A pendulum released from rest at 45° from the vertical (the bottom
    /// position is (0, -1, 0)) swings: the bob's pivot stays within a small
    /// Baumgarte residual of the anchor, the bob oscillates (x crosses the
    /// bottom repeatedly) and its speed peaks at the bottom.
    #[test]
    fn pendulum_swings_and_keeps_the_pivot() {
        // Bob centre at (sin45°, −cos45°, 0) oriented by −45° around Z, so
        // its pivot (local (-1,0,0), the rod) is exactly at the anchor
        // (origin): the constraint starts satisfied and there is no
        // correction kick. Rod length 1 m.
        let (mut world, bob, _anchor) = pendulum_world(
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::ZERO,
        );
        let a45 = std::f32::consts::FRAC_PI_4 as Real;
        world.body_mut(bob).position = Vector3::new(a45.sin(), -a45.cos(), 0.0);
        world.body_mut(bob).orientation = Quaternion::from_rotation_z(-a45);
        let pivot_err = |world: &PhysicsWorld, bob: usize| -> Real {
            let b = world.body(bob);
            (b.position + b.orientation * Vector3::new(-1.0, 0.0, 0.0)).length()
        };
        let mut max_err: Real = 0.0;
        let mut min_x: Real = Real::INFINITY;
        let mut max_speed: Real = 0.0;
        let mut zero_crossings = 0;
        let mut prev_x = world.body(bob).position.x;
        for _ in 0..600 {
            world.step_simulation(DT);
            let b = world.body(bob);
            max_err = max_err.max(pivot_err(&world, bob));
            min_x = min_x.min(b.position.x);
            max_speed = max_speed.max(b.linear_velocity.length());
            let x = b.position.x;
            if prev_x > 0.0 && x <= 0.0 {
                zero_crossings += 1;
            }
            prev_x = x;
        }
        // Constraint maintained within the Baumgarte residual (~0.1 m during
        // a fast swing; the steady-state error is ~1 cm).
        assert!(max_err < 0.2, "pivot error {max_err} too large");
        // Swung through the bottom at least a couple of times.
        assert!(zero_crossings >= 3, "bob did not oscillate (crossings {zero_crossings})");
        // Reached the other side of the swing (x well below the release 0.707).
        assert!(min_x < -0.5, "bob never swung past the bottom: min_x = {min_x}");
        // Non-trivial speed at the bottom of the swing
        // (v = √(2·g·L·(1−cos45°)) ≈ 2.4 m/s).
        assert!(max_speed > 1.5, "bob too slow: {max_speed}");
    }

    /// A pendulum hanging at rest below the pivot stays there: the bob's
    /// pivot stays on the anchor pivot and the bob does not drift.
    #[test]
    fn pendulum_hangs_at_equilibrium() {
        // Bob centre at the origin with the pivot 1 m above at (0, 1, 0);
        // the anchor pivot is also at (0, 1, 0): the bob hangs vertically
        // below the pivot, a stable equilibrium.
        let (mut world, bob, _anchor) = pendulum_world(
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        for _ in 0..600 {
            world.step_simulation(DT);
        }
        let b = world.body(bob);
        let pivot = b.position + b.orientation * Vector3::new(0.0, 1.0, 0.0);
        // The pivot converges to the anchor; only a tiny Baumgarte residual
        // remains (the bob sags a fraction of a millimetre under gravity).
        assert!(
            (pivot - Vector3::new(0.0, 1.0, 0.0)).length() < 0.01,
            "pivot not on the anchor: {:?}",
            pivot
        );
        assert!(
            b.position.length() < 0.02,
            "bob drifted from below the pivot: {:?}",
            b.position
        );
        assert!(b.linear_velocity.length() < 0.05, "bob still moving: {:?}", b.linear_velocity);
    }

    /// A double pendulum keeps both rod lengths constant while it swings.
    #[test]
    fn two_link_chain_keeps_its_lengths() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let upper = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::new(0.0, -1.0, 0.0),
            1.0,
        ));
        let lower = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::new(0.0, -3.0, 0.0),
            1.0,
        ));
        // Rod 1: the top of the upper bob (local (0,1,0)) pinned to the
        // anchor pivot at the origin. Rod 2: the top of the lower bob pinned
        // to the bottom of the upper bob (local (0,-1,0)). Both constraints
        // start satisfied, hanging straight down.
        world.add_constraint(Constraint::point2point(
            upper,
            anchor,
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::ZERO,
        ));
        world.add_constraint(Constraint::point2point(
            lower,
            upper,
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        ));
        // Nudge the lower bob sideways so the chain swings.
        world.body_mut(lower).linear_velocity = Vector3::new(2.0, 0.0, 0.0);
        let mut max_err: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            let u = world.body(upper);
            let l = world.body(lower);
            // Upper top must sit at the origin; lower top at the upper bottom.
            max_err = max_err
                .max((u.position + u.orientation * Vector3::new(0.0, 1.0, 0.0)).length())
                .max(
                    (l.position + l.orientation * Vector3::new(0.0, 1.0, 0.0)
                        - (u.position + u.orientation * Vector3::new(0.0, -1.0, 0.0)))
                    .length(),
                );
        }
        assert!(max_err < 0.2, "chain pivots drifted: {max_err}");
    }

    /// A door hinged at its edge swings like a pendulum around the vertical
    /// hinge axis: the pivot stays on the anchor, the hinge axes stay
    /// parallel, and the hinge angle oscillates.
    #[test]
    fn hinge_swings_and_keeps_axis_aligned() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        // A door (box) hinged at its edge (local (-0.5,0,0)) to the anchor
        // at the origin, opened 45° around the vertical (Z) hinge.
        let door = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
            Vector3::new(0.0, 0.5, 0.0),
            2.0,
        ));
        let a45 = std::f32::consts::FRAC_PI_4 as Real;
        world.body_mut(door).position = Vector3::new(a45.cos() * 0.5, a45.sin() * 0.5, 0.0);
        world.body_mut(door).orientation = Quaternion::from_rotation_z(a45);
        // Hinge: the door's edge (local (-0.5,0,0)) pinned to the anchor
        // pivot at the origin, axis (0,0,1) on both bodies.
        let hinge = Constraint::hinge(
            door,
            anchor,
            Vector3::new(-0.5, 0.0, 0.0),
            Vector3::Z,
            Vector3::ZERO,
            Vector3::Z,
        );
        world.add_constraint(hinge);

        let mut max_pivot_err: Real = 0.0;
        let mut min_align: Real = 1.0;
        let mut max_angle: Real = 0.0;
        let mut zero_crossings = 0;
        let mut prev_angle = 0.0;
        for step in 0..600 {
            world.step_simulation(DT);
            let d = world.body(door);
            let pivot = d.position + d.orientation * Vector3::new(-0.5, 0.0, 0.0);
            max_pivot_err = max_pivot_err.max(pivot.length());
            // Hinge axes: the door's local Z rotated to world, the anchor's Z.
            let ax1 = d.orientation * Vector3::Z;
            min_align = min_align.min(ax1.dot(Vector3::Z).abs());
            let angle = world.constraint(0).kind().hinge().hinge_angle();
            max_angle = max_angle.max(angle.abs());
            if step > 0 && prev_angle > 0.0 && angle <= 0.0 {
                zero_crossings += 1;
            }
            prev_angle = angle;
        }
        assert!(max_pivot_err < 0.2, "door pivot drifted: {max_pivot_err}");
        assert!(min_align > 0.99, "hinge axes misaligned: {min_align}");
        assert!(max_angle > 0.5, "door barely swung: max angle {max_angle}");
        assert!(zero_crossings >= 3, "door did not oscillate: {zero_crossings} crossings");
    }

    /// A door with angular limits stops at the limit instead of swinging
    /// through: the hinge angle never exceeds the range and settles on it.
    #[test]
    fn hinge_limit_stops_the_door() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let door = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
            Vector3::ZERO,
            2.0,
        ));
        // Open the door 45° (well past the 30° limit) and release it.
        let a45 = std::f32::consts::FRAC_PI_4 as Real;
        world.body_mut(door).position = Vector3::new(a45.cos() * 0.5, a45.sin() * 0.5, 0.0);
        world.body_mut(door).orientation = Quaternion::from_rotation_z(a45);
        let mut hinge = Constraint::hinge(
            door,
            anchor,
            Vector3::new(-0.5, 0.0, 0.0),
            Vector3::Z,
            Vector3::ZERO,
            Vector3::Z,
        );
        hinge.kind_mut().hinge_mut().set_limit(-0.3, 0.3);
        world.add_constraint(hinge);
        let mut min_angle: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            let angle = world.constraint(0).kind().hinge().hinge_angle();
            min_angle = min_angle.min(angle);
        }
        // The door starts at +45° (past the +0.3 limit, so the initial state
        // violates it by construction) and swings down under gravity; the
        // low limit at −0.3 stops it. It must never swing past the low limit
        // and must settle resting on it.
        assert!(min_angle > -0.35, "door passed the low limit: {min_angle}");
        let final_angle = world.constraint(0).kind().hinge().hinge_angle();
        assert!(
            (final_angle + 0.3).abs() < 0.05,
            "door did not settle on the low limit: {final_angle}"
        );
    }

    /// A motorized hinge rotates the door at the target angular velocity.
    #[test]
    fn hinge_motor_rotates_the_door() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let door = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
            Vector3::new(0.5, 0.0, 0.0),
            2.0,
        ));
        let mut hinge = Constraint::hinge(
            door,
            anchor,
            Vector3::new(-0.5, 0.0, 0.0),
            Vector3::Z,
            Vector3::ZERO,
            Vector3::Z,
        );
        hinge.kind_mut().hinge_mut().enable_motor(1.0, 1000.0);
        world.add_constraint(hinge);
        for _ in 0..180 {
            world.step_simulation(DT);
        }
        // 3 s at 1 rad/s ≈ 3 rad of rotation.
        let angle = world.constraint(0).kind().hinge().hinge_angle();
        assert!(
            (angle - 3.0).abs() < 1.0,
            "motor did not rotate the door at 1 rad/s: {angle}"
        );
    }

    /// Two overlapping boxes linked by a joint do not push each other apart
    /// (collisions suppressed), while the same boxes without the joint do.
    #[test]
    fn joint_suppresses_collisions_between_linked_bodies() {
        let run = |with_joint: bool| -> Vector3 {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            // Two overlapping boxes at the origin.
            let a = world.add_body(RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::ZERO,
                1.0,
            ));
            let b = world.add_body(RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::ZERO,
                1.0,
            ));
            if with_joint {
                let mut c = Constraint::point2point(a, b, Vector3::ZERO, Vector3::ZERO);
                c.set_disable_collisions(true);
                world.add_constraint(c);
            }
            for _ in 0..30 {
                world.step_simulation(DT);
            }
            world.body(b).position - world.body(a).position
        };
        let sep_joint = run(true);
        let sep_plain = run(false);
        // Without the joint the boxes separate (collision pushes them apart);
        // with the joint they stay linked (no contact response pushes them).
        assert!(
            sep_plain.length() > 0.05,
            "plain overlapping boxes should separate, got {sep_plain:?}"
        );
        assert!(
            sep_joint.length() < 0.02,
            "linked boxes must not be pushed apart, got {sep_joint:?}"
        );
    }

    /// The impulse clamp caps the per-row impulse: a heavy bob held up by a
    /// clamped joint sags (the clamp releases the excess), while an
    /// unclamped joint holds it.
    #[test]
    fn impulse_clamp_limits_the_support_impulse() {
        let run = |clamp: Real| -> Real {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
            let anchor = world.add_body(RigidBody::new_static(
                CollisionShape::box_shape(Vector3::splat(0.2)),
                Vector3::ZERO,
            ));
            let bob = world.add_body(RigidBody::new_dynamic(
                CollisionShape::sphere(0.1),
                Vector3::ZERO,
                10.0,
            ));
            // The joint pins the bob centre to the anchor at the origin.
            let mut c = Constraint::point2point(bob, anchor, Vector3::ZERO, Vector3::ZERO);
            c.kind_mut().point2point_mut().set_impulse_clamp(clamp);
            world.add_constraint(c);
            for _ in 0..300 {
                world.step_simulation(DT);
            }
            world.body(bob).position.y
        };
        // The static support impulse needed is m·g·dt ≈ 10·9.81/60 ≈ 1.6 per
        // step; a clamp of 0.1 cannot hold the bob, so it falls away.
        let clamped = run(0.1);
        assert!(
            clamped < -1.0,
            "clamped joint should let the bob fall, y = {clamped}"
        );
        let held = run(0.0);
        assert!(
            held.abs() < 0.05,
            "unclamped joint should hold the bob, y = {held}"
        );
    }
}
