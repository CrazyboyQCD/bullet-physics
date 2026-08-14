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
    /// Fixed (weld) joint: both pivots coincide and the relative orientation
    /// is locked — all six degrees of freedom are removed (Bullet's
    /// `btFixedConstraint`, which locks every axis of a
    /// `btGeneric6DofConstraint`).
    Fixed(FixedConstraint),
    /// Cone-and-twist joint: the pivots coincide, the relative orientation
    /// stays inside an elliptical cone around the X axes plus a one-sided
    /// twist limit around the cone axis (Bullet's `btConeTwistConstraint`).
    ConeTwist(ConeTwistConstraint),
    /// Slider (prismatic) joint: the bodies slide along the frame X axes
    /// with optional linear/angular limits and motors (Bullet's
    /// `btSliderConstraint`).
    Slider(SliderConstraint),
    /// Generic six-degree-of-freedom joint: three linear and three angular
    /// axes with independent limits and motors (Bullet's
    /// `btGeneric6DofConstraint`).
    Generic6Dof(Generic6DofConstraint),
}

impl ConstraintKind {
    /// Mutable access to the [`Point2PointConstraint`] parameters.
    pub fn point2point_mut(&mut self) -> &mut Point2PointConstraint {
        match self {
            ConstraintKind::Point2Point(p2p) => p2p,
            ConstraintKind::Hinge(_) => panic!("not a point-to-point constraint"),
            ConstraintKind::Fixed(_) | ConstraintKind::ConeTwist(_) | ConstraintKind::Slider(_) => {
                panic!("not a point-to-point constraint")
            }
            ConstraintKind::Generic6Dof(_) => panic!("not a point-to-point constraint"),
        }
    }

    /// Access to the [`HingeConstraint`] parameters.
    pub fn hinge(&self) -> &HingeConstraint {
        match self {
            ConstraintKind::Hinge(hinge) => hinge,
            ConstraintKind::Point2Point(_) => panic!("not a hinge constraint"),
            ConstraintKind::Fixed(_) | ConstraintKind::ConeTwist(_) | ConstraintKind::Slider(_) => {
                panic!("not a hinge constraint")
            }
            ConstraintKind::Generic6Dof(_) => panic!("not a hinge constraint"),
        }
    }

    /// Mutable access to the [`HingeConstraint`] parameters.
    pub fn hinge_mut(&mut self) -> &mut HingeConstraint {
        match self {
            ConstraintKind::Hinge(hinge) => hinge,
            ConstraintKind::Point2Point(_) => panic!("not a hinge constraint"),
            ConstraintKind::Fixed(_) | ConstraintKind::ConeTwist(_) | ConstraintKind::Slider(_) => {
                panic!("not a hinge constraint")
            }
            ConstraintKind::Generic6Dof(_) => panic!("not a hinge constraint"),
        }
    }

    /// Access to the [`FixedConstraint`] parameters.
    pub fn fixed(&self) -> &FixedConstraint {
        match self {
            ConstraintKind::Fixed(fixed) => fixed,
            _ => panic!("not a fixed constraint"),
        }
    }

    /// Mutable access to the [`FixedConstraint`] parameters.
    pub fn fixed_mut(&mut self) -> &mut FixedConstraint {
        match self {
            ConstraintKind::Fixed(fixed) => fixed,
            _ => panic!("not a fixed constraint"),
        }
    }

    /// Access to the [`ConeTwistConstraint`] parameters.
    pub fn cone_twist(&self) -> &ConeTwistConstraint {
        match self {
            ConstraintKind::ConeTwist(ct) => ct,
            _ => panic!("not a cone-twist constraint"),
        }
    }

    /// Mutable access to the [`ConeTwistConstraint`] parameters.
    pub fn cone_twist_mut(&mut self) -> &mut ConeTwistConstraint {
        match self {
            ConstraintKind::ConeTwist(ct) => ct,
            _ => panic!("not a cone-twist constraint"),
        }
    }

    /// Access to the [`SliderConstraint`] parameters.
    pub fn slider(&self) -> &SliderConstraint {
        match self {
            ConstraintKind::Slider(slider) => slider,
            _ => panic!("not a slider constraint"),
        }
    }

    /// Mutable access to the [`SliderConstraint`] parameters.
    pub fn slider_mut(&mut self) -> &mut SliderConstraint {
        match self {
            ConstraintKind::Slider(slider) => slider,
            _ => panic!("not a slider constraint"),
        }
    }

    /// Access to the [`Generic6DofConstraint`] parameters.
    pub fn generic_6dof(&self) -> &Generic6DofConstraint {
        match self {
            ConstraintKind::Generic6Dof(dof) => dof,
            _ => panic!("not a generic 6DOF constraint"),
        }
    }

    /// Mutable access to the [`Generic6DofConstraint`] parameters.
    pub fn generic_6dof_mut(&mut self) -> &mut Generic6DofConstraint {
        match self {
            ConstraintKind::Generic6Dof(dof) => dof,
            _ => panic!("not a generic 6DOF constraint"),
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

    /// A fixed (weld) joint: the frames `frame_a` (in body A's
    /// center-of-mass frame) and `frame_b` (in body B's) stay coincident —
    /// both the pivot and the relative orientation are locked, removing all
    /// six degrees of freedom (Bullet's `btFixedConstraint`).
    pub fn fixed(
        body_a: usize,
        body_b: usize,
        frame_a: JointFrame,
        frame_b: JointFrame,
    ) -> Self {
        Self {
            kind: ConstraintKind::Fixed(FixedConstraint::new(frame_a, frame_b)),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    /// A cone-and-twist joint: the pivots stay coincident and the relative
    /// orientation is limited to an elliptical cone around the frame X axes
    /// (`swing_span1` around Y, `swing_span2` around Z) plus a one-sided
    /// twist limit `twist_span` around the cone axis (Bullet's
    /// `btConeTwistConstraint`). All spans are in radians.
    pub fn cone_twist(
        body_a: usize,
        body_b: usize,
        frame_a: JointFrame,
        frame_b: JointFrame,
        swing_span1: Real,
        swing_span2: Real,
        twist_span: Real,
    ) -> Self {
        Self {
            kind: ConstraintKind::ConeTwist(ConeTwistConstraint::new(
                frame_a,
                frame_b,
                swing_span1,
                swing_span2,
                twist_span,
            )),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    /// A slider (prismatic) joint: the bodies slide along the frame X axes.
    /// Bullet's defaults apply — free linear travel, locked rotation — so
    /// set the limits/motors to taste (see [`SliderConstraint`]).
    pub fn slider(body_a: usize, body_b: usize, frame_a: JointFrame, frame_b: JointFrame) -> Self {
        Self {
            kind: ConstraintKind::Slider(SliderConstraint::new(frame_a, frame_b)),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    /// A generic six-degree-of-freedom joint (Bullet's
    /// `btGeneric6DofConstraint`): three linear axes along the frame A
    /// directions and three angular axes around the relative-rotation
    /// axes, each with independent limits and an optional motor. Bullet's
    /// defaults apply — linear locked, angular free — so a fresh joint
    /// behaves like a ball joint.
    pub fn generic_6dof(
        body_a: usize,
        body_b: usize,
        frame_a: JointFrame,
        frame_b: JointFrame,
    ) -> Self {
        Self {
            kind: ConstraintKind::Generic6Dof(Generic6DofConstraint::new(frame_a, frame_b)),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
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

/// Wrap an angle into `[-π, π]` (Bullet's `btNormalizeAngle`).
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

/// Return `angle` adjusted to the range `[lo, hi]` by adding/subtracting
/// `2π` so it is closest to one of the limits (Bullet's
/// `btAdjustAngleToLimits`); all arguments are expected normalized. When
/// `lo >= hi` (a free or locked axis) the angle is returned unchanged.
fn adjust_angle_to_limits(angle: Real, lo: Real, hi: Real) -> Real {
    let two_pi = 2.0 * bullet_math::PI;
    if lo >= hi {
        return angle;
    }
    if angle < lo {
        let diff_lo = normalize_angle(lo - angle).abs();
        let diff_hi = normalize_angle(hi - angle).abs();
        if diff_lo < diff_hi {
            angle
        } else {
            angle + two_pi
        }
    } else if angle > hi {
        let diff_hi = normalize_angle(angle - hi).abs();
        let diff_lo = normalize_angle(angle - lo).abs();
        if diff_lo < diff_hi {
            angle - two_pi
        } else {
            angle
        }
    } else {
        angle
    }
}

/// Bullet's `matrixToEulerXYZ`: the Euler angles `(x, y, z)` of a rotation
/// matrix, extracted with `btGetMatrixElem(mat, index) = mat[index % 3]
/// [index / 3]` (row `index % 3`, column `index / 3` in Bullet's row-major
/// storage). glam stores columns, so Bullet's `mat[i][j]` is
/// `m.col(j)[i]`; the extraction is ported verbatim (the signs are
/// consistent with the constraint rows built from these angles — do not
/// "fix" them).
fn euler_xyz(m: &Matrix3) -> Vector3 {
    // Bullet's `mat[2][0]`, `mat[2][1]`, `mat[1][0]`, ... in glam's column
    // storage (row i, column j = `m.col(j)[i]`).
    let r20 = m.col(0).z; // mat[2][0]
    let r21 = m.col(1).z; // mat[2][1]
    let r22 = m.col(2).z; // mat[2][2]
    let r10 = m.col(0).y; // mat[1][0]
    let r11 = m.col(1).y; // mat[1][1]
    let r01 = m.col(1).x; // mat[0][1]
    let r00 = m.col(0).x; // mat[0][0]
    if r20 < 1.0 {
        if r20 > -1.0 {
            Vector3::new((-r21).atan2(r22), r20.asin(), (-r10).atan2(r00))
        } else {
            Vector3::new(-(r01).atan2(r11), -bullet_math::FRAC_PI_2, 0.0)
        }
    } else {
        Vector3::new((r01).atan2(r11), bullet_math::FRAC_PI_2, 0.0)
    }
}

/// The relative-orientation information Bullet's `calculateAngleInfo`
/// computes for the generic 6DOF family: the Euler XYZ angles of the
/// relative rotation `trA⁻¹ · trB` and the three normalized angular
/// constraint axes (`m_calculatedAxis`), each of which constrains the
/// corresponding Euler-angle rate.
fn relative_rotation_info(tr_a: &Matrix3, tr_b: &Matrix3) -> (Vector3, [Vector3; 3]) {
    let relative = tr_a.transpose() * *tr_b;
    let euler = euler_xyz(&relative);
    // Bullet: `axis0 = trB.col(0)`, `axis2 = trA.col(2)`, then
    // `ax[1] = axis2×axis0`, `ax[0] = ax[1]×axis2`, `ax[2] = axis0×ax[1]`.
    let axis0 = tr_b.col(0);
    let axis2 = tr_a.col(2);
    let mut axes = [Vector3::ZERO; 3];
    axes[1] = axis2.cross(axis0);
    axes[0] = axes[1].cross(axis2);
    axes[2] = axis0.cross(axes[1]);
    for a in axes.iter_mut() {
        *a = a.normalize();
    }
    (euler, axes)
}

/// Mirror Bullet's accumulated-impulse window `[lower, upper]` (infinities
/// included) through our row convention, where our accumulated impulse is
/// the negative of Bullet's (`λ_ours = −λ_bullet`), so a window `[L, U]`
/// becomes `[−U, −L]`.
fn mirrored_row_limits(lower: Real, upper: Real) -> RowLimits {
    let lower_inf = lower.is_infinite() && lower < 0.0;
    let upper_inf = upper.is_infinite() && upper > 0.0;
    match (lower_inf, upper_inf) {
        (true, true) => RowLimits::Unbounded,
        // Bullet `[−inf, 0]` (a low limit) → our `[0, +inf]`.
        (true, false) => RowLimits::LowerBounded(0.0),
        // Bullet `[0, +inf]` (a high limit) → our `[−inf, 0]`.
        (false, true) => RowLimits::UpperBounded(0.0),
        // A symmetric window (motor): unchanged, since `[−c, c]` mirrors to
        // itself.
        (false, false) => RowLimits::Bilateral((upper - lower) / 2.0),
    }
}

/// A fixed (weld) joint locking all six degrees of freedom between two
/// frames (Bullet's `btFixedConstraint`, which is a `btGeneric6DofConstraint`
/// with every limit zeroed).
///
/// Rows: three linear pivot rows keeping the frame origins coincident
/// (point-to-point style, world axes) plus three angular rows driving the
/// relative orientation back to identity. The angular rows use the locked
/// branch of the generic 6DOF constraint: the Euler angles of `trA⁻¹·trB`
/// driven to zero along the calculated axes (see [`relative_rotation_info`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FixedConstraint {
    /// Constraint frame on body A (pivot + basis in A's COM frame).
    pub frame_a: JointFrame,
    /// Constraint frame on body B (pivot + basis in B's COM frame).
    pub frame_b: JointFrame,
}

impl FixedConstraint {
    /// A fixed joint keeping `frame_a` and `frame_b` coincident.
    pub fn new(frame_a: JointFrame, frame_b: JointFrame) -> Self {
        Self { frame_a, frame_b }
    }

    /// Flatten this fixed joint into its solver rows (the locked-6DOF
    /// recipe): three world-axis linear rows plus three angular rows along
    /// the relative-rotation axes.
    fn prepare_rows(
        &self,
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

        // Three linear pivot rows along the world axes (identical to the
        // point-to-point joint): keep the frame origins coincident.
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

        // Three angular rows locking the relative orientation: the locked
        // 6DOF branch (`lo == hi == 0`, so the violation is the Euler angle
        // itself). Bullet's rotational limit error is `−k·err`, mirrored to
        // our `+k·err` bias; the limits are `[−inf, +inf]` (unbounded).
        let (euler, axes) = relative_rotation_info(&tr_a, &tr_b);
        for i in 0..3 {
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                Vector3::ZERO,
                axes[i],
                Vector3::ZERO,
                axes[i],
                a,
                b,
                k * euler[i],
                RowLimits::Unbounded,
            ));
        }
    }
}

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
const SIMD_EPSILON: Real = {
    #[cfg(not(feature = "f64"))]
    {
        std::f32::EPSILON
    }
    #[cfg(feature = "f64")]
    {
        std::f64::EPSILON
    }
};

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
        let mut q_target_cone = Quaternion::from_rotation_arc(CONE_TWIST_AXIS, v_twisted);
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
                q_target_cone = Quaternion::from_axis_angle(swing_axis, swing_angle);
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
                q_target_twist = Quaternion::from_axis_angle(twist_axis, twist_angle);
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
            let mut q_delta = Quaternion::from_mat3(&r_delta);
            q_delta = q_delta.normalize();
            let swing_axis = q_delta.xyz();
            let len2 = swing_axis.length_squared();
            if len2 > SIMD_EPSILON {
                self.swing_axis = swing_axis / len2.sqrt();
                self.swing_correction = quaternion_angle(&q_delta);
                if self.swing_correction.abs() > SIMD_EPSILON {
                    self.solve_swing_limit = true;
                }
            }
            return;
        }

        // Non-motor branch: decompose the relative rotation into cone (no
        // twist) and pure twist, from B's perspective.
        let q_a = Quaternion::from_mat3(tr_a);
        let q_b = Quaternion::from_mat3(tr_b);
        let mut q_ab = q_b.inverse() * q_a;
        q_ab = q_ab.normalize();
        let mut v_cone_no_twist = q_ab * CONE_TWIST_AXIS;
        v_cone_no_twist = v_cone_no_twist.normalize();
        let mut q_ab_cone = Quaternion::from_rotation_arc(CONE_TWIST_AXIS, v_cone_no_twist);
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
                            let span2 = z.atan2(x);
                            if span2 > self.swing_span2 {
                                x = self.swing_span2.cos();
                                z = self.swing_span2.sin();
                            } else if span2 < -self.swing_span2 {
                                x = self.swing_span2.cos();
                                z = -self.swing_span2.sin();
                            }
                        }
                    }
                } else {
                    // Hinge around the Z axis.
                    if x.abs() > SIMD_EPSILON || y.abs() > SIMD_EPSILON {
                        self.solve_swing_limit = true;
                        if self.swing_span1 >= CONE_TWIST_FIX_THRESH {
                            z = 0.0;
                            let span1 = y.atan2(x);
                            if span1 > self.swing_span1 {
                                x = self.swing_span1.cos();
                                y = self.swing_span1.sin();
                            } else if span1 < -self.swing_span1 {
                                x = self.swing_span1.cos();
                                y = -self.swing_span1.sin();
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
            let mut swing_axis = q_cone.xyz();
            swing_axis = swing_axis.normalize();
            let x_ellipse = swing_axis.y;
            let y_ellipse = -swing_axis.z;
            let mut swing_limit = self.swing_span1;
            if x_ellipse.abs() > SIMD_EPSILON {
                let surface_slope2 = (y_ellipse * y_ellipse) / (x_ellipse * x_ellipse);
                let mut norm = 1.0 / (self.swing_span2 * self.swing_span2);
                norm += surface_slope2 / (self.swing_span1 * self.swing_span1);
                let swing_limit2 = (1.0 + surface_slope2) / norm;
                swing_limit = swing_limit2.sqrt();
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
        let mut twist_axis = q_min_twist.xyz();
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
fn quaternion_angle(q: &Quaternion) -> Real {
    2.0 * q.w.clamp(-1.0, 1.0).acos()
}

/// The rotation of a frame's basis, as a quaternion (Bullet's
/// `btTransform::getRotation`).
impl JointFrame {
    fn basis_quat(&self) -> Quaternion {
        Quaternion::from_mat3(&self.basis)
    }
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
        // World frames and pivots (Bullet's `calculateTransforms`).
        let tr_a = Matrix3::from_quat(a.orientation) * self.frame_a.basis;
        let tr_b = Matrix3::from_quat(b.orientation) * self.frame_b.basis;
        let pivot_a = a.position + a.orientation * self.frame_a.origin;
        let pivot_b = b.position + b.orientation * self.frame_b.origin;
        let delta = pivot_b - pivot_a;
        let ax1_a = tr_a.col(0);
        // Per-axis offsets (m_depth) and limit tests.
        let mut depth = [0.0; 3];
        for i in 0..3 {
            depth[i] = delta.dot(tr_a.col(i));
        }
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
                axis_b0.dot(axis_a1).atan2(axis_b0.dot(axis_a0)),
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
    fn prepare_rows(
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
        for i in 0..3 {
            if self.test_angular_limit_motor(i) {
                let m = self.angular[i];
                // Bullet's bounce velocity: `angVelA·ax1 − angVelB·ax1`.
                let row_vel = a.angular_velocity.dot(axes[i]) - b.angular_velocity.dot(axes[i]);
                self.push_limit_motor_row(
                    body_a,
                    body_b,
                    a,
                    b,
                    inv_dt,
                    axes[i],
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
        ConstraintKind::Fixed(fixed) => {
            fixed.prepare_rows(
                constraint.body_a,
                constraint.body_b,
                bodies,
                inv_dt,
                erp,
                out,
            );
        }
        ConstraintKind::ConeTwist(cone_twist) => {
            cone_twist.prepare_rows(
                constraint.body_a,
                constraint.body_b,
                bodies,
                inv_dt,
                erp,
                out,
            );
        }
        ConstraintKind::Slider(slider) => {
            slider.prepare_rows(
                constraint.body_a,
                constraint.body_b,
                bodies,
                inv_dt,
                erp,
                out,
            );
        }
        ConstraintKind::Generic6Dof(dof) => {
            dof.prepare_rows(
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

    /// A fixed joint welds a box to a static anchor: an initial spin and
    /// sideways push are killed, the relative orientation returns to
    /// identity and the pivot stays on the anchor under gravity.
    #[test]
    fn fixed_joint_locks_a_rigid_assembly() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        // A box hanging 1 m below the anchor; both pivots sit 0.5 above each
        // body's centre, coinciding at (0, 1.5, 0) at assembly.
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        world.add_constraint(Constraint::fixed(
            box_body,
            anchor,
            JointFrame::new(Vector3::new(0.0, 0.5, 0.0), Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(0.0, 1.5, 0.0), Matrix3::IDENTITY),
        ));
        // Kick it: a spin around Z and a sideways velocity.
        world.body_mut(box_body).angular_velocity = Vector3::new(0.0, 0.0, 4.0);
        world.body_mut(box_body).linear_velocity = Vector3::new(2.0, 0.0, 0.0);
        let mut max_rel_angle: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            // The anchor's frame is identity, so the relative orientation is
            // the box's own orientation.
            max_rel_angle = max_rel_angle.max(quaternion_angle(&world.body(box_body).orientation));
        }
        let b = world.body(box_body);
        // The weld holds: the box returns to the anchor's orientation and
        // hangs straight below the pivot at (0, 1, 0).
        assert!(
            max_rel_angle < 0.5,
            "relative orientation wandered during the kick: {max_rel_angle}"
        );
        assert!(
            (b.position - Vector3::new(0.0, 1.0, 0.0)).length() < 0.03,
            "box drifted from below the anchor: {:?}",
            b.position
        );
        assert!(
            b.angular_velocity.length() < 0.05,
            "box still spinning: {:?}",
            b.angular_velocity
        );
        let pivot = b.position + b.orientation * Vector3::new(0.0, 0.5, 0.0);
        assert!(
            (pivot - Vector3::new(0.0, 1.5, 0.0)).length() < 0.03,
            "pivot off the anchor: {:?}",
            pivot
        );
    }

    /// Two boxes welded side by side tumble and land on the ground as one
    /// rigid unit: their relative pose stays fixed even under impact.
    #[test]
    fn fixed_joint_welds_two_dynamic_bodies() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // Two boxes, one above the other with a small gap, welded by a fixed
        // joint whose frames meet halfway between them.
        let a = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.4, 0.4, 0.4)),
            Vector3::new(0.0, 1.2, 0.0),
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.4, 0.4, 0.4)),
            Vector3::new(0.0, 1.6, 0.0),
            1.0,
        ));
        world.add_constraint(Constraint::fixed(
            a,
            b,
            JointFrame::new(Vector3::new(0.0, 0.2, 0.0), Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(0.0, -0.2, 0.0), Matrix3::IDENTITY),
        ));
        // Tumble the pair a little.
        world.body_mut(a).angular_velocity = Vector3::new(1.0, 2.0, 0.0);
        world.body_mut(b).angular_velocity = Vector3::new(1.0, 2.0, 0.0);
        let mut max_rel_angle: Real = 0.0;
        let mut max_pivot_err: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            let ba = world.body(a);
            let bb = world.body(b);
            let rel_quat = Quaternion::from_mat3(&Matrix3::from_quat(bb.orientation)).inverse()
                * Quaternion::from_mat3(&Matrix3::from_quat(ba.orientation));
            max_rel_angle = max_rel_angle.max(quaternion_angle(&rel_quat));
            // The weld residual: the two frame origins must stay coincident
            // (their world positions differ by the rigid offset).
            let pivot_a = ba.position + ba.orientation * Vector3::new(0.0, 0.2, 0.0);
            let pivot_b = bb.position + bb.orientation * Vector3::new(0.0, -0.2, 0.0);
            max_pivot_err = max_pivot_err.max((pivot_b - pivot_a).length());
        }
        assert!(
            max_rel_angle < 0.15,
            "welded pair twisted apart: {max_rel_angle}"
        );
        assert!(
            max_pivot_err < 0.15,
            "welded pair stretched apart: {max_pivot_err}"
        );
    }

    /// A cone-twist joint limits the swing: a bob pinned at the anchor and
    /// spun around Z swings against the cone and the angle between the cone
    /// axes never exceeds the swing span (plus the Baumgarte overshoot).
    #[test]
    fn cone_twist_limits_the_swing() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let bob = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::ZERO,
            1.0,
        ));
        // Cone axis = X on both bodies; both pivots at the origin (the bob's
        // centre is pinned to the anchor). Swing limited to 25°, twist free.
        let cone = Constraint::cone_twist(
            bob,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            0.44,
            0.44,
            -1.0,
        );
        world.add_constraint(cone);
        // Spin the bob around Z: its X axis swings out of the cone.
        world.body_mut(bob).angular_velocity = Vector3::new(0.0, 0.0, 4.0);
        let mut max_swing: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            // The swing angle is the angle between the cone axes (X).
            let ax_a = world.body(anchor).orientation * Vector3::X;
            let ax_b = world.body(bob).orientation * Vector3::X;
            max_swing = max_swing.max(ax_a.dot(ax_b).clamp(-1.0, 1.0).acos());
        }
        assert!(
            max_swing < 0.55,
            "cone exceeded: swing angle {max_swing} rad (limit 0.44)"
        );
        // It actually swung (moved out of the hanging position and back).
        assert!(max_swing > 0.3, "bob barely swung: {max_swing}");
    }

    /// A cone-twist joint with a locked twist span (0) stops the twist: a
    /// spin around the cone axis is killed and the twist angle never grows.
    #[test]
    fn cone_twist_locks_the_twist() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let bob = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::ZERO,
            1.0,
        ));
        // Free swing, locked twist (span 0): the bob can swing but not twist.
        let cone = Constraint::cone_twist(
            bob,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            0.9,
            0.9,
            0.0,
        );
        world.add_constraint(cone);
        // Spin the bob around the cone axis (X).
        world.body_mut(bob).angular_velocity = Vector3::new(6.0, 0.0, 0.0);
        let mut max_twist: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            max_twist = max_twist.max(world.constraint(0).kind().cone_twist().twist_angle());
        }
        assert!(
            max_twist < 0.1,
            "locked twist drifted: {max_twist} rad"
        );
        assert!(
            world.body(bob).angular_velocity.x.abs() < 0.1,
            "bob still twisting: {:?}",
            world.body(bob).angular_velocity
        );
    }

    /// A slider joint (free linear travel, locked rotation): a box pushed
    /// sideways with a spin slides along the axis, stays on the axis, and
    /// keeps its orientation.
    #[test]
    fn slider_slides_along_its_axis() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        ));
        // Slider along X with the default limits (free linear, locked
        // angular); the box centre is 1 m along the axis, inside the limits.
        // The static anchor is the reference body A (the motor and limits
        // act on body B relative to body A, like Bullet).
        world.add_constraint(Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        ));
        // Push it along the axis with a sideways drift and a spin.
        world.body_mut(box_body).linear_velocity = Vector3::new(1.5, 0.5, 0.2);
        world.body_mut(box_body).angular_velocity = Vector3::new(0.0, 0.0, 3.0);
        let mut max_off_axis: Real = 0.0;
        let mut max_rel_angle: Real = 0.0;
        for _ in 0..180 {
            world.step_simulation(DT);
            let p = world.body(box_body).position;
            max_off_axis = max_off_axis.max(p.y.abs()).max(p.z.abs());
            max_rel_angle = max_rel_angle
                .max(quaternion_angle(&world.body(box_body).orientation));
        }
        let p = world.body(box_body).position;
        assert!(p.x > 1.1, "box did not slide: x = {}", p.x);
        assert!(max_off_axis < 0.05, "box drifted off the axis: {max_off_axis}");
        assert!(max_rel_angle < 0.15, "slider allowed rotation: {max_rel_angle}");
    }

    /// A slider with linear limits stops the box at the upper limit: it
    /// slides until the offset reaches the limit and never passes it.
    #[test]
    fn slider_limit_stops_the_slide() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.2, 0.0, 0.0),
            1.0,
        ));
        let mut slider = Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        slider.kind_mut().slider_mut().set_lin_limits(-0.3, 0.3);
        world.add_constraint(slider);
        // Push the box hard along the axis toward the upper limit. The
        // slider limit is a velocity bias (Bullet's `m_constraintError =
        // k·depth`): it stops the box at the limit and rebounds it, so the
        // box can never travel far past the limit, and it coasts back
        // inside once the row deactivates (no friction in this test).
        world.body_mut(box_body).linear_velocity = Vector3::new(4.0, 0.0, 0.0);
        let mut max_x: Real = 0.0;
        let mut limit_fired = false;
        for _ in 0..180 {
            world.step_simulation(DT);
            let s = world.constraint(0).kind().slider();
            limit_fired |= s.solve_lin_limit;
            max_x = max_x.max(world.body(box_body).position.x);
        }
        let final_x = world.body(box_body).position.x;
        // One step of free travel at 4 m/s overshoots to ~0.33; the limit
        // pulls it back from there.
        assert!(max_x < 0.34, "box passed the upper limit: {max_x}");
        assert!(limit_fired, "the limit row never fired");
        // After the rebound the box coasts back inside the range.
        assert!(final_x < 0.3, "box did not return inside the limits: {final_x}");
    }

    /// A slider linear motor drives the box along the axis at the target
    /// velocity.
    #[test]
    fn slider_motor_drives_along_the_axis() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        let mut slider = Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        slider
            .kind_mut()
            .slider_mut()
            .enable_lin_motor(1.5, 100.0);
        world.add_constraint(slider);
        // 1.5 m/s is above the sleeping threshold (0.8), so the motorized
        // box keeps sliding instead of deactivating after 2 s.
        for _ in 0..150 {
            world.step_simulation(DT);
        }
        let p = world.body(box_body).position;
        // 2.5 s at 1.5 m/s along the axis.
        assert!(
            (p.x - 3.75).abs() < 0.3,
            "motor did not drive the box at 1.5 m/s: x = {p:?}"
        );
        assert!(
            world.body(box_body).linear_velocity.x.abs() < 1.8,
            "box velocity off target: {:?}",
            world.body(box_body).linear_velocity
        );
    }

    /// A slider with angular limits stops the box's rotation around the
    /// axis: the angular position stays inside the limits.
    #[test]
    fn slider_angular_limit_stops_the_twist() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        let mut slider = Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        slider.kind_mut().slider_mut().set_ang_limits(-0.4, 0.4);
        world.add_constraint(slider);
        // Spin the box around the slider axis.
        world.body_mut(box_body).angular_velocity = Vector3::new(3.0, 0.0, 0.0);
        let mut max_ang: Real = 0.0;
        for _ in 0..180 {
            world.step_simulation(DT);
            let s = world.constraint(0).kind().slider();
            max_ang = max_ang.max(s.ang_pos.abs());
        }
        assert!(max_ang < 0.5, "box passed the angular limit: {max_ang}");
    }

    /// A fully locked generic 6DOF behaves like a fixed joint: an initial
    /// spin and sideways push are killed, the relative orientation returns
    /// to identity and the pivot stays on the anchor.
    #[test]
    fn generic_6dof_locked_is_rigid() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        // All six limits locked at zero (the default linear limits already
        // lock; lock the angular ones too).
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::new(0.0, 0.5, 0.0), Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(0.0, 1.5, 0.0), Matrix3::IDENTITY),
        );
        dof.kind_mut()
            .generic_6dof_mut()
            .set_angular_limits(Vector3::ZERO, Vector3::ZERO);
        world.add_constraint(dof);
        world.body_mut(box_body).angular_velocity = Vector3::new(0.0, 0.0, 4.0);
        world.body_mut(box_body).linear_velocity = Vector3::new(2.0, 0.0, 0.0);
        let mut max_rel_angle: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            max_rel_angle =
                max_rel_angle.max(quaternion_angle(&world.body(box_body).orientation));
        }
        let b = world.body(box_body);
        assert!(
            max_rel_angle < 0.5,
            "relative orientation wandered during the kick: {max_rel_angle}"
        );
        assert!(
            (b.position - Vector3::new(0.0, 1.0, 0.0)).length() < 0.03,
            "box drifted from below the anchor: {:?}",
            b.position
        );
    }

    /// A generic 6DOF with one angular limit stops the spin around that
    /// axis: the relative Euler angle stays inside the limit.
    #[test]
    fn generic_6dof_angular_limit_stops_the_spin() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        // Linear locked (default), angular X limited to ±0.4, Y/Z free.
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        dof.kind_mut()
            .generic_6dof_mut()
            .set_angular_limits(Vector3::new(-0.4, -1.0, -1.0), Vector3::new(0.4, 1.0, 1.0));
        world.add_constraint(dof);
        // Spin the box around X: the X Euler angle swings against its limit.
        world.body_mut(box_body).angular_velocity = Vector3::new(4.0, 0.0, 0.0);
        let mut max_angle: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            let a = world.constraint(0).kind().generic_6dof().get_angle(0);
            max_angle = max_angle.max(a.abs());
        }
        assert!(max_angle < 0.5, "box passed the angular limit: {max_angle}");
        assert!(max_angle > 0.3, "box barely rotated: {max_angle}");
    }

    /// A generic 6DOF angular motor rotates the joint around the Z euler
    /// axis at the target velocity.
    #[test]
    fn generic_6dof_motor_rotates_an_axis() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        // Angular Z motor at 1 rad/s (axis 5 = angular Z).
        dof.kind_mut().generic_6dof_mut().enable_motor(5, 1.0, 100.0);
        world.add_constraint(dof);
        for _ in 0..180 {
            world.step_simulation(DT);
        }
        let angle = world.constraint(0).kind().generic_6dof().get_angle(2);
        // 3 s at 1 rad/s of relative rotation around the Z euler axis.
        assert!(
            (angle.abs() - 3.0).abs() < 1.0,
            "motor did not rotate the joint at 1 rad/s: angle {angle}"
        );
    }

    /// A generic 6DOF with a linear limit stops the box sliding along that
    /// axis: the pivot offset stays inside the limit.
    #[test]
    fn generic_6dof_linear_limit_stops_the_slide() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.2, 0.0, 0.0),
            1.0,
        ));
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        // Linear X limited to ±0.3, Y/Z locked (default), angular free
        // (default).
        dof.kind_mut()
            .generic_6dof_mut()
            .set_linear_limits(Vector3::new(-0.3, 0.0, 0.0), Vector3::new(0.3, 0.0, 0.0));
        world.add_constraint(dof);
        // Push the box hard along X; the limit is a velocity bias, so the
        // box rebounds and coasts back (no friction, like Bullet).
        world.body_mut(box_body).linear_velocity = Vector3::new(4.0, 0.0, 0.0);
        let mut max_x: Real = 0.0;
        for _ in 0..180 {
            world.step_simulation(DT);
            max_x = max_x.max(world.body(box_body).position.x);
        }
        assert!(max_x < 0.34, "box passed the upper limit: {max_x}");
        assert!(
            world.body(box_body).position.x < 0.3,
            "box did not return inside the limits: {:?}",
            world.body(box_body).position
        );
    }
}
