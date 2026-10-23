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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::{math, Matrix3, Quaternion, Real, Vector3};
use bullet_math::compat::{quat_from_axis_angle, quat_from_mat3, quat_from_rotation_arc, quat_xyz};

use crate::body::RigidBody;
use crate::handles::BodyHandle;
use crate::solver::sequential_impulse::{ConstraintRow, RowLimits};

mod conetwist;
mod fixed;
mod gear;
mod generic6dof;
mod generic6dof_spring2;
mod hinge;
mod p2p;
mod shared;
mod slider;
#[cfg(test)]
mod tests;

pub use conetwist::{ConeTwistConstraint, ConeTwistMotor};
pub use fixed::FixedConstraint;
pub use gear::GearConstraint;
pub use generic6dof::{Generic6DofConstraint, RotationalLimitMotor, TranslationalLimitMotor};
pub use generic6dof_spring2::{
    Generic6DofSpring2Constraint, RotateOrder, RotationalLimitMotor2, TranslationalLimitMotor2,
};
pub use hinge::{HingeConstraint, HingeMotor};
pub use p2p::Point2PointConstraint;
pub use slider::{SliderConstraint, SliderMotor};

pub(crate) use shared::euler_xyz;
use shared::normalize_angle;

/// The specific kind of a [`Constraint`] and its parameters (Bullet's
/// constraint subclasses).
// The variants carry their full parameter payload by value (cloneable,
// comparable) matching the engine's value-semantics style; the f64 build
// doubles the largest variant's size past clippy's threshold.
#[allow(clippy::large_enum_variant)]
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
    /// Generic six-degree-of-freedom joint with per-axis springs, servo
    /// motors, elastic limits and configurable Euler order (Bullet's
    /// `btGeneric6DofSpring2Constraint`, the 2014 rewrite). The joint type
    /// PMX/MMD-style model physics is built on.
    Generic6DofSpring2(Generic6DofSpring2Constraint),
    /// Hinge-2 (wheel) joint: a spring-2 configuration with steering about
    /// `axis1`, free spin about `axis2` and the suspension spring along
    /// `axis1` (Bullet's `btHinge2Constraint`, a
    /// `btGeneric6DofSpring2Constraint` subclass upstream — the rows are
    /// the spring-2 rows). Built by [`PhysicsWorld::add_hinge2`].
    Hinge2(Generic6DofSpring2Constraint),
    /// Universal (shaft) joint: two perpendicular bend axes with all other
    /// degrees of freedom locked (Bullet's `btUniversalConstraint`, a
    /// `btGeneric6DofConstraint` subclass upstream — the rows are the
    /// classic 6DOF rows). Built by [`PhysicsWorld::add_universal`].
    Universal(Generic6DofConstraint),
    /// Gear coupling of two rotation axes (Bullet's `btGearConstraint`):
    /// one angular-only bilateral row `ω_A·axisA + ratio·ω_B·axisB = 0`
    /// with zero bias. Built by [`Constraint::gear`].
    Gear(GearConstraint),
}

impl ConstraintKind {
    /// Name of this kind, used by the accessors' panic messages.
    fn kind_name(&self) -> &'static str {
        match self {
            ConstraintKind::Point2Point(_) => "point-to-point",
            ConstraintKind::Hinge(_) => "hinge",
            ConstraintKind::Fixed(_) => "fixed",
            ConstraintKind::ConeTwist(_) => "cone-twist",
            ConstraintKind::Slider(_) => "slider",
            ConstraintKind::Generic6Dof(_) => "generic 6DOF",
            ConstraintKind::Generic6DofSpring2(_) => "generic 6DOF spring 2",
            ConstraintKind::Hinge2(_) => "hinge 2",
            ConstraintKind::Universal(_) => "universal",
            ConstraintKind::Gear(_) => "gear",
        }
    }

    /// Access to the [`Point2PointConstraint`] parameters (panics on any
    /// other kind).
    pub fn point2point(&self) -> &Point2PointConstraint {
        match self {
            ConstraintKind::Point2Point(p2p) => p2p,
            other => panic!(
                "not a point-to-point constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Mutable access to the [`Point2PointConstraint`] parameters (panics on
    /// any other kind).
    pub fn point2point_mut(&mut self) -> &mut Point2PointConstraint {
        match self {
            ConstraintKind::Point2Point(p2p) => p2p,
            other => panic!(
                "not a point-to-point constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Access to the [`HingeConstraint`] parameters (panics on any other
    /// kind).
    pub fn hinge(&self) -> &HingeConstraint {
        match self {
            ConstraintKind::Hinge(hinge) => hinge,
            other => panic!("not a hinge constraint (found {})", other.kind_name()),
        }
    }

    /// Mutable access to the [`HingeConstraint`] parameters (panics on any
    /// other kind).
    pub fn hinge_mut(&mut self) -> &mut HingeConstraint {
        match self {
            ConstraintKind::Hinge(hinge) => hinge,
            other => panic!("not a hinge constraint (found {})", other.kind_name()),
        }
    }

    /// Access to the [`FixedConstraint`] parameters (panics on any other
    /// kind).
    pub fn fixed(&self) -> &FixedConstraint {
        match self {
            ConstraintKind::Fixed(fixed) => fixed,
            other => panic!("not a fixed constraint (found {})", other.kind_name()),
        }
    }

    /// Mutable access to the [`FixedConstraint`] parameters (panics on any
    /// other kind).
    pub fn fixed_mut(&mut self) -> &mut FixedConstraint {
        match self {
            ConstraintKind::Fixed(fixed) => fixed,
            other => panic!("not a fixed constraint (found {})", other.kind_name()),
        }
    }

    /// Access to the [`ConeTwistConstraint`] parameters (panics on any other
    /// kind).
    pub fn cone_twist(&self) -> &ConeTwistConstraint {
        match self {
            ConstraintKind::ConeTwist(ct) => ct,
            other => panic!(
                "not a cone-twist constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Mutable access to the [`ConeTwistConstraint`] parameters (panics on
    /// any other kind).
    pub fn cone_twist_mut(&mut self) -> &mut ConeTwistConstraint {
        match self {
            ConstraintKind::ConeTwist(ct) => ct,
            other => panic!(
                "not a cone-twist constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Access to the [`SliderConstraint`] parameters (panics on any other
    /// kind).
    pub fn slider(&self) -> &SliderConstraint {
        match self {
            ConstraintKind::Slider(slider) => slider,
            other => panic!("not a slider constraint (found {})", other.kind_name()),
        }
    }

    /// Mutable access to the [`SliderConstraint`] parameters (panics on any
    /// other kind).
    pub fn slider_mut(&mut self) -> &mut SliderConstraint {
        match self {
            ConstraintKind::Slider(slider) => slider,
            other => panic!("not a slider constraint (found {})", other.kind_name()),
        }
    }

    /// Access to the [`Generic6DofConstraint`] parameters (panics on any
    /// other kind).
    pub fn generic_6dof(&self) -> &Generic6DofConstraint {
        match self {
            ConstraintKind::Generic6Dof(dof) => dof,
            other => panic!(
                "not a generic 6DOF constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Mutable access to the [`Generic6DofConstraint`] parameters (panics on
    /// any other kind).
    pub fn generic_6dof_mut(&mut self) -> &mut Generic6DofConstraint {
        match self {
            ConstraintKind::Generic6Dof(dof) => dof,
            other => panic!(
                "not a generic 6DOF constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Access to the [`Generic6DofSpring2Constraint`] parameters (panics on
    /// any other kind).
    pub fn generic_6dof_spring2(&self) -> &Generic6DofSpring2Constraint {
        match self {
            ConstraintKind::Generic6DofSpring2(dof) => dof,
            other => panic!(
                "not a generic 6DOF spring 2 constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Mutable access to the [`Generic6DofSpring2Constraint`] parameters
    /// (panics on any other kind).
    pub fn generic_6dof_spring2_mut(&mut self) -> &mut Generic6DofSpring2Constraint {
        match self {
            ConstraintKind::Generic6DofSpring2(dof) => dof,
            other => panic!(
                "not a generic 6DOF spring 2 constraint (found {})",
                other.kind_name()
            ),
        }
    }

    /// Access to the [`Hinge2`][ConstraintKind::Hinge2] joint's spring-2
    /// parameters (panics on any other kind).
    pub fn hinge2(&self) -> &Generic6DofSpring2Constraint {
        match self {
            ConstraintKind::Hinge2(dof) => dof,
            other => panic!("not a hinge-2 constraint (found {})", other.kind_name()),
        }
    }

    /// Mutable access to the [`Hinge2`][ConstraintKind::Hinge2] joint's
    /// spring-2 parameters (panics on any other kind).
    pub fn hinge2_mut(&mut self) -> &mut Generic6DofSpring2Constraint {
        match self {
            ConstraintKind::Hinge2(dof) => dof,
            other => panic!("not a hinge-2 constraint (found {})", other.kind_name()),
        }
    }

    /// Access to the [`Universal`][ConstraintKind::Universal] joint's 6DOF
    /// parameters (panics on any other kind).
    pub fn universal(&self) -> &Generic6DofConstraint {
        match self {
            ConstraintKind::Universal(dof) => dof,
            other => panic!("not a universal constraint (found {})", other.kind_name()),
        }
    }

    /// Mutable access to the [`Universal`][ConstraintKind::Universal]
    /// joint's 6DOF parameters (panics on any other kind).
    pub fn universal_mut(&mut self) -> &mut Generic6DofConstraint {
        match self {
            ConstraintKind::Universal(dof) => dof,
            other => panic!("not a universal constraint (found {})", other.kind_name()),
        }
    }

    /// Access to the [`Gear`][ConstraintKind::Gear] joint's parameters
    /// (panics on any other kind).
    pub fn gear(&self) -> &GearConstraint {
        match self {
            ConstraintKind::Gear(gear) => gear,
            other => panic!("not a gear constraint (found {})", other.kind_name()),
        }
    }

    /// Mutable access to the [`Gear`][ConstraintKind::Gear] joint's
    /// parameters (panics on any other kind).
    pub fn gear_mut(&mut self) -> &mut GearConstraint {
        match self {
            ConstraintKind::Gear(gear) => gear,
            other => panic!("not a gear constraint (found {})", other.kind_name()),
        }
    }
}

/// A joint between two bodies (Bullet's `btTypedConstraint`). The common
/// flags (`enabled`, `disable_collisions`) live here; the geometry and tuning
/// live in [`ConstraintKind`].
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    kind: ConstraintKind,
    /// Body handles (the constructors fix the math direction order at
    /// creation).
    body_a: BodyHandle,
    body_b: BodyHandle,
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
        body_a: BodyHandle,
        body_b: BodyHandle,
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
        body_a: BodyHandle,
        body_b: BodyHandle,
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

    pub fn body_a(&self) -> BodyHandle {
        self.body_a
    }

    pub fn body_b(&self) -> BodyHandle {
        self.body_b
    }

    /// A fixed (weld) joint: the frames `frame_a` (in body A's
    /// center-of-mass frame) and `frame_b` (in body B's) stay coincident —
    /// both the pivot and the relative orientation are locked, removing all
    /// six degrees of freedom (Bullet's `btFixedConstraint`).
    pub fn fixed(
        body_a: BodyHandle,
        body_b: BodyHandle,
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
        body_a: BodyHandle,
        body_b: BodyHandle,
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
    pub fn slider(
        body_a: BodyHandle,
        body_b: BodyHandle,
        frame_a: JointFrame,
        frame_b: JointFrame,
    ) -> Self {
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
        body_a: BodyHandle,
        body_b: BodyHandle,
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

    /// A generic six-degree-of-freedom joint with per-axis springs, servo
    /// motors, elastic limits and a configurable Euler rotation order
    /// (Bullet's `btGeneric6DofSpring2Constraint`). Bullet's defaults apply —
    /// linear axes locked, angular axes free — so a fresh joint behaves like
    /// a ball joint; set limits/springs to taste.
    pub fn generic_6dof_spring2(
        body_a: BodyHandle,
        body_b: BodyHandle,
        frame_a: JointFrame,
        frame_b: JointFrame,
        rotate_order: RotateOrder,
    ) -> Self {
        Self {
            kind: ConstraintKind::Generic6DofSpring2(Generic6DofSpring2Constraint::new(
                frame_a,
                frame_b,
                rotate_order,
            )),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    /// A gear coupling of two rotation axes (Bullet's `btGearConstraint`):
    /// the axes are in each body's center-of-mass frame, the ratio relates
    /// the angular velocities (`ω_A·axisA + ratio·ω_B·axisB = 0`, so
    /// parallel axes mesh like external gears). Add it with
    /// `PhysicsWorld::add_constraint` — the constructor reads no body state.
    pub fn gear(
        body_a: BodyHandle,
        body_b: BodyHandle,
        axis_a: Vector3,
        axis_b: Vector3,
        ratio: Real,
    ) -> Self {
        Self {
            kind: ConstraintKind::Gear(GearConstraint::new(axis_a, axis_b, ratio)),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    pub fn kind(&self) -> &ConstraintKind {
        &self.kind
    }

    /// Crate-internal assembly for the world-level factories whose joints
    /// need body poses at construction (hinge-2 / universal).
    pub(crate) fn from_kind(kind: ConstraintKind, body_a: BodyHandle, body_b: BodyHandle) -> Self {
        Self {
            kind,
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    pub fn kind_mut(&mut self) -> &mut ConstraintKind {
        &mut self.kind
    }

    // Parameter accessors delegating to the kind, so tuning code stays one
    // dispatch away instead of `kind_mut().xxx_mut()` (Bullet calls the
    // setters directly on its constraint subclasses). Reads go through
    // `kind()` — each kind's constructor owns that name here. All of these
    // panic with the found kind on a mismatch.

    /// Mutable access to the [`Point2PointConstraint`] parameters (Bullet's
    /// `btPoint2PointConstraint` setters, e.g. `setMaxImpulseClamp` / ERP).
    pub fn point2point_mut(&mut self) -> &mut Point2PointConstraint {
        self.kind_mut().point2point_mut()
    }

    /// Mutable access to the [`HingeConstraint`] parameters (limits, motor).
    pub fn hinge_mut(&mut self) -> &mut HingeConstraint {
        self.kind_mut().hinge_mut()
    }

    /// Mutable access to the [`FixedConstraint`] parameters.
    pub fn fixed_mut(&mut self) -> &mut FixedConstraint {
        self.kind_mut().fixed_mut()
    }

    /// Mutable access to the [`ConeTwistConstraint`] parameters (limits,
    /// motor).
    pub fn cone_twist_mut(&mut self) -> &mut ConeTwistConstraint {
        self.kind_mut().cone_twist_mut()
    }

    /// Mutable access to the [`SliderConstraint`] parameters (limits,
    /// motors).
    pub fn slider_mut(&mut self) -> &mut SliderConstraint {
        self.kind_mut().slider_mut()
    }

    /// Mutable access to the [`Generic6DofConstraint`] parameters (limits,
    /// motors).
    pub fn generic_6dof_mut(&mut self) -> &mut Generic6DofConstraint {
        self.kind_mut().generic_6dof_mut()
    }

    /// Mutable access to the [`Generic6DofSpring2Constraint`] parameters
    /// (limits, motors, springs, servos).
    pub fn generic_6dof_spring2_mut(&mut self) -> &mut Generic6DofSpring2Constraint {
        self.kind_mut().generic_6dof_spring2_mut()
    }

    /// Mutable access to the [`Hinge2`][ConstraintKind::Hinge2] joint's
    /// spring-2 parameters (suspension spring, steering limits, motors).
    pub fn hinge2_mut(&mut self) -> &mut Generic6DofSpring2Constraint {
        self.kind_mut().hinge2_mut()
    }

    /// Mutable access to the [`Universal`][ConstraintKind::Universal]
    /// joint's 6DOF parameters (limits, motors).
    pub fn universal_mut(&mut self) -> &mut Generic6DofConstraint {
        self.kind_mut().universal_mut()
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable the constraint (Bullet's `setEnabled`); disabled
    /// constraints produce no solver rows. Collision suppression between
    /// their bodies is independent of `enabled` — Bullet's ignore list
    /// (installed by `addConstraint(..., true)`) keeps suppressing until
    /// the constraint is removed or `set_disable_collisions(false)` clears
    /// it.
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

    /// The rotation of a frame's basis, as a quaternion (Bullet's
    /// `btTransform::getRotation`).
    pub(crate) fn basis_quat(&self) -> Quaternion {
        quat_from_mat3(&self.basis)
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

/// Flatten a [`Constraint`] into its solver rows for this step (Bullet's
/// `getInfo1` + `getInfo2` + `convertJoint`). Rows are appended to `out`.
/// `inv_dt` is `1 / dt` (Bullet's `info2.fps`); `erp` is the global ERP
/// (Bullet's `infoGlobal.m_erp`, 0.2); `global_cfm` is the global compliance
/// (Bullet's `infoGlobal.m_globalCfm`, 0 — the default row cfm for joint
/// rows; only the spring-2 kind consumes it today). Joints with per-step
/// state (the hinge angle and limit test) update it here, before the solve.
pub fn prepare_rows(
    constraint: &mut Constraint,
    bodies: &[RigidBody],
    inv_dt: Real,
    erp: Real,
    global_cfm: Real,
    out: &mut Vec<ConstraintRow>,
) {
    if !constraint.enabled {
        return;
    }
    // Solver rows address the bodies by world slot (the row pipeline is
    // handle-agnostic).
    let (body_a_idx, body_b_idx) = (constraint.body_a.index(), constraint.body_b.index());
    match &mut constraint.kind {
        ConstraintKind::Point2Point(p2p) => {
            let a = &bodies[body_a_idx];
            let b = &bodies[body_b_idx];
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
            // `m_J1linearAxis` = ±e_j with the skew-symmetric angular
            // jacobians built from the lever arms).
            for axis in 0..3 {
                let mut normal = Vector3::ZERO;
                normal[axis] = 1.0;
                out.push(ConstraintRow::prepare(
                    body_a_idx,
                    body_b_idx,
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
            hinge.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, out);
        }
        ConstraintKind::Fixed(fixed) => {
            fixed.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, out);
        }
        ConstraintKind::ConeTwist(cone_twist) => {
            cone_twist.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, out);
        }
        ConstraintKind::Slider(slider) => {
            slider.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, out);
        }
        ConstraintKind::Generic6Dof(dof) => {
            dof.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, out);
        }
        ConstraintKind::Generic6DofSpring2(dof) => {
            dof.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, global_cfm, out);
        }
        ConstraintKind::Hinge2(dof) => {
            dof.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, global_cfm, out);
        }
        ConstraintKind::Universal(dof) => {
            dof.prepare_rows(body_a_idx, body_b_idx, bodies, inv_dt, erp, out);
        }
        ConstraintKind::Gear(gear) => {
            gear.prepare_rows(body_a_idx, body_b_idx, bodies, out);
        }
    }
}
