//! Multibody constraints — port of `BulletDynamics/Featherstone`'s
//! `btMultiBodyConstraint` subclasses (`btMultiBodyPoint2Point`,
//! `btMultiBodyFixedConstraint`, `btMultiBodySliderConstraint`,
//! `btMultiBodyJointMotor`, `btMultiBodyGearConstraint`,
//! `btMultiBodyJointLimitConstraint`, `btMultiBodySphericalJointLimit`,
//! `btMultiBodySphericalJointMotor`).
//!
//! A [`MultiBodyConstraint`] couples one link of a
//! [`crate::multibody::MultiBody`] to a link of another multibody (or to the
//! base of the same/different multibody via `link = None`), or — for the
//! `*_to_rigid` constructors — to a rigid body sitting on side B (Bullet's
//! `btMultiBodyPoint2Point(btMultiBody*, int, btRigidBody*, ...)` family;
//! `m_rigidBodyA` has no public constructor upstream and is not ported).
//! Every step the
//! constraint emits a set of 1D rows ([`SolverRow`]); each row carries the
//! generalized jacobian of the constrained DOFs on both bodies, the velocity
//! response of each body to a unit impulse along the row (computed through
//! [`MultiBody::calc_acceleration_deltas`], which reuses the ABA
//! factorization), the effective-mass inverse `jac_diag_ab_inv` and the
//! Baumgarte/velocity RHS. The solver then runs projected Gauss-Seidel over
//! the rows and writes the accumulated impulses back into the generalized
//! velocities. A rigid side B applies its impulses directly to the body's
//! real velocities during the iterations (like the mixed contact rows).
//!
//! Deviation from Bullet: our rigid bodies take gravity into their real
//! velocities before the solve, so the row assembly sees it; Bullet's
//! multibody-vs-rigidbody rows read `rb->getLinearVelocity()` *without* the
//! current gravity step (unlike its rigid-rigid joint rows, which add
//! `m_externalForceImpulse`), leaving a per-frame gravity leak that decays
//! geometrically (~1.4 cm sag on a welded ball, f32 and f64 alike) — not
//! ported, see `docs/bullet-compare.md` § "Multibody mb-vs-rb constraint
//! probes".

use bullet_math::{math, Matrix3, Quaternion, Real, Vector3};

use super::body::{MultiBody, MultiBodyScratch};
use super::link::JointType;
use crate::body::RigidBody;
use crate::handles::{BodyHandle, MultiBodyHandle};

/// Bullet's default `m_splitImpulsePenetrationThreshold`: joint-limit rows
/// only combine their positional and velocity errors while the violation is
/// shallower than this; deeper violations fall into the split-impulse branch
/// (positional error dropped — the multibody solver ignores
/// `m_rhsPenetration`, like Bullet).
const SPLIT_PENETRATION_THRESHOLD: Real = -0.04;

/// The rigid side-B payload of a [`SolverRow`] (the per-row fields Bullet
/// stores on `btMultiBodySolverConstraint` when side B is a `btRigidBody`).
/// Like the mixed contact rows, the rigid side's projection is snapshotted
/// at row-assembly time; the PGS iteration then reads the side's **impulse
/// delta** (current projection minus the snapshot) because the RHS already
/// carries the assembly-time relative velocity.
#[derive(Debug, Clone, Copy)]
pub struct RigidSideB {
    /// Index into the world's rigid body list.
    pub body: usize,
    /// Inertia-scaled angular response `invI · (−torqueAxis1)` (Bullet's
    /// `m_angularComponentB`).
    pub angular_component: Vector3,
    /// Inverse mass.
    pub inv_mass: Real,
    /// Row projection (`contactNormal2·v + relpos2CrossNormal·ω`) at
    /// row-assembly time.
    pub initial_proj: Real,
}

/// One solver row of a multibody constraint (Bullet's
/// `btMultiBodySolverConstraint`): a 1D constraint along a generalized axis
/// between body A's link `link_a` and body B's link `link_b`.
pub struct SolverRow {
    /// Index into `SolverData::jacobians` / `delta_velocities_unit_impulse`
    /// for body A's block (`6 + num_dofs_a` entries).
    pub jac_a_index: usize,
    /// Index into `SolverData::delta_velocities` for body A's accumulated
    /// response block (`-1` until assigned, Bullet's `m_deltaVelAindex`).
    pub delta_vel_a_index: i32,
    /// Same for body B (unused when `rigid_b` is set).
    pub jac_b_index: usize,
    pub delta_vel_b_index: i32,
    /// Number of DOFs (6 + joints) of each body (0 for a rigid side B).
    pub ndof_a: usize,
    pub ndof_b: usize,
    /// Lever-arm × normal / angular normal used for feedback writeback
    /// (Bullet's `m_relpos1CrossNormal` / `m_contactNormal1`).
    pub rel_pos1_cross_normal: Vector3,
    pub contact_normal1: Vector3,
    pub rel_pos2_cross_normal: Vector3,
    pub contact_normal2: Vector3,
    /// Inverse effective mass `1/(J·M⁻¹·Jᵀ)` along the row (0 disables the
    /// row on singularities, Bullet's `m_jacDiagABInv`).
    pub jac_diag_ab_inv: Real,
    /// Velocity target (`m_rhs`).
    pub rhs: Real,
    /// Constraint force mixing (always 0 for multibody rows).
    pub cfm: Real,
    /// Accumulated impulse bounds.
    pub lower_limit: Real,
    pub upper_limit: Real,
    /// Accumulated impulse (zero-initialized; Bullet does not warm-start
    /// multibody rows unless `SOLVER_USE_ARTICULATED_WARMSTARTING`, which is
    /// off by default).
    pub applied_impulse: Real,
    /// Indices of the two multibodies inside the world (side B invalid when
    /// `rigid_b` is set).
    pub multi_body_a: usize,
    pub multi_body_b: usize,
    /// Constrained links (`None` = the base).
    pub link_a: Option<usize>,
    pub link_b: Option<usize>,
    /// Rigid-body payload of side B when the constraint couples a multibody
    /// to a rigid body (Bullet's `m_multiBodyB == 0` branch).
    pub rigid_b: Option<RigidSideB>,
    /// The owning constraint and row, for impulse writeback.
    pub org_constraint: usize,
    pub org_dof_index: usize,
}

/// Shared solver scratch for one solve (Bullet's
/// `btMultiBodyJacobianData`): the flattened jacobian pool, the per-row unit
/// impulse responses and the per-body accumulated delta velocities.
#[derive(Default)]
pub struct SolverData {
    /// Flattened jacobians, one `6 + dof_count` block per body per row.
    pub jacobians: Vec<Real>,
    /// Flattened velocity responses to a unit impulse along each row's
    /// jacobian, one block per body per row (parallel to `jacobians`).
    pub delta_velocities_unit_impulse: Vec<Real>,
    /// Accumulated velocity deltas per constrained body (indexed by the
    /// companion ids), coupling the rows during the PGS iterations.
    pub delta_velocities: Vec<Real>,
}

impl SolverData {
    pub fn reset(&mut self) {
        self.jacobians.clear();
        self.delta_velocities_unit_impulse.clear();
        self.delta_velocities.clear();
    }
}

/// Parameter block of a point-to-point (ball-socket) constraint (Bullet's
/// `btMultiBodyPoint2Point`).
#[derive(Debug, Clone)]
pub struct MultiBodyPoint2Point {
    /// Pivot point in link A's local frame.
    pub pivot_in_a: Vector3,
    /// Pivot point in link B's local frame.
    pub pivot_in_b: Vector3,
}

/// Parameter block of a fixed (weld) constraint (Bullet's
/// `btMultiBodyFixedConstraint`): the pivots coincide and the frames align.
#[derive(Debug, Clone)]
pub struct MultiBodyFixedConstraint {
    pub pivot_in_a: Vector3,
    pub pivot_in_b: Vector3,
    /// Frame attached to link A (local frame).
    pub frame_in_a: Matrix3,
    /// Frame attached to link B (local frame).
    pub frame_in_b: Matrix3,
}

/// Parameter block of a slider constraint (Bullet's
/// `btMultiBodySliderConstraint`): the bodies slide along `joint_axis`, with
/// the other two linear and all three angular DOFs locked.
#[derive(Debug, Clone)]
pub struct MultiBodySliderConstraint {
    pub pivot_in_a: Vector3,
    pub pivot_in_b: Vector3,
    pub frame_in_a: Matrix3,
    pub frame_in_b: Matrix3,
    /// Slide axis in link A's local frame.
    pub joint_axis: Vector3,
}

/// Parameter block of a 1-DOF joint motor (Bullet's
/// `btMultiBodyJointMotor`): drives the joint velocity of a link toward
/// `desired_velocity` (with an optional PD position term), clamped to
/// `±max_motor_impulse`.
#[derive(Debug, Clone)]
pub struct MultiBodyJointMotor {
    /// Which DOF of the link's joint is driven.
    pub link_dof: usize,
    pub desired_velocity: Real,
    pub desired_position: Real,
    /// Velocity gain (Bullet's `m_kd`).
    pub kd: Real,
    /// Position gain (Bullet's `m_kp`).
    pub kp: Real,
    /// Position-stabilization ERP for the motor (Bullet's `m_erp`, default
    /// 1).
    pub erp: Real,
    /// Maximum RHS magnitude (Bullet's `m_rhsClamp`, default infinity).
    pub rhs_clamp: Real,
}

/// Parameter block of a gear constraint (Bullet's
/// `btMultiBodyGearConstraint`): couples the joint of link A on body A to
/// the joint of link B on body B with `q̇_a + ratio·q̇_b = 0` (plus the
/// optional auxiliary-link velocity).
#[derive(Debug, Clone)]
pub struct MultiBodyGearConstraint {
    pub gear_ratio: Real,
    /// Optional auxiliary link on body A whose velocity is added to link A's
    /// (Bullet's `m_gearAuxLink`, default `None`).
    pub gear_aux_link: Option<usize>,
    /// Position error reduction (Bullet's `m_erp`, default 0 = velocity
    /// only).
    pub erp: Real,
    /// Target relative joint position (Bullet's `m_relativePositionTarget`).
    pub relative_position_target: Real,
}

/// Parameter block of a 1-DOF joint limit (Bullet's
/// `btMultiBodyJointLimitConstraint`): clamps the position of a revolute or
/// prismatic joint to `[lower_bound, upper_bound]` with two unilateral rows.
#[derive(Debug, Clone)]
pub struct MultiBodyJointLimit {
    pub lower_bound: Real,
    pub upper_bound: Real,
}

/// Parameter block of a spherical joint limit (Bullet's
/// `btMultiBodySphericalJointLimit`): keeps a spherical joint's swing angles
/// within `swingx_range`/`swingy_range` and its twist within `twist_range`
/// (three unilateral rows along the link frame's world columns). The target
/// velocity/position gains (`kd`/`kp`) and per-DOF impulse limits are only
/// consulted when `use_multi_dof_params` is set (Bullet's
/// `m_use_multi_dof_params`); otherwise the scalar `max_applied_impulse` and
/// the first gain are used.
#[derive(Debug, Clone)]
pub struct MultiBodySphericalJointLimit {
    pub swingx_range: Real,
    pub swingy_range: Real,
    pub twist_range: Real,
    /// Per-axis velocity target (Bullet's `m_desiredVelocity`; the current
    /// Bullet row builder does not feed it into the rows, kept for API
    /// fidelity).
    pub desired_velocity: Vector3,
    /// Desired joint orientation (Bullet's `m_desiredPosition`, unused by the
    /// limit rows).
    pub desired_position: Quaternion,
    pub use_multi_dof_params: bool,
    /// Per-axis velocity gain (Bullet's `m_kd`, default `(1,1,1)`).
    pub kd: Vector3,
    /// Per-axis position gain (Bullet's `m_kp`, default `(0.2,0.2,0.2)`).
    pub kp: Vector3,
    pub erp: Real,
    pub rhs_clamp: Real,
    pub max_applied_impulse_multidof: Vector3,
}

/// Parameter block of a spherical joint motor (Bullet's
/// `btMultiBodySphericalJointMotor`): drives a spherical joint's orientation
/// toward `desired_position` with three bilateral rows along the link frame's
/// world columns, each damped by `damping[i]` and clamped to
/// `±max_applied_impulse`.
#[derive(Debug, Clone)]
pub struct MultiBodySphericalJointMotor {
    /// Per-axis velocity target (Bullet's `m_desiredVelocity`; the current
    /// Bullet row builder does not feed it into the rows, kept for API
    /// fidelity).
    pub desired_velocity: Vector3,
    /// Desired joint orientation (Bullet's `m_desiredPosition`, default
    /// identity).
    pub desired_position: Quaternion,
    pub use_multi_dof_params: bool,
    pub kd: Vector3,
    pub kp: Vector3,
    pub erp: Real,
    pub rhs_clamp: Real,
    pub max_applied_impulse_multidof: Vector3,
    /// Per-axis velocity damping applied to the row's velocity term
    /// (Bullet's `m_damping`, default `(1,1,1)`).
    pub damping: Vector3,
}

/// The kind of a [`MultiBodyConstraint`] and its parameters.
#[derive(Debug, Clone)]
pub enum MultiBodyConstraintKind {
    Point2Point(MultiBodyPoint2Point),
    Fixed(MultiBodyFixedConstraint),
    Slider(MultiBodySliderConstraint),
    JointMotor(MultiBodyJointMotor),
    Gear(MultiBodyGearConstraint),
    JointLimit(MultiBodyJointLimit),
    SphericalJointLimit(MultiBodySphericalJointLimit),
    SphericalJointMotor(MultiBodySphericalJointMotor),
}

/// A constraint between two multibodies (or two links of the same body),
/// or between a multibody (side A) and a rigid body (side B, Bullet's
/// `btMultiBodyPoint2Point(btMultiBody*, int, btRigidBody*, ...)` family).
pub struct MultiBodyConstraint {
    pub(crate) kind: MultiBodyConstraintKind,
    pub(crate) body_a: MultiBodyHandle,
    /// Side-B multibody handle. A placeholder whenever side B is a rigid
    /// body — every side-B multibody read is gated by `rigid_body_b`
    /// (Bullet's `m_multiBodyB == 0` branch, where the field is null).
    pub(crate) body_b: MultiBodyHandle,
    /// Rigid-body handle of side B when the constraint is multibody-vs-
    /// rigidbody (Bullet's `m_rigidBodyB`; `m_rigidBodyA` has no public
    /// constructor upstream and is not ported).
    pub(crate) rigid_body_b: Option<BodyHandle>,
    pub(crate) link_a: Option<usize>,
    pub(crate) link_b: Option<usize>,
    pub(crate) num_rows: usize,
    pub(crate) jac_size_a: usize,
    pub(crate) jac_size_both: usize,
    /// Whether the row window is unilateral (Bullet's `m_isUnilateral`; the
    /// motor is unilateral, the others bilateral).
    pub(crate) is_unilateral: bool,
    /// Per-row impulse clamp (Bullet's `m_maxAppliedImpulse`, default 100).
    pub(crate) max_applied_impulse: Real,
    /// Data block: `num_rows` accumulated impulses, then `num_rows ·
    /// jac_size_both` jacobian entries, then `num_rows` positions.
    pub(crate) data: Vec<Real>,
    pub(crate) pos_offset: usize,
    pub(crate) num_dofs_finalized: usize,
}

impl MultiBodyConstraint {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: MultiBodyConstraintKind,
        body_a: MultiBodyHandle,
        body_b: MultiBodyHandle,
        rigid_b: Option<BodyHandle>,
        link_a: Option<usize>,
        link_b: Option<usize>,
        num_rows: usize,
        is_unilateral: bool,
        bodies: &[MultiBody],
    ) -> Self {
        let mut c = Self {
            kind,
            body_a,
            body_b,
            rigid_body_b: rigid_b,
            link_a,
            link_b,
            num_rows,
            jac_size_a: 0,
            jac_size_both: 0,
            is_unilateral,
            max_applied_impulse: 100.0,
            data: vec![0.0; num_rows],
            pos_offset: 0,
            num_dofs_finalized: 0,
        };
        c.update_jacobian_sizes(bodies);
        c
    }

    fn update_jacobian_sizes(&mut self, bodies: &[MultiBody]) {
        self.jac_size_a = 6 + bodies[self.body_a.index()].num_dofs();
        // A rigid side B has no generalized jacobian block (Bullet's
        // `updateJacobianSizes` with `m_bodyB == 0`).
        self.jac_size_both = match self.rigid_body_b {
            Some(_) => self.jac_size_a,
            None => self.jac_size_a + 6 + bodies[self.body_b.index()].num_dofs(),
        };
    }

    /// Allocate the data block (impulses + jacobians + positions), Bullet's
    /// `allocateJacobiansMultiDof`.
    fn allocate_jacobians(&mut self, bodies: &[MultiBody]) {
        self.update_jacobian_sizes(bodies);
        self.pos_offset = (1 + self.jac_size_both) * self.num_rows;
        self.data.resize((2 + self.jac_size_both) * self.num_rows, 0.0);
    }

    /// The jacobian block of body A for `row` (immutable).
    pub fn jacobian_a(&self, row: usize) -> &[Real] {
        let start = self.num_rows + row * self.jac_size_both;
        &self.data[start..start + self.jac_size_a]
    }

    /// The jacobian block of body B for `row`.
    pub fn jacobian_b(&self, row: usize) -> &[Real] {
        let start = self.num_rows + row * self.jac_size_both + self.jac_size_a;
        &self.data[start..start + self.jac_size_both - self.jac_size_a]
    }

    fn jacobian_a_mut(&mut self, row: usize) -> &mut [Real] {
        let start = self.num_rows + row * self.jac_size_both;
        &mut self.data[start..start + self.jac_size_a]
    }

    fn jacobian_b_mut(&mut self, row: usize) -> &mut [Real] {
        let start = self.num_rows + row * self.jac_size_both + self.jac_size_a;
        let len = self.jac_size_both - self.jac_size_a;
        &mut self.data[start..start + len]
    }

    /// Bullet's `internalSetAppliedImpulse` — write the accumulated impulse
    /// of `dof` back (used for warm starting; multibody rows are not
    /// warm-started by default, but the writeback keeps the API faithful).
    pub fn set_applied_impulse(&mut self, dof: usize, impulse: Real) {
        self.data[dof] = impulse;
    }

    pub fn get_applied_impulse(&self, dof: usize) -> Real {
        self.data[dof]
    }

    /// Current constraint position of `row` (Bullet's `getPosition`; the
    /// joint-limit constraint stores how far the joint is past each bound).
    pub fn get_position(&self, row: usize) -> Real {
        self.data[self.pos_offset + row]
    }

    fn set_position(&mut self, row: usize, pos: Real) {
        self.data[self.pos_offset + row] = pos;
    }

    pub fn kind(&self) -> &MultiBodyConstraintKind {
        &self.kind
    }

    pub fn kind_mut(&mut self) -> &mut MultiBodyConstraintKind {
        &mut self.kind
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn max_applied_impulse(&self) -> Real {
        self.max_applied_impulse
    }

    /// Whether the row window is unilateral (Bullet's `m_isUnilateral`; the
    /// joint motor is unilateral, the rest are bilateral).
    pub fn is_unilateral(&self) -> bool {
        self.is_unilateral
    }

    pub fn set_max_applied_impulse(&mut self, max_imp: Real) {
        self.max_applied_impulse = max_imp;
    }

    pub fn body_a(&self) -> MultiBodyHandle {
        self.body_a
    }

    pub fn body_b(&self) -> MultiBodyHandle {
        self.body_b
    }

    /// The rigid-body handle of side B, when this constraint couples the
    /// multibody to a rigid body.
    pub fn rigid_body_b(&self) -> Option<BodyHandle> {
        self.rigid_body_b
    }

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// A point-to-point constraint keeping `pivot_in_a` (link A's frame)
    /// coincident with `pivot_in_b` (link B's frame) — three bilateral rows
    /// (Bullet's `btMultiBodyPoint2Point`).
    pub fn point2point(
        body_a: MultiBodyHandle,
        link_a: Option<usize>,
        body_b: MultiBodyHandle,
        link_b: Option<usize>,
        pivot_in_a: Vector3,
        pivot_in_b: Vector3,
        bodies: &[MultiBody],
    ) -> Self {
        Self::new(
            MultiBodyConstraintKind::Point2Point(MultiBodyPoint2Point { pivot_in_a, pivot_in_b }),
            body_a,
            body_b,
            None,
            link_a,
            link_b,
            3,
            false,
            bodies,
        )
    }

    /// A fixed constraint welding link A to link B (pivots coincident and
    /// frames aligned) — six bilateral rows (Bullet's
    /// `btMultiBodyFixedConstraint`).
    #[allow(clippy::too_many_arguments)]
    pub fn fixed(
        body_a: MultiBodyHandle,
        link_a: Option<usize>,
        body_b: MultiBodyHandle,
        link_b: Option<usize>,
        pivot_in_a: Vector3,
        pivot_in_b: Vector3,
        frame_in_a: Matrix3,
        frame_in_b: Matrix3,
        bodies: &[MultiBody],
    ) -> Self {
        Self::new(
            MultiBodyConstraintKind::Fixed(MultiBodyFixedConstraint {
                pivot_in_a,
                pivot_in_b,
                frame_in_a,
                frame_in_b,
            }),
            body_a,
            body_b,
            None,
            link_a,
            link_b,
            6,
            false,
            bodies,
        )
    }

    /// A slider constraint allowing translation along `joint_axis` (link A's
    /// frame) — five bilateral rows (Bullet's `btMultiBodySliderConstraint`).
    #[allow(clippy::too_many_arguments)]
    pub fn slider(
        body_a: MultiBodyHandle,
        link_a: Option<usize>,
        body_b: MultiBodyHandle,
        link_b: Option<usize>,
        pivot_in_a: Vector3,
        pivot_in_b: Vector3,
        frame_in_a: Matrix3,
        frame_in_b: Matrix3,
        joint_axis: Vector3,
        bodies: &[MultiBody],
    ) -> Self {
        Self::new(
            MultiBodyConstraintKind::Slider(MultiBodySliderConstraint {
                pivot_in_a,
                pivot_in_b,
                frame_in_a,
                frame_in_b,
                joint_axis,
            }),
            body_a,
            body_b,
            None,
            link_a,
            link_b,
            5,
            false,
            bodies,
        )
    }

    /// A point-to-point constraint between a multibody link and a rigid
    /// body — the rigid body sits on side B (Bullet's
    /// `btMultiBodyPoint2Point(btMultiBody*, int, btRigidBody*, ...)`
    /// constructor).
    pub fn point2point_to_rigid(
        body_a: MultiBodyHandle,
        link_a: Option<usize>,
        rigid_b: BodyHandle,
        pivot_in_a: Vector3,
        pivot_in_b: Vector3,
        bodies: &[MultiBody],
    ) -> Self {
        let c = Self::new(
            MultiBodyConstraintKind::Point2Point(MultiBodyPoint2Point { pivot_in_a, pivot_in_b }),
            body_a,
            // Placeholder side-B multibody (Bullet's `m_multiBodyB = 0`);
            // every side-B read is gated by `rigid_body_b`.
            body_a,
            Some(rigid_b),
            link_a,
            None,
            3,
            false,
            bodies,
        );
        debug_assert_eq!(c.rigid_body_b, Some(rigid_b));
        c
    }

    /// A fixed (weld) constraint between a multibody link and a rigid body —
    /// the rigid body sits on side B (Bullet's
    /// `btMultiBodyFixedConstraint(btMultiBody*, int, btRigidBody*, ...)`
    /// constructor).
    #[allow(clippy::too_many_arguments)]
    pub fn fixed_to_rigid(
        body_a: MultiBodyHandle,
        link_a: Option<usize>,
        rigid_b: BodyHandle,
        pivot_in_a: Vector3,
        pivot_in_b: Vector3,
        frame_in_a: Matrix3,
        frame_in_b: Matrix3,
        bodies: &[MultiBody],
    ) -> Self {
        let c = Self::new(
            MultiBodyConstraintKind::Fixed(MultiBodyFixedConstraint {
                pivot_in_a,
                pivot_in_b,
                frame_in_a,
                frame_in_b,
            }),
            body_a,
            // Placeholder side-B multibody (Bullet's `m_multiBodyB = 0`);
            // every side-B read is gated by `rigid_body_b`.
            body_a,
            Some(rigid_b),
            link_a,
            None,
            6,
            false,
            bodies,
        );
        debug_assert_eq!(c.rigid_body_b, Some(rigid_b));
        c
    }

    /// A slider constraint between a multibody link and a rigid body — the
    /// rigid body sits on side B (Bullet's
    /// `btMultiBodySliderConstraint(btMultiBody*, int, btRigidBody*, ...)`
    /// constructor).
    #[allow(clippy::too_many_arguments)]
    pub fn slider_to_rigid(
        body_a: MultiBodyHandle,
        link_a: Option<usize>,
        rigid_b: BodyHandle,
        pivot_in_a: Vector3,
        pivot_in_b: Vector3,
        frame_in_a: Matrix3,
        frame_in_b: Matrix3,
        joint_axis: Vector3,
        bodies: &[MultiBody],
    ) -> Self {
        let c = Self::new(
            MultiBodyConstraintKind::Slider(MultiBodySliderConstraint {
                pivot_in_a,
                pivot_in_b,
                frame_in_a,
                frame_in_b,
                joint_axis,
            }),
            body_a,
            // Placeholder side-B multibody (Bullet's `m_multiBodyB = 0`);
            // every side-B read is gated by `rigid_body_b`.
            body_a,
            Some(rigid_b),
            link_a,
            None,
            5,
            false,
            bodies,
        );
        debug_assert_eq!(c.rigid_body_b, Some(rigid_b));
        c
    }

    /// A joint motor driving link `link`'s joint DOF `link_dof` toward
    /// `desired_velocity` with a maximum impulse `max_motor_impulse` — one
    /// unilateral row. The motor acts on body `body` alone (Bullet's
    /// `btMultiBodyJointMotor`; `body_b` is the same body).
    pub fn joint_motor(
        body: MultiBodyHandle,
        link: usize,
        link_dof: usize,
        desired_velocity: Real,
        max_motor_impulse: Real,
        bodies: &[MultiBody],
    ) -> Self {
        let mut c = Self::new(
            MultiBodyConstraintKind::JointMotor(MultiBodyJointMotor {
                link_dof,
                desired_velocity,
                desired_position: 0.0,
                kd: 1.0,
                kp: 0.0,
                erp: 1.0,
                rhs_clamp: Real::INFINITY,
            }),
            body,
            body,
            None,
            Some(link),
            bodies[body.index()].link(link).parent,
            1,
            true,
            bodies,
        );
        c.max_applied_impulse = max_motor_impulse;
        c
    }

    /// A gear constraint coupling link A's joint to link B's joint with the
    /// ratio `q̇_a + ratio·q̇_b = 0` — one bilateral row (Bullet's
    /// `btMultiBodyGearConstraint`).
    pub fn gear(
        body_a: MultiBodyHandle,
        link_a: usize,
        body_b: MultiBodyHandle,
        link_b: usize,
        gear_ratio: Real,
        bodies: &[MultiBody],
    ) -> Self {
        let mut c = Self::new(
            MultiBodyConstraintKind::Gear(MultiBodyGearConstraint {
                gear_ratio,
                gear_aux_link: None,
                erp: 0.0,
                relative_position_target: 0.0,
            }),
            body_a,
            body_b,
            None,
            Some(link_a),
            Some(link_b),
            1,
            false,
            bodies,
        );
        c.allocate_jacobians(bodies);
        c.finalize_gear(bodies);
        c
    }

    /// A 1-DOF joint limit clamping a revolute or prismatic joint to
    /// `[lower, upper]` — two unilateral rows on the joint's single DOF
    /// (Bullet's `btMultiBodyJointLimitConstraint`; `body_b` is the same
    /// body, `link_b` its parent).
    pub fn joint_limit(
        body: MultiBodyHandle,
        link: usize,
        lower: Real,
        upper: Real,
        bodies: &[MultiBody],
    ) -> Self {
        Self::new(
            MultiBodyConstraintKind::JointLimit(MultiBodyJointLimit {
                lower_bound: lower,
                upper_bound: upper,
            }),
            body,
            body,
            None,
            Some(link),
            bodies[body.index()].link(link).parent,
            2,
            true,
            bodies,
        )
    }

    /// A spherical joint limit keeping the joint's swing angles within
    /// `swingx_range`/`swingy_range` and its twist within `twist_range` — up
    /// to three unilateral angular rows along the link frame's world columns
    /// (Bullet's `btMultiBodySphericalJointLimit`).
    #[allow(clippy::too_many_arguments)]
    pub fn spherical_joint_limit(
        body: MultiBodyHandle,
        link: usize,
        swingx_range: Real,
        swingy_range: Real,
        twist_range: Real,
        max_applied_impulse: Real,
        bodies: &[MultiBody],
    ) -> Self {
        let mut c = Self::new(
            MultiBodyConstraintKind::SphericalJointLimit(MultiBodySphericalJointLimit {
                swingx_range,
                swingy_range,
                twist_range,
                desired_velocity: Vector3::ZERO,
                desired_position: Quaternion::IDENTITY,
                use_multi_dof_params: false,
                kd: Vector3::ONE,
                kp: Vector3::splat(0.2),
                erp: 1.0,
                rhs_clamp: Real::INFINITY,
                max_applied_impulse_multidof: Vector3::splat(max_applied_impulse),
            }),
            body,
            body,
            None,
            Some(link),
            bodies[body.index()].link(link).parent,
            3,
            true,
            bodies,
        );
        c.max_applied_impulse = max_applied_impulse;
        c
    }

    /// A spherical joint motor driving the joint orientation toward
    /// `desired_position` — three bilateral angular rows along the link
    /// frame's world columns (Bullet's `btMultiBodySphericalJointMotor`).
    pub fn spherical_joint_motor(
        body: MultiBodyHandle,
        link: usize,
        max_motor_impulse: Real,
        bodies: &[MultiBody],
    ) -> Self {
        let mut c = Self::new(
            MultiBodyConstraintKind::SphericalJointMotor(MultiBodySphericalJointMotor {
                desired_velocity: Vector3::ZERO,
                desired_position: Quaternion::IDENTITY,
                use_multi_dof_params: false,
                kd: Vector3::ONE,
                kp: Vector3::splat(0.2),
                erp: 1.0,
                rhs_clamp: Real::INFINITY,
                max_applied_impulse_multidof: Vector3::splat(max_motor_impulse),
                damping: Vector3::ONE,
            }),
            body,
            body,
            None,
            Some(link),
            bodies[body.index()].link(link).parent,
            3,
            true,
            bodies,
        );
        c.max_applied_impulse = max_motor_impulse;
        c
    }

    /// Set the gear's auxiliary link (a link of body A whose joint velocity
    /// is added to link A's, Bullet's `setGearAuxLink`).
    pub fn set_gear_aux_link(&mut self, link: Option<usize>) {
        if let MultiBodyConstraintKind::Gear(g) = &mut self.kind {
            g.gear_aux_link = link;
        }
    }

    /// Set the gear's position error reduction (Bullet's `setErp`).
    pub fn set_gear_erp(&mut self, erp: Real) {
        if let MultiBodyConstraintKind::Gear(g) = &mut self.kind {
            g.erp = erp;
        }
    }

    /// Set the gear's target relative position (Bullet's
    /// `setRelativePositionTarget`).
    pub fn set_gear_relative_position_target(&mut self, target: Real) {
        if let MultiBodyConstraintKind::Gear(g) = &mut self.kind {
            g.relative_position_target = target;
        }
    }

    /// Set the joint motor's velocity target and gain (Bullet's
    /// `setVelocityTarget`).
    pub fn set_velocity_target(&mut self, vel: Real, kd: Real) {
        if let MultiBodyConstraintKind::JointMotor(m) = &mut self.kind {
            m.desired_velocity = vel;
            m.kd = kd;
        }
    }

    /// Set the joint motor's position target and gain (Bullet's
    /// `setPositionTarget`).
    pub fn set_position_target(&mut self, pos: Real, kp: Real) {
        if let MultiBodyConstraintKind::JointMotor(m) = &mut self.kind {
            m.desired_position = pos;
            m.kp = kp;
        }
    }

    /// Set the joint motor's position ERP (Bullet's `setErp`).
    pub fn set_motor_erp(&mut self, erp: Real) {
        if let MultiBodyConstraintKind::JointMotor(m) = &mut self.kind {
            m.erp = erp;
        }
    }

    /// Set the joint motor's RHS clamp (Bullet's `setRhsClamp`).
    pub fn set_rhs_clamp(&mut self, clamp: Real) {
        if let MultiBodyConstraintKind::JointMotor(m) = &mut self.kind {
            m.rhs_clamp = clamp;
        }
    }

    /// Set the 1-DOF joint limit's lower bound (Bullet's `setLowerBound`).
    pub fn set_lower_bound(&mut self, lower: Real) {
        if let MultiBodyConstraintKind::JointLimit(j) = &mut self.kind {
            j.lower_bound = lower;
        }
    }

    /// Set the 1-DOF joint limit's upper bound (Bullet's `setUpperBound`).
    pub fn set_upper_bound(&mut self, upper: Real) {
        if let MultiBodyConstraintKind::JointLimit(j) = &mut self.kind {
            j.upper_bound = upper;
        }
    }

    /// Set the spherical limit/motor velocity target and uniform gain
    /// (Bullet's `setVelocityTarget`).
    pub fn set_spherical_velocity_target(&mut self, vel: Vector3, kd: Real) {
        match &mut self.kind {
            MultiBodyConstraintKind::SphericalJointLimit(s) => {
                s.desired_velocity = vel;
                s.kd = Vector3::splat(kd);
                s.use_multi_dof_params = false;
            }
            MultiBodyConstraintKind::SphericalJointMotor(s) => {
                s.desired_velocity = vel;
                s.kd = Vector3::splat(kd);
                s.use_multi_dof_params = false;
            }
            _ => {}
        }
    }

    /// Set the spherical limit/motor velocity target with per-axis gains
    /// (Bullet's `setVelocityTargetMultiDof`).
    pub fn set_spherical_velocity_target_multidof(&mut self, vel: Vector3, kd: Vector3) {
        match &mut self.kind {
            MultiBodyConstraintKind::SphericalJointLimit(s) => {
                s.desired_velocity = vel;
                s.kd = kd;
                s.use_multi_dof_params = true;
            }
            MultiBodyConstraintKind::SphericalJointMotor(s) => {
                s.desired_velocity = vel;
                s.kd = kd;
                s.use_multi_dof_params = true;
            }
            _ => {}
        }
    }

    /// Set the spherical limit/motor position target and uniform gain
    /// (Bullet's `setPositionTarget`).
    pub fn set_spherical_position_target(&mut self, pos: Quaternion, kp: Real) {
        match &mut self.kind {
            MultiBodyConstraintKind::SphericalJointLimit(s) => {
                s.desired_position = pos;
                s.kp = Vector3::splat(kp);
                s.use_multi_dof_params = false;
            }
            MultiBodyConstraintKind::SphericalJointMotor(s) => {
                s.desired_position = pos;
                s.kp = Vector3::splat(kp);
                s.use_multi_dof_params = false;
            }
            _ => {}
        }
    }

    /// Set the spherical limit/motor position target with per-axis gains
    /// (Bullet's `setPositionTargetMultiDof`).
    pub fn set_spherical_position_target_multidof(&mut self, pos: Quaternion, kp: Vector3) {
        match &mut self.kind {
            MultiBodyConstraintKind::SphericalJointLimit(s) => {
                s.desired_position = pos;
                s.kp = kp;
                s.use_multi_dof_params = true;
            }
            MultiBodyConstraintKind::SphericalJointMotor(s) => {
                s.desired_position = pos;
                s.kp = kp;
                s.use_multi_dof_params = true;
            }
            _ => {}
        }
    }

    /// Set the spherical limit/motor ERP (Bullet's `setErp`).
    pub fn set_spherical_erp(&mut self, erp: Real) {
        match &mut self.kind {
            MultiBodyConstraintKind::SphericalJointLimit(s) => s.erp = erp,
            MultiBodyConstraintKind::SphericalJointMotor(s) => s.erp = erp,
            _ => {}
        }
    }

    /// Set the spherical limit/motor RHS clamp (Bullet's `setRhsClamp`).
    pub fn set_spherical_rhs_clamp(&mut self, clamp: Real) {
        match &mut self.kind {
            MultiBodyConstraintKind::SphericalJointLimit(s) => s.rhs_clamp = clamp,
            MultiBodyConstraintKind::SphericalJointMotor(s) => s.rhs_clamp = clamp,
            _ => {}
        }
    }

    /// Set the spherical limit/motor per-axis impulse limits (Bullet's
    /// `setMaxAppliedImpulseMultiDof`).
    pub fn set_spherical_max_applied_impulse_multidof(&mut self, max: Vector3) {
        match &mut self.kind {
            MultiBodyConstraintKind::SphericalJointLimit(s) => {
                s.max_applied_impulse_multidof = max;
                s.use_multi_dof_params = true;
            }
            MultiBodyConstraintKind::SphericalJointMotor(s) => {
                s.max_applied_impulse_multidof = max;
                s.use_multi_dof_params = true;
            }
            _ => {}
        }
    }

    /// Set the spherical motor's per-axis velocity damping (Bullet's
    /// `setDamping`).
    pub fn set_spherical_motor_damping(&mut self, damping: Vector3) {
        if let MultiBodyConstraintKind::SphericalJointMotor(s) = &mut self.kind {
            s.damping = damping;
        }
    }

    fn finalize_gear(&mut self, bodies: &[MultiBody]) {
        // The gear jacobians never change, so initialize them here (Bullet's
        // `finalizeMultiDof`).
        let link_dof = 0;
        let link_a = self.link_a.unwrap();
        let link_b = self.link_b.unwrap();
        let offset_a = 6 + bodies[self.body_a.index()].link(link_a).dof_offset + link_dof;
        let offset_b = 6 + bodies[self.body_b.index()].link(link_b).dof_offset + link_dof;
        self.jacobian_a_mut(0)[offset_a] = 1.0;
        let ratio = if let MultiBodyConstraintKind::Gear(g) = &self.kind {
            g.gear_ratio
        } else {
            0.0
        };
        self.jacobian_b_mut(0)[offset_b] = ratio;
        self.num_dofs_finalized = self.jac_size_both;
    }

    fn finalize_motor(&mut self, bodies: &[MultiBody]) {
        let link_dof = match &self.kind {
            MultiBodyConstraintKind::JointMotor(m) => m.link_dof,
            _ => 0,
        };
        let link = self.link_a.unwrap();
        let offset = 6 + bodies[self.body_a.index()].link(link).dof_offset + link_dof;
        self.jacobian_a_mut(0)[offset] = 1.0;
        self.num_dofs_finalized = self.jac_size_both;
    }

    /// The 1-DOF joint limit jacobians never change: row 0 (lower bound) is
    /// `+1` on the joint DOF of body A, row 1 (upper bound) is `-1` on the
    /// joint DOF of body B (the same body, link B = parent) — Bullet's
    /// `btMultiBodyJointLimitConstraint::finalizeMultiDof`.
    fn finalize_joint_limit(&mut self, bodies: &[MultiBody]) {
        let link = self.link_a.unwrap();
        let offset = 6 + bodies[self.body_a.index()].link(link).dof_offset;
        self.jacobian_a_mut(0)[offset] = 1.0;
        self.jacobian_b_mut(1)[offset] = -1.0;
        self.num_dofs_finalized = self.jac_size_both;
    }

    /// The spherical limit jacobians mirror the joint limit (Bullet's
    /// `btMultiBodySphericalJointLimit::finalizeMultiDof`). They are never
    /// read by the row builder (which computes the jacobians from the
    /// constraint normals), but `num_dofs_finalized` gates row creation.
    fn finalize_spherical_limit(&mut self, bodies: &[MultiBody]) {
        let link = self.link_a.unwrap();
        let offset = 6 + bodies[self.body_a.index()].link(link).dof_offset;
        self.jacobian_a_mut(0)[offset] = 1.0;
        self.jacobian_b_mut(1)[offset] = -1.0;
        self.num_dofs_finalized = self.jac_size_both;
    }

    /// The spherical motor finalizes only body A's row-0 jacobian (Bullet's
    /// `btMultiBodySphericalJointMotor::finalizeMultiDof`); like the limit,
    /// the row builder recomputes the jacobians from the normals.
    fn finalize_spherical_motor(&mut self, bodies: &[MultiBody]) {
        let link = self.link_a.unwrap();
        let offset = 6 + bodies[self.body_a.index()].link(link).dof_offset;
        self.jacobian_a_mut(0)[offset] = 1.0;
        self.num_dofs_finalized = self.jac_size_both;
    }

    // ------------------------------------------------------------------
    // Row creation
    // ------------------------------------------------------------------

    /// Build this constraint's rows for the current state (Bullet's
    /// `createConstraintRows`). `my_index` is this constraint's position in
    /// the world's constraint list (for impulse writeback).
    #[allow(clippy::too_many_arguments)]
    pub fn create_constraint_rows(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        rigid_bodies: &[RigidBody],
        dt: Real,
        erp: Real,
    ) {
        match &self.kind {
            MultiBodyConstraintKind::Point2Point(_) => {
                self.create_p2p_rows(my_index, rows, data, scratch, bodies, rigid_bodies, dt, erp)
            }
            MultiBodyConstraintKind::Fixed(_) => {
                self.create_fixed_rows(my_index, rows, data, scratch, bodies, rigid_bodies, dt, erp)
            }
            MultiBodyConstraintKind::Slider(_) => {
                self.create_slider_rows(my_index, rows, data, scratch, bodies, rigid_bodies, dt, erp)
            }
            MultiBodyConstraintKind::JointMotor(_) => {
                self.create_motor_row(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::Gear(_) => {
                self.create_gear_row(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::JointLimit(_) => {
                self.create_joint_limit_rows(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::SphericalJointLimit(_) => {
                self.create_spherical_limit_rows(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::SphericalJointMotor(_) => {
                self.create_spherical_motor_rows(my_index, rows, data, scratch, bodies, dt, erp)
            }
        }
    }

    /// Port of `btMultiBodyConstraint::fillMultiBodyConstraint`: populate a
    /// row's jacobians (from `jac_org_a`/`jac_org_b` if given, else computed
    /// from the constraint normals), unit impulse responses, effective mass,
    /// RHS and limits. Returns the relative velocity along the row
    /// (`rel_vel`), like Bullet (callers such as the joint limit override
    /// `rhs` from it).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::useless_conversion)] // Vec3A -> Vector3 narrowing is a no-op in f64 builds
    fn fill_row(
        &self,
        row: &mut SolverRow,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        rigid_bodies: &[RigidBody],
        jac_org_a: Option<&[Real]>,
        jac_org_b: Option<&[Real]>,
        constraint_normal_ang: Vector3,
        constraint_normal_lin: Vector3,
        pos_a_world: Vector3,
        pos_b_world: Vector3,
        pos_error: Real,
        dt: Real,
        erp: Real,
        lower_limit: Real,
        upper_limit: Real,
        ang_constraint: bool,
        relaxation: Real,
        is_friction: bool,
        desired_velocity: Real,
        damping: Real,
    ) -> Real {
        let multi_body_a = self.body_a.index();
        let multi_body_b = self.body_b.index();
        let ndof_a = 6 + bodies[multi_body_a].num_dofs();
        let ndof_b = match self.rigid_body_b {
            Some(_) => 0, // the rigid side has no generalized block
            None => 6 + bodies[multi_body_b].num_dofs(),
        };

        // --- Body A ---
        let rel_pos1 = match self.link_a {
            None => pos_a_world - bodies[multi_body_a].base_pos(),
            Some(l) => pos_a_world - Vector3::from(bodies[multi_body_a].link(l).cached_world_transform.translation),
        };
        row.delta_vel_a_index = bodies[multi_body_a].companion_id;
        if row.delta_vel_a_index < 0 {
            row.delta_vel_a_index = data.delta_velocities.len() as i32;
            bodies[multi_body_a].companion_id = row.delta_vel_a_index;
            data.delta_velocities.resize(data.delta_velocities.len() + ndof_a, 0.0);
        }
        row.jac_a_index = data.jacobians.len();
        data.jacobians.resize(data.jacobians.len() + ndof_a, 0.0);
        let jac_a_start = row.jac_a_index;
        match jac_org_a {
            Some(j) => {
                data.jacobians[jac_a_start..jac_a_start + ndof_a].copy_from_slice(&j[..ndof_a]);
            }
            None => {
                let jac = &mut data.jacobians[jac_a_start..jac_a_start + ndof_a];
                bodies[multi_body_a].fill_constraint_jacobian(
                    self.link_a.map_or(-1, |l| l as i32),
                    pos_a_world,
                    constraint_normal_ang,
                    constraint_normal_lin,
                    jac,
                    scratch,
                );
            }
        }
        // Velocity response to a unit impulse along this row.
        let delta_start = data.delta_velocities_unit_impulse.len();
        data.delta_velocities_unit_impulse
            .resize(delta_start + ndof_a, 0.0);
        {
            let jac = &data.jacobians[jac_a_start..jac_a_start + ndof_a];
            let delta = &mut data.delta_velocities_unit_impulse[delta_start..delta_start + ndof_a];
            bodies[multi_body_a].calc_acceleration_deltas(jac, delta, scratch);
        }
        let torque_axis0 = if ang_constraint {
            constraint_normal_ang
        } else {
            rel_pos1.cross(constraint_normal_lin)
        };
        row.rel_pos1_cross_normal = torque_axis0;
        row.contact_normal1 = constraint_normal_lin;

        // --- Body B ---
        // Bullet branches on `m_multiBodyB` here: the rigid side stores its
        // inertia-scaled response and an assembly-time velocity snapshot
        // instead of a generalized jacobian block.
        let rigid_b_side = self.rigid_body_b.map(|h| &rigid_bodies[h.index()]);
        let (rel_pos2, mb_b_block) = match rigid_b_side {
            Some(rb) => {
                let rel_pos2 = pos_b_world - rb.position;
                let torque_axis1 = if ang_constraint {
                    constraint_normal_ang
                } else {
                    rel_pos2.cross(constraint_normal_lin)
                };
                row.rigid_b = Some(RigidSideB {
                    body: self.rigid_body_b.unwrap().index(),
                    angular_component: rb.inv_inertia_world() * (-torque_axis1),
                    inv_mass: rb.inv_mass(),
                    initial_proj: (-constraint_normal_lin).dot(rb.linear_velocity)
                        + (-torque_axis1).dot(rb.angular_velocity),
                });
                row.rel_pos2_cross_normal = -torque_axis1;
                row.contact_normal2 = -constraint_normal_lin;
                (rel_pos2, None)
            }
            None => {
                let rel_pos2 = match self.link_b {
                    None => pos_b_world - bodies[multi_body_b].base_pos(),
                    Some(l) => pos_b_world
                        - Vector3::from(bodies[multi_body_b].link(l).cached_world_transform.translation),
                };
                row.delta_vel_b_index = bodies[multi_body_b].companion_id;
                if row.delta_vel_b_index < 0 {
                    row.delta_vel_b_index = data.delta_velocities.len() as i32;
                    bodies[multi_body_b].companion_id = row.delta_vel_b_index;
                    data.delta_velocities.resize(data.delta_velocities.len() + ndof_b, 0.0);
                }
                row.jac_b_index = data.jacobians.len();
                data.jacobians.resize(data.jacobians.len() + ndof_b, 0.0);
                let jac_b_start = row.jac_b_index;
                match jac_org_b {
                    Some(j) => {
                        data.jacobians[jac_b_start..jac_b_start + ndof_b]
                            .copy_from_slice(&j[..ndof_b]);
                    }
                    None => {
                        let jac = &mut data.jacobians[jac_b_start..jac_b_start + ndof_b];
                        bodies[multi_body_b].fill_constraint_jacobian(
                            self.link_b.map_or(-1, |l| l as i32),
                            pos_b_world,
                            -constraint_normal_ang,
                            -constraint_normal_lin,
                            jac,
                            scratch,
                        );
                    }
                }
                let delta_b_start = data.delta_velocities_unit_impulse.len();
                data.delta_velocities_unit_impulse
                    .resize(delta_b_start + ndof_b, 0.0);
                {
                    let jac = &data.jacobians[jac_b_start..jac_b_start + ndof_b];
                    let delta = &mut data.delta_velocities_unit_impulse[delta_b_start..delta_b_start + ndof_b];
                    bodies[multi_body_b].calc_acceleration_deltas(jac, delta, scratch);
                }
                let torque_axis1 = if ang_constraint {
                    constraint_normal_ang
                } else {
                    rel_pos2.cross(constraint_normal_lin)
                };
                row.rel_pos2_cross_normal = -torque_axis1;
                row.contact_normal2 = -constraint_normal_lin;
                (rel_pos2, Some((jac_b_start, delta_b_start)))
            }
        };

        // --- Effective mass ---
        let mut denom0 = 0.0;
        {
            let jac = &data.jacobians[jac_a_start..jac_a_start + ndof_a];
            let delta = &data.delta_velocities_unit_impulse[delta_start..delta_start + ndof_a];
            for i in 0..ndof_a {
                denom0 += jac[i] * delta[i];
            }
        }
        let mut denom1 = 0.0;
        match (mb_b_block, rigid_b_side) {
            (Some((jac_b_start, delta_b_start)), _) => {
                let jac = &data.jacobians[jac_b_start..jac_b_start + ndof_b];
                let delta = &data.delta_velocities_unit_impulse[delta_b_start..delta_b_start + ndof_b];
                for i in 0..ndof_b {
                    denom1 += jac[i] * delta[i];
                }
            }
            (_, Some(rb)) => {
                let angular_component = row.rigid_b.unwrap().angular_component;
                if ang_constraint {
                    denom1 = constraint_normal_ang.dot(-angular_component);
                } else {
                    let vec = (-angular_component).cross(rel_pos2);
                    denom1 = rb.inv_mass() + constraint_normal_lin.dot(vec);
                }
            }
            _ => unreachable!("side/data mismatch"),
        }
        let d = denom0 + denom1;
        row.jac_diag_ab_inv = if d > Real::EPSILON { relaxation / d } else { 0.0 };

        // --- RHS ---
        let penetration = if is_friction { 0.0 } else { pos_error };
        let mut rel_vel = 0.0;
        {
            let jac = &data.jacobians[jac_a_start..jac_a_start + ndof_a];
            let vel = bodies[multi_body_a].velocity_vector();
            for i in 0..ndof_a {
                rel_vel += vel[i] * jac[i];
            }
        }
        match (mb_b_block, rigid_b_side) {
            (Some((jac_b_start, _)), _) => {
                let jac = &data.jacobians[jac_b_start..jac_b_start + ndof_b];
                let vel = bodies[multi_body_b].velocity_vector();
                for i in 0..ndof_b {
                    rel_vel += vel[i] * jac[i];
                }
            }
            (_, Some(rb)) => {
                rel_vel += row.contact_normal2.dot(rb.linear_velocity)
                    + row.rel_pos2_cross_normal.dot(rb.angular_velocity);
            }
            _ => unreachable!("side/data mismatch"),
        }
        let velocity_error = (desired_velocity - rel_vel) * damping;
        // Split impulse is not implemented for multibodies: always use m_erp.
        let positional_error = -penetration * erp / dt;
        let penetration_impulse = positional_error * row.jac_diag_ab_inv;
        let velocity_impulse = velocity_error * row.jac_diag_ab_inv;
        row.rhs = penetration_impulse + velocity_impulse;
        row.cfm = 0.0;
        row.lower_limit = lower_limit;
        row.upper_limit = upper_limit;
        rel_vel
    }

    fn new_row(&self, my_index: usize, row_index: usize) -> SolverRow {
        SolverRow {
            jac_a_index: 0,
            delta_vel_a_index: -1,
            jac_b_index: 0,
            delta_vel_b_index: -1,
            ndof_a: 0,
            ndof_b: 0,
            rel_pos1_cross_normal: Vector3::ZERO,
            contact_normal1: Vector3::ZERO,
            rel_pos2_cross_normal: Vector3::ZERO,
            contact_normal2: Vector3::ZERO,
            jac_diag_ab_inv: 0.0,
            rhs: 0.0,
            cfm: 0.0,
            lower_limit: 0.0,
            upper_limit: 0.0,
            applied_impulse: 0.0,
            multi_body_a: self.body_a.index(),
            multi_body_b: self.body_b.index(),
            link_a: self.link_a,
            link_b: self.link_b,
            rigid_b: None,
            org_constraint: my_index,
            org_dof_index: row_index,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_p2p_rows(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        rigid_bodies: &[RigidBody],
        dt: Real,
        erp: Real,
    ) {
        let (pivot_in_a, pivot_in_b) = match &self.kind {
            MultiBodyConstraintKind::Point2Point(p) => (p.pivot_in_a, p.pivot_in_b),
            _ => unreachable!(),
        };
        for i in 0..3 {
            let mut row = self.new_row(my_index, i);
            row.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
            row.ndof_b = match self.rigid_body_b {
                Some(_) => 0,
                None => 6 + bodies[self.body_b.index()].num_dofs(),
            };
            // contactNormalOnB[i] = -1 (Bullet).
            let mut contact_normal_on_b = Vector3::ZERO;
            contact_normal_on_b[i] = -1.0;

            let pivot_a_world = bodies[self.body_a.index()].local_pos_to_world(
                self.link_a.map_or(-1, |l| l as i32),
                pivot_in_a,
            );
            // `rb->getCenterOfMassTransform() * pivotInB` for a rigid side B.
            let pivot_b_world = match self.rigid_body_b {
                Some(rb) => {
                    let body = &rigid_bodies[rb.index()];
                    body.position + body.orientation * pivot_in_b
                }
                None => bodies[self.body_b.index()].local_pos_to_world(
                    self.link_b.map_or(-1, |l| l as i32),
                    pivot_in_b,
                ),
            };
            let pos_error = (pivot_a_world - pivot_b_world).dot(contact_normal_on_b);

            self.fill_row(
                &mut row,
                data,
                scratch,
                bodies,
                rigid_bodies,
                None,
                None,
                Vector3::ZERO,
                contact_normal_on_b,
                pivot_a_world,
                pivot_b_world,
                pos_error,
                dt,
                erp,
                -self.max_applied_impulse,
                self.max_applied_impulse,
                false,
                1.0,
                false,
                0.0,
                1.0,
            );
            rows.push(row);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_fixed_rows(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        rigid_bodies: &[RigidBody],
        dt: Real,
        erp: Real,
    ) {
        let (pivot_in_a, pivot_in_b, frame_in_a, frame_in_b) = match &self.kind {
            MultiBodyConstraintKind::Fixed(f) => {
                (f.pivot_in_a, f.pivot_in_b, f.frame_in_a, f.frame_in_b)
            }
            _ => unreachable!(),
        };
        let pivot_a_world = bodies[self.body_a.index()].local_pos_to_world(self.link_a.map_or(-1, |l| l as i32), pivot_in_a);
        let frame_a_world = bodies[self.body_a.index()].local_frame_to_world(self.link_a.map_or(-1, |l| l as i32), &frame_in_a);
        // Bullet's rigid-B branch composes the frame transposed:
        // `frameBworld = m_frameInB.transpose() * rot(rb)` (unlike the
        // multibody side's `localFrameToWorld` = `R * frame`); with identity
        // frames both reduce to the body orientation. Ported verbatim.
        let (pivot_b_world, frame_b_world) = match self.rigid_body_b {
            Some(rb) => {
                let body = &rigid_bodies[rb.index()];
                (
                    body.position + body.orientation * pivot_in_b,
                    frame_in_b.transpose() * Matrix3::from_quat(body.orientation),
                )
            }
            None => (
                bodies[self.body_b.index()].local_pos_to_world(self.link_b.map_or(-1, |l| l as i32), pivot_in_b),
                bodies[self.body_b.index()].local_frame_to_world(self.link_b.map_or(-1, |l| l as i32), &frame_in_b),
            ),
        };
        // The relative rotation between the two world frames (Bullet's
        // `relRot = frameAworld.inverse() * frameBworld`).
        let rel_rot = frame_a_world.inverse() * frame_b_world;
        let angle_diff = crate::constraint::euler_xyz(&rel_rot);

        for i in 0..6 {
            let mut row = self.new_row(my_index, i);
            row.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
            row.ndof_b = match self.rigid_body_b {
                Some(_) => 0,
                None => 6 + bodies[self.body_b.index()].num_dofs(),
            };
            let constraint_normal_lin = Vector3::ZERO;
            let constraint_normal_ang = Vector3::ZERO;
            let pos_error;
            if i < 3 {
                let mut n = Vector3::ZERO;
                n[i] = 1.0;
                pos_error = (pivot_a_world - pivot_b_world).dot(n);
                self.fill_row(
                    &mut row,
                    data,
                    scratch,
                    bodies,
                    rigid_bodies,
                    None,
                    None,
                    constraint_normal_ang,
                    n,
                    pivot_a_world,
                    pivot_b_world,
                    pos_error,
                    dt,
                    erp,
                    -self.max_applied_impulse,
                    self.max_applied_impulse,
                    false,
                    1.0,
                    false,
                    0.0,
                    1.0,
                );
            } else {
                let n_ang = frame_a_world.col(i % 3);
                pos_error = angle_diff[i % 3];
                self.fill_row(
                    &mut row,
                    data,
                    scratch,
                    bodies,
                    rigid_bodies,
                    None,
                    None,
                    n_ang,
                    constraint_normal_lin,
                    pivot_a_world,
                    pivot_b_world,
                    pos_error,
                    dt,
                    erp,
                    -self.max_applied_impulse,
                    self.max_applied_impulse,
                    true,
                    1.0,
                    false,
                    0.0,
                    1.0,
                );
            }
            rows.push(row);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_slider_rows(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        rigid_bodies: &[RigidBody],
        dt: Real,
        erp: Real,
    ) {
        let (pivot_in_a, pivot_in_b, frame_in_a, frame_in_b, joint_axis) = match &self.kind {
            MultiBodyConstraintKind::Slider(s) => {
                (s.pivot_in_a, s.pivot_in_b, s.frame_in_a, s.frame_in_b, s.joint_axis)
            }
            _ => unreachable!(),
        };
        let pivot_a_world = bodies[self.body_a.index()].local_pos_to_world(self.link_a.map_or(-1, |l| l as i32), pivot_in_a);
        let frame_a_world = bodies[self.body_a.index()].local_frame_to_world(self.link_a.map_or(-1, |l| l as i32), &frame_in_a);
        let joint_axis = bodies[self.body_a.index()].local_dir_to_world(self.link_a.map_or(-1, |l| l as i32), joint_axis);
        // Bullet's rigid-B branch composes the frame transposed (see
        // `create_fixed_rows`); ported verbatim.
        let (pivot_b_world, frame_b_world) = match self.rigid_body_b {
            Some(rb) => {
                let body = &rigid_bodies[rb.index()];
                (
                    body.position + body.orientation * pivot_in_b,
                    frame_in_b.transpose() * Matrix3::from_quat(body.orientation),
                )
            }
            None => (
                bodies[self.body_b.index()].local_pos_to_world(self.link_b.map_or(-1, |l| l as i32), pivot_in_b),
                bodies[self.body_b.index()].local_frame_to_world(self.link_b.map_or(-1, |l| l as i32), &frame_in_b),
            ),
        };

        // Two orthonormal axes perpendicular to the joint axis (Bullet).
        let mut constraint_axis = [Vector3::ZERO; 2];
        'outer: for i in 0..3 {
            let mut c = frame_a_world.col(i).cross(joint_axis);
            if c.length() > 0.000001 {
                c = c.normalize();
                constraint_axis[0] = c;
                constraint_axis[1] = joint_axis.cross(c).normalize();
                break 'outer;
            }
        }

        let rel_rot = frame_a_world.inverse() * frame_b_world;
        let angle_diff = crate::constraint::euler_xyz(&rel_rot);

        for i in 0..5 {
            let mut row = self.new_row(my_index, i);
            row.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
            row.ndof_b = match self.rigid_body_b {
                Some(_) => 0,
                None => 6 + bodies[self.body_b.index()].num_dofs(),
            };
            let constraint_normal_lin = Vector3::ZERO;
            let constraint_normal_ang = Vector3::ZERO;
            if i < 2 {
                let n = constraint_axis[i];
                let pos_error = (pivot_a_world - pivot_b_world).dot(n);
                self.fill_row(
                    &mut row,
                    data,
                    scratch,
                    bodies,
                    rigid_bodies,
                    None,
                    None,
                    constraint_normal_ang,
                    n,
                    pivot_a_world,
                    pivot_b_world,
                    pos_error,
                    dt,
                    erp,
                    -self.max_applied_impulse,
                    self.max_applied_impulse,
                    false,
                    1.0,
                    false,
                    0.0,
                    1.0,
                );
            } else {
                let n_ang = frame_a_world.col(i % 3);
                let pos_error = angle_diff[i % 3];
                self.fill_row(
                    &mut row,
                    data,
                    scratch,
                    bodies,
                    rigid_bodies,
                    None,
                    None,
                    n_ang,
                    constraint_normal_lin,
                    pivot_a_world,
                    pivot_b_world,
                    pos_error,
                    dt,
                    erp,
                    -self.max_applied_impulse,
                    self.max_applied_impulse,
                    true,
                    1.0,
                    false,
                    0.0,
                    1.0,
                );
            }
            rows.push(row);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_motor_row(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        dt: Real,
        erp: Real,
    ) {
        // The motor jacobians never change: finalize once (Bullet checks
        // `m_numDofsFinalized != m_jacSizeBoth`).
        if self.num_dofs_finalized != self.jac_size_both {
            self.allocate_jacobians(bodies);
            self.finalize_motor(bodies);
        }
        if self.num_dofs_finalized != self.jac_size_both {
            return;
        }
        if self.max_applied_impulse == 0.0 {
            return;
        }
        let (link_dof, desired_velocity, desired_position, kd, kp, motor_erp, rhs_clamp) =
            match &self.kind {
                MultiBodyConstraintKind::JointMotor(m) => (
                    m.link_dof,
                    m.desired_velocity,
                    m.desired_position,
                    m.kd,
                    m.kp,
                    m.erp,
                    m.rhs_clamp,
                ),
                _ => unreachable!(),
            };

        let mut row = self.new_row(my_index, 0);
        row.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
        row.ndof_b = 6 + bodies[self.body_b.index()].num_dofs();

        let link = self.link_a.unwrap();
        let current_position = bodies[self.body_a.index()].joint_pos_multidof(link)[link_dof];
        let current_velocity = bodies[self.body_a.index()].joint_vel_multidof(link)[link_dof];
        let position_stabilization_term = motor_erp * (desired_position - current_position) / dt;
        let velocity_error = desired_velocity - current_velocity;
        let mut rhs = kp * position_stabilization_term + current_velocity + kd * velocity_error;
        rhs = rhs.clamp(-rhs_clamp, rhs_clamp);

        self.fill_row(
            &mut row,
            data,
            scratch,
            bodies,
            &[],
            Some(self.jacobian_a(0)),
            Some(self.jacobian_b(0)),
            Vector3::ZERO,
            Vector3::ZERO,
            Vector3::ZERO,
            Vector3::ZERO,
            0.0,
            dt,
            erp,
            -self.max_applied_impulse,
            self.max_applied_impulse,
            false,
            1.0,
            false,
            rhs,
            1.0,
        );

        // The constraint normals are used for feedback only; fill them for
        // fidelity (Bullet sets them from the world joint axis).
        let link_world_rot = bodies[self.body_a.index()].link_world_orientation(link);
        let link_ref = bodies[self.body_a.index()].link(link);
        match link_ref.joint_type {
            JointType::Revolute => {
                let axis_in_world = link_world_rot * link_ref.axis_top(0);
                row.contact_normal1 = Vector3::ZERO;
                row.contact_normal2 = Vector3::ZERO;
                row.rel_pos1_cross_normal = axis_in_world;
                row.rel_pos2_cross_normal = -axis_in_world;
            }
            JointType::Prismatic => {
                let axis_in_world = link_world_rot * link_ref.axis_bottom(0);
                row.contact_normal1 = axis_in_world;
                row.contact_normal2 = -axis_in_world;
                row.rel_pos1_cross_normal = Vector3::ZERO;
                row.rel_pos2_cross_normal = Vector3::ZERO;
            }
            _ => {}
        }
        rows.push(row);
    }

    #[allow(clippy::too_many_arguments)]
    fn create_gear_row(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        dt: Real,
        erp: Real,
    ) {
        if self.max_applied_impulse == 0.0 {
            return;
        }
        let (gear_ratio, gear_aux_link, gear_erp, relative_position_target) = match &self.kind {
            MultiBodyConstraintKind::Gear(g) => (
                g.gear_ratio,
                g.gear_aux_link,
                g.erp,
                g.relative_position_target,
            ),
            _ => unreachable!(),
        };
        let mut row = self.new_row(my_index, 0);
        row.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
        row.ndof_b = 6 + bodies[self.body_b.index()].num_dofs();

        let link_a = self.link_a.unwrap();
        let link_b = self.link_b.unwrap();
        let link_dof = 0;
        let mut aux_vel = 0.0;
        if let Some(aux) = gear_aux_link {
            aux_vel = bodies[self.body_a.index()].joint_vel_multidof(aux)[link_dof];
        }
        let mut pos_error = 0.0;
        if gear_erp != 0.0 {
            let mut current_position_a = bodies[self.body_a.index()].joint_pos_multidof(link_a)[link_dof];
            if let Some(aux) = gear_aux_link {
                current_position_a -= bodies[self.body_a.index()].joint_pos_multidof(aux)[link_dof];
            }
            let current_position_b = gear_ratio * bodies[self.body_b.index()].joint_pos_multidof(link_b)[link_dof];
            let diff = current_position_b + current_position_a;
            pos_error = -gear_erp * (relative_position_target - diff);
        }
        let desired_relative_velocity = aux_vel;

        self.fill_row(
            &mut row,
            data,
            scratch,
            bodies,
            &[],
            Some(self.jacobian_a(0)),
            Some(self.jacobian_b(0)),
            Vector3::ZERO,
            Vector3::ZERO,
            Vector3::ZERO,
            Vector3::ZERO,
            pos_error,
            dt,
            erp,
            -self.max_applied_impulse,
            self.max_applied_impulse,
            false,
            1.0,
            false,
            desired_relative_velocity,
            1.0,
        );
        // Feedback normals (Bullet sets them from the joint axis).
        let link_world_rot = bodies[self.body_a.index()].link_world_orientation(link_a);
        let link_ref = bodies[self.body_a.index()].link(link_a);
        match link_ref.joint_type {
            JointType::Revolute => {
                let axis_in_world = link_world_rot * link_ref.axis_top(0);
                row.contact_normal1 = Vector3::ZERO;
                row.contact_normal2 = Vector3::ZERO;
                row.rel_pos1_cross_normal = axis_in_world;
                row.rel_pos2_cross_normal = -axis_in_world;
            }
            JointType::Prismatic => {
                let axis_in_world = link_world_rot * link_ref.axis_bottom(0);
                row.contact_normal1 = axis_in_world;
                row.contact_normal2 = -axis_in_world;
                row.rel_pos1_cross_normal = Vector3::ZERO;
                row.rel_pos2_cross_normal = Vector3::ZERO;
            }
            _ => {}
        }
        rows.push(row);
    }

    /// Port of `btMultiBodyJointLimitConstraint::createConstraintRows` — two
    /// unilateral rows on the joint's single DOF (row 0 clamps the lower
    /// bound, row 1 the upper). The jacobians are the finalized unit rows; the
    /// RHS is the classic limit formula (positional + velocity error),
    /// overriding the fill's default. Like Bullet, a violation deeper than
    /// `m_splitImpulsePenetrationThreshold` drops the positional term (the
    /// multibody solver ignores `m_rhsPenetration`).
    #[allow(clippy::too_many_arguments)]
    fn create_joint_limit_rows(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        dt: Real,
        erp: Real,
    ) {
        if self.num_dofs_finalized != self.jac_size_both {
            self.allocate_jacobians(bodies);
            self.finalize_joint_limit(bodies);
        }
        if self.num_dofs_finalized != self.jac_size_both {
            return;
        }
        let (lower_bound, upper_bound) = match &self.kind {
            MultiBodyConstraintKind::JointLimit(j) => (j.lower_bound, j.upper_bound),
            _ => unreachable!(),
        };
        let link = self.link_a.unwrap();
        let joint_pos = bodies[self.body_a.index()].joint_pos_multidof(link)[0];
        // Row 0: how far the joint is below the lower bound; row 1: how far
        // above the upper bound (negative = violated, rows skipped inside).
        self.set_position(0, joint_pos - lower_bound);
        self.set_position(1, upper_bound - joint_pos);

        for row in 0..2 {
            let penetration = self.get_position(row);
            if penetration > 0.0 {
                continue;
            }
            let direction = if row == 1 { -1.0 } else { 1.0 };
            let mut r = self.new_row(my_index, row);
            r.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
            r.ndof_b = 6 + bodies[self.body_b.index()].num_dofs();

            let rel_vel = self.fill_row(
                &mut r,
                data,
                scratch,
                bodies,
                &[],
                Some(self.jacobian_a(row)),
                Some(self.jacobian_b(row)),
                Vector3::ZERO,
                Vector3::ZERO,
                Vector3::ZERO,
                Vector3::ZERO,
                0.0, // pos_error: the limit overrides the RHS below
                dt,
                erp,
                0.0,
                self.max_applied_impulse,
                false,
                1.0,
                false,
                0.0,
                1.0,
            );

            // Feedback normals (only used for joint feedback reporting).
            let link_world_rot = bodies[self.body_a.index()].link_world_orientation(link);
            let link_ref = bodies[self.body_a.index()].link(link);
            match link_ref.joint_type {
                JointType::Revolute => {
                    let axis = direction * (link_world_rot * link_ref.axis_top(0));
                    r.contact_normal1 = Vector3::ZERO;
                    r.contact_normal2 = Vector3::ZERO;
                    r.rel_pos1_cross_normal = axis;
                    r.rel_pos2_cross_normal = -axis;
                }
                JointType::Prismatic => {
                    let axis = direction * (link_world_rot * link_ref.axis_bottom(0));
                    r.contact_normal1 = axis;
                    r.contact_normal2 = -axis;
                    r.rel_pos1_cross_normal = Vector3::ZERO;
                    r.rel_pos2_cross_normal = Vector3::ZERO;
                }
                _ => {}
            }

            // Override the RHS with the joint-limit formula (Bullet uses
            // m_erp == m_erp2 == 0.2, so the world erp applies to both).
            let positional_error = -penetration * erp / dt; // penetration <= 0
            let velocity_error = -rel_vel;
            let rhs = if penetration > SPLIT_PENETRATION_THRESHOLD {
                (positional_error + velocity_error) * r.jac_diag_ab_inv
            } else {
                velocity_error * r.jac_diag_ab_inv
            };
            r.rhs = rhs;
            rows.push(r);
        }
    }

    /// Port of `btMultiBodySphericalJointLimit::createConstraintRows` — up to
    /// three unilateral angular rows keeping the joint's swing (about the
    /// link frame's world X/Y) and twist (about its world Z) within the given
    /// ranges. The rows are built from the current joint quaternion split
    /// into its cone and twist parts; each row's window is one-sided so the
    /// limit can only push the joint back inside.
    #[allow(clippy::too_many_arguments)]
    fn create_spherical_limit_rows(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        dt: Real,
        erp: Real,
    ) {
        if self.num_dofs_finalized != self.jac_size_both {
            self.allocate_jacobians(bodies);
            self.finalize_spherical_limit(bodies);
        }
        if self.num_dofs_finalized != self.jac_size_both {
            return;
        }
        if self.max_applied_impulse == 0.0 {
            return;
        }
        let (swingx_range, swingy_range, twist_range, use_multi_dof_params, kd, kp, max_multi_dof) =
            match &self.kind {
                MultiBodyConstraintKind::SphericalJointLimit(s) => (
                    s.swingx_range,
                    s.swingy_range,
                    s.twist_range,
                    s.use_multi_dof_params,
                    s.kd,
                    s.kp,
                    s.max_applied_impulse_multidof,
                ),
                _ => unreachable!(),
            };
        let link = self.link_a.unwrap();
        let jp = bodies[self.body_a.index()].joint_pos_multidof(link);
        let current_quat = Quaternion::from_xyzw(jp[0], jp[1], jp[2], jp[3]);
        let v_twist = Vector3::Z;

        // Split the joint quaternion into the cone (swing) and the residual
        // twist: q = qABCone * qABTwist with qABCone mapping Z onto the
        // rotated Z axis.
        let v_cone_no_twist = (current_quat * v_twist).normalize();
        let qab_cone = shortest_arc_quat(v_twist, v_cone_no_twist).normalize();
        let qab_twist = (qab_cone.inverse() * current_quat).normalize();
        let desired_quat = qab_twist;
        let rel_rot = current_quat.inverse() * desired_quat;
        let mut angle_diff = crate::constraint::euler_xyz(&Matrix3::from_quat(rel_rot));

        // Twist axis/angle (Bullet's `getAngle` = 2·acos(w)); flip the
        // quaternion to take the short way around, then sign by the twist
        // axis against +Z.
        let mut q_min_twist = qab_twist;
        let mut twist_angle = 2.0 * math::acos(qab_twist.w);
        if twist_angle > bullet_math::PI {
            q_min_twist = -qab_twist;
            twist_angle = 2.0 * math::acos(q_min_twist.w);
        }
        let mut v_twist_axis = Vector3::new(q_min_twist.x, q_min_twist.y, q_min_twist.z);
        if twist_angle > Real::EPSILON {
            v_twist_axis = v_twist_axis.normalize();
        }
        if v_twist_axis.dot(v_twist) < 0.0 {
            twist_angle = -twist_angle;
        }
        angle_diff.z = twist_angle;

        let limit_ranges = [swingx_range, swingy_range, twist_range];
        let frame_a_world = bodies[self.body_a.index()].local_frame_to_world(link as i32, &Matrix3::IDENTITY);
        for row in 0..3 {
            let allowed = limit_ranges[row];
            let mut damp = 1.0;
            if angle_diff[row] > -allowed && angle_diff[row] < allowed {
                angle_diff[row] = 0.0;
                damp = 0.0;
            } else {
                if angle_diff[row] > allowed {
                    angle_diff[row] -= allowed;
                }
                if angle_diff[row] < -allowed {
                    angle_diff[row] += allowed;
                }
            }

            // Bullet computes a `velocityError` with kd here but never uses
            // it: the fill is called with desiredVelocity = 0 and the damp
            // factor as damping. Kept for fidelity of the row construction.
            let _ = kd;
            let kp_row = if use_multi_dof_params { kp[row % 3] } else { kp[0] };
            let pos_error = kp_row * angle_diff[row % 3];
            let mut max_impulse =
                if use_multi_dof_params { max_multi_dof[row % 3] } else { self.max_applied_impulse };
            let mut min_impulse = -max_impulse;
            // One-sided window: the limit only pushes back into the range.
            if pos_error > 0.0 {
                max_impulse = 0.0;
            } else {
                min_impulse = 0.0;
            }
            if pos_error.abs() > Real::EPSILON {
                let mut r = self.new_row(my_index, row);
                r.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
                r.ndof_b = 6 + bodies[self.body_b.index()].num_dofs();
                let constraint_normal_ang = frame_a_world.col(row % 3);
                self.fill_row(
                    &mut r,
                    data,
                    scratch,
                    bodies,
                    &[],
                    None,
                    None,
                    constraint_normal_ang,
                    Vector3::ZERO,
                    Vector3::ZERO,
                    Vector3::ZERO,
                    pos_error,
                    dt,
                    erp,
                    min_impulse,
                    max_impulse,
                    true, // pure angular
                    1.0,
                    false,
                    0.0,
                    damp,
                );
                rows.push(r);
            }
        }
    }

    /// Port of `btMultiBodySphericalJointMotor::createConstraintRows` — three
    /// bilateral angular rows driving the joint orientation toward
    /// `desired_position` (default identity), each along a world column of
    /// the link frame and damped by `damping[i]`.
    #[allow(clippy::too_many_arguments)]
    fn create_spherical_motor_rows(
        &mut self,
        my_index: usize,
        rows: &mut Vec<SolverRow>,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
        dt: Real,
        erp: Real,
    ) {
        if self.num_dofs_finalized != self.jac_size_both {
            self.allocate_jacobians(bodies);
            self.finalize_spherical_motor(bodies);
        }
        if self.num_dofs_finalized != self.jac_size_both {
            return;
        }
        if self.max_applied_impulse == 0.0 {
            return;
        }
        let (desired_position, use_multi_dof_params, kd, kp, max_multi_dof, damping) =
            match &self.kind {
                MultiBodyConstraintKind::SphericalJointMotor(m) => (
                    m.desired_position,
                    m.use_multi_dof_params,
                    m.kd,
                    m.kp,
                    m.max_applied_impulse_multidof,
                    m.damping,
                ),
                _ => unreachable!(),
            };
        let link = self.link_a.unwrap();
        let jp = bodies[self.body_a.index()].joint_pos_multidof(link);
        let current_quat = Quaternion::from_xyzw(jp[0], jp[1], jp[2], jp[3]);
        let rel_rot = current_quat.inverse() * desired_position;
        let angle_diff = crate::constraint::euler_xyz(&Matrix3::from_quat(rel_rot));

        let frame_a_world = bodies[self.body_a.index()].local_frame_to_world(link as i32, &Matrix3::IDENTITY);
        for row in 0..3 {
            // Bullet computes a `velocityError` with kd here but never uses
            // it (the fill receives desiredVelocity = 0); kept for fidelity.
            let _ = kd;
            let kp_row = if use_multi_dof_params { kp[row % 3] } else { kp[0] };
            let pos_error = kp_row * angle_diff[row % 3];
            let max_impulse =
                if use_multi_dof_params { max_multi_dof[row % 3] } else { self.max_applied_impulse };
            let mut r = self.new_row(my_index, row);
            r.ndof_a = 6 + bodies[self.body_a.index()].num_dofs();
            r.ndof_b = 6 + bodies[self.body_b.index()].num_dofs();
            let constraint_normal_ang = frame_a_world.col(row % 3);
            self.fill_row(
                &mut r,
                data,
                scratch,
                bodies,
                &[],
                None,
                None,
                constraint_normal_ang,
                Vector3::ZERO,
                Vector3::ZERO,
                Vector3::ZERO,
                pos_error,
                dt,
                erp,
                -max_impulse,
                max_impulse,
                true, // pure angular
                1.0,
                false,
                0.0,
                damping[row % 3],
            );
            rows.push(r);
        }
    }
}

/// The shortest-arc quaternion rotating `v0` onto `v1` (Bullet's
/// `shortestArcQuat`, Game Programming Gems 2.10). Callers are expected to
/// normalize the result.
fn shortest_arc_quat(v0: Vector3, v1: Vector3) -> Quaternion {
    let c = v0.cross(v1);
    let d = v0.dot(v1);
    if d < -1.0 + Real::EPSILON {
        // v0 and v1 are opposite: pick any orthogonal vector.
        let (n, _unused) = plane_space1(v0);
        return Quaternion::from_xyzw(n.x, n.y, n.z, 0.0);
    }
    let s = math::sqrt((1.0 + d) * 2.0);
    let rs = 1.0 / s;
    Quaternion::from_xyzw(c.x * rs, c.y * rs, c.z * rs, s * 0.5)
}

/// Bullet's `btPlaneSpace1`: an orthonormal pair `(p, q)` spanning the plane
/// perpendicular to `n`.
pub(crate) fn plane_space1(n: Vector3) -> (Vector3, Vector3) {
    if n.z.abs() > bullet_math::FRAC_1_SQRT_2 {
        // Choose p in the y-z plane.
        let a = n.y * n.y + n.z * n.z;
        let k = 1.0 / math::sqrt(a);
        let p = Vector3::new(0.0, -n.z * k, n.y * k);
        let q = Vector3::new(a * k, -n.x * p.z, n.x * p.y);
        (p, q)
    } else {
        // Choose p in the x-y plane.
        let a = n.x * n.x + n.y * n.y;
        let k = 1.0 / math::sqrt(a);
        let p = Vector3::new(-n.y * k, n.x * k, 0.0);
        let q = Vector3::new(-n.z * p.y, n.z * p.x, a * k);
        (p, q)
    }
}
