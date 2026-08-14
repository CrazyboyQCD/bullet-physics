//! Multibody constraints 鈥?port of `BulletDynamics/Featherstone`'s
//! `btMultiBodyConstraint` subclasses (`btMultiBodyPoint2Point`,
//! `btMultiBodyFixedConstraint`, `btMultiBodySliderConstraint`,
//! `btMultiBodyJointMotor`, `btMultiBodyGearConstraint`).
//!
//! A [`MultiBodyConstraint`] couples one link of a
//! [`crate::multibody::MultiBody`] to a link of another multibody (or to the
//! base of the same/different multibody via `link = None`). Every step the
//! constraint emits a set of 1D rows ([`SolverRow`]); each row carries the
//! generalized jacobian of the constrained DOFs on both bodies, the velocity
//! response of each body to a unit impulse along the row (computed through
//! [`MultiBody::calc_acceleration_deltas`], which reuses the ABA
//! factorization), the effective-mass inverse `jac_diag_ab_inv` and the
//! Baumgarte/velocity RHS. The solver then runs projected Gauss-Seidel over
//! the rows and writes the accumulated impulses back into the generalized
//! velocities.
//!
//! Deviation from Bullet: only multibody-vs-multibody constraints are
//! supported (Bullet also allows multibody-vs-rigidbody); the joint feedback
//! (`btMultiBodyJointFeedback`) reporting is not implemented.

use bullet_math::{Matrix3, Real, Vector3};

use super::body::{MultiBody, MultiBodyScratch};
use super::link::JointType;

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
    /// Same for body B.
    pub jac_b_index: usize,
    pub delta_vel_b_index: i32,
    /// Number of DOFs (6 + joints) of each body.
    pub ndof_a: usize,
    pub ndof_b: usize,
    /// Lever-arm 脳 normal / angular normal used for feedback writeback
    /// (Bullet's `m_relpos1CrossNormal` / `m_contactNormal1`).
    pub rel_pos1_cross_normal: Vector3,
    pub contact_normal1: Vector3,
    pub rel_pos2_cross_normal: Vector3,
    pub contact_normal2: Vector3,
    /// Inverse effective mass `1/(J路M鈦宦孤稪岬€)` along the row (0 disables the
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
    /// Indices of the two multibodies inside the world.
    pub multi_body_a: usize,
    pub multi_body_b: usize,
    /// Constrained links (`None` = the base).
    pub link_a: Option<usize>,
    pub link_b: Option<usize>,
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
/// `卤max_motor_impulse`.
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
/// the joint of link B on body B with `q虈_a + ratio路q虈_b = 0` (plus the
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

/// The kind of a [`MultiBodyConstraint`] and its parameters.
#[derive(Debug, Clone)]
pub enum MultiBodyConstraintKind {
    Point2Point(MultiBodyPoint2Point),
    Fixed(MultiBodyFixedConstraint),
    Slider(MultiBodySliderConstraint),
    JointMotor(MultiBodyJointMotor),
    Gear(MultiBodyGearConstraint),
}

/// A constraint between two multibodies (or two links of the same body).
pub struct MultiBodyConstraint {
    pub(crate) kind: MultiBodyConstraintKind,
    pub(crate) body_a: usize,
    pub(crate) body_b: usize,
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
    /// Data block: `num_rows` accumulated impulses, then `num_rows 路
    /// jac_size_both` jacobian entries, then `num_rows` positions.
    pub(crate) data: Vec<Real>,
    pub(crate) pos_offset: usize,
    pub(crate) num_dofs_finalized: usize,
}

impl MultiBodyConstraint {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: MultiBodyConstraintKind,
        body_a: usize,
        body_b: usize,
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
        self.jac_size_a = 6 + bodies[self.body_a].num_dofs();
        self.jac_size_both = self.jac_size_a + 6 + bodies[self.body_b].num_dofs();
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

    /// Bullet's `internalSetAppliedImpulse` 鈥?write the accumulated impulse
    /// of `dof` back (used for warm starting; multibody rows are not
    /// warm-started by default, but the writeback keeps the API faithful).
    pub fn set_applied_impulse(&mut self, dof: usize, impulse: Real) {
        self.data[dof] = impulse;
    }

    pub fn get_applied_impulse(&self, dof: usize) -> Real {
        self.data[dof]
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

    pub fn body_a(&self) -> usize {
        self.body_a
    }

    pub fn body_b(&self) -> usize {
        self.body_b
    }

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// A point-to-point constraint keeping `pivot_in_a` (link A's frame)
    /// coincident with `pivot_in_b` (link B's frame) 鈥?three bilateral rows
    /// (Bullet's `btMultiBodyPoint2Point`).
    pub fn point2point(
        body_a: usize,
        link_a: Option<usize>,
        body_b: usize,
        link_b: Option<usize>,
        pivot_in_a: Vector3,
        pivot_in_b: Vector3,
        bodies: &[MultiBody],
    ) -> Self {
        Self::new(
            MultiBodyConstraintKind::Point2Point(MultiBodyPoint2Point { pivot_in_a, pivot_in_b }),
            body_a,
            body_b,
            link_a,
            link_b,
            3,
            false,
            bodies,
        )
    }

    /// A fixed constraint welding link A to link B (pivots coincident and
    /// frames aligned) 鈥?six bilateral rows (Bullet's
    /// `btMultiBodyFixedConstraint`).
    #[allow(clippy::too_many_arguments)]
    pub fn fixed(
        body_a: usize,
        link_a: Option<usize>,
        body_b: usize,
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
            link_a,
            link_b,
            6,
            false,
            bodies,
        )
    }

    /// A slider constraint allowing translation along `joint_axis` (link A's
    /// frame) 鈥?five bilateral rows (Bullet's `btMultiBodySliderConstraint`).
    #[allow(clippy::too_many_arguments)]
    pub fn slider(
        body_a: usize,
        link_a: Option<usize>,
        body_b: usize,
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
            link_a,
            link_b,
            5,
            false,
            bodies,
        )
    }

    /// A joint motor driving link `link`'s joint DOF `link_dof` toward
    /// `desired_velocity` with a maximum impulse `max_motor_impulse` 鈥?one
    /// unilateral row. The motor acts on body `body` alone (Bullet's
    /// `btMultiBodyJointMotor`; `body_b` is the same body).
    pub fn joint_motor(
        body: usize,
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
            Some(link),
            bodies[body].link(link).parent,
            1,
            true,
            bodies,
        );
        c.max_applied_impulse = max_motor_impulse;
        c
    }

    /// A gear constraint coupling link A's joint to link B's joint with the
    /// ratio `q虈_a + ratio路q虈_b = 0` 鈥?one bilateral row (Bullet's
    /// `btMultiBodyGearConstraint`).
    pub fn gear(
        body_a: usize,
        link_a: usize,
        body_b: usize,
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

    fn finalize_gear(&mut self, bodies: &[MultiBody]) {
        // The gear jacobians never change, so initialize them here (Bullet's
        // `finalizeMultiDof`).
        let link_dof = 0;
        let link_a = self.link_a.unwrap();
        let link_b = self.link_b.unwrap();
        let offset_a = 6 + bodies[self.body_a].link(link_a).dof_offset + link_dof;
        let offset_b = 6 + bodies[self.body_b].link(link_b).dof_offset + link_dof;
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
        let offset = 6 + bodies[self.body_a].link(link).dof_offset + link_dof;
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
        dt: Real,
        erp: Real,
    ) {
        match &self.kind {
            MultiBodyConstraintKind::Point2Point(_) => {
                self.create_p2p_rows(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::Fixed(_) => {
                self.create_fixed_rows(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::Slider(_) => {
                self.create_slider_rows(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::JointMotor(_) => {
                self.create_motor_row(my_index, rows, data, scratch, bodies, dt, erp)
            }
            MultiBodyConstraintKind::Gear(_) => {
                self.create_gear_row(my_index, rows, data, scratch, bodies, dt, erp)
            }
        }
    }

    /// Port of `btMultiBodyConstraint::fillMultiBodyConstraint`: populate a
    /// row's jacobians (from `jac_org_a`/`jac_org_b` if given, else computed
    /// from the constraint normals), unit impulse responses, effective mass,
    /// RHS and limits.
    #[allow(clippy::too_many_arguments)]
    fn fill_row(
        &self,
        row: &mut SolverRow,
        data: &mut SolverData,
        scratch: &mut MultiBodyScratch,
        bodies: &mut [MultiBody],
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
    ) {
        let multi_body_a = self.body_a;
        let multi_body_b = self.body_b;
        let ndof_a = 6 + bodies[multi_body_a].num_dofs();
        let ndof_b = 6 + bodies[multi_body_b].num_dofs();

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
            let jac = data.jacobians[jac_a_start..jac_a_start + ndof_a].to_vec();
            let delta = &mut data.delta_velocities_unit_impulse[delta_start..delta_start + ndof_a];
            bodies[multi_body_a].calc_acceleration_deltas(&jac, delta, scratch);
        }
        let torque_axis0 = if ang_constraint {
            constraint_normal_ang
        } else {
            rel_pos1.cross(constraint_normal_lin)
        };
        row.rel_pos1_cross_normal = torque_axis0;
        row.contact_normal1 = constraint_normal_lin;

        // --- Body B ---
        let rel_pos2 = match self.link_b {
            None => pos_b_world - bodies[multi_body_b].base_pos(),
            Some(l) => pos_b_world - Vector3::from(bodies[multi_body_b].link(l).cached_world_transform.translation),
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
                data.jacobians[jac_b_start..jac_b_start + ndof_b].copy_from_slice(&j[..ndof_b]);
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
            let jac = data.jacobians[jac_b_start..jac_b_start + ndof_b].to_vec();
            let delta = &mut data.delta_velocities_unit_impulse[delta_b_start..delta_b_start + ndof_b];
            bodies[multi_body_b].calc_acceleration_deltas(&jac, delta, scratch);
        }
        let torque_axis1 = if ang_constraint {
            constraint_normal_ang
        } else {
            rel_pos2.cross(constraint_normal_lin)
        };
        row.rel_pos2_cross_normal = -torque_axis1;
        row.contact_normal2 = -constraint_normal_lin;

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
        {
            let jac = &data.jacobians[jac_b_start..jac_b_start + ndof_b];
            let delta = &data.delta_velocities_unit_impulse[delta_b_start..delta_b_start + ndof_b];
            for i in 0..ndof_b {
                denom1 += jac[i] * delta[i];
            }
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
            let jac = &data.jacobians[jac_b_start..jac_b_start + ndof_b];
            let vel = bodies[multi_body_b].velocity_vector();
            for i in 0..ndof_b {
                rel_vel += vel[i] * jac[i];
            }
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
            multi_body_a: self.body_a,
            multi_body_b: self.body_b,
            link_a: self.link_a,
            link_b: self.link_b,
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
        dt: Real,
        erp: Real,
    ) {
        let (pivot_in_a, pivot_in_b) = match &self.kind {
            MultiBodyConstraintKind::Point2Point(p) => (p.pivot_in_a, p.pivot_in_b),
            _ => unreachable!(),
        };
        for i in 0..3 {
            let mut row = self.new_row(my_index, i);
            row.ndof_a = 6 + bodies[self.body_a].num_dofs();
            row.ndof_b = 6 + bodies[self.body_b].num_dofs();
            // contactNormalOnB[i] = -1 (Bullet).
            let mut contact_normal_on_b = Vector3::ZERO;
            contact_normal_on_b[i] = -1.0;

            let pivot_a_world = bodies[self.body_a].local_pos_to_world(
                self.link_a.map_or(-1, |l| l as i32),
                pivot_in_a,
            );
            let pivot_b_world = bodies[self.body_b].local_pos_to_world(
                self.link_b.map_or(-1, |l| l as i32),
                pivot_in_b,
            );
            let pos_error = (pivot_a_world - pivot_b_world).dot(contact_normal_on_b);

            self.fill_row(
                &mut row,
                data,
                scratch,
                bodies,
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
        dt: Real,
        erp: Real,
    ) {
        let (pivot_in_a, pivot_in_b, frame_in_a, frame_in_b) = match &self.kind {
            MultiBodyConstraintKind::Fixed(f) => {
                (f.pivot_in_a, f.pivot_in_b, f.frame_in_a, f.frame_in_b)
            }
            _ => unreachable!(),
        };
        let pivot_a_world = bodies[self.body_a].local_pos_to_world(self.link_a.map_or(-1, |l| l as i32), pivot_in_a);
        let frame_a_world = bodies[self.body_a].local_frame_to_world(self.link_a.map_or(-1, |l| l as i32), &frame_in_a);
        let pivot_b_world = bodies[self.body_b].local_pos_to_world(self.link_b.map_or(-1, |l| l as i32), pivot_in_b);
        let frame_b_world = bodies[self.body_b].local_frame_to_world(self.link_b.map_or(-1, |l| l as i32), &frame_in_b);
        // The relative rotation between the two world frames (Bullet's
        // `relRot = frameAworld.inverse() * frameBworld`).
        let rel_rot = frame_a_world.inverse() * frame_b_world;
        let angle_diff = crate::constraint::euler_xyz(&rel_rot);

        for i in 0..6 {
            let mut row = self.new_row(my_index, i);
            row.ndof_a = 6 + bodies[self.body_a].num_dofs();
            row.ndof_b = 6 + bodies[self.body_b].num_dofs();
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
        dt: Real,
        erp: Real,
    ) {
        let (pivot_in_a, pivot_in_b, frame_in_a, frame_in_b, joint_axis) = match &self.kind {
            MultiBodyConstraintKind::Slider(s) => {
                (s.pivot_in_a, s.pivot_in_b, s.frame_in_a, s.frame_in_b, s.joint_axis)
            }
            _ => unreachable!(),
        };
        let pivot_a_world = bodies[self.body_a].local_pos_to_world(self.link_a.map_or(-1, |l| l as i32), pivot_in_a);
        let frame_a_world = bodies[self.body_a].local_frame_to_world(self.link_a.map_or(-1, |l| l as i32), &frame_in_a);
        let joint_axis = bodies[self.body_a].local_dir_to_world(self.link_a.map_or(-1, |l| l as i32), joint_axis);
        let pivot_b_world = bodies[self.body_b].local_pos_to_world(self.link_b.map_or(-1, |l| l as i32), pivot_in_b);
        let frame_b_world = bodies[self.body_b].local_frame_to_world(self.link_b.map_or(-1, |l| l as i32), &frame_in_b);

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
            row.ndof_a = 6 + bodies[self.body_a].num_dofs();
            row.ndof_b = 6 + bodies[self.body_b].num_dofs();
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
        row.ndof_a = 6 + bodies[self.body_a].num_dofs();
        row.ndof_b = 6 + bodies[self.body_b].num_dofs();

        let link = self.link_a.unwrap();
        let current_position = bodies[self.body_a].joint_pos_multidof(link)[link_dof];
        let current_velocity = bodies[self.body_a].joint_vel_multidof(link)[link_dof];
        let position_stabilization_term = motor_erp * (desired_position - current_position) / dt;
        let velocity_error = desired_velocity - current_velocity;
        let mut rhs = kp * position_stabilization_term + current_velocity + kd * velocity_error;
        rhs = rhs.clamp(-rhs_clamp, rhs_clamp);

        let jac_a = self.jacobian_a(0).to_vec();
        let jac_b = self.jacobian_b(0).to_vec();
        self.fill_row(
            &mut row,
            data,
            scratch,
            bodies,
            Some(&jac_a),
            Some(&jac_b),
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
        let link_world_rot = bodies[self.body_a].link_world_orientation(link);
        let link_ref = bodies[self.body_a].link(link);
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
        row.ndof_a = 6 + bodies[self.body_a].num_dofs();
        row.ndof_b = 6 + bodies[self.body_b].num_dofs();

        let link_a = self.link_a.unwrap();
        let link_b = self.link_b.unwrap();
        let link_dof = 0;
        let mut aux_vel = 0.0;
        if let Some(aux) = gear_aux_link {
            aux_vel = bodies[self.body_a].joint_vel_multidof(aux)[link_dof];
        }
        let mut pos_error = 0.0;
        if gear_erp != 0.0 {
            let mut current_position_a = bodies[self.body_a].joint_pos_multidof(link_a)[link_dof];
            if let Some(aux) = gear_aux_link {
                current_position_a -= bodies[self.body_a].joint_pos_multidof(aux)[link_dof];
            }
            let current_position_b = gear_ratio * bodies[self.body_b].joint_pos_multidof(link_b)[link_dof];
            let diff = current_position_b + current_position_a;
            pos_error = -gear_erp * (relative_position_target - diff);
        }
        let desired_relative_velocity = aux_vel;

        let jac_a = self.jacobian_a(0).to_vec();
        let jac_b = self.jacobian_b(0).to_vec();
        self.fill_row(
            &mut row,
            data,
            scratch,
            bodies,
            Some(&jac_a),
            Some(&jac_b),
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
        let link_world_rot = bodies[self.body_a].link_world_orientation(link_a);
        let link_ref = bodies[self.body_a].link(link_a);
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
}
