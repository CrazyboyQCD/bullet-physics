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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::{math, Quaternion, Real, Vector3};

use super::body::{MultiBody, MultiBodyScratch};
use crate::body::RigidBody;
use crate::handles::{BodyHandle, MultiBodyHandle};

mod fixed;
mod gear;
mod joint_limit;
mod motor;
mod p2p;
mod slider;
mod spherical;

pub use fixed::MultiBodyFixedConstraint;
pub use gear::MultiBodyGearConstraint;
pub use joint_limit::MultiBodyJointLimit;
pub use motor::MultiBodyJointMotor;
pub use p2p::MultiBodyPoint2Point;
pub use slider::MultiBodySliderConstraint;
pub use spherical::{MultiBodySphericalJointLimit, MultiBodySphericalJointMotor};

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
