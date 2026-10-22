//! The finalized tree — port of `MultiBodyTree::MultiBodyImpl`
//! (`details/MultiBodyTreeImpl.cpp`): three-level kinematics, the inverse
//! dynamics pass, and the world-frame getters. The mass matrix and the
//! jacobian surface follow in M2.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::builder::JointType;
use crate::math::{body_t_parent_from_axis_angle, transform_x, transform_y, transform_z};
use bullet_math::{Matrix3, Real, Vector3};

/// Calculate/get rejection reasons. The upstream `int` returns and the
/// `NotFinalized` runtime flag map to the type system here (a
/// `MultiBodyTree` only exists finalized); the one upstream `abort()` in
/// `jointNumDoFs` (`MultiBodyTreeImpl.cpp:768`) is unreachable after
/// construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TreeError {
    /// A q/u/dot_u/joint_forces slice did not match `num_dofs`.
    SliceLength { expected: usize, got: usize },
}

/// Kinematic update depth (`MultiBodyTreeImpl.hpp:137-142`); the
/// `POSITION_ONLY` level joins with the M2 jacobian surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum KinUpdate {
    PositionVelocity,
    PositionVelocityAcceleration,
}

/// A finalized inverse-dynamics tree. Construct via [`TreeBuilder::finalize`].
#[derive(Debug)]
pub struct MultiBodyTree {
    // ── static configuration (immutable after finalize) ──
    pub(crate) num_bodies: usize,
    pub(crate) num_dofs: usize,
    pub(crate) world_gravity: Vector3,
    pub(crate) joint_type: Vec<JointType>,
    pub(crate) parent: Vec<isize>,
    pub(crate) q_index: Vec<isize>,
    pub(crate) parent_pos_parent_body_ref: Vec<Vector3>,
    pub(crate) body_t_parent_ref: Vec<Matrix3>,
    pub(crate) jac_jr: Vec<Vector3>,
    pub(crate) jac_jt: Vec<Vector3>,
    /// `body_T_parent_refᵀ · jac_jt`, precomputed for prismatic joints in
    /// `calculateStaticData` (`MultiBodyTreeImpl.cpp:223`); zero elsewhere.
    pub(crate) parent_jac_jt: Vec<Vector3>,
    pub(crate) mass: Vec<Real>,
    /// Mass × center of mass in the body-fixed frame
    /// (`m_body_mass_com`).
    pub(crate) mass_com: Vec<Vector3>,
    pub(crate) inertia: Vec<Matrix3>,
    // ── index sets (`generateIndexSets`) ──
    pub(crate) child_indices: Vec<Vec<usize>>,
    pub(crate) revolute_list: Vec<usize>,
    pub(crate) prismatic_list: Vec<usize>,
    pub(crate) floating_list: Vec<usize>,
    pub(crate) spherical_list: Vec<usize>,
    // ── kinematic state ──
    pub(crate) body_t_parent: Vec<Matrix3>,
    pub(crate) body_t_world: Vec<Matrix3>,
    pub(crate) parent_pos_parent_body: Vec<Vector3>,
    pub(crate) body_ang_vel_rel: Vec<Vector3>,
    pub(crate) parent_vel_rel: Vec<Vector3>,
    pub(crate) body_ang_acc_rel: Vec<Vector3>,
    pub(crate) parent_acc_rel: Vec<Vector3>,
    pub(crate) body_pos: Vec<Vector3>,
    pub(crate) body_vel: Vec<Vector3>,
    /// Includes gravitational acceleration unless gravity is zeroed
    /// (`MultiBodyTreeImpl.cpp:539-542`).
    pub(crate) body_acc: Vec<Vector3>,
    pub(crate) body_ang_vel: Vec<Vector3>,
    pub(crate) body_ang_acc: Vec<Vector3>,
    // ── dynamic state ──
    pub(crate) eom_lhs_rotational: Vec<Vector3>,
    pub(crate) eom_lhs_translational: Vec<Vector3>,
    pub(crate) force_at_joint: Vec<Vector3>,
    pub(crate) moment_at_joint: Vec<Vector3>,
    pub(crate) body_force_user: Vec<Vector3>,
    pub(crate) body_moment_user: Vec<Vector3>,
    // ── policy flags carried from the builder ──
    pub(crate) accept_invalid_mass_parameters: bool,
    pub(crate) mass_parameters_are_valid: bool,
}

impl MultiBodyTree {
    /// Number of bodies, including the root environment body.
    pub fn num_bodies(&self) -> usize {
        self.num_bodies
    }

    /// Dimension of the q vector.
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }

    /// Whether any body carried invalid mass properties (accepted or not).
    pub fn mass_parameters_are_valid(&self) -> bool {
        self.mass_parameters_are_valid
    }

    /// Whether invalid mass properties are accepted
    /// (`getAcceptInvalidMassProperties`).
    pub fn accept_invalid_mass_parameters(&self) -> bool {
        self.accept_invalid_mass_parameters
    }

    /// Set gravitational acceleration in the world frame; default
    /// `(0, 0, −9.8)` (`MultiBodyTreeImpl.cpp:21-23`).
    pub fn set_gravity_in_world_frame(&mut self, gravity: Vector3) {
        self.world_gravity = gravity;
    }

    /// Parent index of a body; `None` when the index is out of range. The
    /// root body reports −1.
    pub fn parent_index(&self, body: usize) -> Option<isize> {
        self.parent.get(body).copied()
    }

    /// Joint type of a body.
    pub fn joint_type(&self, body: usize) -> Option<JointType> {
        self.joint_type.get(body).copied()
    }

    /// Offset of this body's dofs in the q vector; `None` for fixed joints
    /// (upstream stores −1) and out-of-range indices.
    pub fn dof_offset(&self, body: usize) -> Option<usize> {
        match self.q_index.get(body) {
            Some(q) if *q >= 0 => Some(*q as usize),
            _ => None,
        }
    }

    fn check_slices(&self, slices: [&[Real]; 3]) -> Result<(), TreeError> {
        for slice in slices {
            if slice.len() != self.num_dofs {
                return Err(TreeError::SliceLength {
                    expected: self.num_dofs,
                    got: slice.len(),
                });
            }
        }
        Ok(())
    }

    /// Static (q,u,dot_u-independent) relative kinematics
    /// (`MultiBodyTreeImpl.cpp:204-276`). The jacobian zeroing there is an
    /// M2 concern.
    pub(crate) fn calculate_static_data(&mut self) {
        for i in 0..self.num_bodies {
            match self.joint_type[i] {
                JointType::Revolute => {
                    self.parent_vel_rel[i] = Vector3::ZERO;
                    self.parent_acc_rel[i] = Vector3::ZERO;
                    self.parent_pos_parent_body[i] = self.parent_pos_parent_body_ref[i];
                }
                JointType::Prismatic => {
                    self.body_t_parent[i] = self.body_t_parent_ref[i];
                    self.parent_jac_jt[i] = self.body_t_parent_ref[i].transpose() * self.jac_jt[i];
                    self.body_ang_vel_rel[i] = Vector3::ZERO;
                    self.body_ang_acc_rel[i] = Vector3::ZERO;
                }
                JointType::Fixed => {
                    self.parent_pos_parent_body[i] = self.parent_pos_parent_body_ref[i];
                    self.body_t_parent[i] = self.body_t_parent_ref[i];
                    self.body_ang_vel_rel[i] = Vector3::ZERO;
                    self.parent_vel_rel[i] = Vector3::ZERO;
                    self.body_ang_acc_rel[i] = Vector3::ZERO;
                    self.parent_acc_rel[i] = Vector3::ZERO;
                }
                JointType::Floating => {}
                JointType::Spherical => {
                    self.parent_pos_parent_body[i] = self.parent_pos_parent_body_ref[i];
                    self.parent_vel_rel[i] = Vector3::ZERO;
                    self.parent_acc_rel[i] = Vector3::ZERO;
                }
            }
        }
    }

    /// `calculateKinematics` at the full level with gravity temporarily
    /// zeroed, so accelerations come out gravity-free
    /// (`MultiBodyTree.cpp:138-158`).
    pub fn calculate_kinematics(
        &mut self,
        q: &[Real],
        u: &[Real],
        dot_u: &[Real],
    ) -> Result<(), TreeError> {
        let gravity = self.world_gravity;
        self.world_gravity = Vector3::ZERO;
        let result =
            self.calculate_kinematics_internal(q, u, dot_u, KinUpdate::PositionVelocityAcceleration);
        self.world_gravity = gravity;
        result
    }

    /// Position kinematics only. Upstream calls
    /// `calculateKinematics(q, q, q, POSITION_VELOCITY)` — velocities are
    /// evaluated at u = q (`MultiBodyTree.cpp:160-174`), a quirk preserved
    /// verbatim (they are not meaningful outputs for this entry point).
    pub fn calculate_position_kinematics(&mut self, q: &[Real]) -> Result<(), TreeError> {
        self.calculate_kinematics_internal(q, q, q, KinUpdate::PositionVelocity)
    }

    /// Position and velocity kinematics. Upstream passes `(q, u, u)`; the
    /// third argument is unused at the `POSITION_VELOCITY` level
    /// (`MultiBodyTree.cpp:176-190`).
    pub fn calculate_position_and_velocity_kinematics(
        &mut self,
        q: &[Real],
        u: &[Real],
    ) -> Result<(), TreeError> {
        self.calculate_kinematics_internal(q, u, u, KinUpdate::PositionVelocity)
    }

    /// `MultiBodyTreeImpl.cpp:386-585`, verbatim: per-joint-type relative
    /// updates in the order revolute → prismatic → floating → spherical,
    /// then the absolute recursion from the root.
    fn calculate_kinematics_internal(
        &mut self,
        q: &[Real],
        u: &[Real],
        dot_u: &[Real],
        update: KinUpdate,
    ) -> Result<(), TreeError> {
        self.check_slices([q, u, dot_u])?;

        // 1. relative kinematics.
        // 1.1 revolute.
        for &i in &self.revolute_list {
            let qi = self.q_index[i] as usize;
            let t = body_t_parent_from_axis_angle(self.jac_jr[i], q[qi]);
            self.body_t_parent[i] = t * self.body_t_parent_ref[i];
            if update >= KinUpdate::PositionVelocity {
                self.body_ang_vel_rel[i] = self.jac_jr[i] * u[qi];
            }
            if update >= KinUpdate::PositionVelocityAcceleration {
                self.body_ang_acc_rel[i] = self.jac_jr[i] * dot_u[qi];
            }
        }
        // 1.2 prismatic.
        for &i in &self.prismatic_list {
            let qi = self.q_index[i] as usize;
            self.parent_pos_parent_body[i] =
                self.parent_pos_parent_body_ref[i] + self.parent_jac_jt[i] * q[qi];
            if update >= KinUpdate::PositionVelocity {
                self.parent_vel_rel[i] =
                    self.body_t_parent_ref[i].transpose() * self.jac_jt[i] * u[qi];
            }
            if update >= KinUpdate::PositionVelocityAcceleration {
                self.parent_acc_rel[i] = self.parent_jac_jt[i] * dot_u[qi];
            }
        }
        // 1.3 fixed joints: nothing to do.
        // 1.4 floating (6-dof) joints.
        for &i in &self.floating_list {
            let qi = self.q_index[i] as usize;
            self.body_t_parent[i] = transform_z(q[qi + 2]) * transform_y(q[qi + 1])
                * transform_x(q[qi]);
            let mut pos = Vector3::new(q[qi + 3], q[qi + 4], q[qi + 5]);
            self.parent_pos_parent_body[i] = self.body_t_parent[i] * pos;
            if update >= KinUpdate::PositionVelocity {
                self.body_ang_vel_rel[i] = Vector3::new(u[qi], u[qi + 1], u[qi + 2]);
                pos = Vector3::new(u[qi + 3], u[qi + 4], u[qi + 5]);
                self.parent_vel_rel[i] = self.body_t_parent[i].transpose() * pos;
            }
            if update >= KinUpdate::PositionVelocityAcceleration {
                self.body_ang_acc_rel[i] =
                    Vector3::new(dot_u[qi], dot_u[qi + 1], dot_u[qi + 2]);
                pos = Vector3::new(dot_u[qi + 3], dot_u[qi + 4], dot_u[qi + 5]);
                self.parent_acc_rel[i] = self.body_t_parent[i].transpose() * pos;
            }
        }
        // 1.5 spherical (3-dof) joints.
        for &i in &self.spherical_list {
            let qi = self.q_index[i] as usize;
            let t = transform_x(q[qi]) * transform_y(q[qi + 1]) * transform_z(q[qi + 2]);
            self.body_t_parent[i] = t * self.body_t_parent_ref[i];
            self.parent_pos_parent_body[i] = self.body_t_parent[i] * Vector3::ZERO;
            if update >= KinUpdate::PositionVelocity {
                self.body_ang_vel_rel[i] = Vector3::new(u[qi], u[qi + 1], u[qi + 2]);
                self.parent_vel_rel[i] = self.body_t_parent[i].transpose() * self.parent_vel_rel[i];
            }
            if update >= KinUpdate::PositionVelocityAcceleration {
                self.body_ang_acc_rel[i] =
                    Vector3::new(dot_u[qi], dot_u[qi + 1], dot_u[qi + 2]);
                self.parent_acc_rel[i] = self.body_t_parent[i].transpose() * self.parent_acc_rel[i];
            }
        }

        // 2. absolute kinematics, root first.
        {
            self.body_pos[0] = self.body_t_parent[0] * self.parent_pos_parent_body[0];
            self.body_t_world[0] = self.body_t_parent[0];
            if update >= KinUpdate::PositionVelocity {
                self.body_ang_vel[0] = self.body_ang_vel_rel[0];
                self.body_vel[0] = self.parent_vel_rel[0];
            }
            if update >= KinUpdate::PositionVelocityAcceleration {
                self.body_ang_acc[0] = self.body_ang_acc_rel[0];
                self.body_acc[0] = self.body_t_parent[0] * self.parent_acc_rel[0];
                self.body_acc[0] = self.body_acc[0] - self.body_t_parent[0] * self.world_gravity;
            }
        }

        for i in 1..self.num_bodies {
            let p = self.parent[i] as usize;
            self.body_pos[i] =
                self.body_t_parent[i] * (self.body_pos[p] + self.parent_pos_parent_body[i]);
            self.body_t_world[i] = self.body_t_parent[i] * self.body_t_world[p];

            if update >= KinUpdate::PositionVelocity {
                self.body_ang_vel[i] =
                    self.body_t_parent[i] * self.body_ang_vel[p] + self.body_ang_vel_rel[i];
                self.body_vel[i] = self.body_t_parent[i]
                    * (self.body_vel[p]
                        + self.body_ang_vel[p].cross(self.parent_pos_parent_body[i])
                        + self.parent_vel_rel[i]);
            }
            if update >= KinUpdate::PositionVelocityAcceleration {
                self.body_ang_acc[i] = self.body_t_parent[i] * self.body_ang_acc[p]
                    - self
                        .body_ang_vel_rel[i]
                        .cross(self.body_t_parent[i] * self.body_ang_vel[p])
                    + self.body_ang_acc_rel[i];
                self.body_acc[i] = self.body_t_parent[i]
                    * (self.body_acc[p]
                        + self.body_ang_acc[p].cross(self.parent_pos_parent_body[i])
                        + self
                            .body_ang_vel[p]
                            .cross(self.body_ang_vel[p].cross(self.parent_pos_parent_body[i]))
                        + 2.0 * self.body_ang_vel[p].cross(self.parent_vel_rel[i])
                        + self.parent_acc_rel[i]);
            }
        }

        Ok(())
    }

    /// Port of `calculateInverseDynamics` (`MultiBodyTreeImpl.cpp:278-384`):
    /// full-level kinematics, per-body Newton-Euler EOM left-hand sides,
    /// backward parent-joint wrench recursion, then the projection onto the
    /// free directions per joint type. Gravity participates through the
    /// kinematics acceleration terms (unlike [`Self::calculate_kinematics`]).
    pub fn calculate_inverse_dynamics(
        &mut self,
        q: &[Real],
        u: &[Real],
        dot_u: &[Real],
        joint_forces: &mut [Real],
    ) -> Result<(), TreeError> {
        self.check_slices([q, u, dot_u])?;
        if joint_forces.len() != self.num_dofs {
            return Err(TreeError::SliceLength {
                expected: self.num_dofs,
                got: joint_forces.len(),
            });
        }

        self.calculate_kinematics_internal(q, u, dot_u, KinUpdate::PositionVelocityAcceleration)?;

        // Equation-of-motion left-hand sides (rate of change of angular and
        // linear momentum), minus user wrenches.
        for i in 0..self.num_bodies {
            self.eom_lhs_rotational[i] = self.inertia[i] * self.body_ang_acc[i]
                + self.mass_com[i].cross(self.body_acc[i])
                + self.body_ang_vel[i].cross(self.inertia[i] * self.body_ang_vel[i])
                - self.body_moment_user[i];
            self.eom_lhs_translational[i] = self.body_ang_acc[i].cross(self.mass_com[i])
                + self.mass[i] * self.body_acc[i]
                + self
                    .body_ang_vel[i]
                    .cross(self.body_ang_vel[i].cross(self.mass_com[i]))
                - self.body_force_user[i];
        }

        // Wrench at the parent joint, summed backward over children.
        for body_idx in (0..self.num_bodies).rev() {
            let mut sum_f_children = Vector3::ZERO;
            let mut sum_m_children = Vector3::ZERO;
            for &child in &self.child_indices[body_idx] {
                let child_joint_force_in_this_frame =
                    self.body_t_parent[child].transpose() * self.force_at_joint[child];
                sum_f_children -= child_joint_force_in_this_frame;
                sum_m_children -= self.body_t_parent[child].transpose()
                    * self.moment_at_joint[child]
                    + self.parent_pos_parent_body[child]
                        .cross(child_joint_force_in_this_frame);
            }
            self.force_at_joint[body_idx] =
                self.eom_lhs_translational[body_idx] - sum_f_children;
            self.moment_at_joint[body_idx] = self.eom_lhs_rotational[body_idx] - sum_m_children;
        }

        // Projection onto the free directions, per joint type.
        for &i in &self.revolute_list {
            let qi = self.q_index[i] as usize;
            joint_forces[qi] = self.jac_jr[i].dot(self.moment_at_joint[i]);
        }
        for &i in &self.prismatic_list {
            let qi = self.q_index[i] as usize;
            joint_forces[qi] = self.jac_jt[i].dot(self.force_at_joint[i]);
        }
        for &i in &self.floating_list {
            let qi = self.q_index[i] as usize;
            joint_forces[qi] = self.moment_at_joint[i].x;
            joint_forces[qi + 1] = self.moment_at_joint[i].y;
            joint_forces[qi + 2] = self.moment_at_joint[i].z;
            joint_forces[qi + 3] = self.force_at_joint[i].x;
            joint_forces[qi + 4] = self.force_at_joint[i].y;
            joint_forces[qi + 5] = self.force_at_joint[i].z;
        }
        for &i in &self.spherical_list {
            let qi = self.q_index[i] as usize;
            joint_forces[qi] = self.moment_at_joint[i].x;
            joint_forces[qi + 1] = self.moment_at_joint[i].y;
            joint_forces[qi + 2] = self.moment_at_joint[i].z;
        }

        Ok(())
    }

    /// Accumulate an external force at the body-fixed frame origin.
    pub fn add_user_force(&mut self, body: usize, force: Vector3) {
        self.body_force_user[body] += force;
    }

    /// Accumulate an external moment at the body-fixed frame origin.
    pub fn add_user_moment(&mut self, body: usize, moment: Vector3) {
        self.body_moment_user[body] += moment;
    }

    /// Zero all user forces and moments.
    pub fn clear_all_user_forces_and_moments(&mut self) {
        for i in 0..self.num_bodies {
            self.body_force_user[i] = Vector3::ZERO;
            self.body_moment_user[i] = Vector3::ZERO;
        }
    }

    fn check_body(&self, body: usize) -> bool {
        body < self.num_bodies
    }

    /// Origin of the body-fixed frame in world coordinates.
    pub fn body_origin(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose() * self.body_pos[body])
    }

    /// Center of mass in world coordinates (the frame origin for zero-mass
    /// bodies, `MultiBodyTreeImpl.cpp:1055-1069`).
    pub fn body_com(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        let t = self.body_t_world[body].transpose();
        if self.mass[body] > 0.0 {
            Some(t * (self.body_pos[body] + self.mass_com[body] / self.mass[body]))
        } else {
            Some(t * self.body_pos[body])
        }
    }

    /// Body-to-world rotation.
    pub fn body_transform(&self, body: usize) -> Option<Matrix3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose())
    }

    /// Absolute angular velocity in world coordinates.
    pub fn body_angular_velocity(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose() * self.body_ang_vel[body])
    }

    /// Linear velocity of the body-fixed frame origin in world coordinates.
    pub fn body_linear_velocity(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose() * self.body_vel[body])
    }

    /// Linear velocity of the center of mass in world coordinates.
    pub fn body_linear_velocity_com(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        let com = if self.mass[body] > 0.0 {
            self.mass_com[body] / self.mass[body]
        } else {
            Vector3::ZERO
        };
        Some(
            self.body_t_world[body].transpose()
                * (self.body_vel[body] + self.body_ang_vel[body].cross(com)),
        )
    }

    /// Absolute angular acceleration in world coordinates.
    pub fn body_angular_acceleration(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose() * self.body_ang_acc[body])
    }

    /// Absolute linear acceleration in world coordinates. Includes
    /// gravitational acceleration unless gravity was zeroed
    /// (`MultiBodyTree.cpp` note on `getBodyLinearAcceleration`).
    pub fn body_linear_acceleration(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose() * self.body_acc[body])
    }
}

#[cfg(test)]
mod tests {
    use super::MultiBodyTree;
    use crate::builder::{JointType, TreeBuilder};
    use crate::TreeError;
    use bullet_math::{FRAC_PI_2, Matrix3, Real, Vector3};

    fn identity() -> Matrix3 {
        Matrix3::IDENTITY
    }

    fn inertia(v: Real) -> Matrix3 {
        Matrix3::from_cols(
            Vector3::new(v, 0.0, 0.0),
            Vector3::new(0.0, v, 0.0),
            Vector3::new(0.0, 0.0, v),
        )
    }

    fn close(a: Vector3, b: Vector3, eps: Real) -> bool {
        (a.x - b.x).abs() < eps && (a.y - b.y).abs() < eps && (a.z - b.z).abs() < eps
    }

    /// Root fixed at the origin, then two Z-revolute links, each with its
    /// joint one unit along the parent x.
    fn two_r_arm() -> MultiBodyTree {
        let mut b = TreeBuilder::new();
        b.add_body(
            0,
            -1,
            JointType::Fixed,
            Vector3::ZERO,
            identity(),
            Vector3::ZERO,
            0.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        for i in 1..=2usize {
            b.add_body(
                i,
                i as isize - 1,
                JointType::Revolute,
                Vector3::new(1.0, 0.0, 0.0),
                identity(),
                Vector3::Z,
                1.0,
                Vector3::ZERO,
                inertia(1.0),
            )
            .unwrap();
        }
        b.finalize().unwrap()
    }

    #[test]
    fn two_r_position_hand_computed() {
        let mut tree = two_r_arm();
        // The axis-angle form is the transposed rotation: the body-to-world
        // map is the standard CCW Rz(q) (upstream convention, pinned here).
        tree.set_gravity_in_world_frame(Vector3::ZERO);
        tree.calculate_position_kinematics(&[FRAC_PI_2, 0.0]).unwrap();
        // Body 1's frame origin sits on its joint axis: fixed in world.
        assert!(close(
            tree.body_origin(1).unwrap(),
            Vector3::new(1.0, 0.0, 0.0),
            1e-6
        ));
        // Body 2's joint is one unit along body 1's x: Rz(pi/2)·(1,0,0).
        assert!(close(
            tree.body_origin(2).unwrap(),
            Vector3::new(1.0, 1.0, 0.0),
            1e-6
        ));
        // The exposed transform is body-to-world (T_worldᵀ).
        let t = tree.body_transform(1).unwrap();
        assert!(close(t.x_axis, Vector3::new(0.0, 1.0, 0.0), 1e-6));
        assert!(close(t.y_axis, Vector3::new(-1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn pendulum_gravity_torque_hand_computed() {
        let mut b = TreeBuilder::new();
        b.add_body(
            0,
            -1,
            JointType::Fixed,
            Vector3::ZERO,
            identity(),
            Vector3::ZERO,
            0.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        b.add_body(
            1,
            0,
            JointType::Revolute,
            Vector3::ZERO,
            identity(),
            Vector3::new(0.0, 1.0, 0.0),
            2.0,
            Vector3::new(1.0, 0.0, 0.0),
            inertia(0.5),
        )
        .unwrap();
        let mut tree = b.finalize().unwrap();

        // At q=0 the arm points along +x; gravity −z. The static-hold torque
        // about the y axis is τ = −m·g·r = −2·9.8·1.
        let mut tau = [0.0];
        tree.calculate_inverse_dynamics(&[0.0], &[0.0], &[0.0], &mut tau)
            .unwrap();
        assert!((tau[0] - (-2.0 * 9.8)).abs() < 1e-5);

        // Arm parallel to gravity: zero torque.
        tree.calculate_inverse_dynamics(&[FRAC_PI_2], &[0.0], &[0.0], &mut tau)
            .unwrap();
        assert!(tau[0].abs() < 1e-5);
    }

    #[test]
    fn zero_state_zero_gravity_gives_zero_forces() {
        let mut tree = two_r_arm();
        tree.set_gravity_in_world_frame(Vector3::ZERO);
        let mut tau = [0.0; 2];
        tree.calculate_inverse_dynamics(&[0.0; 2], &[0.0; 2], &[0.0; 2], &mut tau)
            .unwrap();
        assert_eq!(tau, [0.0; 2]);
    }

    #[test]
    fn user_force_enters_the_joint_projection() {
        let mut b = TreeBuilder::new();
        b.add_body(
            0,
            -1,
            JointType::Fixed,
            Vector3::ZERO,
            identity(),
            Vector3::ZERO,
            0.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        b.add_body(
            1,
            0,
            JointType::Prismatic,
            Vector3::ZERO,
            identity(),
            Vector3::new(1.0, 0.0, 0.0),
            0.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        let mut tree = b.finalize().unwrap();
        tree.set_gravity_in_world_frame(Vector3::ZERO);
        // A user force −2x on the body: the EOM left-hand side subtracts it,
        // so the joint must deliver +2 along the slide axis.
        tree.add_user_force(1, Vector3::new(-2.0, 0.0, 0.0));
        let mut tau = [0.0];
        tree.calculate_inverse_dynamics(&[0.0], &[0.0], &[0.0], &mut tau)
            .unwrap();
        assert!((tau[0] - 2.0).abs() < 1e-6);
        tree.clear_all_user_forces_and_moments();
        tree.calculate_inverse_dynamics(&[0.0], &[0.0], &[0.0], &mut tau)
            .unwrap();
        assert_eq!(tau[0], 0.0);
    }

    #[test]
    fn floating_root_pin_and_velocity() {
        let mut b = TreeBuilder::new();
        b.add_body(
            0,
            -1,
            JointType::Floating,
            Vector3::ZERO,
            identity(),
            Vector3::ZERO,
            1.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        let mut tree = b.finalize().unwrap();
        tree.set_gravity_in_world_frame(Vector3::ZERO);
        // Verbatim upstream chain: the floating relative update rotates the
        // displacement by body_T_parent, and the root block rotates it once
        // more — the doubled application is upstream behavior, pinned here
        // so any future "fix" is a conscious deviation.
        let q = [FRAC_PI_2, 0.0, 0.0, 1.0, 2.0, 3.0];
        let u = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        tree.calculate_position_and_velocity_kinematics(&q, &u).unwrap();
        // The stored body_pos carries the doubled T application, but the
        // world origin cancels it (T_worldᵀ·T·T·d = T·d): the displacement
        // rotated once into the world.
        assert!(close(
            tree.body_origin(0).unwrap(),
            Vector3::new(1.0, 3.0, -2.0),
            1e-6
        ));
        assert!(close(
            tree.body_linear_velocity(0).unwrap(),
            Vector3::new(0.0, -1.0, 0.0),
            1e-6
        ));
        assert!(close(
            tree.body_angular_velocity(0).unwrap(),
            Vector3::new(1.0, 0.0, 0.0),
            1e-6
        ));
    }

    #[test]
    fn kinematics_wrapper_hides_gravity_in_accelerations() {
        let mut tree = two_r_arm();
        tree.calculate_kinematics(&[0.0; 2], &[0.0; 2], &[0.0; 2]).unwrap();
        // Gravity restored after the call, and accelerations are free of it.
        assert!(close(
            tree.body_linear_acceleration(1).unwrap(),
            Vector3::ZERO,
            1e-6
        ));
        // The inverse-dynamics path keeps gravity folded in: the bodies hang
        // with the negated world gravity in their acceleration terms.
        let mut tau = [0.0; 2];
        tree.calculate_inverse_dynamics(&[0.0; 2], &[0.0; 2], &[0.0; 2], &mut tau)
            .unwrap();
        assert!(close(
            tree.body_linear_acceleration(1).unwrap(),
            Vector3::new(0.0, 0.0, 9.8),
            1e-6
        ));
    }

    #[test]
    fn slice_length_mismatch_rejected() {
        let mut tree = two_r_arm();
        assert_eq!(
            tree.calculate_kinematics(&[0.0], &[0.0; 2], &[0.0; 2]),
            Err(TreeError::SliceLength {
                expected: 2,
                got: 1
            })
        );
        let mut tau = [0.0; 3];
        assert_eq!(
            tree.calculate_inverse_dynamics(&[0.0; 2], &[0.0; 2], &[0.0; 2], &mut tau),
            Err(TreeError::SliceLength {
                expected: 2,
                got: 3
            })
        );
        assert_eq!(tree.body_origin(9), None);
    }
}
