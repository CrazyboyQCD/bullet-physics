//! 1-DOF joint motor multibody constraint — port of `btMultiBodyJointMotor`.

use super::*;
use crate::multibody::link::JointType;

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

impl MultiBodyConstraint {
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_motor_row(
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
}
