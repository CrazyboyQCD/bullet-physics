//! Gear multibody constraint — port of `btMultiBodyGearConstraint`.

use super::*;
use crate::multibody::link::JointType;

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

impl MultiBodyConstraint {
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_gear_row(
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
}
