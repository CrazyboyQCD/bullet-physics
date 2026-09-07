//! 1-DOF joint limit multibody constraint — port of
//! `btMultiBodyJointLimitConstraint`.

use super::*;
use crate::multibody::link::JointType;
use super::SPLIT_PENETRATION_THRESHOLD;

/// Parameter block of a 1-DOF joint limit (Bullet's
/// `btMultiBodyJointLimitConstraint`): clamps the position of a revolute or
/// prismatic joint to `[lower_bound, upper_bound]` with two unilateral rows.
#[derive(Debug, Clone)]
pub struct MultiBodyJointLimit {
    pub lower_bound: Real,
    pub upper_bound: Real,
}

impl MultiBodyConstraint {
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

    /// Port of `btMultiBodyJointLimitConstraint::createConstraintRows` — two
    /// unilateral rows on the joint's single DOF (row 0 clamps the lower
    /// bound, row 1 the upper). The jacobians are the finalized unit rows; the
    /// RHS is the classic limit formula (positional + velocity error),
    /// overriding the fill's default. Like Bullet, a violation deeper than
    /// `m_splitImpulsePenetrationThreshold` drops the positional term (the
    /// multibody solver ignores `m_rhsPenetration`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_joint_limit_rows(
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
}
