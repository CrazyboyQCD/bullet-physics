//! Slider multibody constraint — port of `btMultiBodySliderConstraint`.

use super::*;
use bullet_math::Matrix3;

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

impl MultiBodyConstraint {
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_slider_rows(
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
}
