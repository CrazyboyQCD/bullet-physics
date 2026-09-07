//! Point-to-point (ball-socket) multibody constraint — port of
//! `btMultiBodyPoint2Point`.

use super::*;

/// Parameter block of a point-to-point (ball-socket) constraint (Bullet's
/// `btMultiBodyPoint2Point`).
#[derive(Debug, Clone)]
pub struct MultiBodyPoint2Point {
    /// Pivot point in link A's local frame.
    pub pivot_in_a: Vector3,
    /// Pivot point in link B's local frame.
    pub pivot_in_b: Vector3,
}

impl MultiBodyConstraint {
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_p2p_rows(
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
}
