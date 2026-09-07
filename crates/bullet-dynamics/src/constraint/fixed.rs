//! Fixed joint (Bullet's `btFixedConstraint`: a locked 6-DOF).

use super::*;
use super::shared::relative_rotation_info;

/// A fixed (weld) joint locking all six degrees of freedom between two
/// frames (Bullet's `btFixedConstraint`, which is a `btGeneric6DofConstraint`
/// with every limit zeroed).
///
/// Rows: three linear pivot rows keeping the frame origins coincident
/// (point-to-point style, world axes) plus three angular rows driving the
/// relative orientation back to identity. The angular rows use the locked
/// branch of the generic 6DOF constraint: the Euler angles of `trA⁻¹·trB`
/// driven to zero along the calculated axes (see [`relative_rotation_info`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FixedConstraint {
    /// Constraint frame on body A (pivot + basis in A's COM frame).
    pub frame_a: JointFrame,
    /// Constraint frame on body B (pivot + basis in B's COM frame).
    pub frame_b: JointFrame,
}

impl FixedConstraint {
    /// A fixed joint keeping `frame_a` and `frame_b` coincident.
    pub fn new(frame_a: JointFrame, frame_b: JointFrame) -> Self {
        Self { frame_a, frame_b }
    }

    /// Flatten this fixed joint into its solver rows (the locked-6DOF
    /// recipe): three world-axis linear rows plus three angular rows along
    /// the relative-rotation axes.
    pub(crate) fn prepare_rows(
        &self,
        body_a: usize,
        body_b: usize,
        bodies: &[RigidBody],
        inv_dt: Real,
        erp: Real,
        out: &mut Vec<ConstraintRow>,
    ) {
        let a = &bodies[body_a];
        let b = &bodies[body_b];
        let tr_a = Matrix3::from_quat(a.orientation) * self.frame_a.basis;
        let tr_b = Matrix3::from_quat(b.orientation) * self.frame_b.basis;
        let pivot_a = a.position + a.orientation * self.frame_a.origin;
        let pivot_b = b.position + b.orientation * self.frame_b.origin;
        let r_a = pivot_a - a.position;
        let r_b = pivot_b - b.position;
        let k = inv_dt * erp;

        // Three linear pivot rows along the world axes (identical to the
        // point-to-point joint): keep the frame origins coincident.
        for axis in 0..3 {
            let mut normal = Vector3::ZERO;
            normal[axis] = 1.0;
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                normal,
                r_a.cross(normal),
                normal,
                r_b.cross(normal),
                a,
                b,
                k * (pivot_a - pivot_b)[axis],
                RowLimits::Unbounded,
            ));
        }

        // Three angular rows locking the relative orientation: the locked
        // 6DOF branch (`lo == hi == 0`, so the violation is the Euler angle
        // itself). Bullet's rotational limit error is `−k·err`, mirrored to
        // our `+k·err` bias; the limits are `[−inf, +inf]` (unbounded).
        let (euler, axes) = relative_rotation_info(&tr_a, &tr_b);
        for i in 0..3 {
            out.push(ConstraintRow::prepare(
                body_a,
                body_b,
                Vector3::ZERO,
                axes[i],
                Vector3::ZERO,
                axes[i],
                a,
                b,
                k * euler[i],
                RowLimits::Unbounded,
            ));
        }
    }
}

