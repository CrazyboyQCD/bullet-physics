//! Spherical joint limit/motor multibody constraints — ports of
//! `btMultiBodySphericalJointLimit` and `btMultiBodySphericalJointMotor`.
//! The two share the `set_spherical_*` target/gain setters (Bullet's
//! `m_use_multi_dof_params` semantics apply to both).

use super::*;
use bullet_math::{Matrix3, Quaternion};

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

impl MultiBodyConstraint {
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

    /// Port of `btMultiBodySphericalJointLimit::createConstraintRows` — up to
    /// three unilateral angular rows keeping the joint's swing (about the
    /// link frame's world X/Y) and twist (about its world Z) within the given
    /// ranges. The rows are built from the current joint quaternion split
    /// into its cone and twist parts; each row's window is one-sided so the
    /// limit can only push the joint back inside.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_spherical_limit_rows(
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
    pub(crate) fn create_spherical_motor_rows(
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
