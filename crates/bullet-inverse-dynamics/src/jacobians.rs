//! Per-body jacobians — port of `calculateJacobians` and the
//! `addRelativeJacobianComponent` helper (`MultiBodyTreeImpl.cpp:587-678`),
//! plus the world-frame jacobian getters (`:1252-1283`). Storage is a flat
//! row-major 3×num_dofs array per body (`mat3x` equivalent); the exact
//! upstream expression order of `mul(mat33, mat3x)` (`IDMath.cpp:97-115`)
//! is replicated entry-wise.

use crate::math::tilde_operator;
use crate::tree::{KinUpdate, MultiBodyTree, TreeError};
use bullet_math::{Matrix3, Real, Vector3};

/// `mul(const mat33 &a, const mat3x &b, mat3x *result)` — per-column explicit
/// dot products in the reference order (`IDMath.cpp:97-115`).
pub(crate) fn mul_mat3_flat(a: &Matrix3, b: &[Real], out: &mut [Real]) {
    let ac = a.to_cols_array_2d();
    let (b_cols, _) = b.as_chunks::<3>();
    let (out_cols, _) = out.as_chunks_mut::<3>();
    for (b, o) in b_cols.iter().zip(out_cols) {
        o[0] = ac[0][0] * b[0] + ac[1][0] * b[1] + ac[2][0] * b[2];
        o[1] = ac[0][1] * b[0] + ac[1][1] * b[1] + ac[2][1] * b[2];
        o[2] = ac[0][2] * b[0] + ac[1][2] * b[1] + ac[2][2] * b[2];
    }
}

impl MultiBodyTree {
    /// `addRelativeJacobianComponent` (`MultiBodyTreeImpl.cpp:589-634`). The
    /// prismatic column reuses `parent_jac_jt`, bit-identical to the upstream
    /// inline recompute of `body_T_parent_refᵀ · Jac_JT`
    /// (`calculateStaticData`, `:223`).
    fn add_relative_jacobian_component(&mut self, body: usize) {
        let qi = self.q_index[body];
        if qi < 0 {
            return;
        }
        let qi = qi as usize;
        match self.joint_type[body] {
            crate::builder::JointType::Fixed => {}
            crate::builder::JointType::Revolute => {
                let axis = self.jac_jr[body];
                self.body_jac_r[body][qi * 3..qi * 3 + 3]
                    .copy_from_slice(&[axis.x, axis.y, axis.z]);
            }
            crate::builder::JointType::Prismatic => {
                let col = self.parent_jac_jt[body];
                self.body_jac_t[body][qi * 3..qi * 3 + 3]
                    .copy_from_slice(&[col.x, col.y, col.z]);
            }
            crate::builder::JointType::Floating => {
                self.body_jac_r[body][qi * 3] = 1.0;
                self.body_jac_r[body][(qi + 1) * 3 + 1] = 1.0;
                self.body_jac_r[body][(qi + 2) * 3 + 2] = 1.0;
                // jac_t[:, qi+3+j] = row j of body_T_parent; exactly the
                // three dof columns this joint owns.
                let t = self.body_t_parent[body].to_cols_array_2d();
                let (slots, _) =
                    self.body_jac_t[body][(qi + 3) * 3..(qi + 6) * 3].as_chunks_mut::<3>();
                for (j, dst) in slots.iter_mut().enumerate() {
                    *dst = [t[0][j], t[1][j], t[2][j]];
                }
            }
            crate::builder::JointType::Spherical => {
                self.body_jac_r[body][qi * 3] = 1.0;
                self.body_jac_r[body][(qi + 1) * 3 + 1] = 1.0;
                self.body_jac_r[body][(qi + 2) * 3 + 2] = 1.0;
            }
        }
    }

    /// `calculateJacobians(q, u)` at the `POSITION_VELOCITY` level
    /// (`MultiBodyTree.cpp:193-207`). Position + velocity kinematics must be
    /// current (e.g. [`Self::calculate_position_and_velocity_kinematics`]) —
    /// upstream does not refresh them here.
    pub fn calculate_jacobians(&mut self, q: &[Real], u: &[Real]) -> Result<(), TreeError> {
        self.calculate_jacobians_internal(q, u, KinUpdate::PositionVelocity)
    }

    /// `calculateJacobians(q)` at the `POSITION_ONLY` level
    /// (`MultiBodyTree.cpp:209-223`): the velocity-dependent `dot(J)·u`
    /// terms are left untouched.
    pub fn calculate_position_jacobians(&mut self, q: &[Real]) -> Result<(), TreeError> {
        self.calculate_jacobians_internal(q, q, KinUpdate::PositionOnly)
    }

    /// `MultiBodyTreeImpl.cpp:652-675`, verbatim: root component, then per
    /// body `Jac_R = T·parent Jac_R`, `Jac_T = parent Jac_T −
    /// tilde(parent_pos)·parent Jac_R`, own component, left-multiply by `T`;
    /// the `dot(J)·u` pair at the velocity level.
    fn calculate_jacobians_internal(
        &mut self,
        q: &[Real],
        u: &[Real],
        update: KinUpdate,
    ) -> Result<(), TreeError> {
        self.check_slices([q, u])?;

        self.add_relative_jacobian_component(0);
        for i in 1..self.num_bodies {
            let p = self.parent[i] as usize;
            let t = self.body_t_parent[i];
            let tilde = tilde_operator(self.parent_pos_parent_body[i]);

            // Jac_R = T · parent Jac_R.
            mul_mat3_flat(&t, &self.body_jac_r[p], &mut self.mat3x_scratch);
            self.body_jac_r[i].copy_from_slice(&self.mat3x_scratch);
            // Jac_T = parent Jac_T − tilde(parent_pos) · parent Jac_R.
            {
                // Parent indices precede children (the recursion itself
                // relies on it), so split_at_mut keeps the borrow checker
                // happy without cloning.
                let (parents, current) = self.body_jac_t.split_at_mut(i);
                let jac_t = &mut current[0];
                jac_t.copy_from_slice(&parents[p]);
                mul_mat3_flat(&tilde, &self.body_jac_r[p], &mut self.mat3x_scratch);
                for (dst, sub) in jac_t.iter_mut().zip(&self.mat3x_scratch) {
                    *dst -= *sub;
                }
            }
            self.add_relative_jacobian_component(i);
            // Jac_T = T · Jac_T.
            mul_mat3_flat(&t, &self.body_jac_t[i], &mut self.mat3x_scratch);
            self.body_jac_t[i].copy_from_slice(&self.mat3x_scratch);

            if update >= KinUpdate::PositionVelocity {
                let t = self.body_t_parent[i];
                let p_pos = self.parent_pos_parent_body[i];
                let p_dot_r = self.body_dot_jac_r_u[p];
                let p_ang = self.body_ang_vel[p];
                let ang_vel_rel = self.body_ang_vel_rel[i];
                self.body_dot_jac_r_u[i] = t * p_dot_r - ang_vel_rel.cross(t * p_ang);
                let p_dot_t = self.body_dot_jac_t_u[p];
                let p_vel_rel = self.parent_vel_rel[i];
                self.body_dot_jac_t_u[i] = t
                    * (p_dot_t
                        + p_dot_r.cross(p_pos)
                        + p_ang.cross(p_ang.cross(p_pos))
                        + 2.0 * p_ang.cross(p_vel_rel));
            }
        }
        Ok(())
    }

    /// World-frame translational jacobian, row-major 3×num_dofs
    /// (`getBodyJacobianTrans`, `MultiBodyTreeImpl.cpp:1269-1275`).
    pub fn body_jacobian_trans(&self, body: usize, out: &mut [Real]) -> Result<(), TreeError> {
        if body >= self.num_bodies {
            return Err(TreeError::BodyIndex {
                got: body,
                num_bodies: self.num_bodies,
            });
        }
        if out.len() != 3 * self.num_dofs {
            return Err(TreeError::JacobianSlice {
                expected: 3 * self.num_dofs,
                got: out.len(),
            });
        }
        mul_mat3_flat(
            &self.body_t_world[body].transpose(),
            &self.body_jac_t[body],
            out,
        );
        Ok(())
    }

    /// World-frame rotational jacobian, row-major 3×num_dofs
    /// (`getBodyJacobianRot`, `:1277-1283`).
    pub fn body_jacobian_rot(&self, body: usize, out: &mut [Real]) -> Result<(), TreeError> {
        if body >= self.num_bodies {
            return Err(TreeError::BodyIndex {
                got: body,
                num_bodies: self.num_bodies,
            });
        }
        if out.len() != 3 * self.num_dofs {
            return Err(TreeError::JacobianSlice {
                expected: 3 * self.num_dofs,
                got: out.len(),
            });
        }
        mul_mat3_flat(
            &self.body_t_world[body].transpose(),
            &self.body_jac_r[body],
            out,
        );
        Ok(())
    }

    /// `dot(J_T)·u` in world coordinates
    /// (`getBodyDotJacobianTransU`, `:1253-1259`).
    pub fn body_dot_jacobian_trans_u(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose() * self.body_dot_jac_t_u[body])
    }

    /// `dot(J_R)·u` in world coordinates
    /// (`getBodyDotJacobianRotU`, `:1261-1267`).
    pub fn body_dot_jacobian_rot_u(&self, body: usize) -> Option<Vector3> {
        if !self.check_body(body) {
            return None;
        }
        Some(self.body_t_world[body].transpose() * self.body_dot_jac_r_u[body])
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::{JointType, TreeBuilder};
    use crate::tree::{MultiBodyTree, TreeError};
    use bullet_math::{FRAC_PI_2, Matrix3, Real, Vector3};

    fn inertia(v: Real) -> Matrix3 {
        Matrix3::from_cols(
            Vector3::new(v, 0.0, 0.0),
            Vector3::new(0.0, v, 0.0),
            Vector3::new(0.0, 0.0, v),
        )
    }

    /// Fixed root + prismatic along X at z-offset 0.3.
    fn prismatic_tree() -> MultiBodyTree {
        let mut b = TreeBuilder::new();
        b.add_body(0, -1, JointType::Fixed, Vector3::ZERO, Matrix3::IDENTITY,
            Vector3::ZERO, 0.0, Vector3::ZERO, inertia(1.0)).unwrap();
        b.add_body(1, 0, JointType::Prismatic, Vector3::new(0.0, 0.0, 0.3),
            Matrix3::IDENTITY, Vector3::X, 1.0, Vector3::ZERO, inertia(1.0)).unwrap();
        b.finalize().unwrap()
    }

    #[test]
    fn revolute_jacobian_column_is_the_axis() {
        let mut b = TreeBuilder::new();
        b.add_body(0, -1, JointType::Fixed, Vector3::ZERO, Matrix3::IDENTITY,
            Vector3::ZERO, 0.0, Vector3::ZERO, inertia(1.0)).unwrap();
        b.add_body(1, 0, JointType::Revolute, Vector3::ZERO, Matrix3::IDENTITY,
            Vector3::new(0.0, 1.0, 0.0), 1.0, Vector3::ZERO, inertia(1.0)).unwrap();
        let mut tree = b.finalize().unwrap();
        tree.calculate_position_kinematics(&[0.0]).unwrap();
        tree.calculate_position_jacobians(&[0.0]).unwrap();
        let mut jac = [0.0; 3];
        tree.body_jacobian_rot(1, &mut jac).unwrap();
        // World jac_R column = body_T_worldᵀ · (0,1,0); identity frame at
        // q=0 → (0,1,0).
        assert_eq!(jac, [0.0, 1.0, 0.0]);
        // Translational jacobian zero (revolute about a point at the origin).
        tree.body_jacobian_trans(1, &mut jac).unwrap();
        assert_eq!(jac, [0.0; 3]);
        // POSITION_ONLY leaves the dot terms at their finalize values.
        assert_eq!(tree.body_dot_jacobian_rot_u(1), Some(Vector3::ZERO));
    }

    #[test]
    fn prismatic_jacobian_follows_the_reference_transform() {
        let mut tree = prismatic_tree();
        tree.calculate_position_kinematics(&[0.25]).unwrap();
        tree.calculate_position_jacobians(&[0.25]).unwrap();
        let mut jac = [0.0; 3];
        tree.body_jacobian_trans(1, &mut jac).unwrap();
        // Identity reference frame: the column is T_refᵀ·(1,0,0) rotated to
        // world = (1,0,0), independent of the joint position.
        assert_eq!(jac, [1.0, 0.0, 0.0]);
        tree.body_jacobian_rot(1, &mut jac).unwrap();
        assert_eq!(jac, [0.0; 3]);
    }

    #[test]
    fn floating_jacobians_at_identity_and_quarter_turn() {
        let mut b = TreeBuilder::new();
        b.add_body(0, -1, JointType::Floating, Vector3::ZERO, Matrix3::IDENTITY,
            Vector3::ZERO, 1.0, Vector3::ZERO, inertia(1.0)).unwrap();
        let mut tree = b.finalize().unwrap();
        let mut jac = [0.0; 18];

        // At q = 0 the translational columns are the world axes.
        tree.calculate_position_and_velocity_kinematics(&[0.0; 6], &[0.0; 6]).unwrap();
        tree.calculate_jacobians(&[0.0; 6], &[0.0; 6]).unwrap();
        tree.body_jacobian_trans(0, &mut jac).unwrap();
        assert_eq!(&jac[9..12], &[1.0, 0.0, 0.0]);
        assert_eq!(&jac[12..15], &[0.0, 1.0, 0.0]);
        assert_eq!(&jac[15..18], &[0.0, 0.0, 1.0]);
        tree.body_jacobian_rot(0, &mut jac).unwrap();
        assert_eq!(&jac[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&jac[3..6], &[0.0, 1.0, 0.0]);
        assert_eq!(&jac[6..9], &[0.0, 0.0, 1.0]);

        // A quarter turn about y: world jac_T column 3 (body x axis) points
        // along −z. Trig noise kept out of the assertions by tolerance.
        let q = [0.0, FRAC_PI_2, 0.0, 0.0, 0.0, 0.0];
        tree.calculate_position_and_velocity_kinematics(&q, &q).unwrap();
        tree.calculate_jacobians(&q, &[0.0; 6]).unwrap();
        tree.body_jacobian_trans(0, &mut jac).unwrap();
        assert!((jac[9] - (-1.0)).abs() < 1e-6);
        assert!(jac[10].abs() < 1e-6);
        assert!(jac[11].abs() < 1e-6);
    }

    #[test]
    fn jacobian_slice_and_index_enforced() {
        let tree = prismatic_tree();
        assert_eq!(
            tree.body_jacobian_trans(1, &mut [0.0; 4]),
            Err(TreeError::JacobianSlice { expected: 3, got: 4 })
        );
        assert_eq!(
            tree.body_jacobian_rot(9, &mut [0.0; 3]),
            Err(TreeError::BodyIndex { got: 9, num_bodies: 2 })
        );
        assert_eq!(tree.body_dot_jacobian_trans_u(9), None);
        assert_eq!(tree.body_dot_jacobian_rot_u(5), None);
    }
}
