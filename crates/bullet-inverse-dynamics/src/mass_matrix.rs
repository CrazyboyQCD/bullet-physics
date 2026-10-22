//! Joint-space mass matrix — port of `MultiBodyTree::calculateMassMatrix`
//! (`MultiBodyTreeImpl.cpp:772-1000`), the "method 3" of Walker & Orin 1982
//! (later named CRBA by Featherstone) formulated on body-frame origins, plus
//! the runtime mass-property setters (`:1186-1206`).

use crate::builder::JointType;
use crate::math::{body_t_parent_from_axis_angle, transform_x, transform_y, transform_z,
                  tilde_operator};
use crate::tree::{MultiBodyTree, TreeError};
use bullet_math::{Matrix3, Real, Vector3};

/// Upstream `jointNumDoFs` (`MultiBodyTreeImpl.cpp:751-768`).
pub(crate) fn joint_num_dofs(joint_type: JointType) -> usize {
    match joint_type {
        JointType::Fixed => 0,
        JointType::Revolute | JointType::Prismatic => 1,
        JointType::Floating => 6,
        JointType::Spherical => 3,
    }
}

/// `setThreeDoFJacobians` (`MultiBodyTreeImpl.cpp:680-704`).
pub(crate) fn set_three_dof_jacobians(dof: usize, jac_jr: &mut Vector3, jac_jt: &mut Vector3) {
    *jac_jr = Vector3::new(
        (dof == 0) as u8 as Real,
        (dof == 1) as u8 as Real,
        (dof == 2) as u8 as Real,
    );
    *jac_jt = Vector3::ZERO;
}

/// `setSixDoFJacobians` (`MultiBodyTreeImpl.cpp:706-749`).
pub(crate) fn set_six_dof_jacobians(dof: usize, jac_jr: &mut Vector3, jac_jt: &mut Vector3) {
    if dof < 3 {
        set_three_dof_jacobians(dof, jac_jr, jac_jt);
    } else {
        *jac_jr = Vector3::ZERO;
        let t = dof - 3;
        *jac_jt = Vector3::new(
            (t == 0) as u8 as Real,
            (t == 1) as u8 as Real,
            (t == 2) as u8 as Real,
        );
    }
}

impl MultiBodyTree {
    /// Joint-space mass matrix, `num_dofs²` row-major into `mass_matrix`
    /// (`MultiBodyTree.cpp:133-136`: update/init/lower all true).
    pub fn calculate_mass_matrix(&mut self, q: &[Real], mass_matrix: &mut [Real]) -> Result<(), TreeError> {
        self.calculate_mass_matrix_ex(q, true, true, true, mass_matrix)
    }

    /// `MultiBodyTree::calculateMassMatrix` with explicit flags
    /// (`MultiBodyTreeImpl.cpp:772-1000`).
    pub fn calculate_mass_matrix_ex(
        &mut self,
        q: &[Real],
        update_kinematics: bool,
        initialize_matrix: bool,
        set_lower_triangular_matrix: bool,
        mass_matrix: &mut [Real],
    ) -> Result<(), TreeError> {
        let expected = self.num_dofs * self.num_dofs;
        if q.len() != self.num_dofs {
            return Err(TreeError::SliceLength {
                expected: self.num_dofs,
                got: q.len(),
            });
        }
        if mass_matrix.len() != expected {
            return Err(TreeError::MassMatrixSlice {
                expected,
                got: mass_matrix.len(),
            });
        }
        if initialize_matrix {
            mass_matrix.fill(0.0);
        }
        if update_kinematics {
            self.update_relative_position(q);
        }

        // Subtree composites, backward (`MultiBodyTreeImpl.cpp:863-897`).
        for i in (0..self.num_bodies).rev() {
            self.subtree_mass[i] = self.mass[i];
            self.body_subtree_mass_com[i] = self.mass_com[i];
            self.body_subtree_i_body[i] = self.inertia[i];
            for &c in &self.child_indices[i] {
                let body_t_child = self.body_t_parent[c].transpose();
                self.subtree_mass[i] += self.subtree_mass[c];
                // Upstream folds the child terms first (`a += b + c`), not
                // left-to-right — preserve the association.
                self.body_subtree_mass_com[i] = self.body_subtree_mass_com[i]
                    + (body_t_child * self.body_subtree_mass_com[c]
                        + self.parent_pos_parent_body[c] * self.subtree_mass[c]);
                self.body_subtree_i_body[i] = self.body_subtree_i_body[i]
                    + body_t_child * self.body_subtree_i_body[c] * self.body_t_parent[c];
                if self.subtree_mass[c] > 0.0 {
                    // Huygens-Steiner shift written as the tilde² difference
                    // (`:883-895`); the division is the btVector3 reciprocal
                    // form `v * (1.0/s)` (`btVector3.h:843-856`).
                    let r_com = body_t_child * self.body_subtree_mass_com[c]
                        * (1.0 / self.subtree_mass[c]);
                    let tilde_r_child_com = tilde_operator(r_com);
                    let tilde_r_body_com =
                        tilde_operator(self.parent_pos_parent_body[c] + r_com);
                    self.body_subtree_i_body[i] = self.body_subtree_i_body[i]
                        + self.subtree_mass[c]
                            * (tilde_r_child_com * tilde_r_child_com
                                - tilde_r_body_com * tilde_r_body_com);
                }
            }
        }

        // Column fill, bodies backward, columns descending, then the ancestor
        // chain (`:899-987`).
        let n = self.num_dofs;
        for i in (0..self.num_bodies).rev() {
            let joint_type = self.joint_type[i];
            let q_index_min = self.q_index[i];
            let q_index_max = q_index_min + joint_num_dofs(joint_type) as isize - 1;
            let mut jac_jr = self.jac_jr[i];
            let mut jac_jt = self.jac_jt[i];
            let mut col = q_index_max;
            while col >= q_index_min {
                let col_u = col as usize;
                if joint_type == JointType::Floating {
                    set_six_dof_jacobians((col - q_index_min) as usize, &mut jac_jr, &mut jac_jt);
                }
                if joint_type == JointType::Spherical {
                    set_three_dof_jacobians(
                        (col - q_index_min) as usize,
                        &mut jac_jr,
                        &mut jac_jt,
                    );
                }
                let mut body_eom_rot = self.body_subtree_i_body[i] * jac_jr
                    + self.body_subtree_mass_com[i].cross(jac_jt);
                let mut body_eom_trans = self.subtree_mass[i] * jac_jt
                    - self.body_subtree_mass_com[i].cross(jac_jr);
                mass_matrix[col_u * n + col_u] =
                    jac_jr.dot(body_eom_rot) + jac_jt.dot(body_eom_trans);

                // 1. rest of the dofs of this body. The upstream `const double
                // Mrc` detour is value-neutral (idScalar sum converts exactly
                // to double and back).
                let mut row = col - 1;
                while row >= q_index_min {
                    let row_u = row as usize;
                    if joint_type == JointType::Spherical {
                        set_three_dof_jacobians(
                            (row - q_index_min) as usize,
                            &mut jac_jr,
                            &mut jac_jt,
                        );
                        mass_matrix[col_u * n + row_u] =
                            jac_jr.dot(body_eom_rot) + jac_jt.dot(body_eom_trans);
                    }
                    if joint_type == JointType::Floating {
                        set_six_dof_jacobians(
                            (row - q_index_min) as usize,
                            &mut jac_jr,
                            &mut jac_jt,
                        );
                        mass_matrix[col_u * n + row_u] =
                            jac_jr.dot(body_eom_rot) + jac_jt.dot(body_eom_trans);
                    }
                    row -= 1;
                }
                // 2. ancestor dofs: rotate the EOM pair into each ancestor's
                // frame and shift the rotational part by r×f.
                let mut child_idx = i as isize;
                let mut parent_idx = self.parent[i];
                while parent_idx >= 0 {
                    let cb = child_idx as usize;
                    let pb = parent_idx as usize;
                    let parent_t_child = self.body_t_parent[cb].transpose();
                    body_eom_rot = parent_t_child * body_eom_rot;
                    body_eom_trans = parent_t_child * body_eom_trans;
                    body_eom_rot += self.parent_pos_parent_body[cb].cross(body_eom_trans);

                    let parent_joint_type = self.joint_type[pb];
                    let p_q_index_min = self.q_index[pb];
                    let p_q_index_max =
                        p_q_index_min + joint_num_dofs(parent_joint_type) as isize - 1;
                    let mut p_jac_jr = self.jac_jr[pb];
                    let mut p_jac_jt = self.jac_jt[pb];
                    let mut row = p_q_index_max;
                    while row >= p_q_index_min {
                        let row_u = row as usize;
                        if parent_joint_type == JointType::Spherical {
                            set_three_dof_jacobians(
                                (row - p_q_index_min) as usize,
                                &mut p_jac_jr,
                                &mut p_jac_jt,
                            );
                        }
                        if parent_joint_type == JointType::Floating {
                            set_six_dof_jacobians(
                                (row - p_q_index_min) as usize,
                                &mut p_jac_jr,
                                &mut p_jac_jt,
                            );
                        }
                        mass_matrix[col_u * n + row_u] =
                            p_jac_jr.dot(body_eom_rot) + p_jac_jt.dot(body_eom_trans);
                        row -= 1;
                    }
                    child_idx = parent_idx;
                    parent_idx = self.parent[child_idx as usize];
                }
                col -= 1;
            }
        }

        if set_lower_triangular_matrix {
            for col in 0..n {
                for row in 0..col {
                    mass_matrix[row * n + col] = mass_matrix[col * n + row];
                }
            }
        }
        Ok(())
    }

    /// Position-level relative kinematics, the `update_kinematics` branch of
    /// `calculateMassMatrix` (`MultiBodyTreeImpl.cpp:808-862`). Absolute
    /// state is left untouched, exactly like upstream (getters read it
    /// stale until the next kinematics call).
    fn update_relative_position(&mut self, q: &[Real]) {
        for &i in &self.revolute_list {
            let qi = self.q_index[i] as usize;
            let t = body_t_parent_from_axis_angle(self.jac_jr[i], q[qi]);
            self.body_t_parent[i] = t * self.body_t_parent_ref[i];
        }
        for &i in &self.prismatic_list {
            let qi = self.q_index[i] as usize;
            self.parent_pos_parent_body[i] =
                self.parent_pos_parent_body_ref[i] + self.parent_jac_jt[i] * q[qi];
        }
        for &i in &self.floating_list {
            let qi = self.q_index[i] as usize;
            self.body_t_parent[i] = transform_z(q[qi + 2]) * transform_y(q[qi + 1])
                * transform_x(q[qi]);
            let pos = Vector3::new(q[qi + 3], q[qi + 4], q[qi + 5]);
            self.parent_pos_parent_body[i] = self.body_t_parent[i] * pos;
        }
        for &i in &self.spherical_list {
            let qi = self.q_index[i] as usize;
            let t = transform_x(q[qi]) * transform_y(q[qi + 1]) * transform_z(q[qi + 2]);
            self.body_t_parent[i] = t * self.body_t_parent_ref[i];
            self.parent_pos_parent_body[i] = self.body_t_parent[i] * Vector3::ZERO;
        }
    }

    /// Overwrite a body's mass (`setBodyMass`, `MultiBodyTreeImpl.cpp:1186-1191`).
    pub fn set_body_mass(&mut self, body: usize, mass: Real) -> Result<(), TreeError> {
        if body >= self.num_bodies {
            return Err(TreeError::BodyIndex {
                got: body,
                num_bodies: self.num_bodies,
            });
        }
        self.mass[body] = mass;
        Ok(())
    }

    /// Overwrite a body's first mass moment, mass × center of mass in the
    /// body-fixed frame (`setBodyFirstMassMoment`, `:1193-1199`).
    pub fn set_body_first_mass_moment(
        &mut self,
        body: usize,
        first_mass_moment: Vector3,
    ) -> Result<(), TreeError> {
        if body >= self.num_bodies {
            return Err(TreeError::BodyIndex {
                got: body,
                num_bodies: self.num_bodies,
            });
        }
        self.mass_com[body] = first_mass_moment;
        Ok(())
    }

    /// Overwrite a body's second mass moment, the body-frame inertia
    /// (`setBodySecondMassMoment`, `:1200-1206`).
    pub fn set_body_second_mass_moment(
        &mut self,
        body: usize,
        second_mass_moment: Matrix3,
    ) -> Result<(), TreeError> {
        if body >= self.num_bodies {
            return Err(TreeError::BodyIndex {
                got: body,
                num_bodies: self.num_bodies,
            });
        }
        self.inertia[body] = second_mass_moment;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::joint_num_dofs;
    use crate::builder::{JointType, TreeBuilder};
    use crate::tree::{MultiBodyTree, TreeError};
    use bullet_math::{FRAC_PI_2, Matrix3, Real, Vector3};

    fn inertia(xx: Real, yy: Real, zz: Real) -> Matrix3 {
        Matrix3::from_cols(
            Vector3::new(xx, 0.0, 0.0),
            Vector3::new(0.0, yy, 0.0),
            Vector3::new(0.0, 0.0, zz),
        )
    }

    /// body 0: fixed root; body 1: revolute about Z at x=1, mass 1 at
    /// (0.5,0,0), origin inertia diag(0.2,0.15,0.1). No children.
    fn two_link_arm() -> MultiBodyTree {
        let mut b = TreeBuilder::new();
        b.add_body(0, -1, JointType::Fixed, Vector3::ZERO, Matrix3::IDENTITY,
            Vector3::ZERO, 0.0, Vector3::ZERO, inertia(1.0, 1.0, 1.0)).unwrap();
        b.add_body(1, 0, JointType::Revolute, Vector3::new(1.0, 0.0, 0.0),
            Matrix3::IDENTITY, Vector3::Z, 1.0, Vector3::new(0.5, 0.0, 0.0),
            inertia(0.2, 0.15, 0.1)).unwrap();
        b.finalize().unwrap()
    }

    #[test]
    fn joint_num_dofs_matches_dof_accounting() {
        assert_eq!(joint_num_dofs(JointType::Fixed), 0);
        assert_eq!(joint_num_dofs(JointType::Revolute), 1);
        assert_eq!(joint_num_dofs(JointType::Prismatic), 1);
        assert_eq!(joint_num_dofs(JointType::Floating), 6);
        assert_eq!(joint_num_dofs(JointType::Spherical), 3);
    }

    #[test]
    fn childless_revolute_picks_the_axis_inertia_entry() {
        // The supplied body inertia is frame-origin-referenced (upstream
        // semantics), so for a childless body the joint-space mass is the
        // axis entry of the given tensor — independent of q and mass.
        let mut tree = two_link_arm();
        let mut mm = [0.0; 1];
        tree.calculate_mass_matrix(&[0.0], &mut mm).unwrap();
        assert_eq!(mm[0], 0.1);
        tree.calculate_mass_matrix(&[FRAC_PI_2], &mut mm).unwrap();
        assert_eq!(mm[0], 0.1);
    }

    #[test]
    fn mass_matrix_slice_length_enforced() {
        let mut tree = two_link_arm();
        let mut mm = [0.0; 4];
        assert_eq!(
            tree.calculate_mass_matrix(&[0.0], &mut mm),
            Err(TreeError::MassMatrixSlice { expected: 1, got: 4 })
        );
        assert_eq!(
            tree.calculate_mass_matrix(&[0.0, 0.0], &mut [0.0; 1]),
            Err(TreeError::SliceLength { expected: 1, got: 2 })
        );
    }

    /// Single floating body, mass 1, com (0.1, 0.2, 0.3), origin inertia
    /// diag(0.3, 0.25, 0.2): M = blkdiag(m·I₃, I) plus the first-moment
    /// coupling M[rot_j][trans_i] = e_j·(com×e_i) — all entries are exact
    /// input copies.
    fn floating_body() -> MultiBodyTree {
        let mut b = TreeBuilder::new();
        b.add_body(0, -1, JointType::Floating, Vector3::ZERO, Matrix3::IDENTITY,
            Vector3::ZERO, 1.0, Vector3::new(0.1, 0.2, 0.3),
            inertia(0.3, 0.25, 0.2)).unwrap();
        b.finalize().unwrap()
    }

    #[test]
    fn floating_body_blocks_and_first_moment_coupling() {
        let mut tree = floating_body();
        let mut mm = [0.0; 36];
        tree.calculate_mass_matrix(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &mut mm).unwrap();
        for i in 0..3 {
            assert_eq!(mm[(3 + i) * 6 + (3 + i)], 1.0, "trans {i}");
            assert_eq!(mm[i * 6 + i], [0.3, 0.25, 0.2][i], "rot {i}");
        }
        // com×e_x = (0, 0.3, −0.2); com×e_y = (−0.3, 0, 0.1);
        // com×e_z = (0.2, −0.1, 0).
        let coupling = [[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]];
        for j in 0..3 {
            for i in 0..3 {
                assert_eq!(mm[(3 + i) * 6 + j], coupling[j][i], "M[rot{j}][trans{i}]");
                assert_eq!(mm[j * 6 + (3 + i)], coupling[j][i], "mirror");
            }
        }
        // Cross entries inside the rotational block stay zero.
        assert_eq!(mm[1], 0.0);
        assert_eq!(mm[8], 0.0);
    }

    #[test]
    fn mass_matrix_is_symmetric_after_mirror() {
        let mut b = TreeBuilder::new();
        b.add_body(0, -1, JointType::Floating, Vector3::ZERO, Matrix3::IDENTITY,
            Vector3::ZERO, 1.0, Vector3::new(0.1, 0.2, 0.3),
            inertia(0.3, 0.25, 0.2)).unwrap();
        b.add_body(1, 0, JointType::Spherical, Vector3::new(0.0, 0.0, 1.0),
            Matrix3::IDENTITY, Vector3::ZERO, 0.5, Vector3::new(0.0, 0.0, -0.5),
            inertia(0.1, 0.1, 0.1)).unwrap();
        let mut tree = b.finalize().unwrap();
        let mut mm = [0.0; 81];
        tree.calculate_mass_matrix(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], &mut mm)
            .unwrap();
        for r in 0..9 {
            for c in 0..9 {
                let a = mm[r * 9 + c];
                let sym = mm[c * 9 + r];
                assert!(
                    (a - sym).abs() <= 1e-9 * a.abs().max(1.0),
                    "M[{r}][{c}]={a} vs M[{c}][{r}]={sym}"
                );
            }
        }
        // Positive diagonal everywhere (a mass matrix property).
        assert!(mm.iter().enumerate().all(|(k, &v)| k % 10 != 0 || v > 0.0));
    }

    #[test]
    fn mass_setters_feed_the_next_composition() {
        // Childless revolute: mass never enters the composite (no children to
        // shift, Jac_JT = 0) — the tensor setter is what moves the value.
        let mut tree = two_link_arm();
        let mut mm = [0.0; 1];
        tree.calculate_mass_matrix(&[0.0], &mut mm).unwrap();
        tree.set_body_mass(1, 5.0).unwrap();
        tree.calculate_mass_matrix(&[0.0], &mut mm).unwrap();
        assert_eq!(mm[0], 0.1);
        tree.set_body_second_mass_moment(1, inertia(0.2, 0.15, 0.7)).unwrap();
        tree.calculate_mass_matrix(&[0.0], &mut mm).unwrap();
        assert_eq!(mm[0], 0.7);

        // Floating body: the translational block is the mass itself.
        let mut tree = floating_body();
        let mut mm = [0.0; 36];
        tree.set_body_mass(0, 2.0).unwrap();
        tree.set_body_first_mass_moment(0, Vector3::new(0.4, 0.0, 0.0)).unwrap();
        tree.calculate_mass_matrix(&[0.0; 6], &mut mm).unwrap();
        assert_eq!(mm[21], 2.0);
        // Coupling tracks the first moment, not the com: M[rot_y][trans_x]
        // = com_z = first_moment_z / mass = 0.
        assert_eq!(mm[3 * 6 + 1], 0.0);

        assert_eq!(
            tree.set_body_mass(5, 1.0),
            Err(TreeError::BodyIndex { got: 5, num_bodies: 1 })
        );
        assert_eq!(
            tree.set_body_first_mass_moment(1, Vector3::ZERO),
            Err(TreeError::BodyIndex { got: 1, num_bodies: 1 })
        );
        assert_eq!(
            tree.set_body_second_mass_moment(9, Matrix3::ZERO),
            Err(TreeError::BodyIndex { got: 9, num_bodies: 1 })
        );
    }
}
