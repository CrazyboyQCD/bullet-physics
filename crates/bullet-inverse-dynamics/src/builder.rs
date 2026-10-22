//! Tree construction — port of `MultiBodyTree::addBody`/`finalize` and the
//! `InitCache` builder cache (`MultiBodyTree.cpp:247-484`,
//! `MultiBodyTreeInitCache.cpp`).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::math::{is_unit_vector, is_valid_inertia_matrix, is_valid_transform_matrix};
use crate::tree::MultiBodyTree;
use bullet_math::{math, Matrix3, Real, Vector3};

/// Joint types (`MultiBodyTree.hpp:10-22`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointType {
    /// No degree of freedom; moves with the parent.
    Fixed,
    /// One rotational degree of freedom relative to the parent.
    Revolute,
    /// One translational degree of freedom relative to the parent.
    Prismatic,
    /// Six degrees of freedom relative to the parent.
    Floating,
    /// Three rotational degrees of freedom relative to the parent.
    Spherical,
}

/// `addBody` rejection reasons. The Rust port adds one gate the reference
/// lacks: `body` must equal the insertion position (upstream assumes proper
/// indices without checking, `MultiBodyTreeInitCache.cpp:119-120`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AddBodyError {
    /// A second body claims parent −1 (`MultiBodyTreeInitCache.cpp:44-49`).
    MultipleRoots { existing: usize },
    /// `body` differs from the insertion position.
    IndexMismatch { expected: usize },
    /// Revolute/prismatic axis is a unit vector scaled below
    /// `sqrt(REAL_MIN)`, which cannot be normalized
    /// (`MultiBodyTree.cpp:272-276`).
    AxisTooShort,
    /// Negative mass with `accept_invalid_mass_parameters == false`
    /// (`MultiBodyTree.cpp:292-300`).
    NegativeMass,
    /// Inertia matrix failed the upstream gates with
    /// `accept_invalid_mass_parameters == false`
    /// (`MultiBodyTree.cpp:302-310`).
    InvalidInertiaMatrix,
    /// Reference transform is not a proper rotation
    /// (`MultiBodyTree.cpp:312-315`; no accept override exists for this
    /// gate upstream).
    InvalidTransformMatrix,
}

/// `finalize` rejection reasons. Upstream additionally guards
/// `num_dofs < 0` with a commented-out return (`MultiBodyTree.cpp:352-356`);
/// the port replaces that dead check with the type system (dofs are
/// unsigned).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuildError {
    /// A non-root body's parent index failed the
    /// `0 <= parent < num_bodies - 1` gate
    /// (`MultiBodyTreeImpl.cpp:176-199`).
    BadParentIndex { child: usize, parent: isize },
}

#[derive(Clone, Copy, Debug)]
struct InertiaData {
    mass: Real,
    body_pos_body_com: Vector3,
    body_i_body: Matrix3,
}

#[derive(Clone, Copy, Debug)]
struct JointData {
    parent: isize,
    joint_type: JointType,
    parent_pos_parent_child_ref: Vector3,
    child_t_parent_ref: Matrix3,
    child_axis_of_motion: Vector3,
}

/// Builder-side storage for `addBody` calls (`InitCache` equivalent).
#[derive(Debug)]
pub struct TreeBuilder {
    inertias: Vec<InertiaData>,
    joints: Vec<JointData>,
    num_dofs: usize,
    root_index: Option<usize>,
    accept_invalid_mass_parameters: bool,
    mass_parameters_are_valid: bool,
}

impl Default for TreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeBuilder {
    pub fn new() -> Self {
        Self {
            inertias: Vec::new(),
            joints: Vec::new(),
            num_dofs: 0,
            root_index: None,
            accept_invalid_mass_parameters: false,
            mass_parameters_are_valid: true,
        }
    }

    /// Upstream `setAcceptInvalidMassParameters`: when set, invalid mass
    /// properties are recorded and accepted instead of rejected
    /// (`MultiBodyTree.cpp:29-37`).
    pub fn accept_invalid_mass_parameters(&mut self, flag: bool) -> &mut Self {
        self.accept_invalid_mass_parameters = flag;
        self
    }

    /// Whether any added body carried invalid mass properties (upstream
    /// keeps the flag internal; exposed here so the information is not
    /// silently dropped).
    pub fn mass_parameters_are_valid(&self) -> bool {
        self.mass_parameters_are_valid
    }

    /// Port of `MultiBodyTree::addBody` (`MultiBodyTree.cpp:247-320`) plus
    /// `InitCache::addBody` (`MultiBodyTreeInitCache.cpp:13-71`). Bodies are
    /// appended; `body` must be the next free index. The root body is the
    /// one with `parent == -1` (by convention index 0).
    #[allow(clippy::too_many_arguments)] // upstream addBody parameter list
    pub fn add_body(
        &mut self,
        body: usize,
        parent: isize,
        joint_type: JointType,
        parent_pos_parent_body_ref: Vector3,
        body_t_parent_ref: Matrix3,
        axis_of_motion: Vector3,
        mass: Real,
        body_com: Vector3,
        body_inertia: Matrix3,
    ) -> Result<(), AddBodyError> {
        if body != self.inertias.len() {
            return Err(AddBodyError::IndexMismatch {
                expected: self.inertias.len(),
            });
        }

        let mut axis = axis_of_motion;
        match joint_type {
            JointType::Revolute | JointType::Prismatic => {
                if !is_unit_vector(axis) {
                    // btPow(x, 2), verbatim (`MultiBodyTree.cpp:269-271`); the
                    // constant exponent is black-boxed so LLVM cannot rewrite
                    // pow(x, 2.0) into x·x (the two disagree on some inputs).
                    let length = math::sqrt(
                        math::powf(axis.x, core::hint::black_box(2.0))
                            + math::powf(axis.y, core::hint::black_box(2.0))
                            + math::powf(axis.z, core::hint::black_box(2.0)),
                    );
                    if length < math::sqrt(Real::MIN_POSITIVE) {
                        return Err(AddBodyError::AxisTooShort);
                    }
                    axis = (1.0 / length) * axis;
                }
            }
            JointType::Fixed | JointType::Floating | JointType::Spherical => {}
        }

        if mass < 0.0 {
            self.mass_parameters_are_valid = false;
            if !self.accept_invalid_mass_parameters {
                return Err(AddBodyError::NegativeMass);
            }
        }

        if !is_valid_inertia_matrix(&body_inertia, joint_type == JointType::Fixed) {
            self.mass_parameters_are_valid = false;
            if !self.accept_invalid_mass_parameters {
                return Err(AddBodyError::InvalidInertiaMatrix);
            }
        }

        if !is_valid_transform_matrix(&body_t_parent_ref) {
            return Err(AddBodyError::InvalidTransformMatrix);
        }

        self.num_dofs += match joint_type {
            JointType::Revolute | JointType::Prismatic => 1,
            JointType::Fixed => 0,
            JointType::Spherical => 3,
            JointType::Floating => 6,
        };

        if parent == -1 {
            if let Some(existing) = self.root_index {
                return Err(AddBodyError::MultipleRoots { existing });
            }
            self.root_index = Some(body);
        }

        self.joints.push(JointData {
            parent,
            joint_type,
            parent_pos_parent_child_ref: parent_pos_parent_body_ref,
            child_t_parent_ref: body_t_parent_ref,
            child_axis_of_motion: axis,
        });
        self.inertias.push(InertiaData {
            mass,
            body_pos_body_com: body_com,
            body_i_body: body_inertia,
        });
        Ok(())
    }

    /// Port of `MultiBodyTree::finalize` (`MultiBodyTree.cpp:347-484`) in
    /// the reference order: parent index set, per-body static data (with the
    /// zero jacobian initialization for fixed/floating/spherical,
    /// `:415-465`), dof index sets, joint-geometry precomputation, and the
    /// user-force zeroing.
    pub fn finalize(self) -> Result<MultiBodyTree, BuildError> {
        let num_bodies = self.joints.len();
        let num_dofs = self.num_dofs;

        // 2. build the parent index set (`InitCache::buildIndexSets`). The
        // insertion position is the child index (validated in `add_body`).
        let mut parent = vec![0isize; num_bodies];
        for (child, joint) in self.joints.iter().enumerate() {
            parent[child] = joint.parent;
        }

        // 3. per-body kinematic/dynamic static data.
        let mut tree = MultiBodyTree {
            num_bodies,
            num_dofs,
            world_gravity: Vector3::new(0.0, 0.0, -9.8),
            joint_type: Vec::with_capacity(num_bodies),
            parent,
            q_index: vec![-1isize; num_bodies],
            parent_pos_parent_body_ref: Vec::with_capacity(num_bodies),
            body_t_parent_ref: Vec::with_capacity(num_bodies),
            jac_jr: Vec::with_capacity(num_bodies),
            jac_jt: Vec::with_capacity(num_bodies),
            parent_jac_jt: vec![Vector3::ZERO; num_bodies],
            mass: Vec::with_capacity(num_bodies),
            mass_com: Vec::with_capacity(num_bodies),
            inertia: Vec::with_capacity(num_bodies),
            child_indices: vec![Vec::new(); num_bodies],
            revolute_list: Vec::new(),
            prismatic_list: Vec::new(),
            floating_list: Vec::new(),
            spherical_list: Vec::new(),
            body_t_parent: vec![Matrix3::ZERO; num_bodies],
            body_t_world: vec![Matrix3::ZERO; num_bodies],
            parent_pos_parent_body: vec![Vector3::ZERO; num_bodies],
            body_ang_vel_rel: vec![Vector3::ZERO; num_bodies],
            parent_vel_rel: vec![Vector3::ZERO; num_bodies],
            body_ang_acc_rel: vec![Vector3::ZERO; num_bodies],
            parent_acc_rel: vec![Vector3::ZERO; num_bodies],
            body_pos: vec![Vector3::ZERO; num_bodies],
            body_vel: vec![Vector3::ZERO; num_bodies],
            body_acc: vec![Vector3::ZERO; num_bodies],
            body_ang_vel: vec![Vector3::ZERO; num_bodies],
            body_ang_acc: vec![Vector3::ZERO; num_bodies],
            eom_lhs_rotational: vec![Vector3::ZERO; num_bodies],
            eom_lhs_translational: vec![Vector3::ZERO; num_bodies],
            force_at_joint: vec![Vector3::ZERO; num_bodies],
            moment_at_joint: vec![Vector3::ZERO; num_bodies],
            body_force_user: vec![Vector3::ZERO; num_bodies],
            body_moment_user: vec![Vector3::ZERO; num_bodies],
            accept_invalid_mass_parameters: self.accept_invalid_mass_parameters,
            mass_parameters_are_valid: self.mass_parameters_are_valid,
            subtree_mass: vec![0.0; num_bodies],
            body_subtree_mass_com: vec![Vector3::ZERO; num_bodies],
            body_subtree_i_body: vec![Matrix3::ZERO; num_bodies],
            body_jac_r: vec![vec![0.0; 3 * num_dofs]; num_bodies],
            body_jac_t: vec![vec![0.0; 3 * num_dofs]; num_bodies],
            body_dot_jac_r_u: vec![Vector3::ZERO; num_bodies],
            body_dot_jac_t_u: vec![Vector3::ZERO; num_bodies],
            mat3x_scratch: vec![0.0; 3 * num_dofs],
        };

        for (joint, inertia) in self.joints.iter().zip(&self.inertias) {
            tree.joint_type.push(joint.joint_type);
            tree.parent_pos_parent_body_ref
                .push(joint.parent_pos_parent_child_ref);
            tree.body_t_parent_ref.push(joint.child_t_parent_ref);
            tree.mass.push(inertia.mass);
            tree.mass_com.push(inertia.mass * inertia.body_pos_body_com);
            tree.inertia.push(inertia.body_i_body);
            // Joint jacobians: always 3-vectors here, per the upstream
            // "avoid variable sized matrices" note. Fixed/floating/spherical
            // store zeros ("this is not really correct" upstream TODO); the
            // per-dof columns are supplied dynamically by the mass-matrix
            // and jacobian paths (MultiBodyTree.cpp:413-465).
            let (jr, jt) = match joint.joint_type {
                JointType::Revolute => (joint.child_axis_of_motion, Vector3::ZERO),
                JointType::Prismatic => (Vector3::ZERO, joint.child_axis_of_motion),
                JointType::Fixed | JointType::Spherical | JointType::Floating => {
                    (Vector3::ZERO, Vector3::ZERO)
                }
            };
            tree.jac_jr.push(jr);
            tree.jac_jt.push(jt);
        }

        // 4. dof indices and per-joint-type index arrays
        // (`MultiBodyTreeImpl.cpp:128-202`).
        let mut q_index_cursor: isize = 0;
        for index in 0..num_bodies {
            match tree.joint_type[index] {
                JointType::Revolute => {
                    tree.revolute_list.push(index);
                    tree.q_index[index] = q_index_cursor;
                    q_index_cursor += 1;
                }
                JointType::Prismatic => {
                    tree.prismatic_list.push(index);
                    tree.q_index[index] = q_index_cursor;
                    q_index_cursor += 1;
                }
                JointType::Fixed => {}
                JointType::Floating => {
                    tree.floating_list.push(index);
                    tree.q_index[index] = q_index_cursor;
                    q_index_cursor += 6;
                }
                JointType::Spherical => {
                    tree.spherical_list.push(index);
                    tree.q_index[index] = q_index_cursor;
                    q_index_cursor += 3;
                }
            }
        }
        debug_assert_eq!(q_index_cursor, num_dofs as isize);

        // Child indices with the upstream parent gate: `0 <= parent <
        // num_bodies - 1` for every non-root body; parent −1 on a non-root
        // body is reported as the "multiple roots" case there.
        for child in 1..num_bodies {
            let parent_index = tree.parent[child];
            if parent_index >= 0 && parent_index < num_bodies as isize - 1 {
                tree.child_indices[parent_index as usize].push(child);
            } else {
                return Err(BuildError::BadParentIndex {
                    child,
                    parent: parent_index,
                });
            }
        }

        // 5. pre-computations (`MultiBodyTreeImpl.cpp:204-276`).
        tree.calculate_static_data();

        // 6. user forces start zeroed.
        tree.clear_all_user_forces_and_moments();

        Ok(tree)
    }
}

#[cfg(test)]
mod tests {
    use super::{AddBodyError, JointType, TreeBuilder};
    use crate::{BuildError, MultiBodyTree};
    use bullet_math::{Matrix3, Real, Vector3};

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

    fn add_fixed_root(b: &mut TreeBuilder) {
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
    }

    #[test]
    fn two_link_arm_finalizes_with_index_sets() {
        let mut b = TreeBuilder::new();
        add_fixed_root(&mut b);
        b.add_body(
            1,
            0,
            JointType::Revolute,
            Vector3::new(1.0, 0.0, 0.0),
            identity(),
            Vector3::Z,
            1.0,
            Vector3::ZERO,
            inertia(2.0),
        )
        .unwrap();
        let tree = b.finalize().unwrap();
        assert_eq!(tree.num_bodies(), 2);
        assert_eq!(tree.num_dofs(), 1);
        assert_eq!(tree.parent_index(1), Some(0));
        assert_eq!(tree.parent_index(0), Some(-1));
        assert_eq!(tree.joint_type(1), Some(JointType::Revolute));
        assert_eq!(tree.dof_offset(0), None);
        assert_eq!(tree.dof_offset(1), Some(0));
        assert!(tree.mass_parameters_are_valid());
    }

    #[test]
    fn dof_counts_per_joint_type() {
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
        b.add_body(
            1,
            0,
            JointType::Spherical,
            Vector3::ZERO,
            identity(),
            Vector3::ZERO,
            1.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        b.add_body(
            2,
            1,
            JointType::Fixed,
            Vector3::ZERO,
            identity(),
            Vector3::ZERO,
            1.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        let tree = b.finalize().unwrap();
        assert_eq!(tree.num_dofs(), 9);
        assert_eq!(tree.dof_offset(0), Some(0));
        assert_eq!(tree.dof_offset(1), Some(6));
        assert_eq!(tree.dof_offset(2), None);
    }

    #[test]
    fn multiple_roots_rejected() {
        let mut b = TreeBuilder::new();
        add_fixed_root(&mut b);
        let err = b
            .add_body(
                1,
                -1,
                JointType::Fixed,
                Vector3::ZERO,
                identity(),
                Vector3::ZERO,
                1.0,
                Vector3::ZERO,
                inertia(1.0),
            )
            .unwrap_err();
        assert_eq!(err, AddBodyError::MultipleRoots { existing: 0 });
    }

    #[test]
    fn index_mismatch_rejected() {
        let mut b = TreeBuilder::new();
        add_fixed_root(&mut b);
        let err = b
            .add_body(
                5,
                0,
                JointType::Fixed,
                Vector3::ZERO,
                identity(),
                Vector3::ZERO,
                1.0,
                Vector3::ZERO,
                inertia(1.0),
            )
            .unwrap_err();
        assert_eq!(err, AddBodyError::IndexMismatch { expected: 1 });
    }

    #[test]
    fn finalize_succeeds_on_a_well_formed_chain() {
        let mut b = TreeBuilder::new();
        add_fixed_root(&mut b);
        b.add_body(
            1,
            0,
            JointType::Revolute,
            Vector3::ZERO,
            identity(),
            Vector3::Z,
            1.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        b.add_body(
            2,
            1,
            JointType::Fixed,
            Vector3::ZERO,
            identity(),
            Vector3::ZERO,
            1.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        let result: Result<MultiBodyTree, BuildError> = b.finalize();
        assert!(result.is_ok());
    }

    #[test]
    fn axis_normalized_and_too_short_rejected() {
        let mut b = TreeBuilder::new();
        add_fixed_root(&mut b);
        b.add_body(
            1,
            0,
            JointType::Revolute,
            Vector3::ZERO,
            identity(),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
            Vector3::ZERO,
            inertia(1.0),
        )
        .unwrap();
        let tree = b.finalize().unwrap();
        assert_eq!(tree.jac_jr[1], Vector3::new(0.0, 1.0, 0.0));

        let mut b2 = TreeBuilder::new();
        add_fixed_root(&mut b2);
        assert_eq!(
            b2.add_body(
                1,
                0,
                JointType::Prismatic,
                Vector3::ZERO,
                identity(),
                Vector3::ZERO,
                1.0,
                Vector3::ZERO,
                inertia(1.0),
            ),
            Err(AddBodyError::AxisTooShort)
        );
    }

    #[test]
    fn mass_and_inertia_gates() {
        let mut b = TreeBuilder::new();
        add_fixed_root(&mut b);
        assert_eq!(
            b.add_body(
                1,
                0,
                JointType::Revolute,
                Vector3::ZERO,
                identity(),
                Vector3::Z,
                -1.0,
                Vector3::ZERO,
                inertia(1.0),
            ),
            Err(AddBodyError::NegativeMass)
        );
        // Triangle inequality violated.
        let bad_inertia = Matrix3::from_cols(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 3.0),
        );
        assert_eq!(
            b.add_body(
                1,
                0,
                JointType::Revolute,
                Vector3::ZERO,
                identity(),
                Vector3::Z,
                1.0,
                Vector3::ZERO,
                bad_inertia,
            ),
            Err(AddBodyError::InvalidInertiaMatrix)
        );
        // Accepted under the flag, and the flag records the taint.
        b.accept_invalid_mass_parameters(true);
        b.add_body(
            1,
            0,
            JointType::Revolute,
            Vector3::ZERO,
            identity(),
            Vector3::Z,
            -1.0,
            Vector3::ZERO,
            bad_inertia,
        )
        .unwrap();
        assert!(!b.mass_parameters_are_valid());
        let tree = b.finalize().unwrap();
        assert!(!tree.mass_parameters_are_valid());
        assert!(tree.accept_invalid_mass_parameters());
    }

    #[test]
    fn invalid_transform_rejected_even_with_accept_flag() {
        let mut b = TreeBuilder::new();
        add_fixed_root(&mut b);
        b.accept_invalid_mass_parameters(true);
        let scaled = Matrix3::from_cols(
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(
            b.add_body(
                1,
                0,
                JointType::Revolute,
                Vector3::ZERO,
                scaled,
                Vector3::Z,
                1.0,
                Vector3::ZERO,
                inertia(1.0),
            ),
            Err(AddBodyError::InvalidTransformMatrix)
        );
    }
}
