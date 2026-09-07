//! The articulated rigid body — port of
//! `BulletDynamics/Featherstone/btMultiBody.h` / `btMultiBody.cpp`.
//!
//! A [`MultiBody`] is a tree of links connected by revolute, prismatic,
//! spherical, planar or fixed joints, with an optional floating base. The
//! dynamics use Featherstone's articulated-body algorithm (ABA) over spatial
//! algebra: forward kinematics, velocity/force/acceleration propagation, a
//! 6x6 spatial-inertia solve for the base (`solve_imatrix`) and per-link
//! joint-space solves for the accelerations. Constraint impulses are applied
//! through [`MultiBody::calc_acceleration_deltas`], which reuses the ABA
//! factorization cached during the previous
//! [`MultiBody::compute_accelerations_aba`] call.
//!
//! Deviation from Bullet: link colliders, kinematic links (per-link
//! static/kinematic flags), the RK4 integrator, global velocities and the
//! interpolation (predicted) pose are not implemented — all links are always
//! dynamic, the integrator is the explicit semi-implicit Euler used by
//! Bullet's default path, and the base is fixed only through
//! [`MultiBody::new`]'s `fixed_base` flag.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::spatial::{
    symmetric_spatial_outer_product_force, SpatialForceVector, SpatialMotionVector,
    SpatialTransformationMatrix, SymmetricSpatialDyad,
};
use bullet_math::{math, Matrix3, Quaternion, Real, Transform, Vector3};
use bullet_math::compat::{matrix3_from_diagonal, quat_from_axis_angle, quat_from_mat3, transform_from_rotation_translation};

use super::link::{JointType, MultiBodyLink};
use bullet_collision::shapes::CollisionShape;

const INITIAL_SLEEP_EPSILON: Real = 0.05; // squared velocity (m² s⁻²)
const INITIAL_SLEEP_TIMEOUT: Real = 2.0; // seconds
/// Bullet's `ANGULAR_MOTION_THRESHOLD` (in `btTransformUtil.h`).
const ANGULAR_MOTION_THRESHOLD: Real = 0.5 * bullet_math::FRAC_PI_2;

/// Scratch buffers for the articulated-body algorithm, owned by the world to
/// avoid reallocation every step (Bullet passes scratch arrays from
/// `btMultiBodyDynamicsWorld`).
#[derive(Default, Clone)]
pub struct MultiBodyScratch {
    /// `Y` joint-space bias terms followed by the `qdd` output
    /// (`2·dof_count + 7` entries in Bullet).
    pub r: Vec<Real>,
    /// Spatial link velocities `v̂_i` (one per link + base).
    pub spat_vel: Vec<SpatialMotionVector>,
    /// Accumulated bias forces `ẑᵢᴬ` (one per link + base).
    pub zero_acc: Vec<SpatialForceVector>,
    /// Coriolis accelerations `ĉ_i` (one per link, NOT defined for the base).
    pub coriolis: Vec<SpatialMotionVector>,
    /// Spatial link accelerations `â_i` (one per link + base).
    pub spat_acc: Vec<SpatialMotionVector>,
    /// Rotation from the world frame to each link frame.
    pub rot_from_world: Vec<Matrix3>,
    /// Articulated-body inertias `Îᵢᴬ` (one per link + base).
    pub spat_inertia: Vec<SymmetricSpatialDyad>,
    /// Per-link ABA D matrices (`dof_count^2` entries, cleared per link).
    pub aba_d: Vec<Real>,
    /// Joint-space force vector reused by `calc_acceleration_deltas`
    /// (`dof_count` entries).
    pub dof_force: Vec<Real>,
    /// Chain walked by `fill_constraint_jacobian` (link indices, base
    /// excluded).
    pub jac_chain: Vec<usize>,
    /// Per-link local positions/normals reused by `fill_constraint_jacobian`
    /// (`num_links + 1` entries each).
    pub jac_p_local: Vec<Vector3>,
    pub jac_n_lin: Vec<Vector3>,
    pub jac_n_ang: Vec<Vector3>,
    /// Unit-impulse responses reused by `fill_constraint_jacobian`
    /// (`dof_count` entries).
    pub jac_results: Vec<Real>,
}

impl MultiBodyScratch {
    /// Ensure the buffers can hold `num_links + 1` links and `dof_count`
    /// DOFs (Bullet resizes them inside the ABA each call).
    fn ensure(&mut self, num_links: usize, dof_count: usize) {
        let n = num_links + 1;
        if self.r.len() < 2 * dof_count + 7 {
            self.r.resize(2 * dof_count + 7, 0.0);
        }
        if self.spat_vel.len() < n {
            self.spat_vel.resize(n, SpatialMotionVector::zero());
        }
        if self.zero_acc.len() < n {
            self.zero_acc.resize(n, SpatialForceVector::zero());
        }
        if self.coriolis.len() < num_links {
            self.coriolis.resize(num_links, SpatialMotionVector::zero());
        }
        if self.spat_acc.len() < n {
            self.spat_acc.resize(n, SpatialMotionVector::zero());
        }
        if self.rot_from_world.len() < n {
            self.rot_from_world.resize(n, Matrix3::IDENTITY);
        }
        if self.spat_inertia.len() < n {
            self.spat_inertia.resize(n, SymmetricSpatialDyad::zero());
        }
        if self.aba_d.len() < dof_count * dof_count {
            self.aba_d.resize(dof_count * dof_count, 0.0);
        }
        if self.dof_force.len() < dof_count {
            self.dof_force.resize(dof_count, 0.0);
        }
        if self.jac_chain.len() < num_links {
            self.jac_chain.resize(num_links, 0);
        }
        if self.jac_p_local.len() < n {
            self.jac_p_local.resize(n, Vector3::ZERO);
        }
        if self.jac_n_lin.len() < n {
            self.jac_n_lin.resize(n, Vector3::ZERO);
        }
        if self.jac_n_ang.len() < n {
            self.jac_n_ang.resize(n, Vector3::ZERO);
        }
        if self.jac_results.len() < dof_count {
            self.jac_results.resize(dof_count, 0.0);
        }
    }
}

/// A collider attached to a multibody's base or a link — the value form of
/// Bullet's `btMultiBodyLinkCollider`: the collision shape plus surface
/// parameters. The link/base association lives in the slot the collider is
/// stored in; the world registers each collider as a broad-phase proxy.
#[derive(Debug, Clone)]
pub struct LinkCollider {
    pub shape: CollisionShape,
    /// Friction contribution (combined per contact like rigid bodies).
    pub friction: Real,
    /// Restitution contribution.
    pub restitution: Real,
    /// Collision filter group (Bullet's `m_collisionFilterGroup` on the
    /// `btMultiBodyLinkCollider` proxy; default
    /// [`filter_groups::DEFAULT`](crate::body::filter_groups::DEFAULT)).
    pub collision_group: u16,
    /// Collision filter mask (default
    /// [`filter_groups::ALL`](crate::body::filter_groups::ALL)).
    pub collision_mask: u16,
    /// Whether pairs this collider touches emit collision events (the
    /// per-side opt-in, like [`RigidBody::collision_events`]).
    pub collision_events: bool,
}

/// A Featherstone articulated rigid body.
#[derive(Clone)]
pub struct MultiBody {
    /// Position of the base COM, world frame.
    pub base_pos: Vector3,
    /// Rotation mapping world vectors into the base frame (Bullet's
    /// `m_baseQuat` — note it is the *inverse* of the base's world
    /// orientation).
    pub base_quat: Quaternion,
    base_mass: Real,
    /// Diagonal base inertia, base frame.
    base_inertia: Vector3,
    /// External force/torque on the base, world frame (cleared each step).
    pub base_force: Vector3,
    pub base_torque: Vector3,
    /// Constraint (solver) force/torque on the base, world frame.
    base_constraint_force: Vector3,
    base_constraint_torque: Vector3,
    links: Vec<MultiBodyLink>,
    /// Generalized velocities: `[base_omega (world), base_vel (world),
    /// joint vels...]` (Bullet's `m_realBuf[0..6+dof_count)`).
    velocities: Vec<Real>,
    /// Accumulated constraint delta velocities, applied by
    /// `process_delta_vee_2` (Bullet's `m_deltaV`).
    delta_v: Vec<Real>,
    /// Split-impulse delta velocities (Bullet's `m_splitV`).
    split_v: Vec<Real>,
    /// Per-link inverse joint-space matrices `D⁻¹`, scattered at
    /// `dof_offset²` (Bullet's `m_realBuf[6+dof_count..]`).
    inv_d: Vec<Real>,
    /// The `h` spatial force vectors, one per DOF (Bullet's `m_vectorBuf`).
    h: Vec<SpatialForceVector>,
    /// Cached rotation matrices from each parent frame to each child frame
    /// (index 0 = base; Bullet's `m_matrixBuf`).
    rot_from_parent: Vec<Matrix3>,
    /// Cached composite spatial inertia of the base (block layout
    /// `[topLeft, topRight, bottomLeft, bottomRight]`), set by the ABA when
    /// `num_links > 0` and the base is dynamic.
    cached_inertia: [Matrix3; 4],
    cached_inertia_valid: bool,
    fixed_base: bool,
    // Sleep parameters.
    awake: bool,
    can_sleep: bool,
    can_wakeup: bool,
    sleep_timer: Real,
    sleep_epsilon: Real,
    sleep_timeout: Real,
    // Dynamics parameters.
    linear_damping: Real,
    angular_damping: Real,
    use_gyro_term: bool,
    max_applied_impulse: Real,
    max_coordinate_velocity: Real,
    has_self_collision: bool,
    /// Report joint reaction forces in the world frame instead of the link
    /// frame (Bullet's `m_jointFeedbackInWorldSpace`, default false).
    joint_feedback_in_world_space: bool,
    /// Report the torque about the joint frame instead of the link COM frame
    /// (Bullet's `m_jointFeedbackInJointFrame`, default false).
    joint_feedback_in_joint_frame: bool,
    /// Set by the last articulated-body pass when any link carries a joint
    /// feedback slot; the world runs the constraint pass when set (Bullet's
    /// `m_internalNeedsJointFeedback`).
    needs_joint_feedback: bool,
    /// Collider on the base (Bullet's base `btMultiBodyLinkCollider`).
    base_collider: Option<LinkCollider>,
    /// Colliders on the links, parallel to `links` (Bullet's per-link
    /// `m_collider`).
    link_colliders: Vec<Option<LinkCollider>>,
    pos_updated: bool,
    dof_count: usize,
    pos_var_count: usize,
    use_rk4: bool,
    /// Scratch index used by the solver to find this body's accumulated
    /// delta-velocity block (Bullet's `m_companionId`); reset each solve.
    pub companion_id: i32,
}

impl MultiBody {
    /// Create a multibody with `n_links` (not counting the base), base mass
    /// `mass`, diagonal base inertia `inertia` (base frame), and a base that
    /// is fixed (`fixed_base = true`) or free.
    pub fn new(n_links: usize, mass: Real, inertia: Vector3, fixed_base: bool, can_sleep: bool) -> Self {
        let mut body = Self {
            base_pos: Vector3::ZERO,
            base_quat: Quaternion::IDENTITY,
            base_mass: mass,
            base_inertia: inertia,
            base_force: Vector3::ZERO,
            base_torque: Vector3::ZERO,
            base_constraint_force: Vector3::ZERO,
            base_constraint_torque: Vector3::ZERO,
            links: Vec::with_capacity(n_links),
            velocities: Vec::new(),
            delta_v: Vec::new(),
            split_v: Vec::new(),
            inv_d: Vec::new(),
            h: Vec::new(),
            rot_from_parent: Vec::new(),
            cached_inertia: [Matrix3::ZERO; 4],
            cached_inertia_valid: false,
            fixed_base,
            awake: true,
            can_sleep,
            can_wakeup: true,
            sleep_timer: 0.0,
            sleep_epsilon: INITIAL_SLEEP_EPSILON,
            sleep_timeout: INITIAL_SLEEP_TIMEOUT,
            linear_damping: 0.04,
            angular_damping: 0.04,
            use_gyro_term: true,
            max_applied_impulse: 1000.0,
            max_coordinate_velocity: 100.0,
            has_self_collision: true,
            joint_feedback_in_world_space: false,
            joint_feedback_in_joint_frame: false,
            needs_joint_feedback: false,
            base_collider: None,
            link_colliders: vec![None; n_links],
            pos_updated: false,
            dof_count: 0,
            pos_var_count: 0,
            use_rk4: false,
            companion_id: -1,
        };
        for _ in 0..n_links {
            body.links.push(MultiBodyLink::new());
        }
        body.rot_from_parent.resize(n_links + 1, Matrix3::IDENTITY);
        body.clear_constraint_forces();
        body.clear_forces_and_torques();
        body
    }

    // ------------------------------------------------------------------
    // Setup
    // ------------------------------------------------------------------

    fn update_links_dof_offsets(&mut self) {
        let mut dof_offset = 0;
        let mut cfg_offset = 0;
        for link in &mut self.links {
            link.dof_offset = dof_offset;
            link.cfg_offset = cfg_offset;
            dof_offset += link.dof_count;
            cfg_offset += link.pos_var_count;
        }
    }

    /// Configure link `i` as a fixed (weld) joint to `parent`.
    #[allow(clippy::too_many_arguments)]
    pub fn setup_fixed(
        &mut self,
        i: usize,
        mass: Real,
        inertia: Vector3,
        parent: Option<usize>,
        rot_parent_to_this: Quaternion,
        parent_com_to_this_pivot_offset: Vector3,
        this_pivot_to_this_com_offset: Vector3,
    ) {
        let link = &mut self.links[i];
        link.mass = mass;
        link.inertia_local = inertia;
        link.parent = parent;
        link.set_axis_top(0, Vector3::ZERO);
        link.set_axis_bottom(0, Vector3::ZERO);
        link.zero_rot_parent_to_this = rot_parent_to_this;
        link.d_vector = this_pivot_to_this_com_offset;
        link.e_vector = parent_com_to_this_pivot_offset;
        link.joint_type = JointType::Fixed;
        link.dof_count = 0;
        link.pos_var_count = 0;
        link.cached_rot_parent_to_this = rot_parent_to_this;
        link.update_cache_multidof(None);
        self.update_links_dof_offsets();
    }

    /// Configure link `i` as a prismatic (sliding) joint to `parent` along
    /// `joint_axis` (in the link's frame).
    #[allow(clippy::too_many_arguments)]
    pub fn setup_prismatic(
        &mut self,
        i: usize,
        mass: Real,
        inertia: Vector3,
        parent: Option<usize>,
        rot_parent_to_this: Quaternion,
        joint_axis: Vector3,
        parent_com_to_this_pivot_offset: Vector3,
        this_pivot_to_this_com_offset: Vector3,
    ) {
        self.dof_count += 1;
        self.pos_var_count += 1;
        let link = &mut self.links[i];
        link.mass = mass;
        link.inertia_local = inertia;
        link.parent = parent;
        link.zero_rot_parent_to_this = rot_parent_to_this;
        link.set_axis_top(0, Vector3::ZERO);
        link.set_axis_bottom(0, joint_axis);
        link.e_vector = parent_com_to_this_pivot_offset;
        link.d_vector = this_pivot_to_this_com_offset;
        link.cached_rot_parent_to_this = rot_parent_to_this;
        link.joint_type = JointType::Prismatic;
        link.dof_count = 1;
        link.pos_var_count = 1;
        link.joint_pos[0] = 0.0;
        link.joint_torque[0] = 0.0;
        link.update_cache_multidof(None);
        self.update_links_dof_offsets();
    }

    /// Configure link `i` as a revolute (hinge) joint to `parent` around
    /// `joint_axis` (in the link's frame).
    #[allow(clippy::too_many_arguments)]
    pub fn setup_revolute(
        &mut self,
        i: usize,
        mass: Real,
        inertia: Vector3,
        parent: Option<usize>,
        rot_parent_to_this: Quaternion,
        joint_axis: Vector3,
        parent_com_to_this_pivot_offset: Vector3,
        this_pivot_to_this_com_offset: Vector3,
    ) {
        self.dof_count += 1;
        self.pos_var_count += 1;
        let link = &mut self.links[i];
        link.mass = mass;
        link.inertia_local = inertia;
        link.parent = parent;
        link.zero_rot_parent_to_this = rot_parent_to_this;
        link.set_axis_top(0, joint_axis);
        link.set_axis_bottom(0, joint_axis.cross(this_pivot_to_this_com_offset));
        link.d_vector = this_pivot_to_this_com_offset;
        link.e_vector = parent_com_to_this_pivot_offset;
        link.joint_type = JointType::Revolute;
        link.dof_count = 1;
        link.pos_var_count = 1;
        link.joint_pos[0] = 0.0;
        link.joint_torque[0] = 0.0;
        link.update_cache_multidof(None);
        self.update_links_dof_offsets();
    }

    /// Configure link `i` as a spherical (ball) joint to `parent`.
    #[allow(clippy::too_many_arguments)]
    pub fn setup_spherical(
        &mut self,
        i: usize,
        mass: Real,
        inertia: Vector3,
        parent: Option<usize>,
        rot_parent_to_this: Quaternion,
        parent_com_to_this_pivot_offset: Vector3,
        this_pivot_to_this_com_offset: Vector3,
    ) {
        self.dof_count += 3;
        self.pos_var_count += 4;
        let link = &mut self.links[i];
        link.mass = mass;
        link.inertia_local = inertia;
        link.parent = parent;
        link.zero_rot_parent_to_this = rot_parent_to_this;
        link.d_vector = this_pivot_to_this_com_offset;
        link.e_vector = parent_com_to_this_pivot_offset;
        link.joint_type = JointType::Spherical;
        link.dof_count = 3;
        link.pos_var_count = 4;
        link.set_axis_top(0, Vector3::X);
        link.set_axis_top(1, Vector3::Y);
        link.set_axis_top(2, Vector3::Z);
        link.set_axis_bottom(0, link.axis_top(0).cross(this_pivot_to_this_com_offset));
        link.set_axis_bottom(1, link.axis_top(1).cross(this_pivot_to_this_com_offset));
        link.set_axis_bottom(2, link.axis_top(2).cross(this_pivot_to_this_com_offset));
        link.joint_pos[0] = 0.0;
        link.joint_pos[1] = 0.0;
        link.joint_pos[2] = 0.0;
        link.joint_pos[3] = 1.0;
        link.joint_torque[0] = 0.0;
        link.joint_torque[1] = 0.0;
        link.joint_torque[2] = 0.0;
        link.update_cache_multidof(None);
        self.update_links_dof_offsets();
    }

    /// Configure link `i` as a planar joint to `parent`: rotation around
    /// `rotation_axis` plus two orthogonal translations on the plane.
    #[allow(clippy::too_many_arguments)]
    pub fn setup_planar(
        &mut self,
        i: usize,
        mass: Real,
        inertia: Vector3,
        parent: Option<usize>,
        rot_parent_to_this: Quaternion,
        rotation_axis: Vector3,
        parent_com_to_this_com_offset: Vector3,
    ) {
        self.dof_count += 3;
        self.pos_var_count += 3;
        {
            let link = &mut self.links[i];
            link.mass = mass;
            link.inertia_local = inertia;
            link.parent = parent;
            link.zero_rot_parent_to_this = rot_parent_to_this;
            link.d_vector = Vector3::ZERO;
            link.e_vector = parent_com_to_this_com_offset;

            let mut vec_non_parallel_to_rot_axis = Vector3::X;
            if rotation_axis.normalize().dot(vec_non_parallel_to_rot_axis) > 0.999 {
                vec_non_parallel_to_rot_axis = Vector3::Y;
            }

            link.joint_type = JointType::Planar;
            link.dof_count = 3;
            link.pos_var_count = 3;
            let n = rotation_axis.normalize();
            link.set_axis_top(0, n);
            link.set_axis_top(1, Vector3::ZERO);
            link.set_axis_top(2, Vector3::ZERO);
            link.set_axis_bottom(0, Vector3::ZERO);
            let mut cr = link.axis_top(0).cross(vec_non_parallel_to_rot_axis);
            link.set_axis_bottom(1, cr);
            cr = link.axis_bottom(1).cross(link.axis_top(0));
            link.set_axis_bottom(2, cr);
            link.joint_pos[0] = 0.0;
            link.joint_pos[1] = 0.0;
            link.joint_pos[2] = 0.0;
            link.joint_torque[0] = 0.0;
            link.joint_torque[1] = 0.0;
            link.joint_torque[2] = 0.0;
            link.update_cache_multidof(None);
        }
        self.update_links_dof_offsets();
        // Normalize the translational axes after the first cache update
        // (Bullet does the same; at q = 0 the cache does not depend on them).
        let link = &mut self.links[i];
        link.set_axis_bottom(1, link.axis_bottom(1).normalize());
        link.set_axis_bottom(2, link.axis_bottom(2).normalize());
    }

    /// Allocate the internal buffers once every link is configured (Bullet's
    /// `finalizeMultiDof`).
    pub fn finalize_multidof(&mut self) {
        let n = 6 + self.dof_count;
        self.delta_v = vec![0.0; n];
        self.split_v = vec![0.0; n];
        self.velocities = vec![0.0; n];
        self.inv_d = vec![0.0; self.dof_count * self.dof_count];
        self.h = vec![SpatialForceVector::zero(); self.dof_count];
        self.rot_from_parent.resize(self.links.len() + 1, Matrix3::IDENTITY);
        self.update_links_dof_offsets();
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub fn num_links(&self) -> usize {
        self.links.len()
    }

    pub fn num_dofs(&self) -> usize {
        self.dof_count
    }

    pub fn num_pos_vars(&self) -> usize {
        self.pos_var_count
    }

    pub fn link(&self, i: usize) -> &MultiBodyLink {
        &self.links[i]
    }

    pub fn link_mut(&mut self, i: usize) -> &mut MultiBodyLink {
        &mut self.links[i]
    }

    pub fn links(&self) -> &[MultiBodyLink] {
        &self.links
    }

    pub fn base_mass(&self) -> Real {
        self.base_mass
    }

    pub fn base_inertia(&self) -> Vector3 {
        self.base_inertia
    }

    pub fn base_pos(&self) -> Vector3 {
        self.base_pos
    }

    /// Base linear velocity, world frame.
    pub fn base_vel(&self) -> Vector3 {
        Vector3::new(self.velocities[3], self.velocities[4], self.velocities[5])
    }

    /// Base angular velocity, world frame.
    pub fn base_omega(&self) -> Vector3 {
        Vector3::new(self.velocities[0], self.velocities[1], self.velocities[2])
    }

    /// The world→base rotation quaternion.
    pub fn world_to_base_rot(&self) -> Quaternion {
        self.base_quat
    }

    /// The base's world transform.
    pub fn base_world_transform(&self) -> Transform {
        transform_from_rotation_translation(self.base_quat.inverse(), self.base_pos)
    }

    pub fn set_base_pos(&mut self, pos: Vector3) {
        self.base_pos = pos;
    }

    pub fn set_world_to_base_rot(&mut self, rot: Quaternion) {
        self.base_quat = rot;
    }

    // `Vector3::from` narrows the glam `Vec3A` affine columns to `Vector3`;
    // it is an identity conversion (and clippy-useless) in f64 builds.
    #[allow(clippy::useless_conversion)]
    pub fn set_base_world_transform(&mut self, tr: &Transform) {
        self.base_pos = Vector3::from(tr.translation);
        let m = Matrix3::from_cols(
            Vector3::from(tr.matrix3.col(0)),
            Vector3::from(tr.matrix3.col(1)),
            Vector3::from(tr.matrix3.col(2)),
        );
        self.base_quat = quat_from_mat3(&m).inverse();
    }

    /// The world orientation of link `i`, extracted from its cached world
    /// transform (Bullet's `m_cachedWorldTransform.getRotation()`).
    #[allow(clippy::useless_conversion)] // see set_base_world_transform
    pub fn link_world_orientation(&self, i: usize) -> Quaternion {
        let m3 = self.links[i].cached_world_transform.matrix3;
        let m = Matrix3::from_cols(
            Vector3::from(m3.col(0)),
            Vector3::from(m3.col(1)),
            Vector3::from(m3.col(2)),
        );
        quat_from_mat3(&m)
    }

    pub fn set_base_vel(&mut self, vel: Vector3) {
        self.velocities[3] = vel.x;
        self.velocities[4] = vel.y;
        self.velocities[5] = vel.z;
    }

    pub fn set_base_omega(&mut self, omega: Vector3) {
        self.velocities[0] = omega.x;
        self.velocities[1] = omega.y;
        self.velocities[2] = omega.z;
    }

    /// Joint position variables of link `i` (a slice of `joint_pos`).
    pub fn joint_pos_multidof(&self, i: usize) -> &[Real] {
        &self.links[i].joint_pos[..self.links[i].pos_var_count]
    }

    /// Joint velocities of link `i` (a slice into the velocity vector).
    pub fn joint_vel_multidof(&self, i: usize) -> &[Real] {
        let start = 6 + self.links[i].dof_offset;
        &self.velocities[start..start + self.links[i].dof_count]
    }

    pub fn set_joint_pos_multidof(&mut self, i: usize, q: &[Real]) {
        let link = &mut self.links[i];
        for (pos, &v) in q.iter().enumerate().take(link.pos_var_count) {
            link.joint_pos[pos] = v;
        }
        link.update_cache_multidof(None);
    }

    pub fn set_joint_vel_multidof(&mut self, i: usize, qdot: &[Real]) {
        let start = 6 + self.links[i].dof_offset;
        for (k, &v) in qdot.iter().enumerate().take(self.links[i].dof_count) {
            self.velocities[start + k] = v;
        }
    }

    /// The full generalized velocity vector `[ω, v, q̇...]`.
    pub fn velocity_vector(&self) -> &[Real] {
        &self.velocities
    }

    pub fn velocity_vector_mut(&mut self) -> &mut [Real] {
        &mut self.velocities
    }

    pub fn link_mass(&self, i: usize) -> Real {
        self.links[i].mass
    }

    pub fn link_inertia(&self, i: usize) -> Vector3 {
        self.links[i].inertia_local
    }

    /// Vector from COM(parent(i)) to COM(i), in frame i's coordinates.
    pub fn r_vector(&self, i: usize) -> Vector3 {
        self.links[i].cached_r_vector
    }

    /// Rotates vectors in frame parent(i) to vectors in frame i.
    pub fn parent_to_local_rot(&self, i: usize) -> Quaternion {
        self.links[i].cached_rot_parent_to_this
    }

    // ------------------------------------------------------------------
    // Frame transforms
    // ------------------------------------------------------------------

    /// Transform a point given in link `i`'s local frame to the world frame.
    pub fn local_pos_to_world(&self, i: i32, local_pos: Vector3) -> Vector3 {
        let mut result = local_pos;
        let mut idx = i;
        while idx != -1 {
            let k = idx as usize;
            result += self.r_vector(k);
            result = self.parent_to_local_rot(k).inverse() * result;
            idx = self.links[k].parent.map_or(-1, |p| p as i32);
        }
        self.base_quat.inverse() * result + self.base_pos
    }

    /// Transform a point given in the world frame to link `i`'s local frame.
    pub fn world_pos_to_local(&self, i: i32, world_pos: Vector3) -> Vector3 {
        if i == -1 {
            // world to base
            return self.base_quat * (world_pos - self.base_pos);
        }
        let k = i as usize;
        // find position in the parent frame, then transform to this frame
        let parent = self.links[k].parent.map_or(-1, |p| p as i32);
        self.parent_to_local_rot(k) * self.world_pos_to_local(parent, world_pos) - self.r_vector(k)
    }

    /// Transform a direction given in link `i`'s local frame to the world
    /// frame.
    pub fn local_dir_to_world(&self, i: i32, local_dir: Vector3) -> Vector3 {
        let mut result = local_dir;
        let mut idx = i;
        while idx != -1 {
            let k = idx as usize;
            result = self.parent_to_local_rot(k).inverse() * result;
            idx = self.links[k].parent.map_or(-1, |p| p as i32);
        }
        self.base_quat.inverse() * result
    }

    /// Transform a direction given in the world frame to link `i`'s local
    /// frame.
    pub fn world_dir_to_local(&self, i: i32, world_dir: Vector3) -> Vector3 {
        if i == -1 {
            return self.base_quat * world_dir;
        }
        let k = i as usize;
        let parent = self.links[k].parent.map_or(-1, |p| p as i32);
        self.parent_to_local_rot(k) * self.world_dir_to_local(parent, world_dir)
    }

    /// Transform a frame (rotation matrix) given in link `i`'s local frame
    /// to the world frame.
    pub fn local_frame_to_world(&self, i: i32, local_frame: &Matrix3) -> Matrix3 {
        let c0 = self.local_dir_to_world(i, local_frame.col(0));
        let c1 = self.local_dir_to_world(i, local_frame.col(1));
        let c2 = self.local_dir_to_world(i, local_frame.col(2));
        Matrix3::from_cols(c0, c1, c2)
    }

    /// Compute each link's (and the base's) velocity in its own local frame
    /// (Bullet's `compTreeLinkVelocities`): `omega[0]`/`vel[0]` are the base's
    /// local-frame velocities, `omega[i+1]`/`vel[i+1]` the link `i`'s.
    pub fn comp_tree_link_velocities(&self, omega: &mut [Vector3], vel: &mut [Vector3]) {
        omega[0] = self.base_quat * self.base_omega();
        vel[0] = self.base_quat * self.base_vel();
        for i in 0..self.num_links() {
            let link = &self.links[i];
            let parent = link.parent;
            let (o, v) = bullet_math::spatial_transform(
                &Matrix3::from_quat(link.cached_rot_parent_to_this),
                link.cached_r_vector,
                omega[parent.map_or(0, |p| p + 1)],
                vel[parent.map_or(0, |p| p + 1)],
            );
            omega[i + 1] = o;
            vel[i + 1] = v;
            let joint_vel = self.joint_vel_multidof(i);
            for (dof, &qv) in joint_vel.iter().enumerate() {
                omega[i + 1] += qv * link.axis_top(dof);
                vel[i + 1] += qv * link.axis_bottom(dof);
            }
        }
    }

    // ------------------------------------------------------------------
    // Forces
    // ------------------------------------------------------------------

    pub fn clear_constraint_forces(&mut self) {
        self.base_constraint_force = Vector3::ZERO;
        self.base_constraint_torque = Vector3::ZERO;
        for link in &mut self.links {
            link.applied_constraint_force = Vector3::ZERO;
            link.applied_constraint_torque = Vector3::ZERO;
        }
    }

    pub fn clear_forces_and_torques(&mut self) {
        self.base_force = Vector3::ZERO;
        self.base_torque = Vector3::ZERO;
        for link in &mut self.links {
            link.applied_force = Vector3::ZERO;
            link.applied_torque = Vector3::ZERO;
            link.joint_torque = [0.0; 6];
        }
    }

    pub fn clear_velocities(&mut self) {
        self.velocities.fill(0.0);
    }

    pub fn add_base_force(&mut self, f: Vector3) {
        self.base_force += f;
    }

    pub fn add_base_torque(&mut self, t: Vector3) {
        self.base_torque += t;
    }

    pub fn add_link_force(&mut self, i: usize, f: Vector3) {
        self.links[i].applied_force += f;
    }

    pub fn add_link_torque(&mut self, i: usize, t: Vector3) {
        self.links[i].applied_torque += t;
    }

    pub fn add_base_constraint_force(&mut self, f: Vector3) {
        self.base_constraint_force += f;
    }

    pub fn add_base_constraint_torque(&mut self, t: Vector3) {
        self.base_constraint_torque += t;
    }

    pub fn add_link_constraint_force(&mut self, i: usize, f: Vector3) {
        self.links[i].applied_constraint_force += f;
    }

    pub fn add_link_constraint_torque(&mut self, i: usize, t: Vector3) {
        self.links[i].applied_constraint_torque += t;
    }

    pub fn add_joint_torque_multidof(&mut self, i: usize, dof: usize, q: Real) {
        self.links[i].joint_torque[dof] += q;
    }

    pub fn add_joint_torque(&mut self, i: usize, q: Real) {
        self.links[i].joint_torque[0] += q;
    }

    pub fn joint_torque_multidof(&self, i: usize) -> &[Real] {
        &self.links[i].joint_torque[..self.links[i].dof_count]
    }

    pub fn base_force(&self) -> Vector3 {
        self.base_force
    }

    pub fn base_torque(&self) -> Vector3 {
        self.base_torque
    }

    // ------------------------------------------------------------------
    // Dynamics
    // ------------------------------------------------------------------

    /// Bullet's `safeNorm`: `max(|x|, |y|, |z|)` — a cheap length estimate
    /// used by the damping terms.
    fn safe_norm(v: Vector3) -> Real {
        v.x.abs().max(v.y.abs()).max(v.z.abs())
    }

    /// Apply `delta_vee * multiplier` to the generalized velocities,
    /// clamping every component to `±max_coordinate_velocity` (Bullet's
    /// `applyDeltaVeeMultiDof`).
    pub fn apply_delta_vee(&mut self, delta_vee: &[Real], multiplier: Real) {
        for (v, &d) in self.velocities.iter_mut().zip(delta_vee) {
            *v += d * multiplier;
            *v = (*v).clamp(-self.max_coordinate_velocity, self.max_coordinate_velocity);
        }
    }

    /// Accumulate into `delta_v` (Bullet's `applyDeltaVeeMultiDof2`), used
    /// for split impulses.
    pub fn apply_delta_vee_2(&mut self, delta_vee: &[Real], multiplier: Real) {
        for (v, &d) in self.delta_v.iter_mut().zip(delta_vee) {
            *v += d * multiplier;
        }
    }

    /// `addSplitV`: apply the accumulated split delta to the velocities.
    pub fn add_split_v(&mut self) {
        for i in 0..self.velocities.len() {
            let d = self.split_v[i];
            self.apply_delta_vee_component(i, d, 1.0);
        }
    }

    /// `substractSplitV`: remove the split delta and clear it.
    pub fn subtract_split_v(&mut self) {
        for i in 0..self.velocities.len() {
            let d = self.split_v[i];
            self.apply_delta_vee_component(i, d, -1.0);
        }
        self.split_v.fill(0.0);
    }

    pub fn apply_delta_split_vee(&mut self, delta_vee: &[Real], multiplier: Real) {
        for (v, &d) in self.split_v.iter_mut().zip(delta_vee) {
            *v += d * multiplier;
        }
    }

    /// `processDeltaVeeMultiDof2`: apply the accumulated `delta_v` and clear
    /// it.
    pub fn process_delta_vee_2(&mut self) {
        for i in 0..self.velocities.len() {
            let d = self.delta_v[i];
            self.apply_delta_vee_component(i, d, 1.0);
        }
        self.delta_v.fill(0.0);
    }

    /// One component of `applyDeltaVeeMultiDof` (clamped); split out so the
    /// split-impulse helpers can mix reads of one field with writes of
    /// another without cloning.
    fn apply_delta_vee_component(&mut self, i: usize, d: Real, multiplier: Real) {
        let v = &mut self.velocities[i];
        *v += d * multiplier;
        *v = (*v).clamp(-self.max_coordinate_velocity, self.max_coordinate_velocity);
    }

    fn mul_matrix(p_a: &[Real], p_b: &[Real], rows_a: usize, cols_a: usize, rows_b: usize, cols_b: usize, p_c: &mut [Real]) {
        for row in 0..rows_a {
            for col in 0..cols_b {
                let mut acc = 0.0;
                for inner in 0..rows_b {
                    acc += p_a[row * cols_a + inner] * p_b[col + inner * cols_b];
                }
                p_c[row * cols_b + col] = acc;
            }
        }
    }

    /// Featherstone's articulated-body algorithm: compute the generalized
    /// accelerations and (when `is_constraint_pass` is false and `dt > 0`)
    /// integrate the velocities by `dt`. Also caches the factorization
    /// (`rot_from_parent`, `h`, `invD`, composite base inertia) consumed by
    /// [`MultiBody::calc_acceleration_deltas`]. Port of
    /// `computeAccelerationsArticulatedBodyAlgorithmMultiDof`.
    ///
    /// The `qdd` output (6 + dof_count entries, starting at
    /// `scratch.r[dof_count]`) holds the base angular/linear accelerations in
    /// the world frame followed by the joint accelerations.
    pub fn compute_accelerations_aba(
        &mut self,
        dt: Real,
        scratch: &mut MultiBodyScratch,
        is_constraint_pass: bool,
    ) {
        let num_links = self.num_links();
        let k1_lin = self.linear_damping;
        let k2_lin = self.linear_damping;
        let k1_ang = self.angular_damping;
        let k2_ang = self.angular_damping;

        self.needs_joint_feedback = false;

        scratch.ensure(num_links, self.dof_count);

        // `output` holds the qdd results; `Y` the joint-space bias terms.
        let (y, output) = scratch.r.split_at_mut(self.dof_count);
        let y = &mut y[..self.dof_count];
        let output = &mut output[..6 + self.dof_count];

        let base_omega = self.base_omega();
        let base_vel = self.base_vel();

        // Cached rotations from each parent frame to each child frame.
        self.rot_from_parent[0] = Matrix3::from_quat(self.base_quat);

        // Spatial velocity of the base in base-local coordinates.
        scratch.spat_vel[0] = SpatialMotionVector::new(
            self.rot_from_parent[0] * base_omega,
            self.rot_from_parent[0] * base_vel,
        );

        if self.fixed_base {
            scratch.zero_acc[0] = SpatialForceVector::zero();
        } else {
            // External forces, negated and rotated into the base frame.
            let base_force = if is_constraint_pass {
                self.base_constraint_force
            } else {
                self.base_force
            };
            let base_torque = if is_constraint_pass {
                self.base_constraint_torque
            } else {
                self.base_torque
            };
            scratch.zero_acc[0] = SpatialForceVector::new(
                -(self.rot_from_parent[0] * base_force),
                -(self.rot_from_parent[0] * base_torque),
            );
            // Damping terms.
            let w = scratch.spat_vel[0].top;
            let v = scratch.spat_vel[0].bottom;
            let ang_damp = (k1_ang + k2_ang * Self::safe_norm(w)) * (self.base_inertia * w);
            let lin_damp = (k1_lin + k2_lin * Self::safe_norm(v)) * (self.base_mass * v);
            scratch.zero_acc[0].bottom += ang_damp;
            scratch.zero_acc[0].top += lin_damp;
            // `p += v̂ × I v̂`, done the simple way.
            if self.use_gyro_term {
                scratch.zero_acc[0].bottom += w.cross(self.base_inertia * w);
            }
            scratch.zero_acc[0].top += self.base_mass * w.cross(v);
        }

        // The base spatial inertia (block-diagonal because the base frame
        // origin is the base COM).
        scratch.spat_inertia[0] = SymmetricSpatialDyad::new(
            Matrix3::ZERO,
            matrix3_from_diagonal(Vector3::splat(self.base_mass)),
            matrix3_from_diagonal(self.base_inertia),
        );

        scratch.rot_from_world[0] = self.rot_from_parent[0];

        for i in 0..num_links {
            let parent = self.links[i].parent;
            let parent_idx = parent.map_or(0, |p| p + 1);
            self.rot_from_parent[i + 1] = Matrix3::from_quat(self.links[i].cached_rot_parent_to_this);
            scratch.rot_from_world[i + 1] = self.rot_from_parent[i + 1] * scratch.rot_from_world[parent_idx];

            let from_parent = SpatialTransformationMatrix {
                rot: self.rot_from_parent[i + 1],
                trn: self.links[i].cached_r_vector,
            };
            // v̂_i = X_pi · v̂_p
            scratch.spat_vel[i + 1] = from_parent.transform(&scratch.spat_vel[parent_idx]);

            // v̂_i += q̇ · ŝ_i
            let mut spat_joint_vel = SpatialMotionVector::zero();
            let vel_start = 6 + self.links[i].dof_offset;
            let joint_vel = &self.velocities[vel_start..vel_start + self.links[i].dof_count];
            for (dof, &qv) in joint_vel.iter().enumerate() {
                spat_joint_vel += self.links[i].axes[dof] * qv;
            }
            scratch.spat_vel[i + 1] += spat_joint_vel;

            // ĉ_i = v̂_i × v̂_joint
            scratch.coriolis[i] = scratch.spat_vel[i + 1].cross(&spat_joint_vel);

            // ẑᵢᴬ: external forces + damping + gyro + Coriolis
            {
                let link_force = if is_constraint_pass {
                    self.links[i].applied_constraint_force
                } else {
                    self.links[i].applied_force
                };
                let link_torque = if is_constraint_pass {
                    self.links[i].applied_constraint_torque
                } else {
                    self.links[i].applied_torque
                };
                scratch.zero_acc[i + 1] = SpatialForceVector::new(
                    -(scratch.rot_from_world[i + 1] * link_force),
                    -(scratch.rot_from_world[i + 1] * link_torque),
                );
                let w = scratch.spat_vel[i + 1].top;
                let v = scratch.spat_vel[i + 1].bottom;
                let ang_damp = (k1_ang + k2_ang * Self::safe_norm(w)) * (self.links[i].inertia_local * w);
                let lin_damp = (k1_lin + k2_lin * Self::safe_norm(v)) * (self.links[i].mass * v);
                scratch.zero_acc[i + 1].bottom += ang_damp;
                scratch.zero_acc[i + 1].top += lin_damp;
                if self.use_gyro_term {
                    scratch.zero_acc[i + 1].bottom += w.cross(self.links[i].inertia_local * w);
                }
                if !is_constraint_pass {
                    scratch.zero_acc[i + 1].top += self.links[i].mass * w.cross(v);
                }
            }

            // Îᵢᴬ (block-diagonal at the link COM).
            scratch.spat_inertia[i + 1] = SymmetricSpatialDyad::new(
                Matrix3::ZERO,
                matrix3_from_diagonal(Vector3::splat(self.links[i].mass)),
                matrix3_from_diagonal(self.links[i].inertia_local),
            );
        }

        // 'Downward' loop (part of TreeForwardDynamics in Mirtich).
        for i in (0..num_links).rev() {
            let parent = self.links[i].parent;
            let parent_idx = parent.map_or(0, |p| p + 1);
            let from_parent = SpatialTransformationMatrix {
                rot: self.rot_from_parent[i + 1],
                trn: self.links[i].cached_r_vector,
            };
            let dof_offset = self.links[i].dof_offset;
            let dof_count = self.links[i].dof_count;

            for dof in 0..dof_count {
                let h_dof = scratch.spat_inertia[i + 1].apply(&self.links[i].axes[dof]);
                self.h[dof_offset + dof] = h_dof;
                let joint_torque = if is_constraint_pass {
                    0.0
                } else {
                    self.links[i].joint_torque[dof]
                };
                y[dof_offset + dof] = joint_torque
                    - self.links[i].axes[dof].dot_force(&scratch.zero_acc[i + 1])
                    - scratch.coriolis[i].dot_force(&h_dof);
            }
            // D matrix (dof_count x dof_count) — pooled in the scratch so
            // the ABA does not allocate per link per step.
            let d_len = dof_count * dof_count;
            scratch.aba_d.clear();
            scratch.aba_d.resize(d_len, 0.0);
            let d = &mut scratch.aba_d[..];
            for dof in 0..dof_count {
                for dof2 in 0..dof_count {
                    d[dof * dof_count + dof2] = self.links[i]
                        .axes[dof]
                        .dot_force(&self.h[dof_offset + dof2]);
                }
            }
            // D⁻¹ per joint type.
            let inv_di_start = dof_offset * dof_offset;
            match self.links[i].joint_type {
                JointType::Prismatic | JointType::Revolute => {
                    self.inv_d[inv_di_start] = if d[0] >= Real::EPSILON { 1.0 / d[0] } else { 0.0 };
                }
                JointType::Spherical | JointType::Planar => {
                    let d3 = Matrix3::from_cols(
                        Vector3::new(d[0], d[3], d[6]),
                        Vector3::new(d[1], d[4], d[7]),
                        Vector3::new(d[2], d[5], d[8]),
                    );
                    let inv_d3 = d3.inverse();
                    for row in 0..3 {
                        for col in 0..3 {
                            self.inv_d[inv_di_start + row * 3 + col] = inv_d3.col(col)[row];
                        }
                    }
                }
                JointType::Fixed => {}
            }

            // h · D⁻¹
            let mut spat_force_temps = [SpatialForceVector::zero(); 6];
            for (dof, temp) in spat_force_temps.iter_mut().enumerate().take(dof_count) {
                for dof2 in 0..dof_count {
                    *temp +=
                        self.h[dof_offset + dof2] * self.inv_d[inv_di_start + dof2 * dof_count + dof];
                }
            }
            // Î -= h D⁻¹ hᵀ
            let mut dyad_temp = scratch.spat_inertia[i + 1];
            for (dof, temp) in spat_force_temps.iter().enumerate().take(dof_count) {
                dyad_temp -= symmetric_spatial_outer_product_force(&self.h[dof_offset + dof], temp);
            }
            scratch.spat_inertia[parent_idx] += from_parent.transform_inverse_dyad(&dyad_temp);

            // D⁻¹ Y
            let mut inv_d_times_y = [0.0; 6];
            for (dof, slot) in inv_d_times_y.iter_mut().enumerate().take(dof_count) {
                let mut acc = 0.0;
                for dof2 in 0..dof_count {
                    acc += self.inv_d[inv_di_start + dof * dof_count + dof2] * y[dof_offset + dof2];
                }
                *slot = acc;
            }
            // ẑ_p += X⁻¹ (ẑ_i + Î ĉ + h D⁻¹ Y)
            spat_force_temps[0] = scratch.zero_acc[i + 1]
                + scratch.spat_inertia[i + 1].apply(&scratch.coriolis[i]);
            for (h_row, &dy) in self.h[dof_offset..dof_offset + dof_count]
                .iter()
                .zip(&inv_d_times_y)
            {
                spat_force_temps[0] += *h_row * dy;
            }
            spat_force_temps[1] = from_parent.transform_inverse_force(&spat_force_temps[0]);
            scratch.zero_acc[parent_idx] += spat_force_temps[1];
        }

        // Second 'upward' loop (part of TreeForwardDynamics in Mirtich).
        if self.fixed_base {
            scratch.spat_acc[0] = SpatialMotionVector::zero();
        } else {
            if num_links > 0 {
                self.cached_inertia_valid = true;
                self.cached_inertia[0] = scratch.spat_inertia[0].top_left;
                self.cached_inertia[1] = scratch.spat_inertia[0].top_right;
                self.cached_inertia[2] = scratch.spat_inertia[0].bottom_left;
                self.cached_inertia[3] = scratch.spat_inertia[0].top_left.transpose();
            }
            let result = self.solve_imatrix(&scratch.zero_acc[0]);
            scratch.spat_acc[0] = -result;
        }

        for i in 0..num_links {
            let parent = self.links[i].parent;
            let parent_idx = parent.map_or(0, |p| p + 1);
            let from_parent = SpatialTransformationMatrix {
                rot: self.rot_from_parent[i + 1],
                trn: self.links[i].cached_r_vector,
            };
            scratch.spat_acc[i + 1] = from_parent.transform(&scratch.spat_acc[parent_idx]);

            let dof_offset = self.links[i].dof_offset;
            let dof_count = self.links[i].dof_count;
            let mut y_minus_ht_a = [0.0; 6];
            for dof in 0..dof_count {
                y_minus_ht_a[dof] = y[dof_offset + dof]
                    - scratch.spat_acc[i + 1].dot_force(&self.h[dof_offset + dof]);
            }
            let inv_di_start = dof_offset * dof_offset;
            Self::mul_matrix(
                &self.inv_d[inv_di_start..inv_di_start + dof_count * dof_count],
                &y_minus_ht_a,
                dof_count,
                dof_count,
                dof_count,
                1,
                &mut output[6 + dof_offset..6 + dof_offset + dof_count],
            );
            scratch.spat_acc[i + 1] += scratch.coriolis[i];
            for dof in 0..dof_count {
                scratch.spat_acc[i + 1] += self.links[i].axes[dof] * output[6 + dof_offset + dof];
            }

            // Joint reaction forces (Bullet's `m_jointFeedback` writeback in
            // the second upward loop): the spatial force transmitted through
            // the inboard joint, `Îᵢ · âᵢ + ẑᵢ`, at the link COM. The
            // external-force pass stores (`=`), the constraint pass adds
            // (`+=`), so after a step the feedback holds both contributions.
            if self.links[i].joint_feedback.is_some() {
                self.needs_joint_feedback = true;
                let reaction = scratch.spat_inertia[i + 1].apply(&scratch.spat_acc[i + 1])
                    + scratch.zero_acc[i + 1];
                let mut force = reaction.top;
                let mut torque = reaction.bottom;
                if self.joint_feedback_in_joint_frame {
                    // Shift the torque's reference point to the joint frame:
                    // the force is unchanged, the moment arm is d_vector
                    // (joint → COM, link frame).
                    torque -= force.cross(self.links[i].d_vector);
                }
                if self.joint_feedback_in_world_space {
                    // Link frame → world frame (the transpose of the cached
                    // world→link rotation).
                    let to_world = scratch.rot_from_world[i + 1].transpose();
                    force = to_world * force;
                    torque = to_world * torque;
                }
                let fb = self.links[i].joint_feedback.as_mut().unwrap();
                if is_constraint_pass {
                    fb.top += force;
                    fb.bottom += torque;
                } else {
                    *fb = SpatialForceVector::new(force, torque);
                }
            }
        }

        // Transform base accelerations back to the world frame.
        let omegadot_out = self.rot_from_parent[0].transpose() * scratch.spat_acc[0].top;
        output[0] = omegadot_out.x;
        output[1] = omegadot_out.y;
        output[2] = omegadot_out.z;
        let vdot_out = self.rot_from_parent[0].transpose()
            * (scratch.spat_acc[0].bottom + scratch.spat_vel[0].top.cross(scratch.spat_vel[0].bottom));
        output[3] = vdot_out.x;
        output[4] = vdot_out.y;
        output[5] = vdot_out.z;

        // Final step: add the accelerations (times dt) to the velocities.
        if !is_constraint_pass && dt > 0.0 {
            self.apply_delta_vee(output, dt);
        }
    }

    /// Solve `I · x = rhs` for the composite base spatial inertia, where `I`
    /// is the 6x6 matrix stored as four 3x3 blocks (Bullet's
    /// `solveImatrix`).
    fn solve_imatrix(&self, rhs: &SpatialForceVector) -> SpatialMotionVector {
        if self.num_links() == 0 {
            // Plain rigid body: rhs / invI is easier.
            let angular = if self.base_inertia.x >= Real::EPSILON
                && self.base_inertia.y >= Real::EPSILON
                && self.base_inertia.z >= Real::EPSILON
            {
                rhs.angular() / self.base_inertia
            } else {
                Vector3::ZERO
            };
            let linear = if self.base_mass >= Real::EPSILON {
                rhs.linear() / self.base_mass
            } else {
                Vector3::ZERO
            };
            SpatialMotionVector::new(angular, linear)
        } else {
            if !self.cached_inertia_valid {
                return SpatialMotionVector::zero();
            }
            // Special routine for the inverse of a spatial inertia matrix.
            let top_left = &self.cached_inertia[0];
            let top_right = &self.cached_inertia[1];
            let bottom_left = &self.cached_inertia[2];
            let bottom_right = &self.cached_inertia[3];

            let b_inv = top_right.inverse() * -1.0;
            let mut tmp = bottom_right * b_inv;
            let inv_i_upper_right = (tmp * top_left + bottom_left).inverse();
            tmp = inv_i_upper_right * bottom_right;
            let inv_i_upper_left = tmp * b_inv;
            let inv_i_lower_right = inv_i_upper_left.transpose();
            tmp = top_left * inv_i_upper_left;
            tmp -= Matrix3::IDENTITY;
            let inv_i_lower_left = b_inv * tmp;

            let vtop = inv_i_upper_left * rhs.linear() + inv_i_upper_right * rhs.angular();
            let vbot = inv_i_lower_left * rhs.linear() + inv_i_lower_right * rhs.angular();
            SpatialMotionVector::new(vtop, vbot)
        }
    }

    /// Compute the generalized-velocity response of this multibody to a
    /// force vector `force` laid out like a jacobian (`[τ, F, joint
    /// torques...]`): `output = M⁻¹ · force` in the world frame. Must have
    /// been preceded by a [`MultiBody::compute_accelerations_aba`] call to
    /// populate the cached factorization. Port of
    /// `calcAccelerationDeltasMultiDof`.
    pub fn calc_acceleration_deltas(
        &self,
        force: &[Real],
        output: &mut [Real],
        scratch: &mut MultiBodyScratch,
    ) {
        let num_links = self.num_links();
        scratch.ensure(num_links, self.dof_count);
        // Pooled joint-space force vector (cleared per call, no alloc).
        scratch.dof_force.clear();
        scratch.dof_force.resize(self.dof_count, 0.0);
        let y = &mut scratch.dof_force[..];
        // zero_acc uses the first num_links+1 slots, spat_acc the next
        // (Bullet reuses one scratch block; we use separate buffers).
        let rot_from_parent = &self.rot_from_parent;
        let inv_d = &self.inv_d;
        let h = &self.h;

        // ẑᵢᴬ: the applied force on the base, zero elsewhere.
        if self.is_base_static_or_kinematic() {
            scratch.zero_acc[0] = SpatialForceVector::zero();
        } else {
            let from_parent = SpatialTransformationMatrix {
                rot: rot_from_parent[0],
                trn: Vector3::ZERO,
            };
            scratch.zero_acc[0] = from_parent.transform_rotation_only(&SpatialForceVector::new(
                Vector3::new(-force[3], -force[4], -force[5]),
                Vector3::new(-force[0], -force[1], -force[2]),
            ));
        }
        for i in 0..num_links {
            scratch.zero_acc[i + 1] = SpatialForceVector::zero();
        }

        // 'Downward' loop.
        for i in (0..num_links).rev() {
            if self.is_link_and_all_ancestors_kinematic(i) {
                continue;
            }
            let parent = self.links[i].parent;
            let parent_idx = parent.map_or(0, |p| p + 1);
            let from_parent = SpatialTransformationMatrix {
                rot: rot_from_parent[i + 1],
                trn: self.links[i].cached_r_vector,
            };
            let dof_offset = self.links[i].dof_offset;
            let dof_count = self.links[i].dof_count;
            let inv_di_start = dof_offset * dof_offset;

            for dof in 0..dof_count {
                y[dof_offset + dof] =
                    force[6 + dof_offset + dof] - self.links[i].axes[dof].dot_force(&scratch.zero_acc[i + 1]);
            }
            let mut inv_d_times_y = [0.0; 6];
            for dof in 0..dof_count {
                let mut acc = 0.0;
                for dof2 in 0..dof_count {
                    acc += inv_d[inv_di_start + dof * dof_count + dof2] * y[dof_offset + dof2];
                }
                inv_d_times_y[dof] = acc;
            }
            let mut spat_force_temps = [SpatialForceVector::zero(); 6];
            spat_force_temps[0] = scratch.zero_acc[i + 1];
            for dof in 0..dof_count {
                spat_force_temps[0] += h[dof_offset + dof] * inv_d_times_y[dof];
            }
            spat_force_temps[1] = from_parent.transform_inverse_force(&spat_force_temps[0]);
            scratch.zero_acc[parent_idx] += spat_force_temps[1];
        }

        // Second 'upward' loop.
        if self.is_base_static_or_kinematic() {
            scratch.spat_acc[0] = SpatialMotionVector::zero();
        } else {
            let result = self.solve_imatrix(&scratch.zero_acc[0]);
            scratch.spat_acc[0] = -result;
        }

        for i in 0..num_links {
            if self.is_link_and_all_ancestors_kinematic(i) {
                continue;
            }
            let parent = self.links[i].parent;
            let parent_idx = parent.map_or(0, |p| p + 1);
            let from_parent = SpatialTransformationMatrix {
                rot: rot_from_parent[i + 1],
                trn: self.links[i].cached_r_vector,
            };
            scratch.spat_acc[i + 1] = from_parent.transform(&scratch.spat_acc[parent_idx]);

            let dof_offset = self.links[i].dof_offset;
            let dof_count = self.links[i].dof_count;
            let inv_di_start = dof_offset * dof_offset;
            let mut y_minus_ht_a = [0.0; 6];
            for dof in 0..dof_count {
                y_minus_ht_a[dof] = y[dof_offset + dof] - scratch.spat_acc[i + 1].dot_force(&h[dof_offset + dof]);
            }
            Self::mul_matrix(
                &inv_d[inv_di_start..inv_di_start + dof_count * dof_count],
                &y_minus_ht_a,
                dof_count,
                dof_count,
                dof_count,
                1,
                &mut output[6 + dof_offset..6 + dof_offset + dof_count],
            );
            for dof in 0..dof_count {
                scratch.spat_acc[i + 1] += self.links[i].axes[dof] * output[6 + dof_offset + dof];
            }
        }

        // Transform base accelerations back to the world frame.
        let omegadot_out = rot_from_parent[0].transpose() * scratch.spat_acc[0].top;
        output[0] = omegadot_out.x;
        output[1] = omegadot_out.y;
        output[2] = omegadot_out.z;
        let vdot_out = rot_from_parent[0].transpose() * scratch.spat_acc[0].bottom;
        output[3] = vdot_out.x;
        output[4] = vdot_out.y;
        output[5] = vdot_out.z;
    }

    /// Timestep the positions given the current velocities (Bullet's
    /// `stepPositionsMultiDof` with the internal state buffers).
    pub fn step_positions(&mut self, dt: Real) {
        let num_links = self.num_links();

        // Step the base position by adding dt * velocity.
        self.base_pos += dt * self.base_vel();

        // Exponential-map quaternion integration (Bullet's local functor
        // `pQuatUpdateFun`, based on `btTransformUtil::integrateTransform`).
        let p_quat_update_fun = |omega_in: Vector3, quat: &mut Quaternion, base_body: bool, dt: Real| {
            // baseBody: quat is the world→base map and omega is in world
            // coordinates; otherwise quat is the parent→this map and omega is
            // in the joint's local coordinates.
            let angvel = if base_body { omega_in } else { *quat * omega_in };
            let mut f_angle = angvel.length();
            // Limit the angular motion.
            if f_angle * dt > ANGULAR_MOTION_THRESHOLD {
                f_angle = 0.5 * bullet_math::FRAC_PI_2 / dt;
            }
            let axis = if f_angle < 0.001 {
                // Taylor expansion of the sinc function (1/48 ≈ 0.0208333…).
                angvel * (0.5 * dt - (dt * dt * dt) * (1.0 / 48.0) * f_angle * f_angle)
            } else {
                angvel * (math::sin(0.5 * f_angle * dt) / f_angle)
            };
            let w = math::cos(f_angle * dt * 0.5);
            if base_body {
                *quat *= Quaternion::from_xyzw(-axis.x, -axis.y, -axis.z, w);
            } else {
                *quat = Quaternion::from_xyzw(axis.x, axis.y, axis.z, w) * *quat;
            }
            *quat = quat.normalize();
        };

        // Step the base orientation.
        p_quat_update_fun(self.base_omega(), &mut self.base_quat, true, dt);

        // Step the joint positions.
        for i in 0..num_links {
            let joint_vel = self.joint_vel_multidof(i).to_vec();
            match self.links[i].joint_type {
                JointType::Prismatic | JointType::Revolute => {
                    self.links[i].joint_pos[0] += dt * joint_vel[0];
                }
                JointType::Spherical => {
                    let mut joint_ori = Quaternion::from_xyzw(
                        self.links[i].joint_pos[0],
                        self.links[i].joint_pos[1],
                        self.links[i].joint_pos[2],
                        self.links[i].joint_pos[3],
                    );
                    p_quat_update_fun(
                        Vector3::new(joint_vel[0], joint_vel[1], joint_vel[2]),
                        &mut joint_ori,
                        false,
                        dt,
                    );
                    self.links[i].joint_pos[0] = joint_ori.x;
                    self.links[i].joint_pos[1] = joint_ori.y;
                    self.links[i].joint_pos[2] = joint_ori.z;
                    self.links[i].joint_pos[3] = joint_ori.w;
                }
                JointType::Planar => {
                    self.links[i].joint_pos[0] += dt * joint_vel[0];
                    let q0_coors = joint_vel[1] * self.links[i].axis_bottom(1)
                        + joint_vel[2] * self.links[i].axis_bottom(2);
                    let no_q0_coors =
                        quat_from_axis_angle(self.links[i].axis_top(0), self.links[i].joint_pos[0]) * q0_coors;
                    self.links[i].joint_pos[1] += self.links[i].axis_bottom(1).dot(no_q0_coors) * dt;
                    self.links[i].joint_pos[2] += self.links[i].axis_bottom(2).dot(no_q0_coors) * dt;
                }
                JointType::Fixed => {}
            }
            self.links[i].update_cache_multidof(None);
        }
    }

    /// Fill a constraint jacobian for a 1D constraint acting on link `link`
    /// at `contact_point` (world), with angular normal `normal_ang` and
    /// linear normal `normal_lin` (world): `jac = [ω coeffs, v coeffs,
    /// q̇ coeffs...]` in the same layout as the velocity vector. Port of
    /// `fillConstraintJacobianMultiDof`.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_constraint_jacobian(
        &self,
        link: i32,
        contact_point: Vector3,
        normal_ang: Vector3,
        normal_lin: Vector3,
        jac: &mut [Real],
        scratch: &mut MultiBodyScratch,
    ) {
        let num_links = self.num_links();
        // Walk the chain from the constrained link to the base — buffers
        // pooled in the scratch (this runs per contact row per side).
        scratch.ensure(num_links, self.dof_count);
        let mut l = link;
        scratch.jac_chain.clear();
        while l != -1 {
            scratch.jac_chain.push(l as usize);
            l = self.links[l as usize].parent.map_or(-1, |p| p as i32);
        }
        let chain = &scratch.jac_chain[..];
        // Local-frame positions and normals, one per link + base.
        let n = num_links + 1;
        for buf in [&mut scratch.jac_p_local, &mut scratch.jac_n_lin, &mut scratch.jac_n_ang] {
            buf.clear();
            buf.resize(n, Vector3::ZERO);
        }
        scratch.jac_results.clear();
        scratch.jac_results.resize(self.dof_count, 0.0);
        let p_minus_com_local = &mut scratch.jac_p_local[..];
        let n_local_lin = &mut scratch.jac_n_lin[..];
        let n_local_ang = &mut scratch.jac_n_ang[..];
        let results = &mut scratch.jac_results[..];

        scratch.rot_from_world[0] = Matrix3::from_quat(self.base_quat);

        // Omega coefficients first.
        let p_minus_com_world = contact_point - self.base_pos;
        let omega_coeffs_world = p_minus_com_world.cross(normal_lin);
        jac[0] = omega_coeffs_world.x + normal_ang.x;
        jac[1] = omega_coeffs_world.y + normal_ang.y;
        jac[2] = omega_coeffs_world.z + normal_ang.z;
        // Then the v coefficients.
        jac[3] = normal_lin.x;
        jac[4] = normal_lin.y;
        jac[5] = normal_lin.z;

        // Link-local versions for the base.
        p_minus_com_local[0] = scratch.rot_from_world[0] * p_minus_com_world;
        n_local_lin[0] = scratch.rot_from_world[0] * normal_lin;
        n_local_ang[0] = scratch.rot_from_world[0] * normal_ang;

        // Set the remaining jac values to zero for now.
        for v in jac.iter_mut().skip(6) {
            *v = 0.0;
        }

        if num_links > 0 && link > -1 {
            // Calculate the required normals & positions in the local frames,
            // walking from the base down to the constrained link.
            for a in 0..chain.len() {
                let i = chain[chain.len() - 1 - a];
                let parent = self.links[i].parent;
                let parent_idx = parent.map_or(0, |p| p + 1);
                let mtx = Matrix3::from_quat(self.links[i].cached_rot_parent_to_this);
                scratch.rot_from_world[i + 1] = mtx * scratch.rot_from_world[parent_idx];
                n_local_lin[i + 1] = mtx * n_local_lin[parent_idx];
                n_local_ang[i + 1] = mtx * n_local_ang[parent_idx];
                p_minus_com_local[i + 1] = mtx * p_minus_com_local[parent_idx] - self.links[i].cached_r_vector;

                let dof_offset = self.links[i].dof_offset;
                let pml = p_minus_com_local[i + 1];
                let nll = n_local_lin[i + 1];
                let nla = n_local_ang[i + 1];
                match self.links[i].joint_type {
                    JointType::Revolute => {
                        results[dof_offset] =
                            nll.dot(self.links[i].axis_top(0).cross(pml) + self.links[i].axis_bottom(0));
                        results[dof_offset] += nla.dot(self.links[i].axis_top(0));
                    }
                    JointType::Prismatic => {
                        results[dof_offset] = nll.dot(self.links[i].axis_bottom(0));
                    }
                    JointType::Spherical => {
                        results[dof_offset] =
                            nll.dot(self.links[i].axis_top(0).cross(pml) + self.links[i].axis_bottom(0));
                        results[dof_offset + 1] =
                            nll.dot(self.links[i].axis_top(1).cross(pml) + self.links[i].axis_bottom(1));
                        results[dof_offset + 2] =
                            nll.dot(self.links[i].axis_top(2).cross(pml) + self.links[i].axis_bottom(2));
                        results[dof_offset] += nla.dot(self.links[i].axis_top(0));
                        results[dof_offset + 1] += nla.dot(self.links[i].axis_top(1));
                        results[dof_offset + 2] += nla.dot(self.links[i].axis_top(2));
                    }
                    JointType::Planar => {
                        results[dof_offset] = nll.dot(self.links[i].axis_top(0).cross(pml));
                        results[dof_offset + 1] = nll.dot(self.links[i].axis_bottom(1));
                        results[dof_offset + 2] = nll.dot(self.links[i].axis_bottom(2));
                    }
                    JointType::Fixed => {}
                }
            }
            // Copy the results into the output, walking the chain back up.
            let mut idx = link;
            while idx != -1 {
                let k = idx as usize;
                for dof in 0..self.links[k].dof_count {
                    jac[6 + self.links[k].dof_offset + dof] = results[self.links[k].dof_offset + dof];
                }
                idx = self.links[k].parent.map_or(-1, |p| p as i32);
            }
        }
    }

    /// Compute the world transforms of every link (Bullet's
    /// `forwardKinematics`), refreshing `cached_world_transform`.
    pub fn forward_kinematics(&mut self) {
        let num_links = self.num_links();
        let rot_from_parent = &mut self.rot_from_parent;
        rot_from_parent[0] = Matrix3::from_quat(self.base_quat);
        for i in 0..num_links {
            rot_from_parent[i + 1] = Matrix3::from_quat(self.links[i].cached_rot_parent_to_this);
        }
        // world_to_local[k] rotates world vectors into frame k; local_origin
        // is the frame origin in world coordinates.
        let mut world_to_local = vec![Quaternion::IDENTITY; num_links + 1];
        let mut local_origin = vec![Vector3::ZERO; num_links + 1];
        world_to_local[0] = self.base_quat;
        local_origin[0] = self.base_pos;
        for k in 0..num_links {
            let parent = self.links[k].parent;
            let parent_idx = parent.map_or(0, |p| p + 1);
            world_to_local[k + 1] = self.links[k].cached_rot_parent_to_this * world_to_local[parent_idx];
            local_origin[k + 1] = local_origin[parent_idx] + (world_to_local[k + 1].inverse() * self.links[k].cached_r_vector);
        }
        for link in 0..num_links {
            let index = link + 1;
            let rot = Quaternion::from_xyzw(
                -world_to_local[index].x,
                -world_to_local[index].y,
                -world_to_local[index].z,
                world_to_local[index].w,
            );
            self.links[link].cached_world_transform =
                transform_from_rotation_translation(rot, local_origin[index]);
        }
    }

    // ------------------------------------------------------------------
    // Sleeping
    // ------------------------------------------------------------------

    pub fn has_fixed_base(&self) -> bool {
        self.fixed_base
    }

    /// The base is static or kinematic. Without link colliders this equals
    /// `fixed_base` (Bullet would also consult the base collider's flags).
    pub fn is_base_static_or_kinematic(&self) -> bool {
        self.fixed_base
    }

    /// Without link colliders, no link is ever kinematic (Bullet checks each
    /// link collider's `isKinematic`).
    pub fn is_link_and_all_ancestors_kinematic(&self, _i: usize) -> bool {
        false
    }

    pub fn is_awake(&self) -> bool {
        self.awake
    }

    pub fn can_sleep(&self) -> bool {
        self.can_sleep
    }

    pub fn set_can_sleep(&mut self, can_sleep: bool) {
        if self.can_wakeup {
            self.can_sleep = can_sleep;
        }
    }

    pub fn can_wakeup(&self) -> bool {
        self.can_wakeup
    }

    pub fn set_can_wakeup(&mut self, can_wakeup: bool) {
        self.can_wakeup = can_wakeup;
    }

    pub fn wake_up(&mut self) {
        self.sleep_timer = 0.0;
        self.awake = true;
    }

    pub fn go_to_sleep(&mut self) {
        self.awake = false;
    }

    /// Bullet's `checkMotionAndSleepIfRequired`: accumulate the sleep timer
    /// while the squared generalized speed stays below `sleep_epsilon`.
    pub fn check_motion_and_sleep(&mut self, timestep: Real) {
        if !self.can_sleep {
            self.awake = true;
            self.sleep_timer = 0.0;
            return;
        }
        // Motion is omega² + v² + (sum of squares of joint velocities).
        let mut motion = 0.0;
        for &v in &self.velocities {
            motion += v * v;
        }
        if motion < self.sleep_epsilon {
            self.sleep_timer += timestep;
            if self.sleep_timer > self.sleep_timeout {
                self.go_to_sleep();
            }
        } else {
            self.sleep_timer = 0.0;
            if self.can_wakeup && !self.awake {
                self.wake_up();
            }
        }
    }

    pub fn set_sleep_threshold(&mut self, sleep_epsilon: Real) {
        self.sleep_epsilon = sleep_epsilon;
    }

    pub fn set_sleep_timeout(&mut self, sleep_timeout: Real) {
        self.sleep_timeout = sleep_timeout;
    }

    pub fn set_linear_damping(&mut self, damp: Real) {
        self.linear_damping = damp;
    }

    pub fn set_angular_damping(&mut self, damp: Real) {
        self.angular_damping = damp;
    }

    pub fn linear_damping(&self) -> Real {
        self.linear_damping
    }

    pub fn angular_damping(&self) -> Real {
        self.angular_damping
    }

    pub fn set_use_gyro_term(&mut self, use_gyro: bool) {
        self.use_gyro_term = use_gyro;
    }

    pub fn use_gyro_term(&self) -> bool {
        self.use_gyro_term
    }

    pub fn set_max_coordinate_velocity(&mut self, max_vel: Real) {
        self.max_coordinate_velocity = max_vel;
    }

    pub fn max_coordinate_velocity(&self) -> Real {
        self.max_coordinate_velocity
    }

    pub fn set_max_applied_impulse(&mut self, max_imp: Real) {
        self.max_applied_impulse = max_imp;
    }

    pub fn max_applied_impulse(&self) -> Real {
        self.max_applied_impulse
    }

    pub fn set_has_self_collision(&mut self, has: bool) {
        self.has_self_collision = has;
    }

    pub fn has_self_collision(&self) -> bool {
        self.has_self_collision
    }

    /// Enable (from zero) the joint reaction-force feedback of link `i`
    /// (assigning a `btMultiBodyJointFeedback` to the link in Bullet). Read
    /// it back with [`MultiBody::link_joint_feedback`] after `step_simulation`.
    pub fn enable_link_joint_feedback(&mut self, i: usize) {
        self.links[i].joint_feedback = Some(SpatialForceVector::zero());
    }

    /// Disable the joint reaction-force feedback of link `i`.
    pub fn disable_link_joint_feedback(&mut self, i: usize) {
        self.links[i].joint_feedback = None;
    }

    /// The latest joint reaction forces of link `i`, if feedback is enabled.
    /// `top` = linear force, `bottom` = angular torque.
    pub fn link_joint_feedback(&self, i: usize) -> Option<SpatialForceVector> {
        self.links[i].joint_feedback
    }

    /// Whether any link's feedback was written by the last articulated-body
    /// pass (Bullet's `internalNeedsJointFeedback`); the world then runs the
    /// post-solve constraint pass.
    pub fn needs_joint_feedback(&self) -> bool {
        self.needs_joint_feedback
    }

    /// Report joint reaction forces in the world frame (Bullet's solver-info
    /// flag `m_jointFeedbackInWorldSpace`).
    pub fn set_joint_feedback_in_world_space(&mut self, world_space: bool) {
        self.joint_feedback_in_world_space = world_space;
    }

    /// Report the torque about the joint frame (Bullet's solver-info flag
    /// `m_jointFeedbackInJointFrame`).
    pub fn set_joint_feedback_in_joint_frame(&mut self, joint_frame: bool) {
        self.joint_feedback_in_joint_frame = joint_frame;
    }

    /// Attach a collider to the base (Bullet's base
    /// `btMultiBodyLinkCollider`). The world registers it as a broad-phase
    /// proxy at the next step.
    pub fn set_base_collider(&mut self, shape: CollisionShape, friction: Real, restitution: Real) {
        self.base_collider = Some(LinkCollider {
            shape,
            friction,
            restitution,
            collision_group: crate::body::filter_groups::DEFAULT,
            collision_mask: crate::body::filter_groups::ALL,
            collision_events: false,
        });
    }

    /// Attach a collider to link `i` (Bullet's link `m_collider`).
    pub fn set_link_collider(
        &mut self,
        i: usize,
        shape: CollisionShape,
        friction: Real,
        restitution: Real,
    ) {
        self.link_colliders[i] = Some(LinkCollider {
            shape,
            friction,
            restitution,
            collision_group: crate::body::filter_groups::DEFAULT,
            collision_mask: crate::body::filter_groups::ALL,
            collision_events: false,
        });
    }

    pub fn base_collider(&self) -> Option<&LinkCollider> {
        self.base_collider.as_ref()
    }

    pub fn link_collider(&self, i: usize) -> Option<&LinkCollider> {
        self.link_colliders[i].as_ref()
    }

    /// The base's world orientation (the inverse of `base_quat`, which maps
    /// world vectors into the base frame).
    pub fn base_world_orientation(&self) -> Quaternion {
        self.base_quat.inverse()
    }

    /// The world position of link `i`'s frame origin, from the cached FK
    /// transform (`m_cachedWorldTransform.getOrigin()`).
    pub fn link_world_position(&self, i: usize) -> Vector3 {
        #[allow(clippy::useless_conversion)] // Vec3A -> Vector3, identity in f64
        Vector3::from(self.links[i].cached_world_transform.translation)
    }

    /// Whether the colliders of links `a` and `b` (`-1` = base) of this body
    /// may collide — port of `btMultiBodyLinkCollider::checkCollideWithOverride`
    /// for two colliders of the same multibody. Adjacent links collide by
    /// default; `disable_parent_collision` / `disable_all_parent_collision`
    /// (Bullet's link flags `BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION` /
    /// `_DISABLE_ALL_PARENT_COLLISION`) opt out per link, and
    /// [`set_has_self_collision`](Self::set_has_self_collision) disables
    /// self-collision for the whole body.
    pub fn colliders_collide(&self, a: i32, b: i32) -> bool {
        if !self.has_self_collision {
            return false;
        }
        // Does link `link` disable collision with ancestor `other`?
        let excludes = |link: usize, other: i32| -> bool {
            let l = &self.links[link];
            if l.disable_all_parent_collision {
                // Walk the ancestor chain including the base (Bullet's loop
                // reaches -1 and compares it against the base collider).
                let mut p = link as i32;
                loop {
                    if p == -1 {
                        break;
                    }
                    p = match self.links[p as usize].parent {
                        Some(pp) => pp as i32,
                        None => -1,
                    };
                    if p == other {
                        return true;
                    }
                }
                false
            } else if l.disable_parent_collision {
                match l.parent {
                    Some(pp) => pp as i32 == other,
                    None => other == -1,
                }
            } else {
                false
            }
        };
        if a >= 0 && excludes(a as usize, b) {
            return false;
        }
        if b >= 0 && excludes(b as usize, a) {
            return false;
        }
        true
    }

    /// Disable collisions between link `i`'s collider and its direct parent
    /// (`all_ancestors = false`) or all of its ancestors including the base
    /// (`true`) — Bullet's link flags
    /// `BT_MULTIBODYLINKFLAGS_DISABLE_(ALL_)PARENT_COLLISION`.
    pub fn set_link_parent_collision_disabled(&mut self, i: usize, all_ancestors: bool) {
        let link = &mut self.links[i];
        if all_ancestors {
            link.disable_all_parent_collision = true;
        } else {
            link.disable_parent_collision = true;
        }
    }

    pub fn set_pos_updated(&mut self, updated: bool) {
        self.pos_updated = updated;
    }

    pub fn is_pos_updated(&self) -> bool {
        self.pos_updated
    }

    pub fn use_rk4_integration(&self) -> bool {
        self.use_rk4
    }

    pub fn set_use_rk4_integration(&mut self, use_rk4: bool) {
        self.use_rk4 = use_rk4;
    }
}
