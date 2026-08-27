//! A single link (and its inboard joint) of a Featherstone articulated body —
//! port of `BulletDynamics/Featherstone/btMultiBodyLink.h`.
//!
//! Each link stores its mass, diagonal local inertia, the joint connecting it
//! to its parent, and the cached kinematics derived from the joint position
//! variables (`m_cachedRotParentToThis` / `m_cachedRVector`). The link frame
//! origin is the link's center of mass (COM), which keeps the spatial inertia
//! block-diagonal.

use bullet_math::{
    Quaternion, Real, SpatialForceVector, SpatialMotionVector, Transform, Vector3,
};
use bullet_math::compat::quat_from_axis_angle;

/// Link flags (Bullet's `btMultiBodyLinkFlags`). Parent-collision flags are
/// honored by the collision layer; this crate's multibody subsystem does not
/// do collision detection itself, so the flags are stored for forward
/// compatibility.
pub const FLAG_DISABLE_PARENT_COLLISION: u32 = 1;
pub const FLAG_DISABLE_ALL_PARENT_COLLISION: u32 = 2;

/// The type of the inboard joint of a link (Bullet's
/// `eFeatherstoneJointType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType {
    Revolute,
    Prismatic,
    Spherical,
    Planar,
    Fixed,
}

/// One link of a [`crate::multibody::MultiBody`], excluding the base.
#[derive(Debug, Clone)]
pub struct MultiBodyLink {
    /// Mass of the link.
    pub mass: Real,
    /// Diagonal inertia of the link in its local (COM) frame.
    pub inertia_local: Vector3,
    /// Index of the parent link, or `None` if the parent is the base.
    pub parent: Option<usize>,
    /// Rotates vectors in the parent frame to vectors in this frame when the
    /// joint position is zero. Constant.
    pub zero_rot_parent_to_this: Quaternion,
    /// Vector from the inboard joint position to this link's COM, in this
    /// link's local frame. Constant (zero for planar joints).
    pub d_vector: Vector3,
    /// Joint-type dependent, constant: for revolute/prismatic/fixed/spherical
    /// it is the vector from the parent's COM to the pivot point, in the
    /// PARENT's frame; for planar, the vector from the parent's COM to this
    /// link's COM when `q = 0` (local frame).
    pub e_vector: Vector3,
    /// Spatial joint axes (`m_axes`), one per DOF, expressed in the local
    /// frame. Constant. Only the first `dof_count` entries are meaningful.
    pub axes: [SpatialMotionVector; 6],
    /// Offset of this link's DOFs inside the multibody's flattened velocity
    /// vector (Bullet's `m_dofOffset`).
    pub dof_offset: usize,
    /// Offset of this link's position variables inside the multibody's
    /// flattened configuration (Bullet's `m_cfgOffset`).
    pub cfg_offset: usize,
    /// Cached rotation from the parent frame to this frame, derived from the
    /// joint position.
    pub cached_rot_parent_to_this: Quaternion,
    /// Cached vector from the parent's COM to this link's COM, in this link's
    /// local frame.
    pub cached_r_vector: Vector3,
    /// External force on the link, in the world frame.
    pub applied_force: Vector3,
    /// External torque on the link, in the world frame.
    pub applied_torque: Vector3,
    /// Constraint (solver) force on the link, in the world frame.
    pub applied_constraint_force: Vector3,
    /// Constraint (solver) torque on the link, in the world frame.
    pub applied_constraint_torque: Vector3,
    /// Joint position variables: `[q]` for 1-DOF joints, `[x,y,z,w]`
    /// (quaternion) for spherical, `[q0,q1,q2]` for planar (Bullet's
    /// `m_jointPos[7]`).
    pub joint_pos: [Real; 7],
    /// User-applied joint torques, one per DOF, cleared every step
    /// (Bullet's `m_jointTorque[6]`).
    pub joint_torque: [Real; 6],
    /// Number of velocity DOFs of the inboard joint.
    pub dof_count: usize,
    /// Number of position variables of the inboard joint.
    pub pos_var_count: usize,
    /// The inboard joint type.
    pub joint_type: JointType,
    /// World transform of the link, refreshed by
    /// [`crate::multibody::MultiBody::forward_kinematics`].
    pub cached_world_transform: Transform,
    /// Skip collisions between this link's collider and its direct parent's
    /// (Bullet's `BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION`).
    pub disable_parent_collision: bool,
    /// Skip collisions between this link's collider and **all** of its
    /// ancestors including the base (Bullet's
    /// `BT_MULTIBODYLINKFLAGS_DISABLE_ALL_PARENT_COLLISION`).
    pub disable_all_parent_collision: bool,
    /// Joint reaction-force feedback (Bullet's `m_jointFeedback`, default
    /// null). Written by the articulated-body algorithm: the external-force
    /// pass stores (`=`) its reaction, the post-solve constraint pass adds
    /// (`+=`) the constraint contribution. `top` = linear force,
    /// `bottom` = angular torque, in the link frame unless the body's
    /// world-space / joint-frame flags redirect it.
    pub joint_feedback: Option<SpatialForceVector>,
}

impl Default for MultiBodyLink {
    fn default() -> Self {
        let mut axes = [SpatialMotionVector::zero(); 6];
        axes[0] = SpatialMotionVector::new(Vector3::ZERO, Vector3::X);
        Self {
            mass: 1.0,
            inertia_local: Vector3::ONE,
            parent: None,
            zero_rot_parent_to_this: Quaternion::IDENTITY,
            d_vector: Vector3::ZERO,
            e_vector: Vector3::ZERO,
            axes,
            dof_offset: 0,
            cfg_offset: 0,
            cached_rot_parent_to_this: Quaternion::IDENTITY,
            cached_r_vector: Vector3::ZERO,
            applied_force: Vector3::ZERO,
            applied_torque: Vector3::ZERO,
            applied_constraint_force: Vector3::ZERO,
            applied_constraint_torque: Vector3::ZERO,
            joint_pos: [0.0; 7],
            joint_torque: [0.0; 6],
            dof_count: 0,
            pos_var_count: 0,
            joint_type: JointType::Fixed,
            cached_world_transform: Transform::IDENTITY,
            disable_parent_collision: false,
            disable_all_parent_collision: false,
            joint_feedback: None,
        }
    }
}

impl MultiBodyLink {
    pub fn new() -> Self {
        let mut link = Self::default();
        link.joint_pos[3] = 1.0; // quaternion w
        link
    }

    #[inline]
    pub fn axis_top(&self, dof: usize) -> Vector3 {
        self.axes[dof].top
    }

    #[inline]
    pub fn axis_bottom(&self, dof: usize) -> Vector3 {
        self.axes[dof].bottom
    }

    #[inline]
    pub fn set_axis_top(&mut self, dof: usize, v: Vector3) {
        self.axes[dof].top = v;
    }

    #[inline]
    pub fn set_axis_bottom(&mut self, dof: usize, v: Vector3) {
        self.axes[dof].bottom = v;
    }

    /// Refresh `cached_rot_parent_to_this` and `cached_r_vector` from the
    /// joint position (Bullet's `updateCacheMultiDof`). `pq` optionally
    /// points at this link's position variables in a scratch configuration
    /// array (used by `stepPositionsMultiDof` when integrating into a
    /// temporary buffer).
    pub fn update_cache_multidof(&mut self, pq: Option<&[Real]>) {
        let pos = match pq {
            Some(p) => p,
            None => &self.joint_pos,
        };
        let cached_rot;
        let cached_vector;
        match self.joint_type {
            JointType::Revolute => {
                cached_rot = quat_from_axis_angle(self.axis_top(0), -pos[0]) * self.zero_rot_parent_to_this;
                cached_vector = self.d_vector + (cached_rot * self.e_vector);
            }
            JointType::Prismatic => {
                // The cached rotation never changes for a prismatic joint.
                cached_rot = self.cached_rot_parent_to_this;
                cached_vector =
                    self.d_vector + (self.cached_rot_parent_to_this * self.e_vector) + pos[0] * self.axis_bottom(0);
            }
            JointType::Spherical => {
                // Bullet stores the joint quaternion as (x, y, z, w) and
                // negates w when applying it.
                cached_rot = Quaternion::from_xyzw(pos[0], pos[1], pos[2], -pos[3]) * self.zero_rot_parent_to_this;
                cached_vector = self.d_vector + (cached_rot * self.e_vector);
            }
            JointType::Planar => {
                cached_rot = quat_from_axis_angle(self.axis_top(0), -pos[0]) * self.zero_rot_parent_to_this;
                let q_rot = quat_from_axis_angle(self.axis_top(0), -pos[0]);
                let disp = pos[1] * self.axis_bottom(1) + pos[2] * self.axis_bottom(2);
                cached_vector = q_rot * disp + (cached_rot * self.e_vector);
            }
            JointType::Fixed => {
                cached_rot = self.zero_rot_parent_to_this;
                cached_vector = self.d_vector + (cached_rot * self.e_vector);
            }
        }
        self.cached_rot_parent_to_this = cached_rot;
        self.cached_r_vector = cached_vector;
    }
}
