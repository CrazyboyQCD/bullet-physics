//! Position-based soft bodies — port of Bullet's classic
//! `BulletSoftBody` system (cloth/ropes/volumetric bodies with node-link
//! dynamics, anchored and coupled to rigid bodies through contacts).
//!
//! The simulation model is Bullet's predict-solve loop: `predict_motion`
//! integrates the nodes to predicted positions, collision handlers record
//! contacts against rigid bodies and other soft bodies, and
//! `solve_constraints` iterates position kernels over links/anchors/
//! contacts before writing the velocities back from the corrected
//! positions. Stepping is owned by [`crate::world::PhysicsWorld`]
//! (insertion points follow `btSoftRigidDynamicsWorld`; see
//! `docs/design/softbody.md` §4).
//!
//! Reference classes: `btSoftBody`, `btSoftBodyHelpers`,
//! `btSoftRigidDynamicsWorld`.

pub mod body;
pub mod cluster;
pub mod collide;
pub mod config;
pub mod forces;
pub mod geometry;
pub mod helpers;
pub mod joints;
pub mod quickhull;
pub mod rng;
pub mod solver;
#[cfg(test)]
mod tests;

pub use body::{
    Cluster, Face, Impulse, Link, Material, Node, Pose, Side, SoftBody, Tetra,
};
pub use body::ImplicitFn;
pub use cluster::solve_clusters;
pub use collide::{Anchor, ContactOwner, Cti, RContact, SContact};
pub use config::{
    AeroModel, Config, PSolverKind, SoftCollision, SolverPreset, SolverState, VSolverKind,
};
pub use helpers::{
    create_ellipsoid, create_from_convex_hull, create_from_tri_mesh, create_patch, create_rope,
    patch_fixeds, ROPE_FIX_END, ROPE_FIX_START,
};
pub use joints::{
    AngularControl, AngularJointSpecs, AJoint, CJoint, DefaultAngularControl, JointSpecs,
    LinearJointSpecs, LJoint, SoftJoint,
};
pub use rng::MsvcRand;
