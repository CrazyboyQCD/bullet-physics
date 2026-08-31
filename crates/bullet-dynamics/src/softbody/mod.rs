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
//! `docs/softbody.md` §4).
//!
//! Reference classes: `btSoftBody`, `btSoftBodyHelpers`,
//! `btSoftRigidDynamicsWorld`.

pub mod body;
pub mod collide;
pub mod config;
pub mod helpers;
pub mod solver;
#[cfg(test)]
mod tests;

pub use body::{Link, Material, Node, SoftBody};
pub use collide::{Anchor, Cti, RContact};
pub use config::{
    AeroModel, Config, PSolverKind, SoftCollision, SolverPreset, SolverState, VSolverKind,
};
pub use helpers::{create_rope, ROPE_FIX_END, ROPE_FIX_START};
