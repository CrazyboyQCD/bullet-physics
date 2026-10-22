//! # bullet-physics
//!
//! A pure Rust physics engine reimplementing the core of Bullet Physics.
//!
//! This crate is a facade aggregating the workspace crates:
//! [`bullet-math`], [`bullet-collision`], and [`bullet-dynamics`].
//!
//! ## Precision features
//!
//! - `f32` (default): single-precision scalar and SIMD math types.
//! - `f64`: double-precision math types (matches Bullet's default
//!   `BT_USE_DOUBLE_PRECISION`).
//!
//! ## Optional subsystems
//!
//! - `inverse-dynamics` (off by default): re-exports
//!   [`bullet-inverse-dynamics`] as [`inverse_dynamics`] — tree-structured
//!   multibody inverse dynamics as an independent leaf crate.
//!
//! ## License
//!
//! zlib (same as upstream Bullet Physics).

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

pub use bullet_collision as collision;
pub use bullet_dynamics as dynamics;
pub use bullet_math as math;
pub use bullet_collision::raycast;
pub use bullet_collision::shapes;

#[cfg(feature = "inverse-dynamics")]
pub use bullet_inverse_dynamics as inverse_dynamics;

/// Convenience re-exports for common imports.
pub mod prelude {
    pub use bullet_collision::raycast::RayHit;
    pub use bullet_collision::shapes::CollisionShape;
    pub use bullet_dynamics::{
        create_rope, BodyHandle, CollisionEvent, CollisionFlags, CollisionObject, ContactForceEvent,
        ContactPointEvent, ConstraintTag, DebugDrawFlags, DebugPrim, KinematicCharacterController,
        MotionState, MultiBody, MultiBodyConstraint, MultiBodyHandle, PhysicsWorld, RaycastVehicle,
        RigidBody, SoftBody, SoftBodyHandle, SolverInfo, StepStage, StepTimings, VehicleTuning,
        STEP_STAGES,
    };
    pub use bullet_dynamics::body::filter_groups;
    pub use bullet_math::{Quaternion, Real, Vector3};
}
