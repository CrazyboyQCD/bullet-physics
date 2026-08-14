//! # bullet
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
//! ## License
//!
//! MIT OR Apache-2.0.

#![deny(unsafe_code)]

pub use bullet_collision as collision;
pub use bullet_dynamics as dynamics;
pub use bullet_math as math;
pub use bullet_collision::raycast;
pub use bullet_collision::shapes;

/// Convenience re-exports for common imports.
pub mod prelude {
    pub use bullet_collision::raycast::RayHit;
    pub use bullet_collision::shapes::CollisionShape;
    pub use bullet_dynamics::{KinematicCharacterController, PhysicsWorld, RigidBody};
    pub use bullet_math::{Quaternion, Real, Vector3};
}
