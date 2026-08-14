//! Featherstone articulated-body dynamics — port of
//! `BulletDynamics/Featherstone`.
//!
//! This is a self-contained subsystem that does **not** touch the rigid-body
//! [`crate::world::PhysicsWorld`]: a [`MultiBody`] is a tree of links
//! (revolute/prismatic/spherical/planar/fixed joints) with an optional fixed
//! base, integrated with Featherstone's articulated-body algorithm over
//! spatial algebra ([`bullet_math::spatial`]); [`MultiBodyConstraint`]s
//! (point-to-point, fixed, slider, joint motor, gear) are enforced by the
//! multibody sequential-impulse solver in [`crate::multibody::world`].
//!
//! Reference classes: `btMultiBody`, `btMultiBodyConstraint`,
//! `btMultiBodyConstraintSolver`, `btMultiBodyDynamicsWorld`.

pub mod body;
pub mod constraint;
pub mod link;
pub mod world;

pub use body::{MultiBody, MultiBodyScratch};
pub use constraint::{
    MultiBodyConstraint, MultiBodyConstraintKind, MultiBodyFixedConstraint, MultiBodyGearConstraint,
    MultiBodyJointMotor, MultiBodyPoint2Point, MultiBodySliderConstraint,
};
pub use link::{JointType, MultiBodyLink};
pub use world::MultiBodyWorld;
