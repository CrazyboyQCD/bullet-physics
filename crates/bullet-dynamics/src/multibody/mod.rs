//! Featherstone articulated-body dynamics — port of
//! `BulletDynamics/Featherstone`.
//!
//! A [`MultiBody`] is a tree of links (revolute/prismatic/spherical/planar/
//! fixed joints) with an optional fixed base, integrated with Featherstone's
//! articulated-body algorithm over spatial algebra ([`bullet_math::spatial`]);
//! [`MultiBodyConstraint`]s (point-to-point, fixed, slider, joint motor,
//! gear, joint limits) are enforced by the multibody rows solved inside
//! [`crate::world::PhysicsWorld`] (the fused world of phase A; see
//! `docs/multibody-collision.md`).
//!
//! Reference classes: `btMultiBody`, `btMultiBodyConstraint`,
//! `btMultiBodyConstraintSolver`, `btMultiBodyDynamicsWorld`.

pub mod body;
pub mod constraint;
pub mod link;
#[cfg(test)]
mod world_tests;

pub use body::{LinkCollider, MultiBody, MultiBodyScratch};
pub use constraint::{
    MultiBodyConstraint, MultiBodyConstraintKind, MultiBodyFixedConstraint, MultiBodyGearConstraint,
    MultiBodyJointLimit, MultiBodyJointMotor, MultiBodyPoint2Point, MultiBodySliderConstraint,
    MultiBodySphericalJointLimit, MultiBodySphericalJointMotor,
};
pub use link::{JointType, MultiBodyLink};
