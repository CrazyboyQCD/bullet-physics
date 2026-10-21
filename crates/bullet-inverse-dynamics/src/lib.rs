//! Inverse dynamics for tree-structured multibody systems — port of Bullet's
//! `BulletInverseDynamics` (`btInverseDynamics::MultiBodyTree`).
//!
//! The library computes joint forces for given generalized state and
//! derivatives (`calculate_inverse_dynamics`), the joint-space mass matrix
//! (composite rigid body algorithm), forward kinematics at three levels, and
//! per-body 3×N jacobians, for trees built from fixed, revolute, prismatic,
//! floating (6-dof) and spherical (3-dof) joints.
//!
//! Reference implementation: `refs/bullet3/src/BulletInverseDynamics`
//! (zlib, like the rest of Bullet). Design baseline:
//! `docs/design/inverse-dynamics.md`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!(
    "select a scalar math backend: the \"std\" feature (default) or \"libm\" for no_std builds"
);

pub use bullet_math::{Matrix3, Real, Vector3};

pub mod math;
