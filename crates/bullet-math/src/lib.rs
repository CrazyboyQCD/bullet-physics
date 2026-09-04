//! Linear math layer, the Rust equivalent of Bullet's `LinearMath`.
//!
//! Type aliases map to [`glam`]'s SIMD-accelerated types, switching between
//! `f32` and `f64` based on the selected precision feature (`f32` by default,
//! `f64` when the `f64` feature is enabled).
//!
//! `glam` enables SIMD automatically through target features (SSE2/AVX/NEON),
//! so no extra feature flag is required. `Vector3`/`Matrix3` map to glam's
//! default `Vec3`/`Mat3` (SIMD-accelerated, matching Bullet's `btVector3`
//! semantics); `Transform` maps to the aligned `Affine3A`, whose public API
//! accepts and returns `Vec3`/`Mat3`-compatible types.
//!
//! Scalar transcendentals are dispatched through [`math`], which routes to the
//! standard library (default `std` feature) or to the pure-Rust `libm` crate
//! (`libm` feature, for no_std targets).

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!(
    "select a scalar math backend: the \"std\" feature (default) or \"libm\" for no_std builds"
);

/// Scalar precision, chosen at compile time via cargo features.
#[cfg(not(feature = "f64"))]
pub type Real = f32;
#[cfg(feature = "f64")]
pub type Real = f64;

/// 3D vector (SIMD). Equivalent to `btVector3`.
#[cfg(not(feature = "f64"))]
pub use glam::Vec3A as Vector3;
#[cfg(feature = "f64")]
pub use glam::DVec3 as Vector3;

/// Quaternion. Equivalent to `btQuaternion`.
#[cfg(not(feature = "f64"))]
pub use glam::Quat as Quaternion;
#[cfg(feature = "f64")]
pub use glam::DQuat as Quaternion;

/// 3x3 matrix. Equivalent to `btMatrix3x3`.
#[cfg(not(feature = "f64"))]
pub use glam::Mat3A as Matrix3;
#[cfg(feature = "f64")]
pub use glam::DMat3 as Matrix3;

/// Rigid transform (rotation + translation). Equivalent to `btTransform`.
#[cfg(not(feature = "f64"))]
pub use glam::Affine3A as Transform;
#[cfg(feature = "f64")]
pub use glam::DAffine3 as Transform;

pub mod compat;
pub mod math;

/// `pi` at the active precision.
#[cfg(not(feature = "f64"))]
pub const PI: f32 = core::f32::consts::PI;
#[cfg(feature = "f64")]
pub const PI: f64 = core::f64::consts::PI;

/// `pi / 2` at the active precision.
#[cfg(not(feature = "f64"))]
pub const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
#[cfg(feature = "f64")]
pub const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;

/// `1 / sqrt(2)` at the active precision.
#[cfg(not(feature = "f64"))]
pub const FRAC_1_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;
#[cfg(feature = "f64")]
pub const FRAC_1_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;

mod transform;
pub mod spatial;
mod util;

pub use transform::make_transform;
pub use spatial::{
    outer_product, skew, spatial_transform, symmetric_spatial_outer_product,
    symmetric_spatial_outer_product_force, SpatialForceVector, SpatialMotionVector,
    SpatialTransformationMatrix, SymmetricSpatialDyad,
};
pub use util::Aabb;
pub use util::inertia_tensor_world;
