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

/// Scalar precision, chosen at compile time via cargo features.
#[cfg(not(feature = "f64"))]
pub type Real = f32;
#[cfg(feature = "f64")]
pub type Real = f64;

/// 3D vector (SIMD). Equivalent to `btVector3`.
#[cfg(not(feature = "f64"))]
pub use glam::Vec3 as Vector3;
#[cfg(feature = "f64")]
pub use glam::DVec3 as Vector3;

/// Quaternion. Equivalent to `btQuaternion`.
#[cfg(not(feature = "f64"))]
pub use glam::Quat as Quaternion;
#[cfg(feature = "f64")]
pub use glam::DQuat as Quaternion;

/// 3x3 matrix. Equivalent to `btMatrix3x3`.
#[cfg(not(feature = "f64"))]
pub use glam::Mat3 as Matrix3;
#[cfg(feature = "f64")]
pub use glam::DMat3 as Matrix3;

/// Rigid transform (rotation + translation). Equivalent to `btTransform`.
#[cfg(not(feature = "f64"))]
pub use glam::Affine3A as Transform;
#[cfg(feature = "f64")]
pub use glam::DAffine3 as Transform;

/// `pi` at the active precision.
#[cfg(not(feature = "f64"))]
pub const PI: f32 = std::f32::consts::PI;
#[cfg(feature = "f64")]
pub const PI: f64 = std::f64::consts::PI;

/// `pi / 2` at the active precision.
#[cfg(not(feature = "f64"))]
pub const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
#[cfg(feature = "f64")]
pub const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// `1 / sqrt(2)` at the active precision.
#[cfg(not(feature = "f64"))]
pub const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
#[cfg(feature = "f64")]
pub const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

mod transform;
mod util;

pub use transform::make_transform;
pub use util::Aabb;
pub use util::inertia_tensor_world;
