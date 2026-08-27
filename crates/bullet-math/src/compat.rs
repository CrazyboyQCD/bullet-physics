//! Precision-agnostic wrappers over glam constructors whose signatures differ
//! between the scalar-argument f32 types and the f64 types (e.g. `Quat::
//! from_axis_angle` takes a scalar `Vec3` while `DQuat::from_axis_angle`
//! takes `DVec3` directly). Centralizing the casts keeps call sites free of
//! `#[cfg]` and survives glam upgrades unchanged.

use super::{Matrix3, Quaternion, Real, Transform, Vector3};

/// Rotation about `axis` by `angle` radians (`Quat::from_axis_angle`).
#[inline]
pub fn quat_from_axis_angle(axis: Vector3, angle: Real) -> Quaternion {
    #[cfg(not(feature = "f64"))]
    {
        Quaternion::from_axis_angle(axis.into(), angle)
    }
    #[cfg(feature = "f64")]
    {
        Quaternion::from_axis_angle(axis, angle)
    }
}

/// Rotation from a scaled-axis vector (`Quat::from_scaled_axis`).
#[inline]
pub fn quat_from_scaled_axis(v: Vector3) -> Quaternion {
    #[cfg(not(feature = "f64"))]
    {
        Quaternion::from_scaled_axis(v.into())
    }
    #[cfg(feature = "f64")]
    {
        Quaternion::from_scaled_axis(v)
    }
}

/// Shortest rotation taking vector `from` to vector `to`
/// (`Quat::from_rotation_arc`).
#[inline]
pub fn quat_from_rotation_arc(from: Vector3, to: Vector3) -> Quaternion {
    #[cfg(not(feature = "f64"))]
    {
        Quaternion::from_rotation_arc(from.into(), to.into())
    }
    #[cfg(feature = "f64")]
    {
        Quaternion::from_rotation_arc(from, to)
    }
}

/// Quaternion from a rotation matrix (`Quat::from_mat3a` on f32,
/// `DQuat::from_mat3` on f64).
#[inline]
pub fn quat_from_mat3(mat: &Matrix3) -> Quaternion {
    #[cfg(not(feature = "f64"))]
    {
        Quaternion::from_mat3a(mat)
    }
    #[cfg(feature = "f64")]
    {
        Quaternion::from_mat3(mat)
    }
}

/// Diagonal matrix (`Mat3A::from_diagonal` takes a scalar `Vec3` on f32).
#[inline]
pub fn matrix3_from_diagonal(diagonal: Vector3) -> Matrix3 {
    #[cfg(not(feature = "f64"))]
    {
        Matrix3::from_diagonal(diagonal.into())
    }
    #[cfg(feature = "f64")]
    {
        Matrix3::from_diagonal(diagonal)
    }
}

/// Quaternion imaginary part (`Quat::xyz` returns a scalar `Vec3` on f32).
#[inline]
pub fn quat_xyz(q: Quaternion) -> Vector3 {
    #[cfg(not(feature = "f64"))]
    {
        q.xyz().into()
    }
    #[cfg(feature = "f64")]
    {
        q.xyz()
    }
}

/// Transform from a rotation matrix and translation
/// (`Affine3A::from_mat3_translation` takes scalar `Mat3`/`Vec3` on f32).
#[inline]
pub fn transform_from_mat3_translation(mat3: Matrix3, translation: Vector3) -> Transform {
    #[cfg(not(feature = "f64"))]
    {
        Transform::from_mat3_translation(mat3.into(), translation.into())
    }
    #[cfg(feature = "f64")]
    {
        Transform::from_mat3_translation(mat3, translation)
    }
}

/// Transform from an orientation and translation (f32 takes a scalar `Vec3`).
#[inline]
pub fn transform_from_rotation_translation(
    rotation: Quaternion,
    translation: Vector3,
) -> Transform {
    #[cfg(not(feature = "f64"))]
    {
        Transform::from_rotation_translation(rotation, translation.into())
    }
    #[cfg(feature = "f64")]
    {
        Transform::from_rotation_translation(rotation, translation)
    }
}
