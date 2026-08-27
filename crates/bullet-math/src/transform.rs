use super::{Quaternion, Transform, Vector3};

/// Compose a rigid transform from a translation and an orientation.
///
/// Equivalent to `btTransform(rotation, origin)`.
#[inline]
pub fn make_transform(translation: Vector3, orientation: Quaternion) -> Transform {
    // `Affine3A::from_rotation_translation` takes a scalar `Vec3`; the f64
    // path (DAffine3) takes DVec3 directly, so the cast only exists on f32.
    #[cfg(not(feature = "f64"))]
    {
        Transform::from_rotation_translation(orientation, translation.into())
    }
    #[cfg(feature = "f64")]
    {
        Transform::from_rotation_translation(orientation, translation)
    }
}
