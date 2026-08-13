use super::{Quaternion, Transform, Vector3};

/// Compose a rigid transform from a translation and an orientation.
///
/// Equivalent to `btTransform(rotation, origin)`.
#[inline]
pub fn make_transform(translation: Vector3, orientation: Quaternion) -> Transform {
    Transform::from_rotation_translation(orientation, translation)
}
