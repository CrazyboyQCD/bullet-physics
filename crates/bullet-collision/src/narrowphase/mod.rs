//! Narrow-phase collision: exact shape-pair tests and contact generation.

use crate::manifold::ContactManifold;
use bullet_math::{Quaternion, Vector3};
use crate::shapes::CollisionShape;

mod box_box;
pub mod epa;
pub mod gjk;
mod sphere;

/// Detect collision between two convex shapes and generate a contact manifold.
///
/// `normal` of each contact point points from `shape_a` to `shape_b`.
/// Returns `None` when the shapes do not touch.
pub fn collide(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
) -> Option<ContactManifold> {
    match (shape_a, shape_b) {
        (CollisionShape::Sphere(sa), CollisionShape::Sphere(sb)) => {
            sphere::sphere_sphere(sa, position_a, sb, position_b)
        }
        (CollisionShape::Sphere(sa), CollisionShape::BoxShape(bb)) => {
            sphere::sphere_box(sa, position_a, bb, position_b, orientation_b)
        }
        (CollisionShape::BoxShape(ba), CollisionShape::Sphere(sb)) => {
            sphere::sphere_box(sb, position_b, ba, position_a, orientation_a).map(|m| m.flipped())
        }
        (CollisionShape::BoxShape(ba), CollisionShape::BoxShape(bb)) => {
            box_box::box_box(ba, position_a, orientation_a, bb, position_b, orientation_b)
        }
    }
}
