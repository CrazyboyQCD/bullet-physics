//! Plane-vs-convex narrow phase.
//!
//! The plane is the reference shape; the deepest point of the convex shape
//! along the plane normal yields a single contact point. The contact normal
//! points from the plane to the convex shape.

use crate::manifold::{ContactManifold, ContactPoint};
use crate::shapes::{CollisionShape, Plane};
use bullet_math::{Quaternion, Vector3};

/// Contact between a static plane and a convex shape.
pub fn plane_convex(
    plane: &Plane,
    plane_position: Vector3,
    plane_orientation: Quaternion,
    convex: &CollisionShape,
    convex_position: Vector3,
    convex_orientation: Quaternion,
) -> Option<ContactManifold> {
    // World-space plane: `n . x = c` (rotated local normal, shifted constant).
    let n = plane_orientation.mul_vec3(plane.normal);
    let c = plane.constant + n.dot(plane_position);

    // Orient the contact normal from the plane toward the convex shape.
    let normal = if n.dot(convex_position - plane_position) >= 0.0 {
        n
    } else {
        -n
    };
    let c = if normal == n { c } else { -c };

    // Deepest point of the convex shape along -normal.
    let deepest = convex.support_world(convex_position, convex_orientation, -normal);
    let depth = c - normal.dot(deepest);
    if depth < 0.0 {
        return None;
    }
    Some(ContactManifold::from_points(vec![ContactPoint::new(
        deepest,
        normal,
        depth,
    )]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sphere_resting_on_plane() {
        let plane = Plane::new(Vector3::Y, 0.0);
        let sphere = CollisionShape::sphere(0.5);
        // Sphere center 0.4 above the plane: penetration 0.1.
        let m = plane_convex(
            &plane,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &sphere,
            Vector3::new(0.0, 0.4, 0.0),
            Quaternion::IDENTITY,
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-6);
        // Contact sits at the sphere's deepest point, 0.1 below the plane.
        assert_relative_eq!(p.position.y, -0.1, epsilon = 1e-6);
    }

    #[test]
    fn separated_sphere_gives_no_contact() {
        let plane = Plane::new(Vector3::Y, 0.0);
        let sphere = CollisionShape::sphere(0.5);
        assert!(plane_convex(
            &plane,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &sphere,
            Vector3::new(0.0, 2.0, 0.0),
            Quaternion::IDENTITY,
        )
        .is_none());
    }

    #[test]
    fn sphere_below_plane_flips_the_normal() {
        let plane = Plane::new(Vector3::Y, 0.0);
        let sphere = CollisionShape::sphere(0.5);
        // Sphere center 0.3 below the plane: normal points from the plane
        // toward the sphere, i.e. down.
        let m = plane_convex(
            &plane,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &sphere,
            Vector3::new(0.0, -0.3, 0.0),
            Quaternion::IDENTITY,
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, -1.0);
        assert_relative_eq!(p.penetration, 0.2, epsilon = 1e-6);
    }

    #[test]
    fn rotated_plane_uses_the_world_normal() {
        // Plane rotated 90 deg about Z: its normal is now +X.
        let plane = Plane::new(Vector3::Y, 0.0);
        let sphere = CollisionShape::sphere(0.5);
        let q = Quaternion::from_rotation_z(-bullet_math::FRAC_PI_2);
        let m = plane_convex(
            &plane,
            Vector3::ZERO,
            q,
            &sphere,
            Vector3::new(0.4, 0.0, 0.0),
            Quaternion::IDENTITY,
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.x, 1.0);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-6);
    }
}
