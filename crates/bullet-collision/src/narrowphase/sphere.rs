use crate::manifold::{ContactManifold, ContactPoint};
use bullet_math::{Quaternion, Vector3};
use crate::shapes::{BoxShape, Sphere};

/// Sphere-sphere contact.
pub fn sphere_sphere(
    sa: &Sphere,
    pa: Vector3,
    sb: &Sphere,
    pb: Vector3,
) -> Option<ContactManifold> {
    let delta = pb - pa;
    let dist_sq = delta.length_squared();
    let radius_sum = sa.radius + sb.radius;
    if dist_sq >= radius_sum * radius_sum {
        return None;
    }
    let dist = dist_sq.sqrt();
    // Degenerate case (coincident centers): pick an arbitrary normal.
    let normal = if dist > 1e-12 { delta / dist } else { Vector3::Y };
    let penetration = radius_sum - dist;
    let position = pa + normal * (sa.radius - penetration * 0.5);
    Some(ContactManifold::from_points(vec![ContactPoint::new(
        position,
        normal,
        penetration,
    )]))
}

/// Sphere-box contact. `normal` points from the box to the sphere.
pub fn sphere_box(
    s: &Sphere,
    ps: Vector3,
    b: &BoxShape,
    pb: Vector3,
    ob: Quaternion,
) -> Option<ContactManifold> {
    // Sphere center in box-local space.
    let local = ob.inverse().mul_vec3(ps - pb);
    let h = b.half_extents;
    let clamped = local.clamp(-h, h);
    let delta = local - clamped;
    let dist_sq = delta.length_squared();

    if dist_sq >= s.radius * s.radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    let (normal_local, penetration) = if dist > 1e-12 {
        // Sphere center outside the box.
        (delta / dist, s.radius - dist)
    } else {
        // Sphere center inside the box: push out along the closest face.
        let dx = h.x - local.x.abs();
        let dy = h.y - local.y.abs();
        let dz = h.z - local.z.abs();
        let (n, d) = if dx <= dy && dx <= dz {
            (Vector3::new(local.x.signum(), 0.0, 0.0), dx)
        } else if dy <= dz {
            (Vector3::new(0.0, local.y.signum(), 0.0), dy)
        } else {
            (Vector3::new(0.0, 0.0, local.z.signum()), dz)
        };
        (n, s.radius + d)
    };

    let normal = ob.mul_vec3(normal_local);
    let position = ob.mul_vec3(clamped) + pb;
    Some(ContactManifold::from_points(vec![ContactPoint::new(
        position, normal, penetration,
    )]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spheres_overlap() {
        let a = Sphere::new(1.0);
        let b = Sphere::new(1.0);
        let m = sphere_sphere(&a, Vector3::ZERO, &b, Vector3::new(1.5, 0.0, 0.0)).unwrap();
        assert_relative_eq!(m.points[0].penetration, 0.5);
        assert_relative_eq!(m.points[0].normal.x, 1.0);
    }

    #[test]
    fn spheres_separated() {
        let a = Sphere::new(1.0);
        let b = Sphere::new(1.0);
        assert!(sphere_sphere(&a, Vector3::ZERO, &b, Vector3::new(3.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn sphere_box_contact() {
        let s = Sphere::new(0.5);
        let b = BoxShape::new(Vector3::splat(1.0));
        let m = sphere_box(&s, Vector3::new(1.4, 0.0, 0.0), &b, Vector3::ZERO, Quaternion::IDENTITY)
            .unwrap();
        assert_relative_eq!(m.points[0].penetration, 0.1);
        assert_relative_eq!(m.points[0].normal.x, 1.0);
    }
}
