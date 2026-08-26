//! Generic convex-convex contact generation via GJK + EPA.
//!
//! The GJK simplex that encloses the origin of the Minkowski difference seeds
//! an EPA pass that yields the penetration normal and depth; the contact point
//! is the midpoint of the two support witnesses along the normal.

use crate::manifold::{ContactManifold, ContactPoint};
use crate::narrowphase::{epa, gjk};
use crate::shapes::CollisionShape;
use bullet_math::{Quaternion, Vector3};

/// Contact between two convex shapes, using the generic GJK + EPA pipeline.
/// Returns `None` when the shapes are separated (or the overlap is numerical
/// noise below the EPA threshold).
pub fn convex_convex(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
) -> Option<ContactManifold> {
    let simplex = gjk::gjk_intersect(
        shape_a,
        position_a,
        orientation_a,
        shape_b,
        position_b,
        orientation_b,
    )?;
    // GJK only reports success once the origin is enclosed, which needs a full
    // tetrahedron (four vertices).
    if simplex.points().len() < 4 {
        return None;
    }
    let mut seed = [Vector3::ZERO; 4];
    seed.copy_from_slice(&simplex.points()[..4]);

    let support = |dir: Vector3| {
        gjk::support_minkowski(
            shape_a,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
            dir,
        )
    };
    // `normal` points from A to B (EPA convention); `depth` is the overlap.
    let (normal, depth) = epa::penetration(&seed, support);
    if depth <= 0.0 {
        return None;
    }

    // Contact position: the witness point on B, `pointOnB = supportA(normal) -
    // normal * depth`, matching `btGjkPairDetector::getClosestPoints`
    // (`btGjkEpa2::Penetration`'s `witnesses[1]` under our A->B normal
    // convention). The previous midpoint-of-supports formula used B's *global*
    // support along -normal, which for a large shape (e.g. a mesh triangle)
    // lies far from the overlap; the resulting contact position was garbage,
    // wrecking the solver's torque arms (found by the `mesh` parity scene:
    // bodies neither tumbled nor rolled downhill like Bullet's).
    let point_a = shape_a.support_world(position_a, orientation_a, normal);
    let position = point_a - normal * depth;
    Some(ContactManifold::from_points(vec![ContactPoint::new(
        position,
        normal,
        depth,
    )]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_math::Real;
    use approx::assert_relative_eq;

    fn contact(
        a: &CollisionShape,
        pa: Vector3,
        b: &CollisionShape,
        pb: Vector3,
    ) -> Option<(Vector3, Real)> {
        convex_convex(a, pa, Quaternion::IDENTITY, b, pb, Quaternion::IDENTITY)
            .map(|m| {
                let p = m.points[0];
                (p.normal, p.penetration)
            })
    }

    #[test]
    fn probe_debug() {
        use bullet_math::Real as R;
        let _ = R::MAX;
        let a = CollisionShape::capsule(0.5, 1.0);
        let b = CollisionShape::sphere(0.5);
        match contact(&a, Vector3::ZERO, &a, Vector3::new(0.7, 0.0, 0.0)) {
            Some((n, d)) => println!("RES capsule-capsule n={n:?} d={d}"),
            None => println!("RES capsule-capsule None"),
        }
        let hull = CollisionShape::convex_hull(vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ]);
        for pb in [Vector3::new(1.25, 0.0, 0.0), Vector3::new(1.25, 0.3, 0.0)] {
            match contact(&hull, Vector3::ZERO, &b, pb) {
                Some((n, d)) => println!("RES hull-sphere pb={pb:?} n={n:?} d={d}"),
                None => println!("RES hull-sphere pb={pb:?} None"),
            }
        }
    }

    #[test]
    fn capsules_along_x_overlap() {
        let a = CollisionShape::capsule(0.5, 1.0);
        let b = CollisionShape::capsule(0.5, 1.0);
        // Both capsules stand on Y; overlap along X by 0.3.
        let (n, depth) = contact(&a, Vector3::ZERO, &b, Vector3::new(0.7, 0.0, 0.0)).unwrap();
        assert_relative_eq!(n.x, 1.0, epsilon = 1e-4);
        assert_relative_eq!(depth, 0.3, epsilon = 1e-3);
    }

    #[test]
    fn separated_capsules_do_not_touch() {
        let a = CollisionShape::capsule(0.5, 1.0);
        let b = CollisionShape::capsule(0.5, 1.0);
        assert!(contact(&a, Vector3::ZERO, &b, Vector3::new(2.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn cylinder_sphere_overlap() {
        let a = CollisionShape::cylinder(1.0, 1.0);
        let b = CollisionShape::sphere(0.5);
        // Sphere at the rim, overlapping along +X by 0.25; normal from the
        // cylinder (A) to the sphere (B) points along +X.
        let (n, depth) = contact(&a, Vector3::ZERO, &b, Vector3::new(1.25, 0.0, 0.0)).unwrap();
        assert_relative_eq!(n.x, 1.0, epsilon = 1e-4);
        assert_relative_eq!(depth, 0.25, epsilon = 1e-3);
    }

    #[test]
    fn cone_box_overlap_along_y() {
        let a = CollisionShape::cone(1.0, 1.0);
        let b = CollisionShape::box_shape(Vector3::splat(0.5));
        // Box sinks into the cone apex by 0.2 (apex at y = +1); the normal
        // points from the cone (A) to the box (B), i.e. along +Y.
        let (n, depth) = contact(&a, Vector3::ZERO, &b, Vector3::new(0.0, 1.3, 0.0)).unwrap();
        assert_relative_eq!(n.y, 1.0, epsilon = 1e-4);
        assert_relative_eq!(depth, 0.2, epsilon = 1e-3);
    }

    #[test]
    fn hull_hull_overlap() {
        let a = CollisionShape::convex_hull(vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ]);
        let b = CollisionShape::sphere(0.5);
        // Sphere overlaps the octahedron vertex region along +X by 0.25; the
        // normal points from the hull (A) to the sphere (B), i.e. along +X.
        let (n, depth) = contact(&a, Vector3::ZERO, &b, Vector3::new(1.25, 0.0, 0.0)).unwrap();
        assert_relative_eq!(n.x, 1.0, epsilon = 1e-3);
        assert_relative_eq!(depth, 0.25, epsilon = 1e-3);
    }
}
