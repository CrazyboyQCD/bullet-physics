//! Generic convex-convex contact generation via GJK + EPA.
//!
//! The GJK distance walk runs on the shapes' margin **cores** (Bullet's
//! margin architecture, see [`gjk::gjk_contact`]): separated pairs inside
//! the contact band report a margin-band contact directly (Bullet's
//! positive-distance manifold points — EPA never runs for them), pairs
//! overlapping within the margin shell are resolved by the walk exactly,
//! and only deep overlap seeds EPA from the enclosing simplex of the cores
//! for the penetration normal and depth (with the margins added back). The
//! contact point is the witness on B in all cases.

use crate::manifold::{ContactManifold, ContactPoint};
#[cfg(not(feature = "std"))]
use alloc::vec;
use crate::narrowphase::{epa, gjk, EpaScratch};
use crate::shapes::CollisionShape;
use bullet_math::{Quaternion, Real, Vector3};

/// Contact between two convex shapes, using the generic GJK + EPA pipeline.
/// Returns `None` when the shapes are separated beyond `threshold` (or the
/// overlap is numerical noise below the EPA threshold). `seed` optionally
/// warm-starts GJK with a cached separating normal (see
/// [`gjk::gjk_intersect_seeded`]); pass `None` for an unseeded run.
#[allow(clippy::too_many_arguments)]
pub fn convex_convex(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
    seed: Option<Vector3>,
    threshold: Real,
    scratch: &mut EpaScratch,
) -> Option<ContactManifold> {
    let simplex = match gjk::gjk_contact(
        shape_a,
        position_a,
        orientation_a,
        shape_b,
        position_b,
        orientation_b,
        seed,
        threshold,
    ) {
        gjk::GjkOutcome::Separated => return None,
        gjk::GjkOutcome::Band { distance, normal, point_on_b } => {
            // Margin-band contact: `distance` is the signed true separation
            // (positive while separated, negative while the full shapes
            // overlap within the margin shell), so `penetration = -distance`
            // is negative (velocity-only) for separated pairs — exactly like
            // Bullet's positive-distance manifold points — and positive (a
            // real contact) when the walk resolved an overlap.
            return Some(ContactManifold::from_points(vec![ContactPoint::new(
                point_on_b,
                normal,
                -distance,
            )]));
        }
        gjk::GjkOutcome::Overlap(simplex) => simplex,
    };
    let mut seed = [Vector3::ZERO; 4];
    seed.copy_from_slice(&simplex.points()[..4]);

    // EPA walks the same cores the GJK walk converged on; the margins are
    // added back to its depth below (Bullet runs its Minkowski penetration
    // solver on `localGetSupportVertexWithoutMargin` shapes likewise).
    let support = |dir: Vector3| {
        gjk::core_support_minkowski(
            shape_a,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
            dir,
        )
    };
    // `normal` points from A to B (EPA convention); `core_depth` is the core
    // overlap, the true overlap adds the margin shell back.
    let (normal, core_depth) = epa::penetration(&seed, support, scratch);
    if core_depth <= 0.0 {
        return None;
    }
    let depth = core_depth + shape_a.margin() + shape_b.margin();

    // Contact position: the witness point on B, `pointOnB = supportA(normal) -
    // normal * depth`, matching `btGjkPairDetector::getClosestPoints`
    // (`btGjkEpa2::Penetration`'s `witnesses[1]` under our A->B normal
    // convention), computed against the full shapes. The previous
    // midpoint-of-supports formula used B's *global* support along -normal,
    // which for a large shape (e.g. a mesh triangle) lies far from the
    // overlap; the resulting contact position was garbage, wrecking the
    // solver's torque arms (found by the `mesh` parity scene: bodies neither
    // tumbled nor rolled downhill like Bullet's).
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
        let threshold = a
            .contact_breaking_threshold()
            .min(b.contact_breaking_threshold());
        convex_convex(
            a,
            pa,
            Quaternion::IDENTITY,
            b,
            pb,
            Quaternion::IDENTITY,
            None,
            threshold,
            &mut EpaScratch::default(),
        )
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

    /// Margin-core architecture: an overlap that lives inside the margin
    /// shell (0.54 for this box/sphere pair) is resolved by the GJK walk
    /// itself — the contact reports the exact depth, normal, and the
    /// witness on B moved out of the core by B's margin.
    #[test]
    fn margin_shell_overlap_resolved_by_walk() {
        let a = CollisionShape::box_shape(Vector3::splat(1.0));
        let b = CollisionShape::sphere(0.5);
        // Sphere bottom at 0.8, box top at 1.0: overlap 0.2.
        let m = convex_convex(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, 1.3, 0.0),
            Quaternion::IDENTITY,
            None,
            a.contact_breaking_threshold().min(b.contact_breaking_threshold()),
            &mut EpaScratch::default(),
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.2, epsilon = 1e-3);
        // Witness on B: the sphere's bottom point (center minus the radius
        // along the A->B normal).
        assert_relative_eq!(p.position.y, 0.8, epsilon = 1e-3);
    }

    /// Deep overlap past the margin shell runs EPA on the cores and adds
    /// the margins back: the depth is the true full-shape overlap and the
    /// witness is computed against the full shapes.
    #[test]
    fn deep_overlap_runs_epa_with_margins() {
        let a = CollisionShape::box_shape(Vector3::splat(1.0));
        let b = CollisionShape::sphere(0.5);
        // Sphere bottom at 0.1, box top at 1.0: overlap 0.9, well past the
        // 0.54 margin shell — the sphere's point core sits 0.36 deep inside
        // the box's shrunk core.
        let m = convex_convex(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, 0.6, 0.0),
            Quaternion::IDENTITY,
            None,
            a.contact_breaking_threshold().min(b.contact_breaking_threshold()),
            &mut EpaScratch::default(),
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.9, epsilon = 1e-3);
        // Witness on B against the full shapes: supportA(y) - y*depth =
        // (0,1,0) - 0.9*y = the sphere's bottom point.
        assert_relative_eq!(p.position.y, 0.1, epsilon = 1e-3);
    }

    /// A pair separated inside the contact band reports a velocity-only
    /// contact with negative penetration (unchanged band semantics under
    /// the margin-core walk).
    #[test]
    fn margin_shell_separation_reports_band() {
        let a = CollisionShape::capsule(0.5, 1.0);
        let b = CollisionShape::box_shape(Vector3::splat(1.0));
        // Capsule surface at x = 0.5, box face at x = 0.51: full shapes
        // separated by 0.01.
        let m = convex_convex(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(1.51, 0.0, 0.0),
            Quaternion::IDENTITY,
            None,
            a.contact_breaking_threshold().min(b.contact_breaking_threshold()),
            &mut EpaScratch::default(),
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.x, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, -0.01, epsilon = 1e-3);
    }
}
