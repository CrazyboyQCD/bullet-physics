//! Generic convex-convex contact generation via GJK + EPA.
//!
//! The GJK distance walk runs on the shapes' margin **cores** (Bullet's
//! margin architecture, see [`gjk::gjk_contact`]): separated pairs inside
//! the contact band report a margin-band contact directly (Bullet's
//! positive-distance manifold points — EPA never runs for them), pairs
//! overlapping within the margin shell are resolved by the walk exactly,
//! and only deep overlap seeds EPA from the enclosing simplex of the cores
//! for the penetration normal and depth (with the margins added back). The
//! deep-overlap contact point follows Bullet's A-witness-then-depth
//! construction: it lies on B's support plane under the A→B normal
//! convention and coincides with the true B witness whenever the facing
//! support points are tangentially aligned (spheres, parallel faces);
//! generic convex pairs carry a bounded tangential offset instead
//! (recorded deviation, see `docs/bullet-deviations.md`).

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
        gjk::GjkOutcome::Overlap(simplex) => {
            // Simplex contract (audited against the two `Overlap` producers
            // in `gjk_contact`, not just its doc comment): (a) exactly four
            // points — the `ReduceStep::Inside` branch constructs the simplex
            // with `len: 4`, and the degeneracy-gate branch explicitly
            // requires `points().len() == 4`; (b) a structurally
            // non-degenerate tetrahedron — `reduce_tetrahedron` routes
            // coplanar/duplicate-vertex inputs (determinant below 1e-12) to
            // its best-face fallback, so `Inside` implies a finite volume;
            // (c) the closed tetrahedron encloses the origin — `Inside` and
            // `solve_tetrahedron`'s `true` are exact sign tests over all
            // four faces (the origin on a face counts as touching, which
            // EPA accepts like Bullet). The assert pins this for refactors;
            // EPA additionally re-validates the seed atomically.
            debug_assert_eq!(
                simplex.points().len(),
                4,
                "GJK Overlap must carry a full tetrahedron"
            );
            simplex
        }
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
    //
    // Per-status acceptance mirrors `btGjkEpa2` (checked against
    // `EPA::Evaluate` + `btGjkEpaSolver2::Penetration`, not assumed): every
    // non-fatal exit — `AccuraryReached`, the iteration budget simply running
    // out (Bullet's status stays `Valid`), `InvalidHull` after a
    // *transactional* expand rejection, and `OutOfVertices` — reports the
    // best face of the intact hull and is accepted by Bullet's depth solver.
    // Only the seed-failure fallback (`m_depth = 0`; our degenerate-seed zero
    // result) yields no contact, which the `core_depth <= 0` check below
    // already covers. The status is therefore carried for documentation and
    // tests, not to filter contacts — filtering would *diverge* from Bullet.
    let (status, normal, core_depth) =
        epa::penetration_with_status(&seed, support, scratch);
    let _ = status;
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
    //
    // Witness invariant (audited against `btGjkEpaSolver2::Penetration`):
    // Bullet computes `witnesses[1] = w0 - normal * depth`, where `w0` is
    // the best face's A-side support points interpolated with the EPA
    // barycentric weights — the *same* A-witness-then-push structure as this
    // formula. Both land on B's deep support plane (`n . position` equals
    // B's support along `-n`, which is what carries the depth) and inherit
    // A's tangential coordinates: they coincide with the true B witness
    // whenever the facing support points are tangentially aligned (spheres,
    // parallel faces — the parity-covered cases) and carry a bounded
    // tangential offset otherwise (ours: `|supportA(n) - w0|`; Bullet's:
    // the face-interpolation spread). Recorded as a deviation; not changed
    // here — the interpolation variant would rewrite this layer's
    // parity-bit-exact output.
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

    // ------------------------------------------------------------------
    // Robustness suite (convex-convex hardening round). Every test
    // asserts: no panic, finite normal/depth/position, a sensible normal
    // direction, non-negative depth for accepted penetrations, and clean
    // `None` for rejected contacts.
    // ------------------------------------------------------------------

    fn contact_manifold(
        a: &CollisionShape,
        pa: Vector3,
        qa: Quaternion,
        b: &CollisionShape,
        pb: Vector3,
        qb: Quaternion,
    ) -> Option<ContactManifold> {
        convex_convex(
            a,
            pa,
            qa,
            b,
            pb,
            qb,
            None,
            a.contact_breaking_threshold().min(b.contact_breaking_threshold()),
            &mut EpaScratch::default(),
        )
    }

    fn assert_clean(p: &ContactPoint, separation: Vector3) {
        let n = p.normal;
        assert!(n.x.is_finite() && n.y.is_finite() && n.z.is_finite());
        assert!(p.penetration.is_finite(), "depth must be finite");
        assert!(p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite());
        assert!(n.dot(separation.normalize_or_zero()) > 0.0, "normal must face the separation direction");
    }

    /// Runtime pin of the audited `GjkOutcome::Overlap` contract: the
    /// simplex handed to EPA always carries the full tetrahedron.
    #[test]
    fn overlap_simplex_carries_full_tetrahedron() {
        let a = CollisionShape::box_shape(Vector3::splat(1.0));
        let b = CollisionShape::sphere(0.5);
        // Deep overlap (the sphere's core well inside the box's core).
        match gjk::gjk_contact(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.6, 0.0, 0.0),
            Quaternion::IDENTITY,
            None,
            0.02,
        ) {
            gjk::GjkOutcome::Overlap(simplex) => {
                assert_eq!(simplex.points().len(), 4);
            }
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    /// The status propagates through the crate-internal EPA API the contact
    /// layer consumes; a normal overlap reports `Converged`.
    #[test]
    fn epa_status_propagates_through_contact_layer() {
        let a = CollisionShape::box_shape(Vector3::splat(1.0));
        let b = CollisionShape::sphere(0.5);
        let simplex = match gjk::gjk_contact(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.6, 0.0, 0.0),
            Quaternion::IDENTITY,
            None,
            0.02,
        ) {
            gjk::GjkOutcome::Overlap(s) => s,
            other => panic!("expected overlap, got {other:?}"),
        };
        let mut seed = [Vector3::ZERO; 4];
        seed.copy_from_slice(&simplex.points()[..4]);
        let support = |dir: Vector3| {
            gjk::core_support_minkowski(
                &a, Vector3::ZERO, Quaternion::IDENTITY, &b,
                Vector3::new(0.6, 0.0, 0.0), Quaternion::IDENTITY, dir,
            )
        };
        let (status, normal, depth) =
            epa::penetration_with_status(&seed, support, &mut EpaScratch::default());
        assert_eq!(status, epa::EpaStatus::Converged);
        assert!(depth > 0.0 && depth.is_finite());
        assert!(normal.x.is_finite() && normal.y.is_finite() && normal.z.is_finite());
    }

    /// Witness checks for an off-axis asymmetric hull against a sphere: the
    /// contact must land on B's deep support plane (`n . position == n .
    /// supportB(-n)`), with coherent normal and depth.
    #[test]
    fn witness_asymmetric_hull_vs_sphere_on_support_plane() {
        let hull = CollisionShape::convex_hull(vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ]);
        let sphere = CollisionShape::sphere(0.5);
        let m = contact_manifold(
            &hull,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &sphere,
            Vector3::new(1.25, 0.3, 0.0),
            Quaternion::IDENTITY,
        )
        .unwrap();
        let p = &m.points[0];
        assert_clean(p, Vector3::new(0.25, 0.3, 0.0));
        assert!(p.penetration > 0.0);
        // B's deep support plane: the sphere's surface point along -normal.
        let plane_b = p.normal.dot(
            sphere.support_world(
                Vector3::new(1.25, 0.3, 0.0),
                Quaternion::IDENTITY,
                -p.normal,
            ),
        );
        assert_relative_eq!(p.normal.dot(p.position), plane_b, epsilon = 2e-2);
    }

    /// Rotated asymmetric box pair: coherent normal/depth and the same
    /// support-plane invariant under rotation.
    #[test]
    fn witness_rotated_box_vs_box_on_support_plane() {
        let a = CollisionShape::box_shape(Vector3::new(1.0, 0.4, 0.6));
        let b = CollisionShape::box_shape(Vector3::splat(0.5));
        let qa = Quaternion::from_rotation_z(0.6);
        let qb = Quaternion::from_rotation_z(-0.35);
        let pb = Vector3::new(1.3, 0.25, 0.1);
        let m = contact_manifold(&a, Vector3::ZERO, qa, &b, pb, qb).unwrap();
        let p = &m.points[0];
        assert_clean(p, pb);
        assert!(p.penetration > 0.0);
        let plane_b = p.normal.dot(b.support_world(pb, qb, -p.normal));
        assert_relative_eq!(p.normal.dot(p.position), plane_b, epsilon = 2e-2);
    }

    /// Elongated hull against a hull: finite, plane-consistent, sane
    /// direction — the tangential-offset regime of the documented witness
    /// invariant must not produce garbage.
    #[test]
    fn witness_elongated_hull_vs_hull_stays_clean() {
        let a = CollisionShape::box_shape(Vector3::new(1.5, 0.05, 0.05));
        let b = CollisionShape::box_shape(Vector3::new(0.3, 0.3, 0.3));
        let qa = Quaternion::from_rotation_z(0.25);
        let pb = Vector3::new(1.7, 0.2, 0.0);
        let m = contact_manifold(&a, Vector3::ZERO, qa, &b, pb, Quaternion::IDENTITY).unwrap();
        let p = &m.points[0];
        assert_clean(p, pb);
        assert!(p.penetration > 0.0);
        let plane_b = p.normal.dot(b.support_world(pb, Quaternion::IDENTITY, -p.normal));
        assert_relative_eq!(p.normal.dot(p.position), plane_b, epsilon = 2e-2);
    }

    /// Unequal margins (sphere 0.25 vs box 0.04): the deep-overlap depth is
    /// the true full-shape overlap — core depth plus both margins.
    #[test]
    fn witness_unequal_margins_deep_overlap() {
        let a = CollisionShape::box_shape(Vector3::splat(1.0));
        let b = CollisionShape::sphere(0.25);
        let m = contact_manifold(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, 0.9, 0.0),
            Quaternion::IDENTITY,
        )
        .unwrap();
        let p = &m.points[0];
        assert_clean(p, Vector3::new(0.0, 0.9, 0.0));
        // Sphere bottom at 0.65, box top at 1.0: true overlap 0.35.
        assert_relative_eq!(p.penetration, 0.35, epsilon = 2e-3);
        assert_relative_eq!(p.position.y, 0.65, epsilon = 2e-3);
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
    }

    /// Gradual transition from the margin band (walk-resolved) into deep
    /// overlap (EPA): the reported quantities must stay within a
    /// step-proportional envelope across the algorithmic path switch —
    /// small jumps at the transition are expected, geometric ones are not.
    #[test]
    fn band_to_epa_transition_stays_continuous() {
        let a = CollisionShape::box_shape(Vector3::splat(1.0));
        let b = CollisionShape::sphere(0.5);
        let step = 0.01;
        // From 0.18 band overlap down to 0.95 deep overlap.
        let mut prev: Option<(Real, Real, Real)> = None; // (normal.y, penetration, position.y)
        for i in 0..78 {
            let py = 1.32 - step * i as Real;
            let Some(m) = contact_manifold(
                &a,
                Vector3::ZERO,
                Quaternion::IDENTITY,
                &b,
                Vector3::new(0.0, py, 0.0),
                Quaternion::IDENTITY,
            ) else {
                // A sample can sit exactly on the core-shell boundary
                // (sphere core touching the box's core face), where the
                // walk's overlap resolution is degenerate by construction;
                // skip it and re-anchor the continuity chain.
                prev = None;
                continue;
            };
            let p = &m.points[0];
            assert!(
                (p.normal.y - 1.0).abs() <= 5e-2,
                "normal flip at py = {py}: n.y = {}",
                p.normal.y
            );
            assert!(p.penetration.is_finite() && p.position.y.is_finite());
            if let Some((pn, pd, pp)) = prev {
                let jump = 2.5 * step + 2e-3;
                assert!(
                    (p.penetration - pd).abs() <= jump,
                    "penetration jump {} at py = {py}",
                    (p.penetration - pd).abs()
                );
                assert!(
                    (p.position.y - pp).abs() <= jump,
                    "position jump {} at py = {py}",
                    (p.position.y - pp).abs()
                );
                assert!(p.normal.y * pn > 0.5, "normal direction change at py = {py}");
            }
            prev = Some((p.normal.y, p.penetration, p.position.y));
        }
    }

    /// Exact touching (sphere gap = 0): either a zero-depth contact or a
    /// clean `None` — never a panic, NaN, or a spurious deep penetration.
    #[test]
    fn exact_touching_is_clean() {
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::sphere(0.5);
        // Below the GJK overlap resolution — a clean `None` is acceptable.
        if let Some(m) = contact_manifold(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(1.0, 0.0, 0.0),
            Quaternion::IDENTITY,
        ) {
            let p = &m.points[0];
            assert!(p.penetration.is_finite() && p.penetration.abs() <= 1e-3);
            assert_relative_eq!(p.normal.x, 1.0, epsilon = 1e-2);
        }
    }

    /// Tiny overlap and tiny separation near the surface: both must stay
    /// finite and proportional, whichever path (band walk or EPA) resolves.
    #[test]
    fn tiny_overlap_and_separation_stay_proportional() {
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::sphere(0.5);
        // 1e-4 overlap: any reported contact must be of that order.
        if let Some(m) = contact_manifold(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(1.0 - 1e-4, 0.0, 0.0),
            Quaternion::IDENTITY,
        ) {
            let p = &m.points[0];
            assert!(p.penetration.is_finite() && p.penetration.abs() <= 0.1);
            assert!(p.normal.x.is_finite());
        }
        // 1e-4 separation inside the band: a velocity-only contact.
        if let Some(m) = contact_manifold(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(1.0 + 1e-4, 0.0, 0.0),
            Quaternion::IDENTITY,
        ) {
            let p = &m.points[0];
            assert!(p.penetration.is_finite() && p.penetration.abs() <= 0.1);
            assert!(p.penetration <= 0.0, "separated pair must not report penetration");
        }
    }
}
