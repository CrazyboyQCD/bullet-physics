//! Mesh-versus-mesh narrow phase (Bullet's `gimpact_vs_gimpact`).
//!
//! Both meshes keep their BVH in their own local frame, so candidate
//! triangle pairs come from a tree-vs-tree traversal: `other`'s node boxes
//! are mapped into `this`'s frame by the B-to-A local pose product (Bullet's
//! `trans1to0`). Each surviving triangle pair runs the generic GJK+EPA
//! pipeline on the prebuilt three-vertex hulls — the same per-triangle
//! strategy as the convex-vs-mesh path, and the shape of Bullet's own
//! `BULLET_TRIANGLE_COLLISION` build (its default build clips triangle
//! polygons instead; recorded deviation). The contact normal points from
//! mesh A to mesh B, matching the `collide` convention.
//!
//! Known limit of the GJK route (recorded deviation): exactly co-planar
//! triangle overlaps and line-crossing contacts have a zero-volume
//! Minkowski difference, where the EPA fallback cannot resolve a normal —
//! resting meshes live in the separated margin band (reported fine), and
//! single-sided zero-thickness plates can tunnel through each other when a
//! single step drives them deeper than the pair band; closed shells are
//! covered by their far-side triangles long before that.

use bullet_math::{Aabb, Quaternion, Real, Vector3};

use crate::manifold::{ContactManifold, ContactPoint};
use crate::narrowphase::convex_convex::convex_convex;
use crate::narrowphase::gjk;
use crate::shapes::TriangleMesh;

/// Minimum contact band for mesh-vs-mesh pairs (Bullet's GImpact triangle
/// margins sum: two `btGImpactMeshShapePart` primitives carry 0.1 each).
/// The pair threshold scaled by the shape discs can be far narrower, and a
/// zero-thickness shell stepping deeper than the band in one step would
/// skip the only contacts it ever gets — Bullet's absolute margin is what
/// keeps its GImpact pairs catchable.
const MESH_MESH_BAND: Real = 0.1;

/// Contact between two triangle meshes (either or both dynamic).
///
/// Returns `None` when no triangle pair touches within the contact band.
/// `seed` optionally warm-starts every triangle-level GJK run with a cached
/// separating normal.
#[allow(clippy::too_many_arguments)]
pub fn mesh_mesh(
    mesh_a: &TriangleMesh,
    position_a: Vector3,
    orientation_a: Quaternion,
    mesh_b: &TriangleMesh,
    position_b: Vector3,
    orientation_b: Quaternion,
    seed: Option<Vector3>,
    threshold: Real,
) -> Option<ContactManifold> {
    let threshold = threshold.max(MESH_MESH_BAND);
    // B-to-A local pose product (Bullet's `trans1to0 = trans0.inverse() *
    // trans1` in `gimpact_vs_gimpact_find_pairs`).
    let inv_a = orientation_a.inverse();
    let rotation = inv_a * orientation_b;
    let translation = inv_a * (position_b - position_a);

    let mut pairs = Vec::new();
    mesh_a.triangles_overlapping(mesh_b, rotation, translation, threshold, &mut pairs);

    let mut manifold = ContactManifold::new();
    for (ta, tb) in pairs {
        // Exact triangle-box filter before GJK (the mesh-vs-mesh counterpart
        // of `TestTriangleAgainstAabb2`): both triangle AABBs are mapped to
        // world boxes and inflated by the contact band; a miss cannot touch
        // — GJK would only confirm the separation.
        let box_a = transformed_aabb(mesh_a.triangle_aabb(ta as usize), position_a, orientation_a)
            .expanded(threshold);
        let box_b = transformed_aabb(mesh_b.triangle_aabb(tb as usize), position_b, orientation_b);
        if !box_a.overlaps(&box_b) {
            continue;
        }
        let hull_a = mesh_a.triangle_hull(ta as usize);
        let hull_b = mesh_b.triangle_hull(tb as usize);
        // Margin-shell contact, the GImpact triangle-margin semantics:
        // triangle pairs closer than the band are reported as *penetrating*
        // the shell (`penetration = band - signed_separation`, exactly
        // Bullet's clip-method `margin - distance`), so the solver's
        // positional correction can hold meshes apart like it does for real
        // shapes. A plain band contact (velocity-only) cannot stop a
        // falling zero-thickness mesh: past the surface the separation only
        // grows again, so there is nothing to recover from.
        match gjk::gjk_contact(
            hull_a,
            position_a,
            orientation_a,
            hull_b,
            position_b,
            orientation_b,
            seed,
            threshold,
        ) {
            gjk::GjkOutcome::Separated => {}
            gjk::GjkOutcome::Band {
                distance,
                normal,
                point_on_b,
            } => {
                manifold.points.push(ContactPoint::new(
                    point_on_b,
                    normal,
                    threshold - distance,
                ));
            }
            gjk::GjkOutcome::Overlap(_) => {
                // Deep overlap (past the band): the generic EPA pipeline.
                // Degenerate for zero-volume triangle pairs — see the
                // module docs.
                if let Some(m) = convex_convex(
                    hull_a,
                    position_a,
                    orientation_a,
                    hull_b,
                    position_b,
                    orientation_b,
                    seed,
                    threshold,
                ) {
                    manifold.points.extend(m.points);
                }
            }
        }
    }
    if manifold.is_empty() {
        None
    } else {
        // Many triangle pairs report one contact each; reduce to four
        // well-separated points before handing the set to the solver.
        manifold.reduce(4);
        Some(manifold)
    }
}

/// World-space wrap of a local AABB under a pose (the corner transform of
/// `Aabb::transformed`).
fn transformed_aabb(aabb: &Aabb, position: Vector3, orientation: Quaternion) -> Aabb {
    let mut min = Vector3::splat(Real::INFINITY);
    let mut max = Vector3::splat(Real::NEG_INFINITY);
    for i in 0..8 {
        let corner = Vector3::new(
            if i & 1 == 0 { aabb.max.x } else { aabb.min.x },
            if i & 2 == 0 { aabb.max.y } else { aabb.min.y },
            if i & 4 == 0 { aabb.max.z } else { aabb.min.z },
        );
        let p = orientation * corner + position;
        min = min.min(p);
        max = max.max(p);
    }
    Aabb::new(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Mesh discs are huge and never limit a pair; 0.02 is the convex-side
    /// threshold scale used across the narrow-phase tests.
    fn t() -> Real {
        0.02
    }

    /// Two flat quads (two triangles each) spanning the unit square in XZ.
    fn plate() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    /// Parallel plates separated inside the contact band report a
    /// margin-shell contact: `penetration = band - separation` (Bullet's
    /// GImpact triangle-margin semantics — the solver can hold meshes apart
    /// positionally, like real shapes).
    #[test]
    fn parallel_plates_band_contact() {
        let a = plate();
        let b = plate();
        // B floats 0.01 above A: separation 0.01 inside the 0.1 band.
        let m = mesh_mesh(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, 0.01, 0.0),
            Quaternion::IDENTITY,
            None,
            t(),
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, MESH_MESH_BAND - 0.01, epsilon = 1e-3);
    }

    /// B below A still resolves through the tree with the conservative
    /// corner-wrap (the transformed box is inflated by the band) and reports
    /// the flipped normal.
    #[test]
    fn parallel_plates_below_flips_the_normal() {
        let a = plate();
        let b = plate();
        // B floats 0.01 below A: normal from A to B points down.
        let m = mesh_mesh(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, -0.01, 0.0),
            Quaternion::IDENTITY,
            None,
            t(),
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, MESH_MESH_BAND - 0.01, epsilon = 1e-3);
    }

    /// A tilted plate resting its low edge just above the floor finds its
    /// contacts through the tree traversal (every triangle pair stays
    /// separated inside the band, the pose dynamic meshes live in while
    /// resting on each other).
    #[test]
    fn tilted_plate_edge_band_contact() {
        let a = plate();
        let b = plate();
        // Tilt B 10 deg about Z: vertices reach y in [0, sin(10 deg)].
        // Lifted so the lowest vertex sits 0.005 above A — every triangle
        // pair is separated by 0.005..0.175, the low-edge pairs inside the
        // 0.1 band.
        let q = Quaternion::from_rotation_z(10.0_f64 as Real * bullet_math::PI / 180.0);
        let m = mesh_mesh(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, 0.005, 0.0),
            q,
            None,
            t(),
        )
        .unwrap();
        // The surviving contacts hug the low edge (normal from A to B).
        let p = m.points[0];
        assert!(p.normal.y > 0.99);
        assert!((0.0..MESH_MESH_BAND).contains(&p.penetration));
    }

    #[test]
    fn separated_plates_report_nothing() {
        let a = plate();
        let b = plate();
        assert!(mesh_mesh(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, 5.0, 0.0),
            Quaternion::IDENTITY,
            None,
            t(),
        )
        .is_none());
    }

    /// Offset plates that share no overlapping region report nothing even
    /// though their node boxes conservatively overlap.
    #[test]
    fn laterally_offset_plates_report_nothing() {
        let a = plate();
        let b = plate();
        assert!(mesh_mesh(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(0.0, 0.0, 2.5),
            Quaternion::IDENTITY,
            None,
            t(),
        )
        .is_none());
    }
}
