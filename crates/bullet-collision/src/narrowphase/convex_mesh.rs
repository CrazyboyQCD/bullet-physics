//! Convex-versus-triangle-mesh narrow phase.
//!
//! The convex's world AABB (in the mesh's local frame) looks up candidate
//! triangles in the mesh BVH; each candidate is tested with the generic
//! GJK+EPA pipeline by treating the triangle as a three-vertex hull. The
//! contact normal points from the convex shape to the mesh, matching the
//! `collide` convention that the normal points from `shape_a` to `shape_b`
//! (callers flip as needed).

use bullet_math::{math, Aabb, Quaternion, Real, Vector3};

use crate::manifold::{ContactManifold, ContactPoint};
use crate::narrowphase::convex_convex::convex_convex;
use crate::shapes::{CollisionShape, TriangleMesh};

/// Contact between a convex shape and a static triangle mesh.
///
/// Returns `None` when no candidate triangle touches (or the overlaps are
/// numerical noise below the EPA threshold). `seed` optionally warm-starts
/// every triangle-level GJK run with a cached separating normal.
/// `threshold` is the convex-vs-mesh pair contact band (Bullet runs each
/// candidate triangle through the pair manifold's breaking threshold).
#[allow(clippy::too_many_arguments)]
pub fn convex_mesh(
    convex: &CollisionShape,
    position_convex: Vector3,
    orientation_convex: Quaternion,
    mesh: &TriangleMesh,
    position_mesh: Vector3,
    orientation_mesh: Quaternion,
    seed: Option<Vector3>,
    threshold: Real,
) -> Option<ContactManifold> {
    // Sweep the convex's world AABB corners into the mesh's local frame.
    let convex_world = convex.world_aabb(position_convex, orientation_convex);
    let inv_m = orientation_mesh.inverse();
    let mut local_min = Vector3::splat(Real::INFINITY);
    let mut local_max = Vector3::splat(Real::NEG_INFINITY);
    let c = convex_world.center();
    let e = convex_world.extents();
    for i in 0..8 {
        let corner = Vector3::new(
            if i & 1 == 0 { c.x + e.x } else { c.x - e.x },
            if i & 2 == 0 { c.y + e.y } else { c.y - e.y },
            if i & 4 == 0 { c.z + e.z } else { c.z - e.z },
        );
        let p = inv_m * (corner - position_mesh);
        local_min = local_min.min(p);
        local_max = local_max.max(p);
    }

    let mut manifold = ContactManifold::new();
    let query = Aabb::new(local_min, local_max);

    // Sphere fast path (Bullet's `btSphereTriangleCollisionAlgorithm` /
    // `SphereTriangleDetector`): an analytic closest-point test per
    // candidate triangle in the mesh frame, replacing the generic
    // GJK+EPA pipeline that runs the triangle as a three-vertex hull.
    if let CollisionShape::Sphere(s) = convex {
        let center = inv_m * (position_convex - position_mesh);
        for t in mesh.triangles_in_aabb(&query) {
            if !mesh.triangle_aabb(t as usize).overlaps(&query) {
                continue;
            }
            let (a, b, c) = mesh.triangle(t as usize);
            if let Some((point_local, normal_local, penetration)) =
                sphere_triangle_contact(center, s.radius, threshold, a, b, c)
            {
                // The detector's normal points toward the sphere center;
                // this module reports it from the convex toward the mesh.
                manifold.points.push(ContactPoint::new(
                    position_mesh + orientation_mesh * point_local,
                    orientation_mesh * -normal_local,
                    penetration,
                ));
            }
        }
    } else {
        for t in mesh.triangles_in_aabb(&query) {
            // Exact triangle-AABB filter before GJK (Bullet's
            // `TestTriangleAgainstAabb2`): the BVH query is conservative (shared
            // node AABBs return neighbor triangles too), and a triangle whose
            // AABB misses the query AABB cannot touch the convex — GJK would
            // only confirm the separation.
            if !mesh.triangle_aabb(t as usize).overlaps(&query) {
                continue;
            }
            let tri = mesh.triangle_hull(t as usize);
            if let Some(m) = convex_convex(
                convex,
                position_convex,
                orientation_convex,
                tri,
                position_mesh,
                orientation_mesh,
                seed,
                threshold,
            ) {
                manifold.points.extend(m.points);
            }
        }
    }
    if manifold.is_empty() {
        None
    } else {
        // One contact per candidate triangle can easily exceed the manifold
        // cap; reduce to four well-separated points (Bullet manifold
        // reduction) before handing the contact set to the solver.
        manifold.reduce(4);
        Some(manifold)
    }
}

/// Analytic sphere-vs-triangle contact, a port of Bullet's
/// `SphereTriangleDetector::collide` in the triangle's frame. Returns the
/// contact point and center-pointing normal on the triangle side plus the
/// penetration `radius - distance`: positive while overlapping, negative
/// while separated inside the contact band (`radius + threshold`), matching
/// the margin-band convention of the GJK path.
fn sphere_triangle_contact(
    center: Vector3,
    radius: Real,
    threshold: Real,
    a: Vector3,
    b: Vector3,
    c: Vector3,
) -> Option<(Vector3, Vector3, Real)> {
    let radius_with_threshold = radius + threshold;

    let mut normal = (b - a).cross(c - a);
    let l2 = normal.length_squared();
    if l2 < Real::EPSILON * Real::EPSILON {
        // Degenerate triangle (Bullet skips it too).
        return None;
    }
    normal *= math::sqrt(l2).recip();
    let p1_to_center = center - a;
    let mut distance_from_plane = normal.dot(p1_to_center);
    if distance_from_plane < 0.0 {
        // Sphere approaches from the back face.
        distance_from_plane = -distance_from_plane;
        normal = -normal;
    }
    if distance_from_plane >= radius_with_threshold {
        return None;
    }

    let contact_point = if point_in_triangle(center, a, b, c, normal) {
        // Inside the contact wedge: the shell point on the face plane.
        center - normal * distance_from_plane
    } else {
        // Maybe inside one of the edge capsules: keep the nearest edge point
        // closer than the contact radius.
        let capsule_radius_sq = radius_with_threshold * radius_with_threshold;
        let mut min_dist_sq = capsule_radius_sq;
        let mut nearest = None;
        for (from, to) in [(a, b), (b, c), (c, a)] {
            let (dist_sq, on_edge) = segment_closest_point_sq(from, to, center);
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                nearest = Some(on_edge);
            }
        }
        nearest?
    };

    let contact_to_center = center - contact_point;
    let dist_sq = contact_to_center.length_squared();
    if dist_sq >= radius_with_threshold * radius_with_threshold {
        return None;
    }
    if dist_sq > Real::EPSILON {
        let distance = math::sqrt(dist_sq);
        Some((
            contact_point,
            contact_to_center * distance.recip(),
            radius - distance,
        ))
    } else {
        // Center sits on the surface: the direction is undefined, report the
        // face normal and full-radius penetration (Bullet's fallback).
        Some((contact_point, normal, radius))
    }
}

/// Squared distance from `p` to segment `from-to` and the closest segment
/// point (Bullet's `SegmentSqrDistance`).
fn segment_closest_point_sq(from: Vector3, to: Vector3, p: Vector3) -> (Real, Vector3) {
    let v = to - from;
    let mut t = v.dot(p - from);
    if t > 0.0 {
        let dot_vv = v.dot(v);
        if t < dot_vv {
            t /= dot_vv;
        } else {
            t = 1.0;
        }
    } else {
        t = 0.0;
    }
    let nearest = from + v * t;
    ((p - nearest).length_squared(), nearest)
}

/// Whether `p` lies inside the triangle `v1-v2-v3` (Bullet's
/// `pointInTriangle`, normals oriented consistently with `normal`).
fn point_in_triangle(p: Vector3, v1: Vector3, v2: Vector3, v3: Vector3, normal: Vector3) -> bool {
    let edge1 = v2 - v1;
    let edge2 = v3 - v2;
    let edge3 = v1 - v3;
    let edge1_normal = edge1.cross(normal);
    let edge2_normal = edge2.cross(normal);
    let edge3_normal = edge3.cross(normal);
    let r1 = edge1_normal.dot(p - v1);
    let r2 = edge2_normal.dot(p - v2);
    let r3 = edge3_normal.dot(p - v3);
    (r1 > 0.0 && r2 > 0.0 && r3 > 0.0) || (r1 <= 0.0 && r2 <= 0.0 && r3 <= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The convex-vs-mesh pair threshold uses the convex's (smaller) disc.
    fn t(convex: &CollisionShape) -> Real {
        convex.contact_breaking_threshold()
    }

    fn flat_ground() -> TriangleMesh {
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

    #[test]
    fn sphere_resting_on_mesh() {
        let mesh = flat_ground();
        let sphere = CollisionShape::sphere(0.5);
        // Sphere center 0.4 above the surface: penetration 0.1; the normal
        // points from the sphere (convex) toward the mesh, i.e. -Y.
        let m = convex_mesh(
            &sphere,
            Vector3::new(0.5, 0.4, 0.5),
            Quaternion::IDENTITY,
            &mesh,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            None,
            t(&sphere),
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-3);
    }

    /// The analytic sphere fast path must agree with the generic GJK+EPA
    /// pipeline (sphere vs the triangle as a three-vertex hull) on contact
    /// point, normal, and depth across face, edge, vertex, and band poses.
    #[test]
    fn sphere_fast_path_matches_gjk() {
        let mesh = TriangleMesh::new(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.3, 0.2, 0.9), // tilted triangle
            ],
            vec![[0, 1, 2]],
        );
        let sphere = CollisionShape::sphere(0.5);
        let th = t(&sphere);
        for center in [
            Vector3::new(0.4, 0.3, 0.4),   // over the face interior
            Vector3::new(0.95, 0.15, 0.1), // near a vertex region
            Vector3::new(-0.3, 0.3, 0.2),  // off the a-vertex side
            Vector3::new(0.3, 0.55, 0.3),  // separated inside the band
            Vector3::new(0.6, 0.05, 0.4),  // deep overlap
        ] {
            let fast = convex_mesh(
                &sphere,
                center,
                Quaternion::IDENTITY,
                &mesh,
                Vector3::ZERO,
                Quaternion::IDENTITY,
                None,
                th,
            );
            let slow = convex_convex(
                &sphere,
                center,
                Quaternion::IDENTITY,
                mesh.triangle_hull(0),
                Vector3::ZERO,
                Quaternion::IDENTITY,
                None,
                th,
            );
            match (&fast, &slow) {
                (Some(f), Some(s)) => {
                    assert_eq!(f.points.len(), s.points.len(), "at {center:?}");
                    let fp = f.points[0];
                    let sp = s.points[0];
                    assert_relative_eq!(fp.penetration, sp.penetration, epsilon = 5e-3);
                    // EPA converges the triangle hull's surface to ~1e-4, so
                    // edge/vertex-region normals differ from the analytic
                    // closest point by a fraction of a degree.
                    assert!(
                        fp.normal.dot(sp.normal) > 0.999,
                        "normals differ at {center:?}: {fp:?} vs {sp:?}"
                    );
                    assert_relative_eq!(fp.position.x, sp.position.x, epsilon = 1e-2);
                    assert_relative_eq!(fp.position.y, sp.position.y, epsilon = 1e-2);
                    assert_relative_eq!(fp.position.z, sp.position.z, epsilon = 1e-2);
                }
                (None, None) => {}
                _ => panic!("presence mismatch at {center:?}: {fast:?} vs {slow:?}"),
            }
        }
    }

    /// Bullet `SphereTriangleDetector` semantics: contacts over edge and
    /// vertex regions, negative penetration inside the contact band, no
    /// contact beyond it, and two-sided (back-face) reporting.
    #[test]
    fn sphere_triangle_edge_vertex_band_backface() {
        let mesh = TriangleMesh::new(
            vec![
                Vector3::ZERO,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2]],
        );
        let sphere = CollisionShape::sphere(0.5);
        let th = t(&sphere);
        let probe = |center: Vector3| {
            convex_mesh(
                &sphere,
                center,
                Quaternion::IDENTITY,
                &mesh,
                Vector3::ZERO,
                Quaternion::IDENTITY,
                None,
                th,
            )
        };

        // Over vertex 0 at distance 0.4: penetration 0.1, contact at the
        // vertex, normal from the sphere toward the mesh (-Y).
        let p = probe(Vector3::new(0.0, 0.4, 0.0)).unwrap().points[0];
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-4);
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.position.length_squared(), 0.0, epsilon = 1e-6);

        // Over the a-b edge midpoint at distance 0.4: same numbers, contact
        // on the edge.
        let p = probe(Vector3::new(0.5, 0.4, 0.0)).unwrap().points[0];
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-4);
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.position.x, 0.5, epsilon = 1e-6);
        assert_relative_eq!(p.position.y, 0.0, epsilon = 1e-6);

        // Band semantics at the detector level. Note the mesh-level AABB
        // filter (shared with the GJK path) drops a sphere separated along
        // the face normal — its AABB no longer overlaps the triangle's — so
        // the band cases are probed on the detector directly. In-scene band
        // contacts arise while the AABBs still overlap, e.g. across edges.
        let th = t(&sphere);
        let (a, b, c) = (Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let (point, normal, pen) =
            sphere_triangle_contact(Vector3::new(0.3, 0.51, 0.3), 0.5, th, a, b, c).unwrap();
        assert_relative_eq!(pen, -0.01, epsilon = 1e-4);
        assert_relative_eq!(normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(point.y, 0.0, epsilon = 1e-6);
        assert!(sphere_triangle_contact(Vector3::new(0.3, 0.6, 0.3), 0.5, th, a, b, c).is_none());

        // Back face: a sphere below the triangle still reports a contact,
        // with the normal flipped toward it (+Y after the convex→mesh flip).
        let p = probe(Vector3::new(0.3, -0.4, 0.3)).unwrap().points[0];
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-4);
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.position.y, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn separated_sphere_does_not_touch_mesh() {
        let mesh = flat_ground();
        let sphere = CollisionShape::sphere(0.5);
        assert!(convex_mesh(
            &sphere,
            Vector3::new(0.5, 5.0, 0.5),
            Quaternion::IDENTITY,
            &mesh,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            None,
            t(&sphere),
        )
        .is_none());
    }

    #[test]
    fn box_on_heightfield() {
        let hf = CollisionShape::heightfield(vec![0.0; 9], 3, 3, Vector3::ONE);
        let CollisionShape::Heightfield(hf) = &hf else {
            unreachable!()
        };
        let box_shape = CollisionShape::box_shape(Vector3::splat(0.5));
        // Box bottom at y = -0.1: penetration 0.1.
        let m = convex_mesh(
            &box_shape,
            Vector3::new(1.0, 0.4, 1.0),
            Quaternion::IDENTITY,
            hf.as_mesh(),
            Vector3::ZERO,
            Quaternion::IDENTITY,
            None,
            t(&box_shape),
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-3);
    }

    /// A box straddling four heightfield cells yields one contact per touched
    /// triangle; the merged manifold must be reduced to the manifold cap.
    #[test]
    fn box_over_many_triangles_reduces_contacts() {
        let hf = CollisionShape::heightfield(vec![0.0; 25], 5, 5, Vector3::ONE);
        let CollisionShape::Heightfield(hf) = &hf else {
            unreachable!()
        };
        let box_shape = CollisionShape::box_shape(Vector3::splat(0.5));
        let m = convex_mesh(
            &box_shape,
            Vector3::new(2.0, 0.4, 2.0),
            Quaternion::IDENTITY,
            hf.as_mesh(),
            Vector3::ZERO,
            Quaternion::IDENTITY,
            None,
            t(&box_shape),
        )
        .unwrap();
        assert!(m.points.len() <= 4);
        // The deepest contact is still reported first with the -Y normal.
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-3);
    }
}
