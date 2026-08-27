//! Convex-versus-triangle-mesh narrow phase.
//!
//! The convex's world AABB (in the mesh's local frame) looks up candidate
//! triangles in the mesh BVH; each candidate is tested with the generic
//! GJK+EPA pipeline by treating the triangle as a three-vertex hull. The
//! contact normal points from the convex shape to the mesh, matching the
//! `collide` convention that the normal points from `shape_a` to `shape_b`
//! (callers flip as needed).

use bullet_math::{Aabb, Quaternion, Real, Vector3};

use crate::manifold::ContactManifold;
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
