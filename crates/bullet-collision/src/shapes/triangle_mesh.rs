use bullet_math::{Aabb, Quaternion, Real, Vector3};

use super::bvh::Bvh;
use super::CollisionShape;

/// A static triangle mesh collision shape, equivalent to Bullet's
/// `btBvhTriangleMeshShape` (a `btTriangleMesh` plus a built BVH).
///
/// Meshes are concave and intended for static bodies: they cannot be used
/// with GJK, have zero margin, and report zero inertia. The local-space BVH
/// accelerates narrow-phase candidate lookup and ray casting.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleMesh {
    pub vertices: Vec<Vector3>,
    pub indices: Vec<[u32; 3]>,
    bvh: Bvh,
    /// Per-triangle three-vertex convex hulls, built once (the narrow phase
    /// used to rebuild one allocation per candidate triangle per step;
    /// Bullet constructs its `btTriangleShape` on the stack for the same
    /// zero-per-step cost).
    tri_hulls: Vec<CollisionShape>,
    /// Per-triangle local AABBs, built once (the exact pre-GJK filter in the
    /// narrow phase; Bullet's `TestTriangleAgainstAabb2`).
    tri_aabbs: Vec<Aabb>,
    /// Cached local AABB over all vertices (hot: the contact-breaking
    /// threshold of every convex-vs-mesh pair reads it).
    local_bounds: Aabb,
}

impl TriangleMesh {
    /// Build a mesh from vertices and triangle indices (not validated).
    pub fn new(vertices: Vec<Vector3>, indices: Vec<[u32; 3]>) -> Self {
        let bvh = Bvh::build(&vertices, &indices);
        let mut tri_hulls = Vec::with_capacity(indices.len());
        let mut tri_aabbs = Vec::with_capacity(indices.len());
        let mut local_bounds = Aabb::new(Vector3::ZERO, Vector3::ZERO);
        for (i, t) in indices.iter().enumerate() {
            let a = vertices[t[0] as usize];
            let b = vertices[t[1] as usize];
            let c = vertices[t[2] as usize];
            tri_hulls.push(CollisionShape::convex_hull(vec![a, b, c]));
            tri_aabbs.push(Aabb::new(a.min(b.min(c)), a.max(b.max(c))));
            if i == 0 {
                local_bounds = tri_aabbs[0];
            } else {
                local_bounds = local_bounds.merged(&tri_aabbs[i]);
            }
        }
        Self {
            vertices,
            indices,
            bvh,
            tri_hulls,
            tri_aabbs,
            local_bounds,
        }
    }

    /// The three vertices of triangle `index`.
    #[inline]
    pub fn triangle(&self, index: usize) -> (Vector3, Vector3, Vector3) {
        let t = self.indices[index];
        (
            self.vertices[t[0] as usize],
            self.vertices[t[1] as usize],
            self.vertices[t[2] as usize],
        )
    }

    /// The prebuilt three-vertex hull of triangle `index` (local frame).
    #[inline]
    pub(crate) fn triangle_hull(&self, index: usize) -> &CollisionShape {
        &self.tri_hulls[index]
    }

    /// The local AABB of triangle `index`.
    #[inline]
    pub(crate) fn triangle_aabb(&self, index: usize) -> &Aabb {
        &self.tri_aabbs[index]
    }

    /// Triangle indices whose AABB overlaps `aabb` (local space).
    #[inline]
    pub(crate) fn triangles_in_aabb(&self, aabb: &Aabb) -> Vec<u32> {
        let mut out = Vec::new();
        self.bvh.triangles_in_aabb(aabb, &mut out);
        out
    }

    /// Local-space ray cast (same semantics as the shape-level ray cast).
    #[inline]
    pub(crate) fn raycast(&self, from: Vector3, to: Vector3) -> Option<(Real, Vector3)> {
        self.bvh.raycast(&self.vertices, &self.indices, from, to)
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        self.local_bounds
    }

    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        self.local_aabb().transformed(position, orientation)
    }

    /// Static meshes have no mass properties.
    #[inline]
    pub fn local_inertia(&self, _mass: Real) -> Vector3 {
        Vector3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A flat 2x2 grid (two triangles) spanning the unit square in XZ at y=0.
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
    fn local_aabb_covers_the_vertices() {
        let m = flat_ground();
        let a = m.local_aabb();
        assert_relative_eq!(a.min.x, 0.0);
        assert_relative_eq!(a.max.y, 0.0);
        assert_relative_eq!(a.max.z, 1.0);
    }

    #[test]
    fn aabb_query_finds_touching_triangles() {
        let m = flat_ground();
        // A box over the second triangle only. Both triangles share the unit
        // AABB, so the BVH conservatively reports both; the narrow phase does
        // the exact test.
        let query = Aabb::new(
            Vector3::new(0.8, -1.0, 0.8),
            Vector3::new(1.2, 1.0, 1.2),
        );
        let hits = m.triangles_in_aabb(&query);
        assert!(hits.contains(&1));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn raycast_hits_the_surface_with_upward_normal() {
        let m = flat_ground();
        let (t, n) = m.raycast(Vector3::new(0.5, 5.0, 0.5), Vector3::new(0.5, -1.0, 0.5)).unwrap();
        assert_relative_eq!(t, 5.0 / 6.0, epsilon = 1e-5);
        assert_relative_eq!(n.y, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn raycast_misses_beyond_the_mesh() {
        let m = flat_ground();
        assert!(m
            .raycast(Vector3::new(5.0, 5.0, 5.0), Vector3::new(5.0, -1.0, 5.0))
            .is_none());
    }
}
