use bullet_math::{Aabb, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::bvh::Bvh;
use super::CollisionShape;

/// One indexed triangle soup of a mesh part (Bullet's `btIndexedMesh`): an
/// independent vertex/index pair with its own index base. Parts never share
/// vertex buffers — tiled terrain duplicates the seam vertices instead.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleMeshPart {
    /// The part's vertices.
    pub vertices: Vec<Vector3>,
    /// Triangle corner indices into [`TriangleMeshPart::vertices`].
    pub indices: Vec<[u32; 3]>,
}

/// A triangle mesh collision shape, equivalent to Bullet's
/// `btBvhTriangleMeshShape` when used statically and to its
/// `btGImpactMeshShape` when used dynamically (a `btTriangleMesh` plus a
/// built BVH either way — the tree lives in the mesh's local frame and the
/// body pose carries it, so dynamic meshes never rebuild it).
///
/// Meshes are concave: they cannot be used with GJK as a whole, carry zero
/// margin, and collide with convex shapes through per-triangle tests (the
/// local-space BVH accelerates candidate lookup, ray casting, and the
/// mesh-vs-mesh tree traversal). With a static body the mesh behaves exactly
/// like Bullet's BVH triangle mesh; with a dynamic body it becomes Bullet's
/// GImpact case — vertex-point-mass inertia, mesh-vs-mesh and plane-vs-mesh
/// contact paths included.
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
        Self::from_parts(vec![TriangleMeshPart { vertices, indices }])
    }

    /// A scaled copy (Bullet's `btScaledBvhTriangleMeshShape` semantics,
    /// realized at construction time — see
    /// [`CollisionShape::scaled_triangle_mesh`]): every vertex is multiplied
    /// by `scaling` (negative components flip the winding exactly like the
    /// wrapper's per-axis swap) and the BVH, triangle hulls and AABBs are
    /// rebuilt from the scaled stream.
    pub fn scaled(self, scaling: Vector3) -> Self {
        let Self { mut vertices, indices, .. } = self;
        for v in &mut vertices {
            *v *= scaling;
        }
        Self::new(vertices, indices)
    }

    /// Build a mesh from several independent parts (Bullet's
    /// `btTriangleIndexVertexArray` with `btIndexedMesh` entries): parts
    /// never share vertex buffers, so the vertices are concatenated and the
    /// indices offset per part. The runtime — BVH, narrow phase, ray casts,
    /// inertia — sees one triangle stream in part order, exactly like
    /// Bullet's BVH leaves spanning every part; there is no per-part
    /// runtime path, this is a construction-time convenience.
    pub fn from_parts(parts: Vec<TriangleMeshPart>) -> Self {
        let total_vertices = parts.iter().map(|p| p.vertices.len()).sum();
        let total_indices = parts.iter().map(|p| p.indices.len()).sum();
        let mut vertices = Vec::with_capacity(total_vertices);
        let mut indices = Vec::with_capacity(total_indices);
        for part in parts {
            let offset = vertices.len() as u32;
            vertices.extend(part.vertices);
            indices.extend(
                part.indices
                    .into_iter()
                    .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset]),
            );
        }
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
    pub fn triangles_in_aabb(&self, aabb: &Aabb) -> Vec<u32> {
        let mut out = Vec::new();
        self.bvh.triangles_in_aabb(aabb, &mut out);
        out
    }

    /// Append triangle-index pairs `(this, other)` whose node boxes may
    /// overlap when `other`'s tree is mapped into this tree's frame by
    /// `rotation`/`translation` (the B-to-A local pose product, Bullet's
    /// `trans1to0` between GImpact box trees), with the transformed boxes
    /// inflated by `margin` (the pair contact band). Conservative: the
    /// candidate pairs are exact-tested by the narrow-phase caller.
    #[inline]
    pub(crate) fn triangles_overlapping(
        &self,
        other: &TriangleMesh,
        rotation: Quaternion,
        translation: Vector3,
        margin: Real,
        out: &mut Vec<(u32, u32)>,
    ) {
        self.bvh
            .triangles_overlapping(&other.bvh, rotation, translation, margin, out);
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

    /// Mesh inertia as vertex point masses (Bullet's
    /// `btGImpactMeshShape::calculateLocalInertia` with `CALC_EXACT_INERTIA`,
    /// the default build): every vertex carries `mass / vertex_count` and
    /// contributes its origin-frame point-mass tensor. Static bodies never
    /// read this (`RigidBody::new_static` ignores inertia); a dynamic mesh
    /// body is Bullet's GImpact use case.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        if self.vertices.is_empty() {
            return Vector3::ZERO;
        }
        let point_mass = mass / self.vertices.len() as Real;
        let mut inertia = Vector3::ZERO;
        for v in &self.vertices {
            inertia.x += point_mass * (v.y * v.y + v.z * v.z);
            inertia.y += point_mass * (v.x * v.x + v.z * v.z);
            inertia.z += point_mass * (v.x * v.x + v.y * v.y);
        }
        inertia
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

    /// Vertex point-mass inertia (Bullet's `CALC_EXACT_INERTIA`): four unit
    /// masses at the corners of a unit square in XZ at y=0.
    #[test]
    fn local_inertia_matches_vertex_point_masses() {
        let m = flat_ground();
        let i = m.local_inertia(4.0);
        // point_mass = 1; x-extent 1, z-extent 1, y-extent 0.
        // Ixx = sum m*(y^2+z^2) = 4*(0+0.5^2..) -- corners at (0/1, 0, 0/1):
        // z^2 in {0,1}, two vertices each -> sum z^2 = 2, same for x^2.
        assert_relative_eq!(i.x, 2.0, epsilon = 1e-6); // sum(y^2+z^2) = 2
        assert_relative_eq!(i.y, 4.0, epsilon = 1e-6); // sum(x^2+z^2) = 4
        assert_relative_eq!(i.z, 2.0, epsilon = 1e-6); // sum(x^2+y^2) = 2
    }

    #[test]
    fn empty_mesh_inertia_is_zero() {
        let m = TriangleMesh::new(Vec::new(), Vec::new());
        assert_eq!(m.local_inertia(1.0), Vector3::ZERO);
    }

    /// Two unit-square parts tiling x in [0, 2] (duplicated seam vertices at
    /// x=1, like independent `btIndexedMesh` buffers).
    fn two_parts() -> Vec<TriangleMeshPart> {
        let quad = |x0: Real| {
            vec![
                Vector3::new(x0, 0.0, 0.0),
                Vector3::new(x0 + 1.0, 0.0, 0.0),
                Vector3::new(x0 + 1.0, 0.0, 1.0),
                Vector3::new(x0, 0.0, 1.0),
            ]
        };
        let tris = || vec![[0, 1, 2], [0, 2, 3]];
        vec![
            TriangleMeshPart { vertices: quad(0.0), indices: tris() },
            TriangleMeshPart { vertices: quad(1.0), indices: tris() },
        ]
    }

    /// `from_parts` is exactly the concatenated single-buffer mesh: same
    /// vertices, same (offset) indices, same derived data — byte-identical
    /// to the concatenation workaround.
    #[test]
    fn parts_equal_the_concatenated_mesh() {
        let parts = two_parts();
        let concatenated = TriangleMesh::new(
            parts[0]
                .vertices
                .iter()
                .chain(&parts[1].vertices)
                .cloned()
                .collect(),
            vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        );
        let multi = TriangleMesh::from_parts(parts);
        assert_eq!(multi, concatenated);
    }

    /// The runtime paths treat the part boundary as an internal seam: the
    /// BVH reports candidate triangles from both parts across it, ray casts
    /// hit either side, and the vertex point-mass inertia sums both parts
    /// (Bullet's `btOptimizedBvh::build` walks every `IndexedMesh` part the
    /// same way).
    #[test]
    fn seam_crossing_queries_reach_both_parts() {
        let m = TriangleMesh::from_parts(two_parts());
        // The seam column x=1: candidates from both parts share the seam.
        let query = Aabb::new(Vector3::new(0.8, -1.0, -0.1), Vector3::new(1.2, 1.0, 1.1));
        let hits = m.triangles_in_aabb(&query);
        assert!(hits.contains(&1), "part A triangle missing");
        assert!(hits.contains(&2), "part B triangle missing");
        // A ray down the seam hits a triangle (t and normal identical to
        // the interior case — the duplicated seam vertices are exact).
        let (t, n) = m
            .raycast(Vector3::new(1.0, 5.0, 0.5), Vector3::new(1.0, -1.0, 0.5))
            .unwrap();
        assert_relative_eq!(t, 5.0 / 6.0, epsilon = 1e-6);
        assert_relative_eq!(n.y, 1.0, epsilon = 1e-6);
        // Inertia: 8 vertex point masses at the grid corners.
        let i = m.local_inertia(8.0);
        assert_relative_eq!(i.x, 4.0, epsilon = 1e-5); // sum(y²+z²) = 4
        assert_relative_eq!(i.z, 12.0, epsilon = 1e-5); // sum(x²+y²) = 12
    }
}
