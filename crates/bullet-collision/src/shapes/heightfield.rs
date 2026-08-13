use bullet_math::{Aabb, Quaternion, Real, Vector3};

use super::triangle_mesh::TriangleMesh;

/// A static heightfield terrain shape, equivalent to Bullet's
/// `btHeightfieldTerrainShape`. The height grid is materialized into an
/// internal [`TriangleMesh`] at construction, sharing its narrow-phase and ray
/// cast paths.
///
/// `heights` is a `width x depth` row-major grid (row `z` first); the world
/// position of grid point `(x, z)` is `(x * cell_size.x, height, z * cell_size.z)`.
/// Like meshes, heightfields are concave, static-only, and report zero inertia.
#[derive(Debug, Clone, PartialEq)]
pub struct Heightfield {
    mesh: TriangleMesh,
    pub width: usize,
    pub depth: usize,
}

impl Heightfield {
    pub fn new(heights: Vec<Real>, width: usize, depth: usize, cell_size: Vector3) -> Self {
        debug_assert_eq!(heights.len(), width * depth, "height grid must be width * depth");
        let mut vertices = Vec::with_capacity(width * depth);
        for z in 0..depth {
            for x in 0..width {
                let h = heights[z * width + x];
                vertices.push(Vector3::new(x as Real * cell_size.x, h, z as Real * cell_size.z));
            }
        }
        // Two triangles per cell; winding chosen so the normal faces +Y.
        let mut indices = Vec::with_capacity((width - 1) * (depth - 1) * 2);
        for z in 0..depth - 1 {
            for x in 0..width - 1 {
                let v00 = (z * width + x) as u32;
                let v10 = v00 + 1;
                let v01 = v00 + width as u32;
                let v11 = v01 + 1;
                indices.push([v00, v11, v10]);
                indices.push([v00, v01, v11]);
            }
        }
        let mesh = TriangleMesh::new(vertices, indices);
        Self { mesh, width, depth }
    }

    /// The internal triangle mesh (shared narrow-phase path).
    #[inline]
    pub(crate) fn as_mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// Local-space ray cast (same semantics as the shape-level ray cast).
    #[inline]
    pub(crate) fn raycast(&self, from: Vector3, to: Vector3) -> Option<(Real, Vector3)> {
        self.mesh.raycast(from, to)
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        self.mesh.local_aabb()
    }

    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        self.mesh.world_aabb(position, orientation)
    }

    /// Static heightfields have no mass properties.
    #[inline]
    pub fn local_inertia(&self, _mass: Real) -> Vector3 {
        Vector3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A 2x2 flat grid with cell size 1.
    fn flat() -> Heightfield {
        Heightfield::new(vec![0.0, 0.0, 0.0, 0.0], 2, 2, Vector3::ONE)
    }

    #[test]
    fn local_aabb_spans_the_grid() {
        let h = flat();
        let a = h.local_aabb();
        assert_relative_eq!(a.min.x, 0.0);
        assert_relative_eq!(a.max.x, 1.0);
        assert_relative_eq!(a.max.z, 1.0);
        assert_relative_eq!(a.max.y, 0.0);
    }

    #[test]
    fn raycast_hits_with_upward_normal() {
        let h = flat();
        let (t, n) = h.raycast(Vector3::new(0.5, 5.0, 0.5), Vector3::new(0.5, -1.0, 0.5)).unwrap();
        assert_relative_eq!(t, 5.0 / 6.0, epsilon = 1e-5);
        assert_relative_eq!(n.y, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn elevated_cell_changes_the_surface_height() {
        let h = Heightfield::new(vec![0.0, 2.0, 0.0, 0.0], 2, 2, Vector3::ONE);
        // The column at (1, 0) is 2 units high. A ray above the x >= z half
        // hits the sloped triangle: at x = 0.75, z = 0.25 the surface sits at
        // y = 1.0 (interpolated along the slope from (1,2,0)).
        let (t, n) = h.raycast(Vector3::new(0.75, 5.0, 0.25), Vector3::new(0.75, -1.0, 0.25)).unwrap();
        assert_relative_eq!(t, 2.0 / 3.0, epsilon = 1e-4);
        // Sloped face normal facing the ray origin: upward component present.
        assert!(n.y > 0.1, "surface normal should face up, got {n:?}");
    }
}
