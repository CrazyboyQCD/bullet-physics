use bullet_math::{Aabb, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::triangle_mesh::TriangleMesh;

/// Up axis of a heightfield grid (upstream's `m_upAxis`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpAxis {
    X,
    Y,
    Z,
}

/// Construction options for a [`Heightfield`], mirroring the observable
/// surface of upstream's `btHeightfieldTerrainShape` flags. Defaults match
/// the legacy constructor: Y-up, no diagonal or winding flips.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightfieldOptions {
    pub up_axis: UpAxis,
    /// Split every quad along the `v(x,j) - v(x+1,j+1)` diagonal (upstream
    /// `flipQuadEdges`).
    pub flip_quad_edges: bool,
    /// Checkerboard the quad diagonal on cells with even `(row + col)`
    /// (upstream `m_useDiamondSubdivision`). Ignored when
    /// [`Self::flip_quad_edges`] is set.
    pub use_diamond_subdivision: bool,
    /// Alternate the quad diagonal by row parity (upstream
    /// `m_useZigzagSubdivision`). Ignored when [`Self::flip_quad_edges`] or
    /// [`Self::use_diamond_subdivision`] applies.
    pub use_zigzag_subdivision: bool,
    /// Reverse every triangle's winding (upstream
    /// `m_flipTriangleWinding`).
    pub flip_triangle_winding: bool,
}

impl Default for HeightfieldOptions {
    fn default() -> Self {
        Self {
            up_axis: UpAxis::Y,
            flip_quad_edges: false,
            use_diamond_subdivision: false,
            use_zigzag_subdivision: false,
            flip_triangle_winding: false,
        }
    }
}

/// A static heightfield terrain shape, equivalent to Bullet's
/// `btHeightfieldTerrainShape`. The height grid is materialized into an
/// internal [`TriangleMesh`] at construction, sharing its narrow-phase and ray
/// cast paths.
///
/// `heights` is a `width x depth` row-major grid (row index first). The grid
/// is centered on the local origin like upstream: grid point `(x, y)` sits at
/// in-plane coordinates `((x - (width-1)/2) * cell, (y - (depth-1)/2) * cell)`
/// and its height is offset by the midpoint of the data's min/max, so the
/// terrain's bounding box is centered at the body origin. Like meshes,
/// heightfields are concave, static-only, and report zero inertia.
///
/// Reference decisions covered (upstream `btHeightfieldTerrainShape`):
/// centered in-plane origin `i - (n-1)/2` (`getVertex`, with our
/// `cell_size` scaling what upstream leaves to body `localScaling`);
/// height offset by the min/max midpoint (`m_localOrigin`); the quad
/// diagonal choice `flipQuadEdges || (diamond && !(row+col & 1)) || (zigzag
/// && !(row & 1))`; the exact two-triangle windings per diagonal branch;
/// and `flipTriangleWinding` applied as an index permutation
/// (`processAllTriangles`). Not ported (recorded in deviations): the
/// quantized height data types + `heightScale` (we take the crate's `Real`
/// only), the `m_vboundsGrid` acceleration (upstream-internal), and the
/// on-demand cell iteration (we materialize once).
#[derive(Debug, Clone, PartialEq)]
pub struct Heightfield {
    mesh: TriangleMesh,
    pub width: usize,
    pub depth: usize,
}

impl Heightfield {
    pub fn new(heights: Vec<Real>, width: usize, depth: usize, cell_size: Vector3) -> Self {
        Self::with_options(HeightfieldOptions::default(), heights, width, depth, cell_size)
    }

    pub fn with_options(
        options: HeightfieldOptions,
        heights: Vec<Real>,
        width: usize,
        depth: usize,
        cell_size: Vector3,
    ) -> Self {
        debug_assert_eq!(heights.len(), width * depth, "height grid must be width * depth");
        debug_assert!(width >= 2 && depth >= 2, "heightfield needs at least one cell");
        let min_h = heights.iter().copied().fold(Real::INFINITY, Real::min);
        let max_h = heights.iter().copied().fold(Real::NEG_INFINITY, Real::max);
        let mid_h = (min_h + max_h) * 0.5;
        let mut vertices = Vec::with_capacity(width * depth);
        for y in 0..depth {
            for x in 0..width {
                let h = heights[y * width + x] - mid_h;
                let u = (x as Real - (width - 1) as Real * 0.5) * cell_size.x;
                let v = (y as Real - (depth - 1) as Real * 0.5) * cell_size.z;
                vertices.push(match options.up_axis {
                    UpAxis::X => Vector3::new(h, u, v),
                    UpAxis::Y => Vector3::new(u, h, v),
                    UpAxis::Z => Vector3::new(u, v, h),
                });
            }
        }
        // Two triangles per cell. Diagonal selection and the two windings
        // follow upstream's processAllTriangles verbatim; the winding flip
        // is an index permutation applied to both triangles.
        let order = if options.flip_triangle_winding {
            [2usize, 1, 0]
        } else {
            [0, 1, 2]
        };
        let wind = |t: [u32; 3]| [t[order[0]], t[order[1]], t[order[2]]];
        let mut indices = Vec::with_capacity((width - 1) * (depth - 1) * 2);
        for y in 0..depth - 1 {
            for x in 0..width - 1 {
                let v00 = (y * width + x) as u32;
                let v10 = v00 + 1;
                let v01 = v00 + width as u32;
                let v11 = v01 + 1;
                let flipped = options.flip_quad_edges
                    || (options.use_diamond_subdivision && ((y + x) & 1) == 0)
                    || (options.use_zigzag_subdivision && (y & 1) == 0);
                let (a, b) = if flipped {
                    ([v00, v01, v11], [v00, v11, v10])
                } else {
                    ([v00, v01, v10], [v10, v01, v11])
                };
                indices.push(wind(a));
                indices.push(wind(b));
            }
        }
        let mesh = TriangleMesh::new(vertices, indices);
        Self { mesh, width, depth }
    }

    /// The internal triangle mesh (shared narrow-phase path).
    #[inline]
    pub fn as_mesh(&self) -> &TriangleMesh {
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

    /// A 2x2 flat grid with cell size 1, centered on the origin.
    fn flat() -> Heightfield {
        Heightfield::new(vec![0.0, 0.0, 0.0, 0.0], 2, 2, Vector3::ONE)
    }

    #[test]
    fn local_aabb_spans_the_centered_grid() {
        let h = flat();
        let a = h.local_aabb();
        assert_relative_eq!(a.min.x, -0.5);
        assert_relative_eq!(a.max.x, 0.5);
        assert_relative_eq!(a.min.z, -0.5);
        assert_relative_eq!(a.max.z, 0.5);
        assert_relative_eq!(a.max.y, 0.0);
        assert_relative_eq!(a.min.y, 0.0);
    }

    #[test]
    fn heights_are_offset_by_the_midpoint() {
        let h = Heightfield::new(vec![-1.0, 1.0, -1.0, 1.0], 2, 2, Vector3::ONE);
        let a = h.local_aabb();
        assert_relative_eq!(a.min.y, -1.0);
        assert_relative_eq!(a.max.y, 1.0);
    }

    #[test]
    fn raycast_hits_with_upward_normal() {
        let h = flat();
        let (t, n) = h
            .raycast(
                Vector3::new(-0.25, 5.0, -0.25),
                Vector3::new(-0.25, -1.0, -0.25),
            )
            .unwrap();
        assert_relative_eq!(t, 5.0 / 6.0, epsilon = 1e-6);
        assert_relative_eq!(n.y, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn elevated_cell_changes_the_surface_height() {
        let h = Heightfield::new(vec![0.0, 2.0, 0.0, 0.0], 2, 2, Vector3::ONE);
        // Mid height = 1; the column at (1, 0) is 2 high, so its apex sits
        // at local y = +1. The default split's first triangle (v00,v01,v10)
        // covers the x+z < 0 half with the plane y = 2x: at
        // (x, z) = (0.25, -0.375) the surface is at y = 0.5.
        let (t, n) = h
            .raycast(
                Vector3::new(0.25, 5.0, -0.375),
                Vector3::new(0.25, -1.0, -0.375),
            )
            .unwrap();
        assert_relative_eq!(t, 0.75, epsilon = 1e-4);
        // Sloped face normal facing the ray origin: upward component present.
        assert!(n.y > 0.1, "surface normal should face up, got {n:?}");
    }

    #[test]
    fn x_and_z_up_grids_lay_the_grid_along_the_axis() {
        // heights [0, 0, 1, 1]: mid 0.5, so the up coordinate spans ±0.5.
        let pts: Vec<Real> = vec![0.0, 0.0, 1.0, 1.0];
        let hx = Heightfield::with_options(
            HeightfieldOptions {
                up_axis: UpAxis::X,
                ..HeightfieldOptions::default()
            },
            pts.clone(),
            2,
            2,
            Vector3::ONE,
        );
        let a = hx.local_aabb();
        assert_relative_eq!(a.min.x, -0.5);
        assert_relative_eq!(a.max.x, 0.5);
        assert_relative_eq!(a.min.y, -0.5);
        assert_relative_eq!(a.max.y, 0.5);
        assert_relative_eq!(a.min.z, -0.5);

        let hz = Heightfield::with_options(
            HeightfieldOptions {
                up_axis: UpAxis::Z,
                ..HeightfieldOptions::default()
            },
            pts,
            2,
            2,
            Vector3::ONE,
        );
        let a = hz.local_aabb();
        assert_relative_eq!(a.min.z, -0.5);
        assert_relative_eq!(a.max.z, 0.5);
        assert_relative_eq!(a.max.x, 0.5);
    }

    /// Recover the input vertex index of a triangle corner: the diagonal
    /// tests use `heights[i] = i` with midpoint 4, so the local height is
    /// the index minus four.
    fn tri_ids(hf: &Heightfield, t: usize) -> [u32; 3] {
        let (a, b, c) = hf.as_mesh().triangle(t);
        let id = |p: Vector3| (p.y + 4.0).round() as u32;
        [id(a), id(b), id(c)]
    }

    /// 3x3 grid quad diagonals: default split, flip_quad_edges, zigzag rows
    /// and the diamond checkerboard pick the upstream splits.
    #[test]
    fn diagonal_options_pick_the_upstream_splits() {
        let heights: Vec<Real> = (0..9).map(|i| i as Real).collect();
        let build = |o: HeightfieldOptions| {
            Heightfield::with_options(o, heights.clone(), 3, 3, Vector3::ONE)
        };

        // Default: (v00,v01,v10) + (v10,v01,v11) per cell.
        let d = build(HeightfieldOptions::default());
        assert_eq!(tri_ids(&d, 0), [0, 3, 1]);
        assert_eq!(tri_ids(&d, 1), [1, 3, 4]);
        // flip_quad_edges: (v00,v01,v11) + (v00,v11,v10).
        let f = build(HeightfieldOptions {
            flip_quad_edges: true,
            ..HeightfieldOptions::default()
        });
        assert_eq!(tri_ids(&f, 0), [0, 3, 4]);
        assert_eq!(tri_ids(&f, 1), [0, 4, 1]);
        // zigzag: rows alternate (row 0 flipped, row 1 default). Cells
        // iterate row-major: cell (0,0) tris 0-1, cell (1,0) tris 2-3, cell
        // (0,1) tris 4-5, cell (1,1) tris 6-7.
        let z = build(HeightfieldOptions {
            use_zigzag_subdivision: true,
            ..HeightfieldOptions::default()
        });
        assert_eq!(tri_ids(&z, 0), [0, 3, 4]);
        assert_eq!(tri_ids(&z, 2), [1, 4, 5]);
        assert_eq!(tri_ids(&z, 4), [3, 6, 4]);
        // diamond: cells with even (row+col) flip (cells (0,0) and (1,1)
        // flipped, (1,0) and (0,1) default).
        let m = build(HeightfieldOptions {
            use_diamond_subdivision: true,
            ..HeightfieldOptions::default()
        });
        assert_eq!(tri_ids(&m, 0), [0, 3, 4]);
        assert_eq!(tri_ids(&m, 2), [1, 4, 2]);
        assert_eq!(tri_ids(&m, 5), [4, 6, 7]);
        assert_eq!(tri_ids(&m, 7), [4, 8, 5]);
    }

    #[test]
    fn flip_triangle_winding_reverses_every_triangle() {
        let heights: Vec<Real> = (0..9).map(|i| i as Real).collect();
        let plain = Heightfield::new(heights.clone(), 3, 3, Vector3::ONE);
        let flipped = Heightfield::with_options(
            HeightfieldOptions {
                flip_triangle_winding: true,
                ..HeightfieldOptions::default()
            },
            heights,
            3,
            3,
            Vector3::ONE,
        );
        for t in 0..8 {
            let (a0, a1, a2) = plain.as_mesh().triangle(t);
            let (b0, b1, b2) = flipped.as_mesh().triangle(t);
            assert_eq!(a0, b2);
            assert_eq!(a1, b1);
            assert_eq!(a2, b0);
        }
    }

    #[test]
    fn vertex_layout_matches_upstream_centering() {
        let heights: Vec<Real> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let h = Heightfield::new(heights, 3, 3, Vector3::ONE);
        // Mid of min/max = 4; corner (0,0) -> (-1, -4, -1).
        let (v0, _, _) = h.as_mesh().triangle(0);
        assert_relative_eq!(v0.x, -1.0);
        assert_relative_eq!(v0.y, -4.0);
        assert_relative_eq!(v0.z, -1.0);
        // Grid point (1,1) = height 4 = mid, sits at the origin.
        for t in 0..8 {
            let (a, b, c) = h.as_mesh().triangle(t);
            for p in [a, b, c] {
                if p.y == 0.0 {
                    assert_relative_eq!(p.x, 0.0);
                    assert_relative_eq!(p.z, 0.0);
                    return;
                }
            }
        }
        panic!("center vertex not found");
    }
}
