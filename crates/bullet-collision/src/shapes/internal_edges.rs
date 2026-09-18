//! Internal-edge utility — a port of Bullet's `btInternalEdgeUtility`
//! (`btGenerateInternalEdgeInfo` + `btAdjustInternalEdgeContacts`,
//! btInternalEdgeUtility.cpp) with its `btTriangleInfoMap` data model.
//!
//! The utility removes the false contact normals a convex picks up when it
//! touches the *interior* shared edges/corners of a triangle mesh (edge and
//! corner popping, Bullet issue #27): every triangle records the dihedral
//! correction angle of its shared edges at mesh-finalize time, and contact
//! normals that land on a shared edge are rotated onto the
//! adjacent-triangle-aware direction before they enter the manifold.
//!
//! Reference decisions covered (btInternalEdgeUtility.cpp / btTriangleInfoMap.h):
//! - Map defaults verbatim: `convexEpsilon 0`, `planarEpsilon 1e-4`,
//!   `equalVertexThreshold 1e-8`, `edgeDistanceThreshold 0.1`,
//!   `zeroAreaThreshold 1e-8`, `maxEdgeAngleThreshold 2π`; unshared edge
//!   angles start at 2π so the `|angle| < max` gate rejects them.
//! - Connectivity: candidate triangles come from a BVH query with the
//!   triangle's *own* AABB (unexpanded — a shared-edge neighbor's AABB
//!   shares an edge with bit-identical coordinates, and the query is a
//!   closed-interval overlap); self skipped; both cross-square degenerate
//!   skips; shared-vertex scan at `equalVertexThreshold`; `numshared == 3`
//!   aborts the pair; `numshared == 2` runs the edge branch with the
//!   V0V2→V2V0 order fixup, the `edgeCrossA/B` direction flips toward the
//!   opposite vertex, the planar gate, `atan2` angle, `isConvex = nA·ecB < 0`,
//!   `corrected = ±(π − angle2)`, and the `sumvertsA ∈ {1,2,3}` write-out of
//!   `−corrected` + CONVEX + SWAP_NORMALB flags (SWAP decided by rotating
//!   `normalA` through `quat(edge, −corrected)` and comparing with
//!   `normalB`).
//! - Adjustment flow verbatim: nearest-edge selection among the
//!   `|angle| < maxEdgeAngleThreshold` edges, the `bestedge == k` guard (at
//!   most one edge branch can touch the normal), `edgeDistanceThreshold`,
//!   `angle == 0 → concave hit`, else the SWAP-normal computation, the
//!   back-facing test against `convexEpsilon`, `btClampNormal`'s one-sided
//!   angle window (`cur = atan2(n·ec, n·t)`), and the front-facing gate
//!   `clamped·triNormal > 0`; the concave tail snaps the normal to
//!   `triNormal` and bails when it opposes the contact normal (`dot < 0`).
//! - NOT ported (deviations §1): the three `normalAdjustFlags` modes
//!   (upstream marks the double-sided modes experimental, its own callers
//!   pass 0) — the single-sided, front-facing route is hard-wired; and the
//!   `m_positionWorldOnB` re-projection (our contact anchor already *is*
//!   the surface point; the triangle plane does not move when only the
//!   normal rotates).
//! - Normal direction: Bullet's `m_normalWorldOnB` on this path points
//!   **mesh → convex** (the manifold's body0 is the convex — every upstream
//!   caller passes `(cp, colObj1, colObj0, partId1, index1)`, making the
//!   triangle side body1; and `normalInB = m_cachedSeparatingAxis` is the
//!   B→A separation direction). [`adjust_internal_edge_contacts`] therefore
//!   takes/returns that convention, and the narrow-phase caller flips
//!   to/from our convex → mesh normals.
//! - `btHashMap` is replaced by a direct-indexed `Vec` (the workspace's
//!   hash-collection removal): the key is the dense global triangle index,
//!   an O(1) slot read; triangles without a shared edge keep the default
//!   `2π` entries, which the adjustment gate rejects exactly like Bullet's
//!   "in the map but this edge has no neighbor" case.

use bullet_math::{compat::quat_from_axis_angle, math, Real, Vector3};
use super::bvh::Bvh;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Per-triangle edge-angle info (Bullet's `btTriangleInfo`): the recorded
/// dihedral correction angle per shared edge and the CONVEX/SWAP flags.
/// An unshared edge keeps the 2π default, which every adjustment gate
/// rejects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleInfo {
    /// `V0V1_CONVEX` / `V1V2_CONVEX` / `V2V0_CONVEX` / `*_SWAP_NORMALB` bits.
    pub flags: u32,
    /// Shared-edge correction angle on the V0→V1 edge (`m_edgeV0V1Angle`).
    pub angle_v0v1: Real,
    /// Shared-edge correction angle on the V1→V2 edge (`m_edgeV1V2Angle`).
    pub angle_v1v2: Real,
    /// Shared-edge correction angle on the V2→V0 edge (`m_edgeV2V0Angle`).
    pub angle_v2v0: Real,
}

impl TriangleInfo {
    /// Bullet's constructor: no shared edges yet.
    pub const UNBOUND: Self = Self {
        flags: 0,
        angle_v0v1: 2.0 * bullet_math::PI,
        angle_v1v2: 2.0 * bullet_math::PI,
        angle_v2v0: 2.0 * bullet_math::PI,
    };
}

/// Flag: the V0→V1 shared edge is convex (`TRI_INFO_V0V1_CONVEX`).
pub const V0V1_CONVEX: u32 = 1;
/// Flag: the V1→V2 shared edge is convex (`TRI_INFO_V1V2_CONVEX`).
pub const V1V2_CONVEX: u32 = 2;
/// Flag: the V2→V0 shared edge is convex (`TRI_INFO_V2V0_CONVEX`).
pub const V2V0_CONVEX: u32 = 4;
/// Flag: the adjacent normal on V0→V1 came out inverted
/// (`TRI_INFO_V0V1_SWAP_NORMALB`).
pub const V0V1_SWAP_NORMALB: u32 = 8;
/// Flag: the adjacent normal on V1→V2 came out inverted
/// (`TRI_INFO_V1V2_SWAP_NORMALB`).
pub const V1V2_SWAP_NORMALB: u32 = 16;
/// Flag: the adjacent normal on V2→V0 came out inverted
/// (`TRI_INFO_V2V0_SWAP_NORMALB`).
pub const V2V0_SWAP_NORMALB: u32 = 32;

/// Edge-angle table over a whole mesh (Bullet's `btTriangleInfoMap`),
/// direct-indexed by global triangle id — a plain `Vec` replaces Bullet's
/// `btHashMap` (see the module docs). The six tunables carry Bullet's
/// constructor defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleInfoMap {
    /// One entry per mesh triangle, indexed by global triangle id.
    pub infos: Vec<TriangleInfo>,
    /// Convex/back-facing dot threshold (`m_convexEpsilon`).
    pub convex_epsilon: Real,
    /// Planar (zero dihedral) threshold on `|edgeCrossA × edgeCrossB|²`
    /// (`m_planarEpsilon`).
    pub planar_epsilon: Real,
    /// Vertex-equality distance² for connectivity (`m_equalVertexThreshold`).
    pub equal_vertex_threshold: Real,
    /// Max contact-to-edge distance for an edge hit
    /// (`m_edgeDistanceThreshold`).
    pub edge_distance_threshold: Real,
    /// Degenerate-triangle cross-square threshold (`m_zeroAreaThreshold`).
    pub zero_area_threshold: Real,
    /// Edges with a larger dihedral angle are ignored
    /// (`m_maxEdgeAngleThreshold`).
    pub max_edge_angle_threshold: Real,
}

impl TriangleInfoMap {
    /// Bullet's `btTriangleInfoMap` constructor defaults.
    pub fn new(triangle_count: usize) -> Self {
        Self {
            infos: vec![TriangleInfo::UNBOUND; triangle_count],
            convex_epsilon: 0.0,
            planar_epsilon: 0.0001,
            equal_vertex_threshold: 0.0001 * 0.0001,
            edge_distance_threshold: 0.1,
            zero_area_threshold: 0.0001 * 0.0001,
            max_edge_angle_threshold: 2.0 * bullet_math::PI,
        }
    }

    /// Build the edge-angle table for a mesh (Bullet's
    /// `btGenerateInternalEdgeInfo` driving its `btConnectivityProcessor`
    /// over every triangle). `tri_aabbs`/`bvh` are the mesh's prebuilt
    /// tables; the candidate scan uses each triangle's own AABB, so
    /// bit-identical shared-vertex coordinates make the closed-interval
    /// box test hit the neighbors without any epsilon inflation.
    pub(crate) fn build(
        vertices: &[Vector3],
        indices: &[[u32; 3]],
        tri_aabbs: &[bullet_math::Aabb],
        bvh: &Bvh,
    ) -> Self {
        let mut map = Self::new(indices.len());
        let mut candidates: Vec<u32> = Vec::new();
        for (a, index) in indices.iter().enumerate() {
            let tri_a = [
                vertices[index[0] as usize],
                vertices[index[1] as usize],
                vertices[index[2] as usize],
            ];
            // The A-side degenerate check is candidate-independent in Bullet
            // (it re-tests the same cross-square for every B) — hoisting it
            // out skips the whole triangle exactly like the per-pair skips.
            if cross_square(&tri_a) < map.equal_vertex_threshold {
                continue;
            }
            candidates.clear();
            bvh.triangles_in_aabb(&tri_aabbs[a], &mut candidates);
            for &b in &candidates {
                let b = b as usize;
                if b == a {
                    continue;
                }
                let index_b = &indices[b];
                let tri_b = [
                    vertices[index_b[0] as usize],
                    vertices[index_b[1] as usize],
                    vertices[index_b[2] as usize],
                ];
                if cross_square(&tri_b) < map.equal_vertex_threshold {
                    continue;
                }
                connect_pair(&mut map, a, &tri_a, &tri_b);
            }
        }
        map
    }

    /// The recorded info of triangle `index`.
    #[inline]
    pub fn info(&self, index: usize) -> Option<&TriangleInfo> {
        self.infos.get(index)
    }
}

/// `((v1 − v0) × (v2 − v0))²`, the degenerate-triangle test
/// (`crossBSqr` / `crossASqr` in `btConnectivityProcessor`).
#[inline]
fn cross_square(tri: &[Vector3; 3]) -> Real {
    (tri[1] - tri[0]).cross(tri[2] - tri[0]).length_squared()
}

/// Triangle normal, Bullet's `btTriangleShape::calcNormal`.
#[inline]
fn calc_normal(tri: &[Vector3; 3]) -> Vector3 {
    (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize()
}

/// Bullet's `btGetAngle(edgeA, normalA, normalB)`:
/// `atan2(swing·refAxis0, swing·refAxis1)`.
#[inline]
fn get_angle(ref_axis0: Vector3, ref_axis1: Vector3, swing: Vector3) -> Real {
    math::atan2(swing.dot(ref_axis0), swing.dot(ref_axis1))
}

/// One candidate pair of the connectivity scan (the `numshared == 2`
/// branch of `btConnectivityProcessor::processTriangle`), writing the
/// shared-edge entry of triangle `a`.
fn connect_pair(map: &mut TriangleInfoMap, a: usize, tri_a: &[Vector3; 3], tri_b: &[Vector3; 3]) {
    let threshold = map.equal_vertex_threshold;
    let mut shared_a = [0usize; 2];
    let mut shared_b = [0usize; 2];
    let mut numshared = 0usize;
    for (i, va) in tri_a.iter().enumerate() {
        for (j, vb) in tri_b.iter().enumerate() {
            if (*va - *vb).length_squared() < threshold {
                shared_a[numshared] = i;
                shared_b[numshared] = j;
                numshared += 1;
                if numshared >= 3 {
                    // Degenerate overlap (Bullet aborts the pair).
                    return;
                }
            }
        }
    }
    if numshared != 2 {
        // 0 or 1 shared vertices: nothing to record.
        return;
    }
    // Keep the edge reading V2V0 and not V0V2 so the signs are correct.
    if shared_a[0] == 0 && shared_a[1] == 2 {
        shared_a.swap(0, 1);
        shared_b.swap(0, 1);
    }
    let sum_verts_a = shared_a[0] + shared_a[1];
    let other_index_a = 3 - sum_verts_a;
    let other_index_b = 3 - (shared_b[0] + shared_b[1]);

    let mut edge = tri_a[shared_a[1]] - tri_a[shared_a[0]];
    let normal_a = calc_normal(tri_a);
    // btTriangleShape tB(triangle[sb1], triangle[sb0], triangle[otherB]).
    let normal_b = calc_normal(&[tri_b[shared_b[1]], tri_b[shared_b[0]], tri_b[other_index_b]]);
    edge = edge.normalize();
    let mut edge_cross_a = edge.cross(normal_a).normalize();
    if edge_cross_a.dot(tri_a[other_index_a] - tri_a[shared_a[0]]) < 0.0 {
        edge_cross_a = -edge_cross_a;
    }
    let mut edge_cross_b = edge.cross(normal_b).normalize();
    if edge_cross_b.dot(tri_b[other_index_b] - tri_b[shared_b[0]]) < 0.0 {
        edge_cross_b = -edge_cross_b;
    }

    let mut corrected_angle = 0.0;
    let mut is_convex = false;
    let calculated_edge = edge_cross_a.cross(edge_cross_b);
    if calculated_edge.length_squared() >= map.planar_epsilon {
        let calculated_edge = calculated_edge.normalize();
        let calculated_normal_a = calculated_edge.cross(edge_cross_a).normalize();
        let angle2 = get_angle(calculated_normal_a, edge_cross_a, edge_cross_b);
        let ang4 = bullet_math::PI - angle2;
        is_convex = normal_a.dot(edge_cross_b) < 0.0;
        corrected_angle = if is_convex { ang4 } else { -ang4 };
    }

    let (edge_wing, angle_slot, convex_flag, swap_flag) = match sum_verts_a {
        1 => (
            tri_a[0] - tri_a[1],
            0usize,
            V0V1_CONVEX,
            V0V1_SWAP_NORMALB,
        ),
        2 => (
            tri_a[2] - tri_a[0],
            2usize,
            V2V0_CONVEX,
            V2V0_SWAP_NORMALB,
        ),
        _ => (
            tri_a[1] - tri_a[2],
            1usize,
            V1V2_CONVEX,
            V1V2_SWAP_NORMALB,
        ),
    };
    let info = &mut map.infos[a];
    // btQuaternion(axis, angle) divides by the axis length internally; glam's
    // from_axis_angle expects a unit axis, hence the normalize.
    let orn = quat_from_axis_angle(edge_wing.normalize(), -corrected_angle);
    let computed_normal_b = orn * normal_a;
    if computed_normal_b.dot(normal_b) < 0.0 {
        // Bullet also flips the computed normal here; the value is only used
        // for this check, the flag is what carries over.
        info.flags |= swap_flag;
    }
    match angle_slot {
        0 => info.angle_v0v1 = -corrected_angle,
        1 => info.angle_v1v2 = -corrected_angle,
        _ => info.angle_v2v0 = -corrected_angle,
    }
    if is_convex {
        info.flags |= convex_flag;
    }
}

/// Closest point to `point` on the segment `line0`–`line1`, clamped to the
/// endpoints (Bullet's `btNearestPointInLineSegment`).
fn nearest_point_on_segment(line0: Vector3, line1: Vector3, point: Vector3) -> Vector3 {
    let line_delta = line1 - line0;
    if line_delta.x.abs() < Real::EPSILON
        && line_delta.y.abs() < Real::EPSILON
        && line_delta.z.abs() < Real::EPSILON
    {
        return line0;
    }
    let mut t = (point - line0).dot(line_delta) / line_delta.dot(line_delta);
    t = t.clamp(0.0, 1.0);
    line0 + line_delta * t
}

/// Bullet's `btClampNormal`: rotate `local_contact_normal` about `edge` onto
/// the boundary of the dihedral wedge described by `corrected_edge_angle`,
/// when it lies beyond it. The quaternion route normalizes the axis like
/// `btQuaternion(axis, angle)` does internally.
fn clamp_normal(
    edge: Vector3,
    tri_normal_org: Vector3,
    local_contact_normal: Vector3,
    corrected_edge_angle: Real,
) -> Option<Vector3> {
    let edge_cross = edge.cross(tri_normal_org).normalize();
    let cur_angle = get_angle(edge_cross, tri_normal_org, local_contact_normal);
    let out_of_wedge = if corrected_edge_angle < 0.0 {
        cur_angle < corrected_edge_angle
    } else {
        cur_angle > corrected_edge_angle
    };
    if out_of_wedge {
        let orn = quat_from_axis_angle(
            edge.normalize(),
            corrected_edge_angle - cur_angle,
        );
        return Some(orn * local_contact_normal);
    }
    None
}

/// Adjust one contact normal through the internal-edge utility (Bullet's
/// `btAdjustInternalEdgeContacts`, single-sided route with the
/// `normalAdjustFlags` fixed at 0 — see the module docs).
///
/// Everything runs in the mesh's local frame. `local_point` is the contact
/// point and `local_normal` the contact normal, **both in Bullet's
/// `m_normalWorldOnB` convention (mesh → convex)**; the return value is the
/// adjusted local normal, or `None` when the utility leaves the contact
/// alone (no shared edge nearby, front-facing convex edge, or the concave
/// snap opposing the contact).
pub(crate) fn adjust_internal_edge_contacts(
    map: &TriangleInfoMap,
    tri_index: usize,
    tri: (Vector3, Vector3, Vector3),
    local_point: Vector3,
    local_normal: Vector3,
) -> Option<Vector3> {
    let info = map.infos.get(tri_index)?;
    let tri = [tri.0, tri.1, tri.2];
    let tri_normal = calc_normal(&tri);
    // Edge k runs from edges[k].0 to edges[k].1; the clamp/record edge axis
    // is the reversed difference (Bullet: `edge(v0−v1)` / `edge(v1−v2)` /
    // `edge(v2−v0)`).
    let edges = [(0usize, 1usize), (1usize, 2usize), (2usize, 0usize)];
    let angles = [info.angle_v0v1, info.angle_v1v2, info.angle_v2v0];

    // Closest candidate edge (Bullet's bestedge/disttobestedge scan).
    let mut best_edge = usize::MAX;
    let mut best_distance = Real::INFINITY;
    for (k, &(i, j)) in edges.iter().enumerate() {
        if angles[k].abs() >= map.max_edge_angle_threshold {
            continue;
        }
        let nearest = nearest_point_on_segment(tri[i], tri[j], local_point);
        let len = (local_point - nearest).length();
        if len < best_distance {
            best_edge = k;
            best_distance = len;
        }
    }

    let mut is_near_edge = false;
    let mut concave_hits = 0u32;
    let mut adjusted: Option<Vector3> = None;
    for (k, &(i, j)) in edges.iter().enumerate() {
        if angles[k].abs() >= map.max_edge_angle_threshold {
            continue;
        }
        let nearest = nearest_point_on_segment(tri[i], tri[j], local_point);
        let len = (local_point - nearest).length();
        if len >= map.edge_distance_threshold || best_edge != k {
            continue;
        }
        is_near_edge = true;
        if angles[k] == 0.0 {
            concave_hits += 1;
            continue;
        }
        let (convex_flag, swap_flag) = match k {
            0 => (V0V1_CONVEX, V0V1_SWAP_NORMALB),
            1 => (V1V2_CONVEX, V1V2_SWAP_NORMALB),
            _ => (V2V0_CONVEX, V2V0_SWAP_NORMALB),
        };
        let is_edge_convex = info.flags & convex_flag != 0;
        let swap_factor = if is_edge_convex { 1.0 } else { -1.0 };
        let n_a = tri_normal * swap_factor;
        let edge_axis = match k {
            0 => tri[0] - tri[1],
            1 => tri[1] - tri[2],
            _ => tri[2] - tri[0],
        };
        let orn = quat_from_axis_angle(edge_axis.normalize(), angles[k]);
        let mut computed_normal_b = orn * tri_normal;
        if info.flags & swap_flag != 0 {
            computed_normal_b = -computed_normal_b;
        }
        let n_b = computed_normal_b * swap_factor;
        let back_facing = local_normal.dot(n_a) < map.convex_epsilon
            && local_normal.dot(n_b) < map.convex_epsilon;
        if back_facing {
            concave_hits += 1;
            continue;
        }
        if let Some(clamped) =
            clamp_normal(edge_axis, tri_normal * swap_factor, local_normal, angles[k])
        {
            // Front-facing gate; the BT_TRIANGLE_CONVEX_DOUBLE_SIDED flag is
            // not ported (single-sided route, frontFacing = 1).
            if clamped.dot(tri_normal) > 0.0 {
                adjusted = Some(clamped);
            }
        }
    }

    if is_near_edge && concave_hits > 0 {
        // Concave tail: snap to the triangle normal unless it opposes the
        // contact (the BT_TRIANGLE_CONCAVE_DOUBLE_SIDED flag is not ported).
        if tri_normal.dot(local_normal) < 0.0 {
            return None;
        }
        return Some(tri_normal);
    }
    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::TriangleMesh;
    use approx::assert_relative_eq;
    use bullet_math::PI;

    fn mesh(vertices: Vec<Vector3>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        let mut m = TriangleMesh::new(vertices, indices);
        m.generate_internal_edge_info();
        m
    }

    /// Coplanar pair sharing the diagonal V2–V0 of triangle 0.
    fn coplanar_pair() -> TriangleMesh {
        mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    /// A horizontal triangle A folded against a 30°-tilted neighbor along
    /// A's V0–V1 edge. `up` picks the concave variant (the neighbor rising
    /// toward A's side of the edge) or the convex one (falling away, both
    /// coordinates of the free vertex negative).
    fn fold(up: bool) -> TriangleMesh {
        let c = 30f64.to_radians().cos() as Real;
        let s = 30f64.to_radians().sin() as Real;
        let (y, z) = if up { (c, s) } else { (-c, -s) };
        mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 1.0, 0.0),
                Vector3::new(0.0, y, z),
            ],
            vec![[0, 1, 2], [0, 1, 3]],
        )
    }

    #[test]
    fn build_coplanar_pair_records_zero_angle() {
        let m = coplanar_pair();
        let map = m.internal_edge_info().unwrap();
        // Triangle 0 shares its V2–V0 edge (the diagonal) with triangle 1.
        let info = map.info(0).unwrap();
        assert_eq!(info.angle_v2v0, 0.0);
        assert_eq!(info.flags & V2V0_CONVEX, 0);
        assert_eq!(info.flags & V2V0_SWAP_NORMALB, 0);
        // The untouched edges stay unbound at 2π.
        assert_relative_eq!(info.angle_v0v1, 2.0 * PI);
        // Triangle 1 records the same edge from its own side (V0–V1).
        assert_eq!(map.info(1).unwrap().angle_v0v1, 0.0);
    }

    #[test]
    fn build_fold_records_angle_convexity_and_swap() {
        // 30° fold: Bullet records the dihedral through `atan2` + the
        // `π − angle2` correction, measured against the swapped-order
        // neighbor triangle (`tB = (B[sb1], B[sb0], B[other])`). Concave
        // neighbor: +5π/6, no flags; convex neighbor: −π/6 with CONVEX.
        let concave = fold(true);
        let info = concave.internal_edge_info().unwrap().info(0).unwrap();
        assert_relative_eq!(info.angle_v0v1, 5.0 * PI / 6.0, epsilon = 1e-6);
        assert_eq!(info.flags & V0V1_CONVEX, 0);
        assert_eq!(info.flags & V0V1_SWAP_NORMALB, 0);

        let convex = fold(false);
        let info = convex.internal_edge_info().unwrap().info(0).unwrap();
        assert_relative_eq!(info.angle_v0v1, -PI / 6.0, epsilon = 1e-6);
        assert_eq!(info.flags & V0V1_CONVEX, V0V1_CONVEX);
        assert_eq!(info.flags & V0V1_SWAP_NORMALB, 0);
    }

    #[test]
    fn build_single_triangle_stays_unbound() {
        let m = mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2]],
        );
        let map = m.internal_edge_info().unwrap();
        assert_eq!(map.info(0).unwrap(), &TriangleInfo::UNBOUND);
    }

    #[test]
    fn generate_is_idempotent() {
        let mut m = coplanar_pair();
        let first = m.internal_edge_info().unwrap().clone();
        m.generate_internal_edge_info();
        assert_eq!(m.internal_edge_info(), Some(&first));
    }

    #[test]
    fn adjust_coplanar_seam_snaps_to_face_normal() {
        let m = coplanar_pair();
        let map = m.internal_edge_info().unwrap();
        let tri = m.triangle(0);
        // On the shared diagonal, contact normal straight up (mesh → convex
        // convention): the concave tail snaps it to the face normal.
        let adjusted = adjust_internal_edge_contacts(
            map,
            0,
            tri,
            Vector3::new(0.5, 0.5, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(adjusted, Some(Vector3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn adjust_ignores_contacts_far_from_edges() {
        let m = coplanar_pair();
        let map = m.internal_edge_info().unwrap();
        let adjusted = adjust_internal_edge_contacts(
            map,
            0,
            m.triangle(0),
            Vector3::new(0.8, 0.1, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(adjusted, None);
    }

    #[test]
    fn adjust_concave_snap_bails_on_opposing_normal() {
        let m = coplanar_pair();
        let map = m.internal_edge_info().unwrap();
        // Contact from below: the face normal opposes the contact normal,
        // Bullet's `dot < 0` bail keeps the original.
        let adjusted = adjust_internal_edge_contacts(
            map,
            0,
            m.triangle(0),
            Vector3::new(0.5, 0.5, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        );
        assert_eq!(adjusted, None);
    }

    #[test]
    fn adjust_convex_edge_front_facing_contact_untouched() {
        let m = fold(false);
        let map = m.internal_edge_info().unwrap();
        // On the convex shared edge, contact normal along the face normal:
        // inside the wedge, the clamp window rejects it and nothing changes.
        let adjusted = adjust_internal_edge_contacts(
            map,
            0,
            m.triangle(0),
            Vector3::new(0.2, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(adjusted, None);
    }

    #[test]
    fn clamp_normal_rotates_onto_the_wedge_boundary() {
        // n sits 15.5° beyond a −60° wedge about the z axis; clamping
        // rotates it back exactly onto the boundary.
        let n = Vector3::new(0.25, -0.968, 0.0);
        let clamped = clamp_normal(
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 0.0),
            n,
            -60f64.to_radians() as Real,
        )
        .unwrap();
        assert_relative_eq!(clamped.x, 0.5, epsilon = 1e-3);
        assert_relative_eq!(clamped.y, -0.866, epsilon = 1e-3);
        // Boundary property: the clamped normal now measures exactly the
        // corrected angle.
        let edge_cross = Vector3::new(0.0, 0.0, 1.0).cross(Vector3::new(1.0, 0.0, 0.0));
        let cur = get_angle(edge_cross, Vector3::new(1.0, 0.0, 0.0), clamped);
        assert_relative_eq!(cur, -60f64.to_radians() as Real, epsilon = 1e-4);
    }
}
