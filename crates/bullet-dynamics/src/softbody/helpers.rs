//! Soft-body builders — port of `btSoftBodyHelpers` (ropes, patches and the
//! volumetric builders: ellipsoid / triangle mesh / convex hull).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeSet;
#[cfg(feature = "std")]
use std::collections::BTreeSet;
use bullet_math::{math, Real, Vector3};

use super::body::SoftBody;
use super::quickhull::quick_hull;
use super::rng::MsvcRand;

/// Bit flag for [`create_rope`]: pin the first node (Bullet's `fixeds & 1`).
pub const ROPE_FIX_START: u32 = 1;
/// Bit flag for [`create_rope`]: pin the last node (Bullet's `fixeds & 2`).
pub const ROPE_FIX_END: u32 = 2;

/// Pin flags for [`create_patch`] (Bullet's `fixeds` bits 0-8).
pub mod patch_fixeds {
    /// Corner (0, 0).
    pub const CORNER_00: u32 = 1;
    /// Corner (rx-1, 0).
    pub const CORNER_10: u32 = 2;
    /// Corner (0, ry-1).
    pub const CORNER_01: u32 = 4;
    /// Corner (rx-1, ry-1).
    pub const CORNER_11: u32 = 8;
    /// Edge midpoint ((rx-1)/2, 0).
    pub const EDGE_X0_MID: u32 = 16;
    /// Edge midpoint (0, (ry-1)/2).
    pub const EDGE_Y0_MID: u32 = 32;
    /// Edge midpoint (rx-1, (ry-1)/2).
    pub const EDGE_X1_MID: u32 = 64;
    /// Edge midpoint ((rx-1)/2, ry-1).
    pub const EDGE_Y1_MID: u32 = 128;
    /// Center ((rx-1)/2, (ry-1)/2).
    pub const CENTER: u32 = 256;
}

/// Create a rope of `res + 2` unit-mass nodes strung between `from` and
/// `to`, chained by structural links (Bullet's
/// `btSoftBodyHelpers::CreateRope`). `fixeds` combines
/// [`ROPE_FIX_START`]/[`ROPE_FIX_END`] to pin the ends (mass 0).
pub fn create_rope(from: Vector3, to: Vector3, res: usize, fixeds: u32) -> SoftBody {
    // Create nodes.
    let r = res + 2;
    let one: Real = 1.0;
    let mut positions = Vec::with_capacity(r);
    let masses = vec![one; r];
    for i in 0..r {
        let t = i as Real / (r - 1) as Real;
        positions.push(from.lerp(to, t));
    }
    let mut psb = SoftBody::new(&positions, &masses);
    if fixeds & ROPE_FIX_START != 0 {
        psb.set_mass(0, 0.0);
    }
    if fixeds & ROPE_FIX_END != 0 {
        psb.set_mass(r - 1, 0.0);
    }
    // Create links.
    for i in 1..r {
        psb.append_link(i - 1, i, None, false);
    }
    psb
}

/// Create a `resx x resy` patch of unit-mass nodes bilinearly interpolated
/// between the four corners, with structural links (horizontal, vertical,
/// plus the diagonal per cell when `gendiags`) and two faces per cell
/// (Bullet's `btSoftBodyHelpers::CreatePatch`). `fixeds` combines the
/// [`patch_fixeds`] bits to pin nodes. The append order (links and faces,
/// interleaved per cell) matches Bullet exactly — PBD is order-sensitive.
///
/// `perturbation` jitters the y of both edge interpolants using MSVC's
/// `rand()` sequence (Bullet calls the CRT `rand()`; only meaningful when
/// comparing against MSVC-built probes — pass 0 for a deterministic patch).
#[allow(clippy::too_many_arguments)]
pub fn create_patch(
    corner00: Vector3,
    corner10: Vector3,
    corner01: Vector3,
    corner11: Vector3,
    resx: usize,
    resy: usize,
    fixeds: u32,
    gendiags: bool,
    perturbation: Real,
) -> SoftBody {
    assert!(resx >= 2 && resy >= 2, "patch resolution must be >= 2");
    let idx = |x: usize, y: usize| y * resx + x;
    // Create nodes (y first, then x — Bullet's row-major IDX layout). The
    // C++ draws two rand() per node (both edges) unconditionally; the draw
    // only shapes geometry when perturbation != 0, so the deterministic
    // path skips it.
    let mut positions = Vec::with_capacity(resx * resy);
    let masses = vec![1.0 as Real; resx * resy];
    let mut rng = MsvcRand::default_seeded();
    for iy in 0..resy {
        let ty = iy as Real / (resy - 1) as Real;
        let py0 = corner00.lerp(corner01, ty);
        let py1 = corner10.lerp(corner11, ty);
        for ix in 0..resx {
            let tx = ix as Real / (resx - 1) as Real;
            let (temp, temp1) = if perturbation != 0.0 {
                let mut temp1 = py1;
                temp1.y += perturbation * rng.next_rand() as Real / 32767.0;
                let mut temp = py0;
                temp.y += perturbation * rng.next_rand() as Real / 32767.0;
                (temp, temp1)
            } else {
                (py0, py1)
            };
            positions.push(temp.lerp(temp1, tx));
        }
    }
    let mut psb = SoftBody::new(&positions, &masses);
    {
        use patch_fixeds::*;
        if fixeds & CORNER_00 != 0 {
            psb.set_mass(idx(0, 0), 0.0);
        }
        if fixeds & CORNER_10 != 0 {
            psb.set_mass(idx(resx - 1, 0), 0.0);
        }
        if fixeds & CORNER_01 != 0 {
            psb.set_mass(idx(0, resy - 1), 0.0);
        }
        if fixeds & CORNER_11 != 0 {
            psb.set_mass(idx(resx - 1, resy - 1), 0.0);
        }
        if fixeds & EDGE_X0_MID != 0 {
            psb.set_mass(idx((resx - 1) / 2, 0), 0.0);
        }
        if fixeds & EDGE_Y0_MID != 0 {
            psb.set_mass(idx(0, (resy - 1) / 2), 0.0);
        }
        if fixeds & EDGE_X1_MID != 0 {
            psb.set_mass(idx(resx - 1, (resy - 1) / 2), 0.0);
        }
        if fixeds & EDGE_Y1_MID != 0 {
            psb.set_mass(idx((resx - 1) / 2, resy - 1), 0.0);
        }
        if fixeds & CENTER != 0 {
            psb.set_mass(idx((resx - 1) / 2, (resy - 1) / 2), 0.0);
        }
    }
    // Create links and faces (Bullet's per-cell append order: horizontal
    // link, vertical link, two faces, then the diagonal link).
    for iy in 0..resy {
        for ix in 0..resx {
            let mdx = ix + 1 < resx;
            let mdy = iy + 1 < resy;
            // Bullet's naming: node01 = x+1, node10 = y+1.
            let (n00, n01, n10, n11) =
                (idx(ix, iy), idx(ix + 1, iy), idx(ix, iy + 1), idx(ix + 1, iy + 1));
            if mdx {
                psb.append_link(n00, n01, None, false);
            }
            if mdy {
                psb.append_link(n00, n10, None, false);
            }
            if mdx && mdy {
                psb.append_face(n00, n10, n11, None);
                psb.append_face(n11, n01, n00, None);
                if gendiags {
                    psb.append_link(n00, n11, None, false);
                }
            }
        }
    }
    psb
}

/// Deterministic Hammersley point set on the unit sphere (Bullet's
/// `CreateEllipsoid::Hammersley::Generate`): the mirrored radical inverse
/// in z and `(pi + 2*pi*i)/n` in azimuth.
fn hammersley_generate(x: &mut [Vector3]) {
    let n = x.len();
    for (i, slot) in x.iter_mut().enumerate() {
        let mut p: Real = 0.5;
        let mut t: Real = 0.0;
        let mut j = i;
        while j != 0 {
            if j & 1 != 0 {
                t += p;
            }
            p *= 0.5;
            j >>= 1;
        }
        let w = 2.0 * t - 1.0;
        let a = (bullet_math::PI + 2.0 * i as Real * bullet_math::PI) / n as Real;
        let s = math::sqrt(1.0 - w * w);
        *slot = Vector3::new(s * math::cos(a), s * math::sin(a), w);
    }
}

/// Create a volumetric soft body as a hull over `3 + res` Hammersley
/// points on the scaled sphere (Bullet's `CreateEllipsoid`): the hull
/// builder produces the nodes (its vertices) and the link/face surface.
pub fn create_ellipsoid(center: Vector3, radius: Vector3, res: usize) -> SoftBody {
    let mut vtx = vec![Vector3::ZERO; 3 + res];
    hammersley_generate(&mut vtx);
    for v in &mut vtx {
        *v = *v * radius + center;
    }
    create_from_convex_hull(&vtx)
}

/// Create a soft body whose surface is the convex hull of `vertices`
/// (Bullet's `CreateFromConvexHull`): the hull vertices become the nodes,
/// every hull edge a structural link and every hull triangle a face. Call
/// [`SoftBody::set_total_mass_from_faces`] afterwards — the constructor
/// leaves all nodes free-floating at mass zero, exactly like Bullet.
pub fn create_from_convex_hull(vertices: &[Vector3]) -> SoftBody {
    let (verts, triangles) = quick_hull(vertices)
        .expect("create_from_convex_hull: degenerate point set (need 4 non-coplanar points)");
    let positions: Vec<Vector3> = verts.iter().map(|&i| vertices[i]).collect();
    let masses = vec![0.0 as Real; verts.len()];
    let mut psb = SoftBody::new(&positions, &masses);
    // Map hull vertex -> node index (hull vertices are index-ascending, so
    // the map is a running counter over the sorted subset).
    let mut remap = vec![usize::MAX; vertices.len()];
    for (node, &vi) in verts.iter().enumerate() {
        remap[vi] = node;
    }
    for t in &triangles {
        let idx = [remap[t[0]], remap[t[1]], remap[t[2]]];
        if idx[0] < idx[1] {
            psb.append_link(idx[0], idx[1], None, false);
        }
        if idx[1] < idx[2] {
            psb.append_link(idx[1], idx[2], None, false);
        }
        if idx[2] < idx[0] {
            psb.append_link(idx[2], idx[0], None, false);
        }
        psb.append_face(idx[0], idx[1], idx[2], None);
    }
    psb
}

/// Create a soft body from a triangle soup (Bullet's `CreateFromTriMesh`):
/// one node per vertex (mass zero, like Bullet) and one link per undirected
/// mesh edge plus one face per triangle.
pub fn create_from_tri_mesh(vertices: &[Vector3], triangles: &[u32]) -> SoftBody {
    let masses = vec![0.0 as Real; vertices.len()];
    let mut psb = SoftBody::new(vertices, &masses);
    // BTreeSet (no hasher) instead of HashSet: same dedup semantics at
    // modeling-time sizes.
    let mut seen = BTreeSet::new();
    let (triangles, _) = triangles.as_chunks::<3>();
    for t in triangles {
        let idx = [t[0] as usize, t[1] as usize, t[2] as usize];
        for j in 0..3 {
            let (a, b) = (idx[j], idx[(j + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if seen.insert(key) {
                psb.append_link(key.0, key.1, None, false);
            }
        }
        psb.append_face(idx[0], idx[1], idx[2], None);
    }
    psb
}
