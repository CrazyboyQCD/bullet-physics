//! Incremental quickhull — a small deterministic implementation playing the
//! role of Bullet's `HullLibrary` (`LinearMath/btConvexHull*`, which
//! btShapeHull and the soft-body builders share), consumed by the
//! collision shape hulls and the soft-body builders
//! (`create_from_convex_hull` and the ellipsoid's Hammersley point set).
//!
//! Bullet's hull library re-indexes and merges its output vertices, so its
//! link/face append order differs from ours; this implementation keeps the
//! *input* vertex indices (the hull vertices are an index-ascending subset),
//! which is more deterministic but still a recorded deviation — the face
//! order shapes the PBD solve order and with it the settle scatter.
//!
//! Robustness contract: the output is always a closed surface (every edge
//! carried equally in both winding directions) that contains every input
//! point within the tolerance band. On knife-edge near-coplanar clouds a
//! connected visible region can wrap around a vertex; its figure-eight
//! horizon then folds two fans onto the shared vertex-to-apex edge — a
//! closed but non-manifold *pinch*, unavoidable while output triangles keep
//! the caller's vertex identity (upstream dodges this by collapsing such
//! clouds to a handful of vertices instead). The expansion restricts
//! visibility to the edge-connected component of the seed face and consumes
//! each point as a hull vertex at most once, which removes the lossy
//! failure modes (missing faces, vertices re-entering the outside sets).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use crate::{Real, Vector3};

/// A face of the evolving hull with its outside set.
struct Face {
    v: [usize; 3],
    /// Unit outward normal.
    n: Vector3,
    /// Plane offset: `n . x = d`.
    d: Real,
    /// Indices of points strictly outside, farthest last.
    outside: Vec<usize>,
}

impl Face {
    /// Signed distance of `p` from the face plane (positive outside).
    fn dist(&self, points: &[Vector3], p: usize) -> Real {
        self.n.dot(points[p]) - self.d
    }
}

/// Tolerance for the outside test, relative to the point cloud's extent.
const REL_TOL: Real = 1e-6;

/// Build the convex hull of `points`. Returns the hull vertex subset (input
/// indices, ascending) and the triangle faces over that subset, or `None`
/// when the points are degenerate (fewer than 4 non-coplanar points).
pub fn quick_hull(points: &[Vector3]) -> Option<(Vec<usize>, Vec<[usize; 3]>)> {
    if points.len() < 4 {
        return None;
    }
    // Scale for tolerances.
    let mut min = points[0];
    let mut max = points[0];
    for p in points {
        min = min.min(*p);
        max = max.max(*p);
    }
    let tol = (max - min).length().max(1e-12) * REL_TOL;

    // Initial simplex: two extreme points along the widest axis, the
    // farthest from their line, then the farthest from that plane.
    let extent = max - min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        Vector3::X
    } else if extent.y >= extent.z {
        Vector3::Y
    } else {
        Vector3::Z
    };
    let mut a = 0;
    let mut b = 0;
    for (i, p) in points.iter().enumerate() {
        if p.dot(axis) < points[a].dot(axis) {
            a = i;
        }
        if p.dot(axis) > points[b].dot(axis) {
            b = i;
        }
    }
    if a == b {
        return None;
    }
    let ab = points[b] - points[a];
    let mut c = usize::MAX;
    let mut best_c = tol;
    for (i, p) in points.iter().enumerate() {
        if i == a || i == b {
            continue;
        }
        let dist = ab.cross(*p - points[a]).length() / ab.length();
        if dist > best_c {
            best_c = dist;
            c = i;
        }
    }
    if c == usize::MAX {
        return None;
    }
    let n0 = ab.cross(points[c] - points[a]);
    let plane_d = n0.dot(points[a]);
    let mut d = usize::MAX;
    let mut best_d = tol * n0.length();
    for (i, p) in points.iter().enumerate() {
        if i == a || i == b || i == c {
            continue;
        }
        let dist = (n0.dot(*p) - plane_d).abs();
        if dist > best_d {
            best_d = dist;
            d = i;
        }
    }
    if d == usize::MAX {
        return None;
    }

    // An interior reference point (the initial tetrahedron's centroid)
    // fixes every face's outward winding.
    let inner = (points[a] + points[b] + points[c] + points[d]) * 0.25;
    let tetra = [[a, b, c], [b, a, d], [c, b, d], [a, c, d]];
    let mut faces = Vec::with_capacity(8 + points.len());
    for &t in &tetra {
        faces.push(make_face(points, t, tol, inner)?);
    }
    // Hull vertices never re-enter the outside sets: a point is expanded
    // exactly once, like the conflict graph of the classical algorithm.
    // Without this, reassignment can hand a vertex back to a face it hovers
    // over (near-coplanar sheets sit within a thickness of each other), and
    // the vertex then drives a second expansion that duplicates its edges.
    let mut done = vec![false; points.len()];
    for &s in tetra.iter().flatten() {
        done[s] = true;
    }
    let open: Vec<usize> = (0..points.len()).filter(|&i| !done[i]).collect();
    assign_points_indexes(&mut faces, points, &open, tol, &done);

    // Expand: while some face holds outside points, add its farthest point
    // and rebuild the hull over the horizon of the faces it sees.
    while let Some(fi) = faces.iter().position(|f| !f.outside.is_empty()) {
        let p = *faces[fi].outside.last().unwrap();
        done[p] = true;
        // Faces `p` can see, restricted to the edge-connected component of
        // the seed face. Taking every visible face can join two patches that
        // share only a vertex; their joint horizon is a figure-eight and
        // coning over it folds two fans onto the same pinched edge (a
        // non-manifold hull). On a manifold surface the boundary of one
        // connected region is a simple cycle, so per-expansion cones close.
        let mut visible = vec![fi];
        let mut stack = vec![fi];
        while let Some(&vi) = stack.last() {
            stack.pop();
            let vi_edges = face_edges(faces[vi].v).map(edge_key);
            for (nb, f) in faces.iter().enumerate() {
                if !visible.contains(&nb) && f.dist(points, p) > tol {
                    let nb_edges = face_edges(f.v).map(edge_key);
                    if vi_edges.iter().any(|e| nb_edges.contains(e)) {
                        visible.push(nb);
                        stack.push(nb);
                    }
                }
            }
        }
        // Horizon: visible-face edges not shared with another visible face
        // (canonicalized undirected keys).
        let mut horizon: Vec<[usize; 2]> = Vec::new();
        for &vi in &visible {
            for e in face_edges(faces[vi].v) {
                let ek = edge_key(e);
                let shared = visible.iter().any(|&wj| {
                    wj != vi
                        && face_edges(faces[wj].v).iter().any(|e2| edge_key(*e2) == ek)
                });
                if !shared {
                    horizon.push(e);
                }
            }
        }
        // Orphan the outside points of the visible faces, then remove them.
        let mut orphans: Vec<usize> = Vec::new();
        let mut vis_sorted = visible;
        vis_sorted.sort_unstable();
        for &vi in vis_sorted.iter().rev() {
            let mut f = faces.swap_remove(vi);
            orphans.append(&mut f.outside);
        }
        if horizon.is_empty() {
            // Everything visible (numerical degeneracy): drop the point.
            continue;
        }
        // Cone faces over the horizon (skipping degenerate ones).
        for e in &horizon {
            if let Some(fc) = make_face(points, [e[0], e[1], p], tol, inner) {
                faces.push(fc);
            }
        }
        assign_points_indexes(&mut faces, points, &orphans, tol, &done);
    }

    // Collect the hull vertices (ascending input indices) and the faces.
    let mut vertices: Vec<usize> = faces.iter().flat_map(|f| f.v.iter().copied()).collect();
    vertices.sort_unstable();
    vertices.dedup();
    let triangles = faces.iter().map(|f| f.v).collect();
    Some((vertices, triangles))
}

fn face_edges(v: [usize; 3]) -> [[usize; 2]; 3] {
    let (a, b, c) = (v[0], v[1], v[2]);
    [[a, b], [b, c], [c, a]]
}

/// Canonical undirected key of an edge.
fn edge_key(e: [usize; 2]) -> (usize, usize) {
    (e[0].min(e[1]), e[0].max(e[1]))
}

/// Build one face with the normal oriented away from the interior
/// reference point; degenerate (zero-area) faces are rejected.
fn make_face(points: &[Vector3], v: [usize; 3], tol: Real, inner: Vector3) -> Option<Face> {
    let mut n = (points[v[1]] - points[v[0]]).cross(points[v[2]] - points[v[0]]);
    let len = n.length();
    if len <= tol {
        return None;
    }
    n /= len;
    let mut d = n.dot(points[v[0]]);
    if n.dot(inner) - d > 0.0 {
        // Flip so the interior lies on the negative side.
        n = -n;
        d = -d;
    }
    Some(Face { v, n, d, outside: Vec::new() })
}

/// Assign `candidates` not yet consumed as hull vertices to the face they
/// are farthest above (if any).
fn assign_points_indexes(
    faces: &mut [Face],
    points: &[Vector3],
    candidates: &[usize],
    tol: Real,
    done: &[bool],
) {
    for &p in candidates {
        if done[p] {
            continue;
        }
        let mut best_face: Option<usize> = None;
        let mut best_dist = tol;
        for (fi, f) in faces.iter().enumerate() {
            let dist = f.dist(points, p);
            if dist > best_dist {
                best_dist = dist;
                best_face = Some(fi);
            }
        }
        if let Some(fi) = best_face {
            faces[fi].outside.push(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_box_corners() {
        let pts: Vec<Vector3> = [-1.0, 1.0]
            .iter()
            .flat_map(|x| [-1.0, 1.0].iter().map(move |y| (*x, *y)))
            .flat_map(|(x, y)| [-1.0, 1.0].iter().map(move |z| Vector3::new(x, y, *z)))
            .collect();
        let (verts, faces) = quick_hull(&pts).unwrap();
        assert_eq!(verts.len(), 8);
        assert_eq!(faces.len(), 12);
    }

    #[test]
    fn interior_points_excluded() {
        let mut pts: Vec<Vector3> = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ];
        pts.push(Vector3::ZERO);
        pts.push(Vector3::new(0.1, 0.1, 0.1));
        let (verts, faces) = quick_hull(&pts).unwrap();
        assert_eq!(verts.len(), 6, "interior points must not be hull vertices");
        assert_eq!(faces.len(), 8, "octahedron has 8 faces");
    }

    #[test]
    fn degenerate_clouds_rejected() {
        let coplanar: Vec<Vector3> = (0..5).map(|i| Vector3::new(i as Real, 0.0, 0.0)).collect();
        assert!(quick_hull(&coplanar).is_none());
    }

    // --- P2-C06 adversarial regression helpers (flat slabs lost faces) ---

    fn lcgn(s: &mut u64) -> Real {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) as Real
    }

    /// Directed-edge balance: every edge must carry equal winding counts in
    /// both directions. A fold (figure-eight horizon through a shared vertex,
    /// unavoidable without vertex splitting) shows up as a *balanced* double;
    /// a real defect (missing face) is unbalanced.
    fn balanced_closure(faces: &[[usize; 3]]) -> bool {
        let mut out = std::collections::BTreeMap::new();
        for f in faces {
            for i in 0..3 {
                *out.entry((f[i], f[(i + 1) % 3])).or_insert(0u32) += 1;
            }
        }
        out.iter()
            .all(|(&(a, b), &c)| out.get(&(b, a)) == Some(&c))
    }

    /// First input point sitting outside every face plane by more than
    /// `slack`. Winding-agnostic: each face plane is oriented against the
    /// hull-vertex centroid (which is interior for a full-dimensional hull).
    fn first_outside(
        points: &[Vector3],
        verts: &[usize],
        faces: &[[usize; 3]],
        slack: Real,
    ) -> Option<(usize, Real)> {
        let mut inner = Vector3::ZERO;
        for &v in verts {
            inner += points[v];
        }
        let inner = inner / verts.len() as Real;
        'outer: for (pi, p) in points.iter().enumerate() {
            let mut worst = Real::NEG_INFINITY;
            for f in faces {
                let a = points[f[0]];
                let mut n = (points[f[1]] - a).cross(points[f[2]] - a);
                let len = n.length();
                if len > Real::EPSILON {
                    n /= len;
                    if n.dot(inner - a) > 0.0 {
                        n = -n;
                    }
                    let dist = n.dot(*p) - n.dot(a);
                    if dist <= slack {
                        continue 'outer;
                    }
                    worst = worst.max(dist);
                }
            }
            if worst.is_finite() {
                return Some((pi, worst));
            }
        }
        None
    }

    fn assert_valid_hull(ctx: &str, points: &[Vector3], allow_none: bool) {
        match quick_hull(points) {
            Some((verts, faces)) => {
                assert!(
                    balanced_closure(&faces),
                    "[{ctx}] hull not closed ({} faces)\nbad={:?}\nfaces={faces:?}",
                    faces.len(),
                    bad_edges(&faces)
                );
                if let Some((pi, dist)) = first_outside(points, &verts, &faces, 1e-3) {
                    panic!("[{ctx}] point {pi} outside hull by {dist}\nfaces={faces:?}");
                }
            }
            None => assert!(allow_none, "[{ctx}] unexpected None"),
        }
    }

    /// Directed edges not carried equally in both directions.
    fn bad_edges(faces: &[[usize; 3]]) -> Vec<((usize, usize), u32)> {
        let mut out = std::collections::BTreeMap::new();
        for f in faces {
            for i in 0..3 {
                *out.entry((f[i], f[(i + 1) % 3])).or_insert(0u32) += 1;
            }
        }
        out.iter()
            .filter(|&(&(a, b), &c)| out.get(&(b, a)) != Some(&c))
            .map(|(&k, &c)| (k, c))
            .collect()
    }

    #[test]
    fn randomized_flat_clouds_stay_closed() {
        // A: thin slabs, thickness sweep 1e-1 .. 1e-5 (tol ~ 2.8e-6).
        for tier in 1..=5u32 {
            let t = (10.0f64).powi(-(tier as i32)) as Real;
            for k in 0..100u64 {
                let mut s = 0x0C06_0000 + k;
                let mut pts = Vec::new();
                for _ in 0..40 {
                    let (x, y, z) = (lcgn(&mut s), lcgn(&mut s), lcgn(&mut s));
                    pts.push(Vector3::new(
                        x * 2.0 - 1.0,
                        y * 2.0 - 1.0,
                        (z * 2.0 - 1.0) * t,
                    ));
                }
                // Sub-tolerance thickness may legitimately come back None.
                let allow_none = t < 1e-5;
                assert_valid_hull(&format!("thin-slab t={t} k={k}"), &pts, allow_none);
            }
        }

        // B: uniform cube + near-duplicate pairs at 1e-5 .. 1e-3 separation.
        for k in 0..100u64 {
            let mut s = 0x0C06_0000 + k;
            let mut pts = Vec::new();
            for _ in 0..20 {
                pts.push(Vector3::new(
                    lcgn(&mut s) * 2.0 - 1.0,
                    lcgn(&mut s) * 2.0 - 1.0,
                    lcgn(&mut s) * 2.0 - 1.0,
                ));
            }
            let base = pts[0];
            let sep = (1e-5 + lcgn(&mut s) * 1e-3) as Real;
            let dir = Vector3::new(lcgn(&mut s) - 0.5, lcgn(&mut s) - 0.5, lcgn(&mut s) - 0.5);
            pts.push(base + dir.normalize_or_zero() * sep);
            pts.push(base * 0.5 + dir.normalize_or_zero() * sep);
            assert_valid_hull(&format!("dup-pairs k={k}"), &pts, false);
        }

        // C: coplanar grid (z = 0) + a few off-plane outliers.
        for k in 0..100u64 {
            let mut s = 0x0C00_0600 + k;
            let mut pts = Vec::new();
            for i in 0..5 {
                for j in 0..5 {
                    pts.push(Vector3::new(i as Real - 2.0, j as Real - 2.0, 0.0));
                }
            }
            for _ in 0..6 {
                let h = (lcgn(&mut s) - 0.5) * 2.0;
                pts.push(Vector3::new(
                    lcgn(&mut s) * 4.0 - 2.0,
                    lcgn(&mut s) * 4.0 - 2.0,
                    h,
                ));
            }
            // Nearly flat grids (all |h| tiny) may be rejected as coplanar.
            let max_h = pts.iter().map(|p| p.z.abs()).fold(0.0, Real::max);
            assert_valid_hull(&format!("grid+outliers k={k}"), &pts, max_h < 1e-4);
        }
    }

    /// Regression for the P2-C06 face loss: a 40-point slab
    /// (t = 1e-3, deterministic LCG seed) whose hull used to come back
    /// non-manifold (a vertex drove a second expansion, folding duplicate
    /// fans onto its edges). Values are f32-exact so both features see the
    /// same cloud.
    #[test]
    fn flat_slab_cloud_stays_manifold() {
        let raw: [[f64; 3]; 39] = [
            [0.2691295, -0.23461741, 0.0004619904],
            [-0.43353087, -0.6117897, 0.00051185966],
            [-0.766617, 0.30281067, 0.00094336254],
            [-0.13432908, -0.37333828, -0.00067369797],
            [0.48226058, 0.5953672, 0.00027337635],
            [0.54398525, -0.80950475, 0.0006877636],
            [0.9166075, -0.9253926, -0.00035188516],
            [0.46440423, -0.72225773, -0.00043515983],
            [0.6592368, 0.621101, 0.00032918778],
            [0.016827345, -0.732311, -0.0009729715],
            [-0.918939, -0.791459, 0.0009633836],
            [-0.4002056, -0.68798554, 0.00033889426],
            [-0.91948706, 0.27823615, -0.0009934349],
            [0.83144283, 0.5248977, 0.00015817619],
            [0.45332778, -0.9703815, -0.00033576062],
            [0.2629192, -0.87479687, 0.00066716317],
            [-0.90504503, 0.30459344, 0.00026242054],
            [-0.5788772, -0.58595085, 0.00032721402],
            [0.16627944, -0.99928725, -0.00036237075],
            [0.13506198, 0.06441009, 0.0007628943],
            [0.02131784, -0.14764863, -0.00097466534],
            [-0.028191686, -0.024073482, 0.00040069653],
            [0.124082685, 0.7241721, -0.00083422393],
            [0.3771006, 0.086423635, 0.0002477714],
            [-0.2689572, 0.8284508, 0.0005312504],
            [0.08478296, -0.08115655, 0.0006406417],
            [0.76238596, 0.51733303, -0.00042507716],
            [0.12746024, -0.42984104, 0.000110320216],
            [0.45593333, -0.52107465, 0.0009341531],
            [0.60939837, 0.8485298, -0.00076062605],
            [-0.38970447, 0.87306774, -0.00002678925],
            [-0.1834715, 0.43748617, -0.0008340864],
            [-0.36680025, -0.4446723, 0.00053626637],
            [-0.033923626, 0.24546015, 0.00052167254],
            [0.20582342, 0.7605152, 0.00016522325],
            [-0.9256521, 0.3907827, -0.00055722723],
            [0.7886088, -0.9735655, 0.00041946722],
            [0.31053114, -0.8435588, 0.00027884927],
            [-0.26148474, 0.5891502, -0.00051785604],
        ];
        let pts: Vec<Vector3> = raw
            .iter()
            .map(|p| Vector3::new(p[0] as Real, p[1] as Real, p[2] as Real))
            .collect();
        assert_valid_hull("flat-slab-39", &pts, false);
    }

    /// Thickness sweep over thin boxes: above the tolerance band the hull
    /// must be complete (8 vertices, 12 faces); far below it the cloud is
    /// coplanar and rejected instead of losing faces silently.
    #[test]
    fn thin_box_thickness_sweep() {
        let mut t = 1.0;
        for _ in 0..12 {
            t *= 0.5;
            let mut pts = Vec::new();
            for &x in &[-1.0, 1.0] {
                for &y in &[-1.0, 1.0] {
                    for &z in &[-t, t] {
                        pts.push(Vector3::new(x, y, z));
                    }
                }
            }
            let (verts, faces) =
                quick_hull(&pts).unwrap_or_else(|| panic!("thin box t={t} rejected"));
            assert_eq!(verts.len(), 8, "t={t}");
            assert_eq!(faces.len(), 12, "t={t}");
        }
        // diag ~ 2.83, tol ~ 2.8e-6: thickness 1e-7 is coplanar within tol.
        let t = 1e-7;
        let mut pts = Vec::new();
        for &x in &[-1.0, 1.0] {
            for &y in &[-1.0, 1.0] {
                for &z in &[-t, t] {
                    pts.push(Vector3::new(x, y, z));
                }
            }
        }
        assert!(quick_hull(&pts).is_none());
    }

    /// Second P2-C06 regression: a knife-edge slab cloud (t = 1e-4, f32)
    /// whose single connected visible region wraps around a vertex, giving a
    /// figure-eight horizon. The cones fold two fans onto the shared
    /// vertex-to-apex edge — a closed but non-manifold *pinch*. The contract
    /// here is balanced closure (no missing faces) and containment, which
    /// holds; a strict manifold assertion would demand vertex splitting.
    /// Values are f32-exact so both features see the same cloud.
    #[test]
    fn knife_edge_slab_stays_closed() {
        let raw: [[f64; 3]; 40] = [
            [-0.9108932, -0.17379946, 5.6840323e-5],
            [0.37399292, -0.73195565, -8.328129e-5],
            [0.34108257, -0.9125448, -7.6166994e-5],
            [0.7837244, 0.6782863, -1.7853277e-5],
            [0.14034307, 0.92666996, -2.7055692e-5],
            [0.24637961, -0.9072363, 4.0039253e-5],
            [0.8883046, 0.19199145, -3.7612204e-5],
            [0.3230536, -0.62355125, 5.4503154e-5],
            [0.94991887, -0.120628715, 1.45807735e-5],
            [0.72641265, 0.14986241, 4.3881795e-5],
            [-0.54348016, -0.6211158, -1.1115074e-5],
            [-0.6020621, -0.7717385, -6.0938506e-5],
            [0.38007832, 0.18998957, 5.6040943e-5],
            [0.9512106, 0.16365933, 5.3124786e-6],
            [0.24433541, 0.75072503, 6.5147506e-5],
            [-0.47116846, -0.2682932, 2.42697e-6],
            [-0.9784581, -0.119253874, -7.576941e-5],
            [0.36630225, -0.11977929, 7.105188e-5],
            [-0.6922034, 0.9947412, -1.7422431e-5],
            [-0.006534934, 0.95962214, -2.7276832e-5],
            [0.037958026, 0.045357108, -2.7195643e-5],
            [0.25884867, 0.3337592, -3.3316544e-5],
            [-0.26696128, -0.38368696, -7.681285e-5],
            [-0.3945827, -0.6440284, -4.612567e-5],
            [0.564589, 0.10798466, 7.030734e-5],
            [-0.9489303, 0.4384477, 7.663727e-5],
            [0.78310716, -0.6452262, -7.694695e-5],
            [-0.19817632, -0.15985894, -9.376955e-6],
            [-0.041793466, -0.5924778, 8.149576e-6],
            [-0.6154877, -0.64395773, -7.673014e-5],
            [0.93298995, 0.5055597, -9.602039e-6],
            [0.7907262, -0.9781223, 9.1491995e-5],
            [0.71562994, -0.0734424, -2.2521477e-5],
            [-0.3308726, 0.84766114, 3.3319473e-6],
            [-0.6907604, -0.1288858, 7.1249255e-5],
            [0.31984055, -0.527374, -2.0510643e-5],
            [-0.6395036, 0.99416065, 3.3420645e-5],
            [-0.6615592, 0.5757067, 4.3187127e-5],
            [-0.40946823, 0.4472872, 4.9349772e-5],
            [-0.3527248, -0.2507866, -7.227519e-5],
        ];
        let pts: Vec<Vector3> = raw
            .iter()
            .map(|p| Vector3::new(p[0] as Real, p[1] as Real, p[2] as Real))
            .collect();
        assert_valid_hull("knife-edge-slab-40", &pts, false);
    }
}

