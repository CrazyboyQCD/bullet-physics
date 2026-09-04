//! Incremental quickhull — a small deterministic implementation for the
//! soft-body builders (`create_from_convex_hull` and the ellipsoid's
//! Hammersley point set), playing the role of Bullet's `HullLibrary`
//! (`Extras/ConvexDecomposition`).
//!
//! Bullet's hull library re-indexes and merges its output vertices, so its
//! link/face append order differs from ours; this implementation keeps the
//! *input* vertex indices (the hull vertices are an index-ascending subset),
//! which is more deterministic but still a recorded deviation — the face
//! order shapes the PBD solve order and with it the settle scatter.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::{Real, Vector3};

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
    assign_points(&mut faces, points, tol);

    // Expand: while some face holds outside points, add its farthest point
    // and rebuild the hull over the horizon of the faces it sees.
    while let Some(fi) = faces.iter().position(|f| !f.outside.is_empty()) {
        let p = *faces[fi].outside.last().unwrap();
        // Flood fill the faces `p` can see.
        let mut visible = vec![fi];
        let mut stack = vec![fi];
        while stack.pop().is_some() {
            for (nb, f) in faces.iter().enumerate() {
                if !visible.contains(&nb) && f.dist(points, p) > tol {
                    visible.push(nb);
                    stack.push(nb);
                }
            }
        }
        // Horizon: visible-face edges not shared with another visible face
        // (canonicalized undirected keys).
        let key = |e: [usize; 2]| (e[0].min(e[1]), e[0].max(e[1]));
        let mut horizon: Vec<[usize; 2]> = Vec::new();
        for &vi in &visible {
            for e in face_edges(faces[vi].v) {
                let ek = key(e);
                let shared = visible.iter().any(|&wj| {
                    wj != vi
                        && face_edges(faces[wj].v).iter().any(|e2| key(*e2) == ek)
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
        assign_points_indexes(&mut faces, points, &orphans, tol);
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

fn assign_points(faces: &mut [Face], points: &[Vector3], tol: Real) {
    let indexes: Vec<usize> = (0..points.len()).collect();
    assign_points_indexes(faces, points, &indexes, tol);
}

fn assign_points_indexes(faces: &mut [Face], points: &[Vector3], candidates: &[usize], tol: Real) {
    for &p in candidates {
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
}
