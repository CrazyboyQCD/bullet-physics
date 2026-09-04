//! Expanding Polytope Algorithm (EPA): penetration depth from a Minkowski
//! difference whose origin is enclosed.
//!
//! Given a GJK simplex enclosing the origin of the Minkowski difference
//! `A - B`, EPA seeds a polytope (thickened with the six axis-direction
//! support points) and iteratively expands it toward the origin, returning the
//! penetration normal (pointing from A to B) and depth.
//!
//! The face-distance bookkeeping follows Bullet's `btGjkEpa2` (`getedgedist`):
//! a face's distance is the true distance from the origin to the triangle,
//! falling back to the edge or vertex distance when the origin projects
//! outside the triangle. Using the plain plane distance makes the closest-face
//! selection wrong for elongated Minkowski differences and the iteration
//! stalls on wrong faces.

use bullet_math::{math, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

const MAX_ITERATIONS: u32 = 255;
/// Bullet's `EPA_ACCURACY` (precision-dependent, matching `btGjkEpa2`'s
/// `BT_USE_DOUBLE_PRECISION` config: `1e-4` / `1e-12`).
#[cfg(not(feature = "f64"))]
const EPSILON: Real = 1e-4;
#[cfg(feature = "f64")]
const EPSILON: Real = 1e-12;
/// Bullet's `EPA_PLANE_EPS`: face-visibility tolerance (absorbs coplanar
/// support points; precision-dependent as above: `1e-5` / `1e-14`).
#[cfg(not(feature = "f64"))]
const PLANE_EPS: Real = 1e-5;
#[cfg(feature = "f64")]
const PLANE_EPS: Real = 1e-14;
/// Bullet's `EPA_MAX_VERTICES`: the polytope stops growing past this size.
const MAX_VERTICES: usize = 128;

struct Face {
    a: usize,
    b: usize,
    c: usize,
    normal: Vector3,
    /// Edge-aware distance from the origin to the triangle; used for
    /// closest-face selection and convergence.
    distance: Real,
    /// Plain plane distance `normal . vertex`; used for face visibility.
    plane_distance: Real,
}

/// Compute penetration `(normal, depth)` from a GJK simplex enclosing the
/// origin and the Minkowski support function. `support` maps a direction to
/// the farthest point of `A - B` along it.
///
/// The vertex, face and boundary-edge buffers are caller-owned (the
/// allocations used to dominate the EPA profile — `Vec<Face>` churn from
/// the per-iteration `retain`/push cycle and the per-`expand` boundary
/// set): pass the same `EpaScratch` for every pair of a detection phase
/// and steady-state calls allocate nothing; a fresh one simply costs one
/// allocation per call.
pub fn penetration(
    simplex: &[Vector3; 4],
    support: impl Fn(Vector3) -> Vector3,
    scratch: &mut EpaScratch,
) -> (Vector3, Real) {
    penetration_inner(simplex, support, scratch)
}

/// Pooled EPA buffers (capacity survives `clear()`, so steady-state calls
/// allocate nothing). Owned by the caller — `PhysicsWorld` holds one and
/// reuses it across every pair of a detection phase.
#[derive(Default)]
pub struct EpaScratch {
    vertices: Vec<Vector3>,
    faces: Vec<Face>,
    boundary: Vec<(usize, usize)>,
}

fn penetration_inner(
    simplex: &[Vector3; 4],
    support: impl Fn(Vector3) -> Vector3,
    scratch: &mut EpaScratch,
) -> (Vector3, Real) {
    let EpaScratch { vertices, faces, boundary } = scratch;
    vertices.clear();
    faces.clear();
    // Seed the polytope with the GJK tetrahedron (which encloses the origin),
    // then expand it with the six axis-direction support points. Seeding from
    // the tetrahedron alone is fragile: for elongated Minkowski differences it
    // is near-degenerate and the iteration converges to a wrong face. Seeding
    // from the six directions alone does not guarantee the polytope encloses
    // the origin. Bullet's `btGjkEpa2` combines both the same way.
    vertices.extend_from_slice(simplex);
    build_initial_faces(simplex, faces);
    debug_assert!(!faces.is_empty(), "GJK tetrahedron is fully degenerate");

    for d in [
        Vector3::X,
        Vector3::NEG_X,
        Vector3::Y,
        Vector3::NEG_Y,
        Vector3::Z,
        Vector3::NEG_Z,
    ] {
        // expand() is a no-op when the point already lies inside the polytope.
        // Degenerate expansion would leave a hole in the hull (and negative
        // face distances); abort seeding in that case (Bullet's `InvalidHull`).
        let idx = vertices.len();
        let point = support(d);
        vertices.push(point);
        if !expand(faces, vertices, boundary, point, idx) {
            break;
        }
    }

    for _ in 0..MAX_ITERATIONS {
        let (best_idx, best_normal, best_dist) = find_best(faces);

        let new_point = support(best_normal);
        // Bullet: `wdist = dot(best->n, w) - best->d`; only expand when the
        // support point is meaningfully outside the best face.
        let wdist = best_normal.dot(new_point) - best_dist;
        if wdist <= EPSILON {
            // Converged: the closest face bounds the polytope.
            return (best_normal, best_dist);
        }

        // Remove the face and rebuild the hull around the new point.
        if vertices.len() >= MAX_VERTICES {
            // Bullet's `OutOfVertices`: stop growing the polytope.
            break;
        }
        let new_idx = vertices.len();
        vertices.push(new_point);
        if !expand(faces, vertices, boundary, new_point, new_idx) {
            // Degenerate expansion: keep the current hull and return its
            // closest face (matches Bullet's `InvalidHull` early exit).
            break;
        }
        let _ = best_idx;
    }

    let (_, best_normal, best_dist) = find_best(faces);
    (best_normal, best_dist)
}

/// The closest face: smallest absolute distance to the origin (Bullet's
/// `findbest`, which compares `d * d`).
fn find_best(faces: &[Face]) -> (usize, Vector3, Real) {
    let best = faces
        .iter()
        .enumerate()
        .min_by(|a, b| {
            a.1.distance
                .abs()
                .partial_cmp(&b.1.distance.abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .unwrap();
    (best.0, best.1.normal, best.1.distance)
}

fn build_initial_faces(simplex: &[Vector3; 4], out: &mut Vec<Face>) {
    // (a, b, c, opposite) for the four triangular faces of the tetrahedron.
    let quads = [(0usize, 1usize, 2usize, 3usize), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)];
    for &(a, b0, c0, opp) in &quads {
        let mut b = b0;
        let mut c = c0;
        let mut normal = (simplex[b] - simplex[a]).cross(simplex[c] - simplex[a]);
        // Orient the normal outward (away from the opposite vertex, which
        // lies inside the tetrahedron). Swapping b/c keeps the vertex
        // winding consistent with the normal, which the boundary-edge
        // tracking in expand() relies on.
        if normal.dot(simplex[opp] - simplex[a]) > 0.0 {
            normal = -normal;
            core::mem::swap(&mut b, &mut c);
        }
        let len = normal.length();
        if len < 1e-12 {
            // Degenerate face (collinear vertices): skip it.
            continue;
        }
        normal /= len;
        out.push(Face {
            a,
            b,
            c,
            normal,
            plane_distance: normal.dot(simplex[a]),
            distance: face_distance(normal, simplex[a], simplex[b], simplex[c]),
        });
    }
}

/// Distance from the origin to the triangle, via Bullet's `getedgedist`: when
/// the origin projects outside the triangle (past an edge), use the distance
/// to that edge (or its endpoints); otherwise use the plane distance.
#[inline]
fn face_distance(normal: Vector3, pa: Vector3, pb: Vector3, pc: Vector3) -> Real {
    // Early returns instead of `Option::or_else` plumbing — same probe
    // order, identical numerics, no closure chain on the hot path.
    if let Some(d) = edge_distance(pa, pb, normal) {
        return d;
    }
    if let Some(d) = edge_distance(pb, pc, normal) {
        return d;
    }
    if let Some(d) = edge_distance(pc, pa, normal) {
        return d;
    }
    pa.dot(normal)
}

/// Distance from the origin to the segment `a -> b`, when the origin lies
/// outside the triangle past that edge (`None` otherwise). Mirrors Bullet's
/// `getedgedist(face, a, b, dist)`.
#[inline]
fn edge_distance(pa: Vector3, pb: Vector3, normal: Vector3) -> Option<Real> {
    let ba = pb - pa;
    let n_ab = ba.cross(normal); // outward edge normal, on the triangle plane
    if pa.dot(n_ab) < 0.0 {
        // The origin is outside the triangle past the edge a->b.
        let ba_l2 = ba.length_squared();
        let a_dot_ba = pa.dot(ba);
        let b_dot_ba = pb.dot(ba);
        let dist = if a_dot_ba > 0.0 {
            pa.length()
        } else if b_dot_ba < 0.0 {
            pb.length()
        } else {
            let a_dot_b = pa.dot(pb);
            math::sqrt(
                ((pa.length_squared() * pb.length_squared() - a_dot_b * a_dot_b) / ba_l2).max(0.0),
            )
        };
        Some(dist)
    } else {
        None
    }
}

/// Expand the hull toward `new_point`, removing visible faces and stitching
/// new ones along the horizon. Returns `false` when an edge is degenerate
/// (leaving the hull would be invalid); the caller should stop iterating.
fn expand(
    faces: &mut Vec<Face>,
    vertices: &[Vector3],
    boundary: &mut Vec<(usize, usize)>,
    new_point: Vector3,
    new_idx: usize,
) -> bool {
    // Remove faces visible from the new point, tracking the boundary loop:
    // a directed edge whose reverse is already tracked is interior to the
    // visible region (cancel both); otherwise it lies on the horizon. The
    // horizon is small (tens of edges), so linear matching beats a hash set
    // and its SipHash cost — which dominated the profile.
    boundary.clear();

    faces.retain(|f| {
        // Coplanar tolerance (Bullet's `-EPA_PLANE_EPS`): a support point on
        // the face plane is treated as inside, keeping the horizon closed.
        let visible = f.normal.dot(new_point) - f.plane_distance > PLANE_EPS;
        if visible {
            for (u, v) in [(f.a, f.b), (f.b, f.c), (f.c, f.a)] {
                match boundary.iter().position(|&(x, y)| x == v && y == u) {
                    Some(i) => {
                        // Cancel the interior edge. `swap_remove` reorders the
                        // rest, but the horizon order is already arbitrary
                        // (face creation order never feeds back into results:
                        // `find_best` ties aside, the closest face is unique).
                        boundary.swap_remove(i);
                    }
                    None => boundary.push((u, v)),
                }
            }
        }
        !visible
    });

    // Drain keeps the pooled vec's capacity for the next expansion.
    for (u, v) in boundary.drain(..) {
        let cross = (vertices[v] - vertices[u]).cross(new_point - vertices[u]);
        let len = cross.length();
        if len < 1e-12 {
            // Degenerate edge (collinear with the new point): the horizon
            // would not close, so abandon this expansion.
            return false;
        }
        // The boundary edges run counter-clockwise when seen from outside, so
        // the cross product points outward; no sign flip is needed (matching
        // Bullet, which keeps face winding consistent).
        let normal = cross / len;
        let distance = face_distance(normal, vertices[u], vertices[v], new_point);
        faces.push(Face {
            a: u,
            b: v,
            c: new_idx,
            normal,
            distance,
            plane_distance: normal.dot(vertices[u]),
        });
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowphase::gjk;
    use bullet_math::Quaternion;
    use crate::shapes::CollisionShape;
    use approx::assert_relative_eq;

    #[test]
    fn epa_penetration_spheres() {
        let a = CollisionShape::sphere(1.0);
        let b = CollisionShape::sphere(1.0);
        let pa = Vector3::ZERO;
        // Oblique overlap (not axis-aligned) to avoid degenerate simplices.
        let pb = Vector3::new(1.0, 0.5, 0.3);

        let simplex = gjk::gjk_intersect(
            &a,
            pa,
            Quaternion::IDENTITY,
            &b,
            pb,
            Quaternion::IDENTITY,
        )
        .expect("spheres overlap");
        let mut sp = [Vector3::ZERO; 4];
        for (i, v) in simplex.points().iter().take(4).enumerate() {
            sp[i] = *v;
        }

        let support = |dir: Vector3| {
            gjk::support_minkowski(
                &a,
                pa,
                Quaternion::IDENTITY,
                &b,
                pb,
                Quaternion::IDENTITY,
                dir,
            )
        };
        let (normal, depth) = penetration(&sp, support, &mut EpaScratch::default());
        let expected = 2.0 - pb.length();
        assert_relative_eq!(depth, expected, epsilon = 1e-2);
        // Normal should point from A to B (i.e. along `pb`).
        assert!(normal.dot(pb.normalize()) > 0.9);
    }

    #[test]
    fn epa_axis_aligned_overlap_is_exact() {
        let a = CollisionShape::sphere(1.0);
        let b = CollisionShape::sphere(1.0);
        // Overlap along +X only; the sphere Minkowski difference is a sphere.
        let simplex = gjk::gjk_intersect(
            &a,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b,
            Vector3::new(1.5, 0.0, 0.0),
            Quaternion::IDENTITY,
        )
        .expect("spheres overlap");
        let mut sp = [Vector3::ZERO; 4];
        for (i, v) in simplex.points().iter().take(4).enumerate() {
            sp[i] = *v;
        }
        let support = |dir: Vector3| {
            gjk::support_minkowski(
                &a,
                Vector3::ZERO,
                Quaternion::IDENTITY,
                &b,
                Vector3::new(1.5, 0.0, 0.0),
                Quaternion::IDENTITY,
                dir,
            )
        };
        let (normal, depth) = penetration(&sp, support, &mut EpaScratch::default());
        assert_relative_eq!(depth, 0.5, epsilon = 1e-3);
        assert_relative_eq!(normal.x, 1.0, epsilon = 1e-4);
    }
}
