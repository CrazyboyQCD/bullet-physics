//! Gilbert-Johnson-Keerthi convex-convex intersection test.
//!
//! Uses the Minkowski difference `A - B` support mapping and a simplex of up
//! to four points. When the origin is enclosed, the returned simplex seeds an
//! EPA pass to compute penetration depth.

use bullet_math::{Quaternion, Vector3};
use crate::shapes::CollisionShape;

const MAX_ITERATIONS: u32 = 64;

/// A simplex of Minkowski-difference vertices: at most four points in a
/// fixed-size buffer (`pts[..len]`, with `pts[0]` the most recent), so the
/// GJK hot loop performs no heap allocation.
#[derive(Clone, Debug)]
pub struct Simplex {
    pts: [Vector3; 4],
    len: usize,
}

impl Simplex {
    fn new(initial: Vector3) -> Self {
        Simplex {
            pts: [initial; 4],
            len: 1,
        }
    }

    fn add(&mut self, v: Vector3) {
        self.pts.copy_within(0..self.len, 1);
        self.pts[0] = v;
        self.len += 1;
    }

    /// Replace the active vertices (`pts[0]` first).
    fn set(&mut self, new_pts: &[Vector3]) {
        self.pts[..new_pts.len()].copy_from_slice(new_pts);
        self.len = new_pts.len();
    }

    /// The active vertices, most recent first.
    pub fn points(&self) -> &[Vector3] {
        &self.pts[..self.len]
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

/// Support point of the Minkowski difference `A - B` along direction `dir`.
#[inline]
pub fn support_minkowski(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
    dir: Vector3,
) -> Vector3 {
    shape_a.support_world(position_a, orientation_a, dir)
        - shape_b.support_world(position_b, orientation_b, -dir)
}

/// Run GJK. Returns the enclosing simplex if the shapes intersect, `None`
/// otherwise.
pub fn gjk_intersect(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
) -> Option<Simplex> {
    gjk_intersect_seeded(
        shape_a,
        position_a,
        orientation_a,
        shape_b,
        position_b,
        orientation_b,
        None,
    )
}

/// GJK with an optional warm-start seed direction (a cached separating
/// normal from a previous step of the same pair, e.g. from a persistent
/// manifold). Any non-degenerate seed is mathematically valid: the simplex
/// loop converges to the same enclosing simplex regardless of where it
/// starts, a good seed only saves iterations.
pub fn gjk_intersect_seeded(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
    seed: Option<Vector3>,
) -> Option<Simplex> {
    let center_dir = position_b - position_a;
    let fallback = if center_dir.length_squared() > 1e-12 {
        center_dir
    } else {
        Vector3::X
    };
    let initial_dir = seed
        .filter(|s| s.length_squared() > 1e-12)
        .unwrap_or(fallback);

    let mut simplex = Simplex::new(support_minkowski(
        shape_a,
        position_a,
        orientation_a,
        shape_b,
        position_b,
        orientation_b,
        initial_dir,
    ));
    let mut dir = -simplex.points()[0];

    for _ in 0..MAX_ITERATIONS {
        let a = support_minkowski(
            shape_a,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
            dir,
        );
        if a.dot(dir) < 0.0 {
            return None;
        }
        simplex.add(a);
        if solve(&mut simplex, &mut dir) {
            return Some(simplex);
        }
        // Degenerate case: the search direction collapsed (origin lies on the
        // simplex, or numeric noise shrank it). Pick a perpendicular.
        let reference = if simplex.len() >= 2 {
            simplex.points()[0] - simplex.points()[simplex.len() - 1]
        } else {
            simplex.points()[0]
        };
        if dir.length_squared() < reference.length_squared() * 1e-12 {
            dir = perpendicular(reference);
        }
    }
    None
}

/// Solve the simplex for the point closest to the origin, updating the search
/// direction. Returns `true` if the origin is enclosed by the simplex.
fn solve(simplex: &mut Simplex, dir: &mut Vector3) -> bool {
    match simplex.len() {
        2 => solve_segment(simplex, dir),
        3 => solve_triangle(simplex, dir),
        4 => solve_tetrahedron(simplex, dir),
        _ => {
            *dir = -simplex.points()[0];
            false
        }
    }
}

fn solve_segment(simplex: &mut Simplex, dir: &mut Vector3) -> bool {
    let a = simplex.pts[1];
    let b = simplex.pts[0];
    let ab = b - a;
    let ao = -a;
    if ab.dot(ao) > 0.0 {
        let cross = ab.cross(ao);
        // Relative collinearity test: `sin^2(angle)` is `cross^2 / (|ab|^2 |ao|^2)`.
        if cross.length_squared() > ab.length_squared() * ao.length_squared() * 1e-10 {
            *dir = cross.cross(ab);
        } else {
            // Degenerate (collinear) segment: pick an arbitrary perpendicular.
            *dir = perpendicular(ab);
        }
    } else {
        simplex.set(&[a]);
        *dir = ao;
    }
    false
}

/// An arbitrary unit vector perpendicular to `v`.
fn perpendicular(v: Vector3) -> Vector3 {
    let helper = if v.x.abs() < 0.9 { Vector3::X } else { Vector3::Y };
    v.cross(helper).normalize()
}

/// Triangle case, using the barycentric-region test from Ericson's
/// `ClosestPtPointTriangle` (Real-Time Collision Detection, 5.1.5). When the
/// closest point degenerates onto the origin (i.e. the origin lies exactly on
/// the simplex), keep the triangle and search along its normal for a fourth
/// point instead of shrinking it.
fn solve_triangle(simplex: &mut Simplex, dir: &mut Vector3) -> bool {
    let a = simplex.pts[2];
    let b = simplex.pts[1];
    let c = simplex.pts[0];

    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac);
    let ap = -a; // p = origin
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        simplex.set(&[a]);
        *dir = -a;
        return false;
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        simplex.set(&[b]);
        *dir = -b;
        return false;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = a + ab * v;
        return shrink_or_normal(simplex, dir, closest, normal, [b, a]);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        simplex.set(&[c]);
        *dir = -c;
        return false;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = a + ac * w;
        return shrink_or_normal(simplex, dir, closest, normal, [c, a]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b + (c - b) * w;
        return shrink_or_normal(simplex, dir, closest, normal, [c, b]);
    }
    // Interior: keep the triangle.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = a + ab * v + ac * w;
    if closest.length_squared() < 1e-24 {
        *dir = normal;
    } else {
        *dir = -closest;
    }
    false
}

/// Shrink the simplex to `new_p`, or (if the closest point is the origin
/// itself) keep the triangle and search along `normal` for a fourth point.
fn shrink_or_normal(
    simplex: &mut Simplex,
    dir: &mut Vector3,
    closest: Vector3,
    normal: Vector3,
    new_pts: [Vector3; 2],
) -> bool {
    if closest.length_squared() < 1e-24 {
        *dir = normal;
    } else {
        simplex.set(&new_pts);
        *dir = -closest;
    }
    false
}

/// Tetrahedron case: the origin is inside iff, for every face, it lies on the
/// same side as the opposite vertex. Otherwise shrink to the offending face.
fn solve_tetrahedron(simplex: &mut Simplex, dir: &mut Vector3) -> bool {
    let a = simplex.pts[3];
    let b = simplex.pts[2];
    let c = simplex.pts[1];
    let d = simplex.pts[0];

    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;
    let bo = -b;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);
    let bcd = (b - c).cross(d - c);

    // Origin is outside a face when it is on the opposite side of the face's
    // opposite vertex (dot products have opposite signs). Shrink to that face
    // and search along its normal (pointing toward the origin).
    if abc.dot(ao) * abc.dot(ad) < 0.0 {
        simplex.set(&[c, b, a]);
        *dir = toward_origin(abc, ao);
        return false;
    }
    if acd.dot(ao) * acd.dot(ab) < 0.0 {
        simplex.set(&[d, c, a]);
        *dir = toward_origin(acd, ao);
        return false;
    }
    if adb.dot(ao) * adb.dot(ac) < 0.0 {
        simplex.set(&[d, b, a]);
        *dir = toward_origin(adb, ao);
        return false;
    }
    if bcd.dot(bo) * bcd.dot(a - b) < 0.0 {
        simplex.set(&[d, c, b]);
        *dir = toward_origin(bcd, bo);
        return false;
    }
    true
}

/// Orient `normal` so it points toward the origin (in the direction of
/// `toward_origin`).
fn toward_origin(normal: Vector3, toward_origin: Vector3) -> Vector3 {
    if normal.dot(toward_origin) < 0.0 {
        -normal
    } else {
        normal
    }
}

/// Closest point to the origin on the simplex, shrinking it to the closest
/// feature (point, edge, face, or tetrahedron interior). Returns the
/// closest-point vector `origin -> q`. Shared by the GJK-distance walks in
/// `raycast.rs` (convex-hull casts) and `convex_sweep.rs` (CCD casts).
pub(crate) fn closest_origin(simplex: &mut Vec<Vector3>) -> Vector3 {
    match simplex.len() {
        0 => Vector3::ZERO,
        1 => simplex[0],
        2 => {
            let (v, keep) = closest_origin_segment(simplex[0], simplex[1]);
            *simplex = keep;
            v
        }
        3 => {
            let (v, keep) = closest_origin_triangle(simplex[0], simplex[1], simplex[2]);
            *simplex = keep;
            v
        }
        _ => {
            let (v, keep) =
                closest_origin_tetrahedron(simplex[0], simplex[1], simplex[2], simplex[3]);
            *simplex = keep;
            v
        }
    }
}

/// Closest point to the origin on the segment `a-b`, keeping the closest
/// feature (a point or the segment).
fn closest_origin_segment(a: Vector3, b: Vector3) -> (Vector3, Vec<Vector3>) {
    let ab = b - a;
    let t = -a.dot(ab) / ab.length_squared();
    if t <= 0.0 {
        (a, vec![a])
    } else if t >= 1.0 {
        (b, vec![b])
    } else {
        (a + ab * t, vec![b, a])
    }
}

/// Closest point to the origin on the triangle `a-b-c`, keeping the closest
/// feature (point, edge, or face). Uses the barycentric region test from
/// Ericson's `ClosestPtPointTriangle` (Real-Time Collision Detection, 5.1.5).
fn closest_origin_triangle(a: Vector3, b: Vector3, c: Vector3) -> (Vector3, Vec<Vector3>) {
    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, vec![a]);
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, vec![b]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = a + ab * v;
        return (q, vec![b, a]);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, vec![c]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = a + ac * w;
        return (q, vec![c, a]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q = b + (c - b) * w;
        return (q, vec![c, b]);
    }
    // Interior: the closest point is the origin's projection onto the face.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = a + ab * v + ac * w;
    (q, vec![c, b, a])
}

/// Closest point to the origin on the tetrahedron `a-b-c-d`. The origin is
/// inside iff it lies on the same side of every face as the opposite vertex;
/// otherwise shrink to the offending face (deferring to the triangle case).
fn closest_origin_tetrahedron(
    a: Vector3,
    b: Vector3,
    c: Vector3,
    d: Vector3,
) -> (Vector3, Vec<Vector3>) {
    let ao = -a;
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    // Degenerate (coplanar) tetrahedron: the side-of-face products below are
    // all zero (every opposite-vertex vector lies in the face plane), so the
    // "origin inside" conclusion would be wrong. A flat hull has no interior:
    // fall back to the closest of the four faces. The GJK distance walks hit
    // this when the search direction is the normal of the current closest
    // face (the pushed support point is coplanar with it).
    if ab.cross(ac).dot(ad).abs() < 1e-12 {
        let faces = [(a, b, c), (a, b, d), (a, c, d), (b, c, d)];
        let (best, keep) = faces
            .iter()
            .map(|&(x, y, z)| closest_origin_triangle(x, y, z))
            .min_by(|p, q| {
                p.0
                    .length_squared()
                    .partial_cmp(&q.0.length_squared())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        return (best, keep);
    }
    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);
    let bo = -b;
    let bcd = (c - b).cross(d - b);
    if abc.dot(ao) * abc.dot(ad) < 0.0 {
        return closest_origin_triangle(a, b, c);
    }
    if acd.dot(ao) * acd.dot(ab) < 0.0 {
        return closest_origin_triangle(a, c, d);
    }
    if adb.dot(ao) * adb.dot(ac) < 0.0 {
        return closest_origin_triangle(a, d, b);
    }
    if bcd.dot(bo) * bcd.dot(a - b) < 0.0 {
        return closest_origin_triangle(b, c, d);
    }
    // Origin inside the tetrahedron: distance is zero.
    (Vector3::ZERO, vec![a, b, c, d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::CollisionShape;

    fn overlaps(a: &CollisionShape, pa: Vector3, b: &CollisionShape, pb: Vector3) -> bool {
        gjk_intersect(a, pa, Quaternion::IDENTITY, b, pb, Quaternion::IDENTITY).is_some()
    }

    #[test]
    fn gjk_detects_sphere_overlap() {
        let a = CollisionShape::sphere(1.0);
        let b = CollisionShape::sphere(1.0);
        assert!(overlaps(&a, Vector3::ZERO, &b, Vector3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn gjk_detects_sphere_separation() {
        let a = CollisionShape::sphere(1.0);
        let b = CollisionShape::sphere(1.0);
        assert!(!overlaps(&a, Vector3::ZERO, &b, Vector3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn gjk_detects_box_overlap() {
        let a = CollisionShape::box_shape(Vector3::splat(1.0));
        let b = CollisionShape::box_shape(Vector3::splat(1.0));
        assert!(overlaps(&a, Vector3::ZERO, &b, Vector3::new(1.5, 0.0, 0.0)));
        assert!(!overlaps(&a, Vector3::ZERO, &b, Vector3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn gjk_detects_sphere_box_overlap() {
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::box_shape(Vector3::splat(1.0));
        assert!(overlaps(&a, Vector3::new(1.4, 0.0, 0.0), &b, Vector3::ZERO));
        assert!(!overlaps(&a, Vector3::new(2.0, 0.0, 0.0), &b, Vector3::ZERO));
    }
}
