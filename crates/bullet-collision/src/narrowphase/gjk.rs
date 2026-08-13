//! Gilbert-Johnson-Keerthi convex-convex intersection test.
//!
//! Uses the Minkowski difference `A - B` support mapping and a simplex of up
//! to four points. When the origin is enclosed, the returned simplex seeds an
//! EPA pass to compute penetration depth.

use bullet_math::{Quaternion, Vector3};
use crate::shapes::CollisionShape;

const MAX_ITERATIONS: u32 = 64;

/// A simplex of Minkowski-difference vertices. `p[0]` is the most recent.
#[derive(Clone, Debug)]
pub struct Simplex {
    pub p: Vec<Vector3>,
}

impl Simplex {
    fn new(initial: Vector3) -> Self {
        Simplex { p: vec![initial] }
    }

    fn add(&mut self, v: Vector3) {
        self.p.insert(0, v);
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
    let center_dir = position_b - position_a;
    let initial_dir = if center_dir.length_squared() > 1e-12 {
        center_dir
    } else {
        Vector3::X
    };

    let mut simplex = Simplex::new(support_minkowski(
        shape_a,
        position_a,
        orientation_a,
        shape_b,
        position_b,
        orientation_b,
        initial_dir,
    ));
    let mut dir = -simplex.p[0];

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
        let reference = if simplex.p.len() >= 2 {
            simplex.p[0] - simplex.p[simplex.p.len() - 1]
        } else {
            simplex.p[0]
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
    match simplex.p.len() {
        2 => solve_segment(simplex, dir),
        3 => solve_triangle(simplex, dir),
        4 => solve_tetrahedron(simplex, dir),
        _ => {
            *dir = -simplex.p[0];
            false
        }
    }
}

fn solve_segment(simplex: &mut Simplex, dir: &mut Vector3) -> bool {
    let a = simplex.p[1];
    let b = simplex.p[0];
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
        simplex.p = vec![a];
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
    let a = simplex.p[2];
    let b = simplex.p[1];
    let c = simplex.p[0];

    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac);
    let ap = -a; // p = origin
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        simplex.p = vec![a];
        *dir = -a;
        return false;
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        simplex.p = vec![b];
        *dir = -b;
        return false;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = a + ab * v;
        return shrink_or_normal(simplex, dir, closest, normal, vec![b, a]);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        simplex.p = vec![c];
        *dir = -c;
        return false;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = a + ac * w;
        return shrink_or_normal(simplex, dir, closest, normal, vec![c, a]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b + (c - b) * w;
        return shrink_or_normal(simplex, dir, closest, normal, vec![c, b]);
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
    new_p: Vec<Vector3>,
) -> bool {
    if closest.length_squared() < 1e-24 {
        *dir = normal;
    } else {
        simplex.p = new_p;
        *dir = -closest;
    }
    false
}

/// Tetrahedron case: the origin is inside iff, for every face, it lies on the
/// same side as the opposite vertex. Otherwise shrink to the offending face.
fn solve_tetrahedron(simplex: &mut Simplex, dir: &mut Vector3) -> bool {
    let a = simplex.p[3];
    let b = simplex.p[2];
    let c = simplex.p[1];
    let d = simplex.p[0];

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
        simplex.p = vec![c, b, a];
        *dir = toward_origin(abc, ao);
        return false;
    }
    if acd.dot(ao) * acd.dot(ab) < 0.0 {
        simplex.p = vec![d, c, a];
        *dir = toward_origin(acd, ao);
        return false;
    }
    if adb.dot(ao) * adb.dot(ac) < 0.0 {
        simplex.p = vec![d, b, a];
        *dir = toward_origin(adb, ao);
        return false;
    }
    if bcd.dot(bo) * bcd.dot(a - b) < 0.0 {
        simplex.p = vec![d, c, b];
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
