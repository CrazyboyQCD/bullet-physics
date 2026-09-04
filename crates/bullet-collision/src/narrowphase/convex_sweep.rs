//! Swept-sphere (CCD) casts: a sphere moving along a segment against a shape.
//!
//! Mirrors Bullet's `btContinuousConvexCollision` driven by the swept-sphere
//! trick in `btDiscreteDynamicsWorld::integrateTransforms`: the cast shape is a
//! `btSphereShape` of the body's `ccdSweptSphereRadius`, so the body's own
//! geometry is not swept. Casting a sphere against a convex target reduces to a
//! ray-versus-inflated-convex GJK distance search (the same walk `raycast.rs`
//! uses for convex hulls): the distance from a point to the target inflated by
//! the sphere radius is convex along the segment, so a ternary search finds the
//! closest approach and a binary search finds the entry point.

use bullet_math::{math, Aabb, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::narrowphase::gjk::closest_origin;
use crate::shapes::{CollisionShape, Plane, TriangleMesh};

/// A swept-sphere hit against a target shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepHit {
    /// Time-of-impact parameter in `[0, 1]` along the segment.
    pub fraction: Real,
    /// Unit normal at the hit, pointing from the target surface toward the
    /// swept sphere.
    pub normal: Vector3,
    /// Contact point on the target surface, in world space.
    pub point: Vector3,
}

/// Cast a sphere of `radius` from `from` to `to` against `other`.
///
/// `other` may be a convex shape (GJK distance search against the shape
/// inflated by `radius`), a plane (analytic), or a triangle mesh/heightfield
/// (per-triangle GJK casts in local space, accelerated by the mesh BVH).
/// Returns `None` when the segment never touches the inflated target.
pub fn swept_sphere_cast(
    from: Vector3,
    to: Vector3,
    radius: Real,
    other: &CollisionShape,
    other_pos: Vector3,
    other_ori: Quaternion,
) -> Option<SweepHit> {
    let delta = to - from;
    let length = delta.length();
    if length <= Real::EPSILON {
        return None;
    }
    let dir = delta / length;
    match other {
        CollisionShape::Plane(p) => sweep_plane(from, dir, length, radius, p, other_ori),
        CollisionShape::TriangleMesh(mesh) => sweep_mesh(from, to, radius, mesh, other_pos, other_ori),
        CollisionShape::Heightfield(hf) => {
            sweep_mesh(from, to, radius, hf.as_mesh(), other_pos, other_ori)
        }
        _ => sweep_convex(from, dir, length, radius, other, other_pos, other_ori),
    }
}

/// Squared distance from `point` to the convex shape with support function
/// `support`, and the separation direction (from the shape toward `point`),
/// via GJK distance.
///
/// The Minkowski difference `{point} - S` has `support(v) = point - sup_S(-v)`.
/// The simplex is seeded with the support point along a guess direction and
/// grown toward the origin; the closest point either collapses (point inside)
/// or stops improving when the support point repeats.
fn point_convex_distance<F>(support: &F, point: Vector3) -> (Real, Vector3)
where
    F: Fn(Vector3) -> Vector3,
{
    let mink = |v: Vector3| point - support(-v);
    let mut v = point;
    if v.length_squared() < 1e-12 {
        v = Vector3::X;
    }
    let mut simplex: Vec<Vector3> = vec![mink(v)];
    let mut v = simplex[0];
    for _ in 0..64 {
        if v.length_squared() < 1e-10 {
            return (0.0, Vector3::ZERO); // Point inside or touching the shape.
        }
        let w = mink(-v);
        if simplex.iter().any(|p| (*p - w).length_squared() < 1e-12) {
            // No new vertex: the simplex already spans the closest feature.
            return (v.length_squared(), v / v.length());
        }
        simplex.push(w);
        v = closest_origin(&mut simplex);
    }
    (v.length_squared(), v / v.length())
}

/// Cast a sphere against a convex target via GJK distance on the inflated
/// shape (the target ⊕ sphere of `radius`).
fn sweep_convex(
    from: Vector3,
    dir: Vector3,
    length: Real,
    radius: Real,
    other: &CollisionShape,
    other_pos: Vector3,
    other_ori: Quaternion,
) -> Option<SweepHit> {
    let support = |d: Vector3| -> Vector3 {
        let sphere = if d.length_squared() > 1e-12 {
            d / d.length() * radius
        } else {
            Vector3::ZERO
        };
        other.support_world(other_pos, other_ori, d) + sphere
    };
    let dist = |p: Vector3| point_convex_distance(&support, p);
    sweep_with_dist(from, dir, length, radius, dist)
}

/// Cast a sphere against a triangle, in the frame the vertices are expressed
/// in. The distance uses the analytic closest-point-on-triangle test (GJK
/// distance is numerically unreliable for zero-thickness triangles).
fn sweep_triangle(
    from: Vector3,
    dir: Vector3,
    length: Real,
    radius: Real,
    a: Vector3,
    b: Vector3,
    c: Vector3,
) -> Option<SweepHit> {
    let dist = |p: Vector3| -> (Real, Vector3) {
        let (d2, dv) = point_triangle_distance(p, a, b, c);
        if d2 <= radius * radius {
            // Inside the inflated triangle (or touching it).
            (0.0, Vector3::ZERO)
        } else {
            let d = math::sqrt(d2);
            ((d - radius) * (d - radius), dv / d)
        }
    };
    sweep_with_dist(from, dir, length, radius, dist)
}

/// Shared sweep core: ternary search for the closest approach (the distance to
/// a convex shape is convex along a line), then binary search for the first
/// fraction inside the inflated shape. `dist` maps a point to the squared
/// distance to the inflated target and the unit direction from the target
/// toward the point.
fn sweep_with_dist<F>(from: Vector3, dir: Vector3, length: Real, radius: Real, dist: F) -> Option<SweepHit>
where
    F: Fn(Vector3) -> (Real, Vector3),
{
    let dist2 = |p: Vector3| dist(p).0;
    // Ternary search for the closest approach (40 iterations, more than
    // enough on f32/f64).
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..40 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if dist2(from + dir * (m1 * length)) < dist2(from + dir * (m2 * length)) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let s_min = (lo + hi) * 0.5;
    let d_min2 = dist2(from + dir * (s_min * length));
    if d_min2 > 1e-8 {
        return None; // Never touches the inflated target.
    }
    // Binary search on [0, s_min] (the distance is decreasing there) for the
    // first fraction inside the inflated shape.
    let mut low = 0.0;
    let mut high = s_min;
    for _ in 0..60 {
        let mid = (low + high) * 0.5;
        if dist2(from + dir * (mid * length)) <= 1e-8 {
            high = mid;
        } else {
            low = mid;
        }
    }
    let fraction = (low + high) * 0.5;
    // Surface normal: the separation direction at a point slightly before the
    // entry (the direction is numerically stable there), pointing from the
    // target toward the sphere.
    let probe = from + dir * ((fraction - 2e-4) * length).max(0.0);
    let (_, v) = dist(probe);
    let normal = if v.length_squared() > 1e-12 {
        v / v.length()
    } else {
        -dir
    };
    // Contact point on the target surface: the sphere-center entry pulled back
    // by the sphere radius along the normal.
    let entry = from + dir * (fraction * length);
    Some(SweepHit {
        fraction,
        normal,
        point: entry - normal * radius,
    })
}

/// Squared distance from `point` to the triangle `a-b-c` and the vector from
/// the closest point on the triangle toward `point` (Ericson's
/// `ClosestPtPointTriangle`, Real-Time Collision Detection, 5.1.5).
pub fn point_triangle_distance(point: Vector3, a: Vector3, b: Vector3, c: Vector3) -> (Real, Vector3) {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ((point - a).length_squared(), point - a);
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return ((point - b).length_squared(), point - b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = a + ab * v;
        return ((point - q).length_squared(), point - q);
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return ((point - c).length_squared(), point - c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = a + ac * w;
        return ((point - q).length_squared(), point - q);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q = b + (c - b) * w;
        return ((point - q).length_squared(), point - q);
    }
    // Interior: the closest point is the projection onto the face.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = a + ab * v + ac * w;
    ((point - q).length_squared(), point - q)
}

/// Cast a sphere against a plane (analytic): the sphere center crosses the
/// plane's offset slab of width `2 * radius`.
fn sweep_plane(
    from: Vector3,
    dir: Vector3,
    length: Real,
    radius: Real,
    plane: &Plane,
    other_ori: Quaternion,
) -> Option<SweepHit> {
    let n = other_ori * plane.normal;
    let dn0 = n.dot(from) - plane.constant;
    if dn0.abs() <= radius {
        // Already touching: clamp at the start.
        return Some(SweepHit {
            fraction: 0.0,
            normal: if dn0 >= 0.0 { n } else { -n },
            point: from - n * dn0,
        });
    }
    let dd = n.dot(to_delta(dir, length));
    if dd >= 0.0 {
        return None; // Moving away from or parallel to the plane.
    }
    let sign = if dn0 > 0.0 { 1.0 } else { -1.0 };
    let fraction = (sign * radius - dn0) / dd;
    if !(0.0..=1.0).contains(&fraction) {
        return None;
    }
    let normal = if sign > 0.0 { n } else { -n };
    let entry = from + to_delta(dir, length) * fraction;
    Some(SweepHit {
        fraction,
        normal,
        point: entry - normal * radius,
    })
}

/// Cast a sphere against a triangle mesh: query the BVH with the swept AABB
/// and keep the closest per-triangle hit (all in mesh-local space; the sphere
/// is isotropic, so the radius carries over unchanged).
fn sweep_mesh(
    from: Vector3,
    to: Vector3,
    radius: Real,
    mesh: &TriangleMesh,
    pos: Vector3,
    ori: Quaternion,
) -> Option<SweepHit> {
    let inv_ori = ori.inverse();
    let local_from = inv_ori * (from - pos);
    let local_to = inv_ori * (to - pos);
    let delta = local_to - local_from;
    let length = delta.length();
    if length <= Real::EPSILON {
        return None;
    }
    let dir = delta / length;
    let aabb = swept_aabb(local_from, local_to, radius);
    let mut best: Option<SweepHit> = None;
    for tri in mesh.triangles_in_aabb(&aabb) {
        let [a, b, c] = mesh.indices[tri as usize];
        if let Some(hit) = sweep_triangle(
            local_from,
            dir,
            length,
            radius,
            mesh.vertices[a as usize],
            mesh.vertices[b as usize],
            mesh.vertices[c as usize],
        ) && best.is_none_or(|h| hit.fraction < h.fraction)
        {
            best = Some(hit);
        }
    }
    best.map(|h| SweepHit {
        fraction: h.fraction,
        normal: ori * h.normal,
        point: ori * h.point + pos,
    })
}

/// AABB of the swept segment (endpoints plus the sphere radius), conservative
/// for the BVH query.
fn swept_aabb(from: Vector3, to: Vector3, radius: Real) -> Aabb {
    let r = Vector3::splat(radius);
    Aabb::new(from.min(to) - r, from.max(to) + r)
}

/// The full segment displacement (`dir * length`).
#[inline]
fn to_delta(dir: Vector3, length: Real) -> Vector3 {
    dir * length
}

/// Cast a convex `shape` from `from` to `to` against a target, returning the
/// closest hit. Generalizes [`swept_sphere_cast`] to an arbitrary convex cast
/// shape — used by the character controller's step-up/forward/down sweeps,
/// mirroring `btCollisionWorld::convexSweep`:
/// - convex targets: Minkowski GJK distance walk (the same closest-approach
///   search as the sphere cast, with a zero radius);
/// - planes: analytic (the shape's deepest point along the plane normal);
/// - triangle meshes / heightfields: BVH-pruned per-triangle convex casts.
///
/// `shape` must be convex; `other` must be convex, a plane, or a
/// mesh/heightfield. Returns `None` when the segment never touches the target.
pub fn convex_cast(
    shape: &CollisionShape,
    from: Vector3,
    ori: Quaternion,
    to: Vector3,
    other: &CollisionShape,
    other_pos: Vector3,
    other_ori: Quaternion,
) -> Option<SweepHit> {
    match other {
        CollisionShape::Plane(p) => sweep_convex_plane(shape, from, ori, to, p, other_pos, other_ori),
        CollisionShape::TriangleMesh(mesh) => {
            sweep_convex_mesh(shape, from, ori, to, mesh, other_pos, other_ori)
        }
        CollisionShape::Heightfield(hf) => {
            sweep_convex_mesh(shape, from, ori, to, hf.as_mesh(), other_pos, other_ori)
        }
        _ => sweep_convex_shape(shape, from, ori, to, other, other_pos, other_ori),
    }
}

/// Cast `shape` against a convex target via the Minkowski GJK distance: the
/// squared distance between the two shapes along the segment is convex, so the
/// shared ternary/binary search finds the first touching fraction. The
/// separation direction (from the target toward the moving shape) is the hit
/// normal.
fn sweep_convex_shape(
    shape: &CollisionShape,
    from: Vector3,
    ori: Quaternion,
    to: Vector3,
    other: &CollisionShape,
    other_pos: Vector3,
    other_ori: Quaternion,
) -> Option<SweepHit> {
    let delta = to - from;
    let length = delta.length();
    if length <= Real::EPSILON {
        return None;
    }
    let dir = delta / length;
    // Squared shape-shape distance (and separation direction) with the moving
    // shape's position `p` along the segment. The support is B - A (the target
    // minus the moving shape), so `point_convex_distance`'s returned direction
    // — the closest point of {point - S} to the origin — points from the
    // target's closest point toward the moving shape, matching the hit-normal
    // convention "from the target toward the swept shape".
    let dist = |p: Vector3| -> (Real, Vector3) {
        let support = |d: Vector3| {
            other.support_world(other_pos, other_ori, d) - shape.support_world(p, ori, -d)
        };
        point_convex_distance(&support, Vector3::ZERO)
    };
    let mut hit = sweep_with_dist(from, dir, length, 0.0, dist)?;
    // The sweep core reports the shape reference point at the entry (the
    // radius is zero); refine it to a contact point on the target's contact
    // side: the target's support along the hit normal (from the target toward
    // the moving shape).
    hit.point = other.support_world(other_pos, other_ori, hit.normal);
    Some(hit)
}

/// Cast `shape` against a plane (analytic): the shape's deepest point along
/// the plane normal crosses the plane during the sweep.
fn sweep_convex_plane(
    shape: &CollisionShape,
    from: Vector3,
    ori: Quaternion,
    to: Vector3,
    plane: &Plane,
    other_pos: Vector3,
    other_ori: Quaternion,
) -> Option<SweepHit> {
    let delta = to - from;
    let length = delta.length();
    if length <= Real::EPSILON {
        return None;
    }
    let n = other_ori * plane.normal;
    let c = plane.constant + n.dot(other_pos);
    // Normal from the plane toward the shape (the plane narrow phase's
    // orientation convention).
    let normal = if n.dot(from) >= c { n } else { -n };
    let deepest = |p: Vector3| shape.support_world(p, ori, -normal);
    let da = normal.dot(deepest(from)) - c;
    let db = normal.dot(deepest(to)) - c;
    if db >= da {
        // Moving away from or parallel to the plane (this also covers a shape
        // resting exactly ON the plane sweeping sideways: da == db == 0).
        return None;
    }
    if da <= 0.0 {
        // Already touching (and approaching): clamp at the start.
        return Some(SweepHit {
            fraction: 0.0,
            normal,
            point: deepest(from),
        });
    }
    let fraction = da / (da - db);
    if !(0.0..=1.0).contains(&fraction) {
        return None;
    }
    let entry = from + delta * fraction;
    Some(SweepHit {
        fraction,
        normal,
        point: deepest(entry),
    })
}

/// Cast `shape` against a triangle mesh: BVH-pruned per-triangle convex casts
/// in mesh-local space (the hit fraction is preserved under the rigid
/// transform; the normal and point are mapped back to world space).
fn sweep_convex_mesh(
    shape: &CollisionShape,
    from: Vector3,
    ori: Quaternion,
    to: Vector3,
    mesh: &TriangleMesh,
    other_pos: Vector3,
    other_ori: Quaternion,
) -> Option<SweepHit> {
    let inv = other_ori.inverse();
    let from_l = inv * (from - other_pos);
    let to_l = inv * (to - other_pos);
    let ori_l = inv * ori;
    // Swept AABB: the union of the shape's AABBs at the two endpoints (the
    // AABB of a convex set moving linearly is the union of endpoint AABBs).
    let aabb = shape
        .world_aabb(from_l, ori_l)
        .merged(&shape.world_aabb(to_l, ori_l));
    let mut best: Option<SweepHit> = None;
    for tri in mesh.triangles_in_aabb(&aabb) {
        let (a, b, c) = mesh.triangle(tri as usize);
        let tri_shape = CollisionShape::convex_hull(vec![a, b, c]);
        if let Some(hit) = sweep_convex_shape(
            shape,
            from_l,
            ori_l,
            to_l,
            &tri_shape,
            Vector3::ZERO,
            Quaternion::IDENTITY,
        ) && best.is_none_or(|h| hit.fraction < h.fraction)
        {
            best = Some(hit);
        }
    }
    best.map(|h| SweepHit {
        fraction: h.fraction,
        normal: other_ori * h.normal,
        point: other_ori * h.point + other_pos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn cast(from: Vector3, to: Vector3, radius: Real, other: &CollisionShape) -> Option<SweepHit> {
        swept_sphere_cast(from, to, radius, other, Vector3::ZERO, Quaternion::IDENTITY)
    }

    #[test]
    fn head_on_box_hit() {
        let b = CollisionShape::box_shape(Vector3::splat(1.0));
        // Sphere radius 0.5 moving +X to -X: the inflated box spans
        // x in [-1.5, 1.5], entry at x = 1.5 from x = 5 over a 5-unit segment.
        let h = cast(Vector3::new(5.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0), 0.5, &b).unwrap();
        assert_relative_eq!(h.fraction, (5.0 - 1.5) / 5.0, epsilon = 1e-4);
        // The normal points from the box back toward the sphere (+X).
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-3);
        // The contact point sits on the box face x = 1.
        assert_relative_eq!(h.point.x, 1.0, epsilon = 1e-3);
        assert_relative_eq!(h.point.y, 0.0, epsilon = 1e-3);
        assert_relative_eq!(h.point.z, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn offset_cast_misses_the_box() {
        let b = CollisionShape::box_shape(Vector3::splat(1.0));
        // The inflated box spans y in [-1.5, 1.5]; a pass at y = 3 misses.
        assert!(cast(Vector3::new(5.0, 3.0, 0.0), Vector3::new(-5.0, 3.0, 0.0), 0.5, &b).is_none());
    }

    #[test]
    fn plane_entry_fraction_is_analytic() {
        let p = CollisionShape::plane(Vector3::Y, 0.0);
        // Sphere falls from y = 5 to y = -5 (segment length 10); the slab
        // boundary is at y = 0.5, so the entry fraction is (5 - 0.5) / 10.
        let h = cast(Vector3::new(0.0, 5.0, 0.0), Vector3::new(0.0, -5.0, 0.0), 0.5, &p).unwrap();
        assert_relative_eq!(h.fraction, 0.45, epsilon = 1e-5);
        assert_relative_eq!(h.normal.y, 1.0, epsilon = 1e-5);
        assert_relative_eq!(h.point.y, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn plane_moving_away_never_hits() {
        let p = CollisionShape::plane(Vector3::Y, 0.0);
        assert!(cast(Vector3::new(0.0, 2.0, 0.0), Vector3::new(0.0, 5.0, 0.0), 0.5, &p).is_none());
    }

    #[test]
    fn sphere_already_touching_clamps_to_zero() {
        let p = CollisionShape::plane(Vector3::Y, 0.0);
        let h = cast(Vector3::new(0.0, 0.3, 0.0), Vector3::new(0.0, -3.0, 0.0), 0.5, &p).unwrap();
        assert_eq!(h.fraction, 0.0);
        assert_relative_eq!(h.normal.y, 1.0);
    }

    #[test]
    fn sphere_cast_against_a_sloped_triangle() {
        // A single triangle sloped 45 degrees rising along +X (y = x), with
        // the hypotenuse from (2,2,0) to (0,0,2).
        let mesh = CollisionShape::triangle_mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(2.0, 2.0, 0.0),
                Vector3::new(0.0, 0.0, 2.0),
            ],
            vec![[0, 1, 2]],
        );
        // The sphere drops vertically over the interior of the slope
        // (x + z = 1.6 < 2), where the surface normal is (1, -1, 0) / sqrt(2).
        // Entry: |y - x| = r * sqrt(2), i.e. y = 1 + 0.2 * sqrt(2) ≈ 1.2828,
        // over a 4-unit drop from y = 3: fraction = (3 - 1.2828) / 4.
        let h = cast(Vector3::new(1.0, 3.0, 0.6), Vector3::new(1.0, -1.0, 0.6), 0.2, &mesh).unwrap();
        assert_relative_eq!(
            h.fraction,
            (3.0 - (1.0 + 0.4 * bullet_math::FRAC_1_SQRT_2)) / 4.0,
            epsilon = 1e-3
        );
        // The slope surface normal points up and toward -X.
        assert!(h.normal.y > 0.6);
        assert!(h.normal.x < -0.1);
    }

    fn ccast(shape: &CollisionShape, from: Vector3, to: Vector3, other: &CollisionShape) -> Option<SweepHit> {
        convex_cast(shape, from, Quaternion::IDENTITY, to, other, Vector3::ZERO, Quaternion::IDENTITY)
    }

    #[test]
    fn box_cast_head_on_hits_a_box() {
        let moving = CollisionShape::box_shape(Vector3::splat(0.5));
        let target = CollisionShape::box_shape(Vector3::splat(1.0));
        // The moving box (half 0.5) travels +X to -X; the target face at
        // x = 1 is reached when the moving box's -X face (at x - 0.5) hits it:
        // from x = 5 over a 5-unit segment, fraction = (5 - 1.5) / 5.
        let h = ccast(&moving, Vector3::new(5.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0), &target).unwrap();
        assert_relative_eq!(h.fraction, (5.0 - 1.5) / 5.0, epsilon = 1e-4);
        // The normal points from the target back toward the moving box (+X).
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-3);
        // The contact point sits on the target's contact face (x = 1).
        assert_relative_eq!(h.point.x, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn box_cast_misses_when_offset() {
        let moving = CollisionShape::box_shape(Vector3::splat(0.5));
        let target = CollisionShape::box_shape(Vector3::splat(1.0));
        // The combined extent along y is 1.5; a pass at y = 3 misses.
        assert!(ccast(&moving, Vector3::new(5.0, 3.0, 0.0), Vector3::new(-5.0, 3.0, 0.0), &target).is_none());
    }

    #[test]
    fn capsule_cast_lands_on_a_plane() {
        let capsule = CollisionShape::capsule(0.4, 0.8);
        let plane = CollisionShape::plane(Vector3::Y, 0.0);
        // Capsule drops from y = 3 to y = -1 (4-unit segment); its lowest
        // point (center - half_height - radius) reaches the plane at
        // y = 3 - 4f, i.e. fraction = (3 - 1.2)/4 = 0.45.
        let h = ccast(&capsule, Vector3::new(0.0, 3.0, 0.0), Vector3::new(0.0, -1.0, 0.0), &plane).unwrap();
        assert_relative_eq!(h.fraction, (3.0 - 1.2) / 4.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal.y, 1.0, epsilon = 1e-4);
        // The hit point is the capsule's lowest point touching the plane.
        assert_relative_eq!(h.point.y, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn box_cast_against_a_mesh_slope() {
        // A single 45-degree triangle slope (y = x), the falling_mesh winding.
        let mesh = CollisionShape::triangle_mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(2.0, 2.0, 0.0),
                Vector3::new(0.0, 0.0, 2.0),
            ],
            vec![[0, 1, 2]],
        );
        let moving = CollisionShape::box_shape(Vector3::splat(0.4));
        // Drop the box vertically over the slope interior; it must hit the
        // surface (upward-ish normal) rather than pass through.
        let h = ccast(&moving, Vector3::new(1.0, 3.0, 0.6), Vector3::new(1.0, -1.0, 0.6), &mesh).unwrap();
        assert!(h.normal.y > 0.6, "normal {h:?} should point up the slope");
        assert!(h.fraction > 0.0 && h.fraction < 1.0);
    }
}
