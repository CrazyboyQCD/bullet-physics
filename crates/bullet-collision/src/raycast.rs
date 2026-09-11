//! Ray casting against collision shapes (Bullet's `rayTest` semantics).
//!
//! A ray is a segment from `ray_from` to `ray_to`; a hit reports the
//! `fraction` (0..1) where the segment first enters the shape, the world-space
//! hit position, and the surface normal at the hit, pointing toward the ray
//! origin. Convex shapes use exact analytic intersections where one exists;
//! [`ConvexHull`](crate::shapes::ConvexHull) (which stores no faces) uses a
//! GJK distance search. Compounds recurse into their children.

use bullet_math::{math, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::narrowphase::gjk::closest_origin;
use crate::shapes::{CollisionShape, Compound};

/// A ray hit against a shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Hit parameter in `[0, 1]`: `position = lerp(ray_from, ray_to, fraction)`.
    /// `0.0` means the ray starts inside the shape.
    pub fraction: Real,
    /// World-space hit position.
    pub position: Vector3,
    /// Unit surface normal at the hit, pointing out of the shape (toward the
    /// ray origin). When the ray starts inside the shape this is instead the
    /// normalized ray direction, since no surface was entered.
    pub normal: Vector3,
}

/// A hit in local space: `(fraction, local_normal)`.
type LocalHit = (Real, Vector3);

/// Cast a ray against a single shape.
///
/// The shape is posed by `position` + `orientation`; the ray is given in world
/// space. Returns `None` when the segment never enters the shape. A
/// zero-length ray (identical endpoints) only hits when its start point is
/// inside the shape.
pub fn raycast(
    shape: &CollisionShape,
    position: Vector3,
    orientation: Quaternion,
    ray_from: Vector3,
    ray_to: Vector3,
) -> Option<RayHit> {
    let dir = ray_to - ray_from;
    let dir_len2 = dir.length_squared();
    if dir_len2 < Real::EPSILON {
        // Degenerate ray: report the interior case only.
        if local_inside(shape, position, orientation, ray_from) {
            return Some(RayHit {
                fraction: 0.0,
                position: ray_from,
                normal: Vector3::ZERO,
            });
        }
        return None;
    }

    // Transform the ray into the shape's local space once; children of a
    // compound re-transform recursively.
    let inv = orientation.inverse();
    let (fraction, local_normal) = local_raycast(shape, inv, position, ray_from, ray_to)?;

    let normal = if fraction > 0.0 {
        orientation * local_normal
    } else {
        // Interior start: no surface was entered, report the ray direction.
        dir / math::sqrt(dir_len2)
    };
    Some(RayHit {
        fraction,
        position: ray_from + dir * fraction,
        normal,
    })
}

/// Local-space ray cast against `shape` (ray already transformed to its local
/// frame, with `inv` the world-to-local rotation kept for compound children).
fn local_raycast(
    shape: &CollisionShape,
    inv: Quaternion,
    position: Vector3,
    ray_from: Vector3,
    ray_to: Vector3,
) -> Option<LocalHit> {
    let local_from = inv * (ray_from - position);
    let local_to = inv * (ray_to - position);
    match shape {
        CollisionShape::Sphere(s) => sphere_raycast(s.radius, local_from, local_to),
        CollisionShape::BoxShape(b) => box_raycast(b.half_extents, local_from, local_to),
        CollisionShape::Plane(p) => plane_raycast(p.normal, p.constant, local_from, local_to),
        CollisionShape::Capsule(c) => capsule_raycast(c.radius, c.half_height, local_from, local_to),
        CollisionShape::Cylinder(c) => {
            cylinder_raycast(c.radius, c.half_height, local_from, local_to)
        }
        CollisionShape::Cone(c) => cone_raycast(c.radius, c.half_height, local_from, local_to),
        CollisionShape::ConvexHull(h) => {
            convex_hull_raycast(&h.points, local_from, local_to)
        }
        CollisionShape::TriangleMesh(m) => m.raycast(local_from, local_to),
        CollisionShape::Heightfield(h) => h.raycast(local_from, local_to),
        CollisionShape::Compound(compound) => {
            compound_raycast(compound, inv, position, ray_from, ray_to)
        }
        // The uniform scale maps a ray point `p` to `position + (p -
        // position) / f` on the child (the scale maps the child hit back
        // onto the original ray at the same parameter), so casting against
        // the child with the remapped endpoints returns the same fraction
        // and the same local normal (uniform scales keep directions).
        CollisionShape::UniformScaling(child, f) => {
            let map = |p: Vector3| position + (p - position) / f;
            local_raycast(
                child,
                inv,
                position,
                map(ray_from),
                map(ray_to),
            )
        }
        CollisionShape::Convex2d(child) => {
            local_raycast(child, inv, position, ray_from, ray_to)
        }
    }
}

/// Whether `point` (world space) lies inside `shape`. Used for degenerate rays
/// and compound interior starts.
fn local_inside(
    shape: &CollisionShape,
    position: Vector3,
    orientation: Quaternion,
    point: Vector3,
) -> bool {
    match shape {
        CollisionShape::Sphere(s) => {
            (point - position).length_squared() <= s.radius * s.radius
        }
        CollisionShape::BoxShape(b) => {
            let p = orientation.inverse() * (point - position);
            p.x >= -b.half_extents.x
                && p.x <= b.half_extents.x
                && p.y >= -b.half_extents.y
                && p.y <= b.half_extents.y
                && p.z >= -b.half_extents.z
                && p.z <= b.half_extents.z
        }
        CollisionShape::Capsule(c) => {
            let p = orientation.inverse() * (point - position);
            segment_distance_squared(Vector3::ZERO, Vector3::new(0.0, c.half_height, 0.0), p)
                <= c.radius * c.radius
        }
        CollisionShape::Cylinder(c) => {
            let p = orientation.inverse() * (point - position);
            p.y.abs() <= c.half_height
                && (p.x * p.x + p.z * p.z) <= c.radius * c.radius
        }
        CollisionShape::Cone(c) => {
            let p = orientation.inverse() * (point - position);
            cone_radial(c.radius, c.half_height, p) <= 0.0
        }
        CollisionShape::ConvexHull(h) => {
            let p = orientation.inverse() * (point - position);
            hull_distance_squared(&h.points, p) <= 1e-12
        }
        CollisionShape::Plane(_) => false,
        // Open surfaces: rays pass through freely; hits report the surface     
        // normal facing the ray origin from either side.
        CollisionShape::TriangleMesh(_) | CollisionShape::Heightfield(_) => false,
        CollisionShape::Compound(compound) => compound.children.iter().any(|child| {
            let child_pos = position + orientation * child.position;
            let child_ori = orientation * child.orientation;
            local_inside(&child.shape, child_pos, child_ori, point)
        }),
        // The uniform scale maps a point `p` to `position + (p - position) /
        // f` on the child (the inverse of the shape's own scale around
        // `position`).
        CollisionShape::UniformScaling(child, f) => {
            local_inside(child, position, orientation, position + (point - position) / f)
        }
        CollisionShape::Convex2d(child) => {
            local_inside(child, position, orientation, point)
        }
    }
}

/// Ray vs sphere (local space, centered at the origin).
fn sphere_raycast(radius: Real, from: Vector3, to: Vector3) -> Option<LocalHit> {
    let d = to - from;
    let a = d.length_squared();
    let b = 2.0 * from.dot(d);
    let c = from.length_squared() - radius * radius;
    if c <= 0.0 {
        // Start inside (or on) the sphere.
        return Some((0.0, Vector3::X));
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - math::sqrt(disc)) / (2.0 * a);
    if t < 0.0 || t > 1.0 {
        return None;
    }
    let hit = from + d * t;
    Some((t, hit / radius))
}

/// Ray vs axis-aligned box (local space, centered at the origin): the slab
/// method. The entered slab records its axis so the exit-facing normal can be
/// reconstructed.
fn box_raycast(half: Vector3, from: Vector3, to: Vector3) -> Option<LocalHit> {
    let d = to - from;
    let mut tmin = Real::NEG_INFINITY;
    let mut tmax = Real::INFINITY;
    let mut hit_axis = 0;
    let mut hit_sign = 1.0;
    for axis in 0..3 {
        let o = from[axis];
        let v = d[axis];
        let h = half[axis];
        if v.abs() < 1e-12 {
            if o < -h || o > h {
                return None;
            }
        } else {
            let mut t1 = (-h - o) / v;
            let mut t2 = (h - o) / v;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            if t1 > tmin {
                tmin = t1;
                hit_axis = axis;
                // The entered face points against the travel direction.
                hit_sign = if v > 0.0 { -1.0 } else { 1.0 };
            }
            tmax = tmax.min(t2);
            if tmin > tmax {
                return None;
            }
        }
    }
    if tmax < 0.0 {
        return None;
    }
    let mut normal = Vector3::ZERO;
    normal[hit_axis] = hit_sign;
    Some((tmin.max(0.0), normal))
}

/// Ray vs single-sided infinite plane `normal . x = constant` (local space).
/// Only the front face (approached against `normal`) is hit.
fn plane_raycast(normal: Vector3, constant: Real, from: Vector3, to: Vector3) -> Option<LocalHit> {
    let d = to - from;
    let denom = normal.dot(d);
    if denom.abs() < 1e-12 {
        return None; // Parallel: the plane is a surface, a touch is a miss.
    }
    if denom > 0.0 {
        return None; // Back face: single-sided plane.
    }
    let t = (constant - normal.dot(from)) / denom;
    if t < 0.0 || t > 1.0 {
        return None;
    }
    Some((t, normal))
}

/// Ray vs capsule (local space): a segment along Y inflated by `radius`.
/// Intersects the side cylinder and both spherical end caps, taking the
/// earliest entry.
fn capsule_raycast(radius: Real, half_height: Real, from: Vector3, to: Vector3) -> Option<LocalHit> {
    let d = to - from;
    if segment_distance_squared(
        Vector3::ZERO,
        Vector3::new(0.0, half_height, 0.0),
        from,
    ) <= radius * radius
    {
        return Some((0.0, Vector3::X));
    }
    let mut best: Option<LocalHit> = None;
    // Side cylinder: the segment shrunk to its XZ projection.
    let o = Vector3::new(from.x, 0.0, from.z);
    let v = Vector3::new(d.x, 0.0, d.z);
    let a = v.length_squared();
    if a > 0.0 {
        let b = 2.0 * o.dot(v);
        let c = o.length_squared() - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let t = (-b - math::sqrt(disc)) / (2.0 * a);
            if t > 0.0 && t <= 1.0 {
                let y = from.y + d.y * t;
                if (-half_height..=half_height).contains(&y) {
                    let hit = from + d * t;
                    let radial = Vector3::new(hit.x, 0.0, hit.z);
                    best = Some((t, radial / radius));
                }
            }
        }
    }
    // Spherical end caps centered at +/- half_height.
    for (tip_y, sign) in [(half_height, 1.0), (-half_height, -1.0)] {
        let center = Vector3::new(0.0, tip_y, 0.0);
        if let Some((t, n)) = ray_sphere_cap(center, radius, from, to)
            && best.is_none_or(|(bt, _)| t < bt)
        {
            // Accept only hits on the cap side of the seam (y beyond tip_y).
            let hit = from + d * t;
            if hit.y * sign >= tip_y * sign - 1e-6 {
                best = Some((t, n));
            }
        }
    }
    best
}

/// Ray vs a sphere, returning the entry fraction and the outward unit normal.
fn ray_sphere_cap(
    center: Vector3,
    radius: Real,
    from: Vector3,
    to: Vector3,
) -> Option<LocalHit> {
    let oc = from - center;
    let d = to - from;
    let a = d.length_squared();
    let b = 2.0 * oc.dot(d);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - math::sqrt(disc)) / (2.0 * a);
    if t < 0.0 || t > 1.0 {
        return None;
    }
    let hit = from + d * t;
    Some((t, (hit - center) / radius))
}

/// Ray vs solid cylinder (local space, axis along Y).
fn cylinder_raycast(
    radius: Real,
    half_height: Real,
    from: Vector3,
    to: Vector3,
) -> Option<LocalHit> {
    let d = to - from;
    if from.y.abs() <= half_height && (from.x * from.x + from.z * from.z) <= radius * radius {
        return Some((0.0, Vector3::X));
    }
    let mut best: Option<LocalHit> = None;
    // Side surface.
    let o = Vector3::new(from.x, 0.0, from.z);
    let v = Vector3::new(d.x, 0.0, d.z);
    let a = v.length_squared();
    if a > 0.0 {
        let b = 2.0 * o.dot(v);
        let c = o.length_squared() - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let t = (-b - math::sqrt(disc)) / (2.0 * a);
            if t > 0.0 && t <= 1.0 {
                let y = from.y + d.y * t;
                if (-half_height..=half_height).contains(&y) {
                    let hit = from + d * t;
                    let radial = Vector3::new(hit.x, 0.0, hit.z);
                    best = Some((t, radial / radius));
                }
            }
        }
    }
    // Top and bottom discs.
    for (disc_y, sign) in [(half_height, 1.0), (-half_height, -1.0)] {
        if let Some((t, n)) = ray_disc(radius, disc_y, sign, from, to)
            && best.is_none_or(|(bt, _)| t < bt)
        {
            best = Some((t, n));
        }
    }
    best
}

/// Ray vs a disc lying in the plane `y = disc_y`, facing `sign` along Y.
fn ray_disc(
    radius: Real,
    disc_y: Real,
    sign: Real,
    from: Vector3,
    to: Vector3,
) -> Option<LocalHit> {
    let d = to - from;
    if d.y.abs() < 1e-12 {
        return None;
    }
    let t = (disc_y - from.y) / d.y;
    if t < 0.0 || t > 1.0 {
        return None;
    }
    let hit = from + d * t;
    if hit.x * hit.x + hit.z * hit.z > radius * radius {
        return None;
    }
    Some((t, Vector3::new(0.0, sign, 0.0)))
}

/// Ray vs right cone (local space): apex at `+half_height`, base disc at
/// `-half_height` with `radius`. The side radius shrinks linearly with height:
/// `rho(y) = radius * (half_height - y) / (2 * half_height)`.
fn cone_raycast(radius: Real, half_height: Real, from: Vector3, to: Vector3) -> Option<LocalHit> {
    if cone_radial(radius, half_height, from) <= 0.0 {
        return Some((0.0, Vector3::X));
    }
    let d = to - from;
    let mut best: Option<LocalHit> = None;
    // Side surface: substitute the ray into x^2 + z^2 = rho(y)^2.
    let h = half_height;
    let k = radius / (2.0 * h);
    let a = (d.x * d.x + d.z * d.z) - k * k * d.y * d.y;
    let b = 2.0 * (from.x * d.x + from.z * d.z) + 2.0 * k * k * d.y * (h - from.y);
    let c = (from.x * from.x + from.z * from.z) - k * k * (h - from.y) * (h - from.y);
    if let Some(t) = smallest_root(a, b, c)
        && t > 0.0
        && t <= 1.0
    {
        let hit = from + d * t;
        let y = hit.y;
        if (-h..=h).contains(&y) {
            // Gradient of x^2 + z^2 - k^2 (h - y)^2 = 0.
            let normal = Vector3::new(hit.x, k * k * (h - y), hit.z).normalize();
            best = Some((t, normal));
        }
    }
    // Base disc.
    if let Some((t, n)) = ray_disc(radius, -h, -1.0, from, to)
        && best.is_none_or(|(bt, _)| t < bt)
    {
        best = Some((t, n));
    }
    best
}

/// Signed "inside" test for a cone: negative inside, using the same radial
/// function as the side surface. Points outside `[-h, h]` on Y are always
/// outside.
fn cone_radial(radius: Real, half_height: Real, p: Vector3) -> Real {
    if p.y < -half_height || p.y > half_height {
        return Real::INFINITY;
    }
    let rho = radius * (half_height - p.y) / (2.0 * half_height);
    math::sqrt(p.x * p.x + p.z * p.z) - rho
}

/// Smallest real root of `a t^2 + b t + c = 0` in `[0, 1]`, if any.
fn smallest_root(a: Real, b: Real, c: Real) -> Option<Real> {
    let roots: Vec<Real> = if a.abs() < 1e-12 {
        // Linear.
        if b.abs() < 1e-12 {
            return None;
        }
        vec![-c / b]
    } else {
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let sq = math::sqrt(disc);
        vec![(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)]
    };
    roots
        .into_iter()
        .filter(|t| *t >= 0.0 && *t <= 1.0)
        .reduce(Real::min)
}

/// Ray vs a convex hull (local space). The hull stores only vertices, so the
/// entry is located by minimizing the point-to-hull distance along the ray:
/// the squared distance to a convex set is convex, so a ternary search finds
/// the closest approach; if it touches, a binary search on the decreasing
/// side locates the entry point.
fn convex_hull_raycast(points: &[Vector3], from: Vector3, to: Vector3) -> Option<LocalHit> {
    let dir = to - from;
    if hull_distance_squared(points, from) <= 1e-12 {
        return Some((0.0, Vector3::X));
    }
    // Ternary search for the closest approach (40 iterations, more than
    // enough on f32/f64).
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..40 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if hull_distance_squared(points, from + dir * m1)
            < hull_distance_squared(points, from + dir * m2)
        {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let s_min = (lo + hi) * 0.5;
    let d_min2 = hull_distance_squared(points, from + dir * s_min);
    if d_min2 > 1e-8 {
        return None; // Never touches the hull.
    }
    // Binary search on [0, s_min] (the distance is decreasing there) for the
    // first fraction inside the hull.
    let mut low = 0.0;
    let mut high = s_min;
    for _ in 0..60 {
        let mid = (low + high) * 0.5;
        if hull_distance_squared(points, from + dir * mid) <= 1e-8 {
            high = mid;
        } else {
            low = mid;
        }
    }
    let fraction = (low + high) * 0.5;
    // Surface normal: cast from a point slightly outside the entry (the GJK
    // distance direction is stable there) back to its closest hull point.
    let probe = from + dir * (fraction - 2e-4);
    let (_, v) = hull_distance_direction(points, probe);
    let normal = if v.length_squared() > 1e-12 { v } else { -dir / dir.length() };
    Some((fraction, normal))
}

/// Squared distance from `point` to a convex hull, via GJK distance on the
/// Minkowski difference of the point and the hull.
fn hull_distance_squared(points: &[Vector3], point: Vector3) -> Real {
    let (d2, _) = hull_distance_direction(points, point);
    d2
}

/// Squared distance and the separation direction (from the hull toward
/// `point`) via GJK distance.
///
/// The Minkowski difference `A - B` has `A = {point}`, so
/// `support(v) = point - sup_B(-v)`. The simplex is seeded with the support
/// point along the guess direction and grown toward the origin (support along
/// `-v`). The closest point `v` either collapses (point inside) or stops
/// improving when the support point repeats; the loop limit guards against
/// degenerate hulls.
fn hull_distance_direction(points: &[Vector3], point: Vector3) -> (Real, Vector3) {
    let support = |v: Vector3| point - hull_support(points, -v);

    let mut v = point - hull_centroid(points);
    if v.length_squared() < 1e-12 {
        v = Vector3::X;
    }
    let mut simplex: Vec<Vector3> = vec![support(v)];
    let mut v = simplex[0];
    for _ in 0..64 {
        if v.length_squared() < 1e-10 {
            return (0.0, Vector3::ZERO); // Point inside or touching the hull.
        }
        let w = support(-v);
        if simplex.iter().any(|p| (*p - w).length_squared() < 1e-12) {
            // No new vertex: the simplex already spans the closest feature.
            return (v.length_squared(), v / v.length());
        }
        simplex.push(w);
        v = closest_origin(&mut simplex);
    }
    // Exhausted iterations with the final refinement having enclosed the
    // origin (point inside the hull): the head check never saw the collapsed
    // `v`, and the division below would be 0/0. Same predicate as the head
    // check, applied to the final value.
    if v.length_squared() < 1e-10 {
        return (0.0, Vector3::ZERO); // Inside or touching after the last refinement.
    }
    (v.length_squared(), v / v.length())
}

/// Farthest hull vertex along `dir` (local space).
fn hull_support(points: &[Vector3], dir: Vector3) -> Vector3 {
    let mut best = Vector3::ZERO;
    let mut best_dot = Real::NEG_INFINITY;
    for p in points {
        let d = p.dot(dir);
        if d > best_dot {
            best_dot = d;
            best = *p;
        }
    }
    best
}

/// Vertex centroid of the hull (a cheap interior point for the initial GJK
/// direction; always inside a convex hull).
fn hull_centroid(points: &[Vector3]) -> Vector3 {
    let sum = points.iter().fold(Vector3::ZERO, |acc, p| acc + *p);
    sum / points.len() as Real
}

/// Ray vs a compound: cast against every child (recursively, in its local
/// frame) and keep the closest hit.
fn compound_raycast(
    compound: &Compound,
    inv: Quaternion,
    position: Vector3,
    ray_from: Vector3,
    ray_to: Vector3,
) -> Option<LocalHit> {
    let mut best: Option<LocalHit> = None;
    for child in &compound.children {
        // World transform of the child, then back into its local frame.
        let child_pos = position + inv.inverse() * child.position;
        let child_ori = inv.inverse() * child.orientation;
        if let Some(hit) =
            local_raycast(&child.shape, child_ori.inverse(), child_pos, ray_from, ray_to)
        {
            if hit.0 == 0.0 {
                return Some(hit);
            }
            if best.is_none_or(|(bt, _)| hit.0 < bt) {
                best = Some(hit);
            }
        }
    }
    best
}

/// Squared distance from `point` to the segment `a-b`.
fn segment_distance_squared(a: Vector3, b: Vector3, point: Vector3) -> Real {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= 0.0 {
        return (point - a).length_squared();
    }
    let t = ((point - a).dot(ab) / len2).clamp(0.0, 1.0);
    (point - (a + ab * t)).length_squared()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ray(shape: &CollisionShape, from: Vector3, to: Vector3) -> Option<RayHit> {
        raycast(shape, Vector3::ZERO, Quaternion::IDENTITY, from, to)
    }

    #[test]
    fn sphere_hit_from_outside() {
        let s = CollisionShape::sphere(1.0);
        let h = ray(&s, Vector3::new(3.0, 0.0, 0.0), Vector3::new(-3.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(h.fraction, 1.0 / 3.0, epsilon = 1e-5);
        assert_relative_eq!(h.position.x, 1.0, epsilon = 1e-5);
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn sphere_interior_start_reports_fraction_zero() {
        let s = CollisionShape::sphere(1.0);
        let h = ray(&s, Vector3::ZERO, Vector3::new(3.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.fraction, 0.0);
        // Interior starts report the ray direction as the normal.
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn sphere_misses() {
        let s = CollisionShape::sphere(1.0);
        assert!(ray(&s, Vector3::new(3.0, 2.0, 0.0), Vector3::new(5.0, 2.0, 0.0)).is_none());
    }

    #[test]
    fn box_front_face_hit() {
        let b = CollisionShape::box_shape(Vector3::splat(1.0));
        let h = ray(&b, Vector3::new(5.0, 0.0, 0.0), Vector3::new(-5.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(h.fraction, 0.4, epsilon = 1e-5);
        assert_relative_eq!(h.position.x, 1.0, epsilon = 1e-5);
        // The normal points back toward the ray origin (+X).
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn box_interior_start() {
        let b = CollisionShape::box_shape(Vector3::splat(1.0));
        let h = ray(&b, Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.fraction, 0.0);
    }

    #[test]
    fn rotated_box_normal_is_rotated_back() {
        let b = CollisionShape::box_shape(Vector3::new(1.0, 0.5, 1.0));
        let q = Quaternion::from_rotation_z(bullet_math::FRAC_PI_2);
        let h = raycast(&b, Vector3::ZERO, q, Vector3::new(3.0, 0.0, 0.0), Vector3::ZERO).unwrap();
        // The +Y half-width now faces +X; the hit is at x = 0.5.
        assert_relative_eq!(h.fraction, 5.0 / 6.0, epsilon = 1e-4);
        assert_relative_eq!(h.position.x, 0.5, epsilon = 1e-4);
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn plane_front_hit() {
        let p = CollisionShape::plane(Vector3::Y, 0.0);
        let h = ray(&p, Vector3::new(0.0, 5.0, 0.0), Vector3::new(0.0, -5.0, 0.0)).unwrap();
        assert_relative_eq!(h.fraction, 0.5, epsilon = 1e-6);
        assert_relative_eq!(h.normal.y, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn plane_back_face_and_parallel_miss() {
        let p = CollisionShape::plane(Vector3::Y, 0.0);
        assert!(ray(&p, Vector3::new(0.0, -5.0, 0.0), Vector3::new(0.0, 5.0, 0.0)).is_none());
        assert!(ray(&p, Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 0.0)).is_none());
    }

    #[test]
    fn capsule_side_hit() {
        let c = CollisionShape::capsule(0.5, 1.0);
        let h = ray(&c, Vector3::new(3.0, 0.0, 0.0), Vector3::ZERO).unwrap();
        // Side cylinder at |x| = 0.5.
        assert_relative_eq!(h.fraction, 2.5 / 3.0, epsilon = 1e-5);
        assert_relative_eq!(h.position.x, 0.5, epsilon = 1e-5);
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn capsule_end_cap_hit_along_axis() {
        let c = CollisionShape::capsule(0.5, 1.0);
        let h = ray(&c, Vector3::new(0.0, 2.5, 0.0), Vector3::new(0.0, -1.5, 0.0)).unwrap();
        // Top hemisphere centered at y = 1 with radius 0.5: entry at y = 1.5.
        assert_relative_eq!(h.fraction, 0.25, epsilon = 1e-5);
        assert_relative_eq!(h.position.y, 1.5, epsilon = 1e-5);
        assert_relative_eq!(h.normal.y, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn capsule_interior_start() {
        let c = CollisionShape::capsule(0.5, 1.0);
        let h = ray(&c, Vector3::ZERO, Vector3::new(0.0, 2.0, 0.0)).unwrap();
        assert_eq!(h.fraction, 0.0);
    }

    #[test]
    fn cylinder_side_hit() {
        let c = CollisionShape::cylinder(0.5, 1.0);
        let h = ray(&c, Vector3::new(3.0, 0.5, 0.0), Vector3::new(0.0, 0.5, 0.0)).unwrap();
        assert_relative_eq!(h.fraction, 2.5 / 3.0, epsilon = 1e-5);
        assert_relative_eq!(h.position.x, 0.5, epsilon = 1e-5);
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn cylinder_cap_hit_from_above() {
        let c = CollisionShape::cylinder(0.5, 1.0);
        let h = ray(&c, Vector3::new(0.0, 3.0, 0.0), Vector3::ZERO).unwrap();
        assert_relative_eq!(h.fraction, 2.0 / 3.0, epsilon = 1e-5);
        assert_relative_eq!(h.position.y, 1.0, epsilon = 1e-5);
        assert_relative_eq!(h.normal.y, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn cylinder_interior_start() {
        let c = CollisionShape::cylinder(0.5, 1.0);
        let h = ray(&c, Vector3::ZERO, Vector3::new(0.0, 2.0, 0.0)).unwrap();
        assert_eq!(h.fraction, 0.0);
    }

    #[test]
    fn cone_side_hit() {
        let c = CollisionShape::cone(0.5, 1.0);
        let h = ray(&c, Vector3::new(3.0, 0.0, 0.0), Vector3::ZERO).unwrap();
        // Side radius at y = 0 is 0.5 * (1 - 0) / 2 = 0.25.
        assert_relative_eq!(h.fraction, 2.75 / 3.0, epsilon = 1e-5);
        assert_relative_eq!(h.position.x, 0.25, epsilon = 1e-5);
        // The side normal at the hit: (x, k^2 (h - y), 0), k = r / 2h = 0.25.
        let n = Vector3::new(0.25, 0.25 * 0.25 * 1.0, 0.0).normalize();
        assert_relative_eq!(h.normal.x, n.x, epsilon = 1e-4);
        assert_relative_eq!(h.normal.y, n.y, epsilon = 1e-4);
    }

    #[test]
    fn cone_base_disc_hit() {
        let c = CollisionShape::cone(0.5, 1.0);
        let h = ray(&c, Vector3::new(0.0, -3.0, 0.0), Vector3::new(0.0, 1.0, 0.0)).unwrap();
        assert_relative_eq!(h.fraction, 0.5, epsilon = 1e-5);
        assert_relative_eq!(h.position.y, -1.0, epsilon = 1e-5);
        assert_relative_eq!(h.normal.y, -1.0, epsilon = 1e-5);
    }

    #[test]
    fn cone_interior_start() {
        let c = CollisionShape::cone(0.5, 1.0);
        let h = ray(&c, Vector3::ZERO, Vector3::new(0.0, 2.0, 0.0)).unwrap();
        assert_eq!(h.fraction, 0.0);
    }

    fn octahedron() -> CollisionShape {
        CollisionShape::convex_hull(vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ])
    }

    #[test]
    fn convex_hull_hit_along_axis() {
        let h = octahedron();
        let hit = ray(&h, Vector3::new(3.0, 0.0, 0.0), Vector3::new(-3.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(hit.fraction, 1.0 / 3.0, epsilon = 1e-3);
        assert_relative_eq!(hit.position.x, 1.0, epsilon = 1e-3);
        assert_relative_eq!(hit.normal.x, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn convex_hull_interior_start_and_miss() {
        let h = octahedron();
        let hit = ray(&h, Vector3::ZERO, Vector3::new(3.0, 0.0, 0.0)).unwrap();
        assert_eq!(hit.fraction, 0.0);
        assert!(ray(&h, Vector3::new(5.0, 2.0, 0.0), Vector3::new(5.0, 4.0, 0.0)).is_none());
    }

    #[test]
    fn compound_keeps_the_closest_child() {
        let compound = CollisionShape::compound(vec![
            crate::shapes::CompoundChild::new(CollisionShape::sphere(0.5), Vector3::new(-1.0, 0.0, 0.0)),
            crate::shapes::CompoundChild::new(CollisionShape::sphere(0.5), Vector3::new(1.0, 0.0, 0.0)),
        ]);
        // The +X child (surface at x = 1.5) is hit before the -X child.
        let h = ray(&compound, Vector3::new(3.0, 0.0, 0.0), Vector3::new(-3.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(h.fraction, 0.25, epsilon = 1e-4);
        assert_relative_eq!(h.position.x, 1.5, epsilon = 1e-4);
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn compound_interior_start() {
        let compound = CollisionShape::compound(vec![
            crate::shapes::CompoundChild::new(CollisionShape::sphere(0.5), Vector3::ZERO),
        ]);
        let h = ray(&compound, Vector3::ZERO, Vector3::new(3.0, 0.0, 0.0)).unwrap();
        assert_eq!(h.fraction, 0.0);
    }

    #[test]
    fn degenerate_ray_hits_only_when_inside() {
        let s = CollisionShape::sphere(1.0);
        let h = ray(&s, Vector3::ZERO, Vector3::ZERO).unwrap();
        assert_eq!(h.fraction, 0.0);
        assert!(ray(&s, Vector3::new(2.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0)).is_none());
    }

    fn unit_ground_mesh() -> CollisionShape {
        CollisionShape::triangle_mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn triangle_mesh_raycast_through_shape_api() {
        let m = unit_ground_mesh();
        let h = ray(&m, Vector3::new(0.5, 5.0, 0.5), Vector3::new(0.5, -1.0, 0.5)).unwrap();
        assert_relative_eq!(h.fraction, 5.0 / 6.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal.y, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn heightfield_raycast_through_shape_api() {
        let hf = CollisionShape::heightfield(vec![0.0; 9], 3, 3, Vector3::ONE);
        let h = ray(&hf, Vector3::new(1.0, 5.0, 1.0), Vector3::new(1.0, -1.0, 1.0)).unwrap();
        assert_relative_eq!(h.fraction, 5.0 / 6.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal.y, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn rotated_mesh_raycast_rotates_the_normal() {
        let m = unit_ground_mesh();
        // Rotate the mesh 90 deg about Z: the surface normal faces world +X.
        let q = Quaternion::from_rotation_z(bullet_math::FRAC_PI_2);
        let h = raycast(
            &m,
            Vector3::ZERO,
            q,
            Vector3::new(3.0, 0.5, 0.5),
            Vector3::new(-1.0, 0.5, 0.5),
        )
        .unwrap();
        assert_relative_eq!(h.normal.x, 1.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal.y, 0.0, epsilon = 1e-4);
    }
}
