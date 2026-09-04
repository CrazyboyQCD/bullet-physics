//! Gilbert-Johnson-Keerthi convex-convex intersection test.
//!
//! Uses the Minkowski difference `A - B` support mapping and a simplex of up
//! to four points. When the origin is enclosed, the returned simplex seeds an
//! EPA pass to compute penetration depth.

use bullet_math::{math, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
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

/// Support point of the *core* Minkowski difference `core(A) - core(B)`:
/// what the GJK/EPA contact pipeline queries (see [`gjk_contact`]).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn core_support_minkowski(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
    dir: Vector3,
) -> Vector3 {
    shape_a.core_support_world(position_a, orientation_a, dir)
        - shape_b.core_support_world(position_b, orientation_b, -dir)
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

/// Outcome of the GJK contact walk ([`gjk_contact`]): the walk either proves
/// separation (with the exact surface distance and the closest-point
/// witnesses when it falls inside the contact band) or reports overlap with
/// a simplex seeding EPA.
#[derive(Debug, Clone)]
pub enum GjkOutcome {
    /// Separated beyond the contact threshold: no contact.
    Separated,
    /// Inside the contact band (Bullet's positive-distance manifold point):
    /// `distance` is the *signed* true surface separation of the full shapes
    /// — positive while separated inside the band (a velocity-only contact
    /// with negative penetration), negative while the full shapes overlap
    /// within the margin shell (the walk resolved it exactly; EPA is not
    /// needed). `normal` points from A to B and `point_on_b` is the witness
    /// on B, matching the witness-on-B convention of the overlap path.
    Band {
        distance: Real,
        normal: Vector3,
        point_on_b: Vector3,
    },
    /// Overlapping past the margin shell: an origin-enclosing simplex (four
    /// points) of the *cores* that seeds the EPA penetration pass.
    Overlap(Simplex),
}

/// One support vertex of the Minkowski difference together with the shape-B
/// witness it came from (Bullet's `btSupportVector` stores both shapes'
/// witnesses; the A-side witness is redundant — it reconstructs as
/// `p_b + normal * penetration`, exactly what the manifold capture does).
#[derive(Clone, Copy)]
struct SupportVertex {
    w: Vector3,
    p_b: Vector3,
}

/// Length-squared below which the walk is considered to have reached the
/// origin (touching or overlapping): the exact distance is numerically
/// unreliable there, so the pair falls back to the intersection walk + EPA
/// (Bullet's `REL_ERROR2` degeneracy gate).
const TOUCH_TOLERANCE2: Real = 1e-12;

/// GJK as a *distance* walk over the margin **cores** (Bullet's margin
/// architecture, `btGjkPairDetector::getClosestPointsNonVirtual`): each
/// shape is its [`core_support_world`](CollisionShape::core_support_world)
/// shrink, and the margins are added back at the walk's exit. The closest
/// point `v` of the core Minkowski difference to the origin is refined with
/// support queries along `-v`, and the supporting-plane certificate
/// `v . w >= |v|^2` (with `w` the support along `-v`, i.e. the minimizer of
/// `v . x`) proves separation with core distance `|v|` exactly, so the true
/// full-shape separation is `|v| - (marginA + marginB)`. Within the contact
/// `threshold` the separation is reported as a margin-band contact instead
/// of `Separated` — Bullet's `m_maximumDistanceSquared` route, which never
/// calls EPA for separated pairs; cores separated by less than the margin
/// sum mean the full shapes already overlap, and the walk resolves that
/// overlap exactly too (a ball-Minkowski property), so EPA only runs for
/// deep overlap past the margin shell. `seed` optionally warm-starts the
/// walk (a cached separating normal); overlapping cores fall back to the
/// strict enclosing walk and return its simplex for EPA.
#[allow(clippy::too_many_arguments)]
pub fn gjk_contact(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
    seed: Option<Vector3>,
    threshold: Real,
) -> GjkOutcome {
    let margin_a = shape_a.margin();
    let margin_b = shape_b.margin();
    let margin_sum = margin_a + margin_b;
    let support = |dir: Vector3| -> SupportVertex {
        let p_a = shape_a.core_support_world(position_a, orientation_a, dir);
        let p_b = shape_b.core_support_world(position_b, orientation_b, -dir);
        SupportVertex { w: p_a - p_b, p_b }
    };

    let center_dir = position_b - position_a;
    let fallback = if center_dir.length_squared() > 1e-12 {
        center_dir
    } else {
        Vector3::X
    };
    let initial_dir = seed
        .filter(|s| s.length_squared() > 1e-12)
        .unwrap_or(fallback);

    let first = support(initial_dir);
    // The simplex grows to at most four entries; `reduce` shrinks it back
    // to the closest feature before each push.
    let mut buf = [first; 4];
    let mut len = 1usize;
    // Last search direction, kept for the overlap fallback's seed.
    let mut dir = -first.w;

    for _ in 0..MAX_ITERATIONS {
        match reduce(&mut buf, &mut len) {
            ReduceStep::Inside => {
                // The four buffered vertices already enclose the origin — use
                // them directly as the EPA seed (no second walk).
                return GjkOutcome::Overlap(Simplex {
                    pts: [buf[0].w, buf[1].w, buf[2].w, buf[3].w],
                    len: 4,
                });
            }
            ReduceStep::Closest { v, p_b } => {
                let v2 = v.length_squared();
                if v2 < TOUCH_TOLERANCE2 {
                    return match enclosing_walk(|d| support(d).w, fallback, Some(dir)) {
                        Some(simplex) if simplex.points().len() == 4 => {
                            GjkOutcome::Overlap(simplex)
                        }
                        _ => GjkOutcome::Separated,
                    };
                }
                dir = -v;
                let w = support(dir);
                let delta = v.dot(w.w);
                // Early rejection (Bullet's `delta * delta > squaredDistance
                // * m_maximumDistanceSquared`): every core Minkowski point
                // `x` satisfies `|x| >= v_hat . x >= v_hat . w`, so a
                // projection beyond the core-space band radius (margins +
                // threshold) proves the pair is separated past the threshold
                // without converging the walk — the cheap exit far pairs take
                // on their first or second support query.
                let band = margin_sum + threshold;
                if delta > 0.0 && delta * delta > v2 * band * band {
                    return GjkOutcome::Separated;
                }
                // Bullet's `inSimplex` gate: a support that duplicates a
                // buffered vertex proves the walk cannot improve (the
                // supporting plane already contains the closest feature), so
                // `v` is the exact closest point. Without this, flat cores
                // (e.g. a sphere's point core against a triangle hull)
                // cycle on duplicated supports until the iteration cap.
                let duplicate = (0..len).any(|i| w.w.abs_diff_eq(buf[i].w, 1e-6));
                // Separation certificate / convergence, Bullet's combined exit:
                // `delta = v . w` is the support projection along the closest
                // direction, so `delta >= |v|^2` proves the whole core
                // Minkowski difference lies beyond the supporting plane at
                // `v` (exact core separation with distance `|v|`), and
                // `delta >= |v|^2 (1 - REL_ERROR2)` is Bullet's `f0 <= f1`
                // no-progress exit (the remaining guaranteed improvement fell
                // below the relative tolerance).
                if duplicate || delta >= v2 * (1.0 - REL_ERROR2) {
                    let core_distance = math::sqrt(v2);
                    // The true full-shape separation: core distance minus the
                    // margins added back (Bullet's `distance = 1/rlen -
                    // margin`, sign-flipped for our separation-positive
                    // convention).
                    let distance = core_distance - margin_sum;
                    if distance < threshold {
                        let normal = dir / core_distance;
                        // B's witness moves out of the core toward A by its
                        // margin (the closest point of the ball-inflated
                        // full shape).
                        return GjkOutcome::Band {
                            distance,
                            normal,
                            point_on_b: p_b - normal * margin_b,
                        };
                    }
                    return GjkOutcome::Separated;
                }
                buf[len] = w;
                len += 1;
            }
        }
    }
    // Iteration cap: report the best distance found (matches the strict
    // walk's behavior of giving up on degenerate inputs).
    GjkOutcome::Separated
}

/// Relative progress below which the walk stops improving (Bullet's
/// `REL_ERROR2`, single precision).
const REL_ERROR2: Real = 1e-6;

/// The strict origin-enclosing walk, parameterized by the support function
/// of the Minkowski difference: full shapes for [`gjk_intersect_seeded`],
/// shrunk cores for the contact walk's EPA fallback (the seed simplex must
/// be consistent with the supports EPA will query). A failed walk
/// (degenerate input) reports `None`, like `gjk_intersect_seeded`.
fn enclosing_walk(
    support: impl Fn(Vector3) -> Vector3,
    fallback: Vector3,
    seed: Option<Vector3>,
) -> Option<Simplex> {
    let initial_dir = seed
        .filter(|s| s.length_squared() > 1e-12)
        .unwrap_or(fallback);

    let mut simplex = Simplex::new(support(initial_dir));
    let mut dir = -simplex.points()[0];

    for _ in 0..MAX_ITERATIONS {
        let a = support(dir);
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

enum ReduceStep {
    /// The origin is enclosed by the simplex (or touching it): overlap.
    Inside,
    Closest { v: Vector3, p_b: Vector3 },
}

/// Shrink the simplex to its closest feature to the origin, returning the
/// closest point on the Minkowski difference and the interpolated shape
/// witnesses. Mirrors the region tests of the strict walk's `solve_*`
/// routines (Ericson 5.1.5) with barycentric weights for the witnesses.
fn reduce(buf: &mut [SupportVertex; 4], len: &mut usize) -> ReduceStep {
    match *len {
        1 => {
            let s = buf[0];
            ReduceStep::Closest { v: s.w, p_b: s.p_b }
        }
        2 => reduce_segment(buf, len),
        3 => reduce_triangle(buf, len),
        _ => reduce_tetrahedron(buf, len),
    }
}

fn reduce_segment(buf: &mut [SupportVertex; 4], len: &mut usize) -> ReduceStep {
    let a = buf[1];
    let b = buf[0];
    let ab = b.w - a.w;
    let denom = ab.length_squared();
    if denom < 1e-24 {
        *len = 1;
        buf[0] = a;
        return ReduceStep::Closest { v: a.w, p_b: a.p_b };
    }
    let t = -a.w.dot(ab) / denom;
    if t <= 0.0 {
        *len = 1;
        buf[0] = a;
        ReduceStep::Closest { v: a.w, p_b: a.p_b }
    } else if t >= 1.0 {
        *len = 1;
        buf[0] = b;
        ReduceStep::Closest { v: b.w, p_b: b.p_b }
    } else {
        *len = 2;
        buf[0] = a;
        buf[1] = b;
        ReduceStep::Closest {
            v: a.w + ab * t,
            p_b: a.p_b + (b.p_b - a.p_b) * t,
        }
    }
}

/// Closest point of the triangle `buf[2], buf[1], buf[0]` (a, b, c) to the
/// origin with barycentric weights; shrinks the simplex to the closest
/// feature. Returns `(closest, weights over (a, b, c))`.
fn closest_triangle_bary(
    a: SupportVertex,
    b: SupportVertex,
    c: SupportVertex,
) -> (Vector3, [Real; 3]) {
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let ap = -a.w;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a.w, [1.0, 0.0, 0.0]);
    }
    let bp = -b.w;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b.w, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return (a.w + ab * t, [1.0 - t, t, 0.0]);
    }
    let cp = -c.w;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c.w, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return (a.w + ac * t, [1.0 - t, 0.0, t]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b.w + (c.w - b.w) * t, [0.0, 1.0 - t, t]);
    }
    // Interior: keep the whole triangle.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a.w + ab * v + ac * w, [1.0 - v - w, v, w])
}

fn reduce_triangle(buf: &mut [SupportVertex; 4], len: &mut usize) -> ReduceStep {
    let (a, b, c) = (buf[2], buf[1], buf[0]);
    let (v, weights) = closest_triangle_bary(a, b, c);
    // Rebuild the simplex from the vertices with nonzero weight.
    let mut kept = [a; 4];
    let mut n = 0usize;
    for (vertex, weight) in [(a, weights[0]), (b, weights[1]), (c, weights[2])] {
        if weight > 0.0 {
            kept[n] = vertex;
            n += 1;
        }
    }
    if n == 0 {
        n = 1;
        kept[0] = a;
    }
    buf[..n].copy_from_slice(&kept[..n]);
    *len = n;
    let p_b = a.p_b * weights[0] + b.p_b * weights[1] + c.p_b * weights[2];
    ReduceStep::Closest { v, p_b }
}

fn reduce_tetrahedron(buf: &mut [SupportVertex; 4], len: &mut usize) -> ReduceStep {
    let (a, b, c, d) = (buf[3], buf[2], buf[1], buf[0]);
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let ad = d.w - a.w;
    // Coplanar degenerate tetrahedron (flat Minkowski faces, e.g. triangle
    // hulls): fall back to the closest of the four faces, like
    // `closest_origin_tetrahedron`.
    if ab.cross(ac).dot(ad).abs() < 1e-12 {
        let faces = [(a, b, c), (a, b, d), (a, c, d), (b, c, d)];
        let mut best_face = faces[0];
        let mut best_v2 = Real::INFINITY;
        let mut best = closest_triangle_bary(faces[0].0, faces[0].1, faces[0].2);
        for &(x, y, z) in &faces[1..] {
            let cand = closest_triangle_bary(x, y, z);
            let v2 = cand.0.length_squared();
            if v2 < best_v2 {
                best_v2 = v2;
                best = cand;
                best_face = (x, y, z);
            }
        }
        let (_, weights) = best;
        let mut kept = [buf[0]; 4];
        let mut n = 0usize;
        for (vertex, weight) in [
            (best_face.0, weights[0]),
            (best_face.1, weights[1]),
            (best_face.2, weights[2]),
        ] {
            if weight > 0.0 {
                kept[n] = vertex;
                n += 1;
            }
        }
        if n == 0 {
            n = 1;
            kept[0] = best_face.0;
        }
        buf[..n].copy_from_slice(&kept[..n]);
        *len = n;
        let p_b = best_face.0.p_b * weights[0]
            + best_face.1.p_b * weights[1]
            + best_face.2.p_b * weights[2];
        return ReduceStep::Closest { v: best.0, p_b };
    }
    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);
    let ao = -a.w;
    let bo = -b.w;
    let bcd = (c.w - b.w).cross(d.w - b.w);
    if abc.dot(ao) * abc.dot(ad) < 0.0 {
        return triangle_step(buf, len, a, b, c);
    }
    if acd.dot(ao) * acd.dot(ab) < 0.0 {
        return triangle_step(buf, len, a, c, d);
    }
    if adb.dot(ao) * adb.dot(ac) < 0.0 {
        return triangle_step(buf, len, a, d, b);
    }
    if bcd.dot(bo) * bcd.dot(a.w - b.w) < 0.0 {
        return triangle_step(buf, len, b, c, d);
    }
    // Origin inside the tetrahedron.
    ReduceStep::Inside
}

/// Store the three face vertices and defer to the triangle reduction.
#[inline]
fn triangle_step(
    buf: &mut [SupportVertex; 4],
    len: &mut usize,
    a: SupportVertex,
    b: SupportVertex,
    c: SupportVertex,
) -> ReduceStep {
    buf[0] = c;
    buf[1] = b;
    buf[2] = a;
    *len = 3;
    reduce_triangle(buf, len)
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
    enclosing_walk(
        |dir| {
            support_minkowski(
                shape_a,
                position_a,
                orientation_a,
                shape_b,
                position_b,
                orientation_b,
                dir,
            )
        },
        fallback,
        seed,
    )
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
                    .unwrap_or(core::cmp::Ordering::Equal)
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

    fn contact(
        a: &CollisionShape,
        pa: Vector3,
        b: &CollisionShape,
        pb: Vector3,
        threshold: Real,
    ) -> GjkOutcome {
        gjk_contact(
            a,
            pa,
            Quaternion::IDENTITY,
            b,
            pb,
            Quaternion::IDENTITY,
            None,
            threshold,
        )
    }

    /// Separated spheres inside the contact band report the exact distance,
    /// the A->B normal, and the witness on B.
    #[test]
    fn contact_walk_reports_band_separation() {
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::sphere(0.5);
        // Gap of 0.01 along X, inside the 0.0173 band.
        match contact(&a, Vector3::ZERO, &b, Vector3::new(1.01, 0.0, 0.0), 0.0173) {
            GjkOutcome::Band { distance, normal, point_on_b } => {
                assert!((distance - 0.01).abs() < 1e-4, "distance {distance}");
                assert!((normal - Vector3::X).length() < 1e-4);
                // Witness on B: the closest point of B toward A.
                assert!((point_on_b - Vector3::new(0.51, 0.0, 0.0)).length() < 1e-4);
            }
            other => panic!("expected Band, got {other:?}"),
        }
        // Outside the band: Separated.
        assert!(matches!(
            contact(&a, Vector3::ZERO, &b, Vector3::new(1.5, 0.0, 0.0), 0.0173),
            GjkOutcome::Separated
        ));
    }

    /// Overlapping shapes within the margin shell are resolved by the walk
    /// (signed Band distance); a sphere pair reports the exact signed gap,
    /// and a rotated pair converges to the same distance from any seed
    /// (seed invariance of the walk's fixed point).
    #[test]
    fn contact_walk_routes_overlap_and_rotated_band() {
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::sphere(0.5);
        // Overlap 0.3 lives inside the 1.0 margin shell: the core walk
        // resolves it exactly as a Band with negative distance (EPA stays
        // reserved for overlap past the margin shell).
        match contact(&a, Vector3::ZERO, &b, Vector3::new(0.7, 0.0, 0.0), 0.0173) {
            GjkOutcome::Band { distance, .. } => {
                assert!((distance + 0.3).abs() < 1e-4, "distance {distance}");
            }
            other => panic!("expected Band, got {other:?}"),
        }

        let cap = CollisionShape::capsule(0.5, 1.0);
        match contact(&cap, Vector3::ZERO, &cap, Vector3::new(1.01, 0.0, 0.0), 0.5) {
            GjkOutcome::Band { distance, .. } => {
                assert!((distance - 0.01).abs() < 1e-4, "distance {distance}");
            }
            other => panic!("expected Band, got {other:?}"),
        }

        let q = Quaternion::from_rotation_z(0.5);
        let pb = Vector3::new(1.6, 0.2, 0.1);
        let reference = match gjk_contact(
            &cap,
            Vector3::ZERO,
            q,
            &cap,
            pb,
            Quaternion::IDENTITY,
            None,
            0.5,
        ) {
            GjkOutcome::Band { distance, .. } => distance,
            other => panic!("expected Band, got {other:?}"),
        };
        for seed in [
            Vector3::X,
            Vector3::Y,
            Vector3::new(0.3, -0.5, 0.8).normalize(),
        ] {
            match gjk_contact(
                &cap,
                Vector3::ZERO,
                q,
                &cap,
                pb,
                Quaternion::IDENTITY,
                Some(seed),
                0.5,
            ) {
                GjkOutcome::Band { distance, .. } => {
                    assert!(
                        (distance - reference).abs() < 2e-3,
                        "seeded distance {distance} vs reference {reference}"
                    );
                }
                other => panic!("expected Band, got {other:?}"),
            }
        }
    }
}
