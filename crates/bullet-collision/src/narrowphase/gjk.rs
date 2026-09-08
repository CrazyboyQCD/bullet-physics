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
/// (Bullet's `REL_ERROR2` degeneracy gate). The walk scales the gate by the
/// first support's squared length (floored at 1, so unit-scale walks keep
/// the absolute threshold bit-for-bit): on large coordinates the walk's own
/// cancellation noise sits orders of magnitude above the absolute gate,
/// which would otherwise miss the origin and turn a deep overlap into a
/// margin-only Band report instead of the EPA route.
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
    // Origin gate, scaled by the walk's own coordinate magnitude (see
    // `TOUCH_TOLERANCE2`).
    let touch2 = TOUCH_TOLERANCE2 * first.w.length_squared().max(1.0);

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
                if v2 < touch2 {
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
        // Strict-walk invariant: `dir` is finite and non-zero at the support
        // query. The simplex reductions can still hand back a zero direction
        // (a vertex/face collapsed onto the origin), so enforce it here —
        // a perpendicular of the latest support point is a valid restart
        // direction, and a fully zero Minkowski difference (coincident point
        // cores) deterministically falls back to an axis.
        let dir2 = dir.length_squared();
        if dir2 <= 0.0 || dir2.is_nan() {
            dir = perpendicular(simplex.points()[0]);
        }
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
        // Zero denominator means a == b (both d's are zero): the edge is a point.
        if d1 == d3 {
            return (a.w, [1.0, 0.0, 0.0]);
        }
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
        // Zero denominator means a == c.
        if d2 == d6 {
            return (a.w, [1.0, 0.0, 0.0]);
        }
        let t = d2 / (d2 - d6);
        return (a.w + ac * t, [1.0 - t, 0.0, t]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        // Zero denominator means b == c.
        if (d4 - d3) + (d5 - d6) == 0.0 {
            return (b.w, [0.0, 1.0, 0.0]);
        }
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b.w + (c.w - b.w) * t, [0.0, 1.0 - t, t]);
    }
    // Interior: keep the whole triangle. A degenerate triangle (duplicated
    // or collinear vertices) has zero determinant sum and no interior:
    // return the closest vertex one-hot instead of dividing by zero
    // (`sum <= 0.0 || sum.is_nan()` also catches NaN; well-conditioned triangles have
    // sum = |ab x ac|^2 > 0 and never enter this branch). An NaN here would
    // otherwise reach the contact walk as a fake `Inside` classification
    // (every NaN comparison reads as "not outside"), so this guard is what
    // keeps flat Minkowski hulls (thin plates, sphere point cores) from
    // producing NaN seeds.
    let sum = va + vb + vc;
    if sum <= 0.0 || sum.is_nan() {
        let (q, weights) = if a.w.length_squared() <= b.w.length_squared()
            && a.w.length_squared() <= c.w.length_squared()
        {
            (a.w, [1.0, 0.0, 0.0])
        } else if b.w.length_squared() <= c.w.length_squared() {
            (b.w, [0.0, 1.0, 0.0])
        } else {
            (c.w, [0.0, 0.0, 1.0])
        };
        return (q, weights);
    }
    let denom = 1.0 / sum;
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

/// An arbitrary unit vector perpendicular to `v`. The helper axis choice
/// above is kept as-is (successful paths are bit-identical), but the
/// primitive now guarantees a finite, non-zero result: the old
/// `v.cross(helper).normalize()` produced a NaN whenever `v` was (near-)
/// parallel to the helper — the absolute `0.9` test misses short
/// axis-aligned vectors like `(0.5, 0, 0)`, which discrete support mappings
/// do produce for duplicate simplex vertices. If the cross product
/// collapses, the other axis is tried (a non-zero finite `v` cannot be
/// parallel to both); a zero or non-finite `v` (whose comparisons all read
/// as false) falls back to a deterministic axis, because the strict walk
/// must never hand a zero/NaN direction to a support query.
fn perpendicular(v: Vector3) -> Vector3 {
    let helper = if v.x.abs() < 0.9 { Vector3::X } else { Vector3::Y };
    let perp = v.cross(helper);
    if perp.length_squared() > 0.0 {
        return perp.normalize();
    }
    let alt_helper = if v.x.abs() < 0.9 { Vector3::Y } else { Vector3::X };
    let perp = v.cross(alt_helper);
    if perp.length_squared() > 0.0 {
        return perp.normalize();
    }
    Vector3::Z
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
        // Zero denominator means a == b (both d's are zero): the edge is a point.
        let v = if d1 == d3 { 0.0 } else { d1 / (d1 - d3) };
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
        // Zero denominator means a == c.
        let w = if d2 == d6 { 0.0 } else { d2 / (d2 - d6) };
        let closest = a + ac * w;
        return shrink_or_normal(simplex, dir, closest, normal, [c, a]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        // Zero denominator means b == c.
        let w = if (d4 - d3) + (d5 - d6) == 0.0 {
            0.0
        } else {
            (d4 - d3) / ((d4 - d3) + (d5 - d6))
        };
        let closest = b + (c - b) * w;
        return shrink_or_normal(simplex, dir, closest, normal, [c, b]);
    }
    // Interior: keep the triangle. A degenerate triangle (duplicated or
    // collinear vertices) has zero determinant sum: treat it like the
    // origin lying on the simplex and search along the face normal instead
    // of producing NaN (`sum <= 0.0 || sum.is_nan()` also catches NaN; well-conditioned
    // triangles have sum = |ab x ac|^2 > 0 and never enter this branch).
    let sum = va + vb + vc;
    if sum <= 0.0 || sum.is_nan() {
        *dir = normal;
        return false;
    }
    let denom = 1.0 / sum;
    let v = vb * denom;
    let w = vc * denom;
    let closest = a + ab * v + ac * w;
    // Same relative origin test as `shrink_or_normal`: an interior
    // projection that cancels down to the origin (within the simplex's own
    // fp resolution) must escape along the face normal, not shrink to a
    // noise-direction edge.
    let span2 = a
        .length_squared()
        .max(b.length_squared())
        .max(c.length_squared());
    if closest.length_squared() < span2 * 1e-12 {
        *dir = normal;
    } else {
        *dir = -closest;
    }
    false
}

/// Shrink the simplex to `new_p`, or (if the closest point is the origin
/// itself) keep the triangle and search along `normal` for a fourth point.
/// The origin test is relative to the simplex span: an interior projection
/// computed on large coordinates carries fp cancellation noise far above an
/// absolute epsilon, and missing the escape here strands the strict walk on
/// a planar slice of the Minkowski difference (its supports then never
/// leave that plane and the walk exhausts). `span2` is the largest squared
/// vertex length; the `1e-12 * span2` gate sits at the squared-noise floor
/// (`~eps^2 * span2`) and only fires when the projection is the origin to
/// within the walk's own resolution.
fn shrink_or_normal(
    simplex: &mut Simplex,
    dir: &mut Vector3,
    closest: Vector3,
    normal: Vector3,
    new_pts: [Vector3; 2],
) -> bool {
    let span2 = simplex
        .points()
        .iter()
        .map(|p| p.length_squared())
        .fold(0.0, Real::max);
    if closest.length_squared() < span2 * 1e-12 {
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
    // Coincident endpoints: the segment is a point (a NaN `t` from the 0/0
    // below would slip past both range tests and poison the interpolation).
    // Exact zero only; near-zero denominators keep the standard arithmetic
    // (the `t` range test then lands on an endpoint, which is correct).
    if ab.length_squared() == 0.0 {
        return (a, vec![a]);
    }
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
        // Zero denominator here means a == b (both d's are zero): the edge
        // is a point.
        if d1 == d3 {
            return (a, vec![a]);
        }
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
        // Zero denominator means a == c (same argument as above).
        if d2 == d6 {
            return (a, vec![a]);
        }
        let w = d2 / (d2 - d6);
        let q = a + ac * w;
        return (q, vec![c, a]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        // Zero denominator means b == c.
        if (d4 - d3) + (d5 - d6) == 0.0 {
            return (b, vec![b]);
        }
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q = b + (c - b) * w;
        return (q, vec![c, b]);
    }
    // Interior: the closest point is the origin's projection onto the face.
    // A degenerate triangle (duplicated or collinear vertices) has zero
    // determinant sum: no interior exists, so fall back to the closest of
    // the three edges instead of dividing by zero (`sum <= 0.0 || sum.is_nan()` also
    // catches NaN; well-conditioned triangles have sum = |ab x ac|^2 > 0
    // and never enter this branch).
    let sum = va + vb + vc;
    if sum <= 0.0 || sum.is_nan() {
        let (ab_q, ab_keep) = closest_origin_segment(a, b);
        let (bc_q, bc_keep) = closest_origin_segment(b, c);
        let (ca_q, ca_keep) = closest_origin_segment(c, a);
        if ab_q.length_squared() <= bc_q.length_squared()
            && ab_q.length_squared() <= ca_q.length_squared()
        {
            return (ab_q, ab_keep);
        }
        if bc_q.length_squared() <= ca_q.length_squared() {
            return (bc_q, bc_keep);
        }
        return (ca_q, ca_keep);
    }
    let denom = 1.0 / sum;
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

    // -----------------------------------------------------------------
    // Numerical robustness suite (independent-audit hardening round).
    // The helper-level tests pin the degenerate closest-point guards
    // directly; the GJK-level tests cover scale sweeps, contact-boundary
    // routing, and seed stability. Every test asserts: no panic, finite
    // results, bounded termination (all walks are iteration-capped by
    // construction), and a sensible classification.
    // -----------------------------------------------------------------

    fn assert_finite(v: Vector3, what: &str) {
        assert!(
            v.x.is_finite() && v.y.is_finite() && v.z.is_finite(),
            "{what} is not finite: {v:?}"
        );
    }

    fn contact_seeded(
        a: &CollisionShape,
        pa: Vector3,
        b: &CollisionShape,
        pb: Vector3,
        seed: Option<Vector3>,
        threshold: Real,
    ) -> GjkOutcome {
        gjk_contact(
            a,
            pa,
            Quaternion::IDENTITY,
            b,
            pb,
            Quaternion::IDENTITY,
            seed,
            threshold,
        )
    }

    /// `closest_origin_segment`: exact-zero length (the guard), lengths whose
    /// square underflows to zero in f32, and near-identical endpoints all
    /// return a finite endpoint instead of a NaN `t` interpolation.
    #[test]
    fn closest_segment_degenerate_lengths_return_endpoints() {
        // Exact zero: the `length_squared() == 0.0` guard.
        let a = Vector3::new(1.0, 2.0, 3.0);
        let (q, keep) = closest_origin_segment(a, a);
        assert_finite(q, "exact zero segment");
        assert_eq!(keep.len(), 1);
        assert!(keep[0].distance(a) < 1e-9);

        // Squared length underflows to 0.0 in f32 (1e-60), stays nonzero in
        // f64: both paths must land on an endpoint (f32 via the guard, f64
        // via `t` clamping to +-1e30).
        let tiny: Real = 1e-30;
        let (q, _) = closest_origin_segment(Vector3::X, Vector3::new(1.0 + tiny, 0.0, 0.0));
        assert_finite(q, "underflow segment (origin behind a)");
        assert!(q.distance(Vector3::X) < 1e-9, "q {q:?}");
        let (q, _) = closest_origin_segment(Vector3::NEG_X, Vector3::new(-1.0 + tiny, 0.0, 0.0));
        assert_finite(q, "underflow segment (origin beyond b)");
        assert!(q.distance(Vector3::new(-1.0 + tiny, 0.0, 0.0)) < 1e-9, "q {q:?}");

        // Near-identical endpoints (non-underflowing squared length in f64;
        // f32 rounds the offset away into the exact-zero case).
        let small: Real = 1e-20;
        let (q, _) = closest_origin_segment(Vector3::X, Vector3::new(1.0 + small, 0.0, 0.0));
        assert_finite(q, "near-zero segment");
        assert!(q.distance(Vector3::X) < 1e-9, "q {q:?}");
    }

    /// `closest_origin_segment` well-conditioned clamps and interior
    /// projection keep their exact standard-path results.
    #[test]
    fn closest_segment_well_conditioned_preserved() {
        // t <= 0 clamps to a.
        let (q, keep) = closest_origin_segment(Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 0.0, 0.0));
        assert!(q.distance(Vector3::new(2.0, 0.0, 0.0)) < 1e-9);
        assert_eq!(keep.len(), 1);
        // t >= 1 clamps to b.
        let (q, _) = closest_origin_segment(Vector3::new(-3.0, 0.0, 0.0), Vector3::new(-2.0, 0.0, 0.0));
        assert!(q.distance(Vector3::new(-2.0, 0.0, 0.0)) < 1e-9);
        // Interior projection: q = a + ab * 0.2, both endpoints kept.
        let (q, keep) = closest_origin_segment(Vector3::new(1.0, 1.0, 0.0), Vector3::new(2.0, -1.0, 0.0));
        let expected = Vector3::new(1.2, 0.6, 0.0);
        assert!(q.distance(expected) < 1e-6, "q {q:?}");
        assert_eq!(keep.len(), 2);
    }

    /// `closest_origin_triangle` with duplicated vertices: finite, no panic,
    /// and the region tests resolve to either the shared vertex (a == b) or
    /// the correct closest point on the remaining segment.
    #[test]
    fn closest_triangle_duplicate_vertices_stay_finite() {
        // a == b: the edge-ab guard returns the shared vertex.
        let shared = Vector3::new(0.5, 0.3, 0.0);
        let (q, keep) = closest_origin_triangle(shared, shared, Vector3::new(-2.0, 0.3, 0.0));
        assert_finite(q, "a == b");
        assert_eq!(keep.len(), 1);
        assert!(q.distance(shared) < 1e-9);

        // a == c: the remaining segment a-b straddles the origin; the edge-ab
        // region resolves the exact midpoint.
        let (q, keep) = closest_origin_triangle(Vector3::new(0.5, 0.0, 0.0), Vector3::new(-0.5, 0.0, 0.0), Vector3::new(0.5, 0.0, 0.0));
        assert_finite(q, "a == c");
        assert!(q.length() < 1e-7, "q {q:?}");
        assert_eq!(keep.len(), 2);

        // b == c: closest point on the remaining segment a-b.
        let (q, _) = closest_origin_triangle(Vector3::new(0.0, 0.5, 0.0), Vector3::new(0.3, -0.5, 0.0), Vector3::new(0.3, -0.5, 0.0));
        let expected = Vector3::new(0.1376, 0.0413, 0.0);
        assert_finite(q, "b == c");
        assert!(q.distance(expected) < 1e-3, "q {q:?}");

        // All three equal: a pure point.
        let p = Vector3::new(0.7, -0.2, 1.0);
        let (q, keep) = closest_origin_triangle(p, p, p);
        assert_finite(q, "a == b == c");
        assert_eq!(keep.len(), 1);
        assert!(q.distance(p) < 1e-9);
    }

    /// Collinear and sliver triangles: no NaN, and the closest point matches
    /// the analytic answer. The sliver case converges to the same point via
    /// the edge region in f32 and the (numerically tiny) interior region in
    /// f64 — the assertion deliberately does not depend on which branch runs.
    #[test]
    fn closest_triangle_collinear_and_sliver_stay_finite() {
        // Exactly collinear, projection inside the middle edge.
        let (q, _) = closest_origin_triangle(Vector3::new(3.0, 1.0, 0.0), Vector3::new(0.5, 1.0, 0.0), Vector3::new(-3.0, 1.0, 0.0));
        assert_finite(q, "collinear interior");
        assert!(q.distance(Vector3::new(0.0, 1.0, 0.0)) < 1e-7, "q {q:?}");

        // Exactly collinear, projection beyond the last vertex.
        let (q, _) = closest_origin_triangle(Vector3::new(1.0, 1.0, 0.0), Vector3::new(2.0, 1.0, 0.0), Vector3::new(3.0, 1.0, 0.0));
        assert_finite(q, "collinear beyond");
        assert!(q.distance(Vector3::new(1.0, 1.0, 0.0)) < 1e-7, "q {q:?}");

        // Near-collinear (1e-7 height perturbation): answer stays the line's
        // closest point.
        let (q, _) = closest_origin_triangle(Vector3::new(3.0, 1.0, 0.0), Vector3::new(0.5, 1.0, 1e-7), Vector3::new(-3.0, 1.0, 0.0));
        assert_finite(q, "near collinear");
        assert!(q.distance(Vector3::new(0.0, 1.0, 0.0)) < 1e-5, "q {q:?}");

        // Sliver of height 1e-6 with the origin inside it: the exact closest
        // point is the origin. f32 resolves the edge region (q lands on the
        // base edge, error bounded by the sliver thickness); f64 resolves the
        // interior region exactly — both stay on the sliver.
        let (q, _) = closest_origin_triangle(Vector3::new(-1.0, 1e-6, 0.0), Vector3::new(1.0, 1e-6, 0.0), Vector3::new(0.0, -1e-6, 0.0));
        assert_finite(q, "sliver interior");
        assert!(q.length() < 2e-6, "q {q:?}");
    }

    /// Well-conditioned triangles keep their exact standard-path results
    /// (interior face, vertex region, edge region) and keep the right
    /// number of simplex vertices.
    #[test]
    fn closest_triangle_well_conditioned_preserved() {
        // Interior projection: q is the origin, the face is kept.
        let (q, keep) = closest_origin_triangle(Vector3::new(2.0, 2.0, 0.0), Vector3::new(-2.0, 2.0, 0.0), Vector3::new(0.0, -2.0, 0.0));
        assert!(q.length() < 1e-7, "q {q:?}");
        assert_eq!(keep.len(), 3);

        // Vertex region.
        let a = Vector3::new(1.0, 0.0, 0.0);
        let (q, keep) = closest_origin_triangle(a, Vector3::new(1.0, 2.0, 0.0), Vector3::new(3.0, 1.0, 0.0));
        assert!(q.distance(a) < 1e-7);
        assert_eq!(keep.len(), 1);

        // Edge region: projection onto edge ab at (1, 0).
        let (q, keep) = closest_origin_triangle(Vector3::new(1.0, -1.0, 0.0), Vector3::new(1.0, 1.0, 0.0), Vector3::new(3.0, 0.0, 0.0));
        assert!(q.distance(Vector3::new(1.0, 0.0, 0.0)) < 1e-7, "q {q:?}");
        assert_eq!(keep.len(), 2);
    }

    /// `closest_origin_tetrahedron`: coplanar input falls back to the best
    /// face (finite), an enclosing tetrahedron reports zero with all four
    /// vertices kept, and an outside-origin shrinks to the offending face.
    #[test]
    fn closest_tetrahedron_degenerate_and_analytic_cases() {
        // Coplanar (all z = 1): best face is the unit triangle containing
        // the projection (0, 0).
        let a = Vector3::new(1.0, 0.0, 1.0);
        let b = Vector3::new(-1.0, 0.0, 1.0);
        let c = Vector3::new(0.0, 1.0, 1.0);
        let d = Vector3::new(0.0, -1.0, 1.0);
        let (q, keep) = closest_origin_tetrahedron(a, b, c, d);
        assert_finite(q, "coplanar");
        assert!(q.distance(Vector3::new(0.0, 0.0, 1.0)) < 1e-7, "q {q:?}");
        assert!(!keep.is_empty());

        // Regular tetrahedron enclosing the origin.
        let (q, keep) = closest_origin_tetrahedron(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, -1.0, 1.0),
        );
        assert!(q.length_squared() == 0.0, "q {q:?}");
        assert_eq!(keep.len(), 4);

        // Outside the abc face (d lifted above the plane). The projection
        // of the origin onto the face lands exactly on edge ab, so the
        // closest feature is that edge.
        let (q, keep) = closest_origin_tetrahedron(a, b, c, Vector3::new(0.0, -1.0, 2.0));
        assert_finite(q, "outside face");
        assert!(q.distance(Vector3::new(0.0, 0.0, 1.0)) < 1e-7, "q {q:?}");
        assert_eq!(keep.len(), 2);
    }

    /// The contact-walk `reduce_*` guards on duplicated / collinear /
    /// coplanar support buffers: finite closest points, shrunk simplex
    /// lengths, and exact `Inside` for a real tetrahedron.
    #[test]
    fn reduce_degenerate_support_buffers_stay_finite() {
        let sv = |w: Vector3, p_b: Vector3| SupportVertex { w, p_b };

        // Duplicated triangle vertices (a == b): shrinks to the shared
        // vertex with its witness.
        let mut buf = [
            sv(Vector3::new(-0.5, 0.0, 0.0), Vector3::new(-0.2, 0.0, 0.0)),
            sv(Vector3::new(0.5, 0.0, 0.0), Vector3::new(0.2, 0.0, 0.0)),
            sv(Vector3::new(0.5, 0.0, 0.0), Vector3::new(0.2, 0.0, 0.0)),
            sv(Vector3::ZERO, Vector3::ZERO),
        ];
        let mut len = 3usize;
        match reduce(&mut buf, &mut len) {
            ReduceStep::Closest { v, p_b } => {
                assert_finite(v, "reduce duplicate");
                assert_eq!(len, 1);
                assert!(v.distance(Vector3::new(0.5, 0.0, 0.0)) < 1e-7);
                assert!(p_b.distance(Vector3::new(0.2, 0.0, 0.0)) < 1e-7);
            }
            ReduceStep::Inside => panic!("duplicated triangle reported Inside"),
        }

        // Collinear triangle: shrinks to the closest vertex.
        let mut buf = [
            sv(Vector3::new(-2.0, 1.0, 0.0), Vector3::ZERO),
            sv(Vector3::new(0.0, 1.0, 0.0), Vector3::ONE),
            sv(Vector3::new(2.0, 1.0, 0.0), Vector3::ZERO),
            sv(Vector3::ZERO, Vector3::ZERO),
        ];
        let mut len = 3usize;
        match reduce(&mut buf, &mut len) {
            ReduceStep::Closest { v, p_b } => {
                assert_finite(v, "reduce collinear");
                assert_eq!(len, 1);
                assert!(v.distance(Vector3::new(0.0, 1.0, 0.0)) < 1e-7);
                assert!(p_b.distance(Vector3::ONE) < 1e-7);
            }
            ReduceStep::Inside => panic!("collinear triangle reported Inside"),
        }

        // Coplanar tetrahedron: falls back to the best face, finite result.
        let mut buf = [
            sv(Vector3::new(1.0, 0.0, 1.0), Vector3::ZERO),
            sv(Vector3::new(-1.0, 0.0, 1.0), Vector3::ZERO),
            sv(Vector3::new(0.0, 1.0, 1.0), Vector3::ZERO),
            sv(Vector3::new(0.0, -1.0, 1.0), Vector3::ZERO),
        ];
        let mut len = 4usize;
        match reduce(&mut buf, &mut len) {
            ReduceStep::Closest { v, p_b } => {
                assert_finite(v, "reduce coplanar");
                assert!(len <= 3);
                assert!(v.distance(Vector3::new(0.0, 0.0, 1.0)) < 1e-7, "v {v:?}");
                assert_finite(p_b, "reduce coplanar witness");
            }
            ReduceStep::Inside => panic!("coplanar tetrahedron reported Inside"),
        }

        // A real enclosing tetrahedron reports Inside.
        let mut buf = [
            sv(Vector3::new(1.0, 1.0, 1.0), Vector3::ZERO),
            sv(Vector3::new(1.0, -1.0, -1.0), Vector3::ZERO),
            sv(Vector3::new(-1.0, 1.0, -1.0), Vector3::ZERO),
            sv(Vector3::new(-1.0, -1.0, 1.0), Vector3::ZERO),
        ];
        let mut len = 4usize;
        assert!(matches!(reduce(&mut buf, &mut len), ReduceStep::Inside));
    }

    /// Coincident full shapes (a volume Minkowski difference): the strict
    /// enclosing walk proves overlap with a 4-point simplex.
    #[test]
    fn gjk_intersect_coincident_shapes_report_overlap() {
        let sphere = CollisionShape::sphere(0.5);
        let simplex = gjk_intersect(
            &sphere,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &sphere,
            Vector3::ZERO,
            Quaternion::IDENTITY,
        )
        .expect("coincident spheres overlap");
        assert_eq!(simplex.points().len(), 4);
        for p in simplex.points() {
            assert_finite(*p, "coincident sphere simplex");
        }

        let b1 = CollisionShape::box_shape(Vector3::splat(0.5));
        let simplex = gjk_intersect(
            &b1,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &b1,
            Vector3::ZERO,
            Quaternion::IDENTITY,
        )
        .expect("coincident boxes overlap");
        assert_eq!(simplex.points().len(), 4);
        for p in simplex.points() {
            assert_finite(*p, "coincident box simplex");
        }
    }

    /// Coincident *point cores* (sphere pairs) degenerate the contact
    /// walk's Minkowski difference to the origin itself; the walk stays
    /// bounded and returns without NaN. This is a documented degeneracy:
    /// the enclosing fallback cannot prove enclosure of a single point, so
    /// the walk reports `Separated` — unreachable from the world pipeline,
    /// which routes sphere-sphere pairs through the analytic algorithm.
    #[test]
    fn gjk_contact_coincident_point_cores_terminate_finitely() {
        let sphere = CollisionShape::sphere(0.5);
        match contact(&sphere, Vector3::ZERO, &sphere, Vector3::ZERO, 0.0173) {
            GjkOutcome::Band { distance, normal, .. } => {
                assert!(distance.is_finite());
                assert_finite(normal, "coincident sphere band");
            }
            GjkOutcome::Separated | GjkOutcome::Overlap(_) => {}
        }

        let capsule = CollisionShape::capsule(0.5, 0.5);
        match contact(&capsule, Vector3::ZERO, &capsule, Vector3::ZERO, 0.0173) {
            GjkOutcome::Band { distance, normal, .. } => {
                assert!(distance.is_finite());
                assert_finite(normal, "coincident capsule band");
            }
            GjkOutcome::Separated | GjkOutcome::Overlap(_) => {}
        }
    }

    /// Controlled contact-boundary cases on exact point-core arithmetic
    /// (spheres): exact touching, a +1e-5 gap, a -1e-5 overlap, and a
    /// clearly separated pair all route to the designed outcome.
    #[test]
    fn gjk_contact_touch_gap_overlap_boundary() {
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::sphere(0.5);
        let threshold: Real = 0.0173;

        // Exact touch: Band with distance 0.
        match contact(&a, Vector3::ZERO, &b, Vector3::new(1.0, 0.0, 0.0), threshold) {
            GjkOutcome::Band { distance, normal, point_on_b } => {
                assert!(distance.abs() < 1e-6, "distance {distance}");
                assert!((normal - Vector3::X).length() < 1e-6);
                assert!((point_on_b - Vector3::new(0.5, 0.0, 0.0)).length() < 1e-5);
            }
            other => panic!("expected Band, got {other:?}"),
        }
        // Tiny gap inside the band.
        match contact(&a, Vector3::ZERO, &b, Vector3::new(1.0 + 1e-5, 0.0, 0.0), threshold) {
            GjkOutcome::Band { distance, .. } => {
                assert!((distance - 1e-5).abs() < 1e-6, "distance {distance}");
            }
            other => panic!("expected Band, got {other:?}"),
        }
        // Tiny overlap inside the margin shell.
        match contact(&a, Vector3::ZERO, &b, Vector3::new(1.0 - 1e-5, 0.0, 0.0), threshold) {
            GjkOutcome::Band { distance, .. } => {
                assert!((distance + 1e-5).abs() < 1e-6, "distance {distance}");
            }
            other => panic!("expected Band, got {other:?}"),
        }
        // Beyond the band: Separated.
        assert!(matches!(
            contact(&a, Vector3::ZERO, &b, Vector3::new(1.05, 0.0, 0.0), threshold),
            GjkOutcome::Separated
        ));
    }

    /// Routing exactly around the walk's band threshold: just below reports
    /// a Band contact with the true distance, just above reports Separated.
    #[test]
    fn gjk_contact_band_threshold_routing() {
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::sphere(0.5);
        let threshold: Real = 0.02;

        match contact(&a, Vector3::ZERO, &b, Vector3::new(1.0199, 0.0, 0.0), threshold) {
            GjkOutcome::Band { distance, .. } => {
                assert!((distance - 0.0199).abs() < 1e-5, "distance {distance}");
            }
            other => panic!("expected Band, got {other:?}"),
        }
        assert!(matches!(
            contact(&a, Vector3::ZERO, &b, Vector3::new(1.0201, 0.0, 0.0), threshold),
            GjkOutcome::Separated
        ));
    }

    /// Sphere-pair scale sweep (the margin scales with the radius, so the
    /// walk is analytically identical across scales): classification and
    /// distance must track the scaled analytic values at 1e-3..1e3.
    #[test]
    fn gjk_scale_sweep_sphere_pairs() {
        let scales: [Real; 5] = [1e-3, 1e-2, 1.0, 1e2, 1e3];
        // (center separation in radii sums, expected outcome, expected distance)
        let configs: [(Real, bool, Real); 4] = [
            (3.0, false, 2.0),  // separated
            (1.0, true, 0.0),   // exact touch
            (0.9, true, -0.1),  // shallow overlap
            (0.5, true, -0.5),  // deep overlap (still inside the margin shell)
        ];
        for scale in scales {
            // The sphere's radius is its margin, so scaled shapes keep the
            // walk analytically identical: classification and the scaled
            // distance must track the unit-scale values at every scale.
            let a = CollisionShape::sphere(0.5 * scale);
            let b = CollisionShape::sphere(0.5 * scale);
            for (centers, expect_band, expected_distance) in configs {
                let threshold = 0.02 * scale;
                match contact(&a, Vector3::ZERO, &b, Vector3::new(centers * scale, 0.0, 0.0), threshold) {
                    GjkOutcome::Band { distance, normal, .. } => {
                        assert!(expect_band, "scale {scale} centers {centers}: Band but expected Separated");
                        assert!(distance.is_finite(), "distance {distance}");
                        let rel = (distance / scale - expected_distance).abs();
                        assert!(rel < 1e-4, "scale {scale} centers {centers}: distance {distance}");
                        assert_finite(normal, "sweep normal");
                        assert!((normal.length() - 1.0).abs() < 1e-5, "normal {normal:?}");
                        assert!(normal.x.abs() > 0.999, "normal {normal:?}");
                    }
                    GjkOutcome::Separated => {
                        assert!(!expect_band, "scale {scale} centers {centers}: Separated but expected Band");
                    }
                    GjkOutcome::Overlap(_) => {
                        panic!("scale {scale} centers {centers}: sphere overlap must stay in the margin shell");
                    }
                }
            }
        }
    }

    /// Box-pair scale sweep against the margin model: cores are the boxes
    /// shrunk by the fixed 0.04 margin (clamped per axis), so the reported
    /// distance is `centers - 2*core_half - margin_sum` exactly. At small
    /// scales the fixed margin dominates (Bullet implicit-shape semantics;
    /// Bullet clamps it at construction via `setSafeMargin`, a documented
    /// deviation) — tiny boxes overlap into deep-negative Band distances.
    #[test]
    fn gjk_scale_sweep_box_pairs() {
        let margin: Real = 0.04;
        let scales: [Real; 5] = [1e-3, 1e-2, 1.0, 1e2, 1e3];
        for scale in scales {
            let half = Vector3::splat(0.5 * scale);
            let a = CollisionShape::box_shape(half);
            let b = CollisionShape::box_shape(half);
            let core_half = (0.5 * scale - margin).max(0.0);
            let threshold = 0.02 * scale;

            // Face-to-face configs: the walk's distance is exact against the
            // margin model, and the Band/Separated split derives from it
            // (Band iff distance < threshold).
            let configs = [
                (3.0 * scale, 3.0 * scale - 2.0 * core_half - 2.0 * margin),
                (1.0 * scale, 1.0 * scale - 2.0 * core_half - 2.0 * margin),
            ];
            for (centers, expected_distance) in configs {
                let expect_band = expected_distance < threshold;
                match contact(&a, Vector3::ZERO, &b, Vector3::new(centers, 0.0, 0.0), threshold) {
                    GjkOutcome::Band { distance, normal, .. } => {
                        assert!(expect_band, "scale {scale}: Band but expected otherwise");
                        assert!(distance.is_finite(), "distance {distance}");
                        let tol = (1e-4 * scale).max(1e-6);
                        assert!(
                            (distance - expected_distance).abs() < tol,
                            "scale {scale}: distance {distance} vs {expected_distance}"
                        );
                        assert_finite(normal, "box sweep normal");
                    }
                    GjkOutcome::Separated => {
                        assert!(!expect_band, "scale {scale}: Separated but expected Band");
                    }
                    GjkOutcome::Overlap(_) => {
                        panic!("scale {scale}: face-axis configs never overlap cores");
                    }
                }
            }

            // Deep overlap: cores intersect at scale >= 0.16, routing to
            // EPA; below that the clamped point cores stay in the Band.
            let deep = contact(&a, Vector3::ZERO, &b, Vector3::new(0.3 * scale, 0.0, 0.0), threshold);
            if core_half > 0.0 {
                assert!(
                    matches!(deep, GjkOutcome::Overlap(ref s) if s.points().len() == 4),
                    "scale {scale}: expected Overlap, got {deep:?}"
                );
            } else {
                assert!(
                    matches!(deep, GjkOutcome::Band { .. }),
                    "scale {scale}: expected Band, got {deep:?}"
                );
            }
        }
    }

    /// Duplicate-support gate across scales (flat cube hull vs a point-core
    /// sphere): the 1e-6 absolute gate terminates the walk correctly at
    /// 1e-3..1e2 because support spacing stays far above it, and the point
    /// core makes the duplicate exit *exactly* right even at small scales.
    #[test]
    fn gjk_duplicate_support_gate_across_scales() {
        let scales: [Real; 4] = [1e-3, 1e-2, 1.0, 1e2];
        for scale in scales {
            let e = 0.5 * scale;
            let hull = CollisionShape::convex_hull(vec![
                Vector3::new(-e, -e, -e),
                Vector3::new(e, -e, -e),
                Vector3::new(-e, e, -e),
                Vector3::new(e, e, -e),
                Vector3::new(-e, -e, e),
                Vector3::new(e, -e, e),
                Vector3::new(-e, e, e),
                Vector3::new(e, e, e),
            ]);
            let sphere = CollisionShape::sphere(0.5 * scale);
            let threshold = 0.02 * scale;
            // Face gap 0.01 * scale: core distance 0.51 * scale, distance
            // 0.01 * scale after adding the sphere's margin back.
            match contact(&hull, Vector3::ZERO, &sphere, Vector3::new(1.01 * scale, 0.0, 0.0), threshold) {
                GjkOutcome::Band { distance, normal, .. } => {
                    let tol = (1e-4 * scale).max(1e-9);
                    assert!(
                        (distance - 0.01 * scale).abs() < tol,
                        "scale {scale}: distance {distance}"
                    );
                    assert_finite(normal, "hull sweep normal");
                    assert!(normal.x.abs() > 0.999, "normal {normal:?}");
                }
                other => panic!("scale {scale}: expected Band, got {other:?}"),
            }
        }
    }

    /// Seeded vs unseeded: classification must never diverge and key
    /// geometric results must agree. Point-core walks are seed-independent
    /// bit-for-bit (the support is a constant); box-core walks may take
    /// different iteration paths but converge to the same fixed point.
    #[test]
    fn gjk_seeded_vs_unseeded_stability() {
        let seeds = [
            None,
            Some(Vector3::X),
            Some(Vector3::NEG_X),
            Some(Vector3::Y),
            Some(Vector3::new(1.0, 7.0, 3.0).normalize()),
        ];

        // Sphere band: exact equality across every seed.
        let a = CollisionShape::sphere(0.5);
        let b = CollisionShape::sphere(0.5);
        let mut reference_distance = None;
        for seed in seeds {
            match contact_seeded(&a, Vector3::ZERO, &b, Vector3::new(1.01, 0.0, 0.0), seed, 0.0173) {
                GjkOutcome::Band { distance, normal, .. } => {
                    assert!((normal.x - 1.0).abs() < 1e-6, "normal {normal:?}");
                    match reference_distance {
                        None => reference_distance = Some(distance),
                        Some(prev) => assert!(
                            (distance - prev).abs() < 1e-9,
                            "seed {seed:?}: distance {distance} vs {prev}"
                        ),
                    }
                }
                other => panic!("sphere band seed {seed:?}: expected Band, got {other:?}"),
            }
        }

        // Box band: same classification, distances within the walk's
        // relative convergence tolerance.
        let box_shape = CollisionShape::box_shape(Vector3::splat(0.5));
        let mut reference_distance = None;
        for seed in seeds {
            match contact_seeded(&box_shape, Vector3::ZERO, &box_shape, Vector3::new(1.02, 0.0, 0.0), seed, 0.05) {
                GjkOutcome::Band { distance, normal, .. } => {
                    assert!(normal.x.abs() > 0.999, "normal {normal:?}");
                    match reference_distance {
                        None => reference_distance = Some(distance),
                        Some(prev) => assert!(
                            (distance - prev).abs() < 1e-5,
                            "seed {seed:?}: distance {distance} vs {prev}"
                        ),
                    }
                }
                other => panic!("box band seed {seed:?}: expected Band, got {other:?}"),
            }
        }

        // Box deep overlap and separated pair: classification only.
        for seed in seeds {
            assert!(
                matches!(
                    contact_seeded(&box_shape, Vector3::ZERO, &box_shape, Vector3::new(0.6, 0.0, 0.0), seed, 0.02),
                    GjkOutcome::Overlap(ref s) if s.points().len() == 4
                ),
                "seed {seed:?}: expected Overlap"
            );
            assert!(
                matches!(
                    contact_seeded(&box_shape, Vector3::ZERO, &box_shape, Vector3::new(3.0, 0.0, 0.0), seed, 0.02),
                    GjkOutcome::Separated
                ),
                "seed {seed:?}: expected Separated"
            );
        }
    }

    /// Degenerate hull pairs: zero-volume Minkowski differences (crossing,
    /// parallel, and collinear segments) terminate finitely with a sensible
    /// classification. Parallel separated segments stay Separated; crossing
    /// segments report the contact; collinear overlapping segments are a
    /// documented 1D degeneracy (the enclosing walk cannot prove enclosure
    /// of a zero-volume set, and the engine exposes no 1D shapes).
    #[test]
    fn gjk_degenerate_segment_hulls_terminate_finitely() {
        let seg_x = CollisionShape::convex_hull(vec![Vector3::new(-1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)]);
        let seg_y = CollisionShape::convex_hull(vec![Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 1.0, 0.0)]);

        // Crossing segments: overlap, classified with a finite outcome.
        assert!(gjk_intersect(
            &seg_x,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &seg_y,
            Vector3::ZERO,
            Quaternion::IDENTITY
        )
        .is_some());
        match contact(&seg_x, Vector3::ZERO, &seg_y, Vector3::ZERO, 0.0173) {
            GjkOutcome::Overlap(simplex) => {
                assert_eq!(simplex.points().len(), 4);
                for p in simplex.points() {
                    assert_finite(*p, "crossing segment simplex");
                }
            }
            GjkOutcome::Band { distance, normal, .. } => {
                assert!(distance.abs() < 1e-3, "distance {distance}");
                assert_finite(normal, "crossing segment band");
            }
            GjkOutcome::Separated => panic!("crossing segments cannot be separated"),
        }

        // Parallel separated segments (offset 2 on Y): no contact.
        assert!(gjk_intersect(
            &seg_x,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &seg_x,
            Vector3::new(0.0, 2.0, 0.0),
            Quaternion::IDENTITY
        )
        .is_none());
        assert!(matches!(
            contact(&seg_x, Vector3::ZERO, &seg_x, Vector3::new(0.0, 2.0, 0.0), 0.0173),
            GjkOutcome::Separated
        ));

        // Collinear overlapping segments: bounded, finite, no panic.
        let _ = gjk_intersect(
            &seg_x,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &seg_x,
            Vector3::new(0.5, 0.0, 0.0),
            Quaternion::IDENTITY,
        );
        match contact(&seg_x, Vector3::ZERO, &seg_x, Vector3::new(0.5, 0.0, 0.0), 0.0173) {
            GjkOutcome::Band { distance, normal, .. } => {
                assert!(distance.is_finite());
                assert_finite(normal, "collinear segment band");
            }
            GjkOutcome::Separated | GjkOutcome::Overlap(_) => {}
        }
    }

    /// Elongated convex shapes (2mm x 2m rod hull vs a unit box): the
    /// near-degenerate support set must keep producing finite, correctly
    /// routed results at touch, band, and separation.
    #[test]
    fn gjk_elongated_rod_vs_box_stays_finite() {
        let e: Real = 1e-3;
        let rod = CollisionShape::convex_hull(vec![
            Vector3::new(-e, -1.0, -e),
            Vector3::new(e, -1.0, -e),
            Vector3::new(-e, 1.0, -e),
            Vector3::new(e, 1.0, -e),
            Vector3::new(-e, -1.0, e),
            Vector3::new(e, -1.0, e),
            Vector3::new(-e, 1.0, e),
            Vector3::new(e, 1.0, e),
        ]);
        let b = CollisionShape::box_shape(Vector3::splat(0.5));
        let threshold: Real = 0.0173;

        // Rod tip (y = 1.0) 0.01 outside the box surface: center 1.51 puts
        // the box bottom at 1.01 (gap 0.01) and the shrunk-core bottom at
        // 1.05, so the core gap is 0.05 and the Band distance is 0.01 after
        // the margin is added back.
        match contact(&rod, Vector3::ZERO, &b, Vector3::new(0.0, 1.51, 0.0), threshold) {
            GjkOutcome::Band { distance, normal, .. } => {
                assert!((distance - 0.01).abs() < 1e-4, "distance {distance}");
                assert_finite(normal, "rod band");
                assert!(normal.y.abs() > 0.999, "normal {normal:?}");
            }
            other => panic!("expected Band, got {other:?}"),
        }

        // Exact touch at the box surface (bottom exactly at the rod tip).
        match contact(&rod, Vector3::ZERO, &b, Vector3::new(0.0, 1.50, 0.0), threshold) {
            GjkOutcome::Band { distance, .. } => {
                assert!(distance.abs() < 1e-5, "distance {distance}");
            }
            other => panic!("expected Band, got {other:?}"),
        }

        // Box bottom 0.05 below the rod tip (0.01 into the shrunk core,
        // whose bottom sits at 0.99): deep enough to route to EPA.
        assert!(
            matches!(
                contact(&rod, Vector3::ZERO, &b, Vector3::new(0.0, 1.45, 0.0), threshold),
                GjkOutcome::Overlap(ref s) if s.points().len() == 4
            ),
            "expected Overlap for the core-penetrating rod"
        );

        // Well separated along the rod axis.
        assert!(matches!(
            contact(&rod, Vector3::ZERO, &b, Vector3::new(0.0, 2.5, 0.0), threshold),
            GjkOutcome::Separated
        ));
    }

    // -----------------------------------------------------------------
    // Strict-walk degeneracy suite: the enclosing walk must never hand a
    // zero or non-finite search direction to the support query, however the
    // simplex reductions collapse (duplicate vertices, vertices on the
    // origin, collinear triangles). The invariant is enforced once, at the
    // top of the `enclosing_walk` loop; `perpendicular` is the always-safe
    // primitive it leans on.
    // -----------------------------------------------------------------

    /// `perpendicular` always returns a finite unit-length vector, and a
    /// true perpendicular for finite non-zero input. `(0.5, 0, 0)` is the
    /// regression case: the old helper test normalized a zero cross product
    /// into NaN for short axis-aligned inputs.
    #[test]
    fn perpendicular_always_finite_and_orthogonal() {
        for v in [
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::X,
            Vector3::NEG_X,
            Vector3::Y,
            Vector3::NEG_Y,
            Vector3::Z,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(2.0, 1.0, 1.0),
            Vector3::new(1e-4, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            Vector3::new(-0.2, -0.2, -0.2),
            Vector3::ZERO,
        ] {
            let p = perpendicular(v);
            assert_finite(p, "perpendicular result");
            assert!(
                p.length_squared() > 0.5 && p.length_squared() < 1.5,
                "not unit: {p:?}"
            );
            if v.length_squared() > 0.0 {
                assert!(p.dot(v).abs() < 1e-5, "not orthogonal: {p:?} vs {v:?}");
            }
        }
        // Non-finite input must not propagate NaN.
        let p = perpendicular(Vector3::new(Real::NAN, 0.0, 0.0));
        assert_finite(p, "NaN input");
    }

    /// Reduction-level contract: `solve` returns a *finite* direction even
    /// for duplicate segments and collinear triangles. Zero is tolerated at
    /// this layer (the walk's loop head repairs it); NaN never is.
    #[test]
    fn solve_reductions_produce_finite_directions() {
        // Duplicate segment vertices (discrete supports repeat).
        let mut s = Simplex::new(Vector3::new(2.0, 0.0, 0.0));
        s.add(Vector3::new(2.0, 0.0, 0.0));
        let mut dir = Vector3::ZERO;
        assert!(!solve(&mut s, &mut dir));
        assert_finite(dir, "duplicate segment");

        // Collinear triangle: the degenerate branch hands back the (zero)
        // face normal after the closest point collapses onto the origin.
        let mut s = Simplex::new(Vector3::new(3.0, 0.0, 0.0));
        s.add(Vector3::new(1.0, 0.0, 0.0));
        s.add(Vector3::new(-2.0, 0.0, 0.0));
        let mut dir = Vector3::ZERO;
        assert!(!solve(&mut s, &mut dir));
        assert_finite(dir, "collinear triangle");
    }

    /// The enclosing walk's loop-head invariant, observed end to end: a
    /// support closure that collapses onto the origin (producing zero
    /// directions from the segment/triangle reductions) is never queried
    /// with a zero or NaN direction.
    #[test]
    fn enclosing_walk_never_queries_with_zero_direction() {
        use core::cell::{Cell, RefCell};
        let calls = Cell::new(0u32);
        let queried: RefCell<Vec<Vector3>> = RefCell::new(Vec::new());
        let outcome = enclosing_walk(
            |d: Vector3| {
                queried.borrow_mut().push(d);
                let n = calls.get();
                calls.set(n + 1);
                // Duplicate support first, then collapse to the origin: the
                // segment reduction hands back `dir = 0` once a vertex sits
                // on the origin, and the triangle reduction returns its zero
                // face normal for the resulting collinear simplex.
                if n < 1 { Vector3::new(2.0, 0.0, 0.0) } else { Vector3::ZERO }
            },
            Vector3::X,
            None,
        );
        // A point-like Minkowski difference cannot enclose the origin.
        assert!(outcome.is_none());
        assert!(
            queried
                .borrow()
                .iter()
                .all(|d| d.length_squared() > 0.0 && d.x.is_finite()),
            "zero/NaN direction handed to support: {:?}",
            queried.borrow()
        );
    }

    /// Duplicate supports on real shapes: axis-aligned box pairs across the
    /// exact-touch boundary and hull pairs whose supports repeat along the
    /// contact axis keep their classification with finite simplices.
    #[test]
    fn strict_walk_duplicate_supports_stay_classified() {
        let b = CollisionShape::box_shape(Vector3::splat(0.5));
        // Overlapping and separated box pairs, with the exact-touch case in
        // between: at touch the Minkowski difference is a box with the origin
        // at the *center* of a face, and the discrete corner supports cannot
        // assemble an enclosing tetrahedron within the iteration budget, so
        // the strict walk reports no intersection. That is the documented
        // touching boundary of GJK on shapes with discrete supports — the
        // contact pipeline covers it through the band semantics (touch
        // distance 0 < threshold), and the classification is stable either
        // way. The assertions pin the actual behavior; finite simplices are
        // the hard requirement.
        for (offset, expect_hit) in [(0.98, true), (1.0, false), (1.05, false)] {
            let hit = gjk_intersect(
                &b,
                Vector3::ZERO,
                Quaternion::IDENTITY,
                &b,
                Vector3::new(offset, 0.0, 0.0),
                Quaternion::IDENTITY,
            );
            assert_eq!(hit.is_some(), expect_hit, "box offset {offset}");
            if let Some(simplex) = hit {
                for p in simplex.points() {
                    assert_finite(*p, "box touch simplex");
                }
            }
        }

        // Cube hulls across the same boundary: supports are the same
        // discrete corners, so the walk sees duplicated vertices there too.
        let e = 0.5;
        let hull = CollisionShape::convex_hull(vec![
            Vector3::new(-e, -e, -e),
            Vector3::new(e, -e, -e),
            Vector3::new(-e, e, -e),
            Vector3::new(e, e, -e),
            Vector3::new(-e, -e, e),
            Vector3::new(e, -e, e),
            Vector3::new(-e, e, e),
            Vector3::new(e, e, e),
        ]);
        for (offset, expect_hit) in [(0.98, true), (1.0, false), (1.05, false)] {
            let hit = gjk_intersect(
                &hull,
                Vector3::ZERO,
                Quaternion::IDENTITY,
                &hull,
                Vector3::new(offset, 0.0, 0.0),
                Quaternion::IDENTITY,
            );
            assert_eq!(hit.is_some(), expect_hit, "hull offset {offset}");
        }
    }
}
