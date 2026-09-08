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
//!
//! Failure semantics: hull expansions are *atomic*. A degenerate expansion is
//! validated against the horizon before the hull is touched, so a rejected
//! expansion leaves the polytope exactly as it was (never a hole), and the
//! run ends with the intact hull's closest face — Bullet's `InvalidHull`
//! early exit. A fully degenerate seed (no valid tetrahedron face) returns
//! the zero normal and zero depth. Every path terminates at `MAX_ITERATIONS`
//! / `MAX_VERTICES` and produces finite values; the internal [`EpaStatus`]
//! distinguishes convergence from the bounded exits (the public
//! [`penetration`] keeps its `(normal, depth)` shape — its only caller
//! rejects non-positive depths).

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

/// Internal outcome of an EPA run. Kept crate-private: the public
/// [`penetration`] returns the best `(normal, depth)` of the *intact* hull in
/// every exit, and its callers reject non-positive depths, so a public
/// result type is not warranted. Tests assert on this to pin the failure
/// paths, and `convex_convex` consumes it to mirror Bullet's per-status
/// acceptance semantics.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EpaStatus {
    /// The closest face bounds the polytope (support point within
    /// `EPSILON` of it): normal convergence. Bullet's `AccuraryReached`.
    Converged,
    /// A degenerate expansion was rejected before the hull was mutated
    /// (including the fully degenerate seed returning the zero result) —
    /// Bullet's `InvalidHull`. Bullet's expand is transactional too (the
    /// hull is committed only after a fully valid horizon), so this mirrors
    /// its "return the intact hull's best face" exit.
    DegenerateExpansion,
    /// `MAX_ITERATIONS` expansions ran without converging. Bullet's loop
    /// simply ends after `EPA_MAX_ITERATIONS` with the status still `Valid`
    /// and reports the current best face — accepted there as well.
    IterationLimit,
    /// `MAX_VERTICES` reached; the polytope stopped growing (Bullet's
    /// `OutOfVertices`, which also reports the best face and is accepted).
    VertexLimit,
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
    let (_, normal, depth) = penetration_inner(simplex, support, scratch);
    (normal, depth)
}

/// Crate-internal variant of [`penetration`] that also reports the
/// [`EpaStatus`], so the contact layer can mirror Bullet's acceptance
/// semantics (every non-fatal exit reports the intact hull's best face and
/// is accepted; only a zero/negative depth — the degenerate seed's zero
/// result or Bullet's `FallBack` with `m_depth = 0` — yields no contact).
pub(crate) fn penetration_with_status(
    simplex: &[Vector3; 4],
    support: impl Fn(Vector3) -> Vector3,
    scratch: &mut EpaScratch,
) -> (EpaStatus, Vector3, Real) {
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
) -> (EpaStatus, Vector3, Real) {
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
    if faces.is_empty() {
        // Fully degenerate tetrahedron (collinear/coplanar vertices): no
        // polytope to expand. Non-panicking exit with the zero result —
        // Bullet's `InvalidHull`.
        return (EpaStatus::DegenerateExpansion, Vector3::ZERO, 0.0);
    }

    let mut status = EpaStatus::IterationLimit;
    for d in [
        Vector3::X,
        Vector3::NEG_X,
        Vector3::Y,
        Vector3::NEG_Y,
        Vector3::Z,
        Vector3::NEG_Z,
    ] {
        // expand() is a no-op when the point already lies inside the polytope.
        // Degenerate expansion is rejected atomically (the hull is untouched);
        // the run then continues from the intact, partially seeded polytope
        // (Bullet's `InvalidHull` seeding exit).
        let idx = vertices.len();
        let point = support(d);
        vertices.push(point);
        if !expand(faces, vertices, boundary, point, idx) {
            // Report the seeding failure explicitly (Bullet's `InvalidHull`);
            // without this the status variable below would still hold its
            // default and misreport the run as an iteration-budget exit.
            status = EpaStatus::DegenerateExpansion;
            break;
        }
    }
    for _ in 0..MAX_ITERATIONS {
        let (_, best_normal, best_dist) = find_best(faces);

        let new_point = support(best_normal);
        // Bullet: `wdist = dot(best->n, w) - best->d`; only expand when the
        // support point is meaningfully outside the best face.
        let wdist = best_normal.dot(new_point) - best_dist;
        if wdist <= EPSILON {
            // Converged: the closest face bounds the polytope.
            return (EpaStatus::Converged, best_normal, best_dist);
        }

        // Remove the face and rebuild the hull around the new point.
        if vertices.len() >= MAX_VERTICES {
            // Bullet's `OutOfVertices`: stop growing the polytope.
            status = EpaStatus::VertexLimit;
            break;
        }
        let new_idx = vertices.len();
        vertices.push(new_point);
        if !expand(faces, vertices, boundary, new_point, new_idx) {
            // Degenerate expansion rejected atomically: the hull is intact,
            // so its closest face is a valid answer (Bullet's `InvalidHull`
            // early exit).
            status = EpaStatus::DegenerateExpansion;
            break;
        }
    }

    let (_, best_normal, best_dist) = find_best(faces);
    (status, best_normal, best_dist)
}

/// The closest face: smallest absolute distance to the origin (Bullet's
/// `findbest`, which compares `d * d`). Callers must pass a non-empty face
/// set: `penetration_inner` guarantees this (the degenerate-seed exit returns
/// before any `find_best`, and an atomic `expand` never empties the hull —
/// a fully visible hull leaves its horizon ring as the new faces).
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
/// new ones along the horizon. Returns `false` when the expansion is
/// degenerate; the hull is then **unchanged** (the expansion is validated in
/// full before any mutation), and the caller should stop iterating.
fn expand(
    faces: &mut Vec<Face>,
    vertices: &[Vector3],
    boundary: &mut Vec<(usize, usize)>,
    new_point: Vector3,
    new_idx: usize,
) -> bool {
    // Two-phase expansion. Phase 1 classifies the visible faces and collects
    // the horizon loop *without touching `faces`*; a degenerate horizon edge
    // aborts here and the hull stays intact (a mid-stitch abort would leave a
    // hole). The visibility predicate is re-evaluated in phase 2 on the same
    // inputs with the same float ops, so the classification is identical.
    // Phase 2 commits: remove the visible faces, then stitch — every stitch
    // edge was validated in phase 1, so the commit cannot fail halfway.
    boundary.clear();

    // Phase 1: collect the boundary loop of the visible region. A directed
    // edge whose reverse is already tracked is interior to the visible region
    // (cancel both); otherwise it lies on the horizon. The horizon is small
    // (tens of edges), so linear matching beats a hash set and its SipHash
    // cost — which dominated the profile.
    let mut any_visible = false;
    for f in faces.iter() {
        if !face_visible(f, new_point) {
            continue;
        }
        any_visible = true;
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
    if !any_visible {
        // The point lies inside (or on) the polytope: nothing to do.
        return true;
    }

    // Validate the whole horizon before mutating anything.
    for &(u, v) in boundary.iter() {
        let cross = (vertices[v] - vertices[u]).cross(new_point - vertices[u]);
        if cross.length() < 1e-12 {
            // Degenerate edge (collinear with the new point): the horizon
            // would not close, so reject the expansion atomically.
            return false;
        }
    }

    // Phase 2: commit. Remove the visible faces...
    faces.retain(|f| !face_visible(f, new_point));

    // ...then stitch the horizon. Drain keeps the pooled vec's capacity for
    // the next expansion.
    for (u, v) in boundary.drain(..) {
        let cross = (vertices[v] - vertices[u]).cross(new_point - vertices[u]);
        let len = cross.length();
        debug_assert!(len >= 1e-12, "horizon edge validated in phase 1");
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

/// Face visibility from `new_point`: strictly outside the face plane by more
/// than the coplanar tolerance (Bullet's `-EPA_PLANE_EPS`): a support point
/// on the face plane is treated as inside, keeping the horizon closed.
/// Called twice per face per expansion (collect + commit) with identical
/// inputs, so both phases always agree.
#[inline]
fn face_visible(f: &Face, new_point: Vector3) -> bool {
    f.normal.dot(new_point) - f.plane_distance > PLANE_EPS
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

    // ------------------------------------------------------------------
    // Adversarial failure paths. Every failure-path test asserts: no panic,
    // finite normal/depth, and no invalid (negative/huge) depth unless the
    // run is explicitly reported as a failed status (zero result).
    // ------------------------------------------------------------------

    /// Convex-hull support over a finite point set (argmax of `d . p`).
    fn point_support(points: &[Vector3]) -> impl Fn(Vector3) -> Vector3 + '_ {
        move |dir| {
            let mut best = points[0];
            let mut best_d = best.dot(dir);
            for &p in &points[1..] {
                let d = p.dot(dir);
                if d > best_d {
                    best = p;
                    best_d = d;
                }
            }
            best
        }
    }

    fn check_finite(normal: Vector3, depth: Real, scale: Real) {
        assert!(normal.x.is_finite() && normal.y.is_finite() && normal.z.is_finite());
        assert!(depth.is_finite(), "depth must not be NaN/Inf");
        assert!(
            depth >= -1e-3 * scale && depth <= 1e3 * scale,
            "depth out of the plausible band: {depth}"
        );
    }

    #[test]
    fn epa_degenerate_seed_returns_zero_cleanly() {
        // All four simplex vertices identical: every tetrahedron face is
        // degenerate, the seed has no valid face, and the run must exit
        // without panicking and without touching `find_best` on an empty
        // hull.
        let v = Vector3::new(0.3, -0.2, 0.5);
        let simplex = [v; 4];
        let points = [v];
        let support = point_support(&points);
        let (status, normal, depth) =
            penetration_inner(&simplex, &support, &mut EpaScratch::default());
        assert_eq!(status, EpaStatus::DegenerateExpansion);
        assert_eq!(normal, Vector3::ZERO);
        assert_eq!(depth, 0.0);

        // The public wrapper reports the same zero result, still no panic.
        let (normal, depth) = penetration(&simplex, &support, &mut EpaScratch::default());
        assert_eq!(normal, Vector3::ZERO);
        assert_eq!(depth, 0.0);
    }

    #[test]
    fn epa_collinear_simplex_fail_clean() {
        // Four distinct but collinear vertices: all faces are degenerate.
        let simplex = [
            Vector3::ONE,
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::NEG_ONE,
            Vector3::new(-2.0, -2.0, -2.0),
        ];
        let support = point_support(&simplex);
        let (status, normal, depth) =
            penetration_inner(&simplex, support, &mut EpaScratch::default());
        assert_eq!(status, EpaStatus::DegenerateExpansion);
        assert_eq!(normal, Vector3::ZERO);
        assert_eq!(depth, 0.0);
    }

    #[test]
    fn epa_near_degenerate_seed_stays_clean() {
        // Two duplicate vertices, two distinct: half the initial faces are
        // degenerate. Whatever exit the run takes, it must be finite and
        // panic-free.
        let a = Vector3::new(1.0, 0.0, 0.2);
        let c = Vector3::new(-0.5, 1.0, -0.3);
        let d = Vector3::new(-0.4, -0.9, 0.4);
        let simplex = [a, a, c, d];
        let points = [a, c, d, a * 1.5, c * -1.2];
        let support = point_support(&points);
        let (status, normal, depth) =
            penetration_inner(&simplex, support, &mut EpaScratch::default());
        let _ = status; // any exit is acceptable; the run must stay clean
        check_finite(normal, depth, 2.0);
    }

    #[test]
    fn epa_coplanar_simplex_stays_clean() {
        // Four coplanar vertices (a flat quad): zero-volume "hull". The run
        // must terminate with finite values and no panic.
        let simplex = [
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(-1.0, 1.0, 0.0),
            Vector3::new(-1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
        ];
        let support = point_support(&simplex);
        let (status, normal, depth) =
            penetration_inner(&simplex, support, &mut EpaScratch::default());
        let _ = status; // degenerate seed or a bounded run — both acceptable
        check_finite(normal, depth, 2.0);
    }

    #[test]
    fn epa_support_never_offers_new_point_converges() {
        // The support function's point set *is* the simplex: after the seed
        // no expansion can find a point outside the best face, so the run
        // converges immediately (the exact-touching analogue for a polytope
        // that cannot grow).
        let simplex = [
            Vector3::new(0.6, 0.6, 0.6),
            Vector3::new(-0.6, -0.6, 0.6),
            Vector3::new(-0.6, 0.6, -0.6),
            Vector3::new(0.6, -0.6, -0.6),
        ];
        let support = point_support(&simplex);
        let (status, normal, depth) =
            penetration_inner(&simplex, support, &mut EpaScratch::default());
        assert_eq!(status, EpaStatus::Converged);
        check_finite(normal, depth, 2.0);
        assert!(depth >= -1e-3, "non-enclosing seed must not report depth");
    }

    #[test]
    fn epa_thin_plate_horizon_stays_clean() {
        // Two thin plates overlapping by less than their thickness: the
        // horizon edges are nearly collinear with the support points, the
        // classic degenerate-horizon trigger. The expansion may be rejected,
        // but never partially applied and never with a panic/NaN.
        let a = CollisionShape::box_shape(Vector3::new(0.5, 0.0005, 0.5));
        let b = CollisionShape::box_shape(Vector3::new(0.5, 0.0005, 0.5));
        let pa = Vector3::ZERO;
        let pb = Vector3::new(0.05, 0.0004, 0.03);
        let simplex = match gjk::gjk_intersect(
            &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY,
        ) {
            Some(s) => s,
            None => return, // GJK reported separation; EPA is not reached
        };
        let mut sp = [Vector3::ZERO; 4];
        for (i, v) in simplex.points().iter().take(4).enumerate() {
            sp[i] = *v;
        }
        let support = |dir: Vector3| {
            gjk::support_minkowski(
                &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY, dir,
            )
        };
        let (normal, depth) = penetration(&sp, support, &mut EpaScratch::default());
        check_finite(normal, depth, 1.0);
    }

    #[test]
    fn epa_tiny_penetration_is_finite_and_tiny() {
        // Overlap of ~1e-4 — the same order as `EPA_ACCURACY` (f32): the
        // run may converge early, but must stay finite and near-zero.
        let r = 1.0;
        let a = CollisionShape::sphere(r);
        let b = CollisionShape::sphere(r);
        let pa = Vector3::ZERO;
        let pb = Vector3::new(2.0 - 1e-4, 0.0, 0.0);
        let Some(simplex) = gjk::gjk_intersect(
            &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY,
        ) else {
            return; // below the GJK overlap resolution — not an EPA failure
        };
        let mut sp = [Vector3::ZERO; 4];
        for (i, v) in simplex.points().iter().take(4).enumerate() {
            sp[i] = *v;
        }
        let support = |dir: Vector3| {
            gjk::support_minkowski(
                &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY, dir,
            )
        };
        let (normal, depth) = penetration(&sp, support, &mut EpaScratch::default());
        check_finite(normal, depth, 1.0);
        assert!(depth >= -1e-3, "tiny overlap must not report a real depth");
        assert!(depth < 0.1, "tiny overlap must not report a large depth");
    }

    #[test]
    fn epa_elongated_convex_stays_clean() {
        // A 200:1 sliver box against a small sphere crossing its thin axis:
        // the edge-aware `face_distance` fallback is load-bearing here.
        let a = CollisionShape::box_shape(Vector3::new(1.0, 0.005, 0.005));
        let b = CollisionShape::sphere(0.02);
        let pa = Vector3::ZERO;
        let pb = Vector3::new(0.3, 0.01, 0.0);
        let Some(simplex) = gjk::gjk_intersect(
            &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY,
        ) else {
            return;
        };
        let mut sp = [Vector3::ZERO; 4];
        for (i, v) in simplex.points().iter().take(4).enumerate() {
            sp[i] = *v;
        }
        let support = |dir: Vector3| {
            gjk::support_minkowski(
                &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY, dir,
            )
        };
        let (normal, depth) = penetration(&sp, support, &mut EpaScratch::default());
        check_finite(normal, depth, 1.0);
        let n2 = normal.length_squared();
        assert!(
            (n2 - 1.0).abs() < 1e-2 || n2 < 1e-8,
            "normal must be unit or the zero result, got |n|² = {n2}"
        );
    }

    #[test]
    fn epa_scale_sweep_spheres() {
        // Penetration of two equal spheres across five magnitudes. The
        // depth scales with the scene; f32's `EPA_ACCURACY = 1e-4` floor
        // dominates the smallest scales (Bullet has the same constant), so
        // the assertions are finiteness + band + direction, not precision.
        for &scale in &[1e-3_f64, 1e-2, 1.0, 1e2, 1e3] {
            let r = scale as Real;
            let a = CollisionShape::sphere(r);
            let b = CollisionShape::sphere(r);
            let pa = Vector3::ZERO;
            // 20 % overlap along an oblique offset (avoids axis-aligned
            // degenerate simplices).
            let offset = (r * 1.6, r * 0.3, r * 0.2);
            let pb = Vector3::new(offset.0, offset.1, offset.2);
            let expected = 2.0 * r - pb.length();

            let Some(simplex) = gjk::gjk_intersect(
                &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY,
            ) else {
                // No enclosing simplex from GJK at this scale — nothing to
                // assert about EPA (recorded in the report, not invented).
                continue;
            };
            let mut sp = [Vector3::ZERO; 4];
            for (i, v) in simplex.points().iter().take(4).enumerate() {
                sp[i] = *v;
            }
            let support = |dir: Vector3| {
                gjk::support_minkowski(
                    &a, pa, Quaternion::IDENTITY, &b, pb, Quaternion::IDENTITY, dir,
                )
            };
            let (normal, depth) = penetration(&sp, support, &mut EpaScratch::default());
            check_finite(normal, depth, r.max(1.0));
            assert!(
                depth > -expected.abs() * 1e-2 && depth < expected * 4.0 + r * 1e-2,
                "scale {scale}: depth {depth} outside the plausible band around {expected}"
            );
            if expected > 1e-3 * r {
                assert!(normal.dot(pb.normalize()) > 0.0, "scale {scale}: normal must face the separation direction");
            }
        }
    }
}
