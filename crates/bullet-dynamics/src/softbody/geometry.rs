//! Soft-body geometry helpers — port of the static inline functions
//! `btSoftBodyInternals.h` weaves through the collision and force kernels
//! (`ProjectOnAxis`, `ProjectOnPlane`, `ProjectOrigin`, `BaryCoord`,
//! `BaryEval`, `AreaOf`, `ApplyClampedForce`, `rayFromToTriangle`).

use bullet_math::{math, Real, Vector3};

/// `SIMD_EPSILON` gates in these helpers are precision-dependent like every
/// Bullet epsilon.
#[cfg(feature = "f64")]
const EPS: Real = 2.2e-16;
#[cfg(not(feature = "f64"))]
const EPS: Real = 1.19e-7;

/// `v - ProjectOnAxis(v, a)` — `a` is expected to be normalized.
#[inline]
pub fn project_on_axis(v: Vector3, a: Vector3) -> Vector3 {
    a * v.dot(a)
}

/// `v - ProjectOnPlane(v, a)` — `a` is expected to be normalized.
#[inline]
pub fn project_on_plane(v: Vector3, a: Vector3) -> Vector3 {
    v - project_on_axis(v, a)
}

/// Closest point of segment `[a, b]` to the origin (Bullet's two-argument
/// `ProjectOrigin`), updating `prj`/`sqd` when closer.
#[inline]
fn project_origin_segment(a: Vector3, b: Vector3, prj: &mut Vector3, sqd: &mut Real) {
    let d = b - a;
    let m2 = d.length_squared();
    if m2 > EPS {
        let t = (-a.dot(d) / m2).clamp(0.0, 1.0);
        let p = a + d * t;
        let l2 = p.length_squared();
        if l2 < *sqd {
            *prj = p;
            *sqd = l2;
        }
    }
}

/// Closest point of triangle `(a, b, c)` to the origin (Bullet's three-
/// argument `ProjectOrigin`), updating `prj`/`sqd` when closer.
#[inline]
pub fn project_origin_triangle(a: Vector3, b: Vector3, c: Vector3, prj: &mut Vector3, sqd: &mut Real) {
    let q = (b - a).cross(c - a);
    let m2 = q.length_squared();
    if m2 > EPS {
        let n = q / math::sqrt(m2);
        let k = a.dot(n);
        let k2 = k * k;
        if k2 < *sqd {
            let p = n * k;
            let inside = (a - p).cross(b - p).dot(q) > 0.0
                && (b - p).cross(c - p).dot(q) > 0.0
                && (c - p).cross(a - p).dot(q) > 0.0;
            if inside {
                *prj = p;
                *sqd = k2;
            } else {
                project_origin_segment(a, b, prj, sqd);
                project_origin_segment(b, c, prj, sqd);
                project_origin_segment(c, a, prj, sqd);
            }
        }
    }
}

/// Barycentric evaluation `a*w.x + b*w.y + c*w.z` (Bullet's generic
/// `BaryEval<T>`; this crate needs it for `Vector3` and `Real`).
#[inline]
pub fn bary_eval_vec3(a: Vector3, b: Vector3, c: Vector3, w: Vector3) -> Vector3 {
    a * w.x + b * w.y + c * w.z
}

/// Scalar flavor of [`bary_eval_vec3`].
#[inline]
pub fn bary_eval_real(a: Real, b: Real, c: Real, w: Vector3) -> Real {
    a * w.x + b * w.y + c * w.z
}

/// Barycentric coordinates of `p` in triangle `(a, b, c)` (Bullet's
/// `BaryCoord`, area-weighted with a cyclic shift).
#[inline]
pub fn bary_coord(a: Vector3, b: Vector3, c: Vector3, p: Vector3) -> Vector3 {
    let w = [
        (a - p).cross(b - p).length(),
        (b - p).cross(c - p).length(),
        (c - p).cross(a - p).length(),
    ];
    let isum = 1.0 / (w[0] + w[1] + w[2]);
    Vector3::new(w[1] * isum, w[2] * isum, w[0] * isum)
}

/// Doubled triangle area (Bullet's `AreaOf` — the cross product length, so
/// callers that want the true area halve it).
#[inline]
pub fn area_of(x0: Vector3, x1: Vector3, x2: Vector3) -> Real {
    (x1 - x0).cross(x2 - x0).length()
}

/// Six times the signed tetrahedron volume (Bullet's `VolumeOf`: the triple
/// product of the three edges leaving `x0`; `append_tetra` stores it raw and
/// the mass helpers take its absolute value).
#[inline]
pub fn volume_of(x0: Vector3, x1: Vector3, x2: Vector3, x3: Vector3) -> Real {
    let a = x1 - x0;
    let b = x2 - x0;
    let c = x3 - x0;
    a.dot(b.cross(c))
}

/// Bullet's `ApplyClampedForce`: apply `f` to node `im` unless it would
/// reverse the node's velocity within dt, in which case only cancel the
/// velocity along the force direction.
#[inline]
pub fn apply_clamped_force(
    force: Vector3,
    im: Real,
    v: Vector3,
    dt: Real,
    node_force: &mut Vector3,
) {
    let dtim = dt * im;
    if (force * dtim).length_squared() > v.length_squared() {
        // Clamp: cancel the velocity component along the force.
        *node_force -= project_on_axis(v, force.normalize()) / dtim;
    } else {
        *node_force += force;
    }
}

/// Bullet's `RayFromToCaster::rayFromToTriangle`: segment-vs-triangle hit
/// time `t ∈ (teps, maxt)` along the full `rayFrom -> rayTo` segment, `-1`
/// on miss. Edge tests use a slightly negative tolerance (`ceps`).
#[allow(clippy::too_many_arguments)]
pub fn ray_from_to_triangle(
    ray_from: Vector3,
    _ray_to: Vector3,
    dir: Vector3,
    a: Vector3,
    b: Vector3,
    c: Vector3,
    maxt: Real,
) -> Real {
    let ceps = -EPS * 10.0;
    let teps = EPS * 10.0;

    let n = (b - a).cross(c - a);
    let d = a.dot(n);
    let den = dir.dot(n);
    // Bullet: `if (!btFuzzyZero(den))` — |den| >= SIMD_EPSILON.
    if den.abs() >= EPS {
        let num = ray_from.dot(n) - d;
        let t = -num / den;
        if t > teps && t < maxt {
            let hit = ray_from + dir * t;
            if n.dot((a - hit).cross(b - hit)) > ceps
                && n.dot((b - hit).cross(c - hit)) > ceps
                && n.dot((c - hit).cross(a - hit)) > ceps
            {
                return t;
            }
        }
    }
    -1.0
}
