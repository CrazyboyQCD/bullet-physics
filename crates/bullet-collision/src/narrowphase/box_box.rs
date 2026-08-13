use crate::manifold::{ContactManifold, ContactPoint};
use bullet_math::{Matrix3, Quaternion, Real, Vector3};
use crate::shapes::BoxShape;

const EPSILON: Real = 1e-6;
const FUDGE: Real = 1.05;

/// Extract the `i`-th column of a rotation matrix (a body-local axis).
#[inline]
fn col(m: &Matrix3, i: usize) -> Vector3 {
    match i {
        0 => m.x_axis,
        1 => m.y_axis,
        _ => m.z_axis,
    }
}

/// Extract the `i`-th component of a vector.
#[inline]
fn comp(v: Vector3, i: usize) -> Real {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Closest points between two infinite lines `pa + ua*t` and `pb + ub*s`.
fn line_closest_approach(
    pa: Vector3,
    ua: Vector3,
    pb: Vector3,
    ub: Vector3,
) -> (Real, Real) {
    let p = pb - pa;
    let uaub = ua.dot(ub);
    let q1 = ua.dot(p);
    let q2 = -ub.dot(p);
    let mut d = 1.0 - uaub * uaub;
    if d <= 1e-4 {
        (0.0, 0.0)
    } else {
        d = 1.0 / d;
        ((q1 + uaub * q2) * d, (uaub * q1 + q2) * d)
    }
}

/// Clip a polygon against the half-plane `sign * coord[axis] < h`.
fn clip_poly(poly: &[[Real; 2]], axis: usize, sign: Real, h: Real) -> Vec<[Real; 2]> {
    let n = poly.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..n {
        let cur = poly[i];
        let next = poly[(i + 1) % n];
        let cur_in = sign * cur[axis] < h;
        let next_in = sign * next[axis] < h;
        if cur_in {
            out.push(cur);
        }
        if cur_in != next_in {
            let t = (sign * h - cur[axis]) / (next[axis] - cur[axis]);
            let mut p = [0.0; 2];
            p[axis] = sign * h;
            p[1 - axis] = cur[1 - axis] + (next[1 - axis] - cur[1 - axis]) * t;
            out.push(p);
        }
    }
    out
}

/// Intersect an axis-aligned rectangle `[-h0, h0] x [-h1, h1]` with a convex
/// quad (Sutherland-Hodgman). Returns the clipped polygon vertices.
fn intersect_rect_quad(h: [Real; 2], quad: [[Real; 2]; 4]) -> Vec<[Real; 2]> {
    let mut poly: Vec<[Real; 2]> = quad.to_vec();
    poly = clip_poly(&poly, 0, -1.0, h[0]);
    poly = clip_poly(&poly, 0, 1.0, h[0]);
    poly = clip_poly(&poly, 1, -1.0, h[1]);
    poly = clip_poly(&poly, 1, 1.0, h[1]);
    poly
}

/// Keep the deepest contact point plus the `maxc - 1` points farthest from it,
/// so remaining contacts are well spread (analogous to Bullet's `cullPoints2`).
fn cull_points(points: Vec<(Vector3, Real)>, maxc: usize) -> Vec<(Vector3, Real)> {
    if points.len() <= maxc {
        return points;
    }
    let i0 = points
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1 .1
                .partial_cmp(&b.1 .1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let deepest = points[i0];
    let mut rest: Vec<(usize, Real)> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != i0)
        .map(|(i, (p, _))| (i, (*p - deepest.0).length_squared()))
        .collect();
    rest.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut result = vec![deepest];
    for &(i, _) in rest.iter().take(maxc - 1) {
        result.push(points[i]);
    }
    result
}

/// Box-box contact, adapted from Bullet's `btBoxBoxDetector` (ODE `dBoxBox2`,
/// ZLib license). Contact normals point from box A to box B.
pub fn box_box(
    a: &BoxShape,
    pa: Vector3,
    oa: Quaternion,
    b: &BoxShape,
    pb: Vector3,
    ob: Quaternion,
) -> Option<ContactManifold> {
    let ra = Matrix3::from_quat(oa);
    let rb = Matrix3::from_quat(ob);
    box_box_sat(pa, &ra, a.half_extents, pb, &rb, b.half_extents)
}

#[allow(clippy::too_many_arguments)]
fn box_box_sat(
    p1: Vector3,
    r1: &Matrix3,
    ha: Vector3,
    p2: Vector3,
    r2: &Matrix3,
    hb: Vector3,
) -> Option<ContactManifold> {
    // Relative translation, in box-1 coordinates.
    let p = p2 - p1;
    let pp = r1.transpose() * p;

    // Relative rotation R = R1^T * R2.
    let r = r1.transpose() * *r2;
    let r11 = r.x_axis.x;
    let r12 = r.y_axis.x;
    let r13 = r.z_axis.x;
    let r21 = r.x_axis.y;
    let r22 = r.y_axis.y;
    let r23 = r.z_axis.y;
    let r31 = r.x_axis.z;
    let r32 = r.y_axis.z;
    let r33 = r.z_axis.z;

    let mut q11 = r11.abs();
    let mut q12 = r12.abs();
    let mut q13 = r13.abs();
    let mut q21 = r21.abs();
    let mut q22 = r22.abs();
    let mut q23 = r23.abs();
    let mut q31 = r31.abs();
    let mut q32 = r32.abs();
    let mut q33 = r33.abs();

    // SAT state: `s` holds the maximum (least negative) overlap found so far.
    let mut s = Real::NEG_INFINITY;
    let mut invert_normal = false;
    let mut code = 0;
    let mut normal_from_r2 = false;
    let mut normal_axis = 0usize;
    let mut normal_is_cross = false;
    let mut normal_c = Vector3::ZERO;

    macro_rules! tst_face {
        ($expr1:expr, $expr2:expr, $fr2:expr, $ax:expr, $cc:expr) => {{
            let s2 = ($expr1).abs() - ($expr2);
            if s2 > 0.0 {
                return None;
            }
            if s2 > s {
                s = s2;
                normal_is_cross = false;
                normal_from_r2 = $fr2;
                normal_axis = $ax;
                invert_normal = ($expr1) < 0.0;
                code = $cc;
            }
        }};
    }

    macro_rules! tst_cross {
        ($expr1:expr, $expr2:expr, $n1:expr, $n2:expr, $n3:expr, $cc:expr) => {{
            let s2 = ($expr1).abs() - ($expr2);
            if s2 > EPSILON {
                return None;
            }
            let l = ($n1 * $n1 + $n2 * $n2 + $n3 * $n3).sqrt();
            if l > EPSILON {
                let s2 = s2 / l;
                if s2 * FUDGE > s {
                    s = s2;
                    normal_is_cross = true;
                    normal_c = Vector3::new($n1 / l, $n2 / l, $n3 / l);
                    invert_normal = ($expr1) < 0.0;
                    code = $cc;
                }
            }
        }};
    }

    // Separating axes: box-1 faces.
    tst_face!(pp.x, ha.x + hb.x * q11 + hb.y * q12 + hb.z * q13, false, 0, 1);
    tst_face!(pp.y, ha.y + hb.x * q21 + hb.y * q22 + hb.z * q23, false, 1, 2);
    tst_face!(pp.z, ha.z + hb.x * q31 + hb.y * q32 + hb.z * q33, false, 2, 3);

    // Separating axes: box-2 faces.
    tst_face!(
        r2.x_axis.dot(p),
        ha.x * q11 + ha.y * q21 + ha.z * q31 + hb.x,
        true,
        0,
        4
    );
    tst_face!(
        r2.y_axis.dot(p),
        ha.x * q12 + ha.y * q22 + ha.z * q32 + hb.y,
        true,
        1,
        5
    );
    tst_face!(
        r2.z_axis.dot(p),
        ha.x * q13 + ha.y * q23 + ha.z * q33 + hb.z,
        true,
        2,
        6
    );

    // Edge-edge cross-product axes (with a small fudge to avoid degenerate
    // near-parallel edge cases).
    let fudge2: Real = 1e-5;
    q11 += fudge2;
    q12 += fudge2;
    q13 += fudge2;
    q21 += fudge2;
    q22 += fudge2;
    q23 += fudge2;
    q31 += fudge2;
    q32 += fudge2;
    q33 += fudge2;

    // u1 x (v1, v2, v3)
    tst_cross!(
        pp.z * r21 - pp.y * r31,
        ha.y * q31 + ha.z * q21 + hb.y * q13 + hb.z * q12,
        0.0,
        -r31,
        r21,
        7
    );
    tst_cross!(
        pp.z * r22 - pp.y * r32,
        ha.y * q32 + ha.z * q22 + hb.x * q13 + hb.z * q11,
        0.0,
        -r32,
        r22,
        8
    );
    tst_cross!(
        pp.z * r23 - pp.y * r33,
        ha.y * q33 + ha.z * q23 + hb.x * q12 + hb.y * q11,
        0.0,
        -r33,
        r23,
        9
    );
    // u2 x (v1, v2, v3)
    tst_cross!(
        pp.x * r31 - pp.z * r11,
        ha.x * q31 + ha.z * q11 + hb.y * q23 + hb.z * q22,
        r31,
        0.0,
        -r11,
        10
    );
    tst_cross!(
        pp.x * r32 - pp.z * r12,
        ha.x * q32 + ha.z * q12 + hb.x * q23 + hb.z * q21,
        r32,
        0.0,
        -r12,
        11
    );
    tst_cross!(
        pp.x * r33 - pp.z * r13,
        ha.x * q33 + ha.z * q13 + hb.x * q22 + hb.y * q21,
        r33,
        0.0,
        -r13,
        12
    );
    // u3 x (v1, v2, v3)
    tst_cross!(
        pp.y * r11 - pp.x * r21,
        ha.x * q21 + ha.y * q11 + hb.y * q33 + hb.z * q32,
        -r21,
        r11,
        0.0,
        13
    );
    tst_cross!(
        pp.y * r12 - pp.x * r22,
        ha.x * q22 + ha.y * q12 + hb.x * q33 + hb.z * q31,
        -r22,
        r12,
        0.0,
        14
    );
    tst_cross!(
        pp.y * r13 - pp.x * r23,
        ha.x * q23 + ha.y * q13 + hb.x * q32 + hb.y * q31,
        -r23,
        r13,
        0.0,
        15
    );

    // Compute the separating axis in world space, pointing from box A to box B
    // (verified geometrically; matches the contact-normal convention).
    let normal_sat = if normal_is_cross {
        *r1 * normal_c
    } else if normal_from_r2 {
        col(r2, normal_axis)
    } else {
        col(r1, normal_axis)
    };
    let normal_sat = if invert_normal { -normal_sat } else { normal_sat };
    let depth = -s;

    // Edge-edge contact.
    if code > 6 {
        let mut pa = p1;
        for j in 0..3 {
            let c = col(r1, j);
            let sign = if normal_sat.dot(c) > 0.0 { 1.0 } else { -1.0 };
            pa += sign * comp(ha, j) * c;
        }
        let mut pb = p2;
        for j in 0..3 {
            let c = col(r2, j);
            let sign = if normal_sat.dot(c) > 0.0 { -1.0 } else { 1.0 };
            pb += sign * comp(hb, j) * c;
        }
        let ua = col(r1, (code - 7) / 3);
        let ub = col(r2, (code - 7) % 3);
        let (alpha, beta) = line_closest_approach(pa, ua, pb, ub);
        pa += ua * alpha;
        pb += ub * beta;
        return Some(ContactManifold::from_points(vec![ContactPoint::new(
            pb, normal_sat, depth,
        )]));
    }

    // Face-face contact: clip the incident face against the reference face.
    let (ra, rb, pa_ref, pb_inc, sa, sb) = if code <= 3 {
        (r1, r2, p1, p2, ha, hb)
    } else {
        (r2, r1, p2, p1, hb, ha)
    };

    let normal2 = if code <= 3 { normal_sat } else { -normal_sat };
    let nr = rb.transpose() * normal2;
    let anr = nr.abs();

    let (lanr, a1, a2) = if anr.y > anr.x {
        if anr.y > anr.z {
            (1, 0, 2)
        } else {
            (2, 0, 1)
        }
    } else if anr.x > anr.z {
        (0, 1, 2)
    } else {
        (2, 0, 1)
    };

    let center = if comp(nr, lanr) < 0.0 {
        pb_inc - pa_ref + comp(sb, lanr) * col(rb, lanr)
    } else {
        pb_inc - pa_ref - comp(sb, lanr) * col(rb, lanr)
    };

    let code_n = if code <= 3 { code - 1 } else { code - 4 };
    let (code1, code2) = if code_n == 0 {
        (1, 2)
    } else if code_n == 1 {
        (0, 2)
    } else {
        (0, 1)
    };

    let c1 = center.dot(col(ra, code1));
    let c2 = center.dot(col(ra, code2));
    let m11 = col(ra, code1).dot(col(rb, a1));
    let m12 = col(ra, code1).dot(col(rb, a2));
    let m21 = col(ra, code2).dot(col(rb, a1));
    let m22 = col(ra, code2).dot(col(rb, a2));

    let k1 = m11 * comp(sb, a1);
    let k2 = m21 * comp(sb, a1);
    let k3 = m12 * comp(sb, a2);
    let k4 = m22 * comp(sb, a2);

    let quad = [
        [c1 - k1 - k3, c2 - k2 - k4],
        [c1 - k1 + k3, c2 - k2 + k4],
        [c1 + k1 + k3, c2 + k2 + k4],
        [c1 + k1 - k3, c2 + k2 - k4],
    ];

    let rect = [comp(sa, code1), comp(sa, code2)];
    let ret = intersect_rect_quad(rect, quad);
    if ret.is_empty() {
        return None;
    }

    // Recover 3D contact positions and per-point depths.
    let det1 = 1.0 / (m11 * m22 - m12 * m21);
    let (m11, m12, m21, m22) = (m11 * det1, m12 * det1, m21 * det1, m22 * det1);

    let mut penetrating: Vec<(Vector3, Real)> = Vec::new();
    for pt in &ret {
        let rx = pt[0];
        let ry = pt[1];
        let k1 = m22 * (rx - c1) - m12 * (ry - c2);
        let k2 = -m21 * (rx - c1) + m11 * (ry - c2);
        let point = center + k1 * col(rb, a1) + k2 * col(rb, a2);
        let dep = comp(sa, code_n) - normal2.dot(point);
        if dep >= 0.0 {
            penetrating.push((point, dep));
        }
    }
    if penetrating.is_empty() {
        return None;
    }

    let penetrating = cull_points(penetrating, 4);
    let mut contacts = Vec::with_capacity(penetrating.len());
    for (point, dep) in penetrating {
        // `point` is expressed relative to the reference-box center; move it
        // back to world space. The contact normal points from box A to box B.
        let pos = point + pa_ref;
        contacts.push(ContactPoint::new(pos, normal_sat, dep));
    }
    Some(ContactManifold::from_points(contacts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn face_face_contact_along_x() {
        let a = BoxShape::new(Vector3::splat(1.0));
        let b = BoxShape::new(Vector3::splat(1.0));
        let m = box_box(&a, Vector3::ZERO, Quaternion::IDENTITY, &b, Vector3::new(1.9, 0.0, 0.0), Quaternion::IDENTITY)
            .unwrap();
        assert!(!m.points.is_empty());
        for p in &m.points {
            assert_relative_eq!(p.normal.x, 1.0, epsilon = 1e-5);
            assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-4);
        }
    }

    #[test]
    fn separated_boxes() {
        let a = BoxShape::new(Vector3::splat(1.0));
        let b = BoxShape::new(Vector3::splat(1.0));
        assert!(box_box(&a, Vector3::ZERO, Quaternion::IDENTITY, &b, Vector3::new(3.0, 0.0, 0.0), Quaternion::IDENTITY)
            .is_none());
    }
}
