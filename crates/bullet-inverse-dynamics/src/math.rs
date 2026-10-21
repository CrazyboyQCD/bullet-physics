//! Math helpers for the inverse-dynamics tree — port of
//! `BulletInverseDynamics/IDMath.{hpp,cpp}`.
//!
//! Matrix conventions follow the reference implementation verbatim: the
//! `transform_*` builders and [`body_t_parent_from_axis_angle`] produce
//! *transposed* (parent-to-body) rotations, which is why the ported formulas
//! look unusual next to textbook rotation matrices (e.g. the negated sine in
//! the axis-angle form, `IDMath.cpp:258-279`). Do not "fix" them.

use bullet_math::{math, Matrix3, Real, Vector3};

/// Rows `[[1,0,0],[0,c,s],[0,-s,c]]` (`IDMath.cpp:150-171`).
pub fn transform_x(alpha: Real) -> Matrix3 {
    let c = math::cos(alpha);
    let s = math::sin(alpha);
    Matrix3::from_cols(
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, c, -s),
        Vector3::new(0.0, s, c),
    )
}

/// Rows `[[c,0,-s],[0,1,0],[s,0,c]]` (`IDMath.cpp:173-194`).
pub fn transform_y(beta: Real) -> Matrix3 {
    let c = math::cos(beta);
    let s = math::sin(beta);
    Matrix3::from_cols(
        Vector3::new(c, 0.0, s),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(-s, 0.0, c),
    )
}

/// Rows `[[c,s,0],[-s,c,0],[0,0,1]]` (`IDMath.cpp:196-217`).
pub fn transform_z(gamma: Real) -> Matrix3 {
    let c = math::cos(gamma);
    let s = math::sin(gamma);
    Matrix3::from_cols(
        Vector3::new(c, -s, 0.0),
        Vector3::new(s, c, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    )
}

/// The cross-product (skew-symmetric) matrix of `v`,
/// rows `[[0,-z,y],[z,0,-x],[-y,x,0]]` (`IDMath.cpp:219-232`).
pub fn tilde_operator(v: Vector3) -> Matrix3 {
    Matrix3::from_cols(
        Vector3::new(0.0, v.z, -v.y),
        Vector3::new(-v.z, 0.0, v.x),
        Vector3::new(v.y, -v.x, 0.0),
    )
}

/// Parent-to-body rotation for `angle` about `axis`, the transposed Rodrigues
/// form with `s = -sin(angle)` (`IDMath.cpp:258-279`).
pub fn body_t_parent_from_axis_angle(axis: Vector3, angle: Real) -> Matrix3 {
    let c = math::cos(angle);
    let s = -math::sin(angle);
    let one_m_c = 1.0 - c;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    Matrix3::from_cols(
        Vector3::new(
            x * x * one_m_c + c,
            x * y * one_m_c + z * s,
            x * z * one_m_c - y * s,
        ),
        Vector3::new(
            x * y * one_m_c - z * s,
            y * y * one_m_c + c,
            y * z * one_m_c + x * s,
        ),
        Vector3::new(
            x * z * one_m_c + y * s,
            y * z * one_m_c - x * s,
            z * z * one_m_c + c,
        ),
    )
}

/// Determinant with the upstream cofactor-expansion order
/// (`IDMath.cpp:341-345`).
pub fn determinant(m: &Matrix3) -> Real {
    m.x_axis.x * m.y_axis.y * m.z_axis.z
        + m.y_axis.x * m.z_axis.y * m.x_axis.z
        + m.z_axis.x * m.x_axis.y * m.y_axis.z
        - m.z_axis.x * m.y_axis.y * m.x_axis.z
        - m.x_axis.x * m.z_axis.y * m.y_axis.z
        - m.y_axis.x * m.x_axis.y * m.z_axis.z
}

#[cfg(test)]
mod tests {
    use super::{
        body_t_parent_from_axis_angle, determinant, tilde_operator, transform_x, transform_y,
        transform_z,
    };
    use bullet_math::{FRAC_PI_2, Matrix3, Real, Vector3};

    fn get(m: &Matrix3, row: usize, col: usize) -> Real {
        let c = m.col(col);
        [c.x, c.y, c.z][row]
    }

    #[test]
    fn zero_angle_transforms_are_identity() {
        for m in [
            transform_x(0.0),
            transform_y(0.0),
            transform_z(0.0),
            body_t_parent_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.0),
        ] {
            for row in 0..3 {
                for col in 0..3 {
                    let want = if row == col { 1.0 } else { 0.0 };
                    assert_eq!(get(&m, row, col), want);
                }
            }
        }
    }

    #[test]
    fn elementary_transforms_match_the_reference_rows() {
        // Row-indexed entries of the reference formulas, evaluated at a
        // quarter turn where sin/cos are both non-trivial.
        let a = FRAC_PI_2;
        let (c, s) = (bullet_math::math::cos(a), bullet_math::math::sin(a));
        let tx = transform_x(a);
        assert_eq!(get(&tx, 1, 1), c);
        assert_eq!(get(&tx, 1, 2), s);
        assert_eq!(get(&tx, 2, 1), -s);
        assert_eq!(get(&tx, 2, 2), c);
        let ty = transform_y(a);
        assert_eq!(get(&ty, 0, 0), c);
        assert_eq!(get(&ty, 0, 2), -s);
        assert_eq!(get(&ty, 2, 0), s);
        assert_eq!(get(&ty, 2, 2), c);
        let tz = transform_z(a);
        assert_eq!(get(&tz, 0, 0), c);
        assert_eq!(get(&tz, 0, 1), s);
        assert_eq!(get(&tz, 1, 0), -s);
        assert_eq!(get(&tz, 1, 1), c);
    }

    #[test]
    fn inverse_angle_round_trip_is_identity() {
        let a: Real = 0.7;
        for (m, mi) in [
            (transform_x(a), transform_x(-a)),
            (transform_y(a), transform_y(-a)),
            (transform_z(a), transform_z(-a)),
        ] {
            let p = m * mi;
            for row in 0..3 {
                for col in 0..3 {
                    let want = if row == col { 1.0 } else { 0.0 };
                    assert!((get(&p, row, col) - want).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn tilde_matrix_matches_reference_entries_and_cross_product() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let m = tilde_operator(v);
        // Reference rows: [[0,-3,2],[3,0,-1],[-2,1,0]].
        assert_eq!(get(&m, 0, 0), 0.0);
        assert_eq!(get(&m, 0, 1), -3.0);
        assert_eq!(get(&m, 0, 2), 2.0);
        assert_eq!(get(&m, 1, 0), 3.0);
        assert_eq!(get(&m, 1, 1), 0.0);
        assert_eq!(get(&m, 1, 2), -1.0);
        assert_eq!(get(&m, 2, 0), -2.0);
        assert_eq!(get(&m, 2, 1), 1.0);
        assert_eq!(get(&m, 2, 2), 0.0);
        let w = Vector3::new(-0.5, 0.25, 4.0);
        let cross = m * w;
        assert_eq!(cross.x, v.y * w.z - v.z * w.y);
        assert_eq!(cross.y, v.z * w.x - v.x * w.z);
        assert_eq!(cross.z, v.x * w.y - v.y * w.x);
    }

    #[test]
    fn axis_angle_about_z_at_quarter_turn_is_transposed_rotation() {
        // s = -sin(pi/2) = -1 flips the off-diagonal signs relative to the
        // textbook form: the result is Rz(-pi/2), the parent-to-body map.
        let m = body_t_parent_from_axis_angle(Vector3::Z, FRAC_PI_2);
        assert!(get(&m, 0, 0).abs() < 1e-6);
        assert!((get(&m, 0, 1) - 1.0).abs() < 1e-6);
        assert!((get(&m, 1, 0) + 1.0).abs() < 1e-6);
        assert!(get(&m, 1, 1).abs() < 1e-6);
        // The z axis is the rotation axis: the diagonal entry is 1 up to the
        // cos(pi/2) tail folded into the f32 sum (never assert it exact).
        assert!((get(&m, 2, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn axis_angle_is_orthonormal_with_unit_determinant() {
        let axis = Vector3::new(1.0, 2.0, -0.5).normalize();
        let angle: Real = 1.234;
        let m = body_t_parent_from_axis_angle(axis, angle);
        let p = m * m.transpose();
        for row in 0..3 {
            for col in 0..3 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!((get(&p, row, col) - want).abs() < 1e-5);
            }
        }
        assert!((determinant(&m) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn determinant_matches_reference_expansion() {
        assert_eq!(determinant(&transform_z(0.0)), 1.0);
        let m = Matrix3::from_cols(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 3.0),
        );
        assert_eq!(determinant(&m), 6.0);
    }
}
