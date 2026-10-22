//! Math helpers for the inverse-dynamics tree — port of
//! `BulletInverseDynamics/IDMath.{hpp,cpp}`.
//!
//! Matrix conventions follow the reference implementation verbatim: the
//! `transform_*` builders and [`body_t_parent_from_axis_angle`] produce
//! *transposed* (parent-to-body) rotations, which is why the ported formulas
//! look unusual next to textbook rotation matrices (e.g. the negated sine in
//! the axis-angle form, `IDMath.cpp:258-279`). Do not "fix" them.

use bullet_math::{math, Matrix3, Real, Vector3};

/// Upstream `kIsZero`: `5 * epsilon` at the active precision
/// (`IDMath.cpp:8`).
pub(crate) const K_IS_ZERO: Real = 5.0 * Real::EPSILON;

/// Upstream `kAxisLengthEpsilon`: `10 * kIsZero` (`IDMath.cpp:11`).
pub(crate) const K_AXIS_LENGTH_EPSILON: Real = 10.0 * K_IS_ZERO;

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

/// `|‖v‖² − 1| < kIsZero` (`IDMath.cpp:496-499`).
pub(crate) fn is_unit_vector(v: Vector3) -> bool {
    (v.x * v.x + v.y * v.y + v.z * v.z - 1.0).abs() < K_IS_ZERO
}

/// Fuzzy positive-semi-definiteness on the leading principal minors with
/// `kIsZero` slack (`IDMath.cpp:321-339`).
pub(crate) fn is_positive_semi_definite_fuzzy(m: &Matrix3) -> bool {
    if m.x_axis.x < -K_IS_ZERO {
        return false;
    }
    if m.x_axis.x * m.y_axis.y - m.y_axis.x * m.x_axis.y < -K_IS_ZERO {
        return false;
    }
    if (m.x_axis.x * (m.y_axis.y * m.z_axis.z - m.y_axis.z * m.z_axis.y)
        - m.y_axis.x * (m.x_axis.y * m.z_axis.z - m.y_axis.z * m.x_axis.z)
        + m.z_axis.x * (m.x_axis.y * m.z_axis.y - m.y_axis.y * m.x_axis.z))
        < -K_IS_ZERO
    {
        return false;
    }
    true
}

/// Upstream inertia gates (`IDMath.cpp:347-443`): fuzzy PSD, the triangle
/// inequality — skipped for bodies joined by a fixed joint, and preserving
/// the reference implementation's duplicated `I00 + I11 < I22` branch
/// (`:385-396`; `I00 + I22 < I11` is never checked) — non-negative diagonal,
/// and symmetry within `kIsZero`.
pub(crate) fn is_valid_inertia_matrix(i: &Matrix3, has_fixed_joint: bool) -> bool {
    if !is_positive_semi_definite_fuzzy(i) {
        return false;
    }
    if !has_fixed_joint {
        if i.x_axis.x + i.y_axis.y < i.z_axis.z {
            return false;
        }
        if i.x_axis.x + i.y_axis.y < i.z_axis.z {
            return false;
        }
        if i.y_axis.y + i.z_axis.z < i.x_axis.x {
            return false;
        }
    }
    for diag in [i.x_axis.x, i.y_axis.y, i.z_axis.z] {
        if diag < 0.0 {
            return false;
        }
    }
    if (i.y_axis.x - i.x_axis.y).abs() > K_IS_ZERO {
        return false;
    }
    if (i.z_axis.x - i.x_axis.z).abs() > K_IS_ZERO {
        return false;
    }
    if (i.z_axis.y - i.y_axis.z).abs() > K_IS_ZERO {
        return false;
    }
    true
}

/// Rotation-matrix gates (`IDMath.cpp:445-494`): unit-length columns within
/// `kAxisLengthEpsilon`, pairwise-orthogonal columns, strict positive
/// determinant.
pub(crate) fn is_valid_transform_matrix(m: &Matrix3) -> bool {
    for col in [m.x_axis, m.y_axis, m.z_axis] {
        let length_minus_1 = (col.x * col.x + col.y * col.y + col.z * col.z - 1.0).abs();
        if length_minus_1 > K_AXIS_LENGTH_EPSILON {
            return false;
        }
    }
    if (m.x_axis.x * m.y_axis.x + m.x_axis.y * m.y_axis.y + m.x_axis.z * m.y_axis.z).abs()
        > K_AXIS_LENGTH_EPSILON
    {
        return false;
    }
    if (m.x_axis.x * m.z_axis.x + m.x_axis.y * m.z_axis.y + m.x_axis.z * m.z_axis.z).abs()
        > K_AXIS_LENGTH_EPSILON
    {
        return false;
    }
    if (m.y_axis.x * m.z_axis.x + m.y_axis.y * m.z_axis.y + m.y_axis.z * m.z_axis.z).abs()
        > K_AXIS_LENGTH_EPSILON
    {
        return false;
    }
    determinant(m) > 0.0
}

#[cfg(test)]
mod tests {
    use super::{
        body_t_parent_from_axis_angle, determinant, is_positive_semi_definite_fuzzy,
        is_unit_vector, is_valid_inertia_matrix, is_valid_transform_matrix, tilde_operator,
        transform_x, transform_y, transform_z, K_AXIS_LENGTH_EPSILON, K_IS_ZERO,
    };
    use bullet_math::{FRAC_PI_2, Matrix3, Real, Vector3};

    fn get(m: &Matrix3, row: usize, col: usize) -> Real {
        let c = m.col(col);
        [c.x, c.y, c.z][row]
    }

    /// Row-major literal to glam column-major.
    fn m(rows: [[Real; 3]; 3]) -> Matrix3 {
        Matrix3::from_cols(
            Vector3::new(rows[0][0], rows[1][0], rows[2][0]),
            Vector3::new(rows[0][1], rows[1][1], rows[2][1]),
            Vector3::new(rows[0][2], rows[1][2], rows[2][2]),
        )
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

    #[test]
    fn tolerance_constants_follow_the_reference_ratios() {
        assert_eq!(K_IS_ZERO, 5.0 * Real::EPSILON);
        assert_eq!(K_AXIS_LENGTH_EPSILON, 10.0 * K_IS_ZERO);
    }

    #[test]
    fn unit_vector_gate() {
        assert!(is_unit_vector(Vector3::X));
        assert!(is_unit_vector(Vector3::new(0.6, 0.8, 0.0)));
        // Just outside kIsZero (5e) is rejected.
        assert!(!is_unit_vector(Vector3::new(1.0 + 1e-5, 0.0, 0.0)));
    }

    #[test]
    fn psd_fuzzy_gate() {
        // Diagonal non-negative: PSD.
        assert!(is_positive_semi_definite_fuzzy(&m([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])));
        // First minor negative beyond slack: rejected.
        assert!(!is_positive_semi_definite_fuzzy(&m([
            [-1e-4, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])));
        // Zero matrix: PSD within slack.
        assert!(is_positive_semi_definite_fuzzy(&m([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])));
    }

    #[test]
    fn inertia_gate_preserves_the_duplicated_triangle_branch() {
        // Triangle inequality violated through I(0,0)+I(1,1) < I(2,2):
        // rejected.
        assert!(!is_valid_inertia_matrix(
            &m([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]),
            false
        ));
        // I(0,0)+I(2,2) < I(1,1) is never checked upstream (the duplicated
        // branch occupies that slot) — accepted.
        assert!(is_valid_inertia_matrix(
            &m([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]),
            false
        ));
        // Same shape passes for a fixed-joint body (checks skipped).
        assert!(is_valid_inertia_matrix(
            &m([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]),
            true
        ));
        // Asymmetric entry beyond kIsZero: rejected.
        assert!(!is_valid_inertia_matrix(
            &m([[2.0, 0.0, 0.0], [1e-3, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            false
        ));
        // Negative diagonal: rejected.
        assert!(!is_valid_inertia_matrix(
            &m([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            false
        ));
    }

    #[test]
    fn transform_gate() {
        // A proper rotation from the axis-angle builder passes.
        let r = body_t_parent_from_axis_angle(Vector3::new(1.0, 2.0, -0.5).normalize(), 0.9);
        assert!(is_valid_transform_matrix(&r));
        // Scaled column (non-unit): rejected.
        assert!(!is_valid_transform_matrix(&m([
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])));
        // Reflection (det < 0): rejected.
        assert!(!is_valid_transform_matrix(&m([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])));
        // Degenerate (non-orthogonal, zero) columns: rejected.
        assert!(!is_valid_transform_matrix(&m([
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ])));
    }
}
