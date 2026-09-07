//! Shared joint math helpers used by several joint kinds (angle normalization,
// Euler extraction, motor fade, mirrored row limits). All `pub(crate)`.

use super::*;

/// The hinge angle between two world frames (Bullet's `getHingeAngle` with
/// `m_referenceSign = +1`): the rotation of frame B's Y axis around frame A's
/// Z axis.
pub(crate) fn get_hinge_angle(tr_a: &Matrix3, tr_b: &Matrix3) -> Real {
    let ref_axis0 = tr_a.col(0);
    let ref_axis1 = tr_a.col(1);
    let swing_axis = tr_b.col(1);
    math::atan2(swing_axis.dot(ref_axis0), swing_axis.dot(ref_axis1))
}

/// Wrap an angle into `[-π, π]` (Bullet's `btNormalizeAngle`).
pub(crate) fn normalize_angle(angle: Real) -> Real {
    let two_pi = 2.0 * bullet_math::PI;
    let mut a = angle % two_pi; // truncating, like `btFmod`.
    if a < -bullet_math::PI {
        a += two_pi;
    } else if a > bullet_math::PI {
        a -= two_pi;
    }
    a
}

/// Return `angle` adjusted to the range `[lo, hi]` by adding/subtracting
/// `2π` so it is closest to one of the limits (Bullet's
/// `btAdjustAngleToLimits`); all arguments are expected normalized. When
/// `lo >= hi` (a free or locked axis) the angle is returned unchanged.
pub(crate) fn adjust_angle_to_limits(angle: Real, lo: Real, hi: Real) -> Real {
    let two_pi = 2.0 * bullet_math::PI;
    if lo >= hi {
        return angle;
    }
    if angle < lo {
        let diff_lo = normalize_angle(lo - angle).abs();
        let diff_hi = normalize_angle(hi - angle).abs();
        if diff_lo < diff_hi {
            angle
        } else {
            angle + two_pi
        }
    } else if angle > hi {
        let diff_hi = normalize_angle(angle - hi).abs();
        let diff_lo = normalize_angle(angle - lo).abs();
        if diff_lo < diff_hi {
            angle - two_pi
        } else {
            angle
        }
    } else {
        angle
    }
}

/// Bullet's `matrixToEulerXYZ`: the Euler angles `(x, y, z)` of a rotation
/// matrix, extracted with `btGetMatrixElem(mat, index) = mat[index % 3]
/// [index / 3]` (row `index % 3`, column `index / 3` in Bullet's row-major
/// storage). glam stores columns, so Bullet's `mat[i][j]` is
/// `m.col(j)[i]`; the extraction is ported verbatim (the signs are
/// consistent with the constraint rows built from these angles — do not
/// "fix" them). Also used by the multibody fixed/slider constraints.
pub(crate) fn euler_xyz(m: &Matrix3) -> Vector3 {
    // Bullet's `mat[2][0]`, `mat[2][1]`, `mat[1][0]`, ... in glam's column
    // storage (row i, column j = `m.col(j)[i]`).
    let r20 = m.col(0).z; // mat[2][0]
    let r21 = m.col(1).z; // mat[2][1]
    let r22 = m.col(2).z; // mat[2][2]
    let r10 = m.col(0).y; // mat[1][0]
    let r11 = m.col(1).y; // mat[1][1]
    let r01 = m.col(1).x; // mat[0][1]
    let r00 = m.col(0).x; // mat[0][0]
    if r20 < 1.0 {
        if r20 > -1.0 {
            Vector3::new(math::atan2(-r21, r22), math::asin(r20), math::atan2(-r10, r00))
        } else {
            Vector3::new(-math::atan2(r01, r11), -bullet_math::FRAC_PI_2, 0.0)
        }
    } else {
        Vector3::new(math::atan2(r01, r11), bullet_math::FRAC_PI_2, 0.0)
    }
}

/// The relative-orientation information Bullet's `calculateAngleInfo`
/// computes for the generic 6DOF family: the Euler XYZ angles of the
/// relative rotation `trA⁻¹ · trB` and the three normalized angular
/// constraint axes (`m_calculatedAxis`), each of which constrains the
/// corresponding Euler-angle rate.
pub(crate) fn relative_rotation_info(tr_a: &Matrix3, tr_b: &Matrix3) -> (Vector3, [Vector3; 3]) {
    let relative = tr_a.transpose() * *tr_b;
    let euler = euler_xyz(&relative);
    // Bullet: `axis0 = trB.col(0)`, `axis2 = trA.col(2)`, then
    // `ax[1] = axis2×axis0`, `ax[0] = ax[1]×axis2`, `ax[2] = axis0×ax[1]`.
    let axis0 = tr_b.col(0);
    let axis2 = tr_a.col(2);
    let mut axes = [Vector3::ZERO; 3];
    axes[1] = axis2.cross(axis0);
    axes[0] = axes[1].cross(axis2);
    axes[2] = axis0.cross(axes[1]);
    for a in axes.iter_mut() {
        *a = a.normalize();
    }
    (euler, axes)
}

/// Mirror Bullet's accumulated-impulse window `[lower, upper]` (infinities
/// included) through our row convention, where our accumulated impulse is
/// the negative of Bullet's (`λ_ours = −λ_bullet`), so a window `[L, U]`
/// becomes `[−U, −L]`.
pub(crate) fn mirrored_row_limits(lower: Real, upper: Real) -> RowLimits {
    let lower_inf = lower.is_infinite() && lower < 0.0;
    let upper_inf = upper.is_infinite() && upper > 0.0;
    match (lower_inf, upper_inf) {
        (true, true) => RowLimits::Unbounded,
        // Bullet `[−inf, 0]` (a low limit) → our `[0, +inf]`.
        (true, false) => RowLimits::LowerBounded(0.0),
        // Bullet `[0, +inf]` (a high limit) → our `[−inf, 0]`.
        (false, true) => RowLimits::UpperBounded(0.0),
        // A symmetric window (motor): unchanged, since `[−c, c]` mirrors to
        // itself.
        (false, false) => RowLimits::Bilateral((upper - lower) / 2.0),
    }
}

pub(crate) fn get_motor_factor(pos: Real, low_lim: Real, upp_lim: Real, vel: Real, time_fact: Real) -> Real {
    if low_lim > upp_lim {
        return 1.0;
    }
    if low_lim == upp_lim {
        return 0.0;
    }
    let delta_max = vel / time_fact;
    if delta_max < 0.0 {
        if pos >= low_lim && pos < low_lim - delta_max {
            (low_lim - pos) / delta_max
        } else if pos < low_lim {
            0.0
        } else {
            1.0
        }
    } else if delta_max > 0.0 {
        if pos <= upp_lim && pos > upp_lim - delta_max {
            (upp_lim - pos) / delta_max
        } else if pos > upp_lim {
            0.0
        } else {
            1.0
        }
    } else {
        0.0
    }
}

