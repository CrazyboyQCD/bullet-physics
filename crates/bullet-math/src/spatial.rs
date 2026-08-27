//! Spatial (6-D) vector algebra used by the Featherstone articulated-body
//! algorithm — port of Bullet's `LinearMath/btSpatialAlgebra.h`.
//!
//! A spatial vector packs an angular and a linear 3-vector into one 6-D
//! quantity. Two flavors exist:
//!
//! - [`SpatialMotionVector`]: a motion (twist) — `top = angular`, `bottom =
//!   linear` velocity/acceleration (Bullet's `m_topVec = angular`,
//!   `m_bottomVec = linear`).
//! - [`SpatialForceVector`]: a wrench — `top = linear` force, `bottom =
//!   angular` torque (Bullet's `m_topVec = linear`, `m_bottomVec = angular`).
//!
//! The dot product of a motion and a force vector is the power, which is
//! invariant under spatial transforms.
//!
//! Field naming follows Bullet's source (`top`/`bottom`), but the explicit
//! `angular`/`linear` accessors make the meaning unambiguous.

use super::{Matrix3, Real, Vector3};

/// A spatial motion vector: `top = angular`, `bottom = linear` (a twist).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialMotionVector {
    /// Angular part (Bullet's `m_topVec`).
    pub top: Vector3,
    /// Linear part (Bullet's `m_bottomVec`).
    pub bottom: Vector3,
}

impl SpatialMotionVector {
    #[inline]
    pub fn new(angular: Vector3, linear: Vector3) -> Self {
        Self {
            top: angular,
            bottom: linear,
        }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::default()
    }

    #[inline]
    pub fn angular(&self) -> Vector3 {
        self.top
    }

    #[inline]
    pub fn linear(&self) -> Vector3 {
        self.bottom
    }

    #[inline]
    pub fn set_angular(&mut self, v: Vector3) {
        self.top = v;
    }

    #[inline]
    pub fn set_linear(&mut self, v: Vector3) {
        self.bottom = v;
    }

    /// Power with a wrench: `m_bottom·f_top + m_top·f_bottom` (Bullet's
    /// `dot(const btSpatialForceVector&)`).
    #[inline]
    pub fn dot_force(&self, b: &SpatialForceVector) -> Real {
        self.bottom.dot(b.top) + self.top.dot(b.bottom)
    }

    /// Spatial cross product (Mirtich's `×`), result has the same flavor as
    /// `b`: `top' = a.top × b.top`, `bottom' = a.bottom × b.top + a.top ×
    /// b.bottom` (Bullet's templated `cross`).
    #[inline]
    pub fn cross(&self, b: &SpatialMotionVector) -> SpatialMotionVector {
        SpatialMotionVector {
            top: self.top.cross(b.top),
            bottom: self.bottom.cross(b.top) + self.top.cross(b.bottom),
        }
    }

    #[inline]
    pub fn scale(&self, s: Real) -> Self {
        Self {
            top: self.top * s,
            bottom: self.bottom * s,
        }
    }
}

impl std::ops::Add for SpatialMotionVector {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            top: self.top + rhs.top,
            bottom: self.bottom + rhs.bottom,
        }
    }
}

impl std::ops::AddAssign for SpatialMotionVector {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.top += rhs.top;
        self.bottom += rhs.bottom;
    }
}

impl std::ops::Sub for SpatialMotionVector {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            top: self.top - rhs.top,
            bottom: self.bottom - rhs.bottom,
        }
    }
}

impl std::ops::Neg for SpatialMotionVector {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            top: -self.top,
            bottom: -self.bottom,
        }
    }
}

impl std::ops::Mul<Real> for SpatialMotionVector {
    type Output = Self;
    #[inline]
    fn mul(self, s: Real) -> Self {
        self.scale(s)
    }
}

impl std::ops::Mul<SpatialMotionVector> for Real {
    type Output = SpatialMotionVector;
    #[inline]
    fn mul(self, v: SpatialMotionVector) -> SpatialMotionVector {
        v.scale(self)
    }
}

/// A spatial force vector (wrench): `top = linear`, `bottom = angular`
/// (Bullet's `btSpatialForceVector`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpatialForceVector {
    /// Linear part (Bullet's `m_topVec`).
    pub top: Vector3,
    /// Angular part (Bullet's `m_bottomVec`).
    pub bottom: Vector3,
}

impl SpatialForceVector {
    #[inline]
    pub fn new(linear: Vector3, angular: Vector3) -> Self {
        Self {
            top: linear,
            bottom: angular,
        }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::default()
    }

    #[inline]
    pub fn linear(&self) -> Vector3 {
        self.top
    }

    #[inline]
    pub fn angular(&self) -> Vector3 {
        self.bottom
    }

    #[inline]
    pub fn set_linear(&mut self, v: Vector3) {
        self.top = v;
    }

    #[inline]
    pub fn set_angular(&mut self, v: Vector3) {
        self.bottom = v;
    }

    #[inline]
    pub fn scale(&self, s: Real) -> Self {
        Self {
            top: self.top * s,
            bottom: self.bottom * s,
        }
    }
}

impl std::ops::Add for SpatialForceVector {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            top: self.top + rhs.top,
            bottom: self.bottom + rhs.bottom,
        }
    }
}

impl std::ops::AddAssign for SpatialForceVector {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.top += rhs.top;
        self.bottom += rhs.bottom;
    }
}

impl std::ops::Sub for SpatialForceVector {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            top: self.top - rhs.top,
            bottom: self.bottom - rhs.bottom,
        }
    }
}

impl std::ops::Neg for SpatialForceVector {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            top: -self.top,
            bottom: -self.bottom,
        }
    }
}

impl std::ops::Mul<Real> for SpatialForceVector {
    type Output = Self;
    #[inline]
    fn mul(self, s: Real) -> Self {
        self.scale(s)
    }
}

impl std::ops::Mul<SpatialForceVector> for Real {
    type Output = SpatialForceVector;
    #[inline]
    fn mul(self, v: SpatialForceVector) -> SpatialForceVector {
        v.scale(self)
    }
}

/// A symmetric spatial inertia dyad, stored as three 3x3 blocks (the
/// lower-right block is the transpose of the upper-left one, so it is not
/// stored). Port of `btSymmetricSpatialDyad`.
#[derive(Debug, Clone, Copy)]
pub struct SymmetricSpatialDyad {
    /// `I_topLeft` (Bullet's `m_topLeftMat`).
    pub top_left: Matrix3,
    /// `I_topRight` (Bullet's `m_topRightMat`).
    pub top_right: Matrix3,
    /// `I_bottomLeft` (Bullet's `m_bottomLeftMat`).
    pub bottom_left: Matrix3,
}

impl Default for SymmetricSpatialDyad {
    fn default() -> Self {
        Self {
            top_left: Matrix3::IDENTITY,
            top_right: Matrix3::IDENTITY,
            bottom_left: Matrix3::IDENTITY,
        }
    }
}

impl SymmetricSpatialDyad {
    #[inline]
    pub fn new(top_left: Matrix3, top_right: Matrix3, bottom_left: Matrix3) -> Self {
        Self {
            top_left,
            top_right,
            bottom_left,
        }
    }

    #[inline]
    pub fn zero() -> Self {
        Self {
            top_left: Matrix3::ZERO,
            top_right: Matrix3::ZERO,
            bottom_left: Matrix3::ZERO,
        }
    }

    /// `I · v` for a spatial motion vector `v` (Bullet's `operator*`):
    /// `angular' = bottomLeft·v.top + topLeftᵀ·v.bottom`,
    /// `linear' = topLeft·v.top + topRight·v.bottom` (force flavor, so
    /// `top = linear`, `bottom = angular`).
    #[inline]
    pub fn apply(&self, v: &SpatialMotionVector) -> SpatialForceVector {
        let angular = self.bottom_left * v.top + self.top_left.transpose() * v.bottom;
        let linear = self.top_left * v.top + self.top_right * v.bottom;
        SpatialForceVector {
            top: linear,
            bottom: angular,
        }
    }
}

impl std::ops::SubAssign for SymmetricSpatialDyad {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.top_left -= rhs.top_left;
        self.top_right -= rhs.top_right;
        self.bottom_left -= rhs.bottom_left;
    }
}

impl std::ops::AddAssign for SymmetricSpatialDyad {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.top_left += rhs.top_left;
        self.top_right += rhs.top_right;
        self.bottom_left += rhs.bottom_left;
    }
}

/// A spatial rigid transform (rotation + translation), port of
/// `btSpatialTransformationMatrix`. `rot` rotates vectors from the 'from'
/// frame into the 'to' frame; `trn` is the vector from the origin of the
/// 'from' frame to the origin of the 'to' frame, expressed in 'to'
/// coordinates.
#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformationMatrix {
    /// Rotation from the 'from' frame to the 'to' frame.
    pub rot: Matrix3,
    /// Origin offset expressed in the 'to' frame (Bullet's `m_trnVec`).
    pub trn: Vector3,
}

impl SpatialTransformationMatrix {
    /// Forward transform of a spatial vector (Bullet's `transform`):
    /// `top' = R·top`, `bottom' = −t × top' + R·bottom`.
    #[inline]
    pub fn transform(&self, v: &SpatialMotionVector) -> SpatialMotionVector {
        let top = self.rot * v.top;
        let bottom = -self.trn.cross(top) + self.rot * v.bottom;
        SpatialMotionVector { top, bottom }
    }

    /// Forward transform, rotation only (Bullet's `transformRotationOnly`).
    #[inline]
    pub fn transform_rotation_only(&self, v: &SpatialForceVector) -> SpatialForceVector {
        SpatialForceVector {
            top: self.rot * v.top,
            bottom: self.rot * v.bottom,
        }
    }

    /// Forward transform of a wrench, the dual of [`Self::transform`]:
    /// `top' = R·top`, `bottom' = R·bottom − t × (R·top)`. This is the exact
    /// inverse of [`Self::transform_inverse_force`] and preserves the
    /// power pairing `motion·force` under a coordinate change (Bullet never
    /// needs it in the ABA — forces only travel backwards through the tree —
    /// but it is the mathematically consistent counterpart).
    #[inline]
    pub fn transform_force(&self, v: &SpatialForceVector) -> SpatialForceVector {
        let top = self.rot * v.top;
        let bottom = self.rot * v.bottom - self.trn.cross(top);
        SpatialForceVector { top, bottom }
    }

    /// Inverse transform (Bullet's `transformInverse`):
    /// `top' = Rᵀ·top`, `bottom' = Rᵀ·(bottom + t × top)`.
    #[inline]
    pub fn transform_inverse(&self, v: &SpatialMotionVector) -> SpatialMotionVector {
        let top = self.rot.transpose() * v.top;
        let bottom = self.rot.transpose() * (v.bottom + self.trn.cross(v.top));
        SpatialMotionVector { top, bottom }
    }

    /// Inverse transform of a spatial force vector (same formula as the
    /// motion version; the cross-term keeps the power bilinear form intact).
    #[inline]
    pub fn transform_inverse_force(&self, v: &SpatialForceVector) -> SpatialForceVector {
        let top = self.rot.transpose() * v.top;
        let bottom = self.rot.transpose() * (v.bottom + self.trn.cross(v.top));
        SpatialForceVector { top, bottom }
    }

    /// `Xᵀ · I · X` congruence transform of a symmetric spatial dyad
    /// (Bullet's `transformInverse(const btSymmetricSpatialDyad&)`): the
    /// articulated-body inertia is shifted from the child frame back to the
    /// parent frame.
    #[inline]
    pub fn transform_inverse_dyad(&self, m: &SymmetricSpatialDyad) -> SymmetricSpatialDyad {
        let r_cross = skew(self.trn);
        let rt = self.rot.transpose();
        let top_left = rt * (m.top_left - m.top_right * r_cross) * self.rot;
        let top_right = rt * m.top_right * self.rot;
        let bottom_left = rt
            * (r_cross * (m.top_left - m.top_right * r_cross)
                + m.bottom_left
                - m.top_left.transpose() * r_cross)
            * self.rot;
        SymmetricSpatialDyad {
            top_left,
            top_right,
            bottom_left,
        }
    }
}

/// The skew-symmetric matrix `[t]×` such that `[t]× · v = t × v` (Bullet's
/// `r_cross` block in the dyad transform).
#[inline]
pub fn skew(t: Vector3) -> Matrix3 {
    Matrix3::from_cols(
        Vector3::new(0.0, t.z, -t.y),
        Vector3::new(-t.z, 0.0, t.x),
        Vector3::new(t.y, -t.x, 0.0),
    )
}

/// Outer product `v0 · v1ᵀ` (Bullet's `outerProduct`/`vecMulVecTranspose`).
#[inline]
pub fn outer_product(v0: Vector3, v1: Vector3) -> Matrix3 {
    Matrix3::from_cols(v0 * v1.x, v0 * v1.y, v0 * v1.z)
}

/// `a · bᵀ` symmetric spatial dyad built from the top/bottom parts of two
/// spatial vectors (Bullet's `symmetricSpatialOuterProduct`): `topLeft =
/// outer(a.top, b.bottom)`, `topRight = outer(a.top, b.top)`, `bottomLeft =
/// outer(a.bottom, b.bottom)`.
#[inline]
pub fn symmetric_spatial_outer_product(a: &SpatialMotionVector, b: &SpatialMotionVector) -> SymmetricSpatialDyad {
    SymmetricSpatialDyad {
        top_left: outer_product(a.top, b.bottom),
        top_right: outer_product(a.top, b.top),
        bottom_left: outer_product(a.bottom, b.bottom),
    }
}

/// Same outer product for two force-flavored vectors (`top = linear`,
/// `bottom = angular`), used with `h` (force) vectors in the ABA.
#[inline]
pub fn symmetric_spatial_outer_product_force(a: &SpatialForceVector, b: &SpatialForceVector) -> SymmetricSpatialDyad {
    SymmetricSpatialDyad {
        top_left: outer_product(a.top, b.bottom),
        top_right: outer_product(a.top, b.top),
        bottom_left: outer_product(a.bottom, b.bottom),
    }
}

/// `btMultiBody::spatialTransform`: transform a spatial vector from the
/// 'from' frame to the 'to' frame given the rotation matrix and the origin
/// displacement in 'to' coordinates.
#[inline]
pub fn spatial_transform(
    rotation_matrix: &Matrix3,
    displacement: Vector3,
    top_in: Vector3,
    bottom_in: Vector3,
) -> (Vector3, Vector3) {
    let top_out = rotation_matrix * top_in;
    let bottom_out = -displacement.cross(top_out) + rotation_matrix * bottom_in;
    (top_out, bottom_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::compat::matrix3_from_diagonal;

    #[test]
    fn transform_and_inverse_are_round_trips() {
        let rot = Matrix3::from_rotation_y(0.7);
        let trn = Vector3::new(0.3, -0.2, 0.1);
        let x = SpatialTransformationMatrix { rot, trn };
        let m = SpatialMotionVector::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
        let f = SpatialForceVector::new(Vector3::new(-1.0, 0.5, 2.0), Vector3::new(0.2, -3.0, 1.0));
        let m2 = x.transform(&m);
        let m3 = x.transform_inverse(&m2);
        assert!((m3.top - m.top).length() < 1e-5);
        assert!((m3.bottom - m.bottom).length() < 1e-5);
        // Wrench round trip: the dual forward transform then the inverse.
        let f2 = x.transform_force(&f);
        let f3 = x.transform_inverse_force(&f2);
        assert!((f3.top - f.top).length() < 1e-5);
        assert!((f3.bottom - f.bottom).length() < 1e-5);
    }

    #[test]
    fn dual_force_transform_preserves_power() {
        // Under the (forward motion, dual force) pairing the power is
        // invariant, unlike Bullet's inverse-only wrench transform.
        let rot = Matrix3::from_rotation_y(0.7);
        let trn = Vector3::new(0.3, -0.2, 0.1);
        let x = SpatialTransformationMatrix { rot, trn };
        let m = SpatialMotionVector::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0));
        let f = SpatialForceVector::new(Vector3::new(-1.0, 0.5, 2.0), Vector3::new(0.2, -3.0, 1.0));
        let before = m.dot_force(&f);
        let after = x.transform(&m).dot_force(&x.transform_force(&f));
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn dyad_apply_matches_definition() {
        // Inertia of a point mass at the origin: I·(ω,v) = (m·v, I·ω).
        let m = 2.0;
        let inertia = matrix3_from_diagonal(Vector3::new(3.0, 4.0, 5.0));
        let dyad = SymmetricSpatialDyad::new(
            Matrix3::ZERO,
            matrix3_from_diagonal(Vector3::splat(m)),
            inertia,
        );
        let v = SpatialMotionVector::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        let f = dyad.apply(&v);
        assert!((f.linear() - Vector3::new(0.0, 2.0, 0.0)).length() < 1e-6);
        assert!((f.angular() - Vector3::new(3.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn outer_product_matches_componentwise() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        let m = outer_product(a, b);
        // glam stores columns: col(j) = v0·v1[j].
        assert_eq!(m.col(0), a * 4.0);
        assert_eq!(m.col(1), a * 5.0);
        assert_eq!(m.col(2), a * 6.0);
        // (A·v)_i = v0_i · (v1·v)
        let v = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(m * v, a * 4.0);
    }

    #[test]
    fn skew_is_cross_product() {
        let t = Vector3::new(1.0, 2.0, 3.0);
        let v = Vector3::new(4.0, 5.0, 6.0);
        assert!((skew(t) * v - t.cross(v)).length() < 1e-6);
    }

    #[test]
    fn spatial_transform_matches_bullet_formula() {
        let rot = Matrix3::from_rotation_z(1.2);
        let disp = Vector3::new(0.1, 0.2, 0.3);
        let (top_out, bottom_out) = spatial_transform(&rot, disp, Vector3::ONE, Vector3::new(1.0, 2.0, 3.0));
        let top_expected = rot * Vector3::ONE;
        let bottom_expected = -disp.cross(top_expected) + rot * Vector3::new(1.0, 2.0, 3.0);
        assert!((top_out - top_expected).length() < 1e-6);
        assert!((bottom_out - bottom_expected).length() < 1e-6);
    }
}
