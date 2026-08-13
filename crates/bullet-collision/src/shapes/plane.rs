use bullet_math::{Aabb, Real, Vector3};

/// An infinite static plane collision shape: `normal . x = constant` in local
/// space. Equivalent to `btStaticPlaneShape`.
///
/// Planes are not convex and cannot be used with GJK; they participate in the
/// narrow phase through a dedicated plane-vs-convex test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// Unit-length plane normal (normalized on construction).
    pub normal: Vector3,
    /// Distance of the plane from the origin along `normal`.
    pub constant: Real,
}

impl Plane {
    #[inline]
    pub fn new(normal: Vector3, constant: Real) -> Self {
        Self {
            normal: normal.normalize(),
            constant,
        }
    }

    #[inline]
    pub fn margin(&self) -> Real {
        0.0
    }

    /// Bullet's `BTLARGE`: the plane's AABB is unbounded.
    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        let b = Vector3::splat(1e30);
        Aabb::new(-b, b)
    }

    #[inline]
    pub fn world_aabb(&self, _position: Vector3, _orientation: bullet_math::Quaternion) -> Aabb {
        self.local_aabb()
    }

    #[inline]
    pub fn local_inertia(&self, _mass: Real) -> Vector3 {
        Vector3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn normal_is_normalized() {
        let p = Plane::new(Vector3::new(0.0, 2.0, 0.0), 5.0);
        assert_relative_eq!(p.normal.y, 1.0);
        assert_relative_eq!(p.constant, 5.0);
    }

    #[test]
    fn aabb_is_unbounded() {
        let p = Plane::new(Vector3::Y, 0.0);
        let a = p.local_aabb();
        assert!(a.min.x < -1e9);
        assert!(a.max.x > 1e9);
    }
}
