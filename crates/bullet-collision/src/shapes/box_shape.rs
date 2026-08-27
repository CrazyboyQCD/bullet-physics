use bullet_math::{Aabb, Quaternion, Real, Vector3};

/// A box (cuboid) collision shape. Equivalent to `btBoxShape`.
///
/// Named `BoxShape` to avoid clashing with `std::boxed::Box`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxShape {
    pub half_extents: Vector3,
}

impl BoxShape {
    #[inline]
    pub fn new(half_extents: Vector3) -> Self {
        Self { half_extents }
    }

    /// Bullet's default convex distance margin. Note: in this MVP the margin is
    /// informational only and is not baked into the support/aabb geometry.
    #[inline]
    pub fn margin(&self) -> Real {
        0.04
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        Aabb::new(-self.half_extents, self.half_extents)
    }

    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let h = self.half_extents;
        Vector3::new(
            if dir.x >= 0.0 { h.x } else { -h.x },
            if dir.y >= 0.0 { h.y } else { -h.y },
            if dir.z >= 0.0 { h.z } else { -h.z },
        )
    }

    /// Rotated-box AABB from its 8 corners.
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        let h = self.half_extents;
        let mut min = Vector3::splat(Real::INFINITY);
        let mut max = Vector3::splat(Real::NEG_INFINITY);
        for i in 0..8 {
            let corner = Vector3::new(
                if i & 1 == 0 { h.x } else { -h.x },
                if i & 2 == 0 { h.y } else { -h.y },
                if i & 4 == 0 { h.z } else { -h.z },
            );
            let p = orientation * corner + position;
            min = min.min(p);
            max = max.max(p);
        }
        Aabb::new(min, max)
    }

    /// Diagonal inertia: `m/3 * (hy^2 + hz^2)` etc., over the full extents.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        let h = self.half_extents;
        let m = mass / 3.0;
        Vector3::new(
            m * (h.y * h.y + h.z * h.z),
            m * (h.x * h.x + h.z * h.z),
            m * (h.x * h.x + h.y * h.y),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn support_corners() {
        let b = BoxShape::new(Vector3::new(1.0, 2.0, 3.0));
        let p = b.support_local(Vector3::new(1.0, 1.0, -1.0));
        assert_relative_eq!(p.x, 1.0);
        assert_relative_eq!(p.y, 2.0);
        assert_relative_eq!(p.z, -3.0);
    }

    #[test]
    fn inertia_matches_analytical() {
        let b = BoxShape::new(Vector3::new(1.0, 2.0, 3.0));
        let i = b.local_inertia(12.0);
        // m/3 = 4
        assert_relative_eq!(i.x, 4.0 * (4.0 + 9.0));
        assert_relative_eq!(i.y, 4.0 * (1.0 + 9.0));
        assert_relative_eq!(i.z, 4.0 * (1.0 + 4.0));
    }
}
