use bullet_math::{Aabb, Quaternion, Real, Vector3};

/// A solid cylinder collision shape with its axis along Y.
/// Equivalent to `btCylinderShape`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub radius: Real,
    pub half_height: Real,
}

impl Cylinder {
    #[inline]
    pub fn new(radius: Real, half_height: Real) -> Self {
        Self {
            radius,
            half_height,
        }
    }

    /// Bullet's default convex distance margin (informational only, not baked
    /// into the geometry, matching the box in this MVP).
    #[inline]
    pub fn margin(&self) -> Real {
        0.04
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        let r = Vector3::new(self.radius, self.half_height, self.radius);
        Aabb::new(-r, r)
    }

    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let xy_len = dir.x * dir.x + dir.z * dir.z;
        let side = if xy_len > 0.0 {
            let s = self.radius / xy_len.sqrt();
            Vector3::new(dir.x * s, 0.0, dir.z * s)
        } else {
            Vector3::new(self.radius, 0.0, 0.0)
        };
        let y = if dir.y >= 0.0 {
            self.half_height
        } else {
            -self.half_height
        };
        Vector3::new(side.x, y, side.z)
    }

    /// Rotated-cylinder AABB via support projections along the world axes
    /// (exact: the projected half-width is `r * |u.xz| + hh * |u.y|`).
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        crate::shapes::support_projected_aabb(|u| self.support_local(u), position, orientation)
    }

    /// Analytical inertia of a solid cylinder about the origin:
    /// `(3r^2 + h^2)/12` transverse and `r^2/2` axial.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        let r2 = self.radius * self.radius;
        let h = 2.0 * self.half_height;
        let i_xx = mass * (3.0 * r2 + h * h) / 12.0;
        let i_yy = mass * r2 / 2.0;
        Vector3::new(i_xx, i_yy, i_xx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn support_on_the_disc_rim_and_cap() {
        let c = Cylinder::new(2.0, 1.0);
        let p = c.support_local(Vector3::new(1.0, 1.0, -1.0).normalize());
        assert_relative_eq!(p.x, 2.0 * bullet_math::FRAC_1_SQRT_2);
        assert_relative_eq!(p.y, 1.0);
        assert_relative_eq!(p.z, -2.0 * bullet_math::FRAC_1_SQRT_2);
        // Axial direction only: any point of the cap is a valid support.
        let top = c.support_local(Vector3::Y);
        assert_relative_eq!(top.y, 1.0);
    }

    #[test]
    fn inertia_matches_analytical() {
        let c = Cylinder::new(1.0, 1.0);
        let i = c.local_inertia(6.0);
        assert_relative_eq!(i.x, 6.0 * (3.0 + 4.0) / 12.0);
        assert_relative_eq!(i.y, 6.0 * 0.5);
        assert_relative_eq!(i.z, i.x);
    }

    #[test]
    fn rotated_aabb_is_exact() {
        let c = Cylinder::new(1.0, 1.0);
        // Rotate 90 deg about Z: the axis lies along X; the cross-section
        // radius then contributes to Y, so the Y half-width stays 1.
        let q = Quaternion::from_rotation_z(bullet_math::FRAC_PI_2);
        let a = c.world_aabb(Vector3::ZERO, q);
        assert_relative_eq!(a.min.x, -1.0);
        assert_relative_eq!(a.max.x, 1.0);
        assert_relative_eq!(a.min.y, -1.0);
        assert_relative_eq!(a.max.y, 1.0);
        // Rotate 45 deg: X half-width = r*cos45 + hh*sin45.
        let q45 = Quaternion::from_rotation_z(bullet_math::FRAC_PI_2 * 0.5);
        let a45 = c.world_aabb(Vector3::ZERO, q45);
        assert_relative_eq!(
            a45.max.x,
            bullet_math::FRAC_1_SQRT_2 + bullet_math::FRAC_1_SQRT_2
        );
    }
}
