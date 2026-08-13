use bullet_math::{Aabb, Quaternion, Real, Vector3};

/// A right circular cone collision shape with its axis along Y: the apex at
/// `+half_height` and the base disc centered at `-half_height`.
/// Equivalent to `btConeShape`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    pub radius: Real,
    pub half_height: Real,
}

impl Cone {
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
        Aabb::new(
            Vector3::new(-self.radius, -self.half_height, -self.radius),
            Vector3::new(self.radius, self.half_height, self.radius),
        )
    }

    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let len = dir.length();
        if len > 0.0 {
            let d = dir / len;
            let apex = Vector3::new(0.0, self.half_height, 0.0);
            let xy_len = d.x * d.x + d.z * d.z;
            if xy_len > 0.0 {
                let s = self.radius / xy_len.sqrt();
                let rim = Vector3::new(d.x * s, -self.half_height, d.z * s);
                if rim.dot(d) > apex.dot(d) {
                    rim
                } else {
                    apex
                }
            } else if d.y >= 0.0 {
                // Along the axis: the apex wins for +Y, the base center for -Y.
                apex
            } else {
                Vector3::new(0.0, -self.half_height, 0.0)
            }
        } else {
            Vector3::new(self.radius, self.half_height, 0.0)
        }
    }

    /// Rotated-cone AABB via support projections along the world axes.
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        crate::shapes::support_projected_aabb(|u| self.support_local(u), position, orientation)
    }

    /// Analytical inertia of a solid cone about the shape origin (the
    /// geometric center): `(3r^2/20 + h^2/10)` transverse and `3r^2/10`
    /// axial, where `h = 2 * half_height`.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        let r2 = self.radius * self.radius;
        let h = 2.0 * self.half_height;
        let i_xx = mass * (3.0 * r2 / 20.0 + h * h / 10.0);
        let i_yy = mass * 3.0 * r2 / 10.0;
        Vector3::new(i_xx, i_yy, i_xx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn support_picks_apex_or_rim_whichever_is_farther() {
        let c = Cone::new(1.0, 1.0);
        // Mostly sideways: the rim wins.
        let p = c.support_local(Vector3::new(1.0, 0.1, 0.0).normalize());
        assert_relative_eq!(p.x, 1.0);
        assert_relative_eq!(p.y, -1.0);
        // Straight up: the apex wins.
        let top = c.support_local(Vector3::Y);
        assert_relative_eq!(top.y, 1.0);
        assert_relative_eq!(top.x, 0.0);
    }

    #[test]
    fn inertia_matches_analytical() {
        let c = Cone::new(1.0, 1.0);
        let i = c.local_inertia(1.0);
        assert_relative_eq!(i.x, 3.0 / 20.0 + 4.0 / 10.0);
        assert_relative_eq!(i.y, 3.0 / 10.0);
        assert_relative_eq!(i.z, i.x);
    }

    #[test]
    fn rotated_aabb_tracks_the_apex() {
        let c = Cone::new(1.0, 1.0);
        // Rotate 90 deg about Z: the apex lies along -X, base center +X.
        let q = Quaternion::from_rotation_z(bullet_math::FRAC_PI_2);
        let a = c.world_aabb(Vector3::ZERO, q);
        assert_relative_eq!(a.min.x, -1.0);
        assert_relative_eq!(a.max.x, 1.0);
        assert_relative_eq!(a.min.y, -1.0);
        assert_relative_eq!(a.max.y, 1.0);
    }
}
