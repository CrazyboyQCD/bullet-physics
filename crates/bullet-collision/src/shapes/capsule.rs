use bullet_math::{Aabb, Quaternion, Real, Vector3};
use bullet_math::PI;

/// A capsule collision shape: a sphere-swept segment along the Y axis.
/// Equivalent to `btCapsuleShape`.
///
/// The segment runs from `-half_height` to `+half_height` along Y and is
/// inflated by `radius` (the total length is `2 * (half_height + radius)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    pub radius: Real,
    pub half_height: Real,
}

impl Capsule {
    #[inline]
    pub fn new(radius: Real, half_height: Real) -> Self {
        Self {
            radius,
            half_height,
        }
    }

    /// The capsule's margin is its radius (Bullet treats the radius as the
    /// implicit margin of the swept segment).
    #[inline]
    pub fn margin(&self) -> Real {
        self.radius
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        let r = Vector3::new(self.radius, self.half_height + self.radius, self.radius);
        Aabb::new(-r, r)
    }

    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let len = dir.length();
        if len > 0.0 {
            let tip = Vector3::new(
                0.0,
                if dir.y >= 0.0 {
                    self.half_height
                } else {
                    -self.half_height
                },
                0.0,
            );
            tip + dir / len * self.radius
        } else {
            Vector3::new(self.radius, self.half_height, 0.0)
        }
    }

    /// Rotated-capsule AABB: inflate the transformed segment endpoints by
    /// `radius` (exact for a sphere-swept segment).
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        let top = orientation.mul_vec3(Vector3::new(0.0, self.half_height, 0.0)) + position;
        let bottom = orientation.mul_vec3(Vector3::new(0.0, -self.half_height, 0.0)) + position;
        let r = Vector3::splat(self.radius);
        Aabb::new(top.min(bottom) - r, top.max(bottom) + r)
    }

    /// Analytical inertia of a solid capsule (cylinder plus two hemispheres),
    /// about the segment center, mass-weighted between the parts.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        let h = 2.0 * self.half_height;
        let r2 = self.radius * self.radius;
        let h2 = h * h;
        let v_cyl = PI * r2 * h;
        let v_sph = 4.0 / 3.0 * PI * r2 * self.radius;
        let v = v_cyl + v_sph;
        let m_cyl = mass * v_cyl / v;
        let m_sph = mass * v_sph / v;
        // Cylinder about the origin: (3r^2 + h^2)/12 transverse, r^2/2 axial.
        // The hemisphere pair (a full sphere) is offset by h/2 along Y.
        let i_xx = m_cyl * (3.0 * r2 + h2) / 12.0 + m_sph * (0.4 * r2 + 0.25 * h2);
        let i_yy = m_cyl * r2 / 2.0 + m_sph * 0.4 * r2;
        Vector3::new(i_xx, i_yy, i_xx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn support_points_to_the_segment_tip_plus_radius() {
        let c = Capsule::new(0.5, 1.0);
        let p = c.support_local(Vector3::new(1.0, 1.0, 0.0).normalize());
        assert_relative_eq!(p.x, 0.5 * bullet_math::FRAC_1_SQRT_2);
        assert_relative_eq!(p.y, 1.0 + 0.5 * bullet_math::FRAC_1_SQRT_2);
        assert_relative_eq!(p.z, 0.0);
    }

    #[test]
    fn inertia_matches_analytical() {
        let c = Capsule::new(1.0, 1.0);
        let i = c.local_inertia(1.0);
        // Volumes: cylinder 2*pi, sphere 4/3*pi -> mass split 0.6 / 0.4.
        assert_relative_eq!(i.x, 0.6 * 7.0 / 12.0 + 0.4 * (0.4 + 1.0));
        assert_relative_eq!(i.y, 0.6 * 0.5 + 0.4 * 0.4);
        assert_relative_eq!(i.z, i.x);
    }

    #[test]
    fn rotated_aabb_swings_the_axis() {
        let c = Capsule::new(0.5, 1.0);
        // Rotate 90 deg about Z: the segment now lies along X.
        let q = Quaternion::from_rotation_z(bullet_math::FRAC_PI_2);
        let a = c.world_aabb(Vector3::ZERO, q);
        assert_relative_eq!(a.min.x, -1.5);
        assert_relative_eq!(a.max.x, 1.5);
        assert_relative_eq!(a.min.y, -0.5);
        assert_relative_eq!(a.max.y, 0.5);
    }
}
