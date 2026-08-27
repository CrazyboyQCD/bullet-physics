use bullet_math::{Aabb, Real, Vector3};

/// A sphere collision shape. Equivalent to `btSphereShape`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub radius: Real,
}

impl Sphere {
    #[inline]
    pub fn new(radius: Real) -> Self {
        Self { radius }
    }

    /// The sphere's margin is its full radius (Bullet treats the radius as
    /// an implicit margin).
    #[inline]
    pub fn margin(&self) -> Real {
        self.radius
    }

    /// Core support (the whole radius is the margin): a point at the origin.
    #[inline]
    pub fn core_support_local(&self, _dir: Vector3) -> Vector3 {
        Vector3::ZERO
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        let r = Vector3::splat(self.radius);
        Aabb::new(-r, r)
    }

    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let len = dir.length();
        if len > 0.0 {
            dir / len * self.radius
        } else {
            Vector3::new(self.radius, 0.0, 0.0)
        }
    }

    #[inline]
    pub fn world_aabb(&self, position: Vector3) -> Aabb {
        let r = Vector3::splat(self.radius);
        Aabb::new(position - r, position + r)
    }

    /// Uniform inertia: `2/5 m r^2` on each axis.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        Vector3::splat(0.4 * mass * self.radius * self.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn support_points_towards_direction() {
        let s = Sphere::new(2.0);
        let p = s.support_local(Vector3::X);
        assert_relative_eq!(p.x, 2.0);
        assert_relative_eq!(p.y, 0.0);
    }

    #[test]
    fn inertia_matches_analytical() {
        let s = Sphere::new(1.0);
        let i = s.local_inertia(5.0);
        assert_relative_eq!(i.x, 2.0);
        assert_relative_eq!(i.y, 2.0);
        assert_relative_eq!(i.z, 2.0);
    }
}
