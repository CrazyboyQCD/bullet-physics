use bullet_math::{Aabb, Quaternion, Real, Vector3};

/// A convex hull collision shape defined by a set of vertices.
/// Equivalent to `btConvexHullShape`.
///
/// The vertices must form a convex polytope (Bullet's contract as well); the
/// hull is not validated or rebuilt here. A hull centered on the origin gives
/// the expected inertia behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexHull {
    pub points: Vec<Vector3>,
    /// Cached local AABB over the vertices (hot: `world_aabb`, the inertia
    /// approximation and the contact-breaking threshold all read it).
    local_bounds: Aabb,
}

impl ConvexHull {
    #[inline]
    pub fn new(points: Vec<Vector3>) -> Self {
        let local_bounds = {
            let mut min = Vector3::splat(Real::INFINITY);
            let mut max = Vector3::splat(Real::NEG_INFINITY);
            for p in &points {
                min = min.min(*p);
                max = max.max(*p);
            }
            if points.is_empty() {
                min = Vector3::ZERO;
                max = Vector3::ZERO;
            }
            Aabb::new(min, max)
        };
        Self { points, local_bounds }
    }

    /// Hulls carry no margin: this engine keeps their geometry exact (the
    /// GJK/EPA core equals the full shape). Bullet's default 0.04 hull margin
    /// inflates the user's vertices into a rounded hull — a quirk we do not
    /// replicate (mesh triangles go through this path too).
    #[inline]
    pub fn margin(&self) -> Real {
        0.0
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        self.local_bounds
    }

    /// Support vertex by linear scan. Bullet keeps neighbor lists for O(log n)
    /// queries; a scan is fine while hulls stay small.
    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let mut best = Vector3::ZERO;
        let mut best_dot = Real::NEG_INFINITY;
        for p in &self.points {
            let d = p.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = *p;
            }
        }
        best
    }

    /// Rotated-hull AABB via support projections along the world axes.
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        crate::shapes::support_projected_aabb(|u| self.support_local(u), position, orientation)
    }

    /// AABB-box approximation of the inertia, matching Bullet's
    /// `btConvexHullShape::calculateLocalInertia` (an exact polyhedral tensor
    /// would require hull faces; deferred).
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        let half = self.local_aabb().extents();
        let m = mass / 3.0;
        Vector3::new(
            m * (half.y * half.y + half.z * half.z),
            m * (half.x * half.x + half.z * half.z),
            m * (half.x * half.x + half.y * half.y),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A unit octahedron (all 6 vertices on the axes).
    fn octahedron() -> ConvexHull {
        ConvexHull::new(vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ])
    }

    #[test]
    fn support_scans_for_the_farthest_vertex() {
        let h = octahedron();
        // (1, 0, 0) wins uniquely: its projection along (2, 1, 0) is 2.
        let p = h.support_local(Vector3::new(2.0, 1.0, 0.0));
        assert_relative_eq!(p.x, 1.0);
        assert_relative_eq!(p.y, 0.0);
        assert_relative_eq!(p.z, 0.0);
    }

    #[test]
    fn local_aabb_covers_the_vertices() {
        let h = octahedron();
        let a = h.local_aabb();
        assert_relative_eq!(a.min.x, -1.0);
        assert_relative_eq!(a.max.y, 1.0);
        assert_relative_eq!(a.max.z, 1.0);
    }

    #[test]
    fn inertia_uses_the_bounding_box() {
        let h = octahedron();
        let i = h.local_inertia(3.0);
        // Bounding box is 2x2x2, like a unit-half-extent box.
        assert_relative_eq!(i.x, 2.0);
        assert_relative_eq!(i.y, 2.0);
        assert_relative_eq!(i.z, 2.0);
    }
}
