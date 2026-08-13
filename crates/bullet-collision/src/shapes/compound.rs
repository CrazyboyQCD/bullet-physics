use bullet_math::{Aabb, Quaternion, Real, Vector3};
use crate::shapes::CollisionShape;

/// A child shape of a compound, positioned relative to the compound origin.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundChild {
    pub shape: CollisionShape,
    pub position: Vector3,
    pub orientation: Quaternion,
}

impl CompoundChild {
    #[inline]
    pub fn new(shape: CollisionShape, position: Vector3) -> Self {
        Self {
            shape,
            position,
            orientation: Quaternion::IDENTITY,
        }
    }

    #[inline]
    pub fn with_orientation(
        shape: CollisionShape,
        position: Vector3,
        orientation: Quaternion,
    ) -> Self {
        Self {
            shape,
            position,
            orientation,
        }
    }
}

/// A compound collision shape: a rigid aggregation of child shapes.
/// Equivalent to `btCompoundShape`.
///
/// Children must be convex for the shape to participate in GJK narrow phase;
/// children may themselves be compounds (recursive).
#[derive(Debug, Clone, PartialEq)]
pub struct Compound {
    pub children: Vec<CompoundChild>,
}

impl Compound {
    #[inline]
    pub fn new(children: Vec<CompoundChild>) -> Self {
        Self { children }
    }

    #[inline]
    pub fn margin(&self) -> Real {
        0.0
    }

    /// True when every child is convex (recursively).
    #[inline]
    pub fn is_convex(&self) -> bool {
        self.children
            .iter()
            .all(|c| c.shape.is_convex())
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        let mut aabb: Option<Aabb> = None;
        for child in &self.children {
            let child_aabb = child.shape.local_aabb().transformed(
                child.position,
                child.orientation,
            );
            aabb = Some(match aabb {
                None => child_aabb,
                Some(a) => a.merged(&child_aabb),
            });
        }
        aabb.unwrap_or_default()
    }

    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let mut best = Vector3::ZERO;
        let mut best_dot = Real::NEG_INFINITY;
        for child in &self.children {
            let local_dir = child.orientation.inverse().mul_vec3(dir);
            let p = child.orientation.mul_vec3(child.shape.support_local(local_dir))
                + child.position;
            let d = p.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = p;
            }
        }
        best
    }

    /// Compound AABB: the merge of every child's world AABB.
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        let mut aabb: Option<Aabb> = None;
        for child in &self.children {
            let child_pos = position + orientation.mul_vec3(child.position);
            let child_ori = orientation * child.orientation;
            let child_aabb = child.shape.world_aabb(child_pos, child_ori);
            aabb = Some(match aabb {
                None => child_aabb,
                Some(a) => a.merged(&child_aabb),
            });
        }
        aabb.unwrap_or_default()
    }

    /// AABB-box approximation of the inertia, matching Bullet's
    /// `btCompoundShape::calculateLocalInertia` (exact mass-weighted child
    /// tensors would need per-child mass allocation; deferred).
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
    use crate::shapes::{BoxShape, Sphere};
    use approx::assert_relative_eq;

    fn compound_of_two_spheres() -> Compound {
        Compound::new(vec![
            CompoundChild::new(CollisionShape::Sphere(Sphere::new(0.5)), Vector3::new(-1.0, 0.0, 0.0)),
            CompoundChild::new(CollisionShape::Sphere(Sphere::new(0.5)), Vector3::new(1.0, 0.0, 0.0)),
        ])
    }

    #[test]
    fn local_aabb_merges_children() {
        let c = compound_of_two_spheres();
        let a = c.local_aabb();
        assert_relative_eq!(a.min.x, -1.5);
        assert_relative_eq!(a.max.x, 1.5);
        assert_relative_eq!(a.min.y, -0.5);
        assert_relative_eq!(a.max.y, 0.5);
    }

    #[test]
    fn support_takes_the_farthest_child() {
        let c = compound_of_two_spheres();
        let p = c.support_local(Vector3::X);
        assert_relative_eq!(p.x, 1.5);
        assert_relative_eq!(p.y, 0.0);
    }

    #[test]
    fn is_convex_reflects_children() {
        let c = compound_of_two_spheres();
        assert!(c.is_convex());
        let mixed = Compound::new(vec![
            CompoundChild::new(
                CollisionShape::BoxShape(BoxShape::new(Vector3::ONE)),
                Vector3::ZERO,
            ),
            CompoundChild::new(
                CollisionShape::plane(Vector3::Y, 0.0),
                Vector3::ZERO,
            ),
        ]);
        assert!(!mixed.is_convex());
    }

    #[test]
    fn inertia_uses_the_merged_bounding_box() {
        let c = compound_of_two_spheres();
        let i = c.local_inertia(6.0);
        // Bounding box: half extents (1.5, 0.5, 0.5).
        assert_relative_eq!(i.x, 2.0 * (0.25 + 0.25));
        assert_relative_eq!(i.y, 2.0 * (2.25 + 0.25));
        assert_relative_eq!(i.z, 2.0 * (2.25 + 0.25));
    }
}
