use bullet_math::{Aabb, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
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

    /// Mutable access to child `i` (Bullet's `getChildShape`). The
    /// compound itself keeps no derived caches — AABB, support and
    /// inertia are computed on demand — so mutating `position` /
    /// `orientation` here takes effect immediately for narrow phase;
    /// the *owning body's* inertia and broad-phase bounds are the
    /// consumer's responsibility. Prefer `RigidBody::set_child_transform`
    /// / `PhysicsWorld::set_child_transform`, which keep those in sync.
    #[inline]
    pub fn child_mut(&mut self, i: usize) -> &mut CompoundChild {
        &mut self.children[i]
    }

    /// Moves/rotates child `i` in the compound's local frame (Bullet's
    /// `btCompoundShape::updateChildTransform`).
    #[inline]
    pub fn set_child_transform(&mut self, i: usize, position: Vector3, orientation: Quaternion) {
        let child = self.child_mut(i);
        child.position = position;
        child.orientation = orientation;
    }

    /// Removes child `i` and returns it (Bullet's `removeChildShape`).
    #[inline]
    pub fn remove_child(&mut self, i: usize) -> CompoundChild {
        self.children.remove(i)
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
            let local_dir = child.orientation.inverse() * dir;
            let p = child.orientation * child.shape.support_local(local_dir)
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
            let child_pos = position + orientation * child.position;
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

    #[test]
    fn child_transform_updates_derived_quantities() {
        let mut c = compound_of_two_spheres();
        c.set_child_transform(1, Vector3::new(2.0, 0.0, 0.0), Quaternion::IDENTITY);
        let a = c.local_aabb();
        assert_relative_eq!(a.max.x, 2.5);
        let p = c.support_local(Vector3::X);
        assert_relative_eq!(p.x, 2.5);
    }

    #[test]
    fn remove_child_shrinks_the_compound() {
        let mut c = compound_of_two_spheres();
        let removed = c.remove_child(0);
        assert_eq!(c.children.len(), 1);
        assert_relative_eq!(removed.position.x, -1.0);
        let a = c.local_aabb();
        assert_relative_eq!(a.min.x, 0.5);
        assert_relative_eq!(a.max.x, 1.5);
    }
}
