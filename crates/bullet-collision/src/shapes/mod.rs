//! Collision shapes.
//!
//! Shapes are value-typed enums, cheap to clone and shareable across bodies
//! (equivalent to Bullet sharing one `btCollisionShape` between objects).

use bullet_math::{Aabb, Quaternion, Real, Vector3};

mod box_shape;
mod sphere;

pub use box_shape::BoxShape;
pub use sphere::Sphere;

/// Shape category, mirroring Bullet's `BroadphaseProxyTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeType {
    Sphere,
    Box,
    Capsule,
    Cylinder,
    Cone,
    ConvexHull,
    Compound,
    Plane,
    TriangleMesh,
    Heightfield,
}

/// A collision shape.
#[derive(Debug, Clone, PartialEq)]
pub enum CollisionShape {
    Sphere(Sphere),
    BoxShape(BoxShape),
}

impl CollisionShape {
    #[inline]
    pub fn sphere(radius: Real) -> Self {
        Self::Sphere(Sphere::new(radius))
    }

    #[inline]
    pub fn box_shape(half_extents: Vector3) -> Self {
        Self::BoxShape(BoxShape::new(half_extents))
    }

    #[inline]
    pub fn shape_type(&self) -> ShapeType {
        match self {
            Self::Sphere(_) => ShapeType::Sphere,
            Self::BoxShape(_) => ShapeType::Box,
        }
    }

    /// True if the shape is convex (usable with GJK/EPA).
    #[inline]
    pub fn is_convex(&self) -> bool {
        matches!(self, Self::Sphere(_) | Self::BoxShape(_))
    }

    /// Suggested contact margin (Bullet's `CONVEX_DISTANCE_MARGIN` semantics).
    #[inline]
    pub fn margin(&self) -> Real {
        match self {
            Self::Sphere(s) => s.margin(),
            Self::BoxShape(b) => b.margin(),
        }
    }

    /// Local-space AABB, centered at the origin.
    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        match self {
            Self::Sphere(s) => s.local_aabb(),
            Self::BoxShape(b) => b.local_aabb(),
        }
    }

    /// Support point in local space: the farthest point of the shape along `dir`.
    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        match self {
            Self::Sphere(s) => s.support_local(dir),
            Self::BoxShape(b) => b.support_local(dir),
        }
    }

    /// Support point in world space, given a pose (position + orientation).
    #[inline]
    pub fn support_world(
        &self,
        position: Vector3,
        orientation: Quaternion,
        dir: Vector3,
    ) -> Vector3 {
        let local_dir = orientation.inverse().mul_vec3(dir);
        let local = self.support_local(local_dir);
        orientation.mul_vec3(local) + position
    }

    /// World-space AABB, given a pose.
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        match self {
            Self::Sphere(s) => s.world_aabb(position),
            Self::BoxShape(b) => b.world_aabb(position, orientation),
        }
    }

    /// Diagonal local inertia tensor for the given mass.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        match self {
            Self::Sphere(s) => s.local_inertia(mass),
            Self::BoxShape(b) => b.local_inertia(mass),
        }
    }
}
