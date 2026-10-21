//! Collision shapes.
//!
//! Shapes are value-typed enums, cheap to clone and shareable across bodies
//! (equivalent to Bullet sharing one `btCollisionShape` between objects).

use bullet_math::{Aabb, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod box_shape;
mod bvh;
mod capsule;
mod compound;
mod cone;
mod convex_hull;
mod cylinder;
mod heightfield;
pub(crate) mod internal_edges;
mod plane;
mod shape_hull;
mod sphere;
mod triangle_mesh;

pub use box_shape::BoxShape;
pub use capsule::Capsule;
pub use compound::{Compound, CompoundChild};
pub use cone::Cone;
pub use convex_hull::ConvexHull;
pub use cylinder::Cylinder;
pub use heightfield::{Heightfield, HeightfieldOptions, UpAxis};
pub use internal_edges::{TriangleInfo, TriangleInfoMap};
pub use plane::Plane;
pub use shape_hull::ShapeHull;
pub use sphere::Sphere;
pub use triangle_mesh::{TriangleMesh, TriangleMeshPart};

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
    UniformScalingShape,
    Convex2dShape,
}

/// A collision shape.
#[derive(Debug, Clone, PartialEq)]
pub enum CollisionShape {
    Sphere(Sphere),
    BoxShape(BoxShape),
    Capsule(Capsule),
    Cylinder(Cylinder),
    Cone(Cone),
    ConvexHull(ConvexHull),
    Compound(Compound),
    Plane(Plane),
    TriangleMesh(TriangleMesh),
    Heightfield(Heightfield),
    /// A uniformly scaled convex child (Bullet's `btUniformScalingShape`):
    /// support points and margin scale by the factor, the inertia scales
    /// linearly (Bullet's own "not realistic" note, kept verbatim).
    UniformScaling(Box<CollisionShape>, Real),
    /// A 2D-convex marker over an arbitrary convex child (Bullet's
    /// `btConvex2dShape`): every query forwards to the child verbatim —
    /// the wrapper only tags the type, the caller owns the 2D usage
    /// (Bullet forwards the 3D support untouched).
    Convex2d(Box<CollisionShape>),
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

    /// A capsule with the given radius and half height along its Y axis.
    #[inline]
    pub fn capsule(radius: Real, half_height: Real) -> Self {
        Self::Capsule(Capsule::new(radius, half_height))
    }

    /// A cylinder with the given radius and half height along its Y axis.
    #[inline]
    pub fn cylinder(radius: Real, half_height: Real) -> Self {
        Self::Cylinder(Cylinder::new(radius, half_height))
    }

    /// A cone with the given radius and half height along its Y axis.
    #[inline]
    pub fn cone(radius: Real, half_height: Real) -> Self {
        Self::Cone(Cone::new(radius, half_height))
    }

    /// A convex hull from a set of vertices (must be convex).
    #[inline]
    pub fn convex_hull(points: Vec<Vector3>) -> Self {
        Self::ConvexHull(ConvexHull::new(points))
    }

    /// A compound shape from a list of positioned children.
    #[inline]
    pub fn compound(children: Vec<CompoundChild>) -> Self {
        Self::Compound(Compound::new(children))
    }

    /// An infinite plane `normal . x = constant`.
    #[inline]
    pub fn plane(normal: Vector3, constant: Real) -> Self {
        Self::Plane(Plane::new(normal, constant))
    }

    /// A static triangle mesh (BVH accelerated).
    #[inline]
    pub fn triangle_mesh(vertices: Vec<Vector3>, indices: Vec<[u32; 3]>) -> Self {
        Self::TriangleMesh(TriangleMesh::new(vertices, indices))
    }

    /// A static triangle mesh built from independent parts (Bullet's
    /// `btTriangleIndexVertexArray`): see
    /// [`TriangleMesh::from_parts`] for the part semantics.
    #[inline]
    pub fn triangle_mesh_parts(parts: Vec<TriangleMeshPart>) -> Self {
        Self::TriangleMesh(TriangleMesh::from_parts(parts))
    }

    /// A static heightfield terrain from a `width x depth` height grid.
    #[inline]
    pub fn heightfield(
        heights: Vec<Real>,
        width: usize,
        depth: usize,
        cell_size: Vector3,
    ) -> Self {
        Self::Heightfield(Heightfield::new(heights, width, depth, cell_size))
    }

    /// A scaled copy of a static triangle mesh (Bullet's
    /// `btScaledBvhTriangleMeshShape` semantics): the vertices are scaled
    /// at construction time, so the BVH, narrow phase and rays see the
    /// scaled geometry directly. This is equivalent to Bullet's wrapper
    /// (which rescales every triangle vertex inside its callback and
    /// un-scales the query box, flipping per axis for negative components);
    /// our value-semantics shapes have no shared runtime BVH to reuse, so
    /// the wrapper collapses into a constructor.
    #[inline]
    pub fn scaled_triangle_mesh(mesh: TriangleMesh, scaling: Vector3) -> Self {
        Self::TriangleMesh(mesh.scaled(scaling))
    }

    /// A uniformly scaled convex child (Bullet's `btUniformScalingShape`).
    /// The child's support points and margin scale by `factor` (the margin
    /// scales too — Bullet's `getMargin` multiplies), and the local inertia
    /// scales linearly (Bullet's own "not realistic" note, kept verbatim).
    #[inline]
    pub fn uniform_scaling(child: CollisionShape, factor: Real) -> Self {
        Self::UniformScaling(Box::new(child), factor)
    }

    /// A 2D-convex marker over an arbitrary convex child (Bullet's
    /// `btConvex2dShape`): every query forwards to the child verbatim —
    /// the wrapper only tags the type, the caller owns the 2D usage
    /// (Bullet forwards the child's 3D support untouched).
    #[inline]
    pub fn convex_2d(child: CollisionShape) -> Self {
        Self::Convex2d(Box::new(child))
    }

    #[inline]
    pub fn shape_type(&self) -> ShapeType {
        match self {
            Self::Sphere(_) => ShapeType::Sphere,
            Self::BoxShape(_) => ShapeType::Box,
            Self::Capsule(_) => ShapeType::Capsule,
            Self::Cylinder(_) => ShapeType::Cylinder,
            Self::Cone(_) => ShapeType::Cone,
            Self::ConvexHull(_) => ShapeType::ConvexHull,
            Self::Compound(_) => ShapeType::Compound,
            Self::Plane(_) => ShapeType::Plane,
            Self::TriangleMesh(_) => ShapeType::TriangleMesh,
            Self::Heightfield(_) => ShapeType::Heightfield,
            Self::UniformScaling(_, _) => ShapeType::UniformScalingShape,
            Self::Convex2d(_) => ShapeType::Convex2dShape,
        }
    }

    /// True if the shape is convex (usable with GJK/EPA). A compound is convex
    /// when all of its children are.
    #[inline]
    pub fn is_convex(&self) -> bool {
        match self {
            Self::Sphere(_)
            | Self::BoxShape(_)
            | Self::Capsule(_)
            | Self::Cylinder(_)
            | Self::Cone(_)
            | Self::ConvexHull(_) => true,
            Self::Compound(c) => c.is_convex(),
            Self::UniformScaling(child, _) | Self::Convex2d(child) => child.is_convex(),
            Self::Plane(_) | Self::TriangleMesh(_) | Self::Heightfield(_) => false,
        }
    }

    /// Suggested contact margin (Bullet's `CONVEX_DISTANCE_MARGIN` semantics).
    #[inline]
    pub fn margin(&self) -> Real {
        match self {
            Self::Sphere(s) => s.margin(),
            Self::BoxShape(b) => b.margin(),
            Self::Capsule(c) => c.margin(),
            Self::Cylinder(c) => c.margin(),
            Self::Cone(c) => c.margin(),
            Self::ConvexHull(h) => h.margin(),
            Self::Compound(c) => c.margin(),
            Self::Plane(p) => p.margin(),
            Self::UniformScaling(child, f) => child.margin() * f,
            Self::Convex2d(child) => child.margin(),
            Self::TriangleMesh(_) | Self::Heightfield(_) => 0.0,
        }
    }

    /// Local-space AABB, centered at the origin.
    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        match self {
            Self::Sphere(s) => s.local_aabb(),
            Self::BoxShape(b) => b.local_aabb(),
            Self::Capsule(c) => c.local_aabb(),
            Self::Cylinder(c) => c.local_aabb(),
            Self::Cone(c) => c.local_aabb(),
            Self::ConvexHull(h) => h.local_aabb(),
            Self::Compound(c) => c.local_aabb(),
            Self::Plane(p) => p.local_aabb(),
            Self::UniformScaling(child, f) => scaled_local_aabb(&child.local_aabb(), f),
            Self::Convex2d(child) => child.local_aabb(),
            Self::TriangleMesh(m) => m.local_aabb(),
            Self::Heightfield(h) => h.local_aabb(),
        }
    }

    /// Support point in local space: the farthest point of the shape along `dir`.
    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        match self {
            Self::Sphere(s) => s.support_local(dir),
            Self::BoxShape(b) => b.support_local(dir),
            Self::Capsule(c) => c.support_local(dir),
            Self::Cylinder(c) => c.support_local(dir),
            Self::Cone(c) => c.support_local(dir),
            Self::ConvexHull(h) => h.support_local(dir),
            Self::Compound(c) => c.support_local(dir),
            Self::UniformScaling(child, f) => child.support_local(dir) * f,
            Self::Convex2d(child) => child.support_local(dir),
            // Planes, meshes, and heightfields are not convex and never reach
            // GJK support queries.
            Self::Plane(_) | Self::TriangleMesh(_) | Self::Heightfield(_) => Vector3::ZERO,
        }
    }

    /// Support point of the *core* in local space (Bullet's
    /// `localGetSupportVertexWithoutMargin`): the shape shrunk by its
    /// [`margin`](Self::margin). The full shape is the core inflated by a
    /// ball of that radius, so the GJK/EPA contact pipeline walks the cores
    /// and adds the margins back at the end — an exact decomposition for
    /// spheres and capsules (ball Minkowski sums) and the standard
    /// componentwise-shrink approximation for boxes, cylinders, and cones
    /// (Bullet's implicit shapes store `halfExtents - margin`). Hulls, mesh
    /// triangles, and compounds carry no margin in this engine, so their
    /// core is the full shape.
    #[inline]
    pub fn core_support_local(&self, dir: Vector3) -> Vector3 {
        match self {
            Self::Sphere(s) => s.core_support_local(dir),
            Self::BoxShape(b) => b.core_support_local(dir),
            Self::Capsule(c) => c.core_support_local(dir),
            Self::Cylinder(c) => c.core_support_local(dir),
            Self::Cone(c) => c.core_support_local(dir),
            // Margin-free shapes: the core is the full shape.
            Self::UniformScaling(child, f) => child.core_support_local(dir) * f,
            Self::Convex2d(child) => child.core_support_local(dir),
            other => other.support_local(dir),
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
        let local_dir = orientation.inverse() * dir;
        let local = self.support_local(local_dir);
        orientation * local + position
    }

    /// Core support point in world space, given a pose
    /// ([`core_support_local`](Self::core_support_local) transformed).
    #[inline]
    pub fn core_support_world(
        &self,
        position: Vector3,
        orientation: Quaternion,
        dir: Vector3,
    ) -> Vector3 {
        let local_dir = orientation.inverse() * dir;
        let local = self.core_support_local(local_dir);
        orientation * local + position
    }

    /// World-space AABB, given a pose.
    #[inline]
    pub fn world_aabb(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        match self {
            Self::Sphere(s) => s.world_aabb(position),
            Self::BoxShape(b) => b.world_aabb(position, orientation),
            Self::Capsule(c) => c.world_aabb(position, orientation),
            Self::Cylinder(c) => c.world_aabb(position, orientation),
            Self::Cone(c) => c.world_aabb(position, orientation),
            Self::ConvexHull(h) => h.world_aabb(position, orientation),
            Self::Compound(c) => c.world_aabb(position, orientation),
            Self::Plane(p) => p.world_aabb(position, orientation),
            Self::UniformScaling(child, f) => scaled_world_aabb(
                position,
                &child.world_aabb(position, orientation),
                f,
            ),
            Self::Convex2d(child) => child.world_aabb(position, orientation),
            Self::TriangleMesh(m) => m.world_aabb(position, orientation),
            Self::Heightfield(h) => h.world_aabb(position, orientation),
        }
    }

    /// The contact band of this shape: contacts are generated while two
    /// shapes stay separated by less than the *pair* threshold, taken as the minimum of
    /// the two shapes' values (Bullet's dispatcher with
    /// `CD_USE_RELATIVE_CONTACT_BREAKING_THRESHOLD`, on by default).
    ///
    /// Per shape this is `getAngularMotionDisc() * gContactBreakingThreshold`:
    /// the local-AABB bounding-sphere radius plus the sphere-center offset,
    /// scaled by 0.02. Unlike Bullet our AABBs carry no margin, so the disc is
    /// measured on the true extents (a few percent below Bullet's for the
    /// default 0.04 margins — well inside the band itself). Planes and huge
    /// meshes yield a large value and never limit a pair.
    pub fn contact_breaking_threshold(&self) -> Real {
        let aabb = self.local_aabb();
        let disc = (aabb.max - aabb.min).length() * 0.5 + aabb.center().length();
        disc * crate::manifold::CONTACT_BREAKING_THRESHOLD
    }

    /// Diagonal local inertia tensor for the given mass.
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        match self {
            Self::Sphere(s) => s.local_inertia(mass),
            Self::BoxShape(b) => b.local_inertia(mass),
            Self::Capsule(c) => c.local_inertia(mass),
            Self::Cylinder(c) => c.local_inertia(mass),
            Self::Cone(c) => c.local_inertia(mass),
            Self::ConvexHull(h) => h.local_inertia(mass),
            Self::Compound(c) => c.local_inertia(mass),
            Self::Plane(p) => p.local_inertia(mass),
            Self::UniformScaling(child, f) => child.local_inertia(mass) * f,
            Self::Convex2d(child) => child.local_inertia(mass),
            Self::TriangleMesh(m) => m.local_inertia(mass),
            Self::Heightfield(h) => h.local_inertia(mass),
        }
    }
}

/// Local AABB scaled by a uniform factor around the origin (the
/// [`CollisionShape::UniformScaling`] wrapper): negative factors swap the
/// corners (the shape flips through the origin).
#[inline]
fn scaled_local_aabb(aabb: &Aabb, f: &Real) -> Aabb {
    let (min, max) = if *f >= 0.0 {
        (aabb.min * *f, aabb.max * *f)
    } else {
        (aabb.max * *f, aabb.min * *f)
    };
    Aabb::new(min, max)
}

/// World AABB of a shape scaled by a uniform factor around `position` (the
/// wrapper's child lives at the same pose): each endpoint maps to
/// `position + (endpoint - position) · f`, with the corners swapped for a
/// negative factor.
#[inline]
fn scaled_world_aabb(position: Vector3, aabb: &Aabb, f: &Real) -> Aabb {
    let map = |p: Vector3| position + (p - position) * *f;
    let (min, max) = if *f >= 0.0 {
        (map(aabb.min), map(aabb.max))
    } else {
        (map(aabb.max), map(aabb.min))
    };
    Aabb::new(min, max)
}

/// AABB of a convex shape from its support projections along the three world
/// axes. Exact for any convex shape (the projected half-width is
/// `support(u) . u` along each world axis `u` in local space), and cheaper
/// than sweeping corners for shapes without explicit formulas.
#[inline]
pub(crate) fn support_projected_aabb(
    support: impl Fn(Vector3) -> Vector3,
    position: Vector3,
    orientation: Quaternion,
) -> Aabb {
    let ux = orientation.inverse() * Vector3::X;
    let uy = orientation.inverse() * Vector3::Y;
    let uz = orientation.inverse() * Vector3::Z;
    // min projection along `u` equals `support(-u) . u` (the farthest point
    // along `-u`, projected back onto `u`).
    let min = Vector3::new(
        position.x + support(-ux).dot(ux),
        position.y + support(-uy).dot(uy),
        position.z + support(-uz).dot(uz),
    );
    let max = Vector3::new(
        position.x + support(ux).dot(ux),
        position.y + support(uy).dot(uy),
        position.z + support(uz).dot(uz),
    );
    Aabb::new(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The uniform-scaling wrapper scales a sphere's support points, margin
    /// (Bullet's `getMargin` multiplies), AABB and inertia by the factor —
    /// and its core support (the scaled core, not the scaled full shape)
    /// keeps the ball-Minkowski decomposition exact.
    #[test]
    fn uniform_scaling_scales_support_margin_and_inertia() {
        let shape = CollisionShape::uniform_scaling(CollisionShape::sphere(0.5), 2.0);
        assert!(shape.is_convex());
        assert_eq!(shape.shape_type(), ShapeType::UniformScalingShape);
        // Sphere margin is the full radius, so the wrapper margin is 1.0.
        assert_relative_eq!(shape.margin(), 1.0);
        // Support along +X: 0.5 -> 1.0.
        assert_relative_eq!(shape.support_local(Vector3::X).x, 1.0);
        // Core support is the point core scaled: still the origin.
        assert_relative_eq!(shape.core_support_local(Vector3::X).length(), 0.0);
        // Local AABB: +/- 1.
        let aabb = shape.local_aabb();
        assert_relative_eq!(aabb.min.x, -1.0);
        assert_relative_eq!(aabb.max.x, 1.0);
        // Inertia scales linearly (Bullet's own "not realistic" note).
        assert_relative_eq!(shape.local_inertia(2.0).x, 0.4 * 2.0 * 0.25 * 2.0);
        // The full shape is still exactly a radius-1 ball: core + margin.
        assert_relative_eq!(
            shape.core_support_local(Vector3::X).dot(Vector3::X) + shape.margin(),
            1.0
        );

        // A box child: the fixed 0.04 margin scales to 0.04 * factor.
        let boxed = CollisionShape::uniform_scaling(
            CollisionShape::box_shape(Vector3::splat(0.5)),
            3.0,
        );
        assert_relative_eq!(boxed.margin(), 0.12);

        // A negative factor flips the shape through the origin (corners swap).
        let flipped = CollisionShape::uniform_scaling(
            CollisionShape::box_shape(Vector3::new(1.0, 2.0, 3.0)),
            -1.0,
        );
        let aabb = flipped.local_aabb();
        assert_relative_eq!(aabb.min.x, -1.0);
        assert_relative_eq!(aabb.max.x, 1.0);
        assert_relative_eq!(aabb.min.y, -2.0);
        assert_relative_eq!(aabb.max.y, 2.0);
    }

    /// The convex-2d marker forwards every query to the child verbatim
    /// (Bullet's wrapper only tags the type).
    #[test]
    fn convex_2d_forwards_to_the_child() {
        let child = CollisionShape::box_shape(Vector3::new(1.0, 2.0, 0.5));
        let shape = CollisionShape::convex_2d(child.clone());
        assert!(shape.is_convex());
        assert_eq!(shape.shape_type(), ShapeType::Convex2dShape);
        assert_eq!(shape.margin(), child.margin());
        assert_eq!(shape.support_local(Vector3::ONE), child.support_local(Vector3::ONE));
        assert_eq!(shape.local_aabb(), child.local_aabb());
        assert_eq!(shape.local_inertia(1.5), child.local_inertia(1.5));
    }

    /// The scaled triangle mesh constructor rescales the vertex stream and
    /// rebuilds the derived structures: rays hit the scaled surface, and a
    /// negative component flips the geometry through the origin.
    #[test]
    fn scaled_triangle_mesh_rescales_vertices_and_queries() {
        // A flat 2x2 ground patch at y = 0, scaled 3x vertically.
        let mesh = TriangleMesh::new(
            vec![
                Vector3::new(-1.0, 0.0, -1.0),
                Vector3::new(1.0, 0.0, -1.0),
                Vector3::new(1.0, 0.0, 1.0),
                Vector3::new(-1.0, 0.0, 1.0),
            ],
            vec![[0, 2, 1], [0, 3, 2]],
        );
        let scaled = CollisionShape::scaled_triangle_mesh(mesh, Vector3::new(2.0, 3.0, 2.0));
        assert!(!scaled.is_convex());
        // AABB scales per component.
        let aabb = scaled.local_aabb();
        assert_relative_eq!(aabb.min.x, -2.0);
        assert_relative_eq!(aabb.max.z, 2.0);
        assert_relative_eq!(aabb.max.y, 0.0);

        // A downward ray from above hits the scaled surface (still y = 0
        // here) at a vertex-scaled horizontal offset that would miss the
        // unscaled mesh.
        let hit = scaled.world_aabb(Vector3::ZERO, Quaternion::IDENTITY);
        assert_relative_eq!(hit.max.x, 2.0);

        // Negative scaling flips through the origin.
        let flipped =
            CollisionShape::scaled_triangle_mesh(TriangleMesh::new(
                vec![
                    Vector3::new(-1.0, 0.0, -1.0),
                    Vector3::new(1.0, 0.0, -1.0),
                    Vector3::new(1.0, 0.0, 1.0),
                ],
                vec![[0, 2, 1]],
            ), Vector3::new(-1.0, 1.0, -1.0));
        let aabb = flipped.local_aabb();
        assert_relative_eq!(aabb.min.x, -1.0);
        assert_relative_eq!(aabb.max.x, 1.0);
    }

    /// The wrapper's world support sits at the scaled radius along any
    /// direction (the full-shape pipeline: core support x factor + the
    /// scaled margin).
    #[test]
    fn uniform_scaling_world_support_at_the_scaled_radius() {
        let sphere = CollisionShape::uniform_scaling(CollisionShape::sphere(0.5), 2.0);
        let support = sphere.support_world(Vector3::ZERO, Quaternion::IDENTITY, -Vector3::Y);
        assert_relative_eq!(support.y, -1.0);
        let core = sphere.core_support_world(Vector3::ZERO, Quaternion::IDENTITY, -Vector3::Y);
        assert_relative_eq!(core.y, 0.0, epsilon = 1e-7);
    }
}
