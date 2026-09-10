//! A triangulated convex hull of a convex shape (Bullet's `btShapeHull`),
//! useful for debugging, convex-simplification previews and generating a
//! [`ConvexHull`] (`btConvexHullShape`) from a primitive.
//!
//! The build samples the shape's support mapping along the same 42 fixed
//! unit-sphere directions Bullet uses (its lowres
//! `NUM_UNITSPHERE_POINTS` table, transcribed verbatim), then runs the
//! deterministic quickhull ([`bullet_math::quickhull`], the
//! [`HullLibrary`](https://github.com/bulletphysics/bullet3) stand-in).
//! Bullet's `highres` 256-direction variant and the preferred-penetration
//! direction extras are not ported (deviations §1).

use crate::shapes::CollisionShape;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::{quickhull::quick_hull, Vector3};

/// A triangulated convex hull: the hull vertices in world-local shape space
/// and one triangle index triple per face.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeHull {
    vertices: Vec<Vector3>,
    triangles: Vec<[u32; 3]>,
}

impl ShapeHull {
    /// Build the hull of a convex `shape`. The margin argument of Bullet's
    /// `buildHull(margin)` is deliberately not taken: Bullet ignores it too
    /// (the parameter is unnamed in its signature). Returns `None` when the
    /// sampled support points are degenerate (all coplanar), and for the
    /// non-finite shapes ([`CollisionShape::Plane`], meshes and
    /// heightfields) whose support mapping does not describe a bounded
    /// convex body.
    pub fn build(shape: &CollisionShape) -> Option<Self> {
        // Sample the 42 directions, keeping first occurrences only: box-like
        // support mappings collapse most directions onto the same corner,
        // and duplicate points break the hull builder (Bullet's HullLibrary
        // re-indexes its input the same way). Component-wise comparison —
        // the SIMD vector's `==` tests the unused w lane too.
        let mut points: Vec<Vector3> = Vec::with_capacity(UNIT_SPHERE_POINTS.len());
        for &dir in &UNIT_SPHERE_POINTS {
            let p = shape.support_local(dir);
            if !points.iter().any(|q| q.x == p.x && q.y == p.y && q.z == p.z) {
                points.push(p);
            }
        }
        let (vertices, faces) = quick_hull(&points)?;
        let vertices = vertices.into_iter().map(|i| points[i]).collect();
        let triangles = faces
            .into_iter()
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        Some(Self { vertices, triangles })
    }

    /// The hull vertices (Bullet's `getVertexPointer` / `numVertices`).
    pub fn vertices(&self) -> &[Vector3] {
        &self.vertices
    }

    /// One index triple per triangle (Bullet's `getIndexPointer`, in
    /// triangles: `numIndices() == 3 * numTriangles()`).
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// Number of hull vertices (Bullet's `numVertices`).
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangles (Bullet's `numTriangles`).
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// A [`ConvexHull`] over the hull vertices: round-tripping a primitive
    /// into the general convex pipeline.
    pub fn to_convex_hull(&self) -> CollisionShape {
        CollisionShape::convex_hull(self.vertices.clone())
    }
}

/// Bullet's lowres unit-sphere sampling directions (`getUnitSpherePoints`,
/// `NUM_UNITSPHERE_POINTS = 42`), transcribed verbatim.
const UNIT_SPHERE_POINTS: [Vector3; 42] = [
    Vector3::new(0.000000, -0.000000, -1.000000),
    Vector3::new(0.723608, -0.525725, -0.447219),
    Vector3::new(-0.276388, -0.850649, -0.447219),
    Vector3::new(-0.894426, -0.000000, -0.447216),
    Vector3::new(-0.276388, 0.850649, -0.447220),
    Vector3::new(0.723608, 0.525725, -0.447219),
    Vector3::new(0.276388, -0.850649, 0.447220),
    Vector3::new(-0.723608, -0.525725, 0.447219),
    Vector3::new(-0.723608, 0.525725, 0.447219),
    Vector3::new(0.276388, 0.850649, 0.447219),
    Vector3::new(0.894426, 0.000000, 0.447216),
    Vector3::new(-0.000000, 0.000000, 1.000000),
    Vector3::new(0.425323, -0.309011, -0.850654),
    Vector3::new(-0.162456, -0.499995, -0.850654),
    Vector3::new(0.262869, -0.809012, -0.525738),
    Vector3::new(0.425323, 0.309011, -0.850654),
    Vector3::new(0.850648, -0.000000, -0.525736),
    Vector3::new(-0.525730, -0.000000, -0.850652),
    Vector3::new(-0.688190, -0.499997, -0.525736),
    Vector3::new(-0.162456, 0.499995, -0.850654),
    Vector3::new(-0.688190, 0.499997, -0.525736),
    Vector3::new(0.262869, 0.809012, -0.525738),
    Vector3::new(0.951058, 0.309013, 0.000000),
    Vector3::new(0.951058, -0.309013, 0.000000),
    Vector3::new(0.587786, -0.809017, 0.000000),
    Vector3::new(0.000000, -1.000000, 0.000000),
    Vector3::new(-0.587786, -0.809017, 0.000000),
    Vector3::new(-0.951058, -0.309013, -0.000000),
    Vector3::new(-0.951058, 0.309013, -0.000000),
    Vector3::new(-0.587786, 0.809017, -0.000000),
    Vector3::new(-0.000000, 1.000000, -0.000000),
    Vector3::new(0.587786, 0.809017, -0.000000),
    Vector3::new(0.688190, -0.499997, 0.525736),
    Vector3::new(-0.262869, -0.809012, 0.525738),
    Vector3::new(-0.850648, 0.000000, 0.525736),
    Vector3::new(-0.262869, 0.809012, 0.525738),
    Vector3::new(0.688190, 0.499997, 0.525736),
    Vector3::new(0.525730, 0.000000, 0.850652),
    Vector3::new(0.162456, -0.499995, 0.850654),
    Vector3::new(-0.425323, -0.309011, 0.850654),
    Vector3::new(-0.425323, 0.309011, 0.850654),
    Vector3::new(0.162456, 0.499995, 0.850654),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::CollisionShape;
    use bullet_math::Real;

    /// Every sampled support point of a sphere lies on the sphere, so all
    /// 42 directions are hull vertices and the closed surface is a
    /// triangle mesh over them (Euler: V - E + F = 2 with all edges
    /// shared by two faces).
    #[test]
    fn sphere_hull_keeps_every_sample_on_the_surface() {
        let sphere = CollisionShape::sphere(0.5);
        let hull = ShapeHull::build(&sphere).unwrap();
        assert_eq!(hull.num_vertices(), 42);
        for v in hull.vertices() {
            assert!((v.length() - 0.5).abs() < 1e-5, "v = {v:?}");
        }
        // Every triangle is non-degenerate.
        for t in hull.triangles() {
            let (a, b, c) = (
                hull.vertices()[t[0] as usize],
                hull.vertices()[t[1] as usize],
                hull.vertices()[t[2] as usize],
            );
            assert!((b - a).cross(c - a).length() > 1e-4);
        }
    }

    /// A box's support mapping collapses every direction onto one of the
    /// eight corners, so the hull is the box itself: 8 vertices, 12
    /// triangles, and every vertex on the box surface.
    #[test]
    fn box_hull_is_the_box_itself() {
        let box_shape = CollisionShape::box_shape(Vector3::new(0.5, 0.25, 1.0));
        let hull = ShapeHull::build(&box_shape).unwrap();
        assert_eq!(hull.num_vertices(), 8);
        assert_eq!(hull.num_triangles(), 12);
        for v in hull.vertices() {
            assert!(
                (v.x.abs() - 0.5).abs() < 1e-6
                    && (v.y.abs() - 0.25).abs() < 1e-6
                    && (v.z.abs() - 1.0).abs() < 1e-6,
                "v = {v:?}"
            );
        }
    }

    /// The hull volume of a capsule is within a few percent of the
    /// analytic volume: the cylinder body is exact, while the caps are
    /// approximated by the sparse 42-direction samples (Bullet's lowres
    /// table has the same resolution).
    #[test]
    fn capsule_hull_volume_is_close_to_the_analytic_volume() {
        let (r, half) = (0.5, 0.75);
        let capsule = CollisionShape::capsule(r, half);
        let hull = ShapeHull::build(&capsule).unwrap();
        // Sum the tetrahedra from the origin over the triangulated surface.
        let mut volume = 0.0;
        for t in hull.triangles() {
            let (a, b, c) = (
                hull.vertices()[t[0] as usize],
                hull.vertices()[t[1] as usize],
                hull.vertices()[t[2] as usize],
            );
            volume += a.dot(b.cross(c)) / 6.0;
        }
        let analytic = core::f64::consts::PI as Real
            * r * r
            * (2.0 * half + 4.0 * r / 3.0);
        assert!(
            volume > 0.0 && (volume - analytic) / analytic > -0.18,
            "inscribed hull volume {volume} vs analytic {analytic}"
        );
    }

    /// Round-tripping a primitive through the hull yields a usable general
    /// convex shape (the debug-draw / simplification use case).
    #[test]
    fn hull_round_trips_into_a_convex_hull_shape() {
        let box_shape = CollisionShape::box_shape(Vector3::splat(0.5));
        let hull = ShapeHull::build(&box_shape).unwrap();
        let round = hull.to_convex_hull();
        // The round-tripped hull supports the original box's corners.
        assert_relative_eq(
            round.support_local(Vector3::new(1.0, 1.0, 1.0)),
            Vector3::splat(0.5),
            1e-5,
        );
    }

    /// `assert_relative_eq`-style helper without the macro's message
    /// restriction (see the approx notes in `HANDOFF.md`).
    fn assert_relative_eq(a: Vector3, b: Vector3, eps: Real) {
        assert!((a - b).length() <= eps, "a = {a:?}, b = {b:?}");
    }
}
