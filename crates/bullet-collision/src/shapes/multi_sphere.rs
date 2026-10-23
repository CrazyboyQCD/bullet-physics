use bullet_math::{math, Aabb, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// The convex hull of a collection of spheres (Bullet's
/// `btMultiSphereShape`): capsules, metaball-like volumes, simplified
/// proxies. GJK support = the farthest sphere's `pos + dir·radius`; the
/// support of a union of convex sets is the support of their hull, so every
/// generic convex query sees the hull surface.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiSphere {
    /// Sphere centers (local frame) and radii.
    pub spheres: Vec<(Vector3, Real)>,
}

impl MultiSphere {
    /// Build from sphere centers and radii. Panics on an empty collection
    /// (the support of an empty set is undefined; upstream would return
    /// garbage instead).
    #[inline]
    pub fn new(spheres: &[(Vector3, Real)]) -> Self {
        assert!(
            !spheres.is_empty(),
            "MultiSphere needs at least one sphere"
        );
        Self {
            spheres: spheres.to_vec(),
        }
    }

    /// Bullet's default convex distance margin. Unlike the box (where it is
    /// informational), the upstream multi-sphere support subtracts the
    /// margin *inside* its without-margin query, so the GJK core is the
    /// spheres shrunk by this amount and the with-margin support adds it
    /// back — the observable surface stays the pure sphere union.
    #[inline]
    pub fn margin(&self) -> Real {
        0.04
    }

    /// Full support: normalize (zero direction falls back to `+x`), then the
    /// first sphere maximizing `dir·(pos + dir·r)` (upstream's `maxDot` uses
    /// strict `>`, so ties keep the first). The margin cancels between the
    /// core subtraction and the caller's band, so this is the pure sphere
    /// surface — bit-identical to upstream's round trip.
    #[inline]
    pub fn support_local(&self, dir: Vector3) -> Vector3 {
        let (best, _) = self.select(dir, 0.0);
        best
    }

    /// Core support (Bullet's `localGetSupportingVertexWithoutMargin`):
    /// same selection with every radius shrunk by the
    /// [`margin`](Self::margin) (upstream subtracts `vec·margin` inside the
    /// query).
    #[inline]
    pub fn core_support_local(&self, dir: Vector3) -> Vector3 {
        let (best, _) = self.select(dir, -self.margin());
        best
    }

    /// Shared selection: the first sphere maximizing `dir·(pos + dir·(r +
    /// extra))`, returned as the support point `pos + dir·(r + extra)` with
    /// `dir` normalized (zero direction falls back to `+x`, upstream's
    /// `SIMD_EPSILON²` gate). The `extra` shift is constant across
    /// candidates, so core and full support pick the same sphere.
    #[inline]
    fn select(&self, dir: Vector3, extra: Real) -> (Vector3, usize) {
        let len_sq = dir.length_squared();
        let dir = if len_sq < Real::EPSILON * Real::EPSILON {
            Vector3::X
        } else {
            dir / math::sqrt(len_sq)
        };
        let mut best_index = 0;
        let mut best_dot = -Real::INFINITY;
        for (index, (pos, radius)) in self.spheres.iter().enumerate() {
            let candidate = *pos + dir * (*radius + extra);
            let dot = dir.dot(candidate);
            if dot > best_dot {
                best_dot = dot;
                best_index = index;
            }
        }
        let (pos, radius) = &self.spheres[best_index];
        (*pos + dir * (*radius + extra), best_index)
    }

    /// Local AABB: the sphere bounds expanded by the margin on each side
    /// (upstream's cached box = with-margin support, then `± margin` again).
    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        let mut min = Vector3::splat(Real::INFINITY);
        let mut max = Vector3::splat(-Real::INFINITY);
        for (pos, radius) in &self.spheres {
            let r = Vector3::splat(*radius);
            min = min.min(*pos - r);
            max = max.max(*pos + r);
        }
        Aabb::new(min, max).expanded(self.margin())
    }

    /// Box inertia of the expanded local AABB (upstream's explicit
    /// approximation: "take the inertia of the box that bounds the spheres").
    #[inline]
    pub fn local_inertia(&self, mass: Real) -> Vector3 {
        let aabb = self.local_aabb();
        let l = aabb.max - aabb.min;
        Vector3::new(
            mass / 12.0 * (l.y * l.y + l.z * l.z),
            mass / 12.0 * (l.x * l.x + l.z * l.z),
            mass / 12.0 * (l.x * l.x + l.y * l.y),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn two_spheres() -> MultiSphere {
        MultiSphere::new(&[(Vector3::new(-1.0, 0.0, 0.0), 0.5), (Vector3::new(1.0, 0.0, 0.0), 1.0)])
    }

    #[test]
    fn support_hits_the_far_sphere_along_the_axis() {
        let ms = two_spheres();
        let p = ms.support_local(Vector3::X);
        assert_relative_eq!(p.x, 2.0);
        assert_relative_eq!(p.y, 0.0);
        let p = ms.support_local(-Vector3::X);
        assert_relative_eq!(p.x, -1.5);
    }

    #[test]
    fn support_picks_the_sphere_by_projection_plus_radius() {
        let ms = two_spheres();
        // Diagonal direction: sphere 2 wins (projection c + radius 1 vs
        // projection −c + radius 0.5, c = √2/2).
        let c = bullet_math::FRAC_1_SQRT_2;
        let d = Vector3::new(c, c, 0.0);
        let p = ms.support_local(d);
        assert_relative_eq!(p.x, 1.0 + c, epsilon = 1e-4);
        assert_relative_eq!(p.y, c, epsilon = 1e-4);
    }

    #[test]
    fn support_tie_keeps_the_first_sphere() {
        // Two spheres equidistant along +y (binary-exact offsets): the
        // strict `>` comparison keeps the first (upstream `maxDot`).
        let ms = MultiSphere::new(&[
            (Vector3::new(-0.5, 0.0, 0.0), 1.0),
            (Vector3::new(0.5, 0.0, 0.0), 1.0),
        ]);
        let (_, index) = ms.select(Vector3::Y, 0.0);
        assert_eq!(index, 0);
    }

    #[test]
    fn zero_direction_falls_back_to_x() {
        let ms = two_spheres();
        let p = ms.support_local(Vector3::ZERO);
        assert_relative_eq!(p.x, 2.0);
    }

    #[test]
    fn core_support_is_margin_short_of_the_surface() {
        let ms = two_spheres();
        let full = ms.support_local(Vector3::X);
        let core = ms.core_support_local(Vector3::X);
        assert_relative_eq!(full.x - core.x, ms.margin());
    }

    #[test]
    fn aabb_includes_margin_and_inertia_matches_the_box() {
        let ms = two_spheres();
        let aabb = ms.local_aabb();
        assert_relative_eq!(aabb.min.x, -1.5 - ms.margin());
        assert_relative_eq!(aabb.max.x, 2.0 + ms.margin());
        let inertia = ms.local_inertia(3.0);
        let lx = aabb.max.x - aabb.min.x;
        let ly = aabb.max.y - aabb.min.y;
        let lz = aabb.max.z - aabb.min.z;
        assert_relative_eq!(inertia.x, 3.0 / 12.0 * (ly * ly + lz * lz));
        assert_relative_eq!(inertia.y, 3.0 / 12.0 * (lx * lx + lz * lz));
        assert_relative_eq!(inertia.z, 3.0 / 12.0 * (lx * lx + ly * ly));
    }

    #[test]
    #[should_panic(expected = "at least one sphere")]
    fn empty_collection_panics() {
        let _ = MultiSphere::new(&[]);
    }
}
