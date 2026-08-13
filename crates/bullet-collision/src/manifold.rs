use bullet_math::{Real, Vector3};

/// A single contact point between two bodies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    /// Contact position in world space.
    pub position: Vector3,
    /// Contact normal, unit length, pointing from body A to body B.
    pub normal: Vector3,
    /// Penetration depth (positive when penetrating).
    pub penetration: Real,
    /// Accumulated normal impulse, for warm starting across solver iterations.
    pub normal_impulse: Real,
    /// Accumulated friction impulses along two tangent axes.
    pub tangent_impulse: [Real; 2],
}

impl ContactPoint {
    #[inline]
    pub fn new(position: Vector3, normal: Vector3, penetration: Real) -> Self {
        Self {
            position,
            normal,
            penetration,
            normal_impulse: 0.0,
            tangent_impulse: [0.0; 2],
        }
    }
}

/// A contact manifold holding up to a handful of contact points.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContactManifold {
    pub points: Vec<ContactPoint>,
}

impl ContactManifold {
    #[inline]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    #[inline]
    pub fn from_points(points: Vec<ContactPoint>) -> Self {
        Self { points }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Flip the manifold so the normal points in the opposite direction
    /// (swap body A and body B).
    #[inline]
    pub fn flipped(mut self) -> Self {
        for p in &mut self.points {
            p.normal = -p.normal;
        }
        self
    }

    /// Keep at most `max` well-distributed points (Bullet manifold reduction).
    ///
    /// The deepest-penetrating point is always kept first; the rest are
    /// chosen greedily, each being the candidate farthest from the points
    /// already kept, so a box resting on a triangle mesh yields a stable
    /// planar contact set instead of one point per triangle.
    pub fn reduce(&mut self, max: usize) {
        if self.points.len() <= max {
            return;
        }
        let mut candidates = std::mem::take(&mut self.points);
        candidates.sort_by(|a, b| b.penetration.total_cmp(&a.penetration));
        let mut kept: Vec<ContactPoint> = Vec::with_capacity(max);
        kept.push(candidates[0]);
        let mut used = vec![false; candidates.len()];
        used[0] = true;
        while kept.len() < max {
            let mut best = 0;
            let mut best_sep = Real::NEG_INFINITY;
            for (i, p) in candidates.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let sep = kept
                    .iter()
                    .map(|q| (p.position - q.position).length_squared())
                    .fold(Real::INFINITY, Real::min);
                if sep > best_sep {
                    best_sep = sep;
                    best = i;
                }
            }
            kept.push(candidates[best]);
            used[best] = true;
        }
        self.points = kept;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: Real, y: Real, z: Real, penetration: Real) -> ContactPoint {
        ContactPoint::new(Vector3::new(x, y, z), Vector3::Y, penetration)
    }

    #[test]
    fn reduce_keeps_the_deepest_and_spreads_the_rest() {
        let mut m = ContactManifold::from_points(vec![
            point(0.0, 0.0, 0.0, 0.5),
            point(0.01, 0.0, 0.0, 0.4), // duplicates the deepest point
            point(1.0, 0.0, 0.0, 0.1),
            point(0.0, 1.0, 0.0, 0.1),
            point(0.0, 0.0, 1.0, 0.1),
            point(1.0, 1.0, 1.0, 0.05),
        ]);
        m.reduce(4);
        assert_eq!(m.points.len(), 4);
        // The deepest contact survives and stays first.
        assert_eq!(m.points[0].position, Vector3::ZERO);
        assert_eq!(m.points[0].penetration, 0.5);
        // The near-duplicate was dropped in favor of a spread-out contact set
        // (the deepest point at the origin is exempt from the spread check).
        assert!(m.points.iter().skip(1).all(|p| p.position.length_squared() > 0.5));
    }

    #[test]
    fn reduce_leaves_small_manifolds_alone() {
        let points = vec![point(0.0, 0.0, 0.0, 0.2), point(1.0, 0.0, 0.0, 0.1), point(0.0, 1.0, 0.0, 0.1)];
        let mut m = ContactManifold::from_points(points.clone());
        m.reduce(4);
        assert_eq!(m.points, points);
    }
}
