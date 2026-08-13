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
}
