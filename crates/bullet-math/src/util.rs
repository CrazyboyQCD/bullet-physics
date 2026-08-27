use super::{Matrix3, Quaternion, Real, Vector3};

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Aabb {
    pub min: Vector3,
    pub max: Vector3,
}

impl Aabb {
    #[inline]
    pub const fn new(min: Vector3, max: Vector3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn center(&self) -> Vector3 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn extents(&self) -> Vector3 {
        (self.max - self.min) * 0.5
    }

    /// True if the two boxes overlap (touching counts as overlapping).
    #[inline]
    pub fn overlaps(&self, other: &Aabb) -> bool {
        // Explicit components: Vec3A's mask ops would also test the unused
        // w lane (garbage after min/max arithmetic), which could spuriously
        // report non-overlap.
        self.min.x <= other.max.x
            && self.min.y <= other.max.y
            && self.min.z <= other.max.z
            && other.min.x <= self.max.x
            && other.min.y <= self.max.y
            && other.min.z <= self.max.z
    }

    #[inline]
    pub fn merged(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// The box grown by `margin` on every side. Used to inflate broad-phase
    /// AABBs by the contact-breaking threshold (Bullet's
    /// `btCollisionWorld::updateSingleAabb` contact-threshold padding).
    #[inline]
    pub fn expanded(&self, margin: Real) -> Aabb {
        let pad = Vector3::splat(margin);
        Aabb::new(self.min - pad, self.max + pad)
    }

    /// Transform the AABB by a rigid pose (rotation + translation), sweeping
    /// its eight corners. Exact for a box shape; conservative for any shape
    /// contained in the box. Equivalent to Bullet's `btTransformAabb`.
    #[inline]
    pub fn transformed(&self, position: Vector3, orientation: Quaternion) -> Aabb {
        let c = self.center();
        let e = self.extents();
        let mut min = Vector3::splat(Real::INFINITY);
        let mut max = Vector3::splat(Real::NEG_INFINITY);
        for i in 0..8 {
            let corner = Vector3::new(
                if i & 1 == 0 { c.x + e.x } else { c.x - e.x },
                if i & 2 == 0 { c.y + e.y } else { c.y - e.y },
                if i & 4 == 0 { c.z + e.z } else { c.z - e.z },
            );
            #[cfg(not(feature = "f64"))]
            let p = orientation.mul_vec3a(corner) + position;
            #[cfg(feature = "f64")]
            let p = orientation * corner + position;
            min = min.min(p);
            max = max.max(p);
        }
        Aabb::new(min, max)
    }
}

/// Build the world-space inverse inertia tensor from the diagonal local inverse
/// inertia and the body orientation: `R * diag(inv_inertia) * R^T`.
///
/// Equivalent to `btTransformUtil::calculateInertia` / `btRigidBody::updateInertiaTensor`.
#[inline]
pub fn inertia_tensor_world(local_inv_inertia: Vector3, orientation: Quaternion) -> Matrix3 {
    let r = Matrix3::from_quat(orientation);
    let d = Matrix3::from_cols(
        Vector3::new(local_inv_inertia.x, 0.0, 0.0),
        Vector3::new(0.0, local_inv_inertia.y, 0.0),
        Vector3::new(0.0, 0.0, local_inv_inertia.z),
    );
    r * d * r.transpose()
}
