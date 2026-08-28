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

    /// True if `other` lies entirely inside this box (closed bounds; Bullet's
    /// `btDbvtAabbMm::Contain`). The Dbvt broad phase keeps leaf volumes fat
    /// and skips tree updates while the exact AABB stays contained.
    #[inline]
    pub fn contains(&self, other: &Aabb) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.min.z <= other.min.z
            && other.max.x <= self.max.x
            && other.max.y <= self.max.y
            && other.max.z <= self.max.z
    }

    /// The box grown by `e` on every side, per axis (Bullet's
    /// `btDbvtAabbMm::Expand` with a vector).
    #[inline]
    pub fn expanded_vec(&self, e: Vector3) -> Aabb {
        Aabb::new(self.min - e, self.max + e)
    }

    /// The box grown only along the direction of each component of `e`
    /// (Bullet's `btDbvtAabbMm::SignedExpand`): positive components push the
    /// max side, negative ones the min side. Used to pre-inflate Dbvt leaf
    /// volumes along an object's motion.
    #[inline]
    pub fn signed_expanded(&self, e: Vector3) -> Aabb {
        let mut out = *self;
        if e.x > 0.0 {
            out.max.x += e.x;
        } else {
            out.min.x += e.x;
        }
        if e.y > 0.0 {
            out.max.y += e.y;
        } else {
            out.min.y += e.y;
        }
        if e.z > 0.0 {
            out.max.z += e.z;
        } else {
            out.min.z += e.z;
        }
        out
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
