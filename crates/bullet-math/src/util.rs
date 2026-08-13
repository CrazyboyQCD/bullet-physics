use super::{Matrix3, Quaternion, Vector3};

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
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }

    #[inline]
    pub fn merged(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
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
