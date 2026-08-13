use bullet_math::{Matrix3, Quaternion, Real, Vector3};
use bullet_collision::shapes::CollisionShape;

/// A rigid body with mass, inertia, pose and velocity state.
///
/// Equivalent to Bullet's `btRigidBody` (with an inline `btCollisionShape`).
pub struct RigidBody {
    pub shape: CollisionShape,
    pub position: Vector3,
    pub orientation: Quaternion,
    pub linear_velocity: Vector3,
    pub angular_velocity: Vector3,
    force: Vector3,
    torque: Vector3,
    inv_mass: Real,
    inv_inertia_local: Vector3,
    pub friction: Real,
    pub restitution: Real,
    is_static: bool,
}

impl RigidBody {
    /// A dynamic body with the given mass (> 0).
    pub fn new_dynamic(shape: CollisionShape, position: Vector3, mass: Real) -> Self {
        let inertia = shape.local_inertia(mass);
        let inv_inertia = Vector3::new(
            1.0 / inertia.x.max(Real::EPSILON),
            1.0 / inertia.y.max(Real::EPSILON),
            1.0 / inertia.z.max(Real::EPSILON),
        );
        Self {
            shape,
            position,
            orientation: Quaternion::IDENTITY,
            linear_velocity: Vector3::ZERO,
            angular_velocity: Vector3::ZERO,
            force: Vector3::ZERO,
            torque: Vector3::ZERO,
            inv_mass: 1.0 / mass,
            inv_inertia_local: inv_inertia,
            friction: 0.5,
            restitution: 0.0,
            is_static: false,
        }
    }

    /// A static (immovable) body, e.g. the ground.
    pub fn new_static(shape: CollisionShape, position: Vector3) -> Self {
        Self {
            shape,
            position,
            orientation: Quaternion::IDENTITY,
            linear_velocity: Vector3::ZERO,
            angular_velocity: Vector3::ZERO,
            force: Vector3::ZERO,
            torque: Vector3::ZERO,
            inv_mass: 0.0,
            inv_inertia_local: Vector3::ZERO,
            friction: 0.5,
            restitution: 0.0,
            is_static: true,
        }
    }

    #[inline]
    pub fn is_static(&self) -> bool {
        self.is_static
    }

    #[inline]
    pub fn inv_mass(&self) -> Real {
        self.inv_mass
    }

    /// Builder: set friction coefficient.
    pub fn friction(mut self, f: Real) -> Self {
        self.friction = f;
        self
    }

    /// Builder: set restitution coefficient.
    pub fn restitution(mut self, r: Real) -> Self {
        self.restitution = r;
        self
    }

    /// Builder: set orientation.
    pub fn orientation(mut self, o: Quaternion) -> Self {
        self.orientation = o;
        self
    }

    /// Builder: set linear velocity.
    pub fn velocity(mut self, v: Vector3) -> Self {
        self.linear_velocity = v;
        self
    }

    /// World-space inverse inertia tensor `R * diag(inv_I_local) * R^T`.
    #[inline]
    pub fn inv_inertia_world(&self) -> Matrix3 {
        bullet_math::inertia_tensor_world(self.inv_inertia_local, self.orientation)
    }

    /// Accumulate a force applied at the center of mass.
    #[inline]
    pub fn apply_force(&mut self, force: Vector3) {
        self.force += force;
    }

    /// Accumulate a force applied at a world-space point.
    #[inline]
    pub fn apply_force_at_point(&mut self, force: Vector3, point: Vector3) {
        self.force += force;
        self.torque += (point - self.position).cross(force);
    }

    #[inline]
    pub fn clear_forces(&mut self) {
        self.force = Vector3::ZERO;
        self.torque = Vector3::ZERO;
    }

    /// Integrate velocities from gravity and accumulated forces (semi-implicit
    /// Euler velocity update).
    #[inline]
    pub fn integrate_velocities(&mut self, gravity: Vector3, dt: Real) {
        if self.is_static {
            return;
        }
        self.linear_velocity += (gravity + self.force * self.inv_mass) * dt;
        let iw = self.inv_inertia_world();
        self.angular_velocity += iw * (self.torque * dt);
    }

    /// Integrate the pose from the current velocities.
    #[inline]
    pub fn integrate_position(&mut self, dt: Real) {
        if self.is_static {
            return;
        }
        self.position += self.linear_velocity * dt;
        // Quaternion integration: dq = 0.5 * (0, w) * q * dt.
        let dq = Quaternion::from_xyzw(
            self.angular_velocity.x * dt * 0.5,
            self.angular_velocity.y * dt * 0.5,
            self.angular_velocity.z * dt * 0.5,
            0.0,
        ) * self.orientation;
        self.orientation = (self.orientation + dq).normalize();
    }

    /// Predicted position after integrating the current velocity for `dt`,
    /// used for broad/narrow-phase contact detection (mirrors Bullet's
    /// `predictUnconstraintMotion`).
    #[inline]
    pub fn predicted_position(&self, dt: Real) -> Vector3 {
        if self.is_static {
            self.position
        } else {
            self.position + self.linear_velocity * dt
        }
    }
}
