use bullet_math::{Matrix3, Quaternion, Real, Vector3};
use bullet_collision::shapes::CollisionShape;

/// Seconds a body must stay below the sleeping thresholds before it sleeps
/// (Bullet's hard-coded `m_deactivationTime > btScalar(2.)` check).
pub const DEACTIVATION_TIME: Real = 2.0;

/// Rigid-body activation state (Bullet's `btCollisionObject::ActivationState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationState {
    /// Participates in the simulation and may deactivate when at rest.
    Active,
    /// At rest: excluded from integration until woken by a contact or force.
    IslandSleeping,
    /// Has been slow long enough to qualify for sleeping (checked by the
    /// world, which then calls [`RigidBody::go_to_sleep`]).
    WantsDeactivation,
    /// Never deactivates (e.g. a perpetually driven body).
    DisableDeactivation,
    /// Excluded from the simulation entirely.
    DisableSimulation,
}

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
    activation_state: ActivationState,
    deactivation_time: Real,
    /// Speed below which the body counts as resting (Bullet's defaults
    /// `m_linearSleepingThreshold = 0.8` / `m_angularSleepingThreshold = 1.0`).
    pub linear_sleeping_threshold: Real,
    pub angular_sleeping_threshold: Real,
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
            activation_state: ActivationState::Active,
            deactivation_time: 0.0,
            linear_sleeping_threshold: 0.8,
            angular_sleeping_threshold: 1.0,
        }
    }

    /// A static (immovable) body, e.g. the ground.
    ///
    /// Static bodies start in the sleeping state, like Bullet's static
    /// `btRigidBody` (constructed with `ISLAND_SLEEPING`).
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
            activation_state: ActivationState::IslandSleeping,
            deactivation_time: 0.0,
            linear_sleeping_threshold: 0.8,
            angular_sleeping_threshold: 1.0,
        }
    }

    #[inline]
    pub fn is_static(&self) -> bool {
        self.is_static
    }

    /// Whether the body participates in this step's integration (Bullet's
    /// `isActive()`: everything but sleeping/disabled bodies).
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(
            self.activation_state,
            ActivationState::Active
                | ActivationState::WantsDeactivation
                | ActivationState::DisableDeactivation
        )
    }

    #[inline]
    pub fn activation_state(&self) -> ActivationState {
        self.activation_state
    }

    /// Wake the body up (Bullet's `activate()`): reset to the active state and
    /// restart the deactivation timer. Static and simulation-disabled bodies
    /// cannot be activated; `DisableDeactivation` bodies stay as they are.
    pub fn activate(&mut self) {
        if self.is_static
            || matches!(
                self.activation_state,
                ActivationState::DisableSimulation | ActivationState::DisableDeactivation
            )
        {
            return;
        }
        self.activation_state = ActivationState::Active;
        self.deactivation_time = 0.0;
    }

    /// Put the body to sleep: zero its velocities and stop simulating it
    /// (Bullet's `goToSleep()`).
    pub fn go_to_sleep(&mut self) {
        self.activation_state = ActivationState::IslandSleeping;
        self.linear_velocity = Vector3::ZERO;
        self.angular_velocity = Vector3::ZERO;
    }

    /// Whether the body has been slow for long enough to sleep
    /// (Bullet's `wantsSleeping()`).
    #[inline]
    pub fn wants_deactivation(&self) -> bool {
        self.activation_state == ActivationState::WantsDeactivation
    }

    /// Accumulate rest time while both speeds stay below the sleeping
    /// thresholds; after [`DEACTIVATION_TIME`] seconds the body is marked
    /// [`ActivationState::WantsDeactivation`] (Bullet's `updateDeactivation`).
    pub fn update_deactivation(&mut self, dt: Real) {
        if !self.is_active() {
            return;
        }
        let slow = self.linear_velocity.length_squared()
            < self.linear_sleeping_threshold * self.linear_sleeping_threshold
            && self.angular_velocity.length_squared()
                < self.angular_sleeping_threshold * self.angular_sleeping_threshold;
        self.deactivation_time = if slow { self.deactivation_time + dt } else { 0.0 };
        if self.deactivation_time > DEACTIVATION_TIME {
            self.deactivation_time = 0.0;
            self.activation_state = ActivationState::WantsDeactivation;
        }
    }

    /// Builder: override the sleeping thresholds (defaults 0.8 linear / 1.0
    /// angular, Bullet's `setSleepingThresholds`).
    pub fn sleeping_thresholds(mut self, linear: Real, angular: Real) -> Self {
        self.linear_sleeping_threshold = linear;
        self.angular_sleeping_threshold = angular;
        self
    }

    /// Builder: never deactivate this body (Bullet's `DISABLE_DEACTIVATION`).
    pub fn disable_deactivation(mut self) -> Self {
        self.activation_state = ActivationState::DisableDeactivation;
        self
    }

    /// Builder: exclude the body from the simulation entirely (Bullet's
    /// `DISABLE_SIMULATION`).
    pub fn disable_simulation(mut self) -> Self {
        self.activation_state = ActivationState::DisableSimulation;
        self
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
    ///
    /// Applying a force wakes the body up, like Bullet's `applyCentralForce`.
    #[inline]
    pub fn apply_force(&mut self, force: Vector3) {
        self.activate();
        self.force += force;
    }

    /// Accumulate a force applied at a world-space point.
    ///
    /// Applying a force wakes the body up, like Bullet's `applyForce`.
    #[inline]
    pub fn apply_force_at_point(&mut self, force: Vector3, point: Vector3) {
        self.activate();
        self.force += force;
        self.torque += (point - self.position).cross(force);
    }

    #[inline]
    pub fn clear_forces(&mut self) {
        self.force = Vector3::ZERO;
        self.torque = Vector3::ZERO;
    }

    /// Integrate velocities from gravity and accumulated forces (semi-implicit
    /// Euler velocity update). Sleeping and disabled bodies are not touched.
    #[inline]
    pub fn integrate_velocities(&mut self, gravity: Vector3, dt: Real) {
        if self.is_static || !self.is_active() {
            return;
        }
        self.linear_velocity += (gravity + self.force * self.inv_mass) * dt;
        let iw = self.inv_inertia_world();
        self.angular_velocity += iw * (self.torque * dt);
    }

    /// Integrate the pose from the current velocities. Sleeping and disabled
    /// bodies are not touched.
    #[inline]
    pub fn integrate_position(&mut self, dt: Real) {
        if self.is_static || !self.is_active() {
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
    /// `predictUnconstraintMotion`). Sleeping and static bodies stay put.
    #[inline]
    pub fn predicted_position(&self, dt: Real) -> Vector3 {
        if self.is_static || !self.is_active() {
            self.position
        } else {
            self.position + self.linear_velocity * dt
        }
    }
}
