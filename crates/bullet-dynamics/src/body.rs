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
    /// CCD: bodies moving farther than this per step (0 = disabled) are
    /// swept as a sphere and clamped at the first impact (Bullet's
    /// `m_ccdMotionThreshold`).
    ccd_motion_threshold: Real,
    /// CCD: radius of the swept sphere (Bullet's `m_ccdSweptSphereRadius`).
    ccd_swept_sphere_radius: Real,
    /// Split-impulse positional correction accumulated by the solver this
    /// step (Bullet's `m_pushVelocity` / `m_turnVelocity`): a pseudo velocity
    /// that moves the body out of deep penetration without changing its real
    /// velocity. Applied to the pose by the world after position integration,
    /// then cleared.
    pub(crate) push_linear: Vector3,
    pub(crate) push_angular: Vector3,
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
            ccd_motion_threshold: 0.0,
            ccd_swept_sphere_radius: 0.0,
            push_linear: Vector3::ZERO,
            push_angular: Vector3::ZERO,
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
            ccd_motion_threshold: 0.0,
            ccd_swept_sphere_radius: 0.0,
            push_linear: Vector3::ZERO,
            push_angular: Vector3::ZERO,
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
        if !self.is_active() || self.activation_state == ActivationState::DisableDeactivation {
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

    /// Enable continuous collision detection (Bullet's
    /// `setCcdMotionThreshold` and `setCcdSweptSphereRadius`): bodies moving
    /// farther than `motion_threshold` per step are swept as a sphere of
    /// `swept_sphere_radius` and clamped at the first impact, so fast bodies
    /// cannot tunnel through thin geometry. A common choice is the body's
    /// smallest dimension for the threshold and its bounding-sphere radius for
    /// the swept radius.
    pub fn ccd(mut self, motion_threshold: Real, swept_sphere_radius: Real) -> Self {
        self.ccd_motion_threshold = motion_threshold;
        self.ccd_swept_sphere_radius = swept_sphere_radius;
        self
    }

    /// CCD motion threshold (0 = disabled).
    #[inline]
    pub fn ccd_motion_threshold(&self) -> Real {
        self.ccd_motion_threshold
    }

    /// CCD swept-sphere radius (0 = disabled).
    #[inline]
    pub fn ccd_swept_sphere_radius(&self) -> Real {
        self.ccd_swept_sphere_radius
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

    /// Mass in kilograms (`1 / inv_mass`); infinite for static bodies.
    #[inline]
    pub fn mass(&self) -> Real {
        if self.inv_mass == 0.0 {
            Real::INFINITY
        } else {
            1.0 / self.inv_mass
        }
    }

    /// Apply an impulse at a world-space point offset from the center of mass
    /// (Bullet's `applyImpulse`): an instant velocity change
    /// `Δv = impulse · inv_mass` and `Δω = inv_inertia_world · (rel_pos ×
    /// impulse)`. Unlike `apply_force`, this does not wake the body up — it is
    /// a raw velocity change, matching Bullet.
    #[inline]
    pub fn apply_impulse(&mut self, impulse: Vector3, rel_pos: Vector3) {
        if self.inv_mass == 0.0 {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
        self.angular_velocity += self.inv_inertia_world() * rel_pos.cross(impulse);
    }

    /// Velocity of the point at offset `rel_pos` from the center of mass
    /// (Bullet's `getVelocityInLocalPoint`):
    /// `linear_velocity + angular_velocity × rel_pos`.
    #[inline]
    pub fn velocity_at_local_point(&self, rel_pos: Vector3) -> Vector3 {
        self.linear_velocity + self.angular_velocity.cross(rel_pos)
    }

    /// Effective mass (inverse of the constraint diagonal) along `normal` at
    /// the world-space point `pos` (Bullet's
    /// `computeImpulseDenominator`): the denominator `1/jacDiagAB` of a
    /// unilateral impulse constraint at that point, `inv_mass + nᵀ·(I⁻¹·(r×n))×r`.
    #[inline]
    pub fn compute_impulse_denominator(&self, pos: Vector3, normal: Vector3) -> Real {
        let r0 = pos - self.position;
        let c0 = r0.cross(normal);
        let vec = (self.inv_inertia_world() * c0).cross(r0);
        self.inv_mass + normal.dot(vec)
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A body with deactivation disabled must never fall asleep, no matter how
    /// long it rests below the sleeping thresholds (Bullet's `updateDeactivation`
    /// returns early for `DISABLE_DEACTIVATION`).
    #[test]
    fn disable_deactivation_never_sleeps() {
        let mut body =
            RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0).disable_deactivation();
        // 10 seconds of rest, far beyond DEACTIVATION_TIME = 2.
        for _ in 0..100 {
            body.update_deactivation(0.1);
        }
        assert_eq!(body.activation_state(), ActivationState::DisableDeactivation);
        assert!(body.is_active());
    }

    /// Moving fast again resets the rest timer; resting below DEACTIVATION_TIME
    /// leaves the body active.
    #[test]
    fn update_deactivation_resets_timer_on_motion() {
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0);
        body.update_deactivation(1.9);
        body.linear_velocity = Vector3::new(5.0, 0.0, 0.0);
        body.update_deactivation(0.5);
        assert_eq!(body.activation_state(), ActivationState::Active);
        // Back at rest: only 0.5s accumulated, still far from 2s.
        body.linear_velocity = Vector3::ZERO;
        body.update_deactivation(0.5);
        assert_eq!(body.activation_state(), ActivationState::Active);
    }

    /// Resting below the thresholds for longer than DEACTIVATION_TIME marks
    /// the body as wanting deactivation (the world then puts it to sleep).
    #[test]
    fn update_deactivation_marks_wants_deactivation() {
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0);
        for _ in 0..100 {
            body.update_deactivation(0.1);
        }
        assert_eq!(body.activation_state(), ActivationState::WantsDeactivation);
        assert!(body.wants_deactivation());
    }

    /// An impulse at the center of mass only changes the linear velocity; an
    /// off-center impulse also spins the body (Bullet's `applyImpulse`).
    #[test]
    fn apply_impulse_changes_velocity() {
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 2.0);
        body.apply_impulse(Vector3::new(4.0, 0.0, 0.0), Vector3::ZERO);
        assert_relative_eq!(body.linear_velocity.x, 2.0, epsilon = 1e-6);
        assert_eq!(body.angular_velocity, Vector3::ZERO);
        // Off-center impulse adds angular velocity around r × j.
        body.apply_impulse(Vector3::new(0.0, 2.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(body.linear_velocity.y, 1.0, epsilon = 1e-6);
        assert!(body.angular_velocity.length() > 0.0);
        // Static bodies ignore impulses.
        let mut static_body = RigidBody::new_static(CollisionShape::sphere(0.5), Vector3::ZERO);
        static_body.apply_impulse(Vector3::new(1.0, 0.0, 0.0), Vector3::ZERO);
        assert_eq!(static_body.linear_velocity, Vector3::ZERO);
    }

    /// `velocity_at_local_point` = linear velocity + ω × r (Bullet's
    /// `getVelocityInLocalPoint`).
    #[test]
    fn velocity_at_local_point_adds_rotation() {
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0);
        body.linear_velocity = Vector3::new(1.0, 2.0, 0.0);
        body.angular_velocity = Vector3::new(0.0, 0.0, 2.0);
        let v = body.velocity_at_local_point(Vector3::new(0.0, 0.0, 3.0));
        assert_relative_eq!(v.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(v.y, 2.0, epsilon = 1e-6);
        assert_relative_eq!(v.z, 0.0, epsilon = 1e-6);
    }

    /// `compute_impulse_denominator` is positive for dynamic bodies and zero
    /// for static ones (Bullet's `computeImpulseDenominator`).
    #[test]
    fn compute_impulse_denominator_is_positive() {
        let body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 2.0);
        let denom = body.compute_impulse_denominator(Vector3::new(0.0, 0.5, 0.0), Vector3::Y);
        assert!(denom > 0.0);
        let static_body = RigidBody::new_static(CollisionShape::sphere(0.5), Vector3::ZERO);
        assert_eq!(
            static_body.compute_impulse_denominator(Vector3::new(0.0, 0.5, 0.0), Vector3::Y),
            0.0
        );
    }
}
