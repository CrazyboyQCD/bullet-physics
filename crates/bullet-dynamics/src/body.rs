use bullet_math::{math, Matrix3, Quaternion, Real, Vector3};
use bullet_collision::shapes::{CollisionShape, CompoundChild};

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

/// Collision filter groups (Bullet's
/// `btBroadphaseProxy::CollisionFilterGroups`). A pair collides when
/// `(a.collision_group & b.collision_mask) != 0` **and**
/// `(b.collision_group & a.collision_mask) != 0`.
pub mod filter_groups {
    /// The default group every body belongs to.
    pub const DEFAULT: u16 = 1;
    /// Conventional group for static geometry (Bullet demos).
    pub const STATIC: u16 = 2;
    /// Conventional group for kinematic bodies.
    pub const KINEMATIC: u16 = 4;
    /// Conventional group for trigger volumes.
    pub const TRIGGER: u16 = 16;
    /// Matches every group (Bullet's `AllFilter` = all bits set).
    pub const ALL: u16 = u16::MAX;
}

/// Collision filter for ray queries (Bullet's `RayResultCallback` group /
/// mask members): a body is ray-tested when the symmetric `group & mask`
/// pair test passes — the same rule the broad phase applies to pairs, but
/// with the ray carrying its own group and mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RayFilter {
    /// The group this query belongs to (Bullet's `m_collisionFilterGroup`).
    pub group: u16,
    /// The groups this query tests against (Bullet's `m_collisionFilterMask`).
    pub mask: u16,
}

impl RayFilter {
    /// Accept every body (Bullet's default `RayResultCallback` filter:
    /// group 1, mask all).
    pub const DEFAULT: Self = Self {
        group: filter_groups::DEFAULT,
        mask: filter_groups::ALL,
    };

    /// The symmetric test between this query and a body.
    #[inline]
    pub(crate) fn accepts(self, body: &RigidBody) -> bool {
        self.group & body.collision_mask != 0 && body.collision_group & self.mask != 0
    }
}

/// Collision-object flags (Bullet's `btCollisionObject::CollisionFlags`),
/// a small bit set stored by value like the engine's other flag newtypes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollisionFlags(pub u32);

impl CollisionFlags {
    /// The body is detected and queried normally, but the solver applies no
    /// response for its pairs: other bodies pass through, while contacts,
    /// [`PhysicsWorld::contact_test`](crate::world::PhysicsWorld::contact_test)
    /// queries and collision events (with `sensor = true`) still report
    /// (Bullet's `CF_NO_CONTACT_RESPONSE`, the trigger-volume / sensor flag).
    pub const NO_CONTACT_RESPONSE: Self = Self(0x0001);

    /// The body's gravity does not follow
    /// [`PhysicsWorld::set_gravity`](crate::world::PhysicsWorld::set_gravity)
    /// propagation: the per-body value (synced from the world at `add_body`
    /// time) stays as set (Bullet's `BT_DISABLE_WORLD_GRAVITY`).
    pub const DISABLE_WORLD_GRAVITY: Self = Self(0x0002);
    
    /// The body is a kinematic prop (Bullet's `CF_KINEMATIC_OBJECT`): its
    /// pose is driven by the velocities you set (no gravity, damping or
    /// accumulated forces ever touch them), while its infinite mass pushes
    /// dynamic bodies out of the way through contacts. See
    /// [`RigidBody::new_kinematic`](RigidBody::new_kinematic).
    pub const KINEMATIC: Self = Self(0x0004);

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no flag is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Reciprocal local inertia components (the `new_dynamic` formula,
/// clamped away from zero so a degenerate axis never divides by zero).
fn inv_inertia_components(inertia: Vector3) -> Vector3 {
    Vector3::new(
        1.0 / inertia.x.max(Real::EPSILON),
        1.0 / inertia.y.max(Real::EPSILON),
        1.0 / inertia.z.max(Real::EPSILON),
    )
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
    /// Torsional friction orthogonal to the contact normal (Bullet's
    /// `m_rollingFriction`, default 0): a rolling body touching a surface
    /// with a positive pair value is slowed by angular-friction rows until
    /// it stops, instead of rolling forever. Only read in pairs where the
    /// combined value (`rf_a · friction_b + rf_b · friction_a`, Bullet's
    /// `calculateCombinedRollingFriction` quirk) exceeds zero.
    pub rolling_friction: Real,
    /// Per-body gravity acceleration (Bullet's `m_gravity_acceleration`):
    /// synced from the world at `add_body` time and by
    /// `PhysicsWorld::set_gravity` propagation unless the
    /// [`CollisionFlags::DISABLE_WORLD_GRAVITY`] flag detaches it.
    gravity: Vector3,
    /// Collision filter group: pairs collide when this intersects the other
    /// body's [`collision_mask`](RigidBody::collision_mask) *and* vice versa
    /// (Bullet's `m_collisionFilterGroup`, default [`filter_groups::DEFAULT`]).
    pub collision_group: u16,
    /// Collision filter mask: the groups this body collides with (Bullet's
    /// `m_collisionFilterMask`, default [`filter_groups::ALL`]).
    pub collision_mask: u16,
    /// Collision-object flags (Bullet's `CollisionFlags`; see
    /// [`CollisionFlags::NO_CONTACT_RESPONSE`] for trigger volumes).
    pub collision_flags: CollisionFlags,
    /// Whether pairs this body touches emit collision events (Rapier's
    /// `ActiveEvents::COLLISION_EVENTS` opt-in): quiet scenes pay nothing.
    pub collision_events: bool,
    /// Velocity damping factors (Bullet's `m_linearDamping` / `m_angularDamping`,
    /// defaults 0; set via [`RigidBody::set_damping`], applied every step by
    /// [`RigidBody::integrate_velocities`]).
    linear_damping: Real,
    angular_damping: Real,
    is_static: bool,
    activation_state: ActivationState,
    deactivation_time: Real,
    /// Whether the body has been removed from its world
    /// ([`PhysicsWorld::remove_body`]). A removed body keeps its slot (body
    /// indices are stable), but the world treats it as if it never existed:
    /// no broad-phase proxy, no contacts, no queries, no integration.
    removed: bool,
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
        let inv_inertia = inv_inertia_components(shape.local_inertia(mass));
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
            rolling_friction: 0.0,
            collision_group: filter_groups::DEFAULT,
            collision_mask: filter_groups::ALL,
            collision_flags: CollisionFlags::default(),
            collision_events: false,
            gravity: Vector3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            is_static: false,
            activation_state: ActivationState::Active,
            deactivation_time: 0.0,
            removed: false,
            linear_sleeping_threshold: 0.8,
            angular_sleeping_threshold: 1.0,
            ccd_motion_threshold: 0.0,
            ccd_swept_sphere_radius: 0.0,
            push_linear: Vector3::ZERO,
            push_angular: Vector3::ZERO,
        }
    }

    /// A kinematic prop (Bullet's `CF_KINEMATIC_OBJECT`): infinite mass,
    /// pose driven by the velocities you set - moving platforms, elevators,
    /// doors, slamming pistons. Gravity, damping and accumulated forces
    /// never touch it; contacts push dynamic bodies out of the way using
    /// its velocity. It never sleeps (an idle kinematic body is the
    /// activation source its contacts rely on).
    pub fn new_kinematic(shape: CollisionShape, position: Vector3) -> Self {
        let mut body = Self::new_static(shape, position);
        body.is_static = false;
        body.collision_flags.0 |= CollisionFlags::KINEMATIC.0;
        // The activation source its sleeping contacts rely on; Bullet
        // kinematics live in the non-static list and are typically driven
        // every frame anyway.
        body.activation_state = ActivationState::DisableDeactivation;
        body
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
            rolling_friction: 0.0,
            collision_group: filter_groups::DEFAULT,
            collision_mask: filter_groups::ALL,
            collision_flags: CollisionFlags::default(),
            collision_events: false,
            gravity: Vector3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            is_static: true,
            activation_state: ActivationState::IslandSleeping,
            deactivation_time: 0.0,
            removed: false,
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

    /// Whether the body is a kinematic prop
    /// ([`CollisionFlags::KINEMATIC`]): pose driven by the user-set
    /// velocities, infinite mass to everything else.
    #[inline]
    pub fn is_kinematic(&self) -> bool {
        self.collision_flags.contains(CollisionFlags::KINEMATIC)
    }

    /// Whether the body can never be moved by the solver (Bullet's
    /// `isStaticOrKinematicObject`): the gate for gravity, damping, force
    /// integration, split impulses and CCD on the world paths.
    #[inline]
    pub fn is_static_or_kinematic(&self) -> bool {
        self.is_static || self.is_kinematic()
    }

    /// Whether the body participates in this step's integration (Bullet's
    /// `isActive()`: everything but sleeping/disabled bodies).
    #[inline]
    pub fn is_active(&self) -> bool {
        !self.removed
            && matches!(
                self.activation_state,
                ActivationState::Active
                    | ActivationState::WantsDeactivation
                    | ActivationState::DisableDeactivation
            )
    }

    /// Whether the body has been removed from its world
    /// ([`PhysicsWorld::remove_body`]); removed bodies are inert data.
    #[inline]
    pub fn is_removed(&self) -> bool {
        self.removed
    }

    /// Mark the body as removed and neutralize its dynamic state (called by
    /// [`PhysicsWorld::remove_body`], which also destroys the broad-phase
    /// proxy and cascades over everything referencing the body).
    pub(crate) fn mark_removed(&mut self) {
        self.removed = true;
        self.linear_velocity = Vector3::ZERO;
        self.angular_velocity = Vector3::ZERO;
        self.force = Vector3::ZERO;
        self.torque = Vector3::ZERO;
        self.push_linear = Vector3::ZERO;
        self.push_angular = Vector3::ZERO;
    }

    #[inline]
    pub fn activation_state(&self) -> ActivationState {
        self.activation_state
    }

    /// Wake the body up (Bullet's `activate()`): reset to the active state and
    /// restart the deactivation timer. Static and simulation-disabled bodies
    /// cannot be activated; `DisableDeactivation` bodies stay as they are.
    pub fn activate(&mut self) {
        if self.removed
            || self.is_static
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

    /// Whether this body is an island member: dynamic bodies participate;
    /// static bodies and `DisableSimulation` bodies do not (they are never
    /// simulated and must not be forced into `IslandSleeping`).
    pub(crate) fn participates_in_islands(&self) -> bool {
        !self.removed
            && !self.is_static
            && !self.is_kinematic()
            && self.activation_state != ActivationState::DisableSimulation      
    }

    /// Whether this body keeps its island awake (the `buildIslands`
    /// activation pass: `ACTIVE_TAG` and `DISABLE_DEACTIVATION` count as
    /// awake; `WANTS_DEACTIVATION`/`ISLAND_SLEEPING` are sleep-ready).
    pub(crate) fn is_island_awake(&self) -> bool {
        !self.removed
            && matches!(
                self.activation_state,
                ActivationState::Active | ActivationState::DisableDeactivation
            )
    }

    /// Wake a body whose island is not sleeping (Bullet's `buildIslands`:
    /// `ISLAND_SLEEPING` members of a non-sleeping island become
    /// `WANTS_DEACTIVATION` with a reset deactivation timer, so the body
    /// simulates again but immediately re-qualifies for sleeping). Only
    /// sleeping bodies transition — active members keep their timers.
    pub(crate) fn island_wake(&mut self) {
        if self.removed
            || self.is_static
            || self.activation_state != ActivationState::IslandSleeping
        {
            return;
        }
        self.activation_state = ActivationState::WantsDeactivation;
        self.deactivation_time = 0.0;
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
    /// A body that moves again re-enters the active state — Bullet sets
    /// `ACTIVE_TAG` right where it resets the timer, so a knocked-away
    /// `WANTS_DEACTIVATION` body keeps its own island awake.
    pub fn update_deactivation(&mut self, dt: Real) {
        if !self.is_active() || self.activation_state == ActivationState::DisableDeactivation {
            return;
        }
        let slow = self.linear_velocity.length_squared()
            < self.linear_sleeping_threshold * self.linear_sleeping_threshold
            && self.angular_velocity.length_squared()
                < self.angular_sleeping_threshold * self.angular_sleeping_threshold;
        if slow {
            self.deactivation_time += dt;
        } else {
            self.deactivation_time = 0.0;
            self.activation_state = ActivationState::Active;
        }
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

    /// Builder: set the collision filter group and mask (Bullet's
    /// `setCollisionGroup` / `setCollisionMask`).
    pub fn collision_filter(mut self, group: u16, mask: u16) -> Self {
        self.collision_group = group;
        self.collision_mask = mask;
        self
    }

    /// Builder: set or clear
    /// [`CollisionFlags::NO_CONTACT_RESPONSE`] (a trigger volume: detected
    /// and reported, but it does not stop other bodies).
    pub fn no_contact_response(mut self, on: bool) -> Self {
        self.collision_flags.0 &= !CollisionFlags::NO_CONTACT_RESPONSE.0;
        if on {
            self.collision_flags.0 |= CollisionFlags::NO_CONTACT_RESPONSE.0;
        }
        self
    }

    /// Builder: set or clear the kinematic flag
    /// ([`CollisionFlags::KINEMATIC`]; see [`RigidBody::new_kinematic`]).
    /// Turning the flag on also disables deactivation for the body.
    pub fn kinematic(mut self, on: bool) -> Self {
        self.collision_flags.0 &= !CollisionFlags::KINEMATIC.0;
        if on {
            self.collision_flags.0 |= CollisionFlags::KINEMATIC.0;
            if self.activation_state != ActivationState::DisableDeactivation {
                self.activation_state = ActivationState::DisableDeactivation;
            }
        }
        self
    }
    
    /// Builder: opt this body's pairs in for collision events (Rapier's        
    /// `ActiveEvents::COLLISION_EVENTS`).
    pub fn collision_events(mut self, on: bool) -> Self {
        self.collision_events = on;
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

    /// Body-space inverse inertia diagonal (zero for static/kinematic
    /// bodies).
    #[inline]
    pub fn inv_inertia_local(&self) -> Vector3 {
        self.inv_inertia_local
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

    /// Apply a torque impulse (Bullet's `applyTorqueImpulse`): an instant
    /// angular velocity change `Δω = inv_inertia_world · impulse`. Static
    /// bodies ignore it.
    #[inline]
    pub fn apply_torque_impulse(&mut self, impulse: Vector3) {
        if self.inv_mass == 0.0 {
            return;
        }
        self.angular_velocity += self.inv_inertia_world() * impulse;
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

    /// Set the linear/angular velocity damping factors (Bullet's
    /// `setDamping`). Both are clamped to `[0, 1]`, like Bullet; the
    /// defaults are 0 (no damping).
    pub fn set_damping(&mut self, linear: Real, angular: Real) {
        self.linear_damping = linear.clamp(0.0, 1.0);
        self.angular_damping = angular.clamp(0.0, 1.0);
    }

    /// Move/rotate a child shape of a compound body in the compound's
    /// local frame (the data half of Bullet's
    /// `btCompoundShape::updateChildTransform`): the inverse inertia of a
    /// dynamic body is recomputed from the new child layout, exactly like
    /// at construction; static/kinematic bodies keep their zero inertia.
    /// Broad-phase bounds are *not* touched here (the body knows no
    /// proxies) — prefer `PhysicsWorld::set_child_transform`, which
    /// also refreshes the proxy immediately so sleeping/static compounds
    /// are safe to mutate. Returns `false` when the shape is not a
    /// compound.
    pub fn set_child_transform(
        &mut self,
        child: usize,
        position: Vector3,
        orientation: Quaternion,
    ) -> bool {
        let CollisionShape::Compound(compound) = &mut self.shape else {
            return false;
        };
        compound.set_child_transform(child, position, orientation);
        self.refresh_compound_inertia();
        true
    }

    /// Remove a child shape from a compound body (Bullet's
    /// `removeChildShape`), recomputing the inertia like
    /// `RigidBody::set_child_transform`. Returns the removed child, or
    /// `None` when the shape is not a compound (or the index is out of
    /// bounds; the child list is empty).
    pub fn remove_child_shape(&mut self, child: usize) -> Option<CompoundChild> {
        let CollisionShape::Compound(compound) = &mut self.shape else {
            return None;
        };
        let removed = compound.remove_child(child);
        self.refresh_compound_inertia();
        Some(removed)
    }

    /// Recompute the inverse inertia from the current compound layout
    /// (the `new_dynamic` formula); static/kinematic bodies keep zero.
    fn refresh_compound_inertia(&mut self) {
        if self.inv_mass > 0.0 {
            let mass = 1.0 / self.inv_mass;
            let inertia = match &self.shape {
                CollisionShape::Compound(c) => c.local_inertia(mass),
                _ => unreachable!("refresh_compound_inertia on a non-compound"),
            };
            self.inv_inertia_local = inv_inertia_components(inertia);
        }
    }

    /// Per-body gravity acceleration. It starts as the world gravity at
    /// `add_body` time and follows `PhysicsWorld::set_gravity` propagation
    /// while the body is awake, unless the
    /// [`CollisionFlags::DISABLE_WORLD_GRAVITY`] flag detaches it (Bullet's
    /// `btRigidBody::setGravity` + `BT_DISABLE_WORLD_GRAVITY`).
    pub fn set_gravity(&mut self, gravity: Vector3) {
        self.gravity = gravity;
    }

    /// The body's gravity acceleration (the world's value unless detached or
    /// overridden).
    pub fn gravity(&self) -> Vector3 {
        self.gravity
    }

    pub fn linear_damping(&self) -> Real {
        self.linear_damping
    }

    pub fn angular_damping(&self) -> Real {
        self.angular_damping
    }

    /// Scale the previous-step velocity by the damping factors (Bullet's
    /// `applyDamping`, run by `predictUnconstraintMotion` before the solver
    /// integrates gravity and forces). Bullet applies it to every non-static
    /// body; gating on activity here is equivalent because sleeping bodies
    /// carry zero velocity. The opt-in "additional damping" block of the
    /// same function is not ported (deviations §3).
    #[inline]
    fn apply_damping(&mut self, dt: Real) {
        // Zero factors skip the `powf`; multiplying by
        // `powf(1.0, dt) = 1.0` would be exact anyway.
        if self.linear_damping != 0.0 {
            self.linear_velocity *= math::powf(1.0 - self.linear_damping, dt);
        }
        if self.angular_damping != 0.0 {
            self.angular_velocity *= math::powf(1.0 - self.angular_damping, dt);
        }
    }

    /// Integrate velocities from gravity and accumulated forces (semi-implicit 
    /// Euler velocity update), damping the previous-step velocity first        
    /// (Bullet's `applyDamping`). Sleeping and disabled bodies are not touched;
    /// a kinematic body's velocities are user property and stay untouched too
    /// (Bullet's `integrateVelocities` static-or-kinematic gate).
    #[inline]
    pub fn integrate_velocities(&mut self, dt: Real) {
        if self.is_static || self.is_kinematic() || !self.is_active() {
            return;
        }
        self.apply_damping(dt);
        self.linear_velocity += (self.gravity + self.force * self.inv_mass) * dt;
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

    /// `set_damping` clamps both factors into `[0, 1]` (Bullet's
    /// `setDamping`).
    #[test]
    fn set_damping_clamps_factors() {
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0);
        body.set_damping(-0.5, 2.0);
        assert_eq!(body.linear_damping(), 0.0);
        assert_eq!(body.angular_damping(), 1.0);
    }

    /// Damping scales the velocity before the force integration: with factor
    /// 0.5 and 60 steps of 1/60 s the speeds halve (Bullet's `applyDamping`:
    /// `v *= (1 - d)^dt` per step).
    #[test]
    fn damping_scales_velocity_exponentially() {
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0);
        body.set_damping(0.5, 0.5);
        body.linear_velocity = Vector3::new(10.0, 0.0, 0.0);
        body.angular_velocity = Vector3::new(0.0, 0.0, 20.0);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            body.integrate_velocities(dt);
        }
        assert_relative_eq!(body.linear_velocity.x, 5.0, epsilon = 1e-4);
        assert_relative_eq!(body.angular_velocity.z, 10.0, epsilon = 1e-4);
    }
}
