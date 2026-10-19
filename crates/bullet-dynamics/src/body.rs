use bullet_math::{math, Matrix3, Quaternion, Real, Transform, Vector3};
use bullet_collision::shapes::{CollisionShape, CompoundChild};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// Seconds a body must stay below the sleeping thresholds before it sleeps
/// (Bullet's hard-coded `m_deactivationTime > btScalar(2.)` check).
pub const DEACTIVATION_TIME: Real = 2.0;

/// Bullet's `BT_LARGE_FLOAT` (btScalar.h) — precision-dependent: `1e18`
/// for f32 (kept so `BT_LARGE_FLOAT * BT_LARGE_FLOAT` stays finite),
/// `1e30` for f64.
#[cfg(not(feature = "f64"))]
const BT_LARGE_FLOAT: Real = 1e18;
#[cfg(feature = "f64")]
const BT_LARGE_FLOAT: Real = 1e30;

/// A user callback bridging the simulation pose and the render side
/// (Bullet's `btMotionState`): the world reads the start pose from it once
/// when the body is added (`getWorldTransform`) and pushes the final pose
/// back after every step (`setWorldTransform`). Store one on a body with
/// [`RigidBody::motion_state`]; a body without one pays nothing.
///
/// Mirroring Bullet's default `synchronizeMotionStates` traversal,
/// `synchronize` runs for *active* bodies only — a sleeping body's pose did
/// not change — and never for static or kinematic bodies. Unlike Bullet's
/// substep pipeline the pose handed over is the step's final value, not an
/// interpolation-time blend (single fixed step per call; recorded in
/// `docs/bullet-deviations.md`).
pub trait MotionState {
    /// The pose the body starts from (Bullet's `getWorldTransform`), read
    /// once by [`PhysicsWorld::add_body`](crate::world::PhysicsWorld::add_body)
    /// to place the body.
    fn initial_pose(&self) -> Transform;
    /// The body's final pose this step (Bullet's `setWorldTransform`).
    fn synchronize(&mut self, pose: &Transform);
}

/// The rigid pose as a rigid [`Transform`] (the f32 affine type stores the
/// translation as a `Vec3A`, so the `Vec3` position widens there).
#[inline]
pub(crate) fn pose_transform(position: Vector3, orientation: Quaternion) -> Transform {
    #[cfg(not(feature = "f64"))]
    {
        Transform::from_rotation_translation(orientation, position.into())
    }
    #[cfg(feature = "f64")]
    {
        Transform::from_rotation_translation(orientation, position)
    }
}

/// Split a pose [`Transform`] back into the rigid position/orientation pair
/// (the f32 affine type narrows its translation to the `Vec3A` position).
#[inline]
pub(crate) fn split_pose(pose: Transform) -> (Vector3, Quaternion) {
    let (_, rotation, translation) = pose.to_scale_rotation_translation();
    #[cfg(not(feature = "f64"))]
    {
        (translation.into(), rotation)
    }
    #[cfg(feature = "f64")]
    {
        (translation, rotation)
    }
}

/// Angular-motion clamp of the interpolation (Bullet's
/// `ANGULAR_MOTION_THRESHOLD = 0.5 * SIMD_HALF_PI` in `btTransformUtil.h`).
#[cfg(not(feature = "f64"))]
const ANGULAR_MOTION_THRESHOLD: Real = core::f32::consts::FRAC_PI_4;
#[cfg(feature = "f64")]
const ANGULAR_MOTION_THRESHOLD: Real = core::f64::consts::FRAC_PI_4;

/// Small-angle Taylor coefficient of the interpolation (Bullet's literal
/// `0.020833333333` ≈ 1/48 in `btTransformUtil::integrateTransform`; kept
/// at the upstream precision per scalar width).
#[cfg(not(feature = "f64"))]
const TAYLOR_COEFFICIENT: Real = 0.020_833_333;
#[cfg(feature = "f64")]
const TAYLOR_COEFFICIENT: Real = 0.020_833_333_333;

/// Extrapolate a pose linearly along the given velocities (Bullet's
/// `btTransformUtil::integrateTransform`, verbatim including the small-angle
/// Taylor branch, the `ANGULAR_MOTION_THRESHOLD` clamp and the
/// degenerate-orientation fallback). `dt` is the extrapolation span — the
/// motion-state interpolation passes a *negative* remainder (the fixed-step
/// render pose trails the last simulation snapshot), which the axis-angle
/// formula handles symmetrically.
pub(crate) fn interpolate_pose(
    position: Vector3,
    orientation: Quaternion,
    linear_velocity: Vector3,
    angular_velocity: Vector3,
    dt: Real,
) -> (Vector3, Quaternion) {
    let new_position = position + linear_velocity * dt;
    let angle_sqr = angular_velocity.length_squared();
    let mut angle = 0.0;
    if angle_sqr > Real::EPSILON {
        angle = math::sqrt(angle_sqr);
    }
    // Limit the angular motion.
    if angle * dt > ANGULAR_MOTION_THRESHOLD {
        angle = ANGULAR_MOTION_THRESHOLD / dt;
    }
    let axis = if angle < 0.001 {
        // Taylor expansion of sin(fAngle) / fAngle around zero.
        angular_velocity
            * (0.5 * dt - (dt * dt * dt) * TAYLOR_COEFFICIENT * angle * angle)
    } else {
        // sin(fAngle) / fAngle.
        angular_velocity * (math::sin(0.5 * angle * dt) / angle)
    };
    let dorn = Quaternion::from_xyzw(
        axis.x,
        axis.y,
        axis.z,
        math::cos(angle * dt * 0.5),
    );
    let predicted = dorn * orientation;
    // `safeNormalize`: a degenerate product (length² under EPSILON) yields
    // the zero quaternion, which the caller gate below rejects.
    let length_sqr = predicted.length_squared();
    let predicted = if length_sqr < Real::EPSILON {
        Quaternion::from_xyzw(0.0, 0.0, 0.0, 0.0)
    } else {
        predicted / math::sqrt(length_sqr)
    };
    if predicted.length_squared() > Real::EPSILON {
        (new_position, predicted)
    } else {
        (new_position, orientation)
    }
}

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

    /// The body's contacts anchor friction (Bullet's
    /// `CF_HAS_FRICTION_ANCHOR`): every friction row of the pair carries a
    /// positional bias that pulls the two world anchors of the cached
    /// contact back together along the tangent plane (Bullet's
    /// `setupFrictionConstraint` `BT_CONTACT_FLAG_FRICTION_ANCHOR`
    /// branch), so the body sticks to the surface it rests on instead of
    /// creeping under tangential loads. Multibody contacts ignore the flag
    /// today (deviations §1).
    pub const HAS_FRICTION_ANCHOR: Self = Self(0x0200);

    /// The body's contacts use explicit stiffness/damping soft-constraint
    /// parameters (Bullet's `CF_HAS_CONTACT_STIFFNESS_DAMPING`): the pair's
    /// normal row swaps its Baumgarte factor and adds a compliance CFM
    /// derived from the combined contact stiffness/damping (see
    /// [`RigidBody::set_contact_stiffness_and_damping`]). Set by that
    /// setter; the flag itself is never cleared (Bullet does not either).
    pub const HAS_CONTACT_STIFFNESS_DAMPING: Self = Self(0x0080);

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no flag is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Reciprocal local inertia components (the `new_dynamic` formula). A zero
/// component maps to zero inverse inertia — the axis is locked, like
/// Bullet's `setMassProps` — because inverting an epsilon instead gives a
/// ~8e6 inverse inertia that turns any residual numerical torque into a
/// violent spin.
fn inv_inertia_components(inertia: Vector3) -> Vector3 {
    Vector3::new(
        if inertia.x != 0.0 { 1.0 / inertia.x } else { 0.0 },
        if inertia.y != 0.0 { 1.0 / inertia.y } else { 0.0 },
        if inertia.z != 0.0 { 1.0 / inertia.z } else { 0.0 },
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
    /// Direction-dependent friction scaling in the body's local axes
    /// (Bullet's `m_anisotropicFriction`, default `(1, 1, 1)` = isotropic):
    /// each friction row's tangent direction is rotated into the body's
    /// local frame, scaled component-wise, and rotated back
    /// (`btSequentialImpulseConstraintSolver::applyAnisotropicFriction`),
    /// for both sides of the pair. Assign directly; a non-unity component
    /// (Bullet's unity gate) switches the effect on, matching
    /// `setAnisotropicFriction`'s `CF_ANISOTROPIC_FRICTION` mode. Multibody
    /// contacts ignore the field today (deviations §1).
    pub anisotropic_friction: Vector3,
    /// Torsional friction orthogonal to the contact normal (Bullet's
    /// `m_rollingFriction`, default 0): a rolling body touching a surface
    /// with a positive pair value is slowed by angular-friction rows until
    /// it stops, instead of rolling forever. Only read in pairs where the
    /// combined value (`rf_a · friction_b + rf_b · friction_a`, Bullet's
    /// `calculateCombinedRollingFriction` quirk) exceeds zero.
    pub rolling_friction: Real,
    /// Torsional friction around the contact normal (Bullet's
    /// `m_spinningFriction`, default 0): spins a touching body down about
    /// the normal axis, clamped like the rolling rows. Like Bullet's
    /// `convertContact`, the row is installed inside the rolling-friction
    /// branch — a positive pair value has no effect unless the pair's
    /// combined rolling friction is positive too (upstream quirk, kept
    /// verbatim). Combined with the same quirk formula as rolling
    /// (`sf_a · friction_b + sf_b · friction_a`, clamped ±10).
    pub spinning_friction: Real,
    /// Soft-constraint parameters of this body's contacts (Bullet's
    /// `m_contactStiffness` / `m_contactDamping`, defaults `BT_LARGE_FLOAT`
    /// / `0.1`): only read when either side of a pair carries
    /// [`CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING`]. Written through
    /// [`RigidBody::set_contact_stiffness_and_damping`] (which clamps the
    /// stiffness away from zero like Bullet and sets the flag), read via
    /// [`RigidBody::contact_stiffness`] / [`RigidBody::contact_damping`].
    contact_stiffness: Real,
    contact_damping: Real,
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
    /// The opt-in "additional damping" block of Bullet's `applyDamping`
    /// (Bullet's `m_additionalDamping` plus the four construction-info
    /// scalars; defaults factor 0.005, both thresholds² 0.01, angular
    /// factor 0.01 — Bullet exposes no setters after construction, so this
    /// crate gates them behind the builder too).
    additional_damping: bool,
    additional_damping_factor: Real,
    additional_linear_damping_threshold_sqr: Real,
    additional_angular_damping_threshold_sqr: Real,
    additional_angular_damping_factor: Real,
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
    /// Hit fraction of the last step's CCD motion clamp (Bullet's
    /// `m_hitFraction`, reset to 1 at the start of every integration and
    /// set to 0 on a clamped hit): the motion-state interpolation factor
    /// scales with it, so a clamped body's render pose freezes at the
    /// pre-integration snapshot until the impact is resolved.
    pub hit_fraction: Real,
    /// Split-impulse positional correction accumulated by the solver this
    /// step (Bullet's `m_pushVelocity` / `m_turnVelocity`): a pseudo velocity
    /// that moves the body out of deep penetration without changing its real
    /// velocity. Applied to the pose by the world after position integration,
    /// then cleared.
    pub(crate) push_linear: Vector3,
    pub(crate) push_angular: Vector3,
    /// Optional render-side pose callback (Bullet's `m_optionalMotionState`);
    /// see [`MotionState`].
    pub(crate) motion_state: Option<Box<dyn MotionState>>,
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
            anisotropic_friction: Vector3::ONE,
            rolling_friction: 0.0,
            spinning_friction: 0.0,
            contact_stiffness: BT_LARGE_FLOAT,
            contact_damping: 0.1,
            collision_group: filter_groups::DEFAULT,
            collision_mask: filter_groups::ALL,
            collision_flags: CollisionFlags::default(),
            collision_events: false,
            gravity: Vector3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            additional_damping: false,
            additional_damping_factor: 0.005,
            additional_linear_damping_threshold_sqr: 0.01,
            additional_angular_damping_threshold_sqr: 0.01,
            additional_angular_damping_factor: 0.01,
            is_static: false,
            activation_state: ActivationState::Active,
            deactivation_time: 0.0,
            removed: false,
            linear_sleeping_threshold: 0.8,
            angular_sleeping_threshold: 1.0,
            ccd_motion_threshold: 0.0,
            ccd_swept_sphere_radius: 0.0,
            hit_fraction: 1.0,
            push_linear: Vector3::ZERO,
            push_angular: Vector3::ZERO,
            motion_state: None,
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
            anisotropic_friction: Vector3::ONE,
            rolling_friction: 0.0,
            spinning_friction: 0.0,
            contact_stiffness: BT_LARGE_FLOAT,
            contact_damping: 0.1,
            collision_group: filter_groups::DEFAULT,
            collision_mask: filter_groups::ALL,
            collision_flags: CollisionFlags::default(),
            collision_events: false,
            gravity: Vector3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            additional_damping: false,
            additional_damping_factor: 0.005,
            additional_linear_damping_threshold_sqr: 0.01,
            additional_angular_damping_threshold_sqr: 0.01,
            additional_angular_damping_factor: 0.01,
            is_static: true,
            activation_state: ActivationState::IslandSleeping,
            deactivation_time: 0.0,
            removed: false,
            linear_sleeping_threshold: 0.8,
            angular_sleeping_threshold: 1.0,
            ccd_motion_threshold: 0.0,
            ccd_swept_sphere_radius: 0.0,
            hit_fraction: 1.0,
            push_linear: Vector3::ZERO,
            push_angular: Vector3::ZERO,
            motion_state: None,
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
        // Bullet's `btCollisionObject::setActivationState` guard: a
        // `DisableSimulation` body's state is user-owned and is never
        // rewritten by the island pass (it still merges islands, so this is
        // reachable from the sleep branch of the activation resolution).
        if self.activation_state == ActivationState::DisableSimulation {
            return;
        }
        self.activation_state = ActivationState::IslandSleeping;
        self.linear_velocity = Vector3::ZERO;
        self.angular_velocity = Vector3::ZERO;
    }
    /// Whether this body is an island member: dynamic bodies participate
    /// (including `DisableSimulation` — Bullet's
    /// `STATIC_SIMULATION_ISLAND_OPTIMIZATION` tag pass only skips
    /// static/kinematic objects, so a disabled body still merges islands and
    /// can bridge its neighbors); static and kinematic bodies do not.
    pub(crate) fn participates_in_islands(&self) -> bool {
        !self.removed && !self.is_static && !self.is_kinematic()
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

    /// Builder: set the direction-dependent friction scaling
    /// ([`RigidBody::anisotropic_friction`]).
    pub fn anisotropic_friction(mut self, scaling: Vector3) -> Self {
        self.anisotropic_friction = scaling;
        self
    }

    /// Builder: set the contacts' soft-constraint parameters (see
    /// [`RigidBody::set_contact_stiffness_and_damping`]).
    pub fn contact_stiffness_and_damping(mut self, stiffness: Real, damping: Real) -> Self {
        self.set_contact_stiffness_and_damping(stiffness, damping);
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

    /// Builder: attach a motion state (Bullet's
    /// `btRigidBodyConstructionInfo::m_motionState`).
    /// [`PhysicsWorld::add_body`](crate::world::PhysicsWorld::add_body)
    /// places the body at [`MotionState::initial_pose`] and the step hands
    /// the final pose back via [`MotionState::synchronize`].
    pub fn motion_state(mut self, state: Box<dyn MotionState>) -> Self {
        self.motion_state = Some(state);
        self
    }

    /// Replace or clear the motion state (Bullet's `setMotionState`).
    /// Setting one does not re-place the body; the pose it reports feeds
    /// the next [`MotionState::synchronize`] round trip only for bodies
    /// added afterwards.
    pub fn set_motion_state(&mut self, state: Option<Box<dyn MotionState>>) {
        self.motion_state = state;
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

    /// Builder: opt this body into the "additional damping" block of
    /// [`RigidBody::apply_damping`]
    /// (`btRigidBodyConstructionInfo::m_additionalDamping`) — a
    /// humanoid/ragdoll-stack tuning safety valve that shrinks velocities
    /// below per-step thresholds by an extra factor. Bullet offers no
    /// post-construction setter for this (construction info only), so this
    /// crate gates it behind the builder as well.
    pub fn additional_damping(mut self, on: bool) -> Self {
        self.additional_damping = on;
        self
    }

    /// Builder: override the four additional-damping scalars (Bullet's
    /// construction-info fields, which also carry no setters). Defaults:
    /// `factor = 0.005`, `linear_threshold_sqr = 0.01`,
    /// `angular_threshold_sqr = 0.01`, `angular_factor = 0.01`. Upstream
    /// quirk, mirrored: `applyDamping` never reads the angular factor —
    /// both velocities scale by `factor` — so the last parameter only
    /// rides along like Bullet's.
    pub fn additional_damping_params(
        mut self,
        factor: Real,
        linear_threshold_sqr: Real,
        angular_threshold_sqr: Real,
        angular_factor: Real,
    ) -> Self {
        self.additional_damping_factor = factor;
        self.additional_linear_damping_threshold_sqr = linear_threshold_sqr;
        self.additional_angular_damping_threshold_sqr = angular_threshold_sqr;
        self.additional_angular_damping_factor = angular_factor;
        self
    }

    /// Soft-constraint parameters for this body's contacts (Bullet's
    /// `setContactStiffnessAndDamping`): every contact pair involving this
    /// body replaces its Baumgarte factor and adds a compliance CFM derived
    /// from the pair's combined stiffness/damping (damping adds, stiffness
    /// combines like springs in series), turning the rigid contact into a
    /// damped spring along the normal. The stiffness is clamped away from
    /// zero like Bullet (below `SIMD_EPSILON` → `SIMD_EPSILON`) and the
    /// [`CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING`] flag is set —
    /// Bullet never clears it. Multibody contacts ignore the parameters
    /// today (deviations §1).
    pub fn set_contact_stiffness_and_damping(&mut self, stiffness: Real, damping: Real) {
        self.contact_stiffness = stiffness;
        self.contact_damping = damping;
        self.collision_flags.0 |= CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING.0;
        // "avoid divisions by zero..." (btCollisionObject.h).
        if self.contact_stiffness < Real::EPSILON {
            self.contact_stiffness = Real::EPSILON;
        }
    }

    /// The soft-constraint stiffness of this body's contacts (Bullet's
    /// `getContactStiffness`; see
    /// [`RigidBody::set_contact_stiffness_and_damping`]).
    #[inline]
    pub fn contact_stiffness(&self) -> Real {
        self.contact_stiffness
    }

    /// The soft-constraint damping of this body's contacts (Bullet's
    /// `getContactDamping`; see
    /// [`RigidBody::set_contact_stiffness_and_damping`]).
    #[inline]
    pub fn contact_damping(&self) -> Real {
        self.contact_damping
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
    /// same function (the ragdoll-stack safety valve, see
    /// [`RigidBody::additional_damping`]) runs verbatim when enabled —
    /// including the upstream quirks of comparing the speeds against the
    /// damping factors themselves (`speed < m_linearDamping`) and the
    /// hardcoded `0.005` subtraction rates.
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
        if !self.additional_damping {
            return;
        }
        // "Additional damping can help avoiding lowpass jitter motion, help
        // stability for ragdolls etc." — btRigidBody.cpp, verbatim.
        if self.angular_velocity.length_squared() < self.additional_angular_damping_threshold_sqr
            && self.linear_velocity.length_squared() < self.additional_linear_damping_threshold_sqr
        {
            self.angular_velocity *= self.additional_damping_factor;
            self.linear_velocity *= self.additional_damping_factor;
        }
        let damp_vel: Real = 0.005;
        let speed = self.linear_velocity.length();
        if speed < self.linear_damping {
            if speed > damp_vel {
                let dir = self.linear_velocity / self.linear_velocity.length();
                self.linear_velocity -= dir * damp_vel;
            } else {
                self.linear_velocity = Vector3::ZERO;
            }
        }
        let ang_speed = self.angular_velocity.length();
        if ang_speed < self.angular_damping {
            if ang_speed > damp_vel {
                let dir = self.angular_velocity / self.angular_velocity.length();
                self.angular_velocity -= dir * damp_vel;
            } else {
                self.angular_velocity = Vector3::ZERO;
            }
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

    /// A zero inertia component maps to a zero inverse inertia (the axis is
    /// locked, Bullet's `setMassProps` ternary) instead of inverting an
    /// epsilon into a ~8e6 inverse inertia.
    #[test]
    fn zero_inertia_components_lock_their_axis() {
        let inv = inv_inertia_components(Vector3::new(2.0, 0.0, 4.0));
        assert_relative_eq!(inv.x, 0.5);
        assert_eq!(inv.y, 0.0);
        assert_relative_eq!(inv.z, 0.25);
    }

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

    /// The additional-damping block: velocities below both thresholds²
    /// (0.01) shrink by the factor every step, velocities above stay
    /// untouched, and the flag gates it off entirely (Bullet's
    /// `applyDamping` construction-info defaults). Bullet scales *both*
    /// velocities by `m_additionalDampingFactor` here — its
    /// `m_additionalAngularDampingFactor` field is never read by the
    /// function, and the port mirrors that.
    #[test]
    fn additional_damping_shrinks_low_speeds() {
        let dt = 1.0 / 60.0;
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0)
            .additional_damping(true);
        body.linear_velocity = Vector3::new(0.05, 0.0, 0.0);
        body.angular_velocity = Vector3::new(0.0, 0.02, 0.0);
        body.integrate_velocities(dt);
        assert_relative_eq!(body.linear_velocity.x, 0.05 * 0.005, epsilon = 1e-6);
        assert_relative_eq!(body.angular_velocity.y, 0.02 * 0.005, epsilon = 1e-6);
        // Above both thresholds the block does nothing.
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0)
            .additional_damping(true);
        body.linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        body.angular_velocity = Vector3::new(0.0, 0.5, 0.0);
        body.integrate_velocities(dt);
        assert_relative_eq!(body.linear_velocity.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(body.angular_velocity.y, 0.5, epsilon = 1e-6);
        // Without the flag nothing runs.
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0);
        body.linear_velocity = Vector3::new(0.05, 0.0, 0.0);
        body.integrate_velocities(dt);
        assert_relative_eq!(body.linear_velocity.x, 0.05, epsilon = 1e-6);
    }

    /// The verbatim subtract quirks: with large damping factors the block
    /// compares the speeds against the factors themselves and removes a
    /// hardcoded 0.005 per step (zeroing below it) — Bullet compares
    /// against `m_linearDamping`, not a threshold. The pow damping runs
    /// first, so the expectations below carry its factor.
    #[test]
    fn additional_damping_subtract_block_follows_the_upstream_quirk() {
        let dt = 1.0 / 60.0;
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0)
            .additional_damping(true);
        body.set_damping(0.2, 0.2);
        body.linear_velocity = Vector3::new(0.04, 0.0, 0.0);
        body.angular_velocity = Vector3::new(0.0, 0.0, 0.15);
        // After the pow pass the angular speed stays above the threshold²
        // (0.01), which skips the factor block (the AND gate); both speeds
        // sit under their damping factors and above the 0.005 cut, so both
        // lose exactly 0.005.
        let pow = bullet_math::math::powf(1.0 - 0.2, dt);
        body.integrate_velocities(dt);
        assert_relative_eq!(body.linear_velocity.x, 0.04 * pow - 0.005, epsilon = 1e-5);
        assert_relative_eq!(body.angular_velocity.z, 0.15 * pow - 0.005, epsilon = 1e-5);
        // A speed below the 0.005 cut zeroes out instead of going negative
        // (the factor block shrinks it further below the cut first).
        let mut body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::ZERO, 1.0)
            .additional_damping(true);
        body.set_damping(0.2, 0.2);
        body.linear_velocity = Vector3::new(0.002, 0.0, 0.0);
        body.integrate_velocities(dt);
        assert_eq!(body.linear_velocity, Vector3::ZERO);
    }
}
