//! Raycast vehicle — port of Bullet's `btRaycastVehicle`
//! (`BulletDynamics/Vehicle`).
//!
//! The vehicle turns an ordinary rigid body (the chassis) into a car: each
//! wheel is a ray cast down from a connection point on the chassis; the ray
//! hits the ground and a spring-damper suspension pushes the chassis away
//! from the contact point. Steering rotates the front wheels around the up
//! axis, engine force / braking apply impulses along the wheels' forward
//! direction, and a lateral-friction impulse kills sideways sliding.
//!
//! Usage mirrors the Bullet action pattern, like the character controller:
//! create the chassis body, build the vehicle, then per frame
//!
//! ```ignore
//! vehicle.set_steering_value(steering, wheel);
//! vehicle.apply_engine_force(force, wheel);
//! world.step_simulation(dt);
//! vehicle.update_vehicle(&mut world, dt);
//! ```
//!
//! [`RaycastVehicle::update_vehicle`] must run *after*
//! [`PhysicsWorld::step_simulation`] (Bullet's actions run after
//! `integrateTransforms`): the wheel rays are cast from the fresh chassis
//! pose and the suspension/friction impulses land on the chassis velocity,
//! which the next step integrates. Keep the chassis active
//! (`RigidBody::disable_deactivation`) — a sleeping chassis freezes the car.
//!
//! Known deviations from Bullet (see `docs/bullet-deviations.md`): the wheel
//! rays explicitly exclude the chassis body (Bullet misses interior-start
//! rays; we report them, deviation §1); the ground is always treated as the
//! fixed (static) body, so driving on dynamic objects is not supported —
//! exactly like Bullet's `m_groundObject = &getFixedBody()` `@todo`; and
//! [`RaycastVehicle::reset_suspension`] refreshes the wheel directions from
//! the current chassis pose instead of reusing stale state.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::{math, Matrix3, Quaternion, Real, Transform, Vector3};
use bullet_math::compat::{quat_from_axis_angle, transform_from_mat3_translation};

use crate::body::RigidBody;
use crate::handles::BodyHandle;
use crate::world::PhysicsWorld;

/// Per-vehicle tuning values (Bullet's `btRaycastVehicle::btVehicleTuning`,
/// same defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleTuning {
    /// Spring constant of the suspension, as an acceleration per metre of
    /// compression (Bullet `m_suspensionStiffness = 5.88`).
    pub suspension_stiffness: Real,
    /// Damper coefficient used while the suspension is compressing
    /// (Bullet `m_suspensionCompression = 0.83`).
    pub suspension_compression: Real,
    /// Damper coefficient used while the suspension is extending
    /// (Bullet `m_suspensionDamping = 0.88`).
    pub suspension_damping: Real,
    /// Maximum suspension travel in centimetres (Bullet
    /// `m_maxSuspensionTravelCm = 500`).
    pub max_suspension_travel_cm: Real,
    /// Tyre friction (Bullet `m_frictionSlip = 10.5`): scales the maximum
    /// forward/lateral impulse the ground can deliver per wheel.
    pub friction_slip: Real,
    /// Cap on the suspension force per wheel (Bullet
    /// `m_maxSuspensionForce = 6000`).
    pub max_suspension_force: Real,
}

impl Default for VehicleTuning {
    fn default() -> Self {
        Self {
            suspension_stiffness: 5.88,
            suspension_compression: 0.83,
            suspension_damping: 0.88,
            max_suspension_travel_cm: 500.0,
            friction_slip: 10.5,
            max_suspension_force: 6000.0,
        }
    }
}

/// Per-wheel raycast results, filled by [`RaycastVehicle::update_vehicle`]
/// (Bullet's `btWheelInfo::RaycastInfo`).
#[derive(Debug, Clone, Copy)]
pub struct RaycastInfo {
    /// World-space contact normal at the ground hit, pointing away from the
    /// ground (toward the wheel).
    pub contact_normal_ws: Vector3,
    /// World-space ground hit point.
    pub contact_point_ws: Vector3,
    /// Current suspension length (distance from the connection point to the
    /// wheel centre), clamped to the travel limits.
    pub suspension_length: Real,
    /// World-space ray origin: the wheel's connection point on the chassis.
    pub hard_point_ws: Vector3,
    /// Wheel ray direction in world space (usually down).
    pub wheel_direction_ws: Vector3,
    /// Wheel axle direction in world space (steering does not change it).
    pub wheel_axle_ws: Vector3,
    /// Whether the wheel ray hit anything this step.
    pub is_in_contact: bool,
    /// Handle of the hit body, when in contact. The ground is treated as the
    /// fixed body (see the module docs), so this is informational only.
    pub ground_body: Option<BodyHandle>,
}

/// One wheel of a [`RaycastVehicle`] (Bullet's `btWheelInfo`): the constant
/// connection geometry plus the per-step raycast, suspension and drive state.
/// All fields are public so the vehicle can be retuned after construction,
/// like Bullet's demos do (`wheel.m_suspensionStiffness = ...`).
#[derive(Debug, Clone)]
pub struct WheelInfo {
    /// The per-step raycast results.
    pub raycast_info: RaycastInfo,
    /// Wheel world pose (rendering): basis + origin.
    pub world_basis: Matrix3,
    pub world_origin: Vector3,
    /// Connection point of the suspension on the chassis, in chassis-local
    /// coordinates (constant).
    pub chassis_connection_point_cs: Vector3,
    /// Wheel ray direction in chassis-local coordinates (constant; usually
    /// `(0, -1, 0)`).
    pub wheel_direction_cs: Vector3,
    /// Wheel axle direction in chassis-local coordinates (constant; usually
    /// `(-1, 0, 0)`).
    pub wheel_axle_cs: Vector3,
    /// Suspension length when the wheel hangs freely (constant).
    pub suspension_rest_length: Real,
    /// Maximum suspension travel in centimetres (Bullet
    /// `m_maxSuspensionTravelCm`).
    pub max_suspension_travel_cm: Real,
    /// Wheel radius (constant).
    pub wheel_radius: Real,
    /// Spring constant of this wheel's suspension (Bullet
    /// `m_suspensionStiffness`).
    pub suspension_stiffness: Real,
    /// Damper coefficient while compressing (Bullet
    /// `m_wheelsDampingCompression`).
    pub wheels_damping_compression: Real,
    /// Damper coefficient while extending (Bullet `m_wheelsDampingRelaxation`).
    pub wheels_damping_relaxation: Real,
    /// Tyre friction (Bullet `m_frictionSlip`).
    pub friction_slip: Real,
    /// Current steering angle around the up axis (Bullet `m_steering`).
    pub steering: Real,
    /// Accumulated wheel spin angle (Bullet `m_rotation`).
    pub rotation: Real,
    /// Spin angle added this step (Bullet `m_deltaRotation`).
    pub delta_rotation: Real,
    /// How much of the lateral impulse is applied at the chassis vs. rolling
    /// the body over (Bullet `m_rollInfluence = 0.1`).
    pub roll_influence: Real,
    /// Cap on the suspension force of this wheel (Bullet
    /// `m_maxSuspensionForce`).
    pub max_suspension_force: Real,
    /// Engine force applied at this wheel this step (Bullet `m_engineForce`).
    pub engine_force: Real,
    /// Braking impulse magnitude (Bullet `m_brake`).
    pub brake: Real,
    /// Whether this wheel steers (Bullet `m_bIsFrontWheel`).
    pub is_front_wheel: bool,
    /// `1 / max(0.1, -n·d)`: scales the spring force when the contact normal
    /// is nearly parallel to the wheel direction (Bullet
    /// `m_clippedInvContactDotSuspension`).
    pub clipped_inv_contact_dot_suspension: Real,
    /// Chassis velocity along the contact normal, divided by `-n·d`
    /// (Bullet `m_suspensionRelativeVelocity`); drives the damper.
    pub suspension_relative_velocity: Real,
    /// Spring-damper force of this wheel for the current step, already scaled
    /// by the chassis mass (Bullet `m_wheelsSuspensionForce`).
    pub wheels_suspension_force: Real,
    /// Friction scaling during sliding (Bullet `m_skidInfo`).
    pub skid_info: Real,
}

impl WheelInfo {
    fn new(
        connection_point_cs: Vector3,
        wheel_direction_cs: Vector3,
        wheel_axle_cs: Vector3,
        suspension_rest_length: Real,
        wheel_radius: Real,
        tuning: &VehicleTuning,
        is_front_wheel: bool,
    ) -> Self {
        Self {
            raycast_info: RaycastInfo {
                contact_normal_ws: Vector3::ZERO,
                contact_point_ws: Vector3::ZERO,
                suspension_length: suspension_rest_length,
                hard_point_ws: Vector3::ZERO,
                wheel_direction_ws: Vector3::ZERO,
                wheel_axle_ws: Vector3::ZERO,
                is_in_contact: false,
                ground_body: None,
            },
            world_basis: Matrix3::IDENTITY,
            world_origin: Vector3::ZERO,
            chassis_connection_point_cs: connection_point_cs,
            wheel_direction_cs,
            wheel_axle_cs,
            suspension_rest_length,
            max_suspension_travel_cm: tuning.max_suspension_travel_cm,
            wheel_radius,
            suspension_stiffness: tuning.suspension_stiffness,
            wheels_damping_compression: tuning.suspension_compression,
            wheels_damping_relaxation: tuning.suspension_damping,
            friction_slip: tuning.friction_slip,
            steering: 0.0,
            rotation: 0.0,
            delta_rotation: 0.0,
            roll_influence: 0.1,
            max_suspension_force: tuning.max_suspension_force,
            engine_force: 0.0,
            brake: 0.0,
            is_front_wheel,
            clipped_inv_contact_dot_suspension: 1.0,
            suspension_relative_velocity: 0.0,
            wheels_suspension_force: 0.0,
            skid_info: 1.0,
        }
    }

    /// The wheel's world-space pose for rendering (Bullet's
    /// `m_worldTransform`).
    pub fn world_transform(&self) -> Transform {
        transform_from_mat3_translation(self.world_basis, self.world_origin)
    }
}

/// A raycast vehicle (Bullet's `btRaycastVehicle`): a chassis rigid body
/// carried by suspension raycasts, with steering, engine and brakes.
///
/// See the [module docs](self) for the usage pattern and deviations.
pub struct RaycastVehicle {
    /// The chassis body handle in the world.
    chassis: BodyHandle,
    wheels: Vec<WheelInfo>,
    /// Per-step forward friction direction/impulse per wheel (Bullet's
    /// `m_forwardWS` / `m_forwardImpulse`).
    forward_ws: Vec<Vector3>,
    forward_impulse: Vec<Real>,
    /// Per-step lateral friction direction/impulse per wheel (Bullet's
    /// `m_axle` / `m_sideImpulse`).
    axle: Vec<Vector3>,
    side_impulse: Vec<Real>,
    /// Speed readout in km/h, negative when moving backwards (Bullet's
    /// `m_currentVehicleSpeedKmHour`).
    current_vehicle_speed_km_hour: Real,
    /// Chassis-local axis indices: which of X/Y/Z is right / up / forward
    /// (Bullet's `m_indexRightAxis` etc.). Defaults match Bullet's
    /// constructor: right = X(0), up = Z(2), forward = Y(1); Y-up vehicles
    /// should call [`set_coordinate_system`](Self::set_coordinate_system)
    /// with `(0, 1, 2)`.
    right_axis: usize,
    up_axis: usize,
    forward_axis: usize,
}

impl RaycastVehicle {
    /// Create a vehicle for the chassis body `chassis` (which must already
    /// exist in the world).
    pub fn new(chassis: BodyHandle) -> Self {
        Self {
            chassis,
            wheels: Vec::new(),
            forward_ws: Vec::new(),
            forward_impulse: Vec::new(),
            axle: Vec::new(),
            side_impulse: Vec::new(),
            current_vehicle_speed_km_hour: 0.0,
            right_axis: 0,
            up_axis: 2,
            forward_axis: 1,
        }
    }

    /// Choose which chassis axes are right / up / forward (Bullet's
    /// `setCoordinateSystem`). A Y-up car uses `(0, 1, 2)`.
    pub fn set_coordinate_system(&mut self, right_index: usize, up_index: usize, forward_index: usize) {
        self.right_axis = right_index;
        self.up_axis = up_index;
        self.forward_axis = forward_index;
    }

    pub fn right_axis(&self) -> usize {
        self.right_axis
    }

    pub fn up_axis(&self) -> usize {
        self.up_axis
    }

    pub fn forward_axis(&self) -> usize {
        self.forward_axis
    }

    /// Handle of the chassis body (Bullet's `getRigidBody`).
    pub fn chassis_body(&self) -> BodyHandle {
        self.chassis
    }

    /// Add a wheel (Bullet's `addWheel`). `connection_point_cs` is where the
    /// suspension attaches to the chassis (chassis-local), `wheel_direction_cs`
    /// the ray direction (usually down) and `wheel_axle_cs` the axle direction
    /// (chassis-local). The wheel's world pose is initialized from the current
    /// chassis pose.
    #[allow(clippy::too_many_arguments)] // mirrors Bullet's addWheel signature
    pub fn add_wheel(
        &mut self,
        world: &PhysicsWorld,
        connection_point_cs: Vector3,
        wheel_direction_cs: Vector3,
        wheel_axle_cs: Vector3,
        suspension_rest_length: Real,
        wheel_radius: Real,
        tuning: &VehicleTuning,
        is_front_wheel: bool,
    ) -> usize {
        self.wheels.push(WheelInfo::new(
            connection_point_cs,
            wheel_direction_cs,
            wheel_axle_cs,
            suspension_rest_length,
            wheel_radius,
            tuning,
            is_front_wheel,
        ));
        let index = self.wheels.len() - 1;
        self.update_wheel_transforms_ws(world, index);
        self.update_wheel_transform(world, index);
        index
    }

    pub fn num_wheels(&self) -> usize {
        self.wheels.len()
    }

    pub fn wheel_info(&self, index: usize) -> &WheelInfo {
        &self.wheels[index]
    }

    pub fn wheel_info_mut(&mut self, index: usize) -> &mut WheelInfo {
        &mut self.wheels[index]
    }

    /// Set the steering angle of `wheel` around the up axis (Bullet's
    /// `setSteeringValue`).
    pub fn set_steering_value(&mut self, steering: Real, wheel: usize) {
        self.wheels[wheel].steering = steering;
    }

    pub fn get_steering_value(&self, wheel: usize) -> Real {
        self.wheels[wheel].steering
    }

    /// Apply an engine force to `wheel` (positive = forward, negative =
    /// reverse; Bullet's `applyEngineForce`). The force is only delivered
    /// while the wheel is in contact.
    pub fn apply_engine_force(&mut self, force: Real, wheel: usize) {
        self.wheels[wheel].engine_force = force;
    }

    /// Set the braking impulse magnitude of `wheel` (Bullet's `setBrake`).
    pub fn set_brake(&mut self, brake: Real, wheel_index: usize) {
        self.wheels[wheel_index].brake = brake;
    }

    /// Speed readout: `3.6 · |v|` km/h, negative when moving backwards
    /// (Bullet's `getCurrentSpeedKmHour`).
    pub fn current_speed_km_hour(&self) -> Real {
        self.current_vehicle_speed_km_hour
    }

    /// The chassis forward axis in world space (Bullet's `getForwardVector`).
    pub fn forward_vector(&self, world: &PhysicsWorld) -> Vector3 {
        let chassis = world.body(self.chassis);
        axis_unit_vector(chassis.orientation, self.forward_axis)
    }

    /// Reset the suspension to its rest state (Bullet's `resetSuspension`):
    /// wheels at rest length, no damper velocity. Refreshes the wheel
    /// directions from the current chassis pose first (Bullet reuses stale
    /// world-space state; see the module docs).
    pub fn reset_suspension(&mut self, world: &PhysicsWorld) {
        for i in 0..self.wheels.len() {
            self.update_wheel_transforms_ws(world, i);
            let wheel = &mut self.wheels[i];
            wheel.raycast_info.suspension_length = wheel.suspension_rest_length;
            wheel.suspension_relative_velocity = 0.0;
            wheel.raycast_info.contact_normal_ws = -wheel.raycast_info.wheel_direction_ws;
            wheel.clipped_inv_contact_dot_suspension = 1.0;
        }
    }

    /// Advance the vehicle by `dt` seconds (Bullet's `updateVehicle`).
    /// Call after [`PhysicsWorld::step_simulation`]: the wheel rays are cast
    /// from the freshly integrated chassis pose and the resulting suspension
    /// and friction impulses land on the chassis velocity for the next step.
    pub fn update_vehicle(&mut self, world: &mut PhysicsWorld, dt: Real) {
        // Refresh the wheel poses from the current chassis transform (used by
        // updateFriction for the axle/forward directions).
        for i in 0..self.wheels.len() {
            self.update_wheel_transform(world, i);
        }

        // Speed readout: 3.6·|v| km/h, negative when driving backwards.
        let chassis = world.body(self.chassis);
        let mut speed: Real = 3.6 * chassis.linear_velocity.length();
        let forward_w = axis_unit_vector(chassis.orientation, self.forward_axis);
        if forward_w.dot(chassis.linear_velocity) < 0.0 {
            speed = -speed;
        }
        self.current_vehicle_speed_km_hour = speed;

        // Simulate the suspension: raycast every wheel from the fresh pose.
        for i in 0..self.wheels.len() {
            self.ray_cast(world, i);
        }
        self.update_suspension(world);

        // Apply the suspension forces as impulses at the contact points
        // (clamped to the per-wheel maximum), raising the chassis.
        let chassis_pos = world.body(self.chassis).position;
        for i in 0..self.wheels.len() {
            let wheel = &self.wheels[i];
            let mut suspension_force = wheel.wheels_suspension_force;
            if suspension_force > wheel.max_suspension_force {
                suspension_force = wheel.max_suspension_force;
            }
            let impulse = wheel.raycast_info.contact_normal_ws * (suspension_force * dt);
            let rel_pos = wheel.raycast_info.contact_point_ws - chassis_pos;
            world.body_mut(self.chassis).apply_impulse(impulse, rel_pos);
        }

        // Longitudinal + lateral tyre friction.
        self.update_friction(world, dt);

        // Advance the wheel spin from the chassis velocity at the wheel's
        // hard point (rolling while in contact; the spin decays otherwise).
        for i in 0..self.wheels.len() {
            let chassis = world.body(self.chassis);
            let rel_pos = self.wheels[i].raycast_info.hard_point_ws - chassis.position;
            let vel = chassis.velocity_at_local_point(rel_pos);
            let wheel = &mut self.wheels[i];
            if wheel.raycast_info.is_in_contact {
                let mut fwd = axis_unit_vector(chassis.orientation, self.forward_axis);
                let proj = fwd.dot(wheel.raycast_info.contact_normal_ws);
                fwd -= wheel.raycast_info.contact_normal_ws * proj;
                let proj2 = fwd.dot(vel);
                wheel.delta_rotation = (proj2 * dt) / wheel.wheel_radius;
                wheel.rotation += wheel.delta_rotation;
            } else {
                wheel.rotation += wheel.delta_rotation;
            }
            wheel.delta_rotation *= 0.99; // Damping of rotation when not in contact.
        }
    }

    /// Recompute the wheel's world-space ray origin/direction from the current
    /// chassis pose (Bullet's `updateWheelTransformsWS`). Also clears the
    /// contact flag — the raycast that follows re-sets it.
    fn update_wheel_transforms_ws(&mut self, world: &PhysicsWorld, wheel_index: usize) {
        let chassis = world.body(self.chassis);
        let wheel = &mut self.wheels[wheel_index];
        wheel.raycast_info.is_in_contact = false;
        wheel.raycast_info.hard_point_ws =
            chassis.position + chassis.orientation * wheel.chassis_connection_point_cs;
        wheel.raycast_info.wheel_direction_ws = chassis.orientation * wheel.wheel_direction_cs;
        wheel.raycast_info.wheel_axle_ws = chassis.orientation * wheel.wheel_axle_cs;
    }

    /// Rebuild the wheel's world pose: origin at the suspension end point,
    /// basis aligned to the (steered) axle / up / forward (Bullet's
    /// `updateWheelTransform`).
    fn update_wheel_transform(&mut self, world: &PhysicsWorld, wheel_index: usize) {
        self.update_wheel_transforms_ws(world, wheel_index);
        let wheel = &mut self.wheels[wheel_index];
        let up = -wheel.raycast_info.wheel_direction_ws;
        let right = wheel.raycast_info.wheel_axle_ws;
        let fwd = up.cross(right).normalize();

        // Steering rotates around the up axis; the spin around the axle.
        let steering_quat = quat_from_axis_angle(up.normalize(), wheel.steering);
        let rotating_quat = quat_from_axis_angle(right.normalize(), -wheel.rotation);

        // Chassis-axes basis: the wheel's right/up/forward columns hold
        // -right / up / fwd (Bullet's `basis2`).
        let (mut cx, mut cy, mut cz) = (Vector3::ZERO, Vector3::ZERO, Vector3::ZERO);
        match self.right_axis {
            0 => cx = -right,
            1 => cy = -right,
            _ => cz = -right,
        }
        match self.up_axis {
            0 => cx = up,
            1 => cy = up,
            _ => cz = up,
        }
        match self.forward_axis {
            0 => cx = fwd,
            1 => cy = fwd,
            _ => cz = fwd,
        }
        let basis2 = Matrix3::from_cols(cx, cy, cz);

        wheel.world_basis = Matrix3::from_quat(steering_quat * rotating_quat) * basis2;
        wheel.world_origin = wheel.raycast_info.hard_point_ws
            + wheel.raycast_info.wheel_direction_ws * wheel.raycast_info.suspension_length;
    }

    /// Cast this wheel's ray down from its connection point and update the
    /// suspension state from the hit (Bullet's `rayCast`). Returns the hit
    /// depth (`-1` on a miss; `update_vehicle` ignores it, like Bullet).
    fn ray_cast(&mut self, world: &PhysicsWorld, wheel_index: usize) -> Real {
        self.update_wheel_transforms_ws(world, wheel_index);
        let mut depth: Real = -1.0;

        let raylen: Real = self.wheels[wheel_index].suspension_rest_length + self.wheels[wheel_index].wheel_radius;
        let (rayvector, source) = {
            let wheel = &self.wheels[wheel_index];
            (wheel.raycast_info.wheel_direction_ws * raylen, wheel.raycast_info.hard_point_ws)
        };
        let target = source + rayvector;

        // The chassis itself is excluded: Bullet's rays starting inside the
        // chassis miss (interior starts), ours would report them (deviation
        // §1), so we skip the chassis explicitly.
        if let Some((ground, hit)) = world.raycast_closest_excluding(source, target, Some(self.chassis)) {
            let param = hit.fraction;
            depth = raylen * param;
            let wheel = &mut self.wheels[wheel_index];
            wheel.raycast_info.contact_normal_ws = hit.normal;
            wheel.raycast_info.is_in_contact = true;
            wheel.raycast_info.ground_body = Some(ground);

            let hit_distance = param * raylen;
            wheel.raycast_info.suspension_length = hit_distance - wheel.wheel_radius;
            // Clamp to the maximum suspension travel.
            let min_len = wheel.suspension_rest_length - wheel.max_suspension_travel_cm * 0.01;
            let max_len = wheel.suspension_rest_length + wheel.max_suspension_travel_cm * 0.01;
            wheel.raycast_info.suspension_length = wheel.raycast_info.suspension_length.clamp(min_len, max_len);
            wheel.raycast_info.contact_point_ws = hit.position;

            // Relative velocity along the suspension, and the clipped inverse
            // of `normal · wheel_direction` used to scale the spring force.
            let denominator = wheel
                .raycast_info
                .contact_normal_ws
                .dot(wheel.raycast_info.wheel_direction_ws);
            let chassis = world.body(self.chassis);
            let relpos = wheel.raycast_info.contact_point_ws - chassis.position;
            let chassis_velocity = chassis.velocity_at_local_point(relpos);
            let proj_vel = wheel.raycast_info.contact_normal_ws.dot(chassis_velocity);
            if denominator >= -0.1 {
                wheel.suspension_relative_velocity = 0.0;
                wheel.clipped_inv_contact_dot_suspension = 1.0 / 0.1;
            } else {
                let inv = -1.0 / denominator;
                wheel.suspension_relative_velocity = proj_vel * inv;
                wheel.clipped_inv_contact_dot_suspension = inv;
            }
        } else {
            // Miss: hang the wheel at the rest length.
            let wheel = &mut self.wheels[wheel_index];
            wheel.raycast_info.suspension_length = wheel.suspension_rest_length;
            wheel.suspension_relative_velocity = 0.0;
            wheel.raycast_info.contact_normal_ws = -wheel.raycast_info.wheel_direction_ws;
            wheel.clipped_inv_contact_dot_suspension = 1.0;
        }
        depth
    }

    /// Compute the spring-damper suspension force of every wheel in contact
    /// (Bullet's `updateSuspension`). The spring force is scaled by the
    /// chassis mass: Bullet treats `stiffness · Δlength` as an acceleration
    /// and converts it to a force at the end.
    fn update_suspension(&mut self, world: &PhysicsWorld) {
        let chassis = world.body(self.chassis);
        let chassis_mass = chassis.mass();
        for wheel in &mut self.wheels {
            if wheel.raycast_info.is_in_contact {
                // Spring: stiffness · (rest − current) · clipped inverse.
                let length_diff = wheel.suspension_rest_length - wheel.raycast_info.suspension_length;
                let mut force = wheel.suspension_stiffness
                    * length_diff
                    * wheel.clipped_inv_contact_dot_suspension;
                // Damper: compression damping while sinking, relaxation while
                // extending.
                let projected_rel_vel = wheel.suspension_relative_velocity;
                let susp_damping = if projected_rel_vel < 0.0 {
                    wheel.wheels_damping_compression
                } else {
                    wheel.wheels_damping_relaxation
                };
                force -= susp_damping * projected_rel_vel;
                // Result: per-wheel force, never negative (no pulling down).
                wheel.wheels_suspension_force = force * chassis_mass;
                if wheel.wheels_suspension_force < 0.0 {
                    wheel.wheels_suspension_force = 0.0;
                }
            } else {
                wheel.wheels_suspension_force = 0.0;
            }
        }
    }

    /// Longitudinal (drive/brake) and lateral (anti-slide) tyre friction
    /// (Bullet's `updateFriction`). The lateral impulse is a unilateral
    /// constraint killing the chassis velocity along each wheel's axle; the
    /// longitudinal impulse is the engine force (or a brake-rolling impulse);
    /// both are clamped to the combined friction cone so the tyres can slide.
    fn update_friction(&mut self, world: &mut PhysicsWorld, dt: Real) {
        let num_wheels = self.wheels.len();
        if num_wheels == 0 {
            return;
        }
        self.forward_ws.resize(num_wheels, Vector3::ZERO);
        self.axle.resize(num_wheels, Vector3::ZERO);
        self.forward_impulse.resize(num_wheels, 0.0);
        self.side_impulse.resize(num_wheels, 0.0);

        let num_wheels_on_ground = self
            .wheels
            .iter()
            .filter(|w| w.raycast_info.is_in_contact)
            .count();

        // Lateral friction: kill the velocity along each wheel's axle.
        for i in 0..num_wheels {
            self.side_impulse[i] = 0.0;
            self.forward_impulse[i] = 0.0;
            if !self.wheels[i].raycast_info.is_in_contact {
                continue;
            }
            // The axle direction is the wheel basis' right-axis column:
            // `update_wheel_transform` places the (steered) axle there, and
            // the spin rotation leaves it untouched. Bullet extracts the
            // right-axis *row* instead, which picks up a `cos(rotation)`
            // factor from the wheel spin — the steering direction then
            // oscillates at the spin frequency and the net turning washes out
            // (observed: a steered car drives straight with a small wobble).
            // We deviate from that literal extraction; see deviations §1.
            let basis = self.wheels[i].world_basis;
            let mut axle = -basis.col(self.right_axis);
            let surf_normal = self.wheels[i].raycast_info.contact_normal_ws;
            let proj = axle.dot(surf_normal);
            axle -= surf_normal * proj;
            axle = axle.normalize();
            self.axle[i] = axle;
            self.forward_ws[i] = surf_normal.cross(axle).normalize();
            let chassis = world.body(self.chassis);
            self.side_impulse[i] =
                resolve_single_bilateral(chassis, self.wheels[i].raycast_info.contact_point_ws, axle);
        }

        // Longitudinal impulse: engine force while throttling, otherwise a
        // brake rolling-friction impulse; then clamp both to the friction cone.
        let side_factor: Real = 1.0;
        let fwd_factor: Real = 0.5;
        let mut sliding = false;
        for i in 0..num_wheels {
            let mut rolling_friction: Real = 0.0;
            if self.wheels[i].raycast_info.is_in_contact {
                if self.wheels[i].engine_force != 0.0 {
                    rolling_friction = self.wheels[i].engine_force * dt;
                } else {
                    let max_impulse = if self.wheels[i].brake != 0.0 {
                        self.wheels[i].brake
                    } else {
                        0.0
                    };
                    let chassis = world.body(self.chassis);
                    rolling_friction = calc_rolling_friction(
                        chassis,
                        self.wheels[i].raycast_info.contact_point_ws,
                        self.forward_ws[i],
                        max_impulse,
                        num_wheels_on_ground,
                    );
                }
            }

            self.forward_impulse[i] = 0.0;
            self.wheels[i].skid_info = 1.0;
            if self.wheels[i].raycast_info.is_in_contact {
                let max_imp = self.wheels[i].wheels_suspension_force * dt * self.wheels[i].friction_slip;
                let max_imp_squared = max_imp * max_imp;
                self.forward_impulse[i] = rolling_friction;
                let x = self.forward_impulse[i] * fwd_factor;
                let y = self.side_impulse[i] * side_factor;
                let impulse_squared = x * x + y * y;
                if impulse_squared > max_imp_squared {
                    sliding = true;
                    let factor = max_imp / math::sqrt(impulse_squared);
                    self.wheels[i].skid_info *= factor;
                }
            }
        }

        // A sliding wheel keeps the direction of its total impulse but scales
        // it down to the friction cone.
        if sliding {
            for i in 0..num_wheels {
                if self.side_impulse[i] != 0.0 && self.wheels[i].skid_info < 1.0 {
                    self.forward_impulse[i] *= self.wheels[i].skid_info;
                    self.side_impulse[i] *= self.wheels[i].skid_info;
                }
            }
        }

        // Apply the impulses to the chassis at the contact points.
        let chassis_pos = world.body(self.chassis).position;
        for i in 0..num_wheels {
            let rel_pos = self.wheels[i].raycast_info.contact_point_ws - chassis_pos;
            if self.forward_impulse[i] != 0.0 {
                world
                    .body_mut(self.chassis)
                    .apply_impulse(self.forward_ws[i] * self.forward_impulse[i], rel_pos);
            }
            if self.side_impulse[i] != 0.0 {
                let side_imp = self.axle[i] * self.side_impulse[i];
                // The lateral impulse is applied with reduced leverage along
                // the chassis up axis (roll influence): the car slides rather
                // than rolling over (Bullet's ROLLING_INFLUENCE_FIX).
                let chassis = world.body(self.chassis);
                let chassis_up = axis_unit_vector(chassis.orientation, self.up_axis);
                let mut rel = rel_pos;
                rel -= chassis_up * (chassis_up.dot(rel) * (1.0 - self.wheels[i].roll_influence));
                world.body_mut(self.chassis).apply_impulse(side_imp, rel);
            }
        }
    }
}

/// The unit vector of a chassis-local axis in world space (the `axis`-th
/// column of the body's orientation).
fn axis_unit_vector(orientation: Quaternion, axis: usize) -> Vector3 {
    let mut unit = Vector3::ZERO;
    unit[axis] = 1.0;
    orientation * unit
}

/// Lateral friction impulse: the impulse along `normal` that (damped) removes
/// the relative velocity at `pos` between the chassis and the fixed ground
/// (Bullet's `resolveSingleBilateral` against `btActionInterface::getFixedBody`).
fn resolve_single_bilateral(chassis: &RigidBody, pos: Vector3, normal: Vector3) -> Real {
    let rel_pos = pos - chassis.position;
    let vel = chassis.velocity_at_local_point(rel_pos);
    let jac_diag_ab_inv = 1.0 / chassis.compute_impulse_denominator(pos, normal);
    let rel_vel = normal.dot(vel);
    let contact_damping: Real = 0.2;
    -contact_damping * rel_vel * jac_diag_ab_inv
}

/// Brake rolling friction: the impulse along `forward_dir` that removes the
/// forward velocity at the contact point, shared across the wheels on the
/// ground and clamped to `max_impulse` (Bullet's `calcRollingFriction`).
fn calc_rolling_friction(
    chassis: &RigidBody,
    contact_pos: Vector3,
    forward_dir: Vector3,
    max_impulse: Real,
    num_wheels_on_ground: usize,
) -> Real {
    let rel_pos = contact_pos - chassis.position;
    let vel = chassis.velocity_at_local_point(rel_pos);
    let vrel = forward_dir.dot(vel);
    let jac_diag_ab_inv = 1.0 / chassis.compute_impulse_denominator(contact_pos, forward_dir);
    let mut j1 = -vrel * jac_diag_ab_inv / (num_wheels_on_ground as Real);
    j1 = j1.clamp(-max_impulse, max_impulse);
    j1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use bullet_collision::shapes::CollisionShape;

    const DT: Real = 1.0 / 60.0;

    /// A world with a ground plane and a 4-wheel car, laid out like Bullet's
    /// raycast-vehicle demo: the suspension towers attach *above* the chassis
    /// centre of mass (which is what keeps an accelerating car from flipping
    /// backwards — a below-COM attachment is unstable under drive torque).
    /// Mass 800, Y-up coordinates (right = X, up = Y, forward = Z).
    fn car_world() -> (PhysicsWorld, RaycastVehicle) {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let chassis = world.add_body(
            RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::new(1.0, 0.3, 0.6)),
                Vector3::new(0.0, 1.0, 0.0),
                800.0,
            )
            .disable_deactivation(),
        );
        let mut vehicle = RaycastVehicle::new(chassis);
        vehicle.set_coordinate_system(0, 1, 2);
        let tuning = VehicleTuning::default();
        let conn = |x: Real, z: Real| Vector3::new(x, 0.4, z);
        for (x, z, front) in [
            (0.8, 0.9, true),
            (-0.8, 0.9, true),
            (-0.8, -0.9, false),
            (0.8, -0.9, false),
        ] {
            vehicle.add_wheel(
                &world,
                conn(x, z),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
                1.1,
                0.35,
                &tuning,
                front,
            );
        }
        (world, vehicle)
    }

    /// A parked car settles: all wheels in contact, the chassis held at the
    /// suspension equilibrium height (slightly below the rest height — the
    /// default tuning compresses ~0.42 m under 1 g), no sinking, no drift.
    #[test]
    fn vehicle_settles_on_flat_ground() {
        let (mut world, mut vehicle) = car_world();
        for _ in 0..240 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        let chassis = world.body(vehicle.chassis_body());
        // Equilibrium: the wheel bottom rests on the ground, so
        // COM_y + 0.4 (connection point) − suspension_length − 0.35 (radius) = 0
        // with the default tuning compressing the suspension ~0.42 m under 1 g.
        let expected_y = (1.1 - 0.42) + 0.35 - 0.4; // = 0.63
        assert!(
            (chassis.position.y - expected_y).abs() < 0.15,
            "chassis y = {} (expected ~{expected_y})",
            chassis.position.y
        );
        for i in 0..vehicle.num_wheels() {
            assert!(vehicle.wheel_info(i).raycast_info.is_in_contact, "wheel {i} lost contact");
        }
        assert!(
            chassis.linear_velocity.length() < 0.5,
            "still moving: v = {:?}",
            chassis.linear_velocity
        );
        // The chassis never touches the ground (bottom at y − 0.3 > 0).
        assert!(chassis.position.y - 0.3 > 0.05);
    }

    /// Engine force on the rear wheels accelerates the car forward (+Z).
    #[test]
    fn engine_force_accelerates_the_car() {
        let (mut world, mut vehicle) = car_world();
        for i in 2..4 {
            vehicle.apply_engine_force(800.0, i);
        }
        for _ in 0..180 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        let chassis = world.body(vehicle.chassis_body());
        assert!(chassis.position.z > 2.0, "z = {}", chassis.position.z);
        assert!(chassis.linear_velocity.z > 1.0, "vz = {}", chassis.linear_velocity.z);
        assert!(vehicle.current_speed_km_hour() > 0.0);
    }

    /// Steering the front wheels turns the car: its forward heading deviates
    /// from +Z while it keeps driving.
    #[test]
    fn steering_turns_the_car() {
        let (mut world, mut vehicle) = car_world();
        for i in 2..4 {
            vehicle.apply_engine_force(800.0, i);
        }
        for i in 0..2 {
            vehicle.set_steering_value(0.35, i);
        }
        for _ in 0..240 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        let fwd = vehicle.forward_vector(&world);
        let chassis = world.body(vehicle.chassis_body());
        assert!(fwd.dot(Vector3::Z) < 0.99, "heading still ~+Z: {:?}", fwd);
        assert!(chassis.position.z > 1.0, "z = {}", chassis.position.z);
    }

    /// Braking (with the throttle off) kills the forward speed.
    #[test]
    fn brake_slows_the_car() {
        let (mut world, mut vehicle) = car_world();
        for i in 2..4 {
            vehicle.apply_engine_force(800.0, i);
        }
        for _ in 0..150 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        let speed_before = world.body(vehicle.chassis_body()).linear_velocity.length();
        for i in 0..4 {
            vehicle.apply_engine_force(0.0, i);
        }
        for i in 0..4 {
            vehicle.set_brake(60.0, i);
        }
        for _ in 0..180 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        let speed_after = world.body(vehicle.chassis_body()).linear_velocity.length();
        assert!(
            speed_after < speed_before * 0.5,
            "braking did not slow the car: {speed_before} -> {speed_after}"
        );
        assert!(speed_after < 2.0, "speed_after = {speed_after}");
    }

    /// A car dropped from a height falls, the wheels catch it and the chassis
    /// ends up at the same equilibrium height (no tumble, no tunnel through).
    #[test]
    fn vehicle_falls_and_is_caught_by_the_suspension() {
        let (mut world, mut vehicle) = car_world();
        world
            .body_mut(vehicle.chassis_body())
            .position = Vector3::new(0.0, 4.0, 0.0);
        for _ in 0..360 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        let chassis = world.body(vehicle.chassis_body());
        assert!(
            (chassis.position.y - 0.63).abs() < 0.2,
            "chassis y = {} (expected ~0.63)",
            chassis.position.y
        );
        assert!(chassis.linear_velocity.length() < 1.0, "still falling: {:?}", chassis.linear_velocity);
    }

    /// The wheel world pose tracks the suspension: at rest the wheel centre
    /// sits one wheel radius above the ground.
    #[test]
    fn wheel_transform_matches_the_suspension() {
        let (mut world, mut vehicle) = car_world();
        for _ in 0..240 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        let wheel = vehicle.wheel_info(0);
        let origin = wheel.world_origin;
        assert!(
            (origin.y - 0.35).abs() < 0.1,
            "wheel centre y = {} (expected ~0.35)",
            origin.y
        );
        // Origin equals hard point + direction · suspension length.
        let expected = wheel.raycast_info.hard_point_ws
            + wheel.raycast_info.wheel_direction_ws * wheel.raycast_info.suspension_length;
        let diff = (origin - expected).abs();
        assert!(
            diff.x < 1e-4 && diff.y < 1e-4 && diff.z < 1e-4,
            "origin {:?} != expected {:?}",
            origin,
            expected
        );
    }

    /// A wheel raycast must never hit the vehicle's own chassis (Bullet skips
    /// it via interior-start misses; we exclude it explicitly).
    #[test]
    fn wheel_raycast_ignores_the_chassis() {
        let (mut world, mut vehicle) = car_world();
        // A tall obstacle under the car path would be a valid hit; the chassis
        // itself must not be.
        for _ in 0..60 {
            world.step_simulation(DT);
            vehicle.update_vehicle(&mut world, DT);
        }
        for i in 0..vehicle.num_wheels() {
            let info = vehicle.wheel_info(i);
            assert!(info.raycast_info.is_in_contact);
            assert_ne!(info.raycast_info.ground_body, Some(vehicle.chassis_body()));
        }
    }
}
