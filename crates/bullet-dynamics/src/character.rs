//! Kinematic character controller — port of Bullet's
//! `btKinematicCharacterController` (`BulletDynamics/Character`).
//!
//! The character is a *static* body in the [`PhysicsWorld`] (it blocks dynamic
//! bodies but is never affected by the solver); the controller moves it
//! directly each step with sweep tests against the world (Bullet's ghost
//! object + convex sweep). Call [`KinematicCharacterController::update`] once
//! per frame, after [`PhysicsWorld::step_simulation`], mirroring Bullet's
//! action order (controllers run after `integrateTransforms`).
//!
//! Known deviations from Bullet (see `docs/bullet-deviations.md`): the
//! controller does not push dynamic bodies aside (the character is a solid
//! obstacle), and the sweep margin (`m_addedMargin = 0.02`) is not applied —
//! sweeps use the exact shape geometry.

use bullet_collision::shapes::CollisionShape;
use bullet_math::{Quaternion, Real, Vector3};

use crate::world::PhysicsWorld;

/// Kinematic character controller that slides along surfaces (Bullet's
/// `btKinematicCharacterController`). Owns a clone of the character's convex
/// shape for the world queries; the matching body must be created by the user
/// with the same shape and `RigidBody::new_static`.
pub struct KinematicCharacterController {
    /// The character's body index in the world.
    body: usize,
    shape: CollisionShape,
    /// Up direction (default `Vector3::Y`).
    up: Vector3,
    /// Jump direction (Bullet's `m_jumpAxis`).
    jump_axis: Vector3,
    step_height: Real,
    /// Gravity magnitude (Bullet's default is 3G = 29.4).
    gravity: Real,
    /// Terminal fall speed (Bullet's default 55 m/s).
    fall_speed: Real,
    /// Jump speed (Bullet's default 10 m/s).
    jump_speed: Real,
    /// Cosine of the maximum walkable slope angle.
    max_slope_cosine: Real,
    /// Penetration depth above which `recover_from_penetration` pushes the
    /// character out (Bullet's `m_maxPenetrationDepth = 0.2`).
    max_penetration_depth: Real,
    walk_direction: Vector3,
    normalized_direction: Vector3,
    use_walk_direction: bool,
    velocity_time_interval: Real,
    linear_damping: Real,
    angular_damping: Real,
    angular_velocity: Vector3,
    vertical_velocity: Real,
    vertical_offset: Real,
    current_step_offset: Real,
    was_on_ground: bool,
    was_jumping: bool,
    touching_contact: bool,
    // Per-step pose state.
    current_position: Vector3,
    current_orientation: Quaternion,
    target_position: Vector3,
}

impl KinematicCharacterController {
    /// Create a controller for the static body `body` (whose shape must match
    /// `shape`), with the given `step_height` and up axis.
    pub fn new(body: usize, shape: CollisionShape, step_height: Real, up: Vector3) -> Self {
        let up = if up.length_squared() > 0.0 { up.normalize() } else { Vector3::Y };
        Self {
            body,
            shape,
            up,
            jump_axis: up,
            step_height,
            gravity: 9.8 * 3.0, // Bullet's 3G default.
            fall_speed: 55.0,
            jump_speed: 10.0,
            max_slope_cosine: (std::f32::consts::FRAC_PI_4 as Real).cos(), // 45 degrees.
            max_penetration_depth: 0.2,
            walk_direction: Vector3::ZERO,
            normalized_direction: Vector3::ZERO,
            use_walk_direction: true,
            velocity_time_interval: 0.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            angular_velocity: Vector3::ZERO,
            vertical_velocity: 0.0,
            vertical_offset: 0.0,
            current_step_offset: 0.0,
            was_on_ground: false,
            was_jumping: false,
            touching_contact: false,
            current_position: Vector3::ZERO,
            current_orientation: Quaternion::IDENTITY,
            target_position: Vector3::ZERO,
        }
    }

    /// Set the per-step walk displacement (Bullet's `setWalkDirection`). This
    /// is neither a direction nor a velocity: the amount the character moves
    /// per simulation step. Resets any velocity set by
    /// [`set_velocity_for_time_interval`](Self::set_velocity_for_time_interval).
    pub fn set_walk_direction(&mut self, walk_direction: Vector3) {
        self.use_walk_direction = true;
        self.walk_direction = walk_direction;
        self.normalized_direction = if walk_direction.length_squared() > 1e-12 {
            walk_direction.normalize()
        } else {
            Vector3::ZERO
        };
    }

    /// Move with `velocity` for `time_interval` seconds, then stop (Bullet's
    /// `setVelocityForTimeInterval`). Resets any walk direction.
    pub fn set_velocity_for_time_interval(&mut self, velocity: Vector3, time_interval: Real) {
        self.use_walk_direction = false;
        self.walk_direction = velocity;
        self.normalized_direction = if velocity.length_squared() > 1e-12 {
            velocity.normalize()
        } else {
            Vector3::ZERO
        };
        self.velocity_time_interval += time_interval;
    }

    /// Set the linear velocity; the up-axis component becomes the vertical
    /// (fall/jump) velocity (Bullet's `setLinearVelocity` hack). Note that in
    /// walk-direction mode the horizontal part is treated as a per-step
    /// displacement, matching Bullet.
    pub fn set_linear_velocity(&mut self, v: Vector3) {
        self.walk_direction = v;
        if v.length_squared() > 0.0 {
            let w = v.normalize();
            let c = w.dot(self.up);
            if c != 0.0 {
                let up_component = self.up * (c * v.length());
                self.walk_direction -= up_component;
                self.vertical_velocity = if c < 0.0 {
                    -up_component.length()
                } else {
                    up_component.length()
                };
                if c > 0.0 {
                    self.was_jumping = true;
                }
            }
        } else {
            self.vertical_velocity = 0.0;
        }
    }

    /// Start a jump along the up axis (Bullet's `jump()`).
    pub fn jump(&mut self) {
        self.vertical_velocity = self.jump_speed;
        self.was_jumping = true;
        self.jump_axis = self.up;
    }

    /// Set gravity; the up axis is realigned to `-gravity` (Bullet's
    /// `setGravity`).
    pub fn set_gravity(&mut self, gravity: Vector3) {
        if gravity.length_squared() > 0.0 {
            self.up = (-gravity).normalize();
        }
        self.gravity = gravity.length();
    }

    pub fn gravity(&self) -> Vector3 {
        -self.gravity * self.up
    }

    /// Maximum walkable slope angle, in radians (Bullet's `setMaxSlope`).
    pub fn set_max_slope(&mut self, slope_radians: Real) {
        self.max_slope_cosine = slope_radians.cos();
    }

    pub fn set_step_height(&mut self, h: Real) {
        self.step_height = h;
    }

    pub fn set_fall_speed(&mut self, s: Real) {
        self.fall_speed = s;
    }

    pub fn set_jump_speed(&mut self, s: Real) {
        self.jump_speed = s;
    }

    pub fn set_max_penetration_depth(&mut self, d: Real) {
        self.max_penetration_depth = d;
    }

    pub fn set_linear_damping(&mut self, d: Real) {
        self.linear_damping = d.clamp(0.0, 1.0);
    }

    pub fn set_angular_damping(&mut self, d: Real) {
        self.angular_damping = d.clamp(0.0, 1.0);
    }

    pub fn set_angular_velocity(&mut self, v: Vector3) {
        self.angular_velocity = v;
    }

    /// True when the character is not moving vertically (Bullet's `onGround`:
    /// `|vertical_velocity| < epsilon && |vertical_offset| < epsilon`).
    pub fn on_ground(&self) -> bool {
        self.vertical_velocity.abs() < 1e-7 && self.vertical_offset.abs() < 1e-7
    }

    /// True when the last step had to recover from penetration.
    pub fn touching_contact(&self) -> bool {
        self.touching_contact
    }

    pub fn position(&self) -> Vector3 {
        self.current_position
    }

    /// Teleport the character to `origin` and reset its motion state (Bullet's
    /// `warp`).
    pub fn warp(&mut self, world: &mut PhysicsWorld, origin: Vector3) {
        self.current_position = origin;
        self.target_position = origin;
        self.vertical_velocity = 0.0;
        self.vertical_offset = 0.0;
        self.was_on_ground = false;
        self.was_jumping = false;
        self.walk_direction = Vector3::ZERO;
        self.velocity_time_interval = 0.0;
        world.set_body_transform(self.body, origin, self.current_orientation);
    }

    /// Advance the character by `dt` seconds (Bullet's `updateAction` =
    /// `preStep` + `playerStep`). Call after `PhysicsWorld::step_simulation`.
    pub fn update(&mut self, world: &mut PhysicsWorld, dt: Real) {
        // preStep: read the body pose.
        let body = world.body(self.body);
        self.current_position = body.position;
        self.current_orientation = body.orientation;
        self.target_position = self.current_position;

        // playerStep.
        // Angular velocity (damped) rotates the character.
        if self.angular_velocity.length_squared() > 0.0 {
            self.angular_velocity *= (1.0 - self.angular_damping).powf(dt);
            if self.angular_velocity.length_squared() > 0.0 {
                let dq = Quaternion::from_scaled_axis(self.angular_velocity * dt);
                self.current_orientation = (dq * self.current_orientation).normalize();
            }
        }

        // No motion when a velocity interval has expired (or is zero).
        if !self.use_walk_direction && (self.velocity_time_interval <= 0.0 || self.walk_direction == Vector3::ZERO)
        {
            return;
        }

        self.was_on_ground = self.on_ground();

        if self.walk_direction.length_squared() > 0.0 {
            self.walk_direction *= (1.0 - self.linear_damping).powf(dt);
        }
        self.vertical_velocity *= (1.0 - self.linear_damping).powf(dt);

        // Integrate the fall/jump velocity and clamp it.
        self.vertical_velocity -= self.gravity * dt;
        if self.vertical_velocity > self.jump_speed {
            self.vertical_velocity = self.jump_speed;
        }
        if self.vertical_velocity < -self.fall_speed.abs() {
            self.vertical_velocity = -self.fall_speed.abs();
        }
        self.vertical_offset = self.vertical_velocity * dt;

        self.step_up(world);
        if self.use_walk_direction {
            self.step_forward_and_strafe(world, self.walk_direction);
        } else {
            let dt_moving = dt.min(self.velocity_time_interval);
            self.velocity_time_interval -= dt;
            self.step_forward_and_strafe(world, self.walk_direction * dt_moving);
        }
        self.step_down(world, dt);

        world.set_body_transform(self.body, self.current_position, self.current_orientation);

        // Recover from penetration (Bullet's post-step loop, max 4 passes).
        let mut loops = 0;
        self.touching_contact = false;
        while self.recover_from_penetration(world) {
            self.touching_contact = true;
            loops += 1;
            if loops > 4 {
                break;
            }
        }
    }

    /// Phase 1: sweep up by the step height (while falling) so small steps are
    /// climbed instead of treated as walls; recover from any ceiling hit.
    fn step_up(&mut self, world: &mut PhysicsWorld) {
        let step_height = if self.vertical_velocity < 0.0 { self.step_height } else { 0.0 };
        self.target_position = self.current_position
            + self.up * step_height
            + self.jump_axis * self.vertical_offset.max(0.0);
        if let Some((_, hit)) = world.convex_sweep_closest(
            &self.shape,
            self.current_position,
            self.current_orientation,
            self.target_position,
            Some(self.body),
        ) && hit.normal.dot(self.up) > 0.0
        {
            // A floor/slope above: move up only partway (Bullet's
            // "moved up only a fraction of the step height").
            self.current_step_offset = step_height * hit.fraction;
            self.current_position = self.current_position.lerp(self.target_position, hit.fraction);
            let mut loops = 0;
            self.touching_contact = false;
            while self.recover_from_penetration(world) {
                self.touching_contact = true;
                loops += 1;
                if loops > 4 {
                    break;
                }
            }
            self.target_position = self.current_position;
            if self.vertical_offset > 0.0 {
                // Hit a ceiling while jumping: stop the upward motion.
                self.vertical_offset = 0.0;
                self.vertical_velocity = 0.0;
                self.current_step_offset = self.step_height;
            }
        } else {
            self.current_step_offset = step_height;
            self.current_position = self.target_position;
        }
    }

    /// Phase 2: move `walk_move`, sliding along any surfaces hit (Bullet's
    /// `stepForwardAndStrafe`, up to 10 slide iterations).
    fn step_forward_and_strafe(&mut self, world: &mut PhysicsWorld, walk_move: Vector3) {
        self.target_position = self.current_position + walk_move;
        let mut max_iter = 10;
        while max_iter > 0 {
            max_iter -= 1;
            if self.current_position.distance_squared(self.target_position) <= Real::EPSILON {
                break; // Nothing left to move.
            }
            if let Some((_, hit)) = world.convex_sweep_closest(
                &self.shape,
                self.current_position,
                self.current_orientation,
                self.target_position,
                Some(self.body),
            ) {
                let movement = self.target_position - self.current_position;
                // A resting contact (hit at fraction ~0) that does not oppose
                // the movement — e.g. the floor the character stands on while
                // walking sideways — is not a blockage: move freely. (Bullet
                // avoids the case with its sweep margin.)
                if hit.fraction <= 1e-4 && movement.dot(hit.normal) >= 0.0 {
                    self.current_position = self.target_position;
                    break;
                }
                self.update_target_based_on_collision(hit.normal);
                let dir = self.target_position - self.current_position;
                let d2 = dir.length_squared();
                if d2 <= 1e-12 {
                    break; // Fully blocked.
                }
                // Quake 2: stop when the slide would move against the original
                // walk direction (avoids oscillations in sloping corners).
                if dir.normalize().dot(self.normalized_direction) <= 0.0 {
                    break;
                }
            } else {
                self.current_position = self.target_position;
                break;
            }
        }
    }

    /// Project the target position onto the surface plane so the character
    /// slides along it (Bullet's `updateTargetPositionBasedOnCollision` with
    /// the default `tangentMag = 0, normalMag = 1`).
    fn update_target_based_on_collision(&mut self, hit_normal: Vector3) {
        let movement = self.target_position - self.current_position;
        let length = movement.length();
        if length > 1e-8 {
            let dir = movement / length;
            let slide = dir - hit_normal * dir.dot(hit_normal);
            self.target_position = self.current_position + slide * length;
        }
    }

    /// Phase 3: fall or step down onto the ground (Bullet's `stepDown` with
    /// the double-sweep that distinguishes small step-downs from free falls).
    fn step_down(&mut self, world: &mut PhysicsWorld, dt: Real) {
        if self.vertical_velocity > 0.0 {
            return; // Still rising.
        }
        let orig_position = self.target_position;
        let mut down_velocity = (-self.vertical_velocity).max(0.0) * dt;
        if down_velocity > 0.0 && down_velocity > self.fall_speed && (self.was_on_ground || !self.was_jumping) {
            down_velocity = self.fall_speed;
        }
        let mut step_drop = self.up * (self.current_step_offset + down_velocity);
        self.target_position -= step_drop;

        let mut run_once = false;
        loop {
            let hit1 = world.convex_sweep_closest(
                &self.shape,
                self.current_position,
                self.current_orientation,
                self.target_position,
                Some(self.body),
            );
            // If the first sweep misses, try twice the drop: a hit there means
            // a small step-down (land), a miss means a free fall.
            let hit2 = if hit1.is_none() {
                world.convex_sweep_closest(
                    &self.shape,
                    self.current_position,
                    self.current_orientation,
                    self.target_position - step_drop,
                    Some(self.body),
                )
            } else {
                None
            };
            let has_hit = hit2.is_some();

            let step_height = if self.vertical_velocity < 0.0 { self.step_height } else { 0.0 };
            let down_velocity2 = (-self.vertical_velocity).max(0.0) * dt;
            if down_velocity2 > 0.0
                && down_velocity2 < step_height
                && has_hit
                && !run_once
                && (self.was_on_ground || !self.was_jumping)
            {
                // Small drop while grounded: snap to the full step height for
                // fast stair motion, then re-run the sweeps.
                self.target_position = orig_position;
                down_velocity = step_height;
                step_drop = self.up * (self.current_step_offset + down_velocity);
                self.target_position -= step_drop;
                run_once = true;
                continue;
            }

            if let Some((_, hit)) = hit1 {
                // Landed on the short sweep: interpolate to the hit point and
                // stop falling (Bullet's `callback.hasHit()` landing branch).
                self.current_position = self.current_position.lerp(self.target_position, hit.fraction);
                self.vertical_velocity = 0.0;
                self.vertical_offset = 0.0;
                self.was_jumping = false;
            } else if run_once {
                // The redo (full step drop) swept down: land at its fraction
                // (or the full target when it still missed).
                let frac = hit1.map_or(1.0, |h| h.1.fraction);
                self.current_position = self.current_position.lerp(self.target_position, frac);
                self.vertical_velocity = 0.0;
                self.vertical_offset = 0.0;
                self.was_jumping = false;
            } else {
                // Full drop.
                self.current_position = self.target_position;
            }
            break;
        }
    }

    /// Push the character out of deep penetrations (Bullet's
    /// `recoverFromPenetration`): contacts deeper than `max_penetration_depth`
    /// move the character 20% of the penetration along the contact normal
    /// (from the obstacle toward the character). Returns whether any push
    /// happened.
    fn recover_from_penetration(&mut self, world: &mut PhysicsWorld) -> bool {
        let manifolds = world.contact_test(
            &self.shape,
            self.current_position,
            self.current_orientation,
            Some(self.body),
        );
        let mut penetrated = false;
        for (_, manifold) in &manifolds {
            for cp in &manifold.points {
                if cp.penetration > self.max_penetration_depth {
                    self.current_position -= cp.normal * (cp.penetration * 0.2);
                    penetrated = true;
                }
            }
        }
        world.set_body_transform(self.body, self.current_position, self.current_orientation);
        penetrated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use bullet_math::Real;

    const DT: Real = 1.0 / 60.0;

    /// A world with a ground plane and a capsule character standing on it
    /// (capsule bottom at y = 0, center at y = 1.2).
    fn flat_world() -> (PhysicsWorld, KinematicCharacterController) {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let shape = CollisionShape::capsule(0.4, 0.8);
        let body = world.add_body(RigidBody::new_static(
            shape.clone(),
            Vector3::new(0.0, 1.2, 0.0),
        ));
        let controller = KinematicCharacterController::new(body, shape, 0.5, Vector3::Y);
        (world, controller)
    }

    #[test]
    fn character_walks_on_flat_ground() {
        let (mut world, mut controller) = flat_world();
        controller.set_walk_direction(Vector3::new(0.5, 0.0, 0.0));
        for _ in 0..120 {
            world.step_simulation(DT);
            controller.update(&mut world, DT);
        }
        // Advanced +x and stayed on the ground (capsule center y = 1.2).
        assert!(controller.position().x > 2.0, "x = {}", controller.position().x);
        assert!(
            (controller.position().y - 1.2).abs() < 0.05,
            "y = {}",
            controller.position().y
        );
        assert!(controller.on_ground());
    }

    #[test]
    fn character_stops_at_a_wall() {
        let (mut world, mut controller) = flat_world();
        // A tall static wall across the path: face at x = 4.5.
        world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::new(0.5, 5.0, 5.0)),
            Vector3::new(5.0, 5.0, 0.0),
        ));
        controller.set_walk_direction(Vector3::new(0.5, 0.0, 0.0));
        for _ in 0..300 {
            world.step_simulation(DT);
            controller.update(&mut world, DT);
        }
        // The capsule (radius 0.4) is stopped by the wall face at x = 4.5.
        assert!(
            controller.position().x < 4.2 && controller.position().x > 3.9,
            "x = {}",
            controller.position().x
        );
    }

    #[test]
    fn character_steps_up_onto_a_box() {
        let (mut world, mut controller) = flat_world();
        // A step box 0.3 tall spanning x in [2, 3].
        world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::new(0.5, 0.15, 5.0)),
            Vector3::new(2.5, 0.15, 0.0),
        ));
        controller.set_walk_direction(Vector3::new(0.5, 0.0, 0.0));
        // Walk onto the step (x in [2, 3]) and assert the character climbed it.
        let mut steps = 0;
        while controller.position().x < 2.2 && steps < 60 {
            world.step_simulation(DT);
            controller.update(&mut world, DT);
            steps += 1;
        }
        assert!(steps < 60, "never reached the step");
        assert!(
            (controller.position().y - 1.5).abs() < 0.05,
            "y = {} (should stand on the 0.3 step)",
            controller.position().y
        );
    }

    #[test]
    fn character_falls_and_lands() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let shape = CollisionShape::capsule(0.4, 0.8);
        let body = world.add_body(RigidBody::new_static(
            shape.clone(),
            Vector3::new(0.0, 5.0, 0.0),
        ));
        let mut controller = KinematicCharacterController::new(body, shape, 0.5, Vector3::Y);
        for _ in 0..240 {
            world.step_simulation(DT);
            controller.update(&mut world, DT);
        }
        // Fell to the ground and stopped.
        assert!(
            (controller.position().y - 1.2).abs() < 0.05,
            "y = {}",
            controller.position().y
        );
        assert!(controller.on_ground());
    }

    #[test]
    fn character_jumps_and_lands() {
        let (mut world, mut controller) = flat_world();
        controller.jump();
        let mut max_y = controller.position().y;
        for _ in 0..300 {
            world.step_simulation(DT);
            controller.update(&mut world, DT);
            max_y = max_y.max(controller.position().y);
        }
        // Jump speed 10, gravity 3G: apex ~1.7 m above the ground.
        assert!(max_y > 2.4, "max_y = {max_y}");
        // Landed back on the ground.
        assert!(
            (controller.position().y - 1.2).abs() < 0.05,
            "final y = {}",
            controller.position().y
        );
    }
}
