//! Character controller demo: a capsule character walks across a flat terrain,
//! steps up onto a box, jumps, and is stopped by a wall.
//!
//! The character is a static body moved kinematically by the controller
//! (Bullet's `btKinematicCharacterController` pattern): call
//! `controller.update(&mut world, dt)` once per frame after
//! `world.step_simulation(dt)`.
//!
//! Run with `cargo run --example character`.

use bullet_physics::dynamics::{KinematicCharacterController, PhysicsWorld, RigidBody};
use bullet_physics::math::{Real, Vector3};
use bullet_physics::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

    // Ground plane at y = 0.
    world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // A 0.4-tall step spanning x in [3, 4] (a box the character climbs).
    world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::new(0.5, 0.2, 5.0)),
        Vector3::new(3.5, 0.2, 0.0),
    ));
    // A tall wall at x = 8 that stops the walk.
    world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::new(0.5, 5.0, 5.0)),
        Vector3::new(8.0, 5.0, 0.0),
    ));

    // The character: a capsule (radius 0.4, half height 0.8) as a static body.
    let shape = CollisionShape::capsule(0.4, 0.8);
    let body = world.add_body(RigidBody::new_static(
        shape.clone(),
        Vector3::new(0.0, 1.2, 0.0),
    ));
    let mut controller = KinematicCharacterController::new(body, shape, 0.5, Vector3::Y);
    // Walk direction is a per-step displacement (Bullet's `setWalkDirection`):
    // 0.02/step at 60 Hz = 1.2 m/s.
    controller.set_walk_direction(Vector3::new(0.02, 0.0, 0.0));

    let dt: Real = 1.0 / 60.0;
    let mut max_y: Real = 1.2;
    let mut time = 0.0;
    println!("t     x      y      on_ground");
    while time < 8.0 {
        world.step_simulation(dt);
        controller.update(&mut world, dt);
        time += dt;
        max_y = max_y.max(controller.position().y);
        // Jump once after climbing the step (x > 4.5), while on the ground.
        if controller.position().x > 4.5 && controller.position().x < 5.0 && controller.on_ground() {
            controller.jump();
        }
        if (time * 30.0).round() as i64 % 30 == 0 {
            println!(
                "{:.2}  {:.3}  {:.3}  {}",
                time,
                controller.position().x,
                controller.position().y,
                controller.on_ground()
            );
        }
    }

    // Behaviors verified: the character walked past the step (x > 2.5), jumped
    // at least 0.5 m above its standing height, and was stopped by the wall at
    // x = 8 (wall face 7.5 minus the capsule radius 0.4).
    let pos = controller.position();
    println!("final x = {:.3}, y = {:.3}, max_y = {:.3}", pos.x, pos.y, max_y);
    assert!(pos.x > 2.5, "did not cross the step: x = {}", pos.x);
    assert!(max_y > 1.7, "jump apex too low: {max_y}");
    assert!(pos.x < 7.6, "walked through the wall: x = {}", pos.x);
    assert!(
        controller.on_ground() && (pos.y - 1.2).abs() < 0.05,
        "not resting on the ground: y = {}",
        pos.y
    );
    println!("OK: character walked, stepped up, jumped and stopped at the wall.");
}
