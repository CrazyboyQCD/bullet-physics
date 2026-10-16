//! Minimal end-to-end rigid-body demo: a box falls onto a static ground box
//! under gravity and comes to rest. Run with `cargo run --example falling_box`.

use bullet_physics::dynamics::{PhysicsWorld, RigidBody};
use bullet_physics::math::{Real, Vector3};
use bullet_physics::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.solver_info_mut().num_iterations = 20;

    // Static ground plane (a wide, thin box).
    let ground = RigidBody::new_static(
        CollisionShape::box_shape(Vector3::new(25.0, 0.5, 25.0)),
        Vector3::new(0.0, -0.5, 0.0),
    );
    world.add_body(ground);

    // A 1x1x1 cube dropped from 5 m above the ground.
    let cube = RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::new(0.0, 5.0, 0.0),
        1.0,
    )
    .friction(0.6)
    .restitution(0.0);
    let cube = world.add_body(cube);

    let dt: Real = 1.0 / 60.0;
    let mut time: Real = 0.0;
    println!("t      y       vy");
    while time < 5.0 {
        world.step_simulation(dt);
        time += dt;
        let p = world.body(cube).position;
        let v = world.body(cube).linear_velocity;
        if (time * 60.0).round() as i64 % 30 == 0 {
            println!("{:.2}  {:6.3}  {:7.3}", time, p.y, v.y);
        }
    }

    // The cube should have settled on the ground (ground top at y = 0, so the
    // cube's center rests at y = 0.5).
    let y = world.body(cube).position.y;
    println!("final y = {:.4} (expected ~0.5)", y);
    assert!(
        (y - 0.5).abs() < 0.05,
        "cube did not settle on the ground: y = {y}"
    );
    println!("OK: cube settled on the ground.");
}
