//! End-to-end demo for the stage-2 shapes: a capsule and a sphere fall onto an
//! infinite static plane and come to rest. The capsule-plane pair exercises
//! the dedicated plane-vs-convex narrow phase; the sphere lands through the
//! same path. Run with `cargo run --example falling_capsule`.

use bullet::dynamics::{PhysicsWorld, RigidBody};
use bullet::math::{Real, Vector3};
use bullet::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.set_iterations(20);

    // Infinite static ground plane at y = 0 (new plane shape).
    let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
    world.add_body(ground);

    // A vertical capsule (radius 0.5, half height 1.0) dropped from 5 m.
    let capsule = RigidBody::new_dynamic(
        CollisionShape::capsule(0.5, 1.0),
        Vector3::new(0.0, 5.0, 0.0),
        1.0,
    )
    .friction(0.6);
    let capsule = world.add_body(capsule);

    // A sphere dropped next to it.
    let sphere = RigidBody::new_dynamic(
        CollisionShape::sphere(0.5),
        Vector3::new(1.0, 6.0, 0.0),
        1.0,
    )
    .friction(0.6);
    let sphere = world.add_body(sphere);

    let dt: Real = 1.0 / 60.0;
    let mut time: Real = 0.0;
    println!("t      cap_y   sph_y");
    while time < 5.0 {
        world.step_simulation(dt);
        time += dt;
        let cy = world.body(capsule).position.y;
        let sy = world.body(sphere).position.y;
        if (time * 60.0).round() as i64 % 30 == 0 {
            println!("{:.2}  {:6.3}  {:6.3}", time, cy, sy);
        }
    }

    // Resting heights: the capsule's bottom tip (y = -1.5) touches the plane,
    // so its center rests at 1.5; the sphere rests at 0.5.
    let cy = world.body(capsule).position.y;
    let sy = world.body(sphere).position.y;
    println!("final cap_y = {cy:.4} (expected ~1.5), sph_y = {sy:.4} (expected ~0.5)");
    assert!((cy - 1.5).abs() < 0.1, "capsule did not settle: y = {cy}");
    assert!((sy - 0.5).abs() < 0.1, "sphere did not settle: y = {sy}");
    println!("OK: capsule and sphere settled on the plane.");
}
