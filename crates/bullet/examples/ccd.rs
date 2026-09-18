//! CCD demo: a fast ball is fired at a thin static wall. With CCD enabled the
//! ball is clamped at the wall and bounces back next step; without CCD it
//! tunnels straight through in a single frame.
//! Run with `cargo run --example ccd`.

use bullet::dynamics::{PhysicsWorld, RigidBody};
use bullet::math::{Real, Vector3};
use bullet::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    world.solver_info_mut().num_iterations = 20;

    // Thin static wall: a 0.2-thick box spanning the XY plane region.
    let wall = RigidBody::new_static(
        CollisionShape::box_shape(Vector3::new(5.0, 5.0, 0.1)),
        Vector3::new(0.0, 0.0, 0.0),
    );
    world.add_body(wall);

    // A fast ball with CCD enabled, aimed at the wall.
    let ccd_ball = world.add_body(
        RigidBody::new_dynamic(
            CollisionShape::sphere(0.3),
            Vector3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, 0.0, -120.0))
        .ccd(0.1, 0.3),
    );
    // A control ball without CCD, offset in X so both travel the same path.
    let plain_ball = world.add_body(
        RigidBody::new_dynamic(
            CollisionShape::sphere(0.3),
            Vector3::new(2.0, 0.0, 1.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, 0.0, -120.0)),
    );

    let dt: Real = 1.0 / 60.0;
    world.step_simulation(dt);

    let ccd_z = world.body(ccd_ball).position.z;
    let plain_z = world.body(plain_ball).position.z;
    println!("after one step:");
    println!("  ccd   ball z = {:.4} (wall at z = 0, clamped at z = 0.4)", ccd_z);
    println!("  plain ball z = {:.4} (tunnelled through)", plain_z);

    // The CCD ball rests just in front of the wall (surface at z = 0.1 plus
    // the sphere radius); the plain ball is far behind it.
    assert!(
        ccd_z > 0.35 && ccd_z < 0.45,
        "ccd ball did not stop at the wall: z = {ccd_z}"
    );
    assert!(
        plain_z < -0.5,
        "plain ball should have tunnelled through: z = {plain_z}"
    );
    println!("OK: CCD stopped the fast ball at the wall; the plain ball tunnelled.");
}
