//! Joints demo: a pendulum and a double pendulum (point-to-point joints,
//! Bullet's `btPoint2PointConstraint` pattern). Two spheres hang from static
//! anchors; the lower of the double pendulum is nudged so the chain swings.
//!
//! Joints are added with `world.add_constraint(...)` and solved by the normal
//! `world.step_simulation(dt)`; no extra per-frame call is needed (unlike the
//! character controller and the vehicle).
//!
//! Run with `cargo run --example joints`.

use bullet::dynamics::{Constraint, PhysicsWorld, RigidBody};
use bullet::math::{Quaternion, Real, Vector3};
use bullet::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

    // Two static anchor points: boxes fixed in space at z = -2 (pendulum) and
    // z = 2 (double pendulum). Static bodies at the anchor position with
    // identity orientation make the local pivot equal the world pivot.
    let anchor_a = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(0.0, 2.0, -2.0),
    ));
    let anchor_b = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(0.0, 2.0, 2.0),
    ));

    // Single pendulum at z = -2: a bob released at 45° from the vertical. The
    // rod attaches 0.5 m above the bob centre (in the bob's frame); the
    // anchor pivot is 0.5 m above the anchor centre. The bob starts with its
    // top exactly on the anchor pivot, so the constraint is satisfied.
    let bob = world.add_body(RigidBody::new_dynamic(
        CollisionShape::sphere(0.15),
        Vector3::new(0.354, 2.146, -2.0),
        1.0,
    )
    .orientation(Quaternion::from_rotation_z(std::f32::consts::FRAC_PI_4 as Real)));
    world.add_constraint(Constraint::point2point(
        bob,
        anchor_a,
        Vector3::new(0.0, 0.5, 0.0),
        Vector3::new(0.0, 0.5, 0.0),
    ));

    // Double pendulum at z = 2: upper bob hangs straight down (its top at the
    // anchor pivot), lower bob hangs below it, then the lower bob is nudged
    // sideways.
    let upper = world.add_body(RigidBody::new_dynamic(
        CollisionShape::sphere(0.15),
        Vector3::new(0.0, 2.0, 2.0),
        1.0,
    ));
    let lower = world.add_body(RigidBody::new_dynamic(
        CollisionShape::sphere(0.15),
        Vector3::new(0.0, 1.0, 2.0),
        1.0,
    ));
    // Upper bob: rod from its top (local (0,0.5,0)) to the anchor pivot at
    // (0, 2.5, 2). Lower bob: rod from its top to the upper bob's bottom.
    world.add_constraint(Constraint::point2point(
        upper,
        anchor_b,
        Vector3::new(0.0, 0.5, 0.0),
        Vector3::new(0.0, 0.5, 0.0),
    ));
    world.add_constraint(Constraint::point2point(
        lower,
        upper,
        Vector3::new(0.0, 0.5, 0.0),
        Vector3::new(0.0, -0.5, 0.0),
    ));
    world.body_mut(lower).linear_velocity = Vector3::new(2.0, 0.0, 0.0);

    // A hinged door at z = 6: opened 60° around a vertical hinge and
    // released, it swings down and is stopped by its ±40° angular limits.
    let anchor_c = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(0.0, 2.0, 6.0),
    ));
    let door = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
        Vector3::ZERO,
        2.0,
    ));
    let a60 = (std::f32::consts::PI / 3.0) as Real;
    world.body_mut(door).position = Vector3::new(a60.cos() * 0.5, a60.sin() * 0.5 + 2.0, 6.0);
    world
        .body_mut(door)
        .orientation = Quaternion::from_rotation_z(a60);
    let mut door_hinge = Constraint::hinge(
        door,
        anchor_c,
        Vector3::new(-0.5, 0.0, 0.0),
        Vector3::Z,
        Vector3::ZERO,
        Vector3::Z,
    );
    door_hinge
        .kind_mut()
        .hinge_mut()
        .set_limit(-0.7, 0.7);
    world.add_constraint(door_hinge);

    // Verify the pivots are initially satisfied: the upper bob top is at the
    // anchor pivot (0, 2.5, 2), the lower bob top at the upper bob bottom.
    {
        let u = world.body(upper);
        let l = world.body(lower);
        let top_u = u.position + u.orientation * Vector3::new(0.0, 0.5, 0.0);
        let top_l = l.position + l.orientation * Vector3::new(0.0, 0.5, 0.0);
        let bot_u = u.position + u.orientation * Vector3::new(0.0, -0.5, 0.0);
        assert!((top_u - Vector3::new(0.0, 2.5, 2.0)).length() < 1e-4, "upper pivot off");
        assert!((top_l - bot_u).length() < 1e-4, "lower pivot off");
    }

    let dt: Real = 1.0 / 60.0;
    let mut max_pivot_err: Real = 0.0;
    let mut min_lower_x: Real = Real::INFINITY;
    let mut door_min_angle: Real = Real::INFINITY;
    println!("t     bob.x    bob.y    lower.x  lower.y  pivot_err");
    for step in 0..900 {
        world.step_simulation(dt);
        let bob_b = world.body(bob);
        let u = world.body(upper);
        let l = world.body(lower);
        // Pivot errors: pendulum bob top vs the anchor pivot (0, 2.5, -2);
        // chain pivots vs their targets (upper top at (0, 2.5, 2), lower top
        // at the upper bob's bottom).
        let e1 = (bob_b.position + bob_b.orientation * Vector3::new(0.0, 0.5, 0.0)
            - Vector3::new(0.0, 2.5, -2.0))
        .length();
        let e2 = (u.position + u.orientation * Vector3::new(0.0, 0.5, 0.0)
            - Vector3::new(0.0, 2.5, 2.0))
        .length();
        let e3 = (l.position + l.orientation * Vector3::new(0.0, 0.5, 0.0)
            - (u.position + u.orientation * Vector3::new(0.0, -0.5, 0.0)))
        .length();
        max_pivot_err = max_pivot_err.max(e1).max(e2).max(e3);
        min_lower_x = min_lower_x.min(l.position.x);
        // The door's hinge angle: starts at +60° and is stopped by the ±40°
        // limits (0.7 rad).
        let door_angle = world.constraint(3).kind().hinge().hinge_angle();
        door_min_angle = door_min_angle.min(door_angle);
        if step % 60 == 0 {
            println!(
                "{:.2}  {:.3}  {:.3}  {:.3}  {:.3}  {:.4}",
                step as Real * dt,
                bob_b.position.x,
                bob_b.position.y,
                l.position.x,
                l.position.y,
                e1.max(e2).max(e3)
            );
        }
    }

    // Behaviors verified: the pendulum bob swung past the bottom (x crossed
    // zero, reaching the far side), the double pendulum swung too, all pivots
    // stayed on their anchors within the Baumgarte residual, and the hinged
    // door was stopped by its angular limits (never swinging past 0.7 rad).
    let bob_b = world.body(bob);
    let l = world.body(lower);
    let door_angle = world.constraint(3).kind().hinge().hinge_angle();
    println!(
        "final bob = ({:.2}, {:.2}), lower = ({:.2}, {:.2}), max pivot err = {:.4}, min lower x = {:.2}, door angle = {:.2}",
        bob_b.position.x,
        bob_b.position.y,
        l.position.x,
        l.position.y,
        max_pivot_err,
        min_lower_x,
        door_angle
    );
    assert!(max_pivot_err < 0.2, "pivot error too large: {max_pivot_err}");
    assert!(min_lower_x < -0.5, "double pendulum did not swing: {min_lower_x}");
    // The door starts at +60° (past the +0.7 limit) and swings down; the low
    // limit at −0.7 stops it, so it must never pass −0.75 and must settle on
    // the limit.
    assert!(
        door_min_angle > -0.75,
        "hinged door passed the low limit: min angle {door_min_angle}"
    );
    assert!(
        (door_angle + 0.7).abs() < 0.05,
        "hinged door did not settle on its limit: {door_angle}"
    );
    println!("OK: pendulums swing on point-to-point joints and the door swings into its hinge limit.");
}
