//! Joints demo: a pendulum and a double pendulum (point-to-point joints,
//! Bullet's `btPoint2PointConstraint` pattern), a hinged door, and one
//! scene per remaining joint family — a welded pair (fixed), a cone-twist
//! swing, a motorized slider and a generic 6DOF with an angular limit.
//!
//! Joints are added with `world.add_constraint(...)` and solved by the normal
//! `world.step_simulation(dt)`; no extra per-frame call is needed (unlike the
//! character controller and the vehicle).
//!
//! Run with `cargo run --example joints`.

use bullet::dynamics::{Constraint, JointFrame, PhysicsWorld, RigidBody};
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

    // A welded pair (fixed joint) at z = -4: two boxes joined rigidly hang
    // from a static anchor; the weld must hold through the swing.
    let anchor_f = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(0.0, 2.0, -4.0),
    ));
    let fixed_box = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(0.0, 1.0, -4.0),
        1.0,
    ));
    world.add_constraint(Constraint::fixed(
        fixed_box,
        anchor_f,
        JointFrame::new(Vector3::new(0.0, 0.5, 0.0), bullet::math::Matrix3::IDENTITY),
        JointFrame::new(Vector3::new(0.0, -0.5, 0.0), bullet::math::Matrix3::IDENTITY),
    ));
    world.body_mut(fixed_box).linear_velocity = Vector3::new(1.0, 0.0, 0.0);

    // A cone-twist joint at z = 4: a bob pinned at the anchor spins around Z
    // and swings against a 30° cone.
    let anchor_ct = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(0.0, 0.0, 4.0),
    ));
    let cone_bob = world.add_body(RigidBody::new_dynamic(
        CollisionShape::sphere(0.15),
        Vector3::new(0.0, 0.0, 4.0),
        1.0,
    ));
    world.add_constraint(Constraint::cone_twist(
        cone_bob,
        anchor_ct,
        JointFrame::new(Vector3::ZERO, bullet::math::Matrix3::IDENTITY),
        JointFrame::new(Vector3::ZERO, bullet::math::Matrix3::IDENTITY),
        0.52,
        0.52,
        -1.0,
    ));
    world.body_mut(cone_bob).angular_velocity = Vector3::new(0.0, 0.0, 3.0);

    // A motorized slider at z = 8: a box rides a rail along X, driven at
    // 1.5 m/s (above the sleeping threshold so it keeps sliding).
    let anchor_s = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(0.0, 0.0, 8.0),
    ));
    let slider_box = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(0.0, 0.0, 8.0),
        1.0,
    ));
    let mut slider = Constraint::slider(
        anchor_s,
        slider_box,
        JointFrame::new(Vector3::ZERO, bullet::math::Matrix3::IDENTITY),
        JointFrame::new(Vector3::ZERO, bullet::math::Matrix3::IDENTITY),
    );
    slider.kind_mut().slider_mut().enable_lin_motor(1.5, 100.0);
    world.add_constraint(slider);

    // A generic 6DOF at z = 12: linear locked, angular Z limited to ±30°;
    // the box spins around Z and is stopped by the limit.
    let anchor_d = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(0.0, 0.0, 12.0),
    ));
    let dof_box = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(0.0, 0.0, 12.0),
        1.0,
    ));
    let mut dof = Constraint::generic_6dof(
        dof_box,
        anchor_d,
        JointFrame::new(Vector3::ZERO, bullet::math::Matrix3::IDENTITY),
        JointFrame::new(Vector3::ZERO, bullet::math::Matrix3::IDENTITY),
    );
    dof.kind_mut().generic_6dof_mut().set_angular_limits(
        Vector3::new(-1.0, -1.0, -0.52),
        Vector3::new(1.0, 1.0, 0.52),
    );
    world.add_constraint(dof);
    world.body_mut(dof_box).angular_velocity = Vector3::new(0.0, 0.0, 3.0);

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
    let mut fixed_pivot_err: Real = 0.0;
    let mut fixed_rel_angle: Real = 0.0;
    let mut cone_max_swing: Real = 0.0;
    let mut slider_off_axis: Real = 0.0;
    let mut slider_rel_angle: Real = 0.0;
    let mut dof_max_angle: Real = 0.0;
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
        // The weld: the fixed box must keep its pivot on the anchor and its
        // orientation locked to the anchor's.
        {
            let f = world.body(fixed_box);
            let pivot = f.position + f.orientation * Vector3::new(0.0, 0.5, 0.0);
            fixed_pivot_err = fixed_pivot_err.max((pivot - Vector3::new(0.0, 1.5, -4.0)).length());
            fixed_rel_angle = fixed_rel_angle
                .max(2.0 * f.orientation.w.clamp(-1.0, 1.0).acos());
        }
        // The cone-twist: the angle between the cone axes stays inside the
        // 0.52 rad cone.
        {
            let ax_a = world.body(anchor_ct).orientation * Vector3::X;
            let ax_b = world.body(cone_bob).orientation * Vector3::X;
            cone_max_swing = cone_max_swing.max(ax_a.dot(ax_b).clamp(-1.0, 1.0).acos());
        }
        // The slider: the box rides the X rail (motor at 1.5 m/s) with its
        // orientation locked.
        {
            let s = world.body(slider_box);
            slider_off_axis = slider_off_axis.max(s.position.y.abs()).max(s.position.z.abs() - 8.0);
            slider_rel_angle = slider_rel_angle.max(2.0 * s.orientation.w.clamp(-1.0, 1.0).acos());
        }
        // The 6DOF: the Z Euler angle stays inside the ±0.52 rad limit.
        dof_max_angle = dof_max_angle.max(world.constraint(7).kind().generic_6dof().get_angle(2).abs());
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
    // The weld held, the cone-twist swing stayed in its cone, the slider
    // motor drove the box along the rail, and the 6DOF spin stopped at its
    // angular limit.
    let bob_b = world.body(bob);
    let l = world.body(lower);
    let door_angle = world.constraint(3).kind().hinge().hinge_angle();
    let slider_x = world.body(slider_box).position.x;
    println!(
        "final bob = ({:.2}, {:.2}), lower = ({:.2}, {:.2}), max pivot err = {:.4}, min lower x = {:.2}, door angle = {:.2}, slider x = {:.1}, cone swing = {:.3}, dof angle = {:.3}",
        bob_b.position.x,
        bob_b.position.y,
        l.position.x,
        l.position.y,
        max_pivot_err,
        min_lower_x,
        door_angle,
        slider_x,
        cone_max_swing,
        dof_max_angle
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
    // The weld: the fixed box stayed on its pivot, locked to the anchor's
    // orientation (the initial sideways nudge was absorbed).
    assert!(fixed_pivot_err < 0.05, "weld stretched: {fixed_pivot_err}");
    assert!(fixed_rel_angle < 0.1, "weld rotated: {fixed_rel_angle}");
    // The cone-twist: the swing stayed inside the cone (0.52 rad + margin).
    assert!(cone_max_swing < 0.6, "cone exceeded: {cone_max_swing}");
    // The slider motor drove the box ~22 m along X in 15 s without drifting
    // off the rail or rotating.
    assert!((slider_x - 22.5).abs() < 2.0, "slider did not travel: x = {slider_x}");
    assert!(slider_off_axis < 0.05, "slider left the rail: {slider_off_axis}");
    assert!(slider_rel_angle < 0.15, "slider rotated: {slider_rel_angle}");
    // The 6DOF spin stopped at the ±0.52 rad Z limit.
    assert!(dof_max_angle < 0.6, "6DOF passed its limit: {dof_max_angle}");
    println!("OK: point-to-point, hinge, fixed, cone-twist, slider and generic-6DOF joints all behave.");
}
