//! Multibody (Featherstone) demo: an articulated-body pendulum, a
//! motor-driven slider, a point-to-point tethered pair and a geared pair of
//! revolute links — the Rust port of `BulletDynamics/Featherstone`
//! (`btMultiBody`, `btMultiBodyDynamicsWorld`, `btMultiBodyConstraint`).
//!
//! Unlike the rigid-body [`bullet::dynamics::PhysicsWorld`], articulated
//! bodies are stepped through the fused [`bullet::dynamics::PhysicsWorld`];
//! joints between them are `btMultiBody*`-style constraints solved by the
//! multibody projected Gauss-Seidel solver inside
//! `world.step_simulation(dt)`.
//!
//! Run with `cargo run --example multibody`.

use bullet::dynamics::{MultiBody, MultiBodyConstraint, MultiBodyHandle, PhysicsWorld};
use bullet::math::{Quaternion, Real, Vector3};

fn main() {
    // --- Scene 1: fixed-base revolute pendulum released from the horizontal.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut pendulum = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    pendulum.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    pendulum.finalize_multidof();
    pendulum.set_joint_pos_multidof(0, &[bullet::math::FRAC_PI_2]);
    world.add_multibody(pendulum);

    // --- Scene 2: a fixed-base prismatic link pushed along its rail by a
    // joint motor at 2 m/s.
    let mut slider = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    slider.setup_prismatic(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::X,
        Vector3::ZERO,
        Vector3::ZERO,
    );
    slider.finalize_multidof();
    let slider_idx = world.add_multibody(slider);
    let snapshot = world.multibodies().to_vec();
    let motor = MultiBodyConstraint::joint_motor(slider_idx, 0, 0, 2.0, 100.0, &snapshot);
    world.add_multibody_constraint(motor);

    // --- Scene 3: two free bases tethered by a point-to-point constraint
    // (the pair falls as one body).
    let mut a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let mut b = MultiBody::new(0, 2.0, Vector3::splat(1.5), false, false);
    a.finalize_multidof();
    b.finalize_multidof();
    b.set_base_vel(Vector3::new(0.0, 3.0, 0.0));
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    let snapshot = world.multibodies().to_vec();
    let tether = MultiBodyConstraint::point2point(ia, None, ib, None, Vector3::ZERO, Vector3::ZERO, &snapshot);
    world.add_multibody_constraint(tether);

    // --- Scene 4: two geared revolute links (ratio 2) driven by a motor on
    // the first; the second must spin at half speed in the opposite sense.
    let mut ga = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    ga.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    let mut gb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    gb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    ga.finalize_multidof();
    gb.finalize_multidof();
    let iga = world.add_multibody(ga);
    let igb = world.add_multibody(gb);
    let snapshot = world.multibodies().to_vec();
    let gear = MultiBodyConstraint::gear(iga, 0, igb, 0, 2.0, &snapshot);
    world.add_multibody_constraint(gear);
    let gear_motor = MultiBodyConstraint::joint_motor(iga, 0, 0, 1.0, 100.0, &snapshot);
    world.add_multibody_constraint(gear_motor);

    let dt: Real = 1.0 / 60.0;
    let mut bottom_speeds = Vec::new();
    let mut prev_q: Real = 0.0;
    let mut max_separation: Real = 0.0;
    let mut slider_speed = 0.0;
    let mut gear_speeds = (0.0, 0.0);
    for step in 0..900 {
        world.step_simulation(dt);
        let pend = world.multibody(MultiBodyHandle::from_index(0));
        let q = pend.joint_pos_multidof(0)[0];
        let qd = pend.joint_vel_multidof(0)[0];
        if prev_q.abs() < 0.1 && q.abs() < 0.1 && prev_q * q <= 0.0 {
            bottom_speeds.push(qd.abs());
        }
        prev_q = q;
        if step > 600 {
            // Steady-state readings.
            slider_speed = world.multibody(slider_idx).joint_vel_multidof(0)[0];
            gear_speeds = (
                world.multibody(iga).joint_vel_multidof(0)[0],
                world.multibody(igb).joint_vel_multidof(0)[0],
            );
        }
        let pa = world.multibody(ia).base_pos();
        let pb = world.multibody(ib).base_pos();
        max_separation = max_separation.max((pa - pb).length());
    }

    let first_bottom = bottom_speeds.first().copied().unwrap_or(0.0);
    let slider_pos = world.multibody(slider_idx).joint_pos_multidof(0)[0];
    println!(
        "pendulum bottom speed = {first_bottom:.3} rad/s, slider x = {slider_pos:.2} (v = {slider_speed:.3}), \
         tether max separation = {max_separation:.4} m, gear speeds = ({:.3}, {:.3}) rad/s",
        gear_speeds.0, gear_speeds.1
    );
    // The pendulum dropped 1 m from the horizontal; with I_eff = m·L² + I_z
    // the bottom speed is √(2·m·g·L/I_eff) ≈ 3.13 rad/s.
    assert!(first_bottom > 2.7 && first_bottom < 3.6, "pendulum bottom speed {first_bottom}");
    // The motor held the slider at 2 m/s and it travelled ~12 m in 15 s.
    assert!((slider_speed - 2.0).abs() < 0.1, "slider speed {slider_speed}");
    assert!((slider_pos - 30.0).abs() < 1.0, "slider position {slider_pos}");
    // The tether kept the two falling bases together.
    assert!(max_separation < 0.05, "tether separation {max_separation}");
    // The gear ratio 2 couples q̇_a + 2·q̇_b = 0 → q̇_b = −0.5 at q̇_a = 1.
    assert!((gear_speeds.0 - 1.0).abs() < 0.05, "gear drive speed {}", gear_speeds.0);
    assert!((gear_speeds.1 + 0.5).abs() < 0.05, "gear follower speed {}", gear_speeds.1);
    println!("OK: Featherstone multibody pendulum, motor slider, tether and gear all behave.");
}
