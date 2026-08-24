//! Tests for the multibody stepping stages of [`crate::world::PhysicsWorld`],
//! ported unchanged from the retired multibody-only `MultiBodyWorld` (the
//! fused world of phase A must reproduce its behavior exactly).

use bullet_math::{Quaternion, Real, Vector3};

use super::body::MultiBody;
use super::constraint::MultiBodyConstraint;
use crate::world::PhysicsWorld;

fn assert_close(a: Real, b: Real, tol: Real) {
    assert!(
        (a - b).abs() < tol,
        "expected {a} close to {b} within {tol}"
    );
}

/// A free base (zero links) falls under gravity (damping off, so the
/// velocity reaches exactly g·t).
#[test]
fn free_base_falls_under_gravity() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut mb = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    mb.set_linear_damping(0.0);
    mb.set_angular_damping(0.0);
    mb.finalize_multidof();
    world.add_multibody(mb);
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step_simulation(dt);
    }
    let body = world.multibody(0);
    // v = g·t after 1 s; position follows the semi-implicit Euler scheme
    // (velocity is integrated before position), so
    // y = −g·dt²·n·(n+1)/2 with n = 60.
    assert_close(body.base_vel().y, -9.81, 0.05);
    let n: Real = 60.0;
    let dt: Real = 1.0 / 60.0;
    let expected_y = -9.81 * dt * dt * n * (n + 1.0) / 2.0;
    assert_close(body.base_pos().y, expected_y, 0.05);
}

/// A fixed-base single-link revolute pendulum released from the
/// horizontal: the joint angle oscillates through the bottom and the
/// bottom speed matches the drop height (√(2·g·L)).
#[test]
fn revolute_pendulum_swings() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    // Pivot at the base origin; link COM hangs 1 m below the pivot
    // (d_vector), axis +Z.
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    // Release from the horizontal (q = +π/2).
    mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
    world.add_multibody(mb);
    let dt = 1.0 / 60.0;
    let mut bottom_speeds = Vec::new();
    let mut prev_q = world.multibody(0).joint_pos_multidof(0)[0];
    let mut prev_qd = world.multibody(0).joint_vel_multidof(0)[0];
    let mut crossings = 0;
    for _ in 0..600 {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let q = body.joint_pos_multidof(0)[0];
        let qd = body.joint_vel_multidof(0)[0];
        // Detect a bottom crossing: q changes sign while qd is largest
        // in magnitude (q near 0 going through the bottom).
        if prev_q.abs() < 0.15 && q.abs() < 0.15 && prev_q * q <= 0.0 && crossings < 4 {
            crossings += 1;
            bottom_speeds.push(qd.abs());
        }
        if prev_q * prev_qd < 0.0 && prev_q.abs() > 1.0 {
            // turn-around above the bottom — nothing to do
        }
        prev_q = q;
        prev_qd = qd;
    }
    // The pendulum must have swung through the bottom at least once
    // (released from 90°, the first crossing happens well within 1 s).
    assert!(crossings >= 1, "pendulum never crossed the bottom");
    // The pendulum: from the horizontal the COM drops 1 m, so with
    // I_eff = m·L² + I_z = 2 the bottom angular speed is
    // √(2·m·g·L / I_eff) = √(9.81) ≈ 3.13 rad/s (small damping).
    let first = bottom_speeds.first().copied().unwrap_or(0.0);
    assert!(
        first > 2.7 && first < 3.6,
        "bottom speed {first} not in [2.7, 3.6] (expected ≈ 3.13)"
    );
}

/// Forward kinematics of a two-link chain: link 1 at a fixed angle, the
/// endpoint lands at the expected world position.
#[test]
fn forward_kinematics_two_link_chain() {
    let mut mb = MultiBody::new(2, 1.0, Vector3::ONE, true, false);
    // Link 0: revolute about +Z at the origin; COM 1 m along +X.
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    // Link 1: revolute about +Z; pivot at link 0's COM, COM 1 m further
    // along +X.
    mb.setup_revolute(
        1,
        1.0,
        Vector3::ONE,
        Some(0),
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
    );
    mb.finalize_multidof();
    mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
    mb.set_joint_pos_multidof(1, &[-bullet_math::FRAC_PI_2]);
    mb.forward_kinematics();
    // Link 0's COM: (1,0,0) rotated +90° about Z → (0,1,0).
    let p0 = mb.local_pos_to_world(0, Vector3::ZERO);
    assert_close(p0.x, 0.0, 1e-5);
    assert_close(p0.y, 1.0, 1e-5);
    // Link 1's pivot sits 1 m along link 0's X (which points +Y after the
    // +90° rotation), so the pivot is at (0,2,0); link 1's COM is 1 m
    // further along its own X, which is back to +X at q1 = −90°.
    let p1 = mb.local_pos_to_world(1, Vector3::ZERO);
    assert_close(p1.x, 1.0, 1e-4);
    assert_close(p1.y, 2.0, 1e-4);
}

/// A point-to-point constraint between two free bases keeps the pivots
/// together even under an asymmetric initial velocity.
#[test]
fn point2point_keeps_bodies_together() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let b = MultiBody::new(0, 2.0, Vector3::splat(1.5), false, false);
    // The constraint ctor reads the finalized bodies' dof counts, so
    // pass a snapshot before moving the real bodies into the world.
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    world.multibody_mut(ia).finalize_multidof();
    world.multibody_mut(ib).finalize_multidof();
    let snapshot = world.multibodies().to_vec();
    let constraint = MultiBodyConstraint::point2point(
        ia,
        None,
        ib,
        None,
        Vector3::ZERO,
        Vector3::ZERO,
        &snapshot,
    );
    world.add_multibody_constraint(constraint);
    // Body B starts with an upward velocity; the constraint must keep
    // the pivots coincident while the pair decelerates.
    world.multibody_mut(ib).set_base_vel(Vector3::new(0.0, 3.0, 0.0));
    let dt = 1.0 / 60.0;
    let mut max_separation: Real = 0.0;
    for _ in 0..120 {
        world.step_simulation(dt);
        let pa = world.multibody(ia).base_pos();
        let pb = world.multibody(ib).base_pos();
        max_separation = max_separation.max((pa - pb).length());
    }
    assert!(
        max_separation < 0.05,
        "P2P pivots drifted apart by {max_separation} m"
    );
}

/// A joint motor drives the joint velocity to the target.
#[test]
fn joint_motor_reaches_target_velocity() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    mb.finalize_multidof();
    let body = world.add_multibody(mb);
    let snapshot = world.multibodies().to_vec();
    let motor = MultiBodyConstraint::joint_motor(body, 0, 0, 2.0, 1000.0, &snapshot);
    world.add_multibody_constraint(motor);
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step_simulation(dt);
    }
    let qd = world.multibody(body).joint_vel_multidof(0)[0];
    assert!(
        (qd - 2.0).abs() < 0.1,
        "motor did not reach 2 rad/s, got {qd}"
    );
    // The joint angle must have advanced.
    let q = world.multibody(body).joint_pos_multidof(0)[0];
    assert!(q > 1.0, "motor joint angle did not advance, got {q}");
}

/// A body at rest with zero gravity falls asleep after the deactivation
/// window, and a constraint to an awake body wakes it (the simplified
/// island-merge semantics).
#[test]
fn body_sleeps_after_rest_and_wakes_via_constraint() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut a = MultiBody::new(0, 1.0, Vector3::ONE, false, true);
    let mut b = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    a.finalize_multidof();
    b.finalize_multidof();
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    assert!(world.multibody(ia).is_awake());
    let dt = 1.0 / 60.0;
    for _ in 0..150 {
        // 2.5 s > the 2 s sleep timeout.
        world.step_simulation(dt);
    }
    assert!(!world.multibody(ia).is_awake(), "body should be asleep");
    assert!(world.multibody(ib).is_awake(), "non-sleeping body stays awake");
    // A constraint to the awake body wakes the sleeping one.
    let snapshot = world.multibodies().to_vec();
    let p2p = MultiBodyConstraint::point2point(
        ia,
        None,
        ib,
        None,
        Vector3::ZERO,
        Vector3::ZERO,
        &snapshot,
    );
    world.add_multibody_constraint(p2p);
    world.step_simulation(dt);
    assert!(world.multibody(ia).is_awake(), "body should wake via constraint");
}

/// Two free bases welded by a fixed constraint move as one rigid body.
#[test]
fn fixed_constraint_welds_bodies() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let b = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    world.multibody_mut(ia).finalize_multidof();
    world.multibody_mut(ib).finalize_multidof();
    let snapshot = world.multibodies().to_vec();
    let fixed = MultiBodyConstraint::fixed(
        ia,
        None,
        ib,
        None,
        Vector3::ZERO,
        Vector3::ZERO,
        bullet_math::Matrix3::IDENTITY,
        bullet_math::Matrix3::IDENTITY,
        &snapshot,
    );
    world.add_multibody_constraint(fixed);
    world.multibody_mut(ib).set_base_vel(Vector3::new(2.0, 0.0, 0.0));
    let dt = 1.0 / 60.0;
    let mut max_rel: Real = 0.0;
    for _ in 0..120 {
        world.step_simulation(dt);
        let pa = world.multibody(ia).base_pos();
        let pb = world.multibody(ib).base_pos();
        let qa = world.multibody(ia).base_quat;
        let qb = world.multibody(ib).base_quat;
        max_rel = max_rel.max((pa - pb).length());
        // Relative orientation stays identity.
        let rel = qa.inverse() * qb;
        assert!(rel.x.abs() < 1e-3 && rel.y.abs() < 1e-3 && rel.z.abs() < 1e-3);
    }
    assert!(
        max_rel < 0.05,
        "welded bodies drifted apart by {max_rel} m"
    );
}

/// A slider constraint lets the bodies separate along the slide axis
/// only.
#[test]
fn slider_constrains_perpendicular_motion() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let b = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    world.multibody_mut(ia).finalize_multidof();
    world.multibody_mut(ib).finalize_multidof();
    let snapshot = world.multibodies().to_vec();
    let slider = MultiBodyConstraint::slider(
        ia,
        None,
        ib,
        None,
        Vector3::ZERO,
        Vector3::ZERO,
        bullet_math::Matrix3::IDENTITY,
        bullet_math::Matrix3::IDENTITY,
        Vector3::X,
        &snapshot,
    );
    world.add_multibody_constraint(slider);
    // Body B slides along +X (allowed) with a Y component (blocked).
    world
        .multibody_mut(ib)
        .set_base_vel(Vector3::new(3.0, 3.0, 0.0));
    let dt = 1.0 / 60.0;
    for _ in 0..90 {
        world.step_simulation(dt);
    }
    let pa = world.multibody(ia).base_pos();
    let pb = world.multibody(ib).base_pos();
    let rel = pb - pa;
    assert!(rel.x > 0.5, "slider did not slide along X: {rel:?}");
    assert!(
        rel.y.abs() < 0.05 && rel.z.abs() < 0.05,
        "slider allowed off-axis motion: {rel:?}"
    );
}

/// A joint limit on a fixed-base revolute joint stops the joint at the
/// upper bound: released past it, the limit pushes the position back and
/// the joint never swings far outside `[lower, upper]`.
#[test]
fn joint_limit_clamps_revolute_joint() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    mb.finalize_multidof();
    // Start inside the limit with a velocity toward the upper bound.
    mb.set_joint_pos_multidof(0, &[0.0]);
    let ib = world.add_multibody(mb);
    world.multibody_mut(ib).set_joint_vel_multidof(0, &[1.0]);
    let snapshot = world.multibodies().to_vec();
    let limit = MultiBodyConstraint::joint_limit(ib, 0, -0.3, 0.3, &snapshot);
    world.add_multibody_constraint(limit);
    let dt = 1.0 / 60.0;
    let mut max_q: Real = 0.0;
    let mut min_q: Real = 0.0;
    for _ in 0..600 {
        world.step_simulation(dt);
        let q = world.multibody(ib).joint_pos_multidof(0)[0];
        max_q = max_q.max(q);
        min_q = min_q.min(q);
    }
    // The joint is clamped near the limits (a small overshoot is allowed
    // for the per-step violation, but nothing like the unconstrained
    // 5 rad/s · 10 s travel).
    assert!(
        max_q < 0.5 && min_q > -0.5,
        "joint limit failed: q in [{min_q}, {max_q}]"
    );
    // The limit is unilateral: with no other torque the joint oscillates
    // between the bounds and is not damped to rest, but it must have hit
    // the bounds (velocity was large enough to reach them).
    assert!(max_q > 0.28, "joint never reached the upper bound: {max_q}");
    assert!(min_q < -0.28, "joint never reached the lower bound: {min_q}");
}

/// A spherical joint limit pulls a tilted joint back inside the swing
/// cone: the swing angle stays within the configured ranges.
#[test]
fn spherical_joint_limit_pulls_tilt_back() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    mb.setup_spherical(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::ZERO,
        Vector3::new(0.0, 0.0, -1.0),
    );
    mb.finalize_multidof();
    // Tilt the joint 60 degrees about X — far outside the 0.2 rad limit.
    let half = bullet_math::FRAC_PI_2 / 3.0;
    let q = [half.sin(), 0.0, 0.0, half.cos()];
    mb.set_joint_pos_multidof(0, &q);
    let ib = world.add_multibody(mb);
    let snapshot = world.multibodies().to_vec();
    let limit = MultiBodyConstraint::spherical_joint_limit(ib, 0, 0.2, 0.2, 0.2, 100.0, &snapshot);
    world.add_multibody_constraint(limit);
    let dt = 1.0 / 60.0;
    let mut min_swing_x: Real = 2.0;
    for _ in 0..600 {
        world.step_simulation(dt);
        let body = world.multibody(ib);
        let jp = body.joint_pos_multidof(0);
        let quat = Quaternion::from_xyzw(jp[0], jp[1], jp[2], jp[3]);
        // Swing = rotation of the link's +Z axis (twist is about +Z, so
        // the cone angle is the angle between Z and the rotated Z).
        let z_rot = quat * Vector3::Z;
        let swing = z_rot.angle_between(Vector3::Z);
        min_swing_x = min_swing_x.min(swing);
    }
    // The limit drives the swing well inside the 0.2 rad range.
    assert!(
        min_swing_x < 0.25,
        "spherical limit did not pull the swing inside the cone: {min_swing_x}"
    );
}

/// A spherical joint motor drives a tilted joint back toward the desired
/// orientation (identity by default).
#[test]
fn spherical_joint_motor_returns_to_identity() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    mb.setup_spherical(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::ZERO,
        Vector3::new(0.0, 0.0, -1.0),
    );
    mb.finalize_multidof();
    let half = bullet_math::FRAC_PI_2 / 3.0;
    let q = [half.sin(), 0.0, 0.0, half.cos()];
    mb.set_joint_pos_multidof(0, &q);
    let ib = world.add_multibody(mb);
    let snapshot = world.multibodies().to_vec();
    let motor = MultiBodyConstraint::spherical_joint_motor(ib, 0, 100.0, &snapshot);
    world.add_multibody_constraint(motor);
    let dt = 1.0 / 60.0;
    let mut min_swing: Real = 2.0;
    for _ in 0..600 {
        world.step_simulation(dt);
        let body = world.multibody(ib);
        let jp = body.joint_pos_multidof(0);
        let quat = Quaternion::from_xyzw(jp[0], jp[1], jp[2], jp[3]);
        let z_rot = quat * Vector3::Z;
        let swing = z_rot.angle_between(Vector3::Z);
        min_swing = min_swing.min(swing);
    }
    // The motor drives the tilt to (near) zero.
    assert!(
        min_swing < 0.05,
        "spherical motor did not return the joint to identity: {min_swing}"
    );
}
