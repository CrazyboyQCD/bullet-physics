//! Multibody (Featherstone) probe — the Rust counterpart of
//! `tools/multibody_probe.cpp`. Runs the same scenes with the same
//! parameters and prints per-frame trajectories so the two implementations
//! can be diffed numerically.
//!
//! Usage: `cargo run -p bullet --example multibody_probe -- <scene> [frames]`
//! with scene in {free_base, pendulum, two_link, p2p, motor, fixed,
//! prismatic, spherical, planar, gear, joint_limit, spherical_limit,
//! spherical_motor}.

use bullet::dynamics::multibody::constraint::MultiBodyConstraint;
use bullet::dynamics::MultiBody;
use bullet::dynamics::PhysicsWorld;
use bullet::math::{Quaternion, Real, Vector3};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scene = args.get(1).map(String::as_str).unwrap_or("pendulum");
    let frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let dt: Real = 1.0 / 60.0;
    match scene {
        "free_base" => scene_free_base(dt, frames),
        "pendulum" => scene_pendulum(dt, frames),
        "two_link" => scene_two_link(dt, frames),
        "p2p" => scene_p2p(dt, frames),
        "motor" => scene_motor(dt, frames),
        "fixed" => scene_fixed(dt, frames),
        "prismatic" => scene_prismatic(dt, frames),
        "spherical" => scene_spherical(dt, frames),
        "planar" => scene_planar(dt, frames),
        "gear" => scene_gear(dt, frames),
        "joint_limit" => scene_joint_limit(dt, frames),
        "spherical_limit" => scene_spherical_limit(dt, frames),
        "spherical_motor" => scene_spherical_motor(dt, frames),
        other => {
            eprintln!("unknown scene {other}");
            std::process::exit(1);
        }
    }
}

fn scene_free_base(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut body = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    body.set_linear_damping(0.0);
    body.set_angular_damping(0.0);
    body.finalize_multidof();
    world.add_multibody(body);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let p = body.base_pos();
        let v = body.base_vel();
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            p.x, p.y, p.z, v.x, v.y, v.z
        );
    }
}

fn scene_pendulum(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    body.finalize_multidof();
    body.set_joint_pos_multidof(0, &[bullet::math::FRAC_PI_2]);
    world.add_multibody(body);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        println!("{f} {:.6} {:.6}", body.joint_pos_multidof(0)[0], body.joint_vel_multidof(0)[0]);
    }
}

fn scene_two_link(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut body = MultiBody::new(2, 1.0, Vector3::ONE, true, false);
    body.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    body.setup_revolute(
        1,
        1.0,
        Vector3::ONE,
        Some(0),
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
    );
    body.finalize_multidof();
    world.add_multibody(body);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6}",
            body.joint_pos_multidof(0)[0],
            body.joint_vel_multidof(0)[0],
            body.joint_pos_multidof(1)[0],
            body.joint_vel_multidof(1)[0]
        );
    }
}

fn scene_p2p(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let mut b = MultiBody::new(0, 2.0, Vector3::splat(1.5), false, false);
    a.finalize_multidof();
    b.finalize_multidof();
    b.set_base_vel(Vector3::new(0.0, 3.0, 0.0));
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    let snapshot = world.multibodies().to_vec();
    let c = MultiBodyConstraint::point2point(ia, None, ib, None, Vector3::ZERO, Vector3::ZERO, &snapshot);
    world.add_multibody_constraint(c);
    for f in 0..frames {
        world.step_simulation(dt);
        let pa = world.multibody(ia).base_pos();
        let pb = world.multibody(ib).base_pos();
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            pa.x, pa.y, pa.z, pb.x, pb.y, pb.z
        );
    }
}

fn scene_motor(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    body.finalize_multidof();
    let ib = world.add_multibody(body);
    let snapshot = world.multibodies().to_vec();
    let motor = MultiBodyConstraint::joint_motor(ib, 0, 0, 2.0, 1000.0, &snapshot);
    world.add_multibody_constraint(motor);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(ib);
        println!("{f} {:.6} {:.6}", body.joint_pos_multidof(0)[0], body.joint_vel_multidof(0)[0]);
    }
}

fn scene_fixed(dt: Real, frames: usize) {    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    let mut b = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    a.finalize_multidof();
    b.finalize_multidof();
    b.set_base_vel(Vector3::new(2.0, 0.0, 0.0));
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    let snapshot = world.multibodies().to_vec();
    let c = MultiBodyConstraint::fixed(
        ia,
        None,
        ib,
        None,
        Vector3::ZERO,
        Vector3::ZERO,
        bullet::math::Matrix3::IDENTITY,
        bullet::math::Matrix3::IDENTITY,
        &snapshot,
    );
    world.add_multibody_constraint(c);
    for f in 0..frames {
        world.step_simulation(dt);
        let pa = world.multibody(ia).base_pos();
        let pb = world.multibody(ib).base_pos();
        let qa = world.multibody(ia).world_to_base_rot();
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            pa.x, pa.y, pa.z, pb.x, pb.y, pb.z, qa.x, qa.y, qa.z, qa.w
        );
    }
}

fn scene_prismatic(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_prismatic(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Y,
        Vector3::ZERO,
        Vector3::ZERO,
    );
    body.finalize_multidof();
    world.add_multibody(body);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        println!("{f} {:.6} {:.6}", body.joint_pos_multidof(0)[0], body.joint_vel_multidof(0)[0]);
    }
}

fn scene_spherical(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_spherical(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::ZERO,
        Vector3::new(0.0, 0.0, -1.0),
    );
    body.finalize_multidof();
    // Tilt the joint 60 degrees about X.
    let half = bullet::math::FRAC_PI_2 / 3.0;
    let q = [half.sin(), 0.0, 0.0, half.cos()];
    body.set_joint_pos_multidof(0, &q);
    world.add_multibody(body);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let jp = body.joint_pos_multidof(0);
        let jv = body.joint_vel_multidof(0);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            jp[0], jp[1], jp[2], jp[3], jv[0], jv[1], jv[2]
        );
    }
}

fn scene_planar(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_planar(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
    );
    body.finalize_multidof();
    body.set_joint_vel_multidof(0, &[0.0, 1.0, 0.0]);
    world.add_multibody(body);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let jp = body.joint_pos_multidof(0);
        let jv = body.joint_vel_multidof(0);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            jp[0], jp[1], jp[2], jv[0], jv[1], jv[2]
        );
    }
}

fn scene_gear(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut a = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    a.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    let mut b = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    b.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    a.finalize_multidof();
    b.finalize_multidof();
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    let snapshot = world.multibodies().to_vec();
    let gear = MultiBodyConstraint::gear(ia, 0, ib, 0, 2.0, &snapshot);
    world.add_multibody_constraint(gear);
    let motor = MultiBodyConstraint::joint_motor(ia, 0, 0, 1.0, 1000.0, &snapshot);
    world.add_multibody_constraint(motor);
    for f in 0..frames {
        world.step_simulation(dt);
        let a = world.multibody(ia);
        let b = world.multibody(ib);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6}",
            a.joint_pos_multidof(0)[0],
            a.joint_vel_multidof(0)[0],
            b.joint_pos_multidof(0)[0],
            b.joint_vel_multidof(0)[0]
        );
    }
}

fn scene_joint_limit(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(1.0, 0.0, 0.0),
    );
    body.finalize_multidof();
    body.set_joint_pos_multidof(0, &[0.25]);
    body.set_joint_vel_multidof(0, &[2.0]);
    let ib = world.add_multibody(body);
    let snapshot = world.multibodies().to_vec();
    let limit = MultiBodyConstraint::joint_limit(ib, 0, -0.3, 0.3, &snapshot);
    world.add_multibody_constraint(limit);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(ib);
        println!("{f} {:.6} {:.6}", body.joint_pos_multidof(0)[0], body.joint_vel_multidof(0)[0]);
    }
}

fn scene_spherical_limit(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_spherical(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::ZERO,
        Vector3::new(0.0, 0.0, -1.0),
    );
    body.finalize_multidof();
    let half = bullet::math::FRAC_PI_2 / 3.0;
    let q = [half.sin(), 0.0, 0.0, half.cos()];
    body.set_joint_pos_multidof(0, &q);
    let ib = world.add_multibody(body);
    let snapshot = world.multibodies().to_vec();
    let limit = MultiBodyConstraint::spherical_joint_limit(ib, 0, 0.3, 0.3, 0.5, 100.0, &snapshot);
    world.add_multibody_constraint(limit);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(ib);
        let jp = body.joint_pos_multidof(0);
        let jv = body.joint_vel_multidof(0);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            jp[0], jp[1], jp[2], jp[3], jv[0], jv[1], jv[2]
        );
    }
}

fn scene_spherical_motor(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut body = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    body.setup_spherical(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::ZERO,
        Vector3::new(0.0, 0.0, -1.0),
    );
    body.finalize_multidof();
    let half = bullet::math::FRAC_PI_2 / 3.0;
    let q = [half.sin(), 0.0, 0.0, half.cos()];
    body.set_joint_pos_multidof(0, &q);
    let ib = world.add_multibody(body);
    let snapshot = world.multibodies().to_vec();
    let motor = MultiBodyConstraint::spherical_joint_motor(ib, 0, 100.0, &snapshot);
    world.add_multibody_constraint(motor);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(ib);
        let jp = body.joint_pos_multidof(0);
        let jv = body.joint_vel_multidof(0);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            jp[0], jp[1], jp[2], jp[3], jv[0], jv[1], jv[2]
        );
    }
}
