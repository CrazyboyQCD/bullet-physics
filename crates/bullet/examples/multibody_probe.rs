//! Multibody (Featherstone) probe — the Rust counterpart of
//! `tools/multibody_probe.cpp`. Runs the same scenes with the same
//! parameters and prints per-frame trajectories so the two implementations
//! can be diffed numerically.
//!
//! Usage: `cargo run -p bullet --example multibody_probe -- <scene> [frames]`
//! with scene in {free_base, pendulum, two_link, p2p, motor, fixed,
//! prismatic, spherical, planar, gear, joint_limit, spherical_limit,
//! spherical_motor, link_ground, base_ground, mb_pushes_rb, rb_hits_mb,
//! mb_mb, self_collision, joint_feedback, joint_feedback_contact}.

use bullet::dynamics::multibody::constraint::MultiBodyConstraint;
use bullet::dynamics::{MultiBody, PhysicsWorld, RigidBody};
use bullet::math::{Quaternion, Real, Vector3};
use bullet::shapes::CollisionShape;

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
        "link_ground" => scene_link_ground(dt, frames),
        "base_ground" => scene_base_ground(dt, frames),
        "mb_pushes_rb" => scene_mb_pushes_rb(dt, frames),
        "rb_hits_mb" => scene_rb_hits_mb(dt, frames),
        "mb_mb" => scene_mb_mb(dt, frames),
        "self_collision" => scene_self_collision(dt, frames),
        "joint_feedback" => scene_joint_feedback(dt, frames),
        "joint_feedback_contact" => scene_joint_feedback_contact(dt, frames),
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

// ---------------------------------------------------------------------------
// Collider scenes (multibody-collision.md §7): link/base colliders meeting
// static and dynamic rigid bodies. Mirrors the C++ twins scene for scene.
// ---------------------------------------------------------------------------

/// A static ground box, top surface at y = 0, half extents (50, 0.5, 50).
fn add_ground_box(world: &mut PhysicsWorld) {
    world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::new(50.0, 0.5, 50.0)),
        Vector3::new(0.0, -0.5, 0.0),
    ));
}

/// §7.1 link_ground: a fixed-base pendulum with a sphere collider on the link
/// swings from the horizontal into a static ground box. The pivot sits 1.25 m
/// up so the arc's lowest point (link COM y = 0.25, r = 0.3) penetrates 0.05 m
/// — a shallow, clean first impact.
fn scene_link_ground(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    add_ground_box(&mut world);
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
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
    // q = +pi/2 rotates the COM from straight down to (+1, 0) — the
    // horizontal release, the same pose as the `pendulum` scene.
    mb.set_joint_pos_multidof(0, &[bullet::math::FRAC_PI_2]);
    mb.base_pos = Vector3::new(0.0, 1.25, 0.0);
    mb.set_link_collider(0, CollisionShape::sphere(0.3), 0.5, 0.0);
    world.add_multibody(mb);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let p = body.link_world_position(0);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6}",
            body.joint_pos_multidof(0)[0],
            body.joint_vel_multidof(0)[0],
            p.x,
            p.y,
            p.z
        );
    }
}

/// §7.2 base_ground: a free base with a box collider drops onto the ground
/// from y = 3; records the base position and velocity.
fn scene_base_ground(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    add_ground_box(&mut world);
    let mut mb = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    mb.set_linear_damping(0.0);
    mb.set_angular_damping(0.0);
    mb.finalize_multidof();
    mb.base_pos = Vector3::new(0.0, 3.0, 0.0);
    mb.set_base_collider(CollisionShape::box_shape(Vector3::splat(0.5)), 0.5, 0.0);
    world.add_multibody(mb);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let p = body.base_pos();
        let v = body.base_vel();
        println!("{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}", p.x, p.y, p.z, v.x, v.y, v.z);
    }
}

/// §7.3 mb_pushes_rb: a motorized link with a sphere collider sweeps through
/// a free-floating dynamic box (gravity off, no ground). The link COM starts
/// at (+1, 0) and circles clockwise at 2 rad/s; the box sits at the bottom of
/// the arc.
fn scene_mb_pushes_rb(dt: Real, frames: usize) {
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
    mb.set_link_collider(0, CollisionShape::sphere(0.3), 0.5, 0.0);
    let ib = world.add_multibody(mb);
    let snapshot = world.multibodies().to_vec();
    let motor = MultiBodyConstraint::joint_motor(ib, 0, 0, -2.0, 1000.0, &snapshot);
    world.add_multibody_constraint(motor);
    let box_idx = world.add_body(
        RigidBody::new_dynamic(CollisionShape::box_shape(Vector3::splat(0.5)), Vector3::new(0.0, -1.0, 0.0), 1.0)
            .disable_deactivation(),
    );
    for f in 0..frames {
        world.step_simulation(dt);
        let box_pos = world.body(box_idx).position;
        let box_vel = world.body(box_idx).linear_velocity;
        let lp = world.multibody(ib).link_world_position(0);
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            box_pos.x, box_pos.y, box_pos.z, box_vel.x, box_vel.y, box_vel.z, lp.x, lp.y, lp.z
        );
    }
}

/// §7.4 rb_hits_mb: a dynamic rigid sphere falls at 6 m/s onto the base
/// sphere collider of a free multibody floating at the origin.
fn scene_rb_hits_mb(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut mb = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    mb.set_linear_damping(0.0);
    mb.set_angular_damping(0.0);
    mb.finalize_multidof();
    mb.set_base_collider(CollisionShape::sphere(0.5), 0.5, 0.0);
    world.add_multibody(mb);
    let ball = world.add_body(
        RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::new(0.0, 3.0, 0.0), 1.0)
            .velocity(Vector3::new(0.0, -6.0, 0.0))
            .disable_deactivation(),
    );
    for f in 0..frames {
        world.step_simulation(dt);
        let mbp = world.multibody(0).base_pos();
        let p = world.body(ball).position;
        let v = world.body(ball).linear_velocity;
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            mbp.x, mbp.y, mbp.z, p.x, p.y, p.z, v.x, v.y, v.z
        );
    }
}

/// §7.5 mb_mb: two free single-link multibodies with sphere colliders on
/// their links collide head-on (A drifts +X at 2 m/s into B at rest 3 m away,
/// gravity off). The e/d vectors are zero so each link COM rides on its base.
fn scene_mb_mb(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut a = MultiBody::new(1, 1.0, Vector3::ONE, false, false);
    a.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::ZERO,
    );
    a.set_linear_damping(0.0);
    a.set_angular_damping(0.0);
    a.finalize_multidof();
    a.set_base_vel(Vector3::new(2.0, 0.0, 0.0));
    a.set_link_collider(0, CollisionShape::sphere(0.5), 0.5, 0.0);
    let mut b = MultiBody::new(1, 1.0, Vector3::ONE, false, false);
    b.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::ZERO,
    );
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.0);
    b.finalize_multidof();
    b.base_pos = Vector3::new(3.0, 0.0, 0.0);
    b.set_link_collider(0, CollisionShape::sphere(0.5), 0.5, 0.0);
    let ia = world.add_multibody(a);
    let ib = world.add_multibody(b);
    for f in 0..frames {
        world.step_simulation(dt);
        let pa = world.multibody(ia).base_pos();
        let pb = world.multibody(ib).base_pos();
        println!("{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}", pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
    }
}

/// §7.6 self_collision: a fixed-base 3-link chain — link 0 hanging down from
/// the base, link 1 hanging below link 0, and link 2 a sibling of link 1 whose
/// COM sits at (+0.6, +0.2, 0) (e = (+0.6, +1, 0) from link 0's COM, d =
/// (0, +0.2, 0)), overlapping the base sphere by ~0.17 m — a non-adjacent
/// pair, and the offset keeps a torque arm on both joint axes (a COM on the
/// base-to-sphere line would make the contact force pass through both pivot
/// axes and the row degenerate to zero). Phase `on` (default self-collision):
/// the contact kicks the chain apart while gravity swings it; phase `off`: the
/// overlapping pair is filtered and the chain evolves under pure ABA.
fn scene_self_collision(dt: Real, frames: usize) {
    for phase in 0..2 {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let mut mb = MultiBody::new(3, 1.0, Vector3::ONE, true, false);
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
        mb.setup_revolute(
            1,
            1.0,
            Vector3::ONE,
            Some(0),
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::ZERO,
        );
        mb.setup_revolute(
            2,
            1.0,
            Vector3::ONE,
            Some(0),
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::new(0.6, 1.0, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
        );
        mb.finalize_multidof();
        if phase == 1 {
            mb.set_has_self_collision(false);
        }
        mb.set_base_collider(CollisionShape::sphere(0.5), 0.5, 0.0);
        mb.set_link_collider(0, CollisionShape::sphere(0.3), 0.5, 0.0);
        mb.set_link_collider(1, CollisionShape::sphere(0.3), 0.5, 0.0);
        mb.set_link_collider(2, CollisionShape::sphere(0.3), 0.5, 0.0);
        world.add_multibody(mb);
        let label = if phase == 0 { "on" } else { "off" };
        for f in 0..frames {
            world.step_simulation(dt);
            let body = world.multibody(0);
            println!(
                "{label} {f} {:.6} {:.6} {:.6}",
                body.joint_pos_multidof(0)[0],
                body.joint_pos_multidof(1)[0],
                body.joint_pos_multidof(2)[0]
            );
        }
    }
}

/// Phase E joint feedback: a fixed-base single-link revolute pendulum
/// released from the horizontal with a joint feedback slot on the link;
/// records the joint state and the reaction force/torque in the link frame
/// (the defaults). No contacts — the constraint pass adds ~zero.
fn scene_joint_feedback(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
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
    mb.enable_link_joint_feedback(0);
    world.add_multibody(mb);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let fb = body.link_joint_feedback(0).expect("feedback enabled");
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            body.joint_pos_multidof(0)[0],
            body.joint_vel_multidof(0)[0],
            fb.top.x,
            fb.top.y,
            fb.top.z,
            fb.bottom.x,
            fb.bottom.y,
            fb.bottom.z
        );
    }
}

/// Phase E joint feedback + contact: a link-ground-style pendulum with a
/// feedback slot; records the joint state and reaction force/torque through
/// the ground impact and settle (the constraint pass accumulation of the
/// contact support). The pivot sits 1.05 m up so the sphere (r = 0.3, COM
/// 1 m below the pivot) first touches the ground at a ~41 degree lean — a
/// well-conditioned contact (the 1.25 m pivot settles at a ~16 degree lean
/// where the normal-friction coupling is near-singular and the solver
/// impulses inflate in both engines).
fn scene_joint_feedback_contact(dt: Real, frames: usize) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    add_ground_box(&mut world);
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
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
    mb.set_joint_pos_multidof(0, &[bullet::math::FRAC_PI_2]);
    mb.base_pos = Vector3::new(0.0, 1.05, 0.0);
    mb.set_link_collider(0, CollisionShape::sphere(0.3), 0.0, 0.0);
    mb.enable_link_joint_feedback(0);
    world.add_multibody(mb);
    for f in 0..frames {
        world.step_simulation(dt);
        let body = world.multibody(0);
        let fb = body.link_joint_feedback(0).expect("feedback enabled");
        println!(
            "{f} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            body.joint_pos_multidof(0)[0],
            body.joint_vel_multidof(0)[0],
            fb.top.x,
            fb.top.y,
            fb.top.z,
            fb.bottom.x,
            fb.bottom.y,
            fb.bottom.z
        );
    }
}
