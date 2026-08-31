//! SoftBody probe — the Rust counterpart of `tools/softbody_probe.cpp`.
//! Runs the same rope/cloth scenes with the same parameters and prints
//! per-frame trajectories so the two implementations can be diffed
//! numerically.
//!
//! Usage: `cargo run -p bullet --example softbody_probe -- <scene> [frames]`
//! with scene in {rope_pendulum, rope_ground, cloth_plane}.

use bullet::dynamics::{create_patch, create_rope, PhysicsWorld, RigidBody};
use bullet::math::{Real, Vector3};
use bullet::shapes::CollisionShape;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scene = args.get(1).map(String::as_str).unwrap_or("rope_pendulum");
    let frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let dt: Real = 1.0 / 60.0;
    match scene {
        "rope_pendulum" => scene_rope_pendulum(dt, frames),
        "rope_ground" => scene_rope_ground(dt, frames),
        "cloth_plane" => scene_cloth_plane(dt, frames),
        "patch_catenary" => scene_patch_catenary(dt, frames),
        "soft_soft_vf" => scene_soft_soft_vf(dt, frames),
        other => {
            eprintln!("unknown scene {other}");
            std::process::exit(1);
        }
    }
}

fn make_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world
}

fn add_ground(world: &mut PhysicsWorld) {
    world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
}

/// Minimal `btSoftBodyHelpers::CreatePatch` wrapper (the `create_patch`
/// helper is the canonical port now; kept for the fixeds subset the probe
/// uses).
fn build_patch(
    corner00: Vector3,
    corner10: Vector3,
    corner01: Vector3,
    corner11: Vector3,
    resx: usize,
    resy: usize,
    fixeds: u32,
) -> bullet::dynamics::SoftBody {
    create_patch(corner00, corner10, corner01, corner11, resx, resy, fixeds, true, 0.0)
}

fn scene_rope_pendulum(dt: Real, frames: usize) {
    let mut world = make_world();
    let rope = world.add_soft_body(create_rope(
        Vector3::new(0.0, 8.0, 0.0),
        Vector3::new(4.0, 8.0, 0.0),
        16,
        1,
    ));
    world.soft_body_mut(rope).cfg.piterations = 4;
    let last = world.soft_body(rope).nodes.len() - 1;
    for f in 0..frames {
        world.step_simulation(dt);
        let sb = world.soft_body(rope);
        let tip = sb.nodes[last].x;
        let com = sb.get_center_of_mass();
        println!("{f} {:.6} {:.6} {:.6} {:.6}", tip.x, tip.y, com.x, com.y);
    }
}

fn scene_rope_ground(dt: Real, frames: usize) {
    let mut world = make_world();
    add_ground(&mut world);
    let rope = world.add_soft_body(create_rope(
        Vector3::new(-3.0, 5.0, 0.0),
        Vector3::new(3.0, 5.0, 0.0),
        20,
        0,
    ));
    world.soft_body_mut(rope).cfg.piterations = 4;
    for f in 0..frames {
        world.step_simulation(dt);
        let sb = world.soft_body(rope);
        let min_y = sb.nodes.iter().map(|n| n.x.y).fold(Real::INFINITY, Real::min);
        let com = sb.get_center_of_mass();
        println!("{f} {:.6} {:.6} {:.6}", min_y, com.y, com.x);
    }
}

fn scene_cloth_plane(dt: Real, frames: usize) {
    let mut world = make_world();
    add_ground(&mut world);
    let cloth = world.add_soft_body(build_patch(
        Vector3::new(-2.0, 5.0, -2.0),
        Vector3::new(2.0, 5.0, -2.0),
        Vector3::new(-2.0, 5.0, 2.0),
        Vector3::new(2.0, 5.0, 2.0),
        12,
        12,
        0,
    ));
    world.soft_body_mut(cloth).cfg.piterations = 4;
    for f in 0..frames {
        world.step_simulation(dt);
        let sb = world.soft_body(cloth);
        let min_y = sb.nodes.iter().map(|n| n.x.y).fold(Real::INFINITY, Real::min);
        let com = sb.get_center_of_mass();
        println!("{f} {:.6} {:.6} {:.6} {:.6}", min_y, com.y, com.x, com.z);
    }
}

fn scene_patch_catenary(dt: Real, frames: usize) {
    let mut world = make_world();
    let cloth = world.add_soft_body(create_patch(
        Vector3::new(-2.0, 6.0, -2.0),
        Vector3::new(2.0, 6.0, -2.0),
        Vector3::new(-2.0, 6.0, 2.0),
        Vector3::new(2.0, 6.0, 2.0),
        12,
        12,
        1 | 2 | 4 | 8,
        true,
        0.0,
    ));
    world.soft_body_mut(cloth).cfg.piterations = 4;
    for f in 0..frames {
        world.step_simulation(dt);
        let sb = world.soft_body(cloth);
        let min_y = sb.nodes.iter().map(|n| n.x.y).fold(Real::INFINITY, Real::min);
        let center_y = sb.nodes[5 * 12 + 5].x.y;
        let com_y = sb.get_center_of_mass().y;
        println!("{f} {:.6} {:.6} {:.6}", center_y, min_y, com_y);
    }
}

fn scene_soft_soft_vf(dt: Real, frames: usize) {
    use bullet::dynamics::SoftCollision;
    let mut world = make_world();
    add_ground(&mut world);
    let upper = world.add_soft_body(create_patch(
        Vector3::new(-2.0, 6.0, -2.0),
        Vector3::new(2.0, 6.0, -2.0),
        Vector3::new(-2.0, 6.0, 2.0),
        Vector3::new(2.0, 6.0, 2.0),
        12,
        12,
        0,
        true,
        0.0,
    ));
    let lower = world.add_soft_body(create_patch(
        Vector3::new(-2.0, 3.0, -2.0),
        Vector3::new(2.0, 3.0, -2.0),
        Vector3::new(-2.0, 3.0, 2.0),
        Vector3::new(2.0, 3.0, 2.0),
        12,
        12,
        0,
        true,
        0.0,
    ));
    for i in [upper, lower] {
        let sb = world.soft_body_mut(i);
        sb.cfg.piterations = 4;
        sb.cfg.collisions = SoftCollision::SDF_RS.union(SoftCollision::VF_SS);
    }
    for f in 0..frames {
        world.step_simulation(dt);
        let (min_upper, com_upper) = {
            let sb = world.soft_body(upper);
            (
                sb.nodes.iter().map(|n| n.x.y).fold(Real::INFINITY, Real::min),
                sb.get_center_of_mass().y,
            )
        };
        let (min_lower, com_lower) = {
            let sb = world.soft_body(lower);
            (
                sb.nodes.iter().map(|n| n.x.y).fold(Real::INFINITY, Real::min),
                sb.get_center_of_mass().y,
            )
        };
        println!("{f} {:.6} {:.6} {:.6} {:.6}", min_upper, min_lower, com_upper, com_lower);
    }
}
