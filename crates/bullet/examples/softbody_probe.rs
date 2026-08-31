//! SoftBody probe — the Rust counterpart of `tools/softbody_probe.cpp`.
//! Runs the same rope/cloth scenes with the same parameters and prints
//! per-frame trajectories so the two implementations can be diffed
//! numerically.
//!
//! Usage: `cargo run -p bullet --example softbody_probe -- <scene> [frames]`
//! with scene in {rope_pendulum, rope_ground, cloth_plane}.

use bullet::dynamics::{create_rope, PhysicsWorld, RigidBody, SoftBody};
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

/// Minimal `btSoftBodyHelpers::CreatePatch` node/link builder for the probe
/// (the full helper — faces, edge flags, perturbation — arrives with the
/// cloth phase). Faces are skipped: before that phase they carry no position
/// dynamics (aero/pressure/volume forces are disabled by default and soft-soft
/// contact is off), so the trajectories match the Bullet twin, which builds
/// them.
fn build_patch(
    corner00: Vector3,
    corner10: Vector3,
    corner01: Vector3,
    corner11: Vector3,
    resx: usize,
    resy: usize,
    fixeds: u32,
) -> SoftBody {
    let idx = |x: usize, y: usize| y * resx + x;
    let mut positions = Vec::with_capacity(resx * resy);
    let masses = vec![1.0 as Real; resx * resy];
    for iy in 0..resy {
        let ty = iy as Real / (resy - 1) as Real;
        let py0 = corner00.lerp(corner01, ty);
        let py1 = corner10.lerp(corner11, ty);
        for ix in 0..resx {
            let tx = ix as Real / (resx - 1) as Real;
            positions.push(py0.lerp(py1, tx));
        }
    }
    let mut psb = SoftBody::new(&positions, &masses);
    if fixeds & 1 != 0 {
        psb.set_mass(idx(0, 0), 0.0);
    }
    if fixeds & 2 != 0 {
        psb.set_mass(idx(resx - 1, 0), 0.0);
    }
    if fixeds & 4 != 0 {
        psb.set_mass(idx(0, resy - 1), 0.0);
    }
    if fixeds & 8 != 0 {
        psb.set_mass(idx(resx - 1, resy - 1), 0.0);
    }
    for iy in 0..resy {
        for ix in 0..resx {
            let mdx = ix + 1 < resx;
            let mdy = iy + 1 < resy;
            // Bullet's naming: node01 = x+1, node10 = y+1. Same append order
            // as CreatePatch (PBD is order-sensitive): horizontal, vertical,
            // then the two faces (skipped here), then the diagonal link.
            let (n00, n01, n10, n11) =
                (idx(ix, iy), idx(ix + 1, iy), idx(ix, iy + 1), idx(ix + 1, iy + 1));
            if mdx {
                psb.append_link(n00, n01, None, false);
            }
            if mdy {
                psb.append_link(n00, n10, None, false);
            }
            if mdx && mdy {
                psb.append_link(n00, n11, None, false);
            }
        }
    }
    psb
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
