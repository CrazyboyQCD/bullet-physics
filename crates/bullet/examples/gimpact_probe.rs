//! GImpact probe: dynamic triangle-mesh scenes printed per-frame for
//! numerical comparison against the C++ `tools/gimpact_probe.cpp` (real
//! Bullet with `btGImpactMeshShape`).
//!
//! Usage: `cargo run -p bullet --release --example gimpact_probe -- <scene> [frames]`
//! Scenes: `inertia` | `shell_on_plane` | `shell_on_bvh` | `sphere_on_shell`

use bullet::prelude::*;

const DT: Real = 1.0 / 60.0;

/// A closed 2x2x0.4 box shell (top and bottom faces), matching the C++
/// `makeShellShape`.
fn shell_shape() -> CollisionShape {
    let (hx, hy, hz) = (1.0 as Real, 0.2 as Real, 1.0 as Real);
    CollisionShape::triangle_mesh(
        vec![
            // Top face.
            Vector3::new(-hx, hy, -hz),
            Vector3::new(hx, hy, -hz),
            Vector3::new(hx, hy, hz),
            Vector3::new(-hx, hy, hz),
            // Bottom face.
            Vector3::new(-hx, -hy, -hz),
            Vector3::new(hx, -hy, -hz),
            Vector3::new(hx, -hy, hz),
            Vector3::new(-hx, -hy, hz),
        ],
        vec![[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]],
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scene = args.get(1).map(String::as_str).unwrap_or("shell_on_plane");
    let frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    match scene {
        "inertia" => {
            let shape = shell_shape();
            let i = shape.local_inertia(2.0);
            println!("inertia {:.6} {:.6} {:.6}", i.x, i.y, i.z);
        }
        "shell_on_plane" => {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
            world.add_body(RigidBody::new_static(
                CollisionShape::plane(Vector3::Y, 0.0),
                Vector3::ZERO,
            ));
            let body = world.add_body(RigidBody::new_dynamic(
                shell_shape(),
                Vector3::new(0.0, 0.5, 0.0),
                2.0,
            ));
            for f in 0..frames {
                world.step_simulation(DT);
                let b = world.body(body);
                println!(
                    "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    f,
                    b.position.x,
                    b.position.y,
                    b.position.z,
                    b.linear_velocity.x,
                    b.linear_velocity.y,
                    b.linear_velocity.z
                );
            }
        }
        "shell_on_bvh" => {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
            let ground = CollisionShape::triangle_mesh(
                (0..8)
                    .flat_map(|z| (0..8).map(move |x| (x, z)))
                    .flat_map(|(x, z)| {
                        let (x, z) = (x as Real, z as Real);
                        vec![
                            Vector3::new(x, 0.0, z),
                            Vector3::new(x + 1.0, 0.0, z),
                            Vector3::new(x + 1.0, 0.0, z + 1.0),
                            Vector3::new(x, 0.0, z + 1.0),
                        ]
                    })
                    .collect::<Vec<_>>(),
                (0..64)
                    .flat_map(|i| {
                        let b = (i * 4) as u32;
                        [[b, b + 2, b + 1], [b, b + 3, b + 2]]
                    })
                    .collect::<Vec<_>>(),
            );
            world.add_body(RigidBody::new_static(
                ground,
                Vector3::new(-4.0, 0.0, -4.0),
            ));
            let body = world.add_body(RigidBody::new_dynamic(
                shell_shape(),
                Vector3::new(0.0, 0.5, 0.0),
                2.0,
            ));
            for f in 0..frames {
                world.step_simulation(DT);
                let b = world.body(body);
                println!(
                    "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    f,
                    b.position.x,
                    b.position.y,
                    b.position.z,
                    b.linear_velocity.x,
                    b.linear_velocity.y,
                    b.linear_velocity.z
                );
            }
        }
        "sphere_on_shell" => {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
            world.add_body(RigidBody::new_static(
                shell_shape(),
                Vector3::new(0.0, 0.2, 0.0),
            ));
            let ball = world.add_body(RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(0.0, 2.0, 0.0),
                1.0,
            ));
            for f in 0..frames {
                world.step_simulation(DT);
                let b = world.body(ball);
                println!(
                    "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    f,
                    b.position.x,
                    b.position.y,
                    b.position.z,
                    b.linear_velocity.x,
                    b.linear_velocity.y,
                    b.linear_velocity.z
                );
            }
        }
        other => println!("unknown scene {other}"),
    }
}
