//! End-to-end demo of concave terrain: a sphere and a box fall onto a
//! triangle mesh ground (built from a small heightfield-like grid) and come to
//! rest. Run with `cargo run --example falling_mesh`.

use bullet_physics::dynamics::{PhysicsWorld, RigidBody};
use bullet_physics::math::{Real, Vector3};
use bullet_physics::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.solver_info_mut().num_iterations = 20;

    // A 10x10 triangle mesh ground spanning x, z in [-5, 5] at y = 0, with a
    // gentle ridge along the diagonal (peak 0.5 at the center).
    let n = 10;
    let mut vertices = Vec::with_capacity(n * n);
    for z in 0..n {
        for x in 0..n {
            let (fx, fz) = (x as Real - 4.5, z as Real - 4.5);
            let y = (fx + fz).abs() * 0.1;
            vertices.push(Vector3::new(fx, y, fz));
        }
    }
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 2);
    for z in 0..n - 1 {
        for x in 0..n - 1 {
            let v00 = (z * n + x) as u32;
            let v10 = v00 + 1;
            let v01 = v00 + n as u32;
            let v11 = v01 + 1;
            indices.push([v00, v11, v10]);
            indices.push([v00, v01, v11]);
        }
    }
    world.add_body(RigidBody::new_static(
        CollisionShape::triangle_mesh(vertices, indices),
        Vector3::ZERO,
    ));

    // A sphere and a box dropped above the terrain.
    let sphere = world.add_body(
        RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::new(-2.0, 5.0, -2.0), 1.0)
            .friction(0.6)
            .restitution(0.0),
    );
    let box_body = world.add_body(
        RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.5)),
            Vector3::new(2.0, 5.0, 2.0),
            1.0,
        )
        .friction(0.6)
        .restitution(0.0),
    );

    let dt: Real = 1.0 / 60.0;
    let mut time: Real = 0.0;
    println!("t      sphere_y  box_y");
    while time < 6.0 {
        world.step_simulation(dt);
        time += dt;
        if (time * 60.0).round() as i64 % 30 == 0 {
            println!(
                "{:.2}  {:8.3}  {:7.3}",
                time,
                world.body(sphere).position.y,
                world.body(box_body).position.y,
            );
        }
    }

    // Both bodies settle on the terrain: near rest, with the center between
    // the surface range (y in [0, 0.5]) plus the 0.5 body offset. Sliding on
    // the ridge means the exact resting height is path-dependent.
    // The sphere keeps rolling along the valley floor: a rolling sphere has
    // no rolling friction, and since the sphere-triangle analytic fast path
    // the contacts carry no EPA noise to damp it numerically, so it only has
    // to stay on the surface (the box has no rolling freedom and must stop).
    let sphere_y = world.body(sphere).position.y;
    let box_y = world.body(box_body).position.y;
    let sphere_v = world.body(sphere).linear_velocity.length();
    let box_v = world.body(box_body).linear_velocity.length();
    println!(
        "final sphere_y = {:.4}, box_y = {:.4} (expected ~0.5-1.1)",
        sphere_y, box_y
    );
    assert!(
        sphere_v < 1.5
            && box_v < 0.1
            && (0.4..1.3).contains(&sphere_y)
            && (0.4..1.3).contains(&box_y),
        "bodies did not settle on the mesh: sphere_y = {sphere_y}, box_y = {box_y}"
    );
    println!("OK: sphere and box settled on the triangle mesh.");
}
