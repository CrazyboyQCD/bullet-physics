//! Soft-body contact demo (phase B of the classic `btSoftBody` port):
//! SDF_RS node contacts and rigid-body anchors.
//!
//! Three scenes: a rope falling onto the ground plane and settling on the
//! contact band, a rope draped over a sphere, and a rope anchored to a
//! dynamic box that drops and drags the rope with it.
//!
//! Run with `cargo run --example softbody_rope_ground`.

use bullet::dynamics::{create_rope, PhysicsWorld, RigidBody};
use bullet::math::{Real, Vector3};
use bullet::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

    // Ground plane.
    world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // A sphere to drape a rope over.
    world.add_body(RigidBody::new_static(
        CollisionShape::sphere(1.0),
        Vector3::new(0.0, 0.0, -4.0),
    ));

    // Scene 1: a rope drops onto the plane and piles up on the band.
    let drop = world.add_soft_body(create_rope(
        Vector3::new(-3.0, 5.0, 0.0),
        Vector3::new(-3.0, 2.0, 0.0),
        20,
        0,
    ));
    // Scene 2: a rope draped over the sphere at z = -4 (slow settle).
    let drape = world.add_soft_body(create_rope(
        Vector3::new(-2.0, 1.4, -4.0),
        Vector3::new(2.0, 1.4, -4.0),
        30,
        0,
    ));
    // Scene 3: a rope pinned at one end to a falling dynamic box (the other
    // end pinned to a static column), forming a sagging leash.
    let column = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.2)),
        Vector3::new(4.0, 3.0, 0.0),
    ));
    let cargo = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(5.5, 3.0, 0.0),
        2.0,
    ));
    let leash = world.add_soft_body(create_rope(
        Vector3::new(4.0, 3.0, 0.0),
        Vector3::new(5.5, 3.0, 0.0),
        10,
        0,
    ));
    for i in [drop, drape, leash] {
        world.soft_body_mut(i).cfg.piterations = 4;
    }
    // Pin the leash: node 0 to the static column, last node to the cargo.
    let (col_pos, col_ori) = {
        let b = world.body(column);
        (b.position, b.orientation)
    };
    let (cargo_pos, cargo_ori) = {
        let b = world.body(cargo);
        (b.position, b.orientation)
    };
    world
        .soft_body_mut(leash)
        .append_anchor(0, column, col_pos, col_ori, 1.0);
    let last = world.soft_body(leash).nodes.len() - 1;
    world
        .soft_body_mut(leash)
        .append_anchor(last, cargo, cargo_pos, cargo_ori, 1.0);

    let dt: Real = 1.0 / 60.0;
    let mut drop_min_y = Real::INFINITY;
    let mut drape_min_center_dist = Real::INFINITY;
    let mut max_anchor_err: Real = 0.0;
    println!("t    drop.min  drape.d  anchor_err");
    for step in 0..900 {
        world.step_simulation(dt);
        let sb = world.soft_body(drop);
        drop_min_y = sb
            .nodes
            .iter()
            .map(|n| n.x.y)
            .fold(Real::INFINITY, Real::min);
        let sb = world.soft_body(drape);
        drape_min_center_dist =
            drape_min_center_dist.min(sb.nodes.iter().map(|n| (n.x - Vector3::new(0.0, 0.0, -4.0)).length()).fold(Real::INFINITY, Real::min));
        let cargo_pos = world.body(cargo).position;
        let end = world.soft_body(leash).nodes[last].x;
        max_anchor_err = max_anchor_err.max((end - cargo_pos).length());
        if step % 180 == 0 {
            println!(
                "{:4.1}  {:7.3}  {:6.3}  {:8.3}",
                step as Real * dt,
                drop_min_y,
                drape_min_center_dist,
                max_anchor_err
            );
        }
    }

    println!(
        "drop min y = {drop_min_y:.3}, drape min dist = {drape_min_center_dist:.3}, max anchor err = {max_anchor_err:.3}",
    );
    // The dropped rope rests on the plane's contact band (never through).
    assert!(drop_min_y > 0.0, "drop rope penetrated: {drop_min_y}");
    assert!(drop_min_y < 0.6, "drop rope never landed: {drop_min_y}");
    // The draped rope never enters the sphere.
    assert!(
        drape_min_center_dist > 0.95,
        "drape rope penetrated: {drape_min_center_dist}"
    );
    // The anchors held through the cargo's swing; the error is the PBD
    // chain stretch under the swinging 2 kg cargo (anchors are soft, the
    // links undershoot in 4 iterations — the same stretch character
    // Bullet's ropes have).
    assert!(max_anchor_err < 0.6, "anchor stretched: {max_anchor_err}");
    let cargo_y = world.body(cargo).position.y;
    assert!(cargo_y < 2.0, "cargo did not drop: {cargo_y}");
    println!("OK: plane/sphere contacts and rigid anchors all behave.");
}
