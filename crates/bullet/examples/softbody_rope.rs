//! Soft-body ropes demo (phase A of the classic `btSoftBody` port):
//! node-link position-based dynamics without contacts yet.
//!
//! Three ropes: a pendulum released horizontally, a rope pinned at both
//! ends sagging into a catenary, and a free-falling rope (rigid-body
//! fall, no stretch).
//!
//! Run with `cargo run --example softbody_rope`.

use bullet::dynamics::{create_rope, PhysicsWorld, ROPE_FIX_END, ROPE_FIX_START};
use bullet::math::{Real, Vector3};

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

    // Pendulum: pinned at the left end, released horizontally.
    let pendulum = world.add_soft_body(create_rope(
        Vector3::new(0.0, 6.0, 0.0),
        Vector3::new(6.0, 6.0, 0.0),
        16,
        ROPE_FIX_START,
    ));
    // Catenary: pinned at both ends, sagging under gravity.
    let catenary = world.add_soft_body(create_rope(
        Vector3::new(-2.0, 5.0, -3.0),
        Vector3::new(2.0, 5.0, -3.0),
        30,
        ROPE_FIX_START | ROPE_FIX_END,
    ));
    // Free fall: no pins, the rope drops like a rigid line.
    let free = world.add_soft_body(create_rope(
        Vector3::new(0.0, 8.0, 3.0),
        Vector3::new(4.0, 8.0, 3.0),
        16,
        0,
    ));
    for i in [pendulum, catenary, free] {
        world.soft_body_mut(i).cfg.piterations = 4;
    }

    let dt: Real = 1.0 / 60.0;
    let mut min_pendulum_mid_y = Real::INFINITY;
    let mut pendulum_crossed = false;
    let mut max_stretch: Real = 0.0;
    let mut catenary_sag: Real = 0.0;
    let free_com_start = world.soft_body(free).get_center_of_mass().y;
    println!("t     pend.mid  catenary  free.com.y");
    for step in 0..600 {
        world.step_simulation(dt);
        let pend = world.soft_body(pendulum);
        let mid = pend.nodes[9].x;
        min_pendulum_mid_y = min_pendulum_mid_y.min(mid.y);
        if pend.nodes[17].x.x < -1.0 {
            pendulum_crossed = true;
        }
        catenary_sag = 5.0 - world.soft_body(catenary).nodes[16].x.y;
        for rope in [pendulum, catenary, free] {
            for l in &world.soft_body(rope).links {
                let nodes = &world.soft_body(rope).nodes;
                let len = (nodes[l.n[0]].x - nodes[l.n[1]].x).length();
                max_stretch = max_stretch.max(len - l.rl);
            }
        }
        if step % 120 == 0 {
            let free_y = world.soft_body(free).get_center_of_mass().y;
            println!(
                "{:5.2}  {:7.3}  {:8.3}  {:7.3}",
                step as Real * dt,
                mid.y,
                catenary_sag,
                free_y
            );
        }
    }

    // The pendulum swung through the bottom to the far side; the catenary
    // sagged a plausible amount; the free rope fell ~10 s without any link
    // stretching more than ~16% of its rest length.
    let free_fall = free_com_start - world.soft_body(free).get_center_of_mass().y;
    println!(
        "pendulum min mid y = {:.3}, crossed = {}, catenary sag = {:.3}, free fall = {:.2}, max stretch = {:.4}",
        min_pendulum_mid_y, pendulum_crossed, catenary_sag, free_fall, max_stretch
    );
    assert!(min_pendulum_mid_y < 5.0, "pendulum never swung down");
    assert!(pendulum_crossed, "pendulum never crossed the bottom");
    assert!(catenary_sag > 0.05 && catenary_sag < 2.0, "bad catenary sag: {catenary_sag}");
    // 10 s of free fall = 490 m, clamped by nothing; expect hundreds of m.
    assert!(free_fall > 100.0, "free rope did not fall: {free_fall}");
    assert!(max_stretch < 0.06, "links stretched too far: {max_stretch}");
    println!("OK: pendulum, catenary and free-fall ropes all behave.");
}
