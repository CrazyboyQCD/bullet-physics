//! Conetwist stress probe: a chain of links hanging along -X (gravity -X)
//! with swing limits at 30 degrees, pressed past the limit by a per-step
//! impulse on the tip (optionally slammed first with a lateral velocity
//! kick), so the joints ride their swing limits. Prints the tip state,
//! per-joint relative swing angles, and per-body angular speeds.
//! Twin of `tools/conetwist_stress_probe.cpp` — same scene, same output
//! columns, diffable with `tools/mbdiff.py`.
//!
//! Run: cargo run --example conetwist_stress_probe -- [frames] [links]
//!      [softness] [relaxation] [mode]
//! mode: 0 = free tip body (no joint), 1 = jointed + kick, 2 = jointed only.

use bullet_physics::dynamics::{BodyHandle, Constraint, JointFrame, PhysicsWorld, RigidBody};
use bullet_physics::math::{Matrix3, Real, Vector3};
use bullet_physics::shapes::CollisionShape;

const SWING: Real = bullet_physics::math::PI / 6.0;
const TWIST: Real = bullet_physics::math::PI / 3.0;

fn swing_angle(prev: Matrix3, cur: Matrix3) -> Real {
    let a = prev.x_axis;
    let b = cur.x_axis;
    let d = a.dot(b).clamp(-1.0, 1.0);
    d.acos()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frames: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let links_n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let softness: Real = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let relaxation: Real = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let mode: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(-9.81, 0.0, 0.0));

    let anchor = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.1)),
        Vector3::ZERO,
    ));

    let mut links: Vec<Option<BodyHandle>> = Vec::with_capacity(links_n);
    for i in 0..links_n {
        let mut body = RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.15, 0.1, 0.1)),
            Vector3::new(-0.25 - 0.3 * i as Real, 0.0, 0.0),
            1.0,
        )
        .disable_deactivation();
        if mode != 2 {
            body.linear_velocity = Vector3::new(0.0, 2.0, 0.0);
        }
        let h = world.add_body(body);
        world.body_mut(h).set_damping(0.0, 0.0);
        links.push(Some(h));
    }
    let links: Vec<BodyHandle> = links.into_iter().map(|l| l.unwrap()).collect();

    if mode != 0 {
        let mut frame_a = JointFrame::new(Vector3::new(-0.1, 0.0, 0.0), Matrix3::IDENTITY);
        let frame_b = JointFrame::new(Vector3::new(0.15, 0.0, 0.0), Matrix3::IDENTITY);
        let joint = world.add_constraint(Constraint::cone_twist(
            anchor, links[0], frame_a, frame_b, SWING, SWING, TWIST,
        ));
        world
            .constraint_mut(joint)
            .cone_twist_mut()
            .set_limit(SWING, SWING, TWIST, softness, 0.3, relaxation);
        for pair in links.windows(2) {
            frame_a.origin = Vector3::new(-0.15, 0.0, 0.0);
            let joint = world.add_constraint(Constraint::cone_twist(
                pair[0], pair[1], frame_a, frame_b, SWING, SWING, TWIST,
            ));
            world
                .constraint_mut(joint)
                .cone_twist_mut()
                .set_limit(SWING, SWING, TWIST, softness, 0.3, relaxation);
        }
    }

    let dt = 1.0 / 60.0;
    for f in 0..frames {
        world
            .body_mut(links[links_n - 1])
            .apply_impulse(Vector3::new(0.0, 25.0 * dt, 0.0), Vector3::ZERO);
        world.step_simulation(dt);
        print!("{f}");
        let tip = world.body(links[links_n - 1]);
        let p = tip.position;
        let v = tip.linear_velocity;
        print!(" {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}", p.x, p.y, p.z, v.x, v.y, v.z);
        let mut prev = Matrix3::IDENTITY;
        for h in &links {
            let cur = Matrix3::from_quat(world.body(*h).orientation);
            print!(" {:.6}", swing_angle(prev, cur));
            prev = cur;
        }
        for h in &links {
            let w = world.body(*h).angular_velocity;
            print!(" {:.6}", w.length());
        }
        println!();
    }
}
