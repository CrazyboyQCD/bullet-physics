//! Hinge-2 / universal probe twin: the two thin joint specializations run
//! through the Rust solver for numeric comparison against the real Bullet
//! constraint solver (tools/hinge2_universal_probe.cpp, built by
//! tools/build_hinge2_universal_probe.py). The joints are configurations of
//! the already-probed spring-2 / classic 6DOF row shapes; what this pins is
//! the constructor-derived configuration (frames from world axes, limit
//! tables, suspension constants, the construction-pose equilibrium) feeding
//! those rows.
//!
//! Output protocol (identical to the C++ probe, one line per frame): label,
//! frame, position(3), quaternion(x y z w) of the moving body. Scenes:
//! H2SUS (suspension catch, 240 frames), H2STE (steering spin absorbed by the
//! linear locks, 120), UNIV (locked-axis spin corrected, 60), GEAR (rigid
//! gear coupling, 120, both bodies printed as GEARA/GEARB). Strip CR, then
//! compare with `python tools/mbdiff.py`.
//!
//! Run:
//!     cargo run --profile release-debug -p bullet-physics --example hinge2_universal_probe
//!     cargo run --profile release-debug -p bullet-physics --example hinge2_universal_probe --features f64

use bullet_physics::dynamics::{Constraint, PhysicsWorld, RigidBody};
use bullet_physics::math::{Real, Vector3};
use bullet_physics::shapes::CollisionShape;

const DT: Real = 1.0 / 60.0;
const FRAMES_SUS: usize = 240;
const FRAMES_STE: usize = 120;
const FRAMES_UNIV: usize = 60;
const FRAMES_GEAR: usize = 120;

fn print_body(label: &str, frame: usize, body: &RigidBody) {
    let p = body.position;
    let q = body.orientation;
    println!(
        "{} {} {} {} {} {} {} {} {}",
        label, frame, p.x, p.y, p.z, q.x, q.y, q.z, q.w
    );
}

fn main() {
    // Scene 0: hinge-2 suspension.
    {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, 0.0, -9.8));
        let chassis = world.add_body(RigidBody::new_static(
            CollisionShape::sphere(0.5),
            Vector3::ZERO,
        ));
        let wheel = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.2),
            Vector3::new(0.0, 0.0, -0.3),
            1.0,
        ));
        world.add_hinge2(chassis, wheel, Vector3::ZERO, Vector3::Z, Vector3::X);
        print_body("H2SUS", 0, world.body(wheel));
        for f in 1..=FRAMES_SUS {
            world.step_simulation(DT);
            print_body("H2SUS", f, world.body(wheel));
        }
    }
    // Scene 1: hinge-2 steering limit.
    {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let chassis = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::ZERO,
            10.0,
        ));
        let wheel = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.15),
            Vector3::new(0.9, 0.0, 0.0),
            1.0,
        ));
        world.add_hinge2(chassis, wheel, Vector3::ZERO, Vector3::Z, Vector3::X);
        world.body_mut(wheel).angular_velocity = Vector3::new(0.0, 0.0, 6.0);
        print_body("H2STE", 0, world.body(wheel));
        for f in 1..=FRAMES_STE {
            world.step_simulation(DT);
            print_body("H2STE", f, world.body(wheel));
        }
    }
    // Scene 2: universal joint, locked-axis spin.
    {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(RigidBody::new_static(
            CollisionShape::sphere(0.3),
            Vector3::ZERO,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.3),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        ));
        world.add_universal(a, b, Vector3::new(1.0, 0.0, 0.0), Vector3::Z, Vector3::X);
        world.body_mut(b).angular_velocity = Vector3::new(0.0, 5.0, 0.0);
        print_body("UNIV", 0, world.body(b));
        for f in 1..=FRAMES_UNIV {
            world.step_simulation(DT);
            print_body("UNIV", f, world.body(b));
        }
    }
    // Scene 3: rigid gear — two identical flywheels, A spun up, ratio 2.
    // The one-row solve settles at ω_A = 4.8 / ω_B = −2.4 (equal inertias):
    // parallel axes spinning opposite. Both bodies' poses are printed
    // (GEARA / GEARB) — the spin shows in the quaternions.
    {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::ZERO,
            2.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(3.0, 0.0, 0.0),
            2.0,
        ));
        world.add_constraint(Constraint::gear(a, b, Vector3::Z, Vector3::Z, 2.0));
        world.body_mut(a).angular_velocity = Vector3::new(0.0, 0.0, 6.0);
        print_body("GEARA", 0, world.body(a));
        print_body("GEARB", 0, world.body(b));
        for f in 1..=FRAMES_GEAR {
            world.step_simulation(DT);
            print_body("GEARA", f, world.body(a));
            print_body("GEARB", f, world.body(b));
        }
    }
}
