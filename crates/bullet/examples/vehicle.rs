//! Raycast vehicle demo: a 4-wheel car accelerates, steers and brakes on a
//! flat plane (Bullet's `btRaycastVehicle` pattern). The chassis is an
//! ordinary dynamic box carried by suspension raycasts; the wheels themselves
//! are not bodies.
//!
//! Call `vehicle.update_vehicle(&mut world, dt)` once per frame *after*
//! `world.step_simulation(dt)` (Bullet's actions run after
//! `integrateTransforms`). The chassis keeps `disable_deactivation()` like
//! Bullet's demos, so it never falls asleep while parked.
//!
//! Run with `cargo run --example vehicle`.

use bullet::dynamics::{PhysicsWorld, RaycastVehicle, RigidBody, VehicleTuning};
use bullet::math::{Real, Vector3};
use bullet::shapes::CollisionShape;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

    // Ground plane at y = 0.
    world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));

    // Chassis: a box that stays above the ground, carried by the suspension.
    // The suspension towers attach *above* the centre of mass (like Bullet's
    // demo, `connectionHeight = 1.2`): with the attachment below the COM an
    // accelerating car flips backwards.
    let chassis = world.add_body(
        RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(1.0, 0.3, 0.6)),
            Vector3::new(0.0, 1.0, 0.0),
            800.0,
        )
        .disable_deactivation(),
    );

    let mut vehicle = RaycastVehicle::new(chassis);
    // Right = X, up = Y, forward = Z (Bullet's default is the legacy Z-up
    // axes right = X, up = Z, forward = Y).
    vehicle.set_coordinate_system(0, 1, 2);
    let tuning = VehicleTuning::default();
    // Connection points at the four corners, 0.4 m above the chassis centre.
    let conn = |x: Real, z: Real| Vector3::new(x, 0.4, z);
    let wheel_dir = Vector3::new(0.0, -1.0, 0.0);
    let wheel_axle = Vector3::new(-1.0, 0.0, 0.0);
    // Front wheels (steer), then rear wheels (drive).
    vehicle.add_wheel(&world, conn(0.8, 0.9), wheel_dir, wheel_axle, 1.1, 0.35, &tuning, true);
    vehicle.add_wheel(&world, conn(-0.8, 0.9), wheel_dir, wheel_axle, 1.1, 0.35, &tuning, true);
    vehicle.add_wheel(&world, conn(-0.8, -0.9), wheel_dir, wheel_axle, 1.1, 0.35, &tuning, false);
    vehicle.add_wheel(&world, conn(0.8, -0.9), wheel_dir, wheel_axle, 1.1, 0.35, &tuning, false);

    let dt: Real = 1.0 / 60.0;
    let mut time = 0.0;
    let mut speed_at_brake: Real = 0.0;
    let mut peak_kmh: Real = 0.0;
    println!("t     z      speed_kmh  phase");
    while time < 5.5 {
        // Drive: engine on the rear wheels.
        vehicle.apply_engine_force(600.0, 2);
        vehicle.apply_engine_force(600.0, 3);
        if (2.0..3.5).contains(&time) {
            // Steer the front wheels mid-run.
            vehicle.set_steering_value(0.25, 0);
            vehicle.set_steering_value(0.25, 1);
        }
        if time >= 3.5 {
            // Coast and brake.
            vehicle.apply_engine_force(0.0, 2);
            vehicle.apply_engine_force(0.0, 3);
            vehicle.set_brake(80.0, 0);
            vehicle.set_brake(80.0, 1);
            vehicle.set_brake(80.0, 2);
            vehicle.set_brake(80.0, 3);
            if speed_at_brake == 0.0 {
                speed_at_brake = vehicle.current_speed_km_hour().abs();
            }
        }
        world.step_simulation(dt);
        vehicle.update_vehicle(&mut world, dt);
        time += dt;
        peak_kmh = peak_kmh.max(vehicle.current_speed_km_hour().abs());
        if (time * 10.0).round() as i64 % 10 == 0 {
            println!(
                "{:.2}  {:.3}  {:8.2}  {}",
                time,
                world.body(chassis).position.z,
                vehicle.current_speed_km_hour(),
                if time < 2.0 {
                    "accelerate"
                } else if time < 3.5 {
                    "steer"
                } else {
                    "brake"
                }
            );
        }
    }

    // Behaviors verified: the car accelerated forward (z moved), turned while
    // steering (heading deviated from +Z), stayed upright, and braked (speed
    // dropped).
    let pos = world.body(chassis).position;
    let fwd = vehicle.forward_vector(&world);
    let final_kmh = vehicle.current_speed_km_hour().abs();
    println!(
        "final pos = ({:.2}, {:.2}, {:.2}), fwd = ({:.2}, {:.2}, {:.2}), peak = {:.1} km/h, final = {:.1} km/h",
        pos.x, pos.y, pos.z, fwd.x, fwd.y, fwd.z, peak_kmh, final_kmh
    );
    assert!(pos.z > 3.0, "car did not drive forward: z = {}", pos.z);
    assert!(peak_kmh > 10.0, "car too slow: peak {peak_kmh} km/h");
    assert!(
        fwd.dot(Vector3::Z) < 0.99,
        "car did not turn while steering: fwd = {:?}",
        fwd
    );
    assert!(fwd.y > -0.3, "car flipped over: fwd = {:?}", fwd);
    assert!(
        final_kmh < speed_at_brake,
        "braking did not slow the car: {speed_at_brake} -> {final_kmh} km/h"
    );
    println!("OK: car accelerated, steered and braked.");
}
