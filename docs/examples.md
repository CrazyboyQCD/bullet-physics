# Examples

Runnable demos live in [`crates/bullet-physics/examples/`](../crates/bullet-physics/examples/):
`cargo run -p bullet-physics --example <name>`. The snippets below are the
short forms of the same APIs; every signature is checked against the sources.

## Rigid bodies

A static ground and a dynamic box, stepped at 60 Hz (`falling_box` demo):

```rust
use bullet_physics::dynamics::{PhysicsWorld, RigidBody};
use bullet_physics::math::Vector3;
use bullet_physics::shapes::CollisionShape;

let mut world = PhysicsWorld::new();
world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

world.add_body(RigidBody::new_static(
    CollisionShape::box_shape(Vector3::new(25.0, 0.5, 25.0)),
    Vector3::new(0.0, -0.5, 0.0),
));
world.add_body(
    RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::new(0.0, 5.0, 0.0),
        1.0,
    )
    .friction(0.6)
    .restitution(0.0),
);

for _ in 0..300 {
    world.step_simulation(1.0 / 60.0);
}
```

## Kinematic platform

Infinite mass, pose driven by the velocities you set — it pushes dynamic
bodies and is never pushed (`ccd`, `mixed_stress` demos use this to shove
scenes around):

```rust
let platform = world.add_body(RigidBody::new_kinematic(
    CollisionShape::box_shape(Vector3::new(2.0, 0.25, 2.0)),
    Vector3::new(0.0, 1.0, 0.0),
));
world.body_mut(platform).linear_velocity = Vector3::new(1.5, 0.0, 0.0);
```

## Constraints (joints)

Six joint families — point2point, hinge, fixed, cone-twist, slider, generic
6DOF (+ spring-2) — added with `world.add_constraint` and solved by the
normal `step_simulation` (`joints` demo):

```rust
use bullet_physics::dynamics::{Constraint, PhysicsWorld, RigidBody};
use bullet_physics::math::Vector3;
use bullet_physics::shapes::CollisionShape;

let anchor = world.add_body(RigidBody::new_static(
    CollisionShape::box_shape(Vector3::splat(0.2)),
    Vector3::new(0.0, 2.0, 0.0),
));
let door = world.add_body(RigidBody::new_dynamic(
    CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
    Vector3::new(0.5, 2.0, 0.0),
    2.0,
));

let mut hinge = Constraint::hinge(
    door,
    anchor,
    Vector3::new(-0.5, 0.0, 0.0), // pivot in the door's frame
    Vector3::Z,                   // hinge axis in the door's frame
    Vector3::ZERO,                // pivot in the anchor's frame
    Vector3::Z,                   // hinge axis in the anchor's frame
);
hinge.hinge_mut().set_limit(-0.7, 0.7);
world.add_constraint(hinge);
```

## Featherstone multibody

Articulated bodies (`btMultiBody` semantics) step through the same world;
multibody constraints (motor, gear, point2point, limits) take a snapshot of
the multibody list at creation (`multibody` demo):

```rust
use bullet_physics::dynamics::{MultiBody, MultiBodyConstraint, PhysicsWorld};
use bullet_physics::math::{Quaternion, Vector3};

let mut world = PhysicsWorld::new();
let mut pendulum = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
pendulum.setup_revolute(
    0, 1.0, Vector3::ONE, None, Quaternion::IDENTITY,
    Vector3::Z,                    // joint axis
    Vector3::ZERO,                 // parent-frame offset
    Vector3::new(0.0, -1.0, 0.0),  // link center of mass
);
pendulum.finalize_multidof();
let handle = world.add_multibody(pendulum);

// Drive a joint at 2 rad/s (max impulse 100):
let snapshot = world.multibodies().to_vec();
let motor = MultiBodyConstraint::joint_motor(handle, 0, 0, 2.0, 100.0, &snapshot);
world.add_multibody_constraint(motor);
```

## Soft body rope

Position-based ropes, pins selected by flags (`softbody_rope` demo; cloth,
clusters and cutting have their own demos):

```rust
use bullet_physics::dynamics::{create_rope, PhysicsWorld, ROPE_FIX_START, ROPE_FIX_END};

let rope = world.add_soft_body(create_rope(
    Vector3::new(-2.0, 5.0, -3.0),
    Vector3::new(2.0, 5.0, -3.0),
    30,                            // node count
    ROPE_FIX_START | ROPE_FIX_END, // pin both ends -> catenary
));
world.soft_body_mut(rope).cfg.piterations = 4;
```

## Raycast vehicle

A chassis carried by suspension raycasts; wheels are not bodies. Call
`update_vehicle` once per frame *after* `step_simulation` (`vehicle` demo):

```rust
use bullet_physics::dynamics::{PhysicsWorld, RaycastVehicle, RigidBody, VehicleTuning};
use bullet_physics::math::Vector3;
use bullet_physics::shapes::CollisionShape;

let chassis = world.add_body(
    RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::new(1.0, 0.3, 0.6)),
        Vector3::new(0.0, 1.0, 0.0),
        800.0,
    )
    .disable_deactivation(),
);
let mut vehicle = RaycastVehicle::new(chassis);
vehicle.set_coordinate_system(0, 1, 2); // right = X, up = Y, forward = Z
let tuning = VehicleTuning::default();
vehicle.add_wheel(
    &world,
    Vector3::new(0.8, 0.4, 0.9),      // connection point (chassis frame)
    Vector3::new(0.0, -1.0, 0.0),     // suspension direction
    Vector3::new(-1.0, 0.0, 0.0),     // axle
    1.1, 0.35, &tuning, true,        // suspension rest length, wheel radius, front
);
```

## Character controller

A capsule driven kinematically; `update` runs once per frame after
`step_simulation` (`character` demo):

```rust
use bullet_physics::dynamics::{KinematicCharacterController, PhysicsWorld, RigidBody};
use bullet_physics::math::Vector3;
use bullet_physics::shapes::CollisionShape;

let shape = CollisionShape::capsule(0.4, 0.8);
let body = world.add_body(RigidBody::new_static(
    shape.clone(),
    Vector3::new(0.0, 1.2, 0.0),
));
let mut controller = KinematicCharacterController::new(body, shape, 0.5, Vector3::Y);
controller.set_walk_direction(Vector3::new(0.02, 0.0, 0.0)); // per-step, 1.2 m/s @ 60 Hz
```

## Heightfield terrain

A centered height grid with the upstream `btHeightfieldTerrainShape` option
surface (`heightfield` parity scenes):

```rust
use bullet_physics::collision::shapes::{Heightfield, HeightfieldOptions, UpAxis};
use bullet_physics::math::Vector3;

let heights: Vec<f32> = (0..13 * 13).map(|i| (i % 7) as f32 * 0.1).collect();
let shape = CollisionShape::Heightfield(Heightfield::with_options(
    HeightfieldOptions {
        up_axis: UpAxis::Y,
        flip_quad_edges: false,
        use_diamond_subdivision: false,
        use_zigzag_subdivision: false,
        flip_triangle_winding: false,
    },
    heights,
    13, 13,                        // width, depth
    Vector3::new(0.5, 1.0, 0.5),   // cell size
));
```

## Inverse dynamics (optional feature)

Tree-structured multibody inverse dynamics — a one-shot solver: given
`(q, u, dot_u)` it returns the joint forces that realize them, plus the mass
matrix and jacobians. Enable the default-off `inverse-dynamics` feature
(`inverse_dynamics` demo):

```toml
[dependencies]
bullet-physics = { version = "0.2", features = ["inverse-dynamics"] }
```

```rust
use bullet_physics::inverse_dynamics::builder::{JointType, TreeBuilder};
use bullet_physics::inverse_dynamics::{Matrix3, Vector3};

fn inertia(x: f32, y: f32, z: f32) -> Matrix3 {
    Matrix3::from_cols(
        Vector3::new(x, 0.0, 0.0),
        Vector3::new(0.0, y, 0.0),
        Vector3::new(0.0, 0.0, z),
    )
}

let mut b = TreeBuilder::new();
b.add_body(
    0, -1, JointType::Fixed, Vector3::ZERO, Matrix3::IDENTITY,
    Vector3::ZERO, 0.0, Vector3::ZERO, inertia(1.0, 1.0, 1.0),
).unwrap();
b.add_body(
    1, 0, JointType::Revolute, Vector3::new(1.0, 0.0, 0.0), Matrix3::IDENTITY,
    Vector3::Z, 1.0, Vector3::new(0.5, 0.0, 0.0), inertia(0.2, 0.15, 0.1),
).unwrap();
let mut tree = b.finalize().unwrap();
tree.set_gravity_in_world_frame(Vector3::new(0.0, -9.8, 0.0));

let mut tau = [0.0; 1];
tree.calculate_inverse_dynamics(&[0.0], &[0.0], &[0.0], &mut tau).unwrap();
let mut mass = [0.0; 1];
tree.calculate_mass_matrix(&[0.0], &mut mass).unwrap();
```
