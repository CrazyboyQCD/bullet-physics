//! Joint-phase soft-body demo (phase E of the classic `btSoftBody` port):
//! the linear/angular joints between clusters and rigid frames.
//!
//! Scene 1: a cloth patch hangs from a box through a linear joint on its
//! master cluster; the box drives back and forth and the cloth follows the
//! swing. Scene 2: an angular joint with the SoftDemo motor control spins a
//! volumetric cluster body up to its goal speed about the joint axis.
//!
//! Run with `cargo run --example softbody_joints`.

use bullet::dynamics::{
    AngularControl, AngularJointSpecs, create_patch, AJoint, PhysicsWorld, RigidBody, Side,
    LinearJointSpecs,
};
use bullet::math::{Real, Vector3};
use bullet::prelude::CollisionShape;

/// The SoftDemo motor control (`MotorControl`): drive the joint speed
/// toward `goal`, clamped by `maxtorque` per solve.
#[derive(Clone, Debug)]
struct MotorControl {
    goal: Real,
    maxtorque: Real,
}

impl AngularControl for MotorControl {
    fn speed(&self, _joint: &AJoint, current: Real) -> Real {
        current + self.maxtorque.min((-self.maxtorque).max(self.goal - current))
    }
    fn clone_box(&self) -> Box<dyn AngularControl> {
        Box::new(self.clone())
    }
}

fn ground() -> RigidBody {
    RigidBody::new_static(
        CollisionShape::plane(Vector3::new(0.0, 1.0, 0.0), 0.0),
        Vector3::ZERO,
    )
}

fn main() {
    let dt: Real = 1.0 / 60.0;

    // Scene 1: cloth dragged through a linear joint on its master cluster.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(ground());
    let cloth = world.add_soft_body(create_patch(
        Vector3::new(-2.0, 4.0, -2.0),
        Vector3::new(2.0, 4.0, -2.0),
        Vector3::new(-2.0, 4.0, 2.0),
        Vector3::new(2.0, 4.0, 2.0),
        7,
        7,
        0,
        true,
        0.0,
    ));
    {
        let sb = world.soft_body_mut(cloth);
        sb.cfg.piterations = 4;
        sb.set_total_mass_from_faces(6.0);
        sb.generate_clusters(8, 32);
    }
    // The drag box rides above the cloth (nothing collides with it); the
    // joint pins the master cluster at the patch's center.
    let drag = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(0.0, 5.6, 0.0),
        8.0,
    ));
    world.add_soft_linear_joint(
        cloth,
        LinearJointSpecs {
            position: Vector3::new(0.0, 4.0, 0.0),
            ..LinearJointSpecs::default()
        },
        Side::Rigid(drag),
    );
    let mut min_cloth_y: Real = 10.0;
    let mut follow_sum: Real = 0.0;
    let mut follow_steps = 0;
    for frame in 0..600 {
        // Drive the box back and forth (a sleeping body ignores velocity
        // writes, so keep waking it — the game_arena lesson).
        world.body_mut(drag).activate();
        let phase = frame as Real / 120.0;
        world.body_mut(drag).linear_velocity = Vector3::new(4.0 * (std::f32::consts::TAU as Real * phase).cos(), 0.0, 0.0);
        world.step_simulation(dt);
        let sb = world.soft_body(cloth);
        let mut com = Vector3::ZERO;
        for n in &sb.nodes {
            com += n.x;
        }
        com /= sb.nodes.len() as Real;
        min_cloth_y = min_cloth_y.min(sb.nodes.iter().map(|n| n.x.y).fold(10.0, Real::min));
        // Once the swing settles in, the cloth tracks the box horizontally.
        if frame > 240 {
            let box_x = world.body(drag).position.x;
            follow_sum += (com.x - box_x).abs();
            follow_steps += 1;
        }
    }
    let follow_err = follow_sum / follow_steps as Real;
    println!(
        "drag: follow err={follow_err:.3} min cloth y={min_cloth_y:.3}"
    );
    assert!(follow_err < 1.2, "cloth did not track the drag box: {follow_err}");
    assert!(min_cloth_y > -0.1, "cloth tunneled through the ground: {min_cloth_y}");
    let sb = world.soft_body(cloth);
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()), "cloth blew up");

    // Scene 2: the motor control spins a volumetric body about the joint
    // axis up to its goal speed.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let anchor = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.1)),
        Vector3::ZERO,
    ));
    let ball = world.add_soft_body(bullet::dynamics::create_ellipsoid(
        Vector3::ZERO,
        Vector3::new(1.0, 0.7, 1.0),
        96,
    ));
    {
        let sb = world.soft_body_mut(ball);
        sb.cfg.piterations = 4;
        sb.set_total_mass_from_faces(20.0);
        sb.generate_bending_constraints(2, None);
        sb.generate_clusters(16, 32);
        sb.materials[0].k_lst = 0.5;
        for c in &mut sb.clusters {
            c.matching = 0.35;
        }
    }
    world.add_soft_angular_joint(
        ball,
        AngularJointSpecs {
            axis: Vector3::Y,
            control: Box::new(MotorControl { goal: 1.5, maxtorque: 0.5 }),
            ..AngularJointSpecs::default()
        },
        Side::Rigid(anchor),
    );
    let mut max_av: Real = 0.0;
    for _ in 0..600 {
        world.step_simulation(dt);
        max_av = max_av.max(world.soft_body(ball).clusters[0].av.y.abs());
    }
    let end_av = world.soft_body(ball).clusters[0].av.y;
    println!("motor: end av y={end_av:.3} max av y={max_av:.3}");
    assert!(
        end_av > 0.8 && end_av < 2.2,
        "motor did not hold the goal speed: {end_av}"
    );
    // The spin keeps the shape: the matching holds the rest frame, so the
    // node spread stays bounded and finite.
    let sb = world.soft_body(ball);
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()), "spinning body blew up");
    let com = sb.get_center_of_mass();
    let spread = sb
        .nodes
        .iter()
        .map(|n| (n.x - com).length())
        .fold(0.0, Real::max);
    assert!(spread < 2.5, "spinning body deformed away: {spread}");
    println!("OK: joints drag the cloth and spin the cluster body.");
}
