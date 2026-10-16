//! Volume-phase soft-body demo (phase D of the classic `btSoftBody` port):
//! tetra-driven mass, K-mean clusters and the `CL_RS`/`CL_SS` contact
//! joints.
//!
//! Scene 1: a volumetric ellipsoid falls onto the ground plane through its
//! cluster colliders, squashes and recovers (the cluster matching plus the
//! structural links hold the shape). Scene 2: two cluster ellipsoids stack
//! through `CL_SS`.
//!
//! Run with `cargo run --example softbody_ellipsoid`.

use bullet_physics::dynamics::{create_ellipsoid, PhysicsWorld, RigidBody, SoftCollision};
use bullet_physics::math::{Real, Vector3};
use bullet_physics::prelude::CollisionShape;

fn ground() -> RigidBody {
    RigidBody::new_static(
        CollisionShape::plane(Vector3::new(0.0, 1.0, 0.0), 0.0),
        Vector3::ZERO,
    )
}

fn main() {
    let dt: Real = 1.0 / 60.0;

    // Scene 1: falling volumetric ball.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(ground());
    let ball = world.add_soft_body(create_ellipsoid(
        Vector3::new(0.0, 4.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        96,
    ));
    {
        let sb = world.soft_body_mut(ball);
        sb.cfg.piterations = 4;
        sb.cfg.collisions = SoftCollision::CL_RS;
        sb.set_total_mass_from_faces(30.0);
        sb.generate_bending_constraints(2, None);
        sb.generate_clusters(16, 32);
        // Shape memory (Bullet's demo recipe: `getCluster(0)->m_matching =
        // 0.35`) plus a softened structural material — the squash/recover
        // character of Bullet's volumetric bodies.
        sb.materials[0].k_lst = 0.5;
        for c in &mut sb.clusters {
            c.matching = 0.35;
        }
    }
    let n_nodes = world.soft_body(ball).nodes.len();
    let mut min_y: Real = 4.0;
    let mut min_half_height: Real = 10.0;
    let mut settled_com_y = 4.0;
    for _ in 0..420 {
        world.step_simulation(dt);
        let sb = world.soft_body(ball);
        let mut com_y: Real = 0.0;
        for n in &sb.nodes {
            com_y += n.x.y;
            min_y = min_y.min(n.x.y);
        }
        com_y /= n_nodes as Real;
        let half_height = sb
            .nodes
            .iter()
            .map(|n| (n.x.y - com_y).abs())
            .fold(0.0, Real::max);
        min_half_height = min_half_height.min(half_height);
        settled_com_y = com_y;
    }
    println!(
        "ellipsoid: nodes={n_nodes} min_y={min_y:.3} settled com y={settled_com_y:.3} min half-height={min_half_height:.3}"
    );
    // The ball reaches the ground, settles above it (the cluster margin
    // band keeps the hull off the plane), squashes below its rest radius
    // on impact and keeps a finite shape.
    assert!(min_y < 1.0, "ball never reached the ground: {min_y}");
    assert!(min_y > -0.1, "ball tunneled through the ground: {min_y}");
    assert!(
        settled_com_y > 0.2 && settled_com_y < 2.5,
        "ball did not settle in the contact band: {settled_com_y}"
    );
    assert!(
        min_half_height < 0.97,
        "ball never squashed: {min_half_height}"
    );
    let sb = world.soft_body(ball);
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()), "ball blew up");
    assert!(!sb.clusters.is_empty(), "clusters vanished");

    // Scene 2: two cluster ellipsoids resting side by side push apart
    // through CL_SS (the contact joints' only horizontal action — the
    // friction term applies while approaching, Bullet's CJoint quirk), then
    // settle on the ground each in its CL_RS band.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(ground());
    let left = world.add_soft_body(create_ellipsoid(
        Vector3::new(-0.9, 1.5, 0.0),
        Vector3::new(1.0, 0.8, 1.0),
        64,
    ));
    let right = world.add_soft_body(create_ellipsoid(
        Vector3::new(0.9, 1.5, 0.0),
        Vector3::new(1.0, 0.8, 1.0),
        64,
    ));
    for i in [left, right] {
        let sb = world.soft_body_mut(i);
        sb.cfg.piterations = 4;
        sb.cfg.collisions = SoftCollision::CL_RS.union(SoftCollision::CL_SS);
        sb.set_total_mass_from_faces(20.0);
        sb.generate_bending_constraints(2, None);
        sb.generate_clusters(16, 32);
        sb.materials[0].k_lst = 0.5;
        for c in &mut sb.clusters {
            c.matching = 0.35;
        }
    }
    let mut separation: Real = 0.0;
    let mut min_y: Real = 10.0;
    for _ in 0..600 {
        world.step_simulation(dt);
        let (a, b) = (world.soft_body(left), world.soft_body(right));
        let com = |sb: &bullet_physics::dynamics::SoftBody| {
            let mut c = Vector3::ZERO;
            for n in &sb.nodes {
                c += n.x;
            }
            c / sb.nodes.len() as Real
        };
        separation = (com(a) - com(b)).length();
        min_y = a
            .nodes
            .iter()
            .chain(b.nodes.iter())
            .map(|n| n.x.y)
            .fold(min_y, Real::min);
    }
    println!("pair: separation={separation:.3} min_y={min_y:.3}");
    // The CL_SS joints push the overlapping hulls apart (the separation
    // grows past the contact band), both settle above the ground and
    // nothing explodes.
    assert!(
        separation > 2.6,
        "CL_SS never pushed the pair apart: {separation}"
    );
    assert!(min_y > -0.1, "pair tunneled through the ground: {min_y}");
    let all_finite = [left, right]
        .iter()
        .all(|&i| world.soft_body(i).nodes.iter().all(|n| n.x.is_finite()));
    assert!(all_finite, "pair blew up");
    println!("OK: ellipsoid bounces and the CL_SS pair separates.");
}
