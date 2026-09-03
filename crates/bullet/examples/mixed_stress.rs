//! Mixed-subsystem stress: rigid bodies + multibodies + dynamic triangle-mesh
//! (GImpact) shells sharing one `step_simulation`, verifying that every
//! subsystem interaction stays stable together:
//!
//! - convex-convex (box tower, balls into boxes),
//! - convex-vs-static-mesh and convex-vs-dynamic-mesh (boxes/balls/mb link
//!   colliders against the terrain and the shells),
//! - mesh-vs-mesh (shell-vs-shell, shell-vs-terrain),
//! - multibody contact rows (the swinging chain's link colliders hitting the
//!   shells),
//! - an mb-vs-rigid p2p constraint (anchor multibody holding a rigid ball)
//!   chained into a rigid-vs-rigid p2p joint (ball holding a box), plus the
//!   island merging/waking across all of it.
//!
//! Health checks every 90 frames (finiteness, awake count, max speed, lowest
//! dynamic body) and final assertions: no NaN/inf ever, nothing tunnels under
//! the terrain, velocities settle, and both p2p joints hold their anchors.
//!
//! Usage: `cargo run -p bullet --release --example mixed_stress [-- frames]`

use bullet::dynamics::Constraint;
use bullet::prelude::*;

const DT: Real = 1.0 / 60.0;

/// Wavy 16x16 height-mesh terrain (static TriangleMesh).
fn terrain() -> CollisionShape {
    let n = 16usize;
    let mut verts = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let (xf, zf) = (x as Real, z as Real);
            let y = 0.15 * (0.8 * xf).sin() * (0.7 * zf).cos();
            verts.push(Vector3::new(xf, y, zf));
        }
    }
    let mut idx = Vec::new();
    for z in 0..n - 1 {
        for x in 0..n - 1 {
            let a = (z * n + x) as u32;
            // Winding gives +Y normals (grid handedness matches gimpact_probe).
            idx.push([a, a + n as u32, a + 1]);
            idx.push([a + 1, a + n as u32, a + 1 + n as u32]);
        }
    }
    CollisionShape::triangle_mesh(verts, idx)
}

/// Closed 12-triangle box shell (dynamic TriangleMesh = the GImpact case).
fn shell(half: Real) -> CollisionShape {
    let h = half;
    let v = [
        (-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h),
        (-h, h, -h), (h, h, -h), (h, h, h), (-h, h, h),
    ];
    let verts: Vec<Vector3> = v.iter().map(|&(x, y, z)| Vector3::new(x, y, z)).collect();
    CollisionShape::triangle_mesh(
        verts,
        vec![
            [0, 1, 2], [0, 2, 3],   // bottom (-Y)
            [4, 6, 5], [4, 7, 6],   // top (+Y)
            [0, 5, 1], [0, 4, 5],   // z = -h
            [2, 7, 3], [2, 6, 7],   // z = +h
            [1, 6, 2], [1, 5, 6],   // x = +h
            [0, 3, 7], [0, 7, 4],   // x = -h
        ],
    )
}

fn main() {
    let frames: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(900);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));

    // Static terrain covering the whole play area (grid origin (-7.5, -7.5)).
    world.add_body(RigidBody::new_static(
        terrain(),
        Vector3::new(-7.5, 0.0, -7.5),
    ));

    // Six dynamic shells in a 2x3 patch over the terrain center (mesh-mesh
    // against the terrain and each other; slow drop, one-step travel < band).
    for k in 0..6 {
        let (i, j) = (k % 2, k / 2);
        world.add_body(
            RigidBody::new_dynamic(
                shell(0.3),
                Vector3::new(-0.6 + 1.2 * i as Real, 1.0, -1.2 + 1.2 * j as Real),
                1.0,
            )
            .friction(0.6)
            .restitution(0.0),
        );
    }

    // A three-high box tower next to the shells (convex-convex + box vs
    // terrain mesh).
    for k in 0..3 {
        world.add_body(
            RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.3)),
                Vector3::new(3.0, 0.31 + 0.61 * k as Real, 0.0),
                1.0,
            )
            .friction(0.6)
            .restitution(0.0),
        );
    }

    // Two balls raining onto the shell patch (sphere vs dynamic mesh).
    for (x, z) in [(0.8, 1.6), (-0.8, -1.6)] {
        world.add_body(
            RigidBody::new_dynamic(CollisionShape::sphere(0.3), Vector3::new(x, 3.2, z), 1.0)
                .friction(0.6)
                .restitution(0.0),
        );
    }

    // A fixed-base 2-link revolute chain released near horizontal; its link
    // colliders sweep through the shell patch (mb contact rows vs mesh).
    {
        let length = Vector3::new(1.2, 0.0, 0.0);
        let mut mb = MultiBody::new(2, 1.0, Vector3::ONE, true, true);
        for i in 0..2 {
            mb.setup_revolute(
                i,
                1.0,
                Vector3::ONE,
                if i == 0 { None } else { Some(i - 1) },
                Quaternion::IDENTITY,
                Vector3::Z,
                length,
                length,
            );
        }
        mb.finalize_multidof();
        mb.set_base_pos(Vector3::new(0.0, 2.9, 0.0));
        for i in 0..2 {
            mb.set_joint_pos_multidof(i, &[75.0_f32.to_radians() as Real]);
        }
        let collider = CollisionShape::sphere(0.3);
        mb.set_base_collider(collider.clone(), 0.5, 0.0);
        for i in 0..2 {
            mb.set_link_collider(i, collider.clone(), 0.5, 0.0);
        }
        world.add_multibody(mb);
    }

    // Anchor multibody (0 links) holding a rigid ball through an mb-vs-rigid
    // p2p constraint, the ball itself holding a box through a rigid p2p
    // joint: constraint chains across the mb/rigid boundary. The ball starts
    // with a sideways kick so the whole assembly swings.
    let (anchor_mb, ball, box_i) = {
        let mut anchor = MultiBody::new(0, 1.0, Vector3::ONE, true, true);
        anchor.finalize_multidof();
        anchor.set_base_pos(Vector3::new(-4.0, 4.0, 0.0));
        let ia = world.add_multibody(anchor);
        let ball = world.add_body(
            RigidBody::new_dynamic(CollisionShape::sphere(0.3), Vector3::new(-4.0, 2.8, 0.0), 1.0)
                .friction(0.6)
                .restitution(0.0),
        );
        let snapshot = world.multibodies().to_vec();
        world.add_multibody_constraint(MultiBodyConstraint::point2point_to_rigid(
            ia,
            None,
            ball,
            Vector3::ZERO,
            Vector3::new(0.0, 1.2, 0.0),
            &snapshot,
        ));
        world.body_mut(ball).linear_velocity = Vector3::new(1.5, 0.0, 0.0);

        // Hang the box 0.2 m below the ball: pivots coincide at (-4, 2.3).
        let box_i = world.add_body(
            RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.25)),
                Vector3::new(-4.0, 1.95, 0.0),
                0.8,
            )
            .friction(0.6)
            .restitution(0.0),
        );
        world.add_constraint(Constraint::point2point(
            ball,
            box_i,
            Vector3::new(0.0, -0.5, 0.0),
            Vector3::new(0.0, 0.35, 0.0),
        ));
        (ia, ball, box_i)
    };

    // Run, tracking health: every dynamic body must stay finite and above the
    // terrain's underside (no tunneling), and the step must converge. The
    // chain's end link must sweep down into the shell patch (otherwise the
    // mb-collider-vs-mesh contact path would not be exercised at all).
    let mut violations = 0usize;
    let mut last_max_v: Real = 0.0;
    let mut chain_min_y: Real = Real::MAX;
    for f in 0..frames {
        world.step_simulation(DT);
        chain_min_y = chain_min_y.min(world.multibody(MultiBodyHandle::from_index(0)).link_world_position(1).y);
        let mut awake = 0usize;
        let mut max_v: Real = 0.0;
        let mut min_y: Real = Real::MAX;
        let mut bad = 0usize;
        let mut count = 0usize;
        for b in world.bodies() {
            if b.is_static() {
                continue;
            }
            count += 1;
            let p = b.position;
            let v = b.linear_velocity;
            if !p.is_finite() || !v.is_finite() || !b.angular_velocity.is_finite() || p.y < -0.5 {
                bad += 1;
            }
            let speed = v.length();
            max_v = max_v.max(speed);
            min_y = min_y.min(p.y);
            if b.is_active() {
                awake += 1;
            }
        }
        violations += bad;
        last_max_v = max_v;
        if f % 90 == 0 || f == frames - 1 {
            println!(
                "mixed frame {f} bodies {count} awake {awake} max_v {max_v:.4} min_y {min_y:.3} bad {bad}"
            );
        }
    }

    // p2p residuals: anchor-vs-ball and ball-vs-box anchor points must stay
    // coincident (steady-state Baumgarte residual is centimeters).
    let anchor_pt = world.multibody(anchor_mb).base_world_transform().translation;
    let ball_body = world.body(ball);
    let ball_pt = ball_body.position
        + ball_body.orientation * Vector3::new(0.0, 1.2, 0.0);
    let p2p_mb_err = (anchor_pt - ball_pt).length();
    let box_body = world.body(box_i);
    let hang_ball = ball_body.position + ball_body.orientation * Vector3::new(0.0, -0.5, 0.0);
    let hang_box = box_body.position + box_body.orientation * Vector3::new(0.0, 0.35, 0.0);
    let p2p_rb_err = (hang_ball - hang_box).length();

    println!(
        "mixed result: frames {frames} violations {violations} final max_v {last_max_v:.4} \
         chain_min_y {chain_min_y:.3} p2p_mb_err {p2p_mb_err:.4} p2p_rb_err {p2p_rb_err:.4}"
    );

    assert_eq!(violations, 0, "non-finite state or terrain tunneling");
    assert!(
        last_max_v < 2.0,
        "velocities did not settle (max_v {last_max_v:.3})"
    );
    assert!(
        chain_min_y < 0.84,
        "chain never reached the shell patch (chain_min_y {chain_min_y:.3})"
    );
    assert!(p2p_mb_err < 0.15, "mb-vs-rigid p2p drifted {p2p_mb_err:.3}");
    assert!(p2p_rb_err < 0.15, "rigid p2p drifted {p2p_rb_err:.3}");
    println!("mixed stress OK");
}
