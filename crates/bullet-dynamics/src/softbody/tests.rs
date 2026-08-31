//! Phase-A soft-body tests: rope dynamics against closed-form solutions,
//! mass/constant bookkeeping, the deterministic constraint shuffle, and the
//! world stepping integration (world-driven vs manual predict/solve loops
//! must agree bit-for-bit).

use bullet_math::{Matrix3, Real, Vector3};

use super::body::SoftBody;
use super::helpers::{create_rope, ROPE_FIX_END, ROPE_FIX_START};
use crate::world::PhysicsWorld;

const DT: Real = 1.0 / 60.0;
const GRAVITY: Vector3 = Vector3::new(0.0, -9.81, 0.0);

/// A free rope falls like a rigid point set: every node shares the same
/// acceleration, so the link lengths stay exactly at rest and the center of
/// mass follows the semi-implicit Euler closed form
/// `x_n = x_0 + g*dt^2*n*(n+1)/2`.
#[test]
fn free_rope_falls_with_closed_form_com() {
    let mut rope = create_rope(
        Vector3::new(0.0, 10.0, 0.0),
        Vector3::new(4.0, 10.0, 0.0),
        14,
        0,
    );
    rope.cfg.piterations = 4;
    let rest_lengths: Vec<Real> = rope.links.iter().map(|l| l.rl).collect();
    let start_com = rope.get_center_of_mass();

    // Mirror of the semi-implicit Euler gravity integration. The node
    // comparison carries a tolerance: solve_constraints re-derives the
    // velocity as (x-q)*isdt, and with the rope near y=10 the f32
    // subtraction cancels ~7 digits (Bullet's classic system has the same
    // precision character), so bit-exactness is not meaningful here.
    let mut node_v = Vector3::ZERO;
    let mut node_x = rope.nodes[3].x;
    let mut com_v = Vector3::ZERO;
    let mut com_x = start_com;
    for step in 1..=120 {
        rope.predict_motion(DT, GRAVITY);
        rope.solve_constraints(&mut []);
        node_v += GRAVITY * DT;
        node_x += node_v * DT;
        let n = rope.nodes[3];
        com_v += GRAVITY * DT;
        com_x += com_v * DT;
        let com = rope.get_center_of_mass();
        // Tolerances scale with the magnitude: the re-derivation error
        // accumulates linearly in steps (~2e-5 per step in f32).
        assert!(
            (n.v - node_v).length() < 1e-3 * (1.0 + node_v.length()),
            "step {step}: node v {n:?} vs {node_v:?}"
        );
        assert!(
            (n.x - node_x).length() < 1e-3 * (1.0 + node_x.length()),
            "step {step}: node x {:?} vs {node_x:?}",
            n.x
        );
        assert!(
            (com - com_x).length() < 5e-3 * (1.0 + (com_x - start_com).length()),
            "step {step}: com {com:?} vs {com_x:?}"
        );
    }
    // No link stretched: free fall develops no internal stress.
    for (i, l) in rope.links.iter().enumerate() {
        let len = (rope.nodes[l.n[0]].x - rope.nodes[l.n[1]].x).length();
        assert!(
            (len - rest_lengths[i]).abs() < 1e-4,
            "link {i} stretched: {len} vs rest {}",
            rest_lengths[i]
        );
    }
}

/// A rope pinned at one end and released horizontally swings through the
/// bottom, keeps its links near rest length, and retains most of its energy
/// (no damping configured).
#[test]
fn rope_pendulum_swings_through_bottom() {
    let mut rope = create_rope(
        Vector3::new(0.0, 6.0, 0.0),
        Vector3::new(6.0, 6.0, 0.0),
        15,
        ROPE_FIX_START,
    );
    rope.cfg.piterations = 4;

    let mut min_mid_y = Real::INFINITY;
    let mut max_stretch: Real = 0.0;
    let mut crossed = false;
    let mut tip_peak_y: Real = 0.0;
    for _ in 0..400 {
        rope.predict_motion(DT, GRAVITY);
        rope.solve_constraints(&mut []);
        let mid = &rope.nodes[8];
        min_mid_y = min_mid_y.min(mid.x.y);
        // The free end sweeps past x=0 to the far side.
        let tip = rope.nodes[16].x;
        if tip.x < -1.0 {
            crossed = true;
            tip_peak_y = tip_peak_y.max(tip.y);
        }
        for l in &rope.links {
            let len = (rope.nodes[l.n[0]].x - rope.nodes[l.n[1]].x).length();
            max_stretch = max_stretch.max(len - l.rl);
        }
    }
    // The middle node dove well below its release height (swung through).
    assert!(min_mid_y < 5.0, "rope did not swing down: {min_mid_y}");
    assert!(crossed, "rope did not swing past the bottom");
    // Energy is mostly conserved: the tip climbs back toward its 6 m
    // release height on the far side. A rope is not a rigid pendulum (the
    // free end lags and whips), and the PBD iterations dissipate a little,
    // so expect a clear majority of the height rather than all of it.
    assert!(tip_peak_y > 4.0, "tip lost its energy: {tip_peak_y}");
    // The distance constraints held within ~16% of the 0.375 m rest length.
    assert!(max_stretch < 0.06, "links stretched: {max_stretch}");
}

/// A rope pinned at both ends sags into a catenary under gravity.
#[test]
fn two_pinned_rope_sags() {
    let mut rope = create_rope(
        Vector3::new(-2.0, 5.0, 0.0),
        Vector3::new(2.0, 5.0, 0.0),
        30,
        ROPE_FIX_START | ROPE_FIX_END,
    );
    rope.cfg.piterations = 4;

    let mut sag: Real = 0.0;
    for _ in 0..500 {
        rope.predict_motion(DT, GRAVITY);
        rope.solve_constraints(&mut []);
        let mid = rope.nodes[16].x;
        sag = 5.0 - mid.y;
    }
    assert!(sag > 0.05, "rope did not sag: {sag}");
    assert!(sag < 2.0, "rope sagged too far (links broke?): {sag}");
    // Bounds track the node hull with the margin pad (sampled inside
    // predict_motion, one solve behind the read-back sag; the settle
    // micro-oscillation is within this tolerance).
    let pad = rope.margin;
    assert!((rope.bounds[0].y - (5.0 - sag - pad)).abs() < 1e-2);
}

/// Mass bookkeeping: set/get/total/redistribute (Bullet's `setMass` /
/// `getMass` / `getTotalMass` / `setTotalMass`).
#[test]
fn mass_api() {
    let mut body = SoftBody::new(
        &[Vector3::ZERO, Vector3::X, Vector3::Y],
        &[1.0, 2.0, 0.0],
    );
    assert_eq!(body.get_mass(0), 1.0);
    assert_eq!(body.get_mass(1), 2.0);
    assert_eq!(body.get_mass(2), 0.0, "fixed node reads back as 0");
    assert_eq!(body.get_total_mass(), 3.0);

    // Redistribute to 6 kg weighted by the current masses (1:2:0).
    body.set_total_mass(6.0);
    assert_eq!(body.get_mass(0), 2.0);
    assert_eq!(body.get_mass(1), 4.0);
    assert_eq!(body.get_mass(2), 0.0);
    assert_eq!(body.get_total_mass(), 6.0);

    // Pinning a node marks the runtime constants dirty.
    body.set_mass(1, 0.0);
    assert!(body.update_rt_cst);
    assert_eq!(body.get_mass(1), 0.0);
}

/// Link runtime constants follow Bullet's formulas: `c0 = (im0+im1)/kLST`,
/// `c1 = rl^2`.
#[test]
fn link_constants_match_formula() {
    let mut body = SoftBody::new(
        &[Vector3::ZERO, Vector3::new(3.0, 0.0, 0.0)],
        &[2.0, 5.0],
    );
    body.append_link(0, 1, None, false);
    body.update_constants();
    let l = &body.links[0];
    assert_eq!(l.rl, 3.0);
    assert_eq!(l.c1, 9.0);
    let expected_c0 = (0.5 + 0.2) / 1.0;
    assert!((l.c0 - expected_c0).abs() < 1e-6, "c0 = {}", l.c0);

    // Softer material scales c0 by 1/kLST.
    let soft = body.append_material(super::body::Material { k_lst: 0.5, ..Default::default() });
    body.links[0].mat = soft;
    body.update_link_constants();
    assert!((body.links[0].c0 - expected_c0 / 0.5).abs() < 1e-6);
}

/// `randomize_constraints` shuffles with Bullet's built-in deterministic
/// LCG: identical constructions shuffle identically, and the result differs
/// from the unshuffled order.
#[test]
fn randomize_constraints_deterministic() {
    let order = |b: &SoftBody| -> Vec<(usize, usize)> { b.links.iter().map(|l| (l.n[0], l.n[1])).collect() };

    let mut a = create_rope(Vector3::ZERO, Vector3::X, 15, 0);
    let mut b = create_rope(Vector3::ZERO, Vector3::X, 15, 0);
    let original = order(&a);
    a.randomize_constraints();
    b.randomize_constraints();
    assert_eq!(order(&a), order(&b), "shuffles diverged");
    assert_ne!(order(&a), original, "shuffle is identity");
}

/// A stretched two-node link contracts back to its rest length under the
/// position solve (`static_solve`, the kernel behind every solve pass).
#[test]
fn stretched_link_restores_length() {
    let mut body = SoftBody::new(&[Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0)], &[0.0, 1.0]);
    body.append_link(0, 1, None, false);
    body.update_constants();
    // Stretch the free node to 3x rest length.
    body.nodes[1].x = Vector3::new(3.0, 0.0, 0.0);
    body.static_solve(100, &mut []);
    let len = (body.nodes[0].x - body.nodes[1].x).length();
    assert!((len - 1.0).abs() < 1e-2, "did not contract: {len}");
}

/// The world stepping integration (stages 1s/7s) is bit-identical to a
/// manual predict/solve loop with the same gravity and dt.
#[test]
fn world_stepping_matches_manual_loop() {
    let build = || {
        let mut rope = create_rope(
            Vector3::new(0.0, 6.0, 0.0),
            Vector3::new(6.0, 6.0, 0.0),
            12,
            ROPE_FIX_START,
        );
        rope.cfg.piterations = 4;
        rope
    };

    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    let handle = world.add_soft_body(build());

    let mut manual = build();
    for step in 0..300 {
        world.step_simulation(DT);
        manual.predict_motion(DT, GRAVITY);
        manual.solve_constraints(&mut []);
        for i in 0..manual.nodes.len() {
            let a = world.soft_body(handle).nodes[i].x;
            let b = manual.nodes[i].x;
            // Component-wise: Vec3A equality also compares the unused w
            // lane, which is not part of the physics.
            assert!(
                a.x == b.x && a.y == b.y && a.z == b.z,
                "step {step} node {i}: world {a:?} vs manual {b:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Phase B: contacts and anchors.
// ---------------------------------------------------------------------------

use bullet_collision::shapes::CollisionShape;
use crate::body::RigidBody;

/// A rope falling onto the ground plane settles on the contact band above
/// the surface (never through it), stacked by its link lengths.
#[test]
fn rope_lands_on_plane() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // A vertical rope, 3 m long, at the default margin: the contact band
    // must exceed the per-step fall distance (7.7 m/s impact -> 0.13 m/step),
    // exactly the regime Bullet's 0.25 default targets.
    let mut rope = create_rope(Vector3::new(0.0, 5.0, 0.0), Vector3::new(0.0, 2.0, 0.0), 30, 0);
    rope.cfg.piterations = 4;
    let handle = world.add_soft_body(rope);

    let mut min_y = Real::INFINITY;
    let mut max_y = Real::NEG_INFINITY;
    for _ in 0..1500 {
        world.step_simulation(DT);
        let sb = world.soft_body(handle);
        for n in &sb.nodes {
            min_y = min_y.min(n.x.y);
            max_y = max_y.max(n.x.y);
        }
    }
    // The bottom node rests inside/above the band (surface + margin = 0.25),
    // never through the surface; the rope keeps roughly its 3 m span.
    assert!(min_y > 0.0, "rope penetrated the plane: {min_y}");
    assert!(min_y < 0.6, "rope never settled on the plane: {min_y}");
    // And it came to rest (contacts damp the fall; the band keeps a mild
    // micro-oscillation with the soft kKHR=0.1 hardness — same as Bullet).
    let v = world
        .soft_body(handle)
        .nodes
        .iter()
        .map(|n| n.v.length_squared())
        .fold(0.0, Real::max);
    assert!(v < 0.05, "rope still moving: {v}");
}

/// A rope draped over a sphere is supported by it (never falls through),
/// though low friction lets it slide off the sides eventually.
#[test]
fn rope_rests_on_sphere() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    world.add_body(RigidBody::new_static(
        CollisionShape::sphere(1.0),
        Vector3::ZERO,
    ));
    // A horizontal rope just above the sphere top (resting contact from the
    // start — no impact to tunnel through).
    let mut rope = create_rope(
        Vector3::new(-2.0, 1.4, 0.0),
        Vector3::new(2.0, 1.4, 0.0),
        30,
        0,
    );
    rope.cfg.piterations = 4;
    let handle = world.add_soft_body(rope);

    let mut min_center_dist = Real::INFINITY;
    let mut early_mid_y = Real::INFINITY;
    for step in 0..600 {
        world.step_simulation(DT);
        let sb = world.soft_body(handle);
        if step < 60 {
            // Before sliding off, the middle is held above the sphere top.
            early_mid_y = early_mid_y.min(sb.nodes[16].x.y);
        }
        for n in &sb.nodes {
            min_center_dist = min_center_dist.min(n.x.length());
        }
    }
    // Never through the sphere surface (the margin band holds nodes at
    // distance >= 1.0 - contact-solve slack).
    assert!(
        min_center_dist > 0.95,
        "rope penetrated the sphere: {min_center_dist}"
    );
    // The middle was supported initially (above the sphere top band).
    assert!(early_mid_y > 1.0, "middle dropped through early: {early_mid_y}");
}

/// An anchor pins a rope end to a falling dynamic box; the rope follows and
/// the anchor distance stays bounded.
#[test]
fn anchor_pins_rope_to_dynamic_body() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    let box_body = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(0.0, 4.0, 0.0),
        5.0,
    ));

    let mut rope = create_rope(
        Vector3::new(0.0, 4.0, 0.0),
        Vector3::new(2.0, 4.0, 0.0),
        12,
        0,
    );
    rope.cfg.piterations = 4;
    rope.margin = 0.05;
    let handle = world.add_soft_body(rope);
    let anchor_node = 0;
    let (box_pos, box_ori) = {
        let b = world.body(box_body);
        (b.position, b.orientation)
    };
    world.soft_body_mut(handle).append_anchor(
        anchor_node,
        box_body,
        box_pos,
        box_ori,
        1.0,
    );

    for _ in 0..600 {
        world.step_simulation(DT);
    }
    // The box fell under gravity dragging the rope; the anchored end stayed
    // glued to it (the box has fallen well below its start height).
    let box_pos = world.body(box_body).position;
    assert!(box_pos.y < 1.0, "box did not fall: {box_pos:?}");
    let anchored = world.soft_body(handle).nodes[anchor_node].x;
    assert!(
        (anchored - box_pos).length() < 0.2,
        "anchor stretched: node {anchored:?} vs box {box_pos:?}"
    );
}

/// A heavy rope landing on a light dynamic box (itself resting on the
/// ground) presses it down; nothing explodes.
#[test]
fn rope_pushes_dynamic_box() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let box_body = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::new(0.0, 1.5, 0.0),
        5.0,
    ));

    let mut rope = create_rope(Vector3::new(0.0, 4.0, 0.0), Vector3::new(0.0, 2.0, 0.0), 20, 0);
    rope.cfg.piterations = 4;
    // Lighter than the box (a stable regime; the extreme ratio belongs to
    // the probe comparisons).
    rope.set_total_mass(2.0);
    let handle = world.add_soft_body(rope);

    let mut max_box_v = 0.0 as Real;
    let mut min_node_y = Real::INFINITY;
    for _ in 0..600 {
        world.step_simulation(DT);
        max_box_v = max_box_v
            .max(world.body(box_body).linear_velocity.y.abs());
        min_node_y = min_node_y.min(
            world
                .soft_body(handle)
                .nodes
                .iter()
                .map(|n| n.x.y)
                .fold(Real::INFINITY, Real::min),
        );
    }
    // No explosion: the box stays well below free-fall speeds.
    assert!(max_box_v < 12.0, "box blasted away: {max_box_v}");
    // Nothing sank through the ground.
    assert!(min_node_y > 0.0, "rope sank through: {min_node_y}");
    // The rope's weight visibly presses the box into its contact band:
    // without the rope the box rests at ~0.5 - slop.
    let box_y = world.body(box_body).position.y;
    assert!(box_y < 0.47, "box not pressed down: {box_y}");
    assert!(box_y > 0.2, "box sank: {box_y}");
}

/// The impulse matrix reduces to the scalar effective mass for the
/// no-rotation case (identity check against the closed form).
#[test]
fn impulse_matrix_reduces_to_scalar() {
    use super::collide::impulse_matrix;
    let dt = 1.0 / 60.0;
    let ima: Real = 1.0; // node mass 1
    let imb: Real = 0.5; // body mass 2
    // Static-like body: zero inverse inertia, anchor at the center.
    let m = impulse_matrix(dt, ima, imb, &Matrix3::ZERO, Vector3::ZERO);
    // Effective scalar: 1/(dt*(ima+imb)) on the diagonal.
    let k = 1.0 / (dt * (ima + imb));
    #[cfg(feature = "f64")]
    let expect = Matrix3::from_diagonal(Vector3::splat(k));
    #[cfg(not(feature = "f64"))]
    let expect = Matrix3::from_diagonal(Vector3::splat(k).into());
    assert!((m - expect).abs_diff_eq(Matrix3::ZERO, 1e-4), "m = {m:?}");
}

/// A wide flat box on a ground plane blows up on HEAD *without any soft
/// body* (single-corner contact, spin-up, sink-through — reproduced before
/// the soft-body work began). Recorded here as a known rigid-side bug;
/// skip-worthy until the plane-vs-wide-box narrow phase is fixed.
#[test]
#[ignore = "known rigid bug: wide flat box on plane (see HANDOFF)"]
fn wide_flat_box_on_plane_known_bug() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let box_body = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
        Vector3::new(0.0, 0.4, 0.0),
        5.0,
    ));
    for _ in 0..120 {
        world.step_simulation(DT);
    }
    let y = world.body(box_body).position.y;
    assert!(y > 0.05, "flat box sank: {y}");
}

/// Removing an anchored rigid body cascades to its soft-body anchors
/// (the joint-cascade pattern; a recorded deviation from Bullet).
#[test]
fn remove_body_cascades_anchors() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    let box_body = world.add_body(RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(0.0, 4.0, 0.0),
        5.0,
    ));
    let mut rope = create_rope(Vector3::ZERO, Vector3::X, 5, 0);
    rope.cfg.piterations = 4;
    let handle = world.add_soft_body(rope);
    world
        .soft_body_mut(handle)
        .append_anchor(0, box_body, Vector3::ZERO, bullet_math::Quaternion::IDENTITY, 1.0);
    assert!(world.soft_body(handle).nodes[0].battach);

    world.remove_body(box_body);
    assert!(world.soft_body(handle).anchors.is_empty());
    assert!(!world.soft_body(handle).nodes[0].battach);
    // Stepping without the removed body runs clean.
    for _ in 0..10 {
        world.step_simulation(DT);
    }
}
