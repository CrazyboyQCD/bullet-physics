//! Phase-A soft-body tests: rope dynamics against closed-form solutions,
//! mass/constant bookkeeping, the deterministic constraint shuffle, and the
//! world stepping integration (world-driven vs manual predict/solve loops
//! must agree bit-for-bit).

use bullet_math::{Real, Vector3};

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
        rope.solve_constraints();
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
        rope.solve_constraints();
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
        rope.solve_constraints();
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
    body.static_solve(100);
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
        manual.solve_constraints();
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
