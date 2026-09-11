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
        SoftBody::solve_constraints(0, core::slice::from_mut(&mut rope), &mut [], None);
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
        SoftBody::solve_constraints(0, core::slice::from_mut(&mut rope), &mut [], None);
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
        SoftBody::solve_constraints(0, core::slice::from_mut(&mut rope), &mut [], None);
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
        SoftBody::solve_constraints(0, core::slice::from_mut(&mut manual), &mut [], None);
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
    // The box stays grounded under the rope's weight. Since the
    // persistent-manifold merge (Bullet's four-corner accumulation) the
    // resting box is far sturdier: a 2 kg rope cannot press a 5 kg
    // four-corner-supported box down any more — stock Bullet behaves the
    // same way (rope_press_probe: box_y stays at 0.5000 for all 600
    // frames, vs the old single-point era where the box sank to ~0.47).
    let box_y = world.body(box_body).position.y;
    assert!(box_y > 0.45, "box sank: {box_y}");
    assert!(box_y < 0.51, "box floated away: {box_y}");
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

/// Regression test for the persistent-manifold merge: with the old
/// replace-on-capture semantics the plane-vs-box manifold was a single
/// corner point that flipped between frames, and the solver torque pumped
/// the box through the plane (y reached −14 by frame 120; stock Bullet
/// settles this scene at y = 0.1 via the same accumulation).
#[test]
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

// ---------------------------------------------------------------------
// Cloth phase (faces, aero/pressure/volume, VF_SS, raycast).

#[test]
fn patch_topology_matches_create_patch() {
    use super::helpers::{create_patch, patch_fixeds};
    let psb = create_patch(
        Vector3::ZERO,
        Vector3::X,
        Vector3::Z,
        Vector3::new(1.0, 0.0, 1.0),
        4,
        3,
        patch_fixeds::CORNER_00 | patch_fixeds::CORNER_11,
        true,
        0.0,
    );
    // 4x3 nodes, per-cell: 1 horizontal + 1 vertical + 1 diagonal link and
    // 2 faces (Bullet's CreatePatch layout).
    assert_eq!(psb.nodes.len(), 12);
    assert_eq!(psb.links.len(), 3 * 3 + 4 * 2 + 3 * 2);
    assert_eq!(psb.faces.len(), 3 * 2 * 2);
    // Pinned corners.
    assert_eq!(psb.nodes[0].im, 0.0);
    assert_eq!(psb.nodes[2 * 4 + 3].im, 0.0);
    assert!(psb.nodes[1].im > 0.0);
    // Degenerate faces rejected.
    let mut b = create_patch(
        Vector3::ZERO,
        Vector3::X,
        Vector3::Z,
        Vector3::new(1.0, 0.0, 1.0),
        2,
        2,
        0,
        false,
        0.0,
    );
    let before = b.faces.len();
    b.append_face(0, 0, 1, None);
    assert_eq!(b.faces.len(), before);
}

#[test]
fn face_area_and_volume() {
    use super::helpers::create_patch;
    // A 2x2 patch of unit cells: 2 faces, each with doubled area 1.
    let mut psb = create_patch(
        Vector3::ZERO,
        Vector3::X,
        Vector3::Z,
        Vector3::new(1.0, 0.0, 1.0),
        2,
        2,
        0,
        false,
        0.0,
    );
    psb.update_constants(); // resets rest lengths + areas
    for f in &psb.faces {
        assert!((f.ra - 1.0).abs() < 1e-6, "face ra {}", f.ra);
    }
    // Corners 0/3 sit on both faces (area 2/3), corners 1/2 on one (1/3) —
    // the single cell's diagonal splits the adjacency.
    assert!((psb.nodes[0].area - 2.0 / 3.0).abs() < 1e-5, "area0 {}", psb.nodes[0].area);
    assert!((psb.nodes[3].area - 2.0 / 3.0).abs() < 1e-5, "area3 {}", psb.nodes[3].area);
    assert!((psb.nodes[1].area - 1.0 / 3.0).abs() < 1e-5, "area1 {}", psb.nodes[1].area);
    assert!((psb.nodes[2].area - 1.0 / 3.0).abs() < 1e-5, "area2 {}", psb.nodes[2].area);
    // A closed axis-aligned box from two triangles per side: unit volume.
    let mut boxy = SoftBody::new(
        &[
            Vector3::ZERO,
            Vector3::X,
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::Z,
            Vector3::Y,
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
        ],
        &[1.0; 8],
    );
    let q = |i: usize| i;
    // Bottom (-Y): 0-2-1, 0-3-2; top (+Y): 4-5-6, 4-6-7.
    boxy.append_face(q(0), q(2), q(1), None);
    boxy.append_face(q(0), q(3), q(2), None);
    boxy.append_face(q(4), q(6), q(5), None);
    boxy.append_face(q(4), q(7), q(6), None);
    // Sides.
    boxy.append_face(q(0), q(5), q(1), None);
    boxy.append_face(q(0), q(4), q(5), None);
    boxy.append_face(q(1), q(6), q(2), None);
    boxy.append_face(q(1), q(5), q(6), None);
    boxy.append_face(q(2), q(7), q(3), None);
    boxy.append_face(q(2), q(6), q(7), None);
    boxy.append_face(q(3), q(4), q(0), None);
    boxy.append_face(q(3), q(7), q(4), None);
    assert!((boxy.get_volume() - 1.0).abs() < 1e-6, "volume {}", boxy.get_volume());
}

#[test]
fn soft_soft_vf_separates_overlapping_patches() {
    use super::config::SoftCollision;
    use super::helpers::create_patch;
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    // Two parallel horizontal patches starting inside the VF_SS band (the
    // margin sum is 0.5): the contact holds them margin-sum apart while
    // they fall together (starting deeply inter-penetrating instead blasts
    // them apart — the same violent recovery Bullet's kernel has).
    let a = world.add_soft_body(create_patch(
        Vector3::new(-1.0, 1.0, -1.0),
        Vector3::new(1.0, 1.0, -1.0),
        Vector3::new(-1.0, 1.0, 1.0),
        Vector3::new(1.0, 1.0, 1.0),
        6,
        6,
        0,
        true,
        0.0,
    ));
    let b = world.add_soft_body(create_patch(
        Vector3::new(-1.0, 0.55, -1.0),
        Vector3::new(1.0, 0.55, -1.0),
        Vector3::new(-1.0, 0.55, 1.0),
        Vector3::new(1.0, 0.55, 1.0),
        6,
        6,
        0,
        true,
        0.0,
    ));
    for i in [a, b] {
        let sb = world.soft_body_mut(i);
        sb.cfg.piterations = 4;
        sb.cfg.collisions = SoftCollision::VF_SS;
    }
    let mut min_gap = Real::INFINITY;
    for _ in 0..240 {
        world.step_simulation(DT);
        // A is the upper sheet: its lowest node minus B's highest.
        let gap = {
            let sa = world.soft_body(a);
            let sb = world.soft_body(b);
            let min_a = sa.nodes.iter().map(|n| n.x.y).fold(Real::INFINITY, Real::min);
            let max_b = sb.nodes.iter().map(|n| n.x.y).fold(Real::NEG_INFINITY, Real::max);
            min_a - max_b
        };
        min_gap = min_gap.min(gap);
    }
    // They never crossed (the band keeps the lower sheet below the upper
    // one) and stay within a hand's reach of the margin-sum equilibrium.
    assert!(min_gap > 0.0, "patches crossed: gap {min_gap}");
    assert!(min_gap < 1.0, "patches blasted apart: gap {min_gap}");
    assert!(
        world.soft_body(a).nodes.iter().all(|n| n.x.is_finite()),
        "VF_SS exploded"
    );
}

#[test]
fn soft_body_raycast_hits_faces() {
    use super::helpers::create_patch;
    let mut world = PhysicsWorld::new();
    let cloth = world.add_soft_body(create_patch(
        Vector3::new(-1.0, 3.0, -1.0),
        Vector3::new(1.0, 3.0, -1.0),
        Vector3::new(-1.0, 3.0, 1.0),
        Vector3::new(1.0, 3.0, 1.0),
        4,
        4,
        0,
        true,
        0.0,
    ));
    // Straight down through the patch center: fraction = 2/5 (from y=5 to
    // the y=3 sheet over the 5 m segment).
    let hit = world
        .raycast_soft_closest(Vector3::new(0.0, 5.0, 0.0), Vector3::new(0.0, 0.0, 0.0))
        .expect("ray missed the cloth");
    assert_eq!(hit.0, cloth);
    assert!((hit.2 - 2.0 / 5.0).abs() < 1e-4, "fraction {}", hit.2);
    // Parallel ray misses.
    assert!(world
        .raycast_soft_closest(Vector3::new(-5.0, 3.0, 0.0), Vector3::new(5.0, 3.0, 0.0))
        .is_none());
}

#[test]
fn vertex_aero_pushes_downwind() {
    use super::config::AeroModel;
    use super::helpers::create_patch;
    // A free patch flying in a crosswind: after a few steps the nodes drift
    // downwind (the aero force points along the relative wind on the
    // leeward side of the sail).
    let mut psb = create_patch(
        Vector3::new(-1.0, 10.0, -1.0),
        Vector3::new(1.0, 10.0, -1.0),
        Vector3::new(-1.0, 10.0, 1.0),
        Vector3::new(1.0, 10.0, 1.0),
        6,
        6,
        0,
        true,
        0.0,
    );
    psb.cfg.piterations = 4;
    psb.cfg.aeromodel = AeroModel::VTwoSided;
    psb.cfg.k_lf = 0.3;
    psb.cfg.k_dg = 0.1;
    psb.wind_velocity = Vector3::new(8.0, 0.0, 0.0);
    let mut com_x0 = Vector3::ZERO;
    for (i, n) in psb.nodes.iter().enumerate() {
        com_x0 += n.x * if n.im > 0.0 { 1.0 } else { 0.0 };
        let _ = i;
    }
    let mut world = PhysicsWorld::new();
    let handle = world.add_soft_body(psb);
    for _ in 0..90 {
        world.step_simulation(DT);
    }
    let sb = world.soft_body(handle);
    let mut com = Vector3::ZERO;
    for n in &sb.nodes {
        if n.im > 0.0 {
            com += n.x;
        }
    }
    assert!(com.x > com_x0.x + 0.2, "aero did not carry the sheet: {}", com.x - com_x0.x);
}

#[test]
fn pressure_inflates_closed_pillow() {
    // Two sewn sheets (the demo's pillow, smaller): the volume grows under
    // kPR without exploding.
    let res = 6usize;
    let mut positions = Vec::new();
    let masses = vec![1.0 as Real; 2 * res * res];
    for iy in 0..res {
        for ix in 0..res {
            let (tx, tz) = (ix as Real / (res - 1) as Real, iy as Real / (res - 1) as Real);
            positions.push(Vector3::new(tx, 4.0, tz));
        }
    }
    for iy in 0..res {
        for ix in 0..res {
            let (tx, tz) = (ix as Real / (res - 1) as Real, iy as Real / (res - 1) as Real);
            positions.push(Vector3::new(tx, 3.94, tz));
        }
    }
    let mut psb = SoftBody::new(&positions, &masses);
    let idx = |s: usize, x: usize, y: usize| s * res * res + y * res + x;
    for sheet in 0..2 {
        for iy in 0..res {
            for ix in 0..res {
                if ix + 1 < res {
                    psb.append_link(idx(sheet, ix, iy), idx(sheet, ix + 1, iy), None, false);
                }
                if iy + 1 < res {
                    psb.append_link(idx(sheet, ix, iy), idx(sheet, ix, iy + 1), None, false);
                }
                if ix + 1 < res && iy + 1 < res {
                    let (n00, n01, n10, n11) = (
                        idx(sheet, ix, iy),
                        idx(sheet, ix + 1, iy),
                        idx(sheet, ix, iy + 1),
                        idx(sheet, ix + 1, iy + 1),
                    );
                    if sheet == 0 {
                        psb.append_face(n00, n01, n11, None);
                        psb.append_face(n11, n10, n00, None);
                    } else {
                        psb.append_face(n00, n10, n11, None);
                        psb.append_face(n11, n01, n00, None);
                    }
                }
            }
        }
    }
    for i in 0..res {
        psb.append_link(idx(0, i, 0), idx(1, i, 0), None, false);
        psb.append_link(idx(0, i, res - 1), idx(1, i, res - 1), None, false);
        psb.append_link(idx(0, 0, i), idx(1, 0, i), None, false);
        psb.append_link(idx(0, res - 1, i), idx(1, res - 1, i), None, false);
    }
    psb.cfg.piterations = 4;
    psb.cfg.k_pr = 2.0;
    let initial = psb.get_volume().abs();
    let mut world = PhysicsWorld::new();
    let handle = world.add_soft_body(psb);
    for _ in 0..300 {
        world.step_simulation(DT);
    }
    let volume = world.soft_body(handle).get_volume().abs();
    assert!(volume > initial * 2.0, "no inflation: {volume} vs {initial}");
    assert!(volume < initial * 40.0, "explosion: {volume} vs {initial}");
}

// ---- Phase D: tetras, clusters, pose, quickhull, CL_RS/CL_SS ----

use super::helpers::{create_ellipsoid, create_from_convex_hull};

/// The ellipsoid builder keeps every Hammersley point as a hull vertex
/// (all points of a sphere point set are extreme), and the hull closes:
/// its divergence-theorem volume is a meaningful fraction of the true
/// ellipsoid volume (the hull is inscribed).
#[test]
fn ellipsoid_builder_topology() {
    let sb = create_ellipsoid(Vector3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 2.0, 1.5), 40);
    assert_eq!(sb.nodes.len(), 43, "3 + res nodes survive as hull vertices");
    assert!(sb.faces.len() >= 4);
    assert!(sb.links.len() >= sb.faces.len() * 3 / 2, "hull edges are links");
    let v = sb.get_volume().abs();
    let true_v = 4.0 / 3.0 * bullet_math::PI * 1.0 * 2.0 * 1.5;
    assert!(
        v > true_v * 0.4 && v < true_v,
        "hull volume {v} not an inscribed fraction of {true_v}"
    );
}

/// `set_total_mass_from_faces` distributes mass by incident face area: a
/// node on a dense end of a wedge-shaped sheet carries more mass than one
/// on the sparse end (same node count, larger triangles).
#[test]
fn face_area_mass_distribution() {
    let mut psb = SoftBody::new(
        &[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 0.0), Vector3::new(2.0, 1.0, 0.0)],
        &[0.0; 6],
    );
    psb.append_face(0, 1, 4, None);
    psb.append_face(4, 3, 0, None);
    psb.append_face(1, 2, 5, None);
    psb.append_face(5, 4, 1, None);
    psb.set_total_mass_from_faces(6.0);
    let total = psb.get_total_mass();
    assert!((total - 6.0).abs() < 1e-4, "total mass {total}");
    // All triangles share the doubled area 1, so a node's share scales with
    // its incident-face count: nodes 1 and 4 touch three faces, 0 and 5
    // two, 2 and 3 one.
    for i in [1, 4] {
        assert!((psb.get_mass(i) - 1.5).abs() < 1e-3, "node {i} mass {}", psb.get_mass(i));
    }
    for i in [0, 5] {
        assert!((psb.get_mass(i) - 1.0).abs() < 1e-3, "node {i} mass {}", psb.get_mass(i));
    }
    for i in [2, 3] {
        assert!((psb.get_mass(i) - 0.5).abs() < 1e-3, "node {i} mass {}", psb.get_mass(i));
    }
}

/// `set_volume_mass` weights by incident tetra volumes: the tetra holding
/// fixed nodes keeps them at mass zero, and a node shared by two tetras
/// with equal rest volumes carries double the share of a single-tetra node
/// of the same volume.
#[test]
fn tetra_volume_mass() {
    let mut psb = SoftBody::new(
        &[
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ],
        &[0.0; 5],
    );
    psb.append_tetra(0, 1, 2, 3, None);
    psb.append_tetra(1, 2, 3, 4, None);
    // The first tetra's rest volume (triple product) is 1, the second's 2;
    // `im = rank / sum(|rv|)` then `set_total_mass` lands shares of
    // 4/3 (single small tetra), 2 (shared nodes) and 8/3 (the big one's
    // lone node) summing to 10.
    psb.set_volume_mass(10.0);
    assert!((psb.get_mass(0) - 4.0 / 3.0).abs() < 1e-3);
    for i in [1, 2, 3] {
        assert!((psb.get_mass(i) - 2.0).abs() < 1e-3, "node {i} mass {}", psb.get_mass(i));
    }
    assert!((psb.get_mass(4) - 8.0 / 3.0).abs() < 1e-3);
    assert!((psb.get_total_mass() - 10.0).abs() < 1e-3);
}

/// `generate_bending_constraints(2)` links the patch's diagonal neighbors:
/// for an `n x n` grid, every interior node gains its distance-2 partners
/// (the count matches the adjacency-list derivation).
#[test]
fn bending_links_count_on_grid() {
    let mut psb = super::helpers::create_patch(
        Vector3::ZERO,
        Vector3::new(3.0, 0.0, 0.0),
        Vector3::new(0.0, 3.0, 0.0),
        Vector3::new(3.0, 3.0, 0.0),
        5,
        5,
        0,
        false,
        0.0,
    );
    let structural = psb.links.len();
    let n = psb.generate_bending_constraints(2, None);
    assert_eq!(psb.links.len(), structural + n);
    assert!(n > 0, "grid has distance-2 pairs");
    assert!(psb.links.iter().any(|l| l.bending));
    // A 5x5 lattice: distance-2 pairs (|dx|+|dy|) == 2 are the axis-2
    // separations plus the unit diagonals of each cell.
    let mut expected = 0usize;
    for y in 0..5 {
        for x in 0..5 {
            if x + 2 < 5 {
                expected += 1;
            }
            if y + 2 < 5 {
                expected += 1;
            }
        }
    }
    let diagonals = 4 * 4 * 2; // both diagonal directions per cell
    assert_eq!(n, expected + diagonals, "axis-2 pairs + cell diagonals");
}

/// `generate_clusters` on a K-mean split: the master cluster lands at
/// index 0 holding every node, colliding clusters partition the body, and
/// the connectivity matrix is symmetric with connected clusters flagged.
#[test]
fn cluster_generation_master_and_connectivity() {
    let mut sb = create_ellipsoid(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0), 64);
    sb.generate_clusters(8, 16);
    // K-mean clusters may collapse to empty and get dropped, so the count
    // is `<= k + master`; the master always lands at index 0.
    assert!(sb.clusters.len() <= 9 && sb.clusters.len() >= 2);
    let master = &sb.clusters[0];
    assert_eq!(master.nodes.len(), sb.nodes.len(), "master holds all nodes");
    assert!(!master.collide, "master does not collide");
    let mut covered = vec![false; sb.nodes.len()];
    for c in &sb.clusters[1..] {
        assert!(c.collide);
        for &ni in &c.nodes {
            covered[ni] = true;
        }
    }
    assert!(covered.iter().all(|&c| c), "kmean clusters cover the nodes");
    let n = sb.clusters.len();
    assert_eq!(sb.cluster_connectivity.len(), n * n);
    for a in 0..n {
        for b in 0..n {
            assert_eq!(
                sb.cluster_connectivity[a + b * n],
                sb.cluster_connectivity[b + a * n],
                "connectivity symmetric"
            );
        }
    }
    // Distinct kmean clusters may still be connected through the face
    // merge; the master is connected to everything.
    assert!(sb.cluster_connectivity[n], "master connects to cluster 1");
}

/// `generate_clusters(0)` falls back to one cluster per face (or per
/// tetra) — Bullet's fallback branch installs no master cluster.
#[test]
fn cluster_per_face_fallback() {
    let mut sb = create_ellipsoid(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0), 16);
    let nfaces = sb.faces.len();
    sb.generate_clusters(0, 0);
    assert_eq!(sb.clusters.len(), nfaces, "one per face, no master");
}

/// The polar decomposition recovers a pure rotation's orthogonal factor
/// and the identity symmetric factor.
#[test]
fn polar_decompose_recovers_rotation() {
    use super::cluster::polar_decompose;
    let r = Matrix3::from_quat(bullet_math::compat::quat_from_axis_angle(
        Vector3::new(0.3, 0.8, 0.5).normalize(),
        0.7,
    ));
    let (q, s) = polar_decompose(&r);
    for (qa, ra) in q.to_cols_array().iter().zip(r.to_cols_array().iter()) {
        assert!((qa - ra).abs() < 1e-3, "q != r");
    }
    for (sa, ia) in s.to_cols_array().iter().zip(Matrix3::IDENTITY.to_cols_array().iter()) {
        assert!((sa - ia).abs() < 1e-3, "s != I");
    }
}

/// `update_pose` tracks a rigid rotation of the whole body: after
/// rotating every node, the pose rotation matches (the axis-angle of
/// `pose.rot` recovers the applied rotation).
#[test]
fn pose_tracks_rigid_rotation() {
    let mut sb = create_ellipsoid(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0), 32);
    sb.set_total_mass_from_faces(10.0);
    sb.set_pose(true, true);
    let q = bullet_math::compat::quat_from_axis_angle(
        Vector3::new(0.0, 0.0, 1.0),
        bullet_math::FRAC_PI_2,
    );
    for n in &mut sb.nodes {
        n.x = q * n.x;
        n.v = Vector3::ZERO;
    }
    sb.update_pose();
    // pose.rot maps rest positions onto current ones: applying it to a
    // rest offset must land on the rotated offset.
    let i = sb.nodes.len() / 2;
    let rest = sb.pose.pos[i];
    let world = sb.pose.rot * rest + sb.pose.com;
    let expect = q * (rest + Vector3::ZERO);
    assert!(
        (world - expect).length() < 0.05,
        "pose rotation did not track: {world:?} vs {expect:?}"
    );
}

/// CL_RS: a cluster ellipsoid falls onto the ground plane and settles in
/// the margin band without tunneling (the world-level integration of the
/// cluster colliders + joint solve).
#[test]
fn cluster_ellipsoid_settles_on_plane() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(GRAVITY);
    let ground = world.add_body(RigidBody::new_static(
        CollisionShape::plane(Vector3::new(0.0, 1.0, 0.0), 0.0),
        Vector3::ZERO,
    ));
    let _ = ground;
    let ball = world.add_soft_body(create_ellipsoid(
        Vector3::new(0.0, 3.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        48,
    ));
    {
        let sb = world.soft_body_mut(ball);
        sb.cfg.piterations = 4;
        sb.cfg.collisions = super::config::SoftCollision::CL_RS;
        sb.set_total_mass_from_faces(20.0);
        sb.generate_bending_constraints(2, None);
        sb.generate_clusters(8, 16);
        for c in &mut sb.clusters {
            c.matching = 0.3;
        }
    }
    let mut min_y: Real = 3.0;
    for _ in 0..360 {
        world.step_simulation(DT);
        for n in &world.soft_body(ball).nodes {
            min_y = min_y.min(n.x.y);
        }
    }
    assert!(min_y < 1.0, "never reached the ground: {min_y}");
    assert!(min_y > -0.1, "tunneled: {min_y}");
    let sb = world.soft_body(ball);
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()));
}

/// CL_SS: two overlapping cluster ellipsoids push each other apart through
/// their contact joints (the world-level soft-soft dispatch).
#[test]
fn cluster_pair_pushes_apart() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let a = world.add_soft_body(create_ellipsoid(
        Vector3::new(-0.4, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        32,
    ));
    let b = world.add_soft_body(create_ellipsoid(
        Vector3::new(0.4, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        32,
    ));
    for i in [a, b] {
        let sb = world.soft_body_mut(i);
        sb.cfg.piterations = 4;
        sb.cfg.collisions = super::config::SoftCollision::CL_SS;
        sb.set_total_mass_from_faces(10.0);
        sb.generate_clusters(8, 16);
        for c in &mut sb.clusters {
            c.matching = 0.3;
        }
    }
    let com = |sb: &SoftBody| {
        let mut c = Vector3::ZERO;
        for n in &sb.nodes {
            c += n.x;
        }
        c / sb.nodes.len() as Real
    };
    let start = (com(world.soft_body(a)) - com(world.soft_body(b))).length();
    let mut separation = start;
    for _ in 0..240 {
        world.step_simulation(DT);
        separation = (com(world.soft_body(a)) - com(world.soft_body(b))).length();
    }
    assert!(
        separation > start + 0.3,
        "CL_SS did not separate: {separation} vs {start}"
    );
}

/// `create_from_convex_hull` drops interior points: only the hull
/// vertices become nodes.
#[test]
fn convex_hull_builder_drops_interior() {
    let pts = [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::ZERO,
    ];
    let sb = create_from_convex_hull(&pts);
    assert_eq!(sb.nodes.len(), 6, "interior point excluded");
    assert_eq!(sb.faces.len(), 8, "octahedron faces");
}

// ---------------------------------------------------------------------------
// Phase E: joints, transforms, cut/refine.
// ---------------------------------------------------------------------------

use bullet_math::compat::quat_from_axis_angle;
use super::body::{ImplicitFn, Side};
use super::joints::{AngularControl, AngularJointSpecs, AJoint, JointSpecs, LinearJointSpecs};

/// The SoftDemo motor control (Bullet's `MotorControl`): drive the joint
/// speed toward `goal` with a clamped torque rate.
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

/// A massaged ellipsoid ready for cluster tests (uniform face-weighted
/// mass, bending links, 8 K-mean clusters + master).
fn cluster_ellipsoid(center: Vector3) -> SoftBody {
    let mut sb = create_ellipsoid(center, Vector3::splat(1.0), 64);
    sb.set_total_mass_from_faces(20.0);
    sb.generate_bending_constraints(2, None);
    sb.generate_clusters(8, 32);
    sb
}

/// The transform family maps positions/previous positions rigidly, keeps
/// link rest lengths consistent with the geometry, and refreshes the
/// bounds.
#[test]
fn transform_translates_rotates_and_scales() {
    let mut sb = cluster_ellipsoid(Vector3::ZERO);
    // Translate: rigid offset of every node, rest lengths unchanged.
    let before: Vec<Vector3> = sb.nodes.iter().map(|n| n.x).collect();
    sb.translate(Vector3::new(1.0, 2.0, 3.0));
    for (n, &b) in sb.nodes.iter().zip(&before) {
        assert!((n.x - (b + Vector3::new(1.0, 2.0, 3.0))).length() < 1e-5);
        assert!((n.q - n.x).length() < 1e-5, "previous position must move with x");
    }
    // Rotate: distance structure preserved; the COM maps rigidly (it does
    // not sit at the world origin, so it moves with the rotation).
    let com = sb.get_center_of_mass();
    sb.rotate(quat_from_axis_angle(Vector3::Z, 0.7));
    for l in &sb.links {
        let len = (sb.nodes[l.n[0]].x - sb.nodes[l.n[1]].x).length();
        assert!((len - l.rl).abs() < 1e-4, "rotation changed a rest length");
    }
    let rot = Matrix3::from_quat(quat_from_axis_angle(Vector3::Z, 0.7));
    assert!(
        (sb.get_center_of_mass() - rot * com).length() < 1e-4,
        "COM must transform rigidly"
    );
    // Scale: uniform scaling doubles the rest lengths and the geometry
    // with them (update_constants re-seeds the rest lengths).
    let [n0, n1] = sb.links[0].n;
    let len_before = (sb.nodes[n0].x - sb.nodes[n1].x).length();
    sb.scale(Vector3::splat(2.0));
    let len = (sb.nodes[n0].x - sb.nodes[n1].x).length();
    let rl_now = sb.links[0].rl;
    assert!(
        (len - rl_now).abs() < 1e-4,
        "update_constants must re-seed rest lengths after scaling: {len} vs {rl_now}"
    );
    assert!(
        (len - 2.0 * len_before).abs() < 1e-4,
        "uniform scaling must double the edge: {len} vs {}",
        2.0 * len_before
    );
}

/// `transform_to` teleports the body onto a target frame: the rigid fit
/// composed with the current one maps every node back to where a rigid
/// motion would put it, and the fit itself reports the applied motion.
#[test]
fn transform_to_recovers_rigid_motion() {
    let mut sb = cluster_ellipsoid(Vector3::ZERO);
    let rest: Vec<Vector3> = sb.nodes.iter().map(|n| n.x).collect();
    let rest_com = sb.get_center_of_mass();
    let rot = quat_from_axis_angle(Vector3::new(0.3, 0.5, 0.8).normalize(), 0.9);
    let t = Vector3::new(2.0, -1.0, 0.5);
    sb.transform(Matrix3::from_quat(rot), t);
    // The fit recovers the motion (origin = the mass-weighted COM, which
    // need not sit at the construction center for face-weighted masses).
    let (fit_r, fit_t) = sb.get_rigid_transform();
    let com = sb.get_center_of_mass();
    assert!((fit_t - com).length() < 1e-6, "fit origin {fit_t:?} vs com {com:?}");
    assert!((fit_t - (Matrix3::from_quat(rot) * rest_com + t)).length() < 1e-3);
    let diff = fit_r - Matrix3::from_quat(rot);
    let r_err = diff
        .x_axis
        .abs()
        .max(diff.y_axis.abs())
        .max(diff.z_axis.abs())
        .max_element();
    assert!(r_err < 1e-3, "fit rotation off by {r_err}");
    // Undo: back onto the identity frame at the origin — every node lands
    // on the rest pose relative to its COM.
    sb.transform_to(Matrix3::IDENTITY, Vector3::ZERO);
    for (n, &b) in sb.nodes.iter().zip(&rest) {
        assert!(
            (n.x - (b - rest_com)).length() < 1e-3,
            "node did not return: {:?} vs {:?}",
            n.x,
            b - rest_com
        );
    }
    assert!((sb.get_center_of_mass() - Vector3::ZERO).length() < 1e-4);
}

/// `cut_link` splits every matching edge in place through a fresh node
/// pair; a non-edge pair changes nothing.
#[test]
fn cut_link_splits_topology() {
    let mut rope = create_rope(Vector3::ZERO, Vector3::new(3.0, 0.0, 0.0), 2, 0);
    // No match: (0, 2) is not an edge.
    assert!(!rope.cut_link(0, 2, 0.5));
    assert_eq!(rope.nodes.len(), 4);
    assert_eq!(rope.links.len(), 3);
    // Cut (1, 2) at 0.25: the edge splits into (1, pn0) and (pn1, 2),
    // inheriting the interpolated velocity.
    rope.nodes[1].v = Vector3::new(0.0, 4.0, 0.0);
    rope.nodes[2].v = Vector3::ZERO;
    assert!(rope.cut_link(1, 2, 0.25));
    assert_eq!(rope.nodes.len(), 6, "a fresh node pair is appended");
    assert_eq!(rope.links.len(), 4, "the cut edge becomes two");
    let pn0 = 4;
    let pn1 = 5;
    let mut has_1_pn0 = false;
    let mut has_pn1_2 = false;
    for l in &rope.links {
        if l.n == [1, pn0] || l.n == [pn0, 1] {
            has_1_pn0 = true;
        }
        if l.n == [pn1, 2] || l.n == [2, pn1] {
            has_pn1_2 = true;
        }
    }
    assert!(has_1_pn0 && has_pn1_2, "split halves missing");
    let expected_x = rope.nodes[1].x.lerp(rope.nodes[2].x, 0.25);
    assert!((rope.nodes[pn0].x - expected_x).length() < 1e-5);
    assert!((rope.nodes[pn0].v - Vector3::new(0.0, 3.0, 0.0)).length() < 1e-5);
}

/// The plane implicit function used by the refine tests.
struct PlaneY(Real);
impl ImplicitFn for PlaneY {
    fn eval(&self, x: Vector3) -> Real {
        x.y - self.0
    }
}

/// `refine` drops bending links across the surface, inserts a node at every
/// crossing, and splits the crossed edges — mass stays conserved.
#[test]
fn refine_inserts_crossing_node() {
    let mut rope = create_rope(Vector3::new(0.0, 0.5, 0.0), Vector3::new(0.0, -0.5, 0.0), 2, 0);
    rope.generate_bending_constraints(2, None);
    let bending = rope.links.iter().filter(|l| l.bending).count();
    assert!(bending > 0, "precondition: bending links present");
    let total_mass = rope.get_total_mass();
    let links_before = rope.links.len();
    rope.refine(&PlaneY(0.0), 0.01, false);
    // Structural edges crossing y = 0 gain nodes: 3 nodes spaced evenly
    // (0.5, -0.166, -0.5 -> interior pairs (0,1)(1,2) straddle zero? node
    // values are +0.5, +0.1667, -0.1667, -0.5, so only (1,2) crosses).
    assert_eq!(rope.nodes.len(), 5, "one crossing node");
    assert!((rope.nodes[4].x.y).abs() < 0.01, "crossing sits on the plane");
    assert_eq!(rope.links.iter().filter(|l| !l.bending).count(), 4, "edge split in two");
    assert_eq!(rope.links.iter().filter(|l| l.bending).count(), 0, "crossed bending dropped");
    assert!(rope.links.len() < links_before + 4);
    assert!(
        (rope.get_total_mass() - total_mass).abs() < 1e-3,
        "mass redistributed, not created: {} vs {total_mass}",
        rope.get_total_mass()
    );
}

/// A plane that separates the rope's two halves: refining with `cut`
/// duplicates the in-band crossing node and re-routes the negative-side
/// links onto it (Bullet's `< accuracy` test is signed, so the whole
/// negative half-space qualifies) — the split halves then dangle, and
/// Bullet's orphan cleanup culls the chain end to end (refine/cut is a
/// surface-mesh tool; the chain result is pinned verbatim).
#[test]
fn refine_cut_duplicates_surface_side() {
    let mut rope = create_rope(Vector3::new(0.0, 0.5, 0.0), Vector3::new(0.0, -0.5, 0.0), 2, 0);
    let total_mass = rope.get_total_mass();
    rope.refine(&PlaneY(0.0), 0.01, true);
    // 4 original nodes + the crossing node + its in-band copy.
    assert_eq!(rope.nodes.len(), 6, "only the in-band crossing node duplicates");
    // The copy carries the crossing node's velocity, the mass splits.
    assert!((rope.nodes[5].v - rope.nodes[4].v).length() < 1e-6);
    assert!(
        (rope.get_total_mass() - total_mass).abs() < 1e-3,
        "halved masses doubled across the copies: {} vs {total_mass}",
        rope.get_total_mass()
    );
    // Every remaining chain end has rank 1 after the split, so the orphan
    // cleanup cascades through the whole chain.
    assert!(rope.links.is_empty(), "dangling links survived the cleanup");
}

/// A linear joint pins the master cluster to a rigid frame: the body hangs
/// at the anchor instead of free-falling.
#[test]
fn linear_joint_holds_cluster_at_anchor() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    // Anchor frame parked above the body so nothing collides with it.
    let anchor_body = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.1)),
        Vector3::new(0.0, 9.0, 0.0),
    ));
    let soft = world.add_soft_body(cluster_ellipsoid(Vector3::new(0.0, 5.0, 0.0)));
    world.add_soft_linear_joint(
        soft,
        LinearJointSpecs {
            position: Vector3::new(0.0, 5.0, 0.0),
            ..LinearJointSpecs::default()
        },
        Side::Rigid(anchor_body),
    );
    let anchor = Vector3::new(0.0, 5.0, 0.0);
    for _ in 0..300 {
        world.step_simulation(DT);
    }
    let com = world.soft_body(soft).get_center_of_mass();
    let d = (com - anchor).length();
    assert!(d < 0.6, "cluster fell away from the anchor: {d} ({com:?})");
    // Free fall over 5s would be ~120m; the joint is what holds it.
    assert!(d < 1.0);
}

/// The cluster-joint drift split must reach the nodes (regression for the
/// `terminate` parameter-order bug that swapped the drift impulse with the
/// anchor offset, collapsing the split channel to noise). The split only
/// reallocates the anchor correction between the velocity channel and the
/// drift channel, so the first-step node displacement must be identical
/// whether the joint runs the default `split = 1` or `split = 0` — the
/// bug dropped the drift half, shifting the `split = 1` outcome by ~8e-3.
#[test]
fn linear_joint_drift_split_reaches_nodes() {
    let run = |split: Real| {
        let mut world = PhysicsWorld::new();
        world.set_gravity(GRAVITY);
        let a = world.add_soft_body(cluster_ellipsoid(Vector3::new(0.0, 5.0, 0.0)));
        let b = world.add_soft_body(cluster_ellipsoid(Vector3::new(0.0, 4.5, 0.0)));
        world.add_soft_linear_joint(
            a,
            LinearJointSpecs {
                position: Vector3::new(0.0, 4.75, 0.0),
                tuning: JointSpecs { split, ..JointSpecs::default() },
            },
            Side::Cluster { soft: b, cluster: 0 },
        );
        world.soft_body_mut(b).add_velocity(Vector3::new(0.0, 1.0, 0.0));
        world.step_simulation(DT);
        world.soft_body(a).get_center_of_mass().y
    };
    let drift = run(1.0);
    let no_drift = run(0.0);
    let delta = (drift - no_drift).abs();
    // The correct split applies the drift half (~8e-3, minus the share the
    // position pass re-absorbs -> ~6e-3); the parameter-order bug left only
    // a ~2e-3 residue.
    assert!(delta > 0.0045, "drift split had no effect on the nodes: {delta}");
}

/// An angular joint realigns a rotated cluster frame back onto the axis
/// (the default control passes the speed through, leaving only the
/// misalignment drift).
#[test]
fn angular_joint_aligns_axes() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let anchor_body = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.1)),
        Vector3::ZERO,
    ));
    let soft = world.add_soft_body(cluster_ellipsoid(Vector3::ZERO));
    world.add_soft_angular_joint(
        soft,
        AngularJointSpecs {
            axis: Vector3::Y,
            ..AngularJointSpecs::default()
        },
        Side::Rigid(anchor_body),
    );
    // Tilt the body 0.5 rad about X (a rigid teleport: x/q rotate together,
    // so the PBD pass does not fight it). The cluster frames catch up on
    // the first `predict_motion`.
    let tilt = quat_from_axis_angle(Vector3::X, 0.5);
    world.soft_body_mut(soft).rotate(tilt);
    let y_axis_err = |w: &PhysicsWorld| {
        w.soft_body(soft).clusters[0].framerot.y_axis.cross(Vector3::Y).length()
    };
    world.step_simulation(DT);
    let start_err = y_axis_err(&world);
    assert!(start_err > 0.4, "precondition: axes start apart ({start_err})");
    for _ in 0..600 {
        world.step_simulation(DT);
    }
    let end_err = y_axis_err(&world);
    assert!(end_err < 0.15, "axes did not realign: {end_err}");
}

/// The SoftDemo motor control spins the cluster toward its goal speed.
#[test]
fn angular_joint_motor_spins_cluster() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let anchor_body = world.add_body(RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.1)),
        Vector3::ZERO,
    ));
    let soft = world.add_soft_body(cluster_ellipsoid(Vector3::ZERO));
    world.add_soft_angular_joint(
        soft,
        AngularJointSpecs {
            axis: Vector3::Y,
            control: Box::new(MotorControl { goal: 2.0, maxtorque: 0.5 }),
            ..AngularJointSpecs::default()
        },
        Side::Rigid(anchor_body),
    );
    for _ in 0..480 {
        world.step_simulation(DT);
    }
    let av = world.soft_body(soft).clusters[0].av.y;
    assert!(
        av > 1.0 && av < 3.0,
        "motor did not spin the cluster to the goal: {av}"
    );
}

/// `apply_torque_impulse` changes the angular velocity through the world
/// inertia (the rigid side of the angular joints).
#[test]
fn torque_impulse_updates_angular_velocity() {
    let mut body = RigidBody::new_dynamic(
        CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::ZERO,
        2.0,
    );
    let iw = body.inv_inertia_world();
    body.apply_torque_impulse(Vector3::new(0.0, 0.0, 3.0));
    let expect = iw * Vector3::new(0.0, 0.0, 3.0);
    assert!((body.angular_velocity - expect).length() < 1e-9);
    // Static bodies ignore torque impulses.
    let mut stat = RigidBody::new_static(
        CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::ZERO,
    );
    stat.apply_torque_impulse(Vector3::ONE);
    assert_eq!(stat.angular_velocity, Vector3::ZERO);
}

// ---------------------------------------------------------------------------
// Phase F: soft-vs-multibody coupling.
// ---------------------------------------------------------------------------

use crate::multibody::MultiBody;
use bullet_math::Quaternion;

/// One fixed-base pendulum link with a sphere collider: pivot at
/// `(0, pivot_y, 0)`, the link COM hanging 1 m below along -Y (Bullet's
/// demo geometry — `this_pivot_to_this_com_offset = (0, -1, 0)`), so the
/// ball rests at `(0, pivot_y - 1, 0)`.
fn pendulum_mb(pivot_y: Real, can_sleep: bool) -> MultiBody {
    let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, can_sleep);
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    mb.set_link_collider(0, CollisionShape::sphere(0.7), 0.5, 0.0);
    mb.base_pos = Vector3::new(0.0, pivot_y, 0.0);
    mb
}

/// A rope draped across the pendulum ball is held on the contact band and
/// its weight swings the link — the SDF_RS link contacts support the rope
/// while the jacobian writeback pushes the multibody (phase F).
///
/// The rope starts inside the band (zero impact) and is offset to one side
/// so its weight produces a swing torque. Verified against the real library
/// (probe scene `rope_ball_tunnel`): Bullet's classic SDF_RS holds the
/// middle nodes on the band for ~30 frames, then the rope slides off the
/// side and falls — our port matches frame-for-frame (~3-frame phase slip),
/// so the assertions only cover the supported window and the swing, never
/// a permanent rest (a short rope DROPPED onto a sphere tunnels through in
/// both engines — the classic solve only partially corrects each frame).
#[test]
fn rope_rests_on_mb_ball() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mbi = world.add_multibody(pendulum_mb(5.7, false));
    // Ball top at y = 5.4, band up to 5.65: the rope starts inside the band,
    // spanning the ball and overhanging it to the right (+0.5 offset).
    let rope = world.add_soft_body(create_rope(
        Vector3::new(-0.4, 5.55, 0.0),
        Vector3::new(1.6, 5.55, 0.0),
        16,
        0,
    ));
    world.soft_body_mut(rope).cfg.piterations = 4;
    // Nodes 3..=9 sit above the ball (x in [-0.025, 0.775]). While the band
    // holds them the middle stays above the surface, and even once the rope
    // slides off the side its nodes stay outside the sphere (the phase slip
    // vs the probe is frames, not penetration).
    let mut min_center_dist = Real::INFINITY;
    let mut supported_max: Real = 0.0;
    let mut max_joint_speed: Real = 0.0;
    for step in 0..300 {
        world.step_simulation(DT);
        let sb = world.soft_body(rope);
        if step < 30 {
            for n in &sb.nodes[3..=9] {
                min_center_dist = min_center_dist.min((n.x - Vector3::new(0.0, 4.7, 0.0)).length());
            }
            if step < 20 {
                supported_max = supported_max.max(sb.nodes[3..=9].iter().map(|n| n.x.y).fold(0.0, Real::max));
            }
        }
        let qd = world.multibody(mbi).velocity_vector()[6];
        max_joint_speed = max_joint_speed.max(qd.abs());
    }
    assert!(
        min_center_dist > 0.65,
        "the rope went through the ball: {min_center_dist}"
    );
    assert!(
        supported_max > 5.5,
        "the band never held the rope above the surface: {supported_max}"
    );
    // The rope's weight dragged the ball to the right (probe: COM x 0.6 ->
    // 1.0+ in both engines — a symmetric rope would only press along the
    // swing axis and leave the joint still).
    let com = world.soft_body(rope).get_center_of_mass();
    assert!(
        com.x > 1.0,
        "the rope never dragged the ball aside: com.x = {}",
        com.x
    );
    assert!(
        max_joint_speed > 0.02,
        "the link never felt the rope: {max_joint_speed}"
    );
    let sb = world.soft_body(rope);
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()), "rope blew up");
    assert!(
        world.multibody(mbi).velocity_vector()[6].is_finite(),
        "the link velocity blew up"
    );
}

/// A swinging pendulum ball sweeps a hanging rope aside — the contact
/// pushes nodes through the mb velocity term while the rope damps the
/// swing (the writeback side of the coupling).
#[test]
fn mb_pendulum_sweeps_hanging_rope() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let mut mb = pendulum_mb(6.0, false);
    mb.set_joint_vel_multidof(0, &[2.0]);
    let mbi = world.add_multibody(mb);
    // A rope pinned just where the swinging ball passes (ball center
    // (0, 5, 0), radius 0.7 — the swing reaches x ≈ 0.9 at ~0.5 rad).
    let rope = world.add_soft_body(create_rope(
        Vector3::new(0.8, 5.0, 0.0),
        Vector3::new(0.8, 3.8, 0.0),
        6,
        ROPE_FIX_START,
    ));
    world.soft_body_mut(rope).cfg.piterations = 4;
    let mut max_x: Real = 0.0;
    for _ in 0..300 {
        world.step_simulation(DT);
        let sb = world.soft_body(rope);
        for n in &sb.nodes {
            max_x = max_x.max(n.x.x);
        }
    }
    // The free nodes hang at x ≈ 0.8; the swinging ball must have pushed
    // some past 1.0.
    assert!(max_x > 1.0, "the swing never moved the rope: {max_x}");
    let sb = world.soft_body(rope);
    assert!(sb.nodes.iter().all(|n| n.x.is_finite()), "rope blew up");
    assert!(
        world.multibody(mbi).velocity_vector()[6].is_finite(),
        "the link velocity blew up"
    );
}

/// A sleeping multibody under a newly-added overlapping soft body wakes
/// up: the island pass unions the pair and the never-sleeping soft body
/// pins the island awake.
#[test]
fn sleeping_mb_wakes_under_soft_contact() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mb = pendulum_mb(5.7, true);
    let mbi = world.add_multibody(mb);
    for _ in 0..150 {
        world.step_simulation(DT);
    }
    assert!(!world.multibody(mbi).is_awake(), "precondition: asleep");
    // Drop a rope node region right onto the ball.
    let rope = world.add_soft_body(create_rope(
        Vector3::new(0.0, 5.4, 0.0),
        Vector3::new(0.6, 5.4, 0.0),
        4,
        0,
    ));
    world.soft_body_mut(rope).cfg.piterations = 4;
    world.step_simulation(DT);
    assert!(world.multibody(mbi).is_awake(), "soft contact did not wake the mb");
}

/// A `CL_RS`-mode soft body does not interact with multibody colliders
/// (Bullet's cluster path against a link collider degenerates to a
/// massless one-sided push — unsupported, the rope free-falls through).
#[test]
fn cl_rs_mode_ignores_mb_collider() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    let _mbi = world.add_multibody(pendulum_mb(5.7, false));
    let rope = world.add_soft_body(create_rope(
        Vector3::new(0.3, 6.4, 0.0),
        Vector3::new(0.9, 6.4, 0.0),
        4,
        0,
    ));
    {
        let sb = world.soft_body_mut(rope);
        sb.cfg.collisions = crate::softbody::SoftCollision::CL_RS;
        sb.generate_clusters(4, 32);
    }
    let mut min_y: Real = 10.0;
    for _ in 0..400 {
        world.step_simulation(DT);
        for n in &world.soft_body(rope).nodes {
            min_y = min_y.min(n.x.y);
        }
    }
    assert!(
        min_y < 4.6,
        "CL_RS mode should not hold the rope on the mb ball: {min_y}"
    );
}
