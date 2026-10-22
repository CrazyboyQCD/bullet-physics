//! Inverse-dynamics probe twin: canonical trees run through the Rust port
//! for bit-level comparison against the real BulletInverseDynamics library
//! (tools/inverse_dynamics_probe.cpp, built by
//! tools/build_inverse_dynamics_probe.py).
//!
//! Output protocol (identical to the C++ probe, one line per (scene,
//! state) per group): "S1" is the inverse-dynamics group (generalized
//! forces + world getters with gravity folded into the accelerations),
//! "S2" the calculateKinematics group (gravity-free accelerations), "M"
//! the joint-space mass matrix (num_dofs^2, row-major), "J" the
//! world-frame jacobians (per body: dot(J_T)u(3), dot(J_R)u(3),
//! jac_T(3xN), jac_R(3xN), row-major). Scenes 0-3 are canonical trees;
//! scenes 4-6 are seeded random trees (CrtRand replicating the MSVC
//! rand() sequence + the shared draw protocol — keep the C++ probe in
//! lockstep).
//! Columns after the label for S1/S2:
//!   scene state | tau[6] (padded) | 4 bodies x 30 slots:
//!   origin(3) com(3) transform(9) ang_vel(3) lin_vel(3) lin_vel_com(3)
//!   ang_acc(3) lin_acc(3)
//! printed at C %.17g precision so a plain byte compare (cmp) is the
//! strictest gate.
//!
//! Run:
//!     cargo run -p bullet-physics --example id_probe --features inverse-dynamics
//!     cargo run -p bullet-physics --example id_probe --features inverse-dynamics,f64
//! The scene body orders and parameters are transcribed 1:1 from the C++
//! probe — keep the two files in lockstep.

use bullet_physics::inverse_dynamics::builder::{JointType, TreeBuilder};
use bullet_physics::inverse_dynamics::math;
use bullet_physics::inverse_dynamics::{Matrix3, MultiBodyTree, Real, Vector3};
use bullet_physics::math::CrtRand;

/// C `%.17g`: 17 significant digits, trailing zeros stripped, fixed
/// notation when the decimal exponent is in [-4, 17), scientific
/// (`e+NN`) otherwise.
fn fmt_g17(x: Real) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    if x < 0.0 {
        return format!("-{}", fmt_g17(-x));
    }
    let sci = format!("{:.16e}", x);
    let (mantissa, exp_str) = sci.split_once('e').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    if (-4..17).contains(&exp) {
        let mut s = format!("{:.*}", (16 - exp) as usize, x);
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        s
    } else {
        let mut m = mantissa.to_string();
        while m.ends_with('0') {
            m.pop();
        }
        if m.ends_with('.') {
            m.pop();
        }
        if exp >= 0 {
            format!("{}e+{:02}", m, exp)
        } else {
            format!("{}e-{:02}", m, -exp)
        }
    }
}

fn print_vec(v: Vector3) {
    print!(" {} {} {}", fmt_g17(v.x), fmt_g17(v.y), fmt_g17(v.z));
}

fn print_mat(m: &Matrix3) {
    // glam stores columns; emit the mathematical matrix row-major like the
    // C++ (r, c) loop.
    let cols = m.to_cols_array_2d();
    let rows = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    for row in &rows {
        for v in row {
            print!(" {}", fmt_g17(*v));
        }
    }
}

const MAX_BODIES: usize = 4;
/// tau display floor for the S1/S2 protocol; the per-state arrays are sized
/// by the call sites (canonical scenes ≤ 6, random trees up to 6×6).
const MAX_DOFS: usize = 6;
const BODY_SLOTS: usize = 30;
const MAX_RANDOM_DOFS: usize = 36;

fn print_tree_state(tree: &MultiBodyTree, scene: usize, state: usize, group: &str,
                    tau: &[Real], num_bodies: usize) {
    print!("{} {} {}", group, scene, state);
    let tau_slots = MAX_DOFS.max(tau.len());
    for i in 0..tau_slots {
        let v = if i < tau.len() { tau[i] } else { 0.0 };
        print!(" {}", fmt_g17(v));
    }
    let body_slots = MAX_BODIES.max(num_bodies);
    for b in 0..body_slots {
        if b < num_bodies {
            print_vec(tree.body_origin(b).unwrap());
            print_vec(tree.body_com(b).unwrap());
            print_mat(&tree.body_transform(b).unwrap());
            print_vec(tree.body_angular_velocity(b).unwrap());
            print_vec(tree.body_linear_velocity(b).unwrap());
            print_vec(tree.body_linear_velocity_com(b).unwrap());
            print_vec(tree.body_angular_acceleration(b).unwrap());
            print_vec(tree.body_linear_acceleration(b).unwrap());
        } else {
            for _ in 0..BODY_SLOTS {
                print!(" {}", fmt_g17(0.0));
            }
        }
    }
    println!();
}

fn diagonal(xx: Real, yy: Real, zz: Real) -> Matrix3 {
    Matrix3::from_cols(
        Vector3::new(xx, 0.0, 0.0),
        Vector3::new(0.0, yy, 0.0),
        Vector3::new(0.0, 0.0, zz),
    )
}

#[allow(clippy::too_many_arguments)] // mirrors the C++ probe's add helper
fn add(tree: &mut TreeBuilder, body: usize, parent: isize, jt: JointType, r: Vector3,
       t: Matrix3, axis: Vector3, mass: Real, com: Vector3, inertia: Matrix3) {
    tree.add_body(body, parent, jt, r, t, axis, mass, com, inertia).unwrap();
}

/// Seeded random-tree generator, the twin of the C++ probe's `draw` +
/// `build_random_tree`: every value is `lo + (rand() % 2001) / 2000.0 *
/// width`, consumed in a fixed order; CrtRand replicates the MSVC rand()
/// sequence. Extras/IDRandomUtil is in the zlib-excluded zone — this
/// generator is written natively for the probe.
fn draw(rand: &mut bullet_math::CrtRand, lo: f64, width: f64) -> Real {
    let r = rand.next_rand();
    // The C++ draw() evaluates in double with exact double arguments
    // (idScalar literals widen losslessly) and narrows only at the consuming
    // v3()/parameter/array store — mirror that exactly or f32 builds round
    // differently at each step.
    (lo + (r % 2001) as f64 / 2000.0 * width) as Real
}

fn v3(x: Real, y: Real, z: Real) -> Vector3 {
    Vector3::new(x, y, z)
}

fn run_scene<const N: usize>(scene: usize, tree: &mut MultiBodyTree, num_dofs: usize,
                             qs: &[[Real; N]; 3], us: &[[Real; N]; 3],
                             dot_us: &[[Real; N]; 3]) {
    let num_bodies = tree.num_bodies();
    for s in 0..3 {
        let q = &qs[s][..num_dofs];
        let u = &us[s][..num_dofs];
        let dot_u = &dot_us[s][..num_dofs];
        let mut tau = vec![0.0; num_dofs];
        tree.calculate_inverse_dynamics(q, u, dot_u, &mut tau).unwrap();
        print_tree_state(tree, scene, s, "S1", &tau, num_bodies);
        tree.calculate_kinematics(q, u, dot_u).unwrap();
        print_tree_state(tree, scene, s, "S2", &tau, num_bodies);
        // Position+velocity kinematics are current from calculate_kinematics;
        // the mass-matrix relative refresh is idempotent at the same q.
        let n2 = num_dofs * num_dofs;
        let mut mm = vec![0.0; n2];
        tree.calculate_mass_matrix(q, &mut mm).unwrap();
        print!("M {} {}", scene, s);
        for v in &mm {
            print!(" {}", fmt_g17(*v));
        }
        println!();
        tree.calculate_jacobians(q, u).unwrap();
        print!("J {} {}", scene, s);
        for b in 0..num_bodies {
            print_vec(tree.body_dot_jacobian_trans_u(b).unwrap());
            print_vec(tree.body_dot_jacobian_rot_u(b).unwrap());
            let mut jac = vec![0.0; 3 * num_dofs];
            tree.body_jacobian_trans(b, &mut jac).unwrap();
            // The flat layout is [dof_column][component]; the C++ probe
            // prints btMatrixX (r, c) row-major — emit the same order.
            let (cols, _) = jac.as_chunks::<3>();
            for r in 0..3 {
                for col in cols.iter() {
                    print!(" {}", fmt_g17(col[r]));
                }
            }
            tree.body_jacobian_rot(b, &mut jac).unwrap();
            let (cols, _) = jac.as_chunks::<3>();
            for r in 0..3 {
                for col in cols.iter() {
                    print!(" {}", fmt_g17(col[r]));
                }
            }
        }
        println!();
    }
}

fn main() {
    // scene 0: arm2r — planar two-revolute arm, default gravity.
    {
        let mut b = TreeBuilder::new();
        add(&mut b, 0, -1, JointType::Fixed, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 0.0, v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
        add(&mut b, 1, 0, JointType::Revolute, v3(1.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 1.0), 1.0, v3(0.5, 0.0, 0.0), diagonal(0.2, 0.15, 0.1));
        add(&mut b, 2, 1, JointType::Revolute, v3(1.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 1.0), 0.8, v3(0.4, 0.0, 0.0), diagonal(0.1, 0.08, 0.06));
        let mut tree = b.finalize().unwrap();
        tree.set_gravity_in_world_frame(v3(0.0, 0.0, -9.8));
        let qs = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.7, -0.4, 0.0, 0.0, 0.0, 0.0],
                  [-1.1, 0.9, 0.0, 0.0, 0.0, 0.0]];
        let us = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.5, -0.75, 0.0, 0.0, 0.0, 0.0],
                  [0.3, 2.0, 0.0, 0.0, 0.0, 0.0]];
        let du = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.5, 0.25, 0.0, 0.0, 0.0, 0.0],
                  [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let nd = tree.num_dofs();
        run_scene(0, &mut tree, nd, &qs, &us, &du);
    }
    // scene 1: mixed1 — revolute + prismatic + fixed chain, non-identity
    // reference transforms, default gravity.
    {
        let mut b = TreeBuilder::new();
        add(&mut b, 0, -1, JointType::Fixed, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 0.0, v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
        add(&mut b, 1, 0, JointType::Revolute, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 1.0, 0.0), 1.2, v3(0.0, 0.3, 0.0), diagonal(0.12, 0.1, 0.09));
        add(&mut b, 2, 1, JointType::Prismatic, v3(0.0, 0.0, 0.3),
            math::transform_y(0.5) * math::transform_x(-0.25), v3(1.0, 0.0, 0.0), 0.6,
            v3(0.2, 0.0, 0.0), diagonal(0.05, 0.05, 0.04));
        add(&mut b, 3, 2, JointType::Fixed, v3(0.5, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 0.5, v3(0.0, 0.0, 0.1), diagonal(0.02, 0.02, 0.02));
        let mut tree = b.finalize().unwrap();
        tree.set_gravity_in_world_frame(v3(0.0, 0.0, -9.8));
        let qs = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, 0.6, 0.0, 0.0, 0.0, 0.0],
                  [-0.5, 1.2, 0.0, 0.0, 0.0, 0.0]];
        let us = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.8, -0.4, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let du = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0]];
        let nd = tree.num_dofs();
        run_scene(1, &mut tree, nd, &qs, &us, &du);
    }
    // scene 2: floating_free — single floating base, tilted gravity.
    {
        let mut b = TreeBuilder::new();
        add(&mut b, 0, -1, JointType::Floating, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 2.0, v3(0.1, 0.2, 0.3), diagonal(0.3, 0.25, 0.2));
        let mut tree = b.finalize().unwrap();
        tree.set_gravity_in_world_frame(v3(0.5, -9.81, 0.2));
        let qs = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.4, -0.2, 0.9, 1.0, 2.0, 3.0],
                  [-0.7, 1.3, -2.1, -0.5, 0.25, 1.75]];
        let us = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.6, 0.1, -0.3, 0.4, -0.8, 0.2],
                  [-1.0, 0.5, 0.9, 0.0, 1.5, -0.6]];
        let du = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.1, 0.2, 0.3, -0.4, 0.5, -0.6],
                  [0.0, -1.0, 0.25, 1.0, 0.0, 0.75]];
        let nd = tree.num_dofs();
        run_scene(2, &mut tree, nd, &qs, &us, &du);
    }
    // scene 3: spherical_arm — spherical joints with a fixed child, default
    // gravity.
    {
        let mut b = TreeBuilder::new();
        add(&mut b, 0, -1, JointType::Fixed, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 0.0, v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
        add(&mut b, 1, 0, JointType::Spherical, v3(0.0, 0.0, 1.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 1.0, v3(0.0, 0.0, -0.5), diagonal(0.1, 0.1, 0.1));
        add(&mut b, 2, 1, JointType::Fixed, v3(0.0, 0.0, -0.6),
            math::transform_z(0.75), v3(0.0, 0.0, 0.0), 0.3, v3(0.05, 0.0, 0.0),
            diagonal(0.01, 0.01, 0.01));
        let mut tree = b.finalize().unwrap();
        tree.set_gravity_in_world_frame(v3(0.0, 0.0, -9.8));
        let qs = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.2, 0.5, -0.4, 0.0, 0.0, 0.0],
                  [-0.9, 0.15, 1.1, 0.0, 0.0, 0.0]];
        let us = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, -0.6, 0.9, 0.0, 0.0, 0.0],
                  [0.0, 1.2, -0.4, 0.0, 0.0, 0.0]];
        let du = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.2, 0.4, 0.1, 0.0, 0.0, 0.0],
                  [0.8, 0.0, -0.3, 0.0, 0.0, 0.0]];
        let nd = tree.num_dofs();
        run_scene(3, &mut tree, nd, &qs, &us, &du);
    }
    // scenes 4-6: seeded random trees, one per seed. The draw order and the
    // lo/width pairs are part of the protocol — keep them in lockstep with
    // the C++ probe.
    for t in 0..3u32 {
        let mut rand = CrtRand::seeded(t + 1);
        let mut b = TreeBuilder::new();
        let nb = 2 + (rand.next_rand() % 5) as usize;
        for i in 0..nb {
            let parent: isize = if i == 0 { -1 } else { (rand.next_rand() % i as i32) as isize };
            let jt = match rand.next_rand() % 5 {
                0 => JointType::Fixed,
                1 => JointType::Revolute,
                2 => JointType::Prismatic,
                3 => JointType::Floating,
                _ => JointType::Spherical,
            };
            let r = v3(draw(&mut rand, -0.4, 0.8), draw(&mut rand, -0.4, 0.8),
                draw(&mut rand, -0.4, 0.8));
            let a0 = draw(&mut rand, -0.4, 0.8);
            let a1 = draw(&mut rand, -0.4, 0.8);
            let a2 = draw(&mut rand, -0.4, 0.8);
            let t_ref =
                math::transform_z(a2) * math::transform_y(a1) * math::transform_x(a0);
            let axis = v3(draw(&mut rand, 0.05, 0.95), draw(&mut rand, 0.05, 0.95),
                draw(&mut rand, 0.05, 0.95));
            let mass = draw(&mut rand, 0.1, 1.9);
            let com = v3(draw(&mut rand, -0.5, 1.0), draw(&mut rand, -0.5, 1.0),
                draw(&mut rand, -0.5, 1.0));
            // Enforce the inertia triangle inequality on the largest entry
            // (lowest-index tie-break), so the add_body gate accepts every
            // tree. Upstream keeps the draws in double until the diagonal()
            // construction.
            let mut m = [draw(&mut rand, 0.05, 0.95), draw(&mut rand, 0.05, 0.95),
                draw(&mut rand, 0.05, 0.95)];
            let sum = m[0] + m[1] + m[2];
            let mut mx = 0usize;
            if m[1] > m[mx] {
                mx = 1;
            }
            if m[2] > m[mx] {
                mx = 2;
            }
            if m[mx] > sum - m[mx] {
                m[mx] = (sum - m[mx]) * 0.9;
            }
            let inertia = diagonal(m[0] as Real, m[1] as Real, m[2] as Real);
            add(&mut b, i, parent, jt, r, t_ref, axis, mass, com, inertia);
        }
        let mut tree = b.finalize().unwrap();
        tree.set_gravity_in_world_frame(v3(0.0, 0.0, -9.8));
        let nd = tree.num_dofs();
        let mut qs = [[0.0 as Real; MAX_RANDOM_DOFS]; 3];
        let mut us = [[0.0 as Real; MAX_RANDOM_DOFS]; 3];
        let mut du = [[0.0 as Real; MAX_RANDOM_DOFS]; 3];
        for s in 0..3 {
            for d in 0..nd {
                qs[s][d] = draw(&mut rand, -1.5, 3.0);
                us[s][d] = draw(&mut rand, -1.5, 3.0);
                du[s][d] = draw(&mut rand, -1.5, 3.0);
            }
        }
        run_scene(4 + t as usize, &mut tree, nd, &qs, &us, &du);
    }
}
