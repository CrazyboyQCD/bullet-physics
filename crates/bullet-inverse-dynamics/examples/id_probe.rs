//! Inverse-dynamics probe twin: canonical trees run through the Rust port
//! for bit-level comparison against the real BulletInverseDynamics library
//! (tools/inverse_dynamics_probe.cpp, built by
//! tools/build_inverse_dynamics_probe.py).
//!
//! Output protocol (identical to the C++ probe, one line per (scene,
//! state) per group): "S1" is the inverse-dynamics group (generalized
//! forces + world getters with gravity folded into the accelerations),
//! "S2" the calculateKinematics group (gravity-free accelerations).
//! Columns after the label:
//!   scene state | tau[6] (padded) | 4 bodies x 30 slots:
//!   origin(3) com(3) transform(9) ang_vel(3) lin_vel(3) lin_vel_com(3)
//!   ang_acc(3) lin_acc(3)
//! printed at C %.17g precision so a plain byte compare (cmp) is the
//! strictest gate.
//!
//! Run:
//!     cargo run -p bullet-inverse-dynamics --example id_probe
//!     cargo run -p bullet-inverse-dynamics --example id_probe --features f64
//! The scene body orders and parameters are transcribed 1:1 from the C++
//! probe — keep the two files in lockstep.

use bullet_inverse_dynamics::builder::{JointType, TreeBuilder};
use bullet_inverse_dynamics::math;
use bullet_inverse_dynamics::{Matrix3, MultiBodyTree, Real, Vector3};

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
const MAX_DOFS: usize = 6;
const BODY_SLOTS: usize = 30;

fn print_tree_state(tree: &MultiBodyTree, scene: usize, state: usize, group: &str,
                    tau: &[Real], num_bodies: usize) {
    print!("{} {} {}", group, scene, state);
    for i in 0..MAX_DOFS {
        let v = if i < tau.len() { tau[i] } else { 0.0 };
        print!(" {}", fmt_g17(v));
    }
    for b in 0..MAX_BODIES {
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

fn v3(x: Real, y: Real, z: Real) -> Vector3 {
    Vector3::new(x, y, z)
}

fn run_scene(scene: usize, tree: &mut MultiBodyTree, num_dofs: usize,
             qs: &[[Real; MAX_DOFS]; 3], us: &[[Real; MAX_DOFS]; 3],
             dot_us: &[[Real; MAX_DOFS]; 3]) {
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
}
