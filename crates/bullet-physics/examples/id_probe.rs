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
//!     ... -- --bench   perf mode (docs/perf/inverse-dynamics.md): the
//!                      chain_16 / chain_64 / chain_256 / random_64 scenes x
//!                      {kin, id, mass, jac}, ns/call medians over interleaved
//!                      rounds plus per-call allocation counts (the zero-alloc
//!                      contract is asserted hard on this side) and the
//!                      finalized-tree live bytes.
//! The scene body orders and parameters are transcribed 1:1 from the C++
//! probe — keep the two files in lockstep.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

use bullet_physics::inverse_dynamics::builder::{JointType, TreeBuilder};
use bullet_physics::inverse_dynamics::math;
use bullet_physics::inverse_dynamics::{Matrix3, MultiBodyTree, Real, Vector3};
use bullet_physics::math::CrtRand;

/// Allocator accounting for the `--bench` run mode: a pass-through wrapper of
/// `System` tracking allocation counts and live requested bytes while
/// `TRACKING` is set (the twin of the C++ probe's operator-new hook; both
/// sides report requested bytes).
struct CountingAlloc;

static TRACKING: AtomicBool = AtomicBool::new(false);
static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACKING.load(Ordering::Relaxed) {
            LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            TOTAL_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING.load(Ordering::Relaxed) {
            // Clamp: chunks allocated before tracking was switched on free
            // late and would otherwise drag the total below zero.
            let old = LIVE_BYTES.fetch_sub(layout.size() as i64, Ordering::Relaxed);
            if old < layout.size() as i64 {
                LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
            }
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

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

/// One drawn body: the shared per-body draw sequence of the random-tree
/// builder and the bench chains (identical order on both sides; the parent
/// and joint type are the caller's, so the chains skip their draws).
fn add_drawn_body(b: &mut TreeBuilder, body: usize, parent: isize, jt: JointType,
                  rand: &mut CrtRand) {
    let r = v3(draw(rand, -0.4, 0.8), draw(rand, -0.4, 0.8), draw(rand, -0.4, 0.8));
    let a0 = draw(rand, -0.4, 0.8);
    let a1 = draw(rand, -0.4, 0.8);
    let a2 = draw(rand, -0.4, 0.8);
    let t_ref = math::transform_z(a2) * math::transform_y(a1) * math::transform_x(a0);
    let axis = v3(draw(rand, 0.05, 0.95), draw(rand, 0.05, 0.95), draw(rand, 0.05, 0.95));
    let mass = draw(rand, 0.1, 1.9);
    let com = v3(draw(rand, -0.5, 1.0), draw(rand, -0.5, 1.0), draw(rand, -0.5, 1.0));
    // Enforce the inertia triangle inequality on the largest entry
    // (lowest-index tie-break), so the add_body gate accepts every tree.
    // Upstream keeps the draws in double until the diagonal() construction.
    let mut m = [draw(rand, 0.05, 0.95), draw(rand, 0.05, 0.95), draw(rand, 0.05, 0.95)];
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
    add(b, body, parent, jt, r, t_ref, axis, mass, com, inertia);
}

fn build_random_tree_n(b: &mut TreeBuilder, nb: usize, rand: &mut CrtRand) {
    for i in 0..nb {
        let parent: isize = if i == 0 { -1 } else { (rand.next_rand() % i as i32) as isize };
        let jt = match rand.next_rand() % 5 {
            0 => JointType::Fixed,
            1 => JointType::Revolute,
            2 => JointType::Prismatic,
            3 => JointType::Floating,
            _ => JointType::Spherical,
        };
        add_drawn_body(b, i, parent, jt, rand);
    }
}

/// Fixed-root serial chain: body 0 FIXED, bodies 1..nb-1 drawn revolute links.
fn build_chain(b: &mut TreeBuilder, nb: usize, rand: &mut CrtRand) {
    add(b, 0, -1, JointType::Fixed, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
        v3(0.0, 0.0, 0.0), 0.0, v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
    for i in 1..nb {
        add_drawn_body(b, i, i as isize - 1, JointType::Revolute, rand);
    }
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
    if std::env::args().nth(1).as_deref() == Some("--bench") {
        run_bench();
        return;
    }
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
        build_random_tree_n(&mut b, nb, &mut rand);
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

// --- bench mode --------------------------------------------------------------
// The twin of the C++ probe's --bench (same scenes, seeds, draw stream,
// calibration and protocol). The zero-alloc contract is asserted hard on this
// side: a nonzero per-call allocation count aborts the run with exit code 1.

struct BenchScene {
    name: &'static str,
    seed: u32,
    bodies: usize,
    random: bool,
}

const BENCH_SCENES: [BenchScene; 4] = [
    BenchScene { name: "chain_16", seed: 2101, bodies: 16, random: false },
    BenchScene { name: "chain_64", seed: 2102, bodies: 64, random: false },
    BenchScene { name: "chain_256", seed: 2103, bodies: 256, random: false },
    BenchScene { name: "random_64", seed: 2104, bodies: 64, random: true },
];

const ALGO_NAMES: [&str; 4] = ["kin", "id", "mass", "jac"];

fn call_algo(a: usize, tree: &mut MultiBodyTree, q: &[Real], u: &[Real], du: &[Real],
             tau: &mut [Real], mm: &mut [Real]) -> bool {
    match a {
        0 => tree.calculate_kinematics(q, u, du).is_ok(),
        1 => tree.calculate_inverse_dynamics(q, u, du, tau).is_ok(),
        2 => tree.calculate_mass_matrix(q, mm).is_ok(),
        3 => tree.calculate_jacobians(q, u).is_ok(),
        _ => false,
    }
}

fn bench_scene(s: &BenchScene) {
    TRACKING.store(true, Ordering::Relaxed);
    let live0 = LIVE_BYTES.load(Ordering::Relaxed);
    let mut rand = CrtRand::seeded(s.seed);
    let mut b = TreeBuilder::new();
    if s.random {
        build_random_tree_n(&mut b, s.bodies, &mut rand);
    } else {
        build_chain(&mut b, s.bodies, &mut rand);
    }
    let mut tree = b.finalize().unwrap();
    tree.set_gravity_in_world_frame(v3(0.0, 0.0, -9.8));
    let tree_live = LIVE_BYTES.load(Ordering::Relaxed) - live0;
    let nd = tree.num_dofs();

    // One drawn state, the same draw(-1.5, 3.0) stream as the parity scenes.
    let mut q = vec![0.0 as Real; nd];
    let mut u = vec![0.0 as Real; nd];
    let mut du = vec![0.0 as Real; nd];
    for d in 0..nd {
        q[d] = draw(&mut rand, -1.5, 3.0);
        u[d] = draw(&mut rand, -1.5, 3.0);
        du[d] = draw(&mut rand, -1.5, 3.0);
    }
    let mut tau = vec![0.0 as Real; nd];
    let mut mm = vec![0.0 as Real; nd * nd];

    println!("bench_mem {} tree_live_bytes {}", s.name, tree_live);

    const ROUNDS: usize = 5;
    let mut ns = [[0.0f64; ROUNDS]; 4];
    let mut calls = [0u64; 4];
    for (a, calls_a) in calls.iter_mut().enumerate() {
        for _ in 0..5 {
            assert!(call_algo(a, &mut tree, &q, &u, &du, &mut tau, &mut mm));
        }
        // Auto-calibrated round size: ~50 ms of calls, clamped.
        let mut one = f64::INFINITY;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            assert!(call_algo(a, &mut tree, &q, &u, &du, &mut tau, &mut mm));
            one = one.min(t0.elapsed().as_nanos() as f64);
        }
        let c = (50.0e6 / one.max(1.0)) as u64;
        *calls_a = c.clamp(1, 200_000);
    }
    // Interleaved rounds: the round loop is outer, the algorithm loop inner.
    for r in 0..ROUNDS {
        for (a, row) in ns.iter_mut().enumerate() {
            let t0 = std::time::Instant::now();
            for _ in 0..calls[a] {
                assert!(call_algo(a, &mut tree, &q, &u, &du, &mut tau, &mut mm));
            }
            row[r] = t0.elapsed().as_nanos() as f64 / calls[a] as f64;
        }
    }
    for (a, name) in ALGO_NAMES.iter().enumerate() {
        let mut sorted = ns[a];
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mean: f64 = sorted.iter().sum::<f64>() / ROUNDS as f64;
        println!("bench {} {} median_ns {:.1} min_ns {:.1} mean_ns {:.1} rounds {} calls_per_round {}",
            s.name, name, sorted[ROUNDS / 2], sorted[0], mean, ROUNDS, calls[a]);
    }
    // Per-call allocation counts of one tracked call (monotonic counters).
    for (a, name) in ALGO_NAMES.iter().enumerate() {
        let c0 = ALLOCS.load(Ordering::Relaxed);
        let b0 = TOTAL_BYTES.load(Ordering::Relaxed);
        assert!(call_algo(a, &mut tree, &q, &u, &du, &mut tau, &mut mm));
        let allocs = ALLOCS.load(Ordering::Relaxed) - c0;
        let bytes = TOTAL_BYTES.load(Ordering::Relaxed) - b0;
        if allocs != 0 || bytes != 0 {
            eprintln!("ZERO-ALLOC VIOLATION {} {} allocs {} bytes {}",
                s.name, name, allocs, bytes);
            std::process::exit(1);
        }
        println!("bench_alloc {} {} allocs {} bytes {}", s.name, name, allocs, bytes);
    }
    TRACKING.store(false, Ordering::Relaxed);
}

fn run_bench() {
    for s in &BENCH_SCENES {
        bench_scene(s);
    }
    eprintln!("bench done");
}
