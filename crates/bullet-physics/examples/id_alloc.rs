//! Zero-allocation audit for the inverse-dynamics tree (M3, the contract of
//! `docs/design/inverse-dynamics.md` §9): after `finalize`, no heap
//! allocation happens in any `calculate_*`, mass setter, user-wrench API or
//! getter. Runs under a counting global allocator — the same accounting unit
//! as the bench mode of `id_probe.rs --bench` and the C++ probe's
//! operator-new hook. The build/finalize phase legitimately allocates (all
//! storage is sized there) and is outside the audited windows.
//!
//! Exit code 1 + a `ZERO-ALLOC VIOLATION` line names the offending scene and
//! operation; otherwise one `OK zero-alloc <scene>` line per scene and a
//! final `OK` (the check.py demo-gate protocol).
//!
//! Run:
//!     cargo run -p bullet-physics --example id_alloc --features inverse-dynamics

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

use bullet_physics::inverse_dynamics::builder::{JointType, TreeBuilder};
use bullet_physics::inverse_dynamics::math;
use bullet_physics::inverse_dynamics::{Matrix3, MultiBodyTree, Real, Vector3};
use bullet_physics::math::CrtRand;

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

fn diagonal(xx: Real, yy: Real, zz: Real) -> Matrix3 {
    Matrix3::from_cols(
        Vector3::new(xx, 0.0, 0.0),
        Vector3::new(0.0, yy, 0.0),
        Vector3::new(0.0, 0.0, zz),
    )
}

fn v3(x: f64, y: f64, z: f64) -> Vector3 {
    Vector3::new(x as Real, y as Real, z as Real)
}

#[allow(clippy::too_many_arguments)]
fn add(b: &mut TreeBuilder, body: usize, parent: isize, jt: JointType, r: Vector3,
       t: Matrix3, axis: Vector3, mass: Real, com: Vector3, inertia: Matrix3) {
    b.add_body(body, parent, jt, r, t, axis, mass, com, inertia).unwrap();
}

/// Audited window: count the allocations of one call, abort on any.
fn audited<T>(scene: &str, op: &str, f: impl FnOnce() -> T) -> T {
    let a0 = ALLOCS.load(Ordering::Relaxed);
    let b0 = TOTAL_BYTES.load(Ordering::Relaxed);
    let out = f();
    let da = ALLOCS.load(Ordering::Relaxed) - a0;
    let db = TOTAL_BYTES.load(Ordering::Relaxed) - b0;
    if da != 0 || db != 0 {
        eprintln!("ZERO-ALLOC VIOLATION {scene} {op}: allocs {da} bytes {db}");
        std::process::exit(1);
    }
    out
}

fn audit_scene(scene: &str, tree: &mut MultiBodyTree) {
    let nd = tree.num_dofs();
    let nb = tree.num_bodies();
    // Buffers predate the audited windows.
    let q: Vec<Real> = (0..nd).map(|i| (i as f64 * 0.25 - 1.0) as Real).collect();
    let u: Vec<Real> = (0..nd).map(|i| (i as f64 * 0.125 + 0.5) as Real).collect();
    let du: Vec<Real> = (0..nd).map(|i| (i as f64 * 0.0625 - 0.25) as Real).collect();
    let mut tau = vec![0.0 as Real; nd];
    let mut mm = vec![0.0 as Real; nd * nd];
    let mut jac = vec![0.0 as Real; 3 * nd];

    TRACKING.store(true, Ordering::Relaxed);

    audited(scene, "calculate_inverse_dynamics", || {
        tree.calculate_inverse_dynamics(&q, &u, &du, &mut tau).unwrap();
    });
    audited(scene, "calculate_kinematics", || {
        tree.calculate_kinematics(&q, &u, &du).unwrap();
    });
    audited(scene, "calculate_position_kinematics", || {
        tree.calculate_position_kinematics(&q).unwrap();
    });
    audited(scene, "calculate_position_and_velocity_kinematics", || {
        tree.calculate_position_and_velocity_kinematics(&q, &u).unwrap();
    });
    audited(scene, "calculate_mass_matrix", || {
        tree.calculate_mass_matrix(&q, &mut mm).unwrap();
    });
    audited(scene, "calculate_mass_matrix_ex(refresh,init,upper)", || {
        tree.calculate_mass_matrix_ex(&q, true, true, false, &mut mm).unwrap();
    });
    audited(scene, "calculate_mass_matrix_ex(stale,reuse,lower)", || {
        tree.calculate_mass_matrix_ex(&q, false, false, true, &mut mm).unwrap();
    });
    audited(scene, "calculate_jacobians", || {
        tree.calculate_jacobians(&q, &u).unwrap();
    });
    audited(scene, "calculate_position_jacobians", || {
        tree.calculate_position_jacobians(&q).unwrap();
    });
    // Runtime setters ride the last body (floating_free carries one body).
    let last = nb - 1;
    audited(scene, "set_body_mass", || {
        tree.set_body_mass(last, 1.5).unwrap();
    });
    audited(scene, "set_body_first_mass_moment", || {
        tree.set_body_first_mass_moment(last, v3(0.2, -0.1, 0.05)).unwrap();
    });
    audited(scene, "set_body_second_mass_moment", || {
        tree.set_body_second_mass_moment(last, diagonal(0.1, 0.1, 0.1)).unwrap();
    });
    audited(scene, "add_user_force/moment", || {
        tree.add_user_force(last, v3(1.0, 0.0, -1.0));
        tree.add_user_moment(last, v3(0.0, 0.5, 0.0));
        tree.clear_all_user_forces_and_moments();
    });
    audited(scene, "set_gravity_in_world_frame", || {
        tree.set_gravity_in_world_frame(v3(0.0, 0.0, -9.8));
    });
    for b in 0..nb {
        audited(scene, &format!("body {b} getters"), || {
            let _ = tree.body_origin(b);
            let _ = tree.body_com(b);
            let _ = tree.body_transform(b);
            let _ = tree.body_angular_velocity(b);
            let _ = tree.body_linear_velocity(b);
            let _ = tree.body_linear_velocity_com(b);
            let _ = tree.body_angular_acceleration(b);
            let _ = tree.body_linear_acceleration(b);
            let _ = tree.parent_index(b);
            let _ = tree.joint_type(b);
            let _ = tree.dof_offset(b);
            let _ = tree.body_dot_jacobian_trans_u(b);
            let _ = tree.body_dot_jacobian_rot_u(b);
            tree.body_jacobian_trans(b, &mut jac).unwrap();
            tree.body_jacobian_rot(b, &mut jac).unwrap();
        });
    }
    // Out-of-range getters take the same path (None / Err) — audit one.
    audited(scene, "out-of-range getters", || {
        let _ = tree.body_origin(nb);
        let _ = tree.parent_index(nb);
    });

    TRACKING.store(false, Ordering::Relaxed);
    println!("OK zero-alloc {scene} bodies {nb} dofs {nd}");
}

fn canonical_trees() -> Vec<(&'static str, MultiBodyTree)> {
    let mut trees = Vec::new();
    {
        let mut b = TreeBuilder::new();
        add(&mut b, 0, -1, JointType::Fixed, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 0.0, v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
        add(&mut b, 1, 0, JointType::Revolute, v3(1.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 1.0), 1.0, v3(0.5, 0.0, 0.0), diagonal(0.2, 0.15, 0.1));
        add(&mut b, 2, 1, JointType::Revolute, v3(1.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 1.0), 0.8, v3(0.4, 0.0, 0.0), diagonal(0.1, 0.08, 0.06));
        trees.push(("arm2r", b.finalize().unwrap()));
    }
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
        trees.push(("mixed1", b.finalize().unwrap()));
    }
    {
        let mut b = TreeBuilder::new();
        add(&mut b, 0, -1, JointType::Floating, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 2.0, v3(0.1, 0.2, 0.3), diagonal(0.3, 0.25, 0.2));
        trees.push(("floating_free", b.finalize().unwrap()));
    }
    {
        let mut b = TreeBuilder::new();
        add(&mut b, 0, -1, JointType::Fixed, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 0.0, v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
        add(&mut b, 1, 0, JointType::Spherical, v3(0.0, 0.0, 1.0), Matrix3::IDENTITY,
            v3(0.0, 0.0, 0.0), 1.0, v3(0.0, 0.0, -0.5), diagonal(0.1, 0.1, 0.1));
        add(&mut b, 2, 1, JointType::Fixed, v3(0.0, 0.0, -0.6),
            math::transform_z(0.75), v3(0.0, 0.0, 0.0), 0.3, v3(0.05, 0.0, 0.0),
            diagonal(0.01, 0.01, 0.01));
        trees.push(("spherical_arm", b.finalize().unwrap()));
    }
    trees
}

/// The bench chain (seed 2101) exercises deep ancestor chains in the CRBA
/// column fill and the jacobian recursion.
fn chain_16() -> MultiBodyTree {
    let mut b = TreeBuilder::new();
    add(&mut b, 0, -1, JointType::Fixed, v3(0.0, 0.0, 0.0), Matrix3::IDENTITY,
        v3(0.0, 0.0, 0.0), 0.0, v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
    for i in 1..16 {
        let r = v3(-0.4 + 0.8 * (i as f64 % 5.0) / 5.0, 0.1, -0.2);
        let t_ref = math::transform_z(0.1 * i as Real);
        let axis = Vector3::Z;
        add(&mut b, i, i as isize - 1, JointType::Revolute, r, t_ref, axis,
            1.0 + 0.01 * i as Real, v3(0.5, 0.0, 0.0), diagonal(0.2, 0.15, 0.1));
    }
    b.finalize().unwrap()
}

/// A seeded random tree (the parity scene 4 seed) covers random parents with
/// multiple floating/spherical branches.
fn random_tree() -> MultiBodyTree {
    let mut rand = CrtRand::seeded(1);
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
        // Params analogous to the probe's draw protocol; only coverage
        // matters here, not cross-engine identity.
        let r = v3(-0.3 + 0.1 * i as f64, 0.2, -0.1);
        let t_ref = math::transform_y(0.3) * math::transform_x(-0.2);
        let axis = v3(0.3, 0.8, 0.5);
        let inertia = diagonal(0.15, 0.12, 0.1);
        add(&mut b, i, parent, jt, r, t_ref, axis, 1.0, v3(0.1, 0.0, 0.0), inertia);
    }
    b.finalize().unwrap()
}

fn main() {
    for (name, mut tree) in canonical_trees() {
        audit_scene(name, &mut tree);
    }
    let mut tree = chain_16();
    audit_scene("chain_16", &mut tree);
    let mut tree = random_tree();
    audit_scene("random", &mut tree);
    println!("OK");
}
