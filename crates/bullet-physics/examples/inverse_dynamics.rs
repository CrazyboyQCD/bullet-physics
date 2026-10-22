//! Inverse-dynamics demo (optional `inverse-dynamics` feature): a planar 2R
//! arm under gravity — the generalized forces that hold a pose, the
//! joint-space mass matrix and a world-frame jacobian. Unlike
//! `PhysicsWorld::step_simulation`, inverse dynamics is a one-shot solver:
//! given `(q, u, dot_u)` it returns the joint forces that realize them.
//!
//! Run with:
//!     cargo run -p bullet-physics --example inverse_dynamics --features inverse-dynamics

use bullet_physics::inverse_dynamics::builder::{JointType, TreeBuilder};
use bullet_physics::inverse_dynamics::{Matrix3, Vector3};
use bullet_physics::math::Real;

fn diag(x: Real, y: Real, z: Real) -> Matrix3 {
    Matrix3::from_cols(
        Vector3::new(x, 0.0, 0.0),
        Vector3::new(0.0, y, 0.0),
        Vector3::new(0.0, 0.0, z),
    )
}

fn main() {
    // Fixed root + two Z-revolute links, one unit apart, planar in x-y.
    let mut b = TreeBuilder::new();
    b.add_body(
        0, -1, JointType::Fixed, Vector3::ZERO, Matrix3::IDENTITY,
        Vector3::ZERO, 0.0, Vector3::ZERO, diag(1.0, 1.0, 1.0),
    )
    .unwrap();
    b.add_body(
        1, 0, JointType::Revolute, Vector3::new(1.0, 0.0, 0.0), Matrix3::IDENTITY,
        Vector3::Z, 1.0, Vector3::new(0.5, 0.0, 0.0), diag(0.2, 0.15, 0.1),
    )
    .unwrap();
    b.add_body(
        2, 1, JointType::Revolute, Vector3::new(1.0, 0.0, 0.0), Matrix3::IDENTITY,
        Vector3::Z, 0.8, Vector3::new(0.4, 0.0, 0.0), diag(0.1, 0.08, 0.06),
    )
    .unwrap();
    let mut tree = b.finalize().unwrap();
    tree.set_gravity_in_world_frame(Vector3::new(0.0, -9.8, 0.0));

    // Generalized forces holding the arm horizontal (both links along +x)
    // against gravity, at rest: each joint carries the offset weight of the
    // links in front of it.
    let q = [0.0, 0.0];
    let mut tau = [0.0; 2];
    tree.calculate_inverse_dynamics(&q, &[0.0; 2], &[0.0; 2], &mut tau)
        .unwrap();
    println!("hold torques: {:.3} {:.3} N·m", tau[0], tau[1]);
    let (t0, t1) = (tau[0], tau[1]);
    // Hand-derived: |τ₀| = g·(m₁·0.5 + m₂·1.4), |τ₁| = g·m₂·0.4 — the
    // actuator torques oppose the (clockwise) gravity moments, hence +.
    assert!((t0 - 9.8 * (0.5 + 0.8 * 1.4)).abs() < 1e-3, "shoulder torque {t0}");
    assert!((t1 - 9.8 * 0.8 * 0.4).abs() < 1e-3, "elbow torque {t1}");

    // Joint-space mass matrix at the same pose (2x2, row-major).
    let mut mm = [0.0; 4];
    tree.calculate_mass_matrix(&q, &mut mm).unwrap();
    println!(
        "mass matrix: [ {:.3} {:.3} ; {:.3} {:.3} ]",
        mm[0], mm[1], mm[2], mm[3]
    );

    // World-frame rotational jacobian of the end link: its columns are the
    // two joint axes in world coordinates — both are Z here, exactly.
    tree.calculate_position_kinematics(&q).unwrap();
    tree.calculate_position_jacobians(&q).unwrap();
    let mut jac = [0.0; 6];
    tree.body_jacobian_rot(2, &mut jac).unwrap();
    println!(
        "end-link jac_R: [ {:.1} {:.1} ; {:.1} {:.1} ; {:.1} {:.1} ]",
        jac[0], jac[3], jac[1], jac[4], jac[2], jac[5]
    );
    assert_eq!([jac[2], jac[5]], [1.0, 1.0]);
    println!("OK: inverse dynamics, mass matrix and jacobians behave.");
}
