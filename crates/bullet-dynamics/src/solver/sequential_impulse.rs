//! Sequential-impulse constraint solver: contact constraints with friction and
//! restitution, adapted from Bullet's `btSequentialImpulseConstraintSolver`.

use bullet_collision::manifold::ContactPoint;
use crate::body::RigidBody;
use bullet_math::{Matrix3, Real, Vector3};

/// Allowed penetration slop before the Baumgarte term kicks in.
const SLOP: Real = 0.005;
/// Baumgarte positional-correction factor.
const BAUMGARTE: Real = 0.2;
/// Minimum approach speed (m/s) below which restitution is disabled.
const RESTITUTION_THRESHOLD: Real = 1.0;

/// A pre-built contact constraint between two bodies.
pub struct ContactConstraint {
    body_a: usize,
    body_b: usize,
    normal: Vector3,
    tangent1: Vector3,
    tangent2: Vector3,
    r_a: Vector3,
    r_b: Vector3,
    inv_mass_a: Real,
    inv_mass_b: Real,
    inv_inertia_a: Matrix3,
    inv_inertia_b: Matrix3,
    normal_mass: Real,
    tangent1_mass: Real,
    tangent2_mass: Real,
    penetration: Real,
    restitution: Real,
    friction: Real,
    normal_impulse: Real,
    tangent1_impulse: Real,
    tangent2_impulse: Real,
}

impl ContactConstraint {
    /// Build a constraint from a contact point and the two bodies.
    pub fn prepare(body_a: usize, body_b: usize, cp: &ContactPoint, a: &RigidBody, b: &RigidBody) -> Self {
        let normal = cp.normal;
        let (tangent1, tangent2) = tangent_basis(normal);
        let r_a = cp.position - a.position;
        let r_b = cp.position - b.position;
        let inv_inertia_a = a.inv_inertia_world();
        let inv_inertia_b = b.inv_inertia_world();
        let inv_mass_a = a.inv_mass();
        let inv_mass_b = b.inv_mass();

        let normal_mass = effective_mass(inv_mass_a, inv_mass_b, &inv_inertia_a, &inv_inertia_b, r_a, r_b, normal);
        let tangent1_mass =
            effective_mass(inv_mass_a, inv_mass_b, &inv_inertia_a, &inv_inertia_b, r_a, r_b, tangent1);
        let tangent2_mass =
            effective_mass(inv_mass_a, inv_mass_b, &inv_inertia_a, &inv_inertia_b, r_a, r_b, tangent2);

        let friction = (a.friction * b.friction).sqrt();
        let restitution = a.restitution.max(b.restitution);

        ContactConstraint {
            body_a,
            body_b,
            normal,
            tangent1,
            tangent2,
            r_a,
            r_b,
            inv_mass_a,
            inv_mass_b,
            inv_inertia_a,
            inv_inertia_b,
            normal_mass,
            tangent1_mass,
            tangent2_mass,
            penetration: cp.penetration,
            restitution,
            friction,
            normal_impulse: 0.0,
            tangent1_impulse: 0.0,
            tangent2_impulse: 0.0,
        }
    }
}

/// Two orthonormal vectors perpendicular to `n`.
fn tangent_basis(n: Vector3) -> (Vector3, Vector3) {
    let helper = if n.x.abs() < 0.9 { Vector3::X } else { Vector3::Y };
    let t1 = n.cross(helper).normalize();
    let t2 = n.cross(t1);
    (t1, t2)
}

/// Inverse effective mass along a direction for a pair of bodies.
#[allow(clippy::too_many_arguments)]
fn effective_mass(
    inv_mass_a: Real,
    inv_mass_b: Real,
    inv_inertia_a: &Matrix3,
    inv_inertia_b: &Matrix3,
    r_a: Vector3,
    r_b: Vector3,
    dir: Vector3,
) -> Real {
    let k = inv_mass_a
        + inv_mass_b
        + dir.dot((*inv_inertia_a * (r_a.cross(dir))).cross(r_a))
        + dir.dot((*inv_inertia_b * (r_b.cross(dir))).cross(r_b));
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// Split a slice to obtain two distinct mutable bodies.
fn two_bodies(bodies: &mut [RigidBody], a: usize, b: usize) -> (&mut RigidBody, &mut RigidBody) {
    debug_assert_ne!(a, b);
    if a < b {
        let (l, r) = bodies.split_at_mut(b);
        (&mut l[a], &mut r[0])
    } else {
        let (l, r) = bodies.split_at_mut(a);
        (&mut r[0], &mut l[b])
    }
}

/// Solve all contact constraints with sequential-impulse (PGS) iterations.
pub fn solve(bodies: &mut [RigidBody], constraints: &mut [ContactConstraint], iterations: u32, inv_dt: Real) {
    for _ in 0..iterations {
        for c in constraints.iter_mut() {
            solve_contact(bodies, c, inv_dt);
        }
    }
}

fn solve_contact(bodies: &mut [RigidBody], c: &mut ContactConstraint, inv_dt: Real) {
    let (a, b) = two_bodies(bodies, c.body_a, c.body_b);

    let va = a.linear_velocity + a.angular_velocity.cross(c.r_a);
    let vb = b.linear_velocity + b.angular_velocity.cross(c.r_b);
    let rel = vb - va;
    let vn = rel.dot(c.normal);

    // Target normal velocity: restitution plus Baumgarte penetration correction.
    let mut bias = 0.0;
    if vn < -RESTITUTION_THRESHOLD {
        bias = -c.restitution * vn;
    }
    bias += BAUMGARTE * inv_dt * (c.penetration - SLOP).max(0.0);

    // Normal impulse, clamped to be non-negative.
    let lambda = c.normal_mass * (bias - vn);
    let old = c.normal_impulse;
    c.normal_impulse = (old + lambda).max(0.0);
    let jn = c.normal_impulse - old;

    apply_impulse(a, b, c, c.normal * jn);

    // Friction: two tangent directions, clamped by Coulomb's law.
    let max_friction = c.friction * c.normal_impulse;

    // Tangent 1.
    {
        let (a, b) = two_bodies(bodies, c.body_a, c.body_b);
        let va = a.linear_velocity + a.angular_velocity.cross(c.r_a);
        let vb = b.linear_velocity + b.angular_velocity.cross(c.r_b);
        let vt = (vb - va).dot(c.tangent1);
        let lambda = -c.tangent1_mass * vt;
        let old = c.tangent1_impulse;
        let new = (old + lambda).clamp(-max_friction, max_friction);
        c.tangent1_impulse = new;
        apply_impulse(a, b, c, c.tangent1 * (new - old));
    }

    // Tangent 2.
    {
        let (a, b) = two_bodies(bodies, c.body_a, c.body_b);
        let va = a.linear_velocity + a.angular_velocity.cross(c.r_a);
        let vb = b.linear_velocity + b.angular_velocity.cross(c.r_b);
        let vt = (vb - va).dot(c.tangent2);
        let lambda = -c.tangent2_mass * vt;
        let old = c.tangent2_impulse;
        let new = (old + lambda).clamp(-max_friction, max_friction);
        c.tangent2_impulse = new;
        apply_impulse(a, b, c, c.tangent2 * (new - old));
    }
}

fn apply_impulse(a: &mut RigidBody, b: &mut RigidBody, c: &ContactConstraint, impulse: Vector3) {
    a.linear_velocity -= impulse * c.inv_mass_a;
    b.linear_velocity += impulse * c.inv_mass_b;
    a.angular_velocity -= c.inv_inertia_a * (c.r_a.cross(impulse));
    b.angular_velocity += c.inv_inertia_b * (c.r_b.cross(impulse));
}
