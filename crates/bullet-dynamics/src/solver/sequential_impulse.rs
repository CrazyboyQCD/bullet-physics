//! Sequential-impulse constraint solver: contact constraints with friction and
//! restitution, adapted from Bullet's `btSequentialImpulseConstraintSolver`.

use bullet_collision::manifold::ContactPoint;
use crate::body::RigidBody;
use bullet_math::{Matrix3, Real, Vector3};

/// Allowed penetration slop before the Baumgarte term kicks in.
pub(crate) const SLOP: Real = 0.005;
/// Baumgarte positional-correction factor — also the default joint ERP
/// (Bullet's `infoGlobal.m_erp`, which `btPoint2PointConstraint::getInfo2`
/// uses for `m_constraintError` unless overridden).
pub(crate) const BAUMGARTE: Real = 0.2;
/// Minimum approach speed (m/s) below which restitution is disabled.
pub(crate) const RESTITUTION_THRESHOLD: Real = 1.0;
/// Penetration depth (m) above which the split-impulse pass takes over
/// position correction (Bullet's `m_splitImpulsePenetrationThreshold = -0.04`,
/// sign-flipped to our positive-penetration convention). Deeper contacts are
/// corrected by the split pass only; shallower ones keep the Baumgarte bias.
const SPLIT_IMPULSE_PENETRATION_THRESHOLD: Real = 0.04;
/// Split-impulse positional correction factor (Bullet's `m_erp2`).
const SPLIT_ERP2: Real = 0.2;
/// Orientation scaling for the split-impulse turn velocity (Bullet's
/// `m_splitImpulseTurnErp`, applied in `writebackVelocityAndTransform`).
pub(crate) const SPLIT_TURN_ERP: Real = 0.1;

/// A pre-built contact constraint between two bodies.
pub struct ContactConstraint {
    pub(crate) body_a: usize,
    pub(crate) body_b: usize,
    pub(crate) normal: Vector3,
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
    pub(crate) normal_impulse: Real,
    pub(crate) tangent1_impulse: Real,
    pub(crate) tangent2_impulse: Real,
    /// Index of this row's contact point inside the owning pair's persistent
    /// manifold (`u32::MAX` for an ephemeral point with no cache slot), used
    /// by the post-solve writeback to store the accumulated impulse.
    pub(crate) point_index: u32,
    /// Split-impulse (deep-penetration) state: whether this contact's
    /// positional correction is handled by the split pass instead of the
    /// Baumgarte bias, the target pseudo-impulse (`m_rhsPenetration`) and the
    /// accumulated split impulse (`m_appliedPushImpulse`).
    split_active: bool,
    rhs_penetration: Real,
    push_impulse: Real,
}

/// Bounds on the accumulated impulse of a constraint row (Bullet's
/// `m_lowerLimit` / `m_upperLimit`; the defaults leave them infinite, an
/// optional `m_impulseClamp` makes them symmetric, and hinge limits/motors
/// clamp one or both ends).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowLimits {
    /// `[-inf, +inf]`: a plain equality row (Bullet's default for joints).
    Unbounded,
    /// `[-clamp, +clamp]` (Bullet's `btConstraintSetting::m_impulseClamp`
    /// and the hinge motor's `m_maxMotorImpulse`).
    Bilateral(Real),
    /// `[lower, +inf]` (a hinge low limit, and contacts' non-negative
    /// normal impulse).
    LowerBounded(Real),
    /// `[-inf, upper]` (a hinge high limit).
    UpperBounded(Real),
    /// `[lower, upper]`: a general asymmetric window (a joint row whose
    /// mirrored Bullet window is neither one-sided nor symmetric, e.g. the
    /// cone-twist motor's `[−maxMotorImpulse, 0]`).
    Window(Real, Real),
}

/// One solver row of a joint constraint (Bullet's `btSolverConstraint`
/// flattened for the non-contact pool): a bilateral equality with the
/// jacobian `(lin_a, ang_a)` on body A and `(lin_b, ang_b)` on body B,
/// measured as `J·v = (lin_b·v_b + ang_b·ω_b) − (lin_a·v_a + ang_a·ω_a)` and
/// driven by the position-error bias. Joint rows share the contact iteration
/// loop (Bullet's `solveGroupCacheFriendly` solves both pools).
pub struct ConstraintRow {
    body_a: usize,
    body_b: usize,
    /// Linear jacobian on body A (the impulse applies `−lin_a·λ`).
    lin_a: Vector3,
    /// Angular jacobian on body A (the impulse applies `−invI_a·ang_a·λ`).
    ang_a: Vector3,
    /// Linear jacobian on body B (the impulse applies `+lin_b·λ`).
    lin_b: Vector3,
    /// Angular jacobian on body B (the impulse applies `+invI_b·ang_b·λ`).
    ang_b: Vector3,
    inv_mass_a: Real,
    inv_mass_b: Real,
    inv_inertia_a: Matrix3,
    inv_inertia_b: Matrix3,
    /// Inverse effective mass along the row: `1 / K` with
    /// `K = Σ J·(M⁻¹·Jᵀ)` over both bodies.
    mass: Real,
    /// Velocity target for `J·v` (Bullet's `m_constraintError`, built into
    /// `m_rhs` at `convertJoint`).
    bias: Real,
    limits: RowLimits,
    impulse: Real,
}

impl ConstraintRow {
    /// Build a bilateral row from the per-body jacobians (Bullet's
    /// `convertJoint`): the effective mass from both bodies, the bias from
    /// the caller's position error, and the impulse clamp.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        body_a: usize,
        body_b: usize,
        lin_a: Vector3,
        ang_a: Vector3,
        lin_b: Vector3,
        ang_b: Vector3,
        a: &RigidBody,
        b: &RigidBody,
        bias: Real,
        limits: RowLimits,
    ) -> Self {
        let inv_inertia_a = a.inv_inertia_world();
        let inv_inertia_b = b.inv_inertia_world();
        let inv_mass_a = a.inv_mass();
        let inv_mass_b = b.inv_mass();
        // K = J·(M⁻¹·Jᵀ): linear + angular contributions of both bodies.
        let k = inv_mass_a * lin_a.length_squared()
            + inv_mass_b * lin_b.length_squared()
            + ang_a.dot(inv_inertia_a * ang_a)
            + ang_b.dot(inv_inertia_b * ang_b);
        let mass = if k > 0.0 { 1.0 / k } else { 0.0 };
        Self {
            body_a,
            body_b,
            lin_a,
            ang_a,
            lin_b,
            ang_b,
            inv_mass_a,
            inv_mass_b,
            inv_inertia_a,
            inv_inertia_b,
            mass,
            bias,
            limits,
            // Joints are not warm-started (Bullet resets
            // `internalSetAppliedImpulse(0)` in `convertJoints`).
            impulse: 0.0,
        }
    }
}

impl ContactConstraint {
    /// Build a constraint from a contact point and the two bodies.
    ///
    /// The tangent basis is derived from the point's *current* normal every
    /// step: cached friction impulses are deliberately not persisted across
    /// steps, because the basis rotates with the point set and stale
    /// tangential impulses made resting bodies oscillate (the persistent-
    /// manifold cache only stores the normal impulse — see `world.rs`).
    pub fn prepare(
        body_a: usize,
        body_b: usize,
        cp: &ContactPoint,
        a: &RigidBody,
        b: &RigidBody,
        point_index: u32,
    ) -> Self {
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

        // Combined material properties, matching Bullet's
        // `btManifoldResult::calculateCombinedFriction` /
        // `calculateCombinedRestitution`: plain products of the two bodies'
        // coefficients (verified against refs/bullet3 on 2026-08-14).
        let friction = a.friction * b.friction;
        let restitution = a.restitution * b.restitution;

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
            // The impulses inherited from the persistent contact cache (if any)
            // are re-applied by the warm-start pass below; new contacts start
            // from zero, exactly like Bullet's persistent-manifold flow.
            normal_impulse: cp.normal_impulse,
            tangent1_impulse: cp.tangent_impulse[0],
            tangent2_impulse: cp.tangent_impulse[1],
            point_index,
            // Deep penetration switches position correction from the Baumgarte
            // bias to the split-impulse pass (Bullet's `m_splitImpulse`
            // branch in `setupContactConstraint`); the split impulse itself
            // starts at zero every step, like `m_appliedPushImpulse`.
            split_active: cp.penetration > SPLIT_IMPULSE_PENETRATION_THRESHOLD,
            rhs_penetration: 0.0,
            push_impulse: 0.0,
        }
    }
}

/// Two orthonormal vectors perpendicular to `n`.
fn tangent_basis(n: Vector3) -> (Vector3, Vector3) {
    let helper = if n.x.abs() < 0.9 { Vector3::X } else { Vector3::Y };
    // mul-by-recip keeps the scalar-Vec3 bit pattern (Vec3A::normalize
    // divides, which differs in the last ulp and measurably destabilizes
    // tall box stacks through the friction basis).
    let c = n.cross(helper);
    let t1 = c * c.length_recip();
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

/// Solve all contact constraints and joint rows with sequential-impulse (PGS)
/// iterations. Contacts warm-start from their cached impulses (Bullet's
/// `SOLVER_USE_WARMSTARTING`); joints build from zero each step.
pub fn solve(
    bodies: &mut [RigidBody],
    constraints: &mut [ContactConstraint],
    joints: &mut [ConstraintRow],
    iterations: u32,
    inv_dt: Real,
) {
    warm_start(bodies, constraints);
    for _ in 0..iterations {
        velocity_iteration(bodies, constraints, joints, inv_dt);
    }
    split_impulse_pass(bodies, constraints, iterations, inv_dt);
}

/// Re-apply the impulses accumulated by the previous step (Bullet's
/// `SOLVER_USE_WARMSTARTING` pass), so resting contacts reach a converged
/// solution in a few iterations instead of rebuilding from zero.
pub(crate) fn warm_start(bodies: &mut [RigidBody], constraints: &mut [ContactConstraint]) {
    for c in constraints.iter_mut() {
        let (a, b) = two_bodies(bodies, c.body_a, c.body_b);
        apply_impulse(a, b, c, c.normal * c.normal_impulse);
        apply_impulse(a, b, c, c.tangent1 * c.tangent1_impulse);
        apply_impulse(a, b, c, c.tangent2 * c.tangent2_impulse);
    }
}

/// One PGS sweep over the rigid contact and joint rows.
pub(crate) fn velocity_iteration(
    bodies: &mut [RigidBody],
    constraints: &mut [ContactConstraint],
    joints: &mut [ConstraintRow],
    inv_dt: Real,
) {
    for c in constraints.iter_mut() {
        solve_contact(bodies, c, inv_dt);
    }
    for j in joints.iter_mut() {
        solve_joint_row(bodies, j);
    }
}

/// Split-impulse pass (Bullet's `solveGroupCacheFriendlySplitImpulseIterations`,
/// after the velocity iterations): deep-penetration contacts are pulled out
/// of the surface by a pseudo impulse accumulated on each body's push
/// velocity. It never touches the real velocities; the world applies the
/// push velocity to the pose after position integration, so the penetration
/// is corrected without injecting energy into the motion.
pub(crate) fn split_impulse_pass(
    bodies: &mut [RigidBody],
    constraints: &mut [ContactConstraint],
    iterations: u32,
    inv_dt: Real,
) {
    let split_rhs = |c: &mut ContactConstraint| {
        if c.split_active {
            c.rhs_penetration = c.penetration * SPLIT_ERP2 * inv_dt * c.normal_mass;
        }
    };
    for c in constraints.iter_mut() {
        split_rhs(c);
    }
    for _ in 0..iterations {
        for c in constraints.iter_mut() {
            if !c.split_active {
                continue;
            }
            let (a, b) = two_bodies(bodies, c.body_a, c.body_b);
            // Push relative normal velocity at the contact point (the split
            // counterpart of `rel.dot(normal)` in `solve_contact`).
            let push_va = a.push_linear + a.push_angular.cross(c.r_a);
            let push_vb = b.push_linear + b.push_angular.cross(c.r_b);
            let push_vn = (push_vb - push_va).dot(c.normal);
            // `delta = rhs - push_vn * mass` matches Bullet's
            // `gResolveSplitPenetrationImpulse_scalar_reference` with our
            // normal convention (A -> B) and zero CFM.
            let delta = c.rhs_penetration - push_vn * c.normal_mass;
            let old = c.push_impulse;
            c.push_impulse = (old + delta).max(0.0);
            let j = c.push_impulse - old;
            if j != 0.0 {
                apply_push(a, b, c, c.normal * j);
            }
        }
    }
}

/// Accumulate a split-impulse positional push onto the bodies' push velocities
/// (Bullet's `internalApplyPushImpulse`); same geometry as [`apply_impulse`]
/// but written to `push_linear`/`push_angular` instead of the real velocities.
fn apply_push(a: &mut RigidBody, b: &mut RigidBody, c: &ContactConstraint, impulse: Vector3) {
    a.push_linear -= impulse * c.inv_mass_a;
    b.push_linear += impulse * c.inv_mass_b;
    a.push_angular -= c.inv_inertia_a * (c.r_a.cross(impulse));
    b.push_angular += c.inv_inertia_b * (c.r_b.cross(impulse));
}

fn solve_contact(bodies: &mut [RigidBody], c: &mut ContactConstraint, inv_dt: Real) {
    let (a, b) = two_bodies(bodies, c.body_a, c.body_b);

    let va = a.linear_velocity + a.angular_velocity.cross(c.r_a);
    let vb = b.linear_velocity + b.angular_velocity.cross(c.r_b);
    let rel = vb - va;
    let vn = rel.dot(c.normal);

    // Target normal velocity: restitution plus Baumgarte penetration
    // correction. For deep penetrations the Baumgarte term is dropped: the
    // split-impulse pass corrects the position instead (Bullet's `m_rhs` =
    // velocity-only branch in `setupContactConstraint`), so the velocity
    // response stays a pure restitution/impact response without the energy
    // injection of a large positional bias.
    let mut bias = 0.0;
    if vn < -RESTITUTION_THRESHOLD {
        bias = -c.restitution * vn;
    }
    if !c.split_active {
        bias += BAUMGARTE * inv_dt * (c.penetration - SLOP).max(0.0);
    }
    // Separated (margin-band) contacts: Bullet's solver subtracts the
    // separation rate from the velocity error (`velocityError -= penetration
    // * invTimeStep` where `penetration = distance + linearSlop`; Bullet's
    // default slop is 0, so only genuinely separated points soften). Our
    // engine-wide slop stays a dead zone between separation and the
    // Baumgarte threshold.
    if c.penetration < 0.0 {
        bias += c.penetration * inv_dt;
    }

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

/// Solve one bilateral joint row (Bullet's `gResolveSingleConstraintRowGeneric`:
/// drive the relative velocity along the row toward the position-error bias,
/// clamping the accumulated impulse to the row limits).
fn solve_joint_row(bodies: &mut [RigidBody], c: &mut ConstraintRow) {
    let (a, b) = two_bodies(bodies, c.body_a, c.body_b);

    let row_vel = (c.lin_b.dot(b.linear_velocity) + c.ang_b.dot(b.angular_velocity))
        - (c.lin_a.dot(a.linear_velocity) + c.ang_a.dot(a.angular_velocity));

    let lambda = c.mass * (c.bias - row_vel);
    let old = c.impulse;
    let new = match c.limits {
        RowLimits::Unbounded => old + lambda,
        RowLimits::Bilateral(clamp) => (old + lambda).clamp(-clamp, clamp),
        RowLimits::LowerBounded(lower) => (old + lambda).max(lower),
        RowLimits::UpperBounded(upper) => (old + lambda).min(upper),
        RowLimits::Window(lower, upper) => (old + lambda).clamp(lower, upper),
    };
    c.impulse = new;
    let j = new - old;
    if j != 0.0 {
        a.linear_velocity -= c.lin_a * (j * c.inv_mass_a);
        b.linear_velocity += c.lin_b * (j * c.inv_mass_b);
        a.angular_velocity -= c.inv_inertia_a * (c.ang_a * j);
        b.angular_velocity += c.inv_inertia_b * (c.ang_b * j);
    }
}

fn apply_impulse(a: &mut RigidBody, b: &mut RigidBody, c: &ContactConstraint, impulse: Vector3) {
    a.linear_velocity -= impulse * c.inv_mass_a;
    b.linear_velocity += impulse * c.inv_mass_b;
    a.angular_velocity -= c.inv_inertia_a * (c.r_a.cross(impulse));
    b.angular_velocity += c.inv_inertia_b * (c.r_b.cross(impulse));
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use bullet_collision::shapes::CollisionShape;

    /// Build a ground plane at y = 0 and a dynamic box sunk `penetration` into
    /// it (contact point at the box bottom), with zero velocities so the
    /// velocity iterations stay idle and only the split pass acts.
    fn deep_contact(penetration: Real) -> (Vec<RigidBody>, Vec<ContactConstraint>) {
        let ground = RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        );
        let cube = RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.5)),
            Vector3::new(0.0, 0.5 - penetration, 0.0),
            1.0,
        );
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, -penetration, 0.0), // box bottom contact point
            Vector3::Y,                           // normal A -> B (box -> plane)
            penetration,
        );
        let c = ContactConstraint::prepare(0, 1, &cp, &ground, &cube, u32::MAX);
        (vec![ground, cube], vec![c])
    }

    /// Deep penetration (> threshold): the split pass pushes the body out of
    /// the surface (push velocity up) without touching the real velocity.
    #[test]
    fn split_impulse_pushes_deep_penetration_without_touching_velocity() {
        let (mut bodies, mut constraints) = deep_contact(0.1);
        solve(&mut bodies, &mut constraints, &mut [], 10, 60.0);
        // Split pass accumulated an upward positional push on the box.
        assert!(bodies[1].push_linear.y > 0.0);
        // The real velocity is untouched: zero approach speed means the
        // velocity iterations produce no normal impulse.
        assert_relative_eq!(bodies[1].linear_velocity.y, 0.0, epsilon = 1e-6);
        // The Baumgarte bias is disabled for this contact (split takes over).
        assert!(constraints[0].split_active);
    }

    /// Shallow penetration (< threshold): the split pass stays idle and the
    /// Baumgarte bias handles position correction as before.
    #[test]
    fn split_impulse_ignores_shallow_penetration() {
        let (mut bodies, mut constraints) = deep_contact(0.01);
        solve(&mut bodies, &mut constraints, &mut [], 10, 60.0);
        assert!(!constraints[0].split_active);
        assert_eq!(bodies[1].push_linear, Vector3::ZERO);
    }

    /// The split impulse is clamped below at zero (never pulls bodies into
    /// the surface) and converges to the penetration target over iterations.
    #[test]
    fn split_impulse_converges_and_clamps() {
        let (mut bodies, mut constraints) = deep_contact(0.1);
        // Half the iterations of the default solve.
        solve(&mut bodies, &mut constraints, &mut [], 5, 60.0);
        let push_full = bodies[1].push_linear.y;
        let (mut bodies2, mut constraints2) = deep_contact(0.1);
        solve(&mut bodies2, &mut constraints2, &mut [], 20, 60.0);
        // More iterations converge further toward removing the penetration;
        // both stay positive (no pull-in).
        assert!(push_full > 0.0);
        assert!(bodies2[1].push_linear.y >= push_full);
    }
}

