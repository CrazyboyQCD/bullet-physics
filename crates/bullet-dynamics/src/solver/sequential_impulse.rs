//! Sequential-impulse constraint solver: contact constraints with friction and
//! restitution, adapted from Bullet's `btSequentialImpulseConstraintSolver`.

use bullet_collision::manifold::ContactPoint;
use crate::body::RigidBody;
use bullet_math::{math, FRAC_1_SQRT_2, Matrix3, Real, Vector3};

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
    /// Pre-folded angular impulse components along each row direction
    /// (`comp = invI · (r × dir)`, Bullet's per-row `m_angularComponentA/B`
    /// from `setupContactConstraint`): applying an impulse `dir · j` becomes
    /// `ω -= comp · j`, so the ten-iteration hot path multiplies a vector by
    /// a scalar instead of re-running `invI · (r × impulse)` matrix-vector
    /// products. The row no longer stores the inverse-inertia matrices.
    ang_normal_a: Vector3,
    ang_normal_b: Vector3,
    ang_tangent1_a: Vector3,
    ang_tangent1_b: Vector3,
    ang_tangent2_a: Vector3,
    ang_tangent2_b: Vector3,
    normal_mass: Real,
    tangent1_mass: Real,
    tangent2_mass: Real,
    penetration: Real,
    /// `(penetration - SLOP).max(0.0)`, hoisted out of the iteration loop:
    /// the solve multiplies it by the iteration-invariant `BAUMGARTE * inv_dt`
    /// (same left-to-right association as the inline expression it replaces).
    baumgarte_span: Real,
    /// `min(penetration, 0.0)`: negative only for separated (margin-band)
    /// points, where the solve softens the velocity error by
    /// `separation * inv_dt` (Bullet's `velocityError -= penetration *
    /// invTimeStep`); zero otherwise.
    separation: Real,
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
    /// Whether each side actually receives impulses (`inv_mass > 0`):
    /// static and kinematic bodies carry zero inverse mass and a zero
    /// inverse-inertia, so applying to them adds exact zeros to their
    /// velocities — skipped to save the arithmetic and the body
    /// writeback (Bullet maps static/kinematic sides to a shared fixed
    /// solver body and skips their writeback). Reading is untouched:
    /// kinematic velocities are real and must enter `J·v`.
    applies_a: bool,
    applies_b: bool,
    /// Per-unit-impulse delta of the relative velocity `rel = vb - va`:
    /// applying `dir · j` advances `rel` by `K_dir · j`, where
    /// `K_dir = dir·(invMass_a + invMass_b) + (invI_b·(r_b×dir))×r_b +
    /// (invI_a·(r_a×dir))×r_a` (both terms pre-folded in `prepare`).
    /// `K_normal·normal == 1 / normal_mass` by construction. The solve
    /// loop advances `rel` through these couplings instead of re-reading
    /// both bodies' velocities per tangent block.
    kn: Vector3,
    kt1: Vector3,
    kt2: Vector3,
    /// Index of this contact's rolling-friction rows in the world-owned
    /// pool (`None` unless the combined rolling friction is positive; the
    /// pool lives outside this struct to keep the hot-path footprint of
    /// every row unchanged — most scenes run without rolling friction).
    rolling: Option<u32>,
}

/// The two torsional-friction rows of one contact point (Bullet's
/// `setupTorsionalFrictionConstraint` pair installed for a positive
/// `m_combinedRollingFriction`): pure-angular constraints on the two
/// tangent-plane axes that drive the contact's relative angular velocity
/// toward zero. Each row's accumulated impulse is clamped per iteration to
/// `±min(rolling_friction · normal_impulse, rolling_friction)`, so rolling
/// bodies only decelerate while the contact actually carries load. The
/// rows are rebuilt from zero every step (Bullet's rolling pool stores
/// `m_originalContactPoint = 0`: no warm start) and never touch the
/// linear velocity or the split-impulse pass.
pub struct RollingFriction {
    /// The tangent-plane axes (Bullet's `btPlaneSpace1(normal)` output,
    /// re-normalized like the assembly site does).
    axis: [Vector3; 2],
    /// Pre-folded angular impulse components `invI · axis` per side. A
    /// row impulse applies `ω_a -= comp_a · j` and `ω_b += comp_b · j`,
    /// matching Bullet's `angularComponentA/B` signs for its
    /// `relpos1CrossNormal = -axis` / `relpos2CrossNormal = +axis` rows.
    comp_a: [Vector3; 2],
    comp_b: [Vector3; 2],
    /// Inverse effective mass per row: `1 / (axis·invI_a·axis +
    /// axis·invI_b·axis)` — a pure-rotation row, no linear term.
    mass: [Real; 2],
    /// Accumulated impulses of the current step (reset by `prepare`).
    impulse: [Real; 2],
    /// The combined rolling-friction coefficient (the clamp scale).
    friction: Real,
}

impl RollingFriction {
    /// Build the two rows for a contact with the given normal.
    pub(crate) fn prepare(
        normal: Vector3,
        inv_inertia_a: Matrix3,
        inv_inertia_b: Matrix3,
        friction: Real,
    ) -> Self {
        let (a0, a1) = plane_space(normal);
        // The assembly site normalizes both axes (Bullet's `axis.normalize()`);
        // the `length() > 0.001` gate after it only exists for the anisotropic
        // filter this crate does not have, so both axes are always unit here.
        let axis = [a0 / a0.length(), a1 / a1.length()];
        let mut comp_a = [Vector3::ZERO; 2];
        let mut comp_b = [Vector3::ZERO; 2];
        let mut mass = [0.0; 2];
        for (i, &axis) in axis.iter().enumerate() {
            comp_a[i] = inv_inertia_a * axis;
            comp_b[i] = inv_inertia_b * axis;
            let k = axis.dot(comp_a[i]) + axis.dot(comp_b[i]);
            mass[i] = if k > 0.0 { 1.0 / k } else { 0.0 };
        }
        Self {
            axis,
            comp_a,
            comp_b,
            mass,
            impulse: [0.0; 2],
            friction,
        }
    }
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

        // Pre-fold the angular components once (Bullet's
        // `m_angularComponentA/B = invInertiaTensorWorld * relPos.cross(dir)`).
        let ang_normal_a = inv_inertia_a * (r_a.cross(normal));
        let ang_normal_b = inv_inertia_b * (r_b.cross(normal));
        let ang_tangent1_a = inv_inertia_a * (r_a.cross(tangent1));
        let ang_tangent1_b = inv_inertia_b * (r_b.cross(tangent1));
        let ang_tangent2_a = inv_inertia_a * (r_a.cross(tangent2));
        let ang_tangent2_b = inv_inertia_b * (r_b.cross(tangent2));

        let normal_mass =
            effective_mass(inv_mass_a, inv_mass_b, ang_normal_a, ang_normal_b, r_a, r_b, normal);
        let tangent1_mass = effective_mass(
            inv_mass_a,
            inv_mass_b,
            ang_tangent1_a,
            ang_tangent1_b,
            r_a,
            r_b,
            tangent1,
        );
        let tangent2_mass = effective_mass(
            inv_mass_a,
            inv_mass_b,
            ang_tangent2_a,
            ang_tangent2_b,
            r_a,
            r_b,
            tangent2,
        );

        // Delta-velocity couplings of the relative velocity per unit
        // impulse along each row direction (see the field docs).
        let kn = normal * (inv_mass_a + inv_mass_b)
            + ang_normal_b.cross(r_b)
            + ang_normal_a.cross(r_a);
        let kt1 = tangent1 * (inv_mass_a + inv_mass_b)
            + ang_tangent1_b.cross(r_b)
            + ang_tangent1_a.cross(r_a);
        let kt2 = tangent2 * (inv_mass_a + inv_mass_b)
            + ang_tangent2_b.cross(r_b)
            + ang_tangent2_a.cross(r_a);

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
            ang_normal_a,
            ang_normal_b,
            ang_tangent1_a,
            ang_tangent1_b,
            ang_tangent2_a,
            ang_tangent2_b,
            normal_mass,
            tangent1_mass,
            tangent2_mass,
            penetration: cp.penetration,
            baumgarte_span: (cp.penetration - SLOP).max(0.0),
            separation: cp.penetration.min(0.0),
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
            applies_a: inv_mass_a > 0.0,
            applies_b: inv_mass_b > 0.0,
            kn,
            kt1,
            kt2,
            rolling: None,
        }
    }

    /// Install the contact's rolling-friction rows into the world-owned
    /// pool (Bullet's `addTorsionalFrictionConstraint` pair, gated on the
    /// combined coefficient being positive). A no-op for rows without a
    /// rolling-friction pair value, which is every scene that never sets
    /// the field.
    pub(crate) fn attach_rolling(
        &mut self,
        a: &RigidBody,
        b: &RigidBody,
        pool: &mut Vec<RollingFriction>,
    ) {
        // Bullet's `calculateCombinedRollingFriction`, verbatim quirk and
        // all: each side's rolling friction weighted by the *other* side's
        // linear friction, clamped to ±10 — not the product the linear
        // friction combines with.
        let combined = (a.rolling_friction * b.friction + b.rolling_friction * a.friction)
            .clamp(-10.0, 10.0);
        if combined <= 0.0 {
            return;
        }
        let rows = RollingFriction::prepare(
            self.normal,
            a.inv_inertia_world(),
            b.inv_inertia_world(),
            combined,
        );
        self.rolling = Some(pool.len() as u32);
        pool.push(rows);
    }
}

/// Bullet's `btPlaneSpace1` (btVector3.h): two unit vectors spanning the
/// plane perpendicular to `n`, branched on `|n.z| > SIMDSQRT12` (transcribed
/// component-for-component, including the `q = n × p` composition through
/// `p`'s components, so both engines build the same rolling basis).
fn plane_space(n: Vector3) -> (Vector3, Vector3) {
    // Bullet's `SIMDSQRT12` (btScalar.h): the f32/f64 roundings of the
    // constant match the C++ literal's casts.
    let simdsqrt12: Real = FRAC_1_SQRT_2;
    if n.z.abs() > simdsqrt12 {
        // Choose p in the y-z plane.
        let a = n.y * n.y + n.z * n.z;
        let k = 1.0 / math::sqrt(a);
        let p = Vector3::new(0.0, -n.z * k, n.y * k);
        // q = n × p
        let q = Vector3::new(a * k, -n.x * p.z, n.x * p.y);
        (p, q)
    } else {
        // Choose p in the x-y plane.
        let a = n.x * n.x + n.y * n.y;
        let k = 1.0 / math::sqrt(a);
        let p = Vector3::new(-n.y * k, n.x * k, 0.0);
        // q = n × p
        let q = Vector3::new(-n.z * p.y, n.z * p.x, a * k);
        (p, q)
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

/// Inverse effective mass along a direction for a pair of bodies. The
/// pre-folded angular components `ang_a = invI_a · (r_a × dir)` and
/// `ang_b = invI_b · (r_b × dir)` come straight from `prepare`, which has
/// already built them for the impulse-application path — the identical
/// intermediates this function used to re-derive, so `k` is bit-identical.
#[allow(clippy::too_many_arguments)]
fn effective_mass(
    inv_mass_a: Real,
    inv_mass_b: Real,
    ang_a: Vector3,
    ang_b: Vector3,
    r_a: Vector3,
    r_b: Vector3,
    dir: Vector3,
) -> Real {
    let k = inv_mass_a
        + inv_mass_b
        + dir.dot(ang_a.cross(r_a))
        + dir.dot(ang_b.cross(r_b));
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
    rolling: &mut [RollingFriction],
    iterations: u32,
    inv_dt: Real,
) {
    warm_start(bodies, constraints);
    for _ in 0..iterations {
        velocity_iteration(bodies, constraints, joints, rolling, inv_dt);
    }
    split_impulse_pass(bodies, constraints, iterations, inv_dt);
}

/// Re-apply the impulses accumulated by the previous step (Bullet's
/// `SOLVER_USE_WARMSTARTING` pass), so resting contacts reach a converged
/// solution in a few iterations instead of rebuilding from zero.
pub(crate) fn warm_start(bodies: &mut [RigidBody], constraints: &mut [ContactConstraint]) {
    for c in constraints.iter_mut() {
        let (a, b) = two_bodies(bodies, c.body_a, c.body_b);
        apply_impulse(a, b, c, Dir::Normal, c.normal_impulse);
        apply_impulse(a, b, c, Dir::Tangent1, c.tangent1_impulse);
        apply_impulse(a, b, c, Dir::Tangent2, c.tangent2_impulse);
    }
}

/// One PGS sweep over the rigid contact and joint rows.
pub(crate) fn velocity_iteration(
    bodies: &mut [RigidBody],
    constraints: &mut [ContactConstraint],
    joints: &mut [ConstraintRow],
    rolling: &mut [RollingFriction],
    inv_dt: Real,
) {
    for c in constraints.iter_mut() {
        solve_contact(bodies, c, rolling, inv_dt);
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
                apply_push(a, b, c, j);
            }
        }
    }
}

/// Accumulate a split-impulse positional push onto the bodies' push velocities
/// (Bullet's `internalApplyPushImpulse`): the push runs along the row normal
/// only, so it reuses the pre-folded normal angular components.
#[inline]
fn apply_push(a: &mut RigidBody, b: &mut RigidBody, c: &ContactConstraint, j: Real) {
    if c.applies_a {
        a.push_linear -= c.normal * (j * c.inv_mass_a);
        a.push_angular -= c.ang_normal_a * j;
    }
    if c.applies_b {
        b.push_linear += c.normal * (j * c.inv_mass_b);
        b.push_angular += c.ang_normal_b * j;
    }
}

fn solve_contact(
    bodies: &mut [RigidBody],
    c: &mut ContactConstraint,
    rolling: &mut [RollingFriction],
    inv_dt: Real,
) {
    // One split serves the whole solve (the ten-iteration hot path used to
    // re-split for every tangent block — pure control-flow overhead).
    let (a, b) = two_bodies(bodies, c.body_a, c.body_b);

    let va = a.linear_velocity + a.angular_velocity.cross(c.r_a);
    let vb = b.linear_velocity + b.angular_velocity.cross(c.r_b);
    let mut rel = vb - va;
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
        bias += BAUMGARTE * inv_dt * c.baumgarte_span;
    }
    // Separated (margin-band) contacts: Bullet's solver subtracts the
    // separation rate from the velocity error (`velocityError -= penetration
    // * invTimeStep` where `penetration = distance + linearSlop`; Bullet's
    // default slop is 0, so only genuinely separated points soften). Our
    // engine-wide slop stays a dead zone between separation and the
    // Baumgarte threshold. Both terms are iteration-invariant and
    // precomputed in `prepare`.
    if c.separation < 0.0 {
        bias += c.separation * inv_dt;
    }

    // Normal impulse, clamped to be non-negative.
    let lambda = c.normal_mass * (bias - vn);
    let old = c.normal_impulse;
    c.normal_impulse = (old + lambda).max(0.0);
    let jn = c.normal_impulse - old;

    apply_impulse(a, b, c, Dir::Normal, jn);
    rel += c.kn * jn;

    // Friction: two tangent directions, clamped by Coulomb's law. Each
    // block projects the *incrementally advanced* `rel` (advanced by the
    // previous block's impulse through the pre-folded couplings) instead
    // of re-reading both bodies' velocities — same sequential-impulse
    // semantics, one cross-pair saved per tangent block.
    let max_friction = c.friction * c.normal_impulse;

    // Tangent 1.
    {
        let vt = rel.dot(c.tangent1);
        let lambda = -c.tangent1_mass * vt;
        let old = c.tangent1_impulse;
        let new = (old + lambda).clamp(-max_friction, max_friction);
        c.tangent1_impulse = new;
        let j = new - old;
        apply_impulse(a, b, c, Dir::Tangent1, j);
        rel += c.kt1 * j;
    }

    // Tangent 2.
    {
        let vt = rel.dot(c.tangent2);
        let lambda = -c.tangent2_mass * vt;
        let old = c.tangent2_impulse;
        let new = (old + lambda).clamp(-max_friction, max_friction);
        c.tangent2_impulse = new;
        let j = new - old;
        apply_impulse(a, b, c, Dir::Tangent2, j);
        rel += c.kt2 * j;
    }

    // Rolling friction (Bullet's rolling-pool pass, solved after the
    // friction rows): two pure-angular rows driven toward zero relative
    // angular velocity, clamped by `min(friction · normal_impulse,
    // friction)` and skipped entirely while the contact carries no normal
    // impulse. The rows read the absolute angular velocities directly
    // (this solver has no delta-velocity buffers) and the linear part of
    // the jacobian is zero, so the incremental-`rel` machinery above does
    // not apply.
    if c.normal_impulse > 0.0
        && let Some(index) = c.rolling
    {
        let rf = &mut rolling[index as usize];
        let magnitude = (rf.friction * c.normal_impulse).min(rf.friction);
        for i in 0..2 {
            let ang_rel =
                rf.axis[i].dot(b.angular_velocity) - rf.axis[i].dot(a.angular_velocity);
            let lambda = rf.mass[i] * (0.0 - ang_rel);
            let old = rf.impulse[i];
            let new = (old + lambda).clamp(-magnitude, magnitude);
            rf.impulse[i] = new;
            let j = new - old;
            if j != 0.0 {
                a.angular_velocity -= rf.comp_a[i] * j;
                b.angular_velocity += rf.comp_b[i] * j;
            }
        }
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

/// Which row direction an impulse acts along (selects the pre-folded angular
/// component and the direction vector).
#[derive(Clone, Copy)]
enum Dir {
    Normal,
    Tangent1,
    Tangent2,
}

/// Apply an impulse `dir · j` to both bodies: the linear parts scale the
/// direction by `j · inv_mass`, the angular parts use the row's pre-folded
/// component (`ω -= comp · j`), matching Bullet's
/// `resolveSingleConstraintRowGeneric` shape. Inlined so the three
/// per-constraint call sites fold the `Dir` match at compile time and keep
/// body velocities in registers across the normal/tangent blocks.
#[inline]
fn apply_impulse(
    a: &mut RigidBody,
    b: &mut RigidBody,
    c: &ContactConstraint,
    dir: Dir,
    j: Real,
) {
    let (dir_vec, comp_a, comp_b) = match dir {
        Dir::Normal => (c.normal, c.ang_normal_a, c.ang_normal_b),
        Dir::Tangent1 => (c.tangent1, c.ang_tangent1_a, c.ang_tangent1_b),
        Dir::Tangent2 => (c.tangent2, c.ang_tangent2_a, c.ang_tangent2_b),
    };
    if c.applies_a {
        a.linear_velocity -= dir_vec * (j * c.inv_mass_a);
        a.angular_velocity -= comp_a * j;
    }
    if c.applies_b {
        b.linear_velocity += dir_vec * (j * c.inv_mass_b);
        b.angular_velocity += comp_b * j;
    }
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
        solve(&mut bodies, &mut constraints, &mut [], &mut [], 10, 60.0);
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
        solve(&mut bodies, &mut constraints, &mut [], &mut [], 10, 60.0);
        assert!(!constraints[0].split_active);
        assert_eq!(bodies[1].push_linear, Vector3::ZERO);
    }

    /// The split impulse is clamped below at zero (never pulls bodies into
    /// the surface) and converges to the penetration target over iterations.
    #[test]
    fn split_impulse_converges_and_clamps() {
        let (mut bodies, mut constraints) = deep_contact(0.1);
        // Half the iterations of the default solve.
        solve(&mut bodies, &mut constraints, &mut [], &mut [], 5, 60.0);
        let push_full = bodies[1].push_linear.y;
        let (mut bodies2, mut constraints2) = deep_contact(0.1);
        solve(&mut bodies2, &mut constraints2, &mut [], &mut [], 20, 60.0);
        // More iterations converge further toward removing the penetration;    
        // both stay positive (no pull-in).
        assert!(push_full > 0.0);
        assert!(bodies2[1].push_linear.y >= push_full);
    }

    /// A rolling sphere on a plane with a positive combined rolling
    /// friction: the torsional rows drive the relative angular velocity
    /// along the tangent-plane axes toward zero, so the spin decays; with
    /// the coefficient at zero the spin is untouched (the default path).
    #[test]
    fn rolling_friction_rows_decay_spin() {
        // Ground plane (body A, static) + sphere r = 0.5 (body B), contact
        // at the bottom, normal A -> B = +Y. The sphere spins at
        // `ω = (0, 0, 2)` around the tangent-plane axis `btPlaneSpace1(Y)`
        // picks as its second axis.
        let build = |rolling: Real| {
            let mut ground = RigidBody::new_static(
                CollisionShape::plane(Vector3::Y, 0.0),
                Vector3::ZERO,
            );
            ground.rolling_friction = rolling;
            let mut sphere = RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            );
            sphere.rolling_friction = rolling;
            sphere.angular_velocity = Vector3::new(0.0, 0.0, 2.0);
            let cp = bullet_collision::manifold::ContactPoint::new(
                Vector3::new(0.0, 0.5, 0.0),
                Vector3::Y,
                0.01,
            );
            let mut c = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX);
            let mut pool: Vec<RollingFriction> = Vec::new();
            c.attach_rolling(&ground, &sphere, &mut pool);
            (vec![ground, sphere], vec![c], pool)
        };

        // Without rolling friction the rows are never installed and the
        // spin stays exactly as it started.
        let (mut bodies, mut constraints, mut pool) = build(0.0);
        solve(&mut bodies, &mut constraints, &mut [], &mut pool, 10, 60.0);
        assert!(pool.is_empty());
        assert_relative_eq!(bodies[1].angular_velocity.z, 2.0, epsilon = 1e-6);

        // With a positive coefficient the spin decays. The rows only
        // receive load while the contact carries a normal impulse; one
        // solve's Baumgarte correction charges it once, so the spin drops
        // by exactly `min(friction · normal_impulse, friction) · invI` —
        // 0.03 · 10 = 0.3 here (the rows saturate their clamp). The
        // full deceleration under sustained contact load is covered by
        // the scene test in `world_tests`.
        let (mut bodies, mut constraints, mut pool) = build(0.5);
        solve(&mut bodies, &mut constraints, &mut [], &mut pool, 10, 60.0);
        assert_eq!(pool.len(), 1);
        let w = bodies[1].angular_velocity.z;
        assert_relative_eq!(w, 1.7, epsilon = 1e-5);
        assert!(pool[0].impulse[1].abs() <= 0.03 + 1e-6);
    }

    /// The per-row impulse clamp: a tiny combined coefficient caps the
    /// accumulated angular impulse at the coefficient itself, so a fast
    /// spin only slows slightly per solve.
    #[test]
    fn rolling_friction_clamps_impulse() {
        let ground = RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        );
        let mut sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.5, 0.0),
            1.0,
        );
        // Combined = rf_a·μ_b + rf_b·μ_a = 0.01·0.5 + 0 = 0.005.
        sphere.rolling_friction = 0.01;
        sphere.angular_velocity = Vector3::new(0.0, 0.0, 100.0);
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::Y,
            0.01,
        );
        let mut c = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX);
        let mut pool: Vec<RollingFriction> = Vec::new();
        c.attach_rolling(&ground, &sphere, &mut pool);
        let mut bodies = vec![ground, sphere];
        solve(&mut bodies, std::slice::from_mut(&mut c), &mut [], &mut pool, 10, 60.0);
        // |ΔL| per row ≤ combined (0.005); I = 0.1 → |Δω| ≤ 0.05.
        let dw = (100.0 - bodies[1].angular_velocity.z).abs();
        assert!(dw <= 0.06, "clamp should cap the change: Δω = {}", dw);
    }
}

