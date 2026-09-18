//! Sequential-impulse constraint solver: contact constraints with friction and
//! restitution, adapted from Bullet's `btSequentialImpulseConstraintSolver`.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bullet_collision::manifold::ContactPoint;
use crate::body::RigidBody;
use crate::CollisionFlags;
use crate::solver::SolverInfo;
use bullet_math::{math, FRAC_1_SQRT_2, Matrix3, Real, Vector3};

/// Bullet's `SIMD_EPSILON` — precision-dependent (`f32::EPSILON` /
/// `f64::EPSILON`, exactly FLT/DBL_EPSILON).
const SIMD_EPSILON: Real = Real::EPSILON;

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
    /// Friction-anchor targets per tangent axis (Bullet's
    /// `BT_CONTACT_FLAG_FRICTION_ANCHOR` branch of
    /// `setupFrictionConstraint`): `-((wa - wb)·tangent) · m_frictionERP`,
    /// built once at `prepare` from the cached world anchors of the
    /// persistent-manifold point (Bullet reads the refreshed
    /// `m_positionWorldOnA/B` at row setup). The solve multiplies by
    /// `inv_dt` to form the positional bias that pulls the pair's world
    /// anchors back together along the tangent plane. Both zeros unless a
    /// pair side carries [`CollisionFlags::HAS_FRICTION_ANCHOR`]; a zero
    /// target leaves the friction row exactly as before (`vt - 0.0 == vt`
    /// bit-for-bit).
    anchor: [Real; 2],
    /// Soft-contact channel of the normal row (Bullet's
    /// `setupContactConstraint` `BT_CONTACT_FLAG_CONTACT_STIFFNESS_DAMPING`
    /// branch). `erp` is the position-error factor: [`BAUMGARTE`] (Bullet's
    /// `m_erp2` default) unless a pair side carries
    /// [`CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING`], then the
    /// combined-stiffness/damping `m_contactERP2` swap. `normal_cfm` is the
    /// compliance term (`m_cfm * m_jacDiagABInv`) that pulls the impulse
    /// delta back proportionally to the accumulated impulse. The defaults
    /// leave the solve bit-for-bit unchanged (`0.0 · x == +0.0` for the
    /// non-negative impulses).
    erp: Real,
    normal_cfm: Real,
}

/// The torsional-friction rows of one contact point (Bullet's
/// `setupTorsionalFrictionConstraint` trio installed inside the
/// rolling-friction branch of `convertContact`): pure-angular constraints
/// that drive the contact's relative angular velocity toward zero. Rows 1
/// and 2 sit on the two tangent-plane axes with the combined rolling
/// coefficient; row 0 spins around the contact normal with the combined
/// spinning coefficient, like Bullet's spinning-friction row (which his
/// own assembly site adds unconditionally inside the rolling branch, so
/// it only ever coexists with rolling friction). Each row's accumulated
/// impulse is clamped per iteration to
/// `±min(friction · normal_impulse, friction)`, so rolling or spinning
/// bodies only decelerate while the contact actually carries load. The
/// rows are rebuilt from zero every step (Bullet's rolling pool stores
/// `m_originalContactPoint = 0`: no warm start) and never touch the
/// linear velocity or the split-impulse pass.
pub struct RollingFriction {
    /// The row axes in install order: `[normal, tangent0, tangent1]`
    /// (Bullet's spinning row first, then the `btPlaneSpace1(normal)`
    /// output, re-normalized like the assembly site does).
    axis: [Vector3; 3],
    /// Pre-folded angular impulse components `invI · axis` per side. A
    /// row impulse applies `ω_a -= comp_a · j` and `ω_b += comp_b · j`,
    /// matching Bullet's `angularComponentA/B` signs for its
    /// `relpos1CrossNormal = -axis` / `relpos2CrossNormal = +axis` rows.
    comp_a: [Vector3; 3],
    comp_b: [Vector3; 3],
    /// Inverse effective mass per row: `1 / (axis·invI_a·axis +
    /// axis·invI_b·axis)` — a pure-rotation row, no linear term.
    mass: [Real; 3],
    /// Accumulated impulses of the current step (reset by `prepare`).
    impulse: [Real; 3],
    /// The per-row clamp scale `[spinning, rolling, rolling]` (the
    /// combined coefficient each row was built with).
    friction: [Real; 3],
    /// Whether the spinning row (row 0) is active: Bullet installs it
    /// unconditionally inside the rolling branch, but a non-positive
    /// coefficient gives it a zero/inverted window that clamps its
    /// impulse to a no-op (or worse, upstream): skipping the row is
    /// bit-identical for every sane input.
    spinning: bool,
}

impl RollingFriction {
    /// Build the rows for a contact with the given normal.
    pub(crate) fn prepare(
        normal: Vector3,
        inv_inertia_a: Matrix3,
        inv_inertia_b: Matrix3,
        friction: Real,
        spinning: Real,
    ) -> Self {
        let (a0, a1) = plane_space(normal);
        // The assembly site normalizes both axes (Bullet's `axis.normalize()`);
        // the `length() > 0.001` gate after it only exists for the anisotropic
        // filter this crate does not have, so both axes are always unit here.
        let tangent = [a0 / a0.length(), a1 / a1.length()];
        let axis = [normal, tangent[0], tangent[1]];
        let mut comp_a = [Vector3::ZERO; 3];
        let mut comp_b = [Vector3::ZERO; 3];
        let mut mass = [0.0; 3];
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
            impulse: [0.0; 3],
            friction: [spinning, friction, friction],
            spinning: spinning > 0.0,
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
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        body_a: usize,
        body_b: usize,
        cp: &ContactPoint,
        a: &RigidBody,
        b: &RigidBody,
        point_index: u32,
        anchor_disp: Option<Vector3>,
        inv_dt: Real,
        info: &SolverInfo,
    ) -> Self {
        let normal = cp.normal;
        let (mut tangent1, mut tangent2) = tangent_basis(normal);
        // Anisotropic friction (Bullet's `applyAnisotropicFriction`, applied
        // by `convertContact` to both sides after the friction directions
        // are fixed): rotate the tangent into the body's local frame, scale
        // it component-wise, rotate it back. Each side applies to each axis,
        // matching the `btPlaneSpace1` branch (this engine's friction basis
        // is fixed, so there is no velocity-dependent branch to mirror).
        if is_anisotropic(a) || is_anisotropic(b) {
            tangent1 = apply_anisotropic_friction(tangent1, a);
            tangent1 = apply_anisotropic_friction(tangent1, b);
            tangent2 = apply_anisotropic_friction(tangent2, a);
            tangent2 = apply_anisotropic_friction(tangent2, b);
        }
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

        let mut normal_mass =
            effective_mass(inv_mass_a, inv_mass_b, ang_normal_a, ang_normal_b, r_a, r_b, normal);
        let mut tangent1_mass = effective_mass(
            inv_mass_a,
            inv_mass_b,
            ang_tangent1_a,
            ang_tangent1_b,
            r_a,
            r_b,
            tangent1,
        );
        let mut tangent2_mass = effective_mass(
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

        // Soft-contact channel (Bullet's `setupContactConstraint`
        // parameter selection: the per-pair stiffness/damping flag overrides
        // the global defaults). `erp` starts at Bullet's `m_erp2` and `cfm`
        // at `m_globalCfm * invTimeStep` (`cfm *= invTimeStep` at
        // btSequentialImpulseConstraintSolver.cpp:846, before it enters the
        // effective-mass denominator); the stiffness branch replaces both
        // with the combined `m_contactERP2` / spring formulas. The relaxation
        // numerator `m_sor` enters both branches' `m_jacDiagABInv`. The
        // defaults (`erp2 0.2`, `globalCfm 0`, `sor 1`) leave the original
        // values bit-for-bit: `k + 0.0 == k`, `1.0 / x` is the same division,
        // and a zero `cfm` keeps the compliance solve path gated off.
        let mut erp = info.erp2;
        let mut normal_cfm = 0.0;
        let cfm_global = info.global_cfm * inv_dt;
        if a.collision_flags.contains(CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING)
            || b.collision_flags.contains(CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING)
        {
            let combined_damping = a.contact_damping() + b.contact_damping();
            let combined_stiffness =
                1.0 / (1.0 / a.contact_stiffness() + 1.0 / b.contact_stiffness());
            let dt = inv_dt.recip();
            let denom_raw = dt * combined_stiffness + combined_damping;
            let denom = if denom_raw < SIMD_EPSILON {
                SIMD_EPSILON
            } else {
                denom_raw
            };
            let cfm = (1.0 / denom) * inv_dt;
            erp = (dt * combined_stiffness) / denom;
            // The row's `k` (inverse of `normal_mass`) gains the compliance:
            // `m_jacDiagABInv = relaxation / (k + cfm)`.
            let k = if normal_mass > 0.0 { 1.0 / normal_mass } else { 0.0 };
            normal_mass = if k + cfm > 0.0 { info.sor / (k + cfm) } else { 0.0 };
            normal_cfm = cfm * normal_mass;
        } else if cfm_global != 0.0 || info.sor != 1.0 {
            // Global compliance / relaxation on a plain contact row: the
            // same `m_jacDiagABInv = relaxation / (k + cfm)` shape over the
            // recovered `k` (the defaults skip this branch entirely, keeping
            // the effective mass exactly as `effective_mass` returned it).
            let k = if normal_mass > 0.0 { 1.0 / normal_mass } else { 0.0 };
            normal_mass = if k + cfm_global > 0.0 {
                info.sor / (k + cfm_global)
            } else {
                0.0
            };
            normal_cfm = cfm_global * normal_mass;
        }
        if info.sor != 1.0 {
            // The linear friction rows carry the relaxation numerator too
            // (`addFrictionConstraint` receives the same `relaxation`; the
            // normal row already folded it in above). The multiplicative
            // form differs from Bullet's division only in the last ulp of
            // the non-default regime.
            tangent1_mass *= info.sor;
            tangent2_mass *= info.sor;
        }

        // Friction-anchor targets (Bullet's `setupFrictionConstraint`
        // `BT_CONTACT_FLAG_FRICTION_ANCHOR` branch): the cached world
        // anchors' separation along each tangent axis, scaled by
        // `m_frictionERP` (the `inv_dt` factor is applied at solve time,
        // matching Bullet's `rhs` assembly order). The sign follows from
        // the row conventions: Bullet's friction `rel_vel` is
        // `(v0 - vB) . t` — the negation of ours — and its servo target
        // `-distance * m_frictionERP / dt` there becomes `+distance *
        // m_frictionERP / dt` on our rows, driving the separation
        // `distance` exponentially to zero ("pin the anchors together").
        // `None` keeps both targets at exactly zero, leaving the friction
        // solve bit-for-bit unchanged.
        let anchor = if let Some(disp) = anchor_disp {
            [
                disp.dot(tangent1) * info.friction_erp,
                disp.dot(tangent2) * info.friction_erp,
            ]
        } else {
            [0.0; 2]
        };

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
            baumgarte_span: (cp.penetration - info.linear_slop).max(0.0),
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
            // bias to the split-impulse pass (Bullet's `!m_splitImpulse ||
            // penetration > m_splitImpulsePenetrationThreshold` branch of
            // `setupContactConstraint`); the threshold keeps Bullet's negative
            // sign and is compared against the positive penetration depth. The
            // split impulse itself starts at zero every step, like
            // `m_appliedPushImpulse`.
            split_active: info.split_impulse
                && cp.penetration > -info.split_impulse_penetration_threshold,
            rhs_penetration: 0.0,
            push_impulse: 0.0,
            applies_a: inv_mass_a > 0.0,
            applies_b: inv_mass_b > 0.0,
            kn,
            kt1,
            kt2,
            rolling: None,
            anchor,
            erp,
            normal_cfm,
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
        // The spinning coefficient combines with the same quirk formula
        // (Bullet's `calculateCombinedSpinningFriction`) and its row is
        // installed inside this same branch (Bullet's `convertContact`),
        // so a pair with a positive spinning value but no rolling value
        // gets no spinning row either — the upstream gate kept verbatim.
        let combined_spinning = (a.spinning_friction * b.friction
            + b.spinning_friction * a.friction)
            .clamp(-10.0, 10.0);
        let rows = RollingFriction::prepare(
            self.normal,
            a.inv_inertia_world(),
            b.inv_inertia_world(),
            combined,
            combined_spinning,
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

/// Whether the body's friction rows get direction-dependent scaling
/// (Bullet's `setAnisotropicFriction` unity gate: any component other
/// than 1 switches the effect on).
fn is_anisotropic(body: &RigidBody) -> bool {
    let s = body.anisotropic_friction;
    s.x != 1.0 || s.y != 1.0 || s.z != 1.0
}

/// Bullet's `applyAnisotropicFriction`: rotate the friction direction into
/// the body's local frame (`basisᵀ · dir`, the conjugate quaternion),
/// scale it component-wise by `anisotropic_friction`, and rotate it back
/// (`basis · dir`). The quaternion path differs from Bullet's matrix product
/// in the last ulp only; parity scenes treat that as tolerance noise.
fn apply_anisotropic_friction(dir: Vector3, body: &RigidBody) -> Vector3 {
    let local = (body.orientation.conjugate() * dir) * body.anisotropic_friction;
    body.orientation * local
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
/// `SOLVER_USE_WARMSTARTING`); joints build from zero each step. The
/// iteration count and the per-row parameters come from `info`
/// (`btContactSolverInfo`).
pub fn solve(
    bodies: &mut [RigidBody],
    constraints: &mut [ContactConstraint],
    joints: &mut [ConstraintRow],
    rolling: &mut [RollingFriction],
    inv_dt: Real,
    info: &SolverInfo,
) {
    warm_start(bodies, constraints);
    for _ in 0..info.num_iterations {
        velocity_iteration(bodies, constraints, joints, rolling, inv_dt, info);
    }
    split_impulse_pass(bodies, constraints, inv_dt, info);
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
    info: &SolverInfo,
) {
    for c in constraints.iter_mut() {
        solve_contact(bodies, c, rolling, inv_dt, info);
    }
    let damping = info.damping;
    for j in joints.iter_mut() {
        solve_joint_row(bodies, j, damping);
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
    inv_dt: Real,
    info: &SolverInfo,
) {
    let split_rhs = |c: &mut ContactConstraint| {
        if c.split_active {
            // The split pass shares the row's position-error factor (Bullet
            // uses the same `erp` for `penetrationImpulse` in both branches).
            c.rhs_penetration = c.penetration * c.erp * inv_dt * c.normal_mass;
        }
    };
    for c in constraints.iter_mut() {
        split_rhs(c);
    }
    for _ in 0..info.num_iterations {
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
            // `delta = rhs - push_vn * mass - cfm * push` matches Bullet's
            // `gResolveSplitPenetrationImpulse_scalar_reference` with our
            // normal convention (A -> B); the compliance branch is only
            // taken with the stiffness flag set.
            let delta = if c.normal_cfm != 0.0 {
                c.rhs_penetration - push_vn * c.normal_mass - c.normal_cfm * c.push_impulse
            } else {
                c.rhs_penetration - push_vn * c.normal_mass
            };
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
    info: &SolverInfo,
) {
    // One split serves the whole solve (the ten-iteration hot path used to
    // re-split for every tangent block — pure control-flow overhead).
    let (a, b) = two_bodies(bodies, c.body_a, c.body_b);
    // The soft-contact channel is opt-in; the branch keeps the default path
    // on the original instruction sequence (no extra loads or multiplies).
    let soft = c.normal_cfm != 0.0;

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
    if vn < -info.restitution_velocity_threshold {
        bias = -c.restitution * vn;
    }
    if !c.split_active {
        // The Baumgarte factor lives on the row (`erp`, loaded from
        // `m_erp2` at prepare time — the soft-contact stiffness branch is
        // its only override), so both paths read the same field.
        bias += c.erp * inv_dt * c.baumgarte_span;
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

    // Normal impulse, clamped to be non-negative. The compliance term is
    // Bullet's `- m_appliedImpulse * m_cfm` (`resolveSingleConstraint`
    // family); the branch is only taken when a pair side carries
    // `CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING`.
    let lambda = if soft {
        c.normal_mass * (bias - vn) - c.normal_cfm * c.normal_impulse
    } else {
        c.normal_mass * (bias - vn)
    };
    let old = c.normal_impulse;
    c.normal_impulse = (old + lambda).max(0.0);
    let jn = c.normal_impulse - old;

    apply_impulse(a, b, c, Dir::Normal, jn);
    rel += c.kn * jn;

    // Friction: two tangent directions, clamped by Coulomb's law. Each
    // block projects the *incrementally advanced* `rel` (advanced by the
    // previous block's impulse through the pre-folded couplings) instead
    // of re-reading both bodies' velocities — same sequential-impulse
    // semantics, one cross-pair saved per tangent block. The anchor term
    // is Bullet's friction-anchor positional bias (`positionalError =
    // -distance * m_frictionERP / dt` on Bullet's rows, `+distance *
    // m_frictionERP / dt` on ours — see `prepare`); for unanchored
    // contacts it is zero and `vt - 0.0 == vt` bit-for-bit.
    let max_friction = c.friction * c.normal_impulse;

    // Tangent 1.
    {
        let vt = rel.dot(c.tangent1);
        let lambda = -c.tangent1_mass * (vt - c.anchor[0] * inv_dt);
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
        let lambda = -c.tangent2_mass * (vt - c.anchor[1] * inv_dt);
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
        // Row 0 is the spinning row; without a positive combined spinning
        // coefficient the sweep starts at the two tangent-plane rows.
        for i in usize::from(!rf.spinning)..3 {
            let magnitude = (rf.friction[i] * c.normal_impulse).min(rf.friction[i]);
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
/// clamping the accumulated impulse to the row limits). `damping` is Bullet's
/// `info2.m_damping` (global default 1.0, bit-for-bit neutral): the row's
/// relative velocity is scaled by it before entering the impulse.
fn solve_joint_row(bodies: &mut [RigidBody], c: &mut ConstraintRow, damping: Real) {
    let (a, b) = two_bodies(bodies, c.body_a, c.body_b);

    let row_vel = (c.lin_b.dot(b.linear_velocity) + c.ang_b.dot(b.angular_velocity))
        - (c.lin_a.dot(a.linear_velocity) + c.ang_a.dot(a.angular_velocity));

    // The damping multiply is branch-isolated so the default path keeps its
    // exact instruction sequence (`x * 1.0` would be a live multiply).
    let lambda = if damping != 1.0 {
        c.mass * (c.bias - row_vel * damping)
    } else {
        c.mass * (c.bias - row_vel)
    };
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
        deep_contact_with(penetration, &SolverInfo::default())
    }

    /// [`deep_contact`] with an explicit solver info (tests that flip
    /// `split_impulse` need the row built under the same info).
    fn deep_contact_with(penetration: Real, info: &SolverInfo) -> (Vec<RigidBody>, Vec<ContactConstraint>) {
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
        let c = ContactConstraint::prepare(0, 1, &cp, &ground, &cube, u32::MAX, None, 60.0, info);
        (vec![ground, cube], vec![c])
    }

    /// The friction-anchor servo (Bullet's `BT_CONTACT_FLAG_FRICTION_ANCHOR`
    /// branch): with the pair's cached world anchors separated along a
    /// tangent axis, one iteration drives body B toward body A's anchor —
    /// the separation `distance` decays exponentially to zero — while a
    /// plain friction row (`anchor_disp = None`) leaves resting bodies
    /// untouched. The servo is directional: an anchor displacement along
    /// +X moves B along +X (toward the displaced A-side anchor).
    #[test]
    fn friction_anchor_servo_pulls_body_b_toward_body_a_s_anchor() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let make_cube = || {
            RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            )
        };
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, 0.0, 0.0), // box bottom / plane top
            Vector3::Y,
            0.0,
        );
        // Normal +Y: `tangent_basis` yields t1 along -Z and t2 along -X, so
        // a displacement of +X*d reads `distance = -d` on t2 and the servo
        // target velocity is negative along t2 — i.e. B moves toward +X,
        // closing the gap to the anchor (which sits +X of B's).
        let d = 0.02;
        let anchored = ContactConstraint::prepare(
            0,
            1,
            &cp,
            &ground,
            &make_cube(),
            u32::MAX,
            Some(Vector3::new(d, 0.0, 0.0)),
            60.0,
            &SolverInfo::default(),
        );
        let mut bodies = vec![ground, make_cube()];
        // Approach speed along the normal: the first iteration builds a
        // normal impulse (`max_friction = friction · normal_impulse`) so
        // the friction cone has slack for the servo to spend.
        bodies[1].linear_velocity = Vector3::new(0.0, -1.0, 0.0);
        let mut constraints = vec![anchored];
        let info = SolverInfo { num_iterations: 1, ..SolverInfo::default() };
solve(&mut bodies, &mut constraints, &mut [], &mut [], 60.0, &info);
        assert!(
            bodies[1].linear_velocity.x > 0.01,
            "servo must pull B toward the A-side anchor: vx = {}",
            bodies[1].linear_velocity.x
        );

        // Without the anchor displacement the plain friction row only
        // zeroes the tangential slip (which is zero here) — no drive.
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let plain = ContactConstraint::prepare(0, 1, &cp, &ground, &make_cube(), u32::MAX, None, 60.0, &SolverInfo::default());
        let mut bodies = vec![ground, make_cube()];
        bodies[1].linear_velocity = Vector3::new(0.0, -1.0, 0.0);
        let mut constraints = vec![plain];
        let info = SolverInfo { num_iterations: 1, ..SolverInfo::default() };
solve(&mut bodies, &mut constraints, &mut [], &mut [], 60.0, &info);
        assert_relative_eq!(bodies[1].linear_velocity.x, 0.0, epsilon = 1e-7);
    }

    /// Deep penetration (> threshold): the split pass pushes the body out of
    /// the surface (push velocity up) without touching the real velocity.
    #[test]
    fn split_impulse_pushes_deep_penetration_without_touching_velocity() {
        let (mut bodies, mut constraints) = deep_contact(0.1);
        solve(&mut bodies, &mut constraints, &mut [], &mut [], 60.0, &SolverInfo::default());
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
        solve(&mut bodies, &mut constraints, &mut [], &mut [], 60.0, &SolverInfo::default());
        assert!(!constraints[0].split_active);
        assert_eq!(bodies[1].push_linear, Vector3::ZERO);
    }

    /// The split impulse is clamped below at zero (never pulls bodies into
    /// the surface) and converges to the penetration target over iterations.
    #[test]
    fn split_impulse_converges_and_clamps() {
        let (mut bodies, mut constraints) = deep_contact(0.1);
        // Half the iterations of the default solve.
        let info = SolverInfo { num_iterations: 5, ..SolverInfo::default() };
solve(&mut bodies, &mut constraints, &mut [], &mut [], 60.0, &info);
        let push_full = bodies[1].push_linear.y;
        let (mut bodies2, mut constraints2) = deep_contact(0.1);
        let info = SolverInfo { num_iterations: 20, ..SolverInfo::default() };
solve(&mut bodies2, &mut constraints2, &mut [], &mut [], 60.0, &info);
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
            let mut c = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &SolverInfo::default());
            let mut pool: Vec<RollingFriction> = Vec::new();
            c.attach_rolling(&ground, &sphere, &mut pool);
            (vec![ground, sphere], vec![c], pool)
        };

        // Without rolling friction the rows are never installed and the
        // spin stays exactly as it started.
        let (mut bodies, mut constraints, mut pool) = build(0.0);
        solve(&mut bodies, &mut constraints, &mut [], &mut pool, 60.0, &SolverInfo::default());
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
        solve(&mut bodies, &mut constraints, &mut [], &mut pool, 60.0, &SolverInfo::default());
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
        let mut c = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &SolverInfo::default());
        let mut pool: Vec<RollingFriction> = Vec::new();
        c.attach_rolling(&ground, &sphere, &mut pool);
        let mut bodies = vec![ground, sphere];
        solve(&mut bodies, std::slice::from_mut(&mut c), &mut [], &mut pool, 60.0, &SolverInfo::default());
        // |ΔL| per row ≤ combined (0.005); I = 0.1 → |Δω| ≤ 0.05.
        let dw = (100.0 - bodies[1].angular_velocity.z).abs();
        assert!(dw <= 0.06, "clamp should cap the change: Δω = {}", dw);
    }

    /// The spinning-friction row (Bullet's third torsional row, around the
    /// contact normal): a sphere spinning about the normal decays while the
    /// tangent-plane rolling rows see no motion (the spin is orthogonal to
    /// them), and the spin drop matches the same clamp formula scaled by
    /// the combined spinning coefficient.
    #[test]
    fn spinning_friction_row_decays_normal_axis_spin() {
        let build = || {
            let mut ground = RigidBody::new_static(
                CollisionShape::plane(Vector3::Y, 0.0),
                Vector3::ZERO,
            );
            ground.rolling_friction = 0.5;
            ground.spinning_friction = 0.5;
            let mut sphere = RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            );
            sphere.rolling_friction = 0.5;
            sphere.spinning_friction = 0.5;
            // Spin about the contact normal (+Y): only the spinning row
            // sees it (`plane_space(Y)` tangent axes are in the X-Z plane).
            sphere.angular_velocity = Vector3::new(0.0, 2.0, 0.0);
            let cp = bullet_collision::manifold::ContactPoint::new(
                Vector3::new(0.0, 0.5, 0.0),
                Vector3::Y,
                0.01,
            );
            let c =
                ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &SolverInfo::default());
            let mut pool: Vec<RollingFriction> = Vec::new();
            let mut c = c;
            c.attach_rolling(&ground, &sphere, &mut pool);
            (vec![ground, sphere], vec![c], pool)
        };

        let (mut bodies, mut constraints, mut pool) = build();
        solve(&mut bodies, &mut constraints, &mut [], &mut pool, 60.0, &SolverInfo::default());
        assert_eq!(pool.len(), 1);
        assert!(pool[0].spinning);
        // Same math as the rolling test: the row saturates its clamp at
        // `min(spinning · normal_impulse, spinning) = 0.03`, invI = 10,
        // so one solve drops the spin by 0.3.
        assert_relative_eq!(bodies[1].angular_velocity.y, 1.7, epsilon = 1e-5);
        assert!(pool[0].impulse[0].abs() <= 0.03 + 1e-6);
        // The tangent-plane rows stayed idle (spin orthogonal to them).
        assert_eq!(pool[0].impulse[1], 0.0);
        assert_eq!(pool[0].impulse[2], 0.0);
    }

    /// Bullet's `convertContact` installs the spinning row inside the
    /// rolling-friction branch: without a positive combined rolling value
    /// no torsional row is built at all, so a spinning-only pair keeps
    /// spinning forever (the upstream gate kept verbatim).
    #[test]
    fn spinning_friction_needs_the_rolling_friction_branch() {
        let mut ground = RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        );
        ground.spinning_friction = 0.5;
        let mut sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.5, 0.0),
            1.0,
        );
        sphere.spinning_friction = 0.5;
        sphere.angular_velocity = Vector3::new(0.0, 2.0, 0.0);
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::Y,
            0.01,
        );
        let c = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &SolverInfo::default());
        let mut pool: Vec<RollingFriction> = Vec::new();
        let mut c = c;
        c.attach_rolling(&ground, &sphere, &mut pool);
        let mut bodies = vec![ground, sphere];
        solve(&mut bodies, std::slice::from_mut(&mut c), &mut [], &mut pool, 60.0, &SolverInfo::default());
        assert!(pool.is_empty());
        assert_relative_eq!(bodies[1].angular_velocity.y, 2.0, epsilon = 1e-6);
    }

    /// The soft-contact channel (Bullet's
    /// `BT_CONTACT_FLAG_CONTACT_STIFFNESS_DAMPING` branch): a flagged side
    /// swaps the row's position-error factor to `dt·k/(dt·k + d)` of the
    /// combined parameters and adds the compliance CFM (into the effective
    /// mass and as `- m_cfm · appliedImpulse`), while an unflagged pair
    /// keeps the Baumgarte defaults bit-for-bit.
    #[test]
    fn contact_stiffness_swaps_erp_and_adds_compliance() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let cp = bullet_collision::manifold::ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.0);

        let mut sphere =
            RigidBody::new_dynamic(CollisionShape::sphere(0.5), Vector3::new(0.0, 0.5, 0.0), 1.0);
        sphere.set_contact_stiffness_and_damping(1000.0, 10.0);
        let soft = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &SolverInfo::default());
        // dt = 1/60, k = 1000; the flagged sphere's damping 10 *adds* the
        // plane's default 0.1 (Bullet's `calculateCombinedContactDamping`),
        // and the plane's `BT_LARGE_FLOAT` stiffness contributes ~nothing to
        // the series combination. The plane side contributes nothing to the
        // row's k either, so it stays the sphere's inv_mass 1.
        let denom = 1000.0 / 60.0 + 10.1;
        let cfm = (1.0 / denom) * 60.0;
        assert_relative_eq!(soft.erp, (1000.0 / 60.0) / denom, epsilon = 1e-12);
        assert_relative_eq!(soft.normal_mass, 1.0 / (1.0 + cfm), epsilon = 1e-12);
        assert_relative_eq!(soft.normal_cfm, cfm * soft.normal_mass, epsilon = 1e-12);

        // An unflagged pair keeps the defaults: erp = Baumgarte, no
        // compliance.
        let plain_sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.5, 0.0),
            1.0,
        );
        let plain =
            ContactConstraint::prepare(0, 1, &cp, &ground, &plain_sphere, u32::MAX, None, 60.0, &SolverInfo::default());
        assert_relative_eq!(plain.erp, 0.2, epsilon = 1e-12);
        assert_relative_eq!(plain.normal_cfm, 0.0, epsilon = 1e-12);
    }

    /// Bullet's `applyAnisotropicFriction`: the pair's tangent axes are
    /// rotated into each body's local frame, scaled component-wise, and
    /// rotated back. With the identity rotation the local scaling *is* the
    /// world scaling (a zero local-Z axis collapses the -Z tangent);
    /// a rotated body scales its local axes, not the world's.
    #[test]
    fn anisotropic_friction_rescales_the_tangent_axes() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let cp = bullet_collision::manifold::ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.0);
        let make_cube = || {
            RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            )
        };

        // Isotropic default: the tangent basis is untouched.
        let plain = ContactConstraint::prepare(
            0,
            1,
            &cp,
            &ground,
            &make_cube(),
            u32::MAX,
            None,
            60.0,
            &SolverInfo::default(),
        );
        let (t1, t2) = tangent_basis(Vector3::Y);
        assert!((plain.tangent1 - t1).length() < 1e-12);
        assert!((plain.tangent2 - t2).length() < 1e-12);

        // Zero local-Z scaling: t1 runs along -Z and collapses; t2 keeps
        // its length.
        let mut aniso = make_cube();
        aniso.anisotropic_friction = Vector3::new(1.0, 1.0, 0.0);
        let scaled = ContactConstraint::prepare(0, 1, &cp, &ground, &aniso, u32::MAX, None, 60.0, &SolverInfo::default());
        assert_relative_eq!(scaled.tangent1.length(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(scaled.tangent2.length(), 1.0, epsilon = 1e-12);

        // A 45° yaw mixes the scaling into both axes: each tangent keeps
        // only its projection onto the unscaled local plane.
        let mut rotated = make_cube();
        rotated.orientation = bullet_math::Quaternion::from_rotation_y(bullet_math::FRAC_PI_2 * 0.5);
        rotated.anisotropic_friction = Vector3::new(1.0, 1.0, 0.0);
        let turned = ContactConstraint::prepare(0, 1, &cp, &ground, &rotated, u32::MAX, None, 60.0, &SolverInfo::default());
        let expected_t1 = Vector3::new(0.5, 0.0, -0.5);
        let expected_t2 = Vector3::new(-0.5, 0.0, 0.5);
        assert!(
            (turned.tangent1 - expected_t1).length() < 1e-5,
            "t1 = {}",
            turned.tangent1
        );
        assert!(
            (turned.tangent2 - expected_t2).length() < 1e-5,
            "t2 = {}",
            turned.tangent2
        );
    }

    /// `erp2` is the contact row's Baumgarte factor: the row carries it in
    /// `erp`, so a stiffer factor lands on the row one-to-one.
    #[test]
    fn solver_info_erp2_scales_the_contact_baumgarte_bias() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.49, 0.0),
            1.0,
        );
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, -0.01, 0.0),
            Vector3::Y,
            0.01,
        );
        let info = SolverInfo::default();
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &info);
        assert_relative_eq!(row.erp, 0.2, epsilon = 1e-12);

        let stiff = SolverInfo { erp2: 0.4, ..SolverInfo::default() };
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &stiff);
        assert_relative_eq!(row.erp, 0.4, epsilon = 1e-12);
    }

    /// `linear_slop` is the dead zone before the Baumgarte term: inside the
    /// slop the span is zero; with the slop removed (Bullet's constructor
    /// value) the same depth feeds the full bias.
    #[test]
    fn solver_info_slop_sets_the_baumgarte_dead_zone() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.497, 0.0),
            1.0,
        );
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, -0.003, 0.0),
            Vector3::Y,
            0.003,
        );
        let info = SolverInfo::default();
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &info);
        assert_relative_eq!(row.baumgarte_span, 0.0, epsilon = 1e-12);

        let upstream = SolverInfo { linear_slop: 0.0, ..SolverInfo::default() };
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &upstream);
        assert_relative_eq!(row.baumgarte_span, 0.003, epsilon = 1e-12);
    }

    /// `restitution_velocity_threshold`: at the engine default (1.0) a
    /// 0.5 m/s approach bounces zero times; at the upstream default (0.2)
    /// the same approach carries restitution.
    #[test]
    fn solver_info_restitution_threshold_gates_low_speed_bounces() {
        let build = |info: &SolverInfo| {
            let mut ground = RigidBody::new_static(
                CollisionShape::plane(Vector3::Y, 0.0),
                Vector3::ZERO,
            );
            ground.restitution = 1.0;
            let mut sphere = RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            );
            sphere.restitution = 1.0;
            sphere.linear_velocity = Vector3::new(0.0, -0.5, 0.0);
            let cp = bullet_collision::manifold::ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.0);
            let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, info);
            (vec![ground, sphere], row)
        };
        // One iteration isolates the threshold from the PGS re-projection
        // (later iterations re-read the approach velocity, which has already
        // been corrected — the gate is a per-iteration bias input, like
        // Bullet's assembly-time restitutionCurve).
        let default = SolverInfo { num_iterations: 1, ..SolverInfo::default() };
        let (mut bodies, mut row) = build(&default);
        solve(&mut bodies, std::slice::from_mut(&mut row), &mut [], &mut [], 60.0, &default);
        // The unilateral impulse absorbs the approach but adds no bounce.
        assert_relative_eq!(bodies[1].linear_velocity.y, 0.0, epsilon = 1e-6);

        let info = SolverInfo {
            restitution_velocity_threshold: 0.2,
            num_iterations: 1,
            ..SolverInfo::default()
        };
        let (mut bodies, mut row) = build(&info);
        solve(&mut bodies, std::slice::from_mut(&mut row), &mut [], &mut [], 60.0, &info);
        assert_relative_eq!(bodies[1].linear_velocity.y, 0.5, epsilon = 1e-6);
    }

    /// `split_impulse = false` keeps the deep-contact position correction in
    /// the velocity solve (Bullet's constructor default): the row loses the
    /// split pass and regains the Baumgarte bias, which the velocity solve
    /// turns into an outward impulse; the default keeps the velocities calm
    /// and hands the correction to the split pass.
    #[test]
    fn solver_info_split_switch_routes_deep_penetration_correction() {
        let off = SolverInfo { split_impulse: false, ..SolverInfo::default() };
        let (mut bodies, mut constraints) = deep_contact_with(0.1, &off);
        solve(&mut bodies, &mut constraints, &mut [], &mut [], 60.0, &off);
        assert!(!constraints[0].split_active);
        assert!(bodies[1].linear_velocity.y > 0.0);

        let (mut bodies, mut constraints) = deep_contact(0.1);
        solve(&mut bodies, &mut constraints, &mut [], &mut [], 60.0, &SolverInfo::default());
        assert!(constraints[0].split_active);
        assert_relative_eq!(bodies[1].linear_velocity.y, 0.0, epsilon = 1e-6);
    }

    /// `split_impulse_penetration_threshold` (Bullet's negative sign) picks
    /// the depth beyond which the split pass takes over: a deeper threshold
    /// keeps a 0.05-deep contact on the plain Baumgarte path.
    #[test]
    fn solver_info_split_threshold_picks_the_handover_depth() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.45, 0.0),
            1.0,
        );
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, -0.05, 0.0),
            Vector3::Y,
            0.05,
        );
        let info = SolverInfo::default();
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &info);
        assert!(row.split_active);

        let deeper = SolverInfo {
            split_impulse_penetration_threshold: -0.1,
            ..SolverInfo::default()
        };
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &deeper);
        assert!(!row.split_active);
    }

    /// `friction_erp` scales the friction-anchor servo target one-to-one:
    /// tripling it triples the first-iteration pull toward the anchor.
    #[test]
    fn solver_info_friction_erp_scales_the_anchor_servo() {
        let cp = bullet_collision::manifold::ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.0);
        let disp = Vector3::new(0.005, 0.0, 0.0);

        let run = |friction_erp: Real| -> Real {
            let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
            let mut cube = RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            );
            cube.linear_velocity = Vector3::new(0.0, -1.0, 0.0);
            let info = SolverInfo { friction_erp, ..SolverInfo::default() };
            let row = ContactConstraint::prepare(0, 1, &cp, &ground, &cube, u32::MAX, Some(disp), 60.0, &info);
            let mut bodies = vec![ground, cube];
            solve(&mut bodies, &mut [row], &mut [], &mut [], 60.0, &info);
            bodies[1].linear_velocity.x
        };
        let default_pull = run(0.2);
        let strong_pull = run(0.6);
        assert!(default_pull > 0.0);
        assert_relative_eq!(strong_pull, default_pull * 3.0, epsilon = 1e-4);
    }

    /// `global_cfm` opens the compliance channel: the row gains a non-zero
    /// `normal_cfm` (soft solve path) and a softened effective mass; the
    /// default 0 keeps both exactly at their rigid values.
    #[test]
    fn solver_info_global_cfm_softens_the_normal_row() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.49, 0.0),
            1.0,
        );
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, -0.01, 0.0),
            Vector3::Y,
            0.01,
        );
        let info = SolverInfo::default();
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &info);
        let rigid_mass = row.normal_mass;
        assert_eq!(row.normal_cfm, 0.0);

        let soft = SolverInfo { global_cfm: 0.1, ..SolverInfo::default() };
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &soft);
        assert!(row.normal_cfm > 0.0);
        assert!(row.normal_mass < rigid_mass);
    }

    /// `sor` is the relaxation numerator of the contact and linear friction
    /// rows: a 2x over-relaxation doubles the masses, one-to-one.
    #[test]
    fn solver_info_sor_relaxes_the_contact_and_friction_rows() {
        let ground = RigidBody::new_static(CollisionShape::plane(Vector3::Y, 0.0), Vector3::ZERO);
        let sphere = RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.49, 0.0),
            1.0,
        );
        let cp = bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.0, -0.01, 0.0),
            Vector3::Y,
            0.01,
        );
        let info = SolverInfo::default();
        let row = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &info);

        let relaxed = SolverInfo { sor: 2.0, ..SolverInfo::default() };
        let row2 = ContactConstraint::prepare(0, 1, &cp, &ground, &sphere, u32::MAX, None, 60.0, &relaxed);
        assert_relative_eq!(row2.normal_mass, row.normal_mass * 2.0, epsilon = 1e-6);
        assert_relative_eq!(row2.tangent1_mass, row.tangent1_mass * 2.0, epsilon = 1e-6);
        assert_relative_eq!(row2.tangent2_mass, row.tangent2_mass * 2.0, epsilon = 1e-6);
    }

    /// `damping` scales the joint row's relative-velocity term (Bullet's
    /// `info2.m_damping`): with 0.5 the point-to-point rows correct less of
    /// the relative velocity in one iteration than with the default 1.0.
    #[test]
    fn solver_info_damping_scales_the_joint_row_velocity_term() {
        let run = |damping: Real| -> (Real, Real) {
            let mut a = RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(-0.5, 0.0, 0.0),
                1.0,
            );
            let b = RigidBody::new_dynamic(
                CollisionShape::sphere(0.5),
                Vector3::new(0.5, 0.0, 0.0),
                1.0,
            );
            a.linear_velocity = Vector3::new(1.0, 0.0, 0.0);
            let mut constraint = crate::constraint::Constraint::point2point(
                crate::handles::BodyHandle::from_index(0),
                crate::handles::BodyHandle::from_index(1),
                Vector3::new(0.5, 0.0, 0.0),
                Vector3::new(-0.5, 0.0, 0.0),
            );
            let mut bodies = vec![a, b];
            let mut rows = Vec::new();
            crate::constraint::prepare_rows(&mut constraint, &bodies, 60.0, 0.2, &mut rows);
            let info = SolverInfo { num_iterations: 1, damping, ..SolverInfo::default() };
            solve(&mut bodies, &mut [], &mut rows, &mut [], 60.0, &info);
            (bodies[0].linear_velocity.x, bodies[1].linear_velocity.x)
        };
        let (va1, vb1) = run(1.0);
        let (va05, vb05) = run(0.5);
        // The rows pull the pair toward a common velocity: the default
        // damping corrects more per iteration, so the residual relative
        // velocity shrinks further.
        assert!((vb1 - va1).abs() < (vb05 - va05).abs());
    }
}

