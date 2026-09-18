//! Global solver configuration (Bullet's `btContactSolverInfo`).

use bullet_math::Real;

/// Global solver parameters, read by the contact and joint solvers each step
/// (Bullet's `btContactSolverInfo`, read through `getSolverInfo()`).
///
/// Defaults reproduce this engine's behavior bit-for-bit; three of them
/// intentionally differ from Bullet's constructor (marked DEVIATION below,
/// documented in `docs/bullet-deviations.md` §4). Every parameter can be set
/// back to the upstream value through this struct.
///
/// Fields not ported from `btContactSolverInfoData` (source-verified dead or
/// structurally inapplicable, see deviations §4): `m_tau` /
/// `m_maxErrorReduction` (never read by the CPU solvers),
/// `m_restingContactRestitutionThreshold` ("unused as of 2.81" upstream),
/// `m_minimumSolverBatchSize` (no island batching), `m_maxGyroscopicForce`
/// (no explicit gyroscopic term), `m_singleAxisRollingFrictionThreshold`
/// (upstream default 1e30 keeps the branch dead), the residual/analytics
/// pair, the deformable trio, `m_frictionCFM` (consumed only by the
/// multibody friction rows), the `m_solverMode` bitmask (the two live flags
/// are the `warmstarting`-shaped and `split_impulse` fields below), and
/// `m_timeStep` (written by the step itself upstream).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverInfo {
    /// Position-error reduction for joint (non-contact) rows (Bullet's
    /// `m_erp`, fed to `getInfo2` as `info2.erp`; also the default of the
    /// multibody constraint ERP).
    pub erp: Real,
    /// Position-error reduction for contact rows (Bullet's `m_erp2`, the
    /// `setupContactConstraint` default; overridden per pair by
    /// [`crate::CollisionFlags::HAS_CONTACT_STIFFNESS_DAMPING`]).
    pub erp2: Real,
    /// Penetration dead zone before the Baumgarte term kicks in, in meters
    /// (Bullet's `m_linearSlop`). DEVIATION: 0.005 here vs upstream 0.0 —
    /// the slop is a dead zone in this engine (shallow penetrations get no
    /// positional bias at all) instead of being added to the contact
    /// distance.
    pub linear_slop: Real,
    /// Minimum approach speed (m/s) for restitution (Bullet's
    /// `m_restitutionVelocityThreshold`). DEVIATION: 1.0 here vs upstream
    /// 0.2 — low-speed bounces are suppressed for stability.
    pub restitution_velocity_threshold: Real,
    /// Whether deep-penetration position correction runs through the
    /// split-impulse pseudo-impulse pass (Bullet's `m_splitImpulse`).
    /// DEVIATION: enabled here vs upstream default off — recorded before
    /// this struct existed; the split pass keeps impact transients
    /// smaller (`falling_capsule` parity, 2026-08-14).
    pub split_impulse: bool,
    /// Penetration depth (m) beyond which the split pass takes over
    /// position correction (Bullet's `m_splitImpulsePenetrationThreshold`,
    /// kept in upstream sign: negative, compared against the signed
    /// contact distance internally; the default -0.04 matches upstream).
    pub split_impulse_penetration_threshold: Real,
    /// Orientation scaling applied to the split-impulse turn velocity
    /// during the pose writeback (Bullet's `m_splitImpulseTurnErp`).
    pub split_impulse_turn_erp: Real,
    /// Scale applied to cached impulses at warm start (Bullet's
    /// `m_warmstartingFactor`; the rigid solver reads it on the
    /// manifold-refresh path).
    pub warmstarting_factor: Real,
    /// Positional servo factor of the friction-anchor bias (Bullet's
    /// `m_frictionERP`, consumed by
    /// [`crate::CollisionFlags::HAS_FRICTION_ANCHOR`] rows).
    pub friction_erp: Real,
    /// Contact constraint-force mixing applied to every contact row
    /// (Bullet's `m_globalCfm`): softens the row's effective mass and
    /// bleeds accumulated impulses back proportionally. Zero keeps the
    /// rows rigid (bit-for-bit default path).
    pub global_cfm: Real,
    /// Successive over-relaxation factor of the contact and linear
    /// friction rows (Bullet's `m_sor`, the `relaxation` numerator of
    /// `m_jacDiagABInv`). Joint and torsional rows are not relaxed
    /// (upstream hardcodes 1.0 there).
    pub sor: Real,
    /// Velocity damping of joint rows (Bullet's `m_damping` via
    /// `info2.m_damping`): the row's relative velocity is scaled by this
    /// factor inside the impulse. 1.0 is bit-for-bit neutral.
    pub damping: Real,
    /// Number of PGS iterations per step for the velocity and split passes
    /// (Bullet's `m_numIterations`).
    pub num_iterations: u32,
    /// Inner iterations for the multibody non-contact rows per outer
    /// iteration (Bullet's `m_numNonContactInnerIterations`).
    pub num_non_contact_inner_iterations: u32,
}

impl Default for SolverInfo {
    fn default() -> Self {
        Self {
            erp: 0.2,
            erp2: 0.2,
            linear_slop: 0.005,
            restitution_velocity_threshold: 1.0,
            split_impulse: true,
            split_impulse_penetration_threshold: -0.04,
            split_impulse_turn_erp: 0.1,
            warmstarting_factor: 0.85,
            friction_erp: 0.2,
            global_cfm: 0.0,
            sor: 1.0,
            damping: 1.0,
            num_iterations: 10,
            num_non_contact_inner_iterations: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults: upstream-aligned values plus the three recorded
    /// deviations (slop, restitution threshold, split on).
    #[test]
    fn defaults_reproduce_the_pre_freeze_hardcoded_values() {
        let info = SolverInfo::default();
        assert_eq!(info.erp, 0.2);
        assert_eq!(info.erp2, 0.2);
        assert_eq!(info.linear_slop, 0.005);
        assert_eq!(info.restitution_velocity_threshold, 1.0);
        assert!(info.split_impulse);
        assert_eq!(info.split_impulse_penetration_threshold, -0.04);
        assert_eq!(info.split_impulse_turn_erp, 0.1);
        assert_eq!(info.warmstarting_factor, 0.85);
        assert_eq!(info.friction_erp, 0.2);
        assert_eq!(info.global_cfm, 0.0);
        assert_eq!(info.sor, 1.0);
        assert_eq!(info.damping, 1.0);
        assert_eq!(info.num_iterations, 10);
        assert_eq!(info.num_non_contact_inner_iterations, 1);
    }
}
