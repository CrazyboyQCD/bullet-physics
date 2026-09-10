//! Soft-body configuration — port of `btSoftBody::Config`, `SolverState`
//! and the solver/collision enums of `btSoftBody.h`.
//!
//! Defaults mirror `btSoftBody::initDefaults()` exactly (including the
//! surprisingly low `piterations = 1`; demos usually raise it).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::Real;

/// Aerodynamic force model (Bullet's `eAeroModel`). The declaration order
/// matches Bullet's enum so `aeromodel >= FTwoSided` selects the face
/// kernels like `applyForces` does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum AeroModel {
    /// Vertex normals are oriented toward velocity.
    #[default]
    VPoint,
    /// Vertex normals are flipped to match velocity.
    VTwoSided,
    /// Vertex normals are flipped to match velocity; lift and drag are
    /// applied.
    VTwoSidedLiftDrag,
    /// Vertex normals are taken as they are.
    VOneSided,
    /// Face normals are flipped to match velocity.
    FTwoSided,
    /// Face normals are flipped to match velocity; lift and drag are applied.
    FTwoSidedLiftDrag,
    /// Face normals are taken as they are.
    FOneSided,
}

/// Position-solver kernel kinds (Bullet's `ePSolver`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PSolverKind {
    /// Link distance constraints (`PSolve_Links`).
    Linear,
    /// Rigid-body anchors (`PSolve_Anchors`).
    Anchors,
    /// Contacts against rigid bodies (`PSolve_RContacts`).
    RContacts,
    /// Contacts against other soft bodies (`PSolve_SContacts`).
    SContacts,
}

/// Velocity-solver kernel kinds (Bullet's `eVSolver`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VSolverKind {
    /// Link velocity solve (`VSolve_Links`).
    Linear,
}

/// Solver preset (Bullet's `eSolverPresets`): the kernel sequences installed
/// by `setSolver`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverPreset {
    /// Position solve only (the default): `psequence = [Anchors, RContacts,
    /// SContacts, Linear]`.
    Positions,
    /// Velocity + position + drift solve: `vsequence = [Linear]`,
    /// `psequence = [Anchors, RContacts, SContacts]`, `dsequence = [Linear]`.
    Velocities,
}

/// Collision feature flags (Bullet's `fCollision`). The effective collision
/// mode of a pair is the intersection of both bodies' flags (see
/// `defaultCollisionHandler`), exactly like Bullet.
///
/// Only the classic-system flags are defined here; the `SDF_RD*`/`VF_DD`
/// deformable-system bits are out of scope (see `docs/design/softbody.md` §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftCollision(pub u32);

impl SoftCollision {
    /// SDF (distance-field) based rigid versus soft.
    pub const SDF_RS: Self = Self(0x0001);
    /// Cluster versus convex rigid.
    pub const CL_RS: Self = Self(0x0002);
    /// Rigid-versus-soft mask.
    pub const RVS_MASK: Self = Self(0x000f);
    /// Vertex versus face soft versus soft.
    pub const VF_SS: Self = Self(0x0010);
    /// Cluster versus cluster soft versus soft.
    pub const CL_SS: Self = Self(0x0020);
    /// Cluster soft-body self collision.
    pub const CL_SELF: Self = Self(0x0040);
    /// Soft-versus-soft mask.
    pub const SVS_MASK: Self = Self(0x00f0);
    /// Bullet's `fCollision::Default` = `SDF_RS` alone in this tree —
    /// soft-soft (`VF_SS`) is opt-in per body pair, and the handler
    /// dispatch matches the masked flags *exactly* (a combination like
    /// `VF_SS | CL_SELF` falls through to nothing upstream).
    pub const DEFAULT: Self = Self(0x0001);

    #[inline]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl Default for SoftCollision {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Per-body tunables (Bullet's `btSoftBody::Config`; defaults from
/// `initDefaults()`).
#[derive(Clone, Debug)]
pub struct Config {
    /// Aerodynamic model (default `V_Point`).
    pub aeromodel: AeroModel,
    /// Velocity correction factor for the drift pass (Baumgarte, default 1).
    pub k_vcf: Real,
    /// Damping coefficient [0,1] applied when writing velocities back from
    /// positions (default 0 = none).
    pub k_dp: Real,
    /// Drag coefficient [0,+inf] (default 0).
    pub k_dg: Real,
    /// Lift coefficient [0,+inf] (default 0).
    pub k_lf: Real,
    /// Pressure coefficient [-inf,+inf] (default 0).
    pub k_pr: Real,
    /// Volume conservation coefficient [0,+inf] (default 0).
    pub k_vc: Real,
    /// Dynamic friction coefficient [0,1] (default 0.2).
    pub k_df: Real,
    /// Pose matching coefficient [0,1] (default 0).
    pub k_mt: Real,
    /// Rigid contacts hardness [0,1] (default 1).
    pub k_chr: Real,
    /// Kinetic (static/kinematic) contacts hardness [0,1] (default 0.1).
    pub k_khr: Real,
    /// Soft contacts hardness [0,1] (default 1).
    pub k_shr: Real,
    /// Anchors hardness [0,1] (default 0.7).
    pub k_ahr: Real,
    /// Soft vs rigid hardness [0,1] (cluster only, default 0.1).
    pub k_srhr_cl: Real,
    /// Soft vs kinetic hardness [0,1] (cluster only, default 1).
    pub k_skhr_cl: Real,
    /// Soft vs soft hardness [0,1] (cluster only, default 0.5).
    pub k_sshr_cl: Real,
    /// Soft vs rigid impulse split [0,1] (cluster only, default 0.5).
    pub k_sr_splt_cl: Real,
    /// Soft vs kinetic impulse split [0,1] (cluster only, default 0.5).
    pub k_sk_splt_cl: Real,
    /// Soft vs soft impulse split [0,1] (cluster only, default 0.5).
    pub k_ss_splt_cl: Real,
    /// Maximum volume ratio for pose (default 1).
    pub maxvolume: Real,
    /// Time scale applied to dt (default 1).
    pub timescale: Real,
    /// Velocity solver iterations (default 0).
    pub viterations: u32,
    /// Position solver iterations (default 1; demos typically use 4).
    pub piterations: u32,
    /// Drift solver iterations (default 0).
    pub diterations: u32,
    /// Cluster solver iterations (default 4).
    pub citerations: u32,
    /// Collision feature flags (default `SDF_RS`).
    pub collisions: SoftCollision,
    /// Velocity solver kernel sequence.
    pub vsequence: Vec<VSolverKind>,
    /// Position solver kernel sequence.
    pub psequence: Vec<PSolverKind>,
    /// Drift solver kernel sequence.
    pub dsequence: Vec<PSolverKind>,
}

impl Default for Config {
    fn default() -> Self {
        let mut cfg = Self {
            aeromodel: AeroModel::default(),
            k_vcf: 1.0,
            k_dp: 0.0,
            k_dg: 0.0,
            k_lf: 0.0,
            k_pr: 0.0,
            k_vc: 0.0,
            k_df: 0.2,
            k_mt: 0.0,
            k_chr: 1.0,
            k_khr: 0.1,
            k_shr: 1.0,
            k_ahr: 0.7,
            k_srhr_cl: 0.1,
            k_skhr_cl: 1.0,
            k_sshr_cl: 0.5,
            k_sr_splt_cl: 0.5,
            k_sk_splt_cl: 0.5,
            k_ss_splt_cl: 0.5,
            maxvolume: 1.0,
            timescale: 1.0,
            viterations: 0,
            piterations: 1,
            diterations: 0,
            citerations: 4,
            collisions: SoftCollision::DEFAULT,
            vsequence: Vec::new(),
            psequence: Vec::new(),
            dsequence: Vec::new(),
        };
        cfg.set_solver(SolverPreset::Positions);
        cfg
    }
}

impl Config {
    /// Install a solver preset (Bullet's `setSolver`).
    pub fn set_solver(&mut self, preset: SolverPreset) {
        self.vsequence.clear();
        self.psequence.clear();
        self.dsequence.clear();
        match preset {
            SolverPreset::Positions => {
                self.psequence.extend([
                    PSolverKind::Anchors,
                    PSolverKind::RContacts,
                    PSolverKind::SContacts,
                    PSolverKind::Linear,
                ]);
            }
            SolverPreset::Velocities => {
                self.vsequence.push(VSolverKind::Linear);
                self.psequence.extend([
                    PSolverKind::Anchors,
                    PSolverKind::RContacts,
                    PSolverKind::SContacts,
                ]);
                self.dsequence.push(PSolverKind::Linear);
            }
        }
    }
}

/// Per-step solver scratch (Bullet's `btSoftBody::SolverState`); recomputed
/// at the head of `predict_motion`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolverState {
    /// Scaled dt (`dt * timescale`).
    pub sdt: Real,
    /// `1/sdt`.
    pub isdt: Real,
    /// Velocity margin (`sdt * 3`), dbvt prediction padding.
    pub velmrg: Real,
    /// Radial margin (the body's collision margin).
    pub radmrg: Real,
    /// Update margin (`radmrg * 0.25`), dbvt update threshold.
    pub updmrg: Real,
}
