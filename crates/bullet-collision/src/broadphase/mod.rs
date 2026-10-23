//! Broad-phase collision: coarse AABB-based pair filtering.

use bullet_math::{Aabb, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod axis_sweep3;
mod brute_force;
mod dbvt;

pub use axis_sweep3::AxisSweep3;
pub use brute_force::BruteForceBroadPhase;
pub use dbvt::{DbvtBroadPhase, DbvtTree};

/// Selectable broad-phase backend: an enum dispatch (no `dyn` on the hot
/// path) so a world can be built with either structure. `AxisSweep3` is the
/// default; `Dbvt` scales to very large dynamic scenes and unrestricted
/// world extents (Bullet's default broad phase).
#[derive(Debug)]
pub enum BroadPhaseBackend {
    AxisSweep3(AxisSweep3),
    Dbvt(DbvtBroadPhase),
}

impl Default for BroadPhaseBackend {
    fn default() -> Self {
        Self::AxisSweep3(AxisSweep3::new())
    }
}

impl BroadPhase for BroadPhaseBackend {
    fn create_proxy(&mut self, aabb: Aabb) -> usize {
        match self {
            Self::AxisSweep3(bp) => bp.create_proxy(aabb),
            Self::Dbvt(bp) => bp.create_proxy(aabb),
        }
    }

    fn set_aabb(&mut self, proxy: usize, aabb: Aabb) {
        match self {
            Self::AxisSweep3(bp) => bp.set_aabb(proxy, aabb),
            Self::Dbvt(bp) => bp.set_aabb(proxy, aabb),
        }
    }

    fn destroy_proxy(&mut self, proxy: usize) {
        match self {
            Self::AxisSweep3(bp) => bp.destroy_proxy(proxy),
            Self::Dbvt(bp) => bp.destroy_proxy(proxy),
        }
    }

    fn proxy_aabb(&self, proxy: usize) -> Option<Aabb> {
        match self {
            Self::AxisSweep3(bp) => bp.proxy_aabb(proxy),
            Self::Dbvt(bp) => bp.proxy_aabb(proxy),
        }
    }

    fn compute_pairs(&mut self) -> Vec<(usize, usize)> {
        match self {
            Self::AxisSweep3(bp) => bp.compute_pairs(),
            Self::Dbvt(bp) => bp.compute_pairs(),
        }
    }

    fn compute_pairs_into(&mut self, out: &mut Vec<(usize, usize)>) {
        match self {
            Self::AxisSweep3(bp) => bp.compute_pairs_into(out),
            Self::Dbvt(bp) => bp.compute_pairs_into(out),
        }
    }

    fn raycast_candidates(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<usize> {
        match self {
            Self::AxisSweep3(bp) => bp.raycast_candidates(ray_from, ray_to),
            Self::Dbvt(bp) => bp.raycast_candidates(ray_from, ray_to),
        }
    }

    fn aabb_candidates(&self, aabb: Aabb) -> Vec<usize> {
        match self {
            Self::AxisSweep3(bp) => bp.aabb_candidates(aabb),
            Self::Dbvt(bp) => bp.aabb_candidates(aabb),
        }
    }
}

/// A broad-phase structure producing candidate overlapping pairs.
///
/// Proxies are lightweight handles (indices) associated with a collision
/// object's AABB. Only pairs that survive this coarse test are sent to the
/// narrow phase.
pub trait BroadPhase {
    /// Create a proxy for the given AABB and return its index.
    fn create_proxy(&mut self, aabb: Aabb) -> usize;
    /// Update the AABB of an existing proxy.
    fn set_aabb(&mut self, proxy: usize, aabb: Aabb);
    /// Remove a proxy.
    fn destroy_proxy(&mut self, proxy: usize);
    /// The current AABB of a proxy, if the slot is live.
    fn proxy_aabb(&self, proxy: usize) -> Option<Aabb>;
    /// Compute all currently overlapping `(proxy_a, proxy_b)` pairs.
    ///
    /// Takes `&mut self`: implementations may advance internal state on this
    /// once-per-step query (the AxisSweep3 pair set only reads; the Dbvt
    /// broadphase runs its per-step tree optimization, stage rotation and
    /// separated-pair pruning here, mirroring Bullet's
    /// `calculateOverlappingPairs`).
    fn compute_pairs(&mut self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        self.compute_pairs_into(&mut out);
        out
    }
    /// Same as [`compute_pairs`](Self::compute_pairs), refilling `out` in
    /// place (cleared first) so a per-step caller can recycle the buffer's
    /// capacity instead of paying malloc/free every step.
    fn compute_pairs_into(&mut self, out: &mut Vec<(usize, usize)>);
    /// Indices of proxies whose AABB the segment `ray_from -> ray_to` crosses.
    ///
    /// Conservative: never misses a proxy a shape hit could come from, but may
    /// include extra proxies (e.g. the segment passes through an AABB corner
    /// that no shape covers). Mirrors Bullet's `btAxisSweep3::rayTest`.
    fn raycast_candidates(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<usize>;
    /// Indices of proxies whose AABB overlaps the query AABB (any order).
    ///
    /// Exact overlap test (no false positives); used by sweep and contact
    /// queries that then run the narrow phase on the candidates.
    fn aabb_candidates(&self, aabb: Aabb) -> Vec<usize>;
}

/// AABB-overlap candidates over a proxy AABB arena (shared by both broad
/// phases; the exact reference for tests and brute-force usage).
#[inline]
pub(crate) fn aabb_overlap_candidates(aabbs: &[Option<Aabb>], query: Aabb) -> Vec<usize> {
    aabbs.iter()
        .enumerate()
        .filter_map(|(i, a)| {
            a.as_ref()
                .filter(|aabb| aabb.overlaps(&query))
                .map(|_| i)
        })
        .collect()
}

/// Slab-method segment-versus-AABB test (t in `[0, 1]`).
///
/// Bullet's `TestAabbAgainstRay` equivalent: the segment is clipped against
/// each axis slab and rejected as soon as the entry exceeds the exit
/// parameter. Degenerate zero-length segments still report hits for boxes
/// containing the start point.
#[inline]
pub(crate) fn segment_hits_aabb(aabb: &Aabb, from: Vector3, to: Vector3) -> bool {
    let dir = to - from;
    let mut t_min: Real = 0.0;
    let mut t_max: Real = 1.0;
    for k in 0..3 {
        let d = dir[k];
        let (lo, hi) = if d >= 0.0 {
            (aabb.min[k], aabb.max[k])
        } else {
            (aabb.max[k], aabb.min[k])
        };
        if d == 0.0 {
            // Parallel to the slab: the ray must already lie inside it.
            if from[k] < lo || from[k] > hi {
                return false;
            }
            continue;
        }
        let inv = 1.0 / d;
        t_min = t_min.max((lo - from[k]) * inv);
        t_max = t_max.min((hi - from[k]) * inv);
        if t_min > t_max {
            return false;
        }
    }
    true
}

/// Ray-test candidates over a proxy AABB arena (shared by both broad phases).
#[inline]
pub(crate) fn aabb_raycast_candidates(
    aabbs: &[Option<Aabb>],
    ray_from: Vector3,
    ray_to: Vector3,
) -> Vec<usize> {
    aabbs.iter()
        .enumerate()
        .filter_map(|(i, a)| {
            a.as_ref()
                .filter(|aabb| segment_hits_aabb(aabb, ray_from, ray_to))
                .map(|_| i)
        })
        .collect()
}
