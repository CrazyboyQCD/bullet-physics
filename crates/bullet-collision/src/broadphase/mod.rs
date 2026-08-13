//! Broad-phase collision: coarse AABB-based pair filtering.

use bullet_math::Aabb;

mod axis_sweep3;
mod brute_force;

pub use axis_sweep3::AxisSweep3;
pub use brute_force::BruteForceBroadPhase;

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
    /// Compute all currently overlapping `(proxy_a, proxy_b)` pairs.
    fn compute_pairs(&self) -> Vec<(usize, usize)>;
}
