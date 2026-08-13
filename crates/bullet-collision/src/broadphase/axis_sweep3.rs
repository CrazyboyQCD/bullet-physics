use std::collections::HashSet;

use bullet_math::Aabb;

use super::BroadPhase;

/// Sweep-and-prune broad phase over all three axes (Bullet's `btAxisSweep3`).
///
/// Each axis keeps a min-sorted index list; a sweep over an axis yields every
/// pair that overlaps along that axis in `O(n + k)` time (an early exit on the
/// first non-overlapping neighbor). A pair overlaps in 3D exactly when it
/// overlaps on all three axes, so the axis pair sets are intersected.
///
/// Sorting is rebuilt from scratch per query; incremental insertion/movement
/// (Bullet keeps the arrays ordered on insert/update) is a future
/// optimization. Output order matches [`BruteForceBroadPhase`](super::BruteForceBroadPhase)
/// for identical inputs (lexicographic on `(i, j)`), so tests can compare
/// directly.
///
/// AABB coordinates must be finite; [`f32::total_cmp`] keys keep the sort
/// total even for degenerate input.
pub struct AxisSweep3 {
    aabbs: Vec<Option<Aabb>>,
}

impl Default for AxisSweep3 {
    fn default() -> Self {
        Self::new()
    }
}

impl AxisSweep3 {
    pub fn new() -> Self {
        Self { aabbs: Vec::new() }
    }

    /// All pairs overlapping along the single axis `k`.
    fn axis_pairs(&self, k: usize) -> HashSet<(usize, usize)> {
        let mut order: Vec<usize> = self
            .aabbs
            .iter()
            .enumerate()
            .filter_map(|(i, a)| a.as_ref().map(|_| i))
            .collect();
        order.sort_by(|&a, &b| self.aabbs[a].as_ref().unwrap().min[k].total_cmp(
            &self.aabbs[b].as_ref().unwrap().min[k],
        ));
        let mut pairs = HashSet::new();
        for pos in 0..order.len() {
            let i = order[pos];
            let i_max = self.aabbs[i].as_ref().unwrap().max[k];
            for &j in &order[pos + 1..] {
                if self.aabbs[j].as_ref().unwrap().min[k] > i_max {
                    break; // The list is sorted by min; no later proxy overlaps.
                }
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                pairs.insert((lo, hi));
            }
        }
        pairs
    }
}

impl BroadPhase for AxisSweep3 {
    fn create_proxy(&mut self, aabb: Aabb) -> usize {
        if let Some(i) = self.aabbs.iter().position(|a| a.is_none()) {
            self.aabbs[i] = Some(aabb);
            i
        } else {
            self.aabbs.push(Some(aabb));
            self.aabbs.len() - 1
        }
    }

    fn set_aabb(&mut self, proxy: usize, aabb: Aabb) {
        self.aabbs[proxy] = Some(aabb);
    }

    fn destroy_proxy(&mut self, proxy: usize) {
        self.aabbs[proxy] = None;
    }

    fn compute_pairs(&self) -> Vec<(usize, usize)> {
        let x = self.axis_pairs(0);
        let y = self.axis_pairs(1);
        let z = self.axis_pairs(2);
        let mut pairs: Vec<(usize, usize)> = x
            .intersection(&y)
            .filter(|p| z.contains(p))
            .copied()
            .collect();
        pairs.sort_unstable();
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_math::{Real, Vector3};

    #[test]
    fn finds_overlapping_pairs() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::splat(0.0), Vector3::splat(1.0)));
        let b = bp.create_proxy(Aabb::new(Vector3::splat(0.5), Vector3::splat(1.5)));
        let c = bp.create_proxy(Aabb::new(Vector3::splat(10.0), Vector3::splat(11.0)));
        let pairs = bp.compute_pairs();
        assert_eq!(pairs, vec![(a, b)]);
        assert!(pairs.iter().all(|&(x, y)| x != c && y != c));
    }

    #[test]
    fn proxy_slot_reuse() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        bp.destroy_proxy(a);
        let b = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        assert_eq!(a, b);
    }

    #[test]
    fn moved_proxies_are_recomputed() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)));
        assert!(bp.compute_pairs().is_empty());
        // Move `b` to overlap `a` and re-query.
        bp.set_aabb(b, Aabb::new(Vector3::new(0.5, 0.0, 0.0), Vector3::new(1.5, 1.0, 1.0)));
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
    }

    /// A single-axis overlap is not enough: pairs must overlap on all axes.
    #[test]
    fn requires_overlap_on_all_axes() {
        let mut bp = AxisSweep3::new();
        // Overlap on X and Y but separated on Z.
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0)));
        let b = bp.create_proxy(Aabb::new(Vector3::new(0.5, 0.5, 5.0), Vector3::new(1.5, 1.5, 6.0)));
        assert!(bp.compute_pairs().is_empty());
    }

    #[test]
    fn matches_brute_force_on_scattered_boxes() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        // Deterministic pseudo-random boxes across the volume.
        let mut seed = 0x9e3779b97f4a7c15u64;
        for _ in 0..64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((seed >> 32) % 1000) as Real / 10.0,
                ((seed >> 16) % 1000) as Real / 10.0,
                (seed % 1000) as Real / 10.0,
            );
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let extent = Vector3::new(
                ((seed >> 32) % 100) as Real / 100.0 + 0.1,
                ((seed >> 16) % 100) as Real / 100.0 + 0.1,
                (seed % 100) as Real / 100.0 + 0.1,
            );
            let aabb = Aabb::new(base, base + extent);
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
    }
}
