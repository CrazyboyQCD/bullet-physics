use std::collections::HashSet;

use bullet_math::Aabb;
use bullet_math::Vector3;

use super::{aabb_raycast_candidates, BroadPhase};

/// Sweep-and-prune broad phase over all three axes (Bullet's `btAxisSweep3`).
///
/// Each axis keeps an active-proxy list sorted by `min[k]`; a sweep over an
/// axis yields every pair that overlaps along that axis in `O(n + k)` time (an
/// early exit on the first non-overlapping neighbor). A pair overlaps in 3D
/// exactly when it overlaps on all three axes, so the axis pair sets are
/// intersected.
///
/// The sorted lists are maintained incrementally: `create_proxy` inserts into
/// each axis at its sorted position, `set_aabb` removes and re-inserts, and
/// `destroy_proxy` removes. A proxy whose AABB did not change (e.g. a static
/// terrain body) keeps its positions at zero cost.
///
/// Output order matches [`BruteForceBroadPhase`](super::BruteForceBroadPhase)
/// for identical inputs (lexicographic on `(i, j)`), so tests can compare
/// directly.
///
/// AABB coordinates must be finite; [`f32::total_cmp`] keys keep the sort
/// total even for degenerate input.
pub struct AxisSweep3 {
    aabbs: Vec<Option<Aabb>>,
    /// For each axis, indices of the active proxies sorted by `min[k]`.
    order: [Vec<usize>; 3],
}

impl Default for AxisSweep3 {
    fn default() -> Self {
        Self::new()
    }
}

impl AxisSweep3 {
    pub fn new() -> Self {
        Self {
            aabbs: Vec::new(),
            order: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Insert `proxy` into axis `k` at its sorted position (by `min[k]`).
    fn insert_sorted(&mut self, k: usize, proxy: usize) {
        let key = self.aabbs[proxy].as_ref().unwrap().min[k];
        let pos = self.order[k].partition_point(|&i| {
            self.aabbs[i].as_ref().unwrap().min[k].total_cmp(&key) == std::cmp::Ordering::Less
        });
        self.order[k].insert(pos, proxy);
    }

    /// Remove `proxy` from axis `k`'s sorted list.
    fn remove_unsorted(&mut self, k: usize, proxy: usize) {
        let pos = self.order[k].iter().position(|&i| i == proxy).unwrap();
        self.order[k].remove(pos);
    }

    /// All pairs overlapping along the single axis `k`.
    fn axis_pairs(&self, k: usize) -> HashSet<(usize, usize)> {
        let mut pairs = HashSet::new();
        let order = &self.order[k];
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
        let proxy = if let Some(i) = self.aabbs.iter().position(|a| a.is_none()) {
            self.aabbs[i] = Some(aabb);
            i
        } else {
            self.aabbs.push(Some(aabb));
            self.aabbs.len() - 1
        };
        for k in 0..3 {
            self.insert_sorted(k, proxy);
        }
        proxy
    }

    fn set_aabb(&mut self, proxy: usize, aabb: Aabb) {
        if self.aabbs[proxy].as_ref() == Some(&aabb) {
            return; // Static proxies keep their sorted positions for free.
        }
        self.aabbs[proxy] = Some(aabb);
        for k in 0..3 {
            self.remove_unsorted(k, proxy);
            self.insert_sorted(k, proxy);
        }
    }

    fn destroy_proxy(&mut self, proxy: usize) {
        self.aabbs[proxy] = None;
        for k in 0..3 {
            self.remove_unsorted(k, proxy);
        }
    }

    fn proxy_aabb(&self, proxy: usize) -> Option<Aabb> {
        self.aabbs[proxy]
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

    fn raycast_candidates(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<usize> {
        aabb_raycast_candidates(&self.aabbs, ray_from, ray_to)
    }

    fn aabb_candidates(&self, query: Aabb) -> Vec<usize> {
        // A proxy overlaps the query on axis k exactly when min[k] <= q.max[k]
        // and max[k] >= q.min[k]. The per-axis lists are sorted by min, so the
        // first condition is a prefix; intersect the three axes' prefixes.
        let mut sets: [HashSet<usize>; 3] = [HashSet::new(), HashSet::new(), HashSet::new()];
        for (k, set) in sets.iter_mut().enumerate() {
            let order = &self.order[k];
            let cut = order.partition_point(|&i| {
                self.aabbs[i].as_ref().unwrap().min[k] <= query.max[k]
            });
            *set = order[..cut]
                .iter()
                .copied()
                .filter(|&i| self.aabbs[i].as_ref().unwrap().max[k] >= query.min[k])
                .collect();
        }
        let mut out: Vec<usize> = sets[0]
            .intersection(&sets[1])
            .filter(|p| sets[2].contains(p))
            .copied()
            .collect();
        out.sort_unstable();
        out
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
        bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0)));
        bp.create_proxy(Aabb::new(Vector3::new(0.5, 0.5, 5.0), Vector3::new(1.5, 1.5, 6.0)));
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

    /// Incremental updates must stay equivalent to the brute-force baseline
    /// even after proxies move far, are destroyed, and are re-created.
    #[test]
    fn incremental_updates_match_brute_force() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut aabbs = Vec::new();
        let mut seed = 0x243f6a8885a308d3u64;
        let random_aabb = |seed: &mut u64| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((*seed >> 32) % 500) as Real / 10.0,
                ((*seed >> 16) % 500) as Real / 10.0,
                (*seed % 500) as Real / 10.0,
            );
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let extent = Vector3::new(
                ((*seed >> 32) % 100) as Real / 100.0 + 0.1,
                ((*seed >> 16) % 100) as Real / 100.0 + 0.1,
                (*seed % 100) as Real / 100.0 + 0.1,
            );
            Aabb::new(base, base + extent)
        };
        for _ in 0..32 {
            let aabb = random_aabb(&mut seed);
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
            aabbs.push(aabb);
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
        // Teleport every even proxy elsewhere, destroy odd ones, add new ones.
        for (i, aabb) in aabbs.iter_mut().enumerate() {
            if i % 2 == 0 {
                *aabb = random_aabb(&mut seed);
                sweep.set_aabb(i, *aabb);
                brute.set_aabb(i, *aabb);
            } else {
                sweep.destroy_proxy(i);
                brute.destroy_proxy(i);
            }
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
        // Re-create into the freed slots and teleport again.
        for i in (0..aabbs.len()).filter(|i| i % 2 == 1) {
            let aabb = random_aabb(&mut seed);
            assert_eq!(sweep.create_proxy(aabb), i);
            assert_eq!(brute.create_proxy(aabb), i);
            aabbs[i] = aabb;
        }
        let aabb = random_aabb(&mut seed);
        sweep.set_aabb(0, aabb);
        brute.set_aabb(0, aabb);
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
    }

    /// Setting an identical AABB must be a no-op (static bodies cost nothing).
    #[test]
    fn unchanged_aabb_is_a_noop() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)));
        let before = bp.compute_pairs();
        bp.set_aabb(a, Aabb::new(Vector3::ZERO, Vector3::ONE));
        assert_eq!(bp.compute_pairs(), before);
        // A ray along X pierces both AABBs at y = z = 0.5.
        let ray = Vector3::new(0.5, 0.5, 0.5);
        assert_eq!(bp.raycast_candidates(Vector3::new(-1.0, 0.5, 0.5), ray + Vector3::X * 5.0), vec![a, b]);
    }

    #[test]
    fn raycast_candidates_filters_by_ray_path() {
        let mut bp = AxisSweep3::new();
        let hit = bp.create_proxy(Aabb::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(2.0, 2.0, 2.0)));
        let miss = bp.create_proxy(Aabb::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(11.0, 11.0, 11.0)));
        // Ray passes through `hit` but stops well before `miss`.
        let candidates = bp.raycast_candidates(Vector3::ZERO, Vector3::new(1.5, 1.5, 1.5));
        assert_eq!(candidates, vec![hit]);
        // Ray starting inside the box still reports it.
        let inside = bp.raycast_candidates(Vector3::new(1.5, 1.5, 1.5), Vector3::new(4.0, 4.0, 4.0));
        assert!(inside.contains(&hit) && !inside.contains(&miss));
        // Box behind the ray origin is pruned.
        let behind = bp.raycast_candidates(Vector3::new(3.0, 3.0, 3.0), Vector3::new(4.0, 4.0, 4.0));
        assert!(behind.is_empty());
    }

    #[test]
    fn raycast_candidates_matches_brute_force() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut seed = 0x13198a2e03707344u64;
        for _ in 0..48 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((seed >> 32) % 400) as Real / 10.0,
                ((seed >> 16) % 400) as Real / 10.0,
                (seed % 400) as Real / 10.0,
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
        let from = Vector3::new(5.0, 6.0, 7.0);
        let to = Vector3::new(25.0, 2.0, 30.0);
        assert_eq!(
            sweep.raycast_candidates(from, to),
            brute.raycast_candidates(from, to)
        );
    }

    #[test]
    fn aabb_candidates_matches_brute_force() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut seed = 0xdeadbeefcafef00du64;
        for _ in 0..48 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((seed >> 32) % 400) as Real / 10.0,
                ((seed >> 16) % 400) as Real / 10.0,
                (seed % 400) as Real / 10.0,
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
        // Query AABBs at scattered positions (including overlaps, edge touches
        // and empty regions).
        let queries = [
            Aabb::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(20.0, 20.0, 20.0)),
            Aabb::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(12.0, 12.0, 12.0)),
            Aabb::new(Vector3::new(50.0, 50.0, 50.0), Vector3::new(60.0, 60.0, 60.0)),
            Aabb::new(Vector3::new(39.9, 0.0, 0.0), Vector3::new(40.0, 40.0, 40.0)),
        ];
        for q in queries {
            assert_eq!(sweep.aabb_candidates(q), brute.aabb_candidates(q), "query {q:?}");
        }
    }
}
