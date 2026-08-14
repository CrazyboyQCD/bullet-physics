use bullet_math::{Aabb, Vector3};

use super::{aabb_raycast_candidates, BroadPhase};

/// O(n^2) broad phase. Correctness baseline for tests; replace with a
/// `DbvtBroadPhase` for real workloads.
pub struct BruteForceBroadPhase {
    aabbs: Vec<Option<Aabb>>,
}

impl Default for BruteForceBroadPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl BruteForceBroadPhase {
    pub fn new() -> Self {
        Self { aabbs: Vec::new() }
    }
}

impl BroadPhase for BruteForceBroadPhase {
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
        let mut pairs = Vec::new();
        for i in 0..self.aabbs.len() {
            let Some(a) = &self.aabbs[i] else {
                continue;
            };
            for (j, b) in self.aabbs.iter().enumerate().skip(i + 1) {
                if let Some(b) = b {
                    if a.overlaps(b) {
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs
    }

    fn raycast_candidates(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<usize> {
        aabb_raycast_candidates(&self.aabbs, ray_from, ray_to)
    }

    fn aabb_candidates(&self, aabb: Aabb) -> Vec<usize> {
        super::aabb_overlap_candidates(&self.aabbs, aabb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_math::Vector3;

    #[test]
    fn finds_overlapping_pairs() {
        let mut bp = BruteForceBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::splat(0.0), Vector3::splat(1.0)));
        let b = bp.create_proxy(Aabb::new(Vector3::splat(0.5), Vector3::splat(1.5)));
        let c = bp.create_proxy(Aabb::new(Vector3::splat(10.0), Vector3::splat(11.0)));
        let pairs = bp.compute_pairs();
        assert_eq!(pairs, vec![(a, b)]);
        assert!(pairs.iter().all(|&(x, y)| x != c && y != c));
    }

    #[test]
    fn proxy_slot_reuse() {
        let mut bp = BruteForceBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        bp.destroy_proxy(a);
        let b = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        assert_eq!(a, b);
    }
}
