//! Simulation islands — the union-find core of Bullet's
//! `btUnionFind` + `btSimulationIslandManager`, phase D of the multibody
//! collision integration (detailed design `docs/multibody-collision.md` §9).
//!
//! Each step, dynamic rigid bodies and multibodies become nodes of a
//! union-find merged by broad-phase AABB pairs (conservatively: owners whose
//! AABBs overlap share an island even before they touch), by joints between
//! two dynamic bodies, and by multibody constraints. The per-island
//! sleep/wake decision lives in
//! [`PhysicsWorld::step_islands`](super::world::PhysicsWorld).

/// Union-find with path compression and union by size (Bullet's `btUnionFind`
/// built with `USE_PATH_COMPRESSION`).
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    /// `n` singleton sets `0..n`.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// The representative (island root) of `x`.
    pub(crate) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            // Path halving, like Bullet's two-line compression.
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merge the sets of `a` and `b` (Bullet's `unite`).
    pub(crate) fn unite(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            self.parent[ra] = rb;
            self.size[rb] += self.size[ra];
        } else {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singletons_are_their_own_root() {
        let mut uf = UnionFind::new(3);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(2), 2);
    }

    #[test]
    fn unite_merges_and_find_is_stable() {
        let mut uf = UnionFind::new(4);
        uf.unite(0, 1);
        uf.unite(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));
        uf.unite(1, 3);
        assert_eq!(uf.find(0), uf.find(2));
    }

    #[test]
    fn unite_of_the_same_set_is_a_noop() {
        let mut uf = UnionFind::new(2);
        uf.unite(0, 1);
        let root = uf.find(0);
        uf.unite(1, 0);
        assert_eq!(uf.find(1), root);
    }
}
