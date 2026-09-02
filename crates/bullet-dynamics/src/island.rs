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
#[derive(Default)]
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    /// Rebuild to `n` singleton sets `0..n` on the pooled arrays (the
    /// capacity survives for the next step).
    pub(crate) fn reset(&mut self, n: usize) {
        self.parent.clear();
        self.parent.extend(0..n);
        self.size.clear();
        self.size.resize(n, 1);
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

/// Pooled island-pass scratch owned by `PhysicsWorld` (Bullet rebuilds its
/// `btUnionFind` per `buildAndProcessIslands`): the union-find arrays plus
/// the awake-flag vectors, grown to the node count on demand so steady-state
/// steps allocate nothing (the 1000-body stack profile billed five per-step
/// heap allocations here).
#[derive(Default)]
pub(crate) struct IslandScratch {
    pub(crate) uf: UnionFind,
    pub(crate) island_awake: Vec<bool>,
    pub(crate) body_awake: Vec<bool>,
    pub(crate) mb_awake: Vec<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singletons_are_their_own_root() {
        let mut uf = UnionFind::default();
        uf.reset(3);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(2), 2);
    }

    #[test]
    fn unite_merges_and_find_is_stable() {
        let mut uf = UnionFind::default();
        uf.reset(4);
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
        let mut uf = UnionFind::default();
        uf.reset(2);
        uf.unite(0, 1);
        let root = uf.find(0);
        uf.unite(1, 0);
        assert_eq!(uf.find(1), root);
    }
}
