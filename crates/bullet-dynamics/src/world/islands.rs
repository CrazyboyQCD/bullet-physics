//! Simulation-island construction and sleep/wake propagation (phase D of
//! `docs/multibody-collision.md`), driven from the pair pass in `step.rs`.

use super::*;

impl PhysicsWorld {

    /// Build the per-step union-find islands and apply the island sleep/wake
    /// decision (Bullet's `calculateSimulationIslands` + the activation pass
    /// of `btSimulationIslandManager::buildIslands`; phase D design
    /// `docs/multibody-collision.md` §9).
    ///
    /// Node space: one node per non-static rigid body, then one per
    /// multibody (a multibody's collider chain is connected by construction,
    /// so a single node yields the same partitions as Bullet's per-collider
    /// chain merge). Merge rules: broad-phase AABB pairs between two dynamic
    /// owners (conservative, like `findUnions` on the pair cache), enabled
    /// joints between two dynamic bodies, and multibody constraints
    /// (collider-less, a recorded deviation). An island with any awake
    /// member (`Active`/`DisableDeactivation`, or an awake multibody) wakes
    /// its sleeping members; an island where everyone is sleep-ready goes to
    /// sleep as a unit.
    pub(crate) fn step_islands(&mut self, pairs: &[(usize, usize)]) {
        let num_bodies = self.bodies.len();
        let num_mbs = self.multibodies.len();
        let soft_base = num_bodies + num_mbs;
        let total = soft_base + self.soft_bodies.len();
        // Pooled buffers, grown to the node count on demand (union-find
        // reset to `n` singletons).
        let IslandScratch { uf, island_awake, body_awake, mb_awake } =
            &mut self.island_scratch;
        uf.reset(total);
        island_awake.clear();
        island_awake.resize(total, false);

        // Proxy-pair owners -> island nodes (None: not an island member).
        // A soft body is one node (Bullet unions the soft body's single
        // collision object the same way) and never sleeps, so any island
        // touching it stays awake.
        let bodies = &self.bodies;
        let collider_entries = &self.collider_entries;
        let node_of = |owner: ProxyOwner| match owner {
            ProxyOwner::Body(i) => bodies[i].participates_in_islands().then_some(i),
            ProxyOwner::Collider(e) => Some(num_bodies + collider_entries[e].mb),
            ProxyOwner::Soft(i) => Some(soft_base + i),
            ProxyOwner::Free => None,
        };
        for &(pi, pj) in pairs {
            if let (Some(a), Some(b)) = (
                node_of(self.proxy_owners[pi]),
                node_of(self.proxy_owners[pj]),
            ) {
                uf.unite(a, b);
            }
        }
        for c in &self.constraints {
            if !c.enabled() {
                continue;
            }
            let (a, b) = (c.body_a().index(), c.body_b().index());
            if self.bodies[a].participates_in_islands()
                && self.bodies[b].participates_in_islands()
            {
                uf.unite(a, b);
            }
        }
        for c in &self.mb_constraints {
            // Multibody-vs-rigidbody constraints join the multibody with the
            // rigid body's island node (Bullet's `getIslandIdB` on
            // `m_rigidBodyB`); static bodies sit outside the node space.
            match c.rigid_body_b() {
                None => uf.unite(
                    num_bodies + c.body_a().index(),
                    num_bodies + c.body_b().index(),
                ),
                Some(rb) => {
                    let rb = rb.index();
                    if self.bodies[rb].participates_in_islands() {
                        uf.unite(num_bodies + c.body_a().index(), rb);
                    }
                }
            }
        }

        // Any awake member keeps its whole island awake (per island root).
        for (i, b) in self.bodies.iter().enumerate() {
            if b.participates_in_islands() && b.is_island_awake() {
                island_awake[uf.find(i)] = true;
            }
        }
        for (i, mb) in self.multibodies.iter().enumerate() {
            if mb.is_awake() {
                island_awake[uf.find(num_bodies + i)] = true;
            }
        }
        // Soft bodies never sleep (Bullet's `updateActivationState` only
        // walks the rigid list), so they pin their islands awake.
        for i in 0..self.soft_bodies.len() {
            island_awake[uf.find(soft_base + i)] = true;
        }

        // A moving kinematic body wakes every island it pairs with (Bullet
        // activates both bodies at contact-point creation; done here because
        // the island pass runs before the narrow phase, so the broadphase
        // pair is the overlap signal). An idle kinematic (zero velocities)
        // keeps its resting contacts sleeping, like Bullet.
        for &(pi, pj) in pairs {
            for (pa, pb) in [(pi, pj), (pj, pi)] {
                let ProxyOwner::Body(k) = self.proxy_owners[pa] else {
                    continue;
                };
                let kb = &bodies[k];
                if !kb.is_kinematic()
                    || (kb.linear_velocity.length_squared() == 0.0
                        && kb.angular_velocity.length_squared() == 0.0)
                {
                    continue;
                }
                let node = match self.proxy_owners[pb] {
                    ProxyOwner::Body(i) if bodies[i].participates_in_islands() => i,
                    ProxyOwner::Collider(e) => num_bodies + collider_entries[e].mb,
                    _ => continue,
                };
                island_awake[uf.find(node)] = true;
            }
        }

        // Resolve every member's island state first (the decisions borrow
        // the union-find while applying them borrows the world).
        body_awake.clear();
        body_awake.extend((0..num_bodies).map(|i| island_awake[uf.find(i)]));
        mb_awake.clear();
        mb_awake.extend((0..num_mbs).map(|i| island_awake[uf.find(num_bodies + i)]));
        for (i, &awake) in body_awake.iter().enumerate() {
            let b = &mut self.bodies[i];
            if awake {
                b.island_wake();
            } else if b.participates_in_islands() {
                b.go_to_sleep();
            }
        }
        for (i, &awake) in mb_awake.iter().enumerate() {
            let mb = &mut self.multibodies[i];
            if awake {
                if !mb.is_awake() {
                    mb.wake_up();
                }
            } else {
                mb.go_to_sleep();
            }
        }
    }
}