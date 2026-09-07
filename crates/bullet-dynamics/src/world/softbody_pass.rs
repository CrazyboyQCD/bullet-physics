//! Soft-body pair dispatch (SDF_RS rigid contacts, VF_SS soft-soft and the
//! link-collider pass), driven from the pair pass in `step.rs`.

use super::*;

impl PhysicsWorld {

    /// Soft-vs-rigid collision for one broad-phase pair (Bullet's
    /// `btSoftRigidCollisionAlgorithm` -> `defaultCollisionHandler`): the
    /// dispatch matches `(collisions & RVSmask)` *exactly* — `SDF_RS` nodes
    /// the body, `CL_RS` clusters it, combinations fall through to nothing
    /// (Bullet's switch semantics). Detection uses the rigid body's
    /// *current* pose — the rigid solve has not run yet, matching Bullet's
    /// detection inside `performDiscreteCollisionDetection`.
    pub(crate) fn soft_vs_rigid(&mut self, soft: usize, rigid: usize) {
        // A sleeping island's pair is skipped: nothing on the rigid side
        // moves and the soft body's nodes were at rest relative to it (the
        // island pass would have woken the body otherwise — a soft body in
        // the island pins it awake, so a sleeping rigid island can only be
        // paired while the soft body itself is out of reach).
        if !self.bodies[rigid].is_active() && !self.bodies[rigid].is_static() {
            // Dynamic-but-sleeping: the soft body would have woken it in
            // 4b if they truly overlapped; the pair is a broad-phase
            // margin artifact.
            return;
        }
        let mode = self.soft_bodies[soft].cfg.collisions.0 & SoftCollision::RVS_MASK.0;
        let (soft_bodies, bodies) = (&mut self.soft_bodies, &self.bodies);
        if mode == SoftCollision::CL_RS.0 {
            soft_bodies[soft].collide_cluster_rigid(
                soft,
                rigid,
                &bodies[rigid],
                &mut self.soft_scratch,
                &mut self.epa_scratch,
            );
        } else if mode == SoftCollision::SDF_RS.0 {
            soft_bodies[soft].collide_rigid(rigid, &bodies[rigid], &mut self.soft_scratch);
        }
    }

    /// Soft-vs-multibody collision for one broad-phase pair (phase F):
    /// the SDF_RS node contacts against a link collider at its current FK
    /// pose. A sleeping multibody only pairs with a soft body through
    /// broad-phase margin artifacts (the island pass unions the pair and
    /// pins the multibody awake whenever they truly overlap), so like
    /// `soft_vs_rigid` the sleeping case is skipped. `CL_RS` against a
    /// multibody stays unsupported (Bullet's cluster path treats the link
    /// collider as massless — a degenerate one-sided push; recorded in
    /// deviations).
    pub(crate) fn soft_vs_mb(&mut self, soft: usize, entry: usize) {
        if !self.multibodies[self.collider_entries[entry].mb].is_awake() {
            return;
        }
        let mode = self.soft_bodies[soft].cfg.collisions.0 & SoftCollision::RVS_MASK.0;
        if mode != SoftCollision::SDF_RS.0 {
            return;
        }
        let ent = self.collider_entries[entry];
        let mb = &self.multibodies[ent.mb];
        let Some(col) = (if ent.link < 0 {
            mb.base_collider()
        } else {
            mb.link_collider(ent.link as usize)
        }) else {
            return;
        };
        let (position, orientation) = if ent.link < 0 {
            (mb.base_pos(), mb.base_world_orientation())
        } else {
            let l = ent.link as usize;
            (mb.link_world_position(l), mb.link_world_orientation(l))
        };
        let (soft_bodies, scratch) = (&mut self.soft_bodies, &mut self.soft_scratch);
        soft_bodies[soft].collide_multibody(
            ent.mb,
            ent.link,
            &col.shape,
            position,
            orientation,
            col.friction,
            scratch,
        );
    }

    /// Soft-vs-soft collision for one broad-phase pair (Bullet's
    /// `btSoftSoftCollisionAlgorithm` -> `defaultCollisionHandler(psb)`):
    /// the dispatch matches `(collisions_a & collisions_b) & SVSmask`
    /// *exactly* — `VF_SS` runs the node-face contacts both ways, `CL_SS`
    /// one cluster-cluster tree pass (joints land on body A), combinations
    /// fall through to nothing.
    pub(crate) fn soft_vs_soft(&mut self, a: usize, b: usize) {
        let effective = self.soft_bodies[a].cfg.collisions.0 & self.soft_bodies[b].cfg.collisions.0;
        if effective & SoftCollision::SVS_MASK.0 == SoftCollision::CL_SS.0 {
            let (soft_bodies, scratch) = (&mut self.soft_bodies, &mut self.soft_pair_scratch);
            let epa = &mut self.epa_scratch;
            let (before, rest) = soft_bodies.split_at_mut(a.min(b));
            let _ = before;
            let (sa, sb) = if a < b {
                let (sa, after) = rest.split_first_mut().unwrap();
                let sb = &mut after[b - a - 1];
                (sa, sb)
            } else {
                let (sb, after) = rest.split_first_mut().unwrap();
                let sa = &mut after[a - b - 1];
                (sa, sb)
            };
            if a < b {
                sa.collide_cluster_soft(a, sb, b, scratch, epa);
            } else {
                sb.collide_cluster_soft(b, sa, a, scratch, epa);
            }
            return;
        }
        if effective & SoftCollision::SVS_MASK.0 != SoftCollision::VF_SS.0 {
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (soft_bodies, scratch) = (&mut self.soft_bodies, &mut self.soft_pair_scratch);
        let (_before, rest) = soft_bodies.split_at_mut(lo);
        let (sb_lo, after) = rest.split_first_mut().unwrap();
        let sb_hi = &mut after[hi - lo - 1];
        // Nodes of A against faces of B first, then B's nodes against A's
        // faces (Bullet's two collideTT passes; each contact lands in the
        // node-side body tagged with the face-side body index).
        if a < b {
            collide_vf_ss(&mut *sb_lo, &mut *sb_hi, b, scratch);
            collide_vf_ss(&mut *sb_hi, &mut *sb_lo, a, scratch);
        } else {
            collide_vf_ss(&mut *sb_hi, &mut *sb_lo, b, scratch);
            collide_vf_ss(&mut *sb_lo, &mut *sb_hi, a, scratch);
        }
    }
}
