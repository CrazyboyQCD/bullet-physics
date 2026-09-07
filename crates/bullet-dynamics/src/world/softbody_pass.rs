//! Soft-body pair dispatch (SDF_RS rigid contacts, VF_SS soft-soft and the
//! link-collider pass), driven from the pair pass in `step.rs`. Each
//! dispatch also feeds the collision-event tracking (`track_pair`): the
//! functions return whether anything touched, skipped pairs freeze their
//! prior state, and quiet scenes never record a pair.

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
        let pair = if soft < rigid {
            (ContactKey::soft(soft), ContactKey::body(rigid))
        } else {
            (ContactKey::body(rigid), ContactKey::soft(soft))
        };
        let opted_in = self.soft_bodies[soft].collision_events
            || self.bodies[rigid].collision_events;
        // A sleeping island's pair is skipped: nothing on the rigid side
        // moves and the soft body's nodes were at rest relative to it (the
        // island pass would have woken the body otherwise — a soft body in
        // the island pins it awake, so a sleeping rigid island can only be
        // paired while the soft body itself is out of reach).
        if !self.bodies[rigid].is_active() && !self.bodies[rigid].is_static() {
            // Dynamic-but-sleeping: the soft body would have woken it in
            // 4b if they truly overlapped; the pair is a broad-phase
            // margin artifact. The touch state freezes like the sleeping
            // rigid pairs do.
            self.track_pair(pair, opted_in, None);
            return;
        }
        let mode = self.soft_bodies[soft].cfg.collisions.0 & SoftCollision::RVS_MASK.0;
        let touched;
        {
            let (soft_bodies, bodies) = (&mut self.soft_bodies, &self.bodies);
            if mode == SoftCollision::CL_RS.0 {
                touched = soft_bodies[soft].collide_cluster_rigid(
                    soft,
                    rigid,
                    &bodies[rigid],
                    &mut self.soft_scratch,
                    &mut self.epa_scratch,
                );
            } else if mode == SoftCollision::SDF_RS.0 {
                touched =
                    soft_bodies[soft].collide_rigid(rigid, &bodies[rigid], &mut self.soft_scratch);
            } else {
                touched = false;
            }
        }
        self.track_pair(pair, opted_in, Some(touched));
    }

    /// Soft-vs-multibody collision for one broad-phase pair (phase F):
    /// the SDF_RS node contacts against a link collider at its current FK
    /// pose. A sleeping multibody only pairs with a soft body through
    /// broad-phase margin artifacts (the island pass unions the pair and
    /// pins the multibody awake whenever they truly overlap), so like
    /// `soft_vs_rigid` the sleeping case freezes instead of detecting.
    /// `CL_RS` against a multibody stays unsupported (Bullet's cluster path
    /// treats the link collider as massless — a degenerate one-sided push;
    /// recorded in deviations).
    pub(crate) fn soft_vs_mb(&mut self, soft: usize, entry: usize) {
        let ent = self.collider_entries[entry];
        // Link keys sort before soft keys (kind-first ordering).
        let pair = (ContactKey::link(ent.mb, ent.link), ContactKey::soft(soft));
        let col = if ent.link < 0 {
            self.multibodies[ent.mb].base_collider()
        } else {
            self.multibodies[ent.mb].link_collider(ent.link as usize)
        };
        let Some(col) = col else {
            return;
        };
        let opted_in = col.collision_events || self.soft_bodies[soft].collision_events;
        if !self.multibodies[ent.mb].is_awake() {
            self.track_pair(pair, opted_in, None);
            return;
        }
        let mode = self.soft_bodies[soft].cfg.collisions.0 & SoftCollision::RVS_MASK.0;
        if mode != SoftCollision::SDF_RS.0 {
            // The dispatch produces no detection for this configuration
            // (the unsupported `CL_RS`-against-a-multibody path).
            self.track_pair(pair, opted_in, Some(false));
            return;
        }
        let mb = &self.multibodies[ent.mb];
        let (position, orientation) = if ent.link < 0 {
            (mb.base_pos(), mb.base_world_orientation())
        } else {
            let l = ent.link as usize;
            (mb.link_world_position(l), mb.link_world_orientation(l))
        };
        let touched = {
            let (soft_bodies, scratch) = (&mut self.soft_bodies, &mut self.soft_scratch);
            soft_bodies[soft].collide_multibody(
                ent.mb,
                ent.link,
                &col.shape,
                position,
                orientation,
                col.friction,
                scratch,
            )
        };
        self.track_pair(pair, opted_in, Some(touched));
    }

    /// Soft-vs-soft collision for one broad-phase pair (Bullet's
    /// `btSoftSoftCollisionAlgorithm` -> `defaultCollisionHandler(psb)`):
    /// the dispatch matches `(collisions_a & collisions_b) & SVSmask`
    /// *exactly* — `VF_SS` runs the node-face contacts both ways, `CL_SS`
    /// one cluster-cluster tree pass (joints land on body A), combinations
    /// fall through to nothing.
    pub(crate) fn soft_vs_soft(&mut self, a: usize, b: usize) {
        let pair = (ContactKey::soft(a.min(b)), ContactKey::soft(a.max(b)));
        let opted_in =
            self.soft_bodies[a].collision_events || self.soft_bodies[b].collision_events;
        let effective = self.soft_bodies[a].cfg.collisions.0 & self.soft_bodies[b].cfg.collisions.0;
        if effective & SoftCollision::SVS_MASK.0 == SoftCollision::CL_SS.0 {
            let touched = {
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
                    sa.collide_cluster_soft(a, sb, b, scratch, epa)
                } else {
                    sb.collide_cluster_soft(b, sa, a, scratch, epa)
                }
            };
            self.track_pair(pair, opted_in, Some(touched));
            return;
        }
        if effective & SoftCollision::SVS_MASK.0 != SoftCollision::VF_SS.0 {
            // The dispatch produces no detection for this configuration.
            self.track_pair(pair, opted_in, Some(false));
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let touched = {
            let (soft_bodies, scratch) = (&mut self.soft_bodies, &mut self.soft_pair_scratch);
            let (_before, rest) = soft_bodies.split_at_mut(lo);
            let (sb_lo, after) = rest.split_first_mut().unwrap();
            let sb_hi = &mut after[hi - lo - 1];
            // Nodes of A against faces of B first, then B's nodes against
            // A's faces (Bullet's two collideTT passes; each contact lands
            // in the node-side body tagged with the face-side body index).
            let mut touched;
            if a < b {
                touched = collide_vf_ss(&mut *sb_lo, &mut *sb_hi, b, scratch);
                touched |= collide_vf_ss(&mut *sb_hi, &mut *sb_lo, a, scratch);
            } else {
                touched = collide_vf_ss(&mut *sb_hi, &mut *sb_lo, b, scratch);
                touched |= collide_vf_ss(&mut *sb_lo, &mut *sb_hi, a, scratch);
            }
            touched
        };
        self.track_pair(pair, opted_in, Some(touched));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::filter_groups;

    /// A soft body filtered by the rigid side's mask falls through.
    #[test]
    fn soft_body_respects_collision_mask() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(3.0, 0.5, 3.0)),
                Vector3::ZERO,
            )
            .collision_filter(filter_groups::DEFAULT, 0),
        );
        let rope = crate::softbody::helpers::create_rope(
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            8,
            0,
        );
        world.add_soft_body(rope);
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            world.step_simulation(dt);
        }
        let min_y = world
            .soft_body(SoftBodyHandle::from_index(0))
            .nodes
            .iter()
            .map(|n| n.x.y)
            .fold(Real::INFINITY, Real::min);
        assert!(
            min_y < -0.5,
            "rope did not fall through the filtered plate: min_y = {min_y}"
        );
    }
}
