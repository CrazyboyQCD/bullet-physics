//! Simulation-island construction and sleep/wake propagation (phase D of
//! `docs/multibody-collision.md`), driven from the pair pass in `step.rs`.

use super::*;
use crate::body::filter_groups;

impl PhysicsWorld {

    /// Build the per-step union-find islands and apply the island sleep/wake
    /// decision (Bullet's `calculateSimulationIslands` + the activation pass
    /// of `btSimulationIslandManager::buildIslands`; phase D design
    /// `docs/multibody-collision.md` §9).
    ///
    /// Node space: one node per non-static rigid body (including
    /// `DisableSimulation` — Bullet's tag pass only skips static/kinematic
    /// objects, so a disabled body still merges islands and can bridge its
    /// neighbors), then one per multibody (a multibody's collider chain is
    /// connected by construction, so a single node yields the same partitions
    /// as Bullet's per-collider chain merge). Merge rules: collision-eligible
    /// broad-phase pairs between two island members — group/mask and the
    /// custom overlap filter (Bullet's pair-creation filter
    /// `needsBroadphaseCollision`) plus the sensor exclusion of
    /// `mergesSimulationIslands`; constraint-suppressed pairs still merge
    /// (they live in the pair cache and `findUnions` never consults
    /// `checkCollideWithOverride`) — then enabled joints between two dynamic
    /// bodies, and multibody constraints (collider-less, a recorded
    /// deviation). An island with any awake
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
        let IslandScratch {
            uf,
            island_awake,
            body_awake,
            mb_awake,
            proxy_flags,
        } = &mut self.island_scratch;
        uf.reset(total);
        island_awake.clear();
        island_awake.resize(total, false);

        // Proxy-pair owners -> island nodes (None: not an island member).
        // A soft body is one node (Bullet unions the soft body's single
        // collision object the same way) and never sleeps, so any island
        // touching it stays awake.
        let bodies = &self.bodies;
        let collider_entries = &self.collider_entries;
        let multibodies = &self.multibodies;
        let soft_bodies = &self.soft_bodies;
        let proxy_owners = &self.proxy_owners;
        let pair_filter = self.pair_filter.as_deref();
        let node_of = |owner: ProxyOwner| match owner {
            ProxyOwner::Body(i) => bodies[i].participates_in_islands().then_some(i),
            ProxyOwner::Collider(e) => Some(num_bodies + collider_entries[e].mb),
            ProxyOwner::Soft(i) => Some(soft_base + i),
            ProxyOwner::Free => None,
        };
        // Island-union eligibility per pair (Bullet's pair cache + the
        // `findUnions` gate): the cache only holds group/mask- and
        // overlap-filter-compatible pairs (`needsBroadphaseCollision` runs at
        // pair creation) and `mergesSimulationIslands` additionally excludes
        // sensor objects from merging. Constraint-suppressed pairs
        // deliberately stay eligible: they live in the pair cache and
        // `findUnions` never consults `checkCollideWithOverride`.
        //
        // The per-proxy flags precompute the rare cases (sensor bit,
        // non-default filter bit) once per step, so the all-default common
        // case is one byte test per side instead of the full walk.
        proxy_flags.clear();
        proxy_flags.resize(proxy_owners.len(), 0);
        for (p, owner) in proxy_owners.iter().enumerate() {
            let triple = match *owner {
                ProxyOwner::Body(i) => {
                    let b = &bodies[i];
                    Some((
                        b.collision_group,
                        b.collision_mask,
                        b.collision_flags
                            .contains(CollisionFlags::NO_CONTACT_RESPONSE),
                    ))
                }
                ProxyOwner::Collider(e) => {
                    let ent = &collider_entries[e];
                    let mb = &multibodies[ent.mb];
                    let col = if ent.link < 0 {
                        mb.base_collider()
                    } else {
                        mb.link_collider(ent.link as usize)
                    };
                    col.map(|col| (col.collision_group, col.collision_mask, false))
                }
                ProxyOwner::Soft(i) => {
                    let s = &soft_bodies[i];
                    Some((s.collision_group, s.collision_mask, false))
                }
                ProxyOwner::Free => None,
            };
            let Some((g, m, sensor)) = triple else {
                continue;
            };
            let mut f = 0u8;
            if sensor {
                f |= 1;
            }
            if g != filter_groups::DEFAULT || m != filter_groups::ALL {
                f |= 2;
            }
            proxy_flags[p] = f;
        }
        let proxy_flags: &[u8] = proxy_flags;
        let pair_eligible = |pi: usize, pj: usize| -> bool {
            let (fa, fb) = (proxy_flags[pi], proxy_flags[pj]);
            // A sensor on either side never merges islands.
            if (fa | fb) & 1 != 0 {
                return false;
            }
            // Both sides on the default filter (and no custom overlap
            // filter installed): the group/mask test cannot reject and the
            // filter has nothing to say.
            if (fa | fb) & 2 == 0 && pair_filter.is_none() {
                return true;
            }
            let side = |p: usize| -> Option<(u16, u16, bool)> {
                match proxy_owners[p] {
                    ProxyOwner::Body(i) => {
                        let b = &bodies[i];
                        Some((
                            b.collision_group,
                            b.collision_mask,
                            b.collision_flags
                                .contains(CollisionFlags::NO_CONTACT_RESPONSE),
                        ))
                    }
                    ProxyOwner::Collider(e) => {
                        let ent = &collider_entries[e];
                        let mb = &multibodies[ent.mb];
                        let col = if ent.link < 0 {
                            mb.base_collider()?
                        } else {
                            mb.link_collider(ent.link as usize)?
                        };
                        Some((col.collision_group, col.collision_mask, false))
                    }
                    ProxyOwner::Soft(i) => {
                        let s = &soft_bodies[i];
                        Some((s.collision_group, s.collision_mask, false))
                    }
                    ProxyOwner::Free => None,
                }
            };
            let (Some((ga, ma, sa)), Some((gb, mbm, sb))) = (side(pi), side(pj)) else {
                return false;
            };
            if sa || sb || !filters_collide(ga, ma, gb, mbm) {
                return false;
            }
            if let (ProxyOwner::Body(i), ProxyOwner::Body(j)) =
                (proxy_owners[pi], proxy_owners[pj])
                && let Some(filter) = pair_filter
                && !filter(
                    BodyHandle::from_index(i),
                    &bodies[i],
                    BodyHandle::from_index(j),
                    &bodies[j],
                )
            {
                return false;
            }
            true
        };
        for &(pi, pj) in pairs {
            if !pair_eligible(pi, pj) {
                continue;
            }
            if let (Some(a), Some(b)) = (node_of(proxy_owners[pi]), node_of(proxy_owners[pj])) {
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
            // Cheap kinematic pretest first: the eligibility check below
            // only runs for pairs that actually carry a moving kinematic
            // side (a pair the dispatcher would reject never produced a
            // manifold for Bullet's contact-time `activate()` to fire on).
            let kinematic_side = [(pi, pj), (pj, pi)].into_iter().find_map(|(pa, _)| {
                let ProxyOwner::Body(k) = proxy_owners[pa] else {
                    return None;
                };
                let kb = &bodies[k];
                if kb.is_kinematic()
                    && (kb.linear_velocity.length_squared() != 0.0
                        || kb.angular_velocity.length_squared() != 0.0)
                {
                    Some(pa)
                } else {
                    None
                }
            });
            let Some(pa) = kinematic_side else {
                continue;
            };
            if !pair_eligible(pi, pj) {
                continue;
            }
            let pb = if pa == pi { pj } else { pi };
            let node = match proxy_owners[pb] {
                ProxyOwner::Body(i) if bodies[i].participates_in_islands() => i,
                ProxyOwner::Collider(e) => num_bodies + collider_entries[e].mb,
                _ => continue,
            };
            island_awake[uf.find(node)] = true;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{filter_groups, ActivationState, CollisionFlags};
    use approx::assert_relative_eq;

    /// A body at rest for longer than `DEACTIVATION_TIME` goes to sleep and
    /// stops being integrated.
    #[test]
    fn body_sleeps_after_resting() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
        ));
        // 3 seconds > DEACTIVATION_TIME: the sphere settles and then sleeps.
        let dt: Real = 1.0 / 60.0;
        for _ in 0..(3.0 / dt) as usize {
            world.step_simulation(dt);
        }
        let sphere = world.body(BodyHandle::from_index(1));
        assert_eq!(sphere.activation_state(), ActivationState::IslandSleeping);
        assert_eq!(sphere.linear_velocity, Vector3::ZERO);
        let y_at_sleep = sphere.position.y;
        // Further steps must not move the sleeping body.
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert_relative_eq!(
            world.body(BodyHandle::from_index(1)).position.y,
            y_at_sleep,
            epsilon = 1e-6
        );
    }

    /// A sleeping body whose island gains an active member is woken up
    /// (Bullet's island activation pass: `ISLAND_SLEEPING` members of a
    /// non-sleeping island become `WANTS_DEACTIVATION` — simulating again,
    /// but immediately re-qualifying for sleeping).
    #[test]
    fn contact_wakes_sleeping_body() {
        let mut world = PhysicsWorld::new();
        // A light sphere resting on the plane goes to sleep first...
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        let dt: Real = 1.0 / 60.0;
        for _ in 0..(3.0 / dt) as usize {
            world.step_simulation(dt);
        }
        assert_eq!(
            world.body(BodyHandle::from_index(1)).activation_state(),
            ActivationState::IslandSleeping
        );
        // ...then a fast ball hits it: the sleeping sphere must wake up.
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.25),
                Vector3::new(0.0, 3.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -5.0, 0.0)),
        );
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert!(
            world.body(BodyHandle::from_index(1)).is_active(),
            "sleeping body was not woken by the impact"
        );
        assert_ne!(
            world.body(BodyHandle::from_index(1)).activation_state(),
            ActivationState::IslandSleeping
        );
    }

    /// Applying a force wakes a sleeping body up (Bullet's `applyCentralForce`).
    #[test]
    fn force_wakes_sleeping_body() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        let dt: Real = 1.0 / 60.0;
        for _ in 0..(3.0 / dt) as usize {
            world.step_simulation(dt);
        }
        world
            .body_mut(BodyHandle::from_index(1))
            .apply_force(Vector3::new(0.0, 5.0, 0.0));
        assert_eq!(
            world.body(BodyHandle::from_index(1)).activation_state(),
            ActivationState::Active
        );
    }

    /// Static bodies start asleep, like Bullet's static `btRigidBody`.
    #[test]
    fn static_body_starts_sleeping() {
        let body = RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::ZERO,
        );
        assert_eq!(body.activation_state(), ActivationState::IslandSleeping);
        assert!(!body.is_active());
    }

    /// Phase D: a resting pair on the plane is one island and sleeps as a
    /// unit, then wakes as a unit when a falling body joins the island
    /// (Bullet's `buildIslands` activation pass).
    #[test]
    fn island_sleeps_and_wakes_as_unit() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // Two boxes side by side, touching (one shared face at x = 0).
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            Vector3::new(-0.5, 0.5, 0.0),
            1.0,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            Vector3::new(0.5, 0.5, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..(4.0 / dt) as usize {
            world.step_simulation(dt);
        }
        // The whole island sleeps together.
        assert!(
            !world.body(BodyHandle::from_index(1)).is_active(),
            "left box should sleep"
        );
        assert!(
            !world.body(BodyHandle::from_index(2)).is_active(),
            "right box should sleep"
        );
        // Sleeping means frozen: further steps do not move either box.
        let (pa, pb) = (
            world.body(BodyHandle::from_index(1)).position,
            world.body(BodyHandle::from_index(2)).position,
        );
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert!((world.body(BodyHandle::from_index(1)).position - pa).length() < 1e-9);
        assert!((world.body(BodyHandle::from_index(2)).position - pb).length() < 1e-9);
        // A fast ball lands on the right box: the impact joins its island
        // and both boxes wake.
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.25),
                Vector3::new(-3.0, 0.5, 0.0),
                1.0,
            )
            .velocity(Vector3::new(8.0, 0.0, 0.0)),
        );
        for _ in 0..30 {
            world.step_simulation(dt);
        }
        assert!(
            world.body(BodyHandle::from_index(1)).is_active(),
            "left box should wake with its island"
        );
        assert!(
            world.body(BodyHandle::from_index(2)).is_active(),
            "right box should wake with its island"
        );
    }

    /// Phase D: a `DisableDeactivation` member keeps its whole island awake
    /// forever (Bullet counts `DISABLE_DEACTIVATION` as an awake contributor).
    /// The boxes are stacked (not edge-to-edge): the persistent four-point
    /// box-box contact keeps the broad-phase pair alive robustly, where
    /// edge-glued boxes can drift apart under the single-point plane contact
    /// (the plane fast path reports one contact, so a resting box pivots and
    /// creeps — a trajectory-sensitive way to lose the island link).
    #[test]
    fn disable_deactivation_keeps_island_awake() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            )
            .disable_deactivation(),
        );
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            Vector3::new(0.0, 1.5, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..(10.0 / dt) as usize {
            world.step_simulation(dt);
        }
        assert!(world.body(BodyHandle::from_index(1)).is_active());
        assert!(
            world.body(BodyHandle::from_index(2)).is_active(),
            "neighbor of a disable-deactivation body must never sleep"
        );
    }

    /// Phase D: a joint merges two islands across a distance — a moving
    /// body keeps its joint partner (far away, no contact) from sleeping
    /// (Bullet merges enabled constraints between two dynamic bodies).
    #[test]
    fn joint_links_islands_across_a_distance() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        // Two bodies 2 m apart joined by a point-to-point joint whose pivots
        // coincide at the midpoint, so the constraint starts satisfied (no
        // initial kick).
        let mid = Vector3::new(5.0, 0.0, 0.0);
        let a = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            mid - Vector3::X,
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            mid + Vector3::X,
            1.0,
        ));
        let joint = crate::constraint::Constraint::point2point(a, b, Vector3::X, -Vector3::X);
        world.add_constraint(joint);
        // Body A is driven perpetually; B is far away and still — only the
        // joint keeps its island awake.
        let dt = 1.0 / 60.0;
        for _ in 0..(6.0 / dt) as usize {
            world.body_mut(a).apply_force(Vector3::new(2.0, 0.0, 0.0));
            world.step_simulation(dt);
        }
        assert!(
            world.body(b).is_active(),
            "joint-linked partner of a moving body must stay awake"
        );
    }

    /// A moving kinematic body wakes the sleeping island it runs into; an idle
    /// one lets its resting contacts sleep.
    #[test]
    fn moving_kinematic_wakes_sleeping_island() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.25, 0.25, 0.25)),
            Vector3::new(0.0, 0.25, 0.0),
            1.0,
        ));
        for _ in 0..600 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(!world.body(box_body).is_active(), "precondition: asleep");

        // A slab flying in from the side.
        let slab = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
            Vector3::new(-4.0, 0.25, 0.0),
        ));
        world.body_mut(slab).linear_velocity = Vector3::new(6.0, 0.0, 0.0);

        let mut woke = false;
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
            if world.body(box_body).is_active() {
                woke = true;
                break;
            }
        }
        assert!(woke, "the moving kinematic must wake the sleeping island");
    }

    /// A pair the collision filter excludes (group/mask) must not affect
    /// island construction: Bullet's pair cache never contains it (the
    /// filter runs at pair creation, `needsBroadphaseCollision`), so
    /// `findUnions` cannot unite the two bodies. A permanently-awake body
    /// overlapping a still one through an excluded pair must therefore not
    /// keep it awake.
    #[test]
    fn group_mask_excluded_pair_does_not_pin_island_awake() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO); // B rests by stillness alone
        let b = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            )
            .collision_filter(2, 3),
        );
        // A is awake forever and overlaps B's broad-phase AABB only: the
        // pair is excluded both ways (B.group(2) & A.mask(1) == 0), and the
        // centers sit one margin past contact distance so the solver can
        // never separate them and invalidate the overlap.
        let a = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.72, 0.5, 0.72),
                1.0,
            )
            .collision_filter(filter_groups::DEFAULT, filter_groups::DEFAULT)
            .disable_deactivation(),
        );
        let dt = 1.0 / 60.0;
        for _ in 0..(3.5 / dt) as usize {
            world.step_simulation(dt);
        }
        assert_eq!(
            world.body(b).activation_state(),
            ActivationState::IslandSleeping,
            "an overlap the collision filter excludes pinned the still body awake"
        );
        // Control: making the same pair filter-compatible unites the island
        // again and A pins B awake. The spheres are placed one broad-phase
        // margin apart (AABBs overlap, surfaces do not), so the solver
        // cannot separate them and the union persists.
        world.body_mut(a).collision_group = filter_groups::DEFAULT;
        world.body_mut(a).collision_mask = filter_groups::ALL;
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert_ne!(
            world.body(b).activation_state(),
            ActivationState::IslandSleeping,
            "control failed: B should be pinned awake once the pair is eligible"
        );
    }

    /// A sensor (`CF_NO_CONTACT_RESPONSE`) never merges simulation islands
    /// (Bullet's `mergesSimulationIslands`), so an awake trigger overlapping
    /// a still body leaves it asleep, while the same body without the flag
    /// pins the neighbor awake.
    #[test]
    fn sensor_body_does_not_pin_island_awake() {
        let run = |sensor: bool| -> ActivationState {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            let b = world.add_body(RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            ));
            let a = world.add_body(
                RigidBody::new_dynamic(
                    bullet_collision::shapes::CollisionShape::sphere(0.5),
                    Vector3::new(0.72, 0.5, 0.72),
                    1.0,
                )
                .disable_deactivation(),
            );
            if sensor {
                world.body_mut(a).collision_flags = CollisionFlags::NO_CONTACT_RESPONSE;
            }
            let dt = 1.0 / 60.0;
            for _ in 0..(3.5 / dt) as usize {
                world.step_simulation(dt);
            }
            world.body(b).activation_state()
        };
        assert_ne!(
            run(false),
            ActivationState::IslandSleeping,
            "control failed: an awake plain body should pin its neighbor awake"
        );
        assert_eq!(
            run(true),
            ActivationState::IslandSleeping,
            "an awake sensor pinned the still body awake (sensors do not merge islands)"
        );
    }

    /// The moving-kinematic wake rides the collision filter too: Bullet's
    /// contact-time `activate()` needs a manifold, which an excluded pair
    /// never produces.
    #[test]
    fn moving_kinematic_does_not_wake_group_excluded_body() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // The box must rest on the plane (mask covers the plane's group 1)
        // while excluding the kinematic slab both ways.
        let box_body = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.25, 0.25, 0.25)),
                Vector3::new(0.0, 0.25, 0.0),
                1.0,
            )
            .collision_filter(2, 3),
        );
        for _ in 0..600 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(!world.body(box_body).is_active(), "precondition: asleep");

        let slab = world.add_body(
            RigidBody::new_kinematic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
                Vector3::new(-4.0, 0.25, 0.0),
            )
            .collision_filter(filter_groups::DEFAULT, filter_groups::DEFAULT),
        );
        world.body_mut(slab).linear_velocity = Vector3::new(6.0, 0.0, 0.0);
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(
            !world.body(box_body).is_active(),
            "the kinematic woke a body the collision filter excludes"
        );
    }

    /// A `DisableSimulation` body still merges islands (Bullet's tag pass
    /// only skips static/kinematic objects), so it bridges its neighbors: an
    /// awake body added on one side wakes a sleeping body on the other side
    /// through it, while the disabled body's own state is never touched
    /// (Bullet's `setActivationState` guard).
    #[test]
    fn disable_simulation_body_bridges_islands() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        // The bridge and the still body first: neither counts as an awake
        // contributor, so C falls asleep. All three spheres below sit one
        // broad-phase margin apart (AABBs overlap, surfaces do not), so the
        // solver cannot separate them and break the bridge.
        let b = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(1.02, 0.0, 0.0),
                1.0,
            )
            .disable_simulation(),
        );
        let c = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(2.04, 0.0, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..(3.5 / dt) as usize {
            world.step_simulation(dt);
        }
        assert_eq!(
            world.body(c).activation_state(),
            ActivationState::IslandSleeping,
            "precondition: C asleep (the disabled bridge is not an awake contributor)"
        );
        assert_eq!(
            world.body(b).activation_state(),
            ActivationState::DisableSimulation,
            "the island pass rewrote the disabled body's state"
        );
        // An awake body on the bridge's other side unites the island through
        // it and wakes C.
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.0, 0.0),
                1.0,
            )
            .disable_deactivation(),
        );
        for _ in 0..30 {
            world.step_simulation(dt);
        }
        assert!(
            world.body(c).is_active(),
            "the sleeping body was not woken through the DisableSimulation bridge"
        );
        assert_eq!(
            world.body(b).activation_state(),
            ActivationState::DisableSimulation,
            "the island pass rewrote the disabled body's state"
        );
    }
}