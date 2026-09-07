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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::ActivationState;
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
}