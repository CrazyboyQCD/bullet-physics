//! The per-step pipeline of `PhysicsWorld` (`step_simulation`) and the
// persistent-manifold cache helpers. Split out of `world.rs`; the big
// function stays in one piece (its field borrows are interleaved).

use super::*;

use super::multibody_pass::side_active_owner;

impl PhysicsWorld {

    /// Binary-search the manifold cache: `Ok(index)` when the pair has a
    /// cached manifold, `Err(insert position)` otherwise. Associated
    /// functions taking the cache directly (instead of `&self`/`&mut self`)
    /// so callers can borrow `self.bodies` / `self.refresh_out` /
    /// `self.manifold_cache` disjointly inside the pair pass.
    fn cache_get(
        cache: &[Vec<ManifoldCacheEntry>],
        a: usize,
        b: usize,
    ) -> Option<&PersistentManifold> {
        let (lo, hi) = (a.min(b), a.max(b));
        cache
            .get(lo)?
            .iter()
            .find(|e| e.other == hi)
            .map(|e| &e.manifold)
    }

    fn cache_get_mut(
        cache: &mut [Vec<ManifoldCacheEntry>],
        a: usize,
        b: usize,
    ) -> Option<&mut PersistentManifold> {
        let (lo, hi) = (a.min(b), a.max(b));
        cache
            .get_mut(lo)?
            .iter_mut()
            .find(|e| e.other == hi)
            .map(|e| &mut e.manifold)
    }

    /// The pair's cached manifold, inserting an empty one on first touch
    /// (the `entry().or_default()` of the hash version). New entries append
    /// to the bucket — membership, not order, is what the cache serves.
    fn cache_entry_default(
        cache: &mut Vec<Vec<ManifoldCacheEntry>>,
        a: usize,
        b: usize,
    ) -> &mut PersistentManifold {
        let (lo, hi) = (a.min(b), a.max(b));
        if lo >= cache.len() {
            cache.resize_with(lo + 1, Vec::new);
        }
        let bucket = &mut cache[lo];
        match bucket.iter().position(|e| e.other == hi) {
            Some(slot) => &mut bucket[slot].manifold,
            None => {
                bucket.push(ManifoldCacheEntry {
                    other: hi,
                    manifold: PersistentManifold::default(),
                });
                &mut bucket.last_mut().expect("just pushed").manifold
            }
        }
    }

    /// Drop the pair's entry, if cached (Bullet destroys the manifold with
    /// the pair). Bucket-linear, like the lookups.
    fn cache_remove(cache: &mut [Vec<ManifoldCacheEntry>], a: usize, b: usize) {
        let (lo, hi) = (a.min(b), a.max(b));
        if let Some(bucket) = cache.get_mut(lo) {
            bucket.retain(|e| e.other != hi);
        }
    }


    /// Advance the simulation by `dt` seconds.
    pub fn step_simulation(&mut self, dt: Real) {
        let inv_dt = 1.0 / dt;
        let mut clk = StepClock::new(self.step_timings.enabled, self.step_clock_fn);
        // Collision events are per-step (Rapier's drain): the new step
        // rebuilds the buffer and the touching set from scratch.
        self.contact_events.clear();
        self.touching.clear();
        self.rolling_rows.clear();

        // 1. Integrate velocities from gravity and accumulated forces.
        // Sample the end-of-last-step speeds first: they feed the refresh-
        // vs-redetect gate of the persistent manifolds (a resting supported
        // body re-accelerates under gravity every step, so gating on the
        // post-gravity speed would never refresh anything).
        self.settled_speeds.clear();
        self.settled_speeds.extend(
            self.bodies
                .iter()
                .map(|b| b.linear_velocity.length_squared() + b.angular_velocity.length_squared()),
        );
        for body in &mut self.bodies {
            body.integrate_velocities(dt);
        }

        // 1m. Multibody gravity (Bullet's `applyGravity` over the `btMultiBody`
        // list) and the external-force pass: forward kinematics, constraint
        // forces cleared, articulated-body algorithm (`solveExternalForces`).
        for mb in &mut self.multibodies {
            if !mb.is_awake() {
                continue;
            }
            mb.add_base_force(self.gravity * mb.base_mass());
            for j in 0..mb.num_links() {
                mb.add_link_force(j, self.gravity * mb.link_mass(j));
            }
        }
        self.mb_solve_external_forces(dt);

        // 1s. Soft bodies: integrate the nodes to their predicted positions
        // (Bullet's `predictMotion`, driven from `predictUnconstraintMotion`
        // after the rigid velocity integration; insertion map
        // `docs/softbody.md` §4).
        let gravity = self.gravity;
        for sb in &mut self.soft_bodies {
            sb.predict_motion(dt, gravity);
        }
        clk.mark(&mut self.step_timings, StepStage::IntegrateVelocities);

        // 2. Predicted positions (used only for contact detection).
        self.predicted_poses.clear();
        self.predicted_poses
            .extend(self.bodies.iter().map(|b| b.predicted_position(dt)));

        // 3. Update broad-phase AABBs from predicted poses. Sleeping bodies
        // keep their AABB: they do not move, so it stays correct.
        for (i, body) in self.bodies.iter().enumerate() {
            if !body.is_active() {
                continue;
            }
            let aabb = body
                .shape
                .world_aabb(self.predicted_poses[i], body.orientation)
                .expanded(BROADPHASE_AABB_MARGIN);
            self.broadphase.set_aabb(self.body_proxies[i], aabb);
        }

        // 3m. Register new multibody colliders in the broad phase and refresh
        // the collider AABBs from the FK world transforms (Bullet's
        // `updateCollisionObjectWorldTransforms`; detection uses the current
        // pose, unlike the rigid bodies' predicted pose — recorded
        // deviation). Sleeping multibodies keep their AABBs like sleeping
        // bodies.
        self.register_pending_colliders();
        for e in &self.collider_entries {
            let Some(proxy) = e.proxy else {
                continue;
            };
            let mb = &self.multibodies[e.mb];
            if !mb.is_awake() {
                continue;
            }
            let aabb = if e.link < 0 {
                mb.base_collider()
                    .expect("registered base collider")
                    .shape
                    .world_aabb(mb.base_pos(), mb.base_world_orientation())
            } else {
                let l = e.link as usize;
                mb.link_collider(l)
                    .expect("registered link collider")
                    .shape
                    .world_aabb(mb.link_world_position(l), mb.link_world_orientation(l))
            }
            .expanded(BROADPHASE_AABB_MARGIN);
            self.broadphase.set_aabb(proxy, aabb);
        }

        // 3s. Soft-body proxies follow the node-hull bounds computed by
        // this step's predict (Bullet's `updateSingleAabb` on the soft
        // body's collision object; soft bodies never sleep, so every proxy
        // refreshes — a resting body's bounds are unchanged and the
        // broad-phase absorbs the no-op).
        for (i, proxy) in self.soft_proxies.iter().enumerate() {
            let bounds = self.soft_bodies[i].bounds;
            self.broadphase
                .set_aabb(*proxy, Aabb::new(bounds[0], bounds[1]));
        }
        clk.mark(&mut self.step_timings, StepStage::UpdateAabbs);

        // 4. Broad phase: candidate overlapping pairs.
        let pairs = self.broadphase.compute_pairs();
        clk.mark(&mut self.step_timings, StepStage::Broadphase);

        // 4b. Simulation islands (phase D, docs/multibody-collision.md §9):
        // union rigid bodies and multibodies by broad-phase pairs, joints
        // and multibody constraints, then decide sleep/wake per island —
        // the island-based replacement of the former one-hop wake-up pass
        // (which ran as stage 5b).
        self.step_islands(&pairs);
        clk.mark(&mut self.step_timings, StepStage::Islands);

        // 5. Narrow phase: persistent manifolds -> contact rows. Pairs
        // linked by a joint with `disable_collisions` are skipped (Bullet's
        // `btCollisionObject::checkCollideWithOverride`). Each cached pair
        // whose relative pose stayed within the redetect thresholds refreshes
        // its points against the current poses (no GJK/EPA rerun); otherwise
        // the pair re-runs detection seeded with the cached normal, captures
        // the fresh manifold, and emits rows. Rows keep the index of their
        // source cache point for the post-solve impulse writeback.
        self.contact_rows.clear();
        self.mb_pending_contacts.clear();
        // A fully-sleeping world keeps its manifolds verbatim: no body moves,
        // so no pair can have separated — skip the pair-stamping work until
        // something wakes (the stamp/prune only runs on active steps).
        self.step_counter = self.step_counter.wrapping_add(1);
        let simulate = self.bodies.iter().any(|b| b.is_active())
            || self.multibodies.iter().any(|b| b.is_awake());
        let stamp = self.step_counter;
        let prune = simulate && self.prev_simulate;
        self.prev_simulate = simulate;
        for &(pi, pj) in &pairs {
            let (owner_a, owner_b) = (self.proxy_owners[pi], self.proxy_owners[pj]);
            // 5s. Soft-body pairs (Bullet's soft collision algorithms,
            // dispatched from the same pair pass): soft-vs-rigid runs the
            // SDF_RS node contacts against the current rigid pose (the
            // rigid solve has not run yet, matching Bullet's detection
            // inside `performDiscreteCollisionDetection`); soft-vs-soft
            // (VF_SS) arrives with the cloth phase, soft-vs-link with the
            // multibody phase.
            match (owner_a, owner_b) {
                (ProxyOwner::Soft(s), ProxyOwner::Body(j)) => {
                    let (gs, ms) = (
                        self.soft_bodies[s].collision_group,
                        self.soft_bodies[s].collision_mask,
                    );
                    let (gb, bm) =
                        (self.bodies[j].collision_group, self.bodies[j].collision_mask);
                    if filters_collide(gs, ms, gb, bm) {
                        self.soft_vs_rigid(s, j);
                    }
                    continue;
                }
                (ProxyOwner::Body(j), ProxyOwner::Soft(s)) => {
                    let (gs, ms) = (
                        self.soft_bodies[s].collision_group,
                        self.soft_bodies[s].collision_mask,
                    );
                    let (gb, bm) =
                        (self.bodies[j].collision_group, self.bodies[j].collision_mask);
                    if filters_collide(gb, bm, gs, ms) {
                        self.soft_vs_rigid(s, j);
                    }
                    continue;
                }
                (ProxyOwner::Soft(a), ProxyOwner::Soft(b)) => {
                    let (ga, ma) = (
                        self.soft_bodies[a].collision_group,
                        self.soft_bodies[a].collision_mask,
                    );
                    let (gb, mb) = (
                        self.soft_bodies[b].collision_group,
                        self.soft_bodies[b].collision_mask,
                    );
                    if filters_collide(ga, ma, gb, mb) {
                        self.soft_vs_soft(a, b);
                    }
                    continue;
                }
                // Phase F: soft-vs-multibody link collider through the
                // SDF_RS node contacts (Bullet's `CollideSDF_RS` on a
                // `CO_FEATHERSTONE_LINK` colObj).
                (ProxyOwner::Soft(s), ProxyOwner::Collider(e))
                | (ProxyOwner::Collider(e), ProxyOwner::Soft(s)) => {
                    let (gs, ms) = (
                        self.soft_bodies[s].collision_group,
                        self.soft_bodies[s].collision_mask,
                    );
                    let ent = self.collider_entries[e];
                    let mb = &self.multibodies[ent.mb];
                    let col = if ent.link < 0 {
                        mb.base_collider()
                    } else {
                        mb.link_collider(ent.link as usize)
                    };
                    if let Some(col) = col
                        && filters_collide(gs, ms, col.collision_group, col.collision_mask)
                    {
                        self.soft_vs_mb(s, e);
                    }
                    continue;
                }
                _ => {}
            }
            let (ProxyOwner::Body(i), ProxyOwner::Body(j)) = (owner_a, owner_b) else {
                // Multibody link-collider pair (mixed or collider-collider):
                // run the narrow phase at the current FK pose and collect the
                // manifold for the contact rows (converted at solve time).
                // Self-collision filtering mirrors `checkCollideWithOverride`.
                // The group/mask filter is the same `NeedsCollision` test as
                // for the rigid pairs, on the colliders' filter values.
                let side_filter = |owner: ProxyOwner| -> Option<(u16, u16)> {
                    match owner {
                        ProxyOwner::Body(i) => Some((
                            self.bodies[i].collision_group,
                            self.bodies[i].collision_mask,
                        )),
                        ProxyOwner::Collider(e) => {
                            let ent = self.collider_entries[e];
                            let mb = &self.multibodies[ent.mb];
                            let col = if ent.link < 0 {
                                mb.base_collider()?
                            } else {
                                mb.link_collider(ent.link as usize)?
                            };
                            Some((col.collision_group, col.collision_mask))
                        }
                        ProxyOwner::Soft(_) | ProxyOwner::Free => None,
                    }
                };
                if let (Some((ga, ma)), Some((gb, gm))) =
                    (side_filter(owner_a), side_filter(owner_b))
                    && !filters_collide(ga, ma, gb, gm)
                {
                    continue;
                }
                let resolve = |owner: ProxyOwner| -> Option<(ContactSide, &bullet_collision::shapes::CollisionShape, Vector3, Quaternion, Real, Real)> {
                    match owner {
                        ProxyOwner::Body(i) => {
                            let b = &self.bodies[i];
                            Some((
                                ContactSide::Rigid(i),
                                &b.shape,
                                b.position,
                                b.orientation,
                                b.friction,
                                b.restitution,
                            ))
                        }
                        ProxyOwner::Collider(e) => {
                            let ent = self.collider_entries[e];
                            let mb = &self.multibodies[ent.mb];
                            let col = if ent.link < 0 {
                                mb.base_collider()?
                            } else {
                                mb.link_collider(ent.link as usize)?
                            };
                            let (pos, ori) = if ent.link < 0 {
                                (mb.base_pos(), mb.base_world_orientation())
                            } else {
                                let l = ent.link as usize;
                                (mb.link_world_position(l), mb.link_world_orientation(l))
                            };
                            Some((
                                ContactSide::Link { mb: ent.mb, link: ent.link },
                                &col.shape,
                                pos,
                                ori,
                                col.friction,
                                col.restitution,
                            ))
                        }
                        ProxyOwner::Soft(_) | ProxyOwner::Free => None,
                    }
                };
                let Some((side_a, shape_a, pa, qa, fa, ra)) = resolve(owner_a) else {
                    continue;
                };
                let Some((side_b, shape_b, pb, qb, fb, rb)) = resolve(owner_b) else {
                    continue;
                };
                if let (ContactSide::Link { mb: mba, link: la }, ContactSide::Link { mb: mbb, link: lb }) =
                    (side_a, side_b)
                    && mba == mbb
                    && !self.multibodies[mba].colliders_collide(la, lb)
                {
                    continue;
                }
                let Some(manifold) = narrowphase::collide_seeded(
                    shape_a,
                    pa,
                    qa,
                    shape_b,
                    pb,
                    qb,
                    None,
                    &mut self.epa_scratch,
                )
                else {
                    continue;
                };
                // Skip pairs where nothing can move (a sleeping multibody on
                // static geometry); anything touching an active side is woken
                // in stage 5b before the rows are assembled.
                let a_awake = side_active_owner(side_a, &self.bodies, &self.multibodies);
                let b_awake = side_active_owner(side_b, &self.bodies, &self.multibodies);
                if !a_awake && !b_awake {
                    continue;
                }
                self.mb_pending_contacts.push(PendingContact {
                    side_a,
                    side_b,
                    friction: fa * fb,
                    restitution: ra * rb,
                    manifold,
                });
                continue;
            };
            let pair = (i.min(j), i.max(j));
            if self.no_collision_pairs.binary_search(&pair).is_ok()
                || self.bodies[i].collision_group & self.bodies[j].collision_mask == 0
                || self.bodies[j].collision_group & self.bodies[i].collision_mask == 0
            {
                continue;
            }
            // The custom overlap filter gates the pair on top of the group
            // & mask test (Bullet's `btOverlapFilterCallback` order: both
            // must accept).
            if let Some(filter) = self.pair_filter.as_deref()
                && !filter(
                    BodyHandle::from_index(i),
                    &self.bodies[i],
                    BodyHandle::from_index(j),
                    &self.bodies[j],
                )
            {
                continue;
            }
            // The pair stays in the cache lifecycle even when skipped below:
            // a sleeping island keeps its manifolds (their AABBs still
            // overlap), matching Bullet's persistent manifolds on a
            // sleeping island.
            if simulate
                && let Some(cache) = Self::cache_get_mut(&mut self.manifold_cache, i, j)
            {
                cache.mark_paired(stamp);
            }
            // A pair whose owners are both inactive after the island pass
            // belongs to a sleeping island: skip detection and rows
            // entirely (Bullet solves no manifolds of a sleeping island).
            if !self.bodies[i].is_active() && !self.bodies[j].is_active() {
                // The sleeping pair's touch state is frozen: carry the
                // previous step's entry over so no event churns while the
                // island sleeps. Sensor pairs ride the same carry-over for
                // their persistent overlap list (see [`PhysicsWorld::overlapping`]).
                if (self.bodies[i].collision_events
                    || self.bodies[j].collision_events
                    || self.pair_sensor(i, j))
                    && self
                        .prev_touching
                        .binary_search(&[i.min(j) as u32, i.max(j) as u32])
                        .is_ok()
                {
                    self.touching.push([i.min(j) as u32, i.max(j) as u32]);
                }
                continue;
            }
            let (a, b) = (&self.bodies[i], &self.bodies[j]);
            // A trigger volume (Bullet's `CF_NO_CONTACT_RESPONSE`) is
            // detected and reported but never solved: no contact rows are
            // built for its pairs, so nothing is blocked or woken by it.
            let no_response = a
                .collision_flags
                .contains(CollisionFlags::NO_CONTACT_RESPONSE)
                || b.collision_flags
                    .contains(CollisionFlags::NO_CONTACT_RESPONSE);
            // Both sides immovable (Bullet's broadphase StaticFilter group
            // excludes static-kinematic and kinematic-kinematic pairs
            // outright): detection could only produce rows the solver would
            // discard (both inv masses are zero), so skip them entirely.
            if a.is_static_or_kinematic() && b.is_static_or_kinematic() {
                continue;
            }
            let (pa, pb) = (self.predicted_poses[i], self.predicted_poses[j]);
            // Fast-moving pairs re-detect *when the cache would serve a
            // positional bias*: a refreshed point deeper than the slop feeds
            // the Baumgarte term a re-projected (and, while the bodies move,
            // stale) depth — an energy pump on settling stacks (see
            // [`REDETECT_SPEED_SQUARED`]; `settled_speeds` holds the
            // end-of-last-step magnitude, combining linear and angular).
            // Shallow points carry no positional bias at all, so they are
            // pump-free at any speed — and they dominate rolling/sliding
            // contacts, exactly the pairs worth refreshing.
            let fast = self.settled_speeds[i] > REDETECT_SPEED_SQUARED
                || self.settled_speeds[j] > REDETECT_SPEED_SQUARED;
            let mut served_from_cache = false;
            if let Some(cache) = Self::cache_get_mut(&mut self.manifold_cache, i, j)
                && !cache.needs_redetect(pa, a.orientation, pb, b.orientation)
            {
                // Refresh path: re-project the cached anchors, drop points
                // that separated or drifted past the breaking threshold,
                // and emit the survivors (margin-band points with
                // non-positive penetration included) directly as solver
                // rows.
                cache.refresh(
                    pa,
                    a.orientation,
                    pb,
                    b.orientation,
                    &mut self.refresh_out,
                );
                let deep = self.refresh_out.iter().any(|cp| cp.penetration > SLOP);
                if !fast || !deep {
                    if !no_response {
                        for (k, mut cp) in self.refresh_out.iter().copied().enumerate() {
                            cp.normal_impulse *= WARM_STARTING_FACTOR;
                            let mut row = ContactConstraint::prepare(i, j, &cp, a, b, k as u32);
                            row.attach_rolling(a, b, &mut self.rolling_rows);
                            self.contact_rows.push(row);
                        }
                    }
                    if !self.refresh_out.is_empty()
                        && (a.collision_events || b.collision_events || no_response)
                    {
                        self.touching.push([i.min(j) as u32, i.max(j) as u32]);
                    }
                    served_from_cache = true;
                }
            }
            if served_from_cache {
                continue;
            }
            let seed =
                Self::cache_get(&self.manifold_cache, i, j).and_then(PersistentManifold::seed);
            if let Some(manifold) = narrowphase::collide_seeded(
                &a.shape,
                pa,
                a.orientation,
                &b.shape,
                pb,
                b.orientation,
                seed,
                &mut self.epa_scratch,
            ) {
                // Detection path: merge the fresh manifold into the
                // persistent one (Bullet's `addManifoldPoint`: matched
                // points replace and inherit their impulse, new points
                // append, undetected points survive until the refresh
                // prunes them), then emit rows for the whole merged
                // manifold — Bullet solves every point of the persistent
                // manifold each step, not just the newest detection. The
                // accumulation is what lets a resting flat box grow from
                // its single first-touch point to the full four-corner
                // support set (see the flat-box probe notes). Stamp the
                // new entry right away so this step's prune keeps it.
                let cache = Self::cache_entry_default(&mut self.manifold_cache, i, j);
                cache.mark_paired(stamp);
                let threshold = a
                    .shape
                    .contact_breaking_threshold()
                    .min(b.shape.contact_breaking_threshold());
                cache.capture(&manifold, pa, a.orientation, pb, b.orientation, threshold);
                cache.refresh(pa, a.orientation, pb, b.orientation, &mut self.refresh_out);
                if !no_response {
                    for (k, mut cp) in self.refresh_out.iter().copied().enumerate() {
                        cp.normal_impulse *= WARM_STARTING_FACTOR;
                        let mut row = ContactConstraint::prepare(i, j, &cp, a, b, k as u32);
                        row.attach_rolling(a, b, &mut self.rolling_rows);
                        self.contact_rows.push(row);
                    }
                }
                if a.collision_events || b.collision_events || no_response {
                    self.touching.push([i.min(j) as u32, i.max(j) as u32]);
                }
            } else {
                // Detection found nothing: the pair separated past the
                // breaking threshold — drop the cache entry entirely.
                Self::cache_remove(&mut self.manifold_cache, i, j);
            }
        }

        // 5e. Collision events: diff this step's touching pairs against the
        // previous step's (Rapier's post-step event drain). Only pairs with
        // an opted-in body are tracked, so scenes without listeners keep
        // both sets empty; a pair skipped because its island sleeps carries
        // its state over instead of churning.
        self.touching.sort_unstable();
        let prev = core::mem::take(&mut self.prev_touching);
        let cur = core::mem::take(&mut self.touching);
        let (mut pi, mut ci) = (0usize, 0usize);
        while pi < prev.len() || ci < cur.len() {
            // Both sets are sorted: walk the smaller key forward each round
            // (an Equal key is the surviving pair, emitted by neither side).
            let order = match (prev.get(pi), cur.get(ci)) {
                (Some(p), Some(c)) => p.cmp(c),
                (Some(_), None) => core::cmp::Ordering::Less,
                (None, Some(_)) => core::cmp::Ordering::Greater,
                (None, None) => break,
            };
            let (pair, started) = match order {
                core::cmp::Ordering::Less => {
                    let pair = prev[pi];
                    pi += 1;
                    (pair, false)
                }
                core::cmp::Ordering::Greater => {
                    let pair = cur[ci];
                    ci += 1;
                    (pair, true)
                }
                core::cmp::Ordering::Equal => {
                    pi += 1;
                    ci += 1;
                    continue;
                }
            };
            let (a, b) = (pair[0] as usize, pair[1] as usize);
            // Pairs tracked only for their overlap list (neither side
            // opted into events) emit nothing: events stay per-body
            // opt-in, the list is the sensor query surface.
            if !(self.bodies[a].collision_events || self.bodies[b].collision_events) {
                continue;
            }
            let sensor = self.pair_sensor(a, b);
            let event = if started {
                CollisionEvent::Started {
                    a: BodyHandle::from_index(a),
                    b: BodyHandle::from_index(b),
                    sensor,
                }
            } else {
                CollisionEvent::Stopped {
                    a: BodyHandle::from_index(a),
                    b: BodyHandle::from_index(b),
                    sensor,
                }
            };
            self.contact_events.push(event);
        }
        self.prev_touching = cur;

        // 5a. Joint rows: flatten every enabled joint of an awake island
        // into its solver rows from the current (pre-integration) poses,
        // like Bullet's `convertJoints` after `predictUnconstraintMotion`.
        // A joint whose both bodies are inactive (including a joint to a
        // static anchor whose dynamic side sleeps) belongs to a sleeping
        // island and produces no rows.
        self.joint_rows.clear();
        for c in &mut self.constraints {
            if !self.bodies[c.body_a().index()].is_active()
                && !self.bodies[c.body_b().index()].is_active()
            {
                continue;
            }
            constraint::prepare_rows(c, &self.bodies, inv_dt, BAUMGARTE, &mut self.joint_rows);
        }
        clk.mark(&mut self.step_timings, StepStage::Narrowphase);

        // 6. Solve constraints (correct velocities). Scenes without
        // multibody contacts keep the plain rigid solve (bit-identical to
        // the pre-multibody behavior); otherwise the rigid rows, multibody
        // contact rows and multibody constraint rows share one PGS loop
        // (Bullet's `solveMultiBodyGroup` inside `solveSingleIteration`).
        // The row pools are taken out for the solve and restored after the
        // writeback (their allocations are reused across steps).
        let mut contact_rows = core::mem::take(&mut self.contact_rows);
        let mut joint_rows = core::mem::take(&mut self.joint_rows);
        if self.mb_pending_contacts.is_empty() {
            sequential_impulse::solve(
                &mut self.bodies,
                &mut contact_rows,
                &mut joint_rows,
                &mut self.rolling_rows,
                self.iterations,
                inv_dt,
            );
        } else {
            self.solve_mixed(&mut contact_rows, &mut joint_rows, inv_dt);
        }

        // 6b. Write the solved impulses back into the persistent manifolds
        // so the next step warm starts from them (Bullet stores
        // `m_appliedImpulse` on the manifold point), and drop cache entries
        // whose broad-phase pair disappeared this step (Bullet destroys the
        // manifold when the pair dies).
        // Rows arrive grouped by pair (the pair pass walks pairs in order),
        // so re-locating the cache entry once per pair keeps the writeback
        // at one bucket scan per pair instead of one per contact row.
        let mut located: Option<((usize, usize), usize, usize)> = None;
        for c in &contact_rows {
            if c.point_index == u32::MAX {
                continue;
            }
            let pair = (c.body_a, c.body_b);
            if located.as_ref().is_none_or(|&(p, _, _)| p != pair) {
                let (lo, hi) = (pair.0.min(pair.1), pair.0.max(pair.1));
                located = self
                    .manifold_cache
                    .get(lo)
                    .and_then(|bucket| bucket.iter().position(|e| e.other == hi))
                    .map(|slot| (pair, lo, slot));
            }
            let Some(&(_, lo, slot)) = located.as_ref() else {
                continue;
            };
            if let Some(point) = self.manifold_cache[lo][slot]
                .manifold
                .point_mut(c.point_index as usize)
            {
                point.normal_impulse = c.normal_impulse;
            }
        }
        if prune {
            for bucket in &mut self.manifold_cache {
                bucket.retain(|e| e.manifold.paired_at(stamp));
            }
        }
        self.contact_rows = contact_rows;
        self.joint_rows = joint_rows;

        // 6bm. Multibody internal constraints: in the mixed path they were
        // solved together with the contact rows above; the pure path runs
        // them here.
        if self.mb_pending_contacts.is_empty() {
            self.mb_solve_internal_constraints(dt);
        }
        self.mb_pending_contacts.clear();
        clk.mark(&mut self.step_timings, StepStage::Solve);

        // 6c. Deactivation: bodies below the sleeping thresholds for longer
        // than `DEACTIVATION_TIME` seconds qualify for sleeping
        // (`WantsDeactivation`); the next step's island pass (4b) then
        // decides whether the whole island sleeps — Bullet's
        // `updateActivationState` / `buildIslands` split.
        for body in &mut self.bodies {
            body.update_deactivation(dt);
        }

        // 7. Integrate positions with the corrected velocities. Bodies that
        // exceed their CCD motion threshold are clamped at the first
        // swept-sphere impact so they cannot tunnel through thin geometry
        // (Bullet's `integrateTransforms` CCD motion clamping; the collision
        // response itself runs on the next step's narrow phase).
        let ccd_clamps: Vec<Real> = (0..self.bodies.len())
            .map(|i| {
                let body = &self.bodies[i];
                let threshold = body.ccd_motion_threshold();
                if !body.is_active() || body.is_static_or_kinematic() || threshold <= 0.0 {  
                    return 1.0;
                }
                let radius = body.ccd_swept_sphere_radius();
                if radius <= 0.0 || !body.shape.is_convex() {
                    return 1.0;
                }
                let from = body.position;
                let to = from + body.linear_velocity * dt;
                let square_motion = (to - from).length_squared();
                if square_motion <= threshold * threshold {
                    return 1.0;
                }
                let mut best = 1.0;
                // Broad-phase AABB pretest bounds for the whole sweep: a
                // target whose (margin-padded) proxy AABB does not touch
                // this box cannot be hit by the sphere cast, so the exact
                // cast only runs on plausible targets (conservative on both
                // sides — output-neutral pruning).
                let swept = Aabb::new(from.min(to), from.max(to)).expanded(radius);
                for j in 0..self.bodies.len() {
                    if i == j {
                        continue;
                    }
                    let other = &self.bodies[j];
                    // Removed bodies are gone from every collision path,
                    // including the CCD sweeps (their proxy is destroyed).
                    if other.is_removed() {
                        continue;
                    }
                    // Joint-linked pairs with suppressed collisions skip the
                    // sweep as well (Bullet's dispatcher `needsCollision` →
                    // `checkCollideWithOverride` gates the CCD sweep too).
                    if self.no_collision_pairs.binary_search(&(i.min(j), i.max(j))).is_ok() {
                        continue;
                    }
                    // The group/mask filter gates the CCD sweep too, like
                    // the suppression set above (dispatcher `needsCollision`).
                    if self.bodies[i].collision_group & other.collision_mask == 0
                        || other.collision_group & self.bodies[i].collision_mask == 0
                    {
                        continue;
                    }
                    // The custom overlap filter gates the CCD sweep as well.
                    if let Some(filter) = self.pair_filter.as_deref()
                        && !filter(
                            BodyHandle::from_index(i),
                            &self.bodies[i],
                            BodyHandle::from_index(j),
                            other,
                        )
                    {
                        continue;
                    }
                    // The swept-bounds pretest (see `swept` above).
                    if let Some(aabb) = self.broadphase.proxy_aabb(self.body_proxies[j])
                        && !aabb.overlaps(&swept)
                    {
                        continue;
                    }
                    if let Some(hit) = narrowphase::swept_sphere_cast(
                        from,
                        to,
                        radius,
                        &other.shape,
                        other.position,
                        other.orientation,
                    ) && hit.fraction < best
                    {
                        best = hit.fraction;
                    }
                }
                best
            })
            .collect();

        for (i, body) in self.bodies.iter_mut().enumerate() {
            body.integrate_position(dt);
            // Split-impulse positional correction (Bullet's
            // `writebackVelocityAndTransform`): move the pose along the
            // solver's push velocity so deep penetrations are corrected
            // without changing the real velocities. The turn velocity is
            // scaled by `m_splitImpulseTurnErp`. Push is cleared here; the
            // solver starts every step from zero.
            // Vec3A equality compares the unused w lane too, so test the
            // components explicitly (the scalar Vec3 compared exactly xyz).
            let push_lin = body.push_linear;
            let push_ang = body.push_angular;
            if body.is_active()
                && !body.is_static_or_kinematic()
                && (push_lin.x != 0.0 || push_lin.y != 0.0 || push_lin.z != 0.0)
            {
                body.position += push_lin * dt;
            }
            if body.is_active()
                && !body.is_static_or_kinematic()
                && (push_ang.x != 0.0 || push_ang.y != 0.0 || push_ang.z != 0.0)
            {
                let w = push_ang * SPLIT_TURN_ERP;
                let dq = Quaternion::from_xyzw(
                    w.x * dt * 0.5,
                    w.y * dt * 0.5,
                    w.z * dt * 0.5,
                    0.0,
                ) * body.orientation;
                body.orientation = (body.orientation + dq).normalize();
            }
            body.push_linear = Vector3::ZERO;
            body.push_angular = Vector3::ZERO;
            if ccd_clamps[i] < 1.0 {
                let from = body.position - body.linear_velocity * dt;
                body.position = from + (body.position - from) * ccd_clamps[i];
            }
        }

        // 7s. Soft bodies: solve the position-based constraint set. Bullet
        // runs `solveSoftBodiesConstraints` after the rigid bodies are fully
        // stepped (solved *and* integrated), so contacts read the solved
        // rigid velocities and any impulses applied to rigid bodies land on
        // the next step — kept identical here (see `docs/softbody.md` §0/§4).
        // The cluster contact joints solve globally first (Bullet's static
        // `btSoftBody::solveClusters` over all bodies). The multibody
        // context carries the link contacts' jacobian buffers (phase F).
        let (soft_bodies, bodies, multibodies, mb_scratch, soft_mb_jac, soft_mb_delta) = (
            &mut self.soft_bodies,
            &mut self.bodies,
            &mut self.multibodies,
            &mut self.mb_scratch,
            &mut self.soft_mb_jac,
            &mut self.soft_mb_delta,
        );
        solve_clusters(soft_bodies, bodies);
        let mut mb_ctx = MbSolveContext {
            multibodies,
            scratch: mb_scratch,
            jac: soft_mb_jac,
            delta: soft_mb_delta,
        };
        for i in 0..soft_bodies.len() {
            SoftBody::solve_constraints(i, soft_bodies, bodies, Some(&mut mb_ctx));
        }

        // 7t. Soft-body self collisions (Bullet's per-body
        // `defaultCollisionHandler(psb)` loop after the solve): only bodies
        // with both `CL_SS` and `CL_SELF` act.
        for i in 0..self.soft_bodies.len() {
            let c = self.soft_bodies[i].cfg.collisions;
            if c.contains(SoftCollision::CL_SS) && c.contains(SoftCollision::CL_SELF) {
                let (soft_bodies, scratch) =
                    (&mut self.soft_bodies, &mut self.soft_pair_scratch);
                let epa = &mut self.epa_scratch;
                soft_bodies[i].collide_cluster_self(i, scratch, epa);
            }
        }

        // 8s. Soft-body normals refresh (Bullet's `updateSoftBodies` ->
        // `integrateMotion` = `updateNormals`); the face pass joins with
        // the cloth phase, ropes keep zero normals.
        for sb in &mut self.soft_bodies {
            sb.update_normals();
        }

        // 8. Clear accumulated forces for the next step.
        for body in &mut self.bodies {
            body.clear_forces();
        }

        // 8m. Multibody position integration (`integrateMultiBodyTransforms`;
        // sleeping bodies have their velocities cleared), force clearing and
        // the motion/sleep check.
        self.mb_integrate_transforms(dt);
        for mb in &mut self.multibodies {
            if mb.is_awake() {
                mb.clear_forces_and_torques();
            }
            mb.check_motion_and_sleep(dt);
        }
        clk.mark(&mut self.step_timings, StepStage::IntegratePositions);
    }
}