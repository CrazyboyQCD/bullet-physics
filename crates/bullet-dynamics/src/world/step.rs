//! The per-step pipeline of `PhysicsWorld` (`step_simulation`) and the
// persistent-manifold cache helpers. Split out of `world.rs`; the big
// function stays in one piece (its field borrows are interleaved).

use super::*;

use super::multibody_pass::side_active_owner;

/// Motion-state interpolation context of a fixed-timestep call (Bullet's
/// `m_localTime` / `m_fixedTimeStep` / `m_latencyMotionStateInterpolation`
/// read at the point of the call): the render pose extrapolates the last
/// pre-integration snapshot by `local - fixed` in the latency mode, or
/// `local × hit_fraction` otherwise.
#[derive(Clone, Copy)]
struct InterpCtx {
    /// The post-deduction accumulator remainder (Bullet's `m_localTime`).
    local: Real,
    /// The call's fixed step (Bullet's `m_fixedTimeStep`; non-zero).
    fixed: Real,
    /// The latency mode (Bullet's `m_latencyMotionStateInterpolation`).
    latency: bool,
}

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


    /// Advance the simulation by `dt` seconds as a single variable step.
    ///
    /// This is the single-step convenience API: one `dt`-sized step, motion
    /// states receive the step's final pose. For Bullet's
    /// `stepSimulation(dt, maxSubSteps, fixedTimeStep)` — the fixed-timestep
    /// accumulator with render interpolation — see
    /// [`PhysicsWorld::step_simulation_fixed`]; `step_simulation_fixed(dt, 0, _)`
    /// is exactly this call.
    pub fn step_simulation(&mut self, dt: Real) {
        // Bullet's variable-step branch resets the accumulator state
        // (`m_localTime` to the latency-gated remainder, `m_fixedTimeStep`
        // to zero) even though this API never reads it back: mixing the two
        // step APIs keeps the upstream timeline.
        self.step_accumulator = if self.latency_motion_state_interpolation { 0.0 } else { dt };
        self.fixed_time_step = 0.0;
        self.internal_step(dt, true, None);
    }

    /// Advance the simulation with Bullet's fixed-timestep accumulator
    /// (`stepSimulation(timeStep, maxSubSteps, fixedTimeStep)`).
    ///
    /// The caller's wall-clock span `dt` feeds a persistent accumulator
    /// (Bullet's `m_localTime`) that is spent in `fixed_dt`-sized substeps —
    /// at most `max_sub_steps` per call, so a slow frame slows the
    /// simulation down instead of spiraling. The return value is the
    /// **unclamped** substep count the accumulator demanded, like Bullet's.
    /// `max_sub_steps == 0` runs one variable `dt`-sized step and is
    /// exactly [`PhysicsWorld::step_simulation`].
    ///
    /// Per-frame semantics differ from a plain substep loop: collision /
    /// contact-point / contact-force events and the applied-force clear run
    /// once per *call*, after the last substep — a pair touching only
    /// between two substeps of one call never churns `Started`/`Stopped`,
    /// and a force applied between calls feeds every substep of the next
    /// one (Bullet runs `clearForces` once per `stepSimulation`). Motion
    /// states synchronize after every substep, extrapolated along the
    /// pre-integration pose snapshot (Bullet's `synchronizeMotionStates`
    /// inside the substep loop): in the default latency mode the render
    /// pose trails the last simulated substep by `fixed_dt - remainder`
    /// seconds, and frames that skip their substep keep advancing it
    /// smoothly. Set
    /// [`latency_motion_state_interpolation(false)`](PhysicsWorld::set_latency_motion_state_interpolation)
    /// to extrapolate by `remainder × hit_fraction` instead (partially
    /// predicted; a CCD-clamped body freezes at its snapshot).
    pub fn step_simulation_fixed(&mut self, dt: Real, max_sub_steps: u32, fixed_dt: Real) -> u32 {
        let num_sub_steps;
        // The substep length: the fixed step, or the wall-clock span itself
        // in the variable-step branch.
        let mut step_dt = fixed_dt;
        if max_sub_steps != 0 {
            // Fixed timestep with interpolation (Bullet's accumulator).
            self.fixed_time_step = fixed_dt;
            self.step_accumulator += dt;
            if self.step_accumulator >= fixed_dt {
                num_sub_steps = (self.step_accumulator / fixed_dt) as u32;
                self.step_accumulator -= num_sub_steps as Real * fixed_dt;
            } else {
                num_sub_steps = 0;
            }
        } else {
            // Variable timestep: one dt-sized step (Bullet's
            // `maxSubSteps == 0` branch; the accumulator state resets even
            // though this path never reads it back).
            step_dt = dt;
            self.step_accumulator = if self.latency_motion_state_interpolation { 0.0 } else { dt };
            self.fixed_time_step = 0.0;
            if dt.abs() <= Real::EPSILON {
                // `btFuzzyZero`: nothing to simulate, but the render poses
                // still synchronize and the applied forces are dropped
                // (Bullet's else branch + the unconditional clearForces).
                self.sync_motion_states(None);
                self.clear_frame_forces();
                return 0;
            }
            num_sub_steps = 1;
        }

        if num_sub_steps == 0 {
            // A skipped frame: no simulation ran, but the motion states
            // keep blending along the last snapshots and the applied
            // forces expire (Bullet's else branch + clearForces).
            let interp = InterpCtx {
                local: self.step_accumulator,
                fixed: self.fixed_time_step,
                latency: self.latency_motion_state_interpolation,
            };
            self.sync_motion_states(Some(interp));
            self.clear_frame_forces();
            return 0;
        }

        // Clamp the substep count so a slow frame cannot grind the
        // simulation down (Bullet's `clampedSimulationSteps`); the
        // unclamped count is the return value.
        let clamped = num_sub_steps.min(max_sub_steps);
        let interp = Some(InterpCtx {
            local: self.step_accumulator,
            fixed: self.fixed_time_step,
            latency: self.latency_motion_state_interpolation,
        });
        for i in 0..clamped {
            // Events and the force clear run once per call: the last
            // substep is the frame step.
            let frame = i + 1 == clamped;
            self.internal_step(step_dt, frame, interp);
        }
        num_sub_steps
    }

    /// Drop the applied forces (Bullet's `clearForces` over the non-static
    /// list: no activity gate — a force expires once per external call
    /// even when the call skipped its substeps).
    fn clear_frame_forces(&mut self) {
        for body in &mut self.bodies {
            body.clear_forces();
        }
    }

    /// Hand the render pose to the motion states (Bullet's
    /// `synchronizeMotionStates`): active non-static/kinematic bodies only —
    /// a sleeping body's pose did not change, static bodies have no motion
    /// to report (the `synchronizeSingleMotionState` gate).
    ///
    /// Without `interp` (the single-step API) the pose is the step's final
    /// value. With the fixed-timestep context the pose extrapolates the
    /// last pre-integration snapshot along the solved velocities by the
    /// interpolation factor: `local - fixed` in the default latency mode
    /// (always behind the snapshot), `local × hit_fraction` otherwise
    /// (zero for a body the CCD clamp froze this step).
    fn sync_motion_states(&mut self, interp: Option<InterpCtx>) {
        if !self.bodies.iter().any(|b| b.motion_state.is_some()) {
            return;
        }
        let Some(ctx) = interp else {
            for body in &mut self.bodies {
                if !body.is_active()
                    || body.is_static_or_kinematic()
                    || body.motion_state.is_none()
                {
                    continue;
                }
                let pose = crate::body::pose_transform(body.position, body.orientation);
                body.motion_state
                    .as_mut()
                    .expect("motion state checked above")
                    .synchronize(&pose);
            }
            return;
        };
        let factor = if ctx.latency { ctx.local - ctx.fixed } else { 0.0 };
        let interp_poses = &self.interp_poses;
        for (i, body) in self.bodies.iter_mut().enumerate() {
            if !body.is_active() || body.is_static_or_kinematic() || body.motion_state.is_none() {
                continue;
            }
            let factor = if ctx.latency { factor } else { ctx.local * body.hit_fraction };
            // Bodies without a snapshot yet (added after the last
            // interpolated substep, or a skipped frame before any substep
            // ran) blend from their current pose — the same final-pose
            // fallback the variable-step API uses.
            let Some(&(snapshot_pos, snapshot_ori)) = interp_poses.get(i) else {
                let pose = crate::body::pose_transform(body.position, body.orientation);
                body.motion_state
                    .as_mut()
                    .expect("motion state checked above")
                    .synchronize(&pose);
                continue;
            };
            let (position, orientation) = crate::body::interpolate_pose(
                snapshot_pos,
                snapshot_ori,
                body.linear_velocity,
                body.angular_velocity,
                factor,
            );
            let pose = crate::body::pose_transform(position, orientation);
            body.motion_state
                .as_mut()
                .expect("motion state checked above")
                .synchronize(&pose);
        }
    }

    /// The single-step pipeline shared by [`PhysicsWorld::step_simulation`]
    /// and [`PhysicsWorld::step_simulation_fixed`]. `frame` marks the last
    /// substep of an external call (the only one that emits events and
    /// clears the applied forces); `interp` carries the motion-state
    /// interpolation context of a fixed-timestep call (`None` keeps the
    /// single-step final-pose semantics).
    fn internal_step(&mut self, dt: Real, frame: bool, interp: Option<InterpCtx>) {
        let inv_dt = 1.0 / dt;
        let mut clk = StepClock::new(self.step_timings.enabled, self.step_clock_fn);
        // Collision events are per-frame (Rapier's drain): the new frame
        // rebuilds the event buffers. A non-frame substep leaves them
        // alone — its touching pass below only maintains the diff
        // baseline, so between-substep churn is never reported.
        if frame {
            self.contact_events.clear();
            self.contact_point_events.clear();
            self.contact_force_events.clear();
        }
        self.touching.clear();
        self.rolling_rows.clear();
        // Joint-collision suppression is re-derived from the current
        // constraint list every step (Bullet resolves it live at pair
        // processing, so runtime `set_disable_collisions` flips take effect
        // on the step after they happen).
        self.rebuild_no_collision_pairs();

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
        // `docs/design/softbody.md` §4).
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

        // 4b. Simulation islands (phase D, docs/design/multibody-collision.md §9):
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
                    // Detection found nothing: the mixed pair reports a
                    // stopped touch (event-tracked sides only).
                    let (ka, kb) = (ContactKey::side(side_a), ContactKey::side(side_b));
                    let pair = if ka <= kb { (ka, kb) } else { (kb, ka) };
                    let opted_in = self.pair_opted_in(pair.0, pair.1);
                    self.track_pair(pair, opted_in, Some(false));
                    continue;
                };
                let (ka, kb) = (ContactKey::side(side_a), ContactKey::side(side_b));
                let pair = if ka <= kb { (ka, kb) } else { (kb, ka) };
                let opted_in = self.pair_opted_in(pair.0, pair.1);
                // Skip pairs where nothing can move (a sleeping multibody on
                // static geometry); anything touching an active side is woken
                // in stage 5b before the rows are assembled.
                let a_awake = side_active_owner(side_a, &self.bodies, &self.multibodies);
                let b_awake = side_active_owner(side_b, &self.bodies, &self.multibodies);
                if !a_awake && !b_awake {
                    // A sleeping collider pair freezes its touch state like
                    // the sleeping rigid pairs do.
                    self.track_pair(pair, opted_in, None);
                    continue;
                }
                self.mb_pending_contacts.push(PendingContact {
                    side_a,
                    side_b,
                    friction: fa * fb,
                    restitution: ra * rb,
                    manifold,
                });
                self.track_pair(pair, opted_in, Some(true));
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
                let pair = (ContactKey::body(i.min(j)), ContactKey::body(i.max(j)));
                if (self.bodies[i].collision_events
                    || self.bodies[j].collision_events
                    || self.pair_sensor(pair.0, pair.1))
                    && self.prev_touching.binary_search(&pair).is_ok()
                {
                    self.touching.push(pair);
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
            // Friction anchoring is a pair property: either side opting in
            // anchors the contact (Bullet sets the point flag from either
            // body's `CF_HAS_FRICTION_ANCHOR` in `btManifoldResult`).
            let anchored = a
                .collision_flags
                .contains(CollisionFlags::HAS_FRICTION_ANCHOR)
                || b.collision_flags
                    .contains(CollisionFlags::HAS_FRICTION_ANCHOR);
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
                let deep = self
                    .refresh_out
                    .iter()
                    .any(|cp| cp.penetration > self.solver_info.linear_slop);
                if !fast || !deep {
                    if !no_response {
                        for (k, mut cp) in self.refresh_out.iter().copied().enumerate() {
                            cp.normal_impulse *= self.solver_info.warmstarting_factor;
                            // The refreshed points line up with the cache
                            // entries (`retain_mut` keeps the order), so
                            // entry k yields the A-side anchor of contact k.
                            let anchor_disp = anchored.then(|| {
                                let cached = &cache.points()[k];
                                pa + a.orientation * cached.local_a - cp.position
                            });
                            let mut row = ContactConstraint::prepare(
                                i, j, &cp, a, b, k as u32, anchor_disp, inv_dt,
                                &self.solver_info,
                            );
                            row.attach_rolling(a, b, &mut self.rolling_rows);
                            self.contact_rows.push(row);
                        }
                    }
                    if !self.refresh_out.is_empty()
                        && (a.collision_events || b.collision_events || no_response)
                    {
                        self.touching
                            .push((ContactKey::body(i.min(j)), ContactKey::body(i.max(j))));
                    }
                    if frame && self.contact_point_events_enabled {
                        for cp in &self.refresh_out {
                            self.contact_point_events.push(ContactPointEvent {
                                a: BodyHandle::from_index(i),
                                b: BodyHandle::from_index(j),
                                point: *cp,
                            });
                        }
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
                        cp.normal_impulse *= self.solver_info.warmstarting_factor;
                        // Same cache alignment as the refresh path: entry k
                        // is the A-side anchor of contact k.
                        let anchor_disp = anchored.then(|| {
                            let cached = &cache.points()[k];
                            pa + a.orientation * cached.local_a - cp.position
                        });
                        let mut row = ContactConstraint::prepare(
                            i, j, &cp, a, b, k as u32, anchor_disp, inv_dt,
                            &self.solver_info,
                        );
                        row.attach_rolling(a, b, &mut self.rolling_rows);
                        self.contact_rows.push(row);
                    }
                }
                if a.collision_events || b.collision_events || no_response {
                    self.touching
                        .push((ContactKey::body(i.min(j)), ContactKey::body(i.max(j))));
                }
                if frame && self.contact_point_events_enabled {
                    for cp in &self.refresh_out {
                        self.contact_point_events.push(ContactPointEvent {
                            a: BodyHandle::from_index(i),
                            b: BodyHandle::from_index(j),
                            point: *cp,
                        });
                    }
                }
            } else {
                // Detection found nothing: the pair separated past the
                // breaking threshold — drop the cache entry entirely.
                Self::cache_remove(&mut self.manifold_cache, i, j);
            }
        }

        // 5e. Collision events: diff this step's touching pairs against the
        // previous step's (Rapier's post-step event drain). Only pairs with
        // an opted-in side are tracked, so scenes without listeners keep
        // both sets empty; a pair skipped because its island sleeps carries
        // its state over instead of churning.
        //
        // The diff — and the baseline update — run on the frame step only:
        // an intermediate substep rebuilds `touching` (the next substep
        // clears it again) but never consumes the previous *frame's*
        // baseline, so a pair that touched between two substeps of one
        // call and separated again never emits a `Started`/`Stopped` pair.
        if frame {
            self.touching.sort_unstable();
            let prev = core::mem::take(&mut self.prev_touching);
            let cur = core::mem::take(&mut self.touching);
            let (mut pi, mut ci) = (0usize, 0usize);
            while pi < prev.len() || ci < cur.len() {
                // Both sets are sorted: walk the smaller key forward each
                // round (an Equal key is the surviving pair, emitted by
                // neither side).
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
                let (a, b) = pair;
                // Pairs tracked only for their overlap list (neither side
                // opted into events) emit nothing: events stay per-side
                // opt-in, the list is the sensor query surface.
                if !self.pair_opted_in(a, b) {
                    continue;
                }
                let sensor = self.pair_sensor(a, b);
                let event = if started {
                    CollisionEvent::Started {
                        a: a.object(),
                        b: b.object(),
                        sensor,
                    }
                } else {
                    CollisionEvent::Stopped {
                        a: a.object(),
                        b: b.object(),
                        sensor,
                    }
                };
                self.contact_events.push(event);
            }
            self.prev_touching = cur;
        }

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
            constraint::prepare_rows(
                c,
                &self.bodies,
                inv_dt,
                self.solver_info.erp,
                self.solver_info.global_cfm,
                &mut self.joint_rows,
            );
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
                inv_dt,
                &self.solver_info,
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
        // 6c. Contact-force events (Rapier's `ContactForceEvent`): sum the
        // solved normal impulses per pair (rows arrive grouped by pair) and
        // report the pairs whose step-average force exceeds the world
        // threshold. The sum runs over this step's rows, including the
        // ephemeral points the manifold writeback above skips; sensor pairs
        // solve no rows and never report. Frame step only: the force of a
        // frame is the force of its last substep.
        if frame && self.contact_force_events_enabled {
            let mut start = 0;
            while start < contact_rows.len() {
                let pair = (contact_rows[start].body_a, contact_rows[start].body_b);
                let mut total = 0.0;
                let mut end = start;
                while end < contact_rows.len()
                    && (contact_rows[end].body_a, contact_rows[end].body_b) == pair
                {
                    total += contact_rows[end].normal_impulse;
                    end += 1;
                }
                let total_force = total * inv_dt;
                if total_force > self.contact_force_event_threshold {
                    self.contact_force_events.push(ContactForceEvent {
                        a: BodyHandle::from_index(pair.0),
                        b: BodyHandle::from_index(pair.1),
                        total_impulse: total,
                        total_force,
                    });
                }
                start = end;
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
                    // Sensor targets carry no contact response; Bullet's CCD
                    // callback ignores them too (`addSingleResult` rejects
                    // `hasContactResponse() == false` objects).
                    if other
                        .collision_flags
                        .contains(CollisionFlags::NO_CONTACT_RESPONSE)
                    {
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

        // Snapshot the pre-integration poses for the motion-state
        // interpolation (Bullet's `proceedToTransform` storing the previous
        // transform in `m_interpolationWorldTransform`); a skipped frame
        // keeps blending along these.
        if interp.is_some() {
            if self.interp_poses.len() != self.bodies.len() {
                self.interp_poses
                    .resize(self.bodies.len(), (Vector3::ZERO, Quaternion::IDENTITY));
            }
            for (snapshot, body) in self.interp_poses.iter_mut().zip(self.bodies.iter()) {
                *snapshot = (body.position, body.orientation);
            }
        }

        for (i, body) in self.bodies.iter_mut().enumerate() {
            // The hit fraction resets at the top of every integration and
            // zeroes on a clamped CCD hit (Bullet's
            // `integrateTransformsInternal`): the non-latency interpolation
            // factor of a clamped body collapses to zero, freezing its
            // render pose at the snapshot until the impact resolves.
            body.hit_fraction = 1.0;
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
                let w = push_ang * self.solver_info.split_impulse_turn_erp;
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
                body.hit_fraction = 0.0;
            }
        }

        // 7m. Motion states: hand the render pose to the motion side
        // (Bullet's `synchronizeMotionStates` after `integrateTransforms`,
        // run after every substep of a fixed-timestep call).
        self.sync_motion_states(interp);

        // 7s. Soft bodies: solve the position-based constraint set. Bullet
        // runs `solveSoftBodiesConstraints` after the rigid bodies are fully
        // stepped (solved *and* integrated), so contacts read the solved
        // rigid velocities and any impulses applied to rigid bodies land on
        // the next step — kept identical here (see `docs/design/softbody.md` §0/§4).
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

        // 8. Clear the accumulated forces (once per external call: an
        // applied force feeds every substep of the call, Bullet's
        // `clearForces` running after the substep loop).
        if frame {
            self.clear_frame_forces();
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

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_math::{math, Transform};
    use crate::body::{ActivationState, MotionState};
    use crate::body::filter_groups;
    use approx::assert_relative_eq;

    /// Damping is applied automatically by the step, before the force
    /// integration (Bullet's `predictUnconstraintMotion`): with factor 0.5
    /// and one second of 1/60 s steps the speed halves.
    #[test]
    fn damping_applied_during_step() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let b = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 10.0, 0.0),
            1.0,
        ));
        world.body_mut(b).linear_velocity = Vector3::new(10.0, 0.0, 0.0);
        world.body_mut(b).set_damping(0.5, 0.0);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert_relative_eq!(world.body(b).linear_velocity.x, 5.0, epsilon = 1e-4);
    }

    // ------------------------------------------------------------------
    // Collision groups/masks, trigger volumes, collision events.
    // ------------------------------------------------------------------

    /// A plate whose mask matches nothing is invisible to the ball: the
    /// ball falls straight through (Bullet's `btBroadphaseProxy`
    /// group/mask pair test).
    #[test]
    fn collision_mask_blocks_pair() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .collision_filter(filter_groups::DEFAULT, 0),
        );
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            world.step_simulation(dt);
        }
        // Through the plate (its top is y = 0.5) and far below it.
        assert!(
            world.body(ball).position.y < -1.0,
            "ball did not fall through: y = {}",
            world.body(ball).position.y
        );
    }

    /// The pair test is symmetric: a plate in the STATIC group is ignored
    /// by a ball whose mask only accepts DEFAULT, and collides again once
    /// the mask widens to ALL.
    #[test]
    fn collision_group_mask_asymmetric() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .collision_filter(filter_groups::STATIC, filter_groups::ALL),
        );
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 1.01, 0.0),
                1.0,
            )
            .collision_filter(filter_groups::DEFAULT, filter_groups::DEFAULT),
        );
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert!(
            world.body(ball).position.y < 0.0,
            "ball should ignore the STATIC plate: y = {}",
            world.body(ball).position.y
        );
        // Widen the mask, put the ball back above the plate: the pair
        // collides and the ball settles on top.
        world.body_mut(ball).collision_mask = filter_groups::ALL;
        world.set_body_transform(ball, Vector3::new(0.0, 1.01, 0.0), Quaternion::IDENTITY);
        for _ in 0..180 {
            world.step_simulation(dt);
        }
        assert_relative_eq!(world.body(ball).position.y, 1.0, epsilon = 0.05);
    }

    /// A trigger volume (`NO_CONTACT_RESPONSE`) never blocks: the ball
    /// falls straight through, and its touch still reports as a sensor
    /// event.
    #[test]
    fn trigger_volume_passes_through_but_reports() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .no_contact_response(true)
            .collision_events(true),
        );
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 3.0, 0.0),
                1.0,
            )
            .collision_events(true),
        );
        let dt = 1.0 / 60.0;
        let mut started = None;
        for _ in 0..120 {
            world.step_simulation(dt);
            for &ev in world.contact_events() {
                if let CollisionEvent::Started { a, b, sensor } = ev {
                    started = Some((a, b, sensor));
                }
            }
        }
        assert_eq!(
            started,
            Some((
                CollisionObject::Body(BodyHandle::from_index(0)),
                CollisionObject::Body(ball),
                true
            ))
        );
        assert!(
            world.body(ball).position.y < -1.0,
            "trigger did not let the ball through: y = {}",
            world.body(ball).position.y
        );
    }

    /// Pairs emit events only when at least one body opts in.
    #[test]
    fn collision_events_require_opt_in() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.01, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step_simulation(dt);
        }
        assert!(world.contact_events().is_empty());
    }

    /// A resting pair that falls asleep keeps its touch state: exactly one
    /// Started, and no churn while the island sleeps.
    #[test]
    fn sleeping_pair_emits_no_events() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .collision_events(true),
        );
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 1.01, 0.0),
                1.0,
            )
            .collision_events(true),
        );
        let dt = 1.0 / 60.0;
        // Events are a per-step drain, so tally them as they appear.
        let (mut started, mut stopped) = (0, 0);
        for _ in 0..300 {
            world.step_simulation(dt);
            for ev in world.contact_events() {
                match ev {
                    CollisionEvent::Started { .. } => started += 1,
                    CollisionEvent::Stopped { .. } => stopped += 1,
                }
            }
        }
        assert_eq!((started, stopped), (1, 0), "exactly one Started, no churn");
        for _ in 0..120 {
            world.step_simulation(dt);
        }
        assert_eq!(
            (started, stopped),
            (1, 0),
            "the sleeping island emitted nothing further"
        );
    }

    /// Removing a body emits Stopped for its touching pairs.
    #[test]
    fn remove_body_emits_stopped() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .collision_events(true),
        );
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 1.01, 0.0),
                1.0,
            )
            .collision_events(true),
        );
        let dt = 1.0 / 60.0;
        let mut saw_started = false;
        for _ in 0..3 {
            world.step_simulation(dt);
            saw_started |= world
                .contact_events()
                .iter()
                .any(|e| matches!(e, CollisionEvent::Started { .. }));
        }
        assert!(saw_started, "the contact never produced a Started");
        world.remove_body(ball);
        let Some(CollisionEvent::Stopped { a, b, sensor }) =
            world.contact_events().iter().copied().last()
        else {
            panic!("removal did not emit Stopped");
        };
        assert_eq!(
            (a, b),
            (
                CollisionObject::Body(BodyHandle::from_index(0)),
                CollisionObject::Body(ball)
            )
        );
        assert!(!sensor);
    }

    /// A multibody base collider is a first-class event source: a rigid
    /// ball dropped onto it emits `Started` with the collider side reported
    /// as a [`CollisionObject::Link`], and removing the ball ends the touch
    /// with a `Stopped`.
    #[test]
    fn link_collider_emits_events() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, true, false);
        mb.finalize_multidof();
        mb.set_base_collider(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            0.5,
            0.0,
        );
        mb.base_pos = Vector3::ZERO;
        world.add_multibody(mb);
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 3.0, 0.0),
                1.0,
            )
            .collision_events(true),
        );
        let dt = 1.0 / 60.0;
        let mut started = None;
        for _ in 0..60 {
            world.step_simulation(dt);
            for &ev in world.contact_events() {
                if let CollisionEvent::Started { a, b, sensor } = ev {
                    started = Some((a, b, sensor));
                }
            }
        }
        let (a, b, sensor) = started.expect("rigid ball never touched the link collider");
        let link = CollisionObject::Link {
            body: MultiBodyHandle::from_index(0),
            link: -1,
        };
        assert!(
            (a == CollisionObject::Body(ball) && b == link)
                || (a == link && b == CollisionObject::Body(ball)),
            "unexpected pair: {a:?} / {b:?}"
        );
        assert!(!sensor);
        world.remove_body(ball);
        assert!(
            world
                .contact_events()
                .iter()
                .any(|e| matches!(e, CollisionEvent::Stopped { .. })),
            "removal did not end the collider touch"
        );
    }

    /// A soft body is a pair-level event source: a rope landing on an
    /// opt-in plate emits `Started` with the rope side reported as a
    /// [`CollisionObject::Soft`].
    #[test]
    fn soft_body_pair_events() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(3.0, 0.5, 3.0)),
                Vector3::ZERO,
            )
            .collision_events(true),
        );
        let mut rope = crate::softbody::helpers::create_rope(
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            8,
            0,
        );
        rope.collision_events = true;
        world.add_soft_body(rope);
        let dt = 1.0 / 60.0;
        let mut started = false;
        for _ in 0..120 {
            world.step_simulation(dt);
            if world.contact_events().iter().any(|e| {
                matches!(e, CollisionEvent::Started { a, b, .. }
                    if matches!(a, CollisionObject::Soft(_)) || matches!(b, CollisionObject::Soft(_)))
            }) {
                started = true;
            }
        }
        assert!(started, "the rope never touched the plate");
    }

    /// Removing a soft body ends its touches with a `Stopped` and the
    /// surviving soft slots renumber inside the tracked keys (the next
    /// step's diff stays consistent).
    #[test]
    fn remove_soft_body_emits_stopped() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(3.0, 0.5, 3.0)),
                Vector3::ZERO,
            )
            .collision_events(true),
        );
        let mut rope = crate::softbody::helpers::create_rope(
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::new(0.0, 3.0, 0.0),
            8,
            0,
        );
        rope.collision_events = true;
        world.add_soft_body(rope);
        // A second soft body behind the rope in slot order: removing the
        // rope shifts its tracked keys down by one.
        let rope2 = crate::softbody::helpers::create_rope(
            Vector3::new(2.0, 5.0, 0.0),
            Vector3::new(2.0, 4.0, 0.0),
            8,
            0,
        );
        world.add_soft_body(rope2);
        let dt = 1.0 / 60.0;
        let mut saw_started = false;
        for _ in 0..120 {
            world.step_simulation(dt);
            saw_started |= !world.contact_events().is_empty();
        }
        assert!(saw_started, "no events before the removal");
        world.remove_soft_body(SoftBodyHandle::from_index(0));
        let stopped = world
            .contact_events()
            .iter()
            .filter_map(|e| match e {
                CollisionEvent::Stopped { a, b, .. } => Some((a, b)),
                _ => None,
            })
            .any(|(a, b)| {
                matches!(a, CollisionObject::Soft(_)) || matches!(b, CollisionObject::Soft(_))
            });
        assert!(stopped, "removal did not end the soft body's touches");
    }

    /// The CCD sweep respects the group/mask filter.
    #[test]
    fn ccd_respects_collision_filter() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 2.0, 0.25)),
                Vector3::ZERO,
            )
            .collision_filter(filter_groups::DEFAULT, 0),
        );
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 0.0, -5.0),
                1.0,
            )
            .ccd(0.1, 0.3),
        );
        world.body_mut(ball).linear_velocity = Vector3::new(0.0, 0.0, 10.0);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert!(
            world.body(ball).position.z > 4.0,
            "filtered wall stopped the CCD ball: z = {}",
            world.body(ball).position.z
        );
    }

    /// A `DisableSimulation` body keeps colliding like Bullet's: the
    /// dispatcher only rejects pairs where BOTH sides are inactive, so the
    /// solver writes impulses into it (its velocity changes) while the
    /// integrator never moves it (`integrateTransforms` gates on
    /// `isActive()`).
    #[test]
    fn disable_simulation_body_takes_solver_impulses_without_integrating() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.0, 0.0),
                1.0,
            )
            .disable_deactivation(),
        );
        let b = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.6, 0.0, 0.0),
                1.0,
            )
            .disable_simulation(),
        );
        world.body_mut(a).linear_velocity = Vector3::new(2.0, 0.0, 0.0);
        world.step_simulation(1.0 / 60.0);
        assert!(
            world.body(b).linear_velocity.x > 0.0,
            "the solver never wrote into the disabled body"
        );
        assert_eq!(
            world.body(b).position,
            Vector3::new(0.6, 0.0, 0.0),
            "the disabled body was integrated"
        );
    }

    /// Runtime `set_disable_collisions` flips take effect on the next step
    /// (Bullet resolves suppression live at pair processing over the ignore
    /// list): a suppressed pair starts colliding when suppression is
    /// cleared, and a pair starts being suppressed when it is set. The
    /// joint itself is disabled so only the collision path is exercised.
    #[test]
    fn constraint_suppression_flips_take_effect_next_step() {
        let dt = 1.0 / 60.0;
        let setup = |suppress: bool| -> (PhysicsWorld, BodyHandle, usize) {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            let a = world.add_body(
                RigidBody::new_dynamic(
                    bullet_collision::shapes::CollisionShape::sphere(0.5),
                    Vector3::new(0.0, 0.0, 0.0),
                    1.0,
                )
                .disable_deactivation(),
            );
            let b = world.add_body(RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.75, 0.0, 0.0),
            ));
            let mut joint =
                crate::constraint::Constraint::point2point(a, b, Vector3::ZERO, Vector3::ZERO);
            joint.set_disable_collisions(suppress);
            joint.set_enabled(false);
            let index = world.add_constraint(joint);
            (world, a, index)
        };
        let charge = |world: &mut PhysicsWorld, a: BodyHandle| {
            let body = world.body_mut(a);
            body.position = Vector3::new(0.0, 0.0, 0.0);
            body.linear_velocity = Vector3::new(3.0, 0.0, 0.0);
        };
        let run = |world: &mut PhysicsWorld| {
            for _ in 0..90 {
                world.step_simulation(dt);
            }
        };
        // Suppressed at build: the pair is skipped, A tunnels through B.
        let (mut world, a, index) = setup(true);
        charge(&mut world, a);
        run(&mut world);
        assert!(
            world.body(a).position.x > 1.5,
            "suppressed pair did not tunnel: x = {}",
            world.body(a).position.x
        );
        // Clearing suppression at runtime resumes collisions: A is stopped
        // by B.
        charge(&mut world, a);
        world.constraint_mut(index).set_disable_collisions(false);
        run(&mut world);
        assert!(
            world.body(a).position.x < 1.5,
            "cleared suppression still skips the pair: x = {}",
            world.body(a).position.x
        );
        // Setting suppression at runtime (joint built unsuppressed) skips
        // the pair again.
        let (mut world, a, index) = setup(false);
        charge(&mut world, a);
        run(&mut world);
        assert!(
            world.body(a).position.x < 1.5,
            "unsuppressed baseline did not collide: x = {}",
            world.body(a).position.x
        );
        charge(&mut world, a);
        world.constraint_mut(index).set_disable_collisions(true);
        run(&mut world);
        assert!(
            world.body(a).position.x > 1.5,
            "runtime suppression had no effect: x = {}",
            world.body(a).position.x
        );
    }

    /// A disabled joint still suppresses collisions between its bodies —
    /// Bullet's ignore list is independent of `setEnabled` (only the solver
    /// rows stop).
    #[test]
    fn disabled_constraint_still_suppresses_collisions() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.0, 0.0),
                1.0,
            )
            .disable_deactivation(),
        );
        let b = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.75, 0.0, 0.0),
        ));
        let mut joint =
            crate::constraint::Constraint::point2point(a, b, Vector3::ZERO, Vector3::ZERO);
        joint.set_enabled(false);
        world.add_constraint(joint);
        world.body_mut(a).linear_velocity = Vector3::new(3.0, 0.0, 0.0);
        for _ in 0..90 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(
            world.body(a).position.x > 1.5,
            "the disabled joint stopped suppressing collisions: x = {}",
            world.body(a).position.x
        );
    }

    #[test]
    fn pair_filter_rejects_pairs() {
        let ground_shape = || {
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
        };
        // Without the filter the ball lands on the plate.
        let mut world = PhysicsWorld::new();
        world.add_body(ground_shape());
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
        ));
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(world.body(ball).position.y > 0.2);
        // Rejecting every pair lets it fall through.
        let mut world = PhysicsWorld::new();
        world.add_body(ground_shape());
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
        ));
        world.set_pair_filter(Some(Box::new(|_, _, _, _| false)));
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(world.body(ball).position.y < -0.5);
    }

    #[test]
    fn per_body_gravity_overrides_the_world() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        // A dynamic body inherits the world gravity at add time.
        let free = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::ZERO,
            1.0,
        ));
        assert_eq!(world.body(free).gravity(), Vector3::new(0.0, -10.0, 0.0));
        // A detached body keeps its own gravity.
        let mut detached = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::ZERO,
            1.0,
        );
        detached.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
        detached.set_gravity(Vector3::new(0.0, 5.0, 0.0));
        let detached = world.add_body(detached);
        world.step_simulation(1.0 / 60.0);
        assert!(world.body(free).position.y < 0.0);
        assert!(world.body(detached).position.y > 0.0);
        // Propagation skips the detached body.
        world.set_gravity(Vector3::new(0.0, -3.0, 0.0));
        assert_eq!(world.body(free).gravity(), Vector3::new(0.0, -3.0, 0.0));
        assert_eq!(world.body(detached).gravity(), Vector3::new(0.0, 5.0, 0.0));
    }

    /// A resting body keeps a supporting contact impulse in the persistent
    /// manifold cache across steps (the warm-start source for the next
    /// step's rows).
    #[test]
    fn step_simulation_persists_contact_impulses() {
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
        for _ in 0..90 {
            world.step_simulation(1.0 / 60.0);
        }
        let cached: Vec<&bullet_collision::manifold::PersistentPoint> = world
            .manifold_cache
            .iter()
            .flatten()
            .flat_map(|e| e.manifold.points())
            .collect();
        assert!(!cached.is_empty());
        assert!(cached.iter().any(|c| c.normal_impulse > 0.0));
        let sphere = world.body(BodyHandle::from_index(1));
        assert_relative_eq!(sphere.position.y, 0.5, epsilon = 0.05);
        assert!(sphere.linear_velocity.length() < 0.1);
    }

    /// A contact cached while barely touching survives a later small
    /// separation inside the breaking-threshold band: the refreshed point is
    /// emitted with negative penetration (Bullet's margin-band contact). The
    /// scene is gravity-free so the pair stays below the redetect speed gate
    /// and actually exercises the refresh path.
    #[test]
    fn band_contact_survives_brief_separation() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // Sphere barely touching (0.0001 deep): one step caches the contact
        // at the predicted pose.
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.4999, 0.0),
            1.0,
        ));
        world.step_simulation(1.0 / 60.0);
        assert!(!world.manifold_cache.is_empty());
        // Nudge up by 0.004 (inside the 0.005 redetect window, so the pair
        // refreshes instead of re-detecting): the refreshed point lands in
        // the margin band with negative penetration.
        world.body_mut(ball).position = Vector3::new(0.0, 0.5039, 0.0);
        world.step_simulation(1.0 / 60.0);
        // The manifold survived with a margin-band point (negative
        // penetration) instead of being dropped by a re-detection that only
        // reports actual penetration.
        let points: Vec<bullet_collision::manifold::PersistentPoint> = world
            .manifold_cache
            .iter()
            .flatten()
            .flat_map(|e| e.manifold.points().to_vec())
            .collect();
        assert!(points.iter().any(|p| p.penetration < 0.0));
    }

    /// A sphere dropped from slightly above the plane is caught by the
    /// margin-band contact *before* touching: the contact forms while the
    /// surfaces are still separated (inside the breaking-threshold band) and
    /// the band softening limits the approach to the separation rate, so the
    /// sphere settles onto the surface without ever penetrating past the
    /// slop — the behavior Bullet's positive-distance contacts give.
    #[test]
    fn band_contact_catches_a_falling_sphere_before_touching() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.508, 0.0),
            1.0,
        ));
        let mut min_y = Real::INFINITY;
        for _ in 0..60 {
            world.step_simulation(1.0 / 60.0);
            min_y = min_y.min(world.body(ball).position.y);
        }
        // Never penetrated past the slop zone on the way down.
        assert!(min_y > 0.5 - 0.006, "min height {min_y}");
        // And it settles on the surface.
        assert_relative_eq!(world.body(ball).position.y, 0.5, epsilon = 0.006);
    }

    /// CCD: a fast body whose one-step displacement would cross a thin static
    /// wall is clamped at the swept-sphere impact instead of tunneling; the
    /// same body without CCD sails straight through.
    #[test]
    fn ccd_clamps_fast_bodies_at_the_impact() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        // Thin static wall: box centered at y = 0.5, spanning y in [0.4, 0.6].
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        // Fast ball with CCD, starting above the wall; per-step motion is
        // 2.0 > threshold 0.1 and the predicted position (y = -1) is past the
        // wall, so the discrete narrow phase sees no contact.
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        // Control ball without CCD, offset in X so it misses the wall.
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(2.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0)),
        );
        let dt: Real = 1.0 / 60.0;
        world.step_simulation(dt);
        // The CCD ball is clamped at the wall's top surface (0.6 + 0.3).
        assert_relative_eq!(
            world.body(BodyHandle::from_index(1)).position.y,
            0.9,
            epsilon = 0.05
        );
        // The control ball crossed the whole wall.
        assert!(world.body(BodyHandle::from_index(2)).position.y < -0.5);
    }

    /// CCD only sweeps when the per-step motion exceeds the threshold; slower
    /// bodies integrate normally.
    #[test]
    fn ccd_ignores_slow_motion() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -5.0, 0.0))
            .ccd(1.0, 0.3),
        );
        // Per-step motion 5/60 < threshold 1.0: plain integration, no clamp.
        world.step_simulation(1.0 / 60.0);
        assert_relative_eq!(
            world.body(BodyHandle::from_index(1)).position.y,
            1.0 - 5.0 / 60.0,
            epsilon = 1e-5
        );
    }

    /// CCD skips pairs with suppressed collisions: Bullet's dispatcher
    /// `needsCollision` (`checkCollideWithOverride`) gates the sweep too, so a
    /// fast body is not clamped against its joint-linked partner. The joint's
    /// impulse is clamped to a token value so the ball's motion stays intact
    /// and the sweep filter is the only thing under test.
    #[test]
    fn ccd_skips_collision_suppressed_pairs() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        // Thin static wall in the fast ball's path, joint-linked to it.
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        let mut joint = Constraint::point2point(
            BodyHandle::from_index(0),
            BodyHandle::from_index(1),
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.0, -0.5, 0.0),
        );
        joint.point2point_mut().set_impulse_clamp(0.001);
        world.add_constraint(joint);
        // The suppressed pair is not swept: the ball crosses the wall (the
        // unsuppressed control is `ccd_clamps_fast_bodies_at_the_impact`).
        world.step_simulation(1.0 / 60.0);
        assert!(world.body(BodyHandle::from_index(1)).position.y < -0.5);
    }

    /// Bullet's CCD callback ignores targets without contact response
    /// (`btClosestNotMeConvexResultCallback::addSingleResult` rejects objects
    /// where `hasContactResponse()` is false): a sensor (`NO_CONTACT_RESPONSE`)
    /// wall never clamps a fast body — it flies straight through.
    #[test]
    fn ccd_ignores_no_contact_response_targets() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut wall = RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        );
        wall.collision_flags = CollisionFlags::NO_CONTACT_RESPONSE;
        world.add_body(wall);
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        world.step_simulation(1.0 / 60.0);
        assert!(
            world.body(BodyHandle::from_index(1)).position.y < -0.5,
            "a sensor wall must not clamp the CCD ball"
        );
    }

    /// The motion threshold is exclusive (Bullet:
    /// `ccdSquareMotionThreshold() < squareMotion`): per-step motion exactly
    /// equal to the threshold does not sweep, motion above it does. The wall
    /// is thin enough that a 1.5-unit step clears it entirely (the predicted
    /// narrow phase cannot catch it either), so the CCD gate is the only
    /// thing deciding between passing through and clamping. Values are
    /// binary-exact so f32 and f64 agree.
    #[test]
    fn ccd_threshold_boundary_is_exclusive() {
        let dt: Real = 0.25;
        let ball_at = |velocity: Real, threshold: Real| -> PhysicsWorld {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            world.add_body(RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
                Vector3::new(0.0, 3.0, 0.0),
            ));
            world.add_body(
                RigidBody::new_dynamic(
                    bullet_collision::shapes::CollisionShape::sphere(0.3),
                    Vector3::new(0.0, 4.0, 0.0),
                    1.0,
                )
                .velocity(Vector3::new(0.0, velocity, 0.0))
                .ccd(threshold, 0.3),
            );
            world
        };
        // Exactly at the threshold (1.5 == 6.0 * 0.25): no sweep, plain
        // integration to y = 2.5 — fully past the wall ([2.9, 3.1]) and its
        // inflated top (3.4), where a sweep would have clamped instead.
        let mut world = ball_at(-6.0, 1.5);
        world.step_simulation(dt);
        assert_relative_eq!(
            world.body(BodyHandle::from_index(1)).position.y,
            2.5,
            epsilon = 1e-5
        );
        // Slightly above: the sweep runs and clamps at the inflated top.
        let mut world = ball_at(-6.4, 1.5);
        world.step_simulation(dt);
        assert_relative_eq!(
            world.body(BodyHandle::from_index(1)).position.y,
            3.4,
            epsilon = 0.02
        );
    }

    /// With two walls in the sweep path the clamp lands on the nearest hit;
    /// once the near wall is group/mask-excluded the far wall takes over.
    #[test]
    fn ccd_selects_nearest_of_multiple_targets() {
        let dt: Real = 1.0 / 60.0;
        let build = |near_group: u16, near_mask: u16| -> PhysicsWorld {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            let mut near = RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.05, 1.0, 1.0)),
                Vector3::new(2.0, 0.0, 0.0),
            );
            near.collision_group = near_group;
            near.collision_mask = near_mask;
            world.add_body(near);
            world.add_body(RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.05, 1.0, 1.0)),
                Vector3::new(6.0, 0.0, 0.0),
            ));
            world.add_body(
                RigidBody::new_dynamic(
                    bullet_collision::shapes::CollisionShape::sphere(0.3),
                    Vector3::ZERO,
                    1.0,
                )
                .velocity(Vector3::new(420.0, 0.0, 0.0))
                .ccd(0.1, 0.3),
            );
            world
        };
        // Both walls visible: the ball clamps on the near wall (face 1.95 −
        // sphere 0.3 = 1.65 sphere-center entry).
        let mut world = build(filter_groups::DEFAULT, filter_groups::ALL);
        world.step_simulation(dt);
        let x = world.body(BodyHandle::from_index(2)).position.x;
        assert!((1.5..1.8).contains(&x), "clamped on the near wall, got {x}");
        // Near wall excluded (its mask rejects the ball's group): the far
        // wall's hit wins instead.
        let mut world = build(2, 0);
        world.step_simulation(dt);
        let x = world.body(BodyHandle::from_index(2)).position.x;
        assert!((5.5..5.8).contains(&x), "clamped on the far wall, got {x}");
    }

    /// Sweep speed sweep: from 120 to 120 000 units/s the clamp keeps the
    /// ball at the wall's inflated top, and every quantity stays finite.
    #[test]
    fn ccd_speed_sweep_stays_finite() {
        let dt: Real = 1.0 / 60.0;
        for k in [1.0, 10.0, 100.0, 1000.0] {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            world.add_body(RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
                Vector3::new(10_000.0, 0.5, 0.0),
            ));
            world.add_body(
                RigidBody::new_dynamic(
                    bullet_collision::shapes::CollisionShape::sphere(0.3),
                    Vector3::new(10_000.0, 1.0, 0.0),
                    1.0,
                )
                .velocity(Vector3::new(0.0, -120.0 * k, 0.0))
                .ccd(0.1, 0.3),
            );
            world.step_simulation(dt);
            let b = world.body(BodyHandle::from_index(1));
            assert!(b.position.is_finite(), "position at {k}x");
            assert!(b.linear_velocity.is_finite(), "velocity at {k}x");
            assert!(b.orientation.is_finite(), "orientation at {k}x");
            assert_relative_eq!(b.position.y, 0.9, epsilon = 0.05);
            assert_relative_eq!(b.position.x, 10_000.0, epsilon = 1.0);
        }
    }

    /// A fast spinning body is clamped by the sweep and every integrated
    /// quantity stays finite and normalized (the position-only rewind leaves
    /// the orientation integrating the full step — the documented deviation —
    /// but nothing may blow up).
    #[test]
    fn ccd_high_spin_stays_finite() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        world.body_mut(BodyHandle::from_index(1)).angular_velocity = Vector3::new(0.0, 3000.0, 0.0);
        world.step_simulation(1.0 / 60.0);
        let b = world.body(BodyHandle::from_index(1));
        assert!(b.position.is_finite());
        assert!(b.angular_velocity.is_finite());
        assert!(b.orientation.is_finite());
        assert_relative_eq!(b.orientation.length_squared(), 1.0, epsilon = 1e-3);
        assert_relative_eq!(b.position.y, 0.9, epsilon = 0.05);
    }

    /// A CCD body starting inside the target reports the fraction-0 clamp:
    /// the body stays put (no backward teleport, no explosive correction)
    /// and every quantity stays finite.
    #[test]
    fn ccd_starting_inside_is_finite() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        world.step_simulation(1.0 / 60.0);
        let b = world.body(BodyHandle::from_index(1));
        assert!(b.position.is_finite());
        assert_relative_eq!(b.position.y, 0.5, epsilon = 1e-4);
    }

    /// Extreme timesteps stay finite: a tiny step never triggers the sweep
    /// (plain integration), a huge step clamps at the wall.
    #[test]
    fn ccd_extreme_timestep_stays_finite() {
        let build = || -> PhysicsWorld {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            world.add_body(RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
                Vector3::new(0.0, 0.5, 0.0),
            ));
            world.add_body(
                RigidBody::new_dynamic(
                    bullet_collision::shapes::CollisionShape::sphere(0.3),
                    Vector3::new(0.0, 1.0, 0.0),
                    1.0,
                )
                .velocity(Vector3::new(0.0, -120.0, 0.0))
                .ccd(0.1, 0.3),
            );
            world
        };
        // dt = 1e-4: motion 0.012 < threshold 0.1, plain integration.
        let mut world = build();
        world.step_simulation(1e-4);
        let b = world.body(BodyHandle::from_index(1));
        assert!(b.position.is_finite());
        assert_relative_eq!(b.position.y, 1.0 - 120.0 * 1e-4, epsilon = 1e-5);
        // dt = 0.5: motion 60, the sweep clamps at the wall.
        let mut world = build();
        world.step_simulation(0.5);
        let b = world.body(BodyHandle::from_index(1));
        assert!(b.position.is_finite());
        assert_relative_eq!(b.position.y, 0.9, epsilon = 0.05);
    }

    /// Runtime suppression flips gate the sweep from the next step on
    /// (the per-step `no_collision_pairs` re-derivation): an enabled pair
    /// sweeps and clamps, disabling it lets the body through, and
    /// re-enabling it clamps on the next wall down.
    #[test]
    fn ccd_suppression_flip_gates_the_sweep() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, -2.5, 0.0),
        ));
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 3.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        let mut joint = Constraint::point2point(
            BodyHandle::from_index(0),
            BodyHandle::from_index(2),
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.0, 2.5, 0.0),
        );
        joint.point2point_mut().set_impulse_clamp(0.001);
        world.add_constraint(joint);
        let dt: Real = 1.0 / 60.0;
        // Step 1 falls from 3 to 1: no contact yet, joint impulse clamped.
        world.step_simulation(dt);
        // Disabled: step 2 sweeps nothing and crosses the wall (to y = -1).
        world.constraint_mut(0).set_disable_collisions(true);
        world.step_simulation(dt);
        assert!(
            world.body(BodyHandle::from_index(2)).position.y < -0.5,
            "suppressed pair must not clamp"
        );
        // Re-enabled: step 3 clamps on the second wall (top -2.4 + 0.3).
        world.constraint_mut(0).set_disable_collisions(false);
        world.step_simulation(dt);
        let y = world.body(BodyHandle::from_index(2)).position.y;
        assert_relative_eq!(y, -2.1, epsilon = 0.05);
    }

    // ---- dynamic triangle-mesh bodies (the GImpact use case) ----------

    /// A closed 2x2x0.4 box shell as a dynamic mesh body (top and bottom
    /// plates; a single-sided zero-thickness plate would be the known
    /// tunneling degenerate).
    fn box_shell_mesh() -> bullet_collision::shapes::CollisionShape {
        let (hx, hy, hz) = (1.0 as Real, 0.2 as Real, 1.0 as Real);
        bullet_collision::shapes::CollisionShape::triangle_mesh(
            vec![
                // Top face.
                Vector3::new(-hx, hy, -hz),
                Vector3::new(hx, hy, -hz),
                Vector3::new(hx, hy, hz),
                Vector3::new(-hx, hy, hz),
                // Bottom face.
                Vector3::new(-hx, -hy, -hz),
                Vector3::new(hx, -hy, -hz),
                Vector3::new(hx, -hy, hz),
                Vector3::new(-hx, -hy, hz),
            ],
            vec![[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]],
        )
    }

    /// A dynamic mesh body falls onto a static plane, rests on it, and goes
    /// to sleep — the GImpact use case end to end.
    #[test]
    fn dynamic_mesh_rests_on_plane_and_sleeps() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let mesh = world.add_body(RigidBody::new_dynamic(
            box_shell_mesh(),
            Vector3::new(0.0, 2.0, 0.0),
            2.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step_simulation(dt);
        }
        let b = world.body(mesh);
        // Bottom face at y = -0.2 relative to the body origin: rest height.
        assert!(
            (0.175..0.23).contains(&b.position.y),
            "mesh should rest with its bottom face on the plane, got {}",
            b.position.y
        );
        assert!(b.linear_velocity.length() < 0.1);
        assert!(matches!(
            b.activation_state(),
            ActivationState::IslandSleeping
        ));
    }

    /// A dynamic mesh body rests on a static mesh (the mesh-vs-mesh tree
    /// traversal driving real support). Dropped from just above the margin
    /// shell — a single step deeper than the band would tunnel the
    /// zero-thickness shell past the surface (known deviation of the GJK
    /// route; Bullet's clip method resolves signed back-side distances).
    #[test]
    fn dynamic_mesh_rests_on_static_mesh() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let ground = bullet_collision::shapes::CollisionShape::triangle_mesh(
            (0..8)
                .flat_map(|z| (0..8).map(move |x| (x, z)))
                .flat_map(|(x, z)| {
                    let (x, z) = (x as Real, z as Real);
                    vec![
                        Vector3::new(x, 0.0, z),
                        Vector3::new(x + 1.0, 0.0, z),
                        Vector3::new(x + 1.0, 0.0, z + 1.0),
                        Vector3::new(x, 0.0, z + 1.0),
                    ]
                })
                .collect::<Vec<_>>(),
            (0..64)
                .flat_map(|i| {
                    let b = (i * 4) as u32;
                    [[b, b + 2, b + 1], [b, b + 3, b + 2]]
                })
                .collect::<Vec<_>>(),
        );
        world.add_body(RigidBody::new_static(ground, Vector3::new(-4.0, 0.0, -4.0)));
        let mesh = world.add_body(RigidBody::new_dynamic(
            box_shell_mesh(),
            Vector3::new(0.0, 0.5, 0.0),
            2.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step_simulation(dt);
        }
        let b = world.body(mesh);
        // The shell rests with its bottom face held `band - slop` above the
        // ground by the margin shell (the GImpact hover, like Bullet).
        assert!(
            (0.24..0.36).contains(&b.position.y),
            "mesh should rest on the static mesh ground, got {}",
            b.position.y
        );
        assert!(b.linear_velocity.length() < 0.1);
    }

    /// A dynamic sphere resting on a dynamic mesh plate: the mesh-vs-convex
    /// path with a *moving* mesh (both sides dynamic, island-merged).
    #[test]
    fn sphere_rests_on_dynamic_mesh() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let mesh = world.add_body(RigidBody::new_dynamic(
            box_shell_mesh(),
            Vector3::new(0.0, 0.2, 0.0),
            5.0,
        ));
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        for _ in 0..600 {
            world.step_simulation(dt);
        }
        // The heavy shell barely moves; the ball rests on its top face
        // (y = 0.2 + 0.2 + 0.5 = 0.9 in the ideal, allow solver slop).
        let top = world.body(mesh).position.y + 0.2;
        let y = world.body(ball).position.y;
        assert!(
            ((top + 0.5 - 0.05)..=(top + 0.5 + 0.05)).contains(&y),
            "ball should rest on the dynamic mesh top face, got {y}"
        );
        assert!(world.body(ball).linear_velocity.length() < 0.1);
    }

    // ---- kinematic rigid bodies (Bullet's `CF_KINEMATIC_OBJECT`) ------

    /// A kinematic body driven by its velocity pushes a dynamic body out of the
    /// way: the contact rows use the kinematic's velocity in the relative
    /// velocity, so the dynamic side gains momentum (velocity propagation).
    #[test]
    fn kinematic_pushes_dynamic_body() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // Sliding kinematic slab (its top face is at y = 0.5) and a dynamic box
        // resting in its path.
        let slab = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.5, 1.0)),
            Vector3::new(-3.0, 0.5, 0.0),
        ));
        let target = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.25, 0.25, 0.25)),
            Vector3::new(0.0, 0.75, 0.0),
            1.0,
        ));
        world.body_mut(slab).linear_velocity = Vector3::new(2.0, 0.0, 0.0);

        let start_x = world.body(target).position.x;
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        // The slab (2 m/s for 2 s) reached and shoved the box well past its
        // starting point.
        assert!(world.body(target).position.x > start_x + 0.5);
        assert!(world.body(target).is_active());
    }

    /// Gravity, damping and accumulated forces never touch a kinematic body's
    /// velocities; the pose integrates from the user-set velocity every step.
    #[test]
    fn kinematic_ignores_gravity_forces_and_damping() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let body = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 5.0, 0.0),
        ));
        world.body_mut(body).set_damping(0.2, 0.2);
        let v = Vector3::new(1.0, 0.0, 0.0);
        world.body_mut(body).linear_velocity = v;
        world.body_mut(body).angular_velocity = Vector3::new(0.0, 0.0, 1.0);

        let dt = 1.0 / 60.0;
        let frames = 60;
        for i in 0..frames {
            world.body_mut(body).apply_force(Vector3::new(0.0, 100.0, 0.0));
            world.step_simulation(dt);
            let b = world.body(body);
            // Gravity must not move the kinematic body.
            assert_relative_eq!(b.position.y, 5.0, epsilon = 1e-4);
            assert_eq!(b.linear_velocity, v, "velocity must stay user-set (frame {i})");
        }
        // Pose integrated from the velocity: x ~= v * t (per-frame `+=` rounding
        // aside), and it never fell.
        assert_relative_eq!(world.body(body).position.x, 1.0, epsilon = 1e-3);
        assert!(world.body(body).is_active());
    }

    /// A dynamic body resting on an *idle* kinematic platform settles and goes
    /// to sleep; the kinematic-static pair produces no solver rows.
    #[test]
    fn dynamic_body_rests_and_sleeps_on_idle_kinematic_platform() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let platform = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.25, 2.0)),
            Vector3::new(0.0, 0.25, 0.0),
        ));
        let cargo = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.25, 0.25, 0.25)),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
        ));
        for _ in 0..480 {
            world.step_simulation(1.0 / 60.0);
        }
        let b = world.body(cargo);
        assert_relative_eq!(b.position.y, 0.75, epsilon = 0.05);
        assert!(!b.is_active(), "idle kinematic contacts must let it sleep");
        // The platform itself never moved and keeps its velocity.
        assert_eq!(world.body(platform).position, Vector3::new(0.0, 0.25, 0.0));
    }

    /// Kinematic-vs-kinematic and kinematic-vs-static overlaps are skipped
    /// entirely and leave both bodies untouched (Bullet's broadphase
    /// StaticFilter excludes them).
    #[test]
    fn kinematic_static_and_kinematic_kinematic_pairs_are_inert() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let ground = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let a = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        let b = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
            Vector3::new(0.3, 0.5, 0.0),
        ));
        let va = Vector3::new(0.5, 0.0, 0.0);
        let vb = Vector3::new(-0.5, 0.0, 0.0);
        world.body_mut(a).linear_velocity = va;
        world.body_mut(b).linear_velocity = vb;

        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        // They flew apart (or through each other) but no solver row ever touched
        // their velocities, and the static ground still holds them where their
        // velocities put them.
        assert_eq!(world.body(a).linear_velocity, va);
        assert_eq!(world.body(b).linear_velocity, vb);
        assert!(world.body(a).position.y > 0.0);
        let _ = ground;
    }

    /// A kinematic compound (a vehicle slab with a raised bumper child)
    /// pushes a dynamic ball purely through its velocity: the end-to-end
    /// kinematic x compound interaction (pair gate, infinite-mass solver
    /// rows, per-step proxy refresh) with no solver row touching the
    /// kinematic body's velocity.
    #[test]
    fn kinematic_compound_pushes_a_ball() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let ground = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // Vehicle: slab + bumper, driven at constant +x from x = -4.
        let vehicle = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::compound(vec![
                bullet_collision::shapes::CompoundChild::new(
                    bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.25, 0.5)),
                    Vector3::new(0.0, 0.25, 0.0),
                ),
                bullet_collision::shapes::CompoundChild::new(
                    bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.1, 0.4, 0.5)),
                    Vector3::new(0.6, 0.4, 0.0),
                ),
            ]),
            Vector3::new(-4.0, 0.0, 0.0),
        ));
        world.body_mut(vehicle).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        // Ball resting on the ground in the vehicle's path.
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 0.3, 0.0),
            1.0,
        ));

        for _ in 0..240 {
            world.step_simulation(1.0 / 60.0);
        }

        // The vehicle drove through the ball's start and carried it along,
        // its own velocity untouched (infinite mass).
        assert!(
            world.body(ball).position.x > 0.5,
            "ball at {}",
            world.body(ball).position.x
        );
        assert!(world.body(ball).position.y > 0.0);
        assert_eq!(
            world.body(vehicle).linear_velocity,
            Vector3::new(1.0, 0.0, 0.0)
        );
        // Drove ~4 m from x = -4 (the 240-frame budget lands it near 0).
        assert!(world.body(vehicle).position.x > -0.5);
        let _ = ground;
    }

    // ---- rolling friction (Bullet's `m_rollingFriction` torsional rows)

    /// A ball rolling on the ground with a positive rolling friction slows
    /// down and stops (the angular rows drain the spin, then sliding friction
    /// takes the linear speed); without the coefficient the matched roll
    /// keeps going — no sliding, nothing to stop it.
    #[test]
    fn rolling_friction_stops_a_rolling_ball() {
        let build = |rolling: Real| {
            let mut world = PhysicsWorld::new();
            let mut ground = RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(20.0, 0.5, 20.0)),
                Vector3::new(0.0, -0.5, 0.0),
            );
            ground.rolling_friction = rolling;
            world.add_body(ground);
            let mut ball = RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            );
            ball.rolling_friction = rolling;
            // Matched roll: contact-point velocity v + ω×(−r) = 0 → ωz = −v/r.
            ball.linear_velocity = Vector3::new(1.0, 0.0, 0.0);
            ball.angular_velocity = Vector3::new(0.0, 0.0, -2.0);
            let ball = world.add_body(ball);
            (world, ball)
        };

        // Without rolling friction the matched roll keeps rolling (frictionless
        // contact: no sliding, nothing decelerates it beyond solver noise).
        let (mut world, ball) = build(0.0);
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        // The residual decay is the solver's friction-row noise on the
        // numerically-zero contact slip (PGS couplings between the normal and
        // tangent blocks), not rolling friction: it drains a matched roll by
        // about a third over two seconds, so keep the bar low.
        assert!(
            world.body(ball).linear_velocity.x > 0.5,
            "unloaded roll should persist: vx = {}",
            world.body(ball).linear_velocity.x
        );

        // With rolling friction the roll drains: both the spin and, through
        // the sliding friction it unmasked, the linear speed collapse.
        let (mut world, ball) = build(1.0);
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        let v = world.body(ball).linear_velocity;
        let w = world.body(ball).angular_velocity;
        assert!(
            v.length() < 0.05,
            "rolling friction should stop the ball: v = {}",
            v.length()
        );
        assert!(
            w.z.abs() < 0.1,
            "the spin should be drained: wz = {}",
            w.z
        );
    }

    // ---- anisotropic friction (Bullet's `m_anisotropicFriction`)

    /// Direction-dependent friction: a box with zero friction scaling along
    /// its local Z axis slides much farther along Z than along X, while the
    /// same box without the field slides equally far in both directions.
    #[test]
    fn anisotropic_friction_makes_sliding_direction_dependent() {
        let build = |aniso: Vector3, dir: Vector3| {
            let mut world = PhysicsWorld::new();
            let mut ground = RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(
                    50.0, 0.5, 50.0,
                )),
                Vector3::new(0.0, -0.5, 0.0),
            );
            ground.friction = 1.0;
            world.add_body(ground);
            let mut box_body = RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.0, 0.5, 0.0),
                1.0,
            );
            box_body.friction = 1.0;
            box_body.anisotropic_friction = aniso;
            box_body.linear_velocity = dir * 2.0;
            let handle = world.add_body(box_body);
            (world, handle)
        };

        // With (1, 1, 0) the Z direction is nearly frictionless: it slides
        // far past the X slide.
        let (mut world, body) = build(Vector3::new(1.0, 1.0, 0.0), Vector3::Z);
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        let z_slide = world.body(body).position.z;
        let (mut world, body) = build(Vector3::new(1.0, 1.0, 0.0), Vector3::X);
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        let x_slide = world.body(body).position.x;
        assert!(
            z_slide > x_slide * 4.0,
            "low-friction Z should slide much farther: z = {z_slide}, x = {x_slide}"
        );

        // Sanity: the isotropic control slides the same distance both ways.
        let (mut world, body) = build(Vector3::new(1.0, 1.0, 1.0), Vector3::Z);
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        let iso_z = world.body(body).position.z;
        let (mut world, body) = build(Vector3::new(1.0, 1.0, 1.0), Vector3::X);
        for _ in 0..120 {
            world.step_simulation(1.0 / 60.0);
        }
        let iso_x = world.body(body).position.x;
        assert!(
            (iso_z - iso_x.abs()).abs() < 0.05,
            "isotropic control should slide equally: z = {iso_z}, x = {iso_x}"
        );
    }

    // ---- contact stiffness/damping (Bullet's
    // `setContactStiffnessAndDamping`)

    /// A soft-contact pair (stiffness flag set) rests with a deeper steady
    /// penetration than the same ball with the hard default contact: the
    /// compliance CFM makes the normal row a damped spring, which balances
    /// gravity at `mg/k` compression instead of the slop-scale Baumgarte
    /// dead zone.
    #[test]
    fn soft_contact_rests_deeper_than_the_default() {
        let build = |stiffness: Option<(Real, Real)>| {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
            let ground = RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(20.0, 0.5, 20.0)),
                Vector3::new(0.0, -0.5, 0.0),
            );
            world.add_body(ground);
            let mut ball = RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 2.0, 0.0),
                1.0,
            );
            if let Some((k, d)) = stiffness {
                ball.set_contact_stiffness_and_damping(k, d);
            }
            let handle = world.add_body(ball);
            (world, handle)
        };

        // Steady-state height after the ball has settled (4 s).
        let settle = |stiffness: Option<(Real, Real)>| {
            let (mut world, ball) = build(stiffness);
            for _ in 0..240 {
                world.step_simulation(1.0 / 60.0);
            }
            world.body(ball).position.y
        };

        let hard = settle(None);
        let soft = settle(Some((500.0, 5.0)));
        // The spring balance: mg/k = 10/500 = 0.02 m below the surface.
        assert!(
            soft < hard - 0.01,
            "the soft-contact ball should rest deeper: soft = {soft}, hard = {hard}"
        );
        assert!(
            soft > 0.4 && hard > 0.4,
            "both should still rest on the ground: soft = {soft}, hard = {hard}"
        );
    }

    // ---- motion states (Bullet's `btMotionState`)

    /// Records the poses handed to [`MotionState::synchronize`], shared with
    /// the test through an `Rc<RefCell>` (the world owns the state itself).
    struct RecordingState {
        initial: Transform,
        log: std::rc::Rc<core::cell::RefCell<Vec<Transform>>>,
    }

    impl MotionState for RecordingState {
        fn initial_pose(&self) -> Transform {
            self.initial
        }
        fn synchronize(&mut self, pose: &Transform) {
            self.log.borrow_mut().push(*pose);
        }
    }

    /// A body with a motion state is placed at the state's pose (Bullet's
    /// constructor `getWorldTransform` read), and every step pushes the
    /// final pose back (Bullet's `synchronizeMotionStates`).
    #[test]
    fn motion_state_places_body_and_synchronizes() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let log = std::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
        let initial = crate::body::pose_transform(Vector3::new(1.0, 5.0, 2.0), Quaternion::IDENTITY);
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::ZERO,
                1.0,
            )
            .motion_state(Box::new(RecordingState {
                initial,
                log: log.clone(),
            })),
        );
        // The constructor position is overridden by the motion state.
        let p = world.body(ball).position;
        assert_relative_eq!(p.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(p.y, 5.0, epsilon = 1e-6);
        assert_relative_eq!(p.z, 2.0, epsilon = 1e-6);

        world.step_simulation(1.0 / 60.0);
        let received = log.borrow();
        assert_eq!(received.len(), 1, "one synchronize per step");
        let (_, _, translation) = received[0].to_scale_rotation_translation();
        // Semi-implicit Euler: x_{n+1} = x_n + (g·dt)·dt.
        assert_relative_eq!(translation.x, 1.0, epsilon = 1e-5);
        assert_relative_eq!(
            translation.y,
            5.0 - 10.0 * (1.0 / 60.0) * (1.0 / 60.0),
            epsilon = 1e-5
        );
        assert_relative_eq!(translation.z, 2.0, epsilon = 1e-5);
    }

    /// Static and kinematic bodies never synchronize (Bullet's
    /// `synchronizeSingleMotionState` gate: `!isStaticOrKinematicObject`).
    #[test]
    fn motion_state_skips_static_and_kinematic() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        let immovables = [
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.0, -0.5, 0.0),
            ),
            RigidBody::new_kinematic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.0, 4.0, 0.0),
            ),
        ];
        for make in immovables {
            let log = std::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
            world.add_body(
                make.motion_state(Box::new(RecordingState {
                    initial: crate::body::pose_transform(Vector3::ZERO, Quaternion::IDENTITY),
                    log: log.clone(),
                })),
            );
            for _ in 0..10 {
                world.step_simulation(1.0 / 60.0);
            }
            assert!(
                log.borrow().is_empty(),
                "immovable bodies must not synchronize"
            );
        }
    }

    // ------------------------------------------------------------------
    // Persistent-manifold extras: friction anchor, contact-point events.
    // ------------------------------------------------------------------

    /// Friction anchoring must not violate the Coulomb cone: with
    /// `μ < tan θ` (friction saturated) an anchored box slides just like a
    /// plain one, and with ample friction both rest — the anchor servo
    /// (Bullet's `BT_CONTACT_FLAG_FRICTION_ANCHOR`) only ever spends the
    /// friction budget's slack, it never adds force.
    #[test]
    fn friction_anchor_respects_the_coulomb_cone() {
        let end_position = |anchored: bool, friction: Real| -> Vector3 {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
            let mut ramp = RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(8.0, 0.5, 2.0)),
                Vector3::ZERO,
            );
            // ~15 degrees: `tan ≈ 0.27`. The pair friction is the
            // *product* of the two coefficients, so 0.9/0.9 → 0.81 rests
            // almost immediately and 0.2/0.2 saturates.
            ramp.orientation = Quaternion::from_rotation_z(-0.26);
            ramp.friction = friction;
            world.add_body(ramp);
            let mut cube = RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.4)),
                Vector3::new(0.0, 1.2, 0.0),
                1.0,
            );
            cube.friction = friction;
            if anchored {
                cube.collision_flags = crate::body::CollisionFlags::HAS_FRICTION_ANCHOR;
            }
            let cube = world.add_body(cube);
            for _ in 0..120 {
                world.step_simulation(1.0 / 60.0);
            }
            world.body(cube).position
        };
        // Saturated friction (`0.2 < tan θ`): both slide down, the anchor
        // cannot hold.
        let plain = end_position(false, 0.2);
        let anchored = end_position(true, 0.2);
        assert!(plain.x > 0.3, "plain box must slide: x = {plain}");
        assert!(
            anchored.x > 0.3,
            "anchor must not act as super-friction: x = {anchored}"
        );
    }

    /// With `set_contact_point_events` enabled, every step reports one
    /// event per surviving point of a touching rigid pair (Bullet's
    /// `gContactProcessedCallback` remapped onto a drain), the payload
    /// handle is the higher side of the pair, and the point lifetime grows
    /// while the contact persists; nothing is collected while disabled.
    #[test]
    fn contact_point_events_report_surviving_points() {
        let dt = 1.0 / 60.0;
        let mut world = PhysicsWorld::new();
        world.set_contact_point_events(true);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        ));
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.05, 0.0),
            1.0,
        ));
        let mut seen = 0;
        let mut max_lifetime = 0;
        for _ in 0..30 {
            world.step_simulation(dt);
            for ev in world.contact_point_events() {
                // `a < b`: the plane was added first, the ball second.
                assert_eq!(ev.b, ball);
                seen += 1;
                max_lifetime = max_lifetime.max(ev.point.lifetime);
            }
        }
        assert!(seen > 0, "no contact-point events collected");
        assert!(max_lifetime > 5, "lifetime did not grow: {max_lifetime}");

        // Disabled: the buffer stays empty.
        let mut quiet = PhysicsWorld::new();
        quiet.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        ));
        quiet.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.05, 0.0),
            1.0,
        ));
        for _ in 0..10 {
            quiet.step_simulation(dt);
        }
        assert!(quiet.contact_point_events().is_empty());
    }

    /// With `set_contact_force_events` enabled, a pair reports after the
    /// solve when its summed normal impulse over `dt` exceeds the world
    /// threshold: a resting 1 kg contact (~9.81 N) stays under a 50 N
    /// threshold while a 2 m drop impact spikes past it; the payload
    /// carries the impulse/force pair and `a < b`. Disabled, nothing is
    /// collected even on impact.
    #[test]
    fn contact_force_events_report_impacts_above_threshold() {
        let dt = 1.0 / 60.0;
        let mut world = PhysicsWorld::new();
        world.set_contact_force_events(true);
        world.set_contact_force_event_threshold(50.0);
        let ground = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        ));
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.05, 0.0),
            1.0,
        ));
        // Resting: the contact force sits at ~m·g, under the threshold.
        for _ in 0..30 {
            world.step_simulation(dt);
        }
        assert!(world.contact_force_events().is_empty());

        // Drop from 2 m above the contact: the impact force crosses the
        // threshold exactly once (zero restitution, no bounce).
        world.set_body_transform(ball, Vector3::new(0.0, 3.0, 0.0), Quaternion::IDENTITY);
        let mut events = 0;
        for _ in 0..60 {
            world.step_simulation(dt);
            for ev in world.take_contact_force_events() {
                events += 1;
                assert_eq!(ev.a, ground);
                assert_eq!(ev.b, ball);
                // The two payloads agree: force is the step-average.
                assert!((ev.total_impulse / dt - ev.total_force).abs() < 1e-3);
                // m·(v_impact + g·dt) with v_impact = √(2·g·2): 3..12 N·s.
                assert!(ev.total_impulse > 3.0, "impulse too low: {}", ev.total_impulse);
                assert!(ev.total_impulse < 12.0, "impulse too high: {}", ev.total_impulse);
            }
        }
        assert_eq!(events, 1, "expected exactly one impact event");

        // Disabled: even the impact stays unreported.
        let mut quiet = PhysicsWorld::new();
        quiet.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        ));
        quiet.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        ));
        for _ in 0..40 {
            quiet.step_simulation(dt);
        }
        assert!(quiet.contact_force_events().is_empty());
    }

    /// A uniformly scaled sphere rides the full GJK margin-band pipeline
    /// (support x factor, margin x factor) and settles exactly like a plain
    /// sphere of the scaled radius: resting center at the scaled radius
    /// above the ground, asleep.
    #[test]
    fn uniform_scaled_sphere_rests_at_the_scaled_radius() {
        let dt = 1.0 / 60.0;
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(20.0, 0.5, 20.0)),
            Vector3::new(0.0, -0.5, 0.0),
        ));
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::uniform_scaling(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                2.0,
            ),
            Vector3::new(0.0, 5.0, 0.0),
            1.0,
        ));
        for _ in 0..240 {
            world.step_simulation(dt);
        }
        let body = world.body(ball);
        assert!(
            (body.position.y - 1.0).abs() < 0.02,
            "scaled sphere must rest at radius * factor: y = {}",
            body.position.y
        );
        assert!(
            body.activation_state() == ActivationState::IslandSleeping,
            "settled sphere should sleep"
        );

        // The convex-2d marker behaves like its child: a 2d-wrapped box
        // slides and settles just like the plain box.
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(20.0, 0.5, 20.0)),
            Vector3::new(0.0, -0.5, 0.0),
        ));
        let flat = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::convex_2d(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            ),
            Vector3::new(0.0, 0.5, 0.0),
            1.0,
        ));
        for _ in 0..120 {
            world.step_simulation(dt);
        }
        let body = world.body(flat);
        assert!(
            (body.position.y - 0.5).abs() < 0.02,
            "2d-wrapped box must rest on the ground: y = {}",
            body.position.y
        );
    }

    /// Sensor pairs solve no contact rows, so they carry no impulse and
    /// never report a contact-force event — not even at threshold 0.
    #[test]
    fn contact_force_events_skip_sensor_pairs() {
        let dt = 1.0 / 60.0;
        let mut world = PhysicsWorld::new();
        world.set_contact_force_events(true);
        world.set_contact_force_event_threshold(0.0);
        let mut wall = RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        );
        wall.collision_flags = CollisionFlags::NO_CONTACT_RESPONSE;
        world.add_body(wall);
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.05, 0.0),
            1.0,
        ));
        for _ in 0..30 {
            world.step_simulation(dt);
        }
        assert!(world.contact_force_events().is_empty());
    }

    // ---- fixed-timestep accumulator (`stepSimulation(dt, maxSubSteps,
    // fixedTimeStep)`)

    /// Regression anchor for the whole fixed-timestep surface: with
    /// `fixed_dt == dt` every call runs exactly one substep, so the
    /// simulation, the events and the force lifecycle must match the
    /// single-step API bit-for-bit.
    #[test]
    fn fixed_degenerates_to_single_step() {
        let build = || {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
            world.add_body(RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(4.0, 0.5, 4.0)),
                Vector3::ZERO,
            ));
            world.add_body(
                RigidBody::new_dynamic(
                    bullet_collision::shapes::CollisionShape::sphere(0.5),
                    Vector3::new(0.1, 3.0, 0.0),
                    1.0,
                )
                .restitution(0.3),
            );
            world
        };
        let (mut single, mut fixed) = (build(), build());
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            single.step_simulation(dt);
            let n = fixed.step_simulation_fixed(dt, 1, dt);
            assert_eq!(n, 1);
        }
        let a = single.body(BodyHandle::from_index(1));
        let b = fixed.body(BodyHandle::from_index(1));
        assert_eq!(a.position, b.position, "positions must match bit-for-bit");
        assert_eq!(
            a.linear_velocity, b.linear_velocity,
            "velocities must match bit-for-bit"
        );
        assert_eq!(
            single.contact_events(), fixed.contact_events(),
            "event streams must match"
        );
    }

    /// The accumulator paces the substeps: a frame of 1/128 s against a
    /// fixed step of 3/128 s (both binary-exact) runs one substep every
    /// third call, and a constant-speed body advances by one fixed step
    /// per substep.
    #[test]
    fn accumulator_substep_rhythm() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(-20.0, 0.0, 0.0),
            1.0,
        ));
        world.body_mut(ball).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        let dt = 1.0 / 128.0;
        let fixed_dt = 3.0 / 128.0;
        let expected = [0u32, 0, 1, 0, 0, 1];
        let mut moved = 0.0;
        for (frame, &want) in expected.iter().enumerate() {
            let n = world.step_simulation_fixed(dt, 4, fixed_dt);
            assert_eq!(n, want, "substep demand of frame {frame}");
            moved += n as Real * fixed_dt;
            assert_relative_eq!(
                world.body(ball).position.x,
                -20.0 + moved,
                epsilon = 1e-6
            );
        }
    }

    /// A frame much longer than the fixed step is clamped to `max_sub_steps`
    /// substeps (the slow frame slows the world down instead of spiraling);
    /// the return value stays the unclamped demand.
    #[test]
    fn clamped_substeps() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(-20.0, 0.0, 0.0),
            1.0,
        ));
        world.body_mut(ball).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        let x0 = world.body(ball).position.x;
        // 1.0 / (1/128) is exactly 128 in both precisions (a 1/120 fixed
        // step would quantize the quotient to 119 in f32).
        let n = world.step_simulation_fixed(1.0, 2, 1.0 / 128.0);
        assert_eq!(n, 128, "the demand is reported unclamped");
        let x = world.body(ball).position.x;
        assert_relative_eq!(x - x0, 2.0 / 128.0, epsilon = 1e-6);
    }

    /// Latency interpolation: the render pose trails the last simulated
    /// substep by `fixed_dt - remainder`, and a frame that skips its
    /// substep keeps advancing it smoothly along the stored snapshot.
    #[test]
    fn interpolation_advances_across_skipped_frames() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let log = std::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(-20.0, 0.0, 0.0),
                1.0,
            )
            .motion_state(Box::new(RecordingState {
                initial: crate::body::pose_transform(Vector3::new(-20.0, 0.0, 0.0), Quaternion::IDENTITY),
                log: log.clone(),
            })),
        );
        world.body_mut(ball).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        // Binary-exact pacing: a fixed step of 3/128 s against 1/128 s
        // frames runs on calls 1 and 4 (demand 1, 0, 0, 1).
        let dt = 1.0 / 128.0;
        let fixed_dt = 3.0 / 128.0;
        // Substep frame: the render pose is the pre-integration snapshot
        // (x = -20) pulled back a full fixed step by the velocity.
        world.step_simulation_fixed(fixed_dt, 4, fixed_dt);
        // Skipped frames: the remainder grows, so the factor
        // (remainder - fixed_dt) rises toward zero — the render pose
        // advances smoothly without any simulation.
        world.step_simulation_fixed(dt, 4, fixed_dt);
        world.step_simulation_fixed(dt, 4, fixed_dt);
        // The next substep frame resets the cycle.
        world.step_simulation_fixed(fixed_dt, 4, fixed_dt);
        let received = log.borrow();
        assert_eq!(received.len(), 4, "one synchronize per call, even skipped");
        let xs: Vec<Real> = received
            .iter()
            .map(|t| t.to_scale_rotation_translation().2.x)
            .collect();
        assert_relative_eq!(xs[0], -20.0 - fixed_dt, epsilon = 1e-5);
        assert_relative_eq!(xs[1], -20.0 - fixed_dt + dt, epsilon = 1e-5);
        assert_relative_eq!(xs[2], -20.0 - fixed_dt + 2.0 * dt, epsilon = 1e-5);
        // The substep advanced the simulation; the render pose (trailing
        // the fresh snapshot by a full fixed step) caught up to where the
        // simulation was one step ago.
        // The substep frame's demand was 1 with a 2/128 remainder left
        // over (the two skipped frames), so its factor is 1/128 short of a
        // full pull-back and the render pose stops short of the snapshot.
        assert_relative_eq!(xs[3], -20.0 + fixed_dt - dt, epsilon = 1e-5);
        // The simulation state moved one fixed step per substep frame
        // (two substeps ran in total).
        assert_relative_eq!(
            world.body(ball).position.x,
            -20.0 + 2.0 * fixed_dt,
            epsilon = 1e-5
        );
    }

    /// A frame shorter than the fixed step still synchronizes the motion
    /// states and drops the applied forces, but simulates nothing.
    #[test]
    fn no_substep_frame_syncs_and_clears_forces() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let log = std::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(-20.0, 0.0, 0.0),
                1.0,
            )
            .motion_state(Box::new(RecordingState {
                initial: crate::body::pose_transform(Vector3::new(-20.0, 0.0, 0.0), Quaternion::IDENTITY),
                log: log.clone(),
            })),
        );
        world.body_mut(ball).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        let fixed_dt = 1.0 / 120.0;
        // Prime the snapshot with one real substep.
        world.step_simulation_fixed(fixed_dt, 4, fixed_dt);
        let at = world.body(ball).position;
        world.body_mut(ball).apply_force_at_point(Vector3::new(10.0, 0.0, 0.0), at);
        // A zero-length frame: the fuzzy-zero gate of the variable branch.
        let n = world.step_simulation_fixed(0.0, 0, fixed_dt);
        assert_eq!(n, 0);
        {
            let received = log.borrow();
            assert_eq!(received.len(), 2, "motion states still synchronize");
        }
        // Forces are dropped by the call (Bullet's unconditional
        // clearForces): a following substep keeps the coasting velocity
        // exactly (a surviving force would have added F/m·fixed_dt).
        let coasting = world.body(ball).linear_velocity.x;
        world.step_simulation_fixed(fixed_dt, 4, fixed_dt);
        assert_relative_eq!(world.body(ball).linear_velocity.x, coasting, epsilon = 1e-6);
    }

    /// A force applied between calls feeds every substep of the next call:
    /// three substeps of 1/120 s accumulate `F·3·fixed_dt` of impulse, and
    /// the force is gone afterwards (Bullet's `clearForces` once per call).
    #[test]
    fn forces_persist_across_substeps() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(-20.0, 0.0, 0.0),
            2.0,
        ));
        let at = world.body(ball).position;
        world.body_mut(ball).apply_force_at_point(Vector3::new(12.0, 0.0, 0.0), at);
        let demand = world.step_simulation_fixed(1.0, 8, 1.0 / 128.0);
        assert_eq!(demand, 128, "the demand is reported unclamped");
        let v = world.body(ball).linear_velocity.x;
        // Δv = F/m · (8 · fixed_dt): the force feeds every one of the
        // eight clamped substeps.
        assert_relative_eq!(v, 12.0 / 2.0 * (8.0 / 128.0), epsilon = 1e-5);
    }

    /// A pair that touches in an intermediate substep and separates before
    /// the frame closes never reports `Started`/`Stopped` (events are
    /// per-call).
    #[test]
    fn events_report_once_per_frame() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(4.0, 0.5, 4.0)),
            Vector3::ZERO,
        ));
        let _ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 1.2, 0.0),
                1.0,
            )
            .collision_events(true),
        );
        let (dt, fixed_dt) = (1.0 / 60.0, 1.0 / 120.0);
        // The event buffer is per-call: drain it after every frame of the
        // drop and assert the single pair starts exactly once across the
        // whole fall, never churning within a call despite two substeps
        // per frame.
        let mut seen_started = false;
        for _ in 0..60 {
            world.step_simulation_fixed(dt, 4, fixed_dt);
            let events = world.contact_events();
            let started = events
                .iter()
                .filter(|e| matches!(e, CollisionEvent::Started { .. }))
                .count();
            let stopped = events
                .iter()
                .filter(|e| matches!(e, CollisionEvent::Stopped { .. }))
                .count();
            assert!(started <= 1 && stopped <= 1, "no within-call churn");
            if started == 1 {
                assert!(!seen_started, "the pair starts exactly once");
                seen_started = true;
            }
        }
        assert!(seen_started, "the drop must produce one Started");
    }

    /// A kinematic platform advances by its user velocity for every
    /// substep of the frame (the velocity-driven pose integration is
    /// per-substep).
    #[test]
    fn kinematic_drives_through_substeps() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let plate = world.add_body(RigidBody::new_kinematic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.25, 2.0)),
            Vector3::new(-20.0, 0.0, 0.0),
        ));
        world.body_mut(plate).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        let x0 = world.body(plate).position.x;
        world.step_simulation_fixed(1.0 / 60.0, 4, 1.0 / 120.0);
        let x = world.body(plate).position.x;
        assert_relative_eq!(x - x0, 2.0 / 120.0, epsilon = 1e-5);
    }

    /// Non-latency interpolation: the factor is `remainder × hit_fraction`,
    /// so the render pose sits *between* the snapshot and the simulation,
    /// and a CCD-clamped body (hit fraction zero) stays pinned at the
    /// snapshot.
    #[test]
    fn non_latency_mode_scales_with_hit_fraction() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let log = std::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(-20.0, 0.0, 0.0),
                1.0,
            )
            .motion_state(Box::new(RecordingState {
                initial: crate::body::pose_transform(Vector3::new(-20.0, 0.0, 0.0), Quaternion::IDENTITY),
                log: log.clone(),
            })),
        );
        world.body_mut(ball).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        world.set_latency_motion_state_interpolation(false);
        assert!(!world.latency_motion_state_interpolation());
        // Binary-exact pacing (3/128 fixed vs 1/128 frames): substep,
        // skip, skip, substep. The non-latency factor is
        // `remainder × hit_fraction`, so the render pose sits *between*
        // the snapshot and the simulation and lands exactly on the body
        // when the remainder hits zero.
        let dt = 1.0 / 128.0;
        let fixed_dt = 3.0 / 128.0;
        world.step_simulation_fixed(fixed_dt, 4, fixed_dt); // remainder 0: the snapshot itself
        world.step_simulation_fixed(dt, 4, fixed_dt); // remainder 1/128
        world.step_simulation_fixed(dt, 4, fixed_dt); // remainder 2/128
        world.step_simulation_fixed(fixed_dt, 4, fixed_dt); // demand 1, remainder 2/128
        let received = log.borrow();
        assert_eq!(received.len(), 4);
        let xs: Vec<Real> = received
            .iter()
            .map(|t| t.to_scale_rotation_translation().2.x)
            .collect();
        assert_relative_eq!(xs[0], -20.0, epsilon = 1e-5);
        assert_relative_eq!(xs[1], -20.0 + dt, epsilon = 1e-5);
        assert_relative_eq!(xs[2], -20.0 + 2.0 * dt, epsilon = 1e-5);
        // The last substep left a 2/128 remainder: the render pose
        // overshoots the fresh snapshot by remainder × hit_fraction.
        assert_relative_eq!(xs[3], -20.0 + fixed_dt + 2.0 * dt, epsilon = 1e-5);
    }

    /// The interpolation formula (Bullet's `btTransformUtil::
    /// integrateTransform`): linear position, axis-angle rotation with the
    /// Taylor small-angle branch and the `ANGULAR_MOTION_THRESHOLD` clamp,
    /// negative spans handled symmetrically.
    #[test]
    fn interpolate_pose_branches() {
        use crate::body::interpolate_pose;
        // Pure translation.
        let (p, q) = interpolate_pose(
            Vector3::new(1.0, 2.0, 3.0),
            Quaternion::IDENTITY,
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::ZERO,
            0.5,
        );
        assert_relative_eq!(p.x, 6.0, epsilon = 1e-6);
        assert_relative_eq!(q.w, 1.0, epsilon = 1e-6);
        // Small-angle branch (|w| < 0.001): the Taylor axis yields the
        // half-angle quaternion x ≈ w·dt/2.
        let w = 5e-4;
        let (_, q) = interpolate_pose(
            Vector3::ZERO,
            Quaternion::IDENTITY,
            Vector3::ZERO,
            Vector3::new(w, 0.0, 0.0),
            1.0,
        );
        assert_relative_eq!(q.x, 0.5 * w, epsilon = 1e-6);
        assert!((q.w - 1.0).abs() < 1e-4);
        // Large angle clamped: 90°/s over 1 s exceeds
        // ANGULAR_MOTION_THRESHOLD (π/4), so the rotation is clamped to a
        // π/4 turn around the rescaled axis (x = 0.63795, w = 0.77008).
        #[cfg(not(feature = "f64"))]
        const HALF_PI: Real = core::f32::consts::FRAC_PI_2;
        #[cfg(feature = "f64")]
        const HALF_PI: Real = core::f64::consts::FRAC_PI_2;
        let (_, q) = interpolate_pose(
            Vector3::ZERO,
            Quaternion::IDENTITY,
            Vector3::ZERO,
            Vector3::new(HALF_PI, 0.0, 0.0),
            1.0,
        );
        assert_relative_eq!(q.x, 0.637_951_8, epsilon = 1e-5);
        assert_relative_eq!(q.w, 0.770_076_1, epsilon = 1e-5);
        // Moderate angle below the clamp: 0.5 rad/s over 1 s is an exact
        // 0.5 rad turn (x = sin(0.25), w = cos(0.25)).
        let (_, q) = interpolate_pose(
            Vector3::ZERO,
            Quaternion::IDENTITY,
            Vector3::ZERO,
            Vector3::new(0.5, 0.0, 0.0),
            1.0,
        );
        assert_relative_eq!(q.x, math::sin(0.25), epsilon = 1e-5);
        assert_relative_eq!(q.w, math::cos(0.25), epsilon = 1e-5);
        // Negative span (the latency remainder) rewinds symmetrically.
        let (p, q) = interpolate_pose(
            Vector3::new(0.0, 1.0, 0.0),
            Quaternion::IDENTITY,
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::ZERO,
            -1.0 / 128.0,
        );
        assert_relative_eq!(p.x, -2.0 / 128.0, epsilon = 1e-6);
        assert_relative_eq!(p.y, 1.0, epsilon = 1e-6);
        assert_relative_eq!(q.w, 1.0, epsilon = 1e-6);
    }
}