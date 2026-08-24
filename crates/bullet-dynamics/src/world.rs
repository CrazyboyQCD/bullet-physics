use std::collections::HashMap;
use std::collections::HashSet;

use bullet_collision::broadphase::{AxisSweep3, BroadPhase};
use bullet_collision::manifold::ContactManifold;
use bullet_collision::narrowphase::{self, SweepHit};
use bullet_collision::raycast::{self, RayHit};
use crate::body::RigidBody;
use crate::constraint::{self, Constraint};
use crate::multibody::body::{MultiBody, MultiBodyScratch};
use crate::multibody::constraint::{MultiBodyConstraint, SolverData, SolverRow};
use bullet_math::{Aabb, Quaternion, Real, Vector3};
use crate::solver::sequential_impulse::{self, BAUMGARTE, ContactConstraint, ConstraintRow, SPLIT_TURN_ERP};

/// Scale applied to a cached impulse before the warm-start pass (Bullet's
/// `m_warmstartingFactor`): old impulses are faded slightly so the solver can
/// correct drift instead of blindly trusting the previous step.
const WARM_STARTING_FACTOR: Real = 0.85;
/// Distance below which a fresh contact point is the same physical contact as
/// a cached one (Bullet's `gContactBreakingThreshold`), keeping its impulse.
const CONTACT_BREAKING_THRESHOLD: Real = 0.02;

/// A multibody collider registered in the broad phase: `(multibody, link)`
/// with `link = -1` for the base, plus its proxy id once assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColliderEntry {
    mb: usize,
    link: i32,
    proxy: Option<usize>,
}

/// What a broad-phase proxy slot belongs to (rigid bodies and multibody
/// link colliders share one proxy index space).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProxyOwner {
    Body(usize),
    Collider(usize),
}

/// A cached contact point from the previous step: the accumulated normal
/// impulse is re-applied on the next step's warm-start pass (the
/// persistent-manifold `m_appliedImpulse` cache in Bullet). Friction impulses
/// are deliberately not cached — see [`warm_start_impulses`].
struct CachedContact {
    position: Vector3,
    normal: Vector3,
    normal_impulse: Real,
}

/// A discrete rigid-body simulation world, mirroring `btDiscreteDynamicsWorld`
/// extended with the Featherstone multibody list of `btMultiBodyDynamicsWorld`
/// (phase A of the collision-integration design: one fused world, stepping
/// both subsystems; link-collider collision integration follows in later
/// phases — see `docs/multibody-collision.md`).
pub struct PhysicsWorld {
    bodies: Vec<RigidBody>,
    broadphase: AxisSweep3,
    gravity: Vector3,
    iterations: u32,
    /// Accumulated contact impulses from the previous step, keyed by body
    /// pair; the cache is rebuilt every step from the solved constraints.
    contact_cache: HashMap<(usize, usize), Vec<CachedContact>>,
    /// Joints (Bullet's `btTypedConstraint` list added with `addConstraint`).
    constraints: Vec<Constraint>,
    /// Body pairs whose collision response is suppressed by a joint
    /// (`disableCollisionsBetweenLinkedBodies`, Bullet's
    /// `checkCollideWithOverride`), sorted `(min, max)`.
    no_collision_pairs: HashSet<(usize, usize)>,
    /// Broad-phase proxy id for each rigid body (proxy ids are shared with
    /// the multibody link colliders, so they no longer coincide with body
    /// indices).
    body_proxies: Vec<usize>,
    /// Multibody colliders registered in the broad phase (Bullet's
    /// `btMultiBodyLinkCollider` list), one entry per attached collider.
    collider_entries: Vec<ColliderEntry>,
    /// Owner of every broad-phase proxy slot, for translating proxy pairs
    /// and query candidates back to bodies or colliders.
    proxy_owners: Vec<ProxyOwner>,
    /// Featherstone articulated bodies (Bullet's `btMultiBody` list, added
    /// with `addMultiBody`).
    multibodies: Vec<MultiBody>,
    /// Multibody constraints (Bullet's `btMultiBodyConstraint` list).
    mb_constraints: Vec<MultiBodyConstraint>,
    /// Inner iterations for the multibody non-contact rows per outer
    /// iteration (Bullet's `m_numNonContactInnerIterations`, default 1).
    num_non_contact_inner_iterations: u32,
    /// Multibody constraint error-reduction factor (Bullet's `m_erp`, 0.2).
    mb_erp: Real,
    /// Multibody solver scratch shared across steps.
    mb_scratch: MultiBodyScratch,
    /// Multibody solver pool (jacobians + velocity deltas).
    mb_data: SolverData,
    /// Flattened multibody constraint rows for the current step.
    mb_rows: Vec<SolverRow>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            broadphase: AxisSweep3::new(),
            gravity: Vector3::new(0.0, -9.81, 0.0),
            iterations: 10,
            contact_cache: HashMap::new(),
            constraints: Vec::new(),
            no_collision_pairs: HashSet::new(),
            body_proxies: Vec::new(),
            collider_entries: Vec::new(),
            proxy_owners: Vec::new(),
            multibodies: Vec::new(),
            mb_constraints: Vec::new(),
            num_non_contact_inner_iterations: 1,
            mb_erp: 0.2,
            mb_scratch: MultiBodyScratch::default(),
            mb_data: SolverData::default(),
            mb_rows: Vec::new(),
        }
    }

    pub fn set_gravity(&mut self, gravity: Vector3) {
        self.gravity = gravity;
    }

    pub fn gravity(&self) -> Vector3 {
        self.gravity
    }

    /// Number of solver (PGS) iterations per step.
    pub fn set_iterations(&mut self, iterations: u32) {
        self.iterations = iterations;
    }

    /// Add a body and return its index.
    ///
    /// Broad-phase proxy ids are shared with multibody link colliders, so
    /// they are tracked in [`PhysicsWorld::body_proxies`] (visible to tests);
    /// body removal is not yet supported.
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let aabb = body.shape.world_aabb(body.position, body.orientation);
        let proxy = self.broadphase.create_proxy(aabb);
        let index = self.bodies.len();
        self.bodies.push(body);
        debug_assert_eq!(proxy, self.proxy_owners.len());
        self.proxy_owners.push(ProxyOwner::Body(index));
        self.body_proxies.push(proxy);
        index
    }

    pub fn body(&self, index: usize) -> &RigidBody {
        &self.bodies[index]
    }

    pub fn body_mut(&mut self, index: usize) -> &mut RigidBody {
        &mut self.bodies[index]
    }

    pub fn bodies(&self) -> &[RigidBody] {
        &self.bodies
    }

    /// Inner iterations for the multibody non-contact (joint) rows per outer
    /// iteration (Bullet's `m_numNonContactInnerIterations`, default 1).
    pub fn set_num_non_contact_inner_iterations(&mut self, n: u32) {
        self.num_non_contact_inner_iterations = n;
    }

    /// Multibody constraint error-reduction factor (Bullet's `m_erp`,
    /// default 0.2).
    pub fn set_multibody_erp(&mut self, erp: Real) {
        self.mb_erp = erp;
    }

    /// Add a finalized multibody (Bullet's `addMultiBody`) and return its
    /// index.
    pub fn add_multibody(&mut self, body: MultiBody) -> usize {
        self.multibodies.push(body);
        self.multibodies.len() - 1
    }

    pub fn multibody(&self, index: usize) -> &MultiBody {
        &self.multibodies[index]
    }

    pub fn multibody_mut(&mut self, index: usize) -> &mut MultiBody {
        &mut self.multibodies[index]
    }

    pub fn multibodies(&self) -> &[MultiBody] {
        &self.multibodies
    }

    pub fn num_multibodies(&self) -> usize {
        self.multibodies.len()
    }

    /// Add a multibody constraint (Bullet's `addMultiBodyConstraint`) and
    /// return its index.
    pub fn add_multibody_constraint(&mut self, constraint: MultiBodyConstraint) -> usize {
        self.mb_constraints.push(constraint);
        self.mb_constraints.len() - 1
    }

    pub fn multibody_constraint(&self, index: usize) -> &MultiBodyConstraint {
        &self.mb_constraints[index]
    }

    pub fn multibody_constraint_mut(&mut self, index: usize) -> &mut MultiBodyConstraint {
        &mut self.mb_constraints[index]
    }

    pub fn multibody_constraints(&self) -> &[MultiBodyConstraint] {
        &self.mb_constraints
    }

    pub fn num_multibody_constraints(&self) -> usize {
        self.mb_constraints.len()
    }

    /// Remove a multibody constraint by index (swaps the last one in to keep
    /// the list dense).
    pub fn remove_multibody_constraint(&mut self, index: usize) {
        self.mb_constraints.swap_remove(index);
    }

    /// Register any attached-but-unregistered multibody colliders as
    /// broad-phase proxies (Bullet's `addCollisionObject` when a
    /// `btMultiBodyLinkCollider` is added). Forward kinematics must have run
    /// this step so the world transforms are fresh.
    fn register_pending_colliders(&mut self) {
        let mut wanted: Vec<(usize, i32)> = Vec::new();
        for (mb_i, mb) in self.multibodies.iter().enumerate() {
            if mb.base_collider().is_some() {
                wanted.push((mb_i, -1));
            }
            for l in 0..mb.num_links() {
                if mb.link_collider(l).is_some() {
                    wanted.push((mb_i, l as i32));
                }
            }
        }
        for (mb_i, link) in wanted {
            if self
                .collider_entries
                .iter()
                .any(|e| e.mb == mb_i && e.link == link)
            {
                continue;
            }
            self.collider_entries.push(ColliderEntry {
                mb: mb_i,
                link,
                proxy: None,
            });
        }
        // Assign broad-phase proxies to the new entries.
        for i in 0..self.collider_entries.len() {
            if self.collider_entries[i].proxy.is_some() {
                continue;
            }
            let aabb = {
                let e = &self.collider_entries[i];
                let mb = &self.multibodies[e.mb];
                if e.link < 0 {
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
            };
            let proxy = self.broadphase.create_proxy(aabb);
            self.collider_entries[i].proxy = Some(proxy);
            debug_assert_eq!(proxy, self.proxy_owners.len());
            self.proxy_owners.push(ProxyOwner::Collider(i));
        }
    }

    /// The `(multibody, link, collider)` triple behind a collider proxy
    /// owner (`None` for rigid-body owners).
    fn collider_of(&self, owner: ProxyOwner) -> Option<(usize, i32, &crate::multibody::LinkCollider)> {
        let ProxyOwner::Collider(e) = owner else {
            return None;
        };
        let ent = &self.collider_entries[e];
        let mb = &self.multibodies[ent.mb];
        let col = if ent.link < 0 {
            mb.base_collider()?
        } else {
            mb.link_collider(ent.link as usize)?
        };
        Some((ent.mb, ent.link, col))
    }

    /// Cast a ray against every body and return the closest hit together with
    /// the body index (Bullet's `rayTest` with a closest-result callback).
    ///
    /// The broad phase first prunes proxies whose AABB the segment does not
    /// cross, so the (more expensive) exact per-shape test only runs on
    /// candidates — mirroring `btAxisSweep3::rayTest` feeding a
    /// `ClosestRayResultCallback`. The AABB test is conservative, so the
    /// result is identical to testing every body.
    pub fn raycast_closest(&self, ray_from: Vector3, ray_to: Vector3) -> Option<(usize, RayHit)> {
        self.raycast_closest_excluding(ray_from, ray_to, None)
    }

    /// Like [`raycast_closest`](Self::raycast_closest) but never reports
    /// `ignore` (the raycast vehicle's wheel rays, which must not hit their
    /// own chassis). Bullet skips the chassis implicitly because its
    /// interior-start rays miss; we report interior starts (deviation §1), so
    /// the exclusion is explicit here.
    pub fn raycast_closest_excluding(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        ignore: Option<usize>,
    ) -> Option<(usize, RayHit)> {
        let mut best: Option<(usize, RayHit)> = None;
        for p in self.broadphase.raycast_candidates(ray_from, ray_to) {
            // Link colliders join the world queries together with their
            // contact response (phase C); until then only rigid bodies are
            // tested.
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            if Some(i) == ignore {
                continue;
            }
            let body = &self.bodies[i];
            if let Some(hit) = raycast::raycast(
                &body.shape,
                body.position,
                body.orientation,
                ray_from,
                ray_to,
            ) && best.is_none_or(|(_, b)| hit.fraction < b.fraction)
            {
                best = Some((i, hit));
            }
        }
        best
    }

    /// Indices of bodies whose broad-phase AABB overlaps `aabb` (exact AABB
    /// test; the caller then runs the narrow phase on the candidates).
    /// Multibody link colliders overlap too but are not reported until they
    /// join the queries in phase C.
    pub fn aabb_candidates(&self, aabb: Aabb) -> Vec<usize> {
        self.broadphase
            .aabb_candidates(aabb)
            .into_iter()
            .filter_map(|p| match self.proxy_owners[p] {
                ProxyOwner::Body(i) => Some(i),
                ProxyOwner::Collider(_) => None,
            })
            .collect()
    }

    /// Closest convex sweep of `shape` from `from` to `to` against every body
    /// except `ignore`, returning the hit body index and the closest hit.
    ///
    /// Mirrors `btCollisionWorld::convexSweep` feeding a
    /// `ClosestConvexResultCallback` (the character controller's step
    /// sweeps). The broad phase prunes bodies whose AABB does not touch the
    /// swept AABB of the moving shape.
    pub fn convex_sweep_closest(
        &self,
        shape: &bullet_collision::shapes::CollisionShape,
        from: Vector3,
        ori: Quaternion,
        to: Vector3,
        ignore: Option<usize>,
    ) -> Option<(usize, SweepHit)> {
        let swept = shape
            .world_aabb(from, ori)
            .merged(&shape.world_aabb(to, ori));
        let mut best: Option<(usize, SweepHit)> = None;
        for p in self.broadphase.aabb_candidates(swept) {
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            if Some(i) == ignore {
                continue;
            }
            let body = &self.bodies[i];
            if let Some(hit) = narrowphase::convex_cast(
                shape,
                from,
                ori,
                to,
                &body.shape,
                body.position,
                body.orientation,
            ) && best.is_none_or(|(_, b)| hit.fraction < b.fraction)
            {
                best = Some((i, hit));
            }
        }
        best
    }

    /// Contact manifolds between `shape` at `pos`/`ori` and every body except
    /// `ignore` (the character controller's touching contacts and penetration
    /// recovery, mirroring a ghost object's overlapping-pair manifolds).
    pub fn contact_test(
        &self,
        shape: &bullet_collision::shapes::CollisionShape,
        pos: Vector3,
        ori: Quaternion,
        ignore: Option<usize>,
    ) -> Vec<(usize, ContactManifold)> {
        let aabb = shape.world_aabb(pos, ori);
        let mut out = Vec::new();
        for p in self.broadphase.aabb_candidates(aabb) {
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            if Some(i) == ignore {
                continue;
            }
            let body = &self.bodies[i];
            if let Some(m) = narrowphase::collide(
                shape,
                pos,
                ori,
                &body.shape,
                body.position,
                body.orientation,
            ) {
                out.push((i, m));
            }
        }
        out
    }

    /// Set a body's pose directly and refresh its broad-phase AABB. Used by
    /// the character controller to move its kinematic body; the solver still
    /// treats the body as immovable (static bodies are never integrated).
    pub fn set_body_transform(&mut self, index: usize, position: Vector3, orientation: Quaternion) {
        let body = &mut self.bodies[index];
        body.position = position;
        body.orientation = orientation;
        let aabb = body.shape.world_aabb(position, orientation);
        self.broadphase.set_aabb(self.body_proxies[index], aabb);
    }

    /// Add a joint between two bodies and return its index (Bullet's
    /// `addConstraint`; the joint suppresses collisions between its bodies by
    /// default, see [`Constraint::set_disable_collisions`]).
    pub fn add_constraint(&mut self, constraint: Constraint) -> usize {
        if constraint.disable_collisions() {
            let (a, b) = (constraint.body_a(), constraint.body_b());
            self.no_collision_pairs.insert((a.min(b), a.max(b)));
        }
        self.constraints.push(constraint);
        self.constraints.len() - 1
    }

    pub fn constraint(&self, index: usize) -> &Constraint {
        &self.constraints[index]
    }

    pub fn constraint_mut(&mut self, index: usize) -> &mut Constraint {
        &mut self.constraints[index]
    }

    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Remove a joint, dropping its index and the collision suppression it
    /// applied (Bullet's `removeConstraint`; later joints shift down).
    pub fn remove_constraint(&mut self, index: usize) {
        self.constraints.remove(index);
        self.no_collision_pairs.clear();
        for c in &self.constraints {
            if c.disable_collisions() {
                let (a, b) = (c.body_a(), c.body_b());
                self.no_collision_pairs.insert((a.min(b), a.max(b)));
            }
        }
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step_simulation(&mut self, dt: Real) {
        let inv_dt = 1.0 / dt;

        // 1. Integrate velocities from gravity and accumulated forces.
        for body in &mut self.bodies {
            body.integrate_velocities(self.gravity, dt);
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

        // 2. Predicted positions (used only for contact detection).
        let predicted: Vec<Vector3> = self
            .bodies
            .iter()
            .map(|b| b.predicted_position(dt))
            .collect();

        // 3. Update broad-phase AABBs from predicted poses. Sleeping bodies
        // keep their AABB: they do not move, so it stays correct.
        for (i, body) in self.bodies.iter().enumerate() {
            if !body.is_active() {
                continue;
            }
            let aabb = body.shape.world_aabb(predicted[i], body.orientation);
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
            };
            self.broadphase.set_aabb(proxy, aabb);
        }

        // 4. Broad phase: candidate overlapping pairs.
        let pairs = self.broadphase.compute_pairs();

        // 5. Narrow phase: exact contact manifolds -> constraints. Pairs
        // linked by a joint with `disable_collisions` are skipped (Bullet's
        // `btCollisionObject::checkCollideWithOverride`).
        let mut constraints: Vec<ContactConstraint> = Vec::new();
        let mut contact_pairs: Vec<(usize, usize)> = Vec::new();
        for &(pi, pj) in &pairs {
            let (owner_a, owner_b) = (self.proxy_owners[pi], self.proxy_owners[pj]);
            let (ProxyOwner::Body(i), ProxyOwner::Body(j)) = (owner_a, owner_b) else {
                // Multibody link-collider pair (mixed or collider-collider):
                // run the narrow phase at the current FK pose. Phase B is
                // detection-only — the manifold is computed and dropped; the
                // contact rows land in phase C. Self-collision filtering
                // mirrors `checkCollideWithOverride`.
                let Some((mb_a, l_a, ca)) = self.collider_of(owner_a) else {
                    continue;
                };
                let Some((mb_b, l_b, cb)) = self.collider_of(owner_b) else {
                    continue;
                };
                if mb_a == mb_b && !self.multibodies[mb_a].colliders_collide(l_a, l_b) {
                    continue;
                }
                let (pa, qa) = if l_a < 0 {
                    let mb = &self.multibodies[mb_a];
                    (mb.base_pos(), mb.base_world_orientation())
                } else {
                    let mb = &self.multibodies[mb_a];
                    let l = l_a as usize;
                    (mb.link_world_position(l), mb.link_world_orientation(l))
                };
                let (pb, qb) = if l_b < 0 {
                    let mb = &self.multibodies[mb_b];
                    (mb.base_pos(), mb.base_world_orientation())
                } else {
                    let mb = &self.multibodies[mb_b];
                    let l = l_b as usize;
                    (mb.link_world_position(l), mb.link_world_orientation(l))
                };
                let _manifold = narrowphase::collide(&ca.shape, pa, qa, &cb.shape, pb, qb);
                continue;
            };
            if self.no_collision_pairs.contains(&(i.min(j), i.max(j))) {
                continue;
            }
            let a = &self.bodies[i];
            let b = &self.bodies[j];
            if let Some(mut manifold) = narrowphase::collide(
                &a.shape,
                predicted[i],
                a.orientation,
                &b.shape,
                predicted[j],
                b.orientation,
            ) {
                contact_pairs.push((i, j));
                // Inherit the previous step's accumulated impulses for contact
                // points that persist (Bullet's persistent manifolds).
                warm_start_impulses(
                    &mut manifold.points,
                    self.contact_cache.get(&(i, j)).map(Vec::as_slice),
                );
                for cp in &manifold.points {
                    constraints.push(ContactConstraint::prepare(i, j, cp, a, b));
                }
            }
        }

        // 5a. Joint rows: flatten every enabled joint into its solver rows
        // from the current (pre-integration) poses, like Bullet's
        // `convertJoints` after `predictUnconstraintMotion`.
        let mut joint_rows: Vec<ConstraintRow> = Vec::new();
        for c in &mut self.constraints {
            constraint::prepare_rows(c, &self.bodies, inv_dt, BAUMGARTE, &mut joint_rows);
        }

        // 5b. Wake up sleeping bodies that touch an active body (Bullet's
        // island-based activation propagation: activation travels along the
        // contact graph one step at a time). A contact between two sleeping
        // bodies does not wake them — a stable stack stays asleep.
        for &(i, j) in &contact_pairs {
            let a_active = self.bodies[i].is_active();
            let b_active = self.bodies[j].is_active();
            if a_active && !b_active {
                self.bodies[j].activate();
            } else if b_active && !a_active {
                self.bodies[i].activate();
            }
        }
        // A joint links two bodies into one island: a sleeping body woken by
        // motion of its linked partner would otherwise stay frozen while the
        // constraint pulls on it.
        for c in &self.constraints {
            if !c.enabled() {
                continue;
            }
            let (a, b) = (c.body_a(), c.body_b());
            let a_active = self.bodies[a].is_active();
            let b_active = self.bodies[b].is_active();
            if a_active && !b_active {
                self.bodies[b].activate();
            } else if b_active && !a_active {
                self.bodies[a].activate();
            }
        }

        // 6. Solve constraints (correct velocities).
        sequential_impulse::solve(
            &mut self.bodies,
            &mut constraints,
            &mut joint_rows,
            self.iterations,
            inv_dt,
        );

        // 6b. Write back the solved impulses so the next step can warm start
        // from them (the cache is rebuilt every step, dropping stale pairs).
        let mut next_cache: HashMap<(usize, usize), Vec<CachedContact>> = HashMap::new();
        for c in &constraints {
            next_cache
                .entry((c.body_a, c.body_b))
                .or_default()
                .push(CachedContact {
                    position: c.position,
                    normal: c.normal,
                    normal_impulse: c.normal_impulse,
                });
        }
        self.contact_cache = next_cache;

        // 6bm. Multibody internal constraints: row conversion, PGS over the
        // non-contact pool and writeback into the generalized velocities
        // (Bullet's `solveInternalConstraints` running the multibody part of
        // `solveMultiBodyGroup`; contact rows join in a later phase).
        self.mb_solve_internal_constraints(dt);

        // 6c. Deactivation: bodies below the sleeping thresholds for longer
        // than `DEACTIVATION_TIME` seconds go to sleep (Bullet's
        // `updateActivationState` between solving and integrating).
        for body in &mut self.bodies {
            body.update_deactivation(dt);
            if body.wants_deactivation() {
                body.go_to_sleep();
            }
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
                if !body.is_active() || body.is_static() || threshold <= 0.0 {
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
                for j in 0..self.bodies.len() {
                    if i == j {
                        continue;
                    }
                    let other = &self.bodies[j];
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
            if body.is_active() && !body.is_static() && body.push_linear != Vector3::ZERO {
                body.position += body.push_linear * dt;
            }
            if body.is_active() && !body.is_static() && body.push_angular != Vector3::ZERO {
                let w = body.push_angular * SPLIT_TURN_ERP;
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
    }

    /// Refresh link transforms, clear constraint forces, then integrate the
    /// generalized velocities with the articulated-body algorithm (Bullet's
    /// `solveExternalForces`).
    fn mb_solve_external_forces(&mut self, dt: Real) {
        for mb in &mut self.multibodies {
            mb.forward_kinematics();
        }
        for mb in &mut self.multibodies {
            if !mb.is_awake() {
                continue;
            }
            mb.clear_constraint_forces();
            mb.compute_accelerations_aba(dt, &mut self.mb_scratch, false);
        }
    }

    /// Convert all multibody constraints into rows and solve them with
    /// projected Gauss-Seidel (Bullet's `solveInternalConstraints` +
    /// `btMultiBodyConstraintSolver::solveMultiBodyGroup` for the non-contact
    /// pool).
    fn mb_solve_internal_constraints(&mut self, dt: Real) {
        // Wake bodies linked to an awake body by a constraint (the island
        // merge in `calculateSimulationIslands`).
        let mut wake = vec![false; self.multibodies.len()];
        for c in &self.mb_constraints {
            let a_awake = self.multibodies[c.body_a()].is_awake();
            let b_awake = self.multibodies[c.body_b()].is_awake();
            if a_awake || b_awake {
                wake[c.body_a()] = true;
                wake[c.body_b()] = true;
            }
        }
        for (i, &w) in wake.iter().enumerate() {
            if w {
                self.multibodies[i].wake_up();
            }
        }

        // Reset the solver scratch and companion ids (Bullet's
        // `solveGroupCacheFriendlySetup`).
        self.mb_data.reset();
        self.mb_rows.clear();
        for mb in &mut self.multibodies {
            mb.companion_id = -1;
        }

        // Convert the constraints into rows.
        {
            let PhysicsWorld {
                multibodies,
                mb_constraints: constraints,
                mb_data: data,
                mb_scratch: scratch,
                mb_rows: rows,
                mb_erp: erp,
                ..
            } = self;
            for (i, c) in constraints.iter_mut().enumerate() {
                c.create_constraint_rows(i, rows, data, scratch, multibodies, dt, *erp);
            }
        }

        // PGS iterations over the non-contact rows, alternating direction by
        // iteration parity (Bullet's `solveSingleIteration`).
        let n = self.mb_rows.len();
        let outer = self.iterations as usize;
        let inner = self.num_non_contact_inner_iterations as usize;
        for iteration in 0..outer {
            for _ in 0..inner {
                if iteration % 2 == 0 {
                    for j in 0..n {
                        resolve_row(&mut self.mb_rows[j], &mut self.mb_data);
                    }
                } else {
                    for j in (0..n).rev() {
                        resolve_row(&mut self.mb_rows[j], &mut self.mb_data);
                    }
                }
            }
        }

        // Write the accumulated impulses back into the constraints (Bullet's
        // `writeBackSolverBodyToMultiBody`).
        for row in &self.mb_rows {
            self.mb_constraints[row.org_constraint]
                .set_applied_impulse(row.org_dof_index, row.applied_impulse);
        }
        for row in &self.mb_rows {
            let imp = row.applied_impulse;
            let unit_a = self.mb_data.delta_velocities_unit_impulse
                [row.jac_a_index..row.jac_a_index + row.ndof_a]
                .to_vec();
            let unit_b = self.mb_data.delta_velocities_unit_impulse
                [row.jac_b_index..row.jac_b_index + row.ndof_b]
                .to_vec();
            let mb = &mut self.multibodies[row.multi_body_a];
            mb.apply_delta_vee(&unit_a, imp);
            let mb = &mut self.multibodies[row.multi_body_b];
            mb.apply_delta_vee(&unit_b, imp);
        }
        // Flush any split-impulse deltas (no-op unless split impulses are
        // used; kept for fidelity with `solveInternalConstraints`).
        for mb in &mut self.multibodies {
            mb.process_delta_vee_2();
        }
    }

    /// Integrate the positions of the awake multibodies and freeze the
    /// sleeping ones (Bullet's `integrateMultiBodyTransforms`).
    fn mb_integrate_transforms(&mut self, dt: Real) {
        for mb in &mut self.multibodies {
            if !mb.is_awake() {
                mb.clear_velocities();
                continue;
            }
            mb.add_split_v();
            mb.step_positions(dt);
            mb.subtract_split_v();
        }
    }
}

/// Inherit the previous step's accumulated normal impulse for each fresh
/// contact point. A new point within [`CONTACT_BREAKING_THRESHOLD`] of a
/// cached one is the same physical contact (Bullet's `getCacheEntry` +
/// `replaceContactPoint` persistence) and keeps its impulse, scaled by
/// [`WARM_STARTING_FACTOR`] so the solver can correct small drift (Bullet's
/// `m_warmstartingFactor`).
///
/// Two guards protect against the point set changing between steps (which
/// Bullet's persistent manifolds smooth out and we rebuild per step):
/// - the contact normal must stay (nearly) the same: our mesh narrow phase
///   generates one normal per triangle, so a cached impulse applied along a
///   newly rotated normal would kick the body sideways;
/// - friction impulses are **not** inherited: the tangent basis rotates with
///   the point set, and stale friction kicks make a resting body on a mesh
///   oscillate instead of settling (observed with `falling_mesh`). Friction
///   rebuilds from zero inside the solver iterations, exactly as on the first
///   contact frame.
fn warm_start_impulses(points: &mut [bullet_collision::manifold::ContactPoint], cache: Option<&[CachedContact]>) {
    const NORMAL_EPS: Real = 0.9;
    let breaking_sq = CONTACT_BREAKING_THRESHOLD * CONTACT_BREAKING_THRESHOLD;
    let Some(cache) = cache else {
        return;
    };
    let mut used = vec![false; cache.len()];
    for p in points {
        let mut best: Option<usize> = None;
        let mut best_d2 = Real::INFINITY;
        for (i, c) in cache.iter().enumerate() {
            if used[i] {
                continue;
            }
            let d2 = (p.position - c.position).length_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best = Some(i);
            }
        }
        let Some(i) = best else {
            continue;
        };
        if best_d2 <= breaking_sq && p.normal.dot(cache[i].normal) >= NORMAL_EPS {
            p.normal_impulse = cache[i].normal_impulse * WARM_STARTING_FACTOR;
            used[i] = true;
        }
    }
}

/// Solve one multibody constraint row (Bullet's
/// `resolveSingleConstraintRowGeneric`): compute the impulse needed to meet
/// `rhs` given the currently accumulated velocity deltas of both bodies,
/// clamp it to the row window, and add its effect to the accumulated deltas.
fn resolve_row(row: &mut SolverRow, data: &mut SolverData) {
    let ndof_a = row.ndof_a;
    let ndof_b = row.ndof_b;
    let da_index = row.delta_vel_a_index as usize;
    let db_index = row.delta_vel_b_index as usize;

    let mut delta_impulse = row.rhs - row.applied_impulse * row.cfm;

    let mut delta_vel_a_dot_n = 0.0;
    {
        let jac = &data.jacobians[row.jac_a_index..row.jac_a_index + ndof_a];
        let dv = &data.delta_velocities[da_index..da_index + ndof_a];
        for i in 0..ndof_a {
            delta_vel_a_dot_n += jac[i] * dv[i];
        }
    }
    let mut delta_vel_b_dot_n = 0.0;
    {
        let jac = &data.jacobians[row.jac_b_index..row.jac_b_index + ndof_b];
        let dv = &data.delta_velocities[db_index..db_index + ndof_b];
        for i in 0..ndof_b {
            delta_vel_b_dot_n += jac[i] * dv[i];
        }
    }
    delta_impulse -= (delta_vel_a_dot_n + delta_vel_b_dot_n) * row.jac_diag_ab_inv;
    let sum = row.applied_impulse + delta_impulse;
    if sum < row.lower_limit {
        delta_impulse = row.lower_limit - row.applied_impulse;
        row.applied_impulse = row.lower_limit;
    } else if sum > row.upper_limit {
        delta_impulse = row.upper_limit - row.applied_impulse;
        row.applied_impulse = row.upper_limit;
    } else {
        row.applied_impulse = sum;
    }

    // Accumulate the effect of this impulse into both bodies' delta
    // velocities (Bullet's `applyDeltaVee`).
    {
        let unit = &data.delta_velocities_unit_impulse[row.jac_a_index..row.jac_a_index + ndof_a];
        let dv = &mut data.delta_velocities[da_index..da_index + ndof_a];
        for i in 0..ndof_a {
            dv[i] += unit[i] * delta_impulse;
        }
    }
    {
        let unit = &data.delta_velocities_unit_impulse[row.jac_b_index..row.jac_b_index + ndof_b];
        let dv = &mut data.delta_velocities[db_index..db_index + ndof_b];
        for i in 0..ndof_b {
            dv[i] += unit[i] * delta_impulse;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::body::ActivationState;

    #[test]
    fn raycast_closest_returns_the_nearest_body() {
        let mut world = PhysicsWorld::new();
        // Near sphere at x = 1, far sphere at x = 3.
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(3.0, 0.0, 0.0),
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(1.0, 0.0, 0.0),
        ));
        let (index, hit) = world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .unwrap();
        assert_eq!(index, 1);
        assert_relative_eq!(hit.fraction, 0.1, epsilon = 1e-4);
        assert_relative_eq!(hit.normal.x, -1.0, epsilon = 1e-4);
    }

    #[test]
    fn raycast_closest_misses_empty_world() {
        let world = PhysicsWorld::new();
        assert!(world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .is_none());
    }

    /// The broad-phase AABB test is conservative: a ray crossing a proxy's
    /// AABB corner must not produce a hit when the exact shape test misses.
    #[test]
    fn raycast_closest_prunes_aabb_only_hits() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::ZERO,
        ));
        // Starts on the AABB corner (0.5, 0.5, 0.5), which is outside the
        // radius-0.5 sphere: the AABB filter keeps it, the sphere test drops it.
        let hit = world.raycast_closest(
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(0.6, 0.6, 0.6),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn convex_sweep_closest_hits_the_nearer_body() {
        let mut world = PhysicsWorld::new();
        // Two boxes on the path: a near one (face at x = 3) and a far one
        // (face at x = 4). The probe starts at x = 8 and travels -X, so the
        // far box (closer to the start) is the closest hit.
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 1.0, 1.0)),
            Vector3::new(2.0, 0.0, 0.0),
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 1.0, 1.0)),
            Vector3::new(5.0, 0.0, 0.0),
        ));
        // Sweep a small box from x = 8 to x = -2 (length 10). The far box
        // spans x in [4, 6]; its +X face at x = 6 is hit by the probe's -X
        // face at x - 0.5: entry when x = 6.5, fraction (8 - 6.5) / 10 = 0.15.
        let probe = bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5));
        let (index, hit) = world
            .convex_sweep_closest(
                &probe,
                Vector3::new(8.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(-2.0, 0.0, 0.0),
                None,
            )
            .unwrap();
        assert_eq!(index, 1);
        assert_relative_eq!(hit.fraction, (8.0 - 6.5) / 10.0, epsilon = 1e-4);
        assert_relative_eq!(hit.normal.x, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn convex_sweep_closest_can_ignore_a_body() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(1.0)),
            Vector3::ZERO,
        ));
        let probe = bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5));
        // Sweeping into the box: hit when not ignored...
        assert!(world
            .convex_sweep_closest(
                &probe,
                Vector3::new(5.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(0.0, 0.0, 0.0),
                None,
            )
            .is_some());
        // ...and a miss when the only body is ignored.
        assert!(world
            .convex_sweep_closest(
                &probe,
                Vector3::new(5.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(0.0, 0.0, 0.0),
                Some(0),
            )
            .is_none());
    }

    #[test]
    fn contact_test_reports_overlapping_bodies() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(1.0)),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(1.0)),
            Vector3::new(5.0, 0.0, 0.0),
        ));
        let probe = bullet_collision::shapes::CollisionShape::sphere(0.5);
        // The probe overlaps the first box (probe x-extent [0.4, 1.4] vs box
        // x-extent [-1, 1]) but not the second.
        let hits = world.contact_test(&probe, Vector3::new(0.9, 0.0, 0.0), Quaternion::IDENTITY, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
        assert!(hits[0].1.points.iter().any(|p| p.penetration > 0.0));
        // The probe itself can be ignored.
        assert!(world
            .contact_test(&probe, Vector3::new(0.9, 0.0, 0.0), Quaternion::IDENTITY, Some(0))
            .is_empty());
    }

    /// A fresh contact point near a cached one keeps its impulse (scaled by
    /// the warm-starting factor); a distant point is a new contact from zero.
    #[test]
    fn warm_start_inherits_cached_impulses() {
        let cache = [CachedContact {
            position: Vector3::ZERO,
            normal: Vector3::Y,
            normal_impulse: 10.0,
        }];
        let mut near = [bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.01, 0.0, 0.0),
            Vector3::Y,
            0.1,
        )];
        warm_start_impulses(&mut near, Some(&cache));
        assert_relative_eq!(near[0].normal_impulse, 8.5, epsilon = 1e-4);
        // Friction is not inherited (rebuilds from zero each step).
        assert_eq!(near[0].tangent_impulse, [0.0, 0.0]);

        let mut far = [bullet_collision::manifold::ContactPoint::new(
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::Y,
            0.1,
        )];
        warm_start_impulses(&mut far, Some(&cache));
        assert_eq!(far[0].normal_impulse, 0.0);
        assert_eq!(far[0].tangent_impulse, [0.0, 0.0]);

        // A nearby point whose normal rotated too much (e.g. a box crossing a
        // triangle-mesh ridge) must not inherit the cached impulse.
        let mut rotated = [bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.01, 0.0, 0.0),
            Vector3::X,
            0.1,
        )];
        warm_start_impulses(&mut rotated, Some(&cache));
        assert_eq!(rotated[0].normal_impulse, 0.0);
    }

    /// A resting body keeps a supporting contact impulse in the cache across
    /// steps (the warm-start input for the next step).
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
        let cached: Vec<&CachedContact> = world.contact_cache.values().flatten().collect();
        assert!(!cached.is_empty());
        assert!(cached.iter().any(|c| c.normal_impulse > 0.0));
        let sphere = world.body(1);
        assert_relative_eq!(sphere.position.y, 0.5, epsilon = 0.05);
        assert!(sphere.linear_velocity.length() < 0.1);
    }

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
        let sphere = world.body(1);
        assert_eq!(sphere.activation_state(), ActivationState::IslandSleeping);
        assert_eq!(sphere.linear_velocity, Vector3::ZERO);
        let y_at_sleep = sphere.position.y;
        // Further steps must not move the sleeping body.
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert_relative_eq!(world.body(1).position.y, y_at_sleep, epsilon = 1e-6);
    }

    /// A sleeping body in contact with an active body is woken up (Bullet's
    /// activation propagation along the contact graph).
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
        assert_eq!(world.body(1).activation_state(), ActivationState::IslandSleeping);
        // ...then a fast ball hits it: the sleeping sphere must wake up.
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.25),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, -5.0, 0.0)));
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert_eq!(
            world.body(1).activation_state(),
            ActivationState::Active,
            "sleeping body was not woken by the impact"
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
        world.body_mut(1).apply_force(Vector3::new(0.0, 5.0, 0.0));
        assert_eq!(world.body(1).activation_state(), ActivationState::Active);
    }

    /// Static bodies start asleep, like Bullet's static `btRigidBody`.
    #[test]
    fn static_body_starts_sleeping() {
        let body = RigidBody::new_static(bullet_collision::shapes::CollisionShape::sphere(0.5), Vector3::ZERO);
        assert_eq!(body.activation_state(), ActivationState::IslandSleeping);
        assert!(!body.is_active());
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
        assert_relative_eq!(world.body(1).position.y, 0.9, epsilon = 0.05);
        // The control ball crossed the whole wall.
        assert!(world.body(2).position.y < -0.5);
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
        assert_relative_eq!(world.body(1).position.y, 1.0 - 5.0 / 60.0, epsilon = 1e-5);
    }

    /// A fixed-base pendulum with a sphere collider on the link: the collider
    /// registers as a broad-phase proxy with the FK pose AABB and refreshes
    /// as the body moves (phase B of the multibody collision design).
    #[test]
    fn link_collider_registers_and_updates_aabb() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(0.0, -1.0, 0.0),
        );
        mb.finalize_multidof();
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        world.step_simulation(1.0 / 60.0);
        // The link COM hangs 1 m below the pivot; the sphere AABB spans
        // y ∈ [−1.5, −0.5].
        let entry = world
            .collider_entries
            .iter()
            .find(|e| e.mb == 0 && e.link == 0)
            .expect("collider registered");
        let proxy = entry.proxy.expect("proxy assigned");
        let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_relative_eq!(aabb.min.y, -1.5, epsilon = 1e-4);
        assert_relative_eq!(aabb.max.y, -0.5, epsilon = 1e-4);
        // Moving the base moves the collider with the next FK pass.
        world.multibody_mut(0).base_pos = Vector3::new(0.0, 5.0, 0.0);
        world.step_simulation(1.0 / 60.0);
        let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_relative_eq!(aabb.min.y, 3.5, epsilon = 1e-4);
    }

    /// A sleeping multibody keeps its collider AABB (like sleeping rigid
    /// bodies): the refresh is skipped until the body wakes.
    #[test]
    fn sleeping_multibody_keeps_collider_aabb() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, true);
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(0.0, -1.0, 0.0),
        );
        mb.finalize_multidof();
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        for _ in 0..150 {
            // 2.5 s > the 2 s sleep timeout.
            world.step_simulation(1.0 / 60.0);
        }
        assert!(!world.multibody(0).is_awake());
        let proxy = world.collider_entries[0].proxy.unwrap();
        let before = world.broadphase.proxy_aabb(proxy).unwrap();
        // Teleport the base: the sleeping body skips the AABB refresh.
        world.multibody_mut(0).base_pos = Vector3::new(3.0, 0.0, 0.0);
        world.step_simulation(1.0 / 60.0);
        let after = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_eq!(before, after);
    }

    /// A link collider overlapping a static rigid body produces a translated
    /// mixed broad-phase pair.
    #[test]
    fn link_collider_pairs_with_rigid_body() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(0.0, -1.0, 0.0),
        );
        mb.finalize_multidof();
        // Sink the link so its sphere (r = 0.5, centered 0.6 below the
        // plane) overlaps the ground.
        mb.base_pos = Vector3::new(0.0, 0.4, 0.0);
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        world.step_simulation(1.0 / 60.0);
        let mixed = world
            .broadphase
            .compute_pairs()
            .into_iter()
            .filter(|(a, b)| {
                matches!(world.proxy_owners[*a], ProxyOwner::Body(_))
                    ^ matches!(world.proxy_owners[*b], ProxyOwner::Body(_))
            })
            .count();
        assert_eq!(mixed, 1, "one rigid-collider pair expected");
    }

    /// Self-collision filtering between two overlapping link colliders of
    /// one chain: the broad phase reports the AABB pair (both colliders are
    /// registered); the `checkCollideWithOverride` port
    /// ([`MultiBody::colliders_collide`]) decides whether the narrow phase
    /// runs on it — adjacent links collide by default, `has_self_collision =
    /// false` and the per-link parent-collision flags opt out.
    #[test]
    fn self_collision_filtering() {
        // Two overlapping link colliders on one chain register and produce
        // one collider-collider broad-phase pair.
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(2, 1.0, Vector3::ONE, true, false);
        // Both links sit at the base origin with overlapping spheres.
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::ZERO,
        );
        mb.setup_revolute(
            1,
            1.0,
            Vector3::ONE,
            Some(0),
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::ZERO,
        );
        mb.finalize_multidof();
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        mb.set_link_collider(
            1,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        world.step_simulation(1.0 / 60.0);
        let collider_pairs = world
            .broadphase
            .compute_pairs()
            .into_iter()
            .filter(|(a, b)| {
                matches!(world.proxy_owners[*a], ProxyOwner::Collider(_))
                    && matches!(world.proxy_owners[*b], ProxyOwner::Collider(_))
            })
            .count();
        assert_eq!(collider_pairs, 1);

        // The filter itself (`checkCollideWithOverride` port) on a 3-link
        // chain: base - 0 - 1 - 2. `-1` is the base.
        let mut mb = crate::multibody::MultiBody::new(3, 1.0, Vector3::ONE, true, false);
        for i in 0..3 {
            mb.setup_revolute(
                i,
                1.0,
                Vector3::ONE,
                if i == 0 { None } else { Some(i - 1) },
                Quaternion::IDENTITY,
                Vector3::Z,
                Vector3::ZERO,
                Vector3::ZERO,
            );
        }
        mb.finalize_multidof();
        // Default: everything collides (Bullet's default).
        assert!(mb.colliders_collide(0, 1));
        assert!(mb.colliders_collide(-1, 0));
        assert!(mb.colliders_collide(0, 2)); // non-adjacent
        // Whole-body opt-out.
        mb.set_has_self_collision(false);
        assert!(!mb.colliders_collide(0, 1));
        assert!(!mb.colliders_collide(-1, 0));
        mb.set_has_self_collision(true);
        // Direct-parent opt-out on link 1: only (1, 0) is excluded.
        mb.set_link_parent_collision_disabled(1, false);
        assert!(!mb.colliders_collide(1, 0));
        assert!(mb.colliders_collide(1, 2));
        assert!(mb.colliders_collide(1, -1));
        // All-ancestors opt-out on link 2: parent and grandparent excluded,
        // others not (fresh body to keep the flags independent).
        let mut mb = crate::multibody::MultiBody::new(3, 1.0, Vector3::ONE, true, false);
        for i in 0..3 {
            mb.setup_revolute(
                i,
                1.0,
                Vector3::ONE,
                if i == 0 { None } else { Some(i - 1) },
                Quaternion::IDENTITY,
                Vector3::Z,
                Vector3::ZERO,
                Vector3::ZERO,
            );
        }
        mb.finalize_multidof();
        mb.set_link_parent_collision_disabled(2, true);
        assert!(!mb.colliders_collide(2, 1));
        assert!(!mb.colliders_collide(2, 0));
        // The base is part of the ancestor walk, like Bullet's loop reaching
        // m_parent == -1.
        assert!(!mb.colliders_collide(2, -1));
        // The exclusion is symmetric (Bullet checks both sides), and pairs
        // without a flagged member are unaffected.
        assert!(!mb.colliders_collide(1, 2));
        assert!(mb.colliders_collide(0, 1));
    }
}
