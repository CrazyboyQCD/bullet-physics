use core::time::Duration;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::timing::{default_step_clock, StepClock, StepStage, StepTimings};

use bullet_collision::broadphase::{BroadPhase, BroadPhaseBackend};
use bullet_collision::manifold::{ContactManifold, PersistentManifold, CONTACT_BREAKING_THRESHOLD};
use bullet_collision::narrowphase::{self, epa::EpaScratch, SweepHit};
use bullet_collision::raycast::{self, RayHit};
use bullet_collision::shapes::CompoundChild;
use crate::body::{CollisionFlags, RayFilter, RigidBody};
use crate::constraint::{self, Constraint};
use crate::handles::{BodyHandle, MultiBodyHandle, SoftBodyHandle};
use crate::island::IslandScratch;
use crate::multibody::body::{MultiBody, MultiBodyScratch};
use crate::multibody::contact::{self, ContactSide, MbContactRow, PendingContact};
use crate::multibody::constraint::{MultiBodyConstraint, SolverData, SolverRow};
use crate::softbody::cluster::solve_clusters;
use crate::softbody::collide::collide_vf_ss;
use crate::softbody::solver::MbSolveContext;
use crate::softbody::config::SoftCollision;
use crate::softbody::{AngularJointSpecs, LinearJointSpecs, Side, SoftBody};
use bullet_math::{Aabb, Matrix3, Quaternion, Real, Vector3};
use crate::solver::sequential_impulse::{self, BAUMGARTE, ContactConstraint, ConstraintRow, SLOP, SPLIT_TURN_ERP};

/// One entry of the persistent-manifold cache: the owning broad-phase pair
/// plus the manifold itself. The cache is an adjacency structure — one
/// bucket per body (indexed by `min(a, b)`, each entry naming the other
/// side) — because contact pairs are strongly local: a body only ever
/// holds cached manifolds against its few neighbours, so bucket-linear
/// lookups stay at a handful of comparisons while insertions and removals
/// never shift the large manifold entries around (a sorted array's
/// memmove-per-churn measured +21% on the 1000-body stack).
#[derive(Default)]
struct ManifoldCacheEntry {
    /// The other body of the pair.
    other: usize,
    manifold: PersistentManifold,
}

/// Every broad-phase AABB is padded by the global contact-breaking
/// threshold (Bullet's `btCollisionWorld::updateSingleAabb` — "need to
/// increase the aabb for contact thresholds"), so pairs live in the
/// broadphase while their margin-band contacts can still form.
const BROADPHASE_AABB_MARGIN: Real = CONTACT_BREAKING_THRESHOLD;

/// Scale applied to a cached impulse before the warm-start pass (Bullet's
/// `m_warmstartingFactor`): old impulses are faded slightly so the solver can
/// correct drift instead of blindly trusting the previous step.
const WARM_STARTING_FACTOR: Real = 0.85;
/// Linear/angular speed above which a pair re-runs detection instead of
/// refreshing its cached manifold. The refresh re-projects cached anchors,
/// normals and penetrations — while the bodies move, those go stale at a rate
/// proportional to the speed, and a stale contact geometry feeds the Baumgarte
/// bias a wrong depth every step (an energy pump that visibly destabilizes
/// settling stacks). The check uses the *end-of-last-step* speeds (before the
/// current gravity kick): a resting supported body ends each step near zero
/// velocity even though gravity re-accelerates it every step, so supported
/// contacts still refresh while genuinely moving ones re-detect.
const REDETECT_SPEED_SQUARED: Real = 0.08 * 0.08;

/// A multibody collider registered in the broad phase: `(multibody, link)`
/// with `link = -1` for the base, plus its proxy id once assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColliderEntry {
    mb: usize,
    link: i32,
    proxy: Option<usize>,
}

/// What a broad-phase proxy slot belongs to (rigid bodies, multibody link
/// colliders and soft bodies share one proxy index space).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProxyOwner {
    Body(usize),
    Collider(usize),
    /// A whole soft body (one proxy over its node hull, Bullet's soft-body
    /// collision object with the `m_bounds` AABB).
    Soft(usize),
    /// A destroyed proxy slot (see [`PhysicsWorld::remove_body`]). The
    /// broad phases recycle freed slots, so the entry is cleared until
    /// `create_proxy` hands the slot out again.
    Free,
}

/// A contact-pair lifecycle event, emitted after each step for every pair
/// with at least one body opted in via [`RigidBody::collision_events`]
/// (Rapier's `CollisionEvent` model; Bullet has no equivalent — its
/// `gContactAddedCallback` is a process-global function pointer fired per
/// contact point). `a < b` always.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionEvent {
    /// The pair began touching this step. `sensor` is set when either side
    /// has [`CollisionFlags::NO_CONTACT_RESPONSE`] — a trigger volume: no
    /// contact response is ever solved for it, only events and queries.
    Started { a: BodyHandle, b: BodyHandle, sensor: bool },
    /// The pair stopped touching this step (separated past the breaking
    /// threshold, left the broad phase, got filtered by group/mask, or a
    /// body was removed).
    Stopped { a: BodyHandle, b: BodyHandle, sensor: bool },
}

/// Bullet's broad-phase pair filter (`btBroadphaseProxy::NeedsCollision`):
/// the pair collides when each object's group intersects the other's mask.
#[inline]
fn filters_collide(group_a: u16, mask_a: u16, group_b: u16, mask_b: u16) -> bool {
    group_a & mask_b != 0 && group_b & mask_a != 0
}

/// Custom overlap-filter predicate for rigid pairs (Bullet's
/// `btOverlapFilterCallback::needsCollision`): receives both body handles
/// and references; pairs it rejects are skipped by the narrow phase and the
/// CCD sweep, on top of the group & mask test.
pub type PairFilter = dyn Fn(BodyHandle, &RigidBody, BodyHandle, &RigidBody) -> bool;

/// A discrete rigid-body simulation world, mirroring `btDiscreteDynamicsWorld`
/// extended with the Featherstone multibody list of `btMultiBodyDynamicsWorld`
/// (phase A of the collision-integration design: one fused world, stepping
/// both subsystems; link-collider collision integration follows in later
/// phases — see `docs/multibody-collision.md`).
pub struct PhysicsWorld {
    bodies: Vec<RigidBody>,
    broadphase: BroadPhaseBackend,
    gravity: Vector3,
    iterations: u32,
    /// Contact manifolds cached across steps, keyed by body pair in
    /// broad-phase proxy order (Bullet's persistent manifolds on the pair
    /// cache): the points are refreshed against the current poses and only
    /// pairs that drifted past the redetect thresholds re-run the narrow
    /// phase. Entries whose broad-phase pair disappeared are dropped at the
    /// end of the step (Bullet destroys the manifold with the pair).
    /// Adjacency layout: `manifold_cache[min(i, j)]` holds the entries of
    /// every pair `(i, j)` (see [`ManifoldCacheEntry`]). The hash-based
    /// layout this replaces matched Bullet's hand-rolled integer-pair hash;
    /// the buckets serve the same lookups without the hash + a dependency.
    manifold_cache: Vec<Vec<ManifoldCacheEntry>>,
    /// Solver contact rows of the current step, pooled across steps (the
    /// capacity survives `clear`, so the steady state allocates nothing).
    contact_rows: Vec<ContactConstraint>,
    /// Joint solver rows of the current step, pooled across steps.
    joint_rows: Vec<ConstraintRow>,
    /// Predicted positions of the current step (detection poses), pooled.
    predicted_poses: Vec<Vector3>,
    /// End-of-last-step linear speed squared per body (sampled before the
    /// gravity kick of the current step): the refresh-vs-redetect gate for
    /// the persistent manifolds — see [`REDETECT_SPEED_SQUARED`].
    settled_speeds: Vec<Real>,
    /// Monotonic step counter stamping the manifold-cache lifecycle prune.
    step_counter: u32,
    /// Scratch buffer for [`PersistentManifold::refresh`] output, pooled
    /// across steps.
    refresh_out: Vec<bullet_collision::manifold::ContactPoint>,
    /// Whether the previous step ran with any awake body: the prune must
    /// not fire on the first step after a fully-sleeping stretch (those
    /// entries carry stale stamps until this step's pair pass re-stamps
    /// them).
    prev_simulate: bool,
    /// Custom overlap-filter predicate for rigid pairs (Bullet's
    /// `btOverlapFilterCallback`); see [`PhysicsWorld::set_pair_filter`].
    pair_filter: Option<Box<PairFilter>>,
    /// Joints (Bullet's `btTypedConstraint` list added with `addConstraint`).
    constraints: Vec<Constraint>,
    /// Body pairs whose collision response is suppressed by a joint
    /// (`disableCollisionsBetweenLinkedBodies`, Bullet's
    /// `checkCollideWithOverride`), sorted `(min, max)`, deduplicated.
    no_collision_pairs: Vec<(usize, usize)>,
    /// Opt-in pairs that touched last step, sorted (`[min, max]` body
    /// indices): the baseline of the Started/Stopped diff.
    prev_touching: Vec<[u32; 2]>,
    /// Opt-in pairs that touched this step, sorted after the pair pass.
    touching: Vec<[u32; 2]>,
    /// Collision events of the last step, cleared at the start of each
    /// step (Rapier's post-step drain buffer).
    contact_events: Vec<CollisionEvent>,
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
    /// Mixed/collider contact manifolds collected by the narrow phase this
    /// step (converted into solver rows at solve time, when the multibodies
    /// can be borrowed mutably).
    mb_pending_contacts: Vec<PendingContact>,
    /// Flattened multibody contact rows for the current step.
    mb_contact_rows: Vec<MbContactRow>,
    /// Reused per-sweep buffer for the contact rows' normal-impulse
    /// snapshot (kept across steps so the solve sweeps do not allocate).
    mb_row_scratch: Vec<Real>,
    /// Pooled island-pass buffers (union-find arrays + awake-flag vectors,
    /// grown to the node count on demand; see `island::IslandScratch`).
    island_scratch: IslandScratch,
    /// Position-based soft bodies (Bullet's `btSoftBody` list of
    /// `btSoftRigidDynamicsWorld`); each body owns one broad-phase proxy
    /// over its node-hull bounds (phase B: SDF_RS contacts and anchors;
    /// soft-soft and clusters follow — see `docs/softbody.md` §4/§10).
    soft_bodies: Vec<SoftBody>,
    /// Broad-phase proxy of each soft body.
    soft_proxies: Vec<usize>,
    /// Recycled leaf scratch for the soft-body SDF_RS queries.
    soft_scratch: Vec<usize>,
    /// Recycled (node, face) leaf-pair scratch for the soft-soft VF_SS
    /// queries.
    soft_pair_scratch: Vec<(usize, usize)>,
    /// Recycled jacobian buffer for the soft-vs-multibody link contacts
    /// (phase F; one contact at a time).
    soft_mb_jac: Vec<Real>,
    /// Recycled unit-impulse-response buffer for the soft-vs-multibody
    /// link contacts (phase F).
    soft_mb_delta: Vec<Real>,
    /// Cumulative per-stage step timings; zero until enabled (see
    /// [`PhysicsWorld::enable_step_timing`]).
    pub(crate) step_timings: StepTimings,
    /// Clock source feeding the step timings. Defaults to the std
    /// monotonic clock (zero without the `std` feature); replaced via
    /// [`PhysicsWorld::set_step_clock`].
    pub(crate) step_clock_fn: fn() -> Duration,
    /// Shared EPA scratch buffers, reused across every pair of a
    /// detection phase so steady-state narrow-phase passes allocate
    /// nothing (the thread-local pool this replaced lived for the whole
    /// process; one world keeps its own).
    epa_scratch: EpaScratch,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self::with_broadphase(BroadPhaseBackend::default())
    }

    /// Build a world over a specific broad-phase backend (Bullet's
    /// `btBroadphaseInterface` selection; `new()` uses `AxisSweep3`).
    pub fn with_broadphase(broadphase: BroadPhaseBackend) -> Self {
        Self {
            bodies: Vec::new(),
            broadphase,
            gravity: Vector3::new(0.0, -9.81, 0.0),
            iterations: 10,
            manifold_cache: Vec::new(),
            contact_rows: Vec::new(),
            joint_rows: Vec::new(),
            predicted_poses: Vec::new(),
            settled_speeds: Vec::new(),
            step_counter: 0,
            refresh_out: Vec::new(),
            prev_simulate: false,
            pair_filter: None,
            constraints: Vec::new(),
            no_collision_pairs: Vec::new(),
            prev_touching: Vec::new(),
            touching: Vec::new(),
            contact_events: Vec::new(),
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
            mb_pending_contacts: Vec::new(),
            mb_contact_rows: Vec::new(),
            mb_row_scratch: Vec::new(),
            island_scratch: IslandScratch::default(),
            soft_bodies: Vec::new(),
            soft_proxies: Vec::new(),
            soft_scratch: Vec::new(),
            soft_pair_scratch: Vec::new(),
            soft_mb_jac: Vec::new(),
            soft_mb_delta: Vec::new(),
            step_timings: StepTimings::default(),
            step_clock_fn: default_step_clock,
            epa_scratch: EpaScratch::default(),
        }
    }

    pub fn set_gravity(&mut self, gravity: Vector3) {
        self.gravity = gravity;
        // Bullet's `setGravity`: propagate to every active body that has not
        // opted out. Sleeping bodies keep the stale value until woken — the
        // upstream quirk, copied verbatim.
        for body in &mut self.bodies {
            if body.is_active()
                && !body
                    .collision_flags
                    .contains(CollisionFlags::DISABLE_WORLD_GRAVITY)
            {
                body.set_gravity(gravity);
            }
        }
    }

    /// Install a custom overlap-filter predicate for rigid pairs (Bullet's
    /// `btOverlapFilterCallback::needsCollision`): the closure receives both
    /// body indices and references, and pairs it rejects are skipped by the
    /// narrow phase and the CCD sweep, on top of the group & mask test.
    /// `None` (the default) restores the standard semantics.
    pub fn set_pair_filter(&mut self, filter: Option<Box<PairFilter>>) {
        self.pair_filter = filter;
    }

    pub fn gravity(&self) -> Vector3 {
        self.gravity
    }

    /// Number of solver (PGS) iterations per step.
    pub fn set_iterations(&mut self, iterations: u32) {
        self.iterations = iterations;
    }

    /// Add a body and return its handle.
    ///
    /// Broad-phase proxy ids are shared with multibody link colliders, so
    /// they are tracked in [`PhysicsWorld::body_proxies`] (visible to tests).
    /// Rigid-body handles are stable: removing a body never renumbers the
    /// others (see [`PhysicsWorld::remove_body`]).
    pub fn add_body(&mut self, mut body: RigidBody) -> BodyHandle {
        // Bullet's `addRigidBody`: a dynamic body inherits the world gravity   
        // unless it detached itself (`BT_DISABLE_WORLD_GRAVITY`); kinematic
        // bodies are excluded like static ones (`isStaticOrKinematicObject`).
        if !body.is_static_or_kinematic()
            && !body
                .collision_flags
                .contains(CollisionFlags::DISABLE_WORLD_GRAVITY)
        {
            body.set_gravity(self.gravity);
        }
        let aabb = body
            .shape
            .world_aabb(body.position, body.orientation)
            .expanded(BROADPHASE_AABB_MARGIN);
        let proxy = self.broadphase.create_proxy(aabb);
        let index = self.bodies.len();
        self.bodies.push(body);
        // Broad-phase slots are recycled after `remove_body`, so this covers
        // both a fresh push and a reused slot.
        if proxy == self.proxy_owners.len() {
            self.proxy_owners.push(ProxyOwner::Body(index));
        } else {
            debug_assert_eq!(self.proxy_owners[proxy], ProxyOwner::Free);
            self.proxy_owners[proxy] = ProxyOwner::Body(index);
        }
        self.body_proxies.push(proxy);
        BodyHandle::from_index(index)
    }

    /// The body at `handle`. Panics if the body was removed
    /// ([`PhysicsWorld::remove_body`]) — the slot itself stays readable
    /// through [`PhysicsWorld::bodies`] for bookkeeping, but a removed body
    /// no longer takes part in anything.
    pub fn body(&self, handle: BodyHandle) -> &RigidBody {
        let index = handle.index();
        assert!(
            !self.bodies[index].is_removed(),
            "body {} has been removed",
            index
        );
        &self.bodies[index]
    }

    /// Mutable access to the body at `handle` (panics if removed, see
    /// [`PhysicsWorld::body`]).
    pub fn body_mut(&mut self, handle: BodyHandle) -> &mut RigidBody {
        let index = handle.index();
        assert!(
            !self.bodies[index].is_removed(),
            "body {} has been removed",
            index
        );
        &mut self.bodies[index]
    }

    /// Move/rotate a child shape of a compound body between steps (the
    /// equivalent of Bullet's `btCompoundShape::updateChildTransform` +
    /// `updateSingleAabb`): the dynamic body's inverse inertia is
    /// recomputed from the new child layout, and the broad-phase proxy is
    /// re-expanded from the current pose immediately — necessary because
    /// the per-step AABB pass only covers active bodies, so sleeping or
    /// static compounds would otherwise keep stale bounds after their
    /// children move. Call between steps: the contact assembly reads
    /// child shapes while stepping; solver rows hold no child references
    /// across steps.
    ///
    /// Returns `false` when the body was removed or its shape is not a
    /// compound.
    pub fn set_child_transform(
        &mut self,
        handle: BodyHandle,
        child: usize,
        position: Vector3,
        orientation: Quaternion,
    ) -> bool {
        let index = handle.index();
        let body = &mut self.bodies[index];
        if body.is_removed() || !body.set_child_transform(child, position, orientation) {
            return false;
        }
        let aabb = body
            .shape
            .world_aabb(body.position, body.orientation)
            .expanded(BROADPHASE_AABB_MARGIN);
        self.broadphase.set_aabb(self.body_proxies[index], aabb);
        true
    }

    /// Remove a child shape from a compound body (Bullet's
    /// `removeChildShape` + `updateSingleAabb`), refreshing the inertia
    /// and broad-phase proxy like
    /// [`PhysicsWorld::set_child_transform`](Self::set_child_transform).
    /// Returns the removed child, or `None` when the body was removed,
    /// its shape is not a compound, or the index is out of bounds.
    pub fn remove_child_shape(
        &mut self,
        handle: BodyHandle,
        child: usize,
    ) -> Option<CompoundChild> {
        let index = handle.index();
        let body = &mut self.bodies[index];
        if body.is_removed() {
            return None;
        }
        let removed = body.remove_child_shape(child)?;
        let aabb = body
            .shape
            .world_aabb(body.position, body.orientation)
            .expanded(BROADPHASE_AABB_MARGIN);
        self.broadphase.set_aabb(self.body_proxies[index], aabb);
        Some(removed)
    }

    pub fn bodies(&self) -> &[RigidBody] {
        &self.bodies
    }

    /// Remove a body from the world (Bullet's `removeRigidBody`): destroys
    /// its broad-phase proxy, drops its cached contact manifolds (Bullet's
    /// `cleanProxyFromPairs` + `destroyProxy`) and neutralizes the body.
    ///
    /// Constraints referencing the body are removed automatically — joints,
    /// multibody-vs-rigidbody constraints and the collision suppression the
    /// joints contributed (Bullet leaves that to the caller and dangles
    /// otherwise; recorded deviation). Body handles are stable: the other
    /// bodies keep their handles and the removed slot is never reused, so a
    /// stale handle reads inert data instead of a different body. External
    /// users of the handle (a vehicle's chassis, a character controller) must
    /// stop using it, exactly like Bullet's pointer holders.
    pub fn remove_body(&mut self, handle: BodyHandle) {
        let index = handle.index();
        assert!(
            !self.bodies[index].is_removed(),
            "body {} has already been removed",
            index
        );
        // Destroy the broad-phase proxy first: its pairs die with it, so no
        // future step can pair or query the body.
        let proxy = self.body_proxies[index];
        self.broadphase.destroy_proxy(proxy);
        self.proxy_owners[proxy] = ProxyOwner::Free;
        // Poison the proxy back-reference so any stray set_aabb on the dead
        // body panics instead of writing a recycled proxy.
        self.body_proxies[index] = usize::MAX;
        self.bodies[index].mark_removed();
        for bucket in &mut self.manifold_cache {
            bucket.retain(|e| e.other != index);
        }
        if let Some(bucket) = self.manifold_cache.get_mut(index) {
            bucket.clear();
        }
        self.constraints.retain(|c| {
            c.body_a().index() != index && c.body_b().index() != index
        });
        self.mb_constraints
            .retain(|c| c.rigid_body_b() != Some(handle));
        // Soft-body anchors pinning this body die with it (the anchors'
        // body indices would dangle, exactly like the joint cascade above —
        // a recorded deviation: Bullet leaves the user to remove them).
        for sb in &mut self.soft_bodies {
            let mut unanchored = false;
            sb.anchors.retain(|a| {
                let keep = a.body != handle;
                unanchored |= !keep;
                keep
            });
            if unanchored {
                // Recheck battach: only clear it when no anchor of that
                // node survived (Bullet's removeAnchor semantics).
                for i in 0..sb.nodes.len() {
                    if !sb.anchors.iter().any(|a| a.node == i) {
                        sb.nodes[i].battach = false;
                    }
                }
            }
        }
        // Collision events: removing a body ends its touches — a Stopped
        // is emitted for pairs that were touching (Rapier reports removed
        // colliders the same way); the buffer survives until the next step
        // clears it, so callers see the removal right away.
        let idx = index as u32;
        let mut ended = Vec::new();
        self.prev_touching.retain(|p| {
            let hit = p[0] == idx || p[1] == idx;
            if hit {
                ended.push(*p);
            }
            !hit
        });
        self.touching.retain(|p| p[0] != idx && p[1] != idx);
        for pair in ended {
            let (a, b) = (pair[0] as usize, pair[1] as usize);
            let sensor = self.pair_sensor(a, b);
            self.contact_events
                .push(CollisionEvent::Stopped {
                    a: BodyHandle::from_index(a),
                    b: BodyHandle::from_index(b),
                    sensor,
                });
        }
        self.no_collision_pairs.clear();
        for c in &self.constraints {
            if c.disable_collisions() {
                let (a, b) = (c.body_a().index(), c.body_b().index());
                Self::insert_suppressed(&mut self.no_collision_pairs, a, b);
            }
        }
    }

    /// Collision events of the step that ran last. Every pair with at
    /// least one body opted in via [`RigidBody::collision_events`] reports
    /// `Started` on contact and `Stopped` on separation; sleeping pairs
    /// keep their state and emit nothing while asleep. The buffer is
    /// cleared at the start of the next [`PhysicsWorld::step_simulation`]
    /// (or drained immediately with [`PhysicsWorld::take_contact_events`]).
    pub fn contact_events(&self) -> &[CollisionEvent] {
        &self.contact_events
    }

    /// Take the collision events of the last step, leaving the buffer
    /// empty.
    pub fn take_contact_events(&mut self) -> Vec<CollisionEvent> {
        core::mem::take(&mut self.contact_events)
    }

    /// Insert into the sorted, deduplicated suppression set. A free
    /// associated function so callers can borrow `self.constraints` and
    /// `self.no_collision_pairs` disjointly inside rebuild loops.
    fn insert_suppressed(set: &mut Vec<(usize, usize)>, a: usize, b: usize) {
        let pair = (a.min(b), a.max(b));
        if let Err(pos) = set.binary_search(&pair) {
            set.insert(pos, pair);
        }
    }

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

    /// Whether either body of the pair is a trigger volume
    /// (`CF_NO_CONTACT_RESPONSE`).
    fn pair_sensor(&self, a: usize, b: usize) -> bool {
        self.bodies[a]
            .collision_flags
            .contains(CollisionFlags::NO_CONTACT_RESPONSE)
            || self.bodies[b]
                .collision_flags
                .contains(CollisionFlags::NO_CONTACT_RESPONSE)
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
    /// handle.
    pub fn add_multibody(&mut self, body: MultiBody) -> MultiBodyHandle {
        self.multibodies.push(body);
        MultiBodyHandle::from_index(self.multibodies.len() - 1)
    }

    pub fn multibody(&self, handle: MultiBodyHandle) -> &MultiBody {
        &self.multibodies[handle.index()]
    }

    pub fn multibody_mut(&mut self, handle: MultiBodyHandle) -> &mut MultiBody {
        &mut self.multibodies[handle.index()]
    }

    pub fn multibodies(&self) -> &[MultiBody] {
        &self.multibodies
    }

    pub fn num_multibodies(&self) -> usize {
        self.multibodies.len()
    }

    /// Add a soft body (Bullet's `btSoftRigidDynamicsWorld::addSoftBody`)
    /// and return its handle. The body owns one broad-phase proxy over its
    /// node-hull bounds and collides with rigid bodies through the SDF_RS
    /// node-contacts (phase B).
    pub fn add_soft_body(&mut self, mut body: SoftBody) -> SoftBodyHandle {
        body.update_bounds();
        let aabb = Aabb::new(body.bounds[0], body.bounds[1]);
        let proxy = self.broadphase.create_proxy(aabb);
        // Broad-phase slots are recycled after removals, so this covers
        // both a fresh slot and a recycled Free one (same pattern as
        // `add_body`).
        if proxy == self.proxy_owners.len() {
            self.proxy_owners.push(ProxyOwner::Soft(self.soft_bodies.len()));
        } else {
            self.proxy_owners[proxy] = ProxyOwner::Soft(self.soft_bodies.len());
        }
        let index = self.soft_bodies.len();
        self.soft_bodies.push(body);
        self.soft_proxies.push(proxy);
        SoftBodyHandle::from_index(index)
    }

    /// Remove a soft body: its proxy is destroyed (slot recycled) and any
    /// anchors of *rigid* bodies pinning it become inert (Bullet removes
    /// the soft body's anchors with it). The other soft bodies renumber —
    /// their handles shift down past the removed slot (index semantics,
    /// unlike the stable rigid-body handles).
    pub fn remove_soft_body(&mut self, handle: SoftBodyHandle) {
        let index = handle.index();
        let proxy = self.soft_proxies[index];
        self.broadphase.destroy_proxy(proxy);
        self.proxy_owners[proxy] = ProxyOwner::Free;
        self.soft_bodies.remove(index);
        self.soft_proxies.remove(index);
        // Reindex the surviving owners above the removed slot.
        for p in &self.soft_proxies {
            if let ProxyOwner::Soft(i) = &mut self.proxy_owners[*p]
                && *i > index
            {
                *i -= 1;
            }
        }
    }

    pub fn soft_body(&self, handle: SoftBodyHandle) -> &SoftBody {
        &self.soft_bodies[handle.index()]
    }

    pub fn soft_body_mut(&mut self, handle: SoftBodyHandle) -> &mut SoftBody {
        &mut self.soft_bodies[handle.index()]
    }

    pub fn soft_bodies(&self) -> &[SoftBody] {
        &self.soft_bodies
    }

    pub fn num_soft_bodies(&self) -> usize {
        self.soft_bodies.len()
    }

    /// Install a linear joint between soft body `soft`'s master cluster
    /// (`clusters[0]`, Bullet's `appendLinearJoint(specs, body)` overload)
    /// and any other side — a rigid body or another soft body's master
    /// cluster. Requires the soft body to have clusters
    /// (`generate_clusters` first).
    pub fn add_soft_linear_joint(
        &mut self,
        soft: SoftBodyHandle,
        specs: LinearJointSpecs,
        body1: Side,
    ) {
        let xform1 = self.side_xform(body1);
        self.soft_bodies[soft.index()].append_linear_joint(soft.index(), 0, &specs, body1, xform1);
    }

    /// Install an angular joint between soft body `soft`'s master cluster
    /// and any other side (Bullet's `appendAngularJoint(specs, body)`
    /// overload). Requires the soft body to have clusters.
    pub fn add_soft_angular_joint(
        &mut self,
        soft: SoftBodyHandle,
        specs: AngularJointSpecs,
        body1: Side,
    ) {
        let xform1 = self.side_xform(body1);
        self.soft_bodies[soft.index()].append_angular_joint(soft.index(), 0, &specs, body1, xform1);
    }

    /// The world frame of one joint side: a rigid body's transform or a
    /// cluster's frame.
    fn side_xform(&self, side: Side) -> (Vector3, Matrix3) {
        match side {
            Side::Cluster { soft, cluster } => {
                let c = &self.soft_bodies[soft.index()].clusters[cluster];
                (c.framepos, c.framerot)
            }
            Side::Rigid(i) => (
                self.bodies[i.index()].position,
                Matrix3::from_quat(self.bodies[i.index()].orientation),
            ),
        }
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
        // Fast path: colliders are only ever attached (never detached) and
        // entries are only pushed here, so an unchanged attached-collider
        // count means there is nothing to register. One O(n) count pass
        // replaces the per-step wanted-list rebuild + quadratic duplicate
        // scan below (the steady state of e.g. a 200-chain scene).
        let attached = self.multibodies.iter().map(|mb| {
            mb.base_collider().is_some() as usize
                + (0..mb.num_links())
                    .filter(|&l| mb.link_collider(l).is_some())
                    .count()
        });
        if attached.sum::<usize>() == self.collider_entries.len() {
            return;
        }
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
            }
            .expanded(BROADPHASE_AABB_MARGIN);
            let proxy = self.broadphase.create_proxy(aabb);
            self.collider_entries[i].proxy = Some(proxy);
            // See `add_body`: the slot may be a recycled one.
            if proxy == self.proxy_owners.len() {
                self.proxy_owners.push(ProxyOwner::Collider(i));
            } else {
                debug_assert_eq!(self.proxy_owners[proxy], ProxyOwner::Free);
                self.proxy_owners[proxy] = ProxyOwner::Collider(i);
            }
        }
    }

    /// Cast a ray against every body and return the closest hit together with
    /// the body handle (Bullet's `rayTest` with a closest-result callback).
    ///
    /// The broad phase first prunes proxies whose AABB the segment does not
    /// cross, so the (more expensive) exact per-shape test only runs on
    /// candidates — mirroring `btAxisSweep3::rayTest` feeding a
    /// `ClosestRayResultCallback`. The AABB test is conservative, so the
    /// result is identical to testing every body.
    pub fn raycast_closest(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
    ) -> Option<(BodyHandle, RayHit)> {
        self.raycast_closest_excluding(ray_from, ray_to, None)
    }

    /// Closest ray hit against the soft bodies' surface faces (Bullet's
    /// per-body `btSoftBody::rayTest`): returns the soft-body handle, the
    /// face index and the segment fraction. Rigid-body hits are not
    /// considered — a combined query joins the world-API cleanup (phase E).
    pub fn raycast_soft_closest(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
    ) -> Option<(SoftBodyHandle, usize, Real)> {
        let mut best: Option<(SoftBodyHandle, usize, Real)> = None;
        for (i, sb) in self.soft_bodies.iter().enumerate() {
            if let Some((face, fraction)) = sb.ray_test(ray_from, ray_to) {
                let closer = match best {
                    Some((_, _, f)) => fraction < f,
                    None => true,
                };
                if closer {
                    best = Some((SoftBodyHandle::from_index(i), face, fraction));
                }
            }
        }
        best
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
        ignore: Option<BodyHandle>,
    ) -> Option<(BodyHandle, RayHit)> {
        let mut best: Option<(BodyHandle, RayHit)> = None;
        self.raycast_walk(ray_from, ray_to, ignore, RayFilter::DEFAULT, |i, hit| {
            if best.is_none_or(|(_, b)| hit.fraction < b.fraction) {
                best = Some((i, hit));
            }
        });
        best
    }

    /// Every body hit along the segment, nearest first (Bullet's `rayTest`
    /// with an all-hits callback). Bodies whose collision group/mask pair
    /// test rejects the query's [`RayFilter::DEFAULT`] group and mask are
    /// not tested.
    pub fn raycast_all(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<(BodyHandle, RayHit)> {
        self.raycast_all_filtered(ray_from, ray_to, RayFilter::DEFAULT)
    }

    /// Every hit among the bodies the `filter` accepts, nearest first.
    pub fn raycast_all_filtered(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        filter: RayFilter,
    ) -> Vec<(BodyHandle, RayHit)> {
        let mut hits: Vec<(BodyHandle, RayHit)> = Vec::new();
        self.raycast_walk(ray_from, ray_to, None, filter, |i, hit| {
            hits.push((i, hit));
        });
        hits.sort_by(|a, b| a.1.fraction.total_cmp(&b.1.fraction));
        hits
    }

    /// Closest hit among the bodies the `filter` accepts (Bullet's
    /// `rayTest` with a filtered closest-result callback).
    pub fn raycast_closest_filtered(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        filter: RayFilter,
    ) -> Option<(BodyHandle, RayHit)> {
        let mut best: Option<(BodyHandle, RayHit)> = None;
        self.raycast_walk(ray_from, ray_to, None, filter, |i, hit| {
            if best.is_none_or(|(_, b)| hit.fraction < b.fraction) {
                best = Some((i, hit));
            }
        });
        best
    }

    /// The shared ray walk: the broad phase prunes proxies whose AABB the
    /// segment does not cross, link-collider proxies are skipped (phase C),
    /// and the filter applies the symmetric group & mask test. Each hit is
    /// handed to `out` in no particular order.
    fn raycast_walk(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        ignore: Option<BodyHandle>,
        filter: RayFilter,
        mut out: impl FnMut(BodyHandle, RayHit),
    ) {
        for p in self.broadphase.raycast_candidates(ray_from, ray_to) {
            // Link colliders join the world queries together with their
            // contact response (phase C); until then only rigid bodies are
            // tested.
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            let handle = BodyHandle::from_index(i);
            if Some(handle) == ignore {
                continue;
            }
            let body = &self.bodies[i];
            if !filter.accepts(body) {
                continue;
            }
            if let Some(hit) = raycast::raycast(
                &body.shape,
                body.position,
                body.orientation,
                ray_from,
                ray_to,
            ) {
                out(handle, hit);
            }
        }
    }

    /// Handles of the bodies whose broad-phase AABB overlaps `aabb` (exact
    /// AABB test; the caller then runs the narrow phase on the candidates).
    /// Multibody link colliders overlap too but are not reported until they
    /// join the queries in phase C.
    pub fn aabb_candidates(&self, aabb: Aabb) -> Vec<BodyHandle> {
        self.broadphase
            .aabb_candidates(aabb)
            .into_iter()
            .filter_map(|p| match self.proxy_owners[p] {
                ProxyOwner::Body(i) => Some(BodyHandle::from_index(i)),
                ProxyOwner::Collider(_) | ProxyOwner::Soft(_) | ProxyOwner::Free => None,
            })
            .collect()
    }

    /// Closest convex sweep of `shape` from `from` to `to` against every body
    /// except `ignore`, returning the hit body handle and the closest hit.
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
        ignore: Option<BodyHandle>,
    ) -> Option<(BodyHandle, SweepHit)> {
        let swept = shape
            .world_aabb(from, ori)
            .merged(&shape.world_aabb(to, ori));
        let mut best: Option<(BodyHandle, SweepHit)> = None;
        for p in self.broadphase.aabb_candidates(swept) {
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            let handle = BodyHandle::from_index(i);
            if Some(handle) == ignore {
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
                best = Some((handle, hit));
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
        ignore: Option<BodyHandle>,
    ) -> Vec<(BodyHandle, ContactManifold)> {
        let aabb = shape.world_aabb(pos, ori);
        let mut out = Vec::new();
        for p in self.broadphase.aabb_candidates(aabb) {
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            let handle = BodyHandle::from_index(i);
            if Some(handle) == ignore {
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
                out.push((handle, m));
            }
        }
        out
    }

    /// Set a body's pose directly and refresh its broad-phase AABB. Used by
    /// the character controller to move its kinematic body; the solver still
    /// treats the body as immovable (static bodies are never integrated).
    pub fn set_body_transform(
        &mut self,
        handle: BodyHandle,
        position: Vector3,
        orientation: Quaternion,
    ) {
        let index = handle.index();
        assert!(
            !self.bodies[index].is_removed(),
            "body {} has been removed",
            index
        );
        let body = &mut self.bodies[index];
        body.position = position;
        body.orientation = orientation;
        let aabb = body
            .shape
            .world_aabb(position, orientation)
            .expanded(BROADPHASE_AABB_MARGIN);
        self.broadphase.set_aabb(self.body_proxies[index], aabb);
    }

    /// Add a joint between two bodies and return its index (Bullet's
    /// `addConstraint`; the joint suppresses collisions between its bodies by
    /// default, see [`Constraint::set_disable_collisions`]).
    pub fn add_constraint(&mut self, constraint: Constraint) -> usize {
        if constraint.disable_collisions() {
            Self::insert_suppressed(
                &mut self.no_collision_pairs,
                constraint.body_a().index(),
                constraint.body_b().index(),
            );
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
                let (a, b) = (c.body_a().index(), c.body_b().index());
                Self::insert_suppressed(&mut self.no_collision_pairs, a, b);
            }
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
                // island sleeps.
                if (self.bodies[i].collision_events || self.bodies[j].collision_events)
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
                            self.contact_rows
                                .push(ContactConstraint::prepare(i, j, &cp, a, b, k as u32));
                        }
                    }
                    if !self.refresh_out.is_empty()
                        && (a.collision_events || b.collision_events)
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
                        self.contact_rows
                            .push(ContactConstraint::prepare(i, j, &cp, a, b, k as u32));
                    }
                }
                if a.collision_events || b.collision_events {
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

    /// Soft-vs-rigid collision for one broad-phase pair (Bullet's
    /// `btSoftRigidCollisionAlgorithm` -> `defaultCollisionHandler`): the
    /// dispatch matches `(collisions & RVSmask)` *exactly* — `SDF_RS` nodes
    /// the body, `CL_RS` clusters it, combinations fall through to nothing
    /// (Bullet's switch semantics). Detection uses the rigid body's
    /// *current* pose — the rigid solve has not run yet, matching Bullet's
    /// detection inside `performDiscreteCollisionDetection`.
    fn soft_vs_rigid(&mut self, soft: usize, rigid: usize) {
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
    fn soft_vs_mb(&mut self, soft: usize, entry: usize) {
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
    fn soft_vs_soft(&mut self, a: usize, b: usize) {
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
    fn step_islands(&mut self, pairs: &[(usize, usize)]) {
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
    /// pool) — the pure-multibody path with no collider contacts.
    fn mb_solve_internal_constraints(&mut self, dt: Real) {
        // Reset the solver scratch and companion ids (Bullet's
        // `solveGroupCacheFriendlySetup`).
        self.mb_data.reset();
        self.mb_rows.clear();
        for mb in &mut self.multibodies {
            mb.companion_id = -1;
        }

        // Convert the constraints into rows. Constraints of a sleeping
        // island (both sides asleep after the island pass) produce no rows.
        {
            let PhysicsWorld {
                bodies,
                multibodies,
                mb_constraints: constraints,
                mb_data: data,
                mb_scratch: scratch,
                mb_rows: rows,
                mb_erp: erp,
                ..
            } = self;
            for (i, c) in constraints.iter_mut().enumerate() {
                let side_b_awake = match c.rigid_body_b() {
                    Some(rb) => bodies[rb.index()].is_active(),
                    None => multibodies[c.body_b().index()].is_awake(),
                };
                if !multibodies[c.body_a().index()].is_awake() && !side_b_awake {
                    continue;
                }
                c.create_constraint_rows(i, rows, data, scratch, multibodies, bodies, dt, *erp);
            }
        }

        // PGS iterations over the non-contact rows, alternating direction by
        // iteration parity (Bullet's `solveSingleIteration`).
        let outer = self.iterations as usize;
        let inner = self.num_non_contact_inner_iterations as usize;
        for iteration in 0..outer {
            for _ in 0..inner {
                self.mb_rows_sweep(iteration);
            }
        }

        self.mb_writeback_rows(dt);
        // Joint feedback constraint pass (see `solve_mixed`); no-op unless a
        // link carries a feedback slot.
        for mb in &mut self.multibodies {
            if mb.is_awake() && mb.needs_joint_feedback() {
                mb.compute_accelerations_aba(dt, &mut self.mb_scratch, true);
            }
        }
        // Flush any split-impulse deltas (no-op unless split impulses are
        // used; kept for fidelity with `solveInternalConstraints`).
        for mb in &mut self.multibodies {
            mb.process_delta_vee_2();
        }
    }

    /// The merged solve (Bullet's `solveMultiBodyGroup`): rigid contact +
    /// joint rows, multibody contact rows and multibody constraint rows in
    /// one PGS loop, with the rigid split-impulse pass after the velocity
    /// iterations and the multibody writebacks at the end.
    fn solve_mixed(
        &mut self,
        constraints: &mut [ContactConstraint],
        joint_rows: &mut [ConstraintRow],
        inv_dt: Real,
    ) {
        // Shared solver pool + companion ids for the contact and constraint
        // rows (Bullet's `solveGroupCacheFriendlySetup`).
        self.mb_data.reset();
        self.mb_rows.clear();
        self.mb_contact_rows.clear();
        for mb in &mut self.multibodies {
            mb.companion_id = -1;
        }

        // Convert the collected contacts and the constraints into rows.
        // Contacts of still-sleeping multibodies (not woken by anything
        // active) are skipped — their pair produced no impulse.
        {
            let PhysicsWorld {
                bodies,
                multibodies,
                mb_pending_contacts,
                mb_contact_rows,
                mb_data: data,
                mb_scratch: scratch,
                mb_constraints,
                mb_rows,
                mb_erp: erp,
                ..
            } = self;
            for p in mb_pending_contacts.iter() {
                let awake = |s: &ContactSide| match *s {
                    ContactSide::Rigid(_) => true,
                    ContactSide::Link { mb, .. } => multibodies[mb].is_awake(),
                };
                if !awake(&p.side_a) || !awake(&p.side_b) {
                    continue;
                }
                contact::convert_contact(
                    p,
                    bodies,
                    multibodies,
                    data,
                    scratch,
                    inv_dt,
                    mb_contact_rows,
                );
            }
            let dt = 1.0 / inv_dt;
            for (i, c) in mb_constraints.iter_mut().enumerate() {
                // Constraints of a sleeping island (both sides asleep after
                // the island pass) produce no rows.
                let side_b_awake = match c.rigid_body_b() {
                    Some(rb) => bodies[rb.index()].is_active(),
                    None => multibodies[c.body_b().index()].is_awake(),
                };
                if !multibodies[c.body_a().index()].is_awake() && !side_b_awake {
                    continue;
                }
                c.create_constraint_rows(i, mb_rows, data, scratch, multibodies, bodies, dt, *erp);
            }
        }

        // One PGS iteration: rigid sweep, multibody contact sweep (normals,
        // then bounded frictions), then the multibody constraint rows.
        sequential_impulse::warm_start(&mut self.bodies, constraints);
        let outer = self.iterations as usize;
        let inner = self.num_non_contact_inner_iterations as usize;
        for iteration in 0..outer {
            sequential_impulse::velocity_iteration(
                &mut self.bodies,
                constraints,
                joint_rows,
                inv_dt,
            );
            contact::solve_contact_rows(
                &mut self.mb_contact_rows,
                &mut self.mb_data,
                &mut self.bodies,
                &mut self.mb_row_scratch,
            );
            for _ in 0..inner {
                self.mb_rows_sweep(iteration);
            }
        }
        sequential_impulse::split_impulse_pass(
            &mut self.bodies,
            constraints,
            self.iterations,
            inv_dt,
        );

        // Writebacks: the multibody contact impulses first, then the
        // constraint impulses (Bullet's `writeBackSolverBodyToMultiBody`
        // order).
        {
            let PhysicsWorld {
                mb_contact_rows,
                mb_data: data,
                multibodies,
                ..
            } = self;
            contact::writeback_contact_rows(mb_contact_rows, data, multibodies, 1.0 / inv_dt);
        }
        self.mb_writeback_rows(1.0 / inv_dt);
        // Joint feedback: when any link carries a feedback slot, rerun the
        // articulated-body algorithm on the accumulated constraint forces so
        // the feedback reports the solver contribution too (Bullet's
        // `solveInternalConstraints` constraint pass). The pass only reads
        // accelerations — velocities were already integrated above.
        for mb in &mut self.multibodies {
            if mb.is_awake() && mb.needs_joint_feedback() {
                mb.compute_accelerations_aba(1.0 / inv_dt, &mut self.mb_scratch, true);
            }
        }
        for mb in &mut self.multibodies {
            mb.process_delta_vee_2();
        }
    }

    /// One inner sweep over the multibody constraint rows, alternating
    /// direction by iteration parity (Bullet's `solveSingleIteration`).
    fn mb_rows_sweep(&mut self, iteration: usize) {
        let PhysicsWorld {
            mb_rows,
            mb_data,
            bodies,
            ..
        } = self;
        if iteration.is_multiple_of(2) {
            for row in mb_rows.iter_mut() {
                resolve_row(row, mb_data, bodies);
            }
        } else {
            for row in mb_rows.iter_mut().rev() {
                resolve_row(row, mb_data, bodies);
            }
        }
    }

    /// Write the accumulated constraint impulses back into the constraints
    /// and the generalized velocities (Bullet's
    /// `writeBackSolverBodyToMultiBody`). Rigid side-B bodies already took
    /// their impulses during the iterations (they have no deferred
    /// generalized writeback), so only the multibody sides are written here.
    fn mb_writeback_rows(&mut self, dt: Real) {
        for row in &self.mb_rows {
            self.mb_constraints[row.org_constraint]
                .set_applied_impulse(row.org_dof_index, row.applied_impulse);
        }
        for row in &self.mb_rows {
            let imp = row.applied_impulse;
            let mb = &mut self.multibodies[row.multi_body_a];
            mb.apply_delta_vee(
                &self.mb_data.delta_velocities_unit_impulse
                    [row.jac_a_index..row.jac_a_index + row.ndof_a],
                imp,
            );
            // Accumulate the row impulse as a constraint force for the joint
            // feedback (Bullet's `writeBackSolverBodyToMultiBody`).
            let force_a = row.contact_normal1 * (imp / dt);
            let torque_a = row.rel_pos1_cross_normal * (imp / dt);
            match row.link_a {
                None => {
                    mb.add_base_constraint_force(force_a);
                    mb.add_base_constraint_torque(torque_a);
                }
                Some(l) => {
                    mb.add_link_constraint_force(l, force_a);
                    mb.add_link_constraint_torque(l, torque_a);
                }
            }
            if row.rigid_b.is_some() {
                continue; // the rigid side applied its impulses in-solve
            }
            let mb = &mut self.multibodies[row.multi_body_b];
            mb.apply_delta_vee(
                &self.mb_data.delta_velocities_unit_impulse
                    [row.jac_b_index..row.jac_b_index + row.ndof_b],
                imp,
            );
            let force_b = row.contact_normal2 * (imp / dt);
            let torque_b = row.rel_pos2_cross_normal * (imp / dt);
            match row.link_b {
                None => {
                    mb.add_base_constraint_force(force_b);
                    mb.add_base_constraint_torque(torque_b);
                }
                Some(l) => {
                    mb.add_link_constraint_force(l, force_b);
                    mb.add_link_constraint_torque(l, torque_b);
                }
            }
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

/// Solve one multibody constraint row (Bullet's
/// `resolveSingleConstraintRowGeneric`): compute the impulse needed to meet
/// `rhs` given the currently accumulated velocity deltas of both bodies,
/// clamp it to the row window, and add its effect to the accumulated deltas.
/// A rigid side B reads its velocity **delta** since row assembly (the RHS
/// already carries the assembly-time projection) and applies the impulse
/// directly to the body's velocities, like the mixed contact rows.
fn resolve_row(row: &mut SolverRow, data: &mut SolverData, rigid_bodies: &mut [RigidBody]) {
    let ndof_a = row.ndof_a;
    let ndof_b = row.ndof_b;
    let da_index = row.delta_vel_a_index as usize;
    let db_index = row.delta_vel_b_index as usize;

    if row.jac_diag_ab_inv == 0.0 {
        return; // disabled singular row
    }
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
    match row.rigid_b {
        Some(rigid) => {
            let body = &rigid_bodies[rigid.body];
            delta_vel_b_dot_n = row.contact_normal2.dot(body.linear_velocity)
                + row.rel_pos2_cross_normal.dot(body.angular_velocity)
                - rigid.initial_proj;
        }
        None => {
            let jac = &data.jacobians[row.jac_b_index..row.jac_b_index + ndof_b];
            let dv = &data.delta_velocities[db_index..db_index + ndof_b];
            for i in 0..ndof_b {
                delta_vel_b_dot_n += jac[i] * dv[i];
            }
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
    // velocities (Bullet's `applyDeltaVee`); a rigid side B applies the
    // impulse to its real velocities (Bullet's `internalApplyImpulse` on the
    // solver body).
    {
        let unit = &data.delta_velocities_unit_impulse[row.jac_a_index..row.jac_a_index + ndof_a];
        let dv = &mut data.delta_velocities[da_index..da_index + ndof_a];
        for i in 0..ndof_a {
            dv[i] += unit[i] * delta_impulse;
        }
    }
    match row.rigid_b {
        Some(rigid) => {
            let body = &mut rigid_bodies[rigid.body];
            body.linear_velocity += row.contact_normal2 * (delta_impulse * rigid.inv_mass);
            body.angular_velocity += rigid.angular_component * delta_impulse;
        }
        None => {
            let unit =
                &data.delta_velocities_unit_impulse[row.jac_b_index..row.jac_b_index + ndof_b];
            let dv = &mut data.delta_velocities[db_index..db_index + ndof_b];
            for i in 0..ndof_b {
                dv[i] += unit[i] * delta_impulse;
            }
        }
    }
}

/// Whether a contact side currently participates in the simulation: rigid
/// bodies by their activation state, multibodies by their awake flag (a
/// whole articulated body sleeps as one unit).
fn side_active_owner(side: ContactSide, bodies: &[RigidBody], multibodies: &[MultiBody]) -> bool {
    match side {
        ContactSide::Rigid(i) => bodies[i].is_active(),
        ContactSide::Link { mb, .. } => multibodies[mb].is_awake(),
    }
}

#[cfg(test)]
mod world_tests;
