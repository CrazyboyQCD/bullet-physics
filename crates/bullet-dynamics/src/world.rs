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
use crate::solver::sequential_impulse::{self, BAUMGARTE, ContactConstraint, ConstraintRow, RollingFriction, SLOP, SPLIT_TURN_ERP};

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
    /// Rolling-friction rows of the current step (Bullet's
    /// `m_tmpSolverContactRollingFrictionConstraintPool`), indexed by the
    /// contact rows' `rolling` field; only populated when a scene actually
    /// sets `RigidBody::rolling_friction`.
    rolling_rows: Vec<RollingFriction>,
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
            rolling_rows: Vec::new(),
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
            // Pairs tracked only for their overlap list emit no Stopped:
            // the list query simply stops reporting them (see
            // [`PhysicsWorld::overlapping`]).
            if !(self.bodies[a].collision_events || self.bodies[b].collision_events) {
                continue;
            }
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

    /// The persistent overlap list of a body (the query surface of Bullet's
    /// `btGhostObject`: `getNumOverlappingObjects` / `getOverlappingObject`):
    /// every body whose pair currently counts as touching `handle`, in
    /// ascending handle order. The list is maintained by the pair pass for
    /// sensor bodies (either side carrying
    /// [`CollisionFlags::NO_CONTACT_RESPONSE`], the trigger-volume flag) and
    /// for pairs with an opted-in body ([`RigidBody::collision_events`]);
    /// it persists across steps, freezes while the pair's island sleeps and
    /// is cleared when a participant is removed or the pair separates.
    /// Unlike Bullet's listing — filled from broad-phase AABB overlaps —
    /// entries reflect the narrow-phase touch state: an actual contact
    /// inside the breaking threshold, not a mere bounds overlap. Reading it
    /// for a body in no tracked pair (non-sensor, no events, untouched)
    /// returns an empty list.
    pub fn overlapping(&self, handle: BodyHandle) -> Vec<BodyHandle> {
        let idx = handle.index() as u32;
        let mut out = Vec::new();
        for pair in &self.prev_touching {
            if pair[0] == idx {
                out.push(BodyHandle::from_index(pair[1] as usize));
            } else if pair[1] == idx {
                out.push(BodyHandle::from_index(pair[0] as usize));
            }
        }
        // The tracked pairs are a set (one entry per pair), so the other
        // sides of one body are already distinct; only their order needs
        // normalizing.
        out.sort_unstable();
        out
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

}
mod islands;
mod multibody_pass;
mod query;
mod softbody_pass;
mod step;

#[cfg(test)]
mod world_tests;
