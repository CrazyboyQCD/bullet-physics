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
use crate::softbody::{AngularJointSpecs, LinearJointSpecs, Side, SoftBody, SoftJoint};
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

/// One side of a tracked contact pair (Bullet's collision-object kinds):
/// a rigid body, a multibody link collider, or a whole soft body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionObject {
    /// A rigid body.
    Body(BodyHandle),
    /// A multibody link collider (Bullet's `btMultiBodyLinkCollider`);
    /// `link < 0` is the base collider (the internal sign convention).
    Link { body: MultiBodyHandle, link: i32 },
    /// A whole soft body (pair-level: individual nodes carry no stable
    /// identity).
    Soft(SoftBodyHandle),
}

/// A contact-pair lifecycle event, emitted after each step for every pair
/// with at least one side opted in via [`RigidBody::collision_events`],
/// [`LinkCollider::collision_events`] or [`SoftBody::collision_events`]
/// (Rapier's `CollisionEvent` model; Bullet has no equivalent — its
/// `gContactAddedCallback` is a process-global function pointer fired per
/// contact point). `a < b` always.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionEvent {
    /// The pair began touching this step. `sensor` is set when either side
    /// has [`CollisionFlags::NO_CONTACT_RESPONSE`] — a trigger volume: no
    /// contact response is ever solved for it, only events and queries.
    Started {
        a: CollisionObject,
        b: CollisionObject,
        sensor: bool,
    },
    /// The pair stopped touching this step (separated past the breaking
    /// threshold, left the broad phase, got filtered by group/mask, or a
    /// body was removed).
    Stopped {
        a: CollisionObject,
        b: CollisionObject,
        sensor: bool,
    },
}

/// One side of the tracked-pair id: a kind tag over the object's slot
/// payload, ordered kind-first (bodies, then links, then soft bodies) so a
/// pair canonically sorts as `(min, max)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ContactKey(u8, u32, i32);

impl ContactKey {
    #[inline]
    fn body(i: usize) -> Self {
        Self(0, i as u32, 0)
    }
    #[inline]
    fn link(mb: usize, link: i32) -> Self {
        Self(1, mb as u32, link)
    }
    #[inline]
    fn soft(i: usize) -> Self {
        Self(2, i as u32, 0)
    }
    /// The key of a narrow-phase contact side (the mixed collider pass).
    #[inline]
    pub(crate) fn side(side: ContactSide) -> Self {
        match side {
            ContactSide::Rigid(i) => Self::body(i),
            ContactSide::Link { mb, link } => Self::link(mb, link),
        }
    }
    /// The public handle view of this side.
    fn object(self) -> CollisionObject {
        match self {
            Self(0, i, _) => CollisionObject::Body(BodyHandle::from_index(i as usize)),
            Self(1, mb, link) => CollisionObject::Link {
                body: MultiBodyHandle::from_index(mb as usize),
                link,
            },
            Self(2, i, _) => CollisionObject::Soft(SoftBodyHandle::from_index(i as usize)),
            _ => unreachable!("contact keys carry kinds 0-2"),
        }
    }
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
    /// Opt-in pairs that touched last step, sorted (canonically ordered
    /// [`ContactKey`] pairs): the baseline of the Started/Stopped diff.
    prev_touching: Vec<(ContactKey, ContactKey)>,
    /// Opt-in pairs that touched this step, sorted after the pair pass.
    touching: Vec<(ContactKey, ContactKey)>,
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
        // A body with a motion state is placed at the pose the state
        // reports (Bullet reads `getWorldTransform` in the `btRigidBody`
        // constructor when a motion state is given).
        if let Some(state) = &body.motion_state {
            let (position, orientation) = crate::body::split_pose(state.initial_pose());
            body.position = position;
            body.orientation = orientation;
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
        let removed = ContactKey::body(index);
        let mut ended = Vec::new();
        self.prev_touching.retain(|p| {
            let hit = p.0 == removed || p.1 == removed;
            if hit {
                ended.push(*p);
            }
            !hit
        });
        self.touching.retain(|p| p.0 != removed && p.1 != removed);
        for (a, b) in ended {
            // Pairs tracked only for their overlap list emit no Stopped:
            // the list query simply stops reporting them (see
            // [`PhysicsWorld::overlapping`]).
            if !self.pair_opted_in(a, b) {
                continue;
            }
            let sensor = self.pair_sensor(a, b);
            self.contact_events
                .push(CollisionEvent::Stopped {
                    a: a.object(),
                    b: b.object(),
                    sensor,
                });
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
        let key = ContactKey::body(handle.index());
        let mut out = Vec::new();
        for pair in &self.prev_touching {
            // The tracked pairs span all collision-object kinds; this query
            // lists rigid bodies only.
            let other = if pair.0 == key {
                pair.1
            } else if pair.1 == key {
                pair.0
            } else {
                continue;
            };
            if other.0 == 0 {
                out.push(BodyHandle::from_index(other.1 as usize));
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

    /// Rebuild the joint-collision-suppression set from the current
    /// constraint list. Bullet resolves suppression live at pair processing
    /// (`checkCollideWithOverride` over the ignore list), so a runtime
    /// `set_disable_collisions` flip takes effect on the next step — this
    /// per-step rebuild is the cached equivalent.
    fn rebuild_no_collision_pairs(&mut self) {
        self.no_collision_pairs.clear();
        for c in &self.constraints {
            if c.disable_collisions() {
                let (a, b) = (c.body_a().index(), c.body_b().index());
                Self::insert_suppressed(&mut self.no_collision_pairs, a, b);
            }
        }
    }

    /// Whether either side of the pair is a trigger volume
    /// (`CF_NO_CONTACT_RESPONSE`). Only rigid bodies carry the flag today.
    fn pair_sensor(&self, a: ContactKey, b: ContactKey) -> bool {
        let flag = |k: ContactKey| {
            k.0 == 0
                && self.bodies[k.1 as usize]
                    .collision_flags
                    .contains(CollisionFlags::NO_CONTACT_RESPONSE)
        };
        flag(a) || flag(b)
    }

    /// Whether either side of the pair is opted into collision events.
    fn pair_opted_in(&self, a: ContactKey, b: ContactKey) -> bool {
        self.side_opted_in(a) || self.side_opted_in(b)
    }

    /// The per-side event opt-in (rigid [`RigidBody::collision_events`],
    /// [`LinkCollider::collision_events`], [`SoftBody::collision_events`]).
    fn side_opted_in(&self, key: ContactKey) -> bool {
        match key.0 {
            0 => self.bodies[key.1 as usize].collision_events,
            1 => {
                let mb = &self.multibodies[key.1 as usize];
                let col = if key.2 < 0 {
                    mb.base_collider()
                } else {
                    mb.link_collider(key.2 as usize)
                };
                col.is_some_and(|c| c.collision_events)
            }
            2 => self.soft_bodies[key.1 as usize].collision_events,
            _ => false,
        }
    }

    /// Record or carry over a mixed pair's touch state (the counterpart of
    /// the rigid pair pass's inline bookkeeping): `touched = None` marks a
    /// skipped pair (a sleeping island, a margin artifact) whose prior
    /// touch state freezes, `Some(bool)` is this step's detection result.
    /// `pair` is the canonically ordered key pair.
    fn track_pair(
        &mut self,
        pair: (ContactKey, ContactKey),
        opted_in: bool,
        touched: Option<bool>,
    ) {
        if !opted_in {
            return;
        }
        let carry = match touched {
            Some(t) => t,
            None => self.prev_touching.binary_search(&pair).is_ok(),
        };
        if carry {
            self.touching.push(pair);
        }
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

    /// The two sides of any soft joint (every [`SoftJoint`] variant carries
    /// a `bodies` pair).
    fn joint_sides(joint: &SoftJoint) -> &[Side; 2] {
        match joint {
            SoftJoint::Linear(j) => &j.bodies,
            SoftJoint::Angular(j) => &j.bodies,
            SoftJoint::Contact(j) => &j.bodies,
        }
    }

    fn joint_sides_mut(joint: &mut SoftJoint) -> &mut [Side; 2] {
        match joint {
            SoftJoint::Linear(j) => &mut j.bodies,
            SoftJoint::Angular(j) => &mut j.bodies,
            SoftJoint::Contact(j) => &mut j.bodies,
        }
    }

    /// Remove a soft body: its proxy is destroyed (slot recycled) and any
    /// anchors of *rigid* bodies pinning it become inert (Bullet removes
    /// the soft body's anchors with it). The other soft bodies renumber —
    /// their handles shift down past the removed slot (index semantics,
    /// unlike the stable rigid-body handles); cross-body soft joints
    /// renumber with them and joints anchored to the removed body die.
    pub fn remove_soft_body(&mut self, handle: SoftBodyHandle) {
        let index = handle.index();
        // Collision events: the soft body's touches end with a Stopped,
        // like `remove_body` (its pairs' other sides stay alive). The pair
        // keys are resolved and the per-pair event metadata computed BEFORE
        // any list shifts: the surviving soft slots renumber below, so a
        // post-removal lookup by the old key would read the wrong body —
        // or past the end of the list.
        let removed = ContactKey::soft(index);
        let mut ended = Vec::new();
        self.prev_touching.retain(|p| {
            let hit = p.0 == removed || p.1 == removed;
            if hit {
                ended.push(*p);
            }
            !hit
        });
        self.touching.retain(|p| p.0 != removed && p.1 != removed);
        let ended: Vec<_> = ended
            .into_iter()
            .map(|(a, b)| (a, b, self.pair_opted_in(a, b), self.pair_sensor(a, b)))
            .collect();
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
        // The surviving soft slots renumber, so their tracked keys shift
        // down with them (the removed body's own keys left with the
        // `retain`s above).
        let renumber = |k: &mut ContactKey| {
            if k.0 == 2 && k.1 > index as u32 {
                k.1 -= 1;
            }
        };
        for p in &mut self.prev_touching {
            renumber(&mut p.0);
            renumber(&mut p.1);
        }
        for p in &mut self.touching {
            renumber(&mut p.0);
            renumber(&mut p.1);
        }
        // Cross-body soft joints (user-installed, plus the previous step's
        // contact joints): joints anchored to the removed body die with it,
        // the surviving cluster sides renumber with their bodies. Without
        // this the cluster solver would index a shifted (or out-of-range)
        // body.
        for sb in &mut self.soft_bodies {
            sb.joints.retain(|joint| {
                !Self::joint_sides(joint).iter().any(|s| {
                    matches!(s, Side::Cluster { soft, .. } if soft.index() == index)
                })
            });
            for joint in &mut sb.joints {
                for side in Self::joint_sides_mut(joint) {
                    if let Side::Cluster { soft, .. } = side
                        && soft.index() > index
                    {
                        *soft = SoftBodyHandle::from_index(soft.index() - 1);
                    }
                }
            }
        }
        for (a, b, opted_in, sensor) in ended {
            if !opted_in {
                continue;
            }
            self.contact_events
                .push(CollisionEvent::Stopped {
                    a: a.object(),
                    b: b.object(),
                    sensor,
                });
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
    }

}
mod islands;
mod multibody_pass;
mod query;
mod softbody_pass;
mod step;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::ActivationState;
    use crate::body::filter_groups;
    use approx::assert_relative_eq;

    /// Removing the LAST soft body while its pair is event-tracked: the
    /// ended pair's opt-in/sensor lookups must resolve against the
    /// pre-removal soft-body list (the old key indexes past the end after
    /// the removal), and the Stopped must still be emitted.
    #[test]
    fn remove_last_soft_body_with_tracked_touch_emits_stopped() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.25, 2.0)),
                Vector3::new(0.0, 0.25, 0.0),
            )
            .collision_events(true),
        );
        // Slot 0: parked far away, never touches anything.
        let far = crate::softbody::helpers::create_rope(
            Vector3::new(50.0, 5.0, 0.0),
            Vector3::new(50.0, 3.0, 0.0),
            6,
            0,
        );
        world.add_soft_body(far);
        // Slot 1 (the last): rests on the plate, tracked via the plate's
        // opt-in.
        let near = crate::softbody::helpers::create_rope(
            Vector3::new(0.0, 1.4, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
            6,
            0,
        );
        world.add_soft_body(near);
        let dt = 1.0 / 60.0;
        let mut saw_touch = false;
        for _ in 0..60 {
            world.step_simulation(dt);
            saw_touch |= !world.contact_events().is_empty();
        }
        assert!(saw_touch, "precondition: no touch before the removal");
        world.remove_soft_body(SoftBodyHandle::from_index(1));
        assert!(
            world.contact_events().iter().any(|e| matches!(
                e,
                CollisionEvent::Stopped {
                    a: CollisionObject::Body(_),
                    b: CollisionObject::Soft(h),
                    ..
                } if h.index() == 1
            )),
            "removing the last soft body lost its Stopped event"
        );
    }

    /// Removing a MIDDLE soft body keeps the tracked-pair bookkeeping
    /// coherent: the Stopped for its touches is emitted with the removed
    /// body's identity, and the surviving keys renumber. The end pair's
    /// opt-in must be resolved BEFORE the list shifts — after it, the slot
    /// holds a different body whose flags decide the event's fate.
    #[test]
    fn remove_middle_soft_body_emits_stopped() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.25, 2.0)),
            Vector3::new(0.0, 0.25, 0.0),
        ));
        // Slot 0: far away.
        let far0 = crate::softbody::helpers::create_rope(
            Vector3::new(50.0, 5.0, 0.0),
            Vector3::new(50.0, 3.0, 0.0),
            6,
            0,
        );
        world.add_soft_body(far0);
        // Slot 1 (middle): rests on the plate; THIS body opts into events.
        let mut near = crate::softbody::helpers::create_rope(
            Vector3::new(0.0, 1.4, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
            6,
            0,
        );
        near.collision_events = true;
        world.add_soft_body(near);
        // Slot 2: far away, NOT opted in — after the removal it slides into
        // slot 1, so a post-removal opt-in lookup would read its flags and
        // silently drop the event.
        let far2 = crate::softbody::helpers::create_rope(
            Vector3::new(-50.0, 5.0, 0.0),
            Vector3::new(-50.0, 3.0, 0.0),
            6,
            0,
        );
        world.add_soft_body(far2);
        let dt = 1.0 / 60.0;
        let mut saw_touch = false;
        for _ in 0..60 {
            world.step_simulation(dt);
            saw_touch |= !world.contact_events().is_empty();
        }
        assert!(saw_touch, "precondition: no touch before the removal");
        world.remove_soft_body(SoftBodyHandle::from_index(1));
        assert!(
            world.contact_events().iter().any(|e| matches!(
                e,
                CollisionEvent::Stopped {
                    a: CollisionObject::Body(_),
                    b: CollisionObject::Soft(h),
                    ..
                } if h.index() == 1
            )),
            "removing a middle soft body dropped the opted-in body's Stopped event"
        );
    }

    /// Cross-body soft joints renumber with the removal: a joint on a
    /// surviving body whose other side is a cluster of a shifted body
    /// retargets to the body's new slot (joints anchored to the removed
    /// body die with it), so the cluster solver never indexes a stale or
    /// out-of-range body.
    #[test]
    fn remove_soft_body_renumbers_cross_soft_joint_sides() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let cluster_at = |x: Real, y: Real| {
            let mut sb = crate::softbody::SoftBody::new(
                &[Vector3::new(x, y, 0.0), Vector3::new(x, y - 0.5, 0.0)],
                &[1.0, 1.0],
            );
            sb.generate_clusters(1, 10_000);
            sb
        };
        let s0 = world.add_soft_body(cluster_at(0.0, 5.0));
        // Slot 1: the middle body that will be removed.
        let s1 = crate::softbody::helpers::create_rope(
            Vector3::new(30.0, 5.0, 0.0),
            Vector3::new(30.0, 3.0, 0.0),
            6,
            0,
        );
        world.add_soft_body(s1);
        let s2 = world.add_soft_body(cluster_at(-30.0, 5.0));
        // A user joint on S0 anchored to S2's master cluster.
        world.add_soft_linear_joint(
            s0,
            LinearJointSpecs {
                position: Vector3::new(0.0, 5.0, 0.0),
                ..LinearJointSpecs::default()
            },
            Side::Cluster {
                soft: s2,
                cluster: 0,
            },
        );
        world.remove_soft_body(SoftBodyHandle::from_index(1));
        let sides = match &world.soft_body(s0).joints[0] {
            SoftJoint::Linear(j) => j.bodies,
            _ => panic!("expected the linear joint to survive"),
        };
        assert_eq!(
            sides[1],
            Side::Cluster {
                soft: SoftBodyHandle::from_index(1),
                cluster: 0,
            },
            "the joint side still points at the removed body's old slot"
        );
        // The retargeted joint keeps solving without indexing out of range.
        for _ in 0..10 {
            world.step_simulation(1.0 / 60.0);
        }
    }

    /// A composite "game moment" exercising the consumer-API tier together:
    /// an anti-gravity balloon rises through a trigger volume (per-body
    /// gravity + sensor events), a fast ball is CCD-clamped by a thin plate,
    /// a filter-rejected pair overlaps without colliding, and the settled
    /// state is probed with filtered all-hits rays.
    #[test]
    fn composite_consumer_scene() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -10.0, 0.0));

        let floor = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(20.0, 0.5, 20.0)),
            Vector3::new(0.0, -0.5, 0.0),
        ));
        // The thin CCD target sits off the balloon's column (x >= 1) so the
        // balloon rises unobstructed and the ray below sees only three
        // bodies.
        let _plate = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.5, 0.05, 1.5)),
            Vector3::new(2.5, 1.0, 0.0),
        ));
        let sensor = world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 1.0, 1.0)),
                Vector3::new(0.0, 2.0, 0.0),
            )
            .collision_filter(filter_groups::TRIGGER, filter_groups::ALL)
            .no_contact_response(true)
            .collision_events(true),
        );
        let pillar = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.4, 1.0, 0.4)),
            Vector3::new(-2.0, 0.5, 0.0),
        ));

        // The balloon carries its own upward gravity, detached from the
        // world propagation.
        let mut balloon_body = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 0.6, 0.0),
            1.0,
        );
        balloon_body.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
        balloon_body.set_gravity(Vector3::new(0.0, 3.0, 0.0));
        let balloon = world.add_body(balloon_body);

        // The fast ball drops at 30 m/s (a 0.5 m step) toward the
        // 0.1-thick plate.
        let mut fast_body = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.2),
            Vector3::new(1.5, 8.0, 0.0),
            1.0,
        );
        fast_body.linear_velocity = Vector3::new(0.0, -30.0, 0.0);
        let fast = world.add_body(fast_body.ccd(0.05, 0.15));

        // A ball starting inside the pillar: the pair filter lets it fall
        // through and land on the floor instead of being pushed out.
        let ghost = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(-2.0, 0.8, 0.0),
            1.0,
        ));
        world.set_pair_filter(Some(Box::new(move |a, _: &RigidBody, b, _: &RigidBody| {
            let (lo, hi) = (a.min(b), a.max(b));
            !(lo == pillar.min(ghost) && hi == pillar.max(ghost))
        })));

        let mut saw_sensor_start = false;
        for _ in 0..150 {
            world.step_simulation(1.0 / 60.0);
            for ev in world.contact_events() {
                if matches!(ev, CollisionEvent::Started { sensor: true, .. }) {
                    saw_sensor_start = true;
                }
            }
        }

        // The balloon rose through the sensor, which reported the touch.
        assert!(world.body(balloon).position.y > 3.0);
        assert!(saw_sensor_start);
        // The fast ball was CCD-clamped onto the plate instead of reaching
        // the floor below it.
        let fast_y = world.body(fast).position.y;
        assert!(fast_y > 0.9 && fast_y < 1.45, "fast ball at {fast_y}");
        // The ghost ball fell through the pillar and rests on the floor,
        // still overlapping the pillar's footprint.
        let ghost_pos = world.body(ghost).position;
        assert!((ghost_pos.x + 2.0).abs() < 1e-3);
        assert!(
            ghost_pos.y > 0.2 && ghost_pos.y < 0.45,
            "ghost at {}",
            ghost_pos.y
        );

        // The downward ray through the balloon's column reports every body
        let hits = world.raycast_all(Vector3::new(0.0, 12.0, 0.0), Vector3::new(0.0, -5.0, 0.0));
        let indices: Vec<BodyHandle> = hits.iter().map(|(i, _)| *i).collect();
        assert_eq!(indices, vec![balloon, sensor, floor]);
        assert!(hits.windows(2).all(|w| w[0].1.fraction < w[1].1.fraction));
        // A TRIGGER-group query sees only the sensor.
        let trigger_only = RayFilter {
            group: filter_groups::TRIGGER,
            mask: filter_groups::TRIGGER,
        };
        let hits = world.raycast_all_filtered(
            Vector3::new(0.0, 12.0, 0.0),
            Vector3::new(0.0, -5.0, 0.0),
            trigger_only,
        );
        assert_eq!(
            hits.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![sensor]
        );
        assert_eq!(
            world
                .raycast_closest_filtered(
                    Vector3::new(0.0, 12.0, 0.0),
                    Vector3::new(0.0, -5.0, 0.0),
                    trigger_only,
                )
                .map(|(i, _)| i),
            Some(sensor)
        );
    }

    // ---- body removal -------------------------------------------

    /// A removed body stops colliding (its broad-phase proxy is gone, its
    /// cached manifolds are dropped) and freezes in place; the bodies that
    /// were resting on it fall.
    #[test]
    fn removed_body_stops_colliding_and_falls() {
        let mut world = PhysicsWorld::new();
        let ground = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.5, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        // Settle into contact (the manifold cache now holds the pair).
        for _ in 0..10 {
            world.step_simulation(dt);
        }
        assert!(world.body(ball).position.y > 0.4);
        let ground_y = world.bodies()[ground.index()].position.y;
        world.remove_body(ground);
        // With the ground gone the ball falls freely through the old plane.
        for _ in 0..30 {
            world.step_simulation(dt);
        }
        assert!(
            world.body(ball).position.y < -0.5,
            "ball must fall through the removed ground"
        );
        // The removed body itself is inert data.
        assert_eq!(world.bodies()[ground.index()].position.y, ground_y);
        assert!(world.bodies()[ground.index()].is_removed());
        assert!(!world.bodies()[ground.index()].is_active());
    }

    /// Removed bodies disappear from every world query (ray casts, sweeps,
    /// contact tests all run through the broad phase).
    #[test]
    fn removed_body_is_invisible_to_queries() {
        let mut world = PhysicsWorld::new();
        let sphere = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(1.0, 0.0, 0.0),
        ));
        assert!(
            world
                .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
                .is_some()
        );
        world.remove_body(sphere);
        assert!(
            world
                .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
                .is_none()
        );
    }

    /// Body slots are never reused (indices stay valid) while the freed
    /// broad-phase slot is recycled for the next registration; the owner
    /// table stays consistent.
    #[test]
    fn remove_then_add_recycles_the_proxy_slot() {
        let mut world = PhysicsWorld::new();
        let first = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(1.0, 0.0, 0.0),
        ));
        let second = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(3.0, 0.0, 0.0),
        ));
        let freed = world.body_proxies[first.index()];
        world.remove_body(first);
        // The new body reuses the freed proxy slot and gets a fresh body slot.
        let third = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(1.0, 2.0, 0.0),
        ));
        assert_eq!(third, BodyHandle::from_index(2));
        assert_eq!(world.body_proxies[third.index()], freed);
        // Queries resolve to the right owners afterwards.
        let (index, _) = world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .unwrap();
        assert_eq!(index, second);
        let (index, _) = world
            .raycast_closest(Vector3::new(1.0, 5.0, 0.0), Vector3::new(1.0, -5.0, 0.0))
            .unwrap();
        assert_eq!(index, third);
    }

    /// Joints referencing a removed body go with it; the remaining joints
    /// keep working (their collision suppression survives the rebuild).
    #[test]
    fn remove_body_cascades_joints() {
        let mut world = PhysicsWorld::new();
        let a = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(-1.0, 5.0, 0.0),
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(1.0, 5.0, 0.0),
            1.0,
        ));
        world.add_constraint(crate::constraint::Constraint::point2point(
            a,
            b,
            Vector3::X,
            -Vector3::X,
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        assert_eq!(world.constraints().len(), 1);
        world.remove_body(a);
        assert_eq!(world.constraints().len(), 0);
        // Nothing holds B up anymore: it falls the whole way down to the
        // ground (a surviving joint would keep it near its release height).
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            world.step_simulation(dt);
        }
        assert!(
            world.body(b).position.y < 1.0,
            "released body must fall freely, got {:?}",
            world.body(b).position
        );
    }

    /// Multibody-vs-rigidbody constraints cascade with the rigid side too.
    #[test]
    fn remove_body_cascades_multibody_constraints() {
        let mut world = PhysicsWorld::new();
        let mut mb = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        mb.finalize_multidof();
        let ia = world.add_multibody(mb);
        let rb = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        let snapshot = world.multibodies().to_vec();
        let constraint = MultiBodyConstraint::point2point_to_rigid(
            ia,
            None,
            rb,
            Vector3::ZERO,
            Vector3::new(0.0, 1.0, 0.0),
            &snapshot,
        );
        world.add_multibody_constraint(constraint);
        assert_eq!(world.num_multibody_constraints(), 1);
        world.remove_body(rb);
        assert_eq!(world.num_multibody_constraints(), 0);
        // The multibody itself keeps simulating.
        for _ in 0..10 {
            world.step_simulation(1.0 / 60.0);
        }
    }

    /// A removed body no longer blocks the CCD swept-sphere clamps (the
    /// candidate sweep skips inert slots; its proxy is destroyed).
    #[test]
    fn removed_body_does_not_block_ccd() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        // Thin static wall spanning y in [0.4, 0.6] — same scene as
        // `ccd_clamps_fast_bodies_at_the_impact`.
        let wall = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        world.remove_body(wall);
        world.step_simulation(1.0 / 60.0);
        assert!(
            world.body(ball).position.y < -0.5,
            "CCD clamp must ignore the removed wall"
        );
    }

    /// The remaining bodies keep their indices and simulate normally after
    /// a removal (no renumbering, no swap).
    #[test]
    fn removed_body_keeps_remaining_indices_stable() {
        let mut world = PhysicsWorld::new();
        let ground = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let a = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 5.0, 0.0),
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(2.0, 8.0, 0.0),
            1.0,
        ));
        world.remove_body(a);
        assert_eq!(world.bodies().len(), 3);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            world.step_simulation(dt);
        }
        // B landed on the still-present ground; the indices never moved.
        assert_relative_eq!(world.body(b).position.y, 0.5, epsilon = 0.05);
        assert_eq!(ground, BodyHandle::from_index(0));
        assert!(world.bodies()[a.index()].is_removed());
    }

    #[test]
    #[should_panic(expected = "already been removed")]
    fn removing_a_body_twice_panics() {
        let mut world = PhysicsWorld::new();
        let body = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::ZERO,
        ));
        world.remove_body(body);
        world.remove_body(body);
    }

    #[test]
    #[should_panic(expected = "has been removed")]
    fn accessing_a_removed_body_panics() {
        let mut world = PhysicsWorld::new();
        let body = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::ZERO,
        ));
        world.remove_body(body);
        let _ = world.body(body);
    }

    /// Mutating a static compound's child between steps must refresh the
    /// broad-phase proxy immediately: the per-step AABB pass only covers
    /// active bodies, so a moved child would otherwise stay invisible to
    /// pair detection at its new location (a ray walks the proxy first).
    #[test]
    fn static_compound_child_transform_refreshes_the_proxy() {
        let mut world = PhysicsWorld::new();
        let compound = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::compound(vec![
                bullet_collision::shapes::CompoundChild::new(
                    bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
                    Vector3::new(0.0, 0.5, 0.0),
                ),
            ]),
            Vector3::ZERO,
        ));
        assert!(world.set_child_transform(
            compound,
            0,
            Vector3::new(6.0, 0.5, 0.0),
            Quaternion::IDENTITY,
        ));

        // A downward ray at the new location hits the moved child; at the old
        // location it finds nothing (of this compound).
        let hit = world.raycast_closest(Vector3::new(6.0, 3.0, 0.0), Vector3::new(6.0, -1.0, 0.0));
        assert!(hit.is_some(), "moved child invisible: proxy not refreshed");
        let hit = world.raycast_closest(Vector3::new(0.0, 3.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        assert!(hit.is_none(), "stale child bounds still visible");
        world.step_simulation(1.0 / 60.0);
        // The refresh survives a step (update_aabbs skips inactive bodies).
        let hit = world.raycast_closest(Vector3::new(6.0, 3.0, 0.0), Vector3::new(6.0, -1.0, 0.0));
        assert!(hit.is_some(), "moved child lost after a step");
    }

    /// A dynamic compound's inverse inertia follows the child layout (the
    /// construction formula re-run); an unchanged axis stays untouched.
    #[test]
    fn dynamic_compound_child_transform_recomputes_inertia() {
        let mut door = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::compound(vec![
                bullet_collision::shapes::CompoundChild::new(
                    bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 2.0, 0.25)),
                    Vector3::new(-3.0, 0.0, 0.0),
                ),
                bullet_collision::shapes::CompoundChild::new(
                    bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 2.0, 0.25)),
                    Vector3::new(3.0, 0.0, 0.0),
                ),
            ]),
            Vector3::ZERO,
            12.0,
        );
        let wide = door.inv_inertia_local();
        // Pull the second child in (the first stays at x = -3): the bounding
        // box shrinks along x, so I_y and I_z shrink while I_x (which only
        // sees the unchanged y/z extents) stays put.
        assert!(door.set_child_transform(1, Vector3::new(1.0, 0.0, 0.0), Quaternion::IDENTITY));
        let narrow = door.inv_inertia_local();
        assert_relative_eq!(narrow.x, wide.x);
        assert!(narrow.y > wide.y && narrow.z > wide.z);

        // A non-compound body reports the miss.
        let mut ball = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::ZERO,
            1.0,
        );
        assert!(!ball.set_child_transform(0, Vector3::ZERO, Quaternion::IDENTITY));
        assert!(ball.remove_child_shape(0).is_none());
    }

    // ---- ghost/sensor persistent overlap lists (Bullet's `btGhostObject`)

    /// A sensor's persistent overlap list tracks a ball flying through it,
    /// with no events emitted (neither side opted into `collision_events`):
    /// the list is the sensor's own query surface.
    #[test]
    fn sensor_overlap_list_tracks_passage() {
        let mut world = PhysicsWorld::new();
        let trigger = world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .no_contact_response(true),
        );
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        ));
        let dt = 1.0 / 60.0;
        let mut seen = false;
        for _ in 0..60 {
            world.step_simulation(dt);
            assert!(world.contact_events().is_empty());
            if !world.overlapping(trigger).is_empty() {
                assert_eq!(world.overlapping(trigger), vec![ball]);
                seen = true;
            }
        }
        assert!(seen, "the ball never appeared in the sensor's overlap list");
        assert!(world.body(ball).position.y < -1.0, "ball passed through");
        assert!(
            world.overlapping(trigger).is_empty(),
            "the list kept the ball after it left"
        );
    }

    /// The list is persistent across steps: a ball resting (and falling
    /// asleep) inside a sensor keeps its entry — no churn while the island
    /// sleeps, like the touch state the events ride on.
    #[test]
    fn sensor_overlap_list_survives_sleep() {
        let mut world = PhysicsWorld::new();
        let trigger = world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .no_contact_response(true),
        );
        let mut ball = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 0.4, 0.0),
            1.0,
        );
        // No gravity: the ball rests inside the trigger and sleeps there.
        ball.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
        let ball = world.add_body(ball);
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            world.step_simulation(dt);
            assert!(world.contact_events().is_empty());
        }
        assert_eq!(
            world.body(ball).activation_state(),
            ActivationState::IslandSleeping,
            "the ball should have fallen asleep inside the sensor"
        );
        assert_eq!(world.overlapping(trigger), vec![ball]);
    }

    /// Removing a tracked body clears the sensor's list immediately, and
    /// still emits no event (the pair had no opted-in side).
    #[test]
    fn sensor_overlap_list_cleared_on_remove() {
        let mut world = PhysicsWorld::new();
        let trigger = world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .no_contact_response(true),
        );
        let mut ball = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 0.4, 0.0),
            1.0,
        );
        ball.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
        let ball = world.add_body(ball);
        let dt = 1.0 / 60.0;
        for _ in 0..3 {
            world.step_simulation(dt);
        }
        assert_eq!(world.overlapping(trigger), vec![ball]);
        world.remove_body(ball);
        assert!(world.overlapping(trigger).is_empty());
        assert!(world.contact_events().is_empty());
    }

    /// One sensor overlapping several bodies reports them in ascending
    /// handle order.
    #[test]
    fn sensor_overlap_list_is_sorted() {
        let mut world = PhysicsWorld::new();
        let trigger = world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .no_contact_response(true),
        );
        let mut left = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(-1.0, 0.4, 0.0),
            1.0,
        );
        left.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
        let left = world.add_body(left);
        let mut right = RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(1.0, 0.4, 0.0),
            1.0,
        );
        right.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
        let right = world.add_body(right);
        for _ in 0..3 {
            world.step_simulation(1.0 / 60.0);
        }
        assert_eq!(world.overlapping(trigger), vec![left, right]);
        // The reverse lookup works too (the balls are not sensors, but their
        // pairs are tracked through the sensor side).
        assert_eq!(world.overlapping(left), vec![trigger]);
    }
}
