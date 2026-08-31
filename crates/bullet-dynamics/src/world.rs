use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::{Duration, Instant};

use bullet_collision::broadphase::{BroadPhase, BroadPhaseBackend};
use bullet_collision::manifold::{ContactManifold, PersistentManifold, CONTACT_BREAKING_THRESHOLD};
use bullet_collision::narrowphase::{self, SweepHit};
use bullet_collision::raycast::{self, RayHit};
use crate::body::RigidBody;
use crate::constraint::{self, Constraint};
use crate::island::UnionFind;
use crate::multibody::body::{MultiBody, MultiBodyScratch};
use crate::multibody::contact::{self, ContactSide, MbContactRow, PendingContact};
use crate::multibody::constraint::{MultiBodyConstraint, SolverData, SolverRow};
use crate::softbody::SoftBody;
use bullet_math::{Aabb, Quaternion, Real, Vector3};
use crate::solver::sequential_impulse::{self, BAUMGARTE, ContactConstraint, ConstraintRow, SLOP, SPLIT_TURN_ERP};

/// FxHash-style hasher (rustc's `FxHashMap` recipe) for the pair-keyed maps
/// below: the keys are `(usize, usize)` proxy pairs looked up several times
/// per broad-phase pair per step (mark/refresh/seed/writeback — ~24k lookups
/// on the 1000-body stack), where SipHash's collision-resistance is pure
/// overhead (the SipHash of `RandomState` measured ~12% of the step
/// inclusive; a multiply-xor-rotate mix is collision-safe enough for index
/// pairs and several times cheaper). The hash only picks the bucket — every
/// consumer keys its access, so lookups stay semantically identical.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Byte-wise fallback; never hit by the usize-only pair keys.
        for &b in bytes {
            self.add_to_hash(b as u64);
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

impl FxHasher {
    #[inline]
    fn add_to_hash(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(FX_SEED);
    }
}

type FxBuild = BuildHasherDefault<FxHasher>;

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

/// One instrumented span of [`PhysicsWorld::step_simulation`], in execution
/// order (the numbered stage comments inside the step; the sampling points
/// mirror Bullet's `BT_PROFILE` instrumentation in
/// `btDiscreteDynamicsWorld::stepSimulation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepStage {
    /// Gravity + external-force velocity integration (stages 1/1m).
    IntegrateVelocities,
    /// Predicted poses + broad-phase AABB refresh (stages 2/3/3m).
    UpdateAabbs,
    /// Broad-phase pair query (stage 4).
    Broadphase,
    /// Simulation islands + island sleep decisions (stage 4b).
    Islands,
    /// Narrow phase + joint row assembly (stages 5/5a).
    Narrowphase,
    /// Constraint solving + warm-start cache writeback (stages 6/6b/6bm).
    Solve,
    /// Deactivation, CCD clamping, position integration, force clearing
    /// (stages 6c/7/8/8m).
    IntegratePositions,
}

/// Every stage in execution order.
pub const STEP_STAGES: [StepStage; 7] = [
    StepStage::IntegrateVelocities,
    StepStage::UpdateAabbs,
    StepStage::Broadphase,
    StepStage::Islands,
    StepStage::Narrowphase,
    StepStage::Solve,
    StepStage::IntegratePositions,
];

/// Cumulative wall-clock durations per [`StepStage`], accumulated over every
/// step since the last reset. All zeros until enabled with
/// [`PhysicsWorld::enable_step_timing`]; while disabled the per-step sampling
/// cost is one branch per stage, while enabled it is ~8 `Instant::now()`
/// reads. Timing never touches the numerical path, so it cannot change the
/// simulation result.
#[derive(Clone, Debug)]
pub struct StepTimings {
    enabled: bool,
    stages: [Duration; 7],
}

impl Default for StepTimings {
    fn default() -> Self {
        Self {
            enabled: false,
            stages: [Duration::from_secs(0); 7],
        }
    }
}

impl StepTimings {
    /// Cumulative duration of one stage.
    pub fn stage(&self, stage: StepStage) -> Duration {
        self.stages[stage as usize]
    }

    /// Sum over all stages (the instrumented portion of every step).
    pub fn total(&self) -> Duration {
        self.stages.iter().copied().sum()
    }

    fn add(&mut self, stage: StepStage, span: Duration) {
        self.stages[stage as usize] += span;
    }
}

/// Stage-boundary sampler used inside `step_simulation`; a no-op unless
/// timing was enabled when the step started.
struct StepClock {
    enabled: bool,
    last: Option<Instant>,
}

impl StepClock {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last: enabled.then(Instant::now),
        }
    }

    /// Close the previous stage (accumulating the elapsed span under
    /// `stage`) and restart the clock for the next one.
    #[inline]
    fn mark(&mut self, timings: &mut StepTimings, stage: StepStage) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if let Some(last) = self.last.replace(now) {
            timings.add(stage, now - last);
        }
    }
}

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
    manifold_cache: HashMap<(usize, usize), PersistentManifold, FxBuild>,
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
    /// Joints (Bullet's `btTypedConstraint` list added with `addConstraint`).
    constraints: Vec<Constraint>,
    /// Body pairs whose collision response is suppressed by a joint
    /// (`disableCollisionsBetweenLinkedBodies`, Bullet's
    /// `checkCollideWithOverride`), sorted `(min, max)`.
    no_collision_pairs: HashSet<(usize, usize), FxBuild>,
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
    /// Position-based soft bodies (Bullet's `btSoftBody` list of
    /// `btSoftRigidDynamicsWorld`); each body owns one broad-phase proxy
    /// over its node-hull bounds (phase B: SDF_RS contacts and anchors;
    /// soft-soft and clusters follow — see `docs/softbody.md` §4/§10).
    soft_bodies: Vec<SoftBody>,
    /// Broad-phase proxy of each soft body.
    soft_proxies: Vec<usize>,
    /// Recycled leaf scratch for the soft-body SDF_RS queries.
    soft_scratch: Vec<usize>,
    /// Cumulative per-stage step timings; zero until enabled (see
    /// [`PhysicsWorld::enable_step_timing`]).
    step_timings: StepTimings,
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
            manifold_cache: HashMap::with_hasher(FxBuild::default()),
            contact_rows: Vec::new(),
            joint_rows: Vec::new(),
            predicted_poses: Vec::new(),
            settled_speeds: Vec::new(),
            step_counter: 0,
            refresh_out: Vec::new(),
            prev_simulate: false,
            constraints: Vec::new(),
            no_collision_pairs: HashSet::with_hasher(FxBuild::default()),
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
            soft_bodies: Vec::new(),
            soft_proxies: Vec::new(),
            soft_scratch: Vec::new(),
            step_timings: StepTimings::default(),
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

    /// Start accumulating per-stage wall-clock timings in
    /// `step_simulation` (Bullet's quickprof instrumentation points).
    /// Off by default; enabling costs ~8 clock reads per step.
    pub fn enable_step_timing(&mut self) {
        self.step_timings.enabled = true;
    }

    /// Cumulative per-stage durations over all steps since timing was
    /// enabled (zeros while disabled).
    pub fn step_timings(&self) -> &StepTimings {
        &self.step_timings
    }

    /// Add a body and return its index.
    ///
    /// Broad-phase proxy ids are shared with multibody link colliders, so
    /// they are tracked in [`PhysicsWorld::body_proxies`] (visible to tests).
    /// Indices are stable: removing a body never renumbers the others (see
    /// [`PhysicsWorld::remove_body`]).
    pub fn add_body(&mut self, body: RigidBody) -> usize {
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
        index
    }

    /// The body at `index`. Panics if the body was removed
    /// ([`PhysicsWorld::remove_body`]) — the slot itself stays readable
    /// through [`PhysicsWorld::bodies`] for bookkeeping, but a removed body
    /// no longer takes part in anything.
    pub fn body(&self, index: usize) -> &RigidBody {
        assert!(
            !self.bodies[index].is_removed(),
            "body {index} has been removed"
        );
        &self.bodies[index]
    }

    /// Mutable access to the body at `index` (panics if removed, see
    /// [`PhysicsWorld::body`]).
    pub fn body_mut(&mut self, index: usize) -> &mut RigidBody {
        assert!(
            !self.bodies[index].is_removed(),
            "body {index} has been removed"
        );
        &mut self.bodies[index]
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
    /// otherwise; recorded deviation). Body indices are stable: the other
    /// bodies keep their indices and the removed slot is never reused, so a
    /// stale index reads inert data instead of a different body. External
    /// users of the index (a vehicle's chassis, a character controller) must
    /// stop using it, exactly like Bullet's pointer holders.
    pub fn remove_body(&mut self, index: usize) {
        assert!(
            !self.bodies[index].is_removed(),
            "body {index} has already been removed"
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
        self.manifold_cache
            .retain(|&(i, j), _| i != index && j != index);
        self.constraints
            .retain(|c| c.body_a() != index && c.body_b() != index);
        self.mb_constraints
            .retain(|c| c.rigid_body_b() != Some(index));
        // Soft-body anchors pinning this body die with it (the anchors'
        // body indices would dangle, exactly like the joint cascade above —
        // a recorded deviation: Bullet leaves the user to remove them).
        for sb in &mut self.soft_bodies {
            let mut unanchored = false;
            sb.anchors.retain(|a| {
                let keep = a.body != index;
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
        self.no_collision_pairs.clear();
        for c in &self.constraints {
            if c.disable_collisions() {
                let (a, b) = (c.body_a(), c.body_b());
                self.no_collision_pairs.insert((a.min(b), a.max(b)));
            }
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

    /// Add a soft body (Bullet's `btSoftRigidDynamicsWorld::addSoftBody`)
    /// and return its index. The body owns one broad-phase proxy over its
    /// node-hull bounds and collides with rigid bodies through the SDF_RS
    /// node-contacts (phase B).
    pub fn add_soft_body(&mut self, mut body: SoftBody) -> usize {
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
        index
    }

    /// Remove a soft body: its proxy is destroyed (slot recycled) and any
    /// anchors of *rigid* bodies pinning it become inert (Bullet removes
    /// the soft body's anchors with it).
    pub fn remove_soft_body(&mut self, index: usize) {
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

    pub fn soft_body(&self, index: usize) -> &SoftBody {
        &self.soft_bodies[index]
    }

    pub fn soft_body_mut(&mut self, index: usize) -> &mut SoftBody {
        &mut self.soft_bodies[index]
    }

    pub fn soft_bodies(&self) -> &[SoftBody] {
        &self.soft_bodies
    }

    pub fn num_soft_bodies(&self) -> usize {
        self.soft_bodies.len()
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
                ProxyOwner::Collider(_) | ProxyOwner::Soft(_) | ProxyOwner::Free => None,
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
        assert!(
            !self.bodies[index].is_removed(),
            "body {index} has been removed"
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
        let mut clk = StepClock::new(self.step_timings.enabled);

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
                    self.soft_vs_rigid(s, j);
                    continue;
                }
                (ProxyOwner::Body(j), ProxyOwner::Soft(s)) => {
                    self.soft_vs_rigid(s, j);
                    continue;
                }
                (ProxyOwner::Soft(_), ProxyOwner::Soft(_))
                | (ProxyOwner::Soft(_), ProxyOwner::Collider(_))
                | (ProxyOwner::Collider(_), ProxyOwner::Soft(_)) => continue,
                _ => {}
            }
            let (ProxyOwner::Body(i), ProxyOwner::Body(j)) = (owner_a, owner_b) else {
                // Multibody link-collider pair (mixed or collider-collider):
                // run the narrow phase at the current FK pose and collect the
                // manifold for the contact rows (converted at solve time).
                // Self-collision filtering mirrors `checkCollideWithOverride`.
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
                let Some(manifold) = narrowphase::collide(shape_a, pa, qa, shape_b, pb, qb)
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
            if self.no_collision_pairs.contains(&(i.min(j), i.max(j))) {
                continue;
            }
            // The pair stays in the cache lifecycle even when skipped below:
            // a sleeping island keeps its manifolds (their AABBs still
            // overlap), matching Bullet's persistent manifolds on a
            // sleeping island.
            if simulate
                && let Some(cache) = self.manifold_cache.get_mut(&(i, j))
            {
                cache.mark_paired(stamp);
            }
            // A pair whose owners are both inactive after the island pass
            // belongs to a sleeping island: skip detection and rows
            // entirely (Bullet solves no manifolds of a sleeping island).
            if !self.bodies[i].is_active() && !self.bodies[j].is_active() {
                continue;
            }
            let (a, b) = (&self.bodies[i], &self.bodies[j]);
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
            if let Some(cache) = self.manifold_cache.get_mut(&(i, j))
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
                    for (k, mut cp) in self.refresh_out.iter().copied().enumerate() {
                        cp.normal_impulse *= WARM_STARTING_FACTOR;
                        self.contact_rows
                            .push(ContactConstraint::prepare(i, j, &cp, a, b, k as u32));
                    }
                    served_from_cache = true;
                }
            }
            if served_from_cache {
                continue;
            }
            let seed = self
                .manifold_cache
                .get(&(i, j))
                .and_then(PersistentManifold::seed);
            if let Some(manifold) = narrowphase::collide_seeded(
                &a.shape,
                pa,
                a.orientation,
                &b.shape,
                pb,
                b.orientation,
                seed,
            ) {
                // Detection path: anchor the fresh manifold in both local
                // frames (inheriting impulses of persisting points), then
                // emit rows from the detection geometry. Stamp the new entry
                // right away so this step's prune keeps it.
                let cache = self.manifold_cache.entry((i, j)).or_default();
                cache.mark_paired(stamp);
                let threshold = a
                    .shape
                    .contact_breaking_threshold()
                    .min(b.shape.contact_breaking_threshold());
                cache.capture(&manifold, pa, a.orientation, pb, b.orientation, threshold);
                for (k, (detected, cached)) in manifold.points.iter().zip(cache.points()).enumerate() {
                    let mut cp = *detected;
                    cp.normal_impulse = cached.normal_impulse * WARM_STARTING_FACTOR;
                    cp.tangent_impulse = [0.0, 0.0];
                    self.contact_rows
                        .push(ContactConstraint::prepare(i, j, &cp, a, b, k as u32));
                }
            } else {
                // Detection found nothing: the pair separated past the
                // breaking threshold — drop the cache entry entirely.
                self.manifold_cache.remove(&(i, j));
            }
        }

        // 5a. Joint rows: flatten every enabled joint of an awake island
        // into its solver rows from the current (pre-integration) poses,
        // like Bullet's `convertJoints` after `predictUnconstraintMotion`.
        // A joint whose both bodies are inactive (including a joint to a
        // static anchor whose dynamic side sleeps) belongs to a sleeping
        // island and produces no rows.
        self.joint_rows.clear();
        for c in &mut self.constraints {
            if !self.bodies[c.body_a()].is_active() && !self.bodies[c.body_b()].is_active() {
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
        let mut contact_rows = std::mem::take(&mut self.contact_rows);
        let mut joint_rows = std::mem::take(&mut self.joint_rows);
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
        for c in &contact_rows {
            if c.point_index == u32::MAX {
                continue;
            }
            if let Some(point) = self
                .manifold_cache
                .get_mut(&(c.body_a, c.body_b))
                .and_then(|m| m.point_mut(c.point_index as usize))
            {
                point.normal_impulse = c.normal_impulse;
            }
        }
        if prune {
            self.manifold_cache.retain(|_, m| m.paired_at(stamp));
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
                    // Removed bodies are gone from every collision path,
                    // including the CCD sweeps (their proxy is destroyed).
                    if other.is_removed() {
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
                && !body.is_static()
                && (push_lin.x != 0.0 || push_lin.y != 0.0 || push_lin.z != 0.0)
            {
                body.position += push_lin * dt;
            }
            if body.is_active()
                && !body.is_static()
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
        let (soft_bodies, bodies) = (&mut self.soft_bodies, &mut self.bodies);
        for sb in soft_bodies {
            sb.solve_constraints(bodies);
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
    /// soft body's node tree is queried against the rigid body's
    /// margin-padded AABB and every contacting node lands in the soft
    /// body's `rcontacts` (solved at stage 7s). Detection uses the rigid
    /// body's *current* pose — the rigid solve has not run yet, matching
    /// Bullet's detection inside `performDiscreteCollisionDetection`.
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
        let (soft_bodies, bodies) = (&mut self.soft_bodies, &self.bodies);
        soft_bodies[soft].collide_rigid(rigid, &bodies[rigid], &mut self.soft_scratch);
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
        let mut uf = UnionFind::new(total);

        // Proxy-pair owners -> island nodes (None: not an island member).
        // A soft body is one node (Bullet unions the soft body's single
        // collision object the same way) and never sleeps, so any island
        // touching it stays awake.
        let node_of = |world: &Self, owner: ProxyOwner| match owner {
            ProxyOwner::Body(i) => world.bodies[i].participates_in_islands().then_some(i),
            ProxyOwner::Collider(e) => Some(num_bodies + world.collider_entries[e].mb),
            ProxyOwner::Soft(i) => Some(soft_base + i),
            ProxyOwner::Free => None,
        };
        for &(pi, pj) in pairs {
            if let (Some(a), Some(b)) = (
                node_of(self, self.proxy_owners[pi]),
                node_of(self, self.proxy_owners[pj]),
            ) {
                uf.unite(a, b);
            }
        }
        for c in &self.constraints {
            if !c.enabled() {
                continue;
            }
            let (a, b) = (c.body_a(), c.body_b());
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
                None => uf.unite(num_bodies + c.body_a(), num_bodies + c.body_b()),
                Some(rb) => {
                    if self.bodies[rb].participates_in_islands() {
                        uf.unite(num_bodies + c.body_a(), rb);
                    }
                }
            }
        }

        // Any awake member keeps its whole island awake (per island root).
        let mut island_awake = vec![false; total];
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

        // Resolve every member's island state first (the decisions borrow
        // the union-find while applying them borrows the world).
        let body_awake: Vec<bool> = (0..num_bodies)
            .map(|i| island_awake[uf.find(i)])
            .collect();
        let mb_awake: Vec<bool> = (0..num_mbs)
            .map(|i| island_awake[uf.find(num_bodies + i)])
            .collect();
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
                    Some(rb) => bodies[rb].is_active(),
                    None => multibodies[c.body_b()].is_awake(),
                };
                if !multibodies[c.body_a()].is_awake() && !side_b_awake {
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
                    Some(rb) => bodies[rb].is_active(),
                    None => multibodies[c.body_b()].is_awake(),
                };
                if !multibodies[c.body_a()].is_awake() && !side_b_awake {
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
        let cached: Vec<&bullet_collision::manifold::PersistentPoint> =
            world.manifold_cache.values().flat_map(|m| m.points()).collect();
        assert!(!cached.is_empty());
        assert!(cached.iter().any(|c| c.normal_impulse > 0.0));
        let sphere = world.body(1);
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
            .values()
            .flat_map(|m| m.points().to_vec())
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
        assert!(
            world.body(1).is_active(),
            "sleeping body was not woken by the impact"
        );
        assert_ne!(
            world.body(1).activation_state(),
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
        // y ∈ [−1.5, −0.5], padded by the 0.02 broad-phase contact margin.
        let entry = world
            .collider_entries
            .iter()
            .find(|e| e.mb == 0 && e.link == 0)
            .expect("collider registered");
        let proxy = entry.proxy.expect("proxy assigned");
        let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_relative_eq!(aabb.min.y, -1.52, epsilon = 1e-4);
        assert_relative_eq!(aabb.max.y, -0.48, epsilon = 1e-4);
        // Moving the base moves the collider with the next FK pass.
        world.multibody_mut(0).base_pos = Vector3::new(0.0, 5.0, 0.0);
        world.step_simulation(1.0 / 60.0);
        let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_relative_eq!(aabb.min.y, 3.48, epsilon = 1e-4);
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

    /// A fixed-base pendulum whose link sphere falls onto a static plane is
    /// supported by the multibody contact rows: it settles hanging straight
    /// down with the sphere resting on the plane (phase C contact response).
    #[test]
    fn link_sphere_rests_on_ground() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
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
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        // Pivot 1.51 m up: the link center's lowest arc point is y = 0.51,
        // so the sphere (r = 0.5) starts 0.01 m inside the plane — the
        // contact rows must support it hanging straight down.
        mb.base_pos = Vector3::new(0.0, 1.51, 0.0);
        world.add_multibody(mb);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            world.step_simulation(dt);
        }
        let body = world.multibody(0);
        let q = body.joint_pos_multidof(0)[0];
        let qd = body.joint_vel_multidof(0)[0];
        assert!(q.abs() < 0.05, "pendulum did not settle hanging down: q = {q}");
        assert!(qd.abs() < 0.1, "pendulum still swinging: qd = {qd}");
        // The link center rests with the sphere on the plane: x ≈ 0,
        // y ≈ 0.5 (within the slop/damping tolerance).
        let p = body.link_world_position(0);
        assert_relative_eq!(p.x, 0.0, epsilon = 0.02);
        assert_relative_eq!(p.y, 0.5, epsilon = 0.05);
    }

    /// A swinging pendulum link kicks a dynamic rigid box lying in its path:
    /// the multibody contact rows exchange impulse with the rigid solver
    /// (box wakes and moves, the pendulum keeps swinging reduced).
    #[test]
    fn swinging_link_pushes_rigid_box() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
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
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            0.3,
            0.0,
        );
        // Pivot 1 m up; release from the horizontal (+X side).
        mb.base_pos = Vector3::new(0.0, 1.0, 0.0);
        mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
        world.add_multibody(mb);
        // A box resting on the plane right of the pivot, in the swing path
        // (the link center sweeps through y ≈ 0 at x ≈ 1).
        let box_start = Vector3::new(1.0, 0.5, 0.0);
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
                box_start,
                1.0,
            )
            .friction(0.0),
        );
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            world.step_simulation(dt);
        }
        // The box was hit and knocked away from its start pose. At the
        // impact point (≈45° along the arc) the ball's velocity points
        // (−X, −Y), so the box is shoved toward −X / tipped over. The exact
        // slide direction is chaotic though: the ball pins the box into the
        // plane and the frictionless box drifts with the contact details
        // (the persistent-manifold warm start shifts the trajectory within
        // this noise), so the direction check carries a small tolerance.
        let box_end = world.body(1).position;
        assert!(
            (box_end - box_start).length() > 0.1,
            "box was not pushed: {box_end:?}"
        );
        assert!(
            box_end.x < box_start.x + 0.05,
            "box pushed the wrong way: {box_end:?}"
        );
    }

    /// A multibody contact wakes a sleeping rigid body: the sleeping box is
    /// struck by the swinging link and moves.
    #[test]
    fn link_contact_wakes_sleeping_rigid_body() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // The victim: a box that settles on the plane and sleeps.
        let box_idx = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            Vector3::new(1.0, 0.5, 0.0),
            1.0,
        ));
        for _ in 0..180 {
            // 3 s > the 2 s deactivation window.
            world.step_simulation(1.0 / 60.0);
        }
        assert!(!world.body(box_idx).is_active(), "box should be asleep");
        // Swing a link sphere into it.
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
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            0.3,
            0.0,
        );
        mb.base_pos = Vector3::new(0.0, 1.0, 0.0);
        mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
        world.add_multibody(mb);
        for _ in 0..90 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(world.body(box_idx).is_active(), "box should have been woken");
        assert!(
            (world.body(box_idx).position - Vector3::new(1.0, 0.5, 0.0)).length() > 0.1,
            "sleeping box was not knocked away"
        );
    }

    /// A free-base multibody with a box collider falls onto a static plane
    /// and is supported by its base contact rows (rest height, zero
    /// velocity).
    #[test]
    fn free_base_with_collider_lands_on_ground() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        mb.set_linear_damping(0.0);
        mb.finalize_multidof();
        mb.set_base_collider(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            0.5,
            0.0,
        );
        mb.base_pos = Vector3::new(0.0, 3.0, 0.0);
        world.add_multibody(mb);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            world.step_simulation(dt);
        }
        let body = world.multibody(0);
        // The half-extent box rests on the plane: y ≈ 0.5 within the slop
        // tolerance, with a small residual velocity (damped settling).
        assert_relative_eq!(body.base_pos().y, 0.5, epsilon = 0.05);
        assert!(body.base_vel().length() < 0.2, "base still moving: {:?}", body.base_vel());
    }

    /// A fast rigid sphere hitting a free multibody (equal masses, head-on,
    /// restitution 0, gravity off) converges to the shared inelastic
    /// velocity: the mixed-row iteration must count the assembly-time
    /// relative velocity only once — reading the rigid side's impulse delta,
    /// not its full velocity. The double-count regression inflated the
    /// impulse ~2.4x, leaving the ball nearly stopped and the multibody
    /// overspeeding.
    #[test]
    fn rb_mb_contact_impulse_not_doubled() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        mb.set_linear_damping(0.0);
        mb.set_angular_damping(0.0);
        mb.finalize_multidof();
        mb.set_base_collider(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 3.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -6.0, 0.0))
            .disable_deactivation(),
        );
        let dt = 1.0 / 60.0;
        for _ in 0..25 {
            world.step_simulation(dt);
        }
        let ball_vy = world.body(ball).linear_velocity.y;
        let mb_vy = world.multibody(0).base_vel().y;
        assert!(mb_vy < -1.5, "multibody was not kicked: {mb_vy}");
        // Equal-mass head-on, restitution 0: both sides leave at ~-3 m/s
        // (plus a small Baumgarte bounce that separates them slightly).
        assert!(
            (ball_vy - mb_vy).abs() < 1.5,
            "velocities did not converge: ball {ball_vy} vs multibody {mb_vy}"
        );
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
        assert!(!world.body(1).is_active(), "left box should sleep");
        assert!(!world.body(2).is_active(), "right box should sleep");
        // Sleeping means frozen: further steps do not move either box.
        let (pa, pb) = (world.body(1).position, world.body(2).position);
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert!((world.body(1).position - pa).length() < 1e-9);
        assert!((world.body(2).position - pb).length() < 1e-9);
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
        assert!(world.body(1).is_active(), "left box should wake with its island");
        assert!(world.body(2).is_active(), "right box should wake with its island");
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
        assert!(world.body(1).is_active());
        assert!(
            world.body(2).is_active(),
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

    // ---- body removal -----------------------------------------------

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
        let ground_y = world.bodies()[ground].position.y;
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
        assert_eq!(world.bodies()[ground].position.y, ground_y);
        assert!(world.bodies()[ground].is_removed());
        assert!(!world.bodies()[ground].is_active());
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
        let freed = world.body_proxies[first];
        world.remove_body(first);
        // The new body reuses the freed proxy slot and gets a fresh body slot.
        let third = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(1.0, 2.0, 0.0),
        ));
        assert_eq!(third, 2);
        assert_eq!(world.body_proxies[third], freed);
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
        assert_eq!(ground, 0);
        assert!(world.bodies()[a].is_removed());
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
        world.add_body(RigidBody::new_static(
            ground,
            Vector3::new(-4.0, 0.0, -4.0),
        ));
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
}
