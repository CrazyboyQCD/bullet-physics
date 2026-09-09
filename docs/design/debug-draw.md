# Debug Draw Design

> Design specification for the minimal debug-draw API (`docs/api-gaps.md` P3,
> the last feature item before the 1.0 API freeze).
>
> Status: **IMPLEMENTED** (design fixed at `10ac469`; implementation landed
> 2026-09-09). This document strictly separates **current** (verified
> against the source), **decision** (what this document fixes),
> **implementation** (work the implementer must do), and **deferred**
> (deliberately out of 1.0).
>
> One-sentence summary: **debug draw is a read-only diagnostic output
> interface of `PhysicsWorld` — not a renderer framework, and not a Rust
> port of Bullet's `btIDebugDraw`.**

## 1. Goals

1. **Observe** the current state of a `PhysicsWorld`: poses, shapes,
   broad-phase bounds, contacts, constraint anchors, activation states.
2. **Diagnose** collision / solver / broad-phase / sleeping problems by
   seeing what the engine actually computed — not a re-derivation.
3. Stay **decoupled from rendering**: the physics core never knows about
   renderers, windows, GPUs, or colors.
4. **Never change the simulation**: drawing must be a pure observation
   (formalized as the invariance contract in §19).
5. Expose a **small, stable, Rust-native API** that can survive the 1.0
   freeze and grow by addition.
6. Be **directly consumable by test code** (headless collection and
   assertions), not only by graphics applications.

## 2. Non-Goals

1.0 does **not** build, and the core does **not** grow:

- a renderer, GPU abstraction, wgpu/OpenGL/Vulkan layer;
- a Bevy plugin, egui integration, three.js integration;
- an editor UI;
- 3D text / labels / measurement annotations;
- a material, shader, or color system in the physics core;
- a persistent debug cache or a second, parallel copy of simulation state;
- a "physics → renderer framework" (no backend traits, no drawer objects,
  no scene graphs owned by the engine).

Rendering is entirely the consumer's job. The engine's role ends at
handing over facts.

## 3. Design Principles

1. **Observation, not simulation.** Debug draw is a read-only query API.
   It consumes the engine's authoritative state; it never computes new
   physics results and never mutates anything.
2. **Authoritative state only.** Every emitted value must be readable
   from existing world state today (`manifold_cache`, broad-phase proxy
   AABBs, body poses/activation, constraint parameters). If a fact is not
   persistent authoritative state (CCD sweeps, island membership, mb/soft
   contacts), the API does not fabricate it — see §17/§18.
3. **Physics facts, not render operations.** A primitive states *what is
   true* ("contact at P with normal N, penetration d"), never *how to
   draw* ("red, 2 px, triangle fan").
4. **No renderer attributes in the core.** No RGB, alpha, line width,
   materials, or shaders in primitives. The primitive's *kind* (its
   variant) is the only style hook; the renderer maps kinds to styles.
5. **Borrow, don't copy.** Large data (mesh shapes) is lent by reference
   for the duration of the callback; nothing is cloned to feed a draw
   call.
6. **No unnecessary persistent allocation.** The world gains no new
   fields for debug draw; the emit path allocates nothing.
7. **Push callback, `&self` only.** One method, one closure, shared
   borrow (§9).

## 4. Architecture

```
PhysicsWorld  (owns all simulation state)
    │
    │  read-only debug query  (&self, no mutation, no new state)
    ▼
DebugPrim values  (physics facts, kind-tagged, borrowed data)
    │
    ├── renderer        (caller-owned; immediate-mode or buffered)
    ├── test / recorder (headless collection + assertions)
    ├── editor / tooling
    └── any other consumer
```

Invariants this architecture enforces:

- **Physics core ≠ renderer.** The dependency arrow points from consumer
  into the crate; the crate has zero reverse dependencies.
- **Debug draw ≠ a second physics state.** No parallel debug structures
  live in `PhysicsWorld`. Every prim is derived, on demand, from the
  same state the step just used.
- **One-way borrows.** The consumer borrows the world (`&PhysicsWorld`);
  the world never borrows, registers, or calls a renderer.

## 5. Debug Domains

The 1.0 domain set, mirroring the diagnostic coverage Bullet's `DBG_*`
flags are used for, organized like Avian's per-domain toggles:

| Domain | Flag | Source of truth (current) |
|---|---|---|
| Shapes (rigid bodies) | `SHAPES` | `RigidBody.shape / position / orientation` |
| AABBs (all proxies) | `AABBS` | `BroadPhase::proxy_aabb` |
| Contacts (rigid pairs, incl. normals) | `CONTACTS` | `manifold_cache` → `PersistentManifold::points()` |
| Constraints (rigid joints, anchors) | `CONSTRAINTS` | `Constraint::kind()` + per-kind parameters |
| Sleeping / activation | `SLEEPING` | `RigidBody::activation_state()` |

Explicitly **not** 1.0 domains (each with its reason in §22):

- **Island membership** — the union-find is a per-step scratch structure
  (`IslandScratch`, reset to singletons every step); it does not survive
  a step, so there is nothing authoritative to read afterwards (§17).
- **CCD** — the sweep exists only inside the integrate-transforms stage;
  no CCD state persists in the world (§18).
- **Multibody / soft-body contact points** — those contacts are per-step
  transients (`mb_pending_contacts`, the soft dispatch), not persistent
  state (§15).
- **Soft-body element drawing** (nodes/links/faces) — element-scale
  output that needs its own design (§23).
- **Vehicle wheel rays** — vehicles are caller-owned; the ray data is
  already public on `WheelInfo`, so this stays a consumer-side recipe
  (§13, §22).

## 6. `DebugDrawFlags`

**Decision.** A hand-written bitmask type in the exact style of the
existing `CollisionFlags` (no new dependency — the workspace has a
standing no-new-crates discipline):

```rust
/// Categories of debug output (`debug_primitives` emits one stream of
/// primitives; each flag enables one domain).
pub struct DebugDrawFlags(u16);

impl DebugDrawFlags {
    pub const NONE:       Self; // empty — emit nothing
    pub const SHAPES:     Self; // rigid-body collider geometry at world poses
    pub const AABBS:      Self; // stored broad-phase proxy AABBs
    pub const CONTACTS:   Self; // persistent-manifold contact points
    pub const CONSTRAINTS: Self; // rigid-joint world anchors
    pub const SLEEPING:   Self; // per-body activation states
    pub const ALL:        Self; // every domain
}
```

Semantics per flag:

| Flag | Emits | Does **not** emit |
|---|---|---|
| `SHAPES` | one `Shape` prim per live rigid body (static, kinematic, sleeping included) | compound children flattened; mb link colliders; soft bodies |
| `AABBS` | one `Aabb` prim per live broad-phase proxy (rigid, mb link collider, soft body) | Dbvt internal fat leaf volumes; swept/CCD bounds; exact tight shape bounds |
| `CONTACTS` | one `Contact` prim per persistent-manifold point of every cached rigid pair (normal and load included) | solver rows; mb/soft contacts; suppressed (constraint-linked) pairs — they hold no manifolds |
| `CONSTRAINTS` | one `Constraint` prim per rigid joint in `world.constraints` (both world anchors + kind tag) | axis/frame/limit geometry; multibody constraints; soft joints |
| `SLEEPING` | one `Activation` prim per live rigid body carrying its `ActivationState` | island membership; deactivation timers (`WantsDeactivation` progress) |

**Empty-flags semantics (decision):** `DebugDrawFlags::NONE` (or any
empty mask) emits **zero primitives** — and the implementation must
early-out before scanning anything, so a quiet mask costs nothing
(per-domain gating is a listed risk in `api-gaps.md` P3).

Flag bits are an implementation detail; bit values are not part of the
stable surface (§24).

## 7. `DebugPrim`

**Decision.** A small, `Copy`, kind-tagged value enum whose variants
*are* the kinds (there is no separate `DebugKind` enum — the variant is
the classification a renderer styles by):

```rust
/// One debug primitive. Valid only for the duration of the callback
/// invocation (the `Shape` variant borrows the body's shape).
/// `non_exhaustive`: future domains (CCD, islands, soft elements) may add
/// variants post-1.0 without breaking exhaustive `match`es.
#[non_exhaustive]
pub enum DebugPrim<'a> {
    /// SHAPES — a rigid body's collider geometry at its world pose.
    Shape {
        object: CollisionObject,
        position: Vector3,
        orientation: Quaternion,
        shape: &'a CollisionShape,
    },
    /// AABBS — the stored broad-phase AABB of one live proxy.
    Aabb {
        object: CollisionObject,
        aabb: Aabb,
    },
    /// CONTACTS — one persistent-manifold contact point.
    Contact {
        a: CollisionObject,
        b: CollisionObject,
        position: Vector3,      // B-side projected anchor (see §15)
        normal: Vector3,        // unit, A -> B
        penetration: Real,      // > 0 = penetrating
        normal_impulse: Real,   // last solve's accumulated normal impulse
    },
    /// CONSTRAINTS — one rigid joint, both anchors in world space.
    Constraint {
        a: BodyHandle,
        b: BodyHandle,
        kind: ConstraintTag,    // payload-free mirror of ConstraintKind
        anchor_a: Vector3,
        anchor_b: Vector3,
    },
    /// SLEEPING — a rigid body's activation state at its origin.
    Activation {
        object: BodyHandle,
        position: Vector3,
        state: ActivationState,
    },
}

/// Which rigid-joint kind a `Constraint` prim describes (Copy mirror of
/// `ConstraintKind` without its parameter payloads).
pub enum ConstraintTag { Point2Point, Hinge, Fixed, ConeTwist, Slider, Generic6Dof }
```

Design notes:

- **`variant = kind` (decision).** A separate `DebugKind` alongside the
  variants would create two parallel classifications that must be kept
  in sync. The renderer matches the variant and applies its own style.
- **Borrowed shape (decision).** `CollisionShape` is a value enum and
  `Clone`, but cloning it per prim would deep-copy `TriangleMesh` BVHs
  every frame. The prim therefore *borrows* the body's shape; the
  lifetime `'a` ties the prim to the `&PhysicsWorld` borrow, and a
  renderer that needs ownership clones explicitly (visible cost).
- **`CollisionObject` for pair/proxy identity (decision).** Rigid
  contacts and proxies are typed with the existing `CollisionObject`
  enum (`Body` / `Link` / `Soft`) so that later adding mb-link or
  soft-body entries does not have to re-shape these variants. Rigid-only
  data (constraint endpoints, activation) uses `BodyHandle` honestly.
- **Candidate primitives rejected (decision):**
  - *`Point`* — subsumed by `Contact::position` and `Activation::position`;
    there is no standalone debug-point state to report.
  - *`Line`* — a line is a render operation, not a physics fact (the one
    exception, wheel rays, is caller-side data; §13). Normal lines are
    drawn by the renderer from `Contact::position + normal`.
  - *`Shape` as tessellated line soup* — rejected; see §13.

## 8. Primitive Semantics

**Answer to the central question:** a primitive expresses *what physical
fact is true*, not *how to draw*. Concretely:

| Variant | Physical fact stated | Derived from (current source) |
|---|---|---|
| `Shape` | this body's collider geometry sits at this world pose | `RigidBody` fields (`body()`) |
| `Aabb` | the broad phase currently stores these bounds for this proxy — i.e. these bounds drive pair candidacy | `proxy_owners` + `BroadPhase::proxy_aabb` |
| `Contact` | the solver's cached manifold holds this point, with this normal, depth, and last-solve load | `manifold_cache` → `PersistentManifold::points()` |
| `Constraint` | this joint's anchors project to these world positions right now | `Constraint::kind()` payloads + body poses |
| `Activation` | this body is in this activation state | `RigidBody::activation_state()` |

Rules every variant obeys:

1. No renderer attributes (§3.4).
2. Values are read-only views; the consumer may copy anything it wants
   to keep.
3. Numerical values use the crate's scalar aliases — `Real`, `Vector3`,
   `Quaternion`, `Aabb` — so precision follows the build (§20).
4. No ordering guarantee across or within domains; consumers must not
   depend on emission order (tests count/filter, never index).

## 9. `PhysicsWorld` API

**Decision — a push callback, not an iterator:**

```rust
impl PhysicsWorld {
    /// Emit debug primitives for every enabled domain, in no particular
    /// order. Read-only: drawing never affects simulation. Empty flags
    /// return immediately.
    pub fn debug_primitives(
        &self,
        flags: DebugDrawFlags,
        emit: &mut impl FnMut(DebugPrim<'_>),
    );
}
```

Callback vs iterator analysis:

| Axis | Callback (chosen) | Iterator |
|---|---|---|
| Borrow | `&self` for the call; prims live for the invocation | same, plus a stateful lazy machine holding `&self` |
| Allocation | none — no boxing, no buffering | a single lazy iterator over five heterogeneous domains requires either `Box<dyn Iterator>` (a `dyn` on a project that bans it on principle) or a deeply nested enum state machine (high complexity, no gain) |
| Ergonomics | immediate-mode renderers consume in one closure | combinators are marginally nicer, but an `if` inside the callback covers filtering |
| Testability | collect into a `Vec` in three lines | equal |
| Nested world access | the closure may capture `&PhysicsWorld` (shared borrow) and call read methods such as `body(handle)` | equal |

The callback shape also matches the sketches already recorded in
`api-gaps.md` P3 / `api-usability.md` §2 P3 (`world.debug_primitives(mask,
&mut impl FnMut(DebugPrim))`), so the design closes on a direction the
project has already half-committed to.

**Timing semantics (decision):** the method may be called at any point
between steps; each domain reflects the world's current authoritative
state at call time:

- `SHAPES` / `CONSTRAINTS` — direct field reads; always current.
- `AABBS` — the stored proxy bounds, i.e. what the last broad-phase
  update (or an immediate refresh such as `set_child_transform`) left
  behind; exactly the bounds pair candidacy uses.
- `CONTACTS` — the persistent manifold cache as the last completed step
  left it: points refreshed/re-detected and normal impulses written back
  by the solver. The cache persists across steps (entries whose
  broad-phase pair vanished are dropped at step end), so inter-step reads
  are well-defined.
- `SLEEPING` — the activation states the last deactivation check left.

**Re-entrancy:** the callback may call other `&self` world methods
(e.g. `body(handle)` to enrich a display); `&mut` methods are
compile-impossible. No guarantee is made about behavior of `step_simulation`
from inside the callback — do not do it.

## 10. Ownership and Borrowing

- **The caller owns the renderer.** The engine owns the physics state.
  `DebugPrim` contains only temporary, read-only information borrowed
  from the world.
- The callback lifetime is confined to the `debug_primitives` call. It
  cannot produce a long-lived borrow of the world, cannot hold `&mut
  PhysicsWorld`, and therefore cannot create aliasing.
- **No global drawer.** There is no `set_debug_drawer`, no registered
  trait object, no world→renderer edge, nothing to clean up. Two
  renderers = call the method twice (or fan out inside one closure).
- Prims that borrow (`Shape`) are valid only inside the callback; keeping
  the data means copying it, and keeping a *shape* means an explicit,
  visible `CollisionShape` clone by the consumer.

## 11. Allocation Model

- The world produces **no** `Vec<DebugPrim>`; the emit path is
  field reads plus closure invocations — zero allocation, zero new world
  fields.
- The domain gates run before any traversal, so an empty or quiet mask
  scans nothing (§6).
- If the implementation ever needs scratch (not currently foreseen), it
  must be a world-owned pooled buffer like the existing step scratch —
  never a per-call heap allocation, and never a persistent debug cache.
- The design goal is "**no unnecessary persistent allocation**", not
  zero-allocation dogma: the consumer's own buffering is its own choice
  (immediate-mode renderers need none).

## 12. Renderer Integration

The consumer side is deliberately out of scope; the design only fixes
the seams:

- **Immediate-mode renderers** (Bevy `Gizmos` is the canonical case from
  `api-usability.md` §2 P3): draw directly inside the callback; nothing
  is stored.
- **Buffered renderers**: copy the prims (or a projected form) into
  their own storage inside the callback; the world never sees the buffer.
- **Tests / recorders**: collect into a `Vec<DebugPrim<'_>>` and assert —
  this is how the API's own regression tests work (§19), and it proves
  the API is consumable without any graphics stack.
- Styling (colors, widths, visibility of normals, labels) maps from the
  primitive variant in renderer-owned code. The intended mapping is
  one-to-one and trivial: variant → style table entry.

## 13. Shape Visualization

Two candidate models were evaluated:

**A. Physics emits tessellated line segments** (`Shape → [Line]`).
Rejected: tessellation density is a rendering decision (wireframe vs
solid vs gizmo vs decimated); putting it in the core would grow physics
code that only serves drawing, and would allocate per call. Bullet's
debug path effectively does this (shapes draw themselves through
`btIDebugDraw`), and it is one of the reasons its debug rendering is
entangled with the collision shapes.

**B. Physics emits shape + transform (chosen).** The prim names the
`CollisionShape` and the world pose; the renderer decides wireframe,
solid debug mesh, or gizmo. This fits the project's shape architecture
exactly: `CollisionShape` is a value enum whose parameters are plain
data (`shape: CollisionShape` is a public `RigidBody` field), so the
renderer can read geometry directly from the borrowed value — mesh
vertices, hull points, compound children included.

Ownership / lifetime handling under B:

- `DebugPrim::Shape` borrows `&CollisionShape` from the world (`§7`);
  no clone per prim, no `Arc`, no shape registry, no `dyn`.
- The alternative — emitting only a handle and forcing consumers to call
  `world.body(handle).shape` themselves — was rejected: it makes the
  prim not self-contained and pushes every renderer into world-coupled
  code for the most common domain.

Scope notes (decisions):

- **Compound shapes** are emitted as their top-level value. Walking
  children is the renderer's option; a read-only children iterator may
  be added by the implementation if missing (the `Compound` mutators and
  `child_mut` already exist, so this is a small, contained read path) —
  flattened-children *prims* are deferred (§23).
- **Contact margin** (the fixed box margin, the GJK margin band) is not
  visualized by 1.0 prims; margin-shell drawing is deferred.
- **Vehicle wheel rays** are *not* producer-side: `RaycastVehicle` is
  caller-owned (`update_vehicle(&mut self, world, dt)`), so the world
  cannot enumerate them, and `WheelInfo` already exposes every constant
  (`chassis_connection_point_cs`, `wheel_direction_cs`,
  `suspension_rest_length`) plus per-step results (`raycast_info`,
  `world_origin`) needed to reconstruct the rays in three lines of
  consumer code. The original P3 sketch listed wheel rays because
  Bullet's demo renderer draws them; here they belong to the caller, and
  the design keeps it that way.

## 14. AABB Visualization

Three AABB notions exist around the pipeline; the design pins exactly
one of them:

1. **Stored proxy AABB** (chosen) — the bounds each proxy currently
   stores in the broad phase, read through `BroadPhase::proxy_aabb`
   (present for both backends; the values include the global
   contact-breaking-threshold padding the world applies). This is the
   *authoritative* bound: it is literally what pair candidacy runs on,
   so drawing it shows the broad phase's real belief, including for
   sleeping/static bodies whose bounds persist.
2. **Fat AABB** (Dbvt leaf volume) — an internal tree structure; not
   exposed by `proxy_aabb` and deliberately not surfaced. Drawing it
   would leak backend internals into a stable API.
3. **Swept AABB** (CCD) — does not exist as persistent state at all
   (§18); nothing to draw.

Decision: **only the stored proxy AABB** is in 1.0, one prim per live
proxy across all three owner kinds (rigid, mb link collider, soft body)
with its `CollisionObject` identity. No new state is added to remember
bounds "for drawing"; if it is not in the broad phase, it is not drawn.

## 15. Contact Visualization

**Source (current):** the per-pair persistent-manifold cache
(`manifold_cache`) — the same data the solver warm-starts from. Debug
draw shows *what the solver actually used*, per the project's core rule;
it never re-runs the narrow phase and never recomputes normals.

Per cached point (`PersistentPoint`, all fields public, read via the
existing `points()` accessor), one `Contact` prim:

| Field | Semantics (decision, pinned to current conventions) |
|---|---|
| `a`, `b` | the two sides of the cached pair, as `CollisionObject::Body`; the prim reports the same A/B the manifold uses, consistent with its normal direction |
| `position` | the B-side projected anchor: `wb = pos_b + ori_b * local_b` at the body's current pose — the same quantity the manifold refresh path hands the solver as the contact position (`ContactPoint::new(wb, ...)`), so draw and solve can never disagree. (Implementation correction 2026-09-09: the design draft said the A-side anchor `wa`; reading the refresh source showed the solver's contact position is the B-side anchor, so the design follows the solver.) |
| `normal` | the stored unit normal, **A → B** (this project's manifold convention; the sign flip of Bullet's `m_normalWorldOnB`, which points B → A) |
| `penetration` | stored signed separation, **positive = penetrating** (the sign flip of Bullet's `m_distance1`); values in `(−threshold, 0]` are margin-band contacts |
| `normal_impulse` | the accumulated normal impulse from the last solve (warm-start value) — the load signal for diagnostics |

**Contact normals are merged into the `Contact` prim** (decision): the
normal is an intrinsic fact of the point. Bullet's `DBG_DrawContactPoints`
and `DBG_DrawNormals` are separate only because its interface is a
sequence of draw calls; splitting them here would either strip the field
(lossy prim) or emit a second positional prim per point (a render
operation in disguise). Renderers that don't want normal arrows simply
don't draw them. Consequence for the 1.0 scope table: "Contact normals"
= merged into `CONTACTS`, no separate flag.

**Coverage boundary (decision):** rigid-rigid pairs only. Multibody
link-collider contacts and soft-body contacts are collected into
per-step transients (`mb_pending_contacts`, the soft dispatch) and do
not persist past the step; exposing them would require new persistent
state, which this design refuses to add for drawing (§3.2). Deferred.
Pairs suppressed by constraints hold no manifolds and therefore emit
nothing.

## 16. Constraint Visualization

**Current access:** `Constraint::kind() -> &ConstraintKind` (six
variants carrying their full parameter payloads), per-kind read
accessors, and `body_a()`/`body_b()` — all public and read-only. The
parameters (e.g. `Point2PointConstraint { pivot_a, pivot_b }`) are
expressed in the bodies' center-of-mass frames; the world can project
them with the current poses without touching any solver machinery.

**1.0 scope (decision) — anchors only:** one `Constraint` prim per rigid
joint, carrying both world-space anchors plus a `ConstraintTag`. Per
kind, the anchor is the joint's pivot/frame origin projected by its
body's pose (p2p/hinge/fixed/cone-twist pivots; slider and 6-DOF frame
origins). This is a read-only `match` over the kind — no solver rows,
no internals need to change.

**Not in 1.0 (decision):** per-kind axis/frame/limit/motor geometry
(hinge axes, cone ellipses, 6-DOF limit windows). That is six kinds ×
bespoke geometry with limit-state coupling — real implementation mass
for a visualization nicety, deferred. Also not drawn: solver row graphs,
constraint force/impulse heatmaps (the per-point normal impulse on the
`Contact` prim is the 1.0 substitute for load questions).

**Out of 1.0:** multibody constraints and soft-body joints (different
frame semantics; deferred with the other mb/soft debug surfaces).

## 17. Island / Sleeping Visualization

**Sleeping / activation — YES (decision).** `ActivationState`
(`Active` / `IslandSleeping` / `WantsDeactivation` / `DisableDeactivation`
/ `DisableSimulation`) is per-body persistent authoritative state,
readable through `activation_state()`; the `SLEEPING` domain emits one
`Activation` prim per live rigid body. Seeing *which* bodies are asleep
(and static/kinematic, via their states) answers the everyday
"why isn't this stack moving?" question without any island graph.

**Island membership — DEFERRED (decision, source-confirmed).** The
union-find lives in the pooled `IslandScratch` and is `reset` to
singletons every step; island membership is a property of the step in
progress and does not survive it. There is therefore no authoritative
post-step fact to read today. The alternatives were rejected:

- persisting the island graph for drawing = a parallel debug state
  structure in the world (violates §3.2 and the memory discipline);
- recomputing islands on query = re-running simulation logic outside the
  step (risks divergence from what the step actually did).

Island visualization returns only if a concrete need survives both
objections — e.g. as a step-time opt-in sink, which is the same shape as
the deferred CCD sink (§18).

## 18. CCD Visualization

**Current fact:** the engine has no persistent CCD state. The
swept-sphere clamping is computed locally inside the integrate-transforms
stage (start/end poses and hit fraction are stack-local), and only the
clamped position survives into the body state. This was confirmed
structurally: `PhysicsWorld` carries no CCD fields.

**Decision: CCD visualization is DEFERRED.** Putting transient sweep
data (sweep start, sweep end, TOI, target) into the world as persistent
state would create exactly the "second physics state" this design
forbids, and would tax every step for a diagnostic that is rarely on.
The future options, if demand appears, are:

1. an opt-in per-step sink (a closure installed for the CCD stage,
   similar in spirit to the motion-state sync point) — output-side only,
   no state retained; or
2. an offline re-enactment query (`debug_ccd_sweep(body) -> Option<…>`)
   recomputing the sweep on demand — deterministic by the project's
   bit-exactness guarantees.

Neither is designed further now; 1.0 ships without CCD prims.

## 19. Simulation Invariance

**Formal contract (1.0, test-pinned):**

> Given identical initial state, `dt`, and user inputs — compared
> **within one build (same precision feature) and one world
> configuration (same broad-phase backend)** — any sequence of
> `debug_primitives` calls, with any flags at any frequency and timing,
> leaves the simulation bit-identical to a run with no calls: positions,
> rotations, velocities, activation states, collision events,
> manifold-cache contents, and all subsequent step outputs.

**Scope of the bit-identical claim (decision).** This contract compares
*draw vs no-draw* inside the same binary and the same world
configuration. It deliberately makes **no** cross-configuration claim:
broad-phase backend cross-consistency is a different property, verified
at tolerance by the integrated adversarial pass
(`09-integrated-adversarial.md` §11 — identical event counts and
activation states, final poses within 1e-3), and f32/f64 are separate
numerical universes by design. The debug API adds no claims on either.

Why this holds by construction (and stays holding): the API takes
`&PhysicsWorld` only; the world has no interior mutability on the paths
involved; the emit path performs no writes, allocates nothing, and adds
no fields. The contract is nevertheless written down and *tested*
because it is the property that makes the feature safe to leave enabled
in a shipping build.

**Implementation-required regression tests:**

1. **Draw vs no-draw:** two worlds built from the same scene and the
   same configuration (same broad-phase backend, same build); A draws
   with `ALL` after every step, B never draws; after N steps assert
   bit-identical pose/velocity/activation snapshots and identical event
   streams (both f32 and f64, run as separate builds).
2. **Empty flags:** `NONE` (and `0`) invokes the callback zero times.
3. **Per-domain content:** AABBS equal `proxy_aabb` values; CONTACTS
   count equals cached `PersistentPoint`s; normals unit-length and
   consistent with A→B identity; CONSTRAINTS anchors match manual
   projection from `kind()` payloads; SLEEPING states equal
   `activation_state()`.
4. **Borrow honesty:** a `Shape` prim's shape pointer is the body's own
   shape (`ptr::eq`) — proving no per-prim cloning.

## 20. f32 / f64 / no_std

- **Precision (decision):** every numeric field of `DebugPrim` uses the
  crate's precision aliases (`Real`, `Vector3`, `Quaternion`, `Aabb`),
  exactly like the rest of the public surface. A world built with
  `--features f64` emits f64 primitives; there is no widening, no
  narrowing, and no separate f32 debug type. Debug values must not lose
  the precision the solver worked at — a mixed-precision debug API would
  invite exactly the wrong kind of "it looks fine in the visualizer"
  bugs.
- **no_std (decision):** the API uses only `core`/`alloc` types (the
  prim set, the flags, the closure). No `String`, no formatting, no
  clock, no OS, no renderer types. The feature must compile under
  `cargo check --no-default-features --features libm` like the rest of
  the workspace, and the existing verification gates cover it (§33
  checklist of HANDOFF).

## 21. Bullet / Rapier / Avian Design Comparison

| Aspect | Bullet (`btIDebugDraw`) | Rapier (debug render) | Avian (debug render) | This design |
|---|---|---|---|---|
| Core knows the renderer? | yes — the world/shape code draws through a drawer interface the user installs | no — the engine-side pipeline renders into a caller-provided backend trait | no — an ECS plugin reads world state and draws gizmos | no — the world pushes facts through one callback |
| Direction of data | pull-by-shapes (shapes draw themselves) + push lines | engine → backend trait | world → plugin systems | world → consumer closure |
| Domain toggles | `DBG_*` bit flags | per-kind render options | per-domain config struct | `DebugDrawFlags` bitmask |
| Geometry ownership | physics code tessellates | backend draws segments | Bevy gizmos | core emits facts; renderer tessellates |
| Style/colors | drawer-side (`getDebugColor`) | options | config | renderer-side; variant = kind |

**What is borrowed (principles only, not APIs):**

- **From Bullet** — the *diagnostic coverage* checklist: shapes, AABBs,
  contact points, normals, constraints, sleeping states. This is the
  de-facto minimum a physics debugger must show, and our domain set is
  exactly that list. Bullet's *mechanism* (a drawer interface inside the
  core, shapes that render themselves) is rejected (§11 below).
- **From Rapier** — the responsibility split: *the engine outputs debug
  geometry/data; an external system decides how to render it.* The
  engine stays renderer-free; the consumer stays engine-free. Rapier's
  backend *trait* is not adopted — a single `FnMut` closure is the
  lighter Rust-native equivalent for 1.0 (no object registration, no
  per-prim virtual dispatch, immediate-mode friendly); a stateful
  consumer can still buffer inside its closure.
- **From Avian** — the *debug-domain organization*: visualization
  grouped into per-domain categories with independent toggles
  (Shapes / AABBs / Contacts / Constraints / Sleeping), which maps onto
  the flags type. Avian's Component/Resource/Plugin machinery is not
  applicable — this project is not an ECS and has no plugin system.

**Why this is not a Rust port of Bullet's `btIDebugDraw`:**

1. Ownership models differ: `btIDebugDraw` is a raw pointer the world
   stores and calls; here the world stores nothing and the borrow
   checker enforces the direction.
2. The renderer belongs to the caller, not to a global registry.
3. A push callback (or iterator) is the Rust-native shape; C++
   interface objects are not.
4. The physics core should not know renderers exist — not even their
   interface.
5. The 1.0 freeze demands a minimal stable surface: one method, one
   flags type, one prim enum — not an interface hierarchy.

## 22. 1.0 Scope

| Domain | 1.0 | Reason |
|---|---|---|
| Shapes — rigid bodies | **YES** | authoritative pose + value-enum shape readable today via `body()`; borrowed, no tessellation in core |
| Shapes — multibody link colliders | NO (deferred) | read paths exist (`MultiBody::link_collider`) but pose-snapshot timing and thin need; `AABBS` already bounds them |
| Shapes — soft bodies | NO (deferred) | element-scale output (nodes/links/faces) needs its own design |
| AABBs — all proxies | **YES** | `BroadPhase::proxy_aabb` is authoritative for rigid + mb-link + soft proxies in one index space |
| Contacts — rigid pairs | **YES** | the persistent manifold cache is the solver's actual input; `points()` accessor already public |
| Contacts — mb / soft | NO (deferred) | per-step transients; persisting them = parallel debug state |
| Contact normals | **MERGED into `CONTACTS`** | intrinsic field of the contact prim, not a separate stream (§15) |
| Constraints — rigid joints (anchors + kind) | **YES** | `kind()` + public per-kind parameters; read-only projection |
| Constraint axes / frames / limits | NO (deferred) | 6 kinds × bespoke geometry, limit-state coupling; low value per line of engine code |
| Islands — membership | NO (deferred) | per-step scratch (`IslandScratch` reset every step); nothing authoritative survives the step |
| Sleeping / activation | **YES** | per-body `activation_state()` is persistent and authoritative |
| CCD | NO (deferred) | no persistent CCD state exists; sweep is stage-local |
| Vehicle wheel rays | NO (never producer-side) | vehicles are caller-owned; `WheelInfo` public fields make this a 3-line consumer recipe |

The five YES rows are exactly the facts the source can hand over today
with zero new persistent state and zero simulation-logic changes; every
NO row is a fact that either does not persist (islands, CCD, mb/soft
contacts) or belongs to the caller (wheel rays) or to a dedicated design
(soft elements, per-kind constraint geometry).

## 23. Deferred Features

Concrete, need-driven deferrals only (no roadmap sprawl — each entry
states what would trigger it):

- **CCD visualization** — via a per-step opt-in sink or an on-demand
  re-enactment query (§18); trigger: a real CCD debugging need.
- **Island-graph visualization** — via a step-time sink; trigger: an
  island-diagnosis need that activation markers cannot answer.
- **Multibody / soft-body contact points** — trigger: debugging mixed
  scenes; requires deciding the sink-vs-snapshot question first.
- **Soft-body element drawing** (nodes, links, faces, clusters) —
  trigger: soft-body authoring tools; needs its own design doc section.
- **Per-kind constraint frame/axis/limit geometry** — trigger: joint
  tuning workflows that anchor pairs are too coarse for.
- **Multibody-constraint anchors** — trigger: robot debugging with
  mb joints.
- **Compound child flattening prims** + a read-only children iterator
  for consumers — trigger: compound authoring; the iterator alone is
  small and may land with the implementation if trivial.
- **Margin-shell visualization** — trigger: margin-band contact
  investigations.
- **Wheel rays producer-side** — permanently out (caller-owned data);
  the consumer recipe lives in the API docs instead.
- **Permanently out of the core:** colors, materials, text, GPU
  geometry, solver-row/force heatmaps (contact normal impulses on the
  prim are the 1.0 answer to load questions).

## 24. API Stability Considerations

Target: freeze a *small* API that can grow by addition — not a promise
that every future debug feature is non-breaking.

- **`DebugDrawFlags`** — additive by construction: a new domain is a new
  const + a new emit branch. Bit values are not part of the stable
  surface (do not serialize flags). Adding flags never breaks.
- **`DebugPrim` variants** — adding a variant breaks exhaustive
  `match`es. Mitigation (**decided**, see §26): the enum is marked
  `#[non_exhaustive]`, so post-1.0 domains (CCD, islands, soft elements)
  can ship without a major version; consumers write the wildcard arm
  they would need anyway for forward-compat. Existing variants' fields
  are kept minimal *now* (anchors-only constraints, no per-variant
  extras) because adding a field to an existing variant is the breaking
  kind of change.
- **`ConstraintTag`** — a payload-free mirror; grows exactly with the
  joint kind set (frozen at six for 1.0). If a joint kind is ever added,
  the new tag rides the same breaking change as the kind itself.
- **Callback signature** — `&mut impl FnMut(DebugPrim<'_>)` contains the
  evolution surface: the prim's lifetime is already exposed, so adding
  borrowed fields to future variants does not change the method's
  shape. A `dyn`-based sink trait is deliberately avoided to keep this
  signature stable and dependency-free.
- **Shape representation** — the prim exposes the engine's real
  `&CollisionShape`. Freezing that means "shape enum changes are
  breaking" (already true — it is a public value enum), so the debug
  API adds no *new* freeze burden. An abstract shape-view type was
  rejected as over-abstraction for a diagnostic surface.
- **Non-promises made now** (so they can never regress): emission order
  is unspecified; flag bit values are unspecified; the number of prims
  per body (e.g. compound flattening) may grow.

## 25. Example

Conceptual usage only (not an implementation, not a renderer):

```rust
use bullet::prelude::*;

// after world.step_simulation(dt):
world.debug_primitives(
    DebugDrawFlags::SHAPES | DebugDrawFlags::CONTACTS,
    |prim| match prim {
        DebugPrim::Shape { position, orientation, shape, .. } => {
            // application-owned renderer: wireframe `shape` at the pose
        }
        DebugPrim::Contact { position, normal, penetration, .. } => {
            // draw the point, and the normal arrow when penetrating
        }
        _ => {} // forward-compat arm (`#[non_exhaustive]` enum)
    },
);

// wheel rays stay consumer-side (vehicles are caller-owned):
for i in 0..vehicle.num_wheels() {
    let w = vehicle.wheel_info(i);
    let from = w.world_origin; // ray reconstructed from public WheelInfo data
    let _to = from + w.wheel_direction_cs * w.suspension_rest_length;
}
```

A headless test consumes the same API by collecting:

```rust
let mut prims = Vec::new();
world.debug_primitives(DebugDrawFlags::ALL, |p| prims.push(p));
assert!(prims.iter().any(|p| matches!(p, DebugPrim::Contact { .. })));
```

## 26. Open Questions

Only what genuinely cannot be settled from the current source:

1. **`ConstraintTag` naming** — `ConstraintTag` vs a `ConstraintKind`
   -adjacent name; a naming-level choice to settle in the freeze
   naming pass (the type's shape is decided; only its name is open).

Previously open, now **decided** (external design review, 2026-09-09):
`#[non_exhaustive]` on `DebugPrim` is **in** — post-1.0 domain additions
(CCD, islands, soft elements) ship without breaking exhaustive
`match`es. Everything else a reader might expect here was deliberately
*decided* against the source instead: the callback shape (§9), the
borrowed-shape model (§13), the AABB semantics (§14), the merged
normals (§15), anchors-only constraints (§16), and the deferrals of
islands and CCD (§17/§18).
