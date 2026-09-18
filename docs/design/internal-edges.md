# Design: internal edge utility (Bullet `btInternalEdgeUtility`)

> Status: **design baseline** — the implementation must stay inside this
> scope (the debug-draw rule: no silent expansion). Upstream reference:
> `refs/bullet3/src/BulletCollision/CollisionDispatch/btInternalEdgeUtility.{h,cpp}`
> + `btTriangleInfoMap.h`. Purpose: kill the false contact normals a
> sphere/box picks up when it slides across the *interior* shared edges of a
> triangle mesh (edge/corner popping) by rotating the contact normal onto the
> adjacent-triangle-aware direction.

## 1. Data model

New module `crates/bullet-collision/src/shapes/internal_edges.rs`:

```rust
/// Per-triangle internal-edge adjacency info (Bullet's `btTriangleInfo`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleInfo {
    pub flags: u32,            // bit constants below, verbatim
    pub angle_v0v1: Real,      // Bullet m_edgeV0V1Angle
    pub angle_v1v2: Real,
    pub angle_v2v0: Real,
}
// flags: V0V1_CONVEX=1, V1V2_CONVEX=2, V2V0_CONVEX=4,
//        V0V1_SWAP_NORMALB=8, V1V2_SWAP_NORMALB=16, V2V0_SWAP_NORMALB=32
// Default: angles = 2π (TAU), flags = 0 (Bullet constructor).

/// Edge-angle table over a whole mesh (Bullet's `btTriangleInfoMap`).
pub struct TriangleInfoMap {
    pub infos: Vec<TriangleInfo>,      // indexed by global triangle id
    pub convex_epsilon: Real,          // 0.0
    pub planar_epsilon: Real,          // 0.0001
    pub equal_vertex_threshold: Real,  // 0.0001^2 = 1e-8
    pub edge_distance_threshold: Real, // 0.1
    pub zero_area_threshold: Real,     // 0.0001^2 = 1e-8
    pub max_edge_angle_threshold: Real,// 2π
}
```

Decisions:

- **`Vec` direct indexing replaces `btHashMap`** — the per-triangle lookup
  in the adjust pass is an O(1) index read, and the workspace already ruled
  out hash collections (`docs/design/hash-collection-removal.md`). A triangle
  with no adjacency entry keeps the default `2π` angles, which the adjust
  gate (`|angle| < max_edge_angle_threshold`, `2π < 2π`) rejects exactly like
  Bullet's "found in map but this edge has no neighbor" case; a triangle
  index out of the built range never occurs because the map covers every
  triangle (`infos.len() == indices.len()`).
- All six thresholds are **pub fields** with Bullet's constructor defaults —
  tunable like the upstream struct, zero-cost when untouched.
- `no_std`: `alloc::vec::Vec` only, math through `bullet_math::math`, no
  HashMap / thread_local / std types.

## 2. Storage and opt-in surface

`TriangleMesh` gains `internal_edges: Option<TriangleInfoMap>` plus:

```rust
pub fn generate_internal_edge_info(&mut self)   // idempotent (Bullet: "if getTriangleInfoMap() return")
pub fn internal_edge_info(&self) -> Option<&TriangleInfoMap>
```

- **Generating info is the single opt-in.** Bullet needs two opt-ins (build
  the map *and* install a contact-added callback); we fold them into one
  constructor-time call. A mesh without the call is untouched — `Option`
  short-circuits before any narrow-phase code changes (zero-regression by
  construction).
- `scaled()` drops the info (it rebuilds via `new()`): non-uniform scaling
  changes the dihedral angles, so the safe default is "regenerate".
  Documented on `scaled`.
- Heightfields are meshes internally — `as_mesh().generate_internal_edge_info()`
  covers Bullet's `btHeightfieldTerrainShape` overload for free.

## 3. Generation pass (`btGenerateInternalEdgeInfo` + `btConnectivityProcessor`)

One entry `TriangleInfoMap::build(vertices, indices, tri_aabbs, bvh) ->
TriangleInfoMap` (free function or associated; takes the prebuilt mesh
tables so it needs no `TriangleMesh` back-reference):

For every triangle A:

- **Candidate lookup**: BVH query with A's *own* AABB, unexpanded. A
  shared-edge neighbor's AABB shares an edge/face with A's (both built from
  bit-identical vertex coordinates), and `triangles_in_aabb` uses closed
  interval overlap, so the boundary tie hits — no epsilon inflation needed.
  Skip self (Bullet's `processTriangle` self-skip).
- **Degenerate skips** (verbatim): cross-square of both triangles below
  `equal_vertex_threshold` → skip.
- **Shared-vertex scan** (verbatim): pairwise vertex distance² below
  `equal_vertex_threshold`; `numshared == 3` → abort (degenerate overlap),
  `numshared == 2` → the shared-edge branch.
- **Shared-edge branch** (verbatim, including the order fixups): swap
  `sharedVertsA [0,2] → [2,0]` (+B swap) so the edge reads V2V0 not V0V2;
  `edge = A[s1] - A[s0]` normalized; `normalA/B` from `calcNormal`;
  `edgeCrossA/B` flipped to point at the opposite vertex; planar check
  (`calculatedEdge.length² < planar_epsilon` → angle 0, not convex); else
  `angle2 = atan2(edgeCrossB·calculatedNormalA, edgeCrossB·edgeCrossA)`,
  `ang4 = π − angle2`, `isConvex = normalA·edgeCrossB < 0`,
  `correctedAngle = isConvex ? ang4 : −ang4`; then the `sumvertsA ∈
  {1,2,3}` switch writes `−correctedAngle` into the matching angle field,
  the `CONVEX` flag, and the `SWAP_NORMALB` flag (computed via
  `quatRotate(edge, −correctedAngle) · normalA` vs `normalB`).

Complexity: `O(n · (log n + k))` with `k` = per-cell neighbors — one build
at setup time, no per-step cost. Bullet processes each unordered pair twice
(once from each side) and writes only the A-side fields; we keep that shape
so the written angles are produced by the same arithmetic order.

## 4. Contact adjustment (`btAdjustInternalEdgeContacts`)

A **pure local-space function** — no pose, no body, no world state:

```rust
pub(crate) fn adjust_internal_edge_contacts(
    map: &TriangleInfoMap,
    tri: (Vector3, Vector3, Vector3),  // triangle in the mesh's local frame
    local_point: Vector3,              // contact point, mesh local frame
    local_normal: Vector3,             // contact normal (convex → mesh), mesh local frame
) -> Option<Vector3>                   // adjusted local normal; None = keep as-is
```

The caller in `convex_mesh` owns the two quaternion transforms per contact
(world→local point, local→world normal; the mesh's inverse rotation is
hoisted out of the triangle loop). Keeping the core pose-free makes it unit
testable without a world and reusable for the heightfield path.

**Normal-direction mapping (verified against the source, not assumed):**
Bullet's `m_normalWorldOnB` points **mesh → convex** on this path: the
manifold's body0 is the convex and the triangle is body1 — every real
upstream caller hands the triangle side as colObj0
(`btAdjustInternalEdgeContacts(cp, colObj1, colObj0, partId1, index1)` in
`PhysicsServerCommandProcessor` / both world importers), and
`normalInB = m_cachedSeparatingAxis` is the B→A separation direction
(the margin inflation `pointOnA -= axis·marginA; pointOnB += axis·marginB`
confirms the axis). Our mesh narrow phase reports normals in the opposite
(convex → mesh) direction, so the caller **negates on the way in and on
the way out**. With that flip every dot-product comparison maps 1:1; the
concave tail's `d < 0` bail then correctly rejects a face-normal snap that
would oppose the contact, and the common coplanar-seam case (contact
normal ≈ +tri_normal after the flip) takes the concave snap.

Flow (verbatim, single-sided route):

1. `tri_normal = normalize((v1−v0)×(v2−v0))`; nearest-edge selection among
   the edges whose `|angle| < max_edge_angle_threshold`, by distance from
   `local_point` to the edge segment (`btNearestPointInLineSegment`).
2. For the single `bestedge` (the `bestedge == k` guard means at most one
   edge branch ever mutates the normal — structural, same as Bullet):
   - `dist > edge_distance_threshold` → nothing.
   - `angle == 0` → concave hit counter.
   - else: `swapFactor = CONVEX flag ? 1 : −1`; `nA = swapFactor·tri_normal`;
     `nB = swapFactor·rotate(quat(edge_axis, angle), tri_normal)` (negated
     when the `SWAP_NORMALB` flag is set); back-facing test
     `nA·n < convex_epsilon && nB·n < convex_epsilon` → concave hit;
     else convex hit + `btClampNormal` verbatim (`curAngle = atan2(n·edgeCross,
     n·tri_normal_org)` against the `correctedEdgeAngle` window, rotate by
     the diff about the edge axis), applied when the clamped normal stays
     front-facing (`clamped·tri_normal > 0`).
3. Tail: `is_near_edge && concave_hits > 0` → normal becomes `tri_normal`,
   abandoned when it opposes the current contact normal (`dot < 0`).

## 5. Integration points

- `narrowphase/convex_mesh.rs`, both branches (the analytic sphere fast
  path and the generic GJK+EPA branch), per contact point before it enters
  the manifold — i.e. before `reduce(4)`, matching Bullet's
  callback-then-reduce order.
- **Not** `mesh_mesh` (GImpact): upstream's internal-edge utility serves the
  static BVH mesh + heightfield only; GImpact has its own contact
  processing.
- **Persistent-manifold interaction (verified)**: `capture` stores the
  adjusted normal (its match gate `normal·normal ≥ 0.9` is compatible), and
  `refresh` re-projects with the *cached* normal without recomputing — the
  adjusted normal survives every refresh until a re-detection re-runs the
  narrow phase and adjusts again. Same lifecycle as Bullet's callback-fed
  points.
- Ray casts and sweeps are untouched (normal adjustment is a solver-contact
  concern).

## 6. Recorded scope cuts (deviations §1 entries)

1. **`normalAdjustFlags` not ported** (`BT_TRIANGLE_CONVEX_BACKFACE_MODE`,
   `BT_TRIANGLE_CONCAVE_DOUBLE_SIDED`, `BT_TRIANGLE_CONVEX_DOUBLE_SIDED`):
   upstream marks the double-sided modes experimental, its own demos pass 0.
   We hard-wire the single-sided, front-facing path (the flag logic stays
   visible in the source as comments at the two branch points).
2. **Contact position not re-projected**: Bullet rewrites
   `m_positionWorldOnB = m_positionWorldOnA − n_new·m_distance1` after
   rotating the normal (and comments out the `m_distance1` rescale). Our
   contact anchor already *is* the surface point on the triangle — the
   plane does not move when only the normal rotates — so position and
   penetration stay; only the normal changes.
3. **`partId` collapsed into the global triangle index** (parts are a
   construction-time concatenation; single index space, `btGetHash`
   unnecessary).

## 7. Test plan

Unit tests (module `internal_edges.rs`, f32+f64 via the usual features):

- **Build**: coplanar pair → both angle fields 0, no CONVEX flag; a ±30°
  fold → recorded angle + convex/concave side; a lone triangle → all 2π;
  degenerate triangles skipped; one-shared-vertex pairs build no edge.
  Idempotent re-generate.
- **Adjust**: sphere pressing a coplanar seam → edge-contact normal rotated
  to the face normal (`None`-vs-`Some` shape plus numeric equality); point
  farther than `edge_distance_threshold` from the best edge → `None`;
  convex fold → clamped onto the wedge boundary; concave fold → tail branch
  to the fold normal; opposing contact normal → `None` (Bullet's `d < 0`
  bail).

World-level: a sphere resting exactly over a coplanar seam reports a
straight-down contact normal (the pre-fix behavior wobbles between triangle
normals); a mesh without info keeps today's behavior.

## 8. Parity scene

New parity scene `internal_edge` (both recorders): a static two-triangle
coplanar ground mesh + a sphere dropped onto the seam and pushed across it
(same recipe family as `mesh`), long enough to cross the shared edge
several times.

- C++ side: `btGenerateInternalEdgeInfo(mesh, new btTriangleInfoMap)` +
  `gContactAddedCallback` → `btAdjustInternalEdgeContacts` (the
  `InternalEdgeDemo` recipe).
- Rust twin: `mesh.generate_internal_edge_info()` on the ground body.
- Ledger: `docs/bullet-compare.md` f32/f64, expect exceed-0 exact or a
  classified class; without the callback the C++ scene would show the
  seam-pop the utility exists to remove, so the scene doubles as the
  feature's behavioral proof.

## 9. Explicitly out of scope (1.0)

- The three `normalAdjustFlags` modes (§6.1).
- Internal-edge info for GImpact/dynamic meshes.
- Debug-draw visualization of the adjacency (`BT_INTERNAL_EDGE_DEBUG_DRAW`).
