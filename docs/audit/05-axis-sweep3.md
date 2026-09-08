# Audit — AxisSweep3 (three-axis sweep-and-prune) (05)

- **Date**: 2026-09-08 (conclusions fixed by the pre-audit review, the
  superset-invariant analysis and the adversarial validation round; this
  record consolidates them — no re-audit performed)
- **HEAD at audit**: `7708248` (adversarial suite)
- **Scope**: `crates/bullet-collision/src/broadphase/axis_sweep3.rs` —
  quantization frame (`quantize`/`needs_rebuild`/`rebuild`), edge tables and
  the four bubble sorts, pair cache maintenance (`insert_handle`,
  `set_aabb`, `destroy_proxy`), `compute_pairs`, the query APIs.
- **Reference**: `refs/bullet3/src/BulletCollision/BroadphaseCollision/
  btAxisSweep3Internal.h` (`addHandle` :480-524, `updateHandle` :708-750,
  `sortMinDown/Up`, `sortMaxDown/Up`, `testOverlap2D`, quantize).
- **Verdict**: ✅ **Audit-complete.** No algorithm change was needed. The
  core invariant — *cached quantized pairs ⊇ exact AABB-overlapping pairs*,
  so `compute_pairs`' exact filter reproduces brute force — is both proven
  (statically) and validated (per-operation randomized testing). The
  structural concerns raised in the external review of this round were
  checked against the reference and **dismissed**: the three-axis dynamic
  pair maintenance and the insert-time z-axis-only maintenance are exactly
  Bullet's `updateHandle`/`addHandle` semantics, not deviations.

## 1. Reference decisions covered

| Decision point | Bullet reference (`btAxisSweep3Internal.h`) | This crate | Match |
|---|---|---|---|
| Handle mask / sentinel | `m_bpHandleMask = 0xfffe`, `m_handleSentinel = 0xffff` | `MASK = 0xfffe`, `SENTINEL_POS = 0xffff` | ✅ |
| min/max tie-breaking | low bit marks max (even = min sorts strictly before odd = max) | same | ✅ |
| `addHandle` overlap maintenance | axes 0,1 `sortMinDown/MaxDown(..., false)`, axis 2 `(..., true)` (:519-524) — pair set maintained on the z pass only | `insert_handle`: `update_overlaps = k == 2`, same call order | ✅ verbatim |
| `updateHandle` overlap maintenance | all three axes, expand first (`dmin<0 → sortMinDown`, `dmax>0 → sortMaxUp`), then shrink (`dmin>0 → sortMinUp`, `dmax<0 → sortMaxDown`), all with `true` (:708-750) | identical order and guards | ✅ verbatim (an external review claim that this was a structural deviation from Bullet is wrong — it is the reference semantics) |
| Add crossings | `sortMinDown`/`sortMaxUp` add on crossing a max/min edge with the `testOverlap2D` pretest on the other two axes | same | ✅ |
| Removal crossings | `sortMinUp`/`sortMaxDown` remove unconditionally (`USE_OVERLAP_TEST_ON_REMOVES` off) | same | ✅ |
| Quantize | 16-bit floor scale with the is-max flag bit, clamped to `[flag, 0xfffe|flag]` | same, non-clamping variant with explicit clamp branches | ✅ |
| World bounds | user-supplied fixed AABB (`btAxisSweep3(worldAabbMin, worldAabbMax)`) | **auto-grown frame**: recomputed from live AABBs + 100% headroom; escape triggers a full rebuild | ⚠️ documented deviation (see §3) |
| Edge removal on destroy | dead edges swapped to the array end (`m_pEdges[axis][...]` swap with the last) | two edges removed with an O(n) index fixup (`> hi → -2`, `> lo → -1`) | ⚠️ different mechanism, same invariant (slots unique, sentinels preserved) |
| Pair cache | `btHashedOverlappingPairCache` (unordered hash) | sorted `Vec<(usize, usize)>` with binary-search insert/remove | ⚠️ engineering (sorted cache feeds `compute_pairs` output order directly; semantics equal) |
| Query APIs | `rayTest` walks the edge arrays; no AABB query on the broadphase | `raycast_candidates`/`aabb_candidates` are brute-force scans over stored exact AABBs | ⚠️ conservative-equivalent; documented distinction — AxisSweep3 accelerates pair generation only |

## 2. The core invariant (proven + validated)

`cached quantized pairs ⊇ exact overlapping pairs`, hence
`compute_pairs() == BruteForce.compute_pairs()` pair-for-pair (including
order). Proof skeleton (the full argument lives with the tests):

1. **Monotone quantization**: exact overlap ⟹ quantized overlap (floor of a
   monotone affine map; the flag bit keeps a proxy's min strictly before its
   max at equal base).
2. **Cache = quantized-overlap set** is preserved by every transition:
   - insert: the z-pass min-down crossing catches every z-overlapping pair
     (`B.max_z > new.min_z` is necessary for z-overlap) with the 2D pretest
     on the other axes;
   - set_aabb: every axis that transitions quantized-separated →
     quantized-overlapping fires an add-crossing, and the **last such axis
     in 0,1,2 processing order** sees both other axes in their final state,
     so its 2D pretest accepts; removals fire only on true final-separation
     crossings;
   - destroy/rebuild: retain/clear-and-reinsert.
3. The exact filter at output can only remove, never add.

Empirically: 6 seeds × 2048 interleaved operations (create, sub-quantum
jitter through bin-crossing drift, near/far teleports, destroy/recreate),
compared against brute force after **every** operation, plus direct cache
inspections (strictly sorted, no duplicates, no stale entries, superset via
binary search) — all green on f32/f64.

## 3. Documented deviations (vs Bullet)

- **Auto-grown quantization frame** (vs Bullet's user-supplied world box):
  first proxy establishes it; 100% headroom keeps oscillating bodies inside;
  an escape triggers a full rebuild (clear + reinsert + re-pair —
  correctness-safe by construction). Two perf-only properties pinned by
  tests: the frame is **sticky** (a far round trip leaves the wide frame
  until the next escape) and extreme-scale outliers coarsen quantization.
- **`FRAME_COMPONENT_LIMIT = 1e18`**: components at or beyond it (plane AABBs
  are ±1e30) do not constrain the frame; such proxies quantize to the clamped
  frame edges, conservative like a plane inside Bullet's user bounds. The
  degenerate corner (every proxy finite-min / infinite-max on one axis ⇒
  `frame_max = -INF` on that axis) quantizes fully conservatively on that
  axis and self-heals on the next finite update — correctness unaffected.
- **Sub-quantum skip**: `set_aabb` with unchanged quantized edges skips all
  work; Bullet's `updateHandle` runs the (no-op) sorts instead. Observable
  behavior identical.
- **Queries are brute force** (see §1) — pair generation is the accelerated
  path.
- **`f32` quantization-tie fact** (found by the boundary tests): at spacing
  0.2, the mul and add rounding paths of the same coordinate can round
  differently around an f32 tie, dropping one touching pair vs f64 —
  precision-dependent counts must use binary-exact spacings in tests. Not a
  defect (both engines agree on their own stored AABBs).

## 4. Tests (adversarial suite, `7708248`)

9 tests: randomized state machine, direct pair-cache superset/liveness
inspection, quantization boundary scans (coarse + sub-quantum fine, three
axes), degenerate point proxies (coincident/touched/sub-quantum-separated
with quantized edges pinned), per-axis overlap combinations, dense-cluster
teardown, oscillating escape rebuilds (rebuild count asserted), frame
stickiness + coarse-quantization exactness, extreme/unbounded coordinates
(incl. the half-bounded `frame_max` corner). Perf: the needless
`live_aabbs` clone in the query APIs was removed (`34cd18d`) — behavior
unchanged.

## 5. Verdict

**Audit-complete.** The implementation is verbatim Bullet
`addHandle`/`updateHandle` semantics over an auto-grown frame; the pair
output is brute-force-equivalent by a proven and empirically validated
superset invariant. Remaining differences (frame policy, edge-removal
mechanism, cache structure, query routing) are documented engineering
choices with no observable effect on the world-facing pair set. Audit
record only — production code was last touched by the test-round
housekeeping (`34cd18d`), no algorithm change.
