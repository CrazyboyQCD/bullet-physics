# Audit — DBVT (tree + broadphase) (06)

- **Date**: 2026-09-08
- **HEAD at audit**: `d94c6ce`
- **Scope**: `crates/bullet-collision/src/broadphase/dbvt.rs` (`DbvtTree`,
  `DbvtBroadPhase`) — the soft-body `cdbvt`/`ndbvt`/`fdbvt` reuse the same
  `DbvtTree` core and are covered through it.
- **Reference**: `refs/bullet3/src/BulletCollision/BroadphaseCollision/
  btDbvt.{h,cpp}` (`insertleaf`, `removeleaf`, `update` overloads, `sort`,
  `optimizeIncremental`, `collideTV`, `collideTTpersistentStack`,
  `rayTestInternal`), `btDbvtBroadphase.{h,cpp}` (`createProxy`, `setAabb`,
  `collide`, cleanup pass, ctor defaults, `DBVT_BP_*` configuration macros,
  `gDbvtMargin`).
- **Verdict**: ✅ **Audit-complete.** One reproduced correctness bug found and
  fixed during the adversarial validation round (`ef25fa9`, same day): the
  broad-phase pair-cache cleanup used the exact-AABB overlap test where
  Bullet uses the fat leaf-volume test, which could permanently drop a pair
  that later re-overlapped through within-fat motion. After the fix the
  superset invariant holds under the full randomized suite. Remaining
  differences are tree-shape/performance heuristics (documented below) that
  the structural-invariant and brute-force-oracle tests pin as
  output-neutral.

> Reading guide: as with `01-manifold.md`, "audit-complete" means every
> observable decision point has been compared and the remaining differences
> are precisely characterized and pinned by tests. It does not mean
> bit-equivalence with Bullet — the broad phase's *pair output* is exactly
> brute-force-equivalent (that is what the world consumes), while the
> internal tree shape legitimately differs from Bullet's pointer-order
> heuristics.

## 1. Reference decisions covered

Mechanical comparison of every observable decision point:

| Decision point | Bullet reference | This crate | Match |
|---|---|---|---|
| Margin constant | `gDbvtMargin = 0.05` (`btDbvtBroadphase.cpp:20`) | `DBVT_MARGIN = 0.05` | ✅ |
| Prediction factor | `m_prediction = 0` (ctor) | `PREDICTION = 0.0` | ✅ |
| Fat growth | `update(leaf,vol,velocity,margin)`: `Contain(volume)` skip, else `Expand(margin)` + `SignedExpand(velocity)`, reinsert (`btDbvt.cpp:583-591`) | `update_fat`: `contains` skip, else `expanded_vec(margin).signed_expanded(velocity)`, `update(-1)` | ✅ |
| `setAabb` moving/teleport split | `Intersect(leaf->volume, aabb)` → Moving; else Teleport = exact reinsert (`btDbvtBroadphase.cpp:333-354`) | `fat.overlaps(&aabb)` → `update_fat`; else `update(exact, -1)` | ✅ |
| Collide gating on update | `docollide` only when `m_sets[0].update(...)` returned true (tree changed), `:341-348`; fixed→dynamic always collides (`:324-329`) | `docollide = update_fat(...)` return; fixed branch sets `docollide = true` | ✅ |
| `DBVT_BP_PREVENTFALSEUPDATE` | `= 0` (no identical-AABB early-out) | none | ✅ |
| Immediate collide | `m_deferedcollide = false` (ctor): collides both trees per proxy event, fixed tree first then dynamic (`:407-412`) | `collide_leaf`: dynamic tree first then fixed | ✅ order-inverted (pair set is unordered; not observable) |
| Pair formation | `btDbvtTreeCollider` → `m_paircache->addOverlappingPair` (insert-only at event time) | `collide_leaf` → `pair_insert` (insert-only, dedup, sorted vec) | ✅ equivalent |
| Stage machinery | `STAGECOUNT = 2`, `m_stageRoots[3]`, intrusive `listappend/listremove`, create and setAabb re-append to `m_stageCurrent` | same three stage lists, same append/remove points | ✅ |
| Stage rotation | `m_stageCurrent = (m_stageCurrent + 1) % STAGECOUNT` | `stage_current ^= 1` | ✅ |
| Dynamic→fixed migration | sweep `m_stageRoots[m_stageCurrent]`, `listremove` → `listappend(fixed)`, `m_sets[0].remove`, `m_sets[1].insert(FromMM(exact))`, `stage = STAGECOUNT`; **no pair maintenance** with `DBVT_BP_ACCURATESLEEPING = 0`; then `m_fixedleft = m_sets[1].m_leaves` | identical, incl. no pair maintenance and `fixed_left = fixed.leaves` | ✅ |
| Dynamic-tree rotation budget | `optimizeIncremental(1 + leaves * m_dupdates / 100)`, `m_dupdates = 0` ⇒ 1 pass | `optimize_incremental(1)` | ✅ |
| Fixed-tree rotation budget | passes `1 + leaves * m_fupdates / 100` (`m_fupdates = 1` ⇒ scales with leaf count), budget decremented by the same amount | always 1 pass, `fixed_left -= 1` | ⚠️ heuristic: Bullet rotates large fixed trees proportionally more; ours rotates one leaf-path per tick (tree-quality only, see deviations) |
| Cleanup pass trigger + extent | `m_needcleanup` set on collide/migration; scans `min(size, max(m_newpairs, size * m_cupdates / 100))` pairs per tick (10% ring, `m_cupdates = 10`) | unconditional full pass over the pair set each tick | ⚠️ timing only (eventually-consistent either way); full pass is simpler and immediate |
| **Cleanup pass criterion** | `if (!Intersect(pa->leaf->volume, pb->leaf->volume)) removePair` — **fat leaf volumes**, not exact AABBs | keep while the pair's leaf volumes overlap; output filtered by exact AABB | ✅ after `ef25fa9` (was exact-AABB — the bug in §2) |
| Cache granularity contract | pairs are fat-overlap candidates; exact filtering happens downstream (`needsCollision`/narrow phase) | pairs kept at fat-overlap granularity, `compute_pairs` returns the exact-filtered subset | ✅ observable semantics equal for the world |
| `insertleaf` descend criterion | `Proximity` = L1 on `(mi + mx) - (mi + mx)`, strict `<` picks c0 | `proximity` verbatim | ✅ |
| `insertleaf` ancestor fixup | capture `indexof(root)` before the parent rewrite; upward re-merge while the parent volume does not contain the child's | same, with the ordering trap called out in-code | ✅ |
| `removeleaf` sibling substitution | sibling takes the parent's slot; ancestors re-merged | same; upward loop breaks when the merged volume equals the old one | ⚠️ Bullet stops on `Intersect(old, merged)` (looser, frozen volumes); we re-merge to exact equality (tighter internal volumes). Both preserve "node volume ⊇ subtree leaves" — output-neutral |
| `sort` rotation | `if (p > n)` **pointer-order** comparison, child/sibling/parent rehook, volume swap (`btDbvt.cpp:418-449`) | same rehooks with slab-index order standing in for pointer order | ⚠️ the comparison key differs (slab index is allocation order; Bullet's pointer order is allocator-dependent) — a heuristic, output-neutral |
| `optimizeIncremental` traversal | `m_opath` bit path, `bit = (bit + 1) & 31`, one `m_opath++` per pass | verbatim | ✅ |
| `optimizeIncremental` reinsertion | single-argument `update(node)` ⇒ `lookahead = -1` ⇒ **reinsert from the tree root** (`m_lkhd = -1` is the other overload's default, unused here) | `update(node, volume, 2)` — two levels above the shrink-test stop | ⚠️ tree-shape choice (see deviations); the stale "uses 2" doc claim was corrected in this audit |
| `collideTV` leaf order | explicit stack pushes c0 then c1 ⇒ c1 (right subtree) processes first | `walk_overlapping` visits `children[1]` first | ✅ (soft-body rcontacts append in visit order) |
| `collideTT` same-tree branch | `a == b` expands `(c0,c0), (c1,c1), (c0,c1)` — mirrored pairs skipped, each unordered leaf pair once | verbatim | ✅ |
| `collideTT` cross-tree expansion | `(a0,b0),(a1,b0),(a0,b1),(a1,b1)` for two internals, 2-entry forms for leaf/internal, `Intersect` pruning | verbatim | ✅ |
| Ray test | both trees via `rayTestInternal`, per-node `btRayAabb2` with the caller-supplied world clip box, same slab test for internal nodes and leaves | both trees via `for_each_on_segment`, shared slab test `t ∈ [0,1]` on fat volumes, no world clip box | ⚠️ equivalent-conservative: our test is the same parameterized slab without the (default-tight) clip box, so candidates are a superset |
| AABB query | `collideTV` over both trees | `for_each_overlapping` over both trees | ✅ |
| Proxy storage | heap `btDbvtProxy` per proxy | slab `Vec<Option<Proxy>>` with slot reuse | ⚠️ engineering (slot identity is the typed-handle-era decision; equivalence pinned by tests) |

## 2. Reproduced bug — found and fixed in this audit (`ef25fa9`)

**`compute_pairs` pruned cached pairs on the exact-AABB test.** The cleanup
pass dropped a cached pair as soon as the *exact* AABBs separated. But both
proxies can then keep moving *inside their existing fat volumes* — a path
that skips the tree update (`update_fat` containment fast path) **and** the
collide pass (`docollide = false`) — and close the exact gap again. At that
point nothing re-inserts the pair: it is gone from the cache, so the reported
output misses an exact overlap. Reproduced by two tests in the adversarial
suite (the explicit fat-move test and the margin-boundary scan, third step).

Bullet's cleanup uses `Intersect(leaf->volume, leaf->volume)` — the fat
volumes — precisely so that margin-adjacent pairs survive until they are
truly out of range (a re-approach from there must cross a fat boundary,
which produces a tree update and a collide). The fix keeps cached pairs
while their **leaf volumes** overlap (the fixed tree stores exact volumes,
which is safe: fixed proxies do not move and wake-ups re-collide) while the
reported output remains the exact-filtered subset. The randomized
state machine, run with per-operation brute-force comparison, passes
post-fix; cost: `dbvt_rain_2000` 6.265 → 6.697 ms (0.33x, ledger rerun
`d94c6ce`), inherent to the conservative cache.

## 3. Deviations (engineering choices, output-neutral; not filed in
`bullet-deviations.md` §1 as they are not observable in pair output)

- **Fixed-tree rotation budget**: Bullet scales passes with the fixed leaf
  count (`m_fupdates = 1`); we rotate one path per tick. With thousands of
  resting proxies Bullet keeps the fixed tree proportionally tighter.
  Performance/quality heuristic only.
- **`optimize_incremental` reinsertion level**: root (Bullet) vs
  shrink-stop + 2 (ours). Local tree-shape choice; the structural checker
  and the overlap/pair oracles pin correctness for both.
- **`sort` comparison key**: slab-index order vs Bullet's pointer order.
  Both are "allocated-later floats up" heuristics.
- **Cleanup extent**: full pass per tick vs Bullet's 10% ring window.
- **Tighter internal volumes**: our `remove_leaf` re-merges ancestors to
  exact equality; Bullet freezes them at `Intersect` stop. Ours are
  never looser than the children's union, so all queries stay conservative.
- **Ray candidate test**: parameterized slab without Bullet's world clip
  box (`btRayAabb2` bounds transform); candidates remain a superset.

## 4. Test gaps → closed by the adversarial validation round (`ef25fa9`)

- ~~No randomized broad-phase state machine~~ → 4 seeds × 2048 interleaved
  create/small-move/teleport/destroy/tick operations, per-operation
  brute-force comparison plus per-operation structural tree checks on both
  sets.
- ~~No direct test of the fat-volume candidate contract~~ → explicit
  fat-overlap/exact-separation setup, within-fat move surfacing the cached
  pair, and the reverse (false positive filtered).
- ~~No migration-specific coverage~~ → fixed↔dynamic round trip with
  cross-tree pair formation and fixed-vs-fixed carry-over.
- ~~`optimize_incremental` untested beyond smoke~~ → 600-step rotation
  churn with a structural checker (root/parent/child pointers, live/dead
  slots, free-list agreement, leaf bookkeeping, payload uniqueness) after
  every operation, plus overlap-query and same-tree pair-traversal
  brute-force oracles (exactly-once, no self-pairs).
- ~~Degenerate shapes untested~~ → identical/nested/point/thin/huge volumes,
  1–3 leaf trees, full insert/update/remove cycles back to empty.
- ~~Boundary semantics unpinned~~ → face/edge/corner exact touches,
  epsilon separations, moves around the margin boundary.
- ~~Queries only spot-checked~~ → randomized rays (incl. zero-length,
  inside-start, tangent, endpoint-on-box) and randomized AABB queries (incl.
  zero-volume, huge, exact-touch, empty region) over a mixed dynamic/fixed
  state.
- ~~Extreme coordinates untested~~ → 1e-6…1e30 bases and ±INF plane-like
  proxies through the full state machine including fixed migration.

## 5. Candidates (not reproduced; recorded)

- The fixed-tree rotation budget and the `optimize_incremental`
  reinsertion level differences (§3) mean long-running scenes will not
  converge to Bullet's exact tree shape. No behavioral divergence is
  possible while the structural invariants hold (pinned per operation);
  the observable pair output is brute-force-equivalent.
- `m_needcleanup`-style deferred cleanup is replaced by an unconditional
  full pass. For pair sets in the tens of thousands the per-tick cost is
  linear in the cache size either way; flagged for the broad-phase perf
  audit slot rather than changed here.

## 6. Verdict

**Audit-complete.** The tree core (`btDbvt`) matches the reference on every
observable decision point with two documented tree-shape heuristics; the
broad-phase wrapper (`btDbvtBroadphase`) matches on all defaults and state
transitions, with one reproduced correctness bug (cleanup granularity) found
by the adversarial suite and fixed to Bullet's semantics in `ef25fa9`. The
`DbvtTree` reuse by the soft-body trees (`cdbvt`) is covered through the
same oracles (insert/remove/update_fat/overlap/pair traversal). Recommendation
for the remaining rotation/heuristic deviations: accepted as engineering
choices; revisit only if a broad-phase performance audit shows tree-quality
degradation against Bullet on long-resting scenes.
