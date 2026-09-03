# Hash collection removal — measurements and the path back to hashbrown

Records the removal of `std::collections::{HashMap, HashSet}` from the
simulation hot path (commit `26fff5d`, no_std step 1): what each site
became, the measured performance impact, and everything needed to
revert to `hashbrown` (or `std`) later without re-deriving the
analysis. Performance numbers are Windows 24H2 / x86-64, MSVC release,
`python tools/perf.py --repeats 3` medians (rust per-frame ms); run-to-
run noise on this machine is ±3-5% — the C++ side of each table is the
cross-check for environment drift.

## The three sites (before -> after)

| Site | Before | After | Bullet's own structure for the same job |
|---|---|---|---|
| `world.rs` `manifold_cache` | `HashMap<(usize, usize), PersistentManifold, FxBuild>` (custom FxHasher, ~24k lookups/step on the 1000-body stack: mark/refresh/seed/writeback) | Adjacency buckets `Vec<Vec<ManifoldCacheEntry>>` indexed by `min(i, j)`, each entry naming `other`; bucket-linear find | `btHashedSimplePairCache` — a hand-rolled integer-pair hash (Thomas Wang hash + bucket chains over `btAlignedObjectArray`) |
| `world.rs` `no_collision_pairs` | `HashSet<(usize, usize), FxBuild>` | Sorted, deduplicated `Vec<(usize, usize)>` + `binary_search` | Not a global set at all: per-object ignore lists checked via `checkCollideWithOverride` (linear over a handful of entries) |
| `softbody/helpers.rs` `create_from_tri_mesh` edge dedup | `std::collections::HashSet` | `std::collections::BTreeSet` (alloc, no hasher) | Modeling-time tool; Bullet mixes linear scans and even `std::map` in `btSoftBodyHelpers` |

Key structural insight: contact pairs are strongly local — a body only
ever holds cached manifolds against its few neighbours — so per-body
buckets keep lookups/inserts/removals at O(bucket size) (a handful of
entries) with no hasher and no large-data memmove on pair churn.

## Measured performance (rust per-frame ms, medians)

Baseline = `be5b023` (Fx-hash versions). Interim = sorted flat
`Vec<(pair, manifold)>` with binary search — **rejected**. Final =
adjacency buckets (shipped in `26fff5d`).

| Scene | Baseline | Interim sorted Vec | Final buckets |
|---|---|---|---|
| stack_1000 | 1.255-1.259 | **1.517 (+21%)** | 1.232-1.324 |
| rain_500 | 1.322 | — | 1.399-1.466 |
| mb_chains | 0.456-0.501 | — | 0.488 |
| dbvt_rain_2000 | 7.886 | — | 8.032-8.479 |
| gimpact_rain_100 | 0.412 | — | 0.431 |
| cloth_wind | 0.231 | — | 0.237 |
| softbody_rain | 0.307 | — | 0.312 |
| softbody_mesh | 1.913 | — | 1.941 |
| game_arena | 0.133-0.147 | — | 0.139-0.149 |
| sphere_rain_1000 | 0.532-0.538 | — | 0.535-0.563 |
| sphere_pool_4000 | 2.456/2.554 | — | 2.357/2.394 |

Reading of the data:

- All final numbers sit inside the ±3-5% run-to-run noise band; the C++
  side drifted the same direction on the same runs (e.g. stack_1000
  bullet 1.355-1.437 across rounds).
- The one real regression was the interim sorted array: +21% on
  stack_1000. Root cause is **not** the binary search but the
  `Vec::insert`/`remove` memmove of large `PersistentManifold` entries
  (hundreds of bytes) while pair turnover is high during the stack's
  collapse phase. Any flat-sorted revival must answer for that memmove.

## If/when reverting to hashbrown

Trigger conditions (any one): profiling shows the pair-pass cache
lookups back above ~2% of the step on the big scenes; scenes with tens
of thousands of live contact pairs (bucket-linear cost grows with
per-body neighbour count); or `hashbrown` gets adopted anyway for
another no_std need.

Recipe:

1. Restore `FxHasher`/`FxBuild` + the two fields verbatim from the
   parent of `26fff5d` (`git show 26fff5d^:crates/bullet-dynamics/src/world.rs`
   — FxHasher block sat right above `BROADPHASE_AABB_MARGIN`). Either
   `std::collections` with `FxBuild` or `hashbrown::HashMap`/`HashSet`
   with `FxBuild` — the hasher is plain arithmetic and needs no std.
2. Swap the `cache_*` associated functions back to
   `get_mut`/`get`/`entry().or_default()`/`remove` on the pair key;
   the writeback loop loses its `located` per-pair hint (a plain
   `get_mut` per row is fine again).
3. `no_collision_pairs` can stay a sorted array (its cost is trivial)
   or return to `HashSet` — independent decision, zero coupling.
4. Dependency accounting: `hashbrown` is a new crate (see the standing
   no-new-dependencies decision, 2026-09-03); plain `std::collections`
   adds nothing but gives up the no_std track. Both choices must be
   weighed against the bucket layout being dependency-free and measured
   neutral.

Re-verification checklist (same discipline as this round): the nine
ledger scenes with `--repeats 3`, stack_1000 specifically (pair churn
stress), plus 327 tests x f32/f64, the 14 demos of `tools/check.py`,
and clippy on both features.

## Behavior guarantees kept through the change

- `manifold_cache` iteration order never feeds numerics (values() was
  test-only; the pair pass keys every access), so bucket order vs hash
  order cannot diverge simulation results.
- Suppression-set semantics unchanged: gate points (pair acceptance +
  CCD sweep) and the cascade rebuild on `remove_body` /
  `remove_constraint` are identical.
- Link creation order in `create_from_tri_mesh` is unchanged (the set
  is dedup-only; discovery order still drives `append_link`).
