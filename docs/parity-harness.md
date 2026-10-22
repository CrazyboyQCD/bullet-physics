# Parity Harness — frame-by-frame numerical comparison with Bullet

> Design for verifying that this Rust reimplementation reproduces Bullet's
> behavior numerically, not just semantically. All eight scenes are landed in
> both precisions (f32 + f64 pairs, 2026-08-14; `composite_consumer`
> 2026-09-03; `rolling` 2026-09-07); see `docs/bullet-compare.md`
> for the ledger. This document records the harness design and run protocol.

## Goal and key preconditions

- Run the same scene in Bullet (C++, from `refs/bullet3`) and in this crate,
  record per-frame body trajectories, and diff them within tolerance.
- Every out-of-tolerance difference is logged and classified as either a
  **known trade-off** (documented deviation) or an **implementation problem**
  (a bug to fix).
- **Precondition 1 (reproducibility)**: disable `SOLVER_RANDMIZE_ORDER`
  (Bullet enables it by default, which randomizes constraint solving order and
  makes frame-by-frame comparison impossible). The C++ recorder must clear it:
  `solverInfo.m_solverMode &= ~SOLVER_RANDMIZE_ORDER`.
- Precondition 2: fixed `dt = 1/60`, `iterations = 10`, matching gravity.
- Precision matrix: Bullet's default single precision pairs with our `f32`
  build; a `BT_USE_DOUBLE_PRECISION` build pairs with our `f64` feature.

## Architecture

```
tools/
├── scenes.json           # Single source of truth for scene parameters
├── generate_scenes.py    # scenes.json -> bullet_scenes.h + generated/parity_scenes.rs
├── bullet_scenes.h       # C++ scene tables (generated, DO NOT EDIT)
├── bullet_record.cpp     # C++ recorder (single file, links refs/bullet3 static libs)
├── build_recorder.py     # MSVC build of the C++ recorder (importable)
├── parity.py             # One-command orchestration: self-build + record + diff + ledger
├── compare.py            # Per-frame diff: position distance + quaternion angle (+ --report-md)
└── check.py              # Regression: cargo test f32/f64 + demo assertions (+ --parity)
crates/bullet-physics/
├── generated/parity_scenes.rs  # Rust scene tables (generated, DO NOT EDIT)
└── examples/record_scenes.rs  # Rust recorder (same scenes, same format)
docs/
├── parity-harness.md      # This design + run instructions
└── bullet-compare.md      # Difference ledger (auto-generated + manual classification columns)
```

## Probe sources

The remaining `tools/*.cpp` files are single-file programs compiled directly
against the pinned `refs/bullet3` checkout, each answering one question about
what stock Bullet actually does. Some carry a Rust twin and a build script; the
rest are one-off archaeology kept as the primary source behind a documented
decision.

| Probe | Role |
|---|---|
| `multibody_probe.cpp`, `softbody_probe.cpp` | probe rigs with Rust twins (`crates/bullet-physics/examples/*_probe.rs`) and build scripts (`build_*_probe.py`) |
| `conetwist_stress_probe.cpp` | a gravity-lateral chain/pendulum riding its cone swing limit — quantifies limit chatter and the `limit_softness`/`relaxation_factor` effect (A/B within an engine; frame-level engine comparison is chaotic-class). Rust twin `examples/conetwist_stress_probe.rs`, built by `build_conetwist_stress_probe.py` |
| `inverse_dynamics_probe.cpp` | tree-structured multibodies through the stock `BulletInverseDynamics` library for bit-level comparison with the Rust port — 7 scenes (4 canonical: revolute arm / mixed chain / floating base / spherical arm; 3 seeded random trees, `srand` vs `CrtRand`) × 3 states × 4 groups (ID, gravity-free kinematics, mass matrix, jacobians) at `%.17g`; links the IEEE-precise BID in `target/bullet3-build-id[-double]` (`USE_MSVC_FAST_FLOATINGPOINT=OFF` — the `/fp:fast` default swaps in approximate trig that diverges 1 ulp from ucrt). Rust twin `crates/bullet-physics/examples/id_probe.rs` (`required-features = ["inverse-dynamics"]`, exercising the optional facade re-export), built by `build_inverse_dynamics_probe.py`; results in the "Inverse-dynamics probes" section of `bullet-compare.md` |
| `quickhull_probe.cpp` | feeds the minimal flat-slab cloud from the 2026-10-19 P2-C06 quickhull investigation to the stock `HullLibrary` (`btConvexHull`) and checks its triangle output for closure — upstream collapses such clouds to a coarse (5-vertex) valid hull where our index-preserving builder may fold; built by `build_quickhull_probe.py`, cloud pinned by the quickhull tests in `bullet-math` |
| `gimpact_probe.cpp` | probe rig with a Rust twin (`examples/gimpact_probe.rs`), cited by `design/architecture.md` and `bullet-compare.md` |
| `island_probe.cpp` | activation / sleeping / island semantics — cited by `bullet-deviations.md` §2 and `design/multibody-collision.md` §9, built by `build_island_probe.py` |
| `rope_press_probe.cpp` | the rope-pressing-a-box scenario cited by the softbody tests (`rope_pushes_dynamic_box`) |
| `flat_box_probe.cpp`, `flat_box_manifold_probe.cpp` | the wide-flat-box manifold bug (single-corner contact, diagonal spin, sink through the plane) — the scenario of the ignored test `wide_flat_box_on_plane_known_bug`, plus a dump of the cached manifold point set |
| `swing_push_probe.cpp` | the passive-pendulum-pushing-a-box scenarios from the early multibody/island work (box trajectory and activation state); no live test names it now |

The probes without a build script (`gimpact_probe.cpp`, `rope_press_probe.cpp`,
`flat_box_probe.cpp`, `flat_box_manifold_probe.cpp`, `swing_push_probe.cpp`)
were compiled ad hoc against `refs/bullet3`; they are kept for the record rather
than for reruns.

## Scene protocol

- **Parameter contract**: every scene is defined once in `tools/scenes.json`
  (shapes/poses/masses/friction/restitution, dt as a fraction, iterations,
  recorded bodies); `tools/generate_scenes.py` derives the C++ and Rust scene
  tables, so the two recorders can never drift apart. Current scenes:
  `falling_box`, `falling_capsule`, `stacking`, `mesh`, `ccd`, `sleeping`,
  `composite_consumer`, `rolling` (all landed in both precisions).
  Adding a scene = one JSON block + a parity run + a classification entry.
  Consumer-API scene fields (all optional, ignored by scenes that omit them):
  body `gravity[x,y,z]` (implies `DISABLE_WORLD_GRAVITY`), body
  `no_contact_response`, body `group`/`mask` (`-1` = all), scene
  `pair_filter [nameA, nameB]`, and scene `rays` probes
  (`from`/`to`/`mode all|closest`/`group`/`mask`).
- **Output format**: one line per frame,
  `t px py pz qx qy qz qw` (10 columns, `%.6f`), redirected to
  `*_bullet.csv` / `*_rust.csv`. Scenes with rays additionally write
  `<scene>_rays_*.csv`, one line per hit, `ray_idx body_idx fraction`
  (`%.6f`), sorted by fraction — the C++ `AllHitsRayResultCallback`
  collects in broadphase pair order, so the recorder sorts to match the
  Rust recorder's nearest-first order.
- **Tolerance**: position `1e-2`, orientation angle `1e-2`, ray fraction
  `1e-3` (f32 reachability). Exceedances are logged, not auto-failed —
  differences are collected first.
- **Consumer-API alignment notes**: the C++ side attaches the pair filter as
  a `btOverlapFilterCallback` on the broadphase pair cache — upstream
  *replaces* the default group/mask test, so the callback re-does it before
  the exclusion, matching the Rust one-layer gate (the dispatcher's
  `needsCollision` still tests group & mask a second time). Collision
  events (`Started`/`Stopped`) are a Rapier-style surface with no Bullet
  counterpart, so they stay covered by Rust unit tests only, not this
  harness.

## Difference classification

| Class | Criterion | Disposition |
|---|---|---|
| Known trade-off | Pattern matches an entry in `bullet-deviations.md` (restitution threshold 1.0 vs 0.2, penetration slop, no friction warm starting, per-step manifold rebuild, activation propagation, etc.) | Ledger marks `expected → deviations #N`; no code change |
| Implementation problem | Unbounded drift, directional bias, energy injection, divergence | Open a bug, fix, re-run the scene |
| Unknown | Neither of the above | Parked until manually investigated, then reclassified |

The comparison report lists per-scene max deviation (frame + magnitude) and
mean deviation; classification columns are filled in manually.

## Landing path (get one scene working end-to-end first)

All steps are done; the list records how the harness was landed.

1. ✅ CMake-build `refs/bullet3` (only `LinearMath` + `BulletCollision` +
   `BulletDynamics` static libs; disable example/demo targets).
2. ✅ Write `bullet_record.cpp` replicating `falling_box`; record the trajectory.
3. ✅ Write `record_scenes.rs` for the same scene; record the trajectory.
4. ✅ Run `compare.py`; produce the first difference report; classify; log to
   `bullet-compare.md` (see `tools/parity.py --update-ledger` for the automated
   rerun).
5. ✅ Extend to the remaining scenes; register each
   scene in `tools/parity.py`'s `EXAMPLES` map.

## Risks and mitigations

- **Quaternion double cover**: `qw` may differ in sign between engines — take
  `min(angle(q1, q2), angle(-q1, q2))`.
- **Manifold differences**: both engines run per-pair persistent manifolds
  now (our merge-on-capture + anchored refresh landed 2026-09-01, matching
  `btPersistentManifold`'s semantics; see `bullet-deviations.md` §1).
  Rest states should converge; transient motion may differ slightly — this
  is a recorded trade-off, not a bug.
- **Build time**: first Bullet compile ~5–10 min; incremental afterwards.
- Toolchain verified: CMake 3.31 + MSVC (VS2022) available on the dev machine.
