# Parity Harness — frame-by-frame numerical comparison with Bullet

> Design for verifying that this Rust reimplementation reproduces Bullet's
> behavior numerically, not just semantically. All six scenes are landed in
> both precisions (f32 + f64 pairs, 2026-08-14); see `docs/bullet-compare.md`
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
├── generate_scenes.py    # scenes.json -> bullet_scenes.h + parity_scenes.rs
├── bullet_scenes.h       # C++ scene tables (generated, DO NOT EDIT)
├── bullet_record.cpp     # C++ recorder (single file, links refs/bullet3 static libs)
├── build_recorder.py     # MSVC build of the C++ recorder (importable)
├── parity.py             # One-command orchestration: self-build + record + diff + ledger
├── compare.py            # Per-frame diff: position distance + quaternion angle (+ --report-md)
└── check.py              # Regression: cargo test f32/f64 + demo assertions (+ --parity)
crates/bullet/
├── src/parity_scenes.rs  # Rust scene tables (generated, DO NOT EDIT)
└── examples/record_scenes.rs  # Rust recorder (same scenes, same format)
docs/
├── parity-harness.md      # This design + run instructions
└── bullet-compare.md      # Difference ledger (auto-generated + manual classification columns)
```

## Scene protocol

- **Parameter contract**: every scene is defined once in `tools/scenes.json`
  (shapes/poses/masses/friction/restitution, dt as a fraction, iterations,
  recorded bodies); `tools/generate_scenes.py` derives the C++ and Rust scene
  tables, so the two recorders can never drift apart. Current scenes:
  `falling_box`, `falling_capsule`, `stacking`, `mesh`, `ccd`, `sleeping`
  (all landed in both precisions).
  Adding a scene = one JSON block + a parity run + a classification entry.
- **Output format**: one line per frame,
  `t px py pz qx qy qz qw` (10 columns, `%.6f`), redirected to
  `*_bullet.csv` / `*_rust.csv`.
- **Tolerance**: position `1e-2`, orientation angle `1e-2` (f32 reachability).
  Exceedances are logged, not auto-failed — differences are collected first.

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
- **Manifold differences**: Bullet keeps persistent manifolds; we rebuild per
  step with an impulse cache. Rest states should converge; transient motion
  may differ slightly — this is a recorded trade-off, not a bug.
- **Build time**: first Bullet compile ~5–10 min; incremental afterwards.
- Toolchain verified: CMake 3.31 + MSVC (VS2022) available on the dev machine.
