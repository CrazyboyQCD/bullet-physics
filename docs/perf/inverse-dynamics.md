# Performance ledger — `bullet-inverse-dynamics` vs upstream BID

Bench ledger for the inverse-dynamics crate (M3, `docs/design/inverse-dynamics.md`
§9). Unlike the PhysicsWorld ledgers (`speed.md` / `memory.md`, driven by
`tools/perf.py`), this subsystem is measured by the probe pair's `--bench`
mode — `tools/inverse_dynamics_probe[ _f64].exe --bench` (C++, against the
IEEE-precise BID rig `target/bullet3-build-id[-double]`) and
`cargo run --profile release-debug -p bullet-physics --example id_probe
--features inverse-dynamics[,f64] -- --bench` (Rust twin, same protocol).

Protocol (identical on both sides): scenes `chain_16` / `chain_64` /
`chain_256` (fixed root + drawn revolute links, seeds 2101-2103) and
`random_64` (seed 2104, random parents and joint types) × algorithms
`kin` (calculateKinematics) / `id` (calculateInverseDynamics) / `mass`
(calculateMassMatrix) / `jac` (calculateJacobians). Five interleaved rounds
per algorithm with auto-calibrated call counts (~50 ms each); the numbers
below are medians of the per-round ns/call over three interleaved runs of
each binary (2026-10-22, rustc 1.101.0-nightly, MSVC /O2 reference, both
precisions in one session).

## Speed (ns per call, lower is better)

### f32

| Scene | Algo | Rust | BID (C++) | Ratio |
|---|---|---:|---:|---:|
| chain_16 | kin | 419 | 1296 | 0.32x |
| chain_16 | id | 646 | 1713 | 0.38x |
| chain_16 | mass | 1238 | 2805 | 0.44x |
| chain_16 | jac | 1083 | 4233 | 0.26x |
| chain_64 | kin | 1770 | 5321 | 0.33x |
| chain_64 | id | 2715 | 7077 | 0.38x |
| chain_64 | mass | 14019 | 29893 | 0.47x |
| chain_64 | jac | 13819 | 68214 | 0.20x |
| chain_256 | kin | 7131 | 21458 | 0.33x |
| chain_256 | id | 10841 | 28500 | 0.38x |
| chain_256 | mass | 195156 | 412594 | 0.47x |
| chain_256 | jac | 215883 | 1054430 | 0.20x |
| random_64 | kin | 2051 | 5694 | 0.36x |
| random_64 | id | 2859 | 7482 | 0.38x |
| random_64 | mass | 13517 | 45248 | 0.30x |
| random_64 | jac | 32709 | 158219 | 0.21x |

### f64

| Scene | Algo | Rust | BID (C++) | Ratio |
|---|---|---:|---:|---:|
| chain_16 | kin | 571 | 725 | 0.79x |
| chain_16 | id | 882 | 1111 | 0.79x |
| chain_16 | mass | 1581 | 2474 | 0.64x |
| chain_16 | jac | 1275 | 4165 | 0.31x |
| chain_64 | kin | 2411 | 3156 | 0.76x |
| chain_64 | id | 3714 | 4796 | 0.77x |
| chain_64 | mass | 16818 | 28838 | 0.58x |
| chain_64 | jac | 16789 | 67225 | 0.25x |
| chain_256 | kin | 9745 | 12746 | 0.76x |
| chain_256 | id | 14884 | 19597 | 0.76x |
| chain_256 | mass | 227708 | 426109 | 0.53x |
| chain_256 | jac | 272847 | 1091289 | 0.25x |
| random_64 | kin | 2774 | 3513 | 0.79x |
| random_64 | id | 4048 | 5204 | 0.78x |
| random_64 | mass | 17155 | 45254 | 0.38x |
| random_64 | jac | 41586 | 159800 | 0.26x |

Analysis:

- Every scene × algorithm × precision is faster than the upstream library —
  0.20–0.47x in f32, 0.25–0.79x in f64. The largest gaps are the jacobians
  (0.20–0.31x): the O(bodies × dofs) recursion is dense `mat3x` work where
  the SoA flat-row-major layout and glam's column expressions pay off most;
  the closest race is the f64 kinematics (0.76–0.79x), a pass dominated by
  small per-body rotation products with little memory traffic to hide.
- The f32 reference is *slower* than its own f64 build (e.g. chain_16 kin
  1296 vs 725 ns) — a reference-rig fact (both /O2, MSVC; the float BID path
  evidently misses the vectorization the double path gets). The f32 ratios
  are correspondingly flattering; the f64 ratios are the honest headline.
  Rust f32 vs f64 sit where expected (f32 slightly faster or equal).
- The jac and mass columns confirm the upstream formulation's asymptotics:
  both grow O(n²·dof) on chains (chain_256 mass ≈ 13.5x chain_64, jac ≈ 16x
  on both engines), i.e. the shape of the cost is upstream-inherited, the
  constant is what we improved.

## Memory (finalized-tree footprint + per-call allocations)

`tree_live_bytes` = allocation-hook live delta over build + finalize (the
finalized tree's own storage; caller-side q/u/tau/mm buffers excluded).
Per-call allocation counts (`bench_alloc`) are **0 allocs / 0 bytes on both
engines for all four algorithms** — upstream BID also runs its
`calculate_*` paths allocation-free after finalize; the port's zero-alloc
contract (§9 of the design) matches the upstream behavior and is
gate-enforced here (the `id_alloc` check.py demo asserts every
`calculate_*`, setter and getter; `id_probe --bench` asserts the four
benched algorithms hard).

| Scene | Rust f32 | BID f32 | Rust f64 | BID f64 |
|---|---:|---:|---:|---:|
| chain_16 | 18 084 | 5 940 | 29 144 | 11 880 |
| chain_64 | 146 196 | 97 524 | 264 200 | 195 048 |
| chain_256 | 1 764 564 | 1 569 780 | 3 416 264 | 3 139 560 |
| random_64 | 282 944 | 235 296 | 538 720 | 470 592 |

Analysis:

- At scale the footprints converge (chain_256: 1.12x f32 / 1.09x f64 over
  upstream; random_64: 1.20x / 1.14x). Small trees carry proportionally more
  (chain_16: ~2.5-3x): the SoA layout spends ~35 fixed per-field
  allocations (heap-block granularity dominates tiny trees) where upstream
  packs its AOS records; the per-body jacobian windows and the O(bodies ×
  dofs) storage itself are the same order both sides.
- The `Vec<Vec<Real>>` jacobian storage (one window per body, the upstream
  `std::vector<mat3x>` structure) was a recorded flattening candidate
  (design §10 M3). A/B'd in 2026-10-22 with the bench above (interleaved
  arm-A/arm-B exes, 3 rounds per precision, kin/id/mass as within-run drift
  anchors): every delta sat inside the documented ±5-9 pp cross-build noise
  band (chain_256 jac ~1%, random_64 f64 ~+4%, chain_64 f32 −8%, the
  untouched mass anchor itself drifting 2%) — no measurable win, so the
  per-body windows stay (they also mirror upstream's per-body `mat3x`
  structure 1:1). Candidate closed.

## Reproducing

```bash
python tools/build_inverse_dynamics_probe.py both
./tools/inverse_dynamics_probe.exe --bench
./tools/inverse_dynamics_probe_f64.exe --bench
cargo run --profile release-debug -p bullet-physics --example id_probe \
    --features inverse-dynamics -- --bench
cargo run --profile release-debug -p bullet-physics --example id_probe \
    --features inverse-dynamics,f64 -- --bench
```

Interleave the runs when comparing (perf discipline: single passes are
noise; the f32/f64 probe exes overwrite each other in
`target/release-debug/examples/` — copy each precision's exe aside first,
the mbdiff-rig lesson).
