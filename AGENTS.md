# AGENTS.md

Durable working rules for humans and coding agents in this repository.
Environment setup lives in [docs/agent-setup.md](docs/agent-setup.md).

## Project map

Pure Rust + glam reimplementation of the Bullet Physics core (zlib, like
upstream). Five crates, strict one-way dependencies:

- `crates/bullet-math` — LinearMath: `Real`/vector/quat aliases (f32/f64), AABB, inertia, quickhull.
- `crates/bullet-collision` — shapes, broadphase, narrowphase (GJK/EPA), manifolds, raycast/sweep.
- `crates/bullet-dynamics` — bodies, world, solver, multibody, softbody, vehicle, character.
- `crates/bullet-physics` — facade: re-exports, prelude, examples.
- `crates/bullet-inverse-dynamics` — `BulletInverseDynamics` (MultiBodyTree); depends only on `bullet-math`, re-exported by the facade behind the off-by-default `inverse-dynamics` feature.

`tools/` is the verification rig: parity/perf harnesses, scene generators,
C++ recorders, probes, and `check.py` (the one-command gate).

Key references: `docs/bullet-deviations.md` (semantic differences vs upstream),
`docs/bullet-compare.md` (parity ledger), `docs/perf/` (speed + memory
ledgers), `docs/audit/` (subsystem audit records, project status).

## Environment hard facts

- MSRV **1.88** (`rust-version` in every manifest; the code uses let-chains).
- `.cargo/config.toml` is host-specific and untracked — create it locally if
  linking fails on a fresh clone (see docs/agent-setup.md).
- Local-only, never commit: `refs/bullet3/`, `crates/bullet-physics/generated/`,
  `handoff.md`, `.cargo/`, `target/`, `tools/` build outputs.
- Scene tables regenerate automatically from `tools/scenes.json` /
  `tools/perf_scenes.json` on first use — never hand-edit generated files.

## Verification gates (all green before every commit)

```bash
cargo test --workspace
cargo test --workspace --features f64
cargo clippy --workspace --all-targets
cargo clippy --workspace --all-targets --features f64
python tools/check.py
cargo check -p bullet-physics --no-default-features --features libm,f32
cargo check -p bullet-physics --no-default-features --features libm,f64
```

Numeric regression vs real Bullet:

```bash
python tools/parity.py --scene falling_box --precision f64
python tools/perf.py --scene all --stages        # --update-ledger re-records
```

## Zero-regression standard

- Parity anchors (`falling_box`, `composite_consumer`) must match
  `docs/bullet-compare.md` bit-for-bit; any divergence must land in a
  previously classified category, or the ledger is re-recorded with a
  written justification.
- Touching multibody paths requires the mb probe byte-comparison
  (`tools/build_multibody_probe.py` + `tools/mbdiff.py`).
- Perf attribution needs interleaved runs; a single `--repeats 1` pass is
  noise, not evidence.

## Porting discipline

- Ported Bullet classes ship a "Reference decisions covered" list: every
  observable upstream decision (branches, defaults, caps, ordering) mapped
  to the port; omissions go to `docs/bullet-deviations.md`.
- No "same as Bullet" wording in docs without a source citation.
- Comments, docs, and commit messages in English; Conventional Commits;
  one atomic commit per feature.

## Code rules

- No `unsafe` (denied at the crate root). No `dyn` in hot paths — prefer
  generics that monomorphize.
- Tests live inline in the same file (`#[cfg(test)] mod tests`); never in
  separate test files.
- Never run `cargo fmt --all` (the repo is not rustfmt-clean); new files
  may be normalized with `rustfmt --edition 2024 <file>`.
- f32 rounding ties: counting/quantization tests use binary-exact spacings
  (0.25, 0.5), never 0.2.
- No new dependencies without approval; `no_std` compatibility is required
  (`std` is a default feature, `libm` the fallback math backend).

## Release process

1. Fill the `Unreleased` section of `CHANGELOG.md` — one line per entry.
2. Bump `version` in `[workspace.package]` (all crates share it).
3. Commit and push.
4. Publish leaf-first: `cargo publish --registry crates-io -p bullet-math`,
   then `-p bullet-collision`, `-p bullet-dynamics`, `-p bullet-physics`.
5. Verify: crates.io API `max_version`, downstream smoke build.
