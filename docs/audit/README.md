# Independent Source Audit

This directory records the **independent correctness audit** of the physics
engine source. The feature core is functionally complete (see
[`00-project-status.md`](00-project-status.md)); the current phase is
correctness hardening: subsystem-by-subsystem source review against the
Bullet reference (`refs/bullet3`), adversarial testing, and the resolution of
anything the reviews surface.

**Scope note**: the parity harness and the probe scenes are numerical
evidence for the *tested* cases only. They do not substitute for reading the
source. "Not yet independently audited" does **not** mean "buggy" — it means
no one has done a dedicated source-level review of that subsystem yet.

## Status legend

| Mark | Meaning |
|---|---|
| 🟢 | Implemented, and current evidence (tests + parity/probe coverage) is sufficient for the claims being made |
| 🟡 | Implemented, with documented deviations or known risks (`docs/bullet-deviations.md`) |
| 🔍 | Not yet independently source-audited |
| 🔴 | Known problem that needs handling |
| 🔵 | Planned / later |

## Subsystem status

| Subsystem | Status | Evidence today | Audit record |
|---|---|---|---|
| World / step pipeline | 🟡 | 350 unit tests ×2; parity 8 scenes × f32/f64; stepping order documented in `docs/architecture.md` §7 | — |
| RigidBody | 🟡 | unit tests; kinematic / motion-state / rolling-friction semantics documented | — |
| Solver (sequential impulse) | 🟡 | parity stacks; solver-info defaults audited 2026-09-01 (deviations §1/§2) | — |
| Persistent Manifold | 🟡 | audited 2026-09-08 (`01-manifold.md`): 18 decision points covered, 3 deviation rows recorded, 4 queued risks resolved, no reproduced candidate | `01-manifold.md` |
| GJK | 🔍 | exercised by every parity/probe scene; no dedicated source review | planned: `02-gjk.md` |
| EPA | 🔍 | `btGjkEpa2` line-by-line claim from the porting round; not re-audited | planned: `03-epa.md` |
| Convex-Convex (dispatch + fast paths) | 🔍 | sphere/sphere, sphere/box, box/box SAT covered by scenes | planned: `04-convex-convex.md` |
| AxisSweep3 | 🔍 | default broad phase; perf parity | planned: `05-axis-sweep3.md` |
| DBVT (tree + broadphase) | 🔍 | full-port claim from the porting round; soft-body trees reuse it | planned: `06-dbvt.md` |
| Multibody | 🟡 | 24 mb-probe scenes × f32/f64; islands documented | — |
| SoftBody | 🟡 | soft-body probes; structural offsets quantified (deviations §1) | — |
| API surface | 🟢 | consumer backlog shipped through batch 2 (`docs/api-gaps.md`) | — |
| Parity harness | 🟢 | 8 scenes × f32/f64, ledger current | `docs/parity-harness.md` |
| Performance | 🟢 | perf ledger current (2026-09-04 rerun) | `docs/bullet-perf.md` |
| Adversarial testing | 🔵 | not started | planned |
| Parallel solver (IslandWork) | 🔵 | deliberately deferred (no rayon; on-demand decision 2026-09-03) | — |

Interpretation rules:

- 🔍 entries are **queued for review**, not suspected of anything. Do not
  downgrade one to 🔴 without concrete evidence (a reproducible failure, a
  proven semantic mismatch, or a ledger classification that breaks down).
- 🟡 means the deviations are *documented and accepted*, not silently
  ignored; the authoritative list is `docs/bullet-deviations.md`.
- An audit record file (`0N-<subsystem>.md`) exists only after its review
  has been performed; the filename in the table above is the planned slot.

## How an audit entry is produced

1. Read the subsystem source against the Bullet reference
   (`refs/bullet3/src/...`), mechanically covering every observable decision
   point (branches, defaults, thresholds, ordering, capacity limits).
2. Record: reference decisions covered, deviations found (new or confirming
   `bullet-deviations.md`), test gaps, and any candidate bug — a candidate
   stays a *candidate* until reproduced.
3. File `0N-<subsystem>.md` in this directory and update the table above.
