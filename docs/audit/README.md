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
| Persistent Manifold | 🟡 | **audit-complete** 2026-09-08 (`01-manifold.md`): 18 decision points covered, 4 queued risks resolved, D1–D3 accepted deviations pinned by 11 adversarial tests (`ba3cd9a`; D1 is behavioral: contact-patch coverage / manifold adaptation, no instability reproduced) | `01-manifold.md` |
| GJK | 🟡 | **audit-complete** 2026-09-08 (`02-gjk.md`): strict-walk/contact-walk routing, simplex reduction, scale guards (two reproduced bugs fixed `d876cfa`/`9aebf38`), degeneracy limits pinned; remaining thresholds classified as tolerance policy | `02-gjk.md` |
| EPA | 🟡 | **audit-complete** 2026-09-08 (`03-epa.md`): constants/budgets verbatim, two-phase transactional horizon (`1286304`), status acceptance mirrored from `btGjkEpaSolver2::Penetration`; the queued `INSIDE_EPS`/`FALLBACK` risk closed as dead reference constants | `03-epa.md` |
| Convex-Convex (dispatch + fast paths) | 🟡 | **audit-complete** 2026-09-08 (`04-convex-convex.md`): GJK→EPA contract proven and `debug_assert`-pinned, per-status acceptance mirrored; the witness construction recorded as the documented behavioral deviation (deviations §1) | `04-convex-convex.md` |
| AxisSweep3 | 🟡 | **audit-complete** 2026-09-08 (`05-axis-sweep3.md`): `addHandle`/`updateHandle` semantics verbatim; superset invariant proven and validated by the randomized suite (6 seeds × 2048 ops, per-op brute-force comparison); auto-frame/frame-limit policies documented | `05-axis-sweep3.md` |
| DBVT (tree + broadphase) | 🟡 | **audit-complete** 2026-09-08 (`06-dbvt.md`): one reproduced bug fixed during the audit round — pair-cache cleanup now uses the fat leaf-volume test like Bullet's cleanup pass (`ef25fa9`); adversarial suite (randomized state machine × per-op tree-invariant checks, brute-force oracles) pins the superset invariant; tree-shape heuristics documented as accepted engineering choices | `06-dbvt.md` |
| Multibody | 🟡 | 24 mb-probe scenes × f32/f64; islands documented | — |
| SoftBody | 🟡 | soft-body probes; structural offsets quantified (deviations §1) | — |
| BroadPhase integration | 🔵 | next stage: world-level dispatch over both broad phases and their interaction with the step pipeline | planned: `07-broadphase-integration.md` |
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
