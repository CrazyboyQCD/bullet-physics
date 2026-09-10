# Consumer API priority assessment

> Unified evaluation of the remaining consumer-API gaps (`docs/api-gaps.md`
> backlog + deviations §3), scoring usage frequency and implementation size.
> Decided with the maintainer on 2026-09-07; the order below drives the
> priorities in `HANDOFF.md` §5. Update this document when items ship or
> verdicts change.

## Unified scoring table

| Item | Frequency | Size | Verdict | Rationale |
|---|---|---|---|---|
| **Kinematic rigid bodies** | ★★★★★ | M | **Done** | Daily necessity for moving platforms / elevators / doors / vehicles; no correct workaround (hacks propagate no velocity). Shipped 2026-09-04. |
| **Compound child dynamic transforms** | ★★★★ | S–M | **Done** | Kinematic compounds are the standard moving-vehicle setup; implemented together with kinematics. Shipped 2026-09-07. |
| **Ghost persistent overlap list** | ★★★☆ | S | **Done** | Level-query alternative to hand-rolled event bookkeeping; the pair pass already tracks touch state — pure exposure. Shipped 2026-09-07 (`world.overlapping`). |
| **Rolling friction** | ★★★☆ | S–M | **Done** | Physical-correctness gap for resting balls/cylinders (pool / wheel scenes); damping hacks are unphysical. Shipped 2026-09-07 (`body.rolling_friction`, parity scene `rolling` — rf rows exceed-0 exact). |
| **Motion states** | ★★★ | S–M | **Done** | Render interpolation is an advanced need; revisit depending on whether the kinematic driving API wants them early. Shipped 2026-09-07 (`MotionState` trait, `add_body` pose read + step-tail synchronize; the interpolation blend and the kinematic drive stay unported — deviations). |
| **mb/soft event sources** | ★★☆ | S–M | **Done** | Robot-arm perception and similar composite scenes exist; extend the event-source dispatch. Shipped 2026-09-07 (`CollisionObject` sides on `CollisionEvent`, `LinkCollider`/`SoftBody` opt-in, pair-level soft events). |
| **Persistent-manifold extras** | ★★☆ | S | **Done** | Shipped 2026-09-10: `ContactPoint::lifetime` (+ `PersistentPoint`), `HAS_FRICTION_ANCHOR` friction-anchor servo (rigid solver, Coulomb-cone bounded), `set_contact_point_events` point drain. Rigid paths only (deviations §1); memory ledger addendum +0.125 MB. |
| **btShapeHull** | ★★☆ | S | **Done** | Shipped 2026-09-10 (`ShapeHull::build`, lowres 42-direction table); quickhull moved to `bullet-math` (LinearMath mapping). Highres/penetration extras stay unported (deviations §1). |
| **Character directional jump** | ★★ | S | **Done** | Shipped 2026-09-10: `jump_direction(v)` + `set_jump_axis` with the `m_SetjumpSpeed` overwrite/restore semantics. `setInterpolateUp` / `setMaxJumpHeight` stay unexposed (deviations §3). |
| **Serialize** | ★★ | L | Defer | Savegames work with handwritten pose snapshots (bodies are all plain data); bit-exact round-trip has no consumer ecosystem — post-release. |
| **Anisotropic friction** | ★★ | M | Defer | Rare; conveyor-belt cases are better served by surface velocity. |
| **Contact stiffness/damping** | ★★ | M | Defer | Soft contacts are niche; record in deviations, pull on demand. |
| **Additional damping** | ★☆ | S | Defer | A humanoid-stack tuning safety valve; scenario-specific. |
| **Multi-part triangle meshes** | ★★ | M | Defer | Vertex/index concatenation is a workaround with no runtime benefit. |
| **Soft cluster core (CLinks)** | ★☆ | S | Defer | Clusters are niche; completeness-only. |
| **Extras decomposition toolbox** | ★★★ | **L+** | Won't do | VHACD is library-scale work; industry practice is offline decomposition with runtime consumption of ConvexHull + Compound (already supported) — point the docs at it. |

## Suggested order

~~kinematic → compound child transforms (same round) → ghost list → rolling
friction~~ — all four shipped (2026-09-04 / 2026-09-07). ~~motion states and
mb/soft event sources (batch 2)~~ shipped 2026-09-07. ~~Minimal debug draw
(P3)~~ — shipped 2026-09-09 per `docs/design/debug-draw.md`. ~~Persistent-
manifold extras, btShapeHull, character directional jump~~ — shipped
2026-09-10 (the whole on-demand batch closed out together). Nothing left
in the table: the next project milestone is the **1.0 API freeze
review**; every remaining row is a Defer or Won't-do verdict.

## Key observation

The highest-frequency items are all **physical-behavior correctness** gaps
(kinematic / child transforms / rolling friction), while the two L-size items
(serialize, decomposition toolbox) are **ecosystem/workflow** items — for a
library, consumers building their own is the better deal.
