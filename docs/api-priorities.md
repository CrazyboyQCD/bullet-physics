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
| **Persistent-manifold extras** | ★★☆ | S | Merge | Do together with the P3 debug draw (contact-point iteration is a debug-draw prerequisite). |
| **btShapeHull** | ★★☆ | S | On demand | quickhull already exists (soft clusters use it); a wrapper away; debugging / simplification use. |
| **Character directional jump** | ★★ | S | On demand | Industry practice is user-written character controllers; the built-in character has a narrow audience. |
| **Serialize** | ★★ | L | Defer | Savegames work with handwritten pose snapshots (bodies are all plain data); bit-exact round-trip has no consumer ecosystem — post-release. |
| **Anisotropic friction** | ★★ | M | Defer | Rare; conveyor-belt cases are better served by surface velocity. |
| **Contact stiffness/damping** | ★★ | M | Defer | Soft contacts are niche; record in deviations, pull on demand. |
| **Additional damping** | ★☆ | S | Defer | A humanoid-stack tuning safety valve; scenario-specific. |
| **Multi-part triangle meshes** | ★★ | M | Defer | Vertex/index concatenation is a workaround with no runtime benefit. |
| **Soft cluster core (CLinks)** | ★☆ | S | Defer | Clusters are niche; completeness-only. |
| **Extras decomposition toolbox** | ★★★ | **L+** | Won't do | VHACD is library-scale work; industry practice is offline decomposition with runtime consumption of ConvexHull + Compound (already supported) — point the docs at it. |

## Suggested order

~~kinematic → compound child transforms (same round) → ghost list → rolling
friction~~ — all four shipped (2026-09-04 / 2026-09-07). ~~Next: minimal
debug draw~~ — motion states and mb/soft event sources (batch 2) shipped
first on user request (2026-09-07). Next: **minimal debug draw** (P3) with
persistent-manifold extras folded in (see "Merge" verdict).

## Key observation

The highest-frequency items are all **physical-behavior correctness** gaps
(kinematic / child transforms / rolling friction), while the two L-size items
(serialize, decomposition toolbox) are **ecosystem/workflow** items — for a
library, consumers building their own is the better deal.
