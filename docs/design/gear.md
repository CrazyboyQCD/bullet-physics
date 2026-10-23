# Rigid gear constraint (btGearConstraint)

Status: approved for implementation (2026-10-23, freeze-review row
"Rigid `btGearConstraint`").

## Upstream inventory

`BulletDynamics/ConstraintSolver/btGearConstraint.{h,cpp}` — the entire
behavioral surface:

- `getInfo1`: always 1 row (`nub = 1`).
- `getInfo2`: `globalAxisA = R_A · m_axisInA`, `globalAxisB = R_B · m_axisInB`
  (axes are **local**, re-rotated every step), then
  `m_J1angularAxis = globalAxisA`, `m_J2angularAxis = m_ratio · globalAxisB`.
  Nothing else is written: no linear jacobians, no `m_constraintError`
  (stays at the `convertJoint` memset zero → pure velocity coupling
  `ω_A·axisA + ratio·ω_B·axisB = 0` with no position correction), no limits.
- Setters `setAxisA` / `setAxisB` / `setRatio` + getters.
- `m_useFrameA` is declared and never read (dead field — dropped).
- `setParam` / `getParam`: `btAssert(0)` stubs (nothing to port — we have no
  typed-param surface on rigid constraints).
- Serialization blocks: out of scope (blanket no-serialization deviation).

## Port shape

- `multibody/constraint/gear.rs` already ports the multibody variant; this is
  the rigid sibling. New `constraint/gear.rs` in `bullet-dynamics`:
  `GearConstraint { axis_a: Vector3, axis_b: Vector3, ratio: Real }` with
  setters/getters (upstream's are post-construction mutable).
- `ConstraintKind::Gear` + `kind_name` "gear" + accessors `gear()` /
  `gear_mut()` + `Constraint::gear(body_a, body_b, axis_a, axis_b, ratio)`
  factory (no world-level factory: unlike hinge-2/universal the constructor
  reads no body poses — axes are local; users add via
  `world.add_constraint`).
- `prepare_rows` arm: one `ConstraintRow::prepare` with
  `lin_a = lin_b = ZERO`, `ang_a = R_A·axis_a`, `ang_b = −ratio·R_B·axis_b`,
  `bias = 0`, `RowLimits::Unbounded`. Our row convention (row velocity
  `J_b·v_b − J_a·v_a`, impulse `−λ` on A / `+λ` on B) makes the coupling
  equation need the negated B jacobian to match upstream's
  both-plus-apply convention; the zero target makes the fixed point
  sign-free.
- `ConstraintTag::Gear` + debug-draw arm (draw nothing geometric — upstream
  has no visualizer either; the tag exists for renderer routing, consistent
  with the other tags).

## Reference decisions covered

| Upstream decision | Port |
|---|---|
| 1 row, `nub = 1` | one row per step |
| angular-only jacobians, zero linear | `lin = ZERO` both sides |
| axes stored local, rotated per step | `R·axis` in `prepare_rows` |
| `m_J2angularAxis = ratio·globalAxisB` | `ang_b = ratio·R_B·axis_b` |
| no `m_constraintError` write → zero bias | `bias = 0` (pure velocity coupling) |
| no limits (`±inf` from `convertJoint` init) | `RowLimits::Unbounded` |
| `m_useFrameA` dead | dropped |
| `setParam/getParam` assert stubs | not ported (no param surface) |
| serialization | blanket deviation |

## Tests

- Coupling: two flywheels (one spun up), free otherwise → steady state
  `ω_B·axisB = −ω_A·axisA / ratio` (hand-computed, ratio 2).
- Zero bias: offset bodies (translated) do NOT get pulled — velocity-only.
- Local-axis rotation: body A rotated 90° → coupling axis follows.
- Setters round-trip; `kind_name`; disabled constraint no-ops.
- Static body on either side degenerates gracefully (K one-sided).

## Verification

- Probe: add a `GEAR` scene to `tools/hinge2_universal_probe.cpp` + the Rust
  twin (two dynamic spheres, axisA=Z on body A spun up, ratio 2) — same
  `%.17g` protocol, mbdiff vs the real solver.
- Full gate + parity anchors bit-identical (additive change only).
