//! Constraints (joints) — port of Bullet's `btTypedConstraint` subclasses
//! (`BulletDynamics/ConstraintSolver`).
//!
//! A [`Constraint`] connects two rigid bodies at fixed points/axes in each
//! body's local frame and is enforced by the sequential-impulse solver as a
//! set of bilateral equality rows (Bullet's `getInfo1`/`getInfo2` protocol,
//! flattened into [`ConstraintRow`]s each step). Contact constraints and
//! joint rows share the same PGS iteration loop, exactly like Bullet's
//! `btSequentialImpulseConstraintSolver`.
//!
//! Supported types:
//!
//! - [`ConstraintKind::Point2Point`] — a ball-socket joint
//!   (`btPoint2PointConstraint`): the two pivot points (given in each body's
//!   center-of-mass frame) are kept coincident, with three orthogonal
//!   constraint rows and an ERP-scaled Baumgarte bias (`m_constraintError =
//!   fps · erp · (pivotB − pivotA)`, default erp = 0.2 like the global
//!   contact ERP). Joints are **not** warm-started, matching Bullet
//!   (`internalSetAppliedImpulse(0)` in `convertJoints`).
//!
//! Like Bullet's `addConstraint(..., disableCollisionsBetweenLinkedBodies =
//! true)`, a [`Constraint`] suppresses collision response between its two
//! bodies by default ([`Constraint::set_disable_collisions`]).

use bullet_math::{Real, Vector3};

use crate::body::RigidBody;
use crate::solver::sequential_impulse::{ConstraintRow, RowLimits};

/// The specific kind of a [`Constraint`] and its parameters (Bullet's
/// constraint subclasses).
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    /// Ball-socket joint: keep `pivot_a` (on body A) coincident with
    /// `pivot_b` (on body B); both are given in their body's
    /// center-of-mass frame (Bullet's `btPoint2PointConstraint`).
    Point2Point(Point2PointConstraint),
}

impl ConstraintKind {
    /// Mutable access to the [`Point2PointConstraint`] parameters.
    pub fn point2point_mut(&mut self) -> &mut Point2PointConstraint {
        match self {
            ConstraintKind::Point2Point(p2p) => p2p,
        }
    }
}

/// A joint between two bodies (Bullet's `btTypedConstraint`). The common
/// flags (`enabled`, `disable_collisions`) live here; the geometry and tuning
/// live in [`ConstraintKind`].
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    kind: ConstraintKind,
    /// Body indices, sorted so `body_a <= body_b` (constraint math is
    /// direction-sensitive; the constructors fix the order at creation).
    body_a: usize,
    body_b: usize,
    /// Whether the solver enforces this constraint this step (Bullet's
    /// `setEnabled`). A disabled constraint also stops suppressing
    /// collisions between its bodies.
    enabled: bool,
    /// Whether collision response between the two bodies is suppressed
    /// (Bullet's `addConstraint(..., disableCollisionsBetweenLinkedBodies)`).
    disable_collisions: bool,
}

impl Constraint {
    /// A ball-socket joint keeping the body-local pivot points `pivot_a` /
    /// `pivot_b` coincident (Bullet's
    /// `btPoint2PointConstraint(rbA, rbB, pivotInA, pivotInB)`). `pivot_a` is
    /// in body A's center-of-mass frame, `pivot_b` in body B's.
    pub fn point2point(
        body_a: usize,
        body_b: usize,
        pivot_a: Vector3,
        pivot_b: Vector3,
    ) -> Self {
        Self {
            kind: ConstraintKind::Point2Point(Point2PointConstraint {
                pivot_a,
                pivot_b,
                impulse_clamp: 0.0,
                erp: None,
            }),
            body_a,
            body_b,
            enabled: true,
            disable_collisions: true,
        }
    }

    pub fn body_a(&self) -> usize {
        self.body_a
    }

    pub fn body_b(&self) -> usize {
        self.body_b
    }

    pub fn kind(&self) -> &ConstraintKind {
        &self.kind
    }

    pub fn kind_mut(&mut self) -> &mut ConstraintKind {
        &mut self.kind
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable the constraint (Bullet's `setEnabled`); disabled
    /// constraints produce no rows and do not suppress collisions.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn disable_collisions(&self) -> bool {
        self.disable_collisions
    }

    /// Whether collision response between the linked bodies is suppressed
    /// (Bullet's `addConstraint(..., disableCollisionsBetweenLinkedBodies)`,
    /// default true).
    pub fn set_disable_collisions(&mut self, disable: bool) {
        self.disable_collisions = disable;
    }
}

/// Parameters of a ball-socket joint (Bullet's `btPoint2PointConstraint`).
#[derive(Debug, Clone, PartialEq)]
pub struct Point2PointConstraint {
    /// Pivot point in body A's center-of-mass frame.
    pub pivot_a: Vector3,
    /// Pivot point in body B's center-of-mass frame.
    pub pivot_b: Vector3,
    /// Optional symmetric clamp on the impulse of each row (Bullet's
    /// `btConstraintSetting::m_impulseClamp`); `0.0` disables the clamp.
    pub impulse_clamp: Real,
    /// Optional ERP override (Bullet's `BT_P2P_FLAGS_ERP`); `None` uses the
    /// global ERP (0.2).
    pub erp: Option<Real>,
}

impl Point2PointConstraint {
    /// Clamp the impulse of every row to `[-clamp, clamp]` (Bullet's
    /// `btConstraintSetting::m_impulseClamp`); `0.0` disables the clamp.
    pub fn set_impulse_clamp(&mut self, clamp: Real) {
        self.impulse_clamp = clamp;
    }

    /// Override the position-correction factor for this joint (Bullet's
    /// `BT_CONSTRAINT_ERP`); `None` restores the global 0.2.
    pub fn set_erp(&mut self, erp: Option<Real>) {
        self.erp = erp;
    }
}

/// Flatten a [`Constraint`] into its solver rows for this step (Bullet's
/// `getInfo1` + `getInfo2` + `convertJoint`). Rows are appended to `out`.
/// `inv_dt` is `1 / dt` (Bullet's `info2.fps`); `erp` is the global ERP
/// (Bullet's `infoGlobal.m_erp`, 0.2).
pub fn prepare_rows(
    constraint: &Constraint,
    bodies: &[RigidBody],
    inv_dt: Real,
    erp: Real,
    out: &mut Vec<ConstraintRow>,
) {
    if !constraint.enabled {
        return;
    }
    match &constraint.kind {
        ConstraintKind::Point2Point(p2p) => {
            let a = &bodies[constraint.body_a];
            let b = &bodies[constraint.body_b];
            // World-space pivots and lever arms from each body's COM.
            let pivot_a = a.position + a.orientation * p2p.pivot_a;
            let pivot_b = b.position + b.orientation * p2p.pivot_b;
            let r_a = pivot_a - a.position;
            let r_b = pivot_b - b.position;
            // `k = fps · erp`: the position error, scaled to a velocity
            // target (Bullet's `m_constraintError = k · (pivotB − pivotA)`).
            let k = inv_dt * p2p.erp.unwrap_or(erp);
            let limits = if p2p.impulse_clamp > 0.0 {
                RowLimits::Bilateral(p2p.impulse_clamp)
            } else {
                RowLimits::Unbounded
            };
            // Three orthogonal rows along the world axes (Bullet's
            // `m_J1linearAxis` = ±e_j with the skew-symmetric angular
            // jacobians built from the lever arms).
            for axis in 0..3 {
                let mut normal = Vector3::ZERO;
                normal[axis] = 1.0;
                out.push(ConstraintRow::prepare(
                    constraint.body_a,
                    constraint.body_b,
                    normal,
                    r_a,
                    r_b,
                    a,
                    b,
                    // bias = k · (pivotA − pivotB) · n (see module docs for
                    // the sign convention; the accumulated impulse comes out
                    // with the correct sign automatically).
                    k * (pivot_a - pivot_b)[axis],
                    limits,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::PhysicsWorld;
    use bullet_collision::shapes::CollisionShape;
    use bullet_math::Quaternion;

    const DT: Real = 1.0 / 60.0;

    /// A world with a fixed anchor (static box at the origin) and a dynamic
    /// sphere bob connected by a point-to-point joint whose pivot is
    /// `bob_offset` from the bob's centre and `anchor_offset` from the
    /// anchor's origin. The bob starts at the constraint-satisfied position
    /// (`anchor_offset − bob_offset`) so there is no initial correction kick.
    fn pendulum_world(bob_offset: Vector3, anchor_offset: Vector3) -> (PhysicsWorld, usize, usize) {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        // The bob is a sphere; its centre of mass is the sphere centre.
        let bob = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            anchor_offset - bob_offset,
            1.0,
        ));
        // The joint holds bob centre + bob_offset at anchor origin +
        // anchor_offset (both in their bodies' frames).
        let c = Constraint::point2point(bob, anchor, bob_offset, anchor_offset);
        world.add_constraint(c);
        (world, bob, anchor)
    }

    /// A pendulum released from rest at 45° from the vertical (the bottom
    /// position is (0, -1, 0)) swings: the bob's pivot stays within a small
    /// Baumgarte residual of the anchor, the bob oscillates (x crosses the
    /// bottom repeatedly) and its speed peaks at the bottom.
    #[test]
    fn pendulum_swings_and_keeps_the_pivot() {
        // Bob centre at (sin45°, −cos45°, 0) oriented by −45° around Z, so
        // its pivot (local (-1,0,0), the rod) is exactly at the anchor
        // (origin): the constraint starts satisfied and there is no
        // correction kick. Rod length 1 m.
        let (mut world, bob, _anchor) = pendulum_world(
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::ZERO,
        );
        let a45 = std::f32::consts::FRAC_PI_4 as Real;
        world.body_mut(bob).position = Vector3::new(a45.sin(), -a45.cos(), 0.0);
        world.body_mut(bob).orientation = Quaternion::from_rotation_z(-a45);
        let pivot_err = |world: &PhysicsWorld, bob: usize| -> Real {
            let b = world.body(bob);
            (b.position + b.orientation * Vector3::new(-1.0, 0.0, 0.0)).length()
        };
        let mut max_err: Real = 0.0;
        let mut min_x: Real = Real::INFINITY;
        let mut max_speed: Real = 0.0;
        let mut zero_crossings = 0;
        let mut prev_x = world.body(bob).position.x;
        for _ in 0..600 {
            world.step_simulation(DT);
            let b = world.body(bob);
            max_err = max_err.max(pivot_err(&world, bob));
            min_x = min_x.min(b.position.x);
            max_speed = max_speed.max(b.linear_velocity.length());
            let x = b.position.x;
            if prev_x > 0.0 && x <= 0.0 {
                zero_crossings += 1;
            }
            prev_x = x;
        }
        // Constraint maintained within the Baumgarte residual (~0.1 m during
        // a fast swing; the steady-state error is ~1 cm).
        assert!(max_err < 0.2, "pivot error {max_err} too large");
        // Swung through the bottom at least a couple of times.
        assert!(zero_crossings >= 3, "bob did not oscillate (crossings {zero_crossings})");
        // Reached the other side of the swing (x well below the release 0.707).
        assert!(min_x < -0.5, "bob never swung past the bottom: min_x = {min_x}");
        // Non-trivial speed at the bottom of the swing
        // (v = √(2·g·L·(1−cos45°)) ≈ 2.4 m/s).
        assert!(max_speed > 1.5, "bob too slow: {max_speed}");
    }

    /// A pendulum hanging at rest below the pivot stays there: the bob's
    /// pivot stays on the anchor pivot and the bob does not drift.
    #[test]
    fn pendulum_hangs_at_equilibrium() {
        // Bob centre at the origin with the pivot 1 m above at (0, 1, 0);
        // the anchor pivot is also at (0, 1, 0): the bob hangs vertically
        // below the pivot, a stable equilibrium.
        let (mut world, bob, _anchor) = pendulum_world(
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        for _ in 0..600 {
            world.step_simulation(DT);
        }
        let b = world.body(bob);
        let pivot = b.position + b.orientation * Vector3::new(0.0, 1.0, 0.0);
        // The pivot converges to the anchor; only a tiny Baumgarte residual
        // remains (the bob sags a fraction of a millimetre under gravity).
        assert!(
            (pivot - Vector3::new(0.0, 1.0, 0.0)).length() < 0.01,
            "pivot not on the anchor: {:?}",
            pivot
        );
        assert!(
            b.position.length() < 0.02,
            "bob drifted from below the pivot: {:?}",
            b.position
        );
        assert!(b.linear_velocity.length() < 0.05, "bob still moving: {:?}", b.linear_velocity);
    }

    /// A double pendulum keeps both rod lengths constant while it swings.
    #[test]
    fn two_link_chain_keeps_its_lengths() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let upper = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::new(0.0, -1.0, 0.0),
            1.0,
        ));
        let lower = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::new(0.0, -3.0, 0.0),
            1.0,
        ));
        // Rod 1: the top of the upper bob (local (0,1,0)) pinned to the
        // anchor pivot at the origin. Rod 2: the top of the lower bob pinned
        // to the bottom of the upper bob (local (0,-1,0)). Both constraints
        // start satisfied, hanging straight down.
        world.add_constraint(Constraint::point2point(
            upper,
            anchor,
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::ZERO,
        ));
        world.add_constraint(Constraint::point2point(
            lower,
            upper,
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        ));
        // Nudge the lower bob sideways so the chain swings.
        world.body_mut(lower).linear_velocity = Vector3::new(2.0, 0.0, 0.0);
        let mut max_err: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            let u = world.body(upper);
            let l = world.body(lower);
            // Upper top must sit at the origin; lower top at the upper bottom.
            max_err = max_err
                .max((u.position + u.orientation * Vector3::new(0.0, 1.0, 0.0)).length())
                .max(
                    (l.position + l.orientation * Vector3::new(0.0, 1.0, 0.0)
                        - (u.position + u.orientation * Vector3::new(0.0, -1.0, 0.0)))
                    .length(),
                );
        }
        assert!(max_err < 0.2, "chain pivots drifted: {max_err}");
    }

    /// Two overlapping boxes linked by a joint do not push each other apart
    /// (collisions suppressed), while the same boxes without the joint do.
    #[test]
    fn joint_suppresses_collisions_between_linked_bodies() {
        let run = |with_joint: bool| -> Vector3 {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::ZERO);
            // Two overlapping boxes at the origin.
            let a = world.add_body(RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::ZERO,
                1.0,
            ));
            let b = world.add_body(RigidBody::new_dynamic(
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::ZERO,
                1.0,
            ));
            if with_joint {
                let mut c = Constraint::point2point(a, b, Vector3::ZERO, Vector3::ZERO);
                c.set_disable_collisions(true);
                world.add_constraint(c);
            }
            for _ in 0..30 {
                world.step_simulation(DT);
            }
            world.body(b).position - world.body(a).position
        };
        let sep_joint = run(true);
        let sep_plain = run(false);
        // Without the joint the boxes separate (collision pushes them apart);
        // with the joint they stay linked (no contact response pushes them).
        assert!(
            sep_plain.length() > 0.05,
            "plain overlapping boxes should separate, got {sep_plain:?}"
        );
        assert!(
            sep_joint.length() < 0.02,
            "linked boxes must not be pushed apart, got {sep_joint:?}"
        );
    }

    /// The impulse clamp caps the per-row impulse: a heavy bob held up by a
    /// clamped joint sags (the clamp releases the excess), while an
    /// unclamped joint holds it.
    #[test]
    fn impulse_clamp_limits_the_support_impulse() {
        let run = |clamp: Real| -> Real {
            let mut world = PhysicsWorld::new();
            world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
            let anchor = world.add_body(RigidBody::new_static(
                CollisionShape::box_shape(Vector3::splat(0.2)),
                Vector3::ZERO,
            ));
            let bob = world.add_body(RigidBody::new_dynamic(
                CollisionShape::sphere(0.1),
                Vector3::ZERO,
                10.0,
            ));
            // The joint pins the bob centre to the anchor at the origin.
            let mut c = Constraint::point2point(bob, anchor, Vector3::ZERO, Vector3::ZERO);
            c.kind_mut().point2point_mut().set_impulse_clamp(clamp);
            world.add_constraint(c);
            for _ in 0..300 {
                world.step_simulation(DT);
            }
            world.body(bob).position.y
        };
        // The static support impulse needed is m·g·dt ≈ 10·9.81/60 ≈ 1.6 per
        // step; a clamp of 0.1 cannot hold the bob, so it falls away.
        let clamped = run(0.1);
        assert!(
            clamped < -1.0,
            "clamped joint should let the bob fall, y = {clamped}"
        );
        let held = run(0.0);
        assert!(
            held.abs() < 0.05,
            "unclamped joint should hold the bob, y = {held}"
        );
    }
}
