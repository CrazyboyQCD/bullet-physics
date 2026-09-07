//! Constraint joint tests (moved verbatim from `constraint.rs`).

use super::*;
use super::conetwist::quaternion_angle;

#[cfg(test)]
    use crate::world::PhysicsWorld;
    use bullet_collision::shapes::CollisionShape;
    use bullet_math::Quaternion;

    const DT: Real = 1.0 / 60.0;

    /// A world with a fixed anchor (static box at the origin) and a dynamic
    /// sphere bob connected by a point-to-point joint whose pivot is
    /// `bob_offset` from the bob's centre and `anchor_offset` from the
    /// anchor's origin. The bob starts at the constraint-satisfied position
    /// (`anchor_offset − bob_offset`) so there is no initial correction kick.
    fn pendulum_world(
        bob_offset: Vector3,
        anchor_offset: Vector3,
    ) -> (PhysicsWorld, BodyHandle, BodyHandle) {
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
        let pivot_err = |world: &PhysicsWorld, bob: BodyHandle| -> Real {
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

    /// A door hinged at its edge swings like a pendulum around the vertical
    /// hinge axis: the pivot stays on the anchor, the hinge axes stay
    /// parallel, and the hinge angle oscillates.
    #[test]
    fn hinge_swings_and_keeps_axis_aligned() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        // A door (box) hinged at its edge (local (-0.5,0,0)) to the anchor
        // at the origin, opened 45° around the vertical (Z) hinge.
        let door = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
            Vector3::new(0.0, 0.5, 0.0),
            2.0,
        ));
        let a45 = std::f32::consts::FRAC_PI_4 as Real;
        world.body_mut(door).position = Vector3::new(a45.cos() * 0.5, a45.sin() * 0.5, 0.0);
        world.body_mut(door).orientation = Quaternion::from_rotation_z(a45);
        // Hinge: the door's edge (local (-0.5,0,0)) pinned to the anchor
        // pivot at the origin, axis (0,0,1) on both bodies.
        let hinge = Constraint::hinge(
            door,
            anchor,
            Vector3::new(-0.5, 0.0, 0.0),
            Vector3::Z,
            Vector3::ZERO,
            Vector3::Z,
        );
        world.add_constraint(hinge);

        let mut max_pivot_err: Real = 0.0;
        let mut min_align: Real = 1.0;
        let mut max_angle: Real = 0.0;
        let mut zero_crossings = 0;
        let mut prev_angle = 0.0;
        for step in 0..600 {
            world.step_simulation(DT);
            let d = world.body(door);
            let pivot = d.position + d.orientation * Vector3::new(-0.5, 0.0, 0.0);
            max_pivot_err = max_pivot_err.max(pivot.length());
            // Hinge axes: the door's local Z rotated to world, the anchor's Z.
            let ax1 = d.orientation * Vector3::Z;
            min_align = min_align.min(ax1.dot(Vector3::Z).abs());
            let angle = world.constraint(0).kind().hinge().hinge_angle();
            max_angle = max_angle.max(angle.abs());
            if step > 0 && prev_angle > 0.0 && angle <= 0.0 {
                zero_crossings += 1;
            }
            prev_angle = angle;
        }
        assert!(max_pivot_err < 0.2, "door pivot drifted: {max_pivot_err}");
        assert!(min_align > 0.99, "hinge axes misaligned: {min_align}");
        assert!(max_angle > 0.5, "door barely swung: max angle {max_angle}");
        assert!(zero_crossings >= 3, "door did not oscillate: {zero_crossings} crossings");
    }

    /// A door with angular limits stops at the limit instead of swinging
    /// through: the hinge angle never exceeds the range and settles on it.
    #[test]
    fn hinge_limit_stops_the_door() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let door = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
            Vector3::ZERO,
            2.0,
        ));
        // Open the door 45° (well past the 30° limit) and release it.
        let a45 = std::f32::consts::FRAC_PI_4 as Real;
        world.body_mut(door).position = Vector3::new(a45.cos() * 0.5, a45.sin() * 0.5, 0.0);
        world.body_mut(door).orientation = Quaternion::from_rotation_z(a45);
        let mut hinge = Constraint::hinge(
            door,
            anchor,
            Vector3::new(-0.5, 0.0, 0.0),
            Vector3::Z,
            Vector3::ZERO,
            Vector3::Z,
        );
        hinge.hinge_mut().set_limit(-0.3, 0.3);
        world.add_constraint(hinge);
        let mut min_angle: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            let angle = world.constraint(0).kind().hinge().hinge_angle();
            min_angle = min_angle.min(angle);
        }
        // The door starts at +45° (past the +0.3 limit, so the initial state
        // violates it by construction) and swings down under gravity; the
        // low limit at −0.3 stops it. It must never swing past the low limit
        // and must settle resting on it.
        assert!(min_angle > -0.35, "door passed the low limit: {min_angle}");
        let final_angle = world.constraint(0).kind().hinge().hinge_angle();
        assert!(
            (final_angle + 0.3).abs() < 0.05,
            "door did not settle on the low limit: {final_angle}"
        );
    }

    /// A motorized hinge rotates the door at the target angular velocity.
    #[test]
    fn hinge_motor_rotates_the_door() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let door = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.2)),
            Vector3::new(0.5, 0.0, 0.0),
            2.0,
        ));
        let mut hinge = Constraint::hinge(
            door,
            anchor,
            Vector3::new(-0.5, 0.0, 0.0),
            Vector3::Z,
            Vector3::ZERO,
            Vector3::Z,
        );
        hinge.hinge_mut().enable_motor(1.0, 1000.0);
        world.add_constraint(hinge);
        for _ in 0..180 {
            world.step_simulation(DT);
        }
        // 3 s at 1 rad/s ≈ 3 rad of rotation.
        let angle = world.constraint(0).kind().hinge().hinge_angle();
        assert!(
            (angle - 3.0).abs() < 1.0,
            "motor did not rotate the door at 1 rad/s: {angle}"
        );
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
            c.point2point_mut().set_impulse_clamp(clamp);
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

    /// A fixed joint welds a box to a static anchor: an initial spin and
    /// sideways push are killed, the relative orientation returns to
    /// identity and the pivot stays on the anchor under gravity.
    #[test]
    fn fixed_joint_locks_a_rigid_assembly() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        // A box hanging 1 m below the anchor; both pivots sit 0.5 above each
        // body's centre, coinciding at (0, 1.5, 0) at assembly.
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        world.add_constraint(Constraint::fixed(
            box_body,
            anchor,
            JointFrame::new(Vector3::new(0.0, 0.5, 0.0), Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(0.0, 1.5, 0.0), Matrix3::IDENTITY),
        ));
        // Kick it: a spin around Z and a sideways velocity.
        world.body_mut(box_body).angular_velocity = Vector3::new(0.0, 0.0, 4.0);
        world.body_mut(box_body).linear_velocity = Vector3::new(2.0, 0.0, 0.0);
        let mut max_rel_angle: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            // The anchor's frame is identity, so the relative orientation is
            // the box's own orientation.
            max_rel_angle = max_rel_angle.max(quaternion_angle(&world.body(box_body).orientation));
        }
        let b = world.body(box_body);
        // The weld holds: the box returns to the anchor's orientation and
        // hangs straight below the pivot at (0, 1, 0).
        assert!(
            max_rel_angle < 0.5,
            "relative orientation wandered during the kick: {max_rel_angle}"
        );
        assert!(
            (b.position - Vector3::new(0.0, 1.0, 0.0)).length() < 0.03,
            "box drifted from below the anchor: {:?}",
            b.position
        );
        assert!(
            b.angular_velocity.length() < 0.05,
            "box still spinning: {:?}",
            b.angular_velocity
        );
        let pivot = b.position + b.orientation * Vector3::new(0.0, 0.5, 0.0);
        assert!(
            (pivot - Vector3::new(0.0, 1.5, 0.0)).length() < 0.03,
            "pivot off the anchor: {:?}",
            pivot
        );
    }

    /// Two boxes welded side by side tumble and land on the ground as one
    /// rigid unit: their relative pose stays fixed even under impact.
    #[test]
    fn fixed_joint_welds_two_dynamic_bodies() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // Two boxes, one above the other with a small gap, welded by a fixed
        // joint whose frames meet halfway between them.
        let a = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.4, 0.4, 0.4)),
            Vector3::new(0.0, 1.2, 0.0),
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::new(0.4, 0.4, 0.4)),
            Vector3::new(0.0, 1.6, 0.0),
            1.0,
        ));
        world.add_constraint(Constraint::fixed(
            a,
            b,
            JointFrame::new(Vector3::new(0.0, 0.2, 0.0), Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(0.0, -0.2, 0.0), Matrix3::IDENTITY),
        ));
        // Tumble the pair a little.
        world.body_mut(a).angular_velocity = Vector3::new(1.0, 2.0, 0.0);
        world.body_mut(b).angular_velocity = Vector3::new(1.0, 2.0, 0.0);
        let mut max_rel_angle: Real = 0.0;
        let mut max_pivot_err: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            let ba = world.body(a);
            let bb = world.body(b);
            let rel_quat = quat_from_mat3(&Matrix3::from_quat(bb.orientation)).inverse()
                * quat_from_mat3(&Matrix3::from_quat(ba.orientation));
            max_rel_angle = max_rel_angle.max(quaternion_angle(&rel_quat));
            // The weld residual: the two frame origins must stay coincident
            // (their world positions differ by the rigid offset).
            let pivot_a = ba.position + ba.orientation * Vector3::new(0.0, 0.2, 0.0);
            let pivot_b = bb.position + bb.orientation * Vector3::new(0.0, -0.2, 0.0);
            max_pivot_err = max_pivot_err.max((pivot_b - pivot_a).length());
        }
        assert!(
            max_rel_angle < 0.15,
            "welded pair twisted apart: {max_rel_angle}"
        );
        assert!(
            max_pivot_err < 0.15,
            "welded pair stretched apart: {max_pivot_err}"
        );
    }

    /// A cone-twist joint limits the swing: a bob pinned at the anchor and
    /// spun around Z swings against the cone and the angle between the cone
    /// axes never exceeds the swing span (plus the Baumgarte overshoot).
    #[test]
    fn cone_twist_limits_the_swing() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let bob = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::ZERO,
            1.0,
        ));
        // Cone axis = X on both bodies; both pivots at the origin (the bob's
        // centre is pinned to the anchor). Swing limited to 25°, twist free.
        let cone = Constraint::cone_twist(
            bob,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            0.44,
            0.44,
            -1.0,
        );
        world.add_constraint(cone);
        // Spin the bob around Z: its X axis swings out of the cone.
        world.body_mut(bob).angular_velocity = Vector3::new(0.0, 0.0, 4.0);
        let mut max_swing: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            // The swing angle is the angle between the cone axes (X).
            let ax_a = world.body(anchor).orientation * Vector3::X;
            let ax_b = world.body(bob).orientation * Vector3::X;
            max_swing = max_swing.max(ax_a.dot(ax_b).clamp(-1.0, 1.0).acos());
        }
        assert!(
            max_swing < 0.55,
            "cone exceeded: swing angle {max_swing} rad (limit 0.44)"
        );
        // It actually swung (moved out of the hanging position and back).
        assert!(max_swing > 0.3, "bob barely swung: {max_swing}");
    }

    /// A cone-twist joint with a locked twist span (0) stops the twist: a
    /// spin around the cone axis is killed and the twist angle never grows.
    #[test]
    fn cone_twist_locks_the_twist() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let bob = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.1),
            Vector3::ZERO,
            1.0,
        ));
        // Free swing, locked twist (span 0): the bob can swing but not twist.
        let cone = Constraint::cone_twist(
            bob,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            0.9,
            0.9,
            0.0,
        );
        world.add_constraint(cone);
        // Spin the bob around the cone axis (X).
        world.body_mut(bob).angular_velocity = Vector3::new(6.0, 0.0, 0.0);
        let mut max_twist: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            max_twist = max_twist.max(world.constraint(0).kind().cone_twist().twist_angle());
        }
        assert!(
            max_twist < 0.1,
            "locked twist drifted: {max_twist} rad"
        );
        assert!(
            world.body(bob).angular_velocity.x.abs() < 0.1,
            "bob still twisting: {:?}",
            world.body(bob).angular_velocity
        );
    }

    /// A slider joint (free linear travel, locked rotation): a box pushed
    /// sideways with a spin slides along the axis, stays on the axis, and
    /// keeps its orientation.
    #[test]
    fn slider_slides_along_its_axis() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        ));
        // Slider along X with the default limits (free linear, locked
        // angular); the box centre is 1 m along the axis, inside the limits.
        // The static anchor is the reference body A (the motor and limits
        // act on body B relative to body A, like Bullet).
        world.add_constraint(Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        ));
        // Push it along the axis with a sideways drift and a spin.
        world.body_mut(box_body).linear_velocity = Vector3::new(1.5, 0.5, 0.2);
        world.body_mut(box_body).angular_velocity = Vector3::new(0.0, 0.0, 3.0);
        let mut max_off_axis: Real = 0.0;
        let mut max_rel_angle: Real = 0.0;
        for _ in 0..180 {
            world.step_simulation(DT);
            let p = world.body(box_body).position;
            max_off_axis = max_off_axis.max(p.y.abs()).max(p.z.abs());
            max_rel_angle = max_rel_angle
                .max(quaternion_angle(&world.body(box_body).orientation));
        }
        let p = world.body(box_body).position;
        assert!(p.x > 1.1, "box did not slide: x = {}", p.x);
        assert!(max_off_axis < 0.05, "box drifted off the axis: {max_off_axis}");
        assert!(max_rel_angle < 0.15, "slider allowed rotation: {max_rel_angle}");
    }

    /// A slider with linear limits stops the box at the upper limit: it
    /// slides until the offset reaches the limit and never passes it.
    #[test]
    fn slider_limit_stops_the_slide() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.2, 0.0, 0.0),
            1.0,
        ));
        let mut slider = Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        slider.slider_mut().set_lin_limits(-0.3, 0.3);
        world.add_constraint(slider);
        // Push the box hard along the axis toward the upper limit. The
        // slider limit is a velocity bias (Bullet's `m_constraintError =
        // k·depth`): it stops the box at the limit and rebounds it, so the
        // box can never travel far past the limit, and it coasts back
        // inside once the row deactivates (no friction in this test).
        world.body_mut(box_body).linear_velocity = Vector3::new(4.0, 0.0, 0.0);
        let mut max_x: Real = 0.0;
        let mut limit_fired = false;
        for _ in 0..180 {
            world.step_simulation(DT);
            let s = world.constraint(0).kind().slider();
            limit_fired |= s.solve_lin_limit;
            max_x = max_x.max(world.body(box_body).position.x);
        }
        let final_x = world.body(box_body).position.x;
        // One step of free travel at 4 m/s overshoots to ~0.33; the limit
        // pulls it back from there.
        assert!(max_x < 0.34, "box passed the upper limit: {max_x}");
        assert!(limit_fired, "the limit row never fired");
        // After the rebound the box coasts back inside the range.
        assert!(final_x < 0.3, "box did not return inside the limits: {final_x}");
    }

    /// A slider linear motor drives the box along the axis at the target
    /// velocity.
    #[test]
    fn slider_motor_drives_along_the_axis() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        let mut slider = Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        slider.slider_mut().enable_lin_motor(1.5, 100.0);
        world.add_constraint(slider);
        // 1.5 m/s is above the sleeping threshold (0.8), so the motorized
        // box keeps sliding instead of deactivating after 2 s.
        for _ in 0..150 {
            world.step_simulation(DT);
        }
        let p = world.body(box_body).position;
        // 2.5 s at 1.5 m/s along the axis.
        assert!(
            (p.x - 3.75).abs() < 0.3,
            "motor did not drive the box at 1.5 m/s: x = {p:?}"
        );
        assert!(
            world.body(box_body).linear_velocity.x.abs() < 1.8,
            "box velocity off target: {:?}",
            world.body(box_body).linear_velocity
        );
    }

    /// A slider with angular limits stops the box's rotation around the
    /// axis: the angular position stays inside the limits.
    #[test]
    fn slider_angular_limit_stops_the_twist() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        let mut slider = Constraint::slider(
            anchor,
            box_body,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        slider.slider_mut().set_ang_limits(-0.4, 0.4);
        world.add_constraint(slider);
        // Spin the box around the slider axis.
        world.body_mut(box_body).angular_velocity = Vector3::new(3.0, 0.0, 0.0);
        let mut max_ang: Real = 0.0;
        for _ in 0..180 {
            world.step_simulation(DT);
            let s = world.constraint(0).kind().slider();
            max_ang = max_ang.max(s.ang_pos.abs());
        }
        assert!(max_ang < 0.5, "box passed the angular limit: {max_ang}");
    }

    /// A fully locked generic 6DOF behaves like a fixed joint: an initial
    /// spin and sideways push are killed, the relative orientation returns
    /// to identity and the pivot stays on the anchor.
    #[test]
    fn generic_6dof_locked_is_rigid() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        // All six limits locked at zero (the default linear limits already
        // lock; lock the angular ones too).
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::new(0.0, 0.5, 0.0), Matrix3::IDENTITY),
            JointFrame::new(Vector3::new(0.0, 1.5, 0.0), Matrix3::IDENTITY),
        );
        dof
            .generic_6dof_mut()
            .set_angular_limits(Vector3::ZERO, Vector3::ZERO);
        world.add_constraint(dof);
        world.body_mut(box_body).angular_velocity = Vector3::new(0.0, 0.0, 4.0);
        world.body_mut(box_body).linear_velocity = Vector3::new(2.0, 0.0, 0.0);
        let mut max_rel_angle: Real = 0.0;
        for _ in 0..600 {
            world.step_simulation(DT);
            max_rel_angle =
                max_rel_angle.max(quaternion_angle(&world.body(box_body).orientation));
        }
        let b = world.body(box_body);
        assert!(
            max_rel_angle < 0.5,
            "relative orientation wandered during the kick: {max_rel_angle}"
        );
        assert!(
            (b.position - Vector3::new(0.0, 1.0, 0.0)).length() < 0.03,
            "box drifted from below the anchor: {:?}",
            b.position
        );
    }

    /// A generic 6DOF with one angular limit stops the spin around that
    /// axis: the relative Euler angle stays inside the limit.
    #[test]
    fn generic_6dof_angular_limit_stops_the_spin() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        // Linear locked (default), angular X limited to ±0.4, Y/Z free.
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        dof
            .generic_6dof_mut()
            .set_angular_limits(Vector3::new(-0.4, -1.0, -1.0), Vector3::new(0.4, 1.0, 1.0));
        world.add_constraint(dof);
        // Spin the box around X: the X Euler angle swings against its limit.
        world.body_mut(box_body).angular_velocity = Vector3::new(4.0, 0.0, 0.0);
        let mut max_angle: Real = 0.0;
        for _ in 0..300 {
            world.step_simulation(DT);
            let a = world.constraint(0).kind().generic_6dof().get_angle(0);
            max_angle = max_angle.max(a.abs());
        }
        assert!(max_angle < 0.5, "box passed the angular limit: {max_angle}");
        assert!(max_angle > 0.3, "box barely rotated: {max_angle}");
    }

    /// A generic 6DOF angular motor rotates the joint around the Z euler
    /// axis at the target velocity.
    #[test]
    fn generic_6dof_motor_rotates_an_axis() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::ZERO,
            1.0,
        ));
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        // Angular Z motor at 1 rad/s (axis 5 = angular Z).
        dof.generic_6dof_mut().enable_motor(5, 1.0, 100.0);
        world.add_constraint(dof);
        for _ in 0..180 {
            world.step_simulation(DT);
        }
        let angle = world.constraint(0).kind().generic_6dof().get_angle(2);
        // 3 s at 1 rad/s of relative rotation around the Z euler axis.
        assert!(
            (angle.abs() - 3.0).abs() < 1.0,
            "motor did not rotate the joint at 1 rad/s: angle {angle}"
        );
    }

    /// A generic 6DOF with a linear limit stops the box sliding along that
    /// axis: the pivot offset stays inside the limit.
    #[test]
    fn generic_6dof_linear_limit_stops_the_slide() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let anchor = world.add_body(RigidBody::new_static(
            CollisionShape::box_shape(Vector3::splat(0.2)),
            Vector3::ZERO,
        ));
        let box_body = world.add_body(RigidBody::new_dynamic(
            CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.2, 0.0, 0.0),
            1.0,
        ));
        let mut dof = Constraint::generic_6dof(
            box_body,
            anchor,
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
            JointFrame::new(Vector3::ZERO, Matrix3::IDENTITY),
        );
        // Linear X limited to ±0.3, Y/Z locked (default), angular free
        // (default).
        dof
            .generic_6dof_mut()
            .set_linear_limits(Vector3::new(-0.3, 0.0, 0.0), Vector3::new(0.3, 0.0, 0.0));
        world.add_constraint(dof);
        // Push the box hard along X; the limit is a velocity bias, so the
        // box rebounds and coasts back (no friction, like Bullet).
        world.body_mut(box_body).linear_velocity = Vector3::new(4.0, 0.0, 0.0);
        let mut max_x: Real = 0.0;
        for _ in 0..180 {
            world.step_simulation(DT);
            max_x = max_x.max(world.body(box_body).position.x);
        }
        assert!(max_x < 0.34, "box passed the upper limit: {max_x}");
        assert!(
            world.body(box_body).position.x < 0.3,
            "box did not return inside the limits: {:?}",
            world.body(box_body).position
        );
    }

    /// The delegating [`Constraint`] accessors reach the kind parameters in
    /// one dispatch, and reads go back through `kind()`.
    #[test]
    fn kind_accessors_tune_parameters() {
        let mut c = Constraint::point2point(
            BodyHandle::from_index(0),
            BodyHandle::from_index(1),
            Vector3::ZERO,
            Vector3::ZERO,
        );
        c.point2point_mut().set_impulse_clamp(0.001);
        assert_eq!(c.kind().point2point().impulse_clamp, 0.001);
    }

    /// A mismatched accessor panics naming the found kind.
    #[test]
    #[should_panic = "not a point-to-point constraint (found hinge)"]
    fn wrong_kind_accessor_panics() {
        let mut h = Constraint::hinge(
            BodyHandle::from_index(0),
            BodyHandle::from_index(1),
            Vector3::ZERO,
            Vector3::Y,
            Vector3::ZERO,
            Vector3::Y,
        );
        h.point2point_mut();
    }