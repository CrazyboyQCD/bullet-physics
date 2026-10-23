//! Gear joint coupling two rotation axes (Bullet's `btGearConstraint`).

use super::*;

/// Couples the angular velocities of two bodies around given local axes
/// with a fixed ratio (Bullet's `btGearConstraint`): the one-row constraint
/// `ω_A·axisA + ratio·ω_B·axisB = 0` — with parallel axes this meshes like
/// external gears (opposite spins). Pure velocity coupling: no position
/// error, no limits, no position correction.
#[derive(Debug, Clone, PartialEq)]
pub struct GearConstraint {
    /// Rotation axis in body A's center-of-mass frame.
    pub axis_a: Vector3,
    /// Rotation axis in body B's center-of-mass frame.
    pub axis_b: Vector3,
    /// Angular velocity ratio between the axes.
    pub ratio: Real,
}

impl GearConstraint {
    /// A gear coupling with the given local axes and ratio.
    pub fn new(axis_a: Vector3, axis_b: Vector3, ratio: Real) -> Self {
        Self {
            axis_a,
            axis_b,
            ratio,
        }
    }

    /// Set body A's local rotation axis (Bullet's `setAxisA`).
    pub fn set_axis_a(&mut self, axis: Vector3) {
        self.axis_a = axis;
    }

    /// Set body B's local rotation axis (Bullet's `setAxisB`).
    pub fn set_axis_b(&mut self, axis: Vector3) {
        self.axis_b = axis;
    }

    /// Set the angular velocity ratio (Bullet's `setRatio`).
    pub fn set_ratio(&mut self, ratio: Real) {
        self.ratio = ratio;
    }

    /// Flatten this gear joint into its solver row (Bullet's `getInfo2`):
    /// one angular-only bilateral row with zero bias — the axes are local
    /// and re-rotated into world space every step. Upstream writes
    /// `J1 = globalAxisA`, `J2 = +ratio·globalAxisB` and applies the
    /// impulse as `+δ` on both bodies; ours applies `−λ` on A and `+λ` on B
    /// with the row velocity `J_b·v_b − J_a·v_a`, so the same coupling
    /// equation needs `ang_b = −ratio·axis_b_world`.
    pub(crate) fn prepare_rows(
        &self,
        body_a: usize,
        body_b: usize,
        bodies: &[RigidBody],
        out: &mut Vec<ConstraintRow>,
    ) {
        let a = &bodies[body_a];
        let b = &bodies[body_b];
        let axis_a_world = a.orientation * self.axis_a;
        let axis_b_world = b.orientation * self.axis_b;
        out.push(ConstraintRow::prepare(
            body_a,
            body_b,
            Vector3::ZERO,
            axis_a_world,
            Vector3::ZERO,
            -(self.ratio * axis_b_world),
            a,
            b,
            // Upstream never writes `m_constraintError` (the `convertJoint`
            // memset keeps it zero): velocity coupling only.
            0.0,
            RowLimits::Unbounded,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::PhysicsWorld;
    use approx::assert_relative_eq;
    use bullet_collision::shapes::CollisionShape;
    use bullet_math::FRAC_PI_2;

    const DT: Real = 1.0 / 60.0;

    /// Two identical free flywheels along z, A spun up. The one-row solve
    /// shares the correction by inverse inertia (`λ = p₀ / (I_a⁻¹ +
    /// ratio²·I_b⁻¹)`), so equal flywheels at ratio 2 settle at
    /// `ω_A = 4.8`, `ω_B = −2.4` — parallel axes spinning opposite, like
    /// meshed external gears.
    #[test]
    fn gears_couple_parallel_axes_to_opposite_spins() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::ZERO,
            2.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(3.0, 0.0, 0.0),
            2.0,
        ));
        world.add_constraint(Constraint::gear(a, b, Vector3::Z, Vector3::Z, 2.0));
        world.body_mut(a).angular_velocity = Vector3::new(0.0, 0.0, 6.0);
        for _ in 0..10 {
            world.step_simulation(DT);
        }
        let w_a = world.body(a).angular_velocity.z;
        let w_b = world.body(b).angular_velocity.z;
        // Same sphere shapes and masses → same inertia both sides.
        assert!((w_a - 4.8).abs() < 1e-2, "ω_a = {}", w_a);
        assert!((w_b + 2.4).abs() < 1e-2, "ω_b = {}", w_b);
        assert_relative_eq!(w_b, -w_a / 2.0, epsilon = 1e-3);
    }

    /// Zero bias: a pure velocity coupling never applies linear impulses —
    /// translated bodies keep their positions while their spins couple.
    #[test]
    fn offset_bodies_feel_no_position_pull() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::ZERO,
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(5.0, 1.0, -2.0),
            1.0,
        ));
        world.add_constraint(Constraint::gear(a, b, Vector3::Z, Vector3::Z, 1.0));
        world.body_mut(a).angular_velocity = Vector3::new(0.0, 0.0, 6.0);
        let before = (
            world.body(a).position,
            world.body(b).position,
        );
        for _ in 0..30 {
            world.step_simulation(DT);
        }
        assert_relative_eq!(world.body(a).position.x, before.0.x, epsilon = 1e-5);
        assert_relative_eq!(world.body(a).position.y, before.0.y, epsilon = 1e-5);
        assert_relative_eq!(world.body(b).position.x, before.1.x, epsilon = 1e-5);
        assert_relative_eq!(world.body(b).position.z, before.1.z, epsilon = 1e-5);
        // And the spins coupled along the way.
        assert_relative_eq!(
            world.body(b).angular_velocity.z,
            -world.body(a).angular_velocity.z,
            epsilon = 1e-3
        );
    }

    /// The axes are local: body A rotated 90° about x turns its local `z`
    /// axis into world `−y` (right-hand rule), so its spin about local z
    /// drives B's local-z spin through the world-frame coupling.
    #[test]
    fn local_axes_follow_body_rotation() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::ZERO,
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            CollisionShape::sphere(0.5),
            Vector3::new(4.0, 0.0, 0.0),
            1.0,
        ));
        world.add_constraint(Constraint::gear(a, b, Vector3::Z, Vector3::Z, 1.0));
        world.body_mut(a).orientation = quat_from_axis_angle(Vector3::X, FRAC_PI_2);
        world.body_mut(a).angular_velocity = Vector3::new(0.0, -6.0, 0.0); // local +z
        for _ in 0..10 {
            world.step_simulation(DT);
        }
        // Equal flywheels at ratio 1 split the drive: the coupled
        // projections settle at 3 / −3 (ω_A·(−y) + 1·ω_B·z = 0).
        let w_a = world.body(a).angular_velocity;
        let w_b = world.body(b).angular_velocity;
        assert!((w_a.y + 3.0).abs() < 1e-2, "ω_a = {}", w_a.y);
        assert!((w_b.z + 3.0).abs() < 1e-2, "ω_b = {}", w_b.z);
    }
}
