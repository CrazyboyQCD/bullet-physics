//! The multibody dynamics world — port of
//! `BulletDynamics/Featherstone/btMultiBodyDynamicsWorld` (the multibody-only
//! subset, without rigid-body collision integration).
//!
//! Stepping order mirrors Bullet's `internalSingleStepSimulation` for the
//! Featherstone parts:
//!
//! 1. **gravity** is applied to every awake body's base and links
//!    (`applyGravity`);
//! 2. **`solveExternalForces`**: forward kinematics, constraint forces are
//!    cleared, and each awake body runs the articulated-body algorithm, which
//!    integrates the generalized velocities by `dt` and caches the
//!    factorization needed by the constraint solver;
//! 3. **`solveInternalConstraints`**: every constraint emits its rows, a
//!    projected Gauss-Seidel pass solves them (10 outer × 1 inner iteration
//!    over the non-contact pool, alternating direction like Bullet), and the
//!    accumulated impulses are written back into the generalized velocities;
//! 4. **`integrateTransforms`**: positions are integrated (`stepPositions`)
//!    for awake bodies; sleeping bodies have their velocities cleared;
//! 5. forces are cleared and each body runs
//!    [`MultiBody::check_motion_and_sleep`].
//!
//! Sleeping mirrors Bullet's island semantics in simplified form: a sleeping
//! body linked by a constraint to an awake body is woken (the union-find
//! island merge). There is no collision detection between multibodies or
//! against rigid bodies in this subsystem (see the crate-level docs and
//! `docs/bullet-deviations.md`).

use bullet_math::{Real, Vector3};

use super::body::{MultiBody, MultiBodyScratch};
use super::constraint::{MultiBodyConstraint, SolverData, SolverRow};

/// A world of Featherstone articulated bodies and their constraints.
pub struct MultiBodyWorld {
    multibodies: Vec<MultiBody>,
    constraints: Vec<MultiBodyConstraint>,
    gravity: Vector3,
    /// PGS outer iterations (Bullet's `m_numIterations`, default 10).
    iterations: u32,
    /// Inner iterations over the non-contact pool per outer iteration
    /// (Bullet's `m_numNonContactInnerIterations`, default 1).
    num_non_contact_inner_iterations: u32,
    /// Constraint error-reduction factor (Bullet's `m_erp`, default 0.2).
    erp: Real,
    scratch: MultiBodyScratch,
    data: SolverData,
    rows: Vec<SolverRow>,
}

impl Default for MultiBodyWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiBodyWorld {
    pub fn new() -> Self {
        Self {
            multibodies: Vec::new(),
            constraints: Vec::new(),
            gravity: Vector3::new(0.0, -9.81, 0.0),
            iterations: 10,
            num_non_contact_inner_iterations: 1,
            erp: 0.2,
            scratch: MultiBodyScratch::default(),
            data: SolverData::default(),
            rows: Vec::new(),
        }
    }

    pub fn set_gravity(&mut self, gravity: Vector3) {
        self.gravity = gravity;
    }

    pub fn gravity(&self) -> Vector3 {
        self.gravity
    }

    /// Number of PGS iterations per step (Bullet's `m_numIterations`).
    pub fn set_iterations(&mut self, iterations: u32) {
        self.iterations = iterations;
    }

    /// Inner iterations for the non-contact (joint) rows per outer iteration
    /// (Bullet's `m_numNonContactInnerIterations`, default 1).
    pub fn set_num_non_contact_inner_iterations(&mut self, n: u32) {
        self.num_non_contact_inner_iterations = n;
    }

    /// Constraint error-reduction factor (Bullet's `m_erp`, default 0.2).
    pub fn set_erp(&mut self, erp: Real) {
        self.erp = erp;
    }

    /// Add a multibody (must be finalized) and return its index.
    pub fn add_multibody(&mut self, body: MultiBody) -> usize {
        self.multibodies.push(body);
        self.multibodies.len() - 1
    }

    pub fn multibody(&self, index: usize) -> &MultiBody {
        &self.multibodies[index]
    }

    pub fn multibody_mut(&mut self, index: usize) -> &mut MultiBody {
        &mut self.multibodies[index]
    }

    pub fn multibodies(&self) -> &[MultiBody] {
        &self.multibodies
    }

    pub fn num_multibodies(&self) -> usize {
        self.multibodies.len()
    }

    /// Add a multibody constraint and return its index.
    pub fn add_constraint(&mut self, constraint: MultiBodyConstraint) -> usize {
        self.constraints.push(constraint);
        self.constraints.len() - 1
    }

    pub fn constraint(&self, index: usize) -> &MultiBodyConstraint {
        &self.constraints[index]
    }

    pub fn constraint_mut(&mut self, index: usize) -> &mut MultiBodyConstraint {
        &mut self.constraints[index]
    }

    pub fn constraints(&self) -> &[MultiBodyConstraint] {
        &self.constraints
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Remove a constraint by index (swaps the last one in to keep the list
    /// dense).
    pub fn remove_constraint(&mut self, index: usize) {
        self.constraints.swap_remove(index);
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step_simulation(&mut self, dt: Real) {
        // 1. Gravity (Bullet's `applyGravity`).
        for mb in &mut self.multibodies {
            if !mb.is_awake() {
                continue;
            }
            mb.add_base_force(self.gravity * mb.base_mass());
            for j in 0..mb.num_links() {
                mb.add_link_force(j, self.gravity * mb.link_mass(j));
            }
        }

        self.solve_external_forces(dt);
        self.solve_internal_constraints(dt);
        self.integrate_transforms(dt);

        // Clear the accumulated forces and update the sleep state (Bullet
        // clears forces at the end of `internalSingleStepSimulation` and runs
        // `updateActivationState` at the start of the next one).
        for mb in &mut self.multibodies {
            if mb.is_awake() {
                mb.clear_forces_and_torques();
            }
            mb.check_motion_and_sleep(dt);
        }
    }

    fn forward_kinematics(&mut self) {
        for mb in &mut self.multibodies {
            mb.forward_kinematics();
        }
    }

    /// Refresh link transforms, clear constraint forces, then integrate the
    /// generalized velocities with the articulated-body algorithm (Bullet's
    /// `solveExternalForces`).
    fn solve_external_forces(&mut self, dt: Real) {
        self.forward_kinematics();
        for mb in &mut self.multibodies {
            if !mb.is_awake() {
                continue;
            }
            mb.clear_constraint_forces();
            mb.compute_accelerations_aba(dt, &mut self.scratch, false);
        }
    }

    /// Convert all constraints into rows and solve them with projected
    /// Gauss-Seidel (Bullet's `solveInternalConstraints` +
    /// `btMultiBodyConstraintSolver::solveMultiBodyGroup`).
    fn solve_internal_constraints(&mut self, dt: Real) {
        // Wake bodies linked to an awake body by a constraint (the island
        // merge in `calculateSimulationIslands`).
        let mut wake = vec![false; self.multibodies.len()];
        for c in &self.constraints {
            let a_awake = self.multibodies[c.body_a()].is_awake();
            let b_awake = self.multibodies[c.body_b()].is_awake();
            if a_awake || b_awake {
                wake[c.body_a()] = true;
                wake[c.body_b()] = true;
            }
        }
        for (i, &w) in wake.iter().enumerate() {
            if w {
                self.multibodies[i].wake_up();
            }
        }

        // Reset the solver scratch and companion ids (Bullet's
        // `solveGroupCacheFriendlySetup`).
        self.data.reset();
        self.rows.clear();
        for mb in &mut self.multibodies {
            mb.companion_id = -1;
        }

        // Convert the constraints into rows.
        {
            let MultiBodyWorld {
                multibodies,
                constraints,
                data,
                scratch,
                rows,
                erp,
                ..
            } = self;
            for (i, c) in constraints.iter_mut().enumerate() {
                c.create_constraint_rows(i, rows, data, scratch, multibodies, dt, *erp);
            }
        }

        // PGS iterations over the non-contact rows, alternating direction by
        // iteration parity (Bullet's `solveSingleIteration`).
        let n = self.rows.len();
        let outer = self.iterations as usize;
        let inner = self.num_non_contact_inner_iterations as usize;
        for iteration in 0..outer {
            for _ in 0..inner {
                if iteration % 2 == 0 {
                    for j in 0..n {
                        resolve_row(&mut self.rows[j], &mut self.data);
                    }
                } else {
                    for j in (0..n).rev() {
                        resolve_row(&mut self.rows[j], &mut self.data);
                    }
                }
            }
        }

        // Write the accumulated impulses back into the constraints (Bullet's
        // `writeBackSolverBodyToMultiBody`).
        for row in &self.rows {
            self.constraints[row.org_constraint]
                .set_applied_impulse(row.org_dof_index, row.applied_impulse);
        }
        for row in &self.rows {
            let imp = row.applied_impulse;
            let unit_a = self.data.delta_velocities_unit_impulse
                [row.jac_a_index..row.jac_a_index + row.ndof_a]
                .to_vec();
            let unit_b = self.data.delta_velocities_unit_impulse
                [row.jac_b_index..row.jac_b_index + row.ndof_b]
                .to_vec();
            let mb = &mut self.multibodies[row.multi_body_a];
            mb.apply_delta_vee(&unit_a, imp);
            let mb = &mut self.multibodies[row.multi_body_b];
            mb.apply_delta_vee(&unit_b, imp);
        }
        // Flush any split-impulse deltas (no-op unless split impulses are
        // used; kept for fidelity with `solveInternalConstraints`).
        for mb in &mut self.multibodies {
            mb.process_delta_vee_2();
        }
    }

    /// Integrate the positions of the awake bodies and freeze the sleeping
    /// ones (Bullet's `integrateMultiBodyTransforms`).
    fn integrate_transforms(&mut self, dt: Real) {
        for mb in &mut self.multibodies {
            if !mb.is_awake() {
                mb.clear_velocities();
                continue;
            }
            mb.add_split_v();
            mb.step_positions(dt);
            mb.subtract_split_v();
        }
    }
}

/// Solve one multibody constraint row (Bullet's
/// `resolveSingleConstraintRowGeneric`): compute the impulse needed to meet
/// `rhs` given the currently accumulated velocity deltas of both bodies,
/// clamp it to the row window, and add its effect to the accumulated deltas.
fn resolve_row(row: &mut SolverRow, data: &mut SolverData) {
    let ndof_a = row.ndof_a;
    let ndof_b = row.ndof_b;
    let da_index = row.delta_vel_a_index as usize;
    let db_index = row.delta_vel_b_index as usize;

    let mut delta_impulse = row.rhs - row.applied_impulse * row.cfm;

    let mut delta_vel_a_dot_n = 0.0;
    {
        let jac = &data.jacobians[row.jac_a_index..row.jac_a_index + ndof_a];
        let dv = &data.delta_velocities[da_index..da_index + ndof_a];
        for i in 0..ndof_a {
            delta_vel_a_dot_n += jac[i] * dv[i];
        }
    }
    let mut delta_vel_b_dot_n = 0.0;
    {
        let jac = &data.jacobians[row.jac_b_index..row.jac_b_index + ndof_b];
        let dv = &data.delta_velocities[db_index..db_index + ndof_b];
        for i in 0..ndof_b {
            delta_vel_b_dot_n += jac[i] * dv[i];
        }
    }
    delta_impulse -= (delta_vel_a_dot_n + delta_vel_b_dot_n) * row.jac_diag_ab_inv;
    let sum = row.applied_impulse + delta_impulse;
    if sum < row.lower_limit {
        delta_impulse = row.lower_limit - row.applied_impulse;
        row.applied_impulse = row.lower_limit;
    } else if sum > row.upper_limit {
        delta_impulse = row.upper_limit - row.applied_impulse;
        row.applied_impulse = row.upper_limit;
    } else {
        row.applied_impulse = sum;
    }

    // Accumulate the effect of this impulse into both bodies' delta
    // velocities (Bullet's `applyDeltaVee`).
    {
        let unit = &data.delta_velocities_unit_impulse[row.jac_a_index..row.jac_a_index + ndof_a];
        let dv = &mut data.delta_velocities[da_index..da_index + ndof_a];
        for i in 0..ndof_a {
            dv[i] += unit[i] * delta_impulse;
        }
    }
    {
        let unit = &data.delta_velocities_unit_impulse[row.jac_b_index..row.jac_b_index + ndof_b];
        let dv = &mut data.delta_velocities[db_index..db_index + ndof_b];
        for i in 0..ndof_b {
            dv[i] += unit[i] * delta_impulse;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multibody::constraint::MultiBodyConstraint;
    use bullet_math::Quaternion;

    fn assert_close(a: Real, b: Real, tol: Real) {
        assert!(
            (a - b).abs() < tol,
            "expected {a} close to {b} within {tol}"
        );
    }

    /// A free base (zero links) falls under gravity (damping off, so the
    /// velocity reaches exactly g·t).
    #[test]
    fn free_base_falls_under_gravity() {
        let mut world = MultiBodyWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let mut mb = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        mb.set_linear_damping(0.0);
        mb.set_angular_damping(0.0);
        mb.finalize_multidof();
        world.add_multibody(mb);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        let body = world.multibody(0);
        // v = g·t after 1 s; position follows the semi-implicit Euler scheme
        // (velocity is integrated before position), so
        // y = −g·dt²·n·(n+1)/2 with n = 60.
        assert_close(body.base_vel().y, -9.81, 0.05);
        let n: Real = 60.0;
        let dt: Real = 1.0 / 60.0;
        let expected_y = -9.81 * dt * dt * n * (n + 1.0) / 2.0;
        assert_close(body.base_pos().y, expected_y, 0.05);
    }

    /// A fixed-base single-link revolute pendulum released from the
    /// horizontal: the joint angle oscillates through the bottom and the
    /// bottom speed matches the drop height (√(2·g·L)).
    #[test]
    fn revolute_pendulum_swings() {
        let mut world = MultiBodyWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
        // Pivot at the base origin; link COM hangs 1 m below the pivot
        // (d_vector), axis +Z.
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(0.0, -1.0, 0.0),
        );
        mb.finalize_multidof();
        // Release from the horizontal (q = +π/2).
        mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
        world.add_multibody(mb);
        let dt = 1.0 / 60.0;
        let mut bottom_speeds = Vec::new();
        let mut prev_q = world.multibody(0).joint_pos_multidof(0)[0];
        let mut prev_qd = world.multibody(0).joint_vel_multidof(0)[0];
        let mut crossings = 0;
        for _ in 0..600 {
            world.step_simulation(dt);
            let body = world.multibody(0);
            let q = body.joint_pos_multidof(0)[0];
            let qd = body.joint_vel_multidof(0)[0];
            // Detect a bottom crossing: q changes sign while qd is largest
            // in magnitude (q near 0 going through the bottom).
            if prev_q.abs() < 0.15 && q.abs() < 0.15 && prev_q * q <= 0.0 && crossings < 4 {
                crossings += 1;
                bottom_speeds.push(qd.abs());
            }
            if prev_q * prev_qd < 0.0 && prev_q.abs() > 1.0 {
                // turn-around above the bottom — nothing to do
            }
            prev_q = q;
            prev_qd = qd;
        }
        // The pendulum must have swung through the bottom at least once
        // (released from 90°, the first crossing happens well within 1 s).
        assert!(crossings >= 1, "pendulum never crossed the bottom");
        // The pendulum: from the horizontal the COM drops 1 m, so with
        // I_eff = m·L² + I_z = 2 the bottom angular speed is
        // √(2·m·g·L / I_eff) = √(9.81) ≈ 3.13 rad/s (small damping).
        let first = bottom_speeds.first().copied().unwrap_or(0.0);
        assert!(
            first > 2.7 && first < 3.6,
            "bottom speed {first} not in [2.7, 3.6] (expected ≈ 3.13)"
        );
    }

    /// Forward kinematics of a two-link chain: link 1 at a fixed angle, the
    /// endpoint lands at the expected world position.
    #[test]
    fn forward_kinematics_two_link_chain() {
        let mut mb = MultiBody::new(2, 1.0, Vector3::ONE, true, false);
        // Link 0: revolute about +Z at the origin; COM 1 m along +X.
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
        );
        // Link 1: revolute about +Z; pivot at link 0's COM, COM 1 m further
        // along +X.
        mb.setup_revolute(
            1,
            1.0,
            Vector3::ONE,
            Some(0),
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        );
        mb.finalize_multidof();
        mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
        mb.set_joint_pos_multidof(1, &[-bullet_math::FRAC_PI_2]);
        mb.forward_kinematics();
        // Link 0's COM: (1,0,0) rotated +90° about Z → (0,1,0).
        let p0 = mb.local_pos_to_world(0, Vector3::ZERO);
        assert_close(p0.x, 0.0, 1e-5);
        assert_close(p0.y, 1.0, 1e-5);
        // Link 1's pivot sits 1 m along link 0's X (which points +Y after the
        // +90° rotation), so the pivot is at (0,2,0); link 1's COM is 1 m
        // further along its own X, which is back to +X at q1 = −90°.
        let p1 = mb.local_pos_to_world(1, Vector3::ZERO);
        assert_close(p1.x, 1.0, 1e-4);
        assert_close(p1.y, 2.0, 1e-4);
    }

    /// A point-to-point constraint between two free bases keeps the pivots
    /// together even under an asymmetric initial velocity.
    #[test]
    fn point2point_keeps_bodies_together() {
        let mut world = MultiBodyWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        let b = MultiBody::new(0, 2.0, Vector3::splat(1.5), false, false);
        // The constraint ctor reads the finalized bodies' dof counts, so
        // pass a snapshot before moving the real bodies into the world.
        let ia = world.add_multibody(a);
        let ib = world.add_multibody(b);
        world.multibody_mut(ia).finalize_multidof();
        world.multibody_mut(ib).finalize_multidof();
        let snapshot = world.multibodies().to_vec();
        let constraint = MultiBodyConstraint::point2point(
            ia,
            None,
            ib,
            None,
            Vector3::ZERO,
            Vector3::ZERO,
            &snapshot,
        );
        world.add_constraint(constraint);
        // Body B starts with an upward velocity; the constraint must keep
        // the pivots coincident while the pair decelerates.
        world.multibody_mut(ib).set_base_vel(Vector3::new(0.0, 3.0, 0.0));
        let dt = 1.0 / 60.0;
        let mut max_separation: Real = 0.0;
        for _ in 0..120 {
            world.step_simulation(dt);
            let pa = world.multibody(ia).base_pos();
            let pb = world.multibody(ib).base_pos();
            max_separation = max_separation.max((pa - pb).length());
        }
        assert!(
            max_separation < 0.05,
            "P2P pivots drifted apart by {max_separation} m"
        );
    }

    /// A joint motor drives the joint velocity to the target.
    #[test]
    fn joint_motor_reaches_target_velocity() {
        let mut world = MultiBodyWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = MultiBody::new(1, 1.0, Vector3::ONE, true, false);
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
        );
        mb.finalize_multidof();
        let body = world.add_multibody(mb);
        let snapshot = world.multibodies().to_vec();
        let motor = MultiBodyConstraint::joint_motor(body, 0, 0, 2.0, 1000.0, &snapshot);
        world.add_constraint(motor);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step_simulation(dt);
        }
        let qd = world.multibody(body).joint_vel_multidof(0)[0];
        assert!(
            (qd - 2.0).abs() < 0.1,
            "motor did not reach 2 rad/s, got {qd}"
        );
        // The joint angle must have advanced.
        let q = world.multibody(body).joint_pos_multidof(0)[0];
        assert!(q > 1.0, "motor joint angle did not advance, got {q}");
    }

    /// A body at rest with zero gravity falls asleep after the deactivation
    /// window, and a constraint to an awake body wakes it (the simplified
    /// island-merge semantics).
    #[test]
    fn body_sleeps_after_rest_and_wakes_via_constraint() {
        let mut world = MultiBodyWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut a = MultiBody::new(0, 1.0, Vector3::ONE, false, true);
        let mut b = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        a.finalize_multidof();
        b.finalize_multidof();
        let ia = world.add_multibody(a);
        let ib = world.add_multibody(b);
        assert!(world.multibody(ia).is_awake());
        let dt = 1.0 / 60.0;
        for _ in 0..150 {
            // 2.5 s > the 2 s sleep timeout.
            world.step_simulation(dt);
        }
        assert!(!world.multibody(ia).is_awake(), "body should be asleep");
        assert!(world.multibody(ib).is_awake(), "non-sleeping body stays awake");
        // A constraint to the awake body wakes the sleeping one.
        let snapshot = world.multibodies().to_vec();
        let p2p = MultiBodyConstraint::point2point(
            ia,
            None,
            ib,
            None,
            Vector3::ZERO,
            Vector3::ZERO,
            &snapshot,
        );
        world.add_constraint(p2p);
        world.step_simulation(dt);
        assert!(world.multibody(ia).is_awake(), "body should wake via constraint");
    }

    /// Two free bases welded by a fixed constraint move as one rigid body.
    #[test]
    fn fixed_constraint_welds_bodies() {
        let mut world = MultiBodyWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        let a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        let b = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        let ia = world.add_multibody(a);
        let ib = world.add_multibody(b);
        world.multibody_mut(ia).finalize_multidof();
        world.multibody_mut(ib).finalize_multidof();
        let snapshot = world.multibodies().to_vec();
        let fixed = MultiBodyConstraint::fixed(
            ia,
            None,
            ib,
            None,
            Vector3::ZERO,
            Vector3::ZERO,
            bullet_math::Matrix3::IDENTITY,
            bullet_math::Matrix3::IDENTITY,
            &snapshot,
        );
        world.add_constraint(fixed);
        world.multibody_mut(ib).set_base_vel(Vector3::new(2.0, 0.0, 0.0));
        let dt = 1.0 / 60.0;
        let mut max_rel: Real = 0.0;
        for _ in 0..120 {
            world.step_simulation(dt);
            let pa = world.multibody(ia).base_pos();
            let pb = world.multibody(ib).base_pos();
            let qa = world.multibody(ia).base_quat;
            let qb = world.multibody(ib).base_quat;
            max_rel = max_rel.max((pa - pb).length());
            // Relative orientation stays identity.
            let rel = qa.inverse() * qb;
            assert!(rel.x.abs() < 1e-3 && rel.y.abs() < 1e-3 && rel.z.abs() < 1e-3);
        }
        assert!(
            max_rel < 0.05,
            "welded bodies drifted apart by {max_rel} m"
        );
    }

    /// A slider constraint lets the bodies separate along the slide axis
    /// only.
    #[test]
    fn slider_constrains_perpendicular_motion() {
        let mut world = MultiBodyWorld::new();
        world.set_gravity(Vector3::ZERO);
        let a = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        let b = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        let ia = world.add_multibody(a);
        let ib = world.add_multibody(b);
        world.multibody_mut(ia).finalize_multidof();
        world.multibody_mut(ib).finalize_multidof();
        let snapshot = world.multibodies().to_vec();
        let slider = MultiBodyConstraint::slider(
            ia,
            None,
            ib,
            None,
            Vector3::ZERO,
            Vector3::ZERO,
            bullet_math::Matrix3::IDENTITY,
            bullet_math::Matrix3::IDENTITY,
            Vector3::X,
            &snapshot,
        );
        world.add_constraint(slider);
        // Body B slides along +X (allowed) with a Y component (blocked).
        world
            .multibody_mut(ib)
            .set_base_vel(Vector3::new(3.0, 3.0, 0.0));
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            world.step_simulation(dt);
        }
        let pa = world.multibody(ia).base_pos();
        let pb = world.multibody(ib).base_pos();
        let rel = pb - pa;
        assert!(rel.x > 0.5, "slider did not slide along X: {rel:?}");
        assert!(
            rel.y.abs() < 0.05 && rel.z.abs() < 0.05,
            "slider allowed off-axis motion: {rel:?}"
        );
    }
}
