//! Multibody solver passes (external forces, internal constraints, the mixed
//! rigid+multibody PGS loop, writeback and FK integration), driven from
//! `step.rs`.

use super::*;

impl PhysicsWorld {

    /// Refresh link transforms, clear constraint forces, then integrate the
    /// generalized velocities with the articulated-body algorithm (Bullet's
    /// `solveExternalForces`).
    pub(crate) fn mb_solve_external_forces(&mut self, dt: Real) {
        for mb in &mut self.multibodies {
            mb.forward_kinematics();
        }
        for mb in &mut self.multibodies {
            if !mb.is_awake() {
                continue;
            }
            mb.clear_constraint_forces();
            mb.compute_accelerations_aba(dt, &mut self.mb_scratch, false);
        }
    }

    /// Convert all multibody constraints into rows and solve them with
    /// projected Gauss-Seidel (Bullet's `solveInternalConstraints` +
    /// `btMultiBodyConstraintSolver::solveMultiBodyGroup` for the non-contact
    /// pool) — the pure-multibody path with no collider contacts.
    pub(crate) fn mb_solve_internal_constraints(&mut self, dt: Real) {
        // Reset the solver scratch and companion ids (Bullet's
        // `solveGroupCacheFriendlySetup`).
        self.mb_data.reset();
        self.mb_rows.clear();
        for mb in &mut self.multibodies {
            mb.companion_id = -1;
        }

        // Convert the constraints into rows. Constraints of a sleeping
        // island (both sides asleep after the island pass) produce no rows.
        {
            let PhysicsWorld {
                bodies,
                multibodies,
                mb_constraints: constraints,
                mb_data: data,
                mb_scratch: scratch,
                mb_rows: rows,
                mb_erp: erp,
                ..
            } = self;
            for (i, c) in constraints.iter_mut().enumerate() {
                let side_b_awake = match c.rigid_body_b() {
                    Some(rb) => bodies[rb.index()].is_active(),
                    None => multibodies[c.body_b().index()].is_awake(),
                };
                if !multibodies[c.body_a().index()].is_awake() && !side_b_awake {
                    continue;
                }
                c.create_constraint_rows(i, rows, data, scratch, multibodies, bodies, dt, *erp);
            }
        }

        // PGS iterations over the non-contact rows, alternating direction by
        // iteration parity (Bullet's `solveSingleIteration`).
        let outer = self.iterations as usize;
        let inner = self.num_non_contact_inner_iterations as usize;
        for iteration in 0..outer {
            for _ in 0..inner {
                self.mb_rows_sweep(iteration);
            }
        }

        self.mb_writeback_rows(dt);
        // Joint feedback constraint pass (see `solve_mixed`); no-op unless a
        // link carries a feedback slot.
        for mb in &mut self.multibodies {
            if mb.is_awake() && mb.needs_joint_feedback() {
                mb.compute_accelerations_aba(dt, &mut self.mb_scratch, true);
            }
        }
        // Flush any split-impulse deltas (no-op unless split impulses are
        // used; kept for fidelity with `solveInternalConstraints`).
        for mb in &mut self.multibodies {
            mb.process_delta_vee_2();
        }
    }

    /// The merged solve (Bullet's `solveMultiBodyGroup`): rigid contact +
    /// joint rows, multibody contact rows and multibody constraint rows in
    /// one PGS loop, with the rigid split-impulse pass after the velocity
    /// iterations and the multibody writebacks at the end.
    pub(crate) fn solve_mixed(
        &mut self,
        constraints: &mut [ContactConstraint],
        joint_rows: &mut [ConstraintRow],
        inv_dt: Real,
    ) {
        // Shared solver pool + companion ids for the contact and constraint
        // rows (Bullet's `solveGroupCacheFriendlySetup`).
        self.mb_data.reset();
        self.mb_rows.clear();
        self.mb_contact_rows.clear();
        for mb in &mut self.multibodies {
            mb.companion_id = -1;
        }

        // Convert the collected contacts and the constraints into rows.
        // Contacts of still-sleeping multibodies (not woken by anything
        // active) are skipped — their pair produced no impulse.
        {
            let PhysicsWorld {
                bodies,
                multibodies,
                mb_pending_contacts,
                mb_contact_rows,
                mb_data: data,
                mb_scratch: scratch,
                mb_constraints,
                mb_rows,
                mb_erp: erp,
                ..
            } = self;
            for p in mb_pending_contacts.iter() {
                let awake = |s: &ContactSide| match *s {
                    ContactSide::Rigid(_) => true,
                    ContactSide::Link { mb, .. } => multibodies[mb].is_awake(),
                };
                if !awake(&p.side_a) || !awake(&p.side_b) {
                    continue;
                }
                contact::convert_contact(
                    p,
                    bodies,
                    multibodies,
                    data,
                    scratch,
                    inv_dt,
                    mb_contact_rows,
                );
            }
            let dt = 1.0 / inv_dt;
            for (i, c) in mb_constraints.iter_mut().enumerate() {
                // Constraints of a sleeping island (both sides asleep after
                // the island pass) produce no rows.
                let side_b_awake = match c.rigid_body_b() {
                    Some(rb) => bodies[rb.index()].is_active(),
                    None => multibodies[c.body_b().index()].is_awake(),
                };
                if !multibodies[c.body_a().index()].is_awake() && !side_b_awake {
                    continue;
                }
                c.create_constraint_rows(i, mb_rows, data, scratch, multibodies, bodies, dt, *erp);
            }
        }

        // One PGS iteration: rigid sweep, multibody contact sweep (normals,
        // then bounded frictions), then the multibody constraint rows.
        sequential_impulse::warm_start(&mut self.bodies, constraints);
        let outer = self.iterations as usize;
        let inner = self.num_non_contact_inner_iterations as usize;
        for iteration in 0..outer {
            sequential_impulse::velocity_iteration(
                &mut self.bodies,
                constraints,
                joint_rows,
                &mut self.rolling_rows,
                inv_dt,
            );
            contact::solve_contact_rows(
                &mut self.mb_contact_rows,
                &mut self.mb_data,
                &mut self.bodies,
                &mut self.mb_row_scratch,
            );
            for _ in 0..inner {
                self.mb_rows_sweep(iteration);
            }
        }
        sequential_impulse::split_impulse_pass(
            &mut self.bodies,
            constraints,
            self.iterations,
            inv_dt,
        );

        // Writebacks: the multibody contact impulses first, then the
        // constraint impulses (Bullet's `writeBackSolverBodyToMultiBody`
        // order).
        {
            let PhysicsWorld {
                mb_contact_rows,
                mb_data: data,
                multibodies,
                ..
            } = self;
            contact::writeback_contact_rows(mb_contact_rows, data, multibodies, 1.0 / inv_dt);
        }
        self.mb_writeback_rows(1.0 / inv_dt);
        // Joint feedback: when any link carries a feedback slot, rerun the
        // articulated-body algorithm on the accumulated constraint forces so
        // the feedback reports the solver contribution too (Bullet's
        // `solveInternalConstraints` constraint pass). The pass only reads
        // accelerations — velocities were already integrated above.
        for mb in &mut self.multibodies {
            if mb.is_awake() && mb.needs_joint_feedback() {
                mb.compute_accelerations_aba(1.0 / inv_dt, &mut self.mb_scratch, true);
            }
        }
        for mb in &mut self.multibodies {
            mb.process_delta_vee_2();
        }
    }

    /// One inner sweep over the multibody constraint rows, alternating
    /// direction by iteration parity (Bullet's `solveSingleIteration`).
    fn mb_rows_sweep(&mut self, iteration: usize) {
        let PhysicsWorld {
            mb_rows,
            mb_data,
            bodies,
            ..
        } = self;
        if iteration.is_multiple_of(2) {
            for row in mb_rows.iter_mut() {
                resolve_row(row, mb_data, bodies);
            }
        } else {
            for row in mb_rows.iter_mut().rev() {
                resolve_row(row, mb_data, bodies);
            }
        }
    }

    /// Write the accumulated constraint impulses back into the constraints
    /// and the generalized velocities (Bullet's
    /// `writeBackSolverBodyToMultiBody`). Rigid side-B bodies already took
    /// their impulses during the iterations (they have no deferred
    /// generalized writeback), so only the multibody sides are written here.
    fn mb_writeback_rows(&mut self, dt: Real) {
        for row in &self.mb_rows {
            self.mb_constraints[row.org_constraint]
                .set_applied_impulse(row.org_dof_index, row.applied_impulse);
        }
        for row in &self.mb_rows {
            let imp = row.applied_impulse;
            let mb = &mut self.multibodies[row.multi_body_a];
            mb.apply_delta_vee(
                &self.mb_data.delta_velocities_unit_impulse
                    [row.jac_a_index..row.jac_a_index + row.ndof_a],
                imp,
            );
            // Accumulate the row impulse as a constraint force for the joint
            // feedback (Bullet's `writeBackSolverBodyToMultiBody`).
            let force_a = row.contact_normal1 * (imp / dt);
            let torque_a = row.rel_pos1_cross_normal * (imp / dt);
            match row.link_a {
                None => {
                    mb.add_base_constraint_force(force_a);
                    mb.add_base_constraint_torque(torque_a);
                }
                Some(l) => {
                    mb.add_link_constraint_force(l, force_a);
                    mb.add_link_constraint_torque(l, torque_a);
                }
            }
            if row.rigid_b.is_some() {
                continue; // the rigid side applied its impulses in-solve
            }
            let mb = &mut self.multibodies[row.multi_body_b];
            mb.apply_delta_vee(
                &self.mb_data.delta_velocities_unit_impulse
                    [row.jac_b_index..row.jac_b_index + row.ndof_b],
                imp,
            );
            let force_b = row.contact_normal2 * (imp / dt);
            let torque_b = row.rel_pos2_cross_normal * (imp / dt);
            match row.link_b {
                None => {
                    mb.add_base_constraint_force(force_b);
                    mb.add_base_constraint_torque(torque_b);
                }
                Some(l) => {
                    mb.add_link_constraint_force(l, force_b);
                    mb.add_link_constraint_torque(l, torque_b);
                }
            }
        }
    }

    /// Integrate the positions of the awake multibodies and freeze the
    /// sleeping ones (Bullet's `integrateMultiBodyTransforms`).
    pub(crate) fn mb_integrate_transforms(&mut self, dt: Real) {
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
/// A rigid side B reads its velocity **delta** since row assembly (the RHS
/// already carries the assembly-time projection) and applies the impulse
/// directly to the body's velocities, like the mixed contact rows.
pub(crate) fn resolve_row(row: &mut SolverRow, data: &mut SolverData, rigid_bodies: &mut [RigidBody]) {
    let ndof_a = row.ndof_a;
    let ndof_b = row.ndof_b;
    let da_index = row.delta_vel_a_index as usize;
    let db_index = row.delta_vel_b_index as usize;

    if row.jac_diag_ab_inv == 0.0 {
        return; // disabled singular row
    }
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
    match row.rigid_b {
        Some(rigid) => {
            let body = &rigid_bodies[rigid.body];
            delta_vel_b_dot_n = row.contact_normal2.dot(body.linear_velocity)
                + row.rel_pos2_cross_normal.dot(body.angular_velocity)
                - rigid.initial_proj;
        }
        None => {
            let jac = &data.jacobians[row.jac_b_index..row.jac_b_index + ndof_b];
            let dv = &data.delta_velocities[db_index..db_index + ndof_b];
            for i in 0..ndof_b {
                delta_vel_b_dot_n += jac[i] * dv[i];
            }
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
    // velocities (Bullet's `applyDeltaVee`); a rigid side B applies the
    // impulse to its real velocities (Bullet's `internalApplyImpulse` on the
    // solver body).
    {
        let unit = &data.delta_velocities_unit_impulse[row.jac_a_index..row.jac_a_index + ndof_a];
        let dv = &mut data.delta_velocities[da_index..da_index + ndof_a];
        for i in 0..ndof_a {
            dv[i] += unit[i] * delta_impulse;
        }
    }
    match row.rigid_b {
        Some(rigid) => {
            let body = &mut rigid_bodies[rigid.body];
            body.linear_velocity += row.contact_normal2 * (delta_impulse * rigid.inv_mass);
            body.angular_velocity += rigid.angular_component * delta_impulse;
        }
        None => {
            let unit =
                &data.delta_velocities_unit_impulse[row.jac_b_index..row.jac_b_index + ndof_b];
            let dv = &mut data.delta_velocities[db_index..db_index + ndof_b];
            for i in 0..ndof_b {
                dv[i] += unit[i] * delta_impulse;
            }
        }
    }
}

/// Whether a contact side currently participates in the simulation: rigid
/// bodies by their activation state, multibodies by their awake flag (a
/// whole articulated body sleeps as one unit).
pub(crate) fn side_active_owner(side: ContactSide, bodies: &[RigidBody], multibodies: &[MultiBody]) -> bool {
    match side {
        ContactSide::Rigid(i) => bodies[i].is_active(),
        ContactSide::Link { mb, .. } => multibodies[mb].is_awake(),
    }
}