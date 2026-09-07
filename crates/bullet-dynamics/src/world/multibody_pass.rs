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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::filter_groups;
    use approx::assert_relative_eq;

    /// A multibody base collider filtered by the plane's mask falls
    /// through.
    #[test]
    fn mb_base_collider_respects_mask() {
        let mut world = PhysicsWorld::new();
        world.add_body(
            RigidBody::new_static(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
                Vector3::ZERO,
            )
            .collision_filter(filter_groups::DEFAULT, 0),
        );
        let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        mb.set_linear_damping(0.0);
        mb.finalize_multidof();
        mb.set_base_collider(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            0.5,
            0.0,
        );
        mb.base_pos = Vector3::new(0.0, 3.0, 0.0);
        world.add_multibody(mb);
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            world.step_simulation(dt);
        }
        assert!(
            world.multibody(MultiBodyHandle::from_index(0)).base_pos.y < -1.0,
            "filtered plane stopped the mb base: y = {}",
            world.multibody(MultiBodyHandle::from_index(0)).base_pos.y
        );
    }

    /// A fixed-base pendulum with a sphere collider on the link: the collider
    /// registers as a broad-phase proxy with the FK pose AABB and refreshes
    /// as the body moves (phase B of the multibody collision design).
    #[test]
    fn link_collider_registers_and_updates_aabb() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
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
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        world.step_simulation(1.0 / 60.0);
        // The link COM hangs 1 m below the pivot; the sphere AABB spans
        // y ∈ [−1.5, −0.5], padded by the 0.02 broad-phase contact margin.
        let entry = world
            .collider_entries
            .iter()
            .find(|e| e.mb == 0 && e.link == 0)
            .expect("collider registered");
        let proxy = entry.proxy.expect("proxy assigned");
        let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_relative_eq!(aabb.min.y, -1.52, epsilon = 1e-4);
        assert_relative_eq!(aabb.max.y, -0.48, epsilon = 1e-4);
        // Moving the base moves the collider with the next FK pass.
        world.multibody_mut(MultiBodyHandle::from_index(0)).base_pos = Vector3::new(0.0, 5.0, 0.0);
        world.step_simulation(1.0 / 60.0);
        let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_relative_eq!(aabb.min.y, 3.48, epsilon = 1e-4);
    }

    /// A sleeping multibody keeps its collider AABB (like sleeping rigid
    /// bodies): the refresh is skipped until the body wakes.
    #[test]
    fn sleeping_multibody_keeps_collider_aabb() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, true);
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
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        for _ in 0..150 {
            // 2.5 s > the 2 s sleep timeout.
            world.step_simulation(1.0 / 60.0);
        }
        assert!(!world.multibody(MultiBodyHandle::from_index(0)).is_awake());
        let proxy = world.collider_entries[0].proxy.unwrap();
        let before = world.broadphase.proxy_aabb(proxy).unwrap();
        // Teleport the base: the sleeping body skips the AABB refresh.
        world.multibody_mut(MultiBodyHandle::from_index(0)).base_pos = Vector3::new(3.0, 0.0, 0.0);
        world.step_simulation(1.0 / 60.0);
        let after = world.broadphase.proxy_aabb(proxy).unwrap();
        assert_eq!(before, after);
    }

    /// A link collider overlapping a static rigid body produces a translated
    /// mixed broad-phase pair.
    #[test]
    fn link_collider_pairs_with_rigid_body() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
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
        // Sink the link so its sphere (r = 0.5, centered 0.6 below the
        // plane) overlaps the ground.
        mb.base_pos = Vector3::new(0.0, 0.4, 0.0);
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        world.step_simulation(1.0 / 60.0);
        let mixed = world
            .broadphase
            .compute_pairs()
            .into_iter()
            .filter(|(a, b)| {
                matches!(world.proxy_owners[*a], ProxyOwner::Body(_))
                    ^ matches!(world.proxy_owners[*b], ProxyOwner::Body(_))
            })
            .count();
        assert_eq!(mixed, 1, "one rigid-collider pair expected");
    }

    /// Self-collision filtering between two overlapping link colliders of
    /// one chain: the broad phase reports the AABB pair (both colliders are
    /// registered); the `checkCollideWithOverride` port
    /// ([`MultiBody::colliders_collide`]) decides whether the narrow phase
    /// runs on it — adjacent links collide by default, `has_self_collision =
    /// false` and the per-link parent-collision flags opt out.
    #[test]
    fn self_collision_filtering() {
        // Two overlapping link colliders on one chain register and produce
        // one collider-collider broad-phase pair.
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(2, 1.0, Vector3::ONE, true, false);
        // Both links sit at the base origin with overlapping spheres.
        mb.setup_revolute(
            0,
            1.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::ZERO,
        );
        mb.setup_revolute(
            1,
            1.0,
            Vector3::ONE,
            Some(0),
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::ZERO,
        );
        mb.finalize_multidof();
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        mb.set_link_collider(
            1,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        world.step_simulation(1.0 / 60.0);
        let collider_pairs = world
            .broadphase
            .compute_pairs()
            .into_iter()
            .filter(|(a, b)| {
                matches!(world.proxy_owners[*a], ProxyOwner::Collider(_))
                    && matches!(world.proxy_owners[*b], ProxyOwner::Collider(_))
            })
            .count();
        assert_eq!(collider_pairs, 1);

        // The filter itself (`checkCollideWithOverride` port) on a 3-link
        // chain: base - 0 - 1 - 2. `-1` is the base.
        let mut mb = crate::multibody::MultiBody::new(3, 1.0, Vector3::ONE, true, false);
        for i in 0..3 {
            mb.setup_revolute(
                i,
                1.0,
                Vector3::ONE,
                if i == 0 { None } else { Some(i - 1) },
                Quaternion::IDENTITY,
                Vector3::Z,
                Vector3::ZERO,
                Vector3::ZERO,
            );
        }
        mb.finalize_multidof();
        // Default: everything collides (Bullet's default).
        assert!(mb.colliders_collide(0, 1));
        assert!(mb.colliders_collide(-1, 0));
        assert!(mb.colliders_collide(0, 2)); // non-adjacent
        // Whole-body opt-out.
        mb.set_has_self_collision(false);
        assert!(!mb.colliders_collide(0, 1));
        assert!(!mb.colliders_collide(-1, 0));
        mb.set_has_self_collision(true);
        // Direct-parent opt-out on link 1: only (1, 0) is excluded.
        mb.set_link_parent_collision_disabled(1, false);
        assert!(!mb.colliders_collide(1, 0));
        assert!(mb.colliders_collide(1, 2));
        assert!(mb.colliders_collide(1, -1));
        // All-ancestors opt-out on link 2: parent and grandparent excluded,
        // others not (fresh body to keep the flags independent).
        let mut mb = crate::multibody::MultiBody::new(3, 1.0, Vector3::ONE, true, false);
        for i in 0..3 {
            mb.setup_revolute(
                i,
                1.0,
                Vector3::ONE,
                if i == 0 { None } else { Some(i - 1) },
                Quaternion::IDENTITY,
                Vector3::Z,
                Vector3::ZERO,
                Vector3::ZERO,
            );
        }
        mb.finalize_multidof();
        mb.set_link_parent_collision_disabled(2, true);
        assert!(!mb.colliders_collide(2, 1));
        assert!(!mb.colliders_collide(2, 0));
        // The base is part of the ancestor walk, like Bullet's loop reaching
        // m_parent == -1.
        assert!(!mb.colliders_collide(2, -1));
        // The exclusion is symmetric (Bullet checks both sides), and pairs
        // without a flagged member are unaffected.
        assert!(!mb.colliders_collide(1, 2));
        assert!(mb.colliders_collide(0, 1));
    }

    /// A fixed-base pendulum whose link sphere falls onto a static plane is
    /// supported by the multibody contact rows: it settles hanging straight
    /// down with the sphere resting on the plane (phase C contact response).
    #[test]
    fn link_sphere_rests_on_ground() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
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
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        // Pivot 1.51 m up: the link center's lowest arc point is y = 0.51,
        // so the sphere (r = 0.5) starts 0.01 m inside the plane — the
        // contact rows must support it hanging straight down.
        mb.base_pos = Vector3::new(0.0, 1.51, 0.0);
        world.add_multibody(mb);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            world.step_simulation(dt);
        }
        let body = world.multibody(MultiBodyHandle::from_index(0));
        let q = body.joint_pos_multidof(0)[0];
        let qd = body.joint_vel_multidof(0)[0];
        assert!(
            q.abs() < 0.05,
            "pendulum did not settle hanging down: q = {q}"
        );
        assert!(qd.abs() < 0.1, "pendulum still swinging: qd = {qd}");
        // The link center rests with the sphere on the plane: x ≈ 0,
        // y ≈ 0.5 (within the slop/damping tolerance).
        let p = body.link_world_position(0);
        assert_relative_eq!(p.x, 0.0, epsilon = 0.02);
        assert_relative_eq!(p.y, 0.5, epsilon = 0.05);
    }

    /// A swinging pendulum link kicks a dynamic rigid box lying in its path:
    /// the multibody contact rows exchange impulse with the rigid solver
    /// (box wakes and moves, the pendulum keeps swinging reduced).
    #[test]
    fn swinging_link_pushes_rigid_box() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // Ball mass 3 vs box mass 1: probe-verified against stock Bullet —
        // a mass-3 ball knocks the box ~1.1 m away, a mass-1 ball only
        // ~0.05 m (the persistent four-corner manifold holds a resting box
        // much steadier than the pre-merge single-point behavior).
        let mut mb = crate::multibody::MultiBody::new(1, 3.0, Vector3::ONE, true, false);
        mb.setup_revolute(
            0,
            3.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(0.0, -1.0, 0.0),
        );
        mb.finalize_multidof();
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            0.3,
            0.0,
        );
        // Pivot 1 m up; release from the horizontal (+X side).
        mb.base_pos = Vector3::new(0.0, 1.0, 0.0);
        mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
        world.add_multibody(mb);
        // A box resting on the plane where the ball's arc crosses its right
        // face, deliberately clear of the ball's release pose: when the box
        // sat under the release point, the ball just came to rest on the box
        // top instead of kicking it (stock-Bullet probe verified).
        let box_start = Vector3::new(0.85, 0.3, 0.0);
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.3)),
                box_start,
                1.0,
            )
            .friction(0.0),
        );
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            world.step_simulation(dt);
        }
        // The box was hit and knocked away from its start pose. At the
        // impact point (≈45° along the arc) the ball's velocity points
        // (−X, −Y), so the box is shoved toward −X / tipped over. The exact
        // slide direction is chaotic though: the ball pins the box into the
        // plane and the frictionless box drifts with the contact details
        // (the persistent-manifold warm start shifts the trajectory within
        // this noise), so the direction check carries a small tolerance.
        let box_end = world.body(BodyHandle::from_index(1)).position;
        assert!(
            (box_end - box_start).length() > 0.1,
            "box was not pushed: {box_end:?}"
        );
        assert!(
            box_end.x < box_start.x + 0.05,
            "box pushed the wrong way: {box_end:?}"
        );
    }

    /// A multibody contact wakes a sleeping rigid body: the sleeping box is
    /// struck by the swinging link and moves.
    #[test]
    fn link_contact_wakes_sleeping_rigid_body() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        // The victim: a box that settles on the plane and sleeps, clear of
        // the ball's release pose (see the swinging test for the geometry
        // and mass rationale — both stock-Bullet probe verified).
        let box_idx = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.3)),
            Vector3::new(0.85, 0.3, 0.0),
            1.0,
        ));
        for _ in 0..180 {
            // 3 s > the 2 s deactivation window.
            world.step_simulation(1.0 / 60.0);
        }
        assert!(!world.body(box_idx).is_active(), "box should be asleep");
        // Swing a link sphere into it (mass 3: a mass-1 ball only nudges
        // the four-corner-supported box ~0.01 m — probe-verified).
        let mut mb = crate::multibody::MultiBody::new(1, 3.0, Vector3::ONE, true, false);
        mb.setup_revolute(
            0,
            3.0,
            Vector3::ONE,
            None,
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::new(0.0, -1.0, 0.0),
        );
        mb.finalize_multidof();
        mb.set_link_collider(
            0,
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            0.3,
            0.0,
        );
        mb.base_pos = Vector3::new(0.0, 1.0, 0.0);
        mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
        world.add_multibody(mb);
        for _ in 0..90 {
            world.step_simulation(1.0 / 60.0);
        }
        assert!(
            world.body(box_idx).is_active(),
            "box should have been woken"
        );
        // The wake is the point of this test; the shove is a gentle nudge
        // (~0.02 m here, ~0.09 m in stock Bullet — after the impact the
        // ball comes to rest wedged on the box's top corner, and the exact
        // resting pose is chaotic).
        assert!(
            (world.body(box_idx).position - Vector3::new(0.85, 0.3, 0.0)).length() > 0.015,
            "sleeping box was not knocked away: {:?}",
            world.body(box_idx).position
        );
    }

    /// A free-base multibody with a box collider falls onto a static plane
    /// and is supported by its base contact rows (rest height, zero
    /// velocity).
    #[test]
    fn free_base_with_collider_lands_on_ground() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        mb.set_linear_damping(0.0);
        mb.finalize_multidof();
        mb.set_base_collider(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            0.5,
            0.0,
        );
        mb.base_pos = Vector3::new(0.0, 3.0, 0.0);
        world.add_multibody(mb);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            world.step_simulation(dt);
        }
        let body = world.multibody(MultiBodyHandle::from_index(0));
        // The half-extent box rests on the plane: y ≈ 0.5 within the slop
        // tolerance, with a small residual velocity (damped settling).
        assert_relative_eq!(body.base_pos().y, 0.5, epsilon = 0.05);
        assert!(
            body.base_vel().length() < 0.2,
            "base still moving: {:?}",
            body.base_vel()
        );
    }

    /// A fast rigid sphere hitting a free multibody (equal masses, head-on,
    /// restitution 0, gravity off) converges to the shared inelastic
    /// velocity: the mixed-row iteration must count the assembly-time
    /// relative velocity only once — reading the rigid side's impulse delta,
    /// not its full velocity. The double-count regression inflated the
    /// impulse ~2.4x, leaving the ball nearly stopped and the multibody
    /// overspeeding.
    #[test]
    fn rb_mb_contact_impulse_not_doubled() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
        mb.set_linear_damping(0.0);
        mb.set_angular_damping(0.0);
        mb.finalize_multidof();
        mb.set_base_collider(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            0.5,
            0.0,
        );
        world.add_multibody(mb);
        let ball = world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.5),
                Vector3::new(0.0, 3.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -6.0, 0.0))
            .disable_deactivation(),
        );
        let dt = 1.0 / 60.0;
        for _ in 0..25 {
            world.step_simulation(dt);
        }
        let ball_vy = world.body(ball).linear_velocity.y;
        let mb_vy = world.multibody(MultiBodyHandle::from_index(0)).base_vel().y;
        assert!(mb_vy < -1.5, "multibody was not kicked: {mb_vy}");
        // Equal-mass head-on, restitution 0: both sides leave at ~-3 m/s
        // (plus a small Baumgarte bounce that separates them slightly).
        assert!(
            (ball_vy - mb_vy).abs() < 1.5,
            "velocities did not converge: ball {ball_vy} vs multibody {mb_vy}"
        );
    }
}