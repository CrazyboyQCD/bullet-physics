//! Multibody contact rows — the contact part of Bullet's
//! `btMultiBodyConstraintSolver`: `convertMultiBodyContact` +
//! `setupMultiBodyContactConstraint` (row assembly), the mixed
//! multibody/rigid branches of `resolveSingleConstraintRowGeneric` (the PGS
//! row solve) and `writeBackSolverBodyToMultiBody` (impulse + feedback
//! writeback). See `docs/multibody-collision.md` §3 for the recipe.
//!
//! Unlike the non-contact [`SolverRow`](super::constraint::SolverRow), either
//! side of a contact row can be a rigid body: the rigid side's response is
//! the classic `{ n, invI·(r×n) }` row applied directly to the body's real
//! velocities (our solver mutates real velocities, which is what Bullet's
//! `internalApplyImpulse` does on solver bodies), while the multibody side
//! accumulates into the shared `deltaVelocities` pool and only touches the
//! generalized velocities at writeback (Bullet compiles without
//! `DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS`).

use bullet_collision::manifold::{ContactManifold, ContactPoint};
use bullet_math::{Real, Vector3};

use super::body::{MultiBody, MultiBodyScratch};
use super::constraint::{plane_space1, SolverData};
use crate::body::RigidBody;
use crate::solver::sequential_impulse::{RESTITUTION_THRESHOLD, SLOP};

/// ERP for multibody contact rows — Bullet's `m_erp2` (0.2), the same value
/// the rigid contact solver's Baumgarte uses, so mixed islands correct
/// penetration at one rate.
const MB_CONTACT_ERP: Real = 0.2;
/// Upper impulse bound of a normal row (Bullet's `m_upperLimit = 1e10`).
const NORMAL_ROW_UPPER: Real = 1e10;
/// CFM for contact rows (Bullet's `m_globalCfm = 0` scaled by `1/dt`).
const CFM: Real = 0.0;

/// One side of a multibody contact: a rigid body or a multibody base/link
/// collider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContactSide {
    Rigid(usize),
    /// `link < 0` is the base (Bullet's `m_link = -1`).
    Link { mb: usize, link: i32 },
}

/// Precomputed per-side response data (the per-side fields of Bullet's
/// `btMultiBodySolverConstraint`).
///
/// Sign convention: the row measures `J·v = (v_B − v_A)·dir + arms`, so side
/// A carries `−dir` and side B `+dir` — an approaching pair has a negative
/// row velocity and a positive normal impulse, like the rigid solver's
/// `(vb − va)·n`.
enum SideData {
    /// The rigid side: the signed normal as this side sees it, the angular
    /// jacobian arm `r × normal`, its inertia-scaled response `invI · arm`,
    /// the inverse mass, and the row projection at row-assembly time.
    ///
    /// The iteration reads this side's **impulse delta** (the current
    /// projection minus the snapshot), not the real velocity: the rhs already
    /// carries `-rel_vel`, so counting the full velocity again would apply the
    /// relative velocity twice (Bullet reads its solver body's
    /// delta-velocity pool for the same reason). The snapshot predates the
    /// warm-start impulses, which are part of the delta like in Bullet.
    Rigid {
        contact_normal: Vector3,
        ang: Vector3,
        angular_component: Vector3,
        inv_mass: Real,
        initial_proj: Real,
    },
    /// The multibody side: this row's jacobian / unit-response block in
    /// [`SolverData`] and the body's shared `deltaVelocities` block.
    MultiBody {
        jac_index: usize,
        delta_vel_index: usize,
        ndof: usize,
    },
}

/// One contact solver row with possibly mixed sides (normal or friction).
pub(crate) struct MbContactRow {
    side_a: ContactSide,
    side_b: ContactSide,
    data_a: SideData,
    data_b: SideData,
    /// Contact normal, A → B (our engine-wide convention).
    normal: Vector3,
    /// `rel_pos × normal` per side (A: `+`, B mirrored `−`), for the
    /// feedback writeback (`m_relpos1CrossNormal` / `m_relpos2CrossNormal`).
    rel_pos_a: Vector3,
    rel_pos_b: Vector3,
    jac_diag_ab_inv: Real,
    rhs: Real,
    lower_limit: Real,
    upper_limit: Real,
    applied_impulse: Real,
    /// Coulomb coefficient for friction rows (0 on normal rows).
    friction: Real,
    /// Friction rows: the index of the paired normal row whose accumulated
    /// impulse bounds this row.
    normal_row: Option<usize>,
}

/// A collected mixed/collider contact manifold waiting for row conversion:
/// the two sides plus the combined surface parameters (products, like rigid
/// contacts).
pub(crate) struct PendingContact {
    pub side_a: ContactSide,
    pub side_b: ContactSide,
    pub friction: Real,
    pub restitution: Real,
    pub manifold: ContactManifold,
}

/// Convert one contact point into a row along `dir` (the normal or a tangent
/// direction), mirroring `setupMultiBodyContactConstraint`.
#[allow(clippy::too_many_arguments)]
fn setup_contact_row(
    side_a: ContactSide,
    side_b: ContactSide,
    dir: Vector3,
    cp: &ContactPoint,
    is_friction: bool,
    friction: Real,
    restitution: Real,
    bodies: &[RigidBody],
    multibodies: &mut [MultiBody],
    data: &mut SolverData,
    scratch: &mut MultiBodyScratch,
    inv_dt: Real,
    normal_row: Option<usize>,
) -> MbContactRow {
    let pos = cp.position;

    // --- Side A ---
    let (data_a, rel_pos_a) = match side_a {
        ContactSide::Rigid(i) => {
            let body = &bodies[i];
            let rel = pos - body.position;
            let contact_normal = -dir;
            let ang = rel.cross(contact_normal);
            let angular_component = body.inv_inertia_world() * ang;
            (
                SideData::Rigid {
                    contact_normal,
                    ang,
                    angular_component,
                    inv_mass: body.inv_mass(),
                    initial_proj: contact_normal.dot(body.linear_velocity)
                        + ang.dot(body.angular_velocity),
                },
                rel,
            )
        }
        ContactSide::Link { mb, link } => {
            let body = &mut multibodies[mb];
            let ndof = 6 + body.num_dofs();
            let mut delta_vel_index = body.companion_id;
            if delta_vel_index < 0 {
                delta_vel_index = data.delta_velocities.len() as i32;
                body.companion_id = delta_vel_index;
                data.delta_velocities.resize(data.delta_velocities.len() + ndof, 0.0);
            }
            let jac_index = data.jacobians.len();
            data.jacobians.resize(jac_index + ndof, 0.0);
            data.delta_velocities_unit_impulse
                .resize(jac_index + ndof, 0.0);
            {
                let jac = &mut data.jacobians[jac_index..jac_index + ndof];
                body.fill_constraint_jacobian(link, pos, Vector3::ZERO, -dir, jac, scratch);
            }
            {
                let jac = data.jacobians[jac_index..jac_index + ndof].to_vec();
                let delta =
                    &mut data.delta_velocities_unit_impulse[jac_index..jac_index + ndof];
                body.calc_acceleration_deltas(&jac, delta, scratch);
            }
            let rel = pos - side_com(side_a, multibodies);
            (
                SideData::MultiBody {
                    jac_index,
                    delta_vel_index: delta_vel_index as usize,
                    ndof,
                },
                rel,
            )
        }
    };

    // --- Side B ---
    let (data_b, rel_pos_b) = match side_b {
        ContactSide::Rigid(i) => {
            let body = &bodies[i];
            let rel = pos - body.position;
            let contact_normal = dir;
            let ang = rel.cross(contact_normal);
            let angular_component = body.inv_inertia_world() * ang;
            (
                SideData::Rigid {
                    contact_normal,
                    ang,
                    angular_component,
                    inv_mass: body.inv_mass(),
                    initial_proj: contact_normal.dot(body.linear_velocity)
                        + ang.dot(body.angular_velocity),
                },
                rel,
            )
        }
        ContactSide::Link { mb, link } => {
            let body = &mut multibodies[mb];
            let ndof = 6 + body.num_dofs();
            let mut delta_vel_index = body.companion_id;
            if delta_vel_index < 0 {
                delta_vel_index = data.delta_velocities.len() as i32;
                body.companion_id = delta_vel_index;
                data.delta_velocities.resize(data.delta_velocities.len() + ndof, 0.0);
            }
            let jac_index = data.jacobians.len();
            data.jacobians.resize(jac_index + ndof, 0.0);
            data.delta_velocities_unit_impulse
                .resize(jac_index + ndof, 0.0);
            {
                let jac = &mut data.jacobians[jac_index..jac_index + ndof];
                body.fill_constraint_jacobian(link, pos, Vector3::ZERO, dir, jac, scratch);
            }
            {
                let jac = data.jacobians[jac_index..jac_index + ndof].to_vec();
                let delta =
                    &mut data.delta_velocities_unit_impulse[jac_index..jac_index + ndof];
                body.calc_acceleration_deltas(&jac, delta, scratch);
            }
            let rel = pos - side_com(side_b, multibodies);
            (
                SideData::MultiBody {
                    jac_index,
                    delta_vel_index: delta_vel_index as usize,
                    ndof,
                },
                rel,
            )
        }
    };

    // --- Effective mass: jacDiagABInv = sor / (Σ jac·unit + cfm) ---
    let mut denom = 0.0;
    if let SideData::MultiBody {
        jac_index,
        ndof,
        ..
    } = data_a
    {
        let jac = &data.jacobians[jac_index..jac_index + ndof];
        let unit = &data.delta_velocities_unit_impulse[jac_index..jac_index + ndof];
        for i in 0..ndof {
            denom += jac[i] * unit[i];
        }
    } else if let SideData::Rigid {
        contact_normal,
        angular_component,
        inv_mass,
        ..
    } = data_a
    {
        let vec = angular_component.cross(rel_pos_a);
        denom += inv_mass + contact_normal.dot(vec);
    }
    if let SideData::MultiBody {
        jac_index,
        ndof,
        ..
    } = data_b
    {
        let jac = &data.jacobians[jac_index..jac_index + ndof];
        let unit = &data.delta_velocities_unit_impulse[jac_index..jac_index + ndof];
        for i in 0..ndof {
            denom += jac[i] * unit[i];
        }
    } else if let SideData::Rigid {
        contact_normal,
        angular_component,
        inv_mass,
        ..
    } = data_b
    {
        let vec = angular_component.cross(rel_pos_b);
        denom += inv_mass + contact_normal.dot(vec);
    }
    let jac_diag_ab_inv = if denom > Real::EPSILON {
        1.0 / (denom + CFM)
    } else {
        0.0 // singular row: disabled
    };

    // --- Relative velocity along the row at setup time ---
    let mut rel_vel = 0.0;
    match (&data_a, side_a) {
        (SideData::MultiBody { jac_index, ndof, .. }, ContactSide::Link { mb, .. }) => {
            let jac = &data.jacobians[*jac_index..*jac_index + *ndof];
            let vel = multibodies[mb].velocity_vector();
            for i in 0..*ndof {
                rel_vel += vel[i] * jac[i];
            }
        }
        (SideData::Rigid { contact_normal, ang, .. }, ContactSide::Rigid(i)) => {
            // Full point velocity; our velocities are already
            // gravity-integrated at solve time (Bullet predicts the gravity
            // contribution through the accumulated force instead — recorded
            // deviation, same one-step phase class as the rigid predicted
            // pose).
            let body = &bodies[i];
            rel_vel += contact_normal.dot(body.linear_velocity)
                + ang.dot(body.angular_velocity);
        }
        _ => unreachable!("side/data mismatch"),
    }
    match (&data_b, side_b) {
        (SideData::MultiBody { jac_index, ndof, .. }, ContactSide::Link { mb, .. }) => {
            let jac = &data.jacobians[*jac_index..*jac_index + *ndof];
            let vel = multibodies[mb].velocity_vector();
            for i in 0..*ndof {
                rel_vel += vel[i] * jac[i];
            }
        }
        (SideData::Rigid { contact_normal, ang, .. }, ContactSide::Rigid(i)) => {
            let body = &bodies[i];
            rel_vel += contact_normal.dot(body.linear_velocity)
                + ang.dot(body.angular_velocity);
        }
        _ => unreachable!("side/data mismatch"),
    }

    // --- RHS (Bullet's positionalError / velocityError recipe) ---
    let mut positional_error = 0.0;
    let mut velocity_error;
    if is_friction {
        // No friction anchors: no positional term.
        velocity_error = -rel_vel;
    } else {
        let restitution = if rel_vel < -RESTITUTION_THRESHOLD {
            -restitution * rel_vel
        } else {
            0.0
        };
        velocity_error = restitution - rel_vel;
        // Bullet's `distance = getDistance() + slop` with its
        // negative-on-penetration sign convention; our `penetration` is
        // positive on penetration, so separating distance = slop − pen.
        // Engine-wide slop (deviation: Bullet's default linearSlop is 0).
        let distance = SLOP - cp.penetration;
        if distance > 0.0 {
            velocity_error -= distance * inv_dt;
        } else {
            positional_error = -distance * MB_CONTACT_ERP * inv_dt;
        }
    }
    let rhs = (positional_error + velocity_error) * jac_diag_ab_inv;

    MbContactRow {
        side_a,
        side_b,
        data_a,
        data_b,
        normal: dir,
        rel_pos_a: rel_pos_a.cross(dir),
        rel_pos_b: -(rel_pos_b.cross(dir)),
        jac_diag_ab_inv,
        rhs,
        lower_limit: 0.0,
        // Friction bounds are refreshed from the paired normal impulse each
        // sweep; the normal row is one-sided `[0, 1e10]`.
        upper_limit: if is_friction { 0.0 } else { NORMAL_ROW_UPPER },
        applied_impulse: 0.0,
        friction: if is_friction { friction } else { 0.0 },
        normal_row,
    }
}

/// The center of mass (frame origin) of a contact side.
fn side_com(side: ContactSide, multibodies: &[MultiBody]) -> Vector3 {
    match side {
        ContactSide::Rigid(_) => Vector3::ZERO, // unused for rigid sides
        ContactSide::Link { mb, link } => {
            let m = &multibodies[mb];
            if link < 0 {
                m.base_pos()
            } else {
                m.link_world_position(link as usize)
            }
        }
    }
}

/// Convert every contact point of a pending manifold into three rows
/// (normal + two tangents, Bullet's `convertMultiBodyContact` with the
/// two-friction-direction engine convention).
pub(crate) fn convert_contact(
    pending: &PendingContact,
    bodies: &[RigidBody],
    multibodies: &mut [MultiBody],
    data: &mut SolverData,
    scratch: &mut MultiBodyScratch,
    inv_dt: Real,
    out: &mut Vec<MbContactRow>,
) {
    for cp in &pending.manifold.points {
        let normal_index = out.len();
        out.push(setup_contact_row(
            pending.side_a,
            pending.side_b,
            cp.normal,
            cp,
            false,
            pending.friction,
            pending.restitution,
            bodies,
            multibodies,
            data,
            scratch,
            inv_dt,
            None,
        ));
        let (t1, t2) = plane_space1(cp.normal);
        for dir in [t1, t2] {
            out.push(setup_contact_row(
                pending.side_a,
                pending.side_b,
                dir,
                cp,
                true,
                pending.friction,
                pending.restitution,
                bodies,
                multibodies,
                data,
                scratch,
                inv_dt,
                Some(normal_index),
            ));
        }
    }
}

/// Solve one contact row (Bullet's `resolveSingleConstraintRowGeneric` with
/// mixed sides): compute the impulse needed to meet `rhs` given both sides'
/// current velocities, clamp it to the row window, and apply it.
fn solve_row(row: &mut MbContactRow, data: &mut SolverData, bodies: &mut [RigidBody]) {
    if row.jac_diag_ab_inv == 0.0 {
        return; // disabled singular row
    }
    let mut delta_impulse = row.rhs - row.applied_impulse * CFM;

    // Velocity of each side along the row, given everything applied so far.
    // Both sides contribute only their **impulse delta** since assembly (the
    // multibody side's pool starts at zero, the rigid side's real velocity
    // minus its assembly-time snapshot) — the rhs already contains the
    // assembly-time relative velocity, and Bullet reads delta pools for the
    // same reason (`m_deltaLinearVelocity` / `m_deltaVelocities`).
    let delta_a = match (&row.data_a, row.side_a) {
        (SideData::MultiBody { .. }, ContactSide::Link { .. }) => {
            read_mb_side(&row.data_a, data)
        }
        (
            SideData::Rigid { contact_normal, ang, initial_proj, .. },
            ContactSide::Rigid(i),
        ) => {
            let body = &bodies[i];
            contact_normal.dot(body.linear_velocity) + ang.dot(body.angular_velocity)
                - initial_proj
        }
        _ => unreachable!("side/data mismatch"),
    };
    let delta_b = match (&row.data_b, row.side_b) {
        (SideData::MultiBody { .. }, ContactSide::Link { .. }) => {
            read_mb_side(&row.data_b, data)
        }
        (
            SideData::Rigid { contact_normal, ang, initial_proj, .. },
            ContactSide::Rigid(i),
        ) => {
            let body = &bodies[i];
            contact_normal.dot(body.linear_velocity) + ang.dot(body.angular_velocity)
                - initial_proj
        }
        _ => unreachable!("side/data mismatch"),
    };
    delta_impulse -= (delta_a + delta_b) * row.jac_diag_ab_inv;

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

    apply_side(&row.data_a, row.side_a, data, bodies, delta_impulse);
    apply_side(&row.data_b, row.side_b, data, bodies, delta_impulse);
}

/// The multibody side's accumulated pool velocity along its row direction.
fn read_mb_side(data: &SideData, solver: &SolverData) -> Real {
    let SideData::MultiBody {
        jac_index,
        delta_vel_index,
        ndof,
    } = data
    else {
        unreachable!("not a multibody side");
    };
    let jac = &solver.jacobians[*jac_index..*jac_index + *ndof];
    let dv = &solver.delta_velocities[*delta_vel_index..*delta_vel_index + *ndof];
    let mut dot = 0.0;
    for i in 0..*ndof {
        dot += jac[i] * dv[i];
    }
    dot
}

/// Apply `delta_impulse` to one side (Bullet's `applyDeltaVee` for multibody
/// sides, `internalApplyImpulse` for rigid sides).
fn apply_side(
    data: &SideData,
    side: ContactSide,
    solver: &mut SolverData,
    bodies: &mut [RigidBody],
    delta_impulse: Real,
) {
    match (data, side) {
        (SideData::MultiBody { jac_index, delta_vel_index, ndof }, _) => {
            let unit =
                solver.delta_velocities_unit_impulse[*jac_index..*jac_index + *ndof].to_vec();
            let dv = &mut solver.delta_velocities[*delta_vel_index..*delta_vel_index + *ndof];
            for (v, u) in dv.iter_mut().zip(&unit) {
                *v += u * delta_impulse;
            }
        }
        (
            SideData::Rigid { contact_normal, angular_component, inv_mass, .. },
            ContactSide::Rigid(i),
        ) => {
            let body = &mut bodies[i];
            body.linear_velocity += *contact_normal * (delta_impulse * inv_mass);
            body.angular_velocity += *angular_component * delta_impulse;
        }
        _ => unreachable!("side/data mismatch"),
    }
}

/// One PGS sweep over the contact rows: the normal rows first, then the
/// friction rows with their bounds refreshed from the same-sweep normal
/// impulses (Bullet's `solveSingleIteration` pool order).
pub(crate) fn solve_contact_rows(
    rows: &mut [MbContactRow],
    data: &mut SolverData,
    bodies: &mut [RigidBody],
) {
    for row in rows.iter_mut() {
        if row.normal_row.is_none() {
            solve_row(row, data, bodies);
        }
    }
    // Snapshot the normal impulses for the friction bounds (a friction row
    // reads its paired normal row while holding a `&mut` over `rows`).
    let normal_impulses: Vec<Real> = rows.iter().map(|r| r.applied_impulse).collect();
    for row in rows.iter_mut() {
        if let Some(n) = row.normal_row {
            // Coulomb bound from the paired normal row's accumulated impulse;
            // unsolved normals (zero impulse) disable the friction row.
            let total = normal_impulses[n];
            if total > 0.0 {
                let bound = row.friction * total;
                row.lower_limit = -bound;
                row.upper_limit = bound;
                solve_row(row, data, bodies);
            } else {
                row.applied_impulse = 0.0;
            }
        }
    }
}

/// Write the solved contact impulses back (Bullet's
/// `writeBackSolverBodyToMultiBody`): each multibody side applies its
/// unit-impulse response scaled by the accumulated impulse (with the
/// `±maxCoordinateVelocity` clamp of `apply_delta_vee`), and the impulse is
/// accumulated into the per-link/base constraint forces consumed by the
/// joint feedback constraint pass. Unlike Bullet (whose manifold normal
/// points B→A), our `row.normal` points A→B, so the physical contact force
/// pushes side A along `−normal` and side B along `+normal` (the stored
/// `rel_pos_a` / `rel_pos_b` are pre-crossed with `normal`, hence the
/// matching sign flips on the torques).
pub(crate) fn writeback_contact_rows(
    rows: &[MbContactRow],
    data: &SolverData,
    multibodies: &mut [MultiBody],
    dt: Real,
) {
    for row in rows {
        if row.applied_impulse == 0.0 {
            continue;
        }
        let (n_a, t_a) = (-row.normal, -row.rel_pos_a);
        let (n_b, t_b) = (row.normal, -row.rel_pos_b);
        apply_side_writeback(&row.data_a, row.side_a, data, multibodies, row.applied_impulse, n_a, t_a, dt);
        apply_side_writeback(&row.data_b, row.side_b, data, multibodies, row.applied_impulse, n_b, t_b, dt);
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_side_writeback(
    data: &SideData,
    side: ContactSide,
    solver: &SolverData,
    multibodies: &mut [MultiBody],
    impulse: Real,
    normal: Vector3,
    relpos_cross_normal: Vector3,
    dt: Real,
) {
    let (SideData::MultiBody { jac_index, ndof, .. }, ContactSide::Link { mb, link }) = (data, side)
    else {
        return; // rigid sides live in the rigid solver's writeback
    };
    let body = &mut multibodies[mb];
    let unit = solver.delta_velocities_unit_impulse[*jac_index..*jac_index + *ndof].to_vec();
    body.apply_delta_vee(&unit, impulse);
    let force = normal * (impulse / dt);
    let torque = relpos_cross_normal * (impulse / dt);
    if link < 0 {
        body.add_base_constraint_force(force);
        body.add_base_constraint_torque(torque);
    } else {
        body.add_link_constraint_force(link as usize, force);
        body.add_link_constraint_torque(link as usize, torque);
    }
}
