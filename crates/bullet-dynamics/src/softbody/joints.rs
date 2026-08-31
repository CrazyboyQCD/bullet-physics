//! Soft-body contact joints — port of `btSoftBody::CJoint`
//! (`btSoftBody.cpp` 3675-3747) and the `Body` impulse dispatch it rides on
//! (`btSoftBody.h` 515-620).
//!
//! The linear/angular joint kinds (`LJoint`/`AJoint`) belong to phase E; the
//! contact joint is the one the cluster colliders (`CL_RS`/`CL_SS`/
//! `CL_SELF`) create every step. Its lifetime is one solve: the colliders
//! push fresh joints, the cluster solver `Prepare`/`Solve`s them and
//! `Terminate` deletes them (`m_maxlife = 0`).

use bullet_math::{Matrix3, Real, Vector3};

use crate::body::RigidBody;

use super::body::{Impulse, Side, SoftBody};

/// One half of a computed joint impulse: the target [`Side`], the impulse
/// and the relative anchor it applies at.
pub(crate) type JointHalf = (Side, Impulse, Vector3);

/// Contact joint between two [`Side`]s (Bullet's `CJoint`): a one-step
/// velocity impulse plus an optional drift split, assembled by the cluster
/// colliders' `solve_contact`.
#[derive(Clone, Debug)]
pub struct CJoint {
    /// The two sides (A = the soft/cluster side that owns the joint).
    pub bodies: [Side; 2],
    /// Relative anchors (world, from each side's frame origin).
    pub rpos: [Vector3; 2],
    /// Contact normal (from B toward A).
    pub normal: Vector3,
    /// Friction factor (1 inside the friction cone, the coefficient
    /// outside).
    pub friction: Real,
    /// Impulse matrix `(MassMatrix(a) + MassMatrix(b))^-1`.
    pub massmatrix: Matrix3,
    /// Constraint-force mixing (1).
    pub cfm: Real,
    /// Error reduction (1, scaled by the config's cluster hardness).
    pub erp: Real,
    /// Drift split factor (1, scaled by the config's cluster split).
    pub split: Real,
    /// Per-iteration velocity correction.
    pub drift: Vector3,
    /// Position (drift) correction.
    pub sdrift: Vector3,
    /// Step counter (Bullet's `m_life`).
    pub life: i32,
    /// Deletion threshold (0 — one solve per joint).
    pub maxlife: i32,
    /// Whether `cleanup` should drop the joint.
    pub delete: bool,
}

/// Bullet's two-sided `ImpulseMatrix(ima, iia, ra, imb, iib, rb)`:
/// `(MassMatrix(ima, iia, ra) + MassMatrix(imb, iib, rb))^-1`.
pub(crate) fn impulse_matrix_two(
    (ima, iia, ra): (Real, Matrix3, Vector3),
    (imb, iib, rb): (Real, Matrix3, Vector3),
) -> Matrix3 {
    (super::collide::mass_matrix(ima, &iia, ra) + super::collide::mass_matrix(imb, &iib, rb))
        .inverse()
}

/// Read accessors for one [`Side`] (the slice of Bullet's `Body` getters the
/// contact joint needs): a side may live in any soft body or the rigid
/// array, so all take the owning slices. The frame/inertia getters join
/// with the linear/angular joints of phase E.
pub(crate) mod side {
    use super::*;

    /// The side's velocity at a relative offset (linear + angular cross).
    pub fn velocity(side: Side, rpos: Vector3, softs: &[SoftBody], bodies: &[RigidBody]) -> Vector3 {
        match side {
            Side::Cluster { soft, cluster } => {
                let c = &softs[soft].clusters[cluster];
                c.lv + c.av.cross(rpos)
            }
            Side::Rigid(i) => bodies[i].velocity_at_local_point(rpos),
        }
    }

    /// Activate the side (Bullet's `Joint::Prepare` wakes both bodies;
    /// soft bodies never sleep so only the rigid side acts).
    pub fn activate(side: Side, bodies: &mut [RigidBody]) {
        if let Side::Rigid(i) = side {
            bodies[i].activate();
        }
    }
}

/// Write helpers for one [`Side`] (the slice of Bullet's `Body::apply*`
/// family the contact joint needs; the angular/central variants join with
/// the linear/angular joints of phase E).
pub(crate) mod side_apply {
    use super::super::cluster::{cluster_d_impulse, cluster_v_impulse};
    use super::*;

    /// Apply a velocity-space impulse at a relative offset.
    pub fn v_impulse(side: Side, rpos: Vector3, impulse: Vector3, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match side {
            Side::Cluster { soft, cluster } => {
                cluster_v_impulse(&mut softs[soft].clusters[cluster], rpos, impulse)
            }
            Side::Rigid(i) => bodies[i].apply_impulse(impulse, rpos),
        }
    }

    /// Apply a drift-space impulse at a relative offset.
    pub fn d_impulse(side: Side, rpos: Vector3, impulse: Vector3, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match side {
            Side::Cluster { soft, cluster } => {
                cluster_d_impulse(&mut softs[soft].clusters[cluster], rpos, impulse)
            }
            Side::Rigid(i) => bodies[i].apply_impulse(impulse, rpos),
        }
    }

    /// Apply a combined [`Impulse`] at a relative offset (Bullet's
    /// `Body::applyImpulse`).
    pub fn impulse(
        side: Side,
        impulse: Impulse,
        rpos: Vector3,
        softs: &mut [SoftBody],
        bodies: &mut [RigidBody],
    ) {
        if impulse.as_velocity {
            v_impulse(side, rpos, impulse.velocity, softs, bodies);
        }
        if impulse.as_drift {
            d_impulse(side, rpos, impulse.drift, softs, bodies);
        }
    }
}

impl CJoint {
    /// Bullet's `CJoint::Prepare`: wake both sides, age the joint one step
    /// (a `maxlife` of 0 deletes it after this solve) and scale the drift by
    /// erp/dt split across the solver iterations.
    pub fn prepare(&mut self, dt: Real, iterations: u32, bodies: &mut [RigidBody]) {
        side::activate(self.bodies[0], bodies);
        side::activate(self.bodies[1], bodies);
        let dodrift = self.life == 0;
        self.life += 1;
        self.delete = self.life > self.maxlife;
        if dodrift {
            self.drift *= self.erp / dt;
            if self.split > 0.0 {
                self.sdrift = self.massmatrix * (self.drift * self.split);
                self.drift *= 1.0 - self.split;
            }
            self.drift /= iterations as Real;
        } else {
            self.drift = Vector3::ZERO;
            self.sdrift = Vector3::ZERO;
        }
    }

    /// Read half of Bullet's `CJoint::Solve`: compute the two half-impulses
    /// `(side, impulse, relative anchor)` from immutable state — side A
    /// carries the negated impulse, side B the positive one. The velocity
    /// impulse is the drift plus (when approaching) the normal cancellation
    /// and in-plane friction, mapped through the impulse matrix.
    /// Self-collision joints (both sides in one soft body) only produce
    /// impulses above the body's `max_self_collision_impulse`, scaled by
    /// `self_collision_impulse_factor` — below the cap nothing is applied
    /// (Bullet's quirk, ported verbatim).
    pub(crate) fn compute(
        &self,
        sor: Real,
        softs: &[SoftBody],
        bodies: &[RigidBody],
    ) -> Option<(JointHalf, JointHalf)> {
        let va = side::velocity(self.bodies[0], self.rpos[0], softs, bodies);
        let vb = side::velocity(self.bodies[1], self.rpos[1], softs, bodies);
        let vrel = va - vb;
        let rvac = vrel.dot(self.normal);
        let mut impulse = Impulse {
            as_velocity: true,
            velocity: self.drift,
            ..Impulse::default()
        };
        if rvac < 0.0 {
            let iv = self.normal * rvac;
            let fv = vrel - iv;
            impulse.velocity += iv + fv * self.friction;
        }
        impulse.velocity = self.massmatrix * impulse.velocity * sor;

        let half_a = (self.bodies[0], -impulse, self.rpos[0]);
        let half_b = (self.bodies[1], impulse, self.rpos[1]);
        let self_collision = match (self.bodies[0], self.bodies[1]) {
            (Side::Cluster { soft: s1, .. }, Side::Cluster { soft: s2, .. }) if s1 == s2 => {
                Some(s1)
            }
            _ => None,
        };
        if let Some(soft) = self_collision {
            let cap = softs[soft].max_self_collision_impulse;
            let factor = softs[soft].self_collision_impulse_factor;
            // NaN-guarded like Bullet's `impulse.getX() == impulse.getX()`.
            if impulse.velocity.is_finite() && impulse.velocity.length() >= cap {
                let (a, imp, r) = half_a;
                let (b, imp_b, rb) = half_b;
                Some((
                    (a, imp * factor, r),
                    (b, imp_b * factor, rb),
                ))
            } else {
                None
            }
        } else {
            Some((half_a, half_b))
        }
    }
}

/// Multiply an [`Impulse`] by a scalar (Bullet's `operator*`).
impl std::ops::Mul<Real> for Impulse {
    type Output = Impulse;
    fn mul(self, x: Real) -> Impulse {
        Impulse {
            velocity: self.velocity * x,
            drift: self.drift * x,
            as_velocity: self.as_velocity,
            as_drift: self.as_drift,
        }
    }
}
