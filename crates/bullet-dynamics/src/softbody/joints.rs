//! Soft-body joints — port of `btSoftBody`'s `Joint` hierarchy
//! (`btSoftBody.cpp` 3580-3747) and the `Body` impulse dispatch it rides on
//! (`btSoftBody.h` 515-620).
//!
//! Three joint kinds share the `Prepare`/`Solve`/`Terminate` lifecycle (the
//! [`SoftJoint`] enum stands in for Bullet's virtual dispatch): the contact
//! joint created every step by the cluster colliders (`CL_RS`/`CL_SS`/
//! `CL_SELF`, one solve per joint) and the user-installed linear/angular
//! joints that live until removed. The angular joint's motor hook is the
//! [`AngularControl`] trait (Bullet's `AJoint::IControl`).

use bullet_math::{math, Matrix3, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use crate::body::RigidBody;

use super::body::{Impulse, Side, SoftBody};

/// One half of a computed joint impulse: the target [`Side`], the impulse
/// and the relative anchor it applies at.
pub(crate) type JointHalf = (Side, Impulse, Vector3);

/// Bullet's `Clamp(v, maxlength)`: cap the length, keep the direction.
fn clamp_vec(v: Vector3, maxlength: Real) -> Vector3 {
    let sql = v.length_squared();
    if sql < maxlength * maxlength {
        v
    } else {
        v.normalize_or_zero() * maxlength
    }
}

/// Contact joint between two [`Side`]s (Bullet's `CJoint`): a one-step
/// velocity impulse plus an optional drift split, assembled by the cluster
/// colliders' `solve_contact`.
#[derive(Clone, Debug, Default)]
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
/// joints need): a side may live in any soft body or the rigid array, so all
/// take the owning slices.
pub(crate) mod side {
    use super::*;

    /// The side's velocity at a relative offset (linear + angular cross).
    pub fn velocity(side: Side, rpos: Vector3, softs: &[SoftBody], bodies: &[RigidBody]) -> Vector3 {
        match side {
            Side::Cluster { soft, cluster } => {
                let c = &softs[soft.index()].clusters[cluster];
                c.lv + c.av.cross(rpos)
            }
            Side::Rigid(i) => bodies[i.index()].velocity_at_local_point(rpos),
        }
    }

    /// The side's angular velocity (Bullet's `Body::angularVelocity()`).
    pub fn angular_velocity(side: Side, softs: &[SoftBody], bodies: &[RigidBody]) -> Vector3 {
        match side {
            Side::Cluster { soft, cluster } => softs[soft.index()].clusters[cluster].av,
            Side::Rigid(i) => bodies[i.index()].angular_velocity,
        }
    }

    /// The side's world frame as `(origin, basis)` (Bullet's `Body::xform`):
    /// the cluster frame (`m_framexform`) or the rigid world transform.
    pub fn xform(side: Side, softs: &[SoftBody], bodies: &[RigidBody]) -> (Vector3, Matrix3) {
        match side {
            Side::Cluster { soft, cluster } => {
                let c = &softs[soft.index()].clusters[cluster];
                (c.framepos, c.framerot)
            }
            Side::Rigid(i) => (
                bodies[i.index()].position,
                Matrix3::from_quat(bodies[i.index()].orientation),
            ),
        }
    }

    /// The side's inverse mass (Bullet's `Body::invMass`).
    pub fn inv_mass(side: Side, softs: &[SoftBody], bodies: &[RigidBody]) -> Real {
        match side {
            Side::Cluster { soft, cluster } => softs[soft.index()].clusters[cluster].imass,
            Side::Rigid(i) => bodies[i.index()].inv_mass(),
        }
    }

    /// The side's inverse world inertia (Bullet's `Body::invWorldInertia`).
    pub fn inv_world_inertia(side: Side, softs: &[SoftBody], bodies: &[RigidBody]) -> Matrix3 {
        match side {
            Side::Cluster { soft, cluster } => softs[soft.index()].clusters[cluster].invwi,
            Side::Rigid(i) => bodies[i.index()].inv_inertia_world(),
        }
    }

    /// Activate the side (Bullet's `Joint::Prepare` wakes both bodies;
    /// soft bodies never sleep so only the rigid side acts).
    pub fn activate(side: Side, bodies: &mut [RigidBody]) {
        if let Side::Rigid(i) = side {
            bodies[i.index()].activate();
        }
    }
}

/// Write helpers for one [`Side`] (the slice of Bullet's `Body::apply*`
/// family the joints need).
pub(crate) mod side_apply {
    use super::super::cluster::{cluster_d_impulse, cluster_da_impulse, cluster_v_impulse, cluster_va_impulse};
    use super::*;

    /// Apply a velocity-space impulse at a relative offset.
    pub fn v_impulse(side: Side, rpos: Vector3, impulse: Vector3, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match side {
            Side::Cluster { soft, cluster } => {
                cluster_v_impulse(&mut softs[soft.index()].clusters[cluster], rpos, impulse)
            }
            Side::Rigid(i) => bodies[i.index()].apply_impulse(impulse, rpos),
        }
    }

    /// Apply a drift-space impulse at a relative offset.
    pub fn d_impulse(side: Side, rpos: Vector3, impulse: Vector3, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match side {
            Side::Cluster { soft, cluster } => {
                cluster_d_impulse(&mut softs[soft.index()].clusters[cluster], rpos, impulse)
            }
            Side::Rigid(i) => bodies[i.index()].apply_impulse(impulse, rpos),
        }
    }

    /// Apply an angular velocity-space impulse (Bullet's
    /// `Body::applyVAImpulse`).
    pub fn va_impulse(side: Side, impulse: Vector3, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match side {
            Side::Cluster { soft, cluster } => {
                cluster_va_impulse(&mut softs[soft.index()].clusters[cluster], impulse)
            }
            Side::Rigid(i) => bodies[i.index()].apply_torque_impulse(impulse),
        }
    }

    /// Apply an angular drift-space impulse (Bullet's
    /// `Body::applyDAImpulse` — the rigid side maps to the same torque
    /// impulse).
    pub fn da_impulse(side: Side, impulse: Vector3, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match side {
            Side::Cluster { soft, cluster } => {
                cluster_da_impulse(&mut softs[soft.index()].clusters[cluster], impulse)
            }
            Side::Rigid(i) => bodies[i.index()].apply_torque_impulse(impulse),
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
            let cap = softs[soft.index()].max_self_collision_impulse;
            let factor = softs[soft.index()].self_collision_impulse_factor;
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

/// Bullet's `AngularImpulseMatrix(iia, iib)`: `(iia + iib)^-1`.
pub(crate) fn angular_impulse_matrix(iia: &Matrix3, iib: &Matrix3) -> Matrix3 {
    (*iia + *iib).inverse()
}

/// Construction parameters shared by the linear/angular joints (Bullet's
/// `Joint::Specs`).
#[derive(Clone, Copy, Debug)]
pub struct JointSpecs {
    /// Error reduction per step.
    pub erp: Real,
    /// Constraint-force mixing (how much relative velocity feeds back).
    pub cfm: Real,
    /// Split between the velocity and the drift (position) channel.
    pub split: Real,
}

impl Default for JointSpecs {
    fn default() -> Self {
        Self { erp: 1.0, cfm: 1.0, split: 1.0 }
    }
}

/// Linear joint parameters (Bullet's `LJoint::Specs`): pin the two sides'
/// frames together at one world position.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearJointSpecs {
    /// Shared tuning.
    pub tuning: JointSpecs,
    /// The anchor position in world space at installation time.
    pub position: Vector3,
}

/// A linear joint (Bullet's `LJoint`, `btSoftBody.cpp` 3587-3627): keeps a
/// point fixed relative to both sides' frames, solved as a three-axis
/// velocity impulse plus an optional drift split.
#[derive(Clone, Debug)]
pub struct LJoint {
    /// The two sides (A = the cluster the joint belongs to).
    pub bodies: [Side; 2],
    /// Frame-local anchors (Bullet's `m_refs`).
    pub refs: [Vector3; 2],
    /// Constraint-force mixing.
    pub cfm: Real,
    /// Error reduction.
    pub erp: Real,
    /// Drift split factor.
    pub split: Real,
    /// World anchors relative to each frame origin (refreshed by `prepare`).
    pub rpos: [Vector3; 2],
    /// Per-iteration velocity correction.
    pub drift: Vector3,
    /// Position (drift) correction.
    pub sdrift: Vector3,
    /// Impulse matrix (refreshed by `prepare`).
    pub massmatrix: Matrix3,
}

impl LJoint {
    /// Bullet's `LJoint::Prepare`: refresh the world anchors, the capped
    /// drift and the two-sided impulse matrix, splitting the correction
    /// across the solver iterations.
    pub fn prepare(&mut self, dt: Real, iterations: u32, softs: &[SoftBody], bodies: &mut [RigidBody]) {
        const MAXDRIFT: Real = 4.0;
        side::activate(self.bodies[0], bodies);
        side::activate(self.bodies[1], bodies);
        let (origin0, basis0) = side::xform(self.bodies[0], softs, bodies);
        let (origin1, basis1) = side::xform(self.bodies[1], softs, bodies);
        self.rpos[0] = basis0 * self.refs[0] + origin0;
        self.rpos[1] = basis1 * self.refs[1] + origin1;
        self.drift = clamp_vec(self.rpos[0] - self.rpos[1], MAXDRIFT) * self.erp / dt;
        self.rpos[0] -= origin0;
        self.rpos[1] -= origin1;
        self.massmatrix = impulse_matrix_two(
            (
                side::inv_mass(self.bodies[0], softs, bodies),
                side::inv_world_inertia(self.bodies[0], softs, bodies),
                self.rpos[0],
            ),
            (
                side::inv_mass(self.bodies[1], softs, bodies),
                side::inv_world_inertia(self.bodies[1], softs, bodies),
                self.rpos[1],
            ),
        );
        if self.split > 0.0 {
            self.sdrift = self.massmatrix * (self.drift * self.split);
            self.drift *= 1.0 - self.split;
        }
        self.drift /= iterations as Real;
    }

    /// Read half of Bullet's `LJoint::Solve`: the velocity-space impulse
    /// countering the anchor drift and the full relative velocity (side A
    /// carries the negated impulse).
    pub(crate) fn compute(
        &self,
        sor: Real,
        softs: &[SoftBody],
        bodies: &[RigidBody],
    ) -> (JointHalf, JointHalf) {
        let va = side::velocity(self.bodies[0], self.rpos[0], softs, bodies);
        let vb = side::velocity(self.bodies[1], self.rpos[1], softs, bodies);
        let vr = va - vb;
        let impulse = Impulse {
            as_velocity: true,
            velocity: self.massmatrix * (self.drift + vr * self.cfm) * sor,
            ..Impulse::default()
        };
        ((self.bodies[0], -impulse, self.rpos[0]), (self.bodies[1], impulse, self.rpos[1]))
    }
}

/// Angular joint parameters (Bullet's `AJoint::Specs`): align an axis of the
/// two sides' frames, optionally driven by a motor control. Consumed on
/// installation (the control box moves into the joint).
pub struct AngularJointSpecs {
    /// Shared tuning.
    pub tuning: JointSpecs,
    /// The alignment axis in world space at installation time.
    pub axis: Vector3,
    /// The motor hook (Bullet's `m_icontrol`; the default passes the
    /// current speed through).
    pub control: Box<dyn AngularControl>,
}

impl Default for AngularJointSpecs {
    fn default() -> Self {
        Self { tuning: JointSpecs::default(), axis: Vector3::ZERO, control: Box::new(DefaultAngularControl) }
    }
}

impl core::fmt::Debug for AngularJointSpecs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AngularJointSpecs")
            .field("tuning", &self.tuning)
            .field("axis", &self.axis)
            .field("control", &self.control)
            .finish()
    }
}

/// The angular joint's motor/steering hook (Bullet's `AJoint::IControl`).
/// `prepare` may rewrite the joint's local axis (Bullet's demo
/// `SteerControl` does); `speed` maps the projected relative angular
/// velocity to the target speed.
pub trait AngularControl: core::fmt::Debug + Send + Sync {
    /// Hook before each solve step (Bullet's `IControl::Prepare`).
    fn prepare(&mut self, _joint: &mut AJoint) {}
    /// Target speed along the axis (Bullet's `IControl::Speed`).
    fn speed(&self, _joint: &AJoint, current: Real) -> Real {
        current
    }
    /// Clone hook for the joint's owned control (Bullet shares one static
    /// instance; we copy per joint).
    fn clone_box(&self) -> Box<dyn AngularControl>;
}

/// Bullet's default no-op control (`IControl::Default()`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAngularControl;

impl AngularControl for DefaultAngularControl {
    fn clone_box(&self) -> Box<dyn AngularControl> {
        Box::new(*self)
    }
}

/// An angular joint (Bullet's `AJoint`, `btSoftBody.cpp` 3630-3672): aligns
/// an axis of the two sides' frames and (through the control's target
/// speed) can drive rotation about it.
#[derive(Debug)]
pub struct AJoint {
    /// The two sides (A = the cluster the joint belongs to).
    pub bodies: [Side; 2],
    /// Frame-local axes (Bullet's `m_refs`).
    pub refs: [Vector3; 2],
    /// Constraint-force mixing.
    pub cfm: Real,
    /// Error reduction.
    pub erp: Real,
    /// Drift split factor.
    pub split: Real,
    /// World axes (refreshed by `prepare`).
    pub axis: [Vector3; 2],
    /// The motor hook.
    pub control: Box<dyn AngularControl>,
    /// Per-iteration velocity correction (axis of misalignment).
    pub drift: Vector3,
    /// Position (drift) correction.
    pub sdrift: Vector3,
    /// Angular impulse matrix (refreshed by `prepare`).
    pub massmatrix: Matrix3,
}

impl Clone for AJoint {
    fn clone(&self) -> Self {
        Self {
            control: self.control.clone_box(),
            bodies: self.bodies,
            refs: self.refs,
            cfm: self.cfm,
            erp: self.erp,
            split: self.split,
            axis: self.axis,
            drift: self.drift,
            sdrift: self.sdrift,
            massmatrix: self.massmatrix,
        }
    }
}

impl AJoint {
    /// Bullet's `AJoint::Prepare`: run the control hook, refresh the world
    /// axes and the misalignment drift (capped at pi/16), and split the
    /// correction like the linear joint.
    pub fn prepare(&mut self, dt: Real, iterations: u32, softs: &[SoftBody], bodies: &mut [RigidBody]) {
        const MAXDRIFT: Real = bullet_math::FRAC_PI_2 / 8.0;
        // The control may rewrite `refs` — swap it out for the hook to keep
        // the borrows disjoint (Bullet hands the joint pointer over).
        let mut control = core::mem::replace(&mut self.control, Box::new(DefaultAngularControl));
        control.prepare(self);
        self.control = control;
        side::activate(self.bodies[0], bodies);
        side::activate(self.bodies[1], bodies);
        let (_, basis0) = side::xform(self.bodies[0], softs, bodies);
        let (_, basis1) = side::xform(self.bodies[1], softs, bodies);
        self.axis[0] = basis0 * self.refs[0];
        self.axis[1] = basis1 * self.refs[1];
        self.drift = normalize_any(self.axis[1].cross(self.axis[0]));
        self.drift *= MAXDRIFT.min(math::acos(self.axis[0].dot(self.axis[1]).clamp(-1.0, 1.0)));
        self.drift *= self.erp / dt;
        self.massmatrix = angular_impulse_matrix(
            &side::inv_world_inertia(self.bodies[0], softs, bodies),
            &side::inv_world_inertia(self.bodies[1], softs, bodies),
        );
        if self.split > 0.0 {
            self.sdrift = self.massmatrix * (self.drift * self.split);
            self.drift *= 1.0 - self.split;
        }
        self.drift /= iterations as Real;
    }

    /// Read half of Bullet's `AJoint::Solve`: the pure-angular impulse
    /// countering the axis misalignment drift and the relative angular
    /// velocity off the motor's target speed.
    pub(crate) fn compute(
        &self,
        sor: Real,
        softs: &[SoftBody],
        bodies: &[RigidBody],
    ) -> (JointHalf, JointHalf) {
        let va = side::angular_velocity(self.bodies[0], softs, bodies);
        let vb = side::angular_velocity(self.bodies[1], softs, bodies);
        let vr = va - vb;
        let sp = vr.dot(self.axis[0]);
        let vc = vr - self.axis[0] * self.control.speed(self, sp);
        let impulse = Impulse {
            as_velocity: true,
            velocity: self.massmatrix * (self.drift + vc * self.cfm) * sor,
            ..Impulse::default()
        };
        // Angular impulses apply about the frames' origins — no offset.
        ((self.bodies[0], -impulse, Vector3::ZERO), (self.bodies[1], impulse, Vector3::ZERO))
    }
}

/// Bullet's `NormalizeAny`: normalized direction or zero for degenerate
/// input (`l > SIMD_EPSILON` gate, precision-dependent).
fn normalize_any(v: Vector3) -> Vector3 {
    #[cfg(feature = "f64")]
    const EPS: Real = 2.2e-16;
    #[cfg(not(feature = "f64"))]
    const EPS: Real = 1.19e-7;
    let l = v.length();
    if l > EPS {
        v / l
    } else {
        Vector3::ZERO
    }
}

/// The joint list entry (Bullet's `Joint*` virtual dispatch): contact joints
/// come from the cluster colliders, linear/angular joints from the user API.
#[derive(Clone, Debug)]
pub enum SoftJoint {
    /// A position pin between two frames (Bullet's `LJoint`).
    Linear(LJoint),
    /// An axis alignment / motor joint (Bullet's `AJoint`).
    Angular(AJoint),
    /// A one-step contact joint (Bullet's `CJoint`).
    Contact(CJoint),
}

/// The paired impulses one solve pass computes (the write half happens with
/// immutable joint state so the sides can borrow any soft body).
pub(crate) enum JointImpulses {
    /// Impulses applied at relative offsets (linear/contact joints).
    AtOffset([JointHalf; 2]),
    /// Pure-angular impulses (the angular joint).
    Angular([(Side, Impulse); 2]),
    /// Nothing to apply (self-collision joints below the impulse cap).
    None,
}

impl SoftJoint {
    /// Bullet's `Joint::Prepare` dispatch.
    pub fn prepare(&mut self, dt: Real, iterations: u32, softs: &[SoftBody], bodies: &mut [RigidBody]) {
        match self {
            SoftJoint::Linear(j) => j.prepare(dt, iterations, softs, bodies),
            SoftJoint::Angular(j) => j.prepare(dt, iterations, softs, bodies),
            // CJoint::prepare only wakes rigid bodies and ages the joint.
            SoftJoint::Contact(j) => j.prepare(dt, iterations, bodies),
        }
    }

    /// The read half of the solve pass.
    pub(crate) fn compute(&self, sor: Real, softs: &[SoftBody], bodies: &[RigidBody]) -> JointImpulses {
        match self {
            SoftJoint::Linear(j) => {
                let (a, b) = j.compute(sor, softs, bodies);
                JointImpulses::AtOffset([a, b])
            }
            // Angular impulses apply about the frame origins; the zero
            // offsets mark the angular application path.
            SoftJoint::Angular(j) => {
                let ((s0, i0, _), (s1, i1, _)) = j.compute(sor, softs, bodies);
                JointImpulses::Angular([(s0, i0), (s1, i1)])
            }
            SoftJoint::Contact(j) => match j.compute(sor, softs, bodies) {
                Some((a, b)) => JointImpulses::AtOffset([a, b]),
                None => JointImpulses::None,
            },
        }
    }

    /// The write half of the solve pass.
    pub(crate) fn apply(impulses: JointImpulses, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match impulses {
            JointImpulses::AtOffset([(s0, i0, r0), (s1, i1, r1)]) => {
                side_apply::impulse(s0, i0, r0, softs, bodies);
                side_apply::impulse(s1, i1, r1, softs, bodies);
            }
            JointImpulses::Angular([(s0, i0), (s1, i1)]) => {
                side_apply::va_impulse(s0, i0.velocity, softs, bodies);
                side_apply::va_impulse(s1, i1.velocity, softs, bodies);
            }
            JointImpulses::None => {}
        }
    }

    /// Bullet's `Terminate` dispatch: apply the drift split (the contact
    /// variant also flags its deletion — handled by the caller's filter).
    pub(crate) fn terminate(&self, softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
        match self {
            SoftJoint::Linear(j) => {
                if j.split > 0.0 {
                    side_apply::d_impulse(j.bodies[0], -j.sdrift, j.rpos[0], softs, bodies);
                    side_apply::d_impulse(j.bodies[1], j.sdrift, j.rpos[1], softs, bodies);
                }
            }
            SoftJoint::Angular(j) => {
                if j.split > 0.0 {
                    side_apply::da_impulse(j.bodies[0], -j.sdrift, softs, bodies);
                    side_apply::da_impulse(j.bodies[1], j.sdrift, softs, bodies);
                }
            }
            SoftJoint::Contact(j) => {
                if j.split > 0.0 {
                    side_apply::d_impulse(j.bodies[0], -j.sdrift, j.rpos[0], softs, bodies);
                    side_apply::d_impulse(j.bodies[1], j.sdrift, j.rpos[1], softs, bodies);
                }
            }
        }
    }

    /// Whether the owning list should drop this joint after the solve
    /// (only contact joints terminate).
    pub(crate) fn finished(&self) -> bool {
        matches!(self, SoftJoint::Contact(j) if j.delete)
    }
}

/// Multiply an [`Impulse`] by a scalar (Bullet's `operator*`).
impl core::ops::Mul<Real> for Impulse {
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
