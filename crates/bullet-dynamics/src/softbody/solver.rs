//! Soft-body simulation kernels — port of `btSoftBody`'s `predictMotion` /
//! `solveConstraints` / `staticSolve` and the position/velocity solver
//! functions (phase A: links only; contacts and anchors arrive with the
//! collision phase — see `docs/softbody.md` §10).

use bullet_math::{Aabb, Real, Vector3};

use super::body::SoftBody;
use super::collide::{impulse_matrix, bary_eval, ContactOwner};
use super::config::{PSolverKind, SoftCollision};
use super::geometry::project_on_plane;
use crate::body::RigidBody;
use crate::multibody::body::{MultiBody, MultiBodyScratch};

/// Multibody context for the soft position kernels (phase F): link
/// contacts read their jacobians and unit-impulse responses through the
/// scratch and write the generalized velocities back. `None` on the
/// single-body paths (`static_solve`) skips the multibody branches.
pub struct MbSolveContext<'a> {
    /// The owning world's multibody slice.
    pub multibodies: &'a mut [MultiBody],
    /// The world's pooled multibody scratch (jacobian assembly).
    pub scratch: &'a mut MultiBodyScratch,
    /// Recycled jacobian buffer (one contact at a time).
    pub jac: &'a mut Vec<Real>,
    /// Recycled unit-impulse-response buffer (one contact at a time).
    pub delta: &'a mut Vec<Real>,
}

/// The bullet-source epsilon gating the link position solve
/// (`l.m_c1 + len > SIMD_EPSILON`): below it the pair is treated as
/// degenerate and skipped. Precision-dependent, like Bullet's `SIMD_EPSILON`.
#[cfg(feature = "f64")]
const LINK_EPSILON: Real = 2.2e-16;
#[cfg(not(feature = "f64"))]
const LINK_EPSILON: Real = 1.19e-7;

/// `dn <= SIMD_EPSILON` gates the rigid-contact solve (Bullet's
/// `PSolve_RContacts`); precision-dependent like every Bullet epsilon.
#[cfg(feature = "f64")]
const CONTACT_EPSILON: Real = 2.2e-16;
#[cfg(not(feature = "f64"))]
const CONTACT_EPSILON: Real = 1.19e-7;

impl SoftBody {
    /// Integrate the nodes to their predicted positions under gravity and
    /// accumulated forces (Bullet's `predictMotion`, dt scaled by
    /// `cfg.timescale`).
    ///
    /// Order of operations matches Bullet exactly: refresh runtime constants
    /// if dirty, update the solver state, add gravity to the velocities,
    /// accumulate forces, then semi-implicitly integrate every node
    /// (`q = x; v += f*im*sdt; x += v*sdt`) with the per-axis
    /// `max_displacement` clamp.
    pub fn predict_motion(&mut self, dt: Real, gravity: Vector3) {
        // Update (runtime constants + face tree on topology changes —
        // Bullet clears and rebuilds `m_fdbvt` whenever `m_bUpdateRtCst`,
        // but only when VF_SS collisions are enabled).
        if self.update_rt_cst {
            self.update_rt_cst = false;
            self.update_constants();
            self.fdbvt.clear();
            if self.cfg.collisions.contains(SoftCollision::VF_SS) {
                self.initialize_face_tree();
            }
        }

        // Prepare.
        self.sst.sdt = dt * self.cfg.timescale;
        self.sst.isdt = 1.0 / self.sst.sdt;
        self.sst.velmrg = self.sst.sdt * 3.0;
        self.sst.radmrg = self.margin;
        self.sst.updmrg = self.sst.radmrg * 0.25;

        // Forces: gravity (every dynamic node), then aero/pressure/volume
        // (phase C; all default coefficients are zero).
        self.add_velocity(gravity * self.sst.sdt);
        self.apply_forces();

        // Integrate.
        let sdt = self.sst.sdt;
        let clamp_delta_v = self.max_displacement / sdt;
        for n in &mut self.nodes {
            n.q = n.x;
            let mut delta_v = n.f * n.im * sdt;
            delta_v = delta_v.clamp(
                Vector3::splat(-clamp_delta_v),
                Vector3::splat(clamp_delta_v),
            );
            n.v += delta_v;
            n.x += n.v * sdt;
            n.f = Vector3::ZERO;
        }

        // Clusters (Bullet refreshes the frames/velocities right after the
        // integration; the matching pulls the nodes toward the rest frame).
        self.update_clusters();

        // Bounds (Bullet ends predictMotion with the node/face dbvt updates,
        // the pose match and the contact clear — those arrive with their
        // phases; the world-space bounds are shared by all of them).
        self.update_bounds();

        // Nodes tree (Bullet's `m_ndbvt.update(leaf, FromCR(x, radmrg),
        // v * velmrg, updmrg)`): the fat leaf absorbs the step's motion, so
        // resting nodes cost nothing.
        let radmrg = self.sst.radmrg;
        let velmrg = self.sst.velmrg;
        let updmrg = self.sst.updmrg;
        for i in 0..self.nodes.len() {
            let (x, v, leaf) = {
                let n = &self.nodes[i];
                (n.x, n.v, n.leaf)
            };
            let pad = Vector3::splat(radmrg);
            let volume = Aabb::new(x - pad, x + pad);
            self.ndbvt.update_fat(leaf, volume, v * velmrg, updmrg);
        }
        // Faces tree (`m_fdbvt.update(leaf, VolumeOf(f, radmrg), v * velmrg,
        // updmrg)` — the padded triangle AABB with the face-average
        // velocity as the fat margin).
        if !self.fdbvt.is_empty() {
            for i in 0..self.faces.len() {
                let (leaf, v) = {
                    let f = &self.faces[i];
                    let n = f.n;
                    let v = (self.nodes[n[0]].v + self.nodes[n[1]].v + self.nodes[n[2]].v) / 3.0;
                    (f.leaf, v)
                };
                let fnodes = self.faces[i].n;
                let [x0, x1, x2] = [
                    self.nodes[fnodes[0]].x,
                    self.nodes[fnodes[1]].x,
                    self.nodes[fnodes[2]].x,
                ];
                let min = x0.min(x1).min(x2) - Vector3::splat(radmrg);
                let max = x0.max(x1).max(x2) + Vector3::splat(radmrg);
                self.fdbvt.update_fat(leaf, Aabb::new(min, max), v * velmrg, updmrg);
            }
            self.fdbvt.optimize_incremental(1);
        }

        // Pose (frame tracking is a cluster-phase feature; the volume
        // conservation reads the rest volume captured by `set_pose`).
        self.update_pose();
        if self.pose.bframe && self.cfg.k_mt > 0.0 {
            // Match toward the rest frame (identity pose until phase D
            // installs PolarDecompose, which makes this exact-noop except
            // for the rest-frame offset — kept for interface parity).
            let com = self.pose.com;
            for i in 0..self.nodes.len() {
                if self.nodes[i].im > 0.0 {
                    let x = self.pose.rot * self.pose.pos[i] + com;
                    self.nodes[i].x = self.nodes[i].x.lerp(x, self.cfg.k_mt);
                }
            }
        }

        // Clear contacts.
        self.rcontacts.clear();
        self.scontacts.clear();
        self.ndbvt.optimize_incremental(1);
        self.cdbvt.optimize_incremental(1);
    }

    /// Solve the constraint set (Bullet's `solveConstraints`): assemble the
    /// per-solve link/anchor constants, run the velocity sequence, the
    /// position sequence, and the drift sequence, then write the velocities
    /// back from the corrected positions.
    ///
    /// An associated function over the whole soft-body slice: the
    /// soft-contact kernel reaches into the face-side body (`PSolve_SContacts`
    /// moves nodes on both sides), so `idx` + `softs` replace `&mut self`.
    /// `bodies` is the owning world's rigid-body slice (anchor preparation
    /// and contact impulses reach into it, exactly like Bullet's
    /// `btRigidBody::applyImpulse` calls).
    pub fn solve_constraints(
        idx: usize,
        softs: &mut [SoftBody],
        bodies: &mut [RigidBody],
        mbs: Option<&mut MbSolveContext<'_>>,
    ) {
        let (before, rest) = softs.split_at_mut(idx);
        let (this, after) = rest.split_first_mut().unwrap();
        Self::solve_constraints_inner(this, idx, before, after, bodies, mbs);
    }

    fn solve_constraints_inner(
        this: &mut SoftBody,
        idx: usize,
        before: &mut [SoftBody],
        after: &mut [SoftBody],
        bodies: &mut [RigidBody],
        mut mbs: Option<&mut MbSolveContext<'_>>,
    ) {
        // Apply clusters (Bullet opens the solve by spreading the velocity
        // impulses accumulated by the cluster joints).
        this.apply_clusters(false);
        // Prepare links: gradient c3 = q1 - q0 and c2 = 1/(|c3|^2 * c0).
        for l in &mut this.links {
            l.c3 = this.nodes[l.n[1]].q - this.nodes[l.n[0]].q;
            l.c2 = 1.0 / (l.c3.length_squared() * l.c0);
        }
        // Prepare anchors (Bullet: impulse matrix per anchor + activate the
        // body so a sleeping anchor target takes part).
        for a in &mut this.anchors {
            let body = &mut bodies[a.body];
            let ra = body.orientation * a.local;
            a.c0 = impulse_matrix(
                this.sst.sdt,
                this.nodes[a.node].im,
                body.inv_mass(),
                &body.inv_inertia_world(),
                ra,
            );
            a.c1 = ra;
            a.c2 = this.sst.sdt * this.nodes[a.node].im;
            body.activate();
        }

        let sdt = this.sst.sdt;

        // Solve velocities.
        if this.cfg.viterations > 0 {
            for _ in 0..this.cfg.viterations {
                for k in 0..this.cfg.vsequence.len() {
                    let kind = this.cfg.vsequence[k];
                    run_vsolver(this, kind, 1.0);
                }
            }
            // Update: re-predict positions from the solved velocities.
            for n in &mut this.nodes {
                n.x = n.q + n.v * sdt;
            }
        }

        // Solve positions.
        if this.cfg.piterations > 0 {
            let piterations = this.cfg.piterations as Real;
            for isolve in 0..this.cfg.piterations {
                let ti = isolve as Real / piterations;
                for k in 0..this.cfg.psequence.len() {
                    let kind = this.cfg.psequence[k];
                    run_psolver(this, Some((idx, before, after)), kind, 1.0, ti, bodies, mbs.as_deref_mut());
                }
            }
            let vc = this.sst.isdt * (1.0 - this.cfg.k_dp);
            for n in &mut this.nodes {
                n.v = (n.x - n.q) * vc;
                n.f = Vector3::ZERO;
            }
        }

        // Solve drift.
        if this.cfg.diterations > 0 {
            let vcf = this.cfg.k_vcf * this.sst.isdt;
            for n in &mut this.nodes {
                n.q = n.x;
            }
            for _ in 0..this.cfg.diterations {
                for k in 0..this.cfg.dsequence.len() {
                    let kind = this.cfg.dsequence[k];
                    run_psolver(this, Some((idx, before, after)), kind, 1.0, 0.0, bodies, mbs.as_deref_mut());
                }
            }
            for n in &mut this.nodes {
                n.v += (n.x - n.q) * vcf;
            }
        }
        // Apply clusters (Bullet closes the solve with the node damping and
        // the drift impulses).
        this.damp_clusters();
        this.apply_clusters(true);
    }

    /// Run the position kernel sequence for `iterations` rounds without the
    /// velocity writeback (Bullet's `staticSolve`, used by refine). Soft
    /// contacts are skipped — refine is a single-body relaxation.
    pub fn static_solve(&mut self, iterations: u32, bodies: &mut [RigidBody]) {
        for _ in 0..iterations {
            for k in 0..self.cfg.psequence.len() {
                let kind = self.cfg.psequence[k];
                // No multibody context: refine is a single-body relaxation,
                // link contacts (if any linger) skip their writeback.
                run_psolver(self, None, kind, 1.0, 0.0, bodies, None);
            }
        }
    }

    /// Refresh every runtime constant (Bullet's `updateConstants` =
    /// `resetLinkRestLengths` + `updateLinkConstants` + `updateArea`).
    pub fn update_constants(&mut self) {
        self.reset_link_rest_lengths();
        self.update_link_constants();
        self.update_area(false);
    }

    /// Refresh face and node areas (Bullet's `updateArea`; nodes accumulate
    /// one third of their adjacent faces' doubled areas).
    pub fn update_area(&mut self, _average_area: bool) {
        for f in &mut self.faces {
            let n = f.n;
            f.ra = super::geometry::area_of(self.nodes[n[0]].x, self.nodes[n[1]].x, self.nodes[n[2]].x);
        }
        for n in &mut self.nodes {
            n.area = 0.0;
        }
        for f in &self.faces {
            let n = f.n;
            for &ni in &n {
                self.nodes[ni].area += f.ra;
            }
        }
        for n in &mut self.nodes {
            n.area *= 0.3333333;
        }
    }

    /// Recompute the world-space bounds from the node positions (Bullet's
    /// `updateBounds`): node min/max padded by the collision margin.
    pub fn update_bounds(&mut self) {
        if let Some(first) = self.nodes.first() {
            let mut mins = first.x;
            let mut maxs = first.x;
            for n in &self.nodes[1..] {
                mins = mins.min(n.x);
                maxs = maxs.max(n.x);
            }
            let mrg = Vector3::splat(self.margin);
            self.bounds = [mins - mrg, maxs + mrg];
        } else {
            self.bounds = [Vector3::ZERO, Vector3::ZERO];
        }
    }

    /// Position-solve every link toward its rest length (Bullet's
    /// `PSolve_Links`): `k = (c1 - len) / (c0 * (c1 + len)) * kst` with
    /// `len = |b - a|^2`, moving both ends by their inverse masses.
    fn solve_links(&mut self, kst: Real) {
        for l in 0..self.links.len() {
            let link = &self.links[l];
            if link.c0 <= 0.0 {
                continue;
            }
            let [ia, ib] = link.n;
            let del = self.nodes[ib].x - self.nodes[ia].x;
            let len = del.length_squared();
            if link.c1 + len > LINK_EPSILON {
                let k = ((link.c1 - len) / (link.c0 * (link.c1 + len))) * kst;
                let step_a = del * (k * self.nodes[ia].im);
                let step_b = del * (k * self.nodes[ib].im);
                self.nodes[ia].x -= step_a;
                self.nodes[ib].x += step_b;
            }
        }
    }

    /// Velocity-solve every link (Bullet's `VSolve_Links`):
    /// `j = -dot(c3, v0 - v1) * c2 * kst` applied along the link gradient.
    fn solve_velocity_links(&mut self, kst: Real) {
        for l in 0..self.links.len() {
            let link = &self.links[l];
            let [ia, ib] = link.n;
            let j = -link.c3.dot(self.nodes[ia].v - self.nodes[ib].v) * link.c2 * kst;
            let step_a = link.c3 * (j * self.nodes[ia].im);
            let step_b = link.c3 * (j * self.nodes[ib].im);
            self.nodes[ia].v += step_a;
            self.nodes[ib].v -= step_b;
        }
    }

    /// Pull anchored nodes onto their rigid bodies (Bullet's
    /// `PSolve_Anchors`): `vr = (va - vb) + (wa - x) * kAHR`,
    /// `impulse = c0 * vr * influence`, node moved by `impulse * c2` and
    /// the body receiving `-impulse` at the anchor.
    fn solve_anchors(&mut self, kst: Real, bodies: &mut [RigidBody]) {
        let k_ahr = self.cfg.k_ahr * kst;
        let sdt = self.sst.sdt;
        for a in 0..self.anchors.len() {
            let (node, body_idx, local, influence, c0, c1, c2) = {
                let a = &self.anchors[a];
                (a.node, a.body, a.local, a.influence, a.c0, a.c1, a.c2)
            };
            let body = &mut bodies[body_idx];
            let wa = body.position + body.orientation * local;
            let va = body.velocity_at_local_point(c1) * sdt;
            let (x, q) = {
                let n = &self.nodes[node];
                (n.x, n.q)
            };
            let vb = x - q;
            let vr = (va - vb) + (wa - x) * k_ahr;
            let impulse = c0 * vr * influence;
            self.nodes[node].x += impulse * c2;
            body.apply_impulse(-impulse, c1);
        }
    }

    /// Solve the rigid contacts (Bullet's `PSolve_RContacts`): approaching
    /// nodes (`dn <= EPS`) get the impulse-matrix correction
    /// `c0 * ((vr - fv*c3 + n*(dp*c4)) * kst)` with the depth clamp
    /// `dp = min(dot(x, n) + offset, margin)`, the node moves by
    /// `impulse * c2` and the body receives `+impulse` at the contact
    /// anchor (sign-flipped vs the anchor — the source is written that way).
    ///
    /// Against a multibody link (Bullet's `CO_FEATHERSTONE_LINK` branch,
    /// btSoftBody.cpp 3908-3960) the contact-point velocity is the normal
    /// projection of the jacobian contraction `dot(velocities, jac)` and
    /// the reaction returns as `|impulse| * 0.5` scaled along the unit
    /// response (`applyDeltaVeeMultiDof` with Bullet's hardcoded
    /// multiplier) — a scalar feedback, not a per-component impulse.
    fn solve_rigid_contacts(
        &mut self,
        kst: Real,
        bodies: &mut [RigidBody],
        mut mbs: Option<&mut MbSolveContext<'_>>,
    ) {
        let sdt = self.sst.sdt;
        let mrg = self.margin;
        for c in 0..self.rcontacts.len() {
            let (node, owner, c0, c1, c2, c3, c4, normal, offset) = {
                let c = &self.rcontacts[c];
                (
                    c.node,
                    c.cti.body,
                    c.c0,
                    c.c1,
                    c.c2,
                    c.c3,
                    c.c4,
                    c.cti.normal,
                    c.cti.offset,
                )
            };
            let (x, q) = {
                let n = &self.nodes[node];
                (n.x, n.q)
            };
            let va = match owner {
                ContactOwner::Rigid(body_idx) => {
                    bodies[body_idx].velocity_at_local_point(c1) * sdt
                }
                ContactOwner::Link { mb: mi, link } => {
                    let Some(ctx) = mbs.as_deref_mut() else {
                        continue;
                    };
                    let mb = &ctx.multibodies[mi];
                    let ndof = 6 + mb.num_dofs();
                    ctx.jac.clear();
                    ctx.jac.resize(ndof, 0.0);
                    ctx.delta.clear();
                    ctx.delta.resize(ndof, 0.0);
                    mb.fill_constraint_jacobian(
                        link,
                        x,
                        Vector3::ZERO,
                        normal,
                        ctx.jac,
                        ctx.scratch,
                    );
                    let vel = mb.velocity_vector();
                    let mut vel_jac = 0.0;
                    for (v, jv) in vel.iter().zip(ctx.jac.iter()) {
                        vel_jac += v * jv;
                    }
                    mb.calc_acceleration_deltas(ctx.jac, ctx.delta, ctx.scratch);
                    normal * vel_jac * sdt
                }
            };
            let vb = x - q;
            let vr = vb - va;
            let dn = vr.dot(normal);
            if dn > CONTACT_EPSILON {
                continue;
            }
            let dp = (x.dot(normal) + offset).min(mrg);
            let fv = vr - normal * dn;
            let impulse = c0 * ((vr - fv * c3 + normal * (dp * c4)) * kst);
            self.nodes[node].x -= impulse * c2;
            match owner {
                ContactOwner::Rigid(body_idx) => bodies[body_idx].apply_impulse(impulse, c1),
                ContactOwner::Link { mb: mi, link: _ } => {
                    if let Some(ctx) = mbs.as_deref_mut() {
                        ctx.multibodies[mi].apply_delta_vee(ctx.delta, -impulse.length() * 0.5);
                    }
                }
            }
        }
    }
}

/// Dispatch one position kernel (Bullet's `getSolver` + the sequence loops).
/// `soft_ctx` carries the split soft-body slices the soft-contact kernel
/// needs to reach the face-side body; `None` skips soft contacts (the
/// single-body `static_solve` path).
fn run_psolver(
    this: &mut SoftBody,
    soft_ctx: Option<(usize, &mut [SoftBody], &mut [SoftBody])>,
    kind: PSolverKind,
    kst: Real,
    _ti: Real,
    bodies: &mut [RigidBody],
    mbs: Option<&mut MbSolveContext<'_>>,
) {
    match kind {
        PSolverKind::Linear => this.solve_links(kst),
        PSolverKind::Anchors => this.solve_anchors(kst, bodies),
        PSolverKind::RContacts => this.solve_rigid_contacts(kst, bodies, mbs),
        PSolverKind::SContacts => {
            if let Some((idx, before, after)) = soft_ctx {
                solve_soft_contacts(this, idx, before, after);
            }
        }
    }
}

/// Dispatch one velocity kernel.
fn run_vsolver(this: &mut SoftBody, kind: super::config::VSolverKind, kst: Real) {
    match kind {
        super::config::VSolverKind::Linear => this.solve_velocity_links(kst),
    }
}

/// Solve the soft-soft contacts (Bullet's `PSolve_SContacts`): the node and
/// the face's barycentric point are pushed apart along the contact normal
/// (depth `margin - (n.x - n.p).normal`) and friction damps the in-plane
/// relative step, each side weighted by its `cfm` share. The face-side body
/// is resolved from the split slices (its index vs `idx`).
fn solve_soft_contacts(
    this: &mut SoftBody,
    idx: usize,
    before: &mut [SoftBody],
    after: &mut [SoftBody],
) {
    for c in 0..this.scontacts.len() {
        let (node, face_body, face, weights, normal, margin, friction, cfm) = {
            let c = &this.scontacts[c];
            (
                c.node,
                c.face_body,
                c.face,
                c.weights,
                c.normal,
                c.margin,
                c.friction,
                c.cfm,
            )
        };
        let other = if face_body < idx {
            &mut before[face_body]
        } else {
            &mut after[face_body - idx - 1]
        };
        let fnodes = other.faces[face].n;
        let p = bary_eval(other.nodes[fnodes[0]].x, other.nodes[fnodes[1]].x, other.nodes[fnodes[2]].x, weights);
        let q = bary_eval(other.nodes[fnodes[0]].q, other.nodes[fnodes[1]].q, other.nodes[fnodes[2]].q, weights);
        let vr = (this.nodes[node].x - this.nodes[node].q) - (p - q);
        let mut corr = Vector3::ZERO;
        if vr.dot(normal) < 0.0 {
            let j = margin - (normal.dot(this.nodes[node].x) - normal.dot(p));
            corr += normal * j;
        }
        corr -= project_on_plane(vr, normal) * friction;
        this.nodes[node].x += corr * cfm[0];
        other.nodes[fnodes[0]].x -= corr * (cfm[1] * weights.x);
        other.nodes[fnodes[1]].x -= corr * (cfm[1] * weights.y);
        other.nodes[fnodes[2]].x -= corr * (cfm[1] * weights.z);
    }
}
