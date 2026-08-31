//! Soft-body simulation kernels — port of `btSoftBody`'s `predictMotion` /
//! `solveConstraints` / `staticSolve` and the position/velocity solver
//! functions (phase A: links only; contacts and anchors arrive with the
//! collision phase — see `docs/softbody.md` §10).

use bullet_math::{Aabb, Real, Vector3};

use super::body::SoftBody;
use super::collide::impulse_matrix;
use super::config::PSolverKind;
use crate::body::RigidBody;

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
        // Update (runtime constants).
        if self.update_rt_cst {
            self.update_rt_cst = false;
            self.update_constants();
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
        self.ndbvt.optimize_incremental(1);

        // Clear contacts (Bullet resizes m_rcontacts/m_scontacts to zero at
        // the end of predictMotion).
        self.rcontacts.clear();
    }

    /// Solve the constraint set (Bullet's `solveConstraints`): assemble the
    /// per-solve link/anchor constants, run the velocity sequence, the
    /// position sequence, and the drift sequence, then write the velocities
    /// back from the corrected positions. `bodies` is the owning world's
    /// rigid-body slice (anchor preparation and contact impulses reach into
    /// it, exactly like Bullet's `btRigidBody::applyImpulse` calls).
    pub fn solve_constraints(&mut self, bodies: &mut [RigidBody]) {
        // Prepare links: gradient c3 = q1 - q0 and c2 = 1/(|c3|^2 * c0).
        for l in &mut self.links {
            l.c3 = self.nodes[l.n[1]].q - self.nodes[l.n[0]].q;
            l.c2 = 1.0 / (l.c3.length_squared() * l.c0);
        }
        // Prepare anchors (Bullet: impulse matrix per anchor + activate the
        // body so a sleeping anchor target takes part).
        for a in &mut self.anchors {
            let body = &mut bodies[a.body];
            let ra = body.orientation * a.local;
            a.c0 = impulse_matrix(
                self.sst.sdt,
                self.nodes[a.node].im,
                body.inv_mass(),
                &body.inv_inertia_world(),
                ra,
            );
            a.c1 = ra;
            a.c2 = self.sst.sdt * self.nodes[a.node].im;
            body.activate();
        }

        let sdt = self.sst.sdt;

        // Solve velocities.
        if self.cfg.viterations > 0 {
            for _ in 0..self.cfg.viterations {
                for k in 0..self.cfg.vsequence.len() {
                    let kind = self.cfg.vsequence[k];
                    self.run_vsolver(kind, 1.0);
                }
            }
            // Update: re-predict positions from the solved velocities.
            for n in &mut self.nodes {
                n.x = n.q + n.v * sdt;
            }
        }

        // Solve positions.
        if self.cfg.piterations > 0 {
            let piterations = self.cfg.piterations as Real;
            for isolve in 0..self.cfg.piterations {
                let ti = isolve as Real / piterations;
                for k in 0..self.cfg.psequence.len() {
                    let kind = self.cfg.psequence[k];
                    self.run_psolver(kind, 1.0, ti, bodies);
                }
            }
            let vc = self.sst.isdt * (1.0 - self.cfg.k_dp);
            for n in &mut self.nodes {
                n.v = (n.x - n.q) * vc;
                n.f = Vector3::ZERO;
            }
        }

        // Solve drift.
        if self.cfg.diterations > 0 {
            let vcf = self.cfg.k_vcf * self.sst.isdt;
            for n in &mut self.nodes {
                n.q = n.x;
            }
            for _ in 0..self.cfg.diterations {
                for k in 0..self.cfg.dsequence.len() {
                    let kind = self.cfg.dsequence[k];
                    self.run_psolver(kind, 1.0, 0.0, bodies);
                }
            }
            for n in &mut self.nodes {
                n.v += (n.x - n.q) * vcf;
            }
        }
    }

    /// Run the position kernel sequence for `iterations` rounds without the
    /// velocity writeback (Bullet's `staticSolve`, used by refine).
    pub fn static_solve(&mut self, iterations: u32, bodies: &mut [RigidBody]) {
        for _ in 0..iterations {
            for k in 0..self.cfg.psequence.len() {
                let kind = self.cfg.psequence[k];
                self.run_psolver(kind, 1.0, 0.0, bodies);
            }
        }
    }

    /// Refresh every runtime constant (Bullet's `updateConstants` =
    /// `resetLinkRestLengths` + `updateLinkConstants` + `updateArea`; the
    /// face-area pass arrives with the cloth phase).
    pub fn update_constants(&mut self) {
        self.reset_link_rest_lengths();
        self.update_link_constants();
    }

    /// Accumulate the aero/pressure/volume forces (Bullet's `applyForces`).
    /// Phase A stub: the default coefficients are all zero, so this is a
    /// no-op; the models arrive with the cloth phase (phase C).
    fn apply_forces(&mut self) {}

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

    fn run_psolver(&mut self, kind: PSolverKind, kst: Real, ti: Real, bodies: &mut [RigidBody]) {
        match kind {
            PSolverKind::Linear => self.solve_links(kst),
            PSolverKind::Anchors => self.solve_anchors(kst, bodies),
            PSolverKind::RContacts => self.solve_rigid_contacts(kst, bodies),
            // Soft contacts arrive with the soft-soft phase.
            PSolverKind::SContacts => {
                let _ = ti;
            }
        }
    }

    fn run_vsolver(&mut self, kind: super::config::VSolverKind, kst: Real) {
        match kind {
            super::config::VSolverKind::Linear => self.solve_velocity_links(kst),
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
    fn solve_rigid_contacts(&mut self, kst: Real, bodies: &mut [RigidBody]) {
        let sdt = self.sst.sdt;
        let mrg = self.margin;
        for c in 0..self.rcontacts.len() {
            let (node, body_idx, c0, c1, c2, c3, c4, normal, offset) = {
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
            let body = &mut bodies[body_idx];
            let va = body.velocity_at_local_point(c1) * sdt;
            let (x, q) = {
                let n = &self.nodes[node];
                (n.x, n.q)
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
            body.apply_impulse(impulse, c1);
        }
    }
}
