//! Soft-body clusters — port of `btSoftBody`'s cluster machinery
//! (`btSoftBody.cpp` 1280-1350 / 1521-1711 / 3171-3460): K-mean generation,
//! point-mass inertia, the per-step frame/inertia/velocity refresh, the
//! impulse accumulators the contact joints push through, and the global
//! prepare/solve/cleanup pass the world runs before the PBD solve.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bullet_math::{math, Aabb, Matrix3, Real, Vector3};

use crate::body::RigidBody;

use super::body::{Cluster, SoftBody};
use super::joints::{CJoint, SoftJoint};

/// `SIMD_EPSILON` gates in the cluster kernels are precision-dependent like
/// every Bullet epsilon.
#[cfg(feature = "f64")]
const EPS: Real = 2.2e-16;
#[cfg(not(feature = "f64"))]
const EPS: Real = 1.19e-7;

/// `BT_LARGE_FLOAT` — the mass share Bullet gives anchored nodes inside a
/// cluster so they dominate the COM/inertia.
const LARGE_FLOAT: Real = 1e18;

/// Maximum column-sum norm of `m`.
fn p1_norm(m: &Matrix3) -> Real {
    let c0 = m.x_axis.x.abs() + m.y_axis.x.abs() + m.z_axis.x.abs();
    let c1 = m.x_axis.y.abs() + m.y_axis.y.abs() + m.z_axis.y.abs();
    let c2 = m.x_axis.z.abs() + m.y_axis.z.abs() + m.z_axis.z.abs();
    c0.max(c1).max(c2)
}

/// Maximum row-sum norm of `m`.
fn pinf_norm(m: &Matrix3) -> Real {
    let r0 = m.x_axis.x.abs() + m.x_axis.y.abs() + m.x_axis.z.abs();
    let r1 = m.y_axis.x.abs() + m.y_axis.y.abs() + m.y_axis.z.abs();
    let r2 = m.z_axis.x.abs() + m.z_axis.y.abs() + m.z_axis.z.abs();
    r0.max(r1).max(r2)
}

/// Polar decomposition `m = q * s` (Bullet's `btPolarDecomposition`, Higham's
/// 1986 scaled Newton iteration, tolerance 1e-4, 16 iterations). Returns the
/// orthogonal part `q` and the symmetric part `s`; `q` is what the cluster
/// frame and the pose rotation track.
pub(crate) fn polar_decompose(a: &Matrix3) -> (Matrix3, Matrix3) {
    let tolerance: Real = 0.0001;
    let mut u = *a;
    let mut h = a.inverse();
    for _ in 0..16 {
        let h_norm = p1_norm(&h) * pinf_norm(&h);
        let u_1 = p1_norm(&u);
        let u_norm = u_1 * pinf_norm(&u);
        if h_norm.abs() < EPS || u_norm.abs() < EPS {
            break;
        }
        let gamma = math::powf(h_norm / u_norm, 0.25);
        let inv_gamma = 1.0 / gamma;
        let delta = (u * (gamma - 2.0) + h.transpose() * inv_gamma) * 0.5;
        u += delta;
        h = u.inverse();
        if p1_norm(&delta) <= tolerance * u_1 {
            break;
        }
    }
    let mut s = u.transpose() * *a;
    s = (s + s.transpose()) * 0.5;
    (u, s)
}

/// Velocity impulse at a relative offset (Bullet's `clusterVImpulse`): moves
/// `lv`/`av` immediately and records the accumulation for `apply_clusters`.
pub(crate) fn cluster_v_impulse(cluster: &mut Cluster, rpos: Vector3, impulse: Vector3) {
    let li = cluster.imass * impulse;
    let ai = cluster.invwi * rpos.cross(impulse);
    cluster.vimpulses[0] += li;
    cluster.lv += li;
    cluster.vimpulses[1] += ai;
    cluster.av += ai;
    cluster.nvimpulses += 1;
}

/// Drift impulse at a relative offset (Bullet's `clusterDImpulse`): only the
/// accumulators move — `apply_clusters(true)` spreads them onto the nodes.
pub(crate) fn cluster_d_impulse(cluster: &mut Cluster, rpos: Vector3, impulse: Vector3) {
    let li = cluster.imass * impulse;
    let ai = cluster.invwi * rpos.cross(impulse);
    cluster.dimpulses[0] += li;
    cluster.dimpulses[1] += ai;
    cluster.ndimpulses += 1;
}

/// Angular velocity impulse (Bullet's `clusterVAImpulse`): moves `av`
/// immediately and records the accumulation.
pub(crate) fn cluster_va_impulse(cluster: &mut Cluster, impulse: Vector3) {
    let ai = cluster.invwi * impulse;
    cluster.vimpulses[1] += ai;
    cluster.av += ai;
    cluster.nvimpulses += 1;
}

/// Angular drift impulse (Bullet's `clusterDAImpulse`): only the angular
/// accumulator moves.
pub(crate) fn cluster_da_impulse(cluster: &mut Cluster, impulse: Vector3) {
    let ai = cluster.invwi * impulse;
    cluster.dimpulses[1] += ai;
    cluster.ndimpulses += 1;
}

impl SoftBody {
    /// Generate `k` clusters with a K-mean over the node positions (Bullet's
    /// `generateClusters(k, maxiterations)`): the seeded assignment uses the
    /// deterministic `(i * 29873) % k` split, centers drift under the
    /// over-relaxation `w = 2 - min(1, iterations/16)`, faces pull their
    /// border nodes into every adjacent cluster, and a non-colliding master
    /// cluster covering all nodes is swapped to index 0 when more than one
    /// cluster survives. `k = 0` falls back to one cluster per tetrahedron
    /// (or per face). Returns the resulting cluster count.
    pub fn generate_clusters(&mut self, k: usize, maxiterations: usize) -> usize {
        self.release_clusters();
        let nk = k.min(self.nodes.len());
        self.clusters.clear();
        self.clusters.resize(nk, Cluster::default());
        for c in &mut self.clusters {
            c.collide = true;
        }
        if !self.clusters.is_empty() {
            // Initialize: split the nodes deterministically and center every
            // cluster on the body's centroid.
            let mut centers = vec![Vector3::ZERO; nk];
            let mut cog = Vector3::ZERO;
            for (i, n) in self.nodes.iter().enumerate() {
                cog += n.x;
                self.clusters[(i * 29873) % nk].nodes.push(i);
            }
            cog /= self.nodes.len() as Real;
            centers.fill(cog);
            // Iterate: recompute centers with over-relaxation, reassign.
            let slope: Real = 16.0;
            let mut changed = true;
            let mut iterations = 0usize;
            while changed && iterations < maxiterations {
                let w = 2.0 - (iterations as Real / slope).min(1.0);
                changed = false;
                iterations += 1;
                for (i, center_slot) in centers.iter_mut().enumerate() {
                    let nodes = &self.clusters[i].nodes;
                    if nodes.is_empty() {
                        continue;
                    }
                    let mut center = Vector3::ZERO;
                    for &ni in nodes {
                        center += self.nodes[ni].x;
                    }
                    center /= nodes.len() as Real;
                    center = *center_slot + (center - *center_slot) * w;
                    changed |= (center - *center_slot).length_squared() > EPS;
                    *center_slot = center;
                    self.clusters[i].nodes.clear();
                }
                for i in 0..self.nodes.len() {
                    let nx = self.nodes[i].x;
                    let mut kbest = 0;
                    let mut kdist = cluster_metric(centers[0], nx);
                    for (j, center) in centers.iter().enumerate().skip(1) {
                        let d = cluster_metric(*center, nx);
                        if d < kdist {
                            kbest = j;
                            kdist = d;
                        }
                    }
                    self.clusters[kbest].nodes.push(i);
                }
            }
            // Merge: faces pull their border nodes into adjacent clusters.
            let mut cids = vec![usize::MAX; self.nodes.len()];
            for (ci, c) in self.clusters.iter().enumerate() {
                for &ni in &c.nodes {
                    cids[ni] = ci;
                }
            }
            for f in 0..self.faces.len() {
                let idx = self.faces[f].n;
                for j in 0..3 {
                    let cid = cids[idx[j]];
                    for q in 1..3 {
                        let kid = idx[(j + q) % 3];
                        if cids[kid] != cid && !self.clusters[cid].nodes.contains(&kid) {
                            self.clusters[cid].nodes.push(kid);
                        }
                    }
                }
            }
            // Master: a non-colliding cluster of all nodes, swapped to 0.
            if self.clusters.len() > 1 {
                let master = Cluster {
                    collide: false,
                    nodes: (0..self.nodes.len()).collect(),
                    ..Cluster::default()
                };
                self.clusters.push(master);
                let last = self.clusters.len() - 1;
                self.clusters.swap(0, last);
            }
            // Terminate: drop empty clusters (their cdbvt leaves included).
            let mut i = 0;
            while i < self.clusters.len() {
                if self.clusters[i].nodes.is_empty() {
                    self.release_cluster(i);
                } else {
                    i += 1;
                }
            }
        } else {
            // One cluster per tetrahedron (or per face).
            if !self.tetras.is_empty() {
                self.clusters
                    .resize_with(self.tetras.len(), Cluster::default);
                for (i, t) in self.tetras.iter().enumerate() {
                    self.clusters[i].collide = true;
                    self.clusters[i].nodes = t.n.to_vec();
                }
            } else {
                self.clusters
                    .resize_with(self.faces.len(), Cluster::default);
                for (i, f) in self.faces.iter().enumerate() {
                    self.clusters[i].collide = true;
                    self.clusters[i].nodes = f.n.to_vec();
                }
            }
        }

        if !self.clusters.is_empty() {
            self.initialize_clusters();
            self.update_clusters();
            // Connectivity for self-collision filtering.
            let nc = self.clusters.len();
            self.cluster_connectivity = vec![false; nc * nc];
            for c0 in 0..nc {
                for c1 in 0..nc {
                    let connected = self.clusters[c0].nodes.iter().any(|&a| {
                        self.clusters[c1].nodes.contains(&a)
                    });
                    self.cluster_connectivity[c0 + c1 * nc] = connected;
                }
            }
        }
        self.clusters.len()
    }

    /// Remove the cluster at `index` together with its cdbvt leaf (Bullet's
    /// `releaseCluster`).
    fn release_cluster(&mut self, index: usize) {
        let leaf = self.clusters[index].leaf;
        if leaf != bullet_collision::broadphase::DbvtTree::NONE {
            self.cdbvt.remove(leaf);
        }
        self.clusters.remove(index);
    }

    /// Remove every cluster (Bullet's `releaseClusters`).
    pub fn release_clusters(&mut self) {
        while !self.clusters.is_empty() {
            self.release_cluster(0);
        }
    }

    /// Capture the per-cluster masses, inertia and rest frame (Bullet's
    /// `initializeClusters`).
    pub fn initialize_clusters(&mut self) {
        for ci in 0..self.clusters.len() {
            let nodes: Vec<usize> = self.clusters[ci].nodes.clone();
            let mut masses = Vec::with_capacity(nodes.len());
            let mut contains_anchor = false;
            let mut imass = 0.0 as Real;
            for &ni in &nodes {
                let im = self.nodes[ni].im;
                let mass = if im == 0.0 {
                    contains_anchor = true;
                    LARGE_FLOAT
                } else {
                    1.0 / im
                };
                masses.push(mass);
                imass += mass;
            }
            imass = 1.0 / imass;
            // Mass-weighted COM (Bullet's `clusterCom`).
            let mut com = Vector3::ZERO;
            for (j, &ni) in nodes.iter().enumerate() {
                com += self.nodes[ni].x * masses[j];
            }
            com *= imass;
            // Point-mass inertia around the COM, inverted immediately.
            let mut ii = Matrix3::ZERO;
            for (j, &ni) in nodes.iter().enumerate() {
                let k = self.nodes[ni].x - com;
                let q = k * k;
                let m = masses[j];
                ii.x_axis.x += m * (q.y + q.z);
                ii.y_axis.y += m * (q.x + q.z);
                ii.z_axis.z += m * (q.x + q.y);
                ii.x_axis.y -= m * k.x * k.y;
                ii.x_axis.z -= m * k.x * k.z;
                ii.y_axis.z -= m * k.y * k.z;
            }
            ii.y_axis.x = ii.x_axis.y;
            ii.z_axis.x = ii.x_axis.z;
            ii.z_axis.y = ii.y_axis.z;
            // Rest frame at the COM.
            let framerefs: Vec<Vector3> =
                nodes.iter().map(|&ni| self.nodes[ni].x - com).collect();
            let c = &mut self.clusters[ci];
            c.masses = masses;
            c.imass = imass;
            c.contains_anchor = contains_anchor;
            c.com = com;
            c.lv = Vector3::ZERO;
            c.av = Vector3::ZERO;
            c.leaf = bullet_collision::broadphase::DbvtTree::NONE;
            c.locii = ii.inverse();
            c.framepos = com;
            c.framerot = Matrix3::IDENTITY;
            c.framerefs = framerefs;
        }
    }

    /// Refresh every cluster's frame, inertia, velocities and matching
    /// (Bullet's `updateClusters`, run at the end of `predictMotion`).
    pub fn update_clusters(&mut self) {
        let sdt = self.sst.sdt;
        let radmrg = self.sst.radmrg;
        for ci in 0..self.clusters.len() {
            let c = &mut self.clusters[ci];
            let n = c.nodes.len();
            if n == 0 {
                continue;
            }
            // Frame: the rotation that best maps the rest refs onto the
            // current offsets (regularized covariance + polar decomposition).
            let eps: Real = 0.0001;
            let mut m = Matrix3::ZERO;
            m.x_axis.x = eps;
            m.y_axis.y = eps * 2.0;
            m.z_axis.z = eps * 3.0;
            c.com = {
                // cluster_com needs &self; recompute inline to keep the
                // mutable cluster borrow.
                let mut com = Vector3::ZERO;
                for (j, &ni) in c.nodes.iter().enumerate() {
                    com += self.nodes[ni].x * c.masses[j];
                }
                com * c.imass
            };
            for (j, &ni) in c.nodes.iter().enumerate() {
                let a = self.nodes[ni].x - c.com;
                let b = c.framerefs[j];
                // Outer product `sum(a outer b)` in glam's column-major
                // layout (see `update_pose` for the transposition trap).
                m += Matrix3::from_cols(a * b.x, a * b.y, a * b.z);
            }
            let (r, _s) = polar_decompose(&m);
            c.framepos = c.com;
            c.framerot = r;
            // Inertia: constant local inertia rotated into the world frame.
            c.invwi = c.framerot * c.locii * c.framerot.transpose();
            // Velocities: mass-weighted node averages with the damping
            // factors applied once here.
            c.lv = Vector3::ZERO;
            c.av = Vector3::ZERO;
            for (j, &ni) in c.nodes.iter().enumerate() {
                let v = self.nodes[ni].v * c.masses[j];
                c.lv += v;
                c.av += (self.nodes[ni].x - c.com).cross(v);
            }
            c.lv *= c.imass * (1.0 - c.ldamping);
            c.av = c.invwi * c.av * (1.0 - c.adamping);
            c.vimpulses = [Vector3::ZERO; 2];
            c.dimpulses = [Vector3::ZERO; 2];
            c.nvimpulses = 0;
            c.ndimpulses = 0;
            // Matching: pull the nodes toward the frame's rest pose.
            if c.matching > 0.0 {
                for (j, &ni) in c.nodes.iter().enumerate() {
                    let x = c.framerot * c.framerefs[j] + c.framepos;
                    self.nodes[ni].x = self.nodes[ni].x.lerp(x, c.matching);
                }
            }
            // Dbvt: the cluster's node bounds, kept fat with the linear
            // velocity margin.
            if c.collide {
                let mut mi = self.nodes[c.nodes[0]].x;
                let mut mx = mi;
                for &ni in &c.nodes[1..] {
                    mi = mi.min(self.nodes[ni].x);
                    mx = mx.max(self.nodes[ni].x);
                }
                let bounds = Aabb::new(mi, mx);
                if c.leaf != bullet_collision::broadphase::DbvtTree::NONE {
                    self.cdbvt.update_fat(c.leaf, bounds, c.lv * sdt * 3.0, radmrg);
                } else {
                    c.leaf = self.cdbvt.insert(bounds, ci);
                }
            }
        }
    }

    /// Spread the accumulated cluster impulses onto the nodes (Bullet's
    /// `applyClusters`): each node collects the mass-weighted velocity/drift
    /// of every cluster holding it, averaged by the accumulated mass.
    pub fn apply_clusters(&mut self, drift: bool) {
        if drift {
            for c in &mut self.clusters {
                if c.ndimpulses > 0 {
                    let nd = c.ndimpulses as Real;
                    c.dimpulses[0] /= nd;
                    c.dimpulses[1] /= nd;
                }
            }
        }
        let sdt = self.sst.sdt;
        let mut deltas = vec![Vector3::ZERO; self.nodes.len()];
        let mut weights = vec![0.0 as Real; self.nodes.len()];
        for c in &self.clusters {
            let active = if drift { c.ndimpulses } else { c.nvimpulses };
            if active > 0 {
                let v = (if drift { c.dimpulses[0] } else { c.vimpulses[0] }) * sdt;
                let w = (if drift { c.dimpulses[1] } else { c.vimpulses[1] }) * sdt;
                for (j, &ni) in c.nodes.iter().enumerate() {
                    let q = c.masses[j];
                    deltas[ni] += (v + w.cross(self.nodes[ni].x - c.com)) * q;
                    weights[ni] += q;
                }
            }
        }
        for i in 0..self.nodes.len() {
            if weights[i] > 0.0 {
                self.nodes[i].x += deltas[i] / weights[i];
            }
        }
    }

    /// Damp node velocities toward their cluster's rigid motion (Bullet's
    /// `dampClusters`): nodes moving slower than the cluster frame get the
    /// `ndamping` share of the difference.
    pub fn damp_clusters(&mut self) {
        for c in 0..self.clusters.len() {
            let ndamping = self.clusters[c].ndamping;
            if ndamping > 0.0 {
                let com = self.clusters[c].com;
                let (lv, av) = (self.clusters[c].lv, self.clusters[c].av);
                let nodes = self.clusters[c].nodes.clone();
                for ni in nodes {
                    if self.nodes[ni].im > 0.0 {
                        let vx = lv + av.cross(self.nodes[ni].q - com);
                        let v = self.nodes[ni].v;
                        if vx.length_squared() <= v.length_squared() {
                            self.nodes[ni].v = v + (vx - v) * ndamping;
                        }
                    }
                }
            }
        }
    }
}

/// The global cluster solve (Bullet's static `btSoftBody::solveClusters`
/// over the world's soft bodies, run by the world before the per-body PBD):
/// prepare every joint, iterate `sor = 1` passes over all bodies' joints,
/// then terminate (apply the drift splits) and drop the finished joints.
pub fn solve_clusters(softs: &mut [SoftBody], bodies: &mut [RigidBody]) {
    if softs.is_empty() {
        return;
    }
    let iterations = softs.iter().map(|s| s.cfg.citerations).max().unwrap_or(0);
    for si in 0..softs.len() {
        let sdt = softs[si].sst.sdt;
        for ji in 0..softs[si].joints.len() {
            // The prepare pass reads frame/inertia state from any soft body
            // (the joint's sides may live anywhere), so each joint is swapped
            // out for a placeholder while it refreshes itself.
            let mut joint =
                core::mem::replace(&mut softs[si].joints[ji], SoftJoint::Contact(CJoint::default()));
            joint.prepare(sdt, iterations, softs, bodies);
            softs[si].joints[ji] = joint;
        }
    }
    for _ in 0..iterations {
        let sor = 1.0;
        for i in 0..softs.len() {
            let count = softs[i].joints.len();
            for j in 0..count {
                // Read half: the impulse pair comes from immutable state
                // (the joint's sides may live in any soft body).
                let impulses = softs[i].joints[j].compute(sor, softs, bodies);
                // Write half: applied side by side, each application
                // borrowing one cluster or rigid body at a time.
                SoftJoint::apply(impulses, softs, bodies);
            }
        }
    }
    for i in 0..softs.len() {
        // Terminate (the drift split) and drop the finished joints; the
        // joints are taken out so the per-side applies can borrow `softs`.
        let joints = core::mem::take(&mut softs[i].joints);
        for j in &joints {
            j.terminate(softs, bodies);
        }
        softs[i].joints = joints.into_iter().filter(|j| !j.finished()).collect();
    }
}

/// L1 distance (Bullet's `ClusterMetric`).
fn cluster_metric(x: Vector3, y: Vector3) -> Real {
    let d = x - y;
    d.abs().element_sum()
}
