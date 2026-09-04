//! Soft-body force accumulation — port of `btSoftBody::applyForces`,
//! `getVolume`, `updatePose` and the node/face aero kernels
//! (`btSoftBody.cpp` 708-885 / 1244-1263 / 3049-3085 / 3750-3815).

use bullet_math::{math, Matrix3, Real, Vector3};

use super::body::SoftBody;
use super::config::AeroModel;
use super::geometry::apply_clamped_force;

#[cfg(feature = "f64")]
const EPS: Real = 2.2e-16;
#[cfg(not(feature = "f64"))]
const EPS: Real = 1.19e-7;

impl SoftBody {
    /// Signed enclosed volume by the divergence theorem (Bullet's
    /// `getVolume`: sum of node-relative triple products over the faces,
    /// divided by 6, measured from the first node).
    pub fn get_volume(&self) -> Real {
        let mut vol = 0.0;
        if !self.nodes.is_empty() {
            let org = self.nodes[0].x;
            for f in &self.faces {
                let n = f.n;
                vol += (self.nodes[n[0]].x - org)
                    .dot((self.nodes[n[1]].x - org).cross(self.nodes[n[2]].x - org));
            }
            vol /= 6.0;
        }
        vol
    }

    /// Advance the rest pose to the current configuration (Bullet's
    /// `updatePose`): the frame pose recomputes the COM and the covariance
    /// `Apq`, polar-decomposes it into the rotation `rot` and derives the
    /// scale `scl = aqq * rot^T * Apq` (volume-clamped when `maxvolume >
    /// 1`). The pose match (`k_mt`) and the cluster frames read `rot`.
    pub fn update_pose(&mut self) {
        if self.pose.bframe {
            let com = self.evaluate_com();
            self.pose.com = com;
            let mut apq = Matrix3::ZERO;
            let eps = EPS;
            apq.x_axis.x = eps;
            apq.y_axis.y = eps * 2.0;
            apq.z_axis.z = eps * 3.0;
            for i in 0..self.nodes.len() {
                let a = (self.nodes[i].x - com) * self.pose.wgh[i];
                let b = self.pose.pos[i];
                // Covariance `sum(a outer b)` (Bullet accumulates by row,
                // `Apq[r] += a[r] * b`; glam is column-major so the outer
                // product lands as `Mat3::from_cols`-style accumulation —
                // storing rows into columns would hand the polar
                // decomposition the transposed rotation).
                apq += Matrix3::from_cols(a * b.x, a * b.y, a * b.z);
            }
            let (r, _s) = super::cluster::polar_decompose(&apq);
            self.pose.rot = r;
            self.pose.scl = self.pose.aqq * r.transpose() * apq;
            if self.cfg.maxvolume > 1.0 {
                let det = self.pose.scl.determinant();
                if det != 0.0 {
                    let idet = (1.0 / det).clamp(1.0, self.cfg.maxvolume);
                    self.pose.scl *= idet;
                }
            }
        }
    }

    /// Accumulate the aero/pressure/volume forces (Bullet's `applyForces`;
    /// all branches are gated by their coefficients, zero by default).
    pub fn apply_forces(&mut self) {
        let k_lf = self.cfg.k_lf;
        let k_dg = self.cfg.k_dg;
        let k_pr = self.cfg.k_pr;
        let k_vc = self.cfg.k_vc;
        let as_pressure = k_pr != 0.0;
        let as_volume = k_vc > 0.0;
        let use_medium = k_lf > 0.0 || k_dg > 0.0;
        let use_volume = as_pressure || as_volume;

        let mut ivolumetp = 0.0;
        let mut dvolumetv = 0.0;
        if use_volume {
            let volume = self.get_volume();
            ivolumetp = 1.0 / volume.abs() * k_pr;
            dvolumetv = (self.pose.volume - volume) * k_vc;
        }

        // Per-vertex forces.
        for i in 0..self.nodes.len() {
            if self.nodes[i].im > 0.0 {
                if use_medium {
                    self.add_aero_force_to_node(i);
                }
                if as_pressure {
                    let n = self.nodes[i].n;
                    let area = self.nodes[i].area;
                    self.nodes[i].f += n * (area * ivolumetp);
                }
                if as_volume {
                    let n = self.nodes[i].n;
                    let area = self.nodes[i].area;
                    self.nodes[i].f += n * (area * dvolumetv);
                }
            }
        }

        // Per-face aero forces.
        if use_medium && self.cfg.aeromodel >= AeroModel::FTwoSided {
            for i in 0..self.faces.len() {
                self.add_aero_force_to_face(i);
            }
        }
    }

    /// Vertex aero force (Bullet's `addAeroForceToNode`; medium = wind at
    /// air density — the water medium is not ported).
    fn add_aero_force_to_node(&mut self, node_index: usize) {
        let sdt = self.sst.sdt;
        let k_lf = self.cfg.k_lf;
        let k_dg = self.cfg.k_dg;
        let aeromodel = self.cfg.aeromodel;
        let n = &self.nodes[node_index];
        if n.im <= 0.0 {
            return;
        }
        let density = self.air_density;
        let rel_v = n.v - self.wind_velocity;
        let rel_v_len = rel_v.length();
        let rel_v2 = rel_v.length_squared();
        if rel_v2 <= EPS {
            return;
        }
        let rel_v_nrm = rel_v / rel_v_len;
        let mut nrm = n.n;

        if aeromodel == AeroModel::VTwoSidedLiftDrag {
            if nrm.dot(rel_v) < 0.0 {
                nrm = -nrm;
            }
            let n_dot_v = nrm.dot(rel_v_nrm);
            let tri_area = 0.5 * n.area;

            let mut f_drag =
                rel_v_nrm * (-0.5 * k_dg * density * rel_v2 * tri_area * n_dot_v);
            // Angle of attack under ~10 degrees produces lift.
            let mut f_lift = Vector3::ZERO;
            if 0.0 < n_dot_v && n_dot_v < 0.98480 {
                f_lift = nrm.cross(rel_v_nrm).cross(rel_v_nrm)
                    * (0.5 * k_lf * density * rel_v_len * tri_area * math::sqrt(1.0 - n_dot_v * n_dot_v));
            }

            // Keep the drag from reversing the node within dt.
            let im = self.nodes[node_index].im;
            let v = self.nodes[node_index].v;
            let del_v_by_fdrag = f_drag * im * sdt;
            let del2 = del_v_by_fdrag.length_squared();
            let v2 = v.length_squared();
            if del2 >= v2 && del2 > 0.0 {
                f_drag *= 0.8 * (math::sqrt(v2) / math::sqrt(del2));
            }
            self.nodes[node_index].f += f_drag;
            self.nodes[node_index].f += f_lift;
        } else if matches!(
            aeromodel,
            AeroModel::VPoint | AeroModel::VOneSided | AeroModel::VTwoSided
        ) {
            if aeromodel == AeroModel::VTwoSided && nrm.dot(rel_v) < 0.0 {
                nrm = -nrm;
            }
            let dvn = rel_v.dot(nrm);
            if dvn > 0.0 {
                let c0 = n.area * dvn * rel_v2 / 2.0;
                let c1 = c0 * density;
                let force = nrm * (-c1 * k_lf) + rel_v_nrm * (-c1 * k_dg);
                let im = self.nodes[node_index].im;
                let v = self.nodes[node_index].v;
                apply_clamped_force(force, im, v, sdt, &mut self.nodes[node_index].f);
            }
        }
    }

    /// Face aero force (Bullet's `addAeroForceToFace`; the load is split
    /// evenly over the three nodes).
    fn add_aero_force_to_face(&mut self, face_index: usize) {
        let sdt = self.sst.sdt;
        let k_lf = self.cfg.k_lf;
        let k_dg = self.cfg.k_dg;
        let aeromodel = self.cfg.aeromodel;
        let density = self.air_density;

        let fnodes = self.faces[face_index].n;
        let ra = self.faces[face_index].ra;
        let f_normal = self.faces[face_index].normal;
        let v = (self.nodes[fnodes[0]].v + self.nodes[fnodes[1]].v + self.nodes[fnodes[2]].v) / 3.0;
        let rel_v = v - self.wind_velocity;
        let rel_v_len = rel_v.length();
        let rel_v2 = rel_v.length_squared();
        if rel_v2 <= EPS {
            return;
        }
        let rel_v_nrm = rel_v / rel_v_len;
        let mut nrm = f_normal;

        if aeromodel == AeroModel::FTwoSidedLiftDrag {
            if nrm.dot(rel_v) < 0.0 {
                nrm = -nrm;
            }
            let n_dot_v = nrm.dot(rel_v_nrm);
            let tri_area = 0.5 * ra;

            let mut f_drag =
                rel_v_nrm * (-0.5 * k_dg * density * rel_v2 * tri_area * n_dot_v);
            let mut f_lift = Vector3::ZERO;
            if 0.0 < n_dot_v && n_dot_v < 0.98480 {
                f_lift = nrm.cross(rel_v_nrm).cross(rel_v_nrm)
                    * (0.5 * k_lf * density * rel_v_len * tri_area * math::sqrt(1.0 - n_dot_v * n_dot_v));
            }
            f_drag /= 3.0;
            f_lift /= 3.0;

            for &ni in &fnodes {
                if self.nodes[ni].im > 0.0 {
                    let im = self.nodes[ni].im;
                    let v = self.nodes[ni].v;
                    let del_v_by_fdrag = f_drag * im * sdt;
                    let del2 = del_v_by_fdrag.length_squared();
                    let v2 = v.length_squared();
                    let mut fd = f_drag;
                    if del2 >= v2 && del2 > 0.0 {
                        fd *= 0.8 * (math::sqrt(v2) / math::sqrt(del2));
                    }
                    self.nodes[ni].f += fd;
                    self.nodes[ni].f += f_lift;
                }
            }
        } else if matches!(aeromodel, AeroModel::FOneSided | AeroModel::FTwoSided) {
            if aeromodel == AeroModel::FTwoSided && nrm.dot(rel_v) < 0.0 {
                nrm = -nrm;
            }
            let dvn = rel_v.dot(nrm);
            if dvn > 0.0 {
                let c0 = ra * dvn * rel_v2;
                let c1 = c0 * density;
                let force = (nrm * (-c1 * k_lf) + rel_v_nrm * (-c1 * k_dg)) / 3.0;
                for &ni in &fnodes {
                    let im = self.nodes[ni].im;
                    let v = self.nodes[ni].v;
                    apply_clamped_force(force, im, v, sdt, &mut self.nodes[ni].f);
                }
            }
        }
    }
}
