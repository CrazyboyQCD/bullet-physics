//! Soft-body core data — port of `btSoftBody`'s node/link topology and
//! public body API (construction, append, mass and force helpers).
//!
//! Bullet weaves `Node*` pointers through links/faces/clusters; here every
//! element stores node indices instead (Bullet's own serialization path
//! `pointersToIndices` proves the index encoding is equivalent).

use bullet_collision::broadphase::DbvtTree;
use bullet_math::{Aabb, Matrix3, Quaternion, Real, Vector3};

use crate::handles::{BodyHandle, SoftBodyHandle};

use super::collide::{Anchor, RContact, SContact};
use super::config::{Config, SolverState};
use super::joints::{
    AngularJointSpecs, AJoint, LinearJointSpecs, LJoint, SoftJoint,
};

/// Linear/angular/volume stiffness coefficients of a material (Bullet's
/// `btSoftBody::Material`; the `DebugDraw`-only `m_flags` bit is not
/// ported).
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Linear stiffness coefficient [0,1].
    pub k_lst: Real,
    /// Area/angular stiffness coefficient [0,1].
    pub k_ast: Real,
    /// Volume stiffness coefficient [0,1].
    pub k_vst: Real,
}

impl Default for Material {
    fn default() -> Self {
        // The default material installed by the btSoftBody constructor.
        Self { k_lst: 1.0, k_ast: 1.0, k_vst: 1.0 }
    }
}

/// A mass particle (Bullet's `btSoftBody::Node`).
#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// Position.
    pub x: Vector3,
    /// Previous step position / test position.
    pub q: Vector3,
    /// Velocity.
    pub v: Vector3,
    /// Force accumulator.
    pub f: Vector3,
    /// Normal (area-weighted, refreshed by `update_normals`).
    pub n: Vector3,
    /// Inverse mass (0 = fixed).
    pub im: Real,
    /// Node area (rest-area weighted by adjacent faces).
    pub area: Real,
    /// Whether the node is anchored to a rigid body.
    pub battach: bool,
    /// Node-tree leaf handle (`DbvtTree::NONE` until inserted).
    pub leaf: u32,
}

impl Node {
    fn new(x: Vector3, mass: Real) -> Self {
        Self {
            x,
            q: x,
            v: Vector3::ZERO,
            f: Vector3::ZERO,
            n: Vector3::ZERO,
            im: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            area: 0.0,
            battach: false,
            leaf: DbvtTree::NONE,
        }
    }
}

/// A distance constraint between two nodes (Bullet's `btSoftBody::Link`).
#[derive(Clone, Copy, Debug)]
pub struct Link {
    /// Node indices.
    pub n: [usize; 2],
    /// Material index.
    pub mat: usize,
    /// Rest length.
    pub rl: Real,
    /// Whether this is a bending link (generated, not structural).
    pub bending: bool,
    /// `(im0 + im1) / kLST` — the position-solve mass scaling.
    pub c0: Real,
    /// `rl^2`.
    pub c1: Real,
    /// `1 / (|c3|^2 * c0)`, refreshed per solve (velocity solve).
    pub c2: Real,
    /// Gradient `q1 - q0`, refreshed per solve (velocity solve).
    pub c3: Vector3,
}

/// A triangle of the soft surface (Bullet's `btSoftBody::Face`; the
/// deformable-system `m_pcontact`/`m_n0`/`m_n1` fields are not ported).
#[derive(Clone, Copy, Debug)]
pub struct Face {
    /// Node indices.
    pub n: [usize; 3],
    /// Material index.
    pub mat: usize,
    /// Normal (refreshed by `update_normals`).
    pub normal: Vector3,
    /// Rest area (doubled — Bullet's `AreaOf`, refreshed by `update_area`).
    pub ra: Real,
    /// Face-tree leaf handle (`DbvtTree::NONE` until inserted).
    pub leaf: u32,
}

impl Face {
    fn new(n: [usize; 3], mat: usize, ra: Real) -> Self {
        Self { n, mat, normal: Vector3::ZERO, ra, leaf: DbvtTree::NONE }
    }
}

/// A tetrahedron of a volumetric soft body (Bullet's `btSoftBody::Tetra`).
/// In the classic system tetras drive the volume-weighted mass distribution
/// and the per-tetra cluster fallback; the deformation-gradient fields
/// (`m_Dm_inverse`/`m_F`/`m_P_inv`) belong to the NeoHookean system and are
/// not ported, so neither are the gradient coefficients `c0`/`c1`/`c2`
/// (they feed only that solver).
#[derive(Clone, Copy, Debug)]
pub struct Tetra {
    /// Node indices.
    pub n: [usize; 4],
    /// Material index.
    pub mat: usize,
    /// Rest volume (six times the true volume — Bullet stores the raw triple
    /// product; the mass helpers take its absolute value).
    pub rv: Real,
}

/// A rigid volume-preserving group of nodes (Bullet's `btSoftBody::Cluster`):
/// point-mass inertia, a matching frame and an impulse accumulator that the
/// contact joints (`CL_RS`/`CL_SS`/`CL_SELF`) push impulses through.
#[derive(Clone, Debug)]
pub struct Cluster {
    /// Per-node mass (`BT_LARGE_FLOAT` for anchored nodes).
    pub masses: Vec<Real>,
    /// Node indices.
    pub nodes: Vec<usize>,
    /// Rest offsets from the frame origin.
    pub framerefs: Vec<Vector3>,
    /// Frame origin (the cluster COM).
    pub framepos: Vector3,
    /// Frame rotation (Bullet's `m_framexform` basis).
    pub framerot: Matrix3,
    /// `1 / sum(masses)`.
    pub imass: Real,
    /// Inverse inertia in the rest frame (point-mass approximation).
    pub locii: Matrix3,
    /// Inverse inertia in world frame (`R * locii * R^T`).
    pub invwi: Matrix3,
    /// Cluster center of mass.
    pub com: Vector3,
    /// Accumulated velocity impulses `[linear, angular]`.
    pub vimpulses: [Vector3; 2],
    /// Accumulated drift impulses `[linear, angular]`.
    pub dimpulses: [Vector3; 2],
    /// Number of velocity impulses this step.
    pub nvimpulses: u32,
    /// Number of drift impulses this step.
    pub ndimpulses: u32,
    /// Linear velocity.
    pub lv: Vector3,
    /// Angular velocity.
    pub av: Vector3,
    /// Cluster-tree leaf handle (`DbvtTree::NONE` until inserted).
    pub leaf: u32,
    /// Node damping coefficient (Bullet's `m_ndamping`).
    pub ndamping: Real,
    /// Linear damping coefficient.
    pub ldamping: Real,
    /// Angular damping coefficient.
    pub adamping: Real,
    /// Pose-matching coefficient.
    pub matching: Real,
    /// Whether the cluster contains an anchored node.
    pub contains_anchor: bool,
    /// Whether the cluster takes part in collision (the master cluster does
    /// not).
    pub collide: bool,
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            masses: Vec::new(),
            nodes: Vec::new(),
            framerefs: Vec::new(),
            framepos: Vector3::ZERO,
            framerot: Matrix3::IDENTITY,
            imass: 0.0,
            locii: Matrix3::ZERO,
            invwi: Matrix3::ZERO,
            com: Vector3::ZERO,
            vimpulses: [Vector3::ZERO; 2],
            dimpulses: [Vector3::ZERO; 2],
            nvimpulses: 0,
            ndimpulses: 0,
            lv: Vector3::ZERO,
            av: Vector3::ZERO,
            leaf: DbvtTree::NONE,
            ndamping: 0.0,
            ldamping: 0.0,
            adamping: 0.0,
            matching: 0.0,
            contains_anchor: false,
            collide: true,
        }
    }
}

/// A combined velocity + drift impulse (Bullet's `btSoftBody::Impulse`):
/// `as_velocity`/`as_drift` pick which halves a joint application moves.
#[derive(Clone, Copy, Debug, Default)]
pub struct Impulse {
    /// Velocity-half.
    pub velocity: Vector3,
    /// Drift-half.
    pub drift: Vector3,
    /// Whether `velocity` is active.
    pub as_velocity: bool,
    /// Whether `drift` is active.
    pub as_drift: bool,
}

impl std::ops::Neg for Impulse {
    type Output = Impulse;
    fn neg(self) -> Impulse {
        Impulse {
            velocity: -self.velocity,
            drift: -self.drift,
            as_velocity: self.as_velocity,
            as_drift: self.as_drift,
        }
    }
}

/// One side of a soft-body joint (Bullet's `btSoftBody::Body` tagged union):
/// either a cluster of a soft body or a rigid body, both addressed by
/// handle (Bullet stores pointers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// A cluster: `(soft body handle, cluster index)`.
    Cluster {
        soft: SoftBodyHandle,
        cluster: usize,
    },
    /// A rigid body.
    Rigid(BodyHandle),
}

impl Default for Side {
    fn default() -> Self {
        Side::Rigid(BodyHandle::from_index(0))
    }
}

/// Rest-pose bookkeeping (Bullet's `btSoftBody::Pose`): mass weights,
/// rest offsets from the COM and the rest volume captured by `set_pose`.
/// The rotation/scale fields only advance when a frame pose is enabled
/// (phase D's cluster matching); `update_pose` keeps them at identity until
/// then.
#[derive(Clone, Debug)]
pub struct Pose {
    /// Whether the rest volume is tracked.
    pub bvolume: bool,
    /// Whether a rest frame is tracked.
    pub bframe: bool,
    /// Rest volume (set by `set_pose` when `bvolume`).
    pub volume: Real,
    /// Rest offsets from the COM.
    pub pos: Vec<Vector3>,
    /// Mass weights (fixed nodes get the huge `kmass` share).
    pub wgh: Vec<Real>,
    /// Rest COM (the origin `evaluate_com` reports — the zero vector unless
    /// a frame pose is enabled, exactly like Bullet).
    pub com: Vector3,
    /// Pose rotation (identity until the cluster phase).
    pub rot: Matrix3,
    /// Pose scale (identity until the cluster phase).
    pub scl: Matrix3,
    /// Inverted rest covariance (used by the frame pose).
    pub aqq: Matrix3,
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            bvolume: false,
            bframe: false,
            volume: 0.0,
            pos: Vec::new(),
            wgh: Vec::new(),
            com: Vector3::ZERO,
            rot: Matrix3::IDENTITY,
            scl: Matrix3::IDENTITY,
            aqq: Matrix3::IDENTITY,
        }
    }
}

/// A rope/cloth/volumetric soft body simulated with position-based dynamics
/// (Bullet's `btSoftBody`, classic system). Phase A of the port covers the
/// node/link skeleton; contacts, faces, tetras, clusters and joints follow
/// in later phases — see `docs/softbody.md` §10.
///
/// Unlike Bullet, gravity is supplied by the owning
/// [`crate::world::PhysicsWorld`] at step time instead of a
/// `btSoftBodyWorldInfo`.
pub struct SoftBody {
    /// Tunables (Bullet's `m_cfg`).
    pub cfg: Config,
    /// Per-step solver scratch (Bullet's `m_sst`).
    pub sst: SolverState,
    /// Collision margin of the implicit soft shape (Bullet's internal
    /// `btSoftBodyCollisionShape` margin, default 0.25 — the contact band).
    pub margin: Real,
    /// Maximum node travel per step, all axes (Bullet's
    /// `m_worldInfo->m_maxDisplacement`, anti-explosion clamp; default 1000).
    pub max_displacement: Real,
    /// Nodes.
    pub nodes: Vec<Node>,
    /// Distance constraints.
    pub links: Vec<Link>,
    /// Surface triangles (empty for ropes).
    pub faces: Vec<Face>,
    /// Tetrahedra (empty unless built by a volumetric helper).
    pub tetras: Vec<Tetra>,
    /// Materials.
    pub materials: Vec<Material>,
    /// Node AABB tree (Bullet's `m_ndbvt`): leaves keyed by node index,
    /// volumes padded by the collision margin and the step's velocity.
    pub ndbvt: DbvtTree,
    /// Face AABB tree (Bullet's `m_fdbvt`), built lazily on the first face
    /// insert and kept fat like the node tree.
    pub fdbvt: DbvtTree,
    /// Cluster AABB tree (Bullet's `m_cdbvt`), maintained by
    /// `update_clusters` for the `CL_RS`/`CL_SS` broad-phase pairing.
    pub cdbvt: DbvtTree,
    /// Rigid volume-preserving node groups (Bullet's `m_clusters`).
    pub clusters: Vec<Cluster>,
    /// Cluster adjacency for self-collision filtering (Bullet's
    /// `m_clusterConnectivity`, row-major `c0 + c1 * clusters.len()`).
    pub cluster_connectivity: Vec<bool>,
    /// Joints: contact joints created by the cluster colliders this step
    /// (one solve each) plus the user-installed linear/angular joints
    /// (Bullet's `m_joints`); prepared/solved/terminated by the cluster
    /// solver.
    pub joints: Vec<super::joints::SoftJoint>,
    /// Impulse cap for self-collision contacts (Bullet's
    /// `m_maxSelfCollisionImpulse`, default 100).
    pub max_self_collision_impulse: Real,
    /// Scale applied to self-collision impulses above the cap (Bullet's
    /// `m_selfCollisionImpulseFactor`, default 0.01).
    pub self_collision_impulse_factor: Real,
    /// Rest pose (Bullet's `m_pose`).
    pub pose: Pose,
    /// Wind velocity for the aero forces (Bullet's `m_windVelocity`, which
    /// lives in the shared world info but is per-body in practice).
    pub wind_velocity: Vector3,
    /// Air density for the aero forces (Bullet's
    /// `SoftBodyWorldInfo::air_density`, default 1.2; the water medium is
    /// not ported — see `docs/softbody.md` §1).
    pub air_density: Real,
    /// Contacts against rigid bodies this step (Bullet's `m_rcontacts`,
    /// cleared in `predict_motion`, filled by the collision phase).
    pub rcontacts: Vec<RContact>,
    /// Contacts against other soft bodies' faces this step (Bullet's
    /// `m_scontacts`, cleared in `predict_motion`).
    pub scontacts: Vec<SContact>,
    /// Rigid-body anchors (Bullet's `m_anchors`).
    pub anchors: Vec<Anchor>,
    /// World-space bounds (nodes +/- margin), refreshed in `predict_motion`.
    pub bounds: [Vector3; 2],
    /// Whether runtime constants must be refreshed at the next
    /// `predict_motion` (set by every topology/mass mutation).
    pub update_rt_cst: bool,
    /// Scale of link rest lengths (Bullet's `m_restLengthScale`).
    pub rest_length_scale: Real,
    /// Initial node positions (Bullet's `m_X`): captured once at
    /// construction and never refreshed — `transform_to` fits the rigid
    /// motion from initial to current pose through them (Bullet's quirk:
    /// nodes appended later have no rest sample).
    pub rest_positions: Vec<Vector3>,
    /// Collision filter group (Bullet's `m_collisionFilterGroup` on the
    /// soft body's proxy; default
    /// [`filter_groups::DEFAULT`](crate::body::filter_groups::DEFAULT)).
    pub collision_group: u16,
    /// Collision filter mask (default
    /// [`filter_groups::ALL`](crate::body::filter_groups::ALL)).
    pub collision_mask: u16,
}

impl Default for SoftBody {
    fn default() -> Self {
        Self::new(&[], &[])
    }
}

impl SoftBody {
    /// Create a soft body from node positions and masses (Bullet's
    /// `btSoftBody(worldInfo, node_count, x, m)` constructor; pass `mass = 0`
    /// for fixed nodes).
    pub fn new(positions: &[Vector3], masses: &[Real]) -> Self {
        assert_eq!(positions.len(), masses.len(), "position/mass count mismatch");
        let mut body = Self {
            cfg: Config::default(),
            sst: SolverState::default(),
            margin: 0.25,
            max_displacement: 1000.0,
            nodes: Vec::with_capacity(positions.len()),
            links: Vec::new(),
            faces: Vec::new(),
            tetras: Vec::new(),
            materials: vec![Material::default()],
            ndbvt: DbvtTree::new(),
            fdbvt: DbvtTree::new(),
            cdbvt: DbvtTree::new(),
            clusters: Vec::new(),
            cluster_connectivity: Vec::new(),
            joints: Vec::new(),
            max_self_collision_impulse: 100.0,
            self_collision_impulse_factor: 0.01,
            pose: Pose::default(),
            wind_velocity: Vector3::ZERO,
            air_density: 1.2,
            rcontacts: Vec::new(),
            scontacts: Vec::new(),
            anchors: Vec::new(),
            bounds: [Vector3::ZERO, Vector3::ZERO],
            update_rt_cst: true,
            rest_length_scale: 1.0,
            rest_positions: positions.to_vec(),
            collision_group: crate::body::filter_groups::DEFAULT,
            collision_mask: crate::body::filter_groups::ALL,
        };
        for (i, (&x, &m)) in positions.iter().zip(masses).enumerate() {
            body.nodes.push(Node::new(x, m));
            // Bullet's appendNode: FromCR(x, margin) — a box around the
            // point padded by the collision margin.
            let pad = Vector3::splat(body.margin);
            body.nodes[i].leaf = body.ndbvt.insert(Aabb::new(x - pad, x + pad), i);
        }
        body
    }

    /// Append a node with an ndbvt leaf (Bullet's `appendNode`). Rest
    /// positions (`m_X`) are not refreshed — Bullet's quirk. Returns the
    /// new node index.
    pub fn append_node(&mut self, x: Vector3, mass: Real) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node::new(x, mass));
        let pad = Vector3::splat(self.margin);
        self.nodes[index].leaf = self.ndbvt.insert(Aabb::new(x - pad, x + pad), index);
        index
    }

    /// Append a material and return its index (Bullet's `appendMaterial`).
    pub fn append_material(&mut self, material: Material) -> usize {
        self.materials.push(material);
        self.materials.len() - 1
    }

    /// Whether a link between the two nodes already exists (Bullet's
    /// `checkLink`).
    pub fn check_link(&self, node0: usize, node1: usize) -> bool {
        self.links.iter().any(|l| {
            (l.n[0] == node0 && l.n[1] == node1) || (l.n[0] == node1 && l.n[1] == node0)
        })
    }

    /// Append a link between two nodes (Bullet's `appendLink`; `mat = None`
    /// uses the default material). Pass `check_exist = true` to skip
    /// duplicates. Returns the link index, or `None` when skipped.
    pub fn append_link(
        &mut self,
        node0: usize,
        node1: usize,
        mat: Option<usize>,
        check_exist: bool,
    ) -> Option<usize> {
        if check_exist && self.check_link(node0, node1) {
            return None;
        }
        let rl = (self.nodes[node0].x - self.nodes[node1].x).length();
        self.links.push(Link {
            n: [node0, node1],
            mat: mat.unwrap_or(0),
            rl,
            bending: false,
            c0: 0.0,
            c1: rl * rl,
            c2: 0.0,
            c3: Vector3::ZERO,
        });
        self.update_rt_cst = true;
        Some(self.links.len() - 1)
    }

    /// Set the mass of one node, 0 to pin it (Bullet's `setMass`).
    pub fn set_mass(&mut self, node: usize, mass: Real) {
        self.nodes[node].im = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.update_rt_cst = true;
    }

    /// Append a surface triangle (Bullet's `appendFace(node0, node1, node2)`;
    /// degenerate (repeated-index) triangles are rejected like Bullet).
    /// The face-tree leaf is inserted lazily on the next
    /// `initialize_face_tree`/`predict_motion`.
    pub fn append_face(&mut self, node0: usize, node1: usize, node2: usize, mat: Option<usize>) {
        if node0 == node1 || node1 == node2 || node2 == node0 {
            return;
        }
        let ra = super::geometry::area_of(
            self.nodes[node0].x,
            self.nodes[node1].x,
            self.nodes[node2].x,
        );
        self.faces.push(Face::new([node0, node1, node2], mat.unwrap_or(0), ra));
        self.update_rt_cst = true;
    }

    /// (Re)build the face tree from the current faces (Bullet's
    /// `initializeFaceTree`): leaves hold the exact triangle AABBs — the
    /// margin padding arrives with the first `predict_motion` update.
    pub fn initialize_face_tree(&mut self) {
        self.fdbvt.clear();
        for i in 0..self.faces.len() {
            let n = self.faces[i].n;
            let [x0, x1, x2] = [self.nodes[n[0]].x, self.nodes[n[1]].x, self.nodes[n[2]].x];
            let min = x0.min(x1).min(x2);
            let max = x0.max(x1).max(x2);
            self.faces[i].leaf = self.fdbvt.insert(Aabb::new(min, max), i);
        }
    }

    /// Append a tetrahedron (Bullet's `appendTetra(n0..n3, mat)`); the rest
    /// volume is the raw triple product, sign included.
    pub fn append_tetra(&mut self, node0: usize, node1: usize, node2: usize, node3: usize, mat: Option<usize>) {
        let rv = super::geometry::volume_of(
            self.nodes[node0].x,
            self.nodes[node1].x,
            self.nodes[node2].x,
            self.nodes[node3].x,
        );
        self.tetras.push(Tetra { n: [node0, node1, node2, node3], mat: mat.unwrap_or(0), rv });
        self.update_rt_cst = true;
    }

    /// Redistribute `mass` over the nodes weighted by their incident
    /// tetrahedra volumes (Bullet's `setVolumeMass`): each node's share is
    /// `rank / sum(|rv|)` with `rank` the number of incident tetras, then
    /// `setTotalMass` normalizes the total. Nodes covered by no tetra keep
    /// mass zero.
    pub fn set_volume_mass(&mut self, mass: Real) {
        let mut ranks = vec![0u32; self.nodes.len()];
        for n in &mut self.nodes {
            n.im = 0.0;
        }
        for t in &self.tetras {
            for &ni in &t.n {
                self.nodes[ni].im += t.rv.abs();
                ranks[ni] += 1;
            }
        }
        for (n, rank) in self.nodes.iter_mut().zip(ranks) {
            if n.im > 0.0 {
                n.im = rank as Real / n.im;
            }
        }
        self.set_total_mass(mass);
    }

    /// Distribute `density` over the tetrahedra (Bullet's `setVolumeDensity`:
    /// the total is `sum(|rv| over node incidents) * density / 6`).
    pub fn set_volume_density(&mut self, density: Real) {
        let mut volume = 0.0;
        for t in &self.tetras {
            for _ in 0..4 {
                volume += t.rv.abs();
            }
        }
        self.set_volume_mass(volume * density / 6.0);
    }

    /// Generate bending links between nodes graph-distance `distance` apart
    /// (Bullet's `generateBendingConstraints`): shortest paths over the
    /// structural links (the `distance == 2` fast path over adjacency lists,
    /// Floyd-Warshall otherwise — the classic system has no volume kernel,
    /// so `distance` beyond the graph diameter simply matches nothing).
    /// Returns the number of links appended.
    pub fn generate_bending_constraints(&mut self, distance: usize, mat: Option<usize>) -> usize {
        if distance <= 1 {
            return 0;
        }
        let n = self.nodes.len();
        let inf = u32::MAX / 2;
        let idx = |x: usize, y: usize| y * n + x;
        let mut adj = vec![inf; n * n];
        for i in 0..n {
            adj[idx(i, i)] = 0;
        }
        let link_ends: Vec<[usize; 2]> = self.links.iter().map(|l| l.n).collect();
        for &[ia, ib] in &link_ends {
            adj[idx(ia, ib)] = 1;
            adj[idx(ib, ia)] = 1;
        }
        if distance == 2 {
            // Adjacency-list fast path (Bullet's special case): node ii's
            // distance-2 set is the union of its neighbors' neighbors.
            let mut node_links: Vec<Vec<usize>> = vec![Vec::new(); n];
            for &[ia, ib] in &link_ends {
                if !node_links[ia].contains(&ib) {
                    node_links[ia].push(ib);
                }
                if !node_links[ib].contains(&ia) {
                    node_links[ib].push(ia);
                }
            }
            for ii in 0..n {
                for &k in &node_links[ii] {
                    for &j in &node_links[k] {
                        if ii != j {
                            let sum = adj[idx(ii, k)] + adj[idx(k, j)];
                            if adj[idx(ii, j)] > sum {
                                adj[idx(ii, j)] = sum;
                                adj[idx(j, ii)] = sum;
                            }
                        }
                    }
                }
            }
        } else {
            // Generic Floyd-Warshall over the upper triangle.
            for k in 0..n {
                for j in 0..n {
                    for i in (j + 1)..n {
                        let sum = adj[idx(i, k)] + adj[idx(k, j)];
                        if adj[idx(i, j)] > sum {
                            adj[idx(i, j)] = sum;
                            adj[idx(j, i)] = sum;
                        }
                    }
                }
            }
        }
        let mut nlinks = 0;
        for j in 0..n {
            for i in (j + 1)..n {
                if adj[idx(i, j)] == distance as u32
                    && let Some(li) = self.append_link(i, j, mat, false)
                {
                    self.links[li].bending = true;
                    nlinks += 1;
                }
            }
        }
        nlinks
    }

    /// Refresh face and node normals (Bullet's `updateNormals`: accumulate
    /// the unnormalized cross products, then normalize — node normals stay
    /// zero for zero-area fans).
    pub fn update_normals(&mut self) {
        for n in &mut self.nodes {
            n.n = Vector3::ZERO;
        }
        for f in 0..self.faces.len() {
            let n = self.faces[f].n;
            let normal = (self.nodes[n[1]].x - self.nodes[n[0]].x)
                .cross(self.nodes[n[2]].x - self.nodes[n[0]].x);
            self.faces[f].normal = normal;
            for &ni in &n {
                self.nodes[ni].n += normal;
            }
        }
        const EPS: Real = if cfg!(feature = "f64") { 2.2e-16 } else { 1.19e-7 };
        for n in &mut self.nodes {
            let len = n.n.length();
            if len > EPS {
                n.n /= len;
            }
        }
    }

    /// Bullet's `evaluateCom`: the pose-weighted COM when a frame pose is
    /// enabled, otherwise the zero vector (the quirk `set_pose` builds on —
    /// rest offsets end up in world coordinates unless a frame is tracked).
    pub fn evaluate_com(&self) -> Vector3 {
        let mut com = Vector3::ZERO;
        if self.pose.bframe {
            for i in 0..self.nodes.len() {
                com += self.nodes[i].x * self.pose.wgh[i];
            }
        }
        com
    }

    /// Capture the rest pose (Bullet's `setPose`): mass weights, rest
    /// offsets, rest volume and the inverse rest covariance. The frame
    /// rotation/scale tracking (and with it `PolarDecompose`) belongs to the
    /// cluster phase; `bframe` still installs the weights and COM so the
    /// pose is ready for it.
    pub fn set_pose(&mut self, bvolume: bool, bframe: bool) {
        self.pose.bvolume = bvolume;
        self.pose.bframe = bframe;

        // Weights: fixed nodes get a `kmass` share so they dominate.
        let omass = self.get_total_mass();
        let kmass = omass * self.nodes.len() as Real * 1000.0;
        let mut tmass = omass;
        for n in &self.nodes {
            if n.im <= 0.0 {
                tmass += kmass;
            }
        }
        self.pose.wgh = self
            .nodes
            .iter()
            .map(|n| if n.im > 0.0 { 1.0 / (n.im * tmass) } else { kmass / tmass })
            .collect();

        // Rest offsets and volume.
        let com = self.evaluate_com();
        self.pose.pos = self.nodes.iter().map(|n| n.x - com).collect();
        self.pose.volume = if bvolume { self.get_volume() } else { 0.0 };
        self.pose.com = com;
        self.pose.rot = Matrix3::IDENTITY;
        self.pose.scl = Matrix3::IDENTITY;

        // Inverse rest covariance (the outer-product accumulation
        // `aqq += (q*w) outer q`, symmetric by construction).
        let mut aqq = Matrix3::ZERO;
        for i in 0..self.nodes.len() {
            let q = self.pose.pos[i];
            let mq = q * self.pose.wgh[i];
            aqq += Matrix3::from_cols(mq * q.x, mq * q.y, mq * q.z);
        }
        self.pose.aqq = aqq.inverse();

        self.update_constants();
    }

    /// Mass of one node (0 when fixed; Bullet's `getMass`).
    pub fn get_mass(&self, node: usize) -> Real {
        let im = self.nodes[node].im;
        if im > 0.0 {
            1.0 / im
        } else {
            0.0
        }
    }

    /// Sum of node masses (Bullet's `getTotalMass`).
    pub fn get_total_mass(&self) -> Real {
        (0..self.nodes.len()).map(|i| self.get_mass(i)).sum()
    }

    /// Redistribute the total mass over the nodes, weighted by the current
    /// masses (Bullet's `setTotalMass(mass, false)`; the face-area variant
    /// arrives with the cloth phase). Inverse masses scale by `total/mass`.
    pub fn set_total_mass(&mut self, mass: Real) {
        let total = self.get_total_mass();
        let scale = mass / total;
        for n in &mut self.nodes {
            n.im /= scale;
        }
        self.update_rt_cst = true;
    }

    /// Redistribute the total mass over the nodes, weighted by their
    /// incident faces' doubled areas (Bullet's `setTotalMass(mass,
    /// fromauthors = true)` — the mass model of surface-built bodies like
    /// the ellipsoid, whose tetra set is empty).
    pub fn set_total_mass_from_faces(&mut self, mass: Real) {
        for n in &mut self.nodes {
            n.im = 0.0;
        }
        for f in &self.faces {
            for &ni in &f.n {
                self.nodes[ni].im += f.ra;
            }
        }
        for n in &mut self.nodes {
            if n.im != 0.0 {
                n.im = 1.0 / n.im;
            }
        }
        self.set_total_mass(mass);
    }

    /// Mass-weighted center of mass (Bullet's `getCenterOfMass`).
    pub fn get_center_of_mass(&self) -> Vector3 {
        let mut com = Vector3::ZERO;
        for i in 0..self.nodes.len() {
            com += self.nodes[i].x * self.get_mass(i);
        }
        com / self.get_total_mass()
    }

    /// Add a force to every node (Bullet's `addForce` — the force is added
    /// whole to each node, not divided).
    pub fn add_force(&mut self, force: Vector3) {
        for n in &mut self.nodes {
            n.f += force;
        }
    }

    /// Add a force to one node (Bullet's `addForce(force, node)`).
    pub fn add_force_at_node(&mut self, node: usize, force: Vector3) {
        self.nodes[node].f += force;
    }

    /// Add a velocity to every dynamic node (Bullet's `addVelocity`).
    pub fn add_velocity(&mut self, velocity: Vector3) {
        for n in &mut self.nodes {
            if n.im > 0.0 {
                n.v += velocity;
            }
        }
    }

    /// Overwrite the velocity of every dynamic node (Bullet's `setVelocity`).
    pub fn set_velocity(&mut self, velocity: Vector3) {
        for n in &mut self.nodes {
            if n.im > 0.0 {
                n.v = velocity;
            }
        }
    }

    /// Re-seed link rest lengths from the current node distance (Bullet's
    /// `resetLinkRestLengths`).
    pub fn reset_link_rest_lengths(&mut self) {
        for l in &mut self.links {
            l.rl = (self.nodes[l.n[0]].x - self.nodes[l.n[1]].x).length();
            l.c1 = l.rl * l.rl;
        }
    }

    /// Scale all rest lengths (Bullet's `setRestLengthScale`; the rest
    /// lengths are re-based so scaling back restores them).
    pub fn set_rest_length_scale(&mut self, scale: Real) {
        for l in &mut self.links {
            l.rl = l.rl / self.rest_length_scale * scale;
            l.c1 = l.rl * l.rl;
        }
        self.rest_length_scale = scale;
    }

    /// Refresh the link runtime constants (Bullet's `updateLinkConstants`):
    /// `c0 = (im0 + im1) / kLST`.
    pub fn update_link_constants(&mut self) {
        for l in &mut self.links {
            let m = &self.materials[l.mat];
            l.c0 = (self.nodes[l.n[0]].im + self.nodes[l.n[1]].im) / m.k_lst;
        }
    }

    /// Pin a node to a rigid body at its current world position (Bullet's
    /// `appendAnchor(node, body, disableCollisions, influence)`; the anchor
    /// local point is the node position mapped into body space). The body
    /// pose is passed by value so callers can hand over world borrows
    /// piecewise.
    pub fn append_anchor(
        &mut self,
        node: usize,
        body: BodyHandle,
        body_position: Vector3,
        body_orientation: Quaternion,
        influence: Real,
    ) {
        // local = R^T * (x - body position) (Bullet's frame-inverted map).
        let local = body_orientation.inverse() * (self.nodes[node].x - body_position);
        self.append_anchor_local(node, body, local, influence);
    }

    /// Pin a node to a rigid body at an explicit body-space pivot (Bullet's
    /// `appendAnchor(node, body, localPivot, ...)`).
    pub fn append_anchor_local(&mut self, node: usize, body: BodyHandle, local: Vector3, influence: Real) {
        self.nodes[node].battach = true;
        self.anchors.push(Anchor {
            node,
            body,
            local,
            influence,
            c0: Matrix3::ZERO,
            c1: Vector3::ZERO,
            c2: 0.0,
        });
    }

    /// Remove every anchor of one node (Bullet's `removeAnchor`).
    pub fn remove_anchor(&mut self, node: usize) {
        self.anchors.retain(|a| a.node != node);
        self.nodes[node].battach = false;
    }

    /// Shuffle the constraints with Bullet's built-in deterministic LCG
    /// (`randomizeConstraints`: `seed = 243703`,
    /// `seed = (1664525*seed + 1013904223) & 0xffffffff`, Fisher-Yates-style
    /// swaps against `seed % len`) to reduce solver bias. Faces join the
    /// shuffle with the cloth phase; links alone are deterministic.
    pub fn randomize_constraints(&mut self) {
        let mut seed: u32 = 243703;
        let mut next = |ni: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed % ni
        };
        let ni = self.links.len() as u32;
        for i in 0..self.links.len() {
            let j = next(ni) as usize;
            self.links.swap(i, j);
        }
    }

    /// Install a linear joint between this body's cluster `cluster0` and
    /// `body1` (Bullet's `appendLinearJoint(specs, body0, body1)`): both
    /// frames capture a local anchor at `specs.position`. `xform1` is the
    /// other side's world frame `(origin, basis)` at installation time
    /// (Bullet reads it through the `Body` pointer).
    pub fn append_linear_joint(
        &mut self,
        self_idx: usize,
        cluster0: usize,
        specs: &LinearJointSpecs,
        body1: Side,
        xform1: (Vector3, Matrix3),
    ) {
        let cluster = &self.clusters[cluster0];
        let refs0 = cluster.framerot.transpose() * (specs.position - cluster.framepos);
        let refs1 = xform1.1.transpose() * (specs.position - xform1.0);
        self.joints.push(SoftJoint::Linear(LJoint {
            bodies: [
                Side::Cluster {
                    soft: SoftBodyHandle::from_index(self_idx),
                    cluster: cluster0,
                },
                body1,
            ],
            refs: [refs0, refs1],
            cfm: specs.tuning.cfm,
            erp: specs.tuning.erp,
            split: specs.tuning.split,
            rpos: [Vector3::ZERO; 2],
            drift: Vector3::ZERO,
            sdrift: Vector3::ZERO,
            massmatrix: Matrix3::ZERO,
        }));
    }

    /// Install an angular joint between this body's cluster `cluster0` and
    /// `body1` (Bullet's `appendAngularJoint(specs, body0, body1)`): both
    /// frames capture a local axis from `specs.axis`.
    pub fn append_angular_joint(
        &mut self,
        self_idx: usize,
        cluster0: usize,
        specs: &AngularJointSpecs,
        body1: Side,
        xform1: (Vector3, Matrix3),
    ) {
        let cluster = &self.clusters[cluster0];
        let refs0 = cluster.framerot.transpose() * specs.axis;
        let refs1 = xform1.1.transpose() * specs.axis;
        self.joints.push(SoftJoint::Angular(AJoint {
            bodies: [
                Side::Cluster {
                    soft: SoftBodyHandle::from_index(self_idx),
                    cluster: cluster0,
                },
                body1,
            ],
            refs: [refs0, refs1],
            cfm: specs.tuning.cfm,
            erp: specs.tuning.erp,
            split: specs.tuning.split,
            axis: [Vector3::ZERO; 2],
            control: specs.control.clone_box(),
            drift: Vector3::ZERO,
            sdrift: Vector3::ZERO,
            massmatrix: Matrix3::ZERO,
        }));
    }

    /// Move every node by `trs` (Bullet's `translate`).
    pub fn translate(&mut self, trs: Vector3) {
        self.transform(Matrix3::IDENTITY, trs);
    }

    /// Rotate every node about the world origin (Bullet's `rotate`).
    pub fn rotate(&mut self, rot: Quaternion) {
        self.transform(Matrix3::from_quat(rot), Vector3::ZERO);
    }

    /// Apply a rigid transform to every node (Bullet's `transform`):
    /// positions and previous positions map, normals rotate, the node-tree
    /// leaves reinsert at their exact new volumes, then the derived data
    /// rebuilds — `update_constants` re-seeds the link rest lengths, so a
    /// transformed body keeps rest lengths matching its geometry (Bullet's
    /// semantics).
    pub fn transform(&mut self, basis: Matrix3, origin: Vector3) {
        let margin = self.margin;
        for i in 0..self.nodes.len() {
            let n = &mut self.nodes[i];
            n.x = basis * n.x + origin;
            n.q = basis * n.q + origin;
            n.n = basis * n.n;
            let pad = Vector3::splat(margin);
            let volume = Aabb::new(n.x - pad, n.x + pad);
            self.ndbvt.update_exact(n.leaf, volume);
        }
        self.update_normals();
        self.update_bounds();
        self.update_constants();
    }

    /// Scale every node per-axis (Bullet's `scale`): positions and previous
    /// positions multiply, then the same derived-data refresh as
    /// `transform` (Bullet's trailing `initializeDmInverse` belongs to the
    /// NeoHookean system and is not ported).
    pub fn scale(&mut self, scl: Vector3) {
        let margin = self.margin;
        for i in 0..self.nodes.len() {
            let n = &mut self.nodes[i];
            n.x *= scl;
            n.q *= scl;
            let pad = Vector3::splat(margin);
            let volume = Aabb::new(n.x - pad, n.x + pad);
            self.ndbvt.update_exact(n.leaf, volume);
        }
        self.update_normals();
        self.update_bounds();
        self.update_constants();
    }

    /// Best-fit rigid transform from the rest positions to the current
    /// nodes (Bullet's `getRigidTransform`): the mass-weighted COM plus the
    /// Kabsch rotation. Bullet decomposes the covariance `S` with
    /// `btImplicitQRSVD` and returns `V*U^T`; for full-rank `S` that equals
    /// the transpose of the polar factor, which is what we reuse here —
    /// rank-deficient fits may differ (a transform API, not a solver path).
    pub fn get_rigid_transform(&self) -> (Matrix3, Vector3) {
        let t = self.get_center_of_mass();
        // `S -= OuterProduct(m_X[i], t - x_i)` — the outer-product
        // accumulation in glam's column-major layout (see `update_pose`
        // for the transposition trap).
        let mut s = Matrix3::ZERO;
        let count = self.rest_positions.len().min(self.nodes.len());
        for i in 0..count {
            let a = self.rest_positions[i];
            let b = self.nodes[i].x - t;
            s += Matrix3::from_cols(a * b.x, a * b.y, a * b.z);
        }
        let (w, _) = super::cluster::polar_decompose(&s);
        (w.transpose(), t)
    }

    /// Teleport the body so its rigid fit lands on `(basis, origin)`
    /// (Bullet's `transformTo`): the current best rigid fit (measured from
    /// the initial rest positions) is composed with the inverse of the
    /// target and applied through `transform`.
    pub fn transform_to(&mut self, basis: Matrix3, origin: Vector3) {
        let (cur_basis, cur_origin) = self.get_rigid_transform();
        let new_basis = basis * cur_basis.transpose();
        let new_origin = origin - new_basis * cur_origin;
        self.transform(new_basis, new_origin);
    }

    /// Cut every link and face edge between the two nodes, inserting a pair
    /// of fresh nodes at the interpolated `position` (Bullet's `cutLink`).
    /// Returns whether anything matched; on no match the fresh pair is
    /// removed again.
    pub fn cut_link(&mut self, node0: usize, node1: usize, position: Real) -> bool {
        let x = self.nodes[node0].x.lerp(self.nodes[node1].x, position);
        let v = self.nodes[node0].v.lerp(self.nodes[node1].v, position);
        let pn = [self.append_node(x, 1.0), self.append_node(x, 1.0)];
        self.nodes[pn[0]].v = v;
        self.nodes[pn[1]].v = v;
        let mut done = false;
        // Links: every edge matching (node0, node1) splits into two halves
        // through the fresh pair. Appended copies stay out of this pass
        // (Bullet iterates a size snapshot).
        let nlinks = self.links.len();
        for i in 0..nlinks {
            if let Some(mtch) = match_edge(self.links[i].n[0], self.links[i].n[1], node0, node1) {
                let mut copy = self.links[i];
                self.links[i].n[1] = pn[mtch];
                copy.n[0] = pn[1 - mtch];
                self.links.push(copy);
                done = true;
            }
        }
        // Faces: the same edge match turns the triangle pair into a quad
        // split, sewing the diagonal with two fresh links.
        let nfaces = self.faces.len();
        for i in 0..nfaces {
            let mut k = 2;
            let mut l = 0;
            while l < 3 {
                if let Some(mtch) = match_edge(self.faces[i].n[k], self.faces[i].n[l], node0, node1)
                {
                    let mut copy = self.faces[i];
                    self.faces[i].n[l] = pn[mtch];
                    copy.n[k] = pn[1 - mtch];
                    self.faces.push(copy);
                    let next = self.faces[i].n[(l + 1) % 3];
                    let mat = Some(self.faces[i].mat);
                    self.append_link(pn[0], next, mat, true);
                    self.append_link(pn[1], next, mat, true);
                }
                k = l;
                l += 1;
            }
        }
        if !done {
            for &p in &pn {
                let leaf = self.nodes[p].leaf;
                self.ndbvt.remove(leaf);
            }
            self.nodes.pop();
            self.nodes.pop();
        }
        self.update_rt_cst = true;
        done
    }

    /// Refine the mesh against an implicit surface (Bullet's `refine`):
    /// bending links crossing the surface are dropped, structural
    /// link/face edges crossing it gain a node at the intersection, and
    /// with `cut` the surface-side geometry is duplicated so the body can
    /// split along the surface.
    pub fn refine(&mut self, ifn: &dyn ImplicitFn, accuracy: Real, cut: bool) {
        let ncount = self.nodes.len();
        // Symmetric edge matrix (Bullet's `btSymMatrix`, triangle storage):
        // -2 = no edge, -1 = edge without intersection, >= 0 = index of the
        // inserted intersection node.
        let mut edges = vec![-2i64; ncount * (ncount + 1) / 2];
        fn sym_index(a: usize, b: usize) -> usize {
            let (c, r) = if a > b { (b, a) } else { (a, b) };
            r * (r + 1) / 2 + c
        }

        // Filter out bending links spanning the surface.
        let mut i = 0;
        while i < self.links.len() {
            let l = self.links[i];
            if l.bending {
                let (xa, xb) = (self.nodes[l.n[0]].x, self.nodes[l.n[1]].x);
                if !same_sign(ifn.eval(xa), ifn.eval(xb)) {
                    let last = self.links.len() - 1;
                    self.links.swap(i, last);
                    self.links.pop();
                    continue;
                }
            }
            i += 1;
        }
        // Fill edges from links and faces.
        for l in &self.links {
            edges[sym_index(l.n[0], l.n[1])] = -1;
        }
        for f in &self.faces {
            edges[sym_index(f.n[0], f.n[1])] = -1;
            edges[sym_index(f.n[1], f.n[2])] = -1;
            edges[sym_index(f.n[2], f.n[0])] = -1;
        }
        // Intersect: solve each edge against the implicit surface and grow
        // a node at the crossing, re-splitting both endpoint masses.
        for i in 0..ncount {
            for j in (i + 1)..ncount {
                if edges[sym_index(i, j)] == -1 {
                    let t = implicit_solve(ifn, self.nodes[i].x, self.nodes[j].x, accuracy);
                    if t > 0.0 {
                        let x = self.nodes[i].x.lerp(self.nodes[j].x, t);
                        let v = self.nodes[i].v.lerp(self.nodes[j].v, t);
                        let mut m = 0.0;
                        let (ia, ib) = (self.nodes[i].im, self.nodes[j].im);
                        if ia > 0.0 {
                            if ib > 0.0 {
                                let ma = 1.0 / ia;
                                let mb = 1.0 / ib;
                                let mc = ma + (mb - ma) * t;
                                let f = (ma + mb) / (ma + mb + mc);
                                self.nodes[i].im = 1.0 / (ma * f);
                                self.nodes[j].im = 1.0 / (mb * f);
                                m = mc * f;
                            } else {
                                self.nodes[i].im /= 0.5;
                                m = 1.0 / self.nodes[i].im;
                            }
                        } else if ib > 0.0 {
                            self.nodes[j].im /= 0.5;
                            m = 1.0 / self.nodes[j].im;
                        }
                        let ni = self.append_node(x, m);
                        edges[sym_index(i, j)] = ni as i64;
                        self.nodes[ni].v = v;
                    }
                }
            }
        }
        // Refine links: both ends old and an intersection on the edge —
        // split in place (appended copies stay out, Bullet snapshots size).
        let nlinks = self.links.len();
        for i in 0..nlinks {
            let idx = self.links[i].n;
            if idx[0] < ncount && idx[1] < ncount {
                let m = edges[sym_index(idx[0], idx[1])];
                if m > 0 {
                    let mut copy = self.links[i];
                    self.links[i].n[1] = m as usize;
                    copy.n[0] = m as usize;
                    self.links.push(copy);
                }
            }
        }
        // Refine faces: the first intersected edge turns the triangle into
        // a pair; the same face is reprocessed (other edges may carry
        // intersections) — Bullet's `--i; break`.
        let mut i = 0;
        while i < self.faces.len() {
            let idx = self.faces[i].n;
            let mut matched = false;
            let mut j = 2;
            let mut k = 0;
            while k < 3 && !matched {
                if idx[j] < ncount && idx[k] < ncount {
                    let m = edges[sym_index(idx[j], idx[k])];
                    if m > 0 {
                        let m = m as usize;
                        let l = (k + 1) % 3;
                        let mut copy = self.faces[i];
                        self.faces[i].n = [idx[l], idx[j], m];
                        copy.n = [m, idx[k], idx[l]];
                        self.faces.push(copy);
                        self.append_link(m, idx[l], Some(self.faces[i].mat), false);
                        matched = true;
                    }
                }
                j = k;
                k += 1;
            }
            if !matched {
                i += 1;
            }
        }
        // Cut: duplicate the nodes on the negative side of the surface and
        // redirect the surface-side links/faces onto the copies.
        if cut {
            let pcount = ncount;
            let ntotal = self.nodes.len();
            let mut cnodes = vec![0usize; ntotal];
            // Index juggling with a `&mut self` append inside — mirrors
            // Bullet's pointer loop verbatim.
            #[allow(clippy::needless_range_loop)]
            for i in 0..ntotal {
                let x = self.nodes[i].x;
                if i >= pcount || ifn.eval(x).abs() < accuracy {
                    let v = self.nodes[i].v;
                    let mut m = self.get_mass(i);
                    if m > 0.0 {
                        m *= 0.5;
                        self.nodes[i].im /= 0.5;
                    }
                    let cn = self.append_node(x, m);
                    cnodes[i] = cn;
                    self.nodes[cn].v = v;
                }
            }
            let nlinks = self.links.len();
            for i in 0..nlinks {
                let id = self.links[i].n;
                let mut todetach = None;
                if cnodes[id[0]] != 0 && cnodes[id[1]] != 0 {
                    let copy = self.links[i];
                    self.links.push(copy);
                    todetach = Some(self.links.len() - 1);
                } else if ifn.eval(self.nodes[id[0]].x) < accuracy
                    && ifn.eval(self.nodes[id[1]].x) < accuracy
                {
                    // Bullet guards the redirect with `if (todetach)` — an
                    // integer 0 sentinel that also swallows link index 0
                    // (ported verbatim).
                    if i != 0 {
                        todetach = Some(i);
                    }
                }
                if let Some(td) = todetach {
                    for j in 0..2 {
                        let cn = cnodes[self.links[td].n[j]];
                        if cn != 0 {
                            self.links[td].n[j] = cn;
                        }
                    }
                }
            }
            let nfaces = self.faces.len();
            for i in 0..nfaces {
                let n = self.faces[i].n;
                if ifn.eval(self.nodes[n[0]].x) < accuracy
                    && ifn.eval(self.nodes[n[1]].x) < accuracy
                    && ifn.eval(self.nodes[n[2]].x) < accuracy
                {
                    for j in 0..3 {
                        let cn = cnodes[self.faces[i].n[j]];
                        if cn != 0 {
                            self.faces[i].n[j] = cn;
                        }
                    }
                }
            }
            // Clean orphans: links with a rank-1 endpoint (a dangling node
            // nothing else references) drop out.
            let mut ranks = vec![0i64; self.nodes.len()];
            for l in &self.links {
                for &ni in &l.n {
                    ranks[ni] += 1;
                }
            }
            for f in &self.faces {
                for &ni in &f.n {
                    ranks[ni] += 1;
                }
            }
            let mut i = 0;
            while i < self.links.len() {
                let id = self.links[i].n;
                if ranks[id[0]] == 1 || ranks[id[1]] == 1 {
                    ranks[id[0]] -= 1;
                    ranks[id[1]] -= 1;
                    let last = self.links.len() - 1;
                    self.links.swap(i, last);
                    self.links.pop();
                } else {
                    i += 1;
                }
            }
        }
        self.update_rt_cst = true;
    }
}

/// An implicit surface for [`SoftBody::refine`] (Bullet's
/// `btSoftBody::ImplicitFn`): the function's sign flips across the surface.
pub trait ImplicitFn {
    /// Evaluate the implicit function at `x`.
    fn eval(&self, x: Vector3) -> Real;
}

/// Bullet's `SameSign`: strict product positivity (zeros count as a sign
/// change).
fn same_sign(x: Real, y: Real) -> bool {
    x * y > 0.0
}

/// Bullet's `MatchEdge`: does the undirected edge `(a, b)` equal `(ma, mb)`?
/// Returns the matched end of the pair.
fn match_edge(a: usize, b: usize, ma: usize, mb: usize) -> Option<usize> {
    if a == ma && b == mb {
        Some(0)
    } else if a == mb && b == ma {
        Some(1)
    } else {
        None
    }
}

/// Solve the implicit-function zero crossing along a segment (Bullet's
/// `ImplicitSolve`, `btSoftBodyInternals.h` 1199): a secant iteration
/// bracketed to `[0, 1]`, returning the crossing parameter or -1.
fn implicit_solve(ifn: &dyn ImplicitFn, a: Vector3, b: Vector3, accuracy: Real) -> Real {
    let mut span = [0.0 as Real, 1.0];
    let mut values = [ifn.eval(a), ifn.eval(b)];
    if values[0] > values[1] {
        span.swap(0, 1);
        values.swap(0, 1);
    }
    if values[0] > -accuracy {
        return -1.0;
    }
    if values[1] < accuracy {
        return -1.0;
    }
    for _ in 0..256 {
        let t = span[0] + (span[1] - span[0]) * (values[0] / (values[0] - values[1]));
        let v = ifn.eval(a + (b - a) * t);
        if t <= 0.0 || t >= 1.0 {
            break;
        }
        if v.abs() < accuracy {
            return t;
        }
        if v < 0.0 {
            span[0] = t;
            values[0] = v;
        } else {
            span[1] = t;
            values[1] = v;
        }
    }
    -1.0
}
