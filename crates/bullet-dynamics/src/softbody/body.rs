//! Soft-body core data — port of `btSoftBody`'s node/link topology and
//! public body API (construction, append, mass and force helpers).
//!
//! Bullet weaves `Node*` pointers through links/faces/clusters; here every
//! element stores node indices instead (Bullet's own serialization path
//! `pointersToIndices` proves the index encoding is equivalent).

use bullet_collision::broadphase::DbvtTree;
use bullet_math::{Aabb, Matrix3, Quaternion, Real, Vector3};

use super::collide::{Anchor, RContact, SContact};
use super::config::{Config, SolverState};

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
    /// Materials.
    pub materials: Vec<Material>,
    /// Node AABB tree (Bullet's `m_ndbvt`): leaves keyed by node index,
    /// volumes padded by the collision margin and the step's velocity.
    pub ndbvt: DbvtTree,
    /// Face AABB tree (Bullet's `m_fdbvt`), built lazily on the first face
    /// insert and kept fat like the node tree.
    pub fdbvt: DbvtTree,
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
            materials: vec![Material::default()],
            ndbvt: DbvtTree::new(),
            fdbvt: DbvtTree::new(),
            pose: Pose::default(),
            wind_velocity: Vector3::ZERO,
            air_density: 1.2,
            rcontacts: Vec::new(),
            scontacts: Vec::new(),
            anchors: Vec::new(),
            bounds: [Vector3::ZERO, Vector3::ZERO],
            update_rt_cst: true,
            rest_length_scale: 1.0,
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

        // Inverse rest covariance (assembled per row like Bullet:
        // `aqq[row] += mq[row] * q`).
        let mut aqq = Matrix3::ZERO;
        for i in 0..self.nodes.len() {
            let q = self.pose.pos[i];
            let mq = q * self.pose.wgh[i];
            aqq.x_axis.x += mq.x * q.x;
            aqq.x_axis.y += mq.x * q.y;
            aqq.x_axis.z += mq.x * q.z;
            aqq.y_axis.x += mq.y * q.x;
            aqq.y_axis.y += mq.y * q.y;
            aqq.y_axis.z += mq.y * q.z;
            aqq.z_axis.x += mq.z * q.x;
            aqq.z_axis.y += mq.z * q.y;
            aqq.z_axis.z += mq.z * q.z;
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
        body: usize,
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
    pub fn append_anchor_local(&mut self, node: usize, body: usize, local: Vector3, influence: Real) {
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
}
