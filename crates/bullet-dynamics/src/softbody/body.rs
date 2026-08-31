//! Soft-body core data — port of `btSoftBody`'s node/link topology and
//! public body API (construction, append, mass and force helpers).
//!
//! Bullet weaves `Node*` pointers through links/faces/clusters; here every
//! element stores node indices instead (Bullet's own serialization path
//! `pointersToIndices` proves the index encoding is equivalent).

use bullet_math::{Real, Vector3};

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
    /// Materials.
    pub materials: Vec<Material>,
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
            materials: vec![Material::default()],
            bounds: [Vector3::ZERO, Vector3::ZERO],
            update_rt_cst: true,
            rest_length_scale: 1.0,
        };
        for (&x, &m) in positions.iter().zip(masses) {
            body.nodes.push(Node::new(x, m));
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
