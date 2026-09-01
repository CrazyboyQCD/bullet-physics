//! Soft-body collision — SDF_RS (nodes vs rigid bodies) with exact
//! distances (phase B; soft-soft VF_SS and cluster CL_* arrive later —
//! see `docs/softbody.md` §5/§10).
//!
//! Bullet's `checkContact` samples a sparse signed-distance field cached
//! over the rigid shape; we query the exact point-shape distance instead
//! (recorded deviation): analytic for planes, the GJK distance walk for
//! convex shapes (the query point is a zero-radius sphere), and closest
//! point-on-triangle over the BVH for static meshes/heightfields. The
//! contact semantics match Bullet exactly: a node contacts when
//! `distance - margin < 0`, with `offset` stored so the solver's
//! `dot(x, n) + offset` recovers the detection-time distance.

use bullet_collision::narrowphase::epa;
use bullet_collision::narrowphase::gjk::{core_support_minkowski, gjk_contact, GjkOutcome};
use bullet_collision::narrowphase::point_triangle_distance;
use bullet_collision::shapes::CollisionShape;
use bullet_collision::shapes::TriangleMesh;
use bullet_math::{Aabb, Matrix3, Quaternion, Real, Vector3};

use crate::body::RigidBody;

use super::body::{Cluster, Side, SoftBody};
use super::config::SoftCollision;
use super::geometry::{
    bary_coord, bary_eval_real, bary_eval_vec3, project_origin_triangle, ray_from_to_triangle,
};
use super::joints::{impulse_matrix_two, CJoint, SoftJoint};

/// What a soft-body contact hits (Bullet's `sCti::m_colObj` pointer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactOwner {
    /// A rigid body index.
    Rigid(usize),
    /// A multibody link collider: `(multibody index, link index)` with
    /// `-1` for the base (phase F).
    Link { mb: usize, link: i32 },
}

/// Contact info against one rigid body (Bullet's `sCti`).
#[derive(Clone, Copy, Debug)]
pub struct Cti {
    /// The body the contact is against.
    pub body: ContactOwner,
    /// Outward surface normal (body surface toward the node, world space).
    pub normal: Vector3,
    /// Plane offset such that `dot(x, normal) + offset` equals the
    /// detection-time signed distance minus the margin (Bullet's
    /// `m_offset = -dot(n, x - n*dst)`).
    pub offset: Real,
}

/// One node-vs-rigid contact with its precomputed impulse-matrix row
/// (Bullet's `RContact`; against a multibody link the assembly treats the
/// collider as kinematic — `imb = 0`, zero inertia, zero contact-point
/// velocity — and the multibody only reacts through the solve-side
/// jacobian writeback, exactly like Bullet's classic `CollideSDF_RS`).
#[derive(Clone, Debug)]
pub struct RContact {
    /// Contact info.
    pub cti: Cti,
    /// Owner node index.
    pub node: usize,
    /// Impulse matrix `ImpulseMatrix(sdt, ima, imb, invI, ra)`.
    pub c0: Matrix3,
    /// Relative anchor `ra = x - COM`.
    pub c1: Vector3,
    /// `ima * sdt`.
    pub c2: Real,
    /// Friction factor: 0 while inside the friction cone, `1 - fc` outside.
    pub c3: Real,
    /// Contact hardness: `k_khr` against static, `k_chr` against dynamic.
    pub c4: Real,
}

/// A node pinned to a rigid body (Bullet's `Anchor`).
#[derive(Clone, Debug)]
pub struct Anchor {
    /// Pinned node index.
    pub node: usize,
    /// Rigid body index.
    pub body: usize,
    /// Anchor position in body space.
    pub local: Vector3,
    /// Influence weight [0,1].
    pub influence: Real,
    /// Impulse matrix (assembled per solve).
    pub c0: Matrix3,
    /// Relative anchor `R * local`.
    pub c1: Vector3,
    /// `sdt * node.im`.
    pub c2: Real,
}

/// A node-vs-face contact between two soft bodies (Bullet's `SContact`).
#[derive(Clone, Copy, Debug)]
pub struct SContact {
    /// Node index on the node-side body.
    pub node: usize,
    /// Face index on the face-side body (the body itself is tracked by the
    /// world's pair grouping, mirroring Bullet's `m_face` pointer).
    pub face_body: usize,
    /// Face index.
    pub face: usize,
    /// Barycentric weights on the face.
    pub weights: Vector3,
    /// Contact normal (from the face surface toward the node).
    pub normal: Vector3,
    /// Effective margin (both margins plus the node's step travel).
    pub margin: Real,
    /// Friction coefficient (`max` of both bodies' `kDF`).
    pub friction: Real,
    /// Node-side / face-side response weights (`ma/ms * kSHR`).
    pub cfm: [Real; 2],
}

/// Skew (cross-product) matrix of `v`.
#[inline]
fn cross_matrix(v: Vector3) -> Matrix3 {
    Matrix3::from_cols(
        Vector3::new(0.0, -v.z, v.y),
        Vector3::new(v.z, 0.0, -v.x),
        Vector3::new(-v.y, v.x, 0.0),
    )
}

/// `Mat3A::from_diagonal` takes the narrow `Vec3` (f32) while `DMat3`'s
/// takes `DVec3` directly (f64); pick the right conversion per feature.
#[inline]
fn diagonal(v: Vector3) -> Matrix3 {
    #[cfg(feature = "f64")]
    {
        Matrix3::from_diagonal(v)
    }
    #[cfg(not(feature = "f64"))]
    {
        Matrix3::from_diagonal(v.into())
    }
}

/// Bullet's `MassMatrix`: `Diag(im) - Cross(r) * iwi * Cross(r)`.
#[inline]
pub(crate) fn mass_matrix(im: Real, iwi: &Matrix3, r: Vector3) -> Matrix3 {
    let cr = cross_matrix(r);
    diagonal(Vector3::splat(im)) - cr * *iwi * cr
}

/// Bullet's `ImpulseMatrix(dt, ima, imb, iwi, r)`:
/// `Diag(1/dt) * (Diag(ima) + MassMatrix(imb, iwi, r))^-1`.
#[inline]
pub(crate) fn impulse_matrix(dt: Real, ima: Real, imb: Real, iwi: &Matrix3, r: Vector3) -> Matrix3 {
    let inv = (mass_matrix(imb, iwi, r) + diagonal(Vector3::splat(ima))).inverse();
    diagonal(Vector3::splat(1.0 / dt)) * inv
}

/// The zero-radius sphere standing in for a query point in the GJK walk
/// (Bullet's `btGjkEpaSolver2::SignedDistance(point, 0, shape, ...)`).
fn point_shape_distance(
    shape: &CollisionShape,
    position: Vector3,
    orientation: Quaternion,
    x: Vector3,
    margin: Real,
) -> Option<(Vector3, Real)> {
    match shape {
        CollisionShape::Plane(plane) => {
            // Analytic: distance = n . x - c along the (unit) normal.
            let distance = plane.normal.dot(x) - plane.constant;
            // The plane normal is the outward direction above the surface.
            let dst = distance - margin;
            (dst < 0.0).then(|| {
                let normal = if distance >= 0.0 { plane.normal } else { -plane.normal };
                (normal, dst)
            })
        }
        CollisionShape::TriangleMesh(mesh) => point_mesh_distance(mesh, position, orientation, x, margin),
        CollisionShape::Heightfield(h) => {
            point_mesh_distance(h.as_mesh(), position, orientation, x, margin)
        }
        _ => point_convex_distance(shape, position, orientation, x, margin),
    }
}

/// Exact closest-point-on-triangle over the mesh BVH, in mesh-local space
/// (Bullet instead feeds each triangle as an extruded hull through the SDF
/// route — recorded deviation).
fn point_mesh_distance(
    mesh: &TriangleMesh,
    position: Vector3,
    orientation: Quaternion,
    x: Vector3,
    margin: Real,
) -> Option<(Vector3, Real)> {
    let xl = orientation.inverse() * (x - position);
    let m = Vector3::splat(margin);
    let query = Aabb::new(xl - m, xl + m);
    let mut best: Option<(Real, Vector3)> = None;
    for t in mesh.triangles_in_aabb(&query) {
        let (a, b, c) = mesh.triangle(t as usize);
        let (d2, dir) = point_triangle_distance(xl, a, b, c);
        if best.is_none_or(|(bd2, _)| d2 < bd2) {
            best = Some((d2, dir));
        }
    }
    let (d2, dir) = best?;
    let distance = d2.sqrt();
    let dst = distance - margin;
    if dst >= 0.0 {
        return None;
    }
    // `dir` points from the triangle surface toward the point — the outward
    // normal; when the point sits exactly on the surface, fall back to +Y.
    let len = dir.length();
    let normal_local = if len > 1e-12 { dir / len } else { Vector3::Y };
    Some((orientation * normal_local, dst))
}

/// GJK distance walk of a zero-radius sphere (the query point) against a
/// convex shape (Bullet's
/// `btGjkEpaSolver2::SignedDistance(point, 0, shape, ...)`).
fn point_convex_distance(
    shape: &CollisionShape,
    position: Vector3,
    orientation: Quaternion,
    x: Vector3,
    margin: Real,
) -> Option<(Vector3, Real)> {
    let point = CollisionShape::sphere(0.0);
    match gjk_contact(
        shape,
        position,
        orientation,
        &point,
        x,
        Quaternion::IDENTITY,
        None,
        margin,
    ) {
        // gjk's normal points A->B with A = the rigid shape, i.e. outward
        // from the surface toward the node; `Band.distance` is the true
        // signed surface separation (negative inside).
        GjkOutcome::Band { distance, normal, .. } => Some((normal, distance - margin)),
        // Deep overlap past the margin shell: the node is inside the shape.
        // The walk cannot give the direction there; use the direction from
        // the shape center (adequate for the rare deep-insertion case; the
        // solver pushes the node out along it regardless of exactness).
        GjkOutcome::Overlap(_) => {
            let dir = x - position;
            let len = dir.length();
            if len > 1e-12 {
                Some((dir / len, -margin))
            } else {
                Some((Vector3::Y, -margin))
            }
        }
        GjkOutcome::Separated => None,
    }
}

impl SoftBody {
    /// SDF_RS collision against one rigid body (Bullet's
    /// `CollideSDF_RS::DoNode` fed by `m_ndbvt.collideTV`): every node whose
    /// dbvt leaf overlaps the body's margin-padded AABB is distance-tested
    /// and, on contact, appended to `rcontacts` with its impulse row.
    ///
    /// `scratch` is a caller-recycled buffer for the overlapping leaves.
    pub fn collide_rigid(&mut self, body_idx: usize, body: &RigidBody, scratch: &mut Vec<usize>) {
        self.collide_body(
            ContactOwner::Rigid(body_idx),
            &body.shape,
            body.position,
            body.orientation,
            body.inv_mass(),
            &body.inv_inertia_world(),
            |ra| body.velocity_at_local_point(ra),
            body.friction,
            body.is_static(),
            scratch,
        );
    }

    /// SDF_RS collision against one multibody link collider (phase F).        
    /// The classic-system assembly is Bullet's exact behavior for a
    /// `CO_FEATHERSTONE_LINK` colObj through `CollideSDF_RS`: the rigid       
    /// upcast fails, so the impulse row is built kinematic (`imb = 0`, zero   
    /// inertia, zero contact-point velocity — the node solves against an      
    /// infinitely heavy partner) and the multibody only receives its
    /// reaction through the solve-side jacobian writeback. The anchor
    /// `ra = x - collider_origin` feeds the tangent directions.
    #[allow(clippy::too_many_arguments)]
    pub fn collide_multibody(
        &mut self,
        mb: usize,
        link: i32,
        shape: &CollisionShape,
        position: Vector3,
        orientation: Quaternion,
        friction: Real,
        scratch: &mut Vec<usize>,
    ) {
        self.collide_body(
            ContactOwner::Link { mb, link },
            shape,
            position,
            orientation,
            0.0,
            &Matrix3::ZERO,
            |_| Vector3::ZERO,
            friction,
            false,
            scratch,
        );
    }

    /// The shared `CollideSDF_RS::DoNode` kernel.
    #[allow(clippy::too_many_arguments)]
    fn collide_body(
        &mut self,
        owner: ContactOwner,
        shape: &CollisionShape,
        position: Vector3,
        orientation: Quaternion,
        imb: Real,
        inv_inertia: &Matrix3,
        velocity_at: impl Fn(Vector3) -> Vector3,
        friction: Real,
        is_static: bool,
        scratch: &mut Vec<usize>,
    ) {
        if !self.cfg.collisions.contains(SoftCollision::SDF_RS) {
            return;
        }
        let basemargin = self.margin;
        // Bullet's `timemargin` (interpolation delta) is dead code upstream
        // (both transforms are the same object), so dynmargin == stamargin.
        let volume = shape.world_aabb(position, orientation).expanded(basemargin);
        scratch.clear();
        self.ndbvt.for_each_overlapping(&volume, scratch);

        let sdt = self.sst.sdt;
        for &ni in &*scratch {
            let (x, q, ima) = {
                let n = &self.nodes[ni];
                (n.x, n.q, n.im)
            };
            if self.nodes[ni].battach {
                continue;
            }
            // Bullet picks dynmargin (basemargin + a dead-code interpolation
            // delta upstream) for dynamic nodes and stamargin for pinned
            // ones; both equal basemargin here.
            let m = basemargin;
            let Some((normal, dst)) =
                point_shape_distance(shape, position, orientation, x, m)
            else {
                continue;
            };
            if ima + imb <= 0.0 {
                continue;
            }
            let ra = x - position;
            let va = velocity_at(ra) * sdt;
            let vb = x - q;
            let vr = vb - va;
            let dn = vr.dot(normal);
            let fv = vr - normal * dn;
            let fc = self.cfg.k_df * friction;
            let c3 = if fv.length_squared() < (dn * fc) * (dn * fc) {
                0.0
            } else {
                1.0 - fc
            };
            let c4 = if is_static {
                self.cfg.k_khr
            } else {
                self.cfg.k_chr
            };
            let c0 = impulse_matrix(sdt, ima, imb, inv_inertia, ra);
            self.rcontacts.push(RContact {
                cti: Cti {
                    body: owner,
                    normal,
                    offset: -normal.dot(x - normal * dst),
                },
                node: ni,
                c0,
                c1: ra,
                c2: ima * sdt,
                c3,
                c4,
            });
        }
    }
}

/// Soft-vs-soft collision for one ordered pair (Bullet's
/// `defaultCollisionHandler(psb)` `VF_SS` branch): the node-side body's
/// node tree walks the face-side body's face tree; each overlapping
/// (node, face) pair closer than the combined margin plus the node's step
/// travel lands in the node-side body's `scontacts` (solved at stage 7s).
///
/// Bullet runs this twice per broad-phase pair (A's nodes vs B's faces,
/// then B's nodes vs A's faces) — the world does the same. The upstream
/// "skip faces containing the node" loop is a no-op (it `continue`s the
/// wrong scope) and is ported as such: the face check is intentionally
/// absent, matching Bullet's effective behavior.
pub fn collide_vf_ss(node_side: &mut SoftBody, face_side: &mut SoftBody, face_body: usize, pairs: &mut Vec<(usize, usize)>) {
    let mrg = node_side.margin + face_side.margin;
    let k_df_node = node_side.cfg.k_df;
    let k_shr_node = node_side.cfg.k_shr;
    let k_shr_face = face_side.cfg.k_shr;
    let k_df_face = face_side.cfg.k_df;

    pairs.clear();
    node_side.ndbvt.for_each_overlapping_pair(&face_side.fdbvt, pairs);
    for &(ni, fi) in &*pairs {
        // Project the origin onto the triangle relative to the node.
        let o = node_side.nodes[ni].x;
        let q = node_side.nodes[ni].q;
        let fnodes = face_side.faces[fi].n;
        let mut p = Vector3::ZERO;
        let mut d = Real::INFINITY;
        project_origin_triangle(
            face_side.nodes[fnodes[0]].x - o,
            face_side.nodes[fnodes[1]].x - o,
            face_side.nodes[fnodes[2]].x - o,
            &mut p,
            &mut d,
        );
        let m = mrg + (o - q).length() * 2.0;
        if d >= m * m {
            continue;
        }
        let [xa, xb, xc] = [
            face_side.nodes[fnodes[0]].x,
            face_side.nodes[fnodes[1]].x,
            face_side.nodes[fnodes[2]].x,
        ];
        let w = bary_coord(xa, xb, xc, p + o);
        let ma = node_side.nodes[ni].im;
        let mut mb = bary_eval_real(
            face_side.nodes[fnodes[0]].im,
            face_side.nodes[fnodes[1]].im,
            face_side.nodes[fnodes[2]].im,
            w,
        );
        if face_side.nodes[fnodes[0]].im <= 0.0
            || face_side.nodes[fnodes[1]].im <= 0.0
            || face_side.nodes[fnodes[2]].im <= 0.0
        {
            mb = 0.0;
        }
        let ms = ma + mb;
        if ms <= 0.0 {
            continue;
        }
        node_side.scontacts.push(SContact {
            node: ni,
            face_body,
            face: fi,
            weights: w,
            normal: p / -d.sqrt(),
            margin: m,
            friction: k_df_node.max(k_df_face),
            cfm: [ma / ms * k_shr_node, mb / ms * k_shr_face],
        });
    }
}

impl SoftBody {
    /// Closest-hit ray cast against the surface faces (Bullet's
    /// `rayTest(rayFrom, rayTo, sRayCast)`): returns the hit face index and
    /// the segment fraction. The full-face scan matches Bullet's
    /// `bcountonly` path; the fdbvt pruning is a later perf item.
    pub fn ray_test(&self, ray_from: Vector3, ray_to: Vector3) -> Option<(usize, Real)> {
        let dir = ray_to - ray_from;
        let mut mint: Real = 1.0;
        let mut best: Option<usize> = None;
        for i in 0..self.faces.len() {
            let n = self.faces[i].n;
            let t = ray_from_to_triangle(
                ray_from,
                ray_to,
                dir,
                self.nodes[n[0]].x,
                self.nodes[n[1]].x,
                self.nodes[n[2]].x,
                mint,
            );
            if t > 0.0 {
                mint = t;
                best = Some(i);
            }
        }
        best.map(|i| (i, mint))
    }
}

/// Barycentric helper re-exported for the contact solver
/// (Bullet's `BaryEval` on positions).
pub fn bary_eval(a: Vector3, b: Vector3, c: Vector3, w: Vector3) -> Vector3 {
    bary_eval_vec3(a, b, c, w)
}

/// Build the implicit convex-hull shape of a cluster (Bullet's
/// `btSoftClusterCollisionShape`: the support of the node point set — no
/// hull is computed, the GJK walk sees the nodes' convex hull).
fn cluster_hull_shape(body: &SoftBody, cluster: &Cluster) -> CollisionShape {
    let pts: Vec<Vector3> = cluster.nodes.iter().map(|&ni| body.nodes[ni].x).collect();
    CollisionShape::convex_hull(pts)
}

/// One cluster contact result in Bullet's `ClusterBase::SolveContact`
/// convention: the normal points from B toward A, `distance` is the signed
/// surface separation (negative while penetrating) and the witnesses sit on
/// each surface.
struct ClusterDistance {
    /// From B toward A.
    normal: Vector3,
    /// Signed surface distance.
    distance: Real,
    /// Closest point on A.
    witness_a: Vector3,
    /// Closest point on B.
    witness_b: Vector3,
}

/// GJK/EPA distance between a cluster hull (A, world-space nodes at the
/// identity transform) and a convex rigid shape (B), with the witnesses and
/// normal in Bullet's convention (Bullet's `btGjkEpaSolver2::SignedDistance`
/// fed `(&cshape, identity, rshape, wtr, guess, res)`). Static meshes and
/// heightfields are not supported on this path (Bullet triangulates them
/// into per-triangle hulls upstream — recorded deviation); planes take the
/// exact analytic minimum (a convex hull's plane distance is its extreme
/// vertex, so this matches Bullet's triangulated route exactly).
fn cluster_vs_convex(
    hull: &CollisionShape,
    body: &RigidBody,
    margin: Real,
) -> Option<ClusterDistance> {
    if let CollisionShape::Plane(plane) = &body.shape {
        // Analytic: the closest hull point to the plane is the extreme
        // vertex along the plane normal.
        let mut best = Real::INFINITY;
        let mut witness_a = Vector3::ZERO;
        for p in hull_points(hull) {
            let d = plane.normal.dot(*p) - plane.constant;
            if d < best {
                best = d;
                witness_a = *p;
            }
        }
        let normal = if best >= 0.0 { plane.normal } else { -plane.normal };
        let witness_b = witness_a - normal * best;
        return Some(ClusterDistance { normal, distance: best, witness_a, witness_b });
    }
    // `threshold = margin` mirrors Bullet's `SolveContact` gate
    // (`res.distance < m_margin`): the band reports exactly the pairs the
    // joint would accept, deeper overlap falls through to EPA.
    let threshold = margin;
    match gjk_contact(
        &body.shape,
        body.position,
        body.orientation,
        hull,
        Vector3::ZERO,
        Quaternion::IDENTITY,
        Some(Vector3::X),
        threshold,
    ) {
        // Our GJK reports `normal` from A (= the rigid shape) toward B (=
        // the hull); Bullet's convention is B -> A.
        GjkOutcome::Band { distance, normal, point_on_b } => Some(ClusterDistance {
            normal: -normal,
            distance,
            witness_a: point_on_b + (-normal) * distance,
            witness_b: point_on_b,
        }),
        GjkOutcome::Overlap(simplex) => {
            let mut seed = [Vector3::ZERO; 4];
            seed.copy_from_slice(&simplex.points()[..4]);
            let support = |dir: Vector3| {
                core_support_minkowski(
                    &body.shape,
                    body.position,
                    body.orientation,
                    hull,
                    Vector3::ZERO,
                    Quaternion::IDENTITY,
                    dir,
                )
            };
            let (normal_a_to_b, core_depth) = epa::penetration(&seed, support);
            if core_depth <= 0.0 {
                return None;
            }
            let depth = core_depth + body.shape.margin() + hull.margin();
            let normal = -normal_a_to_b;
            Some(ClusterDistance {
                normal,
                distance: -depth,
                witness_a: hull.support_world(Vector3::ZERO, Quaternion::IDENTITY, normal),
                witness_b: body
                    .shape
                    .support_world(body.position, body.orientation, normal_a_to_b),
            })
        }
        GjkOutcome::Separated => None,
    }
}

/// GJK/EPA distance between two cluster hulls (Bullet's `SignedDistance`
/// with both shapes at the identity transform, guess = `comA - comB`).
fn cluster_vs_cluster(
    hull_a: &CollisionShape,
    hull_b: &CollisionShape,
    com_a: Vector3,
    com_b: Vector3,
    margin: Real,
) -> Option<ClusterDistance> {
    let guess = com_a - com_b;
    let threshold = margin;
    match gjk_contact(
        hull_a,
        Vector3::ZERO,
        Quaternion::IDENTITY,
        hull_b,
        Vector3::ZERO,
        Quaternion::IDENTITY,
        Some(guess),
        threshold,
    ) {
        GjkOutcome::Band { distance, normal, point_on_b } => Some(ClusterDistance {
            normal: -normal,
            distance,
            witness_a: point_on_b + (-normal) * distance,
            witness_b: point_on_b,
        }),
        GjkOutcome::Overlap(simplex) => {
            let mut seed = [Vector3::ZERO; 4];
            seed.copy_from_slice(&simplex.points()[..4]);
            let support = |dir: Vector3| {
                core_support_minkowski(
                    hull_a,
                    Vector3::ZERO,
                    Quaternion::IDENTITY,
                    hull_b,
                    Vector3::ZERO,
                    Quaternion::IDENTITY,
                    dir,
                )
            };
            let (normal_a_to_b, core_depth) = epa::penetration(&seed, support);
            if core_depth <= 0.0 {
                return None;
            }
            let depth = core_depth + hull_a.margin() + hull_b.margin();
            let normal = -normal_a_to_b;
            Some(ClusterDistance {
                normal,
                distance: -depth,
                witness_a: hull_a.support_world(Vector3::ZERO, Quaternion::IDENTITY, normal),
                witness_b: hull_b.support_world(Vector3::ZERO, Quaternion::IDENTITY, normal_a_to_b),
            })
        }
        GjkOutcome::Separated => None,
    }
}

/// The hull point set (only planes need it; every other shape goes through
/// the GJK support).
fn hull_points(hull: &CollisionShape) -> &[Vector3] {
    match hull {
        CollisionShape::ConvexHull(h) => &h.points,
        _ => &[],
    }
}

/// Assemble a contact joint from a solved cluster distance (Bullet's
/// `ClusterBase::SolveContact` verbatim: the friction flag, the depth drift
/// and the two-sided impulse matrix).
#[allow(clippy::too_many_arguments)]
fn solve_cluster_contact(
    res: ClusterDistance,
    side_a: Side,
    origin_a: Vector3,
    velocity_a: Vector3,
    inv_mass_a: Real,
    inv_inertia_a: Matrix3,
    side_b: Side,
    origin_b: Vector3,
    velocity_b: Vector3,
    inv_mass_b: Real,
    inv_inertia_b: Matrix3,
    margin: Real,
    friction: Real,
) -> Option<CJoint> {
    if res.distance >= margin {
        return None;
    }
    let normal = res.normal;
    let ra = res.witness_a - origin_a;
    let rb = res.witness_b - origin_b;
    let vrel = velocity_a - velocity_b;
    let rvac = vrel.dot(normal);
    let depth = res.distance - margin;
    let iv = normal * rvac;
    let fv = vrel - iv;
    let friction_flag = if fv.length_squared() < rvac * friction * rvac * friction {
        1.0
    } else {
        friction
    };
    Some(CJoint {
        bodies: [side_a, side_b],
        rpos: [ra, rb],
        normal,
        friction: friction_flag,
        massmatrix: impulse_matrix_two(
            (inv_mass_a, inv_inertia_a, ra),
            (inv_mass_b, inv_inertia_b, rb),
        ),
        cfm: 1.0,
        erp: 1.0,
        split: 1.0,
        drift: normal * depth,
        sdrift: Vector3::ZERO,
        life: 0,
        maxlife: 0,
        delete: false,
    })
}

/// Process one cluster pair into a joint (shared by `CL_SS` and `CL_SELF`):
/// the GJK/EPA distance, the velocities at the witnesses and the joint
/// assembly, with the caller's erp/split multipliers already folded in.
#[allow(clippy::too_many_arguments)]
fn cluster_pair_joint(
    hull_a: &CollisionShape,
    cluster_a: &Cluster,
    hull_b: &CollisionShape,
    cluster_b: &Cluster,
    side_a: Side,
    side_b: Side,
    margin: Real,
    friction: Real,
    erp_mul: Real,
    split_mul: Real,
) -> Option<CJoint> {
    let res =
        cluster_vs_cluster(hull_a, hull_b, cluster_a.com, cluster_b.com, margin)?;
    let velocity_a = cluster_a.lv + cluster_a.av.cross(res.witness_a - cluster_a.com);
    let velocity_b = cluster_b.lv + cluster_b.av.cross(res.witness_b - cluster_b.com);
    let mut joint = solve_cluster_contact(
        res,
        side_a,
        cluster_a.com,
        velocity_a,
        cluster_a.imass,
        cluster_a.invwi,
        side_b,
        cluster_b.com,
        velocity_b,
        cluster_b.imass,
        cluster_b.invwi,
        margin,
        friction,
    )?;
    joint.erp *= erp_mul;
    joint.split *= split_mul;
    Some(joint)
}

impl SoftBody {
    /// CL_RS collision against one rigid body (Bullet's `CollideCL_RS`):
    /// every colliding cluster whose hull is closer to the rigid shape than
    /// the combined margins spawns a contact joint in `self.joints` (solved
    /// by the world's cluster pass). Anchored clusters skip static bodies;
    /// the friction is the `min` of both sides (Bullet's choice on this
    /// path, unlike SDF_RS's product).
    pub fn collide_cluster_rigid(
        &mut self,
        self_idx: usize,
        body_idx: usize,
        body: &RigidBody,
        scratch: &mut Vec<usize>,
    ) {
        if !self.cfg.collisions.contains(SoftCollision::CL_RS) {
            return;
        }
        let margin = body.shape.margin() + self.margin;
        let friction = self.cfg.k_df.min(body.friction);
        let volume = body
            .shape
            .world_aabb(body.position, body.orientation)
            .expanded(margin);
        scratch.clear();
        self.cdbvt.for_each_overlapping(&volume, scratch);
        let mut new_joints: Vec<SoftJoint> = Vec::new();
        for &ci in &*scratch {
            let cluster = &self.clusters[ci];
            // Anchored clusters do not collide with static bodies.
            if body.is_static() && cluster.contains_anchor {
                continue;
            }
            let hull = cluster_hull_shape(self, cluster);
            let Some(res) = cluster_vs_convex(&hull, body, margin) else {
                continue;
            };
            let velocity_a = cluster.lv + cluster.av.cross(res.witness_a - cluster.com);
            let velocity_b = body.velocity_at_local_point(res.witness_b - body.position);
            if let Some(mut joint) = solve_cluster_contact(
                res,
                Side::Cluster { soft: self_idx, cluster: ci },
                cluster.com,
                velocity_a,
                cluster.imass,
                cluster.invwi,
                Side::Rigid(body_idx),
                body.position,
                velocity_b,
                body.inv_mass(),
                body.inv_inertia_world(),
                margin,
                friction,
            ) {
                if body.is_static() {
                    joint.erp *= self.cfg.k_skhr_cl;
                    joint.split *= self.cfg.k_sk_splt_cl;
                } else {
                    joint.erp *= self.cfg.k_srhr_cl;
                    joint.split *= self.cfg.k_sr_splt_cl;
                }
                new_joints.push(SoftJoint::Contact(joint));
            }
        }
        self.joints.append(&mut new_joints);
    }

    /// CL_SS collision between two soft bodies' clusters (Bullet's
    /// `CollideCL_SS::ProcessSoftSoft`): one tree-tree pass over both
    /// cluster trees; every unconnected overlapping cluster pair closer
    /// than the combined margins spawns a joint on body A (`bodies[0]`).
    pub fn collide_cluster_soft(
        &mut self,
        a_idx: usize,
        other: &SoftBody,
        b_idx: usize,
        pairs: &mut Vec<(usize, usize)>,
    ) {
        let margin = self.margin + other.margin;
        let friction = self.cfg.k_df.min(other.cfg.k_df);
        pairs.clear();
        self.cdbvt.for_each_overlapping_pair(&other.cdbvt, pairs);
        let same_body = a_idx == b_idx;
        let n_clusters = self.clusters.len();
        let mut new_joints: Vec<SoftJoint> = Vec::new();
        for &(ca, cb) in &*pairs {
            // Connectivity skip: clusters sharing a node inside one body
            // never collide with each other.
            if same_body && !self.cluster_connectivity.is_empty() {
                let connected = self.cluster_connectivity[ca + n_clusters * cb];
                if connected {
                    continue;
                }
            }
            let cluster_a = &self.clusters[ca];
            let cluster_b = &other.clusters[cb];
            let hull_a = cluster_hull_shape(self, cluster_a);
            let hull_b = cluster_hull_shape(other, cluster_b);
            if let Some(joint) = cluster_pair_joint(
                &hull_a,
                cluster_a,
                &hull_b,
                cluster_b,
                Side::Cluster { soft: a_idx, cluster: ca },
                Side::Cluster { soft: b_idx, cluster: cb },
                margin,
                friction,
                self.cfg.k_sshr_cl.max(other.cfg.k_sshr_cl),
                (self.cfg.k_ss_splt_cl + other.cfg.k_ss_splt_cl) / 2.0,
            ) {
                new_joints.push(SoftJoint::Contact(joint));
            }
        }
        self.joints.append(&mut new_joints);
    }

    /// CL_SELF collision of one soft body with itself (Bullet's
    /// `defaultCollisionHandler(psb)` self loop: the `CL_SS` algorithm with
    /// both sides on this body). The tree-tree pass on the same tree visits
    /// each unordered leaf pair once (Bullet's `a == b` branch); connected
    /// clusters are skipped via the connectivity matrix.
    pub fn collide_cluster_self(&mut self, self_idx: usize, pairs: &mut Vec<(usize, usize)>) {
        let margin = self.margin + self.margin;
        let friction = self.cfg.k_df;
        pairs.clear();
        self.cdbvt.for_each_overlapping_pair(&self.cdbvt, pairs);
        let n_clusters = self.clusters.len();
        if n_clusters == 0 {
            return;
        }
        let mut new_joints: Vec<SoftJoint> = Vec::new();
        for &(ca, cb) in &*pairs {
            if !self.cluster_connectivity.is_empty()
                && self.cluster_connectivity[ca + n_clusters * cb]
            {
                continue;
            }
            let (lo, hi) = (ca.min(cb), ca.max(cb));
            let hull_lo = cluster_hull_shape(self, &self.clusters[lo]);
            let hull_hi = cluster_hull_shape(self, &self.clusters[hi]);
            if let Some(joint) = cluster_pair_joint(
                &hull_lo,
                &self.clusters[lo],
                &hull_hi,
                &self.clusters[hi],
                Side::Cluster { soft: self_idx, cluster: lo },
                Side::Cluster { soft: self_idx, cluster: hi },
                margin,
                friction,
                self.cfg.k_sshr_cl,
                self.cfg.k_ss_splt_cl,
            ) {
                new_joints.push(SoftJoint::Contact(joint));
            }
        }
        self.joints.append(&mut new_joints);
    }
}
