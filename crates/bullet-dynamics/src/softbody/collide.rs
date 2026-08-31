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

use bullet_collision::narrowphase::gjk::{gjk_contact, GjkOutcome};
use bullet_collision::narrowphase::point_triangle_distance;
use bullet_collision::shapes::CollisionShape;
use bullet_collision::shapes::TriangleMesh;
use bullet_math::{Aabb, Matrix3, Quaternion, Real, Vector3};

use crate::body::RigidBody;

use super::body::SoftBody;
use super::config::SoftCollision;

/// Contact info against one rigid body (Bullet's `sCti`).
#[derive(Clone, Copy, Debug)]
pub struct Cti {
    /// Rigid body index.
    pub body: usize,
    /// Outward surface normal (body surface toward the node, world space).
    pub normal: Vector3,
    /// Plane offset such that `dot(x, normal) + offset` equals the
    /// detection-time signed distance minus the margin (Bullet's
    /// `m_offset = -dot(n, x - n*dst)`).
    pub offset: Real,
}

/// One node-vs-rigid contact with its precomputed impulse-matrix row
/// (Bullet's `RContact`; the multibody jacobian fields arrive with phase F).
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
fn mass_matrix(im: Real, iwi: &Matrix3, r: Vector3) -> Matrix3 {
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
        if !self.cfg.collisions.contains(SoftCollision::SDF_RS) {
            return;
        }
        let basemargin = self.margin;
        // Bullet's `timemargin` (interpolation delta) is dead code upstream
        // (both transforms are the same object), so dynmargin == stamargin.
        let volume = body
            .shape
            .world_aabb(body.position, body.orientation)
            .expanded(basemargin);
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
                point_shape_distance(&body.shape, body.position, body.orientation, x, m)
            else {
                continue;
            };
            let imb = body.inv_mass();
            if ima + imb <= 0.0 {
                continue;
            }
            let ra = x - body.position;
            let va = body.velocity_at_local_point(ra) * sdt;
            let vb = x - q;
            let vr = vb - va;
            let dn = vr.dot(normal);
            let fv = vr - normal * dn;
            let fc = self.cfg.k_df * body.friction;
            let c3 = if fv.length_squared() < (dn * fc) * (dn * fc) {
                0.0
            } else {
                1.0 - fc
            };
            let c4 = if body.is_static() {
                self.cfg.k_khr
            } else {
                self.cfg.k_chr
            };
            let c0 = impulse_matrix(sdt, ima, imb, &body.inv_inertia_world(), ra);
            self.rcontacts.push(RContact {
                cti: Cti {
                    body: body_idx,
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
