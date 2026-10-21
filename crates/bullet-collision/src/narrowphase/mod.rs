//! Narrow-phase collision: exact shape-pair tests and contact generation.

pub use epa::EpaScratch;

use crate::manifold::ContactManifold;
use bullet_math::{Quaternion, Real, Vector3};
use crate::shapes::{CollisionShape, Compound};

mod box_box;
mod convex_convex;
mod convex_mesh;
mod convex_sweep;
pub mod epa;
pub mod gjk;
mod mesh_mesh;
mod plane;
mod sphere;

pub use convex_sweep::{convex_cast, point_triangle_distance, swept_sphere_cast, SweepHit};

/// Detect collision between two shapes and generate a contact manifold.
///
/// `normal` of each contact point points from `shape_a` to `shape_b`.
/// Returns `None` when the shapes do not touch.
pub fn collide(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
) -> Option<ContactManifold> {
    // Convenience path: a fresh EPA scratch per call. Detection loops
    // should thread one shared scratch through `collide_seeded` instead
    // (steady-state calls then allocate nothing).
    let mut scratch = epa::EpaScratch::default();
    collide_seeded(
        shape_a,
        position_a,
        orientation_a,
        shape_b,
        position_b,
        orientation_b,
        None,
        &mut scratch,
    )
}

/// The contact band of a shape pair: contacts are generated while the
/// surfaces stay separated by less than this (Bullet's dispatcher takes the
/// minimum of the two shapes' breaking thresholds).
#[inline]
fn pair_threshold(shape_a: &CollisionShape, shape_b: &CollisionShape) -> Real {
    shape_a
        .contact_breaking_threshold()
        .min(shape_b.contact_breaking_threshold())
}

/// [`collide`] with an optional GJK warm-start seed (a cached separating
/// normal from a previous step of the same pair, forwarded to the generic
/// convex paths; the analytic fast paths ignore it).
#[allow(clippy::too_many_arguments)]
pub fn collide_seeded(
    shape_a: &CollisionShape,
    position_a: Vector3,
    orientation_a: Quaternion,
    shape_b: &CollisionShape,
    position_b: Vector3,
    orientation_b: Quaternion,
    seed: Option<Vector3>,
    scratch: &mut EpaScratch,
) -> Option<ContactManifold> {
    let threshold = pair_threshold(shape_a, shape_b);
    // Compounds are expanded into their children (recursive; each child pair   
    // computes its own threshold like Bullet's per-child collision
    // algorithms).
    if let CollisionShape::Compound(ca) = shape_a {
        return compound_collide(
            ca,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
            seed,
            scratch,
        );
    }
    if let CollisionShape::Compound(cb) = shape_b {
        return compound_collide(
            cb,
            position_b,
            orientation_b,
            shape_a,
            position_a,
            orientation_a,
            seed,
            scratch,
        )
        .map(|m| m.flipped());
    }

    match (shape_a, shape_b) {
        (CollisionShape::Plane(_), CollisionShape::Plane(_)) => None,
        // Mesh-vs-mesh (Bullet's `gimpact_vs_gimpact`): tree-vs-tree
        // candidate pairs plus per-triangle GJK. Covers dynamic-vs-dynamic
        // and dynamic-vs-static-mesh pairs; heightfields join through their
        // internal mesh. Two static heightfields never move and stay `None`.
        (CollisionShape::TriangleMesh(ma), CollisionShape::TriangleMesh(mb)) => {
            mesh_mesh::mesh_mesh(
                ma, position_a, orientation_a, mb, position_b, orientation_b, seed, threshold,
                scratch,
            )
        }
        (CollisionShape::TriangleMesh(ma), CollisionShape::Heightfield(hb)) => {
            mesh_mesh::mesh_mesh(
                ma,
                position_a,
                orientation_a,
                hb.as_mesh(),
                position_b,
                orientation_b,
                seed,
                threshold,
                scratch,
            )
        }
        (CollisionShape::Heightfield(ha), CollisionShape::TriangleMesh(mb)) => mesh_mesh::mesh_mesh(
            ha.as_mesh(),
            position_a,
            orientation_a,
            mb,
            position_b,
            orientation_b,
            seed,
            threshold,
            scratch,
        ),
        // Plane-vs-mesh (Bullet's `gimpacttrimeshpart_vs_plane_collision`):
        // per-vertex plane distances. These must precede the generic
        // plane-convex arm — a mesh has no support point to offer it.
        (CollisionShape::Plane(pa), CollisionShape::TriangleMesh(mb)) => plane::plane_mesh(
            pa, position_a, orientation_a, mb, position_b, orientation_b, threshold,
        ),
        (CollisionShape::Plane(pa), CollisionShape::Heightfield(hb)) => plane::plane_mesh(
            pa,
            position_a,
            orientation_a,
            hb.as_mesh(),
            position_b,
            orientation_b,
            threshold,
        ),
        (CollisionShape::TriangleMesh(mb), CollisionShape::Plane(pa)) => plane::plane_mesh(
            pa,
            position_b,
            orientation_b,
            mb,
            position_a,
            orientation_a,
            threshold,
        )
        .map(|m| m.flipped()),
        (CollisionShape::Heightfield(hb), CollisionShape::Plane(pa)) => plane::plane_mesh(
            pa,
            position_b,
            orientation_b,
            hb.as_mesh(),
            position_a,
            orientation_a,
            threshold,
        )
        .map(|m| m.flipped()),
        // Concave meshes/heightfields collide only with convex shapes (the
        // compound case was expanded above). Concave-concave and plane-concave
        // pairs are unsupported and report no contacts.
        (CollisionShape::TriangleMesh(mesh), other) if other.is_convex() => {
            convex_mesh::convex_mesh(
                other,
                position_b,
                orientation_b,
                mesh,
                position_a,
                orientation_a,
                seed,
                threshold,
                scratch,
            )
            .map(|m| m.flipped())
        }
        (CollisionShape::TriangleMesh(_), _) => None,
        (CollisionShape::Heightfield(hf), other) if other.is_convex() => {
            convex_mesh::convex_mesh(
                other,
                position_b,
                orientation_b,
                hf.as_mesh(),
                position_a,
                orientation_a,
                seed,
                threshold,
                scratch,
            )
            .map(|m| m.flipped())
        }
        (CollisionShape::Heightfield(_), _) => None,
        (other, CollisionShape::TriangleMesh(mesh)) if other.is_convex() => {
            convex_mesh::convex_mesh(
                other,
                position_a,
                orientation_a,
                mesh,
                position_b,
                orientation_b,
                seed,
                threshold,
                scratch,
            )
        }
        (_, CollisionShape::TriangleMesh(_)) => None,
        (other, CollisionShape::Heightfield(hf)) if other.is_convex() => {
            convex_mesh::convex_mesh(
                other,
                position_a,
                orientation_a,
                hf.as_mesh(),
                position_b,
                orientation_b,
                seed,
                threshold,
                scratch,
            )
        }
        (_, CollisionShape::Heightfield(_)) => None,
        (CollisionShape::Plane(pa), _) => plane::plane_convex(
            pa,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
            threshold,
        ),
        (_, CollisionShape::Plane(pb)) => plane::plane_convex(
            pb,
            position_b,
            orientation_b,
            shape_a,
            position_a,
            orientation_a,
            threshold,
        )
        .map(|m| m.flipped()),
        // Dedicated fast paths.
        (CollisionShape::Sphere(sa), CollisionShape::Sphere(sb)) => {
            sphere::sphere_sphere(sa, position_a, sb, position_b)
        }
        (CollisionShape::Sphere(sa), CollisionShape::BoxShape(bb)) => {
            // sphere_box reports the normal from the box to the sphere; flip it
            // so it points from shape_a (sphere) to shape_b (box).
            sphere::sphere_box(sa, position_a, bb, position_b, orientation_b, threshold).map(|m| {
                m.flipped()
            })
        }
        (CollisionShape::BoxShape(ba), CollisionShape::Sphere(sb)) => {
            // sphere_box already reports the normal from the box (shape_a) to
            // the sphere (shape_b); no flip.
            sphere::sphere_box(sb, position_b, ba, position_a, orientation_a, threshold)
        }
        (CollisionShape::BoxShape(ba), CollisionShape::BoxShape(bb)) => {
            box_box::box_box(ba, position_a, orientation_a, bb, position_b, orientation_b)
        }
        // Everything else convex: generic GJK + EPA.
        _ => convex_convex::convex_convex(
            shape_a,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
            seed,
            threshold,
            scratch,
        ),
    }
}

/// Collide every child of a compound against `other`, merging the manifolds.   
#[allow(clippy::too_many_arguments)]
fn compound_collide(
    compound: &Compound,
    position: Vector3,
    orientation: Quaternion,
    other: &CollisionShape,
    other_position: Vector3,
    other_orientation: Quaternion,
    seed: Option<Vector3>,
    scratch: &mut EpaScratch,
) -> Option<ContactManifold> {
    let mut manifold = ContactManifold::new();
    for child in &compound.children {
        let child_pos = position + orientation * child.position;
        let child_ori = orientation * child.orientation;
        if let Some(m) = collide_seeded(
            &child.shape,
            child_pos,
            child_ori,
            other,
            other_position,
            other_orientation,
            seed,
            scratch,
        ) {
            manifold.points.extend(m.points);
        }
    }
    if manifold.is_empty() {
        None
    } else {
        Some(manifold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::CompoundChild;
    use approx::assert_relative_eq;

    fn collide_pair(
        a: &CollisionShape,
        pa: Vector3,
        b: &CollisionShape,
        pb: Vector3,
    ) -> Option<ContactManifold> {
        collide(a, pa, Quaternion::IDENTITY, b, pb, Quaternion::IDENTITY)
    }

    #[test]
    fn compound_contacts_through_children() {
        let compound = CollisionShape::compound(vec![
            CompoundChild::new(CollisionShape::sphere(0.5), Vector3::new(-1.0, 0.0, 0.0)),
            CompoundChild::new(CollisionShape::sphere(0.5), Vector3::new(1.0, 0.0, 0.0)),
        ]);
        // Sphere approaches the +X child along X; center distance 0.3 against
        // radius sum 1.0 means penetration 0.7. The normal points from the
        // compound (A) to the sphere (B), i.e. along +X.
        let m = collide_pair(&compound, Vector3::ZERO, &CollisionShape::sphere(0.5), Vector3::new(1.3, 0.0, 0.0))
            .unwrap();
        assert_eq!(m.points.len(), 1);
        let p = m.points[0];
        assert_relative_eq!(p.normal.x, 1.0, epsilon = 1e-4);
        assert_relative_eq!(p.penetration, 0.7, epsilon = 1e-3);
    }

    #[test]
    fn compound_with_a_plane_child() {
        let compound = CollisionShape::compound(vec![CompoundChild::new(
            CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        )]);
        let sphere = CollisionShape::sphere(0.5);
        // Sphere sinks into the embedded plane by 0.1.
        let m = collide_pair(&compound, Vector3::ZERO, &sphere, Vector3::new(0.0, 0.4, 0.0))
            .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn plane_vs_box_through_collide() {
        let plane = CollisionShape::plane(Vector3::Y, 0.0);
        let box_shape = CollisionShape::box_shape(Vector3::splat(1.0));
        // Box half-extent 1, center at y = 0.5: penetration 0.5.
        let m = collide_pair(&plane, Vector3::ZERO, &box_shape, Vector3::new(0.0, 0.5, 0.0))
            .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0);
        assert_relative_eq!(p.penetration, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn rotated_capsule_on_plane() {
        // Capsule lying on its side (axis along X) resting on a plane.
        let plane = CollisionShape::plane(Vector3::Y, 0.0);
        let capsule = CollisionShape::capsule(0.5, 1.0);
        let q = Quaternion::from_rotation_z(bullet_math::FRAC_PI_2);
        let m = collide(
            &plane,
            Vector3::ZERO,
            Quaternion::IDENTITY,
            &capsule,
            Vector3::new(0.0, 0.4, 0.0),
            q,
        )
        .unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn sphere_vs_mesh_through_collide() {
        let mesh = CollisionShape::triangle_mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let sphere = CollisionShape::sphere(0.5);
        // Normal points from the sphere (shape_a) toward the mesh: -Y.
        let m = collide_pair(&sphere, Vector3::new(0.5, 0.4, 0.5), &mesh, Vector3::ZERO).unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-3);
    }

    #[test]
    fn mesh_vs_sphere_flips_the_normal() {
        let mesh = CollisionShape::triangle_mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let sphere = CollisionShape::sphere(0.5);
        // Normal points from the mesh (shape_a) toward the sphere: +Y.
        let m = collide_pair(&mesh, Vector3::ZERO, &sphere, Vector3::new(0.5, 0.4, 0.5)).unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-3);
    }

    #[test]
    fn heightfield_vs_box_through_collide() {
        let hf = CollisionShape::heightfield(vec![0.0; 9], 3, 3, Vector3::ONE);
        let box_shape = CollisionShape::box_shape(Vector3::splat(0.5));
        // The centered 3x3 grid spans [-1, 1]^2; the box sits over an
        // interior flat cell.
        let m = collide_pair(&hf, Vector3::ZERO, &box_shape, Vector3::new(0.5, 0.4, 0.5)).unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-3);
    }

    fn unit_plate() -> CollisionShape {
        CollisionShape::triangle_mesh(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    /// Mesh-vs-mesh reaches the tree traversal through `collide` and reports
    /// the A-to-B normal with the margin-shell penetration.
    #[test]
    fn mesh_vs_mesh_through_collide() {
        let a = unit_plate();
        let b = unit_plate();
        let m = collide_pair(&a, Vector3::ZERO, &b, Vector3::new(0.0, 0.01, 0.0)).unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        // Margin-shell: band (0.1) minus the 0.01 separation.
        assert_relative_eq!(p.penetration, 0.09, epsilon = 1e-3);
    }

    /// Plane-vs-mesh reaches the per-vertex path through `collide` from both
    /// argument orders. The single-sided plane normal stays the plane's own
    /// (the solver pushes the mesh out of the plane along it), so the
    /// mesh-first order reports the flipped normal.
    #[test]
    fn plane_vs_mesh_through_collide() {
        let mesh = unit_plate();
        let plane = CollisionShape::plane(Vector3::Y, 0.0);
        let m = collide_pair(&plane, Vector3::ZERO, &mesh, Vector3::new(0.0, -0.1, 0.0)).unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-6);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-6);
        // Mesh first: flipped normal.
        let m = collide_pair(&mesh, Vector3::new(0.0, -0.1, 0.0), &plane, Vector3::ZERO).unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-6);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-6);
    }

    /// A dynamic mesh body over a static heightfield reaches the mesh-vs-mesh
    /// path through the heightfield's internal mesh.
    #[test]
    fn mesh_vs_heightfield_through_collide() {
        let hf = CollisionShape::heightfield(vec![0.0; 9], 3, 3, Vector3::ONE);
        let mesh = unit_plate();
        // The centered 3x3 grid spans [-1, 1]^2; the plate spans [0, 1]^2
        // over its +x/+z quadrant.
        let m = collide_pair(&mesh, Vector3::new(0.0, 0.01, 0.0), &hf, Vector3::ZERO).unwrap();
        let p = m.points[0];
        // A-to-B normal: from the mesh (above) toward the heightfield, -Y;
        // margin-shell penetration 0.1 - 0.01.
        assert_relative_eq!(p.normal.y, -1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.09, epsilon = 1e-2);
    }

    /// A GJK seed only changes where the simplex walk starts; the detected
    /// contact (normal, depth, position) must match the unseeded result.
    #[test]
    fn seeded_collide_matches_unseeded() {
        let pairs: Vec<(CollisionShape, CollisionShape, Vector3, Quaternion)> = vec![
            (
                CollisionShape::capsule(0.5, 1.0),
                CollisionShape::capsule(0.5, 1.0),
                Vector3::new(0.7, 0.2, 0.1),
                Quaternion::from_rotation_y(0.6),
            ),
            (
                CollisionShape::cylinder(1.0, 0.8),
                CollisionShape::sphere(0.5),
                Vector3::new(1.2, 0.3, -0.4),
                Quaternion::from_rotation_z(0.4),
            ),
            (
                CollisionShape::cone(1.0, 0.9),
                CollisionShape::box_shape(Vector3::splat(0.5)),
                Vector3::new(0.5, 1.1, 0.2),
                Quaternion::from_rotation_x(0.8),
            ),
        ];
        for (a, b, offset, q) in pairs {
            let unseeded = collide(&a, Vector3::ZERO, q, &b, offset, Quaternion::IDENTITY).unwrap();
            for seed in [
                unseeded.points[0].normal,
                -unseeded.points[0].normal,
                Vector3::new(0.3, -0.5, 0.8).normalize(),
            ] {
                let seeded = collide_seeded(
                    &a,
                    Vector3::ZERO,
                    q,
                    &b,
                    offset,
                    Quaternion::IDENTITY,
                    Some(seed),
                    &mut EpaScratch::default(),
                )
                .unwrap();
                assert_eq!(seeded.points.len(), unseeded.points.len());
                for (s, u) in seeded.points.iter().zip(&unseeded.points) {
                    // EPA accuracy floor (1e-4): different starting simplices
                    // converge to slightly different polytopes.
                    assert_relative_eq!(s.penetration, u.penetration, epsilon = 2e-4);
                    assert!((s.normal - u.normal).length() < 1e-4 || s.normal.dot(u.normal) > 0.9999);
                }
            }
        }
    }
}
