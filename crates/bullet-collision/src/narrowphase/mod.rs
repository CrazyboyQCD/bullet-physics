//! Narrow-phase collision: exact shape-pair tests and contact generation.

use crate::manifold::ContactManifold;
use bullet_math::{Quaternion, Vector3};
use crate::shapes::{CollisionShape, Compound};

mod box_box;
mod convex_convex;
mod convex_mesh;
mod convex_sweep;
pub mod epa;
pub mod gjk;
mod plane;
mod sphere;

pub use convex_sweep::{convex_cast, swept_sphere_cast, SweepHit};

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
    // Compounds are expanded into their children (recursive).
    if let CollisionShape::Compound(ca) = shape_a {
        return compound_collide(
            ca,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
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
        )
        .map(|m| m.flipped());
    }

    match (shape_a, shape_b) {
        (CollisionShape::Plane(_), CollisionShape::Plane(_)) => None,
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
            )
            .map(|m| m.flipped())
        }
        (CollisionShape::Heightfield(_), _) => None,
        (other, CollisionShape::TriangleMesh(mesh)) if other.is_convex() => {
            convex_mesh::convex_mesh(other, position_a, orientation_a, mesh, position_b, orientation_b)
        }
        (_, CollisionShape::TriangleMesh(_)) => None,
        (other, CollisionShape::Heightfield(hf)) if other.is_convex() => {
            convex_mesh::convex_mesh(other, position_a, orientation_a, hf.as_mesh(), position_b, orientation_b)
        }
        (_, CollisionShape::Heightfield(_)) => None,
        (CollisionShape::Plane(pa), _) => plane::plane_convex(
            pa,
            position_a,
            orientation_a,
            shape_b,
            position_b,
            orientation_b,
        ),
        (_, CollisionShape::Plane(pb)) => plane::plane_convex(
            pb,
            position_b,
            orientation_b,
            shape_a,
            position_a,
            orientation_a,
        )
        .map(|m| m.flipped()),
        // Dedicated fast paths.
        (CollisionShape::Sphere(sa), CollisionShape::Sphere(sb)) => {
            sphere::sphere_sphere(sa, position_a, sb, position_b)
        }
        (CollisionShape::Sphere(sa), CollisionShape::BoxShape(bb)) => {
            // sphere_box reports the normal from the box to the sphere; flip it
            // so it points from shape_a (sphere) to shape_b (box).
            sphere::sphere_box(sa, position_a, bb, position_b, orientation_b).map(|m| m.flipped())
        }
        (CollisionShape::BoxShape(ba), CollisionShape::Sphere(sb)) => {
            // sphere_box already reports the normal from the box (shape_a) to
            // the sphere (shape_b); no flip.
            sphere::sphere_box(sb, position_b, ba, position_a, orientation_a)
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
        ),
    }
}

/// Collide every child of a compound against `other`, merging the manifolds.
fn compound_collide(
    compound: &Compound,
    position: Vector3,
    orientation: Quaternion,
    other: &CollisionShape,
    other_position: Vector3,
    other_orientation: Quaternion,
) -> Option<ContactManifold> {
    let mut manifold = ContactManifold::new();
    for child in &compound.children {
        let child_pos = position + orientation.mul_vec3(child.position);
        let child_ori = orientation * child.orientation;
        if let Some(m) = collide(
            &child.shape,
            child_pos,
            child_ori,
            other,
            other_position,
            other_orientation,
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
        let m = collide_pair(&hf, Vector3::ZERO, &box_shape, Vector3::new(1.0, 0.4, 1.0)).unwrap();
        let p = m.points[0];
        assert_relative_eq!(p.normal.y, 1.0, epsilon = 1e-3);
        assert_relative_eq!(p.penetration, 0.1, epsilon = 1e-3);
    }
}
