//! Expanding Polytope Algorithm (EPA): penetration depth from a GJK simplex.
//!
//! Given a GJK simplex that encloses the origin of the Minkowski difference
//! `A - B`, EPA iteratively expands the polytope toward the origin and returns
//! the penetration normal (pointing from A to B) and depth.

use bullet_math::{Real, Vector3};

const MAX_ITERATIONS: u32 = 64;
const EPSILON: Real = 1e-6;

struct Face {
    a: usize,
    b: usize,
    c: usize,
    normal: Vector3,
    distance: Real,
}

/// Compute penetration `(normal, depth)` from a GJK simplex enclosing the
/// origin. `support` is the Minkowski-difference support function.
pub fn penetration(
    simplex: &[Vector3; 4],
    support: impl Fn(Vector3) -> Vector3,
) -> (Vector3, Real) {
    let mut vertices: Vec<Vector3> = simplex.to_vec();
    let mut faces = build_initial_faces(simplex);

    for _ in 0..MAX_ITERATIONS {
        let (min_idx, min_normal, min_dist) = faces
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.distance
                    .partial_cmp(&b.1.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, f)| (i, f.normal, f.distance))
            .unwrap();

        let new_point = support(min_normal);
        let new_dist = min_normal.dot(new_point);

        if new_dist - min_dist < EPSILON {
            // Converged: the closest face bounds the polytope.
            return (min_normal, min_dist);
        }

        // Remove the face and rebuild the hull around the new point.
        let new_idx = vertices.len();
        vertices.push(new_point);
        expand(&mut faces, &vertices, new_point, new_idx);
        let _ = min_idx;
    }

    let best = faces
        .iter()
        .min_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    (best.normal, best.distance)
}

fn build_initial_faces(simplex: &[Vector3; 4]) -> Vec<Face> {
    // (a, b, c, opposite) for the four triangular faces of the tetrahedron.
    let quads = [(0usize, 1usize, 2usize, 3usize), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)];
    quads
        .iter()
        .map(|&(a, b, c, opp)| {
            let mut normal = (simplex[b] - simplex[a]).cross(simplex[c] - simplex[a]);
            // Orient the normal outward (away from the opposite vertex, which
            // lies inside the tetrahedron).
            if normal.dot(simplex[opp] - simplex[a]) > 0.0 {
                normal = -normal;
            }
            normal = normal.normalize();
            let distance = normal.dot(simplex[a]);
            Face {
                a,
                b,
                c,
                normal,
                distance,
            }
        })
        .collect()
}

fn expand(faces: &mut Vec<Face>, vertices: &[Vector3], new_point: Vector3, new_idx: usize) {
    // Remove faces visible from the new point, tracking the boundary loop.
    let mut boundary: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    faces.retain(|f| {
        let visible = f.normal.dot(new_point) > f.distance;
        if visible {
            for (u, v) in [(f.a, f.b), (f.b, f.c), (f.c, f.a)] {
                if boundary.contains(&(v, u)) {
                    boundary.remove(&(v, u));
                } else {
                    boundary.insert((u, v));
                }
            }
        }
        !visible
    });

    for (u, v) in boundary {
        let cross = (vertices[v] - vertices[u]).cross(new_point - vertices[u]);
        let len = cross.length();
        if len < 1e-12 {
            // Degenerate edge (collinear with the new point): skip.
            continue;
        }
        let mut normal = cross / len;
        if normal.dot(vertices[u]) < 0.0 {
            normal = -normal;
        }
        let distance = normal.dot(vertices[u]);
        faces.push(Face {
            a: u,
            b: v,
            c: new_idx,
            normal,
            distance,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowphase::gjk;
    use bullet_math::Quaternion;
    use crate::shapes::CollisionShape;
    use approx::assert_relative_eq;

    #[test]
    fn epa_penetration_spheres() {
        let a = CollisionShape::sphere(1.0);
        let b = CollisionShape::sphere(1.0);
        let pa = Vector3::ZERO;
        // Oblique overlap (not axis-aligned) to avoid degenerate simplices.
        let pb = Vector3::new(1.0, 0.5, 0.3);

        let simplex = gjk::gjk_intersect(
            &a,
            pa,
            Quaternion::IDENTITY,
            &b,
            pb,
            Quaternion::IDENTITY,
        )
        .expect("spheres overlap");
        let mut sp = [Vector3::ZERO; 4];
        for (i, v) in simplex.p.iter().take(4).enumerate() {
            sp[i] = *v;
        }

        let support = |dir: Vector3| {
            gjk::support_minkowski(
                &a,
                pa,
                Quaternion::IDENTITY,
                &b,
                pb,
                Quaternion::IDENTITY,
                dir,
            )
        };
        let (normal, depth) = penetration(&sp, support);
        let expected = 2.0 - pb.length();
        assert_relative_eq!(depth, expected, epsilon = 1e-2);
        // Normal should point from A to B (i.e. along `pb`).
        assert!(normal.dot(pb.normalize()) > 0.9);
    }
}
