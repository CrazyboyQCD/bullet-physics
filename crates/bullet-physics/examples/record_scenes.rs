//! Rust-side scene recorder for the frame-by-frame parity harness.
//!
//! Reads the scene tables generated from tools/scenes.json
//! (generated/parity_scenes.rs), simulates one scene with the same protocol as the
//! C++ recorder (tools/bullet_record.cpp), and writes one CSV per recorded
//! body to the current working directory:
//!
//! ```text
//! <scene>_<body>.csv   (one line per frame: t px py pz qx qy qz qw, %.6f)
//! ```
//!
//! Output is diffed against the Bullet recording by tools/compare.py /
//! tools/parity.py (see docs/parity-harness.md).
//!
//! Usage: cargo run -p bullet-physics --example record_scenes -- [scene_name]

use std::fs::File;
use std::io::Write;

use bullet_physics::dynamics::body::RayFilter;
use bullet_physics::dynamics::{BodyHandle, CollisionFlags, PhysicsWorld, RigidBody};
use bullet_physics::math::{Real, Vector3};
use bullet_physics::shapes::{CollisionShape, TriangleMeshPart};

// Included via #[path] from generated/parity_scenes.rs (auto-generated from
// tools/scenes.json).
#[path = "../generated/parity_scenes.rs"]
mod parity_scenes;

use parity_scenes::{ShapeSpec, SCENES};

fn main() {
    let scene_name = std::env::args().nth(1).unwrap_or_else(|| "falling_box".to_string());
    let suffix = std::env::args().nth(2).unwrap_or_else(|| "rust".to_string());
    let Some((_, spec)) = SCENES.iter().find(|(n, _)| *n == scene_name) else {
        eprintln!("record_scenes: unknown scene '{scene_name}'");
        std::process::exit(2);
    };

    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(
        spec.gravity[0] as Real,
        spec.gravity[1] as Real,
        spec.gravity[2] as Real,
    ));
    world.solver_info_mut().num_iterations = spec.iterations;

    // Pair filter (optional): exclude the named pair from collision, same
    // one-layer gate (group & mask, then the exclusion) as the C++ recorder's
    // broadphase callback.
    if let Some((a_name, b_name)) = spec.pair_filter {
        let mut ia = None;
        let mut ib = None;
        for (i, b) in spec.bodies.iter().enumerate() {
            if b.name == a_name {
                ia = Some(i);
            }
            if b.name == b_name {
                ib = Some(i);
            }
        }
        let (Some(ia), Some(ib)) = (ia, ib) else {
            eprintln!("record_scenes: pair_filter body not found");
            std::process::exit(2);
        };
        let (lo, hi) = (
            BodyHandle::from_index(ia.min(ib)),
            BodyHandle::from_index(ia.max(ib)),
        );
        world.set_pair_filter(Some(Box::new(
            move |a: BodyHandle, _: &RigidBody, b: BodyHandle, _: &RigidBody| {
                !(a.min(b) == lo && a.max(b) == hi)
            },
        )));
    }

    // Build bodies; remember the handle of each body whose trajectory is
    // recorded (scene bodies are added in spec order, matching the C++ side).
    let mut record_idx: Vec<BodyHandle> = Vec::new();
    for b in spec.bodies.iter() {
        let shape = match b.shape {
            ShapeSpec::Box(h) => CollisionShape::box_shape(Vector3::new(
                h[0] as Real, h[1] as Real, h[2] as Real,
            )),
            ShapeSpec::Sphere(r) => CollisionShape::sphere(r as Real),
            ShapeSpec::Capsule(r, hh) => CollisionShape::capsule(r as Real, hh as Real),
            ShapeSpec::Cylinder(r, hh) => CollisionShape::cylinder(r as Real, hh as Real),
            ShapeSpec::Cone(r, hh) => CollisionShape::cone(r as Real, hh as Real),
            ShapeSpec::Plane([nx, ny, nz, c]) => {
                CollisionShape::plane(
                    Vector3::new(nx as Real, ny as Real, nz as Real),
                    c as Real,
                )
            }
            ShapeSpec::Mesh(verts, idx) => {
                let vertices: Vec<Vector3> = verts
                    .chunks(3)
                    .map(|c| Vector3::new(c[0] as Real, c[1] as Real, c[2] as Real))
                    .collect();
                let indices: Vec<[u32; 3]> = idx
                    .chunks(3)
                    .map(|t| [t[0], t[1], t[2]])
                    .collect();
                let mut mesh =
                    bullet_collision::shapes::TriangleMesh::new(vertices, indices);
                if b.internal_edges {
                    // Mesh-opt-in (btGenerateInternalEdgeInfo): edge-contact
                    // normals get the adjacent-triangle-aware adjustment.
                    mesh.generate_internal_edge_info();
                }
                CollisionShape::TriangleMesh(mesh)
            }
            ShapeSpec::UniformScaling(factor, kind, params) => {
                // Uniform wrapper (btUniformScalingShape): kind 0 = sphere
                // (params[0] = radius), kind 1 = box (params = half
                // extents); the child scales with the factor through the
                // dispatch layer.
                let child = if kind == 0 {
                    CollisionShape::sphere(params[0] as Real)
                } else {
                    CollisionShape::box_shape(Vector3::new(
                        params[0] as Real,
                        params[1] as Real,
                        params[2] as Real,
                    ))
                };
                CollisionShape::uniform_scaling(child, factor as Real)
            }
            ShapeSpec::ScaledMesh(verts, idx, scaling) => {
                // btScaledBvhTriangleMeshShape semantics, realized at
                // construction time (vertices scaled, BVH rebuilt).
                let vertices: Vec<Vector3> = verts
                    .chunks(3)
                    .map(|c| Vector3::new(c[0] as Real, c[1] as Real, c[2] as Real))
                    .collect();
                let indices: Vec<[u32; 3]> = idx
                    .chunks(3)
                    .map(|t| [t[0], t[1], t[2]])
                    .collect();
                let mesh =
                    bullet_collision::shapes::TriangleMesh::new(vertices, indices);
                CollisionShape::scaled_triangle_mesh(
                    mesh,
                    Vector3::new(
                        scaling[0] as Real,
                        scaling[1] as Real,
                        scaling[2] as Real,
                    ),
                )
            }
            ShapeSpec::MeshParts(parts) => {
                // Independent IndexedMesh buffers (seam vertices duplicated);
                // from_parts concatenates them into one triangle stream in
                // part order, mirroring btTriangleIndexVertexArray.
                let parts: Vec<TriangleMeshPart> = parts
                    .iter()
                    .map(|(verts, idx)| TriangleMeshPart {
                        vertices: verts
                            .chunks(3)
                            .map(|c| Vector3::new(c[0] as Real, c[1] as Real, c[2] as Real))
                            .collect(),
                        indices: idx.chunks(3).map(|t| [t[0], t[1], t[2]]).collect(),
                    })
                    .collect();
                CollisionShape::triangle_mesh_parts(parts)
            }
        };
        let position = Vector3::new(
            b.position[0] as Real, b.position[1] as Real, b.position[2] as Real,
        );
        let mut body = if b.is_static {
            RigidBody::new_static(shape, position)
        } else {
            RigidBody::new_dynamic(shape, position, b.mass as Real)
                .friction(b.friction as Real)
                .restitution(b.restitution as Real)
                .velocity(Vector3::new(
                    b.velocity[0] as Real, b.velocity[1] as Real, b.velocity[2] as Real,
                ))
        };
        body.rolling_friction = b.rolling_friction as Real;
        body.spinning_friction = b.spinning_friction as Real;
        body.set_damping(b.linear_damping as Real, b.angular_damping as Real);
        if b.additional_damping {
            // Construction-info-only opt-in in Bullet (no post-construction
            // setter) — mirrored by the builder.
            body = body.additional_damping(true);
        }
        body.anisotropic_friction = Vector3::new(
            b.anisotropic_friction[0] as Real,
            b.anisotropic_friction[1] as Real,
            b.anisotropic_friction[2] as Real,
        );
        if b.contact_stiffness_set {
            body.set_contact_stiffness_and_damping(
                b.contact_stiffness as Real,
                b.contact_damping as Real,
            );
        }
        if !b.is_static {
            body.angular_velocity = Vector3::new(
                b.angular_velocity[0] as Real,
                b.angular_velocity[1] as Real,
                b.angular_velocity[2] as Real,
            );
        }
        if !b.is_static && b.ccd_on {
            body = body.ccd(
                b.ccd_motion_threshold as Real,
                b.ccd_swept_sphere_radius as Real,
            );
        }
        // Consumer-API fields, set before add_body like the C++ recorder
        // (flags first so the add-time world-gravity sync respects them).
        if b.no_contact_response {
            body = body.no_contact_response(true);
        }
        if b.disable_world_gravity {
            body.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
            body.set_gravity(Vector3::new(
                b.body_gravity[0] as Real,
                b.body_gravity[1] as Real,
                b.body_gravity[2] as Real,
            ));
        }
        body = body.collision_filter(b.group, b.mask);
        let handle = world.add_body(body);
        if spec.record.contains(&b.name) {
            record_idx.push(handle);
        }
    }

    let dt: Real = spec.dt_num as Real / spec.dt_den as Real;

    // Open one output file per recorded body (cwd, like the C++ recorder).
    let mut out: Vec<File> = Vec::new();
    for r in spec.record {
        let fname = format!("{scene_name}_{r}_{suffix}.csv");
        out.push(File::create(&fname).unwrap_or_else(|e| {
            eprintln!("record_scenes: cannot open {fname}: {e}");
            std::process::exit(2);
        }));
    }

    for frame in 1..=spec.frames {
        // A scene-level substeps spec drives the fixed-timestep accumulator
        // (`step_simulation_fixed`); the default is exactly one dt-sized
        // step.
        if let Some(ss) = spec.substeps {
            world.step_simulation_fixed(dt, ss.max_sub_steps, ss.fixed_dt as Real);
        } else {
            world.step_simulation(dt);
        }
        let line = format!("{:.6}", frame as Real * dt);
        for (k, &i) in record_idx.iter().enumerate() {
            let p = world.body(i).position;
            let q = world.body(i).orientation;
            writeln!(out[k], "{line} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                     p.x, p.y, p.z, q.x, q.y, q.z, q.w)
                .expect("write csv");
        }
    }

    // Rays (probed once after the last frame): one line per hit in
    // <scene>_rays_<suffix>.csv - ray_idx body_idx fraction, sorted by
    // fraction (raycast_all already reports nearest-first; the sort keeps the
    // contract explicit and mirrors the C++ recorder).
    if !spec.rays.is_empty() {
        let fname = format!("{scene_name}_rays_{suffix}.csv");
        let mut rf = File::create(&fname).unwrap_or_else(|e| {
            eprintln!("record_scenes: cannot open {fname}: {e}");
            std::process::exit(2);
        });
        for (i, r) in spec.rays.iter().enumerate() {
            let from = Vector3::new(r.from[0] as Real, r.from[1] as Real, r.from[2] as Real);
            let to = Vector3::new(r.to[0] as Real, r.to[1] as Real, r.to[2] as Real);
            let filter = RayFilter { group: r.group, mask: r.mask };
            if r.all_hits {
                let mut hits: Vec<(BodyHandle, Real)> = world
                    .raycast_all_filtered(from, to, filter)
                    .into_iter()
                    .map(|(idx, hit)| (idx, hit.fraction))
                    .collect();
                hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                for (idx, fraction) in hits {
                    writeln!(rf, "{i} {} {fraction:.6}", idx.index()).expect("write csv");
                }
            } else if let Some((idx, hit)) = world.raycast_closest_filtered(from, to, filter) {
                writeln!(rf, "{i} {} {:.6}", idx.index(), hit.fraction).expect("write csv");
            }
        }
    }
}
