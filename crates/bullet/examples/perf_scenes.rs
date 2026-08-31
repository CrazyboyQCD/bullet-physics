//! Rust-side performance runner for the Bullet wall-clock comparison
//! (docs/bullet-perf.md).
//!
//! Reads the perf scene tables generated from tools/perf_scenes.json
//! (perf_scenes_table.rs), builds one scene, runs `warmup` untimed frames
//! then times `frames` fixed steps, and repeats the whole run (fresh world
//! each time) reporting one line per repeat. The output protocol matches the
//! C++ runner (tools/bullet_perf.cpp) so tools/perf.py can parse both:
//!
//! ```text
//! perf scene <name> bodies <n> frames <f> warmup <w>
//! repeat <i> total_ms <x> last300_ms <y>
//! ```
//!
//! `--stages` instead prints the cumulative per-stage timings of one run
//! (Rust only; the C++ side has no equivalent instrumentation hook):
//!
//! ```text
//! stage integrate_velocities_ms <x> ...
//! ```
//!
//! Run in release for meaningful numbers:
//! `cargo run --release -p bullet --example perf_scenes -- <scene>`.
//!
//! Usage: perf_scenes <scene> [--repeats N] [--stages]

use std::time::Instant;

use bullet::dynamics::{MultiBody, PhysicsWorld, RigidBody, StepStage, STEP_STAGES};
use bullet::math::{Quaternion, Real, Vector3, PI};
use bullet::shapes::CollisionShape;
use bullet::collision::broadphase::{BroadPhaseBackend, DbvtBroadPhase};

// Included via #[path] from src/perf_scenes_table.rs (auto-generated from
// tools/perf_scenes.json).
#[path = "../src/perf_scenes_table.rs"]
mod perf_scenes_table;

use perf_scenes_table::{PerfChainSpec, PerfSceneSpec, PerfShapeSpec, PERF_SCENES};

fn build_world(spec: &PerfSceneSpec) -> PhysicsWorld {
    // The backend mirrors the C++ runner's per-scene broad phase choice
    // (btDbvtBroadphase vs btAxisSweep3).
    let mut world = PhysicsWorld::with_broadphase(if spec.broadphase_dbvt {
        BroadPhaseBackend::Dbvt(DbvtBroadPhase::new())
    } else {
        BroadPhaseBackend::default()
    });
    world.set_gravity(Vector3::new(
        spec.gravity[0] as Real,
        spec.gravity[1] as Real,
        spec.gravity[2] as Real,
    ));
    world.set_iterations(spec.iterations);

    for b in spec.bodies {
        let shape = match b.shape {
            PerfShapeSpec::Box(h) => CollisionShape::box_shape(Vector3::new(
                h[0] as Real,
                h[1] as Real,
                h[2] as Real,
            )),
            PerfShapeSpec::Sphere(r) => CollisionShape::sphere(r as Real),
            PerfShapeSpec::Capsule(r, hh) => CollisionShape::capsule(r as Real, hh as Real),
            PerfShapeSpec::Mesh(verts, idx) => {
                let vertices: Vec<Vector3> = verts
                    .chunks(3)
                    .map(|c| Vector3::new(c[0] as Real, c[1] as Real, c[2] as Real))
                    .collect();
                let indices: Vec<[u32; 3]> = idx.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
                CollisionShape::triangle_mesh(vertices, indices)
            }
        };
        let position = Vector3::new(
            b.position[0] as Real,
            b.position[1] as Real,
            b.position[2] as Real,
        );
        let body = if b.mass == 0.0 {
            RigidBody::new_static(shape, position)
        } else {
            RigidBody::new_dynamic(shape, position, b.mass as Real)
                .friction(b.friction as Real)
                .restitution(b.restitution as Real)
        };
        world.add_body(body);
    }

    if let Some(c) = &spec.chains {
        add_chains(&mut world, c);
    }
    world
}

/// A grid of fixed-base revolute pendulums, matching the C++ runner's chain
/// construction (base positions, joint offsets and initial angles must be
/// identical on both sides; the collider is what makes the C++ island solver
/// process the multibodies).
fn add_chains(world: &mut PhysicsWorld, c: &PerfChainSpec) {
    let rows = c.count.div_ceil(c.cols);
    let length = Vector3::new(c.link_length as Real, 0.0, 0.0);
    for k in 0..c.count {
        let (col, row) = (k % c.cols, k / c.cols);
        let x = (col as Real - (c.cols as Real - 1.0) / 2.0) * c.spacing as Real;
        let z = (row as Real - (rows as Real - 1.0) / 2.0) * c.spacing as Real;
        let mut mb = MultiBody::new(
            c.links,
            c.base_mass as Real,
            Vector3::ONE,
            true,  // fixed base
            true,  // can sleep
        );
        for i in 0..c.links {
            mb.setup_revolute(
                i,
                c.link_mass as Real,
                Vector3::ONE,
                if i == 0 { None } else { Some(i - 1) },
                Quaternion::IDENTITY,
                Vector3::Z,
                length,
                length,
            );
        }
        mb.finalize_multidof();
        mb.set_base_pos(Vector3::new(x, c.base_height as Real, z));
        for i in 0..c.links {
            mb.set_joint_pos_multidof(i, &[c.initial_angle as Real]);
        }
        let collider = CollisionShape::sphere(c.collider_radius as Real);
        mb.set_base_collider(collider.clone(), 0.5, 0.0);
        for i in 0..c.links {
            mb.set_link_collider(i, collider.clone(), 0.5, 0.0);
        }
        world.add_multibody(mb);
    }
}

/// Deterministic per-frame arena tick (the `game_arena` script), mirroring
/// the C++ runner's `SceneRun::step` formula for formula: a launcher ball
/// every 30 frames plus the velocity-driven dozer. Runs before
/// `step_simulation` so the solver sees the scripted frame.
fn arena_tick(world: &mut PhysicsWorld, frame: u64) {
    const DOZER: usize = 1; // bodies[1] is the mass-50 dozer sphere (json contract)
    if frame.is_multiple_of(30) && frame < 610 {
        let k = frame / 30;
        let from = Vector3::new(0.0, 2.5, -8.0);
        let target = Vector3::new(((k % 5) as Real - 2.0) * 0.6, 0.7, 0.0);
        let dir = (target - from).normalize();
        let ball = world.add_body(
            RigidBody::new_dynamic(CollisionShape::sphere(0.25), from, 1.0).friction(0.5),
        );
        world.body_mut(ball).linear_velocity = dir * 12.0;
    }
    // Only x is overridden so y/z still solve naturally; activate() first -
    // writing the velocity of a sleeping body would not wake it (and the
    // wake propagates through contacts).
    world.body_mut(DOZER).activate();
    let vx = 4.0 * (2.0 * PI / 240.0) * (2.0 * PI * frame as Real / 240.0).cos();
    world.body_mut(DOZER).linear_velocity.x = vx;
}

/// One scripted step: the arena tick (when the scene asks for it) plus the
/// solver step, shared by every run mode so the workload is identical.
fn step_once(world: &mut PhysicsWorld, spec: &PerfSceneSpec, frame: u64, dt: Real) {
    if spec.arena_script {
        arena_tick(world, frame);
    }
    world.step_simulation(dt);
}

fn run_scene(spec: &PerfSceneSpec, dt: Real) -> (std::time::Duration, std::time::Duration) {
    let mut world = build_world(spec);
    let mut frame: u64 = 0;
    for _ in 0..spec.warmup {
        step_once(&mut world, spec, frame, dt);
        frame += 1;
    }
    let start = Instant::now();
    let mut last300_start = None;
    for f in 0..spec.frames {
        if f + 300 == spec.frames {
            last300_start = Some(Instant::now());
        }
        step_once(&mut world, spec, frame, dt);
        frame += 1;
    }
    let total = start.elapsed();
    let last300 = last300_start.map_or(total, |t| t.elapsed());
    (total, last300)
}

fn run_stages(spec: &PerfSceneSpec, dt: Real) {
    let mut world = build_world(spec);
    let mut frame: u64 = 0;
    for _ in 0..spec.warmup {
        step_once(&mut world, spec, frame, dt);
        frame += 1;
    }
    world.enable_step_timing();
    for _ in 0..spec.frames {
        step_once(&mut world, spec, frame, dt);
        frame += 1;
    }
    let t = world.step_timings();
    for stage in STEP_STAGES {
        let name = match stage {
            StepStage::IntegrateVelocities => "integrate_velocities",
            StepStage::UpdateAabbs => "update_aabbs",
            StepStage::Broadphase => "broadphase",
            StepStage::Islands => "islands",
            StepStage::Narrowphase => "narrowphase",
            StepStage::Solve => "solve",
            StepStage::IntegratePositions => "integrate_positions",
        };
        println!("stage {name}_ms {:.3}", t.stage(stage).as_secs_f64() * 1000.0);
    }
    println!("stage total_ms {:.3}", t.total().as_secs_f64() * 1000.0);
}

/// Widening helper that stays cast-free under both precisions (same
/// pattern as `frame_relevant` in axis_sweep3.rs).
fn to_f64(v: impl Into<f64>) -> f64 {
    v.into()
}

/// Debug probe: step from frame 0 (including warmup) and dump aggregate
/// stack health every 5 frames (sum of dynamic-body y, sum/max speed,
/// awake count).
fn run_dump(spec: &PerfSceneSpec, dt: Real) {
    let mut world = build_world(spec);
    let frames = spec.warmup + spec.frames;
    for f in 0..frames {
        step_once(&mut world, spec, f as u64, dt);
        if f % 5 == 0 || f == frames - 1 {
            let mut sum_y = 0.0f64;
            let mut sum_v = 0.0f64;
            let mut max_v = 0.0f64;
            let mut awake = 0usize;
            let mut count = 0usize;
            for b in world.bodies() {
                if b.is_static() {
                    continue;
                }
                count += 1;
                sum_y += to_f64(b.position.y);
                let v = to_f64(b.linear_velocity.length());
                sum_v += v;
                max_v = max_v.max(v);
                if b.is_active() {
                    awake += 1;
                }
            }
            println!(
                "dump frame {f} bodies {count} sum_y {sum_y:.4} sum_v {sum_v:.5} max_v {max_v:.5} awake {awake}"
            );
        }
    }
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("perf_scenes: warning: debug build; run with `cargo run --release`");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scene_name = args.first().cloned().unwrap_or_default();
    let mut repeats: u32 = 5;
    let mut stages = false;
    let mut dump = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repeats" => {
                i += 1;
                repeats = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(5);
            }
            "--stages" => stages = true,
            "--dump" => dump = true,
            other => eprintln!("perf_scenes: ignoring unknown arg {other}"),
        }
        i += 1;
    }

    let Some((name, spec)) = PERF_SCENES.iter().find(|(n, _)| *n == scene_name) else {
        let avail: Vec<&str> = PERF_SCENES.iter().map(|(n, _)| *n).collect();
        eprintln!("perf_scenes: unknown scene '{scene_name}'; available: {avail:?}");
        std::process::exit(2);
    };
    let dt: Real = spec.dt_num as Real / spec.dt_den as Real;
    let bodies = spec.bodies.len() + spec.chains.as_ref().map_or(0, |c| c.count);
    println!(
        "perf scene {} bodies {} frames {} warmup {}",
        name, bodies, spec.frames, spec.warmup
    );

    if dump {
        run_dump(spec, dt);
        return;
    }
    if stages {
        run_stages(spec, dt);
        return;
    }
    for r in 0..repeats {
        let (total, last300) = run_scene(spec, dt);
        println!(
            "repeat {} total_ms {:.3} last300_ms {:.3}",
            r,
            total.as_secs_f64() * 1000.0,
            last300.as_secs_f64() * 1000.0
        );
    }
}
