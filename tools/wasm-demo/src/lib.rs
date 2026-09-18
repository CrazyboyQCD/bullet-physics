// Browser-only cdylib: the wasm32 target is the only supported build (the
// workspace's host-side `cargo test/clippy --workspace` runs see an empty
// crate, so precision features forwarded to `bullet` never touch this code).
#![cfg(target_arch = "wasm32")]

//! C ABI bindings that drive the `bullet` engine from a browser (the
//! renderer and UI live in `template.html` on the JS side, Babylon.js).        
//!
//! The physics library stays untouched; this is a thin harness: build a        
//! demo scene, step it at a fixed dt, and stream rigid-body poses and
//! soft-body node positions back through wasm linear memory as plain
//! float buffers. Body/soft metadata (spawn table, face topology) is
//! exported once per scene so the JS side can build meshes without
//! mirroring scene layout.

use bullet::dynamics::{body::ActivationState, Constraint};
use bullet::math::{Real, Vector3};
use bullet::prelude::*;

/// Floats per rigid-body pose: position (3) + orientation xyzw (4).
/// (The spawn-table stride, 16 floats, is a JS-side contract — see template.html.)
const POSE_STRIDE: usize = 7;

/// Spawn-table shape kinds (column 0 of each spec row).
const KIND_BOX: f32 = 0.0;
const KIND_SPHERE: f32 = 1.0;

/// Spawn-table body flags (column 14): 0 dynamic, 1 static, 2 kinematic.
const FLAG_DYNAMIC: f32 = 0.0;
const FLAG_STATIC: f32 = 1.0;
const FLAG_KINEMATIC: f32 = 2.0;

struct SoftRange {
    handle: SoftBodyHandle,
    node_off: usize,
    node_count: usize,
    face_off: usize,
    face_count: usize,
}

pub struct Demo {
    world: PhysicsWorld,
    handles: Vec<BodyHandle>,
    specs: Vec<f32>,
    poses: Vec<f32>,
    softs: Vec<SoftRange>,
    soft_pos: Vec<f32>,
    soft_faces: Vec<u32>,
    soft_meta: Vec<u32>,
    /// Lateral wind applied to every soft node each step (cloth scene).
    wind: f32,
    time: f32,
    kinematic_pusher: Option<BodyHandle>,
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b]
}

impl Demo {
    fn new(scene: u32) -> Demo {
        let mut demo = Demo {
            world: PhysicsWorld::new(),
            handles: Vec::new(),
            specs: Vec::new(),
            poses: Vec::new(),
            softs: Vec::new(),
            soft_pos: Vec::new(),
            soft_faces: Vec::new(),
            soft_meta: vec![0],
            wind: 0.0,
            time: 0.0,
            kinematic_pusher: None,
        };
        demo.world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
        match scene {
            0 => demo.scene_tower(),
            1 => demo.scene_chains(),
            _ => demo.scene_cloth(),
        }
        demo.finish();
        demo
    }

    fn push_spec(
        &mut self,
        kind: f32,
        pos: Vector3,
        params: [f32; 3],
        color: [f32; 3],
        flag: f32,
    ) {
        self.specs.extend_from_slice(&[
            kind,
            pos.x, pos.y, pos.z,
            0.0, 0.0, 0.0, 1.0,
            params[0], params[1], params[2],
            color[0], color[1], color[2],
            flag,
            0.0,
        ]);
    }

    fn add_box(&mut self, pos: Vector3, half: Vector3, mass: Real, color: [f32; 3]) -> BodyHandle {
        let body = RigidBody::new_dynamic(CollisionShape::box_shape(half), pos, mass)
            .friction(0.5)
            .restitution(0.1);
        self.push_spec(KIND_BOX, pos, half.into(), color, FLAG_DYNAMIC);
        let h = self.world.add_body(body);
        self.handles.push(h);
        h
    }

    fn add_static_box(&mut self, pos: Vector3, half: Vector3, color: [f32; 3]) -> BodyHandle {
        let body = RigidBody::new_static(CollisionShape::box_shape(half), pos).friction(0.5);
        self.push_spec(KIND_BOX, pos, half.into(), color, FLAG_STATIC);
        let h = self.world.add_body(body);
        self.handles.push(h);
        h
    }

    fn add_sphere(&mut self, pos: Vector3, radius: Real, mass: Real, color: [f32; 3]) -> BodyHandle {
        let body = RigidBody::new_dynamic(CollisionShape::sphere(radius), pos, mass)
            .friction(0.4)
            .restitution(0.3);
        self.push_spec(KIND_SPHERE, pos, [radius, 0.0, 0.0], color, FLAG_DYNAMIC);
        let h = self.world.add_body(body);
        self.handles.push(h);
        h
    }

    fn ground(&mut self) {
        self.add_static_box(
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(45.0, 1.0, 45.0),
            [0.42, 0.45, 0.5],
        );
    }

    /// 7x7 x 10-layer box tower (490 boxes) + a wrecking ball dropped on it.
    fn scene_tower(&mut self) {
        self.ground();
        let half = 0.5;
        let step = half * 2.0 + 0.06;
        for layer in 0..10 {
            let color = hsv(layer as f32 * 0.11, 0.55, 0.95);
            for ix in 0..7 {
                for iz in 0..7 {
                    let x = (ix as f32 - 3.0) * step;
                    let z = (iz as f32 - 3.0) * step;
                    let y = 0.5 + layer as f32 * step + half;
                    self.add_box(
                        Vector3::new(x, y, z),
                        Vector3::new(half, half, half),
                        1.0,
                        color,
                    );
                }
            }
        }
        let ball = self.add_sphere(Vector3::new(6.0, 20.0, 6.0), 1.1, 30.0, [0.9, 0.35, 0.2]);
        self.world.body_mut(ball).linear_velocity = Vector3::new(-4.0, 0.0, -4.0);
    }

    /// Three hanging hinge chains with wrecking-ball tails + a kinematic
    /// pusher plate sweeping under them (velocity-driven kinematics).
    fn scene_chains(&mut self) {
        self.ground();
        let link_half = 0.32;
        for chain in 0..3 {
            let x = (chain as f32 - 1.0) * 5.0;
            let color = hsv(chain as f32 * 0.3 + 0.05, 0.6, 0.95);
            let anchor = self.add_static_box(
                Vector3::new(x, 16.0, 0.0),
                Vector3::new(0.25, 0.25, 0.25),
                [0.3, 0.3, 0.35],
            );
            let mut prev = anchor;
            let mut prev_pivot = Vector3::new(0.0, -0.25, 0.0);
            let mut kicked = Vec::new();
            for i in 0..9 {
                let y = 16.0 - 0.25 - link_half - i as f32 * link_half * 2.0;
                let h = self.add_box(
                    Vector3::new(x, y, 0.0),
                    Vector3::new(link_half * 0.6, link_half, link_half * 0.6),
                    0.8,
                    color,
                );
                if chain == 1 {
                    kicked.push(h);
                }
                self.world.add_constraint(Constraint::hinge(
                    prev,
                    h,
                    prev_pivot,
                    Vector3::new(1.0, 0.0, 0.0),
                    Vector3::new(0.0, link_half, 0.0),
                    Vector3::new(1.0, 0.0, 0.0),
                ));
                prev = h;
                prev_pivot = Vector3::new(0.0, -link_half, 0.0);
            }
            let y_tail = 16.0 - 0.25 - 9.0 * link_half * 2.0 - 0.9;
            let ball = self.add_sphere(Vector3::new(x, y_tail, 0.0), 0.9, 6.0, color);
            self.world.add_constraint(Constraint::hinge(
                prev,
                ball,
                prev_pivot,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 0.9, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
            ));
            // Kick the middle chain so the scene starts in motion.
            for h in &kicked {
                self.world.body_mut(*h).linear_velocity = Vector3::new(6.0, 0.0, 0.0);
            }
        }
        let plate_pos = Vector3::new(0.0, 2.5, 0.0);
        let plate_body = RigidBody::new_kinematic(
            CollisionShape::box_shape(Vector3::new(2.5, 0.3, 2.5)),
            plate_pos,
        );
        self.push_spec(
            KIND_BOX,
            plate_pos,
            [2.5, 0.3, 2.5],
            [0.95, 0.75, 0.2],
            FLAG_KINEMATIC,
        );
        let plate = self.world.add_body(plate_body);
        self.handles.push(plate);
        self.world.body_mut(plate).linear_velocity = Vector3::new(6.0, 0.0, 0.0);
        self.kinematic_pusher = Some(plate);
    }

    /// Flag cloth: pinned along one full edge, driven by wind. Both engines
    /// pin the same edge node-by-node for a fair side-by-side (CreatePatch's
    /// fixeds bitmask covers corners only, so the edge is pinned manually —
    /// corner-only pins let the sheet whip through the pin line and spike).
    fn scene_cloth(&mut self) {
        self.ground();
        let res = 18;
        let mut cloth = bullet::dynamics::create_patch(
            Vector3::new(-4.0, 9.0, -4.0),
            Vector3::new(4.0, 9.0, -4.0),
            Vector3::new(-4.0, 9.0, 4.0),
            Vector3::new(4.0, 9.0, 4.0),
            res,
            res,
            0,
            false,
            0.0,
        );
        cloth.set_total_mass(9.0);
        for n in &mut cloth.nodes[..res] {
            n.im = 0.0;
        }
        let node_off = self.soft_pos.len() / 3;
        let node_count = cloth.nodes.len();
        let face_off = self.soft_faces.len() / 3;
        let face_count = cloth.faces.len();
        for n in &cloth.nodes {
            self.soft_pos.extend_from_slice(&[n.x.x, n.x.y, n.x.z]);
        }
        for f in &cloth.faces {
            self.soft_faces
                .extend_from_slice(&[f.n[0] as u32, f.n[1] as u32, f.n[2] as u32]);
        }
        let handle = self.world.add_soft_body(cloth);
        self.softs.push(SoftRange {
            handle,
            node_off,
            node_count,
            face_off,
            face_count,
        });

        self.wind = 0.9;
    }

    /// Snapshot static per-scene data for the renderer.
    fn finish(&mut self) {
        self.poses.clear();
        self.poses.resize(self.handles.len() * POSE_STRIDE, 0.0);

        self.soft_meta.clear();
        self.soft_meta.push(self.softs.len() as u32);
        for r in &self.softs {
            self.soft_meta.extend_from_slice(&[
                r.node_off as u32,
                r.node_count as u32,
                r.face_off as u32,
                r.face_count as u32,
            ]);
        }

        self.sync_poses();
        self.sync_soft_nodes();
    }

    fn sync_poses(&mut self) {
        // Bodies can be added at runtime (shooting); keep the buffer in sync.
        let need = self.handles.len() * POSE_STRIDE;
        if self.poses.len() != need {
            self.poses.resize(need, 0.0);
        }
        for (i, h) in self.handles.iter().enumerate() {
            let body = self.world.body(*h);
            let p = body.position;
            let (qx, qy, qz, qw) = {
                let q = body.orientation;
                (q.x, q.y, q.z, q.w)
            };
            let o = i * POSE_STRIDE;
            self.poses[o] = p.x;
            self.poses[o + 1] = p.y;
            self.poses[o + 2] = p.z;
            self.poses[o + 3] = qx;
            self.poses[o + 4] = qy;
            self.poses[o + 5] = qz;
            self.poses[o + 6] = qw;
        }
    }

    fn sync_soft_nodes(&mut self) {
        for r in &self.softs {
            let soft = self.world.soft_body(r.handle);
            for (i, n) in soft.nodes.iter().enumerate() {
                let o = (r.node_off + i) * 3;
                self.soft_pos[o] = n.x.x;
                self.soft_pos[o + 1] = n.x.y;
                self.soft_pos[o + 2] = n.x.z;
            }
        }
    }

    fn step(&mut self, dt: f32) {
        self.time += dt;
        // Reverse the kinematic pusher at the arena edges (velocity-driven
        // kinematics: pose integration is owned by the world).
        if let Some(h) = self.kinematic_pusher {
            let p = self.world.body(h).position;
            let v = self.world.body(h).linear_velocity;
            let nx = p.x + v.x * dt;
            if (nx > 14.0 && v.x > 0.0) || (nx < -14.0 && v.x < 0.0) {
                self.world.body_mut(h).linear_velocity = Vector3::new(-v.x, 0.0, 0.0);
            }
        }
        // Lateral wind on soft nodes (node force accumulators reset per step).
        if self.wind.abs() > 0.0 {
            let gust = (self.time * 1.7).sin() * 0.6 + 0.4;
            let force = Vector3::new(
                (self.wind * gust * 0.024) as Real,
                0.0,
                ((self.time * 1.1).cos() * self.wind * 0.007) as Real,
            );
            for r in &self.softs {
                let soft = self.world.soft_body_mut(r.handle);
                for n in &mut soft.nodes {
                    if n.im > 0.0 {
                        n.f += force;
                    }
                }
            }
        }
        self.world.step_simulation(dt as Real);
        self.sync_poses();
        self.sync_soft_nodes();
    }

    fn shoot(&mut self, ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32) {
        let dir = Vector3::new(dx as Real, dy as Real, dz as Real).normalize_or_zero();
        let pos = Vector3::new(ox as Real, oy as Real, oz as Real) + dir * 1.5;
        let color = hsv(self.time.fract() * 0.7, 0.7, 1.0);
        let body = RigidBody::new_dynamic(CollisionShape::sphere(0.5), pos, 4.0)
            .friction(0.4)
            .restitution(0.3)
            // Fast projectile: continuous collision so it cannot tunnel
            // through the thin tower walls.
            .ccd(1.0, 0.5);
        self.push_spec(KIND_SPHERE, pos, [0.5, 0.0, 0.0], color, FLAG_DYNAMIC);
        let h = self.world.add_body(body);
        self.world.body_mut(h).linear_velocity = dir * 30.0;
        self.handles.push(h);
    }
}

// ---------------------------------------------------------------------------
// C ABI surface. All returned buffers stay valid until the next demo_new.
// ---------------------------------------------------------------------------

/// # Safety
/// Returns a boxed demo; the caller owns it until `demo_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_new(scene: u32) -> *mut Demo {
    Box::into_raw(Box::new(Demo::new(scene)))
}

/// # Safety
/// `demo` must come from `demo_new` and not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_free(demo: *mut Demo) {
    unsafe {
        drop(Box::from_raw(demo));
    }
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_step(demo: *mut Demo, dt: f32) {
    unsafe { (*demo).step(dt) };
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_shoot(
    demo: *mut Demo,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
) {
    unsafe { (*demo).shoot(ox, oy, oz, dx, dy, dz) };
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_set_wind(demo: *mut Demo, wind: f32) {
    unsafe { (*demo).wind = wind };
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_idle(demo: *mut Demo) -> u32 {
    unsafe {
        let demo = &*demo;
        // Soft bodies never sleep: a cloth scene is never idle.
        if !demo.softs.is_empty() {
            return 0;
        }
        for h in &demo.handles {
            match demo.world.body(*h).activation_state() {
                // Asleep or sim-disabled (static bodies start as
                // IslandSleeping, so they do not block idling).
                ActivationState::IslandSleeping | ActivationState::DisableSimulation => {}
                // Active / WantsDeactivation / DisableDeactivation (kinematic)
                // keep the demo busy.
                _ => return 0,
            }
        }
        1
    }
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_body_count(demo: *mut Demo) -> u32 {
    unsafe { (*demo).handles.len() as u32 }
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_body_specs(demo: *mut Demo) -> *const f32 {
    unsafe { (*demo).specs.as_ptr() }
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_poses(demo: *mut Demo) -> *const f32 {
    unsafe { (*demo).poses.as_ptr() }
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_soft_meta(demo: *mut Demo) -> *const u32 {
    unsafe { (*demo).soft_meta.as_ptr() }
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_soft_nodes(demo: *mut Demo) -> *const f32 {
    unsafe { (*demo).soft_pos.as_ptr() }
}

/// # Safety
/// `demo` must be a live pointer from `demo_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn demo_soft_faces(demo: *mut Demo) -> *const u32 {
    unsafe { (*demo).soft_faces.as_ptr() }
}
