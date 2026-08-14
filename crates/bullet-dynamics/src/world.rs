use std::collections::HashMap;

use bullet_collision::broadphase::{AxisSweep3, BroadPhase};
use bullet_collision::manifold::ContactManifold;
use bullet_collision::narrowphase::{self, SweepHit};
use bullet_collision::raycast::{self, RayHit};
use crate::body::RigidBody;
use bullet_math::{Aabb, Quaternion, Real, Vector3};
use crate::solver::sequential_impulse::{self, ContactConstraint, SPLIT_TURN_ERP};

/// Scale applied to a cached impulse before the warm-start pass (Bullet's
/// `m_warmstartingFactor`): old impulses are faded slightly so the solver can
/// correct drift instead of blindly trusting the previous step.
const WARM_STARTING_FACTOR: Real = 0.85;
/// Distance below which a fresh contact point is the same physical contact as
/// a cached one (Bullet's `gContactBreakingThreshold`), keeping its impulse.
const CONTACT_BREAKING_THRESHOLD: Real = 0.02;

/// A cached contact point from the previous step: the accumulated normal
/// impulse is re-applied on the next step's warm-start pass (the
/// persistent-manifold `m_appliedImpulse` cache in Bullet). Friction impulses
/// are deliberately not cached — see [`warm_start_impulses`].
struct CachedContact {
    position: Vector3,
    normal: Vector3,
    normal_impulse: Real,
}

/// A discrete rigid-body simulation world, mirroring `btDiscreteDynamicsWorld`.
pub struct PhysicsWorld {
    bodies: Vec<RigidBody>,
    broadphase: AxisSweep3,
    gravity: Vector3,
    iterations: u32,
    /// Accumulated contact impulses from the previous step, keyed by body
    /// pair; the cache is rebuilt every step from the solved constraints.
    contact_cache: HashMap<(usize, usize), Vec<CachedContact>>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            broadphase: AxisSweep3::new(),
            gravity: Vector3::new(0.0, -9.81, 0.0),
            iterations: 10,
            contact_cache: HashMap::new(),
        }
    }

    pub fn set_gravity(&mut self, gravity: Vector3) {
        self.gravity = gravity;
    }

    pub fn gravity(&self) -> Vector3 {
        self.gravity
    }

    /// Number of solver (PGS) iterations per step.
    pub fn set_iterations(&mut self, iterations: u32) {
        self.iterations = iterations;
    }

    /// Add a body and return its index.
    ///
    /// MVP note: body indices coincide with broad-phase proxy indices, so
    /// removal is not yet supported.
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let aabb = body.shape.world_aabb(body.position, body.orientation);
        let proxy = self.broadphase.create_proxy(aabb);
        debug_assert_eq!(proxy, self.bodies.len());
        self.bodies.push(body);
        proxy
    }

    pub fn body(&self, index: usize) -> &RigidBody {
        &self.bodies[index]
    }

    pub fn body_mut(&mut self, index: usize) -> &mut RigidBody {
        &mut self.bodies[index]
    }

    pub fn bodies(&self) -> &[RigidBody] {
        &self.bodies
    }

    /// Cast a ray against every body and return the closest hit together with
    /// the body index (Bullet's `rayTest` with a closest-result callback).
    ///
    /// The broad phase first prunes proxies whose AABB the segment does not
    /// cross, so the (more expensive) exact per-shape test only runs on
    /// candidates — mirroring `btAxisSweep3::rayTest` feeding a
    /// `ClosestRayResultCallback`. The AABB test is conservative, so the
    /// result is identical to testing every body.
    pub fn raycast_closest(&self, ray_from: Vector3, ray_to: Vector3) -> Option<(usize, RayHit)> {
        let mut best: Option<(usize, RayHit)> = None;
        for i in self.broadphase.raycast_candidates(ray_from, ray_to) {
            let body = &self.bodies[i];
            if let Some(hit) = raycast::raycast(
                &body.shape,
                body.position,
                body.orientation,
                ray_from,
                ray_to,
            ) && best.is_none_or(|(_, b)| hit.fraction < b.fraction)
            {
                best = Some((i, hit));
            }
        }
        best
    }

    /// Indices of bodies whose broad-phase AABB overlaps `aabb` (exact AABB
    /// test; the caller then runs the narrow phase on the candidates).
    pub fn aabb_candidates(&self, aabb: Aabb) -> Vec<usize> {
        self.broadphase.aabb_candidates(aabb)
    }

    /// Closest convex sweep of `shape` from `from` to `to` against every body
    /// except `ignore`, returning the hit body index and the closest hit.
    ///
    /// Mirrors `btCollisionWorld::convexSweep` feeding a
    /// `ClosestConvexResultCallback` (the character controller's step
    /// sweeps). The broad phase prunes bodies whose AABB does not touch the
    /// swept AABB of the moving shape.
    pub fn convex_sweep_closest(
        &self,
        shape: &bullet_collision::shapes::CollisionShape,
        from: Vector3,
        ori: Quaternion,
        to: Vector3,
        ignore: Option<usize>,
    ) -> Option<(usize, SweepHit)> {
        let swept = shape
            .world_aabb(from, ori)
            .merged(&shape.world_aabb(to, ori));
        let mut best: Option<(usize, SweepHit)> = None;
        for i in self.broadphase.aabb_candidates(swept) {
            if Some(i) == ignore {
                continue;
            }
            let body = &self.bodies[i];
            if let Some(hit) = narrowphase::convex_cast(
                shape,
                from,
                ori,
                to,
                &body.shape,
                body.position,
                body.orientation,
            ) && best.is_none_or(|(_, b)| hit.fraction < b.fraction)
            {
                best = Some((i, hit));
            }
        }
        best
    }

    /// Contact manifolds between `shape` at `pos`/`ori` and every body except
    /// `ignore` (the character controller's touching contacts and penetration
    /// recovery, mirroring a ghost object's overlapping-pair manifolds).
    pub fn contact_test(
        &self,
        shape: &bullet_collision::shapes::CollisionShape,
        pos: Vector3,
        ori: Quaternion,
        ignore: Option<usize>,
    ) -> Vec<(usize, ContactManifold)> {
        let aabb = shape.world_aabb(pos, ori);
        let mut out = Vec::new();
        for i in self.broadphase.aabb_candidates(aabb) {
            if Some(i) == ignore {
                continue;
            }
            let body = &self.bodies[i];
            if let Some(m) = narrowphase::collide(
                shape,
                pos,
                ori,
                &body.shape,
                body.position,
                body.orientation,
            ) {
                out.push((i, m));
            }
        }
        out
    }

    /// Set a body's pose directly and refresh its broad-phase AABB. Used by
    /// the character controller to move its kinematic body; the solver still
    /// treats the body as immovable (static bodies are never integrated).
    pub fn set_body_transform(&mut self, index: usize, position: Vector3, orientation: Quaternion) {
        let body = &mut self.bodies[index];
        body.position = position;
        body.orientation = orientation;
        let aabb = body.shape.world_aabb(position, orientation);
        self.broadphase.set_aabb(index, aabb);
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step_simulation(&mut self, dt: Real) {
        let inv_dt = 1.0 / dt;

        // 1. Integrate velocities from gravity and accumulated forces.
        for body in &mut self.bodies {
            body.integrate_velocities(self.gravity, dt);
        }

        // 2. Predicted positions (used only for contact detection).
        let predicted: Vec<Vector3> = self
            .bodies
            .iter()
            .map(|b| b.predicted_position(dt))
            .collect();

        // 3. Update broad-phase AABBs from predicted poses. Sleeping bodies
        // keep their AABB: they do not move, so it stays correct.
        for (i, body) in self.bodies.iter().enumerate() {
            if !body.is_active() {
                continue;
            }
            let aabb = body.shape.world_aabb(predicted[i], body.orientation);
            self.broadphase.set_aabb(i, aabb);
        }

        // 4. Broad phase: candidate overlapping pairs.
        let pairs = self.broadphase.compute_pairs();

        // 5. Narrow phase: exact contact manifolds -> constraints.
        let mut constraints: Vec<ContactConstraint> = Vec::new();
        let mut contact_pairs: Vec<(usize, usize)> = Vec::new();
        for &(i, j) in &pairs {
            let a = &self.bodies[i];
            let b = &self.bodies[j];
            if let Some(mut manifold) = narrowphase::collide(
                &a.shape,
                predicted[i],
                a.orientation,
                &b.shape,
                predicted[j],
                b.orientation,
            ) {
                contact_pairs.push((i, j));
                // Inherit the previous step's accumulated impulses for contact
                // points that persist (Bullet's persistent manifolds).
                warm_start_impulses(
                    &mut manifold.points,
                    self.contact_cache.get(&(i, j)).map(Vec::as_slice),
                );
                for cp in &manifold.points {
                    constraints.push(ContactConstraint::prepare(i, j, cp, a, b));
                }
            }
        }

        // 5b. Wake up sleeping bodies that touch an active body (Bullet's
        // island-based activation propagation: activation travels along the
        // contact graph one step at a time). A contact between two sleeping
        // bodies does not wake them — a stable stack stays asleep.
        for &(i, j) in &contact_pairs {
            let a_active = self.bodies[i].is_active();
            let b_active = self.bodies[j].is_active();
            if a_active && !b_active {
                self.bodies[j].activate();
            } else if b_active && !a_active {
                self.bodies[i].activate();
            }
        }

        // 6. Solve constraints (correct velocities).
        sequential_impulse::solve(&mut self.bodies, &mut constraints, self.iterations, inv_dt);

        // 6b. Write back the solved impulses so the next step can warm start
        // from them (the cache is rebuilt every step, dropping stale pairs).
        let mut next_cache: HashMap<(usize, usize), Vec<CachedContact>> = HashMap::new();
        for c in &constraints {
            next_cache
                .entry((c.body_a, c.body_b))
                .or_default()
                .push(CachedContact {
                    position: c.position,
                    normal: c.normal,
                    normal_impulse: c.normal_impulse,
                });
        }
        self.contact_cache = next_cache;

        // 6c. Deactivation: bodies below the sleeping thresholds for longer
        // than `DEACTIVATION_TIME` seconds go to sleep (Bullet's
        // `updateActivationState` between solving and integrating).
        for body in &mut self.bodies {
            body.update_deactivation(dt);
            if body.wants_deactivation() {
                body.go_to_sleep();
            }
        }

        // 7. Integrate positions with the corrected velocities. Bodies that
        // exceed their CCD motion threshold are clamped at the first
        // swept-sphere impact so they cannot tunnel through thin geometry
        // (Bullet's `integrateTransforms` CCD motion clamping; the collision
        // response itself runs on the next step's narrow phase).
        let ccd_clamps: Vec<Real> = (0..self.bodies.len())
            .map(|i| {
                let body = &self.bodies[i];
                let threshold = body.ccd_motion_threshold();
                if !body.is_active() || body.is_static() || threshold <= 0.0 {
                    return 1.0;
                }
                let radius = body.ccd_swept_sphere_radius();
                if radius <= 0.0 || !body.shape.is_convex() {
                    return 1.0;
                }
                let from = body.position;
                let to = from + body.linear_velocity * dt;
                let square_motion = (to - from).length_squared();
                if square_motion <= threshold * threshold {
                    return 1.0;
                }
                let mut best = 1.0;
                for j in 0..self.bodies.len() {
                    if i == j {
                        continue;
                    }
                    let other = &self.bodies[j];
                    if let Some(hit) = narrowphase::swept_sphere_cast(
                        from,
                        to,
                        radius,
                        &other.shape,
                        other.position,
                        other.orientation,
                    ) && hit.fraction < best
                    {
                        best = hit.fraction;
                    }
                }
                best
            })
            .collect();

        for (i, body) in self.bodies.iter_mut().enumerate() {
            body.integrate_position(dt);
            // Split-impulse positional correction (Bullet's
            // `writebackVelocityAndTransform`): move the pose along the
            // solver's push velocity so deep penetrations are corrected
            // without changing the real velocities. The turn velocity is
            // scaled by `m_splitImpulseTurnErp`. Push is cleared here; the
            // solver starts every step from zero.
            if body.is_active() && !body.is_static() && body.push_linear != Vector3::ZERO {
                body.position += body.push_linear * dt;
            }
            if body.is_active() && !body.is_static() && body.push_angular != Vector3::ZERO {
                let w = body.push_angular * SPLIT_TURN_ERP;
                let dq = Quaternion::from_xyzw(
                    w.x * dt * 0.5,
                    w.y * dt * 0.5,
                    w.z * dt * 0.5,
                    0.0,
                ) * body.orientation;
                body.orientation = (body.orientation + dq).normalize();
            }
            body.push_linear = Vector3::ZERO;
            body.push_angular = Vector3::ZERO;
            if ccd_clamps[i] < 1.0 {
                let from = body.position - body.linear_velocity * dt;
                body.position = from + (body.position - from) * ccd_clamps[i];
            }
        }

        // 8. Clear accumulated forces for the next step.
        for body in &mut self.bodies {
            body.clear_forces();
        }
    }
}

/// Inherit the previous step's accumulated normal impulse for each fresh
/// contact point. A new point within [`CONTACT_BREAKING_THRESHOLD`] of a
/// cached one is the same physical contact (Bullet's `getCacheEntry` +
/// `replaceContactPoint` persistence) and keeps its impulse, scaled by
/// [`WARM_STARTING_FACTOR`] so the solver can correct small drift (Bullet's
/// `m_warmstartingFactor`).
///
/// Two guards protect against the point set changing between steps (which
/// Bullet's persistent manifolds smooth out and we rebuild per step):
/// - the contact normal must stay (nearly) the same: our mesh narrow phase
///   generates one normal per triangle, so a cached impulse applied along a
///   newly rotated normal would kick the body sideways;
/// - friction impulses are **not** inherited: the tangent basis rotates with
///   the point set, and stale friction kicks make a resting body on a mesh
///   oscillate instead of settling (observed with `falling_mesh`). Friction
///   rebuilds from zero inside the solver iterations, exactly as on the first
///   contact frame.
fn warm_start_impulses(points: &mut [bullet_collision::manifold::ContactPoint], cache: Option<&[CachedContact]>) {
    const NORMAL_EPS: Real = 0.9;
    let breaking_sq = CONTACT_BREAKING_THRESHOLD * CONTACT_BREAKING_THRESHOLD;
    let Some(cache) = cache else {
        return;
    };
    let mut used = vec![false; cache.len()];
    for p in points {
        let mut best: Option<usize> = None;
        let mut best_d2 = Real::INFINITY;
        for (i, c) in cache.iter().enumerate() {
            if used[i] {
                continue;
            }
            let d2 = (p.position - c.position).length_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best = Some(i);
            }
        }
        let Some(i) = best else {
            continue;
        };
        if best_d2 <= breaking_sq && p.normal.dot(cache[i].normal) >= NORMAL_EPS {
            p.normal_impulse = cache[i].normal_impulse * WARM_STARTING_FACTOR;
            used[i] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::body::ActivationState;

    #[test]
    fn raycast_closest_returns_the_nearest_body() {
        let mut world = PhysicsWorld::new();
        // Near sphere at x = 1, far sphere at x = 3.
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(3.0, 0.0, 0.0),
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(1.0, 0.0, 0.0),
        ));
        let (index, hit) = world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .unwrap();
        assert_eq!(index, 1);
        assert_relative_eq!(hit.fraction, 0.1, epsilon = 1e-4);
        assert_relative_eq!(hit.normal.x, -1.0, epsilon = 1e-4);
    }

    #[test]
    fn raycast_closest_misses_empty_world() {
        let world = PhysicsWorld::new();
        assert!(world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .is_none());
    }

    /// The broad-phase AABB test is conservative: a ray crossing a proxy's
    /// AABB corner must not produce a hit when the exact shape test misses.
    #[test]
    fn raycast_closest_prunes_aabb_only_hits() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::ZERO,
        ));
        // Starts on the AABB corner (0.5, 0.5, 0.5), which is outside the
        // radius-0.5 sphere: the AABB filter keeps it, the sphere test drops it.
        let hit = world.raycast_closest(
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(0.6, 0.6, 0.6),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn convex_sweep_closest_hits_the_nearer_body() {
        let mut world = PhysicsWorld::new();
        // Two boxes on the path: a near one (face at x = 3) and a far one
        // (face at x = 4). The probe starts at x = 8 and travels -X, so the
        // far box (closer to the start) is the closest hit.
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 1.0, 1.0)),
            Vector3::new(2.0, 0.0, 0.0),
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 1.0, 1.0)),
            Vector3::new(5.0, 0.0, 0.0),
        ));
        // Sweep a small box from x = 8 to x = -2 (length 10). The far box
        // spans x in [4, 6]; its +X face at x = 6 is hit by the probe's -X
        // face at x - 0.5: entry when x = 6.5, fraction (8 - 6.5) / 10 = 0.15.
        let probe = bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5));
        let (index, hit) = world
            .convex_sweep_closest(
                &probe,
                Vector3::new(8.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(-2.0, 0.0, 0.0),
                None,
            )
            .unwrap();
        assert_eq!(index, 1);
        assert_relative_eq!(hit.fraction, (8.0 - 6.5) / 10.0, epsilon = 1e-4);
        assert_relative_eq!(hit.normal.x, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn convex_sweep_closest_can_ignore_a_body() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(1.0)),
            Vector3::ZERO,
        ));
        let probe = bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5));
        // Sweeping into the box: hit when not ignored...
        assert!(world
            .convex_sweep_closest(
                &probe,
                Vector3::new(5.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(0.0, 0.0, 0.0),
                None,
            )
            .is_some());
        // ...and a miss when the only body is ignored.
        assert!(world
            .convex_sweep_closest(
                &probe,
                Vector3::new(5.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(0.0, 0.0, 0.0),
                Some(0),
            )
            .is_none());
    }

    #[test]
    fn contact_test_reports_overlapping_bodies() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(1.0)),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(1.0)),
            Vector3::new(5.0, 0.0, 0.0),
        ));
        let probe = bullet_collision::shapes::CollisionShape::sphere(0.5);
        // The probe overlaps the first box (probe x-extent [0.4, 1.4] vs box
        // x-extent [-1, 1]) but not the second.
        let hits = world.contact_test(&probe, Vector3::new(0.9, 0.0, 0.0), Quaternion::IDENTITY, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
        assert!(hits[0].1.points.iter().any(|p| p.penetration > 0.0));
        // The probe itself can be ignored.
        assert!(world
            .contact_test(&probe, Vector3::new(0.9, 0.0, 0.0), Quaternion::IDENTITY, Some(0))
            .is_empty());
    }

    /// A fresh contact point near a cached one keeps its impulse (scaled by
    /// the warm-starting factor); a distant point is a new contact from zero.
    #[test]
    fn warm_start_inherits_cached_impulses() {
        let cache = [CachedContact {
            position: Vector3::ZERO,
            normal: Vector3::Y,
            normal_impulse: 10.0,
        }];
        let mut near = [bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.01, 0.0, 0.0),
            Vector3::Y,
            0.1,
        )];
        warm_start_impulses(&mut near, Some(&cache));
        assert_relative_eq!(near[0].normal_impulse, 8.5, epsilon = 1e-4);
        // Friction is not inherited (rebuilds from zero each step).
        assert_eq!(near[0].tangent_impulse, [0.0, 0.0]);

        let mut far = [bullet_collision::manifold::ContactPoint::new(
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::Y,
            0.1,
        )];
        warm_start_impulses(&mut far, Some(&cache));
        assert_eq!(far[0].normal_impulse, 0.0);
        assert_eq!(far[0].tangent_impulse, [0.0, 0.0]);

        // A nearby point whose normal rotated too much (e.g. a box crossing a
        // triangle-mesh ridge) must not inherit the cached impulse.
        let mut rotated = [bullet_collision::manifold::ContactPoint::new(
            Vector3::new(0.01, 0.0, 0.0),
            Vector3::X,
            0.1,
        )];
        warm_start_impulses(&mut rotated, Some(&cache));
        assert_eq!(rotated[0].normal_impulse, 0.0);
    }

    /// A resting body keeps a supporting contact impulse in the cache across
    /// steps (the warm-start input for the next step).
    #[test]
    fn step_simulation_persists_contact_impulses() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
        ));
        for _ in 0..90 {
            world.step_simulation(1.0 / 60.0);
        }
        let cached: Vec<&CachedContact> = world.contact_cache.values().flatten().collect();
        assert!(!cached.is_empty());
        assert!(cached.iter().any(|c| c.normal_impulse > 0.0));
        let sphere = world.body(1);
        assert_relative_eq!(sphere.position.y, 0.5, epsilon = 0.05);
        assert!(sphere.linear_velocity.length() < 0.1);
    }

    /// A body at rest for longer than `DEACTIVATION_TIME` goes to sleep and
    /// stops being integrated.
    #[test]
    fn body_sleeps_after_resting() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 2.0, 0.0),
            1.0,
        ));
        // 3 seconds > DEACTIVATION_TIME: the sphere settles and then sleeps.
        let dt: Real = 1.0 / 60.0;
        for _ in 0..(3.0 / dt) as usize {
            world.step_simulation(dt);
        }
        let sphere = world.body(1);
        assert_eq!(sphere.activation_state(), ActivationState::IslandSleeping);
        assert_eq!(sphere.linear_velocity, Vector3::ZERO);
        let y_at_sleep = sphere.position.y;
        // Further steps must not move the sleeping body.
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert_relative_eq!(world.body(1).position.y, y_at_sleep, epsilon = 1e-6);
    }

    /// A sleeping body in contact with an active body is woken up (Bullet's
    /// activation propagation along the contact graph).
    #[test]
    fn contact_wakes_sleeping_body() {
        let mut world = PhysicsWorld::new();
        // A light sphere resting on the plane goes to sleep first...
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        let dt: Real = 1.0 / 60.0;
        for _ in 0..(3.0 / dt) as usize {
            world.step_simulation(dt);
        }
        assert_eq!(world.body(1).activation_state(), ActivationState::IslandSleeping);
        // ...then a fast ball hits it: the sleeping sphere must wake up.
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.25),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, -5.0, 0.0)));
        for _ in 0..60 {
            world.step_simulation(dt);
        }
        assert_eq!(
            world.body(1).activation_state(),
            ActivationState::Active,
            "sleeping body was not woken by the impact"
        );
    }

    /// Applying a force wakes a sleeping body up (Bullet's `applyCentralForce`).
    #[test]
    fn force_wakes_sleeping_body() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        let dt: Real = 1.0 / 60.0;
        for _ in 0..(3.0 / dt) as usize {
            world.step_simulation(dt);
        }
        world.body_mut(1).apply_force(Vector3::new(0.0, 5.0, 0.0));
        assert_eq!(world.body(1).activation_state(), ActivationState::Active);
    }

    /// Static bodies start asleep, like Bullet's static `btRigidBody`.
    #[test]
    fn static_body_starts_sleeping() {
        let body = RigidBody::new_static(bullet_collision::shapes::CollisionShape::sphere(0.5), Vector3::ZERO);
        assert_eq!(body.activation_state(), ActivationState::IslandSleeping);
        assert!(!body.is_active());
    }

    /// CCD: a fast body whose one-step displacement would cross a thin static
    /// wall is clamped at the swept-sphere impact instead of tunneling; the
    /// same body without CCD sails straight through.
    #[test]
    fn ccd_clamps_fast_bodies_at_the_impact() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        // Thin static wall: box centered at y = 0.5, spanning y in [0.4, 0.6].
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
            Vector3::new(0.0, 0.5, 0.0),
        ));
        // Fast ball with CCD, starting above the wall; per-step motion is
        // 2.0 > threshold 0.1 and the predicted position (y = -1) is past the
        // wall, so the discrete narrow phase sees no contact.
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0))
            .ccd(0.1, 0.3),
        );
        // Control ball without CCD, offset in X so it misses the wall.
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(2.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -120.0, 0.0)),
        );
        let dt: Real = 1.0 / 60.0;
        world.step_simulation(dt);
        // The CCD ball is clamped at the wall's top surface (0.6 + 0.3).
        assert_relative_eq!(world.body(1).position.y, 0.9, epsilon = 0.05);
        // The control ball crossed the whole wall.
        assert!(world.body(2).position.y < -0.5);
    }

    /// CCD only sweeps when the per-step motion exceeds the threshold; slower
    /// bodies integrate normally.
    #[test]
    fn ccd_ignores_slow_motion() {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vector3::ZERO);
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
            Vector3::ZERO,
        ));
        world.add_body(
            RigidBody::new_dynamic(
                bullet_collision::shapes::CollisionShape::sphere(0.3),
                Vector3::new(0.0, 1.0, 0.0),
                1.0,
            )
            .velocity(Vector3::new(0.0, -5.0, 0.0))
            .ccd(1.0, 0.3),
        );
        // Per-step motion 5/60 < threshold 1.0: plain integration, no clamp.
        world.step_simulation(1.0 / 60.0);
        assert_relative_eq!(world.body(1).position.y, 1.0 - 5.0 / 60.0, epsilon = 1e-5);
    }
}
