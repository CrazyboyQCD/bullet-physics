use std::collections::HashMap;

use bullet_collision::broadphase::{AxisSweep3, BroadPhase};
use bullet_collision::narrowphase;
use bullet_collision::raycast::{self, RayHit};
use crate::body::RigidBody;
use bullet_math::{Real, Vector3};
use crate::solver::sequential_impulse::{self, ContactConstraint};

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

        // 3. Update broad-phase AABBs from predicted poses.
        for (i, body) in self.bodies.iter().enumerate() {
            let aabb = body.shape.world_aabb(predicted[i], body.orientation);
            self.broadphase.set_aabb(i, aabb);
        }

        // 4. Broad phase: candidate overlapping pairs.
        let pairs = self.broadphase.compute_pairs();

        // 5. Narrow phase: exact contact manifolds -> constraints.
        let mut constraints: Vec<ContactConstraint> = Vec::new();
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

        // 7. Integrate positions with the corrected velocities.
        for body in &mut self.bodies {
            body.integrate_position(dt);
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
}
