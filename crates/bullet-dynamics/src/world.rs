use bullet_collision::broadphase::{BroadPhase, BruteForceBroadPhase};
use bullet_collision::narrowphase;
use bullet_collision::raycast::{self, RayHit};
use crate::body::RigidBody;
use bullet_math::{Real, Vector3};
use crate::solver::sequential_impulse::{self, ContactConstraint};

/// A discrete rigid-body simulation world, mirroring `btDiscreteDynamicsWorld`.
pub struct PhysicsWorld {
    bodies: Vec<RigidBody>,
    broadphase: BruteForceBroadPhase,
    gravity: Vector3,
    iterations: u32,
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
            broadphase: BruteForceBroadPhase::new(),
            gravity: Vector3::new(0.0, -9.81, 0.0),
            iterations: 10,
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
    /// Bodies are tested in insertion order with a brute-force walk; an
    /// `AxisSweep3` broad phase can later prune candidates via its ray test.
    pub fn raycast_closest(&self, ray_from: Vector3, ray_to: Vector3) -> Option<(usize, RayHit)> {
        let mut best: Option<(usize, RayHit)> = None;
        for (i, body) in self.bodies.iter().enumerate() {
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
            if let Some(manifold) = narrowphase::collide(
                &a.shape,
                predicted[i],
                a.orientation,
                &b.shape,
                predicted[j],
                b.orientation,
            ) {
                for cp in &manifold.points {
                    constraints.push(ContactConstraint::prepare(i, j, cp, a, b));
                }
            }
        }

        // 6. Solve constraints (correct velocities).
        sequential_impulse::solve(&mut self.bodies, &mut constraints, self.iterations, inv_dt);

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
}
