//! Query surface of `PhysicsWorld` (ray casts, sweeps, `contact_test`,
// `set_body_transform`), split out of `world.rs`. Child modules see the
// parent's private fields, so the impl blocks move without any visibility
// changes.

use super::*;

impl PhysicsWorld {

    /// Cast a ray against every body and return the closest hit together with
    /// the body handle (Bullet's `rayTest` with a closest-result callback).
    ///
    /// The broad phase first prunes proxies whose AABB the segment does not
    /// cross, so the (more expensive) exact per-shape test only runs on
    /// candidates — mirroring `btAxisSweep3::rayTest` feeding a
    /// `ClosestRayResultCallback`. The AABB test is conservative, so the
    /// result is identical to testing every body.
    pub fn raycast_closest(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
    ) -> Option<(BodyHandle, RayHit)> {
        self.raycast_closest_excluding(ray_from, ray_to, None)
    }

    /// Closest ray hit against the soft bodies' surface faces (Bullet's
    /// per-body `btSoftBody::rayTest`): returns the soft-body handle, the
    /// face index and the segment fraction. Rigid-body hits are not
    /// considered — a combined query joins the world-API cleanup (phase E).
    pub fn raycast_soft_closest(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
    ) -> Option<(SoftBodyHandle, usize, Real)> {
        let mut best: Option<(SoftBodyHandle, usize, Real)> = None;
        for (i, sb) in self.soft_bodies.iter().enumerate() {
            if let Some((face, fraction)) = sb.ray_test(ray_from, ray_to) {
                let closer = match best {
                    Some((_, _, f)) => fraction < f,
                    None => true,
                };
                if closer {
                    best = Some((SoftBodyHandle::from_index(i), face, fraction));
                }
            }
        }
        best
    }

    /// Like [`raycast_closest`](Self::raycast_closest) but never reports
    /// `ignore` (the raycast vehicle's wheel rays, which must not hit their
    /// own chassis). Bullet skips the chassis implicitly because its
    /// interior-start rays miss; we report interior starts (deviation §1), so
    /// the exclusion is explicit here.
    pub fn raycast_closest_excluding(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        ignore: Option<BodyHandle>,
    ) -> Option<(BodyHandle, RayHit)> {
        let mut best: Option<(BodyHandle, RayHit)> = None;
        self.raycast_walk(ray_from, ray_to, ignore, RayFilter::DEFAULT, |i, hit| {
            if best.is_none_or(|(_, b)| hit.fraction < b.fraction) {
                best = Some((i, hit));
            }
        });
        best
    }

    /// Every body hit along the segment, nearest first (Bullet's `rayTest`
    /// with an all-hits callback). Bodies whose collision group/mask pair
    /// test rejects the query's [`RayFilter::DEFAULT`] group and mask are
    /// not tested.
    pub fn raycast_all(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<(BodyHandle, RayHit)> {
        self.raycast_all_filtered(ray_from, ray_to, RayFilter::DEFAULT)
    }

    /// Every hit among the bodies the `filter` accepts, nearest first.
    pub fn raycast_all_filtered(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        filter: RayFilter,
    ) -> Vec<(BodyHandle, RayHit)> {
        let mut hits: Vec<(BodyHandle, RayHit)> = Vec::new();
        self.raycast_walk(ray_from, ray_to, None, filter, |i, hit| {
            hits.push((i, hit));
        });
        hits.sort_by(|a, b| a.1.fraction.total_cmp(&b.1.fraction));
        hits
    }

    /// Closest hit among the bodies the `filter` accepts (Bullet's
    /// `rayTest` with a filtered closest-result callback).
    pub fn raycast_closest_filtered(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        filter: RayFilter,
    ) -> Option<(BodyHandle, RayHit)> {
        let mut best: Option<(BodyHandle, RayHit)> = None;
        self.raycast_walk(ray_from, ray_to, None, filter, |i, hit| {
            if best.is_none_or(|(_, b)| hit.fraction < b.fraction) {
                best = Some((i, hit));
            }
        });
        best
    }

    /// The shared ray walk: the broad phase prunes proxies whose AABB the
    /// segment does not cross, link-collider proxies are skipped (phase C),
    /// and the filter applies the symmetric group & mask test. Each hit is
    /// handed to `out` in no particular order.
    fn raycast_walk(
        &self,
        ray_from: Vector3,
        ray_to: Vector3,
        ignore: Option<BodyHandle>,
        filter: RayFilter,
        mut out: impl FnMut(BodyHandle, RayHit),
    ) {
        for p in self.broadphase.raycast_candidates(ray_from, ray_to) {
            // Link colliders join the world queries together with their
            // contact response (phase C); until then only rigid bodies are
            // tested.
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            let handle = BodyHandle::from_index(i);
            if Some(handle) == ignore {
                continue;
            }
            let body = &self.bodies[i];
            if !filter.accepts(body) {
                continue;
            }
            if let Some(hit) = raycast::raycast(
                &body.shape,
                body.position,
                body.orientation,
                ray_from,
                ray_to,
            ) {
                out(handle, hit);
            }
        }
    }

    /// Handles of the bodies whose broad-phase AABB overlaps `aabb` (exact
    /// AABB test; the caller then runs the narrow phase on the candidates).
    /// Multibody link colliders overlap too but are not reported until they
    /// join the queries in phase C.
    pub fn aabb_candidates(&self, aabb: Aabb) -> Vec<BodyHandle> {
        self.broadphase
            .aabb_candidates(aabb)
            .into_iter()
            .filter_map(|p| match self.proxy_owners[p] {
                ProxyOwner::Body(i) => Some(BodyHandle::from_index(i)),
                ProxyOwner::Collider(_) | ProxyOwner::Soft(_) | ProxyOwner::Free => None,
            })
            .collect()
    }

    /// Closest convex sweep of `shape` from `from` to `to` against every body
    /// except `ignore`, returning the hit body handle and the closest hit.
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
        ignore: Option<BodyHandle>,
    ) -> Option<(BodyHandle, SweepHit)> {
        let swept = shape
            .world_aabb(from, ori)
            .merged(&shape.world_aabb(to, ori));
        let mut best: Option<(BodyHandle, SweepHit)> = None;
        for p in self.broadphase.aabb_candidates(swept) {
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            let handle = BodyHandle::from_index(i);
            if Some(handle) == ignore {
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
                best = Some((handle, hit));
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
        ignore: Option<BodyHandle>,
    ) -> Vec<(BodyHandle, ContactManifold)> {
        let aabb = shape.world_aabb(pos, ori);
        let mut out = Vec::new();
        for p in self.broadphase.aabb_candidates(aabb) {
            let ProxyOwner::Body(i) = self.proxy_owners[p] else {
                continue;
            };
            let handle = BodyHandle::from_index(i);
            if Some(handle) == ignore {
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
                out.push((handle, m));
            }
        }
        out
    }

    /// Set a body's pose directly and refresh its broad-phase AABB. Used by
    /// the character controller to move its kinematic body; the solver still
    /// treats the body as immovable (static bodies are never integrated).
    pub fn set_body_transform(
        &mut self,
        handle: BodyHandle,
        position: Vector3,
        orientation: Quaternion,
    ) {
        let index = handle.index();
        assert!(
            !self.bodies[index].is_removed(),
            "body {} has been removed",
            index
        );
        let body = &mut self.bodies[index];
        body.position = position;
        body.orientation = orientation;
        let aabb = body
            .shape
            .world_aabb(position, orientation)
            .expanded(BROADPHASE_AABB_MARGIN);
        self.broadphase.set_aabb(self.body_proxies[index], aabb);
    }


}