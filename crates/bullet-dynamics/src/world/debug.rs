//! Debug draw: read-only diagnostic primitives. The producer side of the
//! design in `docs/design/debug-draw.md`: one method walks the world's
//! authoritative state and pushes borrowed, kind-tagged physics facts
//! through a caller callback. The world gains no debug state, the emit
//! path allocates nothing, and drawing never affects simulation.

use super::*;
use bullet_collision::shapes::CollisionShape;
use crate::body::ActivationState;
use crate::constraint::ConstraintKind;

/// Categories of debug output: [`PhysicsWorld::debug_primitives`] emits one
/// stream of primitives, and each flag enables one domain. A hand-written
/// bit set in the style of the engine's other flag newtypes
/// ([`CollisionFlags`]); an empty mask emits — and scans — nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DebugDrawFlags(pub u16);

impl DebugDrawFlags {
    /// No domains: emit nothing (the call returns before scanning).
    pub const NONE: Self = Self(0);

    /// Rigid-body collider geometry at world poses (`Shape` prims; static,
    /// kinematic and sleeping bodies included).
    pub const SHAPES: Self = Self(1 << 0);
    /// The stored broad-phase AABB of every live proxy — rigid bodies,
    /// multibody link colliders and soft bodies (`Aabb` prims).
    pub const AABBS: Self = Self(1 << 1);
    /// The persistent-manifold contact points of every cached rigid pair,
    /// with normal and last-solve load (`Contact` prims).
    pub const CONTACTS: Self = Self(1 << 2);
    /// The world-space anchors of every rigid joint (`Constraint` prims).
    pub const CONSTRAINTS: Self = Self(1 << 3);
    /// The per-body activation states (`Activation` prims).
    pub const SLEEPING: Self = Self(1 << 4);

    /// Every domain.
    pub const ALL: Self = Self(0b11111);

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no domain is enabled.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for DebugDrawFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for DebugDrawFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Which rigid-joint kind a [`DebugPrim::Constraint`] describes: a
/// payload-free, `Copy` mirror of [`ConstraintKind`], so a renderer can
/// style joints without carrying the joint parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintTag {
    Point2Point,
    Hinge,
    Fixed,
    ConeTwist,
    Slider,
    Generic6Dof,
}

/// One debug primitive: a *physical fact* read from the world's
/// authoritative state, never a render operation. The variant is the kind a
/// renderer styles by; the core carries no colors, widths or materials.
/// Prims are valid only for the duration of the callback invocation — the
/// `Shape` variant borrows the body's shape instead of cloning it.
#[non_exhaustive]
#[derive(Debug)]
pub enum DebugPrim<'a> {
    /// SHAPES — a rigid body's collider geometry at its world pose.
    Shape {
        /// The owning rigid body.
        object: CollisionObject,
        position: Vector3,
        orientation: Quaternion,
        shape: &'a CollisionShape,
    },
    /// AABBS — the bounds the broad phase currently stores for one live
    /// proxy: exactly the bounds pair candidacy runs on.
    Aabb {
        /// The proxy owner (rigid body, multibody link collider or soft
        /// body).
        object: CollisionObject,
        aabb: Aabb,
    },
    /// CONTACTS — one persistent-manifold contact point, exactly as the
    /// solver's warm-start cache holds it.
    Contact {
        /// Pair side A (the manifold's own A; the normal points A -> B).
        a: CollisionObject,
        /// Pair side B.
        b: CollisionObject,
        /// The B-side anchor re-projected onto the body's current pose
        /// (`wb = pos_b + ori_b * local_b`) — the same quantity the
        /// manifold refresh hands the solver as the contact position.
        position: Vector3,
        /// The cached unit contact normal, A -> B.
        normal: Vector3,
        /// The cached signed separation, positive = penetrating (values in
        /// `(-threshold, 0]` are margin-band contacts).
        penetration: Real,
        /// The accumulated normal impulse from the last solve.
        normal_impulse: Real,
    },
    /// CONSTRAINTS — one rigid joint, both anchors in world space.
    Constraint {
        a: BodyHandle,
        b: BodyHandle,
        kind: ConstraintTag,
        anchor_a: Vector3,
        anchor_b: Vector3,
    },
    /// SLEEPING — a rigid body's activation state at its origin.
    Activation {
        object: BodyHandle,
        position: Vector3,
        state: ActivationState,
    },
}

impl PhysicsWorld {
    /// Emit debug primitives for every enabled domain, in no particular
    /// order (the consumer must not depend on it). Read-only observation:
    /// drawing never affects simulation — the formal contract and its
    /// regression tests live in `docs/design/debug-draw.md` §19. An empty
    /// mask returns before scanning anything.
    pub fn debug_primitives<'s>(
        &'s self,
        flags: DebugDrawFlags,
        emit: &mut impl FnMut(DebugPrim<'s>),
    ) {
        if flags.is_empty() {
            return;
        }

        if flags.contains(DebugDrawFlags::SHAPES) {
            for (i, body) in self.bodies.iter().enumerate() {
                if body.is_removed() {
                    continue;
                }
                emit(DebugPrim::Shape {
                    object: CollisionObject::Body(BodyHandle::from_index(i)),
                    position: body.position,
                    orientation: body.orientation,
                    shape: &body.shape,
                });
            }
        }

        if flags.contains(DebugDrawFlags::AABBS) {
            for (proxy, owner) in self.proxy_owners.iter().enumerate() {
                let object = match *owner {
                    ProxyOwner::Body(i) => CollisionObject::Body(BodyHandle::from_index(i)),
                    ProxyOwner::Collider(c) => {
                        let entry = &self.collider_entries[c];
                        CollisionObject::Link {
                            body: MultiBodyHandle::from_index(entry.mb),
                            link: entry.link,
                        }
                    }
                    ProxyOwner::Soft(s) => CollisionObject::Soft(SoftBodyHandle::from_index(s)),
                    ProxyOwner::Free => continue,
                };
                if let Some(aabb) = self.broadphase.proxy_aabb(proxy) {
                    emit(DebugPrim::Aabb { object, aabb });
                }
            }
        }

        if flags.contains(DebugDrawFlags::CONTACTS) {
            // The cache is an adjacency structure: bucket `lo` holds every
            // pair `(lo, other)`, and the manifold's body A is `lo`, body B
            // is `entry.other` — both broad phases report pairs in the
            // normalized `(min, max)` order, so the cached anchors' local
            // frames stay on fixed sides across steps.
            for (lo, bucket) in self.manifold_cache.iter().enumerate() {
                if bucket.is_empty() {
                    continue;
                }
                if self.bodies[lo].is_removed() {
                    continue;
                }
                for entry in bucket {
                    let b = &self.bodies[entry.other];
                    if b.is_removed() {
                        continue;
                    }
                    let a_obj = CollisionObject::Body(BodyHandle::from_index(lo));
                    let b_obj = CollisionObject::Body(BodyHandle::from_index(entry.other));
                    for point in entry.manifold.points() {
                        let position = b.position + b.orientation * point.local_b;
                        emit(DebugPrim::Contact {
                            a: a_obj,
                            b: b_obj,
                            position,
                            normal: point.normal,
                            penetration: point.penetration,
                            normal_impulse: point.normal_impulse,
                        });
                    }
                }
            }
        }

        if flags.contains(DebugDrawFlags::CONSTRAINTS) {
            for constraint in &self.constraints {
                let (tag, local_a, local_b) = match constraint.kind() {
                    ConstraintKind::Point2Point(p) => {
                        (ConstraintTag::Point2Point, p.pivot_a, p.pivot_b)
                    }
                    ConstraintKind::Hinge(h) => {
                        (ConstraintTag::Hinge, h.frame_a.origin, h.frame_b.origin)
                    }
                    ConstraintKind::Fixed(f) => {
                        (ConstraintTag::Fixed, f.frame_a.origin, f.frame_b.origin)
                    }
                    ConstraintKind::ConeTwist(c) => {
                        (ConstraintTag::ConeTwist, c.frame_a.origin, c.frame_b.origin)
                    }
                    ConstraintKind::Slider(s) => {
                        (ConstraintTag::Slider, s.frame_a.origin, s.frame_b.origin)
                    }
                    ConstraintKind::Generic6Dof(g) => {
                        (ConstraintTag::Generic6Dof, g.frame_a.origin, g.frame_b.origin)
                    }
                };
                let (a, b) = (constraint.body_a(), constraint.body_b());
                let body_a = &self.bodies[a.index()];
                let body_b = &self.bodies[b.index()];
                emit(DebugPrim::Constraint {
                    a,
                    b,
                    kind: tag,
                    anchor_a: body_a.position + body_a.orientation * local_a,
                    anchor_b: body_b.position + body_b.orientation * local_b,
                });
            }
        }

        if flags.contains(DebugDrawFlags::SLEEPING) {
            for (i, body) in self.bodies.iter().enumerate() {
                if body.is_removed() {
                    continue;
                }
                emit(DebugPrim::Activation {
                    object: BodyHandle::from_index(i),
                    position: body.position,
                    state: body.activation_state(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ground box (top face at y = 0) with a sphere resting in its margin
    /// band, so contacts exist from the first detection on.
    fn contact_world() -> (PhysicsWorld, BodyHandle, BodyHandle) {
        let mut world = PhysicsWorld::new();
        let ground = world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::new(0.0, -0.5, 0.0),
        ));
        let ball = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 0.51, 0.0),
            1.0,
        ));
        (world, ground, ball)
    }

    fn collect<'a>(
        world: &'a PhysicsWorld,
        flags: DebugDrawFlags,
    ) -> Vec<DebugPrim<'a>> {
        let mut out: Vec<DebugPrim<'a>> = Vec::new();
        world.debug_primitives(flags, &mut |prim| out.push(prim));
        out
    }

    fn count<'a>(prims: &'a [DebugPrim<'a>]) -> (usize, usize, usize, usize, usize) {
        let mut counts = (0, 0, 0, 0, 0);
        for prim in prims {
            match prim {
                DebugPrim::Shape { .. } => counts.0 += 1,
                DebugPrim::Aabb { .. } => counts.1 += 1,
                DebugPrim::Contact { .. } => counts.2 += 1,
                DebugPrim::Constraint { .. } => counts.3 += 1,
                DebugPrim::Activation { .. } => counts.4 += 1,
            }
        }
        counts
    }

    fn assert_bits_eq(a: Real, b: Real) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    fn assert_vec_bits_eq(a: Vector3, b: Vector3) {
        // Component-wise: the f32 vector type compares all lanes.
        assert_bits_eq(a.x, b.x);
        assert_bits_eq(a.y, b.y);
        assert_bits_eq(a.z, b.z);
    }

    fn assert_quat_bits_eq(a: Quaternion, b: Quaternion) {
        assert_bits_eq(a.x, b.x);
        assert_bits_eq(a.y, b.y);
        assert_bits_eq(a.z, b.z);
        assert_bits_eq(a.w, b.w);
    }

    /// An empty (or default) mask emits zero primitives, and a
    /// single-domain mask emits only that domain's primitives.
    #[test]
    fn empty_mask_emits_nothing() {
        let (mut world, _ground, _ball) = contact_world();
        assert!(collect(&world, DebugDrawFlags::NONE).is_empty());
        assert!(collect(&world, DebugDrawFlags::default()).is_empty());

        // One step so manifolds and proxies exist, then check gating.
        world.step_simulation(1.0 / 60.0);
        let shapes = collect(&world, DebugDrawFlags::SHAPES);
        assert_eq!(count(&shapes), (2, 0, 0, 0, 0));
        for prim in collect(&world, DebugDrawFlags::CONTACTS) {
            assert!(matches!(prim, DebugPrim::Contact { .. }));
        }
        for prim in collect(&world, DebugDrawFlags::AABBS) {
            assert!(matches!(prim, DebugPrim::Aabb { .. }));
        }
    }

    /// SHAPES covers every live rigid body (static included) and borrows
    /// the body's own shape (no per-prim cloning).
    #[test]
    fn shape_prims_borrow_the_body_shape() {
        let (world, ground, ball) = contact_world();
        let prims = collect(&world, DebugDrawFlags::SHAPES);
        assert_eq!(prims.len(), 2);
        for prim in &prims {
            match prim {
                DebugPrim::Shape {
                    object,
                    position,
                    orientation,
                    shape,
                } => {
                    let handle = match object {
                        CollisionObject::Body(h) => *h,
                        _ => panic!("shapes are rigid bodies"),
                    };
                    let body = world.body(handle);
                    assert!(core::ptr::eq(*shape, &body.shape));
                    assert_vec_bits_eq(*position, body.position);
                    assert_quat_bits_eq(*orientation, body.orientation);
                }
                _ => panic!("expected a shape prim"),
            }
        }
        let handles: Vec<_> = prims
            .iter()
            .map(|p| match p {
                DebugPrim::Shape {
                    object: CollisionObject::Body(h),
                    ..
                } => *h,
                _ => unreachable!(),
            })
            .collect();
        assert!(handles.contains(&ground));
        assert!(handles.contains(&ball));
    }

    /// AABBS reports, per live proxy, exactly the bounds the broad phase
    /// stores — for rigid bodies and multibody link colliders alike.
    #[test]
    fn aabb_prims_match_the_broadphase_bounds() {
        let mut world = PhysicsWorld::new();
        world.add_body(RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::new(0.0, -0.5, 0.0),
        ));
        let mut mb = MultiBody::new(0, 1.0, Vector3::ONE, true, false);
        mb.finalize_multidof();
        mb.set_base_collider(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.25, 1.0)),
            0.5,
            0.0,
        );
        mb.base_pos = Vector3::new(0.0, 2.0, 0.0);
        world.add_multibody(mb);

        world.step_simulation(1.0 / 60.0);

        let prims = collect(&world, DebugDrawFlags::AABBS);
        assert_eq!(prims.len(), 2, "one proxy per rigid body and mb collider");
        for prim in &prims {
            match prim {
                DebugPrim::Aabb { object, aabb } => {
                    let proxy = match object {
                        CollisionObject::Body(h) => world.body_proxies[h.index()],
                        CollisionObject::Link { .. } => {
                            world.collider_entries[0].proxy.expect("registered collider")
                        }
                        _ => panic!("unexpected soft owner"),
                    };
                    let stored = world.broadphase.proxy_aabb(proxy).expect("live proxy");
                    assert_eq!(aabb.min, stored.min);
                    assert_eq!(aabb.max, stored.max);
                }
                _ => panic!("expected an aabb prim"),
            }
        }
        // The link collider shows up with its Link identity (base = -1).
        assert!(prims.iter().any(|p| matches!(
            p,
            DebugPrim::Aabb {
                object: CollisionObject::Link { link: -1, .. },
                ..
            }
        )));
    }

    /// CONTACTS mirrors the persistent-manifold cache: one prim per cached
    /// point, fields exactly as cached, position re-projected onto the
    /// current B pose, A/B identities matching the manifold's sides.
    #[test]
    fn contact_prims_mirror_the_manifold_cache() {
        let (mut world, ground, ball) = contact_world();
        for _ in 0..10 {
            world.step_simulation(1.0 / 60.0);
        }
        let cached: usize = world
            .manifold_cache
            .iter()
            .map(|bucket| bucket.iter().map(|e| e.manifold.points().len()).sum::<usize>())
            .sum();
        assert!(cached > 0, "the resting pair should hold manifold points");

        let prims = collect(&world, DebugDrawFlags::CONTACTS);
        assert_eq!(prims.len(), cached);
        for prim in &prims {
            match prim {
                DebugPrim::Contact {
                    a,
                    b,
                    position,
                    normal,
                    penetration,
                    normal_impulse,
                } => {
                    assert_eq!(*a, CollisionObject::Body(ground));
                    assert_eq!(*b, CollisionObject::Body(ball));
                    // Normal is a unit vector.
                    let len_sq = normal.length_squared();
                    assert!((len_sq - 1.0).abs() < 1e-5);
                    // The cache side holds this exact point with this exact
                    // projection (A = bucket index, B = entry.other).
                    let lo = ground.index();
                    let entry = world.manifold_cache[lo]
                        .iter()
                        .find(|e| e.other == ball.index())
                        .expect("the pair's cache entry");
                    let matched = entry.manifold.points().iter().any(|p| {
                        let wb = world.bodies[ball.index()].position
                            + world.bodies[ball.index()].orientation * p.local_b;
                        *position == wb && *normal == p.normal
                            && *penetration == p.penetration
                            && *normal_impulse == p.normal_impulse
                    });
                    assert!(matched, "prim must be a cached point, re-projected");
                }
                _ => panic!("expected a contact prim"),
            }
        }
    }

    /// CONSTRAINTS projects each joint's frame anchors with the bodies'
    /// current poses and reports the right kind tag.
    #[test]
    fn constraint_prims_project_frame_anchors() {
        let mut world = PhysicsWorld::new();
        let a = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        ));
        let b = world.add_body(RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 2.02, 0.0),
            1.0,
        ));
        let pivot_a = Vector3::new(0.0, 0.25, 0.0);
        let pivot_b = Vector3::new(0.0, -0.25, 0.0);
        world.add_constraint(Constraint::point2point(a, b, pivot_a, pivot_b));

        let prims = collect(&world, DebugDrawFlags::CONSTRAINTS);
        assert_eq!(prims.len(), 1);
        match &prims[0] {
            DebugPrim::Constraint {
                a: ca,
                b: cb,
                kind,
                anchor_a,
                anchor_b,
            } => {
                assert_eq!((*ca, *cb), (a, b));
                assert_eq!(*kind, ConstraintTag::Point2Point);
                let body_a = world.body(a);
                let body_b = world.body(b);
                assert_vec_bits_eq(*anchor_a, body_a.position + body_a.orientation * pivot_a);
                assert_vec_bits_eq(*anchor_b, body_b.position + body_b.orientation * pivot_b);
            }
            _ => panic!("expected a constraint prim"),
        }

        // A hinge reads the frame origins instead of the p2p pivots.
        let hinge = Constraint::hinge(
            a,
            b,
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
        );
        world.add_constraint(hinge);
        let prims = collect(&world, DebugDrawFlags::CONSTRAINTS);
        assert_eq!(prims.len(), 2);
        assert!(prims.iter().any(|p| matches!(
            p,
            DebugPrim::Constraint {
                kind: ConstraintTag::Hinge,
                ..
            }
        )));
    }

    /// SLEEPING reports each body's activation state at its origin.
    #[test]
    fn sleeping_prims_report_activation_states() {
        let (world, _ground, _ball) = contact_world();
        let prims = collect(&world, DebugDrawFlags::SLEEPING);
        assert_eq!(prims.len(), 2);
        for prim in &prims {
            match prim {
                DebugPrim::Activation {
                    object,
                    position,
                    state,
                } => {
                    let body = world.body(*object);
                    assert_eq!(*state, body.activation_state());
                    assert_vec_bits_eq(*position, body.position);
                }
                _ => panic!("expected an activation prim"),
            }
        }
        assert!(prims.iter().any(|p| matches!(
            p,
            DebugPrim::Activation {
                state: ActivationState::IslandSleeping,
                ..
            }
        )));
        assert!(prims.iter().any(|p| matches!(
            p,
            DebugPrim::Activation {
                state: ActivationState::Active,
                ..
            }
        )));
    }

    /// Removed bodies leak into no domain.
    #[test]
    fn removed_bodies_are_skipped() {
        let (mut world, _ground, ball) = contact_world();
        world.step_simulation(1.0 / 60.0);
        world.remove_body(ball);
        let prims = collect(&world, DebugDrawFlags::ALL);
        for prim in &prims {
            match prim {
                DebugPrim::Shape { object, .. } => {
                    assert_ne!(*object, CollisionObject::Body(ball));
                }
                DebugPrim::Aabb { object, .. } => {
                    assert_ne!(*object, CollisionObject::Body(ball));
                }
                DebugPrim::Activation { object, .. } => {
                    assert_ne!(*object, ball);
                }
                DebugPrim::Contact { a, b, .. } => {
                    assert_ne!(*a, CollisionObject::Body(ball));
                    assert_ne!(*b, CollisionObject::Body(ball));
                }
                _ => {}
            }
        }
    }

    /// The invariance contract (design doc §19): two identical worlds, one
    /// drawing after every step and one never drawing, stay bit-identical —
    /// poses, velocities, activation states and event streams alike.
    #[test]
    fn drawing_does_not_change_the_simulation() {
        fn scene() -> (PhysicsWorld, BodyHandle, BodyHandle) {
            let (mut world, ground, ball) = contact_world();
            world.add_constraint(Constraint::point2point(
                ground,
                ball,
                Vector3::ZERO,
                Vector3::new(0.0, 0.5, 0.0),
            ));
            (world, ground, ball)
        }

        let (mut drawing, _, _) = scene();
        let (mut silent, _, _) = scene();
        let dt = 1.0 / 60.0;
        for _ in 0..30 {
            drawing.step_simulation(dt);
            silent.step_simulation(dt);
            // Draw every domain on one world, nothing on the other.
            let prims = collect(&drawing, DebugDrawFlags::ALL);
            assert!(!prims.is_empty());
            assert_eq!(drawing.contact_events(), silent.contact_events());
        }
        for slot in 0..drawing.bodies.len() {
            let a = &drawing.bodies[slot];
            let b = &silent.bodies[slot];
            assert_eq!(a.is_removed(), b.is_removed());
            if a.is_removed() {
                continue;
            }
            assert_vec_bits_eq(a.position, b.position);
            assert_quat_bits_eq(a.orientation, b.orientation);
            assert_vec_bits_eq(a.linear_velocity, b.linear_velocity);
            assert_vec_bits_eq(a.angular_velocity, b.angular_velocity);
            assert_eq!(a.activation_state(), b.activation_state());
        }
    }
}
