//! Tests for the fused `PhysicsWorld` step pipeline (rigid bodies,
//! constraints, CCD, ray and sweep queries, collision groups, trigger
//! volumes, collision events, soft bodies). Moved verbatim out of
//! `world.rs` (the file had grown past 4600 lines); content unchanged.

use crate::body::ActivationState;
use crate::body::filter_groups;
use crate::world::*;
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
    assert_eq!(index, BodyHandle::from_index(1));
    assert_relative_eq!(hit.fraction, 0.1, epsilon = 1e-4);
    assert_relative_eq!(hit.normal.x, -1.0, epsilon = 1e-4);
}

/// Damping is applied automatically by the step, before the force
/// integration (Bullet's `predictUnconstraintMotion`): with factor 0.5
/// and one second of 1/60 s steps the speed halves.
#[test]
fn damping_applied_during_step() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let b = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 10.0, 0.0),
        1.0,
    ));
    world.body_mut(b).linear_velocity = Vector3::new(10.0, 0.0, 0.0);
    world.body_mut(b).set_damping(0.5, 0.0);
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step_simulation(dt);
    }
    assert_relative_eq!(world.body(b).linear_velocity.x, 5.0, epsilon = 1e-4);
}

// ------------------------------------------------------------------
// Collision groups/masks, trigger volumes, collision events.
// ------------------------------------------------------------------

/// A plate whose mask matches nothing is invisible to the ball: the
/// ball falls straight through (Bullet's `btBroadphaseProxy`
/// group/mask pair test).
#[test]
fn collision_mask_blocks_pair() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        )
        .collision_filter(filter_groups::DEFAULT, 0),
    );
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 3.0, 0.0),
        1.0,
    ));
    let dt = 1.0 / 60.0;
    for _ in 0..90 {
        world.step_simulation(dt);
    }
    // Through the plate (its top is y = 0.5) and far below it.
    assert!(
        world.body(ball).position.y < -1.0,
        "ball did not fall through: y = {}",
        world.body(ball).position.y
    );
}

/// The pair test is symmetric: a plate in the STATIC group is ignored
/// by a ball whose mask only accepts DEFAULT, and collides again once
/// the mask widens to ALL.
#[test]
fn collision_group_mask_asymmetric() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        )
        .collision_filter(filter_groups::STATIC, filter_groups::ALL),
    );
    let ball = world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.01, 0.0),
            1.0,
        )
        .collision_filter(filter_groups::DEFAULT, filter_groups::DEFAULT),
    );
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step_simulation(dt);
    }
    assert!(
        world.body(ball).position.y < 0.0,
        "ball should ignore the STATIC plate: y = {}",
        world.body(ball).position.y
    );
    // Widen the mask, put the ball back above the plate: the pair
    // collides and the ball settles on top.
    world.body_mut(ball).collision_mask = filter_groups::ALL;
    world.set_body_transform(ball, Vector3::new(0.0, 1.01, 0.0), Quaternion::IDENTITY);
    for _ in 0..180 {
        world.step_simulation(dt);
    }
    assert_relative_eq!(world.body(ball).position.y, 1.0, epsilon = 0.05);
}

/// A trigger volume (`NO_CONTACT_RESPONSE`) never blocks: the ball
/// falls straight through, and its touch still reports as a sensor
/// event.
#[test]
fn trigger_volume_passes_through_but_reports() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        )
        .no_contact_response(true)
        .collision_events(true),
    );
    let ball = world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        )
        .collision_events(true),
    );
    let dt = 1.0 / 60.0;
    let mut started = None;
    for _ in 0..120 {
        world.step_simulation(dt);
        for &ev in world.contact_events() {
            if let CollisionEvent::Started { a, b, sensor } = ev {
                started = Some((a, b, sensor));
            }
        }
    }
    assert_eq!(started, Some((BodyHandle::from_index(0), ball, true)));
    assert!(
        world.body(ball).position.y < -1.0,
        "trigger did not let the ball through: y = {}",
        world.body(ball).position.y
    );
}

/// Pairs emit events only when at least one body opts in.
#[test]
fn collision_events_require_opt_in() {
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
        Vector3::ZERO,
    ));
    world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 1.01, 0.0),
        1.0,
    ));
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step_simulation(dt);
    }
    assert!(world.contact_events().is_empty());
}

/// A resting pair that falls asleep keeps its touch state: exactly one
/// Started, and no churn while the island sleeps.
#[test]
fn sleeping_pair_emits_no_events() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        )
        .collision_events(true),
    );
    world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.01, 0.0),
            1.0,
        )
        .collision_events(true),
    );
    let dt = 1.0 / 60.0;
    // Events are a per-step drain, so tally them as they appear.
    let (mut started, mut stopped) = (0, 0);
    for _ in 0..300 {
        world.step_simulation(dt);
        for ev in world.contact_events() {
            match ev {
                CollisionEvent::Started { .. } => started += 1,
                CollisionEvent::Stopped { .. } => stopped += 1,
            }
        }
    }
    assert_eq!((started, stopped), (1, 0), "exactly one Started, no churn");
    for _ in 0..120 {
        world.step_simulation(dt);
    }
    assert_eq!(
        (started, stopped),
        (1, 0),
        "the sleeping island emitted nothing further"
    );
}

/// Removing a body emits Stopped for its touching pairs.
#[test]
fn remove_body_emits_stopped() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        )
        .collision_events(true),
    );
    let ball = world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 1.01, 0.0),
            1.0,
        )
        .collision_events(true),
    );
    let dt = 1.0 / 60.0;
    let mut saw_started = false;
    for _ in 0..3 {
        world.step_simulation(dt);
        saw_started |= world
            .contact_events()
            .iter()
            .any(|e| matches!(e, CollisionEvent::Started { .. }));
    }
    assert!(saw_started, "the contact never produced a Started");
    world.remove_body(ball);
    let Some(CollisionEvent::Stopped { a, b, sensor }) =
        world.contact_events().iter().copied().last()
    else {
        panic!("removal did not emit Stopped");
    };
    assert_eq!((a, b), (BodyHandle::from_index(0), ball));
    assert!(!sensor);
}

/// The CCD sweep respects the group/mask filter.
#[test]
fn ccd_respects_collision_filter() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 2.0, 0.25)),
            Vector3::ZERO,
        )
        .collision_filter(filter_groups::DEFAULT, 0),
    );
    let ball = world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 0.0, -5.0),
            1.0,
        )
        .ccd(0.1, 0.3),
    );
    world.body_mut(ball).linear_velocity = Vector3::new(0.0, 0.0, 10.0);
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step_simulation(dt);
    }
    assert!(
        world.body(ball).position.z > 4.0,
        "filtered wall stopped the CCD ball: z = {}",
        world.body(ball).position.z
    );
}

/// A soft body filtered by the rigid side's mask falls through.
#[test]
fn soft_body_respects_collision_mask() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(3.0, 0.5, 3.0)),
            Vector3::ZERO,
        )
        .collision_filter(filter_groups::DEFAULT, 0),
    );
    let rope = crate::softbody::helpers::create_rope(
        Vector3::new(0.0, 5.0, 0.0),
        Vector3::new(0.0, 3.0, 0.0),
        8,
        0,
    );
    world.add_soft_body(rope);
    let dt = 1.0 / 60.0;
    for _ in 0..90 {
        world.step_simulation(dt);
    }
    let min_y = world
        .soft_body(SoftBodyHandle::from_index(0))
        .nodes
        .iter()
        .map(|n| n.x.y)
        .fold(Real::INFINITY, Real::min);
    assert!(
        min_y < -0.5,
        "rope did not fall through the filtered plate: min_y = {min_y}"
    );
}

/// A multibody base collider filtered by the plane's mask falls
/// through.
#[test]
fn mb_base_collider_respects_mask() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        )
        .collision_filter(filter_groups::DEFAULT, 0),
    );
    let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    mb.set_linear_damping(0.0);
    mb.finalize_multidof();
    mb.set_base_collider(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
        0.5,
        0.0,
    );
    mb.base_pos = Vector3::new(0.0, 3.0, 0.0);
    world.add_multibody(mb);
    let dt = 1.0 / 60.0;
    for _ in 0..90 {
        world.step_simulation(dt);
    }
    assert!(
        world.multibody(MultiBodyHandle::from_index(0)).base_pos.y < -1.0,
        "filtered plane stopped the mb base: y = {}",
        world.multibody(MultiBodyHandle::from_index(0)).base_pos.y
    );
}

#[test]
fn raycast_closest_misses_empty_world() {
    let world = PhysicsWorld::new();
    assert!(
        world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .is_none()
    );
}

#[test]
fn raycast_all_reports_every_hit_nearest_first() {
    let mut world = PhysicsWorld::new();
    let far = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(3.0, 0.0, 0.0),
    ));
    let near = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(1.0, 0.0, 0.0),
    ));
    let hits = world.raycast_all(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0));
    let indices: Vec<BodyHandle> = hits.iter().map(|(i, _)| *i).collect();
    assert_eq!(indices, vec![near, far]);
    assert!(hits[0].1.fraction < hits[1].1.fraction);
}

#[test]
fn raycast_filters_consume_group_and_mask() {
    let mut world = PhysicsWorld::new();
    // The body's mask does not accept the default ray group...
    world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(1.0, 0.0, 0.0),
        )
        .collision_filter(filter_groups::STATIC, filter_groups::STATIC),
    );
    assert!(
        world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .is_none()
    );
    // ...but a query whose group includes STATIC sees it.
    let filter = RayFilter {
        group: filter_groups::DEFAULT | filter_groups::STATIC,
        mask: filter_groups::ALL,
    };
    assert!(
        world
            .raycast_closest_filtered(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0), filter)
            .is_some()
    );
}

#[test]
fn pair_filter_rejects_pairs() {
    let ground_shape = || {
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.5, 2.0)),
            Vector3::ZERO,
        )
    };
    // Without the filter the ball lands on the plate.
    let mut world = PhysicsWorld::new();
    world.add_body(ground_shape());
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(0.0, 2.0, 0.0),
        1.0,
    ));
    for _ in 0..120 {
        world.step_simulation(1.0 / 60.0);
    }
    assert!(world.body(ball).position.y > 0.2);
    // Rejecting every pair lets it fall through.
    let mut world = PhysicsWorld::new();
    world.add_body(ground_shape());
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(0.0, 2.0, 0.0),
        1.0,
    ));
    world.set_pair_filter(Some(Box::new(|_, _, _, _| false)));
    for _ in 0..120 {
        world.step_simulation(1.0 / 60.0);
    }
    assert!(world.body(ball).position.y < -0.5);
}

#[test]
fn per_body_gravity_overrides_the_world() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    // A dynamic body inherits the world gravity at add time.
    let free = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::ZERO,
        1.0,
    ));
    assert_eq!(world.body(free).gravity(), Vector3::new(0.0, -10.0, 0.0));
    // A detached body keeps its own gravity.
    let mut detached = RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::ZERO,
        1.0,
    );
    detached.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
    detached.set_gravity(Vector3::new(0.0, 5.0, 0.0));
    let detached = world.add_body(detached);
    world.step_simulation(1.0 / 60.0);
    assert!(world.body(free).position.y < 0.0);
    assert!(world.body(detached).position.y > 0.0);
    // Propagation skips the detached body.
    world.set_gravity(Vector3::new(0.0, -3.0, 0.0));
    assert_eq!(world.body(free).gravity(), Vector3::new(0.0, -3.0, 0.0));
    assert_eq!(world.body(detached).gravity(), Vector3::new(0.0, 5.0, 0.0));
}

/// A composite "game moment" exercising the consumer-API tier together:
/// an anti-gravity balloon rises through a trigger volume (per-body
/// gravity + sensor events), a fast ball is CCD-clamped by a thin plate,
/// a filter-rejected pair overlaps without colliding, and the settled
/// state is probed with filtered all-hits rays.
#[test]
fn composite_consumer_scene() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));

    let floor = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(20.0, 0.5, 20.0)),
        Vector3::new(0.0, -0.5, 0.0),
    ));
    // The thin CCD target sits off the balloon's column (x >= 1) so the
    // balloon rises unobstructed and the ray below sees only three
    // bodies.
    let _plate = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.5, 0.05, 1.5)),
        Vector3::new(2.5, 1.0, 0.0),
    ));
    let sensor = world.add_body(
        RigidBody::new_static(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 1.0, 1.0)),
            Vector3::new(0.0, 2.0, 0.0),
        )
        .collision_filter(filter_groups::TRIGGER, filter_groups::ALL)
        .no_contact_response(true)
        .collision_events(true),
    );
    let pillar = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.4, 1.0, 0.4)),
        Vector3::new(-2.0, 0.5, 0.0),
    ));

    // The balloon carries its own upward gravity, detached from the
    // world propagation.
    let mut balloon_body = RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(0.0, 0.6, 0.0),
        1.0,
    );
    balloon_body.collision_flags.0 |= CollisionFlags::DISABLE_WORLD_GRAVITY.0;
    balloon_body.set_gravity(Vector3::new(0.0, 3.0, 0.0));
    let balloon = world.add_body(balloon_body);

    // The fast ball drops at 30 m/s (a 0.5 m step) toward the
    // 0.1-thick plate.
    let mut fast_body = RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.2),
        Vector3::new(1.5, 8.0, 0.0),
        1.0,
    );
    fast_body.linear_velocity = Vector3::new(0.0, -30.0, 0.0);
    let fast = world.add_body(fast_body.ccd(0.05, 0.15));

    // A ball starting inside the pillar: the pair filter lets it fall
    // through and land on the floor instead of being pushed out.
    let ghost = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(-2.0, 0.8, 0.0),
        1.0,
    ));
    world.set_pair_filter(Some(Box::new(move |a, _: &RigidBody, b, _: &RigidBody| {
        let (lo, hi) = (a.min(b), a.max(b));
        !(lo == pillar.min(ghost) && hi == pillar.max(ghost))
    })));

    let mut saw_sensor_start = false;
    for _ in 0..150 {
        world.step_simulation(1.0 / 60.0);
        for ev in world.contact_events() {
            if matches!(ev, CollisionEvent::Started { sensor: true, .. }) {
                saw_sensor_start = true;
            }
        }
    }

    // The balloon rose through the sensor, which reported the touch.
    assert!(world.body(balloon).position.y > 3.0);
    assert!(saw_sensor_start);
    // The fast ball was CCD-clamped onto the plate instead of reaching
    // the floor below it.
    let fast_y = world.body(fast).position.y;
    assert!(fast_y > 0.9 && fast_y < 1.45, "fast ball at {fast_y}");
    // The ghost ball fell through the pillar and rests on the floor,
    // still overlapping the pillar's footprint.
    let ghost_pos = world.body(ghost).position;
    assert!((ghost_pos.x + 2.0).abs() < 1e-3);
    assert!(
        ghost_pos.y > 0.2 && ghost_pos.y < 0.45,
        "ghost at {}",
        ghost_pos.y
    );

    // The downward ray through the balloon's column reports every body
    let hits = world.raycast_all(Vector3::new(0.0, 12.0, 0.0), Vector3::new(0.0, -5.0, 0.0));
    let indices: Vec<BodyHandle> = hits.iter().map(|(i, _)| *i).collect();
    assert_eq!(indices, vec![balloon, sensor, floor]);
    assert!(hits.windows(2).all(|w| w[0].1.fraction < w[1].1.fraction));
    // A TRIGGER-group query sees only the sensor.
    let trigger_only = RayFilter {
        group: filter_groups::TRIGGER,
        mask: filter_groups::TRIGGER,
    };
    let hits = world.raycast_all_filtered(
        Vector3::new(0.0, 12.0, 0.0),
        Vector3::new(0.0, -5.0, 0.0),
        trigger_only,
    );
    assert_eq!(
        hits.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![sensor]
    );
    assert_eq!(
        world
            .raycast_closest_filtered(
                Vector3::new(0.0, 12.0, 0.0),
                Vector3::new(0.0, -5.0, 0.0),
                trigger_only,
            )
            .map(|(i, _)| i),
        Some(sensor)
    );
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
    let hit = world.raycast_closest(Vector3::new(0.5, 0.5, 0.5), Vector3::new(0.6, 0.6, 0.6));
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
    assert_eq!(index, BodyHandle::from_index(1));
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
    assert!(
        world
            .convex_sweep_closest(
                &probe,
                Vector3::new(5.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(0.0, 0.0, 0.0),
                None,
            )
            .is_some()
    );
    // ...and a miss when the only body is ignored.
    assert!(
        world
            .convex_sweep_closest(
                &probe,
                Vector3::new(5.0, 0.0, 0.0),
                Quaternion::IDENTITY,
                Vector3::new(0.0, 0.0, 0.0),
                Some(BodyHandle::from_index(0)),
            )
            .is_none()
    );
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
    let hits = world.contact_test(
        &probe,
        Vector3::new(0.9, 0.0, 0.0),
        Quaternion::IDENTITY,
        None,
    );
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, BodyHandle::from_index(0));
    assert!(hits[0].1.points.iter().any(|p| p.penetration > 0.0));
    // The probe itself can be ignored.
    assert!(
        world
            .contact_test(
                &probe,
                Vector3::new(0.9, 0.0, 0.0),
                Quaternion::IDENTITY,
                Some(BodyHandle::from_index(0)),
            )
            .is_empty()
    );
}

/// A resting body keeps a supporting contact impulse in the persistent
/// manifold cache across steps (the warm-start source for the next
/// step's rows).
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
    let cached: Vec<&bullet_collision::manifold::PersistentPoint> = world
        .manifold_cache
        .iter()
        .flatten()
        .flat_map(|e| e.manifold.points())
        .collect();
    assert!(!cached.is_empty());
    assert!(cached.iter().any(|c| c.normal_impulse > 0.0));
    let sphere = world.body(BodyHandle::from_index(1));
    assert_relative_eq!(sphere.position.y, 0.5, epsilon = 0.05);
    assert!(sphere.linear_velocity.length() < 0.1);
}

/// A contact cached while barely touching survives a later small
/// separation inside the breaking-threshold band: the refreshed point is
/// emitted with negative penetration (Bullet's margin-band contact). The
/// scene is gravity-free so the pair stays below the redetect speed gate
/// and actually exercises the refresh path.
#[test]
fn band_contact_survives_brief_separation() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // Sphere barely touching (0.0001 deep): one step caches the contact
    // at the predicted pose.
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 0.4999, 0.0),
        1.0,
    ));
    world.step_simulation(1.0 / 60.0);
    assert!(!world.manifold_cache.is_empty());
    // Nudge up by 0.004 (inside the 0.005 redetect window, so the pair
    // refreshes instead of re-detecting): the refreshed point lands in
    // the margin band with negative penetration.
    world.body_mut(ball).position = Vector3::new(0.0, 0.5039, 0.0);
    world.step_simulation(1.0 / 60.0);
    // The manifold survived with a margin-band point (negative
    // penetration) instead of being dropped by a re-detection that only
    // reports actual penetration.
    let points: Vec<bullet_collision::manifold::PersistentPoint> = world
        .manifold_cache
        .iter()
        .flatten()
        .flat_map(|e| e.manifold.points().to_vec())
        .collect();
    assert!(points.iter().any(|p| p.penetration < 0.0));
}

/// A sphere dropped from slightly above the plane is caught by the
/// margin-band contact *before* touching: the contact forms while the
/// surfaces are still separated (inside the breaking-threshold band) and
/// the band softening limits the approach to the separation rate, so the
/// sphere settles onto the surface without ever penetrating past the
/// slop — the behavior Bullet's positive-distance contacts give.
#[test]
fn band_contact_catches_a_falling_sphere_before_touching() {
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 0.508, 0.0),
        1.0,
    ));
    let mut min_y = Real::INFINITY;
    for _ in 0..60 {
        world.step_simulation(1.0 / 60.0);
        min_y = min_y.min(world.body(ball).position.y);
    }
    // Never penetrated past the slop zone on the way down.
    assert!(min_y > 0.5 - 0.006, "min height {min_y}");
    // And it settles on the surface.
    assert_relative_eq!(world.body(ball).position.y, 0.5, epsilon = 0.006);
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
    let sphere = world.body(BodyHandle::from_index(1));
    assert_eq!(sphere.activation_state(), ActivationState::IslandSleeping);
    assert_eq!(sphere.linear_velocity, Vector3::ZERO);
    let y_at_sleep = sphere.position.y;
    // Further steps must not move the sleeping body.
    for _ in 0..60 {
        world.step_simulation(dt);
    }
    assert_relative_eq!(
        world.body(BodyHandle::from_index(1)).position.y,
        y_at_sleep,
        epsilon = 1e-6
    );
}

/// A sleeping body whose island gains an active member is woken up
/// (Bullet's island activation pass: `ISLAND_SLEEPING` members of a
/// non-sleeping island become `WANTS_DEACTIVATION` — simulating again,
/// but immediately re-qualifying for sleeping).
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
    assert_eq!(
        world.body(BodyHandle::from_index(1)).activation_state(),
        ActivationState::IslandSleeping
    );
    // ...then a fast ball hits it: the sleeping sphere must wake up.
    world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.25),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, -5.0, 0.0)),
    );
    for _ in 0..60 {
        world.step_simulation(dt);
    }
    assert!(
        world.body(BodyHandle::from_index(1)).is_active(),
        "sleeping body was not woken by the impact"
    );
    assert_ne!(
        world.body(BodyHandle::from_index(1)).activation_state(),
        ActivationState::IslandSleeping
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
    world
        .body_mut(BodyHandle::from_index(1))
        .apply_force(Vector3::new(0.0, 5.0, 0.0));
    assert_eq!(
        world.body(BodyHandle::from_index(1)).activation_state(),
        ActivationState::Active
    );
}

/// Static bodies start asleep, like Bullet's static `btRigidBody`.
#[test]
fn static_body_starts_sleeping() {
    let body = RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::ZERO,
    );
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
    assert_relative_eq!(
        world.body(BodyHandle::from_index(1)).position.y,
        0.9,
        epsilon = 0.05
    );
    // The control ball crossed the whole wall.
    assert!(world.body(BodyHandle::from_index(2)).position.y < -0.5);
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
    assert_relative_eq!(
        world.body(BodyHandle::from_index(1)).position.y,
        1.0 - 5.0 / 60.0,
        epsilon = 1e-5
    );
}

/// CCD skips pairs with suppressed collisions: Bullet's dispatcher
/// `needsCollision` (`checkCollideWithOverride`) gates the sweep too, so a
/// fast body is not clamped against its joint-linked partner. The joint's
/// impulse is clamped to a token value so the ball's motion stays intact
/// and the sweep filter is the only thing under test.
#[test]
fn ccd_skips_collision_suppressed_pairs() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    // Thin static wall in the fast ball's path, joint-linked to it.
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
        Vector3::new(0.0, 0.5, 0.0),
    ));
    world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, -120.0, 0.0))
        .ccd(0.1, 0.3),
    );
    let mut joint = Constraint::point2point(
        BodyHandle::from_index(0),
        BodyHandle::from_index(1),
        Vector3::new(0.0, 0.5, 0.0),
        Vector3::new(0.0, -0.5, 0.0),
    );
    joint.point2point_mut().set_impulse_clamp(0.001);
    world.add_constraint(joint);
    // The suppressed pair is not swept: the ball crosses the wall (the
    // unsuppressed control is `ccd_clamps_fast_bodies_at_the_impact`).
    world.step_simulation(1.0 / 60.0);
    assert!(world.body(BodyHandle::from_index(1)).position.y < -0.5);
}

/// A fixed-base pendulum with a sphere collider on the link: the collider
/// registers as a broad-phase proxy with the FK pose AABB and refreshes
/// as the body moves (phase B of the multibody collision design).
#[test]
fn link_collider_registers_and_updates_aabb() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    mb.set_link_collider(
        0,
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        0.5,
        0.0,
    );
    world.add_multibody(mb);
    world.step_simulation(1.0 / 60.0);
    // The link COM hangs 1 m below the pivot; the sphere AABB spans
    // y ∈ [−1.5, −0.5], padded by the 0.02 broad-phase contact margin.
    let entry = world
        .collider_entries
        .iter()
        .find(|e| e.mb == 0 && e.link == 0)
        .expect("collider registered");
    let proxy = entry.proxy.expect("proxy assigned");
    let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
    assert_relative_eq!(aabb.min.y, -1.52, epsilon = 1e-4);
    assert_relative_eq!(aabb.max.y, -0.48, epsilon = 1e-4);
    // Moving the base moves the collider with the next FK pass.
    world.multibody_mut(MultiBodyHandle::from_index(0)).base_pos = Vector3::new(0.0, 5.0, 0.0);
    world.step_simulation(1.0 / 60.0);
    let aabb = world.broadphase.proxy_aabb(proxy).unwrap();
    assert_relative_eq!(aabb.min.y, 3.48, epsilon = 1e-4);
}

/// A sleeping multibody keeps its collider AABB (like sleeping rigid
/// bodies): the refresh is skipped until the body wakes.
#[test]
fn sleeping_multibody_keeps_collider_aabb() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, true);
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    mb.set_link_collider(
        0,
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        0.5,
        0.0,
    );
    world.add_multibody(mb);
    for _ in 0..150 {
        // 2.5 s > the 2 s sleep timeout.
        world.step_simulation(1.0 / 60.0);
    }
    assert!(!world.multibody(MultiBodyHandle::from_index(0)).is_awake());
    let proxy = world.collider_entries[0].proxy.unwrap();
    let before = world.broadphase.proxy_aabb(proxy).unwrap();
    // Teleport the base: the sleeping body skips the AABB refresh.
    world.multibody_mut(MultiBodyHandle::from_index(0)).base_pos = Vector3::new(3.0, 0.0, 0.0);
    world.step_simulation(1.0 / 60.0);
    let after = world.broadphase.proxy_aabb(proxy).unwrap();
    assert_eq!(before, after);
}

/// A link collider overlapping a static rigid body produces a translated
/// mixed broad-phase pair.
#[test]
fn link_collider_pairs_with_rigid_body() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    // Sink the link so its sphere (r = 0.5, centered 0.6 below the
    // plane) overlaps the ground.
    mb.base_pos = Vector3::new(0.0, 0.4, 0.0);
    mb.set_link_collider(
        0,
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        0.5,
        0.0,
    );
    world.add_multibody(mb);
    world.step_simulation(1.0 / 60.0);
    let mixed = world
        .broadphase
        .compute_pairs()
        .into_iter()
        .filter(|(a, b)| {
            matches!(world.proxy_owners[*a], ProxyOwner::Body(_))
                ^ matches!(world.proxy_owners[*b], ProxyOwner::Body(_))
        })
        .count();
    assert_eq!(mixed, 1, "one rigid-collider pair expected");
}

/// Self-collision filtering between two overlapping link colliders of
/// one chain: the broad phase reports the AABB pair (both colliders are
/// registered); the `checkCollideWithOverride` port
/// ([`MultiBody::colliders_collide`]) decides whether the narrow phase
/// runs on it — adjacent links collide by default, `has_self_collision =
/// false` and the per-link parent-collision flags opt out.
#[test]
fn self_collision_filtering() {
    // Two overlapping link colliders on one chain register and produce
    // one collider-collider broad-phase pair.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = crate::multibody::MultiBody::new(2, 1.0, Vector3::ONE, true, false);
    // Both links sit at the base origin with overlapping spheres.
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::ZERO,
    );
    mb.setup_revolute(
        1,
        1.0,
        Vector3::ONE,
        Some(0),
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::ZERO,
    );
    mb.finalize_multidof();
    mb.set_link_collider(
        0,
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        0.5,
        0.0,
    );
    mb.set_link_collider(
        1,
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        0.5,
        0.0,
    );
    world.add_multibody(mb);
    world.step_simulation(1.0 / 60.0);
    let collider_pairs = world
        .broadphase
        .compute_pairs()
        .into_iter()
        .filter(|(a, b)| {
            matches!(world.proxy_owners[*a], ProxyOwner::Collider(_))
                && matches!(world.proxy_owners[*b], ProxyOwner::Collider(_))
        })
        .count();
    assert_eq!(collider_pairs, 1);

    // The filter itself (`checkCollideWithOverride` port) on a 3-link
    // chain: base - 0 - 1 - 2. `-1` is the base.
    let mut mb = crate::multibody::MultiBody::new(3, 1.0, Vector3::ONE, true, false);
    for i in 0..3 {
        mb.setup_revolute(
            i,
            1.0,
            Vector3::ONE,
            if i == 0 { None } else { Some(i - 1) },
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::ZERO,
        );
    }
    mb.finalize_multidof();
    // Default: everything collides (Bullet's default).
    assert!(mb.colliders_collide(0, 1));
    assert!(mb.colliders_collide(-1, 0));
    assert!(mb.colliders_collide(0, 2)); // non-adjacent
    // Whole-body opt-out.
    mb.set_has_self_collision(false);
    assert!(!mb.colliders_collide(0, 1));
    assert!(!mb.colliders_collide(-1, 0));
    mb.set_has_self_collision(true);
    // Direct-parent opt-out on link 1: only (1, 0) is excluded.
    mb.set_link_parent_collision_disabled(1, false);
    assert!(!mb.colliders_collide(1, 0));
    assert!(mb.colliders_collide(1, 2));
    assert!(mb.colliders_collide(1, -1));
    // All-ancestors opt-out on link 2: parent and grandparent excluded,
    // others not (fresh body to keep the flags independent).
    let mut mb = crate::multibody::MultiBody::new(3, 1.0, Vector3::ONE, true, false);
    for i in 0..3 {
        mb.setup_revolute(
            i,
            1.0,
            Vector3::ONE,
            if i == 0 { None } else { Some(i - 1) },
            Quaternion::IDENTITY,
            Vector3::Z,
            Vector3::ZERO,
            Vector3::ZERO,
        );
    }
    mb.finalize_multidof();
    mb.set_link_parent_collision_disabled(2, true);
    assert!(!mb.colliders_collide(2, 1));
    assert!(!mb.colliders_collide(2, 0));
    // The base is part of the ancestor walk, like Bullet's loop reaching
    // m_parent == -1.
    assert!(!mb.colliders_collide(2, -1));
    // The exclusion is symmetric (Bullet checks both sides), and pairs
    // without a flagged member are unaffected.
    assert!(!mb.colliders_collide(1, 2));
    assert!(mb.colliders_collide(0, 1));
}

/// A fixed-base pendulum whose link sphere falls onto a static plane is
/// supported by the multibody contact rows: it settles hanging straight
/// down with the sphere resting on the plane (phase C contact response).
#[test]
fn link_sphere_rests_on_ground() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let mut mb = crate::multibody::MultiBody::new(1, 1.0, Vector3::ONE, true, false);
    mb.setup_revolute(
        0,
        1.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    mb.set_link_collider(
        0,
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        0.5,
        0.0,
    );
    // Pivot 1.51 m up: the link center's lowest arc point is y = 0.51,
    // so the sphere (r = 0.5) starts 0.01 m inside the plane — the
    // contact rows must support it hanging straight down.
    mb.base_pos = Vector3::new(0.0, 1.51, 0.0);
    world.add_multibody(mb);
    let dt = 1.0 / 60.0;
    for _ in 0..240 {
        world.step_simulation(dt);
    }
    let body = world.multibody(MultiBodyHandle::from_index(0));
    let q = body.joint_pos_multidof(0)[0];
    let qd = body.joint_vel_multidof(0)[0];
    assert!(
        q.abs() < 0.05,
        "pendulum did not settle hanging down: q = {q}"
    );
    assert!(qd.abs() < 0.1, "pendulum still swinging: qd = {qd}");
    // The link center rests with the sphere on the plane: x ≈ 0,
    // y ≈ 0.5 (within the slop/damping tolerance).
    let p = body.link_world_position(0);
    assert_relative_eq!(p.x, 0.0, epsilon = 0.02);
    assert_relative_eq!(p.y, 0.5, epsilon = 0.05);
}

/// A swinging pendulum link kicks a dynamic rigid box lying in its path:
/// the multibody contact rows exchange impulse with the rigid solver
/// (box wakes and moves, the pendulum keeps swinging reduced).
#[test]
fn swinging_link_pushes_rigid_box() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // Ball mass 3 vs box mass 1: probe-verified against stock Bullet —
    // a mass-3 ball knocks the box ~1.1 m away, a mass-1 ball only
    // ~0.05 m (the persistent four-corner manifold holds a resting box
    // much steadier than the pre-merge single-point behavior).
    let mut mb = crate::multibody::MultiBody::new(1, 3.0, Vector3::ONE, true, false);
    mb.setup_revolute(
        0,
        3.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    mb.set_link_collider(
        0,
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        0.3,
        0.0,
    );
    // Pivot 1 m up; release from the horizontal (+X side).
    mb.base_pos = Vector3::new(0.0, 1.0, 0.0);
    mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
    world.add_multibody(mb);
    // A box resting on the plane where the ball's arc crosses its right
    // face, deliberately clear of the ball's release pose: when the box
    // sat under the release point, the ball just came to rest on the box
    // top instead of kicking it (stock-Bullet probe verified).
    let box_start = Vector3::new(0.85, 0.3, 0.0);
    world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.3)),
            box_start,
            1.0,
        )
        .friction(0.0),
    );
    let dt = 1.0 / 60.0;
    for _ in 0..90 {
        world.step_simulation(dt);
    }
    // The box was hit and knocked away from its start pose. At the
    // impact point (≈45° along the arc) the ball's velocity points
    // (−X, −Y), so the box is shoved toward −X / tipped over. The exact
    // slide direction is chaotic though: the ball pins the box into the
    // plane and the frictionless box drifts with the contact details
    // (the persistent-manifold warm start shifts the trajectory within
    // this noise), so the direction check carries a small tolerance.
    let box_end = world.body(BodyHandle::from_index(1)).position;
    assert!(
        (box_end - box_start).length() > 0.1,
        "box was not pushed: {box_end:?}"
    );
    assert!(
        box_end.x < box_start.x + 0.05,
        "box pushed the wrong way: {box_end:?}"
    );
}

/// A multibody contact wakes a sleeping rigid body: the sleeping box is
/// struck by the swinging link and moves.
#[test]
fn link_contact_wakes_sleeping_rigid_body() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // The victim: a box that settles on the plane and sleeps, clear of
    // the ball's release pose (see the swinging test for the geometry
    // and mass rationale — both stock-Bullet probe verified).
    let box_idx = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.3)),
        Vector3::new(0.85, 0.3, 0.0),
        1.0,
    ));
    for _ in 0..180 {
        // 3 s > the 2 s deactivation window.
        world.step_simulation(1.0 / 60.0);
    }
    assert!(!world.body(box_idx).is_active(), "box should be asleep");
    // Swing a link sphere into it (mass 3: a mass-1 ball only nudges
    // the four-corner-supported box ~0.01 m — probe-verified).
    let mut mb = crate::multibody::MultiBody::new(1, 3.0, Vector3::ONE, true, false);
    mb.setup_revolute(
        0,
        3.0,
        Vector3::ONE,
        None,
        Quaternion::IDENTITY,
        Vector3::Z,
        Vector3::ZERO,
        Vector3::new(0.0, -1.0, 0.0),
    );
    mb.finalize_multidof();
    mb.set_link_collider(
        0,
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        0.3,
        0.0,
    );
    mb.base_pos = Vector3::new(0.0, 1.0, 0.0);
    mb.set_joint_pos_multidof(0, &[bullet_math::FRAC_PI_2]);
    world.add_multibody(mb);
    for _ in 0..90 {
        world.step_simulation(1.0 / 60.0);
    }
    assert!(
        world.body(box_idx).is_active(),
        "box should have been woken"
    );
    // The wake is the point of this test; the shove is a gentle nudge
    // (~0.02 m here, ~0.09 m in stock Bullet — after the impact the
    // ball comes to rest wedged on the box's top corner, and the exact
    // resting pose is chaotic).
    assert!(
        (world.body(box_idx).position - Vector3::new(0.85, 0.3, 0.0)).length() > 0.015,
        "sleeping box was not knocked away: {:?}",
        world.body(box_idx).position
    );
}

/// A free-base multibody with a box collider falls onto a static plane
/// and is supported by its base contact rows (rest height, zero
/// velocity).
#[test]
fn free_base_with_collider_lands_on_ground() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    mb.set_linear_damping(0.0);
    mb.finalize_multidof();
    mb.set_base_collider(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
        0.5,
        0.0,
    );
    mb.base_pos = Vector3::new(0.0, 3.0, 0.0);
    world.add_multibody(mb);
    let dt = 1.0 / 60.0;
    for _ in 0..240 {
        world.step_simulation(dt);
    }
    let body = world.multibody(MultiBodyHandle::from_index(0));
    // The half-extent box rests on the plane: y ≈ 0.5 within the slop
    // tolerance, with a small residual velocity (damped settling).
    assert_relative_eq!(body.base_pos().y, 0.5, epsilon = 0.05);
    assert!(
        body.base_vel().length() < 0.2,
        "base still moving: {:?}",
        body.base_vel()
    );
}

/// A fast rigid sphere hitting a free multibody (equal masses, head-on,
/// restitution 0, gravity off) converges to the shared inelastic
/// velocity: the mixed-row iteration must count the assembly-time
/// relative velocity only once — reading the rigid side's impulse delta,
/// not its full velocity. The double-count regression inflated the
/// impulse ~2.4x, leaving the ball nearly stopped and the multibody
/// overspeeding.
#[test]
fn rb_mb_contact_impulse_not_doubled() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    let mut mb = crate::multibody::MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    mb.set_linear_damping(0.0);
    mb.set_angular_damping(0.0);
    mb.finalize_multidof();
    mb.set_base_collider(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        0.5,
        0.0,
    );
    world.add_multibody(mb);
    let ball = world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.5),
            Vector3::new(0.0, 3.0, 0.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, -6.0, 0.0))
        .disable_deactivation(),
    );
    let dt = 1.0 / 60.0;
    for _ in 0..25 {
        world.step_simulation(dt);
    }
    let ball_vy = world.body(ball).linear_velocity.y;
    let mb_vy = world.multibody(MultiBodyHandle::from_index(0)).base_vel().y;
    assert!(mb_vy < -1.5, "multibody was not kicked: {mb_vy}");
    // Equal-mass head-on, restitution 0: both sides leave at ~-3 m/s
    // (plus a small Baumgarte bounce that separates them slightly).
    assert!(
        (ball_vy - mb_vy).abs() < 1.5,
        "velocities did not converge: ball {ball_vy} vs multibody {mb_vy}"
    );
}

/// Phase D: a resting pair on the plane is one island and sleeps as a
/// unit, then wakes as a unit when a falling body joins the island
/// (Bullet's `buildIslands` activation pass).
#[test]
fn island_sleeps_and_wakes_as_unit() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // Two boxes side by side, touching (one shared face at x = 0).
    world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::new(-0.5, 0.5, 0.0),
        1.0,
    ));
    world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::new(0.5, 0.5, 0.0),
        1.0,
    ));
    let dt = 1.0 / 60.0;
    for _ in 0..(4.0 / dt) as usize {
        world.step_simulation(dt);
    }
    // The whole island sleeps together.
    assert!(
        !world.body(BodyHandle::from_index(1)).is_active(),
        "left box should sleep"
    );
    assert!(
        !world.body(BodyHandle::from_index(2)).is_active(),
        "right box should sleep"
    );
    // Sleeping means frozen: further steps do not move either box.
    let (pa, pb) = (
        world.body(BodyHandle::from_index(1)).position,
        world.body(BodyHandle::from_index(2)).position,
    );
    for _ in 0..60 {
        world.step_simulation(dt);
    }
    assert!((world.body(BodyHandle::from_index(1)).position - pa).length() < 1e-9);
    assert!((world.body(BodyHandle::from_index(2)).position - pb).length() < 1e-9);
    // A fast ball lands on the right box: the impact joins its island
    // and both boxes wake.
    world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.25),
            Vector3::new(-3.0, 0.5, 0.0),
            1.0,
        )
        .velocity(Vector3::new(8.0, 0.0, 0.0)),
    );
    for _ in 0..30 {
        world.step_simulation(dt);
    }
    assert!(
        world.body(BodyHandle::from_index(1)).is_active(),
        "left box should wake with its island"
    );
    assert!(
        world.body(BodyHandle::from_index(2)).is_active(),
        "right box should wake with its island"
    );
}

/// Phase D: a `DisableDeactivation` member keeps its whole island awake
/// forever (Bullet counts `DISABLE_DEACTIVATION` as an awake contributor).
/// The boxes are stacked (not edge-to-edge): the persistent four-point
/// box-box contact keeps the broad-phase pair alive robustly, where
/// edge-glued boxes can drift apart under the single-point plane contact
/// (the plane fast path reports one contact, so a resting box pivots and
/// creeps — a trajectory-sensitive way to lose the island link).
#[test]
fn disable_deactivation_keeps_island_awake() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -9.81, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
            Vector3::new(0.0, 0.5, 0.0),
            1.0,
        )
        .disable_deactivation(),
    );
    world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::splat(0.5)),
        Vector3::new(0.0, 1.5, 0.0),
        1.0,
    ));
    let dt = 1.0 / 60.0;
    for _ in 0..(10.0 / dt) as usize {
        world.step_simulation(dt);
    }
    assert!(world.body(BodyHandle::from_index(1)).is_active());
    assert!(
        world.body(BodyHandle::from_index(2)).is_active(),
        "neighbor of a disable-deactivation body must never sleep"
    );
}

/// Phase D: a joint merges two islands across a distance — a moving
/// body keeps its joint partner (far away, no contact) from sleeping
/// (Bullet merges enabled constraints between two dynamic bodies).
#[test]
fn joint_links_islands_across_a_distance() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    // Two bodies 2 m apart joined by a point-to-point joint whose pivots
    // coincide at the midpoint, so the constraint starts satisfied (no
    // initial kick).
    let mid = Vector3::new(5.0, 0.0, 0.0);
    let a = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        mid - Vector3::X,
        1.0,
    ));
    let b = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        mid + Vector3::X,
        1.0,
    ));
    let joint = crate::constraint::Constraint::point2point(a, b, Vector3::X, -Vector3::X);
    world.add_constraint(joint);
    // Body A is driven perpetually; B is far away and still — only the
    // joint keeps its island awake.
    let dt = 1.0 / 60.0;
    for _ in 0..(6.0 / dt) as usize {
        world.body_mut(a).apply_force(Vector3::new(2.0, 0.0, 0.0));
        world.step_simulation(dt);
    }
    assert!(
        world.body(b).is_active(),
        "joint-linked partner of a moving body must stay awake"
    );
}

// ---- body removal -----------------------------------------------

/// A removed body stops colliding (its broad-phase proxy is gone, its
/// cached manifolds are dropped) and freezes in place; the bodies that
/// were resting on it fall.
#[test]
fn removed_body_stops_colliding_and_falls() {
    let mut world = PhysicsWorld::new();
    let ground = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 0.5, 0.0),
        1.0,
    ));
    let dt = 1.0 / 60.0;
    // Settle into contact (the manifold cache now holds the pair).
    for _ in 0..10 {
        world.step_simulation(dt);
    }
    assert!(world.body(ball).position.y > 0.4);
    let ground_y = world.bodies()[ground.index()].position.y;
    world.remove_body(ground);
    // With the ground gone the ball falls freely through the old plane.
    for _ in 0..30 {
        world.step_simulation(dt);
    }
    assert!(
        world.body(ball).position.y < -0.5,
        "ball must fall through the removed ground"
    );
    // The removed body itself is inert data.
    assert_eq!(world.bodies()[ground.index()].position.y, ground_y);
    assert!(world.bodies()[ground.index()].is_removed());
    assert!(!world.bodies()[ground.index()].is_active());
}

/// Removed bodies disappear from every world query (ray casts, sweeps,
/// contact tests all run through the broad phase).
#[test]
fn removed_body_is_invisible_to_queries() {
    let mut world = PhysicsWorld::new();
    let sphere = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(1.0, 0.0, 0.0),
    ));
    assert!(
        world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .is_some()
    );
    world.remove_body(sphere);
    assert!(
        world
            .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
            .is_none()
    );
}

/// Body slots are never reused (indices stay valid) while the freed
/// broad-phase slot is recycled for the next registration; the owner
/// table stays consistent.
#[test]
fn remove_then_add_recycles_the_proxy_slot() {
    let mut world = PhysicsWorld::new();
    let first = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(1.0, 0.0, 0.0),
    ));
    let second = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(3.0, 0.0, 0.0),
    ));
    let freed = world.body_proxies[first.index()];
    world.remove_body(first);
    // The new body reuses the freed proxy slot and gets a fresh body slot.
    let third = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(1.0, 2.0, 0.0),
    ));
    assert_eq!(third, BodyHandle::from_index(2));
    assert_eq!(world.body_proxies[third.index()], freed);
    // Queries resolve to the right owners afterwards.
    let (index, _) = world
        .raycast_closest(Vector3::ZERO, Vector3::new(5.0, 0.0, 0.0))
        .unwrap();
    assert_eq!(index, second);
    let (index, _) = world
        .raycast_closest(Vector3::new(1.0, 5.0, 0.0), Vector3::new(1.0, -5.0, 0.0))
        .unwrap();
    assert_eq!(index, third);
}

/// Joints referencing a removed body go with it; the remaining joints
/// keep working (their collision suppression survives the rebuild).
#[test]
fn remove_body_cascades_joints() {
    let mut world = PhysicsWorld::new();
    let a = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(-1.0, 5.0, 0.0),
        1.0,
    ));
    let b = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(1.0, 5.0, 0.0),
        1.0,
    ));
    world.add_constraint(crate::constraint::Constraint::point2point(
        a,
        b,
        Vector3::X,
        -Vector3::X,
    ));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    assert_eq!(world.constraints().len(), 1);
    world.remove_body(a);
    assert_eq!(world.constraints().len(), 0);
    // Nothing holds B up anymore: it falls the whole way down to the
    // ground (a surviving joint would keep it near its release height).
    let dt = 1.0 / 60.0;
    for _ in 0..90 {
        world.step_simulation(dt);
    }
    assert!(
        world.body(b).position.y < 1.0,
        "released body must fall freely, got {:?}",
        world.body(b).position
    );
}

/// Multibody-vs-rigidbody constraints cascade with the rigid side too.
#[test]
fn remove_body_cascades_multibody_constraints() {
    let mut world = PhysicsWorld::new();
    let mut mb = MultiBody::new(0, 1.0, Vector3::ONE, false, false);
    mb.finalize_multidof();
    let ia = world.add_multibody(mb);
    let rb = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
    ));
    let snapshot = world.multibodies().to_vec();
    let constraint = MultiBodyConstraint::point2point_to_rigid(
        ia,
        None,
        rb,
        Vector3::ZERO,
        Vector3::new(0.0, 1.0, 0.0),
        &snapshot,
    );
    world.add_multibody_constraint(constraint);
    assert_eq!(world.num_multibody_constraints(), 1);
    world.remove_body(rb);
    assert_eq!(world.num_multibody_constraints(), 0);
    // The multibody itself keeps simulating.
    for _ in 0..10 {
        world.step_simulation(1.0 / 60.0);
    }
}

/// A removed body no longer blocks the CCD swept-sphere clamps (the
/// candidate sweep skips inert slots; its proxy is destroyed).
#[test]
fn removed_body_does_not_block_ccd() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::ZERO);
    // Thin static wall spanning y in [0.4, 0.6] — same scene as
    // `ccd_clamps_fast_bodies_at_the_impact`.
    let wall = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.1, 1.0)),
        Vector3::new(0.0, 0.5, 0.0),
    ));
    let ball = world.add_body(
        RigidBody::new_dynamic(
            bullet_collision::shapes::CollisionShape::sphere(0.3),
            Vector3::new(0.0, 1.0, 0.0),
            1.0,
        )
        .velocity(Vector3::new(0.0, -120.0, 0.0))
        .ccd(0.1, 0.3),
    );
    world.remove_body(wall);
    world.step_simulation(1.0 / 60.0);
    assert!(
        world.body(ball).position.y < -0.5,
        "CCD clamp must ignore the removed wall"
    );
}

/// The remaining bodies keep their indices and simulate normally after
/// a removal (no renumbering, no swap).
#[test]
fn removed_body_keeps_remaining_indices_stable() {
    let mut world = PhysicsWorld::new();
    let ground = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let a = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 5.0, 0.0),
        1.0,
    ));
    let b = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(2.0, 8.0, 0.0),
        1.0,
    ));
    world.remove_body(a);
    assert_eq!(world.bodies().len(), 3);
    let dt = 1.0 / 60.0;
    for _ in 0..240 {
        world.step_simulation(dt);
    }
    // B landed on the still-present ground; the indices never moved.
    assert_relative_eq!(world.body(b).position.y, 0.5, epsilon = 0.05);
    assert_eq!(ground, BodyHandle::from_index(0));
    assert!(world.bodies()[a.index()].is_removed());
}

#[test]
#[should_panic(expected = "already been removed")]
fn removing_a_body_twice_panics() {
    let mut world = PhysicsWorld::new();
    let body = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::ZERO,
    ));
    world.remove_body(body);
    world.remove_body(body);
}

#[test]
#[should_panic(expected = "has been removed")]
fn accessing_a_removed_body_panics() {
    let mut world = PhysicsWorld::new();
    let body = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::ZERO,
    ));
    world.remove_body(body);
    let _ = world.body(body);
}

/// A closed 2x2x0.4 box shell as a dynamic mesh body (top and bottom
/// plates; a single-sided zero-thickness plate would be the known
/// tunneling degenerate).
fn box_shell_mesh() -> bullet_collision::shapes::CollisionShape {
    let (hx, hy, hz) = (1.0 as Real, 0.2 as Real, 1.0 as Real);
    bullet_collision::shapes::CollisionShape::triangle_mesh(
        vec![
            // Top face.
            Vector3::new(-hx, hy, -hz),
            Vector3::new(hx, hy, -hz),
            Vector3::new(hx, hy, hz),
            Vector3::new(-hx, hy, hz),
            // Bottom face.
            Vector3::new(-hx, -hy, -hz),
            Vector3::new(hx, -hy, -hz),
            Vector3::new(hx, -hy, hz),
            Vector3::new(-hx, -hy, hz),
        ],
        vec![[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]],
    )
}

/// A dynamic mesh body falls onto a static plane, rests on it, and goes
/// to sleep — the GImpact use case end to end.
#[test]
fn dynamic_mesh_rests_on_plane_and_sleeps() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let mesh = world.add_body(RigidBody::new_dynamic(
        box_shell_mesh(),
        Vector3::new(0.0, 2.0, 0.0),
        2.0,
    ));
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step_simulation(dt);
    }
    let b = world.body(mesh);
    // Bottom face at y = -0.2 relative to the body origin: rest height.
    assert!(
        (0.175..0.23).contains(&b.position.y),
        "mesh should rest with its bottom face on the plane, got {}",
        b.position.y
    );
    assert!(b.linear_velocity.length() < 0.1);
    assert!(matches!(
        b.activation_state(),
        ActivationState::IslandSleeping
    ));
}

/// A dynamic mesh body rests on a static mesh (the mesh-vs-mesh tree
/// traversal driving real support). Dropped from just above the margin
/// shell — a single step deeper than the band would tunnel the
/// zero-thickness shell past the surface (known deviation of the GJK
/// route; Bullet's clip method resolves signed back-side distances).
#[test]
fn dynamic_mesh_rests_on_static_mesh() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    let ground = bullet_collision::shapes::CollisionShape::triangle_mesh(
        (0..8)
            .flat_map(|z| (0..8).map(move |x| (x, z)))
            .flat_map(|(x, z)| {
                let (x, z) = (x as Real, z as Real);
                vec![
                    Vector3::new(x, 0.0, z),
                    Vector3::new(x + 1.0, 0.0, z),
                    Vector3::new(x + 1.0, 0.0, z + 1.0),
                    Vector3::new(x, 0.0, z + 1.0),
                ]
            })
            .collect::<Vec<_>>(),
        (0..64)
            .flat_map(|i| {
                let b = (i * 4) as u32;
                [[b, b + 2, b + 1], [b, b + 3, b + 2]]
            })
            .collect::<Vec<_>>(),
    );
    world.add_body(RigidBody::new_static(ground, Vector3::new(-4.0, 0.0, -4.0)));
    let mesh = world.add_body(RigidBody::new_dynamic(
        box_shell_mesh(),
        Vector3::new(0.0, 0.5, 0.0),
        2.0,
    ));
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step_simulation(dt);
    }
    let b = world.body(mesh);
    // The shell rests with its bottom face held `band - slop` above the
    // ground by the margin shell (the GImpact hover, like Bullet).
    assert!(
        (0.24..0.36).contains(&b.position.y),
        "mesh should rest on the static mesh ground, got {}",
        b.position.y
    );
    assert!(b.linear_velocity.length() < 0.1);
}

/// A dynamic sphere resting on a dynamic mesh plate: the mesh-vs-convex
/// path with a *moving* mesh (both sides dynamic, island-merged).
#[test]
fn sphere_rests_on_dynamic_mesh() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let mesh = world.add_body(RigidBody::new_dynamic(
        box_shell_mesh(),
        Vector3::new(0.0, 0.2, 0.0),
        5.0,
    ));
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 2.0, 0.0),
        1.0,
    ));
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step_simulation(dt);
    }
    // The heavy shell barely moves; the ball rests on its top face
    // (y = 0.2 + 0.2 + 0.5 = 0.9 in the ideal, allow solver slop).
    let top = world.body(mesh).position.y + 0.2;
    let y = world.body(ball).position.y;
    assert!(
        ((top + 0.5 - 0.05)..=(top + 0.5 + 0.05)).contains(&y),
        "ball should rest on the dynamic mesh top face, got {y}"
    );
    assert!(world.body(ball).linear_velocity.length() < 0.1);
}

/// The injected step-timing clock: monotonic reads accumulate exact
/// spans, a backwards read saturates to zero, and disabled timing
/// never reads the clock at all.
#[test]
fn step_clock_injection_contract() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    static IDX: AtomicUsize = AtomicUsize::new(0);
    // First read seeds the sampler's start; the next 7 are stage
    // marks. The third mark goes backwards (115 -> 110).
    const TICKS: [u64; 8] = [100, 110, 115, 110, 130, 140, 150, 160];
    fn fake_clock() -> Duration {
        Duration::from_nanos(TICKS[IDX.fetch_add(1, Ordering::SeqCst)])
    }

    let mut world = PhysicsWorld::new();
    // Disabled timing: the clock is never read.
    world.set_step_clock(fake_clock);
    world.step_simulation(1.0 / 60.0);
    assert_eq!(IDX.load(Ordering::SeqCst), 0);
    assert_eq!(world.step_timings().total(), Duration::ZERO);

    // Enabled: spans accumulate; the backwards span clamps to zero.
    IDX.store(0, Ordering::SeqCst);
    world.enable_step_timing();
    world.step_simulation(1.0 / 60.0);
    let t = world.step_timings();
    assert_eq!(
        t.stage(StepStage::IntegrateVelocities),
        Duration::from_nanos(10)
    );
    assert_eq!(t.stage(StepStage::UpdateAabbs), Duration::from_nanos(5));
    assert_eq!(t.stage(StepStage::Broadphase), Duration::ZERO);
    assert_eq!(t.stage(StepStage::Islands), Duration::from_nanos(20));
    assert_eq!(t.stage(StepStage::Narrowphase), Duration::from_nanos(10));
    assert_eq!(t.stage(StepStage::Solve), Duration::from_nanos(10));
    assert_eq!(
        t.stage(StepStage::IntegratePositions),
        Duration::from_nanos(10)
    );
    assert_eq!(t.total(), Duration::from_nanos(65));

    // Swapping clocks mid-session is tolerated (fresh start point).
    IDX.store(0, Ordering::SeqCst);
    world.set_step_clock(fake_clock);
    world.step_simulation(1.0 / 60.0);
}

// --- kinematic rigid bodies (Bullet's `CF_KINEMATIC_OBJECT`) --------------

/// A kinematic body driven by its velocity pushes a dynamic body out of the
/// way: the contact rows use the kinematic's velocity in the relative
/// velocity, so the dynamic side gains momentum (velocity propagation).
#[test]
fn kinematic_pushes_dynamic_body() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // Sliding kinematic slab (its top face is at y = 0.5) and a dynamic box
    // resting in its path.
    let slab = world.add_body(RigidBody::new_kinematic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(1.0, 0.5, 1.0)),
        Vector3::new(-3.0, 0.5, 0.0),
    ));
    let target = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.25, 0.25, 0.25)),
        Vector3::new(0.0, 0.75, 0.0),
        1.0,
    ));
    world.body_mut(slab).linear_velocity = Vector3::new(2.0, 0.0, 0.0);

    let start_x = world.body(target).position.x;
    for _ in 0..120 {
        world.step_simulation(1.0 / 60.0);
    }
    // The slab (2 m/s for 2 s) reached and shoved the box well past its
    // starting point.
    assert!(world.body(target).position.x > start_x + 0.5);
    assert!(world.body(target).is_active());
}

/// Gravity, damping and accumulated forces never touch a kinematic body's
/// velocities; the pose integrates from the user-set velocity every step.
#[test]
fn kinematic_ignores_gravity_forces_and_damping() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    let body = world.add_body(RigidBody::new_kinematic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::new(0.0, 5.0, 0.0),
    ));
    world.body_mut(body).set_damping(0.2, 0.2);
    let v = Vector3::new(1.0, 0.0, 0.0);
    world.body_mut(body).linear_velocity = v;
    world.body_mut(body).angular_velocity = Vector3::new(0.0, 0.0, 1.0);

    let dt = 1.0 / 60.0;
    let frames = 60;
    for i in 0..frames {
        world.body_mut(body).apply_force(Vector3::new(0.0, 100.0, 0.0));
        world.step_simulation(dt);
        let b = world.body(body);
        // Gravity must not move the kinematic body.
        assert_relative_eq!(b.position.y, 5.0, epsilon = 1e-4);
        assert_eq!(b.linear_velocity, v, "velocity must stay user-set (frame {i})");
    }
    // Pose integrated from the velocity: x ~= v * t (per-frame `+=` rounding
    // aside), and it never fell.
    assert_relative_eq!(world.body(body).position.x, 1.0, epsilon = 1e-3);
    assert!(world.body(body).is_active());
}

/// A moving kinematic body wakes the sleeping island it runs into; an idle
/// one lets its resting contacts sleep.
#[test]
fn moving_kinematic_wakes_sleeping_island() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let box_body = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.25, 0.25, 0.25)),
        Vector3::new(0.0, 0.25, 0.0),
        1.0,
    ));
    for _ in 0..600 {
        world.step_simulation(1.0 / 60.0);
    }
    assert!(!world.body(box_body).is_active(), "precondition: asleep");

    // A slab flying in from the side.
    let slab = world.add_body(RigidBody::new_kinematic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
        Vector3::new(-4.0, 0.25, 0.0),
    ));
    world.body_mut(slab).linear_velocity = Vector3::new(6.0, 0.0, 0.0);

    let mut woke = false;
    for _ in 0..120 {
        world.step_simulation(1.0 / 60.0);
        if world.body(box_body).is_active() {
            woke = true;
            break;
        }
    }
    assert!(woke, "the moving kinematic must wake the sleeping island");
}

/// A dynamic body resting on an *idle* kinematic platform settles and goes
/// to sleep; the kinematic-static pair produces no solver rows.
#[test]
fn dynamic_body_rests_and_sleeps_on_idle_kinematic_platform() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    let platform = world.add_body(RigidBody::new_kinematic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(2.0, 0.25, 2.0)),
        Vector3::new(0.0, 0.25, 0.0),
    ));
    let cargo = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.25, 0.25, 0.25)),
        Vector3::new(0.0, 2.0, 0.0),
        1.0,
    ));
    for _ in 0..480 {
        world.step_simulation(1.0 / 60.0);
    }
    let b = world.body(cargo);
    assert_relative_eq!(b.position.y, 0.75, epsilon = 0.05);
    assert!(!b.is_active(), "idle kinematic contacts must let it sleep");
    // The platform itself never moved and keeps its velocity.
    assert_eq!(world.body(platform).position, Vector3::new(0.0, 0.25, 0.0));
}

/// Kinematic-vs-kinematic and kinematic-vs-static overlaps are skipped
/// entirely and leave both bodies untouched (Bullet's broadphase
/// StaticFilter excludes them).
#[test]
fn kinematic_static_and_kinematic_kinematic_pairs_are_inert() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    let ground = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    let a = world.add_body(RigidBody::new_kinematic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
        Vector3::new(0.0, 0.5, 0.0),
    ));
    let b = world.add_body(RigidBody::new_kinematic(
        bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
        Vector3::new(0.3, 0.5, 0.0),
    ));
    let va = Vector3::new(0.5, 0.0, 0.0);
    let vb = Vector3::new(-0.5, 0.0, 0.0);
    world.body_mut(a).linear_velocity = va;
    world.body_mut(b).linear_velocity = vb;

    for _ in 0..120 {
        world.step_simulation(1.0 / 60.0);
    }
    // They flew apart (or through each other) but no solver row ever touched   
    // their velocities, and the static ground still holds them where their     
    // velocities put them.
    assert_eq!(world.body(a).linear_velocity, va);
    assert_eq!(world.body(b).linear_velocity, vb);
    assert!(world.body(a).position.y > 0.0);
    let _ = ground;
}

/// Mutating a static compound's child between steps must refresh the
/// broad-phase proxy immediately: the per-step AABB pass only covers
/// active bodies, so a moved child would otherwise stay invisible to
/// pair detection at its new location (a ray walks the proxy first).
#[test]
fn static_compound_child_transform_refreshes_the_proxy() {
    let mut world = PhysicsWorld::new();
    let compound = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::compound(vec![
            bullet_collision::shapes::CompoundChild::new(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.5, 0.5)),
                Vector3::new(0.0, 0.5, 0.0),
            ),
        ]),
        Vector3::ZERO,
    ));
    assert!(world.set_child_transform(
        compound,
        0,
        Vector3::new(6.0, 0.5, 0.0),
        Quaternion::IDENTITY,
    ));

    // A downward ray at the new location hits the moved child; at the old
    // location it finds nothing (of this compound).
    let hit = world.raycast_closest(Vector3::new(6.0, 3.0, 0.0), Vector3::new(6.0, -1.0, 0.0));
    assert!(hit.is_some(), "moved child invisible: proxy not refreshed");
    let hit = world.raycast_closest(Vector3::new(0.0, 3.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
    assert!(hit.is_none(), "stale child bounds still visible");
    world.step_simulation(1.0 / 60.0);
    // The refresh survives a step (update_aabbs skips inactive bodies).
    let hit = world.raycast_closest(Vector3::new(6.0, 3.0, 0.0), Vector3::new(6.0, -1.0, 0.0));
    assert!(hit.is_some(), "moved child lost after a step");
}

/// A dynamic compound's inverse inertia follows the child layout (the
/// construction formula re-run); an unchanged axis stays untouched.
#[test]
fn dynamic_compound_child_transform_recomputes_inertia() {
    let mut door = RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::compound(vec![
            bullet_collision::shapes::CompoundChild::new(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 2.0, 0.25)),
                Vector3::new(-3.0, 0.0, 0.0),
            ),
            bullet_collision::shapes::CompoundChild::new(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 2.0, 0.25)),
                Vector3::new(3.0, 0.0, 0.0),
            ),
        ]),
        Vector3::ZERO,
        12.0,
    );
    let wide = door.inv_inertia_local();
    // Pull the second child in (the first stays at x = -3): the bounding
    // box shrinks along x, so I_y and I_z shrink while I_x (which only
    // sees the unchanged y/z extents) stays put.
    assert!(door.set_child_transform(1, Vector3::new(1.0, 0.0, 0.0), Quaternion::IDENTITY));
    let narrow = door.inv_inertia_local();
    assert_relative_eq!(narrow.x, wide.x);
    assert!(narrow.y > wide.y && narrow.z > wide.z);

    // A non-compound body reports the miss.
    let mut ball = RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.5),
        Vector3::ZERO,
        1.0,
    );
    assert!(!ball.set_child_transform(0, Vector3::ZERO, Quaternion::IDENTITY));
    assert!(ball.remove_child_shape(0).is_none());
}

/// A kinematic compound (a vehicle slab with a raised bumper child)
/// pushes a dynamic ball purely through its velocity: the end-to-end
/// kinematic x compound interaction (pair gate, infinite-mass solver
/// rows, per-step proxy refresh) with no solver row touching the
/// kinematic body's velocity.
#[test]
fn kinematic_compound_pushes_a_ball() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vector3::new(0.0, -10.0, 0.0));
    let ground = world.add_body(RigidBody::new_static(
        bullet_collision::shapes::CollisionShape::plane(Vector3::Y, 0.0),
        Vector3::ZERO,
    ));
    // Vehicle: slab + bumper, driven at constant +x from x = -4.
    let vehicle = world.add_body(RigidBody::new_kinematic(
        bullet_collision::shapes::CollisionShape::compound(vec![
            bullet_collision::shapes::CompoundChild::new(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.5, 0.25, 0.5)),
                Vector3::new(0.0, 0.25, 0.0),
            ),
            bullet_collision::shapes::CompoundChild::new(
                bullet_collision::shapes::CollisionShape::box_shape(Vector3::new(0.1, 0.4, 0.5)),
                Vector3::new(0.6, 0.4, 0.0),
            ),
        ]),
        Vector3::new(-4.0, 0.0, 0.0),
    ));
    world.body_mut(vehicle).linear_velocity = Vector3::new(1.0, 0.0, 0.0);
    // Ball resting on the ground in the vehicle's path.
    let ball = world.add_body(RigidBody::new_dynamic(
        bullet_collision::shapes::CollisionShape::sphere(0.3),
        Vector3::new(0.0, 0.3, 0.0),
        1.0,
    ));

    for _ in 0..240 {
        world.step_simulation(1.0 / 60.0);
    }

    // The vehicle drove through the ball's start and carried it along,
    // its own velocity untouched (infinite mass).
    assert!(
        world.body(ball).position.x > 0.5,
        "ball at {}",
        world.body(ball).position.x
    );
    assert!(world.body(ball).position.y > 0.0);
    assert_eq!(
        world.body(vehicle).linear_velocity,
        Vector3::new(1.0, 0.0, 0.0)
    );
    // Drove ~4 m from x = -4 (the 240-frame budget lands it near 0).
    assert!(world.body(vehicle).position.x > -0.5);
    let _ = ground;
}
