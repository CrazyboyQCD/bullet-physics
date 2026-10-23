// Hinge-2 / universal probe: the two thin joint specializations run through
// the real Bullet constraint solver (refs/bullet3/src, zlib region) for
// numeric comparison against the Rust twin
// (crates/bullet-physics/examples/hinge2_universal_probe.rs). The joints are
// configurations of the already-probed spring-2 / classic 6DOF row shapes;
// what this pins is the constructor-derived configuration (frames from world
// axes, limit tables, suspension constants, the construction-pose
// equilibrium) feeding those rows.
//
// Scenes:
//   H2SUS — hinge-2 suspension: static chassis + wheel spawned 0.3 below the
//           anchor, gravity -z; the spring catches the fall (240 frames)
//   H2STE — hinge-2 steering: a spin about the steering axis with the COM
//           offset from the anchor; the linear locks absorb the anchor-point
//           violation in one step and the wheel settles into a slow
//           momentum-conserving orbit (120 frames)
//   UNIV  — universal joint: a spin about a locked/limited axis corrected
//           through the angular limit rows (60 frames)
// One line per frame: label, frame, position(3), quaternion(x y z w) of the
// moving body at %.17g — strip CR, then tools/mbdiff.py compares.
//
// Build with tools/build_hinge2_universal_probe.py (f32 + f64).
#include <cstdio>

#include "btBulletDynamicsCommon.h"
#include "BulletDynamics/ConstraintSolver/btHinge2Constraint.h"
#include "BulletDynamics/ConstraintSolver/btUniversalConstraint.h"

static const int kFramesSus = 240;
static const int kFramesSte = 120;
static const int kFramesUniv = 60;

static btRigidBody *static_body(btDiscreteDynamicsWorld *world, btSphereShape *shape,
								const btVector3 &origin) {
	btTransform t;
	t.setIdentity();
	t.setOrigin(origin);
	btRigidBody *body = new btRigidBody(0.0, nullptr, shape, btVector3(0.0, 0.0, 0.0));
	body->setWorldTransform(t);
	world->addRigidBody(body);
	return body;
}

static btRigidBody *dynamic_body(btDiscreteDynamicsWorld *world, btSphereShape *shape,
								 const btVector3 &origin, btScalar mass) {
	btTransform t;
	t.setIdentity();
	t.setOrigin(origin);
	// btRigidBody uses the passed localInertia verbatim — it does not derive
	// it from the shape. Without calculateLocalInertia the body gets mass
	// with a zero inertia tensor (a point mass), and every joint row loses
	// its angular response.
	btVector3 local_inertia(0.0, 0.0, 0.0);
	shape->calculateLocalInertia(mass, local_inertia);
	btRigidBody *body = new btRigidBody(mass, nullptr, shape, local_inertia);
	body->setWorldTransform(t);
	world->addRigidBody(body);
	return body;
}

static void print_body(const char *label, int frame, const btRigidBody *body) {
	const btVector3 &p = body->getCenterOfMassTransform().getOrigin();
	const btQuaternion &q = body->getCenterOfMassTransform().getRotation();
	std::printf("%s %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", label, frame,
				p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
}

int main() {
	// Scene 0: hinge-2 suspension.
	{
		btDefaultCollisionConfiguration config;
		btCollisionDispatcher dispatcher(&config);
		btDbvtBroadphase broadphase;
		btSequentialImpulseConstraintSolver solver;
		btDiscreteDynamicsWorld world(&dispatcher, &broadphase, &solver, &config);
		world.setGravity(btVector3(0.0, 0.0, -9.8));
		btSphereShape chassis_shape(0.5);
		btSphereShape wheel_shape(0.2);
		btRigidBody *chassis = static_body(&world, &chassis_shape, btVector3(0, 0, 0));
		btRigidBody *wheel = dynamic_body(&world, &wheel_shape, btVector3(0, 0, -0.3), 1.0);
		btHinge2Constraint joint(*chassis, *wheel, btVector3(0, 0, 0),
								 btVector3(0, 0, 1), btVector3(1, 0, 0));
		world.addConstraint(&joint, true);
		print_body("H2SUS", 0, wheel);
		const btScalar dt = 1.0 / 60.0;
		for (int f = 1; f <= kFramesSus; f++) {
			world.stepSimulation(dt, 1, dt);
			print_body("H2SUS", f, wheel);
		}
	}
	// Scene 1: hinge-2 steering limit.
	{
		btDefaultCollisionConfiguration config;
		btCollisionDispatcher dispatcher(&config);
		btDbvtBroadphase broadphase;
		btSequentialImpulseConstraintSolver solver;
		btDiscreteDynamicsWorld world(&dispatcher, &broadphase, &solver, &config);
		world.setGravity(btVector3(0.0, 0.0, 0.0));
		btSphereShape chassis_shape(0.5);
		btSphereShape wheel_shape(0.15);
		btRigidBody *chassis = dynamic_body(&world, &chassis_shape, btVector3(0, 0, 0), 10.0);
		btRigidBody *wheel = dynamic_body(&world, &wheel_shape, btVector3(0.9, 0, 0), 1.0);
		btHinge2Constraint joint(*chassis, *wheel, btVector3(0, 0, 0),
								 btVector3(0, 0, 1), btVector3(1, 0, 0));
		world.addConstraint(&joint, true);
		wheel->setAngularVelocity(btVector3(0, 0, 6.0));
		print_body("H2STE", 0, wheel);
		const btScalar dt = 1.0 / 60.0;
		for (int f = 1; f <= kFramesSte; f++) {
			world.stepSimulation(dt, 1, dt);
			print_body("H2STE", f, wheel);
		}
	}
	// Scene 2: universal joint, locked-axis spin.
	{
		btDefaultCollisionConfiguration config;
		btCollisionDispatcher dispatcher(&config);
		btDbvtBroadphase broadphase;
		btSequentialImpulseConstraintSolver solver;
		btDiscreteDynamicsWorld world(&dispatcher, &broadphase, &solver, &config);
		world.setGravity(btVector3(0.0, 0.0, 0.0));
		btSphereShape body_shape(0.3);
		btRigidBody *a = static_body(&world, &body_shape, btVector3(0, 0, 0));
		btRigidBody *b = dynamic_body(&world, &body_shape, btVector3(1, 0, 0), 1.0);
		btUniversalConstraint joint(*a, *b, btVector3(1, 0, 0),
									btVector3(0, 0, 1), btVector3(1, 0, 0));
		world.addConstraint(&joint, true);
		b->setAngularVelocity(btVector3(0, 5.0, 0.0));
		print_body("UNIV", 0, b);
		const btScalar dt = 1.0 / 60.0;
		for (int f = 1; f <= kFramesUniv; f++) {
			world.stepSimulation(dt, 1, dt);
			print_body("UNIV", f, b);
		}
	}
	return 0;
}
