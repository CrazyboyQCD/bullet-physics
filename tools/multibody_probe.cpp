// Multibody (Featherstone) probe: runs a set of btMultiBody scenes and prints
// per-frame trajectories for numerical comparison against the Rust port.
// Build with tools/build_recorder.py's msvc environment, e.g.:
//   python - <<'EOF'
//   import sys; sys.path.insert(0, 'tools')
//   from build_recorder import msvc_env
//   ... (cl multibody_probe.cpp ... /Fe:tools/multibody_probe.exe /link ...)
//   EOF
// Usage: multibody_probe.exe <scene> [frames]

#include <cstdio>
#include <cstring>
#include <vector>

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>
#include <BulletDynamics/Featherstone/btMultiBodyPoint2Point.h>
#include <BulletDynamics/Featherstone/btMultiBodyFixedConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointMotor.h>
#include <BulletDynamics/Featherstone/btMultiBodyGearConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodySphericalJointLimit.h>
#include <BulletDynamics/Featherstone/btMultiBodySphericalJointMotor.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointFeedback.h>
#include <BulletDynamics/Featherstone/btMultiBodySliderConstraint.h>

static const btScalar DT = btScalar(1) / btScalar(60);
static const int DEFAULT_FRAMES = 300;

static btMultiBodyDynamicsWorld* makeWorld()
{
	btDefaultCollisionConfiguration* config = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(config);
	btDbvtBroadphase* broadphase = new btDbvtBroadphase();
	btMultiBodyConstraintSolver* solver = new btMultiBodyConstraintSolver();
	btMultiBodyDynamicsWorld* world = new btMultiBodyDynamicsWorld(dispatcher, broadphase, solver, config);
	world->setGravity(btVector3(0, -9.81, 0));
	// Reproducible solving order (same as the parity harness).
	world->getSolverInfo().m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
	return world;
}

// Register tiny colliders for the base and every link so the island manager
// processes the multibody constraints (without collision objects Bullet
// never solves them).
static void addColliders(btMultiBodyDynamicsWorld* world, btMultiBody* body)
{
	btSphereShape* shape = new btSphereShape(0.1f);
	btMultiBodyLinkCollider* baseCol = new btMultiBodyLinkCollider(body, -1);
	baseCol->setCollisionShape(shape);
	baseCol->setWorldTransform(body->getBaseWorldTransform());
	world->addCollisionObject(baseCol);
	body->setBaseCollider(baseCol);
	for (int i = 0; i < body->getNumLinks(); ++i)
	{
		btMultiBodyLinkCollider* col = new btMultiBodyLinkCollider(body, i);
		col->setCollisionShape(new btSphereShape(0.1f));
		col->setWorldTransform(body->getLink(i).m_cachedWorldTransform);
		world->addCollisionObject(col);
		body->getLink(i).m_collider = col;
	}
}

// Scene 1: a free base (no links) falls under gravity; damping disabled.
static void scene_free_base(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(0, 1, btVector3(1, 1, 1), false, false);
	body->setLinearDamping(0);
	body->setAngularDamping(0);
	body->finalizeMultiDof();
	world->addMultiBody(body);
	addColliders(world, body);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = body->getBasePos();
		const btVector3& v = body->getBaseVel();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

// Scene 2: a fixed-base single-link revolute pendulum released from the
// horizontal; records joint position and velocity.
static void scene_pendulum(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(0, -1, 0));
	body->finalizeMultiDof();
	btScalar q = SIMD_HALF_PI;
	body->setJointPosMultiDof(0, &q);
	world->addMultiBody(body);
	addColliders(world, body);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		printf("%d %.6f %.6f\n", f, body->getJointPos(0), body->getJointVel(0));
	}
}

// Scene 3: a two-link chain, both released from the horizontal (q = 0), each
// link's COM 1 m along its own X.
static void scene_two_link(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(2, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(1, 0, 0));
	body->setupRevolute(1, 1, btVector3(1, 1, 1), 0, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(1, 0, 0), btVector3(1, 0, 0));
	body->finalizeMultiDof();
	world->addMultiBody(body);
	addColliders(world, body);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		printf("%d %.6f %.6f %.6f %.6f\n", f, body->getJointPos(0), body->getJointVel(0),
			   body->getJointPos(1), body->getJointVel(1));
	}
}

// Scene 4: two free bases (masses 1 and 2) connected by a point-to-point
// constraint; body B starts with an upward velocity.
static void scene_p2p(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* a = new btMultiBody(0, 1, btVector3(1, 1, 1), false, false);
	btMultiBody* b = new btMultiBody(0, 2, btVector3(1.5f, 1.5f, 1.5f), false, false);
	a->finalizeMultiDof();
	b->finalizeMultiDof();
	b->setBaseVel(btVector3(0, 3, 0));
	world->addMultiBody(a);
	world->addMultiBody(b);
	addColliders(world, a);
	addColliders(world, b);
	btMultiBodyPoint2Point* c = new btMultiBodyPoint2Point(a, -1, b, -1, btVector3(0, 0, 0), btVector3(0, 0, 0));
	world->addMultiBodyConstraint(c);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& pa = a->getBasePos();
		const btVector3& pb = b->getBasePos();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);
	}
}

// Scene 5: a joint motor drives a fixed-base revolute joint to 2 rad/s
// (gravity off).
static void scene_motor(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	world->setGravity(btVector3(0, 0, 0));
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(1, 0, 0));
	body->finalizeMultiDof();
	world->addMultiBody(body);
	addColliders(world, body);
	btMultiBodyJointMotor* motor = new btMultiBodyJointMotor(body, 0, 0, 2, 1000);
	world->addMultiBodyConstraint(motor);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		printf("%d %.6f %.6f\n", f, body->getJointPos(0), body->getJointVel(0));
	}
}

// Scene 6: two free bases welded by a fixed constraint; body B starts with a
// +X velocity.
static void scene_fixed(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* a = new btMultiBody(0, 1, btVector3(1, 1, 1), false, false);
	btMultiBody* b = new btMultiBody(0, 1, btVector3(1, 1, 1), false, false);
	a->finalizeMultiDof();
	b->finalizeMultiDof();
	b->setBaseVel(btVector3(2, 0, 0));
	world->addMultiBody(a);
	world->addMultiBody(b);
	addColliders(world, a);
	addColliders(world, b);
	btMultiBodyFixedConstraint* c = new btMultiBodyFixedConstraint(a, -1, b, -1, btVector3(0, 0, 0), btVector3(0, 0, 0),
																  btMatrix3x3::getIdentity(), btMatrix3x3::getIdentity());
	world->addMultiBodyConstraint(c);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& pa = a->getBasePos();
		const btVector3& pb = b->getBasePos();
		const btQuaternion& qa = a->getWorldToBaseRot();
		const btQuaternion& qb = b->getWorldToBaseRot();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", f, pa[0], pa[1], pa[2], pb[0], pb[1], pb[2],
			   qa[0], qa[1], qa[2], qa[3]);
	}
}

// Scene 7: a fixed-base prismatic (sliding) link released from rest; the
// joint slides under gravity.
static void scene_prismatic(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupPrismatic(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						 btVector3(0, 1, 0), btVector3(0, 0, 0), btVector3(0, 0, 0), false);
	body->finalizeMultiDof();
	btScalar qd0 = 0;
	body->setJointVelMultiDof(0, &qd0);
	world->addMultiBody(body);
	addColliders(world, body);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		printf("%d %.6f %.6f\n", f, body->getJointPos(0), body->getJointVel(0));
	}
}

// Scene 8: a fixed-base spherical joint - a 3D pendulum. The joint quaternion
// and its angular velocity are recorded.
static void scene_spherical(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupSpherical(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						 btVector3(0, 0, 0), btVector3(0, 0, -1));
	body->finalizeMultiDof();
	// Tilt the joint 60 degrees about X: COM swings like a spherical pendulum.
	btScalar q[4] = {btSin(SIMD_HALF_PI / 3), 0, 0, btCos(SIMD_HALF_PI / 3)};
	body->setJointPosMultiDof(0, q);
	world->addMultiBody(body);
	addColliders(world, body);
	const btScalar* jp = body->getJointPosMultiDof(0);
	const btScalar* jv = body->getJointVelMultiDof(0);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		jp = body->getJointPosMultiDof(0);
		jv = body->getJointVelMultiDof(0);
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", f, jp[0], jp[1], jp[2], jp[3], jv[0], jv[1], jv[2]);
	}
}

// Scene 9: a fixed-base planar joint: rotation about Z plus two translations
// in the XY plane. The link is released from rest with an initial X velocity.
static void scene_planar(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupPlanar(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
					  btVector3(0, 0, 1), btVector3(0, 0, 0));
	body->finalizeMultiDof();
	btScalar qd[3] = {0, 1, 0};
	body->setJointVelMultiDof(0, qd);
	world->addMultiBody(body);
	addColliders(world, body);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btScalar* jp = body->getJointPosMultiDof(0);
		const btScalar* jv = body->getJointVelMultiDof(0);
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, jp[0], jp[1], jp[2], jv[0], jv[1], jv[2]);
	}
}

// Scene 10: a gear (ratio 2) coupling two fixed-base revolute links; body A's
// link is pushed by a motor, body B must rotate at half the rate.
static void scene_gear(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	world->setGravity(btVector3(0, 0, 0));
	btMultiBody* a = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	a->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
					 btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(1, 0, 0));
	a->finalizeMultiDof();
	btMultiBody* b = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	b->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
					 btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(1, 0, 0));
	b->finalizeMultiDof();
	world->addMultiBody(a);
	world->addMultiBody(b);
	addColliders(world, a);
	addColliders(world, b);
	btMultiBodyGearConstraint* gear = new btMultiBodyGearConstraint(a, 0, b, 0, btVector3(0, 0, 0), btVector3(0, 0, 0),
																   btMatrix3x3::getIdentity(), btMatrix3x3::getIdentity());
	gear->setGearRatio(2);
	world->addMultiBodyConstraint(gear);
	btMultiBodyJointMotor* motor = new btMultiBodyJointMotor(a, 0, 0, 1, 1000);
	world->addMultiBodyConstraint(motor);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		printf("%d %.6f %.6f %.6f %.6f\n", f, a->getJointPos(0), a->getJointVel(0), b->getJointPos(0), b->getJointVel(0));
	}
}

// Scene 11: a fixed-base revolute joint with a joint limit [-0.3, 0.3];
// released inside with a velocity toward the upper bound so the limit stops
// the motion and the joint bounces between the bounds (no damping).
static void scene_joint_limit(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	world->setGravity(btVector3(0, 0, 0));
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(1, 0, 0));
	body->finalizeMultiDof();
	btScalar q = 0.25;
	body->setJointPosMultiDof(0, &q);
	btScalar qd = 2;
	body->setJointVelMultiDof(0, &qd);
	world->addMultiBody(body);
	addColliders(world, body);
	btMultiBodyJointLimitConstraint* limit = new btMultiBodyJointLimitConstraint(body, 0, -0.3, 0.3);
	world->addMultiBodyConstraint(limit);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		printf("%d %.6f %.6f\n", f, body->getJointPos(0), body->getJointVel(0));
	}
}

// Scene 12: a fixed-base spherical joint with a swing/twist limit; the joint
// is tilted 60 degrees about X, far outside the 0.3 rad swing range, and the
// limit pulls the swing back inside the cone.
static void scene_spherical_limit(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	world->setGravity(btVector3(0, 0, 0));
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupSpherical(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						 btVector3(0, 0, 0), btVector3(0, 0, -1));
	body->finalizeMultiDof();
	btScalar q[4] = {btSin(SIMD_HALF_PI / 3), 0, 0, btCos(SIMD_HALF_PI / 3)};
	body->setJointPosMultiDof(0, q);
	world->addMultiBody(body);
	addColliders(world, body);
	btMultiBodySphericalJointLimit* limit = new btMultiBodySphericalJointLimit(body, 0, 0.3, 0.3, 0.5, 100);
	world->addMultiBodyConstraint(limit);
	const btScalar* jp = body->getJointPosMultiDof(0);
	const btScalar* jv = body->getJointVelMultiDof(0);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		jp = body->getJointPosMultiDof(0);
		jv = body->getJointVelMultiDof(0);
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", f, jp[0], jp[1], jp[2], jp[3], jv[0], jv[1], jv[2]);
	}
}

// Scene 13: a fixed-base spherical joint driven by a spherical motor back to
// the identity orientation from a 60 degree tilt about X.
static void scene_spherical_motor(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	world->setGravity(btVector3(0, 0, 0));
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupSpherical(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						 btVector3(0, 0, 0), btVector3(0, 0, -1));
	body->finalizeMultiDof();
	btScalar q[4] = {btSin(SIMD_HALF_PI / 3), 0, 0, btCos(SIMD_HALF_PI / 3)};
	body->setJointPosMultiDof(0, q);
	world->addMultiBody(body);
	addColliders(world, body);
	btMultiBodySphericalJointMotor* motor = new btMultiBodySphericalJointMotor(body, 0, 100);
	world->addMultiBodyConstraint(motor);
	const btScalar* jp = body->getJointPosMultiDof(0);
	const btScalar* jv = body->getJointVelMultiDof(0);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		jp = body->getJointPosMultiDof(0);
		jv = body->getJointVelMultiDof(0);
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", f, jp[0], jp[1], jp[2], jp[3], jv[0], jv[1], jv[2]);
	}
}

// ---------------------------------------------------------------------------
// Collider scenes (multibody-collision.md §7): link/base colliders meeting
// static and dynamic rigid bodies. Unlike addColliders above (tiny filler
// spheres only there to drive the island manager), these attach real shapes
// and need the FK caches refreshed before the first step.
// ---------------------------------------------------------------------------

// Refresh the cached FK transforms after posing (setJointPos/setBasePos) so
// freshly attached colliders start at the posed configuration instead of the
// constructor defaults.
static void refreshFk(btMultiBody* body)
{
	btAlignedObjectArray<btQuaternion> world_to_local;
	btAlignedObjectArray<btVector3> local_origin;
	body->forwardKinematics(world_to_local, local_origin);
}

// A static ground box, top surface at y = 0, half extents (50, 0.5, 50).
static void addGroundBox(btMultiBodyDynamicsWorld* world)
{
	btBoxShape* shape = new btBoxShape(btVector3(50, btScalar(0.5), 50));
	btDefaultMotionState* ms = new btDefaultMotionState(
		btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, btScalar(-0.5), 0)));
	btRigidBody::btRigidBodyConstructionInfo ci(0, ms, shape, btVector3(0, 0, 0));
	world->addRigidBody(new btRigidBody(ci));
}

// Attach a friction-0.5 sphere collider to link `i` (or the base for -1) at
// the current FK pose. Dynamic colliders (moving links, free bases) join the
// DefaultFilter group like the official demos — the addCollisionObject
// default (StaticFilter) never pairs against static geometry (whose mask
// excludes Static), silently disabling the collision.
static void attachSphere(btMultiBodyDynamicsWorld* world, btMultiBody* body, int link, btScalar radius, btScalar friction = btScalar(0.5))
{
	btMultiBodyLinkCollider* col = new btMultiBodyLinkCollider(body, link);
	col->setCollisionShape(new btSphereShape(radius));
	col->setFriction(friction);
	col->setWorldTransform(link < 0 ? body->getBaseWorldTransform()
									: body->getLink(link).m_cachedWorldTransform);
	bool is_dynamic = link >= 0 || !body->hasFixedBase();
	int group = is_dynamic ? int(btBroadphaseProxy::DefaultFilter)
						   : int(btBroadphaseProxy::StaticFilter);
	int mask = is_dynamic ? int(btBroadphaseProxy::AllFilter)
						  : int(btBroadphaseProxy::AllFilter ^ btBroadphaseProxy::StaticFilter);
	world->addCollisionObject(col, group, mask);
	if (link < 0)
		body->setBaseCollider(col);
	else
		body->getLink(link).m_collider = col;
}

// Scene 14 (§7.1 link_ground): a fixed-base pendulum with a sphere collider
// on the link swings from the horizontal into a static ground box. The pivot
// sits 1.25 m up so the arc's lowest point (link COM y = 0.25, r = 0.3)
// penetrates 0.05 m — a shallow, clean first impact. Records the joint state
// plus the link COM trajectory.
static void scene_link_ground(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	addGroundBox(world);
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(0, -1, 0));
	body->finalizeMultiDof();
	// q = +pi/2 rotates the COM from straight down to (+1, 0) — the horizontal
	// release, the same pose as the `pendulum` scene.
	btScalar q = SIMD_HALF_PI;
	body->setJointPosMultiDof(0, &q);
	body->setBasePos(btVector3(0, btScalar(1.25), 0));
	refreshFk(body);
	world->addMultiBody(body);
	attachSphere(world, body, 0, btScalar(0.3));
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = body->getLink(0).m_cachedWorldTransform.getOrigin();
		printf("%d %.6f %.6f %.6f %.6f %.6f\n", f, body->getJointPos(0), body->getJointVel(0), p[0], p[1], p[2]);
	}
}

// Scene 15 (§7.2 base_ground): a free base with a box collider drops onto the
// ground from y = 3; records the base position and velocity (impact impulses
// and the rest height).
static void scene_base_ground(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	addGroundBox(world);
	btMultiBody* body = new btMultiBody(0, 1, btVector3(1, 1, 1), false, false);
	body->setLinearDamping(0);
	body->setAngularDamping(0);
	body->finalizeMultiDof();
	body->setBasePos(btVector3(0, 3, 0));
	refreshFk(body);
	world->addMultiBody(body);
	btMultiBodyLinkCollider* col = new btMultiBodyLinkCollider(body, -1);
	col->setCollisionShape(new btBoxShape(btVector3(btScalar(0.5), btScalar(0.5), btScalar(0.5))));
	col->setFriction(btScalar(0.5));
	col->setWorldTransform(body->getBaseWorldTransform());
	world->addCollisionObject(col, btBroadphaseProxy::DefaultFilter, btBroadphaseProxy::AllFilter);
	body->setBaseCollider(col);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = body->getBasePos();
		const btVector3& v = body->getBaseVel();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

// Scene 16 (§7.3 mb_pushes_rb): a motorized link with a sphere collider
// sweeps through a free-floating dynamic box (gravity off, no ground).
// The link COM starts at (+1, 0) and circles clockwise at 2 rad/s; the box
// sits at the bottom of the arc. Records the box trajectory/velocity and the
// link COM.
static void scene_mb_pushes_rb(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	world->setGravity(btVector3(0, 0, 0));
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(1, 0, 0));
	body->finalizeMultiDof();
	refreshFk(body);
	world->addMultiBody(body);
	attachSphere(world, body, 0, btScalar(0.3));
	btMultiBodyJointMotor* motor = new btMultiBodyJointMotor(body, 0, 0, -2, 1000);
	world->addMultiBodyConstraint(motor);
	btBoxShape* shape = new btBoxShape(btVector3(btScalar(0.5), btScalar(0.5), btScalar(0.5)));
	btDefaultMotionState* ms = new btDefaultMotionState(
		btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -1, 0)));
	btRigidBody::btRigidBodyConstructionInfo ci(1, ms, shape, btVector3(0, 0, 0));
	btRigidBody* box = new btRigidBody(ci);
	box->setActivationState(DISABLE_DEACTIVATION);
	world->addRigidBody(box);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = box->getWorldTransform().getOrigin();
		const btVector3& v = box->getLinearVelocity();
		const btVector3& lp = body->getLink(0).m_cachedWorldTransform.getOrigin();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
			   f, p[0], p[1], p[2], v[0], v[1], v[2], lp[0], lp[1], lp[2]);
	}
}

// Scene 17 (§7.4 rb_hits_mb): a dynamic rigid sphere falls at 6 m/s onto the
// base sphere collider of a free multibody floating at the origin. Records
// both trajectories plus the sphere velocity (impulse exchange).
static void scene_rb_hits_mb(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(0, 1, btVector3(1, 1, 1), false, false);
	body->setLinearDamping(0);
	body->setAngularDamping(0);
	body->finalizeMultiDof();
	refreshFk(body);
	world->addMultiBody(body);
	attachSphere(world, body, -1, btScalar(0.5));
	btSphereShape* shape = new btSphereShape(btScalar(0.5));
	btDefaultMotionState* ms = new btDefaultMotionState(
		btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 3, 0)));
	btRigidBody::btRigidBodyConstructionInfo ci(1, ms, shape, btVector3(0, 0, 0));
	btRigidBody* ball = new btRigidBody(ci);
	ball->setLinearVelocity(btVector3(0, -6, 0));
	ball->setActivationState(DISABLE_DEACTIVATION);
	world->addRigidBody(ball);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& mbp = body->getBasePos();
		const btVector3& p = ball->getWorldTransform().getOrigin();
		const btVector3& v = ball->getLinearVelocity();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
			   f, mbp[0], mbp[1], mbp[2], p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

// Scene 18 (§7.5 mb_mb): two free single-link multibodies with sphere
// colliders on their links collide head-on (A drifts +X at 2 m/s into B at
// rest 3 m away, gravity off). The e/d vectors are zero so each link COM
// rides on its base. Records both base positions.
static void scene_mb_mb(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	world->setGravity(btVector3(0, 0, 0));
	btMultiBody* a = new btMultiBody(1, 1, btVector3(1, 1, 1), false, false);
	a->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
					 btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(0, 0, 0));
	a->setLinearDamping(0);
	a->setAngularDamping(0);
	a->finalizeMultiDof();
	a->setBaseVel(btVector3(2, 0, 0));
	refreshFk(a);
	btMultiBody* b = new btMultiBody(1, 1, btVector3(1, 1, 1), false, false);
	b->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
					 btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(0, 0, 0));
	b->setLinearDamping(0);
	b->setAngularDamping(0);
	b->finalizeMultiDof();
	b->setBasePos(btVector3(3, 0, 0));
	refreshFk(b);
	world->addMultiBody(a);
	world->addMultiBody(b);
	attachSphere(world, a, 0, btScalar(0.5));
	attachSphere(world, b, 0, btScalar(0.5));
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& pa = a->getBasePos();
		const btVector3& pb = b->getBasePos();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);
	}
}

// Scene 19 (§7.6 self_collision): a fixed-base 3-link chain — link 0 hanging
// down from the base, link 1 hanging below link 0, and link 2 a sibling of
// link 1 whose COM sits at (+0.6, +0.2, 0) (e = (+0.6, +1, 0) from link 0's
// COM, d = (0, +0.2, 0)), overlapping the base sphere by ~0.17 m — a
// non-adjacent pair, and the offset keeps a torque arm on both joint axes
// (a COM on the base-to-sphere line would make the contact force pass
// through both pivot axes and the row degenerate to zero). Phase `on` (default self-collision): the contact kicks the chain
// apart while gravity swings it; phase `off` (setHasSelfCollision(false)):
// the overlapping pair is filtered and the chain evolves under pure ABA.
// Records the three joint positions in two labelled phases.
static void scene_self_collision(int frames)
{
	for (int phase = 0; phase < 2; ++phase)
	{
		btMultiBodyDynamicsWorld* world = makeWorld();
		btMultiBody* body = new btMultiBody(3, 1, btVector3(1, 1, 1), true, false);
		body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
							btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(0, -1, 0));
		body->setupRevolute(1, 1, btVector3(1, 1, 1), 0, btQuaternion(0, 0, 0, 1),
							btVector3(0, 0, 1), btVector3(0, -1, 0), btVector3(0, 0, 0));
		body->setupRevolute(2, 1, btVector3(1, 1, 1), 0, btQuaternion(0, 0, 0, 1),
							btVector3(0, 0, 1), btVector3(btScalar(0.6), 1, 0), btVector3(0, btScalar(0.2), 0));
		body->finalizeMultiDof();
		if (phase == 1)
			body->setHasSelfCollision(false);
		refreshFk(body);
		world->addMultiBody(body);
		attachSphere(world, body, -1, btScalar(0.5));
		attachSphere(world, body, 0, btScalar(0.3));
		attachSphere(world, body, 1, btScalar(0.3));
		attachSphere(world, body, 2, btScalar(0.3));
		const char* label = phase == 0 ? "on" : "off";
		for (int f = 0; f < frames; ++f)
		{
			world->stepSimulation(DT, 1, DT);
			printf("%s %d %.6f %.6f %.6f\n", label, f,
				   body->getJointPos(0), body->getJointPos(1), body->getJointPos(2));
		}
	}
}

// Scene 20 (phase E joint feedback): a fixed-base single-link revolute
// pendulum released from the horizontal with a joint feedback slot on the
// link; records the joint state and the reaction force/torque in the link
// frame (the solver-info defaults). Exercises the external-force pass
// write (`=`) — no contacts, so the constraint pass adds ~zero.
static void scene_joint_feedback(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(1, 0, 0));
	body->finalizeMultiDof();
	world->addMultiBody(body);
	addColliders(world, body);
	static btMultiBodyJointFeedback feedback;
	body->getLink(0).m_jointFeedback = &feedback;
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btSpatialForceVector& r = feedback.m_reactionForces;
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", f,
			   body->getJointPos(0), body->getJointVel(0),
			   r.m_topVec[0], r.m_topVec[1], r.m_topVec[2],
			   r.m_bottomVec[0], r.m_bottomVec[1], r.m_bottomVec[2]);
	}
}

// Scene 21 (phase E joint feedback + contact): a link-ground-style pendulum
// with a feedback slot; records the joint state and reaction force/torque
// through the ground impact and settle. Exercises the constraint pass
// accumulation (`+=`) of the contact support on top of the external-force
// write. The pivot sits 1.05 m up so the sphere (r = 0.3, COM 1 m below the
// pivot) first touches the ground at a ~41 degree lean — a well-conditioned
// contact (the default 1.25 m pivot settles at a ~16 degree lean where the
// normal-friction coupling is near-singular and the solver impulses inflate
// in both engines).
static void scene_joint_feedback_contact(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	addGroundBox(world);
	btMultiBody* body = new btMultiBody(1, 1, btVector3(1, 1, 1), true, false);
	body->setupRevolute(0, 1, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
						btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(0, -1, 0));
	body->finalizeMultiDof();
	btScalar q = SIMD_HALF_PI;
	body->setJointPosMultiDof(0, &q);
	body->setBasePos(btVector3(0, btScalar(1.05), 0));
	refreshFk(body);
	world->addMultiBody(body);
	attachSphere(world, body, 0, btScalar(0.3), btScalar(0));
	static btMultiBodyJointFeedback feedback;
	body->getLink(0).m_jointFeedback = &feedback;
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btSpatialForceVector& r = feedback.m_reactionForces;
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", f,
			   body->getJointPos(0), body->getJointVel(0),
			   r.m_topVec[0], r.m_topVec[1], r.m_topVec[2],
			   r.m_bottomVec[0], r.m_bottomVec[1], r.m_bottomVec[2]);
	}
}

// A fixed-base zero-link multibody acting as an immovable anchor at the
// origin, plus a dynamic rigid sphere (mass 1, radius 0.1, analytic sphere
// inertia — matching the Rust port's new_dynamic) at `pos`.
static btRigidBody* addRigidBall(btMultiBodyDynamicsWorld* world, const btVector3& pos)
{
	btSphereShape* shape = new btSphereShape(btScalar(0.1));
	btVector3 inertia(0, 0, 0);
	shape->calculateLocalInertia(1, inertia);
	btDefaultMotionState* ms = new btDefaultMotionState(
		btTransform(btQuaternion(0, 0, 0, 1), pos));
	btRigidBody::btRigidBodyConstructionInfo ci(1, ms, shape, inertia);
	btRigidBody* ball = new btRigidBody(ci);
	ball->setActivationState(DISABLE_DEACTIVATION);
	world->addRigidBody(ball);
	return ball;
}

// Scene 20 (mb-vs-rb p2p): the anchor holds a rigid ball through
// btMultiBodyPoint2Point with the rigid body on side B; the ball hangs 1 m
// below the anchor (pivot at its local (0,1,0)) and swings from a +X
// velocity. Records the ball position and velocity.
static void scene_p2p_rb(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* anchor = new btMultiBody(0, 1, btVector3(1, 1, 1), true, false);
	anchor->setLinearDamping(0);
	anchor->setAngularDamping(0);
	anchor->finalizeMultiDof();
	world->addMultiBody(anchor);
	addColliders(world, anchor);
	btRigidBody* ball = addRigidBall(world, btVector3(0, -1, 0));
	ball->setLinearVelocity(btVector3(1.5, 0, 0));
	btMultiBodyPoint2Point* c = new btMultiBodyPoint2Point(anchor, -1, ball,
		btVector3(0, 0, 0), btVector3(0, 1, 0));
	world->addMultiBodyConstraint(c);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = ball->getWorldTransform().getOrigin();
		const btVector3& v = ball->getLinearVelocity();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

// Scene 21 (mb-vs-rb fixed): the anchor welds a rigid ball 2 m up the +Z
// axis through btMultiBodyFixedConstraint (identity frames); gravity must
// not move it. Records the ball position and velocity.
static void scene_fixed_rb(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* anchor = new btMultiBody(0, 1, btVector3(1, 1, 1), true, false);
	anchor->setLinearDamping(0);
	anchor->setAngularDamping(0);
	anchor->finalizeMultiDof();
	world->addMultiBody(anchor);
	addColliders(world, anchor);
	btRigidBody* ball = addRigidBall(world, btVector3(0, 0, 2));
	btMultiBodyFixedConstraint* c = new btMultiBodyFixedConstraint(anchor, -1, ball,
		btVector3(0, 0, 2), btVector3(0, 0, 0),
		btMatrix3x3::getIdentity(), btMatrix3x3::getIdentity());
	world->addMultiBodyConstraint(c);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = ball->getWorldTransform().getOrigin();
		const btVector3& v = ball->getLinearVelocity();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

// Scene 22 (mb-vs-rb slider): the anchor slides a rigid ball along +X
// through btMultiBodySliderConstraint (identity frames, joint axis +X);
// the ball starts 0.5 m out (clear of the anchor collider) with a +X
// velocity and must neither fall nor leave the axis. Records the ball
// position and velocity.
static void scene_slider_rb(int frames)
{
	btMultiBodyDynamicsWorld* world = makeWorld();
	btMultiBody* anchor = new btMultiBody(0, 1, btVector3(1, 1, 1), true, false);
	anchor->setLinearDamping(0);
	anchor->setAngularDamping(0);
	anchor->finalizeMultiDof();
	world->addMultiBody(anchor);
	addColliders(world, anchor);
	btRigidBody* ball = addRigidBall(world, btVector3(0.5, 0, 0));
	ball->setLinearVelocity(btVector3(1, 0, 0));
	btMultiBodySliderConstraint* c = new btMultiBodySliderConstraint(anchor, -1, ball,
		btVector3(0.5, 0, 0), btVector3(0, 0, 0),
		btMatrix3x3::getIdentity(), btMatrix3x3::getIdentity(), btVector3(1, 0, 0));
	world->addMultiBodyConstraint(c);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = ball->getWorldTransform().getOrigin();
		const btVector3& v = ball->getLinearVelocity();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s <free_base|pendulum|two_link|p2p|motor|fixed|prismatic|spherical|planar|gear|joint_limit|spherical_limit|spherical_motor|link_ground|base_ground|mb_pushes_rb|rb_hits_mb|mb_mb|self_collision|joint_feedback|joint_feedback_contact|p2p_rb|fixed_rb|slider_rb> [frames]\n", argv[0]);
		return 1;
	}
	const char* scene = argv[1];
	int frames = argc > 2 ? atoi(argv[2]) : DEFAULT_FRAMES;
	if (!strcmp(scene, "free_base"))
		scene_free_base(frames);
	else if (!strcmp(scene, "pendulum"))
		scene_pendulum(frames);
	else if (!strcmp(scene, "two_link"))
		scene_two_link(frames);
	else if (!strcmp(scene, "p2p"))
		scene_p2p(frames);
	else if (!strcmp(scene, "motor"))
		scene_motor(frames);
	else if (!strcmp(scene, "fixed"))
		scene_fixed(frames);
	else if (!strcmp(scene, "prismatic"))
		scene_prismatic(frames);
	else if (!strcmp(scene, "spherical"))
		scene_spherical(frames);
	else if (!strcmp(scene, "planar"))
		scene_planar(frames);
	else if (!strcmp(scene, "gear"))
		scene_gear(frames);
	else if (!strcmp(scene, "joint_limit"))
		scene_joint_limit(frames);
	else if (!strcmp(scene, "spherical_limit"))
		scene_spherical_limit(frames);
	else if (!strcmp(scene, "spherical_motor"))
		scene_spherical_motor(frames);
	else if (!strcmp(scene, "link_ground"))
		scene_link_ground(frames);
	else if (!strcmp(scene, "base_ground"))
		scene_base_ground(frames);
	else if (!strcmp(scene, "mb_pushes_rb"))
		scene_mb_pushes_rb(frames);
	else if (!strcmp(scene, "rb_hits_mb"))
		scene_rb_hits_mb(frames);
	else if (!strcmp(scene, "mb_mb"))
		scene_mb_mb(frames);
	else if (!strcmp(scene, "self_collision"))
		scene_self_collision(frames);
	else if (!strcmp(scene, "joint_feedback"))
		scene_joint_feedback(frames);
	else if (!strcmp(scene, "joint_feedback_contact"))
		scene_joint_feedback_contact(frames);
	else if (!strcmp(scene, "p2p_rb"))
		scene_p2p_rb(frames);
	else if (!strcmp(scene, "fixed_rb"))
		scene_fixed_rb(frames);
	else if (!strcmp(scene, "slider_rb"))
		scene_slider_rb(frames);
	else
	{
		fprintf(stderr, "unknown scene %s\n", scene);
		return 1;
	}
	return 0;
}
