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

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s <free_base|pendulum|two_link|p2p|motor|fixed|prismatic|spherical|planar|gear> [frames]\n", argv[0]);
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
	else
	{
		fprintf(stderr, "unknown scene %s\n", scene);
		return 1;
	}
	return 0;
}
