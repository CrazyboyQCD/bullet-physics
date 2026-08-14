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

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s <free_base|pendulum|two_link|p2p|motor|fixed> [frames]\n", argv[0]);
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
	else
	{
		fprintf(stderr, "unknown scene %s\n", scene);
		return 1;
	}
	return 0;
}
