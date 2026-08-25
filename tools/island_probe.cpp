// Island behavior probe: two boxes touching on a static ground plane go to
// sleep, then a fast ball strikes the left box along +X. Prints per-frame
// activation states and positions to reveal how Bullet's island activation
// pass treats a knocked-away WANTS_DEACTIVATION body.
// Build: python - <<'EOF'
//   import sys; sys.path.insert(0, 'tools')
//   from build_recorder import msvc_env
//   ... (cl island_probe.cpp /I ... /Fe:tools/island_probe.exe /link ...lib)
//   EOF
#include <cstdio>

#include <btBulletDynamicsCommon.h>

int main()
{
	btDefaultCollisionConfiguration* config = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(config);
	btDbvtBroadphase* broadphase = new btDbvtBroadphase();
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver();
	btDiscreteDynamicsWorld* world = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, config);
	world->setGravity(btVector3(0, -9.81, 0));
	world->getSolverInfo().m_solverMode &= ~SOLVER_RANDMIZE_ORDER;

	btStaticPlaneShape* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 0);
	btRigidBody* ground = new btRigidBody(btRigidBody::btRigidBodyConstructionInfo(0.f, nullptr, groundShape, btVector3(0, 0, 0)));
	world->addRigidBody(ground);

	btBoxShape* boxShape = new btBoxShape(btVector3(0.5f, 0.5f, 0.5f));
	btVector3 inertia(0, 0, 0);
	boxShape->calculateLocalInertia(1.f, inertia);

	btDefaultMotionState* msL = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(-0.5f, 0.5f, 0)));
	btRigidBody* left = new btRigidBody(btRigidBody::btRigidBodyConstructionInfo(1.f, msL, boxShape, inertia));
	world->addRigidBody(left);
	btDefaultMotionState* msR = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0.5f, 0.5f, 0)));
	btRigidBody* right = new btRigidBody(btRigidBody::btRigidBodyConstructionInfo(1.f, msR, boxShape, inertia));
	world->addRigidBody(right);

	const btScalar dt = btScalar(1) / btScalar(60);
	// 4 s: both boxes settle and sleep.
	for (int i = 0; i < 240; ++i)
		world->stepSimulation(dt, 1, dt);
	printf("after sleep: l=%d r=%d\n", (int)left->getActivationState(), (int)right->getActivationState());

	// Fast ball strikes the left box's -X face along +X.
	btSphereShape* ballShape = new btSphereShape(0.25f);
	btVector3 ballInertia(0, 0, 0);
	ballShape->calculateLocalInertia(1.f, ballInertia);
	btDefaultMotionState* msB = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(-3, 0.5f, 0)));
	btRigidBody* ball = new btRigidBody(btRigidBody::btRigidBodyConstructionInfo(1.f, msB, ballShape, ballInertia));
	ball->setLinearVelocity(btVector3(8, 0, 0));
	world->addRigidBody(ball);

	for (int i = 0; i < 60; ++i)
	{
		int pre = (int)left->getActivationState();
		world->stepSimulation(dt, 1, dt);
		printf("f%d: pre=%d wants=%d t=%.3f l=%d@(%.4f,%.4f) v=(%.2f,%.2f) r=%d@(%.4f,%.4f) v=(%.2f,%.2f) b=%d@(%.4f)\n",
			i, pre, (int)left->wantsSleeping(), left->getDeactivationTime(),
			(int)left->getActivationState(), left->getWorldTransform().getOrigin().x(), left->getWorldTransform().getOrigin().y(),
			left->getLinearVelocity().x(), left->getLinearVelocity().y(),
			(int)right->getActivationState(), right->getWorldTransform().getOrigin().x(), right->getWorldTransform().getOrigin().y(),
			right->getLinearVelocity().x(), right->getLinearVelocity().y(),
			(int)ball->getActivationState(), ball->getWorldTransform().getOrigin().x());
	}
	return 0;
}
