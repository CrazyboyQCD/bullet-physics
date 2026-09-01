// Swing-push probe: replicates the two failing world tests against stock
// Bullet — (1) a passive pendulum released from horizontal with its ball
// initially overlapping the box top, box resting on a frictionless plane;
// (2) the same swing against a box that first settles and sleeps. Prints
// the box trajectory and activation state.
#include <cstdio>

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>

static btMultiBodyDynamicsWorld* makeWorld(btDefaultCollisionConfiguration** cfgOut)
{
        auto* config = new btDefaultCollisionConfiguration();
        *cfgOut = config;
        auto* dispatcher = new btCollisionDispatcher(config);
        auto* broadphase = new btDbvtBroadphase();
        auto* solver = new btMultiBodyConstraintSolver();
        auto* world = new btMultiBodyDynamicsWorld(dispatcher, broadphase, solver, config);
        world->setGravity(btVector3(0, -9.81, 0));
        world->getSolverInfo().m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
        return world;
}

static btMultiBody* makePendulum(btMultiBodyDynamicsWorld* world)
{
        btMultiBody* body = new btMultiBody(1, 3, btVector3(1, 1, 1), true, false);
        body->setupRevolute(0, 3, btVector3(1, 1, 1), -1, btQuaternion(0, 0, 0, 1),
                            btVector3(0, 0, 1), btVector3(0, 0, 0), btVector3(0, -1, 0));
        body->finalizeMultiDof();
        body->setBasePos(btVector3(0, 1, 0));
        const btScalar q = btScalar(1.5707964);
        body->setJointPosMultiDof(0, &q);
        world->addMultiBody(body);
        btAlignedObjectArray<btQuaternion> w2l;
        btAlignedObjectArray<btVector3> lo;
        body->forwardKinematics(w2l, lo);
        auto* col = new btMultiBodyLinkCollider(body, 0);
        col->setCollisionShape(new btSphereShape(btScalar(0.3)));
        col->setWorldTransform(body->getLink(0).m_cachedWorldTransform);
        world->addCollisionObject(col, btBroadphaseProxy::DefaultFilter,
                                  btBroadphaseProxy::DefaultFilter | btBroadphaseProxy::StaticFilter);
        body->getLink(0).m_collider = col;
        return body;
}

static btRigidBody* makeBox(btMultiBodyDynamicsWorld* world, bool sleepFirst)
{
        auto* shape = new btBoxShape(btVector3(btScalar(0.3), btScalar(0.3), btScalar(0.3)));
        btVector3 inertia(0, 0, 0);
        shape->calculateLocalInertia(1.f, inertia);  // box mass stays 1
        auto* ms = new btDefaultMotionState(
                btTransform(btQuaternion(0, 0, 0, 1), btVector3(btScalar(0.85), btScalar(0.3), 0)));
        auto* body = new btRigidBody(btRigidBody::btRigidBodyConstructionInfo(1, ms, shape, inertia));
        body->setFriction(0);
        world->addRigidBody(body);
        if (sleepFirst)
        {
                // 3 s idle > the 2 s deactivation window.
                for (int i = 0; i < 180; ++i)
                        world->stepSimulation(btScalar(1) / btScalar(60), 1, btScalar(1) / btScalar(60));
                printf("  after idle: active=%d\n", body->isActive());
        }
        return body;
}

static void run(const char* tag, bool sleepFirst)
{
        btDefaultCollisionConfiguration* config;
        auto* world = makeWorld(&config);
        auto* groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 0);
        btRigidBody ground(btRigidBody::btRigidBodyConstructionInfo(0, nullptr, groundShape, btVector3(0, 0, 0)));
        world->addRigidBody(&ground);
        auto* box = makeBox(world, sleepFirst);
        auto* pend = makePendulum(world);
        const btScalar dt = btScalar(1) / btScalar(60);
        for (int f = 1; f <= 90; ++f)
        {
                world->stepSimulation(dt, 1, dt);
                if (f % 15 == 0)
                {
                        const btVector3& p = box->getWorldTransform().getOrigin();
                        const btVector3& b = pend->getLink(0).m_cachedWorldTransform.getOrigin();
                        printf("%s f%3d box=(%.4f,%.4f,%.4f) active=%d ball=(%.4f,%.4f)\n",
                               tag, f, p.x(), p.y(), p.z(), box->isActive(), b.x(), b.y());
                }
        }
        const btVector3& p = box->getWorldTransform().getOrigin();
        btVector3 move = p - btVector3(btScalar(0.85), btScalar(0.3), 0);
        printf("%s final: |move|=%.4f active=%d\n\n", tag, move.length(), box->isActive());
}

int main()
{
        run("swing", false);
        run("wake ", true);
        return 0;
}
