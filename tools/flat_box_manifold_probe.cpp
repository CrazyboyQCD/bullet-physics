
#include <cstdio>
#include <btBulletDynamicsCommon.h>

static void dump_manifolds(btCollisionDispatcher& d, const char* tag)
{
        int n = d.getNumManifolds();
        for (int i = 0; i < n; ++i)
        {
                btPersistentManifold* m = d.getManifoldByIndexInternal(i);
                printf("%s pts=%d", tag, m->getNumContacts());
                for (int j = 0; j < m->getNumContacts(); ++j)
                {
                        const btManifoldPoint& p = m->getContactPoint(j);
                        printf(" [pos=(%.3f,%.3f,%.3f) dist=%+.5f]", p.getPositionWorldOnB().x(), p.getPositionWorldOnB().y(), p.getPositionWorldOnB().z(), p.getDistance());
                }
                printf("\n");
        }
}

int main()
{
        btDefaultCollisionConfiguration config;
        btCollisionDispatcher dispatcher(&config);
        btDbvtBroadphase broadphase;
        btSequentialImpulseConstraintSolver solver;
        btDiscreteDynamicsWorld world(&dispatcher, &broadphase, &solver, &config);
        world.setGravity(btVector3(0, -9.81, 0));
        world.getSolverInfo().m_solverMode &= ~SOLVER_RANDMIZE_ORDER;

        btStaticPlaneShape groundShape(btVector3(0, 1, 0), 0);
        btRigidBody ground(btRigidBody::btRigidBodyConstructionInfo(0.f, nullptr, &groundShape, btVector3(0, 0, 0)));
        world.addRigidBody(&ground);

        btBoxShape boxShape(btVector3(1.f, 0.1f, 1.f));
        btVector3 inertia(0, 0, 0);
        boxShape.calculateLocalInertia(5.f, inertia);
        btDefaultMotionState ms(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0.4f, 0)));
        btRigidBody body(btRigidBody::btRigidBodyConstructionInfo(5.f, &ms, &boxShape, inertia));
        world.addRigidBody(&body);

        const btScalar dt = btScalar(1) / btScalar(60);
        for (int i = 1; i <= 300; ++i)
        {
                world.stepSimulation(dt, 1, dt);
                const btVector3& p = body.getWorldTransform().getOrigin();
                if (i <= 30 || i == 40 || i == 100 || i == 200)
                {
                        printf("f%3d y=%8.4f |w|=%6.4f ", i, p.y(), body.getAngularVelocity().length());
                        dump_manifolds(dispatcher, "m:");
                }
        }
        return 0;
}
