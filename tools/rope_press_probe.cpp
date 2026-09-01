// Rope-press probe: replicates rope_pushes_dynamic_box against stock
// Bullet — a 20-node rope (total mass 2) dropped from y=4 onto a box
// (half 0.5, mass 5) resting on the ground from y=1.5. Prints the box y.
#include <cstdio>
#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftBody.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>

int main()
{
        btSoftBodyRigidBodyCollisionConfiguration config;
        btCollisionDispatcher dispatcher(&config);
        btDbvtBroadphase broadphase;
        btSequentialImpulseConstraintSolver solver;
        btSoftRigidDynamicsWorld world(&dispatcher, &broadphase, &solver, &config);
        world.setGravity(btVector3(0, -9.81, 0));
        world.getWorldInfo().m_gravity = btVector3(0, -9.81, 0);
        world.getSolverInfo().m_solverMode &= ~SOLVER_RANDMIZE_ORDER;

        btStaticPlaneShape groundShape(btVector3(0, 1, 0), 0);
        btRigidBody ground(btRigidBody::btRigidBodyConstructionInfo(0, nullptr, &groundShape, btVector3(0, 0, 0)));
        world.addRigidBody(&ground);

        btBoxShape boxShape(btVector3(btScalar(0.5), btScalar(0.5), btScalar(0.5)));
        btVector3 inertia(0, 0, 0);
        boxShape.calculateLocalInertia(5, inertia);
        btDefaultMotionState ms(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, btScalar(1.5), 0)));
        btRigidBody box(btRigidBody::btRigidBodyConstructionInfo(5, &ms, &boxShape, inertia));
        world.addRigidBody(&box);

        btSoftBody* rope = btSoftBodyHelpers::CreateRope(
                world.getWorldInfo(), btVector3(0, btScalar(4), 0),
                btVector3(0, btScalar(2), 0), 19, 8);
        rope->setTotalMass(2);
        rope->m_cfg.piterations = 4;
        world.addSoftBody(rope);

        const btScalar dt = btScalar(1) / btScalar(60);
        for (int f = 1; f <= 600; ++f)
        {
                world.stepSimulation(dt, 1, dt);
                if (f % 100 == 0)
                        printf("f%3d box_y=%.4f\n", f, box.getWorldTransform().getOrigin().y());
        }
        printf("final box_y=%.4f\n", box.getWorldTransform().getOrigin().y());
        return 0;
}
