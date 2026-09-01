// Flat-box-on-plane probe: does stock Bullet reproduce the wide-flat-box
// bug seen in this crate (single-corner contact -> diagonal spin -> sink /
// tunnel through a static plane)? Same scenario as the ignored Rust test
// `wide_flat_box_on_plane_known_bug`: box half extents (1, 0.1, 1), mass 5,
// dropped from y = 0.4 onto btStaticPlaneShape((0,1,0), 0). A cube box
// (1,1,1) runs as the control. Prints every 10 frames.
#include <cstdio>
#include <cmath>

#include <btBulletDynamicsCommon.h>

static void run(const char* label, btScalar hx, btScalar hy, btScalar hz)
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

        btBoxShape boxShape(btVector3(hx, hy, hz));
        btVector3 inertia(0, 0, 0);
        boxShape.calculateLocalInertia(5.f, inertia);
        btDefaultMotionState ms(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0.4f, 0)));
        btRigidBody body(btRigidBody::btRigidBodyConstructionInfo(5.f, &ms, &boxShape, inertia));
        world.addRigidBody(&body);

        const btScalar dt = btScalar(1) / btScalar(60);
        btScalar min_y = 10;
        for (int i = 0; i <= 300; ++i)
        {
                if (i)
                        world.stepSimulation(dt, 1, dt);
                const btVector3& p = body.getWorldTransform().getOrigin();
                const btScalar w = body.getAngularVelocity().length();
                if (p.y() < min_y)
                        min_y = p.y();
                if (i % 10 == 0)
                        printf("%s f%3d y=%8.4f |w|=%6.3f min_y=%8.4f sleep=%d\n",
                               label, i, p.y(), w, min_y,
                               (int)(body.getActivationState() != ACTIVE_TAG));
        }
}

int main()
{
        run("flat", 1.f, 0.1f, 1.f);
        run("cube", 0.5f, 0.5f, 0.5f);
        return 0;
}
