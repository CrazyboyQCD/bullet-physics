// Conetwist stress probe: a 4-link chain hanging along -X (gravity -X) with
// swing limits at 30 degrees, slammed with a lateral velocity kick and kept
// loaded by a steady force on the tip, so every joint rides its swing limit.
// Prints the tip state, per-joint relative swing angles, and per-body angular
// speeds for tolerance diffing (tools/mbdiff.py) and chatter analysis.
// Twin: crates/bullet-physics/examples/conetwist_stress_probe.rs.
//
// argv: [frames] [softness] [relaxation]
#include <cstdlib>
#include <cmath>
#include <cstdio>

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/ConstraintSolver/btConeTwistConstraint.h>

static btScalar swingAngle(const btMatrix3x3& prev, const btMatrix3x3& cur) {
    btVector3 a = prev.getColumn(0);
    btVector3 b = cur.getColumn(0);
    btScalar d = a.dot(b);
    if (d > btScalar(1.0)) d = btScalar(1.0);
    if (d < btScalar(-1.0)) d = btScalar(-1.0);
    return std::acos(d);
}

int main(int argc, char** argv) {
    const int frames = argc > 1 ? std::atoi(argv[1]) : 600;
    const btScalar softness = argc > 2 ? btScalar(std::atof(argv[2])) : btScalar(1.0);
    const btScalar relaxation = argc > 3 ? btScalar(std::atof(argv[3])) : btScalar(1.0);
    const int mode = argc > 4 ? std::atoi(argv[4]) : 1;  // 0 = free body, 1 = joint

    btDefaultCollisionConfiguration config;
    btCollisionDispatcher dispatcher(&config);
    btDbvtBroadphase broadphase;
    btSequentialImpulseConstraintSolver solver;
    btDiscreteDynamicsWorld world(&dispatcher, &broadphase, &solver, &config);
    world.setGravity(btVector3(btScalar(-9.81), 0, 0));

    const int kLinks = 1;
    btBoxShape anchorShape(btVector3(0.1, 0.1, 0.1));
    btBoxShape linkShape(btVector3(0.15, 0.1, 0.1));
    btVector3 localInertia(0, 0, 0);
    linkShape.calculateLocalInertia(btScalar(1.0), localInertia);

    btRigidBody::btRigidBodyConstructionInfo anchorInfo(
        0, nullptr, &anchorShape, btVector3(0, 0, 0));
    anchorInfo.m_startWorldTransform.setOrigin(btVector3(0, 0, 0));
    btRigidBody* anchor = new btRigidBody(anchorInfo);
    world.addRigidBody(anchor);

    btRigidBody* links[kLinks];
    for (int i = 0; i < kLinks; ++i) {
        btRigidBody::btRigidBodyConstructionInfo ci(
            btScalar(1.0), nullptr, &linkShape, localInertia);
        ci.m_startWorldTransform.setOrigin(
            btVector3(btScalar(-0.25 - 0.3 * i), 0, 0));
        links[i] = new btRigidBody(ci);
        links[i]->setDamping(btScalar(0), btScalar(0));
        links[i]->setActivationState(DISABLE_DEACTIVATION);
        if (mode != 2) {
            links[i]->setLinearVelocity(btVector3(0, 2, 0));
        }
        world.addRigidBody(links[i]);
    }

    const btScalar SWING = btScalar(0.5235987755982988);  // 30 degrees
    const btScalar TWIST = btScalar(1.0471975511965976);  // 60 degrees
    if (mode != 0) {
        btTransform frameA, frameB;
        frameA.setIdentity();
        frameB.setIdentity();
        frameA.setOrigin(btVector3(btScalar(-0.1), 0, 0));
        frameB.setOrigin(btVector3(btScalar(0.15), 0, 0));
        btConeTwistConstraint* joint0 =
            new btConeTwistConstraint(*anchor, *links[0], frameA, frameB);
        joint0->setLimit(SWING, SWING, TWIST, softness, btScalar(0.3), relaxation);
        world.addConstraint(joint0);
    }

    const btScalar DT = btScalar(1.0) / 60;
    for (int f = 0; f < frames; ++f) {
        links[kLinks - 1]->applyCentralImpulse(btVector3(0, 25 * DT, 0));
        world.stepSimulation(DT);
        btMatrix3x3 prev(btMatrix3x3::getIdentity());
        printf("%d", f);
        const btVector3 p = links[kLinks - 1]->getCenterOfMassTransform().getOrigin();
        const btVector3 v = links[kLinks - 1]->getLinearVelocity();
        printf(" %.6f %.6f %.6f %.6f %.6f %.6f",
               p.x(), p.y(), p.z(), v.x(), v.y(), v.z());
        for (int i = 0; i < kLinks; ++i) {
            const btMatrix3x3 cur = links[i]->getCenterOfMassTransform().getBasis();
            printf(" %.6f", swingAngle(prev, cur));
            prev = cur;
        }
        for (int i = 0; i < kLinks; ++i) {
            printf(" %.6f", links[i]->getAngularVelocity().length());
        }
        printf("\n");
    }
    return 0;
}
