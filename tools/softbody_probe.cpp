// SoftBody probe: runs btSoftBody rope/cloth scenes against the real Bullet
// library and prints per-frame trajectories for numerical comparison against
// the Rust port (phase B of the classic btSoftBody port — see
// docs/softbody.md §10).
//
// Scenes:
//   rope_pendulum  rope pinned at one end and released horizontally — pure
//                  link PBD, no contact (the phase-A numerical check).
//   rope_ground    rope dropped onto the ground plane — Bullet's sparse-SDF
//                  voxel contact vs our exact analytic plane distance.
//   cloth_plane    12x12 patch dropped onto the ground plane (the Rust twin
//                  builds nodes+links only; faces carry no position dynamics
//                  before the cloth phase).
//
// Build with tools/build_softbody_probe.py; usage:
//   softbody_probe.exe <rope_pendulum|rope_ground|cloth_plane> [frames]

#include <cstdio>
#include <cstring>

#include <btBulletDynamicsCommon.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>

static const btScalar DT = btScalar(1) / btScalar(60);
static const int DEFAULT_FRAMES = 300;

static btSoftRigidDynamicsWorld* makeWorld()
{
	btSoftBodyRigidBodyCollisionConfiguration* config = new btSoftBodyRigidBodyCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(config);
	btDbvtBroadphase* broadphase = new btDbvtBroadphase();
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver();
	btSoftRigidDynamicsWorld* world = new btSoftRigidDynamicsWorld(dispatcher, broadphase, solver, config);
	world->setGravity(btVector3(0, -9.81, 0));
	// Reproducible solving order (same as the parity harness). The soft world
	// info gravity defaults to (0,-10,0) and setGravity does not sync it.
	world->getSolverInfo().m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
	world->getWorldInfo().m_gravity.setValue(0, -9.81, 0);
	return world;
}

static btRigidBody* addGround(btSoftRigidDynamicsWorld* world)
{
	btStaticPlaneShape* shape = new btStaticPlaneShape(btVector3(0, 1, 0), 0);
	btRigidBody* ground = new btRigidBody(0, nullptr, shape);
	world->addRigidBody(ground);
	return ground;
}

// Mass-weighted center of mass (same formula as the Rust twin's
// get_center_of_mass; btSoftBody::getCenterOfMass relies on pose weights that
// the rope/patch builders never initialize).
static btVector3 centerOfMass(const btSoftBody* psb)
{
	btVector3 com(0, 0, 0);
	btScalar total = 0;
	for (int i = 0; i < psb->m_nodes.size(); ++i)
	{
		const btScalar m = psb->m_nodes[i].m_im > 0 ? 1 / psb->m_nodes[i].m_im : 0;
		com += psb->m_nodes[i].m_x * m;
		total += m;
	}
	return total > 0 ? com / total : com;
}

static btScalar minY(const btSoftBody* psb)
{
	btScalar y = psb->m_nodes[0].m_x.y();
	for (int i = 1; i < psb->m_nodes.size(); ++i)
		y = btMin(y, psb->m_nodes[i].m_x.y());
	return y;
}

// Scene 1: a rope pinned at (0,8,0), released from the horizontal (18 nodes,
// piterations = 4). Tip position and mass-weighted COM per frame; the rope
// never reaches the ground (lowest point y=4 > margin), so this is a pure
// link-PBD trajectory check.
static void scene_rope_pendulum(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	btSoftBody* rope = btSoftBodyHelpers::CreateRope(world->getWorldInfo(),
	                                                 btVector3(0, 8, 0), btVector3(4, 8, 0),
	                                                 16, 1);
	rope->m_cfg.piterations = 4;
	world->addSoftBody(rope);
	const int last = rope->m_nodes.size() - 1;
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& tip = rope->m_nodes[last].m_x;
		const btVector3 com = centerOfMass(rope);
		printf("%d %.6f %.6f %.6f %.6f\n", f, tip.x(), tip.y(), com.x(), com.y());
	}
}

// Scene 2: a horizontal 22-node rope dropped from y=5 onto the ground plane.
// Settles inside the 0.25 contact band; quantifies the SDF-voxel vs exact
// plane-distance settle difference.
static void scene_rope_ground(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	addGround(world);
	btSoftBody* rope = btSoftBodyHelpers::CreateRope(world->getWorldInfo(),
	                                                 btVector3(-3, 5, 0), btVector3(3, 5, 0),
	                                                 20, 0);
	rope->m_cfg.piterations = 4;
	world->addSoftBody(rope);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 com = centerOfMass(rope);
		printf("%d %.6f %.6f %.6f\n", f, minY(rope), com.y(), com.x());
	}
}

// Scene 3: a 12x12 patch (4x4 m, diagonal links) dropped flat from y=5 onto
// the ground plane.
static void scene_cloth_plane(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	addGround(world);
	btSoftBody* cloth = btSoftBodyHelpers::CreatePatch(world->getWorldInfo(),
	                                                   btVector3(-2, 5, -2), btVector3(2, 5, -2),
	                                                   btVector3(-2, 5, 2), btVector3(2, 5, 2),
	                                                   12, 12, 0, true);
	cloth->m_cfg.piterations = 4;
	world->addSoftBody(cloth);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 com = centerOfMass(cloth);
		printf("%d %.6f %.6f %.6f %.6f\n", f, minY(cloth), com.y(), com.x(), com.z());
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s <rope_pendulum|rope_ground|cloth_plane> [frames]\n", argv[0]);
		return 1;
	}
	const char* scene = argv[1];
	int frames = argc > 2 ? atoi(argv[2]) : DEFAULT_FRAMES;
	if (!strcmp(scene, "rope_pendulum"))
		scene_rope_pendulum(frames);
	else if (!strcmp(scene, "rope_ground"))
		scene_rope_ground(frames);
	else if (!strcmp(scene, "cloth_plane"))
		scene_cloth_plane(frames);
	else
	{
		fprintf(stderr, "unknown scene %s\n", scene);
		return 1;
	}
	return 0;
}
