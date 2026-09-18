// SoftBody probe: runs btSoftBody rope/cloth scenes against the real Bullet
// library and prints per-frame trajectories for numerical comparison against
// the Rust port (phase B of the classic btSoftBody port — see
// docs/design/softbody.md §10).
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

// Scene 4: a 12x12 patch pinned at its four corners, released horizontally
// — pure cloth catenary (links + gravity, no contact below).
static void scene_patch_catenary(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	btSoftBody* cloth = btSoftBodyHelpers::CreatePatch(world->getWorldInfo(),
	                                                   btVector3(-2, 6, -2), btVector3(2, 6, -2),
	                                                   btVector3(-2, 6, 2), btVector3(2, 6, 2),
	                                                   12, 12, 1 + 2 + 4 + 8, true);
	cloth->m_cfg.piterations = 4;
	world->addSoftBody(cloth);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 com = centerOfMass(cloth);
		printf("%d %.6f %.6f %.6f\n", f, cloth->m_nodes[5 * 12 + 5].m_x.y(), minY(cloth), com.y());
	}
}

// Scene 5: one patch drops onto another mid-air patch — the VF_SS
// (vertex-vs-face soft-soft) path. Both bodies opt in (the default
// collision flags are SDF_RS only).
static void scene_soft_soft_vf(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	addGround(world);
	btSoftBody* upper = btSoftBodyHelpers::CreatePatch(world->getWorldInfo(),
	                                                   btVector3(-2, 6, -2), btVector3(2, 6, -2),
	                                                   btVector3(-2, 6, 2), btVector3(2, 6, 2),
	                                                   12, 12, 0, true);
	btSoftBody* lower = btSoftBodyHelpers::CreatePatch(world->getWorldInfo(),
	                                                   btVector3(-2, 3, -2), btVector3(2, 3, -2),
	                                                   btVector3(-2, 3, 2), btVector3(2, 3, 2),
	                                                   12, 12, 0, true);
	upper->m_cfg.piterations = 4;
	lower->m_cfg.piterations = 4;
	upper->m_cfg.collisions = btSoftBody::fCollision::Default | btSoftBody::fCollision::VF_SS;
	lower->m_cfg.collisions = btSoftBody::fCollision::Default | btSoftBody::fCollision::VF_SS;
	world->addSoftBody(upper);
	world->addSoftBody(lower);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		printf("%d %.6f %.6f %.6f %.6f\n", f, minY(upper), minY(lower),
		       centerOfMass(upper).y(), centerOfMass(lower).y());
	}
}

// Scene 6: a volumetric ellipsoid (Hammersley hull, CL_RS clusters) dropped
// from y=4 onto the ground plane — the cluster collider + CJoint path.
// Both sides enable matching on every cluster and soften the structural
// material (kLST = 0.5) so the squash/recover character is visible.
static void scene_ellipsoid_ground(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	addGround(world);
	btSoftBody* ball = btSoftBodyHelpers::CreateEllipsoid(world->getWorldInfo(),
	                                                      btVector3(0, 4, 0),
	                                                      btVector3(1, 1, 1), 96);
	ball->m_cfg.piterations = 4;
	ball->m_cfg.collisions = btSoftBody::fCollision::CL_RS;
	ball->setTotalMass(30, true);
	ball->generateBendingConstraints(2);
	ball->generateClusters(16, 32);
	ball->m_materials[0]->m_kLST = btScalar(0.5);
	for (int i = 0; i < ball->m_clusters.size(); ++i)
	{
		ball->m_clusters[i]->m_matching = btScalar(0.35);
	}
	world->addSoftBody(ball);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 com = centerOfMass(ball);
		btScalar half = 0;
		for (int i = 0; i < ball->m_nodes.size(); ++i)
		{
			half = btMax<btScalar>(half, btFabs(ball->m_nodes[i].m_x.y() - com.y()));
		}
		printf("%d %.6f %.6f %.6f\n", f, minY(ball), com.y(), half);
	}
}

// Scene 7: two cluster ellipsoids resting side by side push apart through
// CL_SS (the CJoint drift/impulse path), then settle on the ground.
static void scene_cluster_pair(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	addGround(world);
	btSoftBody* a = btSoftBodyHelpers::CreateEllipsoid(world->getWorldInfo(),
	                                                   btVector3(-0.9, 1.5, 0),
	                                                   btVector3(1, 0.8, 1), 64);
	btSoftBody* b = btSoftBodyHelpers::CreateEllipsoid(world->getWorldInfo(),
	                                                   btVector3(0.9, 1.5, 0),
	                                                   btVector3(1, 0.8, 1), 64);
	for (btSoftBody* psb : {a, b})
	{
		psb->m_cfg.piterations = 4;
		psb->m_cfg.collisions = btSoftBody::fCollision::CL_RS | btSoftBody::fCollision::CL_SS;
		psb->setTotalMass(20, true);
		psb->generateBendingConstraints(2);
		psb->generateClusters(16, 32);
		psb->m_materials[0]->m_kLST = btScalar(0.5);
		for (int i = 0; i < psb->m_clusters.size(); ++i)
		{
			psb->m_clusters[i]->m_matching = btScalar(0.35);
		}
		world->addSoftBody(psb);
	}
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 ca = centerOfMass(a);
		const btVector3 cb = centerOfMass(b);
		printf("%d %.6f %.6f %.6f\n", f, (ca - cb).length(),
		       btMin(minY(a), minY(b)), ca.x());
	}
}

// Scene 8: a 18-node rope draped inside the SDF_RS band across a static
// sphere (r = 0.7 at (0, 4.7, 0)), offset +0.5 to one side so its weight
// swings the ball if the contact couples. Both engines hold the middle
// nodes on the band ~30 frames, then the rope slides off the side and
// falls (the port matches within a few frames of phase slip). A short
// rope DROPPED onto the sphere at ~3.9 m/s tunnels through in BOTH
// engines (measured: bullet min_y -186.8 vs ours -181.0 over 400 frames;
// the classic position solve only partially corrects each frame), so the
// scene uses the draped variant for a meaningful comparison.
static void scene_rope_ball_tunnel(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	btSphereShape* shape = new btSphereShape(btScalar(0.7));
	btRigidBody* ball = new btRigidBody(0, nullptr, shape);
	ball->getWorldTransform().setOrigin(btVector3(0, btScalar(4.7), 0));
	world->addRigidBody(ball);
	btSoftBody* rope = btSoftBodyHelpers::CreateRope(world->getWorldInfo(),
	                                                 btVector3(btScalar(-0.4), btScalar(5.55), 0),
	                                                 btVector3(btScalar(1.6), btScalar(5.55), 0),
	                                                 16, 0);
	rope->m_cfg.piterations = 4;
	world->addSoftBody(rope);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 com = centerOfMass(rope);
		printf("%d %.6f %.6f %.6f", f, minY(rope), com.y(), com.x());
		for (int i = 0; i < rope->m_nodes.size(); ++i)
			printf(" %.6f", rope->m_nodes[i].m_x.y());
		printf("\n");
	}
}

// Scene 8: two 7x7 patches mid-air (no ground - keeps the scene inside the
// deterministic-helpers class; the ellipsoid hull vertex-order scatter of
// the Phase D probes never enters), each split into 4 K-mean clusters,
// pinned to each other through a LJoint at (0, 4, 0) plus an AJoint on +Y
// (master clusters, specs defaults erp = cfm = split = 1, default
// IControl); patch B starts with +2 m/s so the joint fight is non-trivial.
// Exercises the user-joint half of the cluster-core solve (prepare /
// citerations x solve / terminate with the drift split). Patches, not
// ropes: a straight rope's master cluster is a colinear point set whose
// inertia matrix is singular (Bullet's initializeClusters inverts it
// unguarded - both engines NaN identically, an upstream capacity edge the
// scene must stay clear of).
static void scene_cluster_joints(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	btSoftBody* a = btSoftBodyHelpers::CreatePatch(world->getWorldInfo(),
	                                               btVector3(-2, 5, -2), btVector3(2, 5, -2),
	                                               btVector3(-2, 5, 2), btVector3(2, 5, 2),
	                                               7, 7, 0, true);
	btSoftBody* b = btSoftBodyHelpers::CreatePatch(world->getWorldInfo(),
	                                               btVector3(-2, 3, -2), btVector3(2, 3, -2),
	                                               btVector3(-2, 3, 2), btVector3(2, 3, 2),
	                                               7, 7, 0, true);
	for (btSoftBody* psb : {a, b})
	{
		psb->m_cfg.piterations = 4;
		psb->generateClusters(4, 32);
		world->addSoftBody(psb);
	}
	b->addVelocity(btVector3(0, 2, 0));
	btSoftBody::LJoint::Specs lin;
	lin.position = btVector3(0, 4, 0);
	a->appendLinearJoint(lin, b);
	btSoftBody::AJoint::Specs ang;
	ang.axis = btVector3(0, 1, 0);
	a->appendAngularJoint(ang, b);
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 c0 = a->m_clusters[0]->m_framexform.getOrigin();
		const btVector3 c1 = b->m_clusters[0]->m_framexform.getOrigin();
		printf("%d %.6f %.6f %.6f %.6f\n", f, c0.x(), c0.y(),
		       (c0 - c1).length(), centerOfMass(a).y());
	}
}

// Diagnostic variant of scene_cluster_pair with a light touch: the
// ellipsoids start 0.3 m apart (inside the 0.5 combined-margin band, no
// deep overlap), isolating the large-overlap blow-up regime.
static void scene_cluster_pair_soft(int frames)
{
	btSoftRigidDynamicsWorld* world = makeWorld();
	addGround(world);
	btSoftBody* a = btSoftBodyHelpers::CreateEllipsoid(world->getWorldInfo(),
	                                                   btVector3(-1.15, 1.5, 0),
	                                                   btVector3(1, 0.8, 1), 64);
	btSoftBody* b = btSoftBodyHelpers::CreateEllipsoid(world->getWorldInfo(),
	                                                   btVector3(1.15, 1.5, 0),
	                                                   btVector3(1, 0.8, 1), 64);
	for (btSoftBody* psb : {a, b})
	{
		psb->m_cfg.piterations = 4;
		psb->m_cfg.collisions = btSoftBody::fCollision::CL_RS | btSoftBody::fCollision::CL_SS;
		psb->setTotalMass(20, true);
		psb->generateBendingConstraints(2);
		psb->generateClusters(16, 32);
		psb->m_materials[0]->m_kLST = btScalar(0.5);
		for (int i = 0; i < psb->m_clusters.size(); ++i)
		{
			psb->m_clusters[i]->m_matching = btScalar(0.35);
		}
		world->addSoftBody(psb);
	}
	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3 ca = centerOfMass(a);
		const btVector3 cb = centerOfMass(b);
		printf("%d %.6f %.6f %.6f\n", f, (ca - cb).length(),
		       btMin(minY(a), minY(b)), ca.x());
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s <rope_pendulum|rope_ground|cloth_plane|patch_catenary|soft_soft_vf|ellipsoid_ground|cluster_pair|cluster_pair_soft|cluster_joints|rope_ball_tunnel> [frames]\n", argv[0]);
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
	else if (!strcmp(scene, "patch_catenary"))
		scene_patch_catenary(frames);
	else if (!strcmp(scene, "soft_soft_vf"))
		scene_soft_soft_vf(frames);
	else if (!strcmp(scene, "ellipsoid_ground"))
		scene_ellipsoid_ground(frames);
	else if (!strcmp(scene, "cluster_pair"))
		scene_cluster_pair(frames);
	else if (!strcmp(scene, "cluster_pair_soft"))
		scene_cluster_pair_soft(frames);
	else if (!strcmp(scene, "cluster_joints"))
		scene_cluster_joints(frames);
	else if (!strcmp(scene, "rope_ball_tunnel"))
		scene_rope_ball_tunnel(frames);
	else
	{
		fprintf(stderr, "unknown scene %s\n", scene);
		return 1;
	}
	return 0;
}
