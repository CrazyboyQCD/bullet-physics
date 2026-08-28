// GImpact probe: dynamic btGImpactMeshShape scenes printed per-frame for
// numerical comparison against the Rust port (examples/gimpact_probe.rs).
// Usage: gimpact_probe.exe <scene> [frames]
//   scenes: inertia | shell_on_plane | shell_on_bvh | sphere_on_shell

#include <cstdio>
#include <vector>

#include <btBulletDynamicsCommon.h>
#include <btGImpactShape.h>
#include <btGImpactCollisionAlgorithm.h>

static const btScalar DT = btScalar(1) / btScalar(60);
static const int DEFAULT_FRAMES = 300;

static btDiscreteDynamicsWorld* makeWorld()
{
	btDefaultCollisionConfiguration* config = new btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new btCollisionDispatcher(config);
	// The GImpact algorithms must be registered with the dispatcher.
	btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);
	btDbvtBroadphase* broadphase = new btDbvtBroadphase();
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver();
	btDiscreteDynamicsWorld* world = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, config);
	world->setGravity(btVector3(0, -10, 0));
	world->getSolverInfo().m_solverMode &= ~SOLVER_RANDMIZE_ORDER;
	return world;
}

// A closed 2x2x0.4 box shell (top and bottom faces), matching the Rust
// `box_shell_mesh` test helper.
static btGImpactMeshShape* makeShellShape()
{
	const btScalar hx = 1, hy = btScalar(0.2), hz = 1;
	btTriangleMesh* trimesh = new btTriangleMesh();
	btVector3 top[4] = {
		btVector3(-hx, hy, -hz), btVector3(hx, hy, -hz), btVector3(hx, hy, hz), btVector3(-hx, hy, hz)};
	btVector3 bot[4] = {
		btVector3(-hx, -hy, -hz), btVector3(hx, -hy, -hz), btVector3(hx, -hy, hz), btVector3(-hx, -hy, hz)};
	trimesh->addTriangle(top[0], top[1], top[2]);
	trimesh->addTriangle(top[0], top[2], top[3]);
	trimesh->addTriangle(bot[0], bot[2], bot[1]);
	trimesh->addTriangle(bot[0], bot[3], bot[2]);
	btGImpactMeshShape* shape = new btGImpactMeshShape(trimesh);
	shape->updateBound();
	return shape;
}

// Scene: print the vertex point-mass inertia of the shell (Bullet's
// CALC_EXACT_INERTIA) for direct comparison.
static void scene_inertia(int)
{
	btGImpactMeshShape* shape = makeShellShape();
	btVector3 inertia(0, 0, 0);
	shape->calculateLocalInertia(2, inertia);
	printf("inertia %.6f %.6f %.6f\n", inertia[0], inertia[1], inertia[2]);
}

// Scene: the shell drops from 0.5 onto a static plane and settles.
static void scene_shell_on_plane(int frames)
{
	btDiscreteDynamicsWorld* world = makeWorld();
	btStaticPlaneShape* plane = new btStaticPlaneShape(btVector3(0, 1, 0), 0);
	btRigidBody* ground = new btRigidBody(0, nullptr, plane);
	world->addRigidBody(ground);

	btGImpactMeshShape* shape = makeShellShape();
	btVector3 inertia(0, 0, 0);
	shape->calculateLocalInertia(2, inertia);
	btRigidBody* body = new btRigidBody(2, nullptr, shape, inertia);
	body->setWorldTransform(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0.5, 0)));
	world->addRigidBody(body);

	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = body->getWorldTransform().getOrigin();
		const btVector3& v = body->getLinearVelocity();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

// Scene: the shell drops from 0.5 onto a static BVH triangle-mesh ground
// (8x8 unit grid centered at the origin).
static void scene_shell_on_bvh(int frames)
{
	btDiscreteDynamicsWorld* world = makeWorld();
	btTriangleMesh* trimesh = new btTriangleMesh();
	for (int z = 0; z < 8; ++z)
	{
		for (int x = 0; x < 8; ++x)
		{
			btVector3 v00(x, 0, z), v10(x + 1, 0, z), v11(x + 1, 0, z + 1), v01(x, 0, z + 1);
			trimesh->addTriangle(v00, v11, v10);
			trimesh->addTriangle(v00, v01, v11);
		}
	}
	btBvhTriangleMeshShape* groundShape = new btBvhTriangleMeshShape(trimesh, true);
	btRigidBody* ground = new btRigidBody(0, nullptr, groundShape);
	ground->setWorldTransform(btTransform(btQuaternion(0, 0, 0, 1), btVector3(-4, 0, -4)));
	world->addRigidBody(ground);

	btGImpactMeshShape* shape = makeShellShape();
	btVector3 inertia(0, 0, 0);
	shape->calculateLocalInertia(2, inertia);
	btRigidBody* body = new btRigidBody(2, nullptr, shape, inertia);
	body->setWorldTransform(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0.5, 0)));
	world->addRigidBody(body);

	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = body->getWorldTransform().getOrigin();
		const btVector3& v = body->getLinearVelocity();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

// Scene: a sphere drops from 2 onto a static GImpact shell.
static void scene_sphere_on_shell(int frames)
{
	btDiscreteDynamicsWorld* world = makeWorld();
	btGImpactMeshShape* shape = makeShellShape();
	btRigidBody* shell = new btRigidBody(0, nullptr, shape);
	shell->setWorldTransform(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0.2, 0)));
	world->addRigidBody(shell);

	btSphereShape* sphereShape = new btSphereShape(0.5f);
	btVector3 inertia(0, 0, 0);
	sphereShape->calculateLocalInertia(1, inertia);
	btRigidBody* ball = new btRigidBody(1, nullptr, sphereShape, inertia);
	ball->setWorldTransform(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 2, 0)));
	world->addRigidBody(ball);

	for (int f = 0; f < frames; ++f)
	{
		world->stepSimulation(DT, 1, DT);
		const btVector3& p = ball->getWorldTransform().getOrigin();
		const btVector3& v = ball->getLinearVelocity();
		printf("%d %.6f %.6f %.6f %.6f %.6f %.6f\n", f, p[0], p[1], p[2], v[0], v[1], v[2]);
	}
}

int main(int argc, char** argv)
{
	const char* scene = argc > 1 ? argv[1] : "shell_on_plane";
	int frames = argc > 2 ? atoi(argv[2]) : DEFAULT_FRAMES;
	if (!strcmp(scene, "inertia")) scene_inertia(frames);
	else if (!strcmp(scene, "shell_on_plane")) scene_shell_on_plane(frames);
	else if (!strcmp(scene, "shell_on_bvh")) scene_shell_on_bvh(frames);
	else if (!strcmp(scene, "sphere_on_shell")) scene_sphere_on_shell(frames);
	else printf("unknown scene %s\n", scene);
	return 0;
}
