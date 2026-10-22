// Inverse-dynamics probe: canonical trees run through the real
// BulletInverseDynamics library (refs/bullet3/src, zlib region) for
// bit-level comparison against the Rust twin
// (crates/bullet-inverse-dynamics/examples/id_probe.rs).
//
// Output protocol: one line per (scene, state) per group — "S1" is the
// inverse-dynamics group (generalized forces + world getters with gravity
// folded into the accelerations), "S2" the calculateKinematics group
// (gravity-free accelerations). Columns after the label:
//   scene state | tau[6] (padded) | 4 bodies x 30 slots:
//   origin(3) com(3) transform(9) ang_vel(3) lin_vel(3) lin_vel_com(3)
//   ang_acc(3) lin_acc(3)
// printed at %.17g so a plain byte compare (cmp) is the strictest gate;
// tools/mbdiff.py works on the same lines.
//
// Build with tools/build_inverse_dynamics_probe.py (IEEE-precise BID in
// target/bullet3-build-id[-double] — the /fp:fast default of bullet3's
// CMakeLists swaps in approximate sin/cos that diverge from ucrt by 1 ulp);
// usage:
//     inverse_dynamics_probe[ _f64].exe
#include <cstdio>

#include "BulletInverseDynamics/MultiBodyTree.hpp"
#include "BulletInverseDynamics/IDMath.hpp"

using namespace btInverseDynamics;

static const int kScenes = 4;
static const int kStates = 3;
static const int kMaxBodies = 4;
static const int kMaxDofs = 6;
static const int kBodySlots = 30;

// The vec3 wrapper exposes no three-scalar constructor.
static vec3 v3(idScalar x, idScalar y, idScalar z) {
	vec3 v;
	v(0) = x;
	v(1) = y;
	v(2) = z;
	return v;
}

static mat33 identity() {
	mat33 m;
	m.setIdentity();
	return m;
}

static mat33 diagonal(idScalar xx, idScalar yy, idScalar zz) {
	mat33 m;
	m.setZero();
	m(0, 0) = xx;
	m(1, 1) = yy;
	m(2, 2) = zz;
	return m;
}

static void add(MultiBodyTree &tree, int body, int parent, JointType jt, const vec3 &r,
				const mat33 &T, const vec3 &axis, idScalar mass, const vec3 &com,
				const mat33 &I) {
	if (0 != tree.addBody(body, parent, jt, r, T, axis, mass, com, I, 0, 0x0)) {
		fprintf(stderr, "addBody failed (scene building)\n");
		exit(2);
	}
}

// Scene builders. Body orders and parameters are transcribed 1:1 in the Rust
// twin — keep the two files in lockstep.
static int build_arm2r(MultiBodyTree &tree) {
	add(tree, 0, -1, FIXED, v3(0.0, 0.0, 0.0), identity(), v3(0.0, 0.0, 0.0), 0.0,
		v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
	add(tree, 1, 0, REVOLUTE, v3(1.0, 0.0, 0.0), identity(), v3(0.0, 0.0, 1.0), 1.0,
		v3(0.5, 0.0, 0.0), diagonal(0.2, 0.15, 0.1));
	add(tree, 2, 1, REVOLUTE, v3(1.0, 0.0, 0.0), identity(), v3(0.0, 0.0, 1.0), 0.8,
		v3(0.4, 0.0, 0.0), diagonal(0.1, 0.08, 0.06));
	return 3;
}

static int build_mixed1(MultiBodyTree &tree) {
	add(tree, 0, -1, FIXED, v3(0.0, 0.0, 0.0), identity(), v3(0.0, 0.0, 0.0), 0.0,
		v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
	add(tree, 1, 0, REVOLUTE, v3(0.0, 0.0, 0.0), identity(), v3(0.0, 1.0, 0.0), 1.2,
		v3(0.0, 0.3, 0.0), diagonal(0.12, 0.1, 0.09));
	add(tree, 2, 1, PRISMATIC, v3(0.0, 0.0, 0.3),
		transformY(0.5) * transformX(-0.25), v3(1.0, 0.0, 0.0), 0.6,
		v3(0.2, 0.0, 0.0), diagonal(0.05, 0.05, 0.04));
	add(tree, 3, 2, FIXED, v3(0.5, 0.0, 0.0), identity(), v3(0.0, 0.0, 0.0), 0.5,
		v3(0.0, 0.0, 0.1), diagonal(0.02, 0.02, 0.02));
	return 4;
}

static int build_floating_free(MultiBodyTree &tree) {
	add(tree, 0, -1, FLOATING, v3(0.0, 0.0, 0.0), identity(), v3(0.0, 0.0, 0.0), 2.0,
		v3(0.1, 0.2, 0.3), diagonal(0.3, 0.25, 0.2));
	return 1;
}

static int build_spherical_arm(MultiBodyTree &tree) {
	add(tree, 0, -1, FIXED, v3(0.0, 0.0, 0.0), identity(), v3(0.0, 0.0, 0.0), 0.0,
		v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
	add(tree, 1, 0, SPHERICAL, v3(0.0, 0.0, 1.0), identity(), v3(0.0, 0.0, 0.0), 1.0,
		v3(0.0, 0.0, -0.5), diagonal(0.1, 0.1, 0.1));
	add(tree, 2, 1, FIXED, v3(0.0, 0.0, -0.6),
		transformZ(0.75), v3(0.0, 0.0, 0.0), 0.3,
		v3(0.05, 0.0, 0.0), diagonal(0.01, 0.01, 0.01));
	return 3;
}

static void print_vec(const vec3 &v) {
	printf(" %.17g %.17g %.17g", v(0), v(1), v(2));
}

static void print_tree_state(MultiBodyTree &tree, int scene, int state, const char *group,
							 const vecx &tau, int num_bodies) {
	printf("%s %d %d", group, scene, state);
	for (int i = 0; i < kMaxDofs; i++) {
		idScalar v = i < tau.size() ? tau(i) : 0.0;
		printf(" %.17g", v);
	}
	for (int b = 0; b < kMaxBodies; b++) {
		vec3 v;
		mat33 m;
		if (b < num_bodies) {
			tree.getBodyOrigin(b, &v);
			print_vec(v);
			tree.getBodyCoM(b, &v);
			print_vec(v);
			tree.getBodyTransform(b, &m);
			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					printf(" %.17g", m(r, c));
			tree.getBodyAngularVelocity(b, &v);
			print_vec(v);
			tree.getBodyLinearVelocity(b, &v);
			print_vec(v);
			tree.getBodyLinearVelocityCoM(b, &v);
			print_vec(v);
			tree.getBodyAngularAcceleration(b, &v);
			print_vec(v);
			tree.getBodyLinearAcceleration(b, &v);
			print_vec(v);
		} else {
			for (int k = 0; k < kBodySlots; k++)
				printf(" %.17g", 0.0);
		}
	}
	printf("\n");
}

static void run_scene(int scene, MultiBodyTree &tree, int num_bodies, int num_dofs,
					  const idScalar qs[kStates][kMaxDofs], const idScalar us[kStates][kMaxDofs],
					  const idScalar dot_us[kStates][kMaxDofs]) {
	vecx q(num_dofs), u(num_dofs), dot_u(num_dofs), tau(num_dofs);
	for (int s = 0; s < kStates; s++) {
		for (int i = 0; i < num_dofs; i++) {
			q(i) = qs[s][i];
			u(i) = us[s][i];
			dot_u(i) = dot_us[s][i];
		}
		if (0 != tree.calculateInverseDynamics(q, u, dot_u, &tau)) {
			fprintf(stderr, "calculateInverseDynamics failed (scene %d state %d)\n", scene, s);
			exit(3);
		}
		print_tree_state(tree, scene, s, "S1", tau, num_bodies);
		if (0 != tree.calculateKinematics(q, u, dot_u)) {
			fprintf(stderr, "calculateKinematics failed (scene %d state %d)\n", scene, s);
			exit(3);
		}
		print_tree_state(tree, scene, s, "S2", tau, num_bodies);
	}
}

int main() {
	// scene 0: arm2r — planar two-revolute arm, default gravity.
	{
		MultiBodyTree tree;
		int nb = build_arm2r(tree);
		if (0 != tree.finalize()) {
			fprintf(stderr, "finalize failed (arm2r)\n");
			return 2;
		}
		tree.setGravityInWorldFrame(v3(0.0, 0.0, -9.8));
		const idScalar qs[kStates][kMaxDofs] = {{0.0}, {0.7, -0.4}, {-1.1, 0.9}};
		const idScalar us[kStates][kMaxDofs] = {{0.0}, {1.5, -0.75}, {0.3, 2.0}};
		const idScalar du[kStates][kMaxDofs] = {{0.0}, {0.5, 0.25}, {-1.0, 0.0}};
		run_scene(0, tree, nb, tree.numDoFs(), qs, us, du);
	}
	// scene 1: mixed1 — revolute + prismatic + fixed chain, non-identity
	// reference transforms, default gravity.
	{
		MultiBodyTree tree;
		int nb = build_mixed1(tree);
		if (0 != tree.finalize()) {
			fprintf(stderr, "finalize failed (mixed1)\n");
			return 2;
		}
		tree.setGravityInWorldFrame(v3(0.0, 0.0, -9.8));
		const idScalar qs[kStates][kMaxDofs] = {{0.0, 0.0}, {0.3, 0.6}, {-0.5, 1.2}};
		const idScalar us[kStates][kMaxDofs] = {{0.0, 0.0}, {0.8, -0.4}, {1.0, 0.0}};
		const idScalar du[kStates][kMaxDofs] = {{0.0, 0.0}, {0.2, 1.0}, {-0.5, -0.5}};
		run_scene(1, tree, nb, tree.numDoFs(), qs, us, du);
	}
	// scene 2: floating_free — single floating base, tilted gravity.
	{
		MultiBodyTree tree;
		int nb = build_floating_free(tree);
		if (0 != tree.finalize()) {
			fprintf(stderr, "finalize failed (floating_free)\n");
			return 2;
		}
		tree.setGravityInWorldFrame(v3(0.5, -9.81, 0.2));
		const idScalar qs[kStates][kMaxDofs] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
												{0.4, -0.2, 0.9, 1.0, 2.0, 3.0},
												{-0.7, 1.3, -2.1, -0.5, 0.25, 1.75}};
		const idScalar us[kStates][kMaxDofs] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
												{0.6, 0.1, -0.3, 0.4, -0.8, 0.2},
												{-1.0, 0.5, 0.9, 0.0, 1.5, -0.6}};
		const idScalar du[kStates][kMaxDofs] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
												{0.1, 0.2, 0.3, -0.4, 0.5, -0.6},
												{0.0, -1.0, 0.25, 1.0, 0.0, 0.75}};
		run_scene(2, tree, nb, tree.numDoFs(), qs, us, du);
	}
	// scene 3: spherical_arm — spherical joints with a fixed child, default
	// gravity.
	{
		MultiBodyTree tree;
		int nb = build_spherical_arm(tree);
		if (0 != tree.finalize()) {
			fprintf(stderr, "finalize failed (spherical_arm)\n");
			return 2;
		}
		tree.setGravityInWorldFrame(v3(0.0, 0.0, -9.8));
		const idScalar qs[kStates][kMaxDofs] = {{0.0, 0.0, 0.0},
												{0.2, 0.5, -0.4},
												{-0.9, 0.15, 1.1}};
		const idScalar us[kStates][kMaxDofs] = {{0.0, 0.0, 0.0},
												{0.3, -0.6, 0.9},
												{0.0, 1.2, -0.4}};
		const idScalar du[kStates][kMaxDofs] = {{0.0, 0.0, 0.0},
												{-0.2, 0.4, 0.1},
												{0.8, 0.0, -0.3}};
		run_scene(3, tree, nb, tree.numDoFs(), qs, us, du);
	}
	return 0;
}
