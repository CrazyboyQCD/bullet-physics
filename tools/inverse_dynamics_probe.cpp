// Inverse-dynamics probe: canonical trees run through the real
// BulletInverseDynamics library (refs/bullet3/src, zlib region) for
// bit-level comparison against the Rust twin
// (crates/bullet-inverse-dynamics/examples/id_probe.rs).
//
// Output protocol: one line per (scene, state) per group — "S1" is the
// inverse-dynamics group (generalized forces + world getters with gravity
// folded into the accelerations), "S2" the calculateKinematics group
// (gravity-free accelerations), "M" the joint-space mass matrix
// (num_dofs^2, row-major), "J" the world-frame jacobians (per body:
// dot(J_T)u(3), dot(J_R)u(3), jac_T(3xN), jac_R(3xN), row-major).
// Scenes 0-3 are canonical trees; scenes 4-6 are seeded random trees
// (srand(seed) + the shared draw protocol — keep the Rust twin in
// lockstep).
// Columns after the label for S1/S2:
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
//     inverse_dynamics_probe[ _f64].exe            parity mode (this file's
//                                                  default output)
//     inverse_dynamics_probe[ _f64].exe --bench    perf mode: the chain_16 /
//                                                  chain_64 / chain_256 /
//                                                  random_64 scenes x
//                                                  {kin, id, mass, jac},
//                                                  ns/call medians over
//                                                  interleaved rounds plus
//                                                  per-call allocation counts
//                                                  (docs/perf/
//                                                  inverse-dynamics.md)
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>

#include <LinearMath/btAlignedAllocator.h>

#include "Bullet3Common/b3Logging.h"
#include "BulletInverseDynamics/MultiBodyTree.hpp"
#include "BulletInverseDynamics/IDMath.hpp"

using namespace btInverseDynamics;

static const int kScenes = 4;
static const int kStates = 3;
static const int kMaxBodies = 4;
// tau display floor for the S1/S2 protocol; random trees can carry up to
// 6 bodies x 6 dofs, so their state arrays are sized kMaxStateDofs.
static const int kMaxDofs = 6;
static const int kMaxStateDofs = 36;
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
	const int rc = tree.addBody(body, parent, jt, r, T, axis, mass, com, I, 0, 0x0);
	if (0 != rc) {
		fprintf(stderr, "addBody failed (body %d parent %d rc %d)\n", body, parent, rc);
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

static void print_mass_matrix(int scene, int state, const matxx &mm, int n) {
	printf("M %d %d", scene, state);
	for (int r = 0; r < n; r++)
		for (int c = 0; c < n; c++)
			printf(" %.17g", mm(r, c));
	printf("\n");
}

static void print_jacobians(int scene, int state, MultiBodyTree &tree, int num_bodies, int n) {
	printf("J %d %d", scene, state);
	for (int b = 0; b < num_bodies; b++) {
		vec3 v;
		tree.getBodyDotJacobianTransU(b, &v);
		print_vec(v);
		tree.getBodyDotJacobianRotU(b, &v);
		print_vec(v);
		mat3x jac(3, n);
		jac.setZero();
		tree.getBodyJacobianTrans(b, &jac);
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < n; c++)
				printf(" %.17g", jac(r, c));
		tree.getBodyJacobianRot(b, &jac);
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < n; c++)
				printf(" %.17g", jac(r, c));
	}
	printf("\n");
}

static void print_tree_state(MultiBodyTree &tree, int scene, int state, const char *group,
							 const vecx &tau, int num_bodies) {
	printf("%s %d %d", group, scene, state);
	const int tau_slots = kMaxDofs > tau.size() ? kMaxDofs : tau.size();
	for (int i = 0; i < tau_slots; i++) {
		idScalar v = i < tau.size() ? tau(i) : 0.0;
		printf(" %.17g", v);
	}
	const int body_slots = kMaxBodies > num_bodies ? kMaxBodies : num_bodies;
	for (int b = 0; b < body_slots; b++) {
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

template <int N>
static void run_scene(int scene, MultiBodyTree &tree, int num_bodies, int num_dofs,
					  const idScalar qs[kStates][N], const idScalar us[kStates][N],
					  const idScalar dot_us[kStates][N]) {
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
		// Position+velocity kinematics are current from calculateKinematics;
		// the mass-matrix relative refresh is idempotent at the same q.
		matxx mm(num_dofs, num_dofs);
		mm.setZero();
		if (0 != tree.calculateMassMatrix(q, &mm)) {
			fprintf(stderr, "calculateMassMatrix failed (scene %d state %d)\n", scene, s);
			exit(3);
		}
		print_mass_matrix(scene, s, mm, num_dofs);
		if (0 != tree.calculateJacobians(q, u)) {
			fprintf(stderr, "calculateJacobians failed (scene %d state %d)\n", scene, s);
			exit(3);
		}
		print_jacobians(scene, s, tree, num_bodies, num_dofs);
	}
}

// Seeded random-tree generator. Draw protocol shared 1:1 with the Rust twin
// (CrtRand replicates the MSVC rand() sequence): every value is
// lo + (rand() % 2001 - 1000) / 1000.0 * width, consumed in a fixed order.
// Extras/IDRandomUtil is in the zlib-excluded zone — this generator is
// written natively for the probe.
static double draw(double lo, double width) {
	return lo + (rand() % 2001) / 2000.0 * width;
}

// One drawn body: the shared per-body draw sequence of the random-tree
// builder and the bench chains (identical order on both sides; the parent
// and joint type are the caller's, so the chains skip their draws).
static void add_drawn_body(MultiBodyTree &tree, int body, int parent, JointType jt) {
	const double rx = draw(-0.4, 0.8), ry = draw(-0.4, 0.8), rz = draw(-0.4, 0.8);
	const vec3 r = v3(rx, ry, rz);
	const double a0 = draw(-0.4, 0.8), a1 = draw(-0.4, 0.8), a2 = draw(-0.4, 0.8);
	const mat33 t_ref = transformZ(a2) * transformY(a1) * transformX(a0);
	const double ax = draw(0.05, 0.95), ay = draw(0.05, 0.95), az = draw(0.05, 0.95);
	const vec3 axis = v3(ax, ay, az);
	const double mass = draw(0.1, 1.9);
	const double cx = draw(-0.5, 1.0), cy = draw(-0.5, 1.0), cz = draw(-0.5, 1.0);
	const vec3 com = v3(cx, cy, cz);
	double m[3] = {draw(0.05, 0.95), draw(0.05, 0.95), draw(0.05, 0.95)};
	// Enforce the inertia triangle inequality on the largest entry
	// (lowest-index tie-break), so the addBody gate accepts every tree.
	double sum = m[0] + m[1] + m[2];
	int mx = 0;
	if (m[1] > m[mx])
		mx = 1;
	if (m[2] > m[mx])
		mx = 2;
	if (m[mx] > sum - m[mx])
		m[mx] = (sum - m[mx]) * 0.9;
	const mat33 inertia = diagonal(m[0], m[1], m[2]);
	add(tree, body, parent, jt, r, t_ref, axis, mass, com, inertia);
}

static void build_random_tree_n(MultiBodyTree &tree, int nb) {
	for (int i = 0; i < nb; i++) {
		const int parent = (i == 0) ? -1 : rand() % i;
		JointType jt;
		switch (rand() % 5) {
			case 0:
				jt = FIXED;
				break;
			case 1:
				jt = REVOLUTE;
				break;
			case 2:
				jt = PRISMATIC;
				break;
			case 3:
				jt = FLOATING;
				break;
			default:
				jt = SPHERICAL;
				break;
		}
		add_drawn_body(tree, i, parent, jt);
	}
}

// Fixed-root serial chain: body 0 FIXED, bodies 1..nb-1 drawn revolute links.
static void build_chain(MultiBodyTree &tree, int nb) {
	add(tree, 0, -1, FIXED, v3(0.0, 0.0, 0.0), identity(), v3(0.0, 0.0, 0.0), 0.0,
		v3(0.0, 0.0, 0.0), diagonal(1.0, 1.0, 1.0));
	for (int i = 1; i < nb; i++)
		add_drawn_body(tree, i, i - 1, REVOLUTE);
}

static int build_random_tree(MultiBodyTree &tree, int *num_bodies_out) {
	const int nb = 2 + rand() % 5;
	build_random_tree_n(tree, nb);
	*num_bodies_out = nb;
	return nb;
}

// --- allocation accounting (--bench run mode) -------------------------------
// Same two hook points as tools/bullet_perf.cpp: the global operator
// new/delete family below plus btAlignedAllocSetCustomAligned (installed only
// by the bench mode) for Bullet's internal btAlignedObjectArray storage. The
// bench mode reports monotonic allocation counts and requested bytes per
// calculate call; live bytes after finalize are the tree footprint (the Rust
// twin reports the same unit — requested bytes).
namespace {
std::atomic<bool> g_track{false};
std::atomic<long long> g_live{0};
std::atomic<long long> g_allocs{0};
std::atomic<long long> g_bytes{0};

void mem_note_alloc(std::size_t n) {
	const long long live =
		g_live.fetch_add(long long(n), std::memory_order_relaxed) + long long(n);
	(void)live;
	g_allocs.fetch_add(1, std::memory_order_relaxed);
	g_bytes.fetch_add(long long(n), std::memory_order_relaxed);
}

void mem_note_free(std::size_t n) {
	// Clamp: chunks allocated before tracking was switched on (the startup
	// window) free late and would otherwise drag the total below zero.
	const long long old = g_live.fetch_sub(long long(n), std::memory_order_relaxed);
	if (old < long long(n))
		g_live.fetch_add(long long(n), std::memory_order_relaxed);
}

void* bt_alloc_custom(std::size_t size, int alignment) {
	const std::size_t align = alignment < 16 ? 16 : std::size_t(alignment);
	unsigned char* raw = static_cast<unsigned char*>(
		_aligned_malloc(size + 2 * sizeof(std::size_t), align));
	if (!raw) return nullptr;
	std::size_t* hdr = reinterpret_cast<std::size_t*>(raw);
	hdr[0] = size;
	hdr[1] = align;
	if (g_track.load(std::memory_order_relaxed)) mem_note_alloc(size);
	return raw + 2 * sizeof(std::size_t);
}

void bt_free_custom(void* ptr) {
	if (!ptr) return;
	unsigned char* raw =
		static_cast<unsigned char*>(ptr) - 2 * sizeof(std::size_t);
	if (g_track.load(std::memory_order_relaxed)) {
		const std::size_t* hdr = reinterpret_cast<const std::size_t*>(raw);
		mem_note_free(hdr[0]);
	}
	_aligned_free(raw);
}
}  // namespace

void* operator new(std::size_t n) {
	void* p = std::malloc(n ? n : 1);
	if (!p) throw std::bad_alloc{};
	if (g_track.load(std::memory_order_relaxed)) mem_note_alloc(n);
	return p;
}

void* operator new[](std::size_t n) {
	void* p = std::malloc(n ? n : 1);
	if (!p) throw std::bad_alloc{};
	if (g_track.load(std::memory_order_relaxed)) mem_note_alloc(n);
	return p;
}

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
	void* p = std::malloc(n ? n : 1);
	if (p && g_track.load(std::memory_order_relaxed)) mem_note_alloc(n);
	return p;
}

void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
	void* p = std::malloc(n ? n : 1);
	if (p && g_track.load(std::memory_order_relaxed)) mem_note_alloc(n);
	return p;
}

void operator delete(void* p) noexcept { std::free(p); }

void operator delete[](void* p) noexcept { std::free(p); }

void operator delete(void* p, std::size_t n) noexcept {
	if (p && g_track.load(std::memory_order_relaxed)) mem_note_free(n);
	std::free(p);
}

void operator delete[](void* p, std::size_t n) noexcept {
	if (p && g_track.load(std::memory_order_relaxed)) mem_note_free(n);
	std::free(p);
}

void* operator new(std::size_t n, std::align_val_t al) {
	void* p = _aligned_malloc(n ? n : 1, std::size_t(al));
	if (!p) throw std::bad_alloc{};
	if (g_track.load(std::memory_order_relaxed)) mem_note_alloc(n);
	return p;
}

void* operator new[](std::size_t n, std::align_val_t al) {
	void* p = _aligned_malloc(n ? n : 1, std::size_t(al));
	if (!p) throw std::bad_alloc{};
	if (g_track.load(std::memory_order_relaxed)) mem_note_alloc(n);
	return p;
}

void operator delete(void* p, std::align_val_t) noexcept { _aligned_free(p); }

void operator delete[](void* p, std::align_val_t) noexcept { _aligned_free(p); }

void operator delete(void* p, std::size_t n, std::align_val_t) noexcept {
	if (p && g_track.load(std::memory_order_relaxed)) mem_note_free(n);
	_aligned_free(p);
}

void operator delete[](void* p, std::size_t n, std::align_val_t) noexcept {
	if (p && g_track.load(std::memory_order_relaxed)) mem_note_free(n);
	_aligned_free(p);
}

void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
	_aligned_free(p);
}

void operator delete[](void* p, std::align_val_t,
					   const std::nothrow_t&) noexcept {
	_aligned_free(p);
}

// --- bench mode --------------------------------------------------------------
// Scenes x algorithms of docs/perf/inverse-dynamics.md. Protocol per scene:
// one `bench_mem` line (finalized-tree live bytes under the allocation hook),
// then per algorithm one timing line (median/min ns per call over 5
// interleaved rounds of auto-calibrated call counts) and one allocation line
// (monotonic count/bytes delta of a single tracked call). The Rust twin
// (examples/id_probe.rs --bench) prints the same lines.

namespace {
struct BenchScene {
	const char* name;
	int seed;
	int bodies;
	bool random;
};
const BenchScene kBenchScenes[] = {
	{"chain_16", 2101, 16, false}, {"chain_64", 2102, 64, false},
	{"chain_256", 2103, 256, false}, {"random_64", 2104, 64, true},
};

enum Algo { A_KIN = 0, A_ID = 1, A_MASS = 2, A_JAC = 3, kAlgos = 4 };
const char* kAlgoNames[kAlgos] = {"kin", "id", "mass", "jac"};

bool call_algo(Algo a, MultiBodyTree &tree, const vecx &q, const vecx &u,
			   const vecx &du, vecx &tau, matxx &mm) {
	switch (a) {
		case A_KIN:
			return 0 == tree.calculateKinematics(q, u, du);
		case A_ID:
			return 0 == tree.calculateInverseDynamics(q, u, du, &tau);
		case A_MASS:
			return 0 == tree.calculateMassMatrix(q, &mm);
		case A_JAC:
			return 0 == tree.calculateJacobians(q, u);
	}
	return false;
}

void bench_scene(const BenchScene &s) {
	const long long live0 = g_live.load(std::memory_order_relaxed);
	srand(s.seed);
	MultiBodyTree tree;
	if (s.random)
		build_random_tree_n(tree, s.bodies);
	else
		build_chain(tree, s.bodies);
	if (0 != tree.finalize()) {
		fprintf(stderr, "finalize failed (%s)\n", s.name);
		exit(2);
	}
	tree.setGravityInWorldFrame(v3(0.0, 0.0, -9.8));
	const long long tree_live =
		g_live.load(std::memory_order_relaxed) - live0;
	const int nd = tree.numDoFs();

	// One drawn state, the same draw(-1.5, 3.0) stream as the parity scenes.
	vecx q(nd), u(nd), du(nd), tau(nd);
	for (int d = 0; d < nd; d++) {
		q(d) = draw(-1.5, 3.0);
		u(d) = draw(-1.5, 3.0);
		du(d) = draw(-1.5, 3.0);
	}
	matxx mm(nd, nd);
	mm.setZero();

	printf("bench_mem %s tree_live_bytes %lld\n", s.name, tree_live);

	const int kRounds = 5;
	double ns[kAlgos][kRounds];
	long long calls[kAlgos];
	for (int a = 0; a < kAlgos; a++) {
		const Algo algo = Algo(a);
		for (int w = 0; w < 5; w++)
			if (!call_algo(algo, tree, q, u, du, tau, mm)) {
				fprintf(stderr, "%s warmup failed (%s)\n", kAlgoNames[a], s.name);
				exit(3);
			}
		// Auto-calibrated round size: ~50 ms of calls, clamped.
		double one = 1e300;
		for (int k = 0; k < 3; k++) {
			const auto t0 = std::chrono::steady_clock::now();
			if (!call_algo(algo, tree, q, u, du, tau, mm)) {
				fprintf(stderr, "%s calibration failed (%s)\n", kAlgoNames[a],
						s.name);
				exit(3);
			}
			const auto t1 = std::chrono::steady_clock::now();
			one = std::min(
				one, std::chrono::duration<double, std::nano>(t1 - t0).count());
		}
		long long c = long long(50.0e6 / std::max(one, 1.0));
		calls[a] = std::max(1LL, std::min(200000LL, c));
	}
	// Interleaved rounds: the round loop is outer, the algorithm loop inner.
	for (int r = 0; r < kRounds; r++) {
		for (int a = 0; a < kAlgos; a++) {
			const Algo algo = Algo(a);
			const auto t0 = std::chrono::steady_clock::now();
			for (long long i = 0; i < calls[a]; i++)
				if (!call_algo(algo, tree, q, u, du, tau, mm)) {
					fprintf(stderr, "%s timed call failed (%s)\n", kAlgoNames[a],
							s.name);
					exit(3);
				}
			const auto t1 = std::chrono::steady_clock::now();
			ns[a][r] = std::chrono::duration<double, std::nano>(t1 - t0).count() /
					   double(calls[a]);
		}
	}
	for (int a = 0; a < kAlgos; a++) {
		double sorted[kRounds];
		for (int r = 0; r < kRounds; r++) sorted[r] = ns[a][r];
		std::sort(sorted, sorted + kRounds);
		double total = 0.0;
		for (int r = 0; r < kRounds; r++) total += ns[a][r];
		printf("bench %s %s median_ns %.1f min_ns %.1f mean_ns %.1f rounds %d calls_per_round %lld\n",
			   s.name, kAlgoNames[a], sorted[kRounds / 2], sorted[0],
			   total / double(kRounds), kRounds, calls[a]);
	}
	// Per-call allocation counts of one tracked call (monotonic counters).
	for (int a = 0; a < kAlgos; a++) {
		const long long c0 = g_allocs.load(std::memory_order_relaxed);
		const long long b0 = g_bytes.load(std::memory_order_relaxed);
		if (!call_algo(Algo(a), tree, q, u, du, tau, mm)) {
			fprintf(stderr, "%s tracked call failed (%s)\n", kAlgoNames[a], s.name);
			exit(3);
		}
		printf("bench_alloc %s %s allocs %lld bytes %lld\n", s.name, kAlgoNames[a],
			   g_allocs.load(std::memory_order_relaxed) - c0,
			   g_bytes.load(std::memory_order_relaxed) - b0);
	}
	fflush(stdout);
}
}  // namespace

static int run_bench() {
	// Route Bullet's internal aligned allocations through the accounting hook.
	btAlignedAllocSetCustomAligned(&bt_alloc_custom, &bt_free_custom);
	g_track.store(true, std::memory_order_relaxed);
	for (const BenchScene &s : kBenchScenes) bench_scene(s);
	fprintf(stderr, "bench done\n");
	return 0;
}

int main(int argc, char **argv) {
	// The port logs nothing; silence the upstream axis-normalization warning
	// so it cannot pollute the compare protocol.
	b3SetCustomWarningMessageFunc([](const char *) {});
	if (argc > 1 && 0 == strcmp(argv[1], "--bench"))
		return run_bench();
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
	// scenes 4-6: seeded random trees, one per seed. The draw order and the
	// lo/width pairs are part of the protocol — keep them in lockstep with
	// the Rust twin.
	for (int t = 0; t < 3; t++) {
		srand(t + 1);
		MultiBodyTree tree;
		int nb = 0;
		build_random_tree(tree, &nb);
		if (0 != tree.finalize()) {
			fprintf(stderr, "finalize failed (random %d)\n", t);
			return 2;
		}
		tree.setGravityInWorldFrame(v3(0.0, 0.0, -9.8));
		const int nd = tree.numDoFs();
		idScalar qs[kStates][kMaxStateDofs] = {};
		idScalar us[kStates][kMaxStateDofs] = {};
		idScalar du[kStates][kMaxStateDofs] = {};
		for (int s = 0; s < kStates; s++)
			for (int d = 0; d < nd; d++) {
				qs[s][d] = draw(-1.5, 3.0);
				us[s][d] = draw(-1.5, 3.0);
				du[s][d] = draw(-1.5, 3.0);
			}
		run_scene(4 + t, tree, nb, nd, qs, us, du);
		fflush(stdout);
		fprintf(stderr, "random scene %d done\n", 4 + t);
	}
	fprintf(stderr, "main done\n");
	return 0;
}
