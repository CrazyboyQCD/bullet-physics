// P2-C06 upstream comparison: feed the minimal failing flat-slab cloud
// (greedily minimized from the Rust twin probe in
// crates/bullet-math/src/quickhull.rs) to the real Bullet HullLibrary
// and check its triangle output for closed-manifold structure.
#include <cstdio>
#include <map>
#include <utility>
#include <vector>
#include "LinearMath/btConvexHull.h"

static const float CLOUD[][3] = {
    {0.2691295f, -0.23461741f, 0.0004619904f},
    {-0.43353087f, -0.6117897f, 0.00051185966f},
    {-0.766617f, 0.30281067f, 0.00094336254f},
    {-0.13432908f, -0.37333828f, -0.00067369797f},
    {0.48226058f, 0.5953672f, 0.00027337635f},
    {0.54398525f, -0.80950475f, 0.0006877636f},
    {0.9166075f, -0.9253926f, -0.00035188516f},
    {0.46440423f, -0.72225773f, -0.00043515983f},
    {0.6592368f, 0.621101f, 0.00032918778f},
    {0.016827345f, -0.732311f, -0.0009729715f},
    {-0.918939f, -0.791459f, 0.0009633836f},
    {-0.4002056f, -0.68798554f, 0.00033889426f},
    {-0.91948706f, 0.27823615f, -0.0009934349f},
    {0.83144283f, 0.5248977f, 0.00015817619f},
    {0.45332778f, -0.9703815f, -0.00033576062f},
    {0.2629192f, -0.87479687f, 0.00066716317f},
    {-0.90504503f, 0.30459344f, 0.00026242054f},
    {-0.5788772f, -0.58595085f, 0.00032721402f},
    {0.16627944f, -0.99928725f, -0.00036237075f},
    {0.13506198f, 0.06441009f, 0.0007628943f},
    {0.02131784f, -0.14764863f, -0.00097466534f},
    {-0.028191686f, -0.024073482f, 0.00040069653f},
    {0.124082685f, 0.7241721f, -0.00083422393f},
    {0.3771006f, 0.086423635f, 0.0002477714f},
    {-0.2689572f, 0.8284508f, 0.0005312504f},
    {0.08478296f, -0.08115655f, 0.0006406417f},
    {0.76238596f, 0.51733303f, -0.00042507716f},
    {0.12746024f, -0.42984104f, 0.000110320216f},
    {0.45593333f, -0.52107465f, 0.0009341531f},
    {0.60939837f, 0.8485298f, -0.00076062605f},
    {-0.38970447f, 0.87306774f, -0.00002678925f},
    {-0.1834715f, 0.43748617f, -0.0008340864f},
    {-0.36680025f, -0.4446723f, 0.00053626637f},
    {-0.033923626f, 0.24546015f, 0.00052167254f},
    {0.20582342f, 0.7605152f, 0.00016522325f},
    {-0.9256521f, 0.3907827f, -0.00055722723f},
    {0.7886088f, -0.9735655f, 0.00041946722f},
    {0.31053114f, -0.8435588f, 0.00027884927f},
    {-0.26148474f, 0.5891502f, -0.00051785604f},
};

int main() {
    unsigned int n = sizeof(CLOUD) / sizeof(CLOUD[0]);
    std::vector<btVector3> pts;
    for (unsigned int i = 0; i < n; ++i)
        pts.push_back(btVector3(CLOUD[i][0], CLOUD[i][1], CLOUD[i][2]));

    HullDesc desc(QF_TRIANGLES, n, pts.data());
    HullResult res;
    HullLibrary lib;
    if (lib.CreateConvexHull(desc, res) != QE_OK) {
        printf("upstream: CreateConvexHull FAILED\n");
        return 1;
    }
    printf("upstream: verts=%u faces=%u indices=%u\n", res.mNumOutputVertices,
           res.mNumFaces, res.mNumIndices);

    std::map<std::pair<unsigned int, unsigned int>, unsigned int> out;
    for (unsigned int i = 0; i + 2 < res.m_Indices.size(); i += 3) {
        unsigned int t[3] = {res.m_Indices[i], res.m_Indices[i + 1],
                             res.m_Indices[i + 2]};
        for (int k = 0; k < 3; ++k)
            out[{t[k], t[(k + 1) % 3]}]++;
    }
    bool closed = true;
    for (const auto& kv : out) {
        if (kv.second != 1 ||
            out.count({kv.first.second, kv.first.first}) != 1)
            closed = false;
    }
    printf("upstream: %s\n", closed ? "CLOSED manifold" : "NOT closed");
    lib.ReleaseResult(res);
    return closed ? 0 : 2;
}
