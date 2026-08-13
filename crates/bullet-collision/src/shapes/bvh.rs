//! Bounding-volume hierarchy over mesh triangles.
//!
//! A minimal recursive AABB tree: internal nodes store a child pair, leaves
//! store a contiguous range into a global permutation of the triangle
//! indices. Builds in `O(n log n)` over the longest-axis median split;
//! queries are `O(log n + k)` for the candidates (plus the exact triangle
//! tests by the caller).

use bullet_math::{Aabb, Real, Vector3};

/// Leaves hold at most this many triangles.
const LEAF_MAX_TRIANGLES: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Bvh {
    nodes: Vec<BvhNode>,
    /// Permutation of the triangle indices; leaves reference slices of it.
    tri_order: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
struct BvhNode {
    aabb: Aabb,
    kind: BvhNodeKind,
}

#[derive(Debug, Clone, PartialEq)]
enum BvhNodeKind {
    Internal { left: u32, right: u32 },
    Leaf { start: u32, count: u32 },
}

impl Bvh {
    /// Build a BVH over the triangles of the soup (`vertices`/`indices`).
    pub(crate) fn build(vertices: &[Vector3], indices: &[[u32; 3]]) -> Self {
        let mut tri_order: Vec<u32> = (0..indices.len() as u32).collect();
        let count = tri_order.len();
        let mut nodes = Vec::new();
        build_node(&mut nodes, &mut tri_order, 0, count, vertices, indices);
        Self { nodes, tri_order }
    }

    /// Append all triangle indices whose AABB overlaps `aabb` to `out`.
    pub(crate) fn triangles_in_aabb(&self, aabb: &Aabb, out: &mut Vec<u32>) {
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            let n = &self.nodes[node];
            if !n.aabb.overlaps(aabb) {
                continue;
            }
            match n.kind {
                BvhNodeKind::Internal { left, right } => {
                    stack.push(left as usize);
                    stack.push(right as usize);
                }
                BvhNodeKind::Leaf { start, count } => {
                    for k in start as usize..(start + count) as usize {
                        out.push(self.tri_order[k]);
                    }
                }
            }
        }
    }

    /// First triangle hit by the ray, in `[0, 1]` fraction, plus the local
    /// normal. Both triangle faces are tested (Bullet's default ray test does
    /// not cull back faces); the normal is flipped to face the ray origin.
    pub(crate) fn raycast(
        &self,
        vertices: &[Vector3],
        indices: &[[u32; 3]],
        from: Vector3,
        to: Vector3,
    ) -> Option<(Real, Vector3)> {
        let dir = to - from;
        let mut best: Option<(Real, Vector3)> = None;
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            let n = &self.nodes[node];
            let t_entry = ray_aabb_entry(&n.aabb, from, dir);
            if best.is_some_and(|(bt, _)| t_entry > bt) {
                continue; // Node entirely behind the best hit so far.
            }
            match n.kind {
                BvhNodeKind::Internal { left, right } => {
                    stack.push(left as usize);
                    stack.push(right as usize);
                }
                BvhNodeKind::Leaf { start, count } => {
                    for k in start as usize..(start + count) as usize {
                        let tri = indices[self.tri_order[k] as usize];
                        if let Some((t, normal)) = ray_triangle(
                            vertices[tri[0] as usize],
                            vertices[tri[1] as usize],
                            vertices[tri[2] as usize],
                            from,
                            dir,
                        ) && best.is_none_or(|(bt, _)| t < bt)
                        {
                            best = Some((t, normal));
                        }
                    }
                }
            }
        }
        best
    }
}

fn build_node(
    nodes: &mut Vec<BvhNode>,
    tri_order: &mut [u32],
    start: usize,
    count: usize,
    vertices: &[Vector3],
    indices: &[[u32; 3]],
) -> u32 {
    let aabb = slice_aabb(tri_order, start, count, vertices, indices);
    let index = nodes.len() as u32;
    nodes.push(BvhNode {
        aabb,
        kind: BvhNodeKind::Leaf {
            start: start as u32,
            count: count as u32,
        },
    });
    if count <= LEAF_MAX_TRIANGLES {
        return index;
    }
    // Split on the longest axis at the median of the triangle centers.
    let (axis, _) = longest_axis(&aabb);
    let slice = &mut tri_order[start..start + count];
    slice.sort_by(|&a, &b| {
        triangle_center(vertices, indices, a)[axis]
            .total_cmp(&triangle_center(vertices, indices, b)[axis])
    });
    let mid = count / 2;
    let left = build_node(nodes, tri_order, start, mid, vertices, indices);
    let right = build_node(nodes, tri_order, start + mid, count - mid, vertices, indices);
    nodes[index as usize].kind = BvhNodeKind::Internal { left, right };
    index
}

fn slice_aabb(
    tri_order: &[u32],
    start: usize,
    count: usize,
    vertices: &[Vector3],
    indices: &[[u32; 3]],
) -> Aabb {
    let mut aabb: Option<Aabb> = None;
    for &t in &tri_order[start..start + count] {
        for v in [0, 1, 2] {
            let p = vertices[indices[t as usize][v] as usize];
            aabb = Some(match aabb {
                None => Aabb::new(p, p),
                Some(a) => a.merged(&Aabb::new(p, p)),
            });
        }
    }
    aabb.unwrap_or_else(|| Aabb::new(Vector3::ZERO, Vector3::ZERO))
}

fn triangle_center(vertices: &[Vector3], indices: &[[u32; 3]], t: u32) -> Vector3 {
    let tri = indices[t as usize];
    (vertices[tri[0] as usize] + vertices[tri[1] as usize] + vertices[tri[2] as usize]) * (1.0 / 3.0)
}

fn longest_axis(aabb: &Aabb) -> (usize, Real) {
    let e = aabb.extents();
    let mut axis = 0;
    let mut max = e.x;
    if e.y > max {
        axis = 1;
        max = e.y;
    }
    if e.z > max {
        axis = 2;
    }
    (axis, max)
}

/// Entry fraction of the ray against an AABB (`0` when the origin is inside),
/// or `+inf` when the ray misses.
fn ray_aabb_entry(aabb: &Aabb, from: Vector3, dir: Vector3) -> Real {
    let mut tmin = Real::NEG_INFINITY;
    let mut tmax = Real::INFINITY;
    for axis in 0..3 {
        let o = from[axis];
        let d = dir[axis];
        let lo = aabb.min[axis];
        let hi = aabb.max[axis];
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return Real::INFINITY;
            }
        } else {
            let mut t1 = (lo - o) / d;
            let mut t2 = (hi - o) / d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmin > tmax {
                return Real::INFINITY;
            }
        }
    }
    tmin.max(0.0)
}

/// Möller-Trumbore ray-triangle intersection in local space. `dir` is the
/// unnormalized ray direction; the returned normal faces the ray origin.
fn ray_triangle(
    a: Vector3,
    b: Vector3,
    c: Vector3,
    from: Vector3,
    dir: Vector3,
) -> Option<(Real, Vector3)> {
    let e1 = b - a;
    let e2 = c - a;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None; // Ray parallel to the triangle.
    }
    let inv_det = 1.0 / det;
    let t_vec = from - a;
    let u = t_vec.dot(p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = t_vec.cross(e1);
    let v = dir.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv_det;
    if t < 0.0 || t > 1.0 {
        return None;
    }
    let n = e1.cross(e2).normalize();
    // Face the ray origin: flip when hitting the back side.
    let n = if n.dot(dir) > 0.0 { -n } else { n };
    Some((t, n))
}
