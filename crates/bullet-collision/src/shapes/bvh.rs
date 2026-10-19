//! Bounding-volume hierarchy over mesh triangles.
//!
//! A minimal recursive AABB tree: internal nodes store a child pair, leaves
//! store a contiguous range into a global permutation of the triangle
//! indices. Builds in `O(n log n)` over the longest-axis median split;
//! queries are `O(log n + k)` for the candidates (plus the exact triangle
//! tests by the caller).

use bullet_math::{Aabb, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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
                    for tri in &self.tri_order[start as usize..(start + count) as usize] {
                        out.push(*tri);
                    }
                }
            }
        }
    }

    /// Append triangle-index pairs `(this, other)` whose node boxes overlap
    /// when `other`'s tree is mapped into this tree's frame by
    /// `rotation`/`translation`. Mirrors Bullet's
    /// `_find_quantized_collision_pairs_recursive`: node pairs are tested by
    /// transforming `other`'s node box corners into this frame (the
    /// `BT_BOX_BOX_TRANSFORM_CACHE` conservative overlap), leaf/leaf pairs
    /// emit their triangle cross products, and mixed pairs split the internal
    /// side. `margin` inflates the transformed box — the pair contact band
    /// (Bullet's GImpact primitive boxes carry their own 0.1 margin for the
    /// same purpose: zero-thickness parallel plates otherwise never overlap
    /// their boxes until fully past each other).
    pub(crate) fn triangles_overlapping(
        &self,
        other: &Bvh,
        rotation: Quaternion,
        translation: Vector3,
        margin: Real,
        out: &mut Vec<(u32, u32)>,
    ) {
        let mut stack = vec![(0usize, 0usize)];
        while let Some((ia, ib)) = stack.pop() {
            let a = &self.nodes[ia];
            let b = &other.nodes[ib];
            if !overlaps_transformed(&a.aabb, &b.aabb, rotation, translation, margin) {
                continue;
            }
            match (&a.kind, &b.kind) {
                (
                    BvhNodeKind::Leaf { start: sa, count: ca },
                    BvhNodeKind::Leaf { start: sb, count: cb },
                ) => {
                    for ta in &self.tri_order[*sa as usize..(*sa + *ca) as usize] {
                        for tb in &other.tri_order[*sb as usize..(*sb + *cb) as usize] {
                            out.push((*ta, *tb));
                        }
                    }
                }
                (BvhNodeKind::Leaf { .. }, BvhNodeKind::Internal { left, right }) => {
                    stack.push((ia, *left as usize));
                    stack.push((ia, *right as usize));
                }
                (BvhNodeKind::Internal { left, right }, BvhNodeKind::Leaf { .. }) => {
                    stack.push((*left as usize, ib));
                    stack.push((*right as usize, ib));
                }
                (
                    BvhNodeKind::Internal { left: la, right: ra },
                    BvhNodeKind::Internal { left: lb, right: rb },
                ) => {
                    stack.push((*la as usize, *lb as usize));
                    stack.push((*la as usize, *rb as usize));
                    stack.push((*ra as usize, *lb as usize));
                    stack.push((*ra as usize, *rb as usize));
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

/// Conservative overlap of `b`'s box with `a`'s box after mapping `b` by
/// `rotation`/`translation` into `a`'s frame and inflating by `margin`: the
/// transformed corners are re-wrapped in an axis-aligned box (Bullet's
/// `btAABB::overlapping_trans_cache` without the full primitive test). The
/// wrap is a superset of the rotated box, so a miss here is a true miss and
/// a hit defers to the exact triangle tests.
fn overlaps_transformed(
    a: &Aabb,
    b: &Aabb,
    rotation: Quaternion,
    translation: Vector3,
    margin: Real,
) -> bool {
    let mut min = Vector3::splat(Real::INFINITY);
    let mut max = Vector3::splat(Real::NEG_INFINITY);
    for i in 0..8 {
        let corner = Vector3::new(
            if i & 1 == 0 { b.max.x } else { b.min.x },
            if i & 2 == 0 { b.max.y } else { b.min.y },
            if i & 4 == 0 { b.max.z } else { b.min.z },
        );
        let p = rotation * corner + translation;
        min = min.min(p);
        max = max.max(p);
    }
    Aabb::new(min - Vector3::splat(margin), max + Vector3::splat(margin)).overlaps(a)
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
                core::mem::swap(&mut t1, &mut t2);
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
