use std::collections::HashSet;

use bullet_math::{Aabb, Real, Vector3};

use super::{aabb_raycast_candidates, BroadPhase};

/// Quantized position mask and sentinel (Bullet's 16-bit `btAxisSweep3`:
/// `m_bpHandleMask = 0xfffe`, `m_handleSentinel = 0xffff`).
const MASK: u16 = 0xfffe;
const SENTINEL_POS: u16 = 0xffff;
/// Owner id of the two boundary sentinel edges (Bullet reserves handle 0).
const SENTINEL_HANDLE: usize = usize::MAX;
/// Components at or beyond this magnitude (plane AABBs are ±1e30) do not
/// constrain the auto-grown quantization frame; such proxies simply quantize
/// to the clamped frame edges, conservative like a plane inside Bullet's
/// user-supplied world bounds.
const FRAME_COMPONENT_LIMIT: f64 = 1e18;

/// Sweep-and-prune broad phase over all three axes (Bullet's `btAxisSweep3`).
///
/// Each axis stores one interleaved list of min/max *edges* sorted by a
/// 16-bit quantized position (Bullet quantizes into a fixed world box; here
/// the frame is grown automatically and everything is re-quantized when a
/// proxy leaves it). The low bit of an edge position marks max edges, so a
/// proxy's min edge always sorts strictly before its max edge.
///
/// Updates are incremental, exactly like Bullet's `updateHandle`: the moved
/// edges keep their slot, their positions are rewritten, and each edge
/// bubbles towards its new sorted position by adjacent swaps
/// (`sortMinDown`/`sortMinUp`/`sortMaxDown`/`sortMaxUp`). Swapping past
/// another proxy's edge adds or removes that pair in a persistent overlap
/// set (with a 2D edge-index pretest on add, `testOverlap2D`), so
/// [`compute_pairs`](BroadPhase::compute_pairs) only walks the cached set.
///
/// Quantization makes the cached set slightly conservative (sub-quantum
/// gaps read as overlapping); the reported pairs are filtered back with the
/// exact stored AABBs so the output matches the exact-float semantics of
/// [`BruteForceBroadPhase`](super::BruteForceBroadPhase) pair for pair.
///
/// The quantization also absorbs jitter: a sub-quantum AABB change rewrites
/// nothing at all (Bullet's resting stacks do no work in the broad phase).
///
/// Output order matches [`BruteForceBroadPhase`](super::BruteForceBroadPhase)
/// for identical inputs (lexicographic on `(i, j)`), so tests can compare
/// directly.
pub struct AxisSweep3 {
    handles: Vec<Option<Handle>>,
    edges: [Vec<Edge>; 3],
    /// Persistently maintained quantized-overlap set (Bullet's pair cache),
    /// normalized `(min, max)` proxy ids.
    pairs: HashSet<(usize, usize)>,
    /// Quantization frame (Bullet's `m_worldAabbMin/Max` + `m_quantize`),
    /// grown from the finite proxy AABBs with 50% headroom per axis.
    frame_min: Vector3,
    frame_max: Vector3,
    quant: Vector3,
    has_frame: bool,
}

struct Handle {
    aabb: Aabb,
    /// Index of this proxy's min/max edge in each axis's edge list.
    min_edges: [usize; 3],
    max_edges: [usize; 3],
}

#[derive(Clone, Copy)]
struct Edge {
    /// Quantized position; the low bit marks a max edge (Bullet's
    /// `Edge::IsMax`).
    pos: u16,
    handle: usize,
}

impl Edge {
    #[inline]
    fn is_max(&self) -> bool {
        self.pos & 1 == 1
    }
}

impl Default for AxisSweep3 {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn pair_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

#[inline]
fn frame_relevant(v: f64) -> bool {
    v.is_finite() && v.abs() < FRAME_COMPONENT_LIMIT
}

impl AxisSweep3 {
    pub fn new() -> Self {
        Self {
            handles: Vec::new(),
            edges: [Vec::new(), Vec::new(), Vec::new()],
            pairs: HashSet::new(),
            frame_min: Vector3::ZERO,
            frame_max: Vector3::ZERO,
            quant: Vector3::ZERO,
            has_frame: false,
        }
    }

    /// Quantize one corner (Bullet's `quantize`, non-clamping variant): the
    /// low bit encodes `is_max`, the value is clamped to `[isMax,
    /// SENTINEL_POS]`.
    fn quantize(&self, point: Vector3, is_max: bool) -> [u16; 3] {
        let v = (point - self.frame_min) * self.quant;
        let flag = u16::from(is_max);
        let mut out = [0u16; 3];
        for k in 0..3 {
            out[k] = if v[k] <= 0.0 {
                flag
            } else if v[k] >= SENTINEL_POS as Real {
                (SENTINEL_POS & MASK) | flag
            } else {
                ((v[k] as u16) & MASK) | flag
            };
        }
        out
    }

    /// Whether `aabb` has a finite component outside the current frame (the
    /// rebuild trigger; plane-like infinite proxies never trigger it).
    fn needs_rebuild(&self, aabb: &Aabb) -> bool {
        if !self.has_frame {
            return true;
        }
        for k in 0..3 {
            if frame_relevant(aabb.min[k] as f64) && aabb.min[k] < self.frame_min[k] {
                return true;
            }
            if frame_relevant(aabb.max[k] as f64) && aabb.max[k] > self.frame_max[k] {
                return true;
            }
        }
        false
    }

    /// Recompute the quantization frame from all live AABBs (plus `extra`,
    /// a proxy being inserted/updated) and rebuild every edge list and the
    /// pair set from scratch (Bullet builds a `btAxisSweep3` over a fixed
    /// user-supplied world box; the auto-grown frame is this crate's
    /// unbounded-API equivalent).
    fn rebuild(&mut self, extra: Option<&Aabb>) {
        let mut lo = Vector3::splat(Real::INFINITY);
        let mut hi = Vector3::splat(Real::NEG_INFINITY);
        let absorb = |aabb: &Aabb, lo: &mut Vector3, hi: &mut Vector3| {
            for k in 0..3 {
                if frame_relevant(aabb.min[k] as f64) {
                    lo[k] = lo[k].min(aabb.min[k]);
                }
                if frame_relevant(aabb.max[k] as f64) {
                    hi[k] = hi[k].max(aabb.max[k]);
                }
            }
        };
        for h in self.handles.iter().flatten() {
            absorb(&h.aabb, &mut lo, &mut hi);
        }
        if let Some(aabb) = extra {
            absorb(aabb, &mut lo, &mut hi);
        }
        for k in 0..3 {
            // 50% headroom per axis (floor for degenerate points) so normal
            // motion stays inside the frame for a while.
            let extent = (hi[k] - lo[k]).max(0.0);
            let margin = (extent * 0.5).max(0.5);
            if !frame_relevant(lo[k] as f64) {
                lo[k] = -0.5;
                hi[k] = 0.5;
            } else {
                lo[k] -= margin;
                hi[k] += margin;
            }
            self.quant[k] = SENTINEL_POS as Real / (hi[k] - lo[k]);
        }
        self.frame_min = lo;
        self.frame_max = hi;
        self.has_frame = true;

        self.pairs.clear();
        let slots: Vec<usize> = self
            .handles
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.as_ref().map(|_| i))
            .collect();
        for k in 0..3 {
            self.edges[k] = vec![
                Edge { pos: 0, handle: SENTINEL_HANDLE },
                Edge { pos: SENTINEL_POS, handle: SENTINEL_HANDLE },
            ];
        }
        for slot in slots {
            let aabb = self.handles[slot].as_ref().unwrap().aabb;
            self.insert_handle(slot, aabb);
        }
    }

    /// Append `slot`'s two edges in front of the max sentinel of every axis
    /// and bubble them down into place, maintaining the pair set on the last
    /// axis only (Bullet's `addHandle`: one z-axis pass with the 2D pretest
    /// adds exactly the 3D-overlapping pairs).
    fn insert_handle(&mut self, slot: usize, aabb: Aabb) {
        let qmin = self.quantize(aabb.min, false);
        let qmax = self.quantize(aabb.max, true);
        for k in 0..3 {
            let n = self.edges[k].len();
            self.edges[k].truncate(n - 1); // drop the max sentinel
            self.edges[k].push(Edge { pos: qmin[k], handle: slot });
            self.edges[k].push(Edge { pos: qmax[k], handle: slot });
            self.edges[k].push(Edge { pos: SENTINEL_POS, handle: SENTINEL_HANDLE });
            let h = self.handles[slot].as_mut().unwrap();
            h.min_edges[k] = n - 1;
            h.max_edges[k] = n;
        }
        for k in 0..3 {
            let (emin, emax) = {
                let h = self.handles[slot].as_ref().unwrap();
                (h.min_edges[k], h.max_edges[k])
            };
            let update_overlaps = k == 2;
            self.sort_min_down(k, emin, update_overlaps);
            self.sort_max_down(k, emax, update_overlaps);
        }
    }

    /// Swap two adjacent edges, fixing the owners' stored edge indices
    /// (Bullet does the fixups inline in its sorts).
    fn swap_edges(&mut self, k: usize, i: usize, j: usize) {
        let a = self.edges[k][i];
        let b = self.edges[k][j];
        if a.handle != SENTINEL_HANDLE {
            let h = self.handles[a.handle].as_mut().unwrap();
            if a.is_max() { h.max_edges[k] = j } else { h.min_edges[k] = j }
        }
        if b.handle != SENTINEL_HANDLE {
            let h = self.handles[b.handle].as_mut().unwrap();
            if b.is_max() { h.max_edges[k] = i } else { h.min_edges[k] = i }
        }
        self.edges[k][i] = b;
        self.edges[k][j] = a;
    }

    /// Bullet's `testOverlap2D`: overlap on the other two axes, decided by
    /// edge-array indices (the arrays are sorted, so index order equals
    /// position order).
    fn test_overlap_2d(&self, a: usize, b: usize, axis0: usize, axis1: usize) -> bool {
        let ha = self.handles[a].as_ref().unwrap();
        let hb = self.handles[b].as_ref().unwrap();
        !(ha.max_edges[axis0] < hb.min_edges[axis0]
            || hb.max_edges[axis0] < ha.min_edges[axis0]
            || ha.max_edges[axis1] < hb.min_edges[axis1]
            || hb.max_edges[axis1] < ha.min_edges[axis1])
    }

    /// A min edge bubbling down can only *add* overlaps (Bullet's
    /// `sortMinDown`); stops at the min sentinel (pos 0).
    fn sort_min_down(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        while i > 0 && self.edges[k][i].pos < self.edges[k][i - 1].pos {
            let prev = self.edges[k][i - 1];
            if prev.is_max() && update_overlaps {
                let handle = self.edges[k][i].handle;
                let axis1 = (k + 1) % 3;
                let axis2 = (k + 2) % 3;
                if self.test_overlap_2d(handle, prev.handle, axis1, axis2) {
                    self.pairs.insert(pair_key(handle, prev.handle));
                }
            }
            self.swap_edges(k, i - 1, i);
            i -= 1;
        }
    }

    /// A min edge bubbling up can only *remove* overlaps (Bullet's
    /// `sortMinUp`; removal is unconditional — `USE_OVERLAP_TEST_ON_REMOVES`
    /// is off — because an axis crossing implies separation).
    fn sort_min_up(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        while self.edges[k][i + 1].handle != SENTINEL_HANDLE
            && self.edges[k][i].pos >= self.edges[k][i + 1].pos
        {
            let next = self.edges[k][i + 1];
            if next.is_max() && update_overlaps {
                let handle = self.edges[k][i].handle;
                self.pairs.remove(&pair_key(handle, next.handle));
            }
            self.swap_edges(k, i, i + 1);
            i += 1;
        }
    }

    /// A max edge bubbling down can only *remove* overlaps (Bullet's
    /// `sortMaxDown`).
    fn sort_max_down(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        while i > 0 && self.edges[k][i].pos < self.edges[k][i - 1].pos {
            let prev = self.edges[k][i - 1];
            if !prev.is_max() && update_overlaps {
                let handle = self.edges[k][i].handle;
                self.pairs.remove(&pair_key(handle, prev.handle));
            }
            self.swap_edges(k, i - 1, i);
            i -= 1;
        }
    }

    /// A max edge bubbling up can only *add* overlaps (Bullet's `sortMaxUp`:
    /// crossing another proxy's min edge with the 2D pretest).
    fn sort_max_up(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        while self.edges[k][i + 1].handle != SENTINEL_HANDLE
            && self.edges[k][i].pos >= self.edges[k][i + 1].pos
        {
            let next = self.edges[k][i + 1];
            if !next.is_max() && update_overlaps {
                let handle = self.edges[k][i].handle;
                let axis1 = (k + 1) % 3;
                let axis2 = (k + 2) % 3;
                if self.test_overlap_2d(handle, next.handle, axis1, axis2) {
                    self.pairs.insert(pair_key(handle, next.handle));
                }
            }
            self.swap_edges(k, i, i + 1);
            i += 1;
        }
    }
}

impl BroadPhase for AxisSweep3 {
    fn create_proxy(&mut self, aabb: Aabb) -> usize {
        if let Some(i) = self.handles.iter().position(|h| h.is_none()) {
            self.handles[i] = Some(Handle {
                aabb,
                min_edges: [0; 3],
                max_edges: [0; 3],
            });
            if self.needs_rebuild(&aabb) {
                self.rebuild(Some(&aabb));
            } else {
                self.insert_handle(i, aabb);
            }
            i
        } else {
            self.handles.push(Some(Handle {
                aabb,
                min_edges: [0; 3],
                max_edges: [0; 3],
            }));
            let slot = self.handles.len() - 1;
            if self.needs_rebuild(&aabb) {
                self.rebuild(Some(&aabb));
            } else {
                self.insert_handle(slot, aabb);
            }
            slot
        }
    }

    fn set_aabb(&mut self, proxy: usize, aabb: Aabb) {
        let Some(handle) = self.handles[proxy].as_mut() else {
            return;
        };
        handle.aabb = aabb;
        if self.needs_rebuild(&aabb) {
            self.rebuild(None);
            return;
        }
        let (qmin, qmax) = (self.quantize(aabb.min, false), self.quantize(aabb.max, true));
        for k in 0..3 {
            let (emin, emax, old_min, old_max) = {
                let h = self.handles[proxy].as_ref().unwrap();
                (
                    h.min_edges[k],
                    h.max_edges[k],
                    self.edges[k][h.min_edges[k]].pos,
                    self.edges[k][h.max_edges[k]].pos,
                )
            };
            // Sub-quantum jitter rewrites nothing (Bullet's resting stacks
            // do no broad-phase work).
            if qmin[k] == old_min && qmax[k] == old_max {
                continue;
            }
            self.edges[k][emin].pos = qmin[k];
            self.edges[k][emax].pos = qmax[k];
            // Expand first (only adds overlaps), then shrink (only removes).
            if qmin[k] < old_min {
                self.sort_min_down(k, emin, true);
            }
            if qmax[k] > old_max {
                self.sort_max_up(k, emax, true);
            }
            if qmin[k] > old_min {
                self.sort_min_up(k, emin, true);
            }
            if qmax[k] < old_max {
                self.sort_max_down(k, emax, true);
            }
        }
    }

    fn destroy_proxy(&mut self, proxy: usize) {
        // Drop the proxy's pairs, then remove its edges and shift every
        // stored index on each axis (destroys are rare; the O(n) fixup is
        // fine — Bullet instead swaps the dead edges to the array end).
        self.pairs.retain(|&(a, b)| a != proxy && b != proxy);
        for k in 0..3 {
            let (imin, imax) = {
                let h = self.handles[proxy].as_ref().unwrap();
                (h.min_edges[k], h.max_edges[k])
            };
            let (lo, hi) = (imin.min(imax), imax.max(imin));
            self.edges[k].remove(hi);
            self.edges[k].remove(lo);
            for h in self.handles.iter_mut().flatten() {
                for idx in [&mut h.min_edges[k], &mut h.max_edges[k]] {
                    if *idx > hi {
                        *idx -= 2;
                    } else if *idx > lo {
                        *idx -= 1;
                    }
                }
            }
        }
        self.handles[proxy] = None;
    }

    fn proxy_aabb(&self, proxy: usize) -> Option<Aabb> {
        self.handles[proxy].as_ref().map(|h| h.aabb)
    }

    fn compute_pairs(&self) -> Vec<(usize, usize)> {
        // The cached set is quantized-conservative; the exact stored AABBs
        // bring the output back to brute-force-exact semantics.
        let mut pairs: Vec<(usize, usize)> = self
            .pairs
            .iter()
            .copied()
            .filter(|&(a, b)| {
                let ha = self.handles[a].as_ref().unwrap();
                let hb = self.handles[b].as_ref().unwrap();
                ha.aabb.overlaps(&hb.aabb)
            })
            .collect();
        pairs.sort_unstable();
        pairs
    }

    fn raycast_candidates(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<usize> {
        aabb_raycast_candidates(&self.live_aabbs(), ray_from, ray_to)
    }

    fn aabb_candidates(&self, query: Aabb) -> Vec<usize> {
        self.live_aabbs()
            .iter()
            .enumerate()
            .filter_map(|(i, a)| a.filter(|aabb| aabb.overlaps(&query)).map(|_| i))
            .collect()
    }
}

impl AxisSweep3 {
    fn live_aabbs(&self) -> Vec<Option<Aabb>> {
        self.handles.iter().map(|h| h.as_ref().map(|x| x.aabb)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_math::{Real, Vector3};

    #[test]
    fn finds_overlapping_pairs() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::splat(0.0), Vector3::splat(1.0)));
        let b = bp.create_proxy(Aabb::new(Vector3::splat(0.5), Vector3::splat(1.5)));
        let c = bp.create_proxy(Aabb::new(Vector3::splat(10.0), Vector3::splat(11.0)));
        let pairs = bp.compute_pairs();
        assert_eq!(pairs, vec![(a, b)]);
        assert!(pairs.iter().all(|&(x, y)| x != c && y != c));
    }

    #[test]
    fn proxy_slot_reuse() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        bp.destroy_proxy(a);
        let b = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        assert_eq!(a, b);
    }

    #[test]
    fn moved_proxies_are_recomputed() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)));
        assert!(bp.compute_pairs().is_empty());
        // Move `b` to overlap `a` and re-query.
        bp.set_aabb(b, Aabb::new(Vector3::new(0.5, 0.0, 0.0), Vector3::new(1.5, 1.0, 1.0)));
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
    }

    /// A single-axis overlap is not enough: pairs must overlap on all axes.
    #[test]
    fn requires_overlap_on_all_axes() {
        let mut bp = AxisSweep3::new();
        // Overlap on X and Y but separated on Z.
        bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0)));
        bp.create_proxy(Aabb::new(Vector3::new(0.5, 0.5, 5.0), Vector3::new(1.5, 1.5, 6.0)));
        assert!(bp.compute_pairs().is_empty());
    }

    #[test]
    fn matches_brute_force_on_scattered_boxes() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        // Deterministic pseudo-random boxes across the volume.
        let mut seed = 0x9e3779b97f4a7c15u64;
        for _ in 0..64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((seed >> 32) % 1000) as Real / 10.0,
                ((seed >> 16) % 1000) as Real / 10.0,
                (seed % 1000) as Real / 10.0,
            );
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let extent = Vector3::new(
                ((seed >> 32) % 100) as Real / 100.0 + 0.1,
                ((seed >> 16) % 100) as Real / 100.0 + 0.1,
                (seed % 100) as Real / 100.0 + 0.1,
            );
            let aabb = Aabb::new(base, base + extent);
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
    }

    /// Incremental updates must stay equivalent to the brute-force baseline
    /// even after proxies move far, are destroyed, and are re-created.
    #[test]
    fn incremental_updates_match_brute_force() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut aabbs = Vec::new();
        let mut seed = 0x243f6a8885a308d3u64;
        let random_aabb = |seed: &mut u64| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((*seed >> 32) % 500) as Real / 10.0,
                ((*seed >> 16) % 500) as Real / 10.0,
                (*seed % 500) as Real / 10.0,
            );
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let extent = Vector3::new(
                ((*seed >> 32) % 100) as Real / 100.0 + 0.1,
                ((*seed >> 16) % 100) as Real / 100.0 + 0.1,
                (*seed % 100) as Real / 100.0 + 0.1,
            );
            Aabb::new(base, base + extent)
        };
        for _ in 0..32 {
            let aabb = random_aabb(&mut seed);
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
            aabbs.push(aabb);
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
        // Teleport every even proxy elsewhere, destroy odd ones, add new ones.
        for (i, aabb) in aabbs.iter_mut().enumerate() {
            if i % 2 == 0 {
                *aabb = random_aabb(&mut seed);
                sweep.set_aabb(i, *aabb);
                brute.set_aabb(i, *aabb);
            } else {
                sweep.destroy_proxy(i);
                brute.destroy_proxy(i);
            }
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
        // Re-create into the freed slots and teleport again.
        for i in (0..aabbs.len()).filter(|i| i % 2 == 1) {
            let aabb = random_aabb(&mut seed);
            assert_eq!(sweep.create_proxy(aabb), i);
            assert_eq!(brute.create_proxy(aabb), i);
            aabbs[i] = aabb;
        }
        let aabb = random_aabb(&mut seed);
        sweep.set_aabb(0, aabb);
        brute.set_aabb(0, aabb);
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
    }

    /// Setting an identical AABB must be a no-op (static bodies cost nothing).
    #[test]
    fn unchanged_aabb_is_a_noop() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)));
        let before = bp.compute_pairs();
        bp.set_aabb(a, Aabb::new(Vector3::ZERO, Vector3::ONE));
        assert_eq!(bp.compute_pairs(), before);
        // A ray along X pierces both AABBs at y = z = 0.5.
        let ray = Vector3::new(0.5, 0.5, 0.5);
        assert_eq!(bp.raycast_candidates(Vector3::new(-1.0, 0.5, 0.5), ray + Vector3::X * 5.0), vec![a, b]);
    }

    /// A sub-quantum AABB change (resting-stack jitter) rewrites no edges:
    /// the pair set stays bit-identical (Bullet's quantization absorbing
    /// float noise).
    #[test]
    fn subquantum_jitter_is_absorbed() {
        let mut bp = AxisSweep3::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(0.5, 0.9, 0.5), Vector3::new(1.5, 1.9, 1.5)));
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
        let jitter = Aabb::new(
            Vector3::new(0.5, 0.9 + 1e-7, 0.5),
            Vector3::new(1.5, 1.9 + 1e-7, 1.5),
        );
        bp.set_aabb(b, jitter);
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
        assert_eq!(bp.proxy_aabb(b), Some(jitter));
    }

    /// Moving proxies far outside the initial quantization frame triggers
    /// rebuilds that must stay equivalent to the brute-force baseline.
    #[test]
    fn frame_growth_matches_brute_force() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut aabbs = Vec::new();
        for i in 0..8 {
            // A marching column: each step far outside the previous frame.
            let x = 100.0 * i as Real;
            let aabb = Aabb::new(Vector3::new(x, 0.0, 0.0), Vector3::new(x + 1.0, 1.0, 1.0));
            aabbs.push(aabb);
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
        // Collapse the column into one overlapping cluster (another rebuild).
        for (i, aabb) in aabbs.iter_mut().enumerate() {
            let x = (i % 2) as Real * 0.5;
            *aabb = Aabb::new(Vector3::new(x, 0.0, 0.0), Vector3::new(x + 1.0, 1.0, 1.0));
            sweep.set_aabb(i, *aabb);
            brute.set_aabb(i, *aabb);
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
    }

    /// Plane-like infinite AABBs must not poison the quantization frame:
    /// finite proxies keep exact brute-force semantics alongside them.
    #[test]
    fn infinite_aabbs_do_not_poison_the_frame() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let big = Vector3::splat(1e30);
        let ground = Aabb::new(-big, big);
        sweep.create_proxy(ground);
        brute.create_proxy(ground);
        let mut seed = 0x5deece66d96b84b5u64;
        for _ in 0..16 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((seed >> 32) % 100) as Real / 10.0,
                ((seed >> 16) % 100) as Real / 10.0,
                (seed % 100) as Real / 10.0,
            );
            let aabb = Aabb::new(base, base + Vector3::splat(0.5));
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        assert_eq!(sweep.compute_pairs(), brute.compute_pairs());
    }

    #[test]
    fn raycast_candidates_filters_by_ray_path() {
        let mut bp = AxisSweep3::new();
        let hit = bp.create_proxy(Aabb::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(2.0, 2.0, 2.0)));
        let miss = bp.create_proxy(Aabb::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(11.0, 11.0, 11.0)));
        // Ray passes through `hit` but stops well before `miss`.
        let candidates = bp.raycast_candidates(Vector3::ZERO, Vector3::new(1.5, 1.5, 1.5));
        assert_eq!(candidates, vec![hit]);
        // Ray starting inside the box still reports it.
        let inside = bp.raycast_candidates(Vector3::new(1.5, 1.5, 1.5), Vector3::new(4.0, 4.0, 4.0));
        assert!(inside.contains(&hit) && !inside.contains(&miss));
        // Box behind the ray origin is pruned.
        let behind = bp.raycast_candidates(Vector3::new(3.0, 3.0, 3.0), Vector3::new(4.0, 4.0, 4.0));
        assert!(behind.is_empty());
    }

    #[test]
    fn raycast_candidates_matches_brute_force() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut seed = 0x13198a2e03707344u64;
        for _ in 0..48 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((seed >> 32) % 400) as Real / 10.0,
                ((seed >> 16) % 400) as Real / 10.0,
                (seed % 400) as Real / 10.0,
            );
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let extent = Vector3::new(
                ((seed >> 32) % 100) as Real / 100.0 + 0.1,
                ((seed >> 16) % 100) as Real / 100.0 + 0.1,
                (seed % 100) as Real / 100.0 + 0.1,
            );
            let aabb = Aabb::new(base, base + extent);
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        let from = Vector3::new(5.0, 6.0, 7.0);
        let to = Vector3::new(25.0, 2.0, 30.0);
        assert_eq!(
            sweep.raycast_candidates(from, to),
            brute.raycast_candidates(from, to)
        );
    }

    #[test]
    fn aabb_candidates_matches_brute_force() {
        let mut sweep = AxisSweep3::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut seed = 0xdeadbeefcafef00du64;
        for _ in 0..48 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let base = Vector3::new(
                ((seed >> 32) % 400) as Real / 10.0,
                ((seed >> 16) % 400) as Real / 10.0,
                (seed % 400) as Real / 10.0,
            );
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let extent = Vector3::new(
                ((seed >> 32) % 100) as Real / 100.0 + 0.1,
                ((seed >> 16) % 100) as Real / 100.0 + 0.1,
                (seed % 100) as Real / 100.0 + 0.1,
            );
            let aabb = Aabb::new(base, base + extent);
            sweep.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        // Query AABBs at scattered positions (including overlaps, edge touches
        // and empty regions).
        let queries = [
            Aabb::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(20.0, 20.0, 20.0)),
            Aabb::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(12.0, 12.0, 12.0)),
            Aabb::new(Vector3::new(50.0, 50.0, 50.0), Vector3::new(60.0, 60.0, 60.0)),
            Aabb::new(Vector3::new(39.9, 0.0, 0.0), Vector3::new(40.0, 40.0, 40.0)),
        ];
        for q in queries {
            assert_eq!(sweep.aabb_candidates(q), brute.aabb_candidates(q), "query {q:?}");
        }
    }
}

