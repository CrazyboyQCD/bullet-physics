use bullet_math::{Aabb, Real, Vector3};

use super::{aabb_raycast_candidates, BroadPhase};

/// Quantized position mask and sentinel (Bullet's 16-bit `btAxisSweep3`:
/// `m_bpHandleMask = 0xfffe`, `m_handleSentinel = 0xffff`).
const MASK: u16 = 0xfffe;
const SENTINEL_POS: u16 = 0xffff;
/// Owner id of the two boundary sentinel edges (Bullet reserves handle 0).
const SENTINEL_HANDLE: u32 = u32::MAX;
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
    aabbs: Vec<Option<Aabb>>,
    /// Per-proxy min/max edge indices per axis, parallel to `aabbs`. Kept
    /// out of `Option` and down to `u32` so the edge-index fixups inside a
    /// swap-heavy `sort_*` bubble are flat array writes (the old
    /// in-`Handle` `usize` triples cost an `Option` unwrap plus a fat
    /// bounds-checked write per swap step — the dominant constant-factor
    /// gap against Bullet's `updateHandle` on dense swinging scenes).
    min_edges: Vec<[u32; 3]>,
    max_edges: Vec<[u32; 3]>,
    edges: [Vec<Edge>; 3],
    /// Persistently maintained quantized-overlap set (Bullet's pair cache),
    /// normalized `(min, max)` proxy ids kept sorted lexicographically at all
    /// times: edge swaps touch a handful of pairs per step, so binary-search
    /// insert/remove is cheap while [`compute_pairs`](BroadPhase::compute_pairs)
    /// gets its output order for free — resting stacks re-sorted the whole set
    /// every frame despite it never changing (a double-digit slice of a full
    /// step on the 1000-body stack profile; an earlier Fx-hash `HashSet`
    /// removed the SipHash tax but still needed the sort for output order).
    pairs: Vec<(usize, usize)>,
    /// Quantization frame (Bullet's `m_worldAabbMin/Max` + `m_quantize`),
    /// grown from the finite proxy AABBs with headroom per axis.
    frame_min: Vector3,
    frame_max: Vector3,
    quant: Vector3,
    has_frame: bool,
}

#[derive(Clone, Copy)]
struct Edge {
    /// Quantized position; the low bit marks a max edge (Bullet's
    /// `Edge::IsMax`).
    pos: u16,
    /// Owning proxy slot (`u32` halves the edge-table footprint against
    /// `usize`; Bullet packs both fields into one `unsigned short` each).
    handle: u32,
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

impl std::fmt::Debug for AxisSweep3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AxisSweep3").finish_non_exhaustive()
    }
}

#[inline]
fn pair_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

#[inline]
fn frame_relevant(v: impl Into<f64>) -> bool {
    let v = v.into();
    v.is_finite() && v.abs() < FRAME_COMPONENT_LIMIT
}

impl AxisSweep3 {
    pub fn new() -> Self {
        Self {
            aabbs: Vec::new(),
            min_edges: Vec::new(),
            max_edges: Vec::new(),
            edges: [Vec::new(), Vec::new(), Vec::new()],
            pairs: Vec::new(),
            frame_min: Vector3::ZERO,
            frame_max: Vector3::ZERO,
            quant: Vector3::ZERO,
            has_frame: false,
        }
    }

    /// Insert into the sorted pair set. Edge swaps touch a handful of pairs
    /// per step, so the `O(log n)` search plus a short shift beats any
    /// re-sorting in `compute_pairs`.
    #[inline]
    fn pair_insert(&mut self, key: (usize, usize)) {
        if let Err(i) = self.pairs.binary_search(&key) {
            self.pairs.insert(i, key);
        }
    }

    /// Remove from the sorted pair set (missing keys are a no-op, matching
    /// the `HashSet::remove` semantics of the previous pair cache).
    #[inline]
    fn pair_remove(&mut self, key: (usize, usize)) {
        if let Ok(i) = self.pairs.binary_search(&key) {
            self.pairs.remove(i);
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
            if frame_relevant(aabb.min[k]) && aabb.min[k] < self.frame_min[k] {
                return true;
            }
            if frame_relevant(aabb.max[k]) && aabb.max[k] > self.frame_max[k] {
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
                if frame_relevant(aabb.min[k]) {
                    lo[k] = lo[k].min(aabb.min[k]);
                }
                if frame_relevant(aabb.max[k]) {
                    hi[k] = hi[k].max(aabb.max[k]);
                }
            }
        };
        for aabb in self.aabbs.iter().flatten() {
            absorb(aabb, &mut lo, &mut hi);
        }
        if let Some(aabb) = extra {
            absorb(aabb, &mut lo, &mut hi);
        }
        for k in 0..3 {
            // 100% headroom per axis (floor for degenerate points). The
            // headroom serves two masters at once: it keeps oscillating
            // bodies (a swinging chain's AABB peaks drift frame to frame)
            // inside the frame so full rebuilds stay rare, and the wider
            // frame coarsens the 16-bit quantization, which halves the
            // edge-swap traffic per unit of motion. Both effects are
            // output-neutral: quantization is a conservative floor/ceil
            // mapping (the edge tables overlap-superset the exact AABBs)
            // and `compute_pairs` reports the exact-filtered, ordered pair
            // set regardless of the frame's granularity.
            let extent = (hi[k] - lo[k]).max(0.0);
            let margin = (extent * 1.0).max(0.5);
            if !frame_relevant(lo[k]) {
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
        let live: Vec<(usize, Aabb)> = self
            .aabbs
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.map(|aabb| (i, aabb)))
            .collect();
        for k in 0..3 {
            self.edges[k] = vec![
                Edge { pos: 0, handle: SENTINEL_HANDLE },
                Edge { pos: SENTINEL_POS, handle: SENTINEL_HANDLE },
            ];
        }
        for (slot, aabb) in live {
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
            self.edges[k].push(Edge { pos: qmin[k], handle: slot as u32 });
            self.edges[k].push(Edge { pos: qmax[k], handle: slot as u32 });
            self.edges[k].push(Edge { pos: SENTINEL_POS, handle: SENTINEL_HANDLE });
            self.min_edges[slot][k] = (n - 1) as u32;
            self.max_edges[slot][k] = n as u32;
        }
        for k in 0..3 {
            let (emin, emax) = (self.min_edges[slot][k] as usize, self.max_edges[slot][k] as usize);
            let update_overlaps = k == 2;
            self.sort_min_down(k, emin, update_overlaps);
            self.sort_max_down(k, emax, update_overlaps);
        }
    }

    /// Bullet's `testOverlap2D`: overlap on the other two axes, decided by
    /// edge-array indices (the arrays are sorted, so index order equals
    /// position order).
    #[inline]
    fn test_overlap_2d(&self, a: usize, b: usize, axis0: usize, axis1: usize) -> bool {
        let (a_min, a_max) = (&self.min_edges[a], &self.max_edges[a]);
        let (b_min, b_max) = (&self.min_edges[b], &self.max_edges[b]);
        !(a_max[axis0] < b_min[axis0]
            || b_max[axis0] < a_min[axis0]
            || a_max[axis1] < b_min[axis1]
            || b_max[axis1] < a_min[axis1])
    }

    /// A min edge bubbling down can only *add* overlaps (Bullet's
    /// `sortMinDown`); stops at the min sentinel (pos 0). The moving edge
    /// is hoisted out of the loop — its position and handle are loop
    /// invariants — and its own edge index is written once at the end
    /// instead of on every swap; only the displaced neighbour needs a
    /// per-step fixup (a flat `u32` write into `min_edges`/`max_edges`).
    fn sort_min_down(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        let moving = self.edges[k][i];
        let axis1 = (k + 1) % 3;
        let axis2 = (k + 2) % 3;
        while i > 0 {
            let prev = self.edges[k][i - 1];
            if moving.pos >= prev.pos {
                break;
            }
            if prev.is_max() {
                if update_overlaps
                    && self.test_overlap_2d(moving.handle as usize, prev.handle as usize, axis1, axis2)
                {
                    self.pair_insert(pair_key(moving.handle as usize, prev.handle as usize));
                }
                self.max_edges[prev.handle as usize][k] = i as u32;
            } else {
                self.min_edges[prev.handle as usize][k] = i as u32;
            }
            self.edges[k][i] = prev;
            self.edges[k][i - 1] = moving;
            i -= 1;
        }
        self.min_edges[moving.handle as usize][k] = i as u32;
    }

    /// A min edge bubbling up can only *remove* overlaps (Bullet's
    /// `sortMinUp`; removal is unconditional — `USE_OVERLAP_TEST_ON_REMOVES`
    /// is off — because an axis crossing implies separation).
    fn sort_min_up(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        let moving = self.edges[k][i];
        while self.edges[k][i + 1].handle != SENTINEL_HANDLE
            && moving.pos >= self.edges[k][i + 1].pos
        {
            let next = self.edges[k][i + 1];
            if next.is_max() && update_overlaps {
                self.pair_remove(pair_key(moving.handle as usize, next.handle as usize));
            }
            if next.is_max() {
                self.max_edges[next.handle as usize][k] = i as u32;
            } else {
                self.min_edges[next.handle as usize][k] = i as u32;
            }
            self.edges[k][i + 1] = moving;
            self.edges[k][i] = next;
            i += 1;
        }
        self.min_edges[moving.handle as usize][k] = i as u32;
    }

    /// A max edge bubbling down can only *remove* overlaps (Bullet's
    /// `sortMaxDown`).
    fn sort_max_down(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        let moving = self.edges[k][i];
        while i > 0 {
            let prev = self.edges[k][i - 1];
            if moving.pos >= prev.pos {
                break;
            }
            if !prev.is_max() && update_overlaps {
                self.pair_remove(pair_key(moving.handle as usize, prev.handle as usize));
            }
            if prev.is_max() {
                self.max_edges[prev.handle as usize][k] = i as u32;
            } else {
                self.min_edges[prev.handle as usize][k] = i as u32;
            }
            self.edges[k][i] = prev;
            self.edges[k][i - 1] = moving;
            i -= 1;
        }
        self.max_edges[moving.handle as usize][k] = i as u32;
    }

    /// A max edge bubbling up can only *add* overlaps (Bullet's `sortMaxUp`:
    /// crossing another proxy's min edge with the 2D pretest).
    fn sort_max_up(&mut self, k: usize, mut i: usize, update_overlaps: bool) {
        let moving = self.edges[k][i];
        let axis1 = (k + 1) % 3;
        let axis2 = (k + 2) % 3;
        while self.edges[k][i + 1].handle != SENTINEL_HANDLE
            && moving.pos >= self.edges[k][i + 1].pos
        {
            let next = self.edges[k][i + 1];
            if !next.is_max() && update_overlaps
                && self.test_overlap_2d(moving.handle as usize, next.handle as usize, axis1, axis2)
            {
                self.pair_insert(pair_key(moving.handle as usize, next.handle as usize));
            }
            if next.is_max() {
                self.max_edges[next.handle as usize][k] = i as u32;
            } else {
                self.min_edges[next.handle as usize][k] = i as u32;
            }
            self.edges[k][i + 1] = moving;
            self.edges[k][i] = next;
            i += 1;
        }
        self.max_edges[moving.handle as usize][k] = i as u32;
    }
}

impl BroadPhase for AxisSweep3 {
    fn create_proxy(&mut self, aabb: Aabb) -> usize {
        if let Some(i) = self.aabbs.iter().position(|h| h.is_none()) {
            self.aabbs[i] = Some(aabb);
            self.min_edges[i] = [0; 3];
            self.max_edges[i] = [0; 3];
            if self.needs_rebuild(&aabb) {
                self.rebuild(Some(&aabb));
            } else {
                self.insert_handle(i, aabb);
            }
            i
        } else {
            self.aabbs.push(Some(aabb));
            self.min_edges.push([0; 3]);
            self.max_edges.push([0; 3]);
            let slot = self.aabbs.len() - 1;
            if self.needs_rebuild(&aabb) {
                self.rebuild(Some(&aabb));
            } else {
                self.insert_handle(slot, aabb);
            }
            slot
        }
    }

    fn set_aabb(&mut self, proxy: usize, aabb: Aabb) {
        let Some(slot_aabb) = self.aabbs[proxy].as_mut() else {
            return;
        };
        *slot_aabb = aabb;
        if self.needs_rebuild(&aabb) {
            self.rebuild(None);
            return;
        }
        let (qmin, qmax) = (self.quantize(aabb.min, false), self.quantize(aabb.max, true));
        for k in 0..3 {
            let (emin, emax, old_min, old_max) = {
                let emin = self.min_edges[proxy][k] as usize;
                let emax = self.max_edges[proxy][k] as usize;
                (emin, emax, self.edges[k][emin].pos, self.edges[k][emax].pos)
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
            let (imin, imax) = (
                self.min_edges[proxy][k] as usize,
                self.max_edges[proxy][k] as usize,
            );
            let (lo, hi) = (imin.min(imax), imax.max(imin));
            self.edges[k].remove(hi);
            self.edges[k].remove(lo);
            for edges in [&mut self.min_edges, &mut self.max_edges] {
                for e in edges.iter_mut() {
                    if e[k] as usize > hi {
                        e[k] -= 2;
                    } else if e[k] as usize > lo {
                        e[k] -= 1;
                    }
                }
            }
        }
        self.aabbs[proxy] = None;
    }

    fn proxy_aabb(&self, proxy: usize) -> Option<Aabb> {
        self.aabbs[proxy]
    }

    fn compute_pairs(&mut self) -> Vec<(usize, usize)> {
        // The cached set is quantized-conservative; the exact stored AABBs
        // bring the output back to brute-force-exact semantics. The set is
        // maintained in sorted order by the edge-swap pair updates, so this
        // is a single ordered walk — no per-step sort.
        self.pairs
            .iter()
            .copied()
            .filter(|&(a, b)| {
                matches!((self.aabbs[a], self.aabbs[b]),
                         (Some(x), Some(y)) if x.overlaps(&y))
            })
            .collect()
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
        self.aabbs.clone()
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

