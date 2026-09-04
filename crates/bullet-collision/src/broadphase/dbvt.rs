//! Dynamic bounding volume tree broad phase (Bullet's `btDbvt` +
//! `btDbvtBroadphase`, by Nathanael Presson).
//!
//! Two AABB trees — one for moving proxies, one for proxies that have been
//! resting for two steps — with lazily maintained *fat* leaf volumes: a
//! proxy whose exact AABB still fits inside its leaf's fat volume touches
//! the tree not at all. Insertion walks down greedily by center proximity
//! (`Select`), and a cheap per-step rotation pass (`optimizeIncremental`)
//! counteracts degradation. Pairs are formed immediately when a leaf is
//! inserted or moves (`m_deferedcollide = false`), and
//! [`compute_pairs`](BroadPhase::compute_pairs) doubles as the per-step
//! tick: tree rotation, stage rotation, dynamic->fixed migration and
//! separated-pair pruning (the exact-filter counterpart of Bullet's
//! `calculateOverlappingPairs` cleanup pass).
//!
//! Unlike [`AxisSweep3`](super::AxisSweep3) there is no object-count
//! ceiling and no quantization frame, and insertion/removal is O(log n) —
//! the structure Bullet defaults to for arbitrary scenes.

use bullet_math::{Aabb, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::{segment_hits_aabb, BroadPhase};

/// Per-axis margin added to leaf volumes when they grow (Bullet's
/// `gDbvtMargin`).
const DBVT_MARGIN: Real = 0.05;
/// Motion prediction factor for leaf volumes (Bullet's `m_prediction`,
/// default 0: leaves only carry the fixed margin unless configured).
const PREDICTION: Real = 0.0;

/// Sentinel node/proxy index ("null pointer").
const NONE: u32 = u32::MAX;

/// Stage slots: two alternating dynamic stages plus the fixed set
/// (Bullet's `DYNAMIC_SET`/`FIXED_SET`/`STAGECOUNT`).
const STAGE_COUNT: usize = 2;
const FIXED_STAGE: u8 = 2;

/// A tree node: an AABB plus either two children or the owning proxy id
/// (Bullet's `btDbvtNode`; `isleaf` is `childs[1] == 0` there).
struct Node {
    volume: Aabb,
    parent: u32,
    kind: NodeKind,
}

enum NodeKind {
    Internal { children: [u32; 2] },
    Leaf { proxy: usize },
}

/// One dynamic-or-fixed AABB tree (Bullet's `btDbvt`), array-based: nodes
/// live in a slab with a free list and all references are slab indices.
/// Leaves carry an arbitrary `usize` payload — the broad phase stores proxy
/// ids, the soft bodies store node/face/cluster indices (Bullet's `m_ndbvt`/
/// `m_fdbvt`/`m_cdbvt` are plain `btDbvt`s too).
pub struct DbvtTree {
    nodes: Vec<Option<Node>>,
    free: Vec<u32>,
    root: u32,
    leaves: u32,
    /// Rotation path counter (Bullet's `m_opath`).
    opath: u32,
}

#[inline]
fn pair_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Bullet's `Select`: descend towards the child whose center is closer to
/// the inserted volume's center (L1 distance on `mi + mx`, the unhalved
/// center sum — the factor cancels in the comparison).
#[inline]
fn proximity(a: &Aabb, b: &Aabb) -> Real {
    let d = (a.min + a.max) - (b.min + b.max);
    d.x.abs() + d.y.abs() + d.z.abs()
}

impl DbvtTree {
    /// Leaf/node index meaning "null pointer".
    pub const NONE: u32 = u32::MAX;

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root == Self::NONE
    }

    /// Drop every leaf (Bullet's `btDbvt::clear`); the node slab keeps its
    /// capacity for the next inserts.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.root = Self::NONE;
        self.leaves = 0;
    }

    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            root: NONE,
            leaves: 0,
            opath: 0,
        }
    }

    #[inline]
    fn is_leaf(&self, n: u32) -> bool {
        matches!(self.nodes[n as usize].as_ref().unwrap().kind, NodeKind::Leaf { .. })
    }

    #[inline]
    fn volume(&self, n: u32) -> Aabb {
        self.nodes[n as usize].as_ref().unwrap().volume
    }

    #[inline]
    fn parent(&self, n: u32) -> u32 {
        self.nodes[n as usize].as_ref().unwrap().parent
    }

    #[inline]
    fn children(&self, n: u32) -> [u32; 2] {
        match &self.nodes[n as usize].as_ref().unwrap().kind {
            NodeKind::Internal { children } => *children,
            NodeKind::Leaf { .. } => [NONE, NONE],
        }
    }

    /// Index of `n` in its parent's child slot (Bullet's `indexof`).
    #[inline]
    fn index_of(&self, n: u32) -> usize {
        let p = self.parent(n);
        (self.children(p)[1] == n) as usize
    }

    fn alloc(&mut self, parent: u32, volume: Aabb, kind: NodeKind) -> u32 {
        let node = Node { volume, parent, kind };
        if let Some(idx) = self.free.pop() {
            self.nodes[idx as usize] = Some(node);
            idx
        } else {
            self.nodes.push(Some(node));
            (self.nodes.len() - 1) as u32
        }
    }

    fn release(&mut self, n: u32) {
        self.nodes[n as usize] = None;
        self.free.push(n);
    }

    /// Bullet's `insertleaf`: descend from `root` (a subtree root) by
    /// proximity, splice a fresh internal node above the reached leaf, then
    /// re-merge ancestor volumes upward while they grow.
    fn insert_leaf(&mut self, root: u32, leaf: u32) {
        if self.root == NONE {
            self.root = leaf;
            self.nodes[leaf as usize].as_mut().unwrap().parent = NONE;
            return;
        }
        let mut node = root;
        if !self.is_leaf(node) {
            loop {
                let [c0, c1] = self.children(node);
                let (v0, v1) = (self.volume(c0), self.volume(c1));
                let leaf_volume = self.volume(leaf);
                node = if proximity(&leaf_volume, &v0) < proximity(&leaf_volume, &v1) {
                    c0
                } else {
                    c1
                };
                if self.is_leaf(node) {
                    break;
                }
            }
        }
        // Bullet computes `indexof(root)` while root's parent is still
        // `prev`; the slot must be captured before the parent rewrite.
        let prev = self.parent(node);
        let slot = if prev != NONE { self.index_of(node) } else { 0 };
        let merged = self.volume(node).merged(&self.volume(leaf));
        let internal = self.alloc(
            prev,
            merged,
            NodeKind::Internal { children: [node, leaf] },
        );
        self.nodes[node as usize].as_mut().unwrap().parent = internal;
        self.nodes[leaf as usize].as_mut().unwrap().parent = internal;
        if prev != NONE {
            self.nodes[prev as usize].as_mut().unwrap().kind = NodeKind::Internal {
                children: {
                    let mut c = self.children(prev);
                    c[slot] = internal;
                    c
                },
            };
        } else {
            self.root = internal;
        }
        // Shrink-test upward (Bullet: re-merge while the parent volume
        // does not already contain the child's).
        let mut child = internal;
        let mut p = prev;
        while p != NONE {
            let child_volume = self.volume(child);
            let p_volume = self.volume(p);
            if p_volume.contains(&child_volume) {
                break;
            }
            let [c0, c1] = self.children(p);
            let merged = self.volume(c0).merged(&self.volume(c1));
            self.nodes[p as usize].as_mut().unwrap().volume = merged;
            child = p;
            p = self.parent(p);
        }
    }

    /// Bullet's `removeleaf`: detach the leaf, let the sibling take the
    /// parent's place, re-merge ancestors while volumes change; returns the
    /// subtree root a re-insertion should start from (the shrink-test stop,
    /// or the tree root).
    fn remove_leaf(&mut self, leaf: u32) -> u32 {
        if self.root == leaf {
            self.root = NONE;
            self.nodes[leaf as usize].as_mut().unwrap().parent = NONE;
            return NONE;
        }
        let parent = self.parent(leaf);
        let prev = self.parent(parent);
        let sibling = self.children(parent)[1 - self.index_of(leaf)];
        if prev != NONE {
            let slot = self.index_of(parent);
            self.nodes[prev as usize].as_mut().unwrap().kind = NodeKind::Internal {
                children: {
                    let mut c = self.children(prev);
                    c[slot] = sibling;
                    c
                },
            };
            self.nodes[sibling as usize].as_mut().unwrap().parent = prev;
            self.release(parent);
            let mut node = prev;
            loop {
                let [c0, c1] = self.children(node);
                let merged = self.volume(c0).merged(&self.volume(c1));
                let old = self.volume(node);
                let stop = old == merged;
                self.nodes[node as usize].as_mut().unwrap().volume = merged;
                if stop {
                    break;
                }
                let up = self.parent(node);
                if up == NONE {
                    break;
                }
                node = up;
            }
            node
        } else {
            self.root = sibling;
            self.nodes[sibling as usize].as_mut().unwrap().parent = NONE;
            self.release(parent);
            self.root
        }
    }

    /// Bullet's `insert`.
    pub fn insert(&mut self, volume: Aabb, payload: usize) -> u32 {
        let leaf = self.alloc(NONE, volume, NodeKind::Leaf { proxy: payload });
        self.insert_leaf(self.root, leaf);
        self.leaves += 1;
        leaf
    }

    /// Bullet's `remove`.
    pub fn remove(&mut self, leaf: u32) {
        self.remove_leaf(leaf);
        self.release(leaf);
        self.leaves -= 1;
    }

    /// Bullet's `update(leaf, volume)`: detach, rewrite the volume, reinsert.
    /// `lookahead < 0` reinserts from the tree root; otherwise from `lookahead`
    /// levels above the shrink-test stop (`optimizeIncremental` uses 2).
    fn update(&mut self, leaf: u32, volume: Aabb, lookahead: i32) {
        let mut root = self.remove_leaf(leaf);
        if root != NONE {
            if lookahead >= 0 {
                for _ in 0..lookahead {
                    if self.parent(root) != NONE {
                        root = self.parent(root);
                    } else {
                        break;
                    }
                }
            } else {
                root = self.root;
            }
        }
        self.nodes[leaf as usize].as_mut().unwrap().volume = volume;
        self.insert_leaf(root, leaf);
    }

    /// Bullet's `update(leaf, volume, velocity, margin)`: a no-op while the
    /// fat leaf volume still contains the exact volume; otherwise grow the
    /// volume by `margin` and along `velocity` and reinsert. Returns whether
    /// the tree changed (the caller then re-collides the leaf).
    pub fn update_fat(
        &mut self,
        leaf: u32,
        volume: Aabb,
        velocity: Vector3,
        margin: Real,
    ) -> bool {
        if self.volume(leaf).contains(&volume) {
            return false;
        }
        let grown = volume
            .expanded_vec(Vector3::splat(margin))
            .signed_expanded(velocity);
        self.update(leaf, grown, -1);
        true
    }

    /// Bullet's plain `update(leaf, volume)`: reinsert the leaf at its exact
    /// new volume (the soft-body `transform`/`scale` path — no fat-volume
    /// caching, unlike the per-step `update_fat`).
    pub fn update_exact(&mut self, leaf: u32, volume: Aabb) {
        self.update(leaf, volume, -1);
    }

    /// Bullet's `sort` rotation for `optimizeIncremental`: if the child was
    /// allocated after its parent (slab-index order stands in for pointer
    /// order), swap their roles to keep the tree shallow. Returns the node
    /// to continue descending from.
    fn sort_rotate(&mut self, n: u32) -> u32 {
        let p = self.parent(n);
        if p == NONE || p < n {
            return n;
        }
        let i = self.index_of(n);
        let j = 1 - i;
        let s = self.children(p)[j];
        let q = self.parent(p);
        if q != NONE {
            let slot = self.index_of(p);
            self.nodes[q as usize].as_mut().unwrap().kind = NodeKind::Internal {
                children: {
                    let mut c = self.children(q);
                    c[slot] = n;
                    c
                },
            };
        } else {
            self.root = n;
        }
        self.nodes[s as usize].as_mut().unwrap().parent = n;
        self.nodes[p as usize].as_mut().unwrap().parent = n;
        self.nodes[n as usize].as_mut().unwrap().parent = q;
        let [n0, n1] = self.children(n);
        self.nodes[p as usize].as_mut().unwrap().kind =
            NodeKind::Internal { children: [n0, n1] };
        self.nodes[n0 as usize].as_mut().unwrap().parent = p;
        self.nodes[n1 as usize].as_mut().unwrap().parent = p;
        let (pn, nn) = {
            let p_volume = self.volume(p);
            let n_volume = self.volume(n);
            (n_volume, p_volume)
        };
        self.nodes[p as usize].as_mut().unwrap().volume = pn;
        self.nodes[n as usize].as_mut().unwrap().volume = nn;
        self.nodes[n as usize].as_mut().unwrap().kind = NodeKind::Internal {
            children: {
                let mut c = self.children(n);
                c[i] = p;
                c[j] = s;
                c
            },
        };
        p
    }

    /// Bullet's `optimizeIncremental`: walk a pseudo-random path to a leaf,
    /// rotating on the way, and reinsert that leaf two levels up.
    pub fn optimize_incremental(&mut self, passes: u32) {
        if self.root == NONE || passes == 0 {
            return;
        }
        for _ in 0..passes {
            let mut node = self.root;
            let mut bit = 0u32;
            while !self.is_leaf(node) {
                node = self.sort_rotate(node);
                let pick = ((self.opath >> bit) & 1) as usize;
                node = self.children(node)[pick];
                bit = (bit + 1) & 31;
            }
            let volume = self.volume(node);
            self.update(node, volume, 2);
            self.opath = self.opath.wrapping_add(1);
        }
    }

    /// Bullet's `collideTV`: visit every leaf whose volume overlaps
    /// `volume` and report its payload.
    pub fn for_each_overlapping(&self, volume: &Aabb, out: &mut Vec<usize>) {
        self.walk_overlapping(self.root, volume, out);
    }

    fn walk_overlapping(&self, n: u32, volume: &Aabb, out: &mut Vec<usize>) {
        if n == NONE || !self.volume(n).overlaps(volume) {
            return;
        }
        match &self.nodes[n as usize].as_ref().unwrap().kind {
            NodeKind::Leaf { proxy } => out.push(*proxy),
            // Bullet's `collideTV` pushes childs[0] then childs[1] on an
            // explicit stack, so childs[1] pops first — the right subtree is
            // visited before the left (leaf order matters to consumers that
            // append in visit order, e.g. the soft body's rcontacts).
            NodeKind::Internal { children } => {
                self.walk_overlapping(children[1], volume, out);
                self.walk_overlapping(children[0], volume, out);
            }
        }
    }

    /// Bullet's `collideTT`: visit every leaf pair `(leaf-of-self,
    /// leaf-of-other)` whose volumes overlap, in Bullet's stack order (each
    /// internal pair pushes `(a0,b0), (a1,b0), (a0,b1), (a1,b1)` so `(a1,b1)`
    /// pops first). When both arguments are the same tree (soft-body
    /// self-collision), Bullet's `a == b` branch expands same-node pairs
    /// into `(c0,c0), (c1,c1), (c0,c1)` — the mirrored `(c1,c0)` is
    /// skipped, so each unordered leaf pair reports exactly once.
    pub fn for_each_overlapping_pair(&self, other: &DbvtTree, out: &mut Vec<(usize, usize)>) {
        let same_tree = core::ptr::eq(self, other);
        let mut stack: Vec<[u32; 2]> = Vec::new();
        if self.root != NONE && other.root != NONE {
            stack.push([self.root, other.root]);
        }
        while let Some([a, b]) = stack.pop() {
            if same_tree && a == b {
                if let NodeKind::Internal { children } =
                    &self.nodes[a as usize].as_ref().unwrap().kind
                {
                    stack.push([children[0], children[0]]);
                    stack.push([children[1], children[1]]);
                    stack.push([children[0], children[1]]);
                }
                continue;
            }
            if !self.volume(a).overlaps(&other.volume(b)) {
                continue;
            }
            let a_children = self.children(a);
            let b_children = other.children(b);
            match (self.is_leaf(a), other.is_leaf(b)) {
                (false, false) => {
                    stack.push([a_children[0], b_children[0]]);
                    stack.push([a_children[1], b_children[0]]);
                    stack.push([a_children[0], b_children[1]]);
                    stack.push([a_children[1], b_children[1]]);
                }
                (false, true) => {
                    stack.push([a_children[0], b]);
                    stack.push([a_children[1], b]);
                }
                (true, false) => {
                    stack.push([a, b_children[0]]);
                    stack.push([a, b_children[1]]);
                }
                (true, true) => out.push((self.leaf_payload(a), other.leaf_payload(b))),
            }
        }
    }

    /// Payload of a leaf node.
    #[inline]
    fn leaf_payload(&self, n: u32) -> usize {
        match &self.nodes[n as usize].as_ref().unwrap().kind {
            NodeKind::Leaf { proxy } => *proxy,
            NodeKind::Internal { .. } => unreachable!("leaf_payload on internal node"),
        }
    }

    /// Collect proxies whose leaf volume the segment crosses (slab test on
    /// node volumes, `t` clamped to `[0, 1]` like the shared helper).
    fn for_each_on_segment(&self, from: Vector3, to: Vector3, out: &mut Vec<usize>) {
        self.walk_segment(self.root, from, to, out);
    }

    fn walk_segment(&self, n: u32, from: Vector3, to: Vector3, out: &mut Vec<usize>) {
        if n == NONE {
            return;
        }
        let volume = self.volume(n);
        if !segment_hits_aabb(&volume, from, to) {
            return;
        }
        match &self.nodes[n as usize].as_ref().unwrap().kind {
            NodeKind::Leaf { proxy } => out.push(*proxy),
            NodeKind::Internal { children } => {
                self.walk_segment(children[0], from, to, out);
                self.walk_segment(children[1], from, to, out);
            }
        }
    }
}

impl Default for DbvtTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Broad-phase proxy (Bullet's `btDbvtProxy`): the exact (world-inflated)
/// AABB, the owning leaf, the stage slot, and the intrusive stage-list
/// links.
struct Proxy {
    aabb: Aabb,
    leaf: u32,
    stage: u8,
    /// Intrusive doubly-linked stage list: `[prev, next]`, proxy ids.
    links: [u32; 2],
}

/// Dynamic-BVH broad phase (Bullet's `btDbvtBroadphase`).
///
/// Output order matches [`BruteForceBroadPhase`](super::BruteForceBroadPhase)
/// for identical inputs (lexicographic on `(i, j)`), so tests can compare
/// directly. `compute_pairs` must be called once per step: besides the
/// exact-filtered pair output it advances the per-step tick (rotation,
/// stage rotation, dynamic->fixed migration, separated-pair pruning).
pub struct DbvtBroadPhase {
    proxies: Vec<Option<Proxy>>,
    /// Moving proxies (Bullet's `m_sets[DYNAMIC_SET]`).
    dynamic: DbvtTree,
    /// Proxies resting for two consecutive steps (Bullet's
    /// `m_sets[FIXED_SET]`); any `set_aabb` moves them back.
    fixed: DbvtTree,
    /// Stage list heads (proxy ids), `[stage0, stage1, fixed]`.
    stage_roots: [u32; STAGE_COUNT + 1],
    stage_current: u8,
    /// Steps of fixed-tree rotation budget left (Bullet's `m_fixedleft`).
    fixed_left: u32,
    /// Sorted pair set, maintained by the immediate collide passes; see
    /// [`AxisSweep3`](super::AxisSweep3) for the sorted-Vec rationale.
    pairs: Vec<(usize, usize)>,
    /// Reused collector for the immediate collide passes.
    collide_scratch: Vec<usize>,
}

impl Default for DbvtBroadPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for DbvtBroadPhase {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DbvtBroadPhase")
            .field("dynamic_leaves", &self.dynamic.leaves)
            .field("fixed_leaves", &self.fixed.leaves)
            .finish_non_exhaustive()
    }
}

impl DbvtBroadPhase {
    pub fn new() -> Self {
        Self {
            proxies: Vec::new(),
            dynamic: DbvtTree::new(),
            fixed: DbvtTree::new(),
            stage_roots: [NONE; STAGE_COUNT + 1],
            stage_current: 0,
            fixed_left: 0,
            pairs: Vec::new(),
            collide_scratch: Vec::new(),
        }
    }

    /// Insert into the sorted pair set (binary search, like the sweep's).
    fn pair_insert(&mut self, key: (usize, usize)) {
        if let Err(i) = self.pairs.binary_search(&key) {
            self.pairs.insert(i, key);
        }
    }

    /// Bullet's `listappend` on the stage lists.
    fn list_append(&mut self, proxy: usize, stage: u8) {
        let root = self.stage_roots[stage as usize];
        {
            let p = self.proxies[proxy].as_mut().unwrap();
            p.links = [NONE, root];
            p.stage = stage;
        }
        if root != NONE {
            self.proxies[root as usize].as_mut().unwrap().links[0] = proxy as u32;
        }
        self.stage_roots[stage as usize] = proxy as u32;
    }

    /// Bullet's `listremove` on the stage lists.
    fn list_remove(&mut self, proxy: usize) {
        let stage = self.proxies[proxy].as_ref().unwrap().stage;
        let [prev, next] = self.proxies[proxy].as_ref().unwrap().links;
        if prev != NONE {
            self.proxies[prev as usize].as_mut().unwrap().links[1] = next;
        } else {
            self.stage_roots[stage as usize] = next;
        }
        if next != NONE {
            self.proxies[next as usize].as_mut().unwrap().links[0] = prev;
        }
    }

    /// Bullet's immediate collide (`m_deferedcollide = false`): form pairs
    /// between `proxy`'s leaf volume and every leaf of both trees.
    fn collide_leaf(&mut self, proxy: usize) {
        let leaf = self.proxies[proxy].as_ref().unwrap().leaf;
        let volume = self.dynamic.volume(leaf);
        self.collide_scratch.clear();
        self.dynamic.for_each_overlapping(&volume, &mut self.collide_scratch);
        self.fixed.for_each_overlapping(&volume, &mut self.collide_scratch);
        let hits = core::mem::take(&mut self.collide_scratch);
        for other in hits {
            if other != proxy {
                self.pair_insert(pair_key(proxy, other));
            }
        }
    }

    /// The per-step tick inside `compute_pairs` (Bullet's `collide()`):
    /// incremental rotation on both trees, stage rotation, migration of
    /// proxies that skipped a full stage cycle (resting for two steps) into
    /// the fixed tree.
    fn tick(&mut self) {
        self.dynamic.optimize_incremental(1);
        if self.fixed_left > 0 {
            self.fixed.optimize_incremental(1);
            self.fixed_left -= 1;
        }
        self.stage_current ^= 1;
        let stage = self.stage_current;
        let mut cur = self.stage_roots[stage as usize];
        let mut migrated = false;
        while cur != NONE {
            let proxy = cur as usize;
            let next = self.proxies[proxy].as_ref().unwrap().links[1];
            let (leaf, aabb) = {
                let p = self.proxies[proxy].as_ref().unwrap();
                (p.leaf, p.aabb)
            };
            self.list_remove(proxy);
            self.list_append(proxy, FIXED_STAGE);
            self.dynamic.remove(leaf);
            let fixed_leaf = self.fixed.insert(aabb, proxy);
            self.proxies[proxy].as_mut().unwrap().leaf = fixed_leaf;
            migrated = true;
            cur = next;
        }
        if migrated {
            self.fixed_left = self.fixed.leaves;
        }
    }
}

impl BroadPhase for DbvtBroadPhase {
    fn create_proxy(&mut self, aabb: Aabb) -> usize {
        let slot = if let Some(i) = self.proxies.iter().position(|p| p.is_none()) {
            i
        } else {
            self.proxies.push(None);
            self.proxies.len() - 1
        };
        let leaf = self.dynamic.insert(aabb, slot);
        self.proxies[slot] = Some(Proxy {
            aabb,
            leaf,
            stage: self.stage_current,
            links: [NONE, NONE],
        });
        let stage = self.stage_current;
        self.list_append(slot, stage);
        self.collide_leaf(slot);
        slot
    }

    fn set_aabb(&mut self, proxy: usize, aabb: Aabb) {
        let Some(p) = self.proxies[proxy].as_ref() else {
            return;
        };
        let (old_aabb, leaf, was_fixed) = (p.aabb, p.leaf, p.stage == FIXED_STAGE);
        let docollide;
        if was_fixed {
            // Fixed -> dynamic set.
            self.fixed.remove(leaf);
            let dynamic_leaf = self.dynamic.insert(aabb, proxy);
            let p = self.proxies[proxy].as_mut().unwrap();
            p.leaf = dynamic_leaf;
            docollide = true;
        } else {
            let fat = self.dynamic.volume(leaf);
            if fat.overlaps(&aabb) {
                // Moving: predictively inflate along the observed motion
                // (Bullet's velocity = half extents * m_prediction, signed
                // by the min-corner delta); the tree update is skipped
                // entirely while the fat volume still contains the AABB.
                let half = aabb.extents() * PREDICTION;
                let delta = aabb.min - old_aabb.min;
                let velocity = Vector3::new(
                    if delta.x < 0.0 { -half.x } else { half.x },
                    if delta.y < 0.0 { -half.y } else { half.y },
                    if delta.z < 0.0 { -half.z } else { half.z },
                );
                docollide = self
                    .dynamic
                    .update_fat(leaf, aabb, velocity, DBVT_MARGIN);
            } else {
                // Teleporting: reinsert at the exact volume, no margin.
                self.dynamic.update(leaf, aabb, -1);
                docollide = true;
            }
        }
        self.list_remove(proxy);
        self.list_append(proxy, self.stage_current);
        self.proxies[proxy].as_mut().unwrap().aabb = aabb;
        if docollide {
            self.collide_leaf(proxy);
        }
    }

    fn destroy_proxy(&mut self, proxy: usize) {
        let (leaf, stage) = {
            let p = self.proxies[proxy].as_ref().unwrap();
            (p.leaf, p.stage)
        };
        if stage == FIXED_STAGE {
            self.fixed.remove(leaf);
        } else {
            self.dynamic.remove(leaf);
        }
        self.list_remove(proxy);
        self.proxies[proxy] = None;
        self.pairs.retain(|&(a, b)| a != proxy && b != proxy);
    }

    fn proxy_aabb(&self, proxy: usize) -> Option<Aabb> {
        self.proxies[proxy].as_ref().map(|p| p.aabb)
    }

    fn compute_pairs(&mut self) -> Vec<(usize, usize)> {
        self.tick();
        // Exact-filtered output over the cached pairs; pairs whose exact
        // AABBs no longer overlap are pruned from the cache here (the
        // counterpart of Bullet's cleanup pass, which scans a slice of the
        // hash pair cache per step — the pair set is small enough that a
        // full single pass is cheaper than the bookkeeping).
        let mut out = Vec::with_capacity(self.pairs.len());
        let pairs = &mut self.pairs;
        let proxies = &self.proxies;
        let mut write = 0;
        for read in 0..pairs.len() {
            let key = pairs[read];
            let overlap = proxies[key.0]
                .as_ref()
                .unwrap()
                .aabb
                .overlaps(&proxies[key.1].as_ref().unwrap().aabb);
            if overlap {
                pairs[write] = key;
                write += 1;
                out.push(key);
            }
        }
        pairs.truncate(write);
        out
    }

    fn raycast_candidates(&self, ray_from: Vector3, ray_to: Vector3) -> Vec<usize> {
        let mut out = Vec::new();
        self.dynamic.for_each_on_segment(ray_from, ray_to, &mut out);
        self.fixed.for_each_on_segment(ray_from, ray_to, &mut out);
        out.sort_unstable();
        out
    }

    fn aabb_candidates(&self, aabb: Aabb) -> Vec<usize> {
        let mut out = Vec::new();
        self.dynamic.for_each_overlapping(&aabb, &mut out);
        self.fixed.for_each_overlapping(&aabb, &mut out);
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_math::{Real, Vector3};

    /// Deterministic pseudo-random boxes (the axis-sweep test recipe).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn aabb(&mut self, spread: u64, size: u64) -> Aabb {
            let base = Vector3::new(
                ((self.next() >> 32) % spread) as Real / 10.0,
                ((self.next() >> 16) % spread) as Real / 10.0,
                (self.next() % spread) as Real / 10.0,
            );
            let extent = Vector3::new(
                ((self.next() >> 32) % size) as Real / 100.0 + 0.1,
                ((self.next() >> 16) % size) as Real / 100.0 + 0.1,
                (self.next() % size) as Real / 100.0 + 0.1,
            );
            Aabb::new(base, base + extent)
        }
    }

    #[test]
    fn finds_overlapping_pairs() {
        let mut bp = DbvtBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::splat(0.0), Vector3::splat(1.0)));
        let b = bp.create_proxy(Aabb::new(Vector3::splat(0.5), Vector3::splat(1.5)));
        let c = bp.create_proxy(Aabb::new(Vector3::splat(10.0), Vector3::splat(11.0)));
        let pairs = bp.compute_pairs();
        assert_eq!(pairs, vec![(a, b)]);
        assert!(pairs.iter().all(|&(x, y)| x != c && y != c));
    }

    #[test]
    fn proxy_slot_reuse() {
        let mut bp = DbvtBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        bp.destroy_proxy(a);
        let b = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        assert_eq!(a, b);
    }

    #[test]
    fn moved_proxies_are_recomputed() {
        let mut bp = DbvtBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)));
        assert!(bp.compute_pairs().is_empty());
        bp.set_aabb(b, Aabb::new(Vector3::new(0.5, 0.0, 0.0), Vector3::new(1.5, 1.0, 1.0)));
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
    }

    /// A single-axis overlap is not enough: pairs must overlap on all axes.
    #[test]
    fn requires_overlap_on_all_axes() {
        let mut bp = DbvtBroadPhase::new();
        bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0)));
        bp.create_proxy(Aabb::new(Vector3::new(0.5, 0.5, 5.0), Vector3::new(1.5, 1.5, 6.0)));
        assert!(bp.compute_pairs().is_empty());
    }

    #[test]
    fn matches_brute_force_on_scattered_boxes() {
        let mut dbvt = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for _ in 0..64 {
            let aabb = rng.aabb(1000, 100);
            dbvt.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        assert_eq!(dbvt.compute_pairs(), brute.compute_pairs());
    }

    /// Incremental updates (teleports, destroys, recreations and the
    /// stage/fixed migration ticks in between) must stay equivalent to the
    /// brute-force baseline.
    #[test]
    fn incremental_updates_match_brute_force() {
        let mut dbvt = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut rng = Rng(0x243f6a8885a308d3);
        let mut aabbs = Vec::new();
        for _ in 0..32 {
            let aabb = rng.aabb(500, 100);
            dbvt.create_proxy(aabb);
            brute.create_proxy(aabb);
            aabbs.push(aabb);
        }
        assert_eq!(dbvt.compute_pairs(), brute.compute_pairs());
        for (i, aabb) in aabbs.iter_mut().enumerate() {
            if i % 2 == 0 {
                *aabb = rng.aabb(500, 100);
                dbvt.set_aabb(i, *aabb);
                brute.set_aabb(i, *aabb);
            } else {
                dbvt.destroy_proxy(i);
                brute.destroy_proxy(i);
            }
        }
        assert_eq!(dbvt.compute_pairs(), brute.compute_pairs());
        // A few tick cycles migrate the untouched proxies into the fixed
        // tree; output must be unaffected.
        for _ in 0..3 {
            assert_eq!(dbvt.compute_pairs(), brute.compute_pairs());
        }
        // Re-create into the freed slots and teleport again (fixed ->
        // dynamic migration path).
        for i in (0..aabbs.len()).filter(|i| i % 2 == 1) {
            let aabb = rng.aabb(500, 100);
            assert_eq!(dbvt.create_proxy(aabb), i);
            assert_eq!(brute.create_proxy(aabb), i);
            aabbs[i] = aabb;
        }
        assert_eq!(dbvt.compute_pairs(), brute.compute_pairs());
    }

    /// Sub-margin jitter must not touch the tree (the fat-leaf containment
    /// fast path) and must not change the reported pairs.
    #[test]
    fn fat_volume_absorbs_small_moves() {
        let mut bp = DbvtBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(0.5, 0.9, 0.5), Vector3::new(1.5, 1.9, 1.5)));
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
        let jitter = Aabb::new(
            Vector3::new(0.5, 0.9 + 1e-3, 0.5),
            Vector3::new(1.5, 1.9 + 1e-3, 1.5),
        );
        bp.set_aabb(b, jitter);
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
        assert_eq!(bp.proxy_aabb(b), Some(jitter));
    }

    /// Teleporting far past the fat volume must rebuild the leaf at the
    /// exact position and re-form the pairs there.
    #[test]
    fn teleport_far_matches_brute_force() {
        let mut dbvt = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut rng = Rng(0x13198a2e03707344);
        let mut aabbs = Vec::new();
        for _ in 0..16 {
            let aabb = rng.aabb(200, 100);
            dbvt.create_proxy(aabb);
            brute.create_proxy(aabb);
            aabbs.push(aabb);
        }
        for (i, slot) in aabbs.iter_mut().enumerate() {
            let aabb = rng.aabb(5000, 100);
            *slot = aabb;
            dbvt.set_aabb(i, aabb);
            brute.set_aabb(i, aabb);
            assert_eq!(dbvt.compute_pairs(), brute.compute_pairs(), "after teleport {i}");
        }
    }

    /// Proxies left alone for two tick cycles migrate into the fixed tree;
    /// a subsequent set_aabb moves them back. Both transitions keep the
    /// output exact.
    #[test]
    fn resting_proxies_migrate_to_fixed_and_back() {
        let mut bp = DbvtBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        let b = bp.create_proxy(Aabb::new(Vector3::new(0.5, 0.0, 0.5), Vector3::new(1.5, 1.0, 1.5)));
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
        // Two ticks without any set_aabb: both proxies rest in the fixed
        // tree now (create_proxy parks them in stage 0, two rotations
        // later the stage list sweeps them).
        bp.compute_pairs();
        let pairs = bp.compute_pairs();
        assert_eq!(pairs, vec![(a, b)]);
        assert_eq!(bp.fixed.leaves, 2, "resting proxies should live in the fixed tree");
        assert_eq!(bp.dynamic.leaves, 0);
        // Wake one: it moves back to the dynamic tree, pair output intact.
        bp.set_aabb(a, Aabb::new(Vector3::ZERO, Vector3::new(1.0, 1.0, 1.0)));
        assert_eq!(bp.dynamic.leaves, 1);
        assert_eq!(bp.fixed.leaves, 1);
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
        // And it settles back into the fixed tree.
        for _ in 0..3 {
            bp.compute_pairs();
        }
        assert_eq!(bp.fixed.leaves, 2);
        assert_eq!(bp.compute_pairs(), vec![(a, b)]);
    }

    /// Destroying a proxy must drop exactly its pairs, including pairs
    /// formed against the fixed tree.
    #[test]
    fn destroy_removes_pairs_including_fixed_side() {
        let mut bp = DbvtBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        bp.create_proxy(Aabb::new(Vector3::splat(0.5), Vector3::splat(1.5)));
        bp.compute_pairs();
        bp.compute_pairs();
        bp.destroy_proxy(a);
        assert!(bp.compute_pairs().is_empty());
    }

    #[test]
    fn raycast_candidates_filters_by_ray_path() {
        let mut bp = DbvtBroadPhase::new();
        let hit = bp.create_proxy(Aabb::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(2.0, 2.0, 2.0)));
        let miss = bp.create_proxy(Aabb::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(11.0, 11.0, 11.0)));
        let candidates = bp.raycast_candidates(Vector3::ZERO, Vector3::new(1.5, 1.5, 1.5));
        assert_eq!(candidates, vec![hit]);
        let inside = bp.raycast_candidates(Vector3::new(1.5, 1.5, 1.5), Vector3::new(4.0, 4.0, 4.0));
        assert!(inside.contains(&hit) && !inside.contains(&miss));
        let behind = bp.raycast_candidates(Vector3::new(3.0, 3.0, 3.0), Vector3::new(4.0, 4.0, 4.0));
        assert!(behind.is_empty());
    }

    #[test]
    fn raycast_candidates_matches_brute_force() {
        let mut dbvt = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut rng = Rng(0xdeadbeefcafef00d);
        for _ in 0..48 {
            let aabb = rng.aabb(400, 100);
            dbvt.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        // Also sweep a few proxies into the fixed tree before querying.
        dbvt.compute_pairs();
        dbvt.compute_pairs();
        let from = Vector3::new(5.0, 6.0, 7.0);
        let to = Vector3::new(25.0, 2.0, 30.0);
        assert_eq!(dbvt.raycast_candidates(from, to), brute.raycast_candidates(from, to));
    }

    #[test]
    fn aabb_candidates_matches_brute_force() {
        let mut dbvt = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut rng = Rng(0x5deece66d96b84b5);
        for _ in 0..48 {
            let aabb = rng.aabb(400, 100);
            dbvt.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        dbvt.compute_pairs();
        dbvt.compute_pairs();
        let queries = [
            Aabb::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(20.0, 20.0, 20.0)),
            Aabb::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(12.0, 12.0, 12.0)),
            Aabb::new(Vector3::new(50.0, 50.0, 50.0), Vector3::new(60.0, 60.0, 60.0)),
            Aabb::new(Vector3::new(39.9, 0.0, 0.0), Vector3::new(40.0, 40.0, 40.0)),
        ];
        for q in queries {
            assert_eq!(dbvt.aabb_candidates(q), brute.aabb_candidates(q), "query {q:?}");
        }
    }
}
