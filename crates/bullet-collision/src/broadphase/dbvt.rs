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
    /// levels above the shrink-test stop. Bullet's `optimizeIncremental`
    /// reinserts from the root (its single-argument `update` has
    /// `lookahead = -1`); our reinsertion two levels up is a local choice
    /// (see the audit doc, deviation table).
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
    /// rotating on the way (`sort`, slab-index order standing in for
    /// Bullet's pointer order), then reinsert that leaf. Bullet reinserts
    /// from the tree root (`update(node)` with `lookahead = -1`); we
    /// reinsert two levels above the shrink-test stop — a local-tree-shape
    /// choice, output-neutral (the oracle tests pin the tree invariants).
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
        // The cache lives at fat-overlap granularity: a pair stays cached
        // while its leaf volumes overlap even when the exact AABBs are
        // separated, because within-fat motion skips both the tree update
        // and the collide pass, so nothing would re-insert the pair once
        // the exact AABBs close the gap again (Bullet's cleanup pass
        // likewise only drops pairs outside the margin). The reported
        // output is the exact-filtered subset.
        let mut out = Vec::with_capacity(self.pairs.len());
        let pairs = &mut self.pairs;
        let proxies = &self.proxies;
        let (dynamic, fixed) = (&self.dynamic, &self.fixed);
        let mut write = 0;
        for read in 0..pairs.len() {
            let key = pairs[read];
            let p0 = proxies[key.0].as_ref().unwrap();
            let p1 = proxies[key.1].as_ref().unwrap();
            let v0 = if p0.stage == FIXED_STAGE {
                fixed.volume(p0.leaf)
            } else {
                dynamic.volume(p0.leaf)
            };
            let v1 = if p1.stage == FIXED_STAGE {
                fixed.volume(p1.leaf)
            } else {
                dynamic.volume(p1.leaf)
            };
            if v0.overlaps(&v1) {
                pairs[write] = key;
                write += 1;
                if p0.aabb.overlaps(&p1.aabb) {
                    out.push(key);
                }
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

    // --------------------------------------------------------------
    // Adversarial validation: the cached fat-overlap pair set must always
    // be a superset of the exact overlapping pairs (compute_pairs' exact
    // filter then reproduces brute force after every state transition),
    // and the trees must keep their structural invariants through every
    // operation.
    // --------------------------------------------------------------

    impl Rng {
        /// Uniform integer in `[0, n)`.
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        /// Uniform `Real` in `[0, 1)`.
        fn unit(&mut self) -> Real {
            (self.next() >> 11) as Real / (1u64 << 53) as Real
        }

        /// Uniform `Real` in `[lo, hi)`.
        fn range(&mut self, lo: Real, hi: Real) -> Real {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Structural checker for a `DbvtTree`: root/leaf bookkeeping,
    /// parent-child pointer consistency, live/dead slot and free-list
    /// agreement, and payload uniqueness.
    fn check_tree_invariants(tree: &DbvtTree) {
        let none = DbvtTree::NONE;
        if tree.root == none {
            assert_eq!(tree.leaves, 0, "empty tree must have no leaves");
            // Free slots may linger (the last removed leaf's own slot).
            assert!(tree.nodes.iter().all(Option::is_none), "live nodes in an empty tree");
            return;
        }
        assert_ne!(tree.leaves, 0);
        assert_eq!(tree.parent(tree.root), none, "root must have no parent");
        let mut seen = vec![false; tree.nodes.len()];
        let mut live_leaf_count = 0u32;
        let mut payloads = Vec::new();
        let mut stack = vec![tree.root];
        while let Some(n) = stack.pop() {
            assert!(n != none, "NONE child reference");
            assert!(!seen[n as usize], "node {n} reached twice");
            seen[n as usize] = true;
            let node = tree.nodes[n as usize]
                .as_ref()
                .unwrap_or_else(|| panic!("node {n} referenced but dead"));
            match &node.kind {
                NodeKind::Leaf { proxy } => {
                    live_leaf_count += 1;
                    payloads.push(*proxy);
                }
                NodeKind::Internal { children } => {
                    for &c in children {
                        assert!(c != none, "internal node with a NONE child");
                        assert!(
                            tree.nodes[c as usize].is_some(),
                            "child {c} referenced but dead"
                        );
                        assert_eq!(tree.parent(c), n, "child/parent pointer mismatch at {c}");
                    }
                    stack.extend(children);
                }
            }
        }
        assert_eq!(live_leaf_count, tree.leaves, "reachable leaf count != leaves counter");
        payloads.sort_unstable();
        let total = payloads.len();
        payloads.dedup();
        assert_eq!(payloads.len(), total, "duplicate leaf payloads");
        // Free list: dead slots only, no duplicates.
        let mut free_seen = vec![false; tree.nodes.len()];
        for &f in &tree.free {
            assert!(tree.nodes[f as usize].is_none(), "free list entry {f} is live");
            assert!(!free_seen[f as usize], "free list duplicate {f}");
            free_seen[f as usize] = true;
        }
        // No leaks: dead exactly when free-listed, live exactly when reachable.
        for (i, slot) in tree.nodes.iter().enumerate() {
            if seen[i] {
                assert!(slot.is_some(), "reachable node {i} is dead");
            } else {
                assert_eq!(slot.is_none(), free_seen[i], "slot {i} live/dead vs free-list");
            }
        }
    }

    /// All `(payload, volume)` leaves, any order.
    fn collect_leaves(tree: &DbvtTree) -> Vec<(usize, Aabb)> {
        let mut out = Vec::new();
        if tree.root != DbvtTree::NONE {
            let mut stack = vec![tree.root];
            while let Some(n) = stack.pop() {
                let node = tree.nodes[n as usize].as_ref().unwrap();
                match &node.kind {
                    NodeKind::Leaf { proxy } => out.push((*proxy, node.volume)),
                    NodeKind::Internal { children } => stack.extend(children),
                }
            }
        }
        out
    }

    /// Brute-force unordered overlapping-leaf-pair oracle over the stored
    /// (fat) volumes, sorted.
    fn brute_tree_pairs(tree: &DbvtTree) -> Vec<(usize, usize)> {
        let leaves = collect_leaves(tree);
        let mut out = Vec::new();
        for i in 0..leaves.len() {
            for j in i + 1..leaves.len() {
                if leaves[i].1.overlaps(&leaves[j].1) {
                    out.push(pair_key(leaves[i].0, leaves[j].0));
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// One interleaved state-machine step applied identically to both broad
    /// phases; returns how many extra tick (compute_pairs) cycles to run
    /// without movement afterwards.
    fn random_step(
        rng: &mut Rng,
        dbvt: &mut DbvtBroadPhase,
        brute: &mut super::super::BruteForceBroadPhase,
        state: &mut [Option<Aabb>],
        live: &mut Vec<usize>,
    ) -> u32 {
        let op = rng.below(100);
        if live.is_empty() || (live.len() < state.len() && op < 20) {
            let aabb = rng.aabb(300, 100);
            let s = dbvt.create_proxy(aabb);
            let b = brute.create_proxy(aabb);
            assert_eq!(s, b);
            state[s] = Some(aabb);
            live.push(s);
        } else if op < 55 {
            // Small move: scales below and above DBVT_MARGIN (0.05), so the
            // fat-containment skip, the margin growth and the teleport
            // reinsert paths all get exercised.
            let i = live[rng.below(live.len() as u64) as usize];
            let cur = state[i].unwrap();
            let scale = match rng.below(4) {
                0 => 0.01,
                1 => 0.04,
                2 => 0.2,
                _ => 2.0,
            };
            let d = Vector3::new(rng.range(-scale, scale), rng.range(-scale, scale), rng.range(-scale, scale));
            let next = Aabb::new(cur.min + d, cur.max + d);
            dbvt.set_aabb(i, next);
            brute.set_aabb(i, next);
            state[i] = Some(next);
        } else if op < 68 {
            // Teleport somewhere else in the volume.
            let i = live[rng.below(live.len() as u64) as usize];
            let cur = state[i].unwrap();
            let center = Vector3::new(rng.range(0.0, 30.0), rng.range(0.0, 30.0), rng.range(0.0, 30.0));
            let half = (cur.max - cur.min) * 0.5;
            let next = Aabb::new(center - half, center + half);
            dbvt.set_aabb(i, next);
            brute.set_aabb(i, next);
            state[i] = Some(next);
        } else if op < 74 {
            // Extra tick cycles with no movement (stage rotation and
            // dynamic -> fixed migration run; movement comes back later).
            return 2;
        } else if live.len() > 4 {
            let k = rng.below(live.len() as u64) as usize;
            let i = live.swap_remove(k);
            dbvt.destroy_proxy(i);
            brute.destroy_proxy(i);
            state[i] = None;
        } else {
            let aabb = rng.aabb(300, 100);
            let s = dbvt.create_proxy(aabb);
            let b = brute.create_proxy(aabb);
            assert_eq!(s, b);
            state[s] = Some(aabb);
            live.push(s);
        }
        0
    }

    /// Thousands of interleaved create/move/teleport/destroy/tick operations
    /// across several seeds; after every single operation the pair output
    /// must match brute force exactly, and both trees must keep their
    /// structural invariants (the empirical core of the superset-invariant
    /// validation, including the dynamic/fixed migration path since every
    /// op is followed by a tick).
    #[test]
    fn randomized_state_machine_matches_brute_force() {
        const SEEDS: u64 = 4;
        const OPS: usize = 2048;
        const POOL: usize = 32;
        for seed in 0..SEEDS {
            let mut rng = Rng(0xdb4e_7f4a_7c15_0001 ^ seed.wrapping_mul(0x517c_c1b7_2722_0a95));
            let mut dbvt = DbvtBroadPhase::new();
            let mut brute = super::super::BruteForceBroadPhase::new();
            let mut state = vec![None; POOL];
            let mut live = Vec::new();
            for op in 0..OPS {
                let extra_ticks = random_step(&mut rng, &mut dbvt, &mut brute, &mut state, &mut live);
                for _ in 0..=extra_ticks {
                    assert_eq!(
                        dbvt.compute_pairs(),
                        brute.compute_pairs(),
                        "seed {seed} diverged at op {op}"
                    );
                }
                check_tree_invariants(&dbvt.dynamic);
                check_tree_invariants(&dbvt.fixed);
            }
        }
    }

    /// Explicit fat-volume candidates: with fat volumes overlapping but
    /// exact AABBs separated, the cached pair is present (create-time
    /// collide) while compute_pairs filters it; moving one proxy within its
    /// existing fat volume (no tree update at all) must surface the pair as
    /// soon as the exact AABBs overlap, and drop it again on the way back.
    #[test]
    fn fat_volume_move_within_stays_exact() {
        let mut bp = DbvtBroadPhase::new();
        let a = bp.create_proxy(Aabb::new(Vector3::ZERO, Vector3::ONE));
        // Grow a's fat volume, then shrink back inside it (tree untouched).
        bp.set_aabb(a, Aabb::new(Vector3::ZERO, Vector3::new(1.1, 1.0, 1.0)));
        bp.set_aabb(a, Aabb::new(Vector3::ZERO, Vector3::ONE));
        // b's exact AABB is 0.03 away from a's: fat volumes overlap
        // (margin 0.05), exact do not.
        let b = bp.create_proxy(Aabb::new(Vector3::new(1.03, 0.0, 0.0), Vector3::new(2.03, 1.0, 1.0)));
        assert!(bp.compute_pairs().is_empty(), "exact-separated pair must be filtered");
        // Move a within its fat volume until the exact AABBs overlap.
        bp.set_aabb(a, Aabb::new(Vector3::ZERO, Vector3::new(1.04, 1.0, 1.0)));
        assert_eq!(bp.compute_pairs(), vec![(a, b)], "exact overlap must surface the cached pair");
        // And back out again within the same fat volume.
        bp.set_aabb(a, Aabb::new(Vector3::ZERO, Vector3::ONE));
        assert!(bp.compute_pairs().is_empty());
    }

    /// Dynamic <-> fixed migration: resting proxies move to the fixed tree,
    /// a moving partner wakes them back, and pairs across the trees are
    /// formed both ways (dynamic-vs-fixed discovery through collide_leaf,
    /// fixed-vs-fixed pairs carried over from the dynamic phase).
    #[test]
    fn dynamic_fixed_migration_round_trip() {
        let mut bp = DbvtBroadPhase::new();
        let home = |x: Real| Aabb::new(Vector3::new(x, 0.0, 0.0), Vector3::new(x + 1.0, 1.0, 1.0));
        let a = bp.create_proxy(home(0.0));
        let b = bp.create_proxy(home(0.5));
        let c = bp.create_proxy(home(2.0));
        let mut brute = super::super::BruteForceBroadPhase::new();
        brute.create_proxy(home(0.0));
        brute.create_proxy(home(0.5));
        brute.create_proxy(home(2.0));
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        // Two resting ticks: everything migrates into the fixed tree.
        bp.compute_pairs();
        bp.compute_pairs();
        assert_eq!(bp.fixed.leaves, 3);
        assert_eq!(bp.dynamic.leaves, 0);
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        // b (fixed) wakes and slides towards a (still fixed): the collide
        // on the dynamic side must discover a in the fixed tree.
        bp.set_aabb(b, home(0.4));
        brute.set_aabb(b, home(0.4));
        assert_eq!(bp.fixed.leaves, 2);
        assert_eq!(bp.dynamic.leaves, 1);
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        // Move away, come back, settle again.
        bp.set_aabb(b, home(2.0));
        brute.set_aabb(b, home(2.0));
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        bp.set_aabb(b, home(0.5));
        brute.set_aabb(b, home(0.5));
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        for _ in 0..3 {
            bp.compute_pairs();
        }
        assert_eq!(bp.fixed.leaves, 3);
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        // a wakes from fixed and moves: both back in the dynamic tree.
        bp.set_aabb(a, home(0.25));
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        // A brand-new proxy overlapping resting c pairs immediately.
        let d = bp.create_proxy(home(2.25));
        brute.create_proxy(home(2.25));
        assert_eq!(bp.compute_pairs(), brute.compute_pairs());
        assert!(bp.compute_pairs().contains(&pair_key(c, d)), "new proxy must pair with resting c");
    }

    /// Random insert/remove/exact-update/fat-update/rotation churn on a
    /// bare tree, with structural invariants checked after every operation
    /// and the overlap query plus same-tree pair traversal compared against
    /// brute-force oracles (exactly once per unordered pair, no self-pairs).
    #[test]
    fn optimize_incremental_maintains_tree_invariants() {
        let mut tree = DbvtTree::new();
        let mut rng = Rng(0x0fdb_4e4f_0db4_0001);
        let mut live: Vec<u32> = Vec::new();
        let mut next_payload = 0usize;
        for step in 0..600 {
            let op = rng.below(100);
            if live.len() < 48 && (op < 40 || live.is_empty()) {
                let aabb = rng.aabb(200, 100);
                live.push(tree.insert(aabb, next_payload));
                next_payload += 1;
            } else if op < 55 && !live.is_empty() {
                let k = rng.below(live.len() as u64) as usize;
                let leaf = live.swap_remove(k);
                tree.remove(leaf);
            } else if op < 75 && !live.is_empty() {
                let leaf = live[rng.below(live.len() as u64) as usize];
                tree.update_fat(leaf, rng.aabb(200, 100), Vector3::ZERO, DBVT_MARGIN);
            } else if op < 85 && !live.is_empty() {
                let leaf = live[rng.below(live.len() as u64) as usize];
                tree.update_exact(leaf, rng.aabb(200, 100));
            } else {
                tree.optimize_incremental(1 + rng.below(4) as u32);
            }
            check_tree_invariants(&tree);
            // Overlap query vs a linear scan of the stored leaf volumes.
            let query = rng.aabb(200, 100);
            let mut found = Vec::new();
            tree.for_each_overlapping(&query, &mut found);
            found.sort_unstable();
            let expect = collect_leaves(&tree)
                .into_iter()
                .filter(|(_, v)| v.overlaps(&query))
                .map(|(p, _)| p)
                .collect::<Vec<_>>();
            assert_eq!(found, expect, "step {step}");
        }
        // Same-tree pair traversal vs the unordered-pair oracle.
        let mut pairs = Vec::new();
        tree.for_each_overlapping_pair(&tree, &mut pairs);
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), pairs.len(), "duplicate or self pairs reported");
        assert!(pairs.iter().all(|&(a, b)| a != b), "self pair reported");
        assert_eq!(sorted, brute_tree_pairs(&tree));
    }

    /// Degenerate shapes and sizes: few leaves, identical/nested/point/
    /// thin/huge volumes, with insert/update/remove churn and oracle
    /// comparisons — no corruption in any regime.
    #[test]
    fn degenerate_shapes_stay_consistent() {
        let shapes = [
            // identical
            Aabb::new(Vector3::splat(0.0), Vector3::splat(1.0)),
            // nested
            Aabb::new(Vector3::splat(-1.0), Vector3::splat(2.0)),
            // point
            Aabb::new(Vector3::splat(0.5), Vector3::splat(0.5)),
            // thin slab
            Aabb::new(Vector3::new(0.0, 0.0, 0.4), Vector3::new(1.0, 1.0, 0.6)),
            // huge
            Aabb::new(Vector3::splat(-1e6), Vector3::splat(1e6)),
        ];
        let mut tree = DbvtTree::new();
        let mut live = Vec::new();
        for round in 0..3 {
            for aabb in shapes {
                live.push(tree.insert(aabb, live.len()));
            }
            check_tree_invariants(&tree);
            assert_eq!(brute_tree_pairs(&tree), {
                let mut p = Vec::new();
                tree.for_each_overlapping_pair(&tree, &mut p);
                p.sort_unstable();
                p
            });
            // Update half, remove the other half.
            for k in (0..live.len()).step_by(2) {
                tree.update_fat(live[k], shapes[(k + round) % shapes.len()], Vector3::ZERO, DBVT_MARGIN);
            }
            check_tree_invariants(&tree);
            for k in (1..live.len()).step_by(2) {
                tree.remove(live[k]);
            }
            live.retain(|l| tree.nodes[*l as usize].is_some());
            check_tree_invariants(&tree);
            // Remove the rest; the tree must return to empty.
            for l in live.drain(..) {
                tree.remove(l);
            }
            assert!(tree.is_empty());
            check_tree_invariants(&tree);
        }
    }

    /// Exact contacts (face/edge/corner touching) and moves around the
    /// margin boundary: touching counts as overlap (`overlaps` uses `<=`);
    /// every step must reproduce the brute-force pair set across the
    /// fat-skip, margin-growth and teleport paths.
    #[test]
    fn exact_contact_and_margin_boundary() {
        let mut bp = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let a = Aabb::new(Vector3::ZERO, Vector3::ONE);
        bp.create_proxy(a);
        brute.create_proxy(a);
        let b = bp.create_proxy(Aabb::new(Vector3::splat(3.0), Vector3::splat(4.0)));
        brute.create_proxy(Aabb::new(Vector3::splat(3.0), Vector3::splat(4.0)));
        assert!(bp.compute_pairs().is_empty());
        // Face / edge / corner touching, then epsilon-separated variants.
        let cases = [
            (Vector3::new(1.0, 0.25, 0.25), true),    // face touch at x = 1
            (Vector3::new(1.0 + 1e-6, 0.25, 0.25), false),
            (Vector3::new(1.0 - 1e-6, 0.25, 0.25), true),
            (Vector3::new(1.0, 1.0, 0.25), true),     // edge touch
            (Vector3::new(1.0, 1.0, 1.0), true),      // corner touch
            (Vector3::new(1.049, 1.0, 1.0), false),   // within a margin of touch, still exact-separated
            (Vector3::new(1.051, 1.0, 1.0), false),   // beyond the margin: teleport path
            (Vector3::new(0.999, 0.999, 0.999), true),
            (Vector3::new(2.5, 2.5, 2.5), false),
        ];
        for (min, expect_pair) in cases {
            let next = Aabb::new(min, min + Vector3::ONE);
            bp.set_aabb(b, next);
            brute.set_aabb(b, next);
            let got = bp.compute_pairs();
            assert_eq!(got, brute.compute_pairs(), "case min {min:?}");
            assert_eq!(got.is_empty(), !expect_pair, "case min {min:?}");
        }
    }

    /// Randomized queries over a mixed dynamic/fixed state: rays (including
    /// zero-length, starting-inside, tangent and endpoint-on-box cases) and
    /// AABB queries (zero-volume, huge, exact-touch, empty region) must
    /// match the brute-force scans.
    #[test]
    fn random_queries_match_brute_force() {
        let mut dbvt = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let mut rng = Rng(0x9001_d00d_cafe_f00d);
        for _ in 0..48 {
            let aabb = rng.aabb(400, 100);
            dbvt.create_proxy(aabb);
            brute.create_proxy(aabb);
        }
        // Settle some proxies into the fixed tree.
        for _ in 0..3 {
            dbvt.compute_pairs();
            brute.compute_pairs();
        }
        // Shake a few back into the dynamic tree.
        for i in (0..48).step_by(7) {
            let aabb = rng.aabb(400, 100);
            dbvt.set_aabb(i, aabb);
            brute.set_aabb(i, aabb);
        }
        for _ in 0..60 {
            let from = Vector3::new(rng.range(-10.0, 50.0), rng.range(-10.0, 50.0), rng.range(-10.0, 50.0));
            let to = Vector3::new(rng.range(-10.0, 50.0), rng.range(-10.0, 50.0), rng.range(-10.0, 50.0));
            assert_eq!(
                dbvt.raycast_candidates(from, to),
                brute.raycast_candidates(from, to),
                "ray {from:?} -> {to:?}"
            );
        }
        // Degenerate rays: zero length (inside a box / in empty space),
        // tangent to a face plane, endpoint exactly on a box.
        let special = [
            (Vector3::new(1.5, 1.5, 1.5), Vector3::new(1.5, 1.5, 1.5)),
            (Vector3::new(-50.0, -50.0, -50.0), Vector3::new(-50.0, -50.0, -50.0)),
            (Vector3::new(-10.0, 1.0, 1.0), Vector3::new(50.0, 1.0, 1.0)),
            (Vector3::new(-10.0, 2.5, 2.5), Vector3::new(2.5, 2.5, 2.5)),
        ];
        for (from, to) in special {
            assert_eq!(
                dbvt.raycast_candidates(from, to),
                brute.raycast_candidates(from, to),
                "special ray {from:?} -> {to:?}"
            );
        }
        for _ in 0..40 {
            let min = Vector3::new(rng.range(-5.0, 45.0), rng.range(-5.0, 45.0), rng.range(-5.0, 45.0));
            let extent = Vector3::splat(rng.range(0.0, 10.0));
            let q = Aabb::new(min, min + extent);
            assert_eq!(dbvt.aabb_candidates(q), brute.aabb_candidates(q), "query {q:?}");
        }
        let special_queries = [
            Aabb::new(Vector3::splat(1.5), Vector3::splat(1.5)),               // zero volume
            Aabb::new(Vector3::splat(-1e4), Vector3::splat(1e4)),              // huge
            Aabb::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(2.0, 2.0, 2.0)), // exact touch
            Aabb::new(Vector3::splat(500.0), Vector3::splat(501.0)),           // empty region
        ];
        for q in special_queries {
            assert_eq!(dbvt.aabb_candidates(q), brute.aabb_candidates(q), "query {q:?}");
        }
    }

    /// Extreme finite magnitudes and plane-like infinite AABBs through the
    /// full state machine (including fixed-tree migration ticks): no panic,
    /// no NaN-driven corruption, brute-force-equivalent output.
    #[test]
    fn extreme_coordinates_match_brute_force() {
        let inf = Real::INFINITY;
        let ninf = Real::NEG_INFINITY;
        let bases: [Real; 6] = [1e-6, 1.0, 1e6, 1e17, 1e30, 1e30];
        for (bi, base) in bases.into_iter().enumerate() {
            let ext = (base * 1e-9).max(1e-3);
            let mut dbvt = DbvtBroadPhase::new();
            let mut brute = super::super::BruteForceBroadPhase::new();
            let p = |off: Real| {
                Aabb::new(
                    Vector3::new(base + off * ext, base, base),
                    Vector3::new(base + (off + 1.0) * ext, base + ext, base + ext),
                )
            };
            let a = dbvt.create_proxy(p(0.0));
            let b = dbvt.create_proxy(p(0.5));
            let c = dbvt.create_proxy(p(10.0));
            brute.create_proxy(p(0.0));
            brute.create_proxy(p(0.5));
            brute.create_proxy(p(10.0));
            assert_eq!(dbvt.compute_pairs(), brute.compute_pairs(), "base[{bi}]");
            // Move b into and out of a, then settle everything into fixed.
            dbvt.set_aabb(b, p(0.25));
            brute.set_aabb(b, p(0.25));
            assert_eq!(dbvt.compute_pairs(), brute.compute_pairs(), "base[{bi}] moved");
            dbvt.set_aabb(b, p(20.0));
            brute.set_aabb(b, p(20.0));
            assert_eq!(dbvt.compute_pairs(), brute.compute_pairs(), "base[{bi}] away");
            for _ in 0..3 {
                dbvt.compute_pairs();
                brute.compute_pairs();
            }
            assert_eq!(dbvt.compute_pairs(), brute.compute_pairs(), "base[{bi}] fixed");
            // Wake b: the fixed -> dynamic migration at an extreme scale.
            dbvt.set_aabb(b, p(0.25));
            brute.set_aabb(b, p(0.25));
            assert_eq!(dbvt.compute_pairs(), brute.compute_pairs(), "base[{bi}] woke");
            let _ = (a, c);
        }
        // Plane-like fully unbounded proxies next to finite bodies.
        let mut dbvt = DbvtBroadPhase::new();
        let mut brute = super::super::BruteForceBroadPhase::new();
        let plane = Aabb::new(Vector3::splat(ninf), Vector3::splat(inf));
        dbvt.create_proxy(plane);
        brute.create_proxy(plane);
        for i in 0..8u32 {
            let off = Vector3::new(1.0 + i as Real, 2.0, 3.0);
            let body = Aabb::new(off, off + Vector3::ONE);
            dbvt.create_proxy(body);
            brute.create_proxy(body);
        }
        assert_eq!(dbvt.compute_pairs(), brute.compute_pairs());
        for _ in 0..3 {
            dbvt.compute_pairs();
            brute.compute_pairs();
        }
        assert_eq!(dbvt.compute_pairs(), brute.compute_pairs());
    }
}
