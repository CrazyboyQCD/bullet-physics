use bullet_math::{math, Quaternion, Real, Vector3};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Distance below which two contact points are the same physical contact,
/// and how far a cached contact may separate (or drift sideways) before it
/// is dropped (Bullet's `gContactBreakingThreshold`).
pub const CONTACT_BREAKING_THRESHOLD: Real = 0.02;

/// A single contact point between two bodies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    /// Contact position in world space.
    pub position: Vector3,
    /// Contact normal, unit length, pointing from body A to body B.
    pub normal: Vector3,
    /// Penetration depth (positive when penetrating).
    pub penetration: Real,
    /// Accumulated normal impulse, for warm starting across solver iterations.
    pub normal_impulse: Real,
    /// Accumulated friction impulses along two tangent axes.
    pub tangent_impulse: [Real; 2],
}

impl ContactPoint {
    #[inline]
    pub fn new(position: Vector3, normal: Vector3, penetration: Real) -> Self {
        Self {
            position,
            normal,
            penetration,
            normal_impulse: 0.0,
            tangent_impulse: [0.0; 2],
        }
    }
}

/// A contact manifold holding up to a handful of contact points.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ContactManifold {
    pub points: Vec<ContactPoint>,
}

impl ContactManifold {
    #[inline]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    #[inline]
    pub fn from_points(points: Vec<ContactPoint>) -> Self {
        Self { points }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Flip the manifold so the normal points in the opposite direction
    /// (swap body A and body B).
    #[inline]
    pub fn flipped(mut self) -> Self {
        for p in &mut self.points {
            p.normal = -p.normal;
        }
        self
    }

    /// Keep at most `max` well-distributed points (Bullet manifold reduction).
    ///
    /// The deepest-penetrating point is always kept first; the rest are
    /// chosen greedily, each being the candidate farthest from the points
    /// already kept, so a box resting on a triangle mesh yields a stable
    /// planar contact set instead of one point per triangle.
    pub fn reduce(&mut self, max: usize) {
        if self.points.len() <= max {
            return;
        }
        let mut candidates = core::mem::take(&mut self.points);
        candidates.sort_by(|a, b| b.penetration.total_cmp(&a.penetration));
        let mut kept: Vec<ContactPoint> = Vec::with_capacity(max);
        kept.push(candidates[0]);
        let mut used = vec![false; candidates.len()];
        used[0] = true;
        while kept.len() < max {
            let mut best = 0;
            let mut best_sep = Real::NEG_INFINITY;
            for (i, p) in candidates.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let sep = kept
                    .iter()
                    .map(|q| (p.position - q.position).length_squared())
                    .fold(Real::INFINITY, Real::min);
                if sep > best_sep {
                    best_sep = sep;
                    best = i;
                }
            }
            kept.push(candidates[best]);
            used[best] = true;
        }
        self.points = kept;
    }
}

/// A contact point cached across steps, anchored in both bodies' local
/// frames so it can be re-projected onto the current poses without
/// re-running the narrow phase (Bullet's persistent-manifold point).\n///
/// The world-space anchors reconstruct as `wa = pos_a + ori_a * local_a`
/// (a point rigidly attached to A) and `wb = pos_b + ori_b * local_b`
/// (attached to B); the signed separation along the cached normal is
/// `(wa - wb) . normal`, positive when penetrating — the convention of
/// [`ContactPoint::penetration`], and the sign flip of Bullet's
/// `m_distance1` (negative = penetrating there).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistentPoint {
    /// World anchor at detection time, in body A's local frame.
    pub local_a: Vector3,
    /// World anchor at detection time, in body B's local frame.
    pub local_b: Vector3,
    /// Cached unit contact normal from A to B.
    pub normal: Vector3,
    /// Signed separation along the normal at the last refresh (positive =
    /// penetrating, values in `(-threshold, 0]` are margin-band contacts).
    pub penetration: Real,
    /// Accumulated normal impulse from the last solve, inherited by points
    /// that persist across steps (warm starting).
    pub normal_impulse: Real,
}

/// Relative-pose drift (translation in meters, rotation in radians)
/// tolerated between detection frames before the cached geometry is
/// considered stale and the pair is re-detected. Keeps the refresh
/// approximation bounded: within this window the deepest features of two
/// resting convex bodies cannot change.
const REDRIFT_TRANSLATION: Real = 0.005;
const REDRIFT_TRANSLATION_SQUARED: Real = REDRIFT_TRANSLATION * REDRIFT_TRANSLATION;
const REDRIFT_ANGLE: Real = 0.01;

/// Persistent-manifold capacity (Bullet's `MANIFOLD_CACHE_SIZE`).
const MANIFOLD_CACHE_SIZE: usize = 4;

/// Rotation angle of a unit quaternion (`2 * acos(w)`, clamped).
#[inline]
fn rotation_angle(q: Quaternion) -> Real {
    2.0 * math::acos(q.w.clamp(-1.0, 1.0))
}

/// A contact manifold cached per overlapping body pair across steps
/// (Bullet's `btPersistentManifold`): the points are anchored in both
/// local frames and refreshed against the current poses; only pairs whose
/// relative pose drifted past the redetect thresholds (or whose points
/// died in the refresh) re-run the narrow phase.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistentManifold {
    points: Vec<PersistentPoint>,
    /// Poses the points were detected at (drift reference).
    detect_pos_a: Vector3,
    detect_ori_a: Quaternion,
    detect_pos_b: Vector3,
    detect_ori_b: Quaternion,
    /// Whether the detection poses are meaningful (false until the first
    /// capture).
    initialized: bool,
    /// Per-pair contact band: how far a cached point may separate (or drift
    /// sideways) before it dies (Bullet stores the manifold breaking
    /// threshold; pairs take the smaller shape disc * 0.02).
    threshold: Real,
    /// Step counter stamp of the last broad-phase pair sighting, for the
    /// lifecycle prune (entries whose pair disappeared are dropped).
    touched: u32,
}

impl PersistentManifold {
    /// Record that the owning pair was seen in this step's broad-phase
    /// output (the lifecycle prune drops entries that were not).
    pub fn mark_paired(&mut self, step: u32) {
        self.touched = step;
    }

    /// Whether the pair was seen in step `step` (see [`Self::mark_paired`]).
    pub fn paired_at(&self, step: u32) -> bool {
        self.touched == step
    }
}

impl Default for PersistentManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentManifold {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            detect_pos_a: Vector3::ZERO,
            detect_ori_a: Quaternion::IDENTITY,
            detect_pos_b: Vector3::ZERO,
            detect_ori_b: Quaternion::IDENTITY,
            initialized: false,
            threshold: CONTACT_BREAKING_THRESHOLD,
            touched: 0,
        }
    }

    pub fn points(&self) -> &[PersistentPoint] {
        &self.points
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Mutable access for the post-solve impulse writeback.
    pub fn point_mut(&mut self, index: usize) -> Option<&mut PersistentPoint> {
        self.points.get_mut(index)
    }

    /// Drop every cached point (used when a re-detection finds no contact).
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Re-project every cached point onto the current poses (Bullet's
    /// `refreshContactPoints`), dropping points that separated beyond the
    /// breaking threshold or whose anchors drifted sideways past it, and
    /// return the surviving points as fresh contact points: position = the
    /// B-side anchor (the deepest witness on B at detection), penetration =
    /// the re-projected signed separation. Margin-band survivors with
    /// non-positive penetration are still returned — the solver treats them
    /// as velocity-only contacts, like Bullet's margin contacts.
    pub fn refresh(
        &mut self,
        pos_a: Vector3,
        ori_a: Quaternion,
        pos_b: Vector3,
        ori_b: Quaternion,
        out: &mut Vec<ContactPoint>,
    ) {
        out.clear();
        let threshold = self.threshold;
        let threshold_sq = threshold * threshold;
        self.points.retain_mut(|p| {
            let wa = pos_a + ori_a * p.local_a;
            let wb = pos_b + ori_b * p.local_b;
            let d = wa - wb;
            let penetration = d.dot(p.normal);
            if penetration < -threshold {
                return false;
            }
            let lateral = d - p.normal * penetration;
            if lateral.length_squared() > threshold_sq {
                return false;
            }
            p.penetration = penetration;
            let mut cp = ContactPoint::new(wb, p.normal, penetration);
            // Carry the accumulated impulse so the solver warm-starts from
            // it (the writeback stores it back after the solve).
            cp.normal_impulse = p.normal_impulse;
            out.push(cp);
            true
        });
        // The refreshed points line up with the surviving cache entries:
        // `retain_mut` keeps the order, so index k of `out` addresses cache
        // point k.
    }

    /// Whether the pair's relative pose drifted past the redetect
    /// thresholds since the points were detected — the cached anchors and
    /// normals may be stale, so the narrow phase must run again.
    pub fn needs_redetect(
        &self,
        pos_a: Vector3,
        ori_a: Quaternion,
        pos_b: Vector3,
        ori_b: Quaternion,
    ) -> bool {
        if !self.initialized {
            return true;
        }
        let drift = (pos_b - self.detect_pos_b) - (pos_a - self.detect_pos_a);
        if drift.length_squared() > REDRIFT_TRANSLATION_SQUARED {
            return true;
        }
        let rot_a = ori_a * self.detect_ori_a.inverse();
        let rot_b = ori_b * self.detect_ori_b.inverse();
        rotation_angle(rot_a) + rotation_angle(rot_b) > REDRIFT_ANGLE
    }

    /// Merge a freshly detected manifold into the cached points (the
    /// merge-half of Bullet's `addManifoldPoint` / `replaceContactPoint`):
    /// a detection point matching a cached one (local anchor within the
    /// breaking threshold, compatible normal) replaces it in place and
    /// inherits the accumulated normal impulse; an unmatched point is
    /// appended; cached points with no detection counterpart are KEPT —
    /// they die only in `refresh` (separated or drifted past the
    /// threshold) or when the capacity overflows, in which case the cache
    /// is sorted deepest-first and truncated to four (Bullet instead lets
    /// the new point always enter, evicting the slot that maximizes the
    /// surviving hull area — see `docs/bullet-deviations.md` §1
    /// "Persistent manifold: overflow policy"). This accumulation is what
    /// grows a resting flat box from its single first-touch point to the
    /// full four-corner support set: `btConvexPlaneCollisionAlgorithm`
    /// reports one point per frame and relies on exactly this merge.
    /// Friction impulses are deliberately not inherited: the tangent basis
    /// is rebuilt from the current normal every step, and stale tangential
    /// impulses along a rotated basis made resting bodies oscillate (see
    /// the warm-starting notes in `world.rs`).
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &mut self,
        manifold: &ContactManifold,
        pos_a: Vector3,
        ori_a: Quaternion,
        pos_b: Vector3,
        ori_b: Quaternion,
        threshold: Real,
    ) {
        self.threshold = threshold;
        let inv_a = ori_a.inverse();
        let inv_b = ori_b.inverse();
        let threshold_sq = threshold * threshold;
        for cp in &manifold.points {
            // Reconstruct the A-side witness from the reported contact
            // point: our detection reports the witness on B plus the
            // normal and depth, so the witness on A sits `penetration`
            // along the normal from it.
            let wb = cp.position;
            let wa = cp.position + cp.normal * cp.penetration;
            let mut point = PersistentPoint {
                local_a: inv_a * (wa - pos_a),
                local_b: inv_b * (wb - pos_b),
                normal: cp.normal,
                penetration: cp.penetration,
                normal_impulse: 0.0,
            };
            if let Some(slot) = self.points.iter_mut().find(|o| {
                (o.local_a - point.local_a).length_squared() < threshold_sq
                    && o.normal.dot(point.normal) >= 0.9
            }) {
                point.normal_impulse = slot.normal_impulse;
                *slot = point;
            } else {
                self.points.push(point);
            }
        }
        // Capacity: keep the deepest points when the manifold overflows
        // (Bullet evicts by surviving-hull-area instead — deviations §1).
        if self.points.len() > MANIFOLD_CACHE_SIZE {
            self.points
                .sort_by(|a, b| b.penetration.total_cmp(&a.penetration));
            self.points.truncate(MANIFOLD_CACHE_SIZE);
        }
        self.detect_pos_a = pos_a;
        self.detect_ori_a = ori_a;
        self.detect_pos_b = pos_b;
        self.detect_ori_b = ori_b;
        self.initialized = true;
    }

    /// A GJK seed direction for re-detecting this pair: the cached normal
    /// of the first (deepest) point, when it is a sane unit vector.
    pub fn seed(&self) -> Option<Vector3> {
        self.points
            .first()
            .map(|p| p.normal)
            .filter(|n| n.length_squared() > 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn point(x: Real, y: Real, z: Real, penetration: Real) -> ContactPoint {
        ContactPoint::new(Vector3::new(x, y, z), Vector3::Y, penetration)
    }

    /// A one-point manifold of a box resting on a plane: detection at
    /// box center (0, 0.49), contact on the plane at the origin.
    fn resting_contact(penetration: Real) -> (ContactManifold, Vector3, Vector3) {
        let pos_a = Vector3::new(0.0, 0.5 - penetration, 0.0); // box (A)
        let pos_b = Vector3::ZERO; // plane (B)
        let manifold = ContactManifold::from_points(vec![ContactPoint::new(
            Vector3::ZERO,
            Vector3::Y,
            penetration,
        )]);
        (manifold, pos_a, pos_b)
    }

    #[test]
    fn refresh_reprojects_separation_and_keeps_band_points() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );

        // Move A away from B along the normal by 0.015: from +0.01
        // penetration to a 0.005 separation (margin band).
        let pos_a_separated = pos_a - Vector3::Y * 0.015;
        let mut refreshed = Vec::new();
        cache.refresh(
            pos_a_separated,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            &mut refreshed,
        );
        assert_eq!(refreshed.len(), 1);
        // The separation is negative now (shapes separated) but within the
        // breaking threshold, so the point survives as a band contact.
        assert_relative_eq!(refreshed[0].penetration, -0.005, epsilon = 1e-6);
        assert_eq!(refreshed[0].position, Vector3::ZERO);
        assert_eq!(refreshed[0].normal, Vector3::Y);
        assert_relative_eq!(cache.points()[0].penetration, -0.005, epsilon = 1e-6);
    }

    #[test]
    fn refresh_drops_separated_and_sideways_drifted_points() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );

        // Moved beyond the breaking threshold: the point dies.
        let mut gone = Vec::new();
        cache.refresh(
            pos_a - Vector3::Y * 0.05,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            &mut gone,
        );
        assert!(gone.is_empty() && cache.is_empty());

        // Recapture, then slide A sideways by more than the threshold: the
        // lateral anchor drift kills the point even though the separation
        // stays small.
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        let mut drifted = Vec::new();
        cache.refresh(
            pos_a + Vector3::X * 0.03,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            &mut drifted,
        );
        assert!(drifted.is_empty() && cache.is_empty());
    }

    #[test]
    fn capture_anchors_round_trip_to_the_detection_points() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        let p = cache.points()[0];
        let wa = pos_a + Quaternion::IDENTITY * p.local_a;
        let wb = pos_b + Quaternion::IDENTITY * p.local_b;
        // wb is the reported contact point; wa sits one penetration above it.
        assert_eq!(wb, Vector3::ZERO);
        assert_relative_eq!(wa.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(wa.y, 0.01, epsilon = 1e-6);
        assert_relative_eq!(wa.z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn capture_merges_and_inherits_the_impulse_of_matching_points() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        cache.point_mut(0).unwrap().normal_impulse = 5.0;

        // Re-detection at the same pose: the point is the same physical
        // contact, so the impulse survives.
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        assert_relative_eq!(cache.points()[0].normal_impulse, 5.0);

        // Re-detection with a flipped normal: a different contact —
        // appended, starting from zero, while the unmatched old point is
        // KEPT alive (Bullet's `addManifoldPoint`); the refresh prunes it
        // once it separates or drifts.
        let flipped = ContactManifold::from_points(vec![ContactPoint::new(
            Vector3::ZERO,
            -Vector3::Y,
            0.01,
        )]);
        cache.capture(
            &flipped,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        assert_eq!(cache.points().len(), 2);
        assert_relative_eq!(cache.points()[0].normal_impulse, 5.0);
        assert_relative_eq!(cache.points()[1].normal_impulse, 0.0);
    }

    #[test]
    fn redetect_triggers_on_relative_drift_only() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        assert!(cache.needs_redetect(pos_a, Quaternion::IDENTITY, pos_b, Quaternion::IDENTITY));
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        // Small drift: keep refreshing.
        assert!(!cache.needs_redetect(
            pos_a + Vector3::Y * 0.004,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY
        ));
        // Translation drift past the threshold.
        assert!(cache.needs_redetect(
            pos_a + Vector3::Y * 0.006,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY
        ));
        // Rotation drift past the threshold (0.02 rad about Z on A).
        let q = Quaternion::from_rotation_z(0.02);
        assert!(cache.needs_redetect(pos_a, q, pos_b, Quaternion::IDENTITY));
        // Moving both bodies together is no relative drift.
        assert!(!cache.needs_redetect(
            pos_a + Vector3::X,
            Quaternion::IDENTITY,
            pos_b + Vector3::X,
            Quaternion::IDENTITY
        ));
    }

    #[test]
    fn reduce_keeps_the_deepest_and_spreads_the_rest() {
        let mut m = ContactManifold::from_points(vec![
            point(0.0, 0.0, 0.0, 0.5),
            point(0.01, 0.0, 0.0, 0.4), // duplicates the deepest point
            point(1.0, 0.0, 0.0, 0.1),
            point(0.0, 1.0, 0.0, 0.1),
            point(0.0, 0.0, 1.0, 0.1),
            point(1.0, 1.0, 1.0, 0.05),
        ]);
        m.reduce(4);
        assert_eq!(m.points.len(), 4);
        // The deepest contact survives and stays first.
        assert_eq!(m.points[0].position, Vector3::ZERO);
        assert_eq!(m.points[0].penetration, 0.5);
        // The near-duplicate was dropped in favor of a spread-out contact set
        // (the deepest point at the origin is exempt from the spread check).
        assert!(m.points.iter().skip(1).all(|p| p.position.length_squared() > 0.5));
    }

    #[test]
    fn reduce_leaves_small_manifolds_alone() {
        let points = vec![point(0.0, 0.0, 0.0, 0.2), point(1.0, 0.0, 0.0, 0.1), point(0.0, 1.0, 0.0, 0.1)];
        let mut m = ContactManifold::from_points(points.clone());
        m.reduce(4);
        assert_eq!(m.points, points);
    }

    // ---- adversarial validation (manifold audit follow-up, 2026-09-08) —

    /// Maximum pairwise distance between anchors.
    fn spread(anchors: &[Vector3]) -> Real {
        let mut max: Real = 0.0;
        for (i, a) in anchors.iter().enumerate() {
            for b in &anchors[i + 1..] {
                max = max.max((*a - *b).length());
            }
        }
        max
    }

    /// Minimal reference model of Bullet's overflow policy
    /// (`addManifoldPoint` -> `sortCachedPoints`): below capacity the point
    /// appends; at capacity the new point ALWAYS enters, replacing the slot
    /// whose removal maximizes the 3-point hull area of the surviving four
    /// (`gContactCalcArea3Points = true` default), with the deepest slot
    /// exempt (`KEEP_DEEPEST_POINT`, its area contribution reads 0 and
    /// participates in the first-max tie-break like `closestAxis4`).
    /// `position` plays the role of `m_localPointA`; Bullet's distance
    /// convention (negative = penetrating) is `-penetration`.
    fn bullet_overflow_reference(cache: &mut Vec<ContactPoint>, new_pt: ContactPoint) {
        if cache.len() < 4 {
            cache.push(new_pt);
            return;
        }
        let dist = |p: &ContactPoint| -p.penetration;
        let mut deepest = 4usize; // sentinel: the new point is the deepest
        let mut max_pen = dist(&new_pt);
        for (i, p) in cache.iter().enumerate() {
            if dist(p) < max_pen {
                max_pen = dist(p);
                deepest = i;
            }
        }
        let mut best = 0usize;
        let mut best_area = -1.0;
        for slot in 0..4 {
            let area = if slot == deepest {
                0.0
            } else {
                match slot {
                    0 => (new_pt.position - cache[1].position)
                        .cross(cache[3].position - cache[2].position)
                        .length_squared(),
                    1 => (new_pt.position - cache[0].position)
                        .cross(cache[3].position - cache[2].position)
                        .length_squared(),
                    2 => (new_pt.position - cache[0].position)
                        .cross(cache[3].position - cache[1].position)
                        .length_squared(),
                    _ => (new_pt.position - cache[0].position)
                        .cross(cache[2].position - cache[1].position)
                        .length_squared(),
                }
            };
            if area > best_area {
                best_area = area;
                best = slot;
            }
        }
        cache[best] = new_pt;
    }

    /// Minimal reference of Bullet's `getCacheEntry`: the NEAREST cached
    /// anchor within the threshold radius, normal never consulted.
    fn nearest_cache_entry_reference(
        cache: &[PersistentPoint],
        incoming_anchor: Vector3,
        threshold: Real,
    ) -> Option<usize> {
        let mut shortest = threshold * threshold;
        let mut nearest = None;
        for (i, p) in cache.iter().enumerate() {
            let d = (p.local_a - incoming_anchor).length_squared();
            if d < shortest {
                shortest = d;
                nearest = Some(i);
            }
        }
        nearest
    }

    /// D1 (audit deviation): on overflow we keep the deepest four points in
    /// depth order; Bullet maximizes the surviving set's hull area instead.
    /// On this input the two policies pick materially different sets: ours
    /// keeps a clustered patch, the area-maximizing reference keeps the
    /// far-spread points.
    #[test]
    fn capture_overflow_keeps_deepest_four_even_when_clustered() {
        let pt = |x: Real, y: Real, z: Real, pen: Real| {
            ContactPoint::new(Vector3::new(x, y, z), Vector3::Y, pen)
        };
        // One deep point; three shallow points clustered within 0.03 of it
        // (beyond the matching radius of each other, so the batch all
        // appends); two shallow points 1.0 away.
        let batch = ContactManifold::from_points(vec![
            pt(0.0, 0.0, 0.0, 0.5),
            pt(0.03, 0.0, 0.0, 0.1),
            pt(0.0, 0.0, 0.03, 0.1),
            pt(0.0, 0.03, 0.0, 0.1),
            pt(1.0, 0.0, 0.0, 0.1),
            pt(0.0, 1.0, 0.0, 0.1),
        ]);
        let origin = Vector3::ZERO;
        let mut cache = PersistentManifold::new();
        cache.capture(
            &batch,
            origin,
            Quaternion::IDENTITY,
            origin,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );

        // Depth-order truncation keeps the deep point plus the first three
        // shallow points in input order — the clustered ones.
        assert_eq!(cache.points().len(), 4);
        let anchors: Vec<Vector3> = cache.points().iter().map(|p| p.local_a).collect();
        assert_relative_eq!(anchors[0].y, 0.5, epsilon = 1e-6); // the deep point
        let shallow = &anchors[1..4];
        assert!(spread(shallow) < 0.1, "shallow keeps must be clustered");
        assert!(
            anchors.iter().all(|a| a.x.max(a.y).max(a.z) < 0.6),
            "the far points must not be kept"
        );

        // Bullet's area-maximizing reference keeps the two far points
        // instead.
        let mut reference: Vec<ContactPoint> = Vec::new();
        for cp in &batch.points {
            bullet_overflow_reference(&mut reference, *cp);
        }
        assert_eq!(reference.len(), 4);
        let ref_anchors: Vec<Vector3> = reference.iter().map(|p| p.position).collect();
        assert!(ref_anchors.iter().any(|a| a.x > 0.5), "reference keeps (1,0,0)");
        assert!(
            ref_anchors.iter().any(|a| a.y > 0.6 && a.x < 0.1),
            "reference keeps (0,1,0)"
        );
        assert!(spread(&ref_anchors) > 0.9, "reference set is spread");
    }

    /// The policy divergence is not specific to clustered inputs: with a
    /// fully spread cache, a shallow near-duplicate arriving beyond the
    /// matching radius is dropped by depth truncation while the reference
    /// admits it and evicts a support point. Our direction keeps the
    /// stronger support set here — the divergence is bidirectional.
    #[test]
    fn capture_overflow_differs_even_on_spread_inputs() {
        let origin = Vector3::ZERO;
        let id = Quaternion::IDENTITY;
        let base = [
            ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.4),
            ContactPoint::new(Vector3::new(1.0, 0.0, 0.0), Vector3::Y, 0.3),
            ContactPoint::new(Vector3::new(0.0, 0.0, 1.0), Vector3::Y, 0.2),
            ContactPoint::new(Vector3::new(1.0, 0.0, 1.0), Vector3::Y, 0.1),
        ];
        let mut cache = PersistentManifold::new();
        for cp in &base {
            cache.capture(
                &ContactManifold::from_points(vec![*cp]),
                origin,
                id,
                origin,
                id,
                CONTACT_BREAKING_THRESHOLD,
            );
        }
        // A shallow near-duplicate of the deep corner arrives (outside the
        // matching radius, so it takes the overflow path).
        let dup = ContactPoint::new(Vector3::new(0.015, 0.0, 0.015), Vector3::Y, 0.05);
        cache.capture(
            &ContactManifold::from_points(vec![dup]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        // Depth truncation keeps the original spread set and drops the dup.
        assert_eq!(cache.points().len(), 4);
        let anchors: Vec<Vector3> = cache.points().iter().map(|p| p.local_a).collect();
        assert!(anchors.iter().any(|a| a.y > 0.3), "deep point kept");
        assert!(
            !anchors
                .iter()
                .any(|a| (a.x - 0.015).abs() < 1e-6 && (a.z - 0.015).abs() < 1e-6),
            "the near-duplicate must be dropped"
        );

        // The reference admits the duplicate and evicts a support point.
        let mut reference: Vec<ContactPoint> = base.to_vec();
        bullet_overflow_reference(&mut reference, dup);
        assert!(
            reference
                .iter()
                .any(|p| (p.position.x - 0.015).abs() < 1e-6),
            "reference admits the near-duplicate"
        );
        assert!(
            !reference.iter().any(|p| p.position.x > 0.9 && p.position.z > 0.9),
            "reference evicted the far support point"
        );
    }

    /// Repeated overflow on a planar patch: the cache never exceeds four,
    /// the deepest seed's anchor is bit-stable across every frame, and the
    /// set frozen in by the overflow policy stays spread. Late patch points
    /// never enter under depth truncation, while the area-maximizing
    /// reference keeps absorbing the patch.
    #[test]
    fn overflow_evolution_on_a_planar_patch() {
        let origin = Vector3::ZERO;
        let id = Quaternion::IDENTITY;
        let mut cache = PersistentManifold::new();
        // The deep seed, then patch points on a 0.05 grid (all beyond the
        // matching radius of each other).
        let mut frames = vec![ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.5)];
        for x in [0.0, 0.05, 0.1] {
            for z in [0.0, 0.05, 0.1] {
                if x == 0.0 && z == 0.0 {
                    continue;
                }
                frames.push(ContactPoint::new(Vector3::new(x, 0.0, z), Vector3::Y, 0.1));
            }
        }
        let mut deep_anchor = None;
        for f in &frames {
            cache.capture(
                &ContactManifold::from_points(vec![*f]),
                origin,
                id,
                origin,
                id,
                CONTACT_BREAKING_THRESHOLD,
            );
            assert!(cache.points().len() <= 4, "capacity must never overflow");
            let deep = &cache.points()[0];
            match deep_anchor {
                None => deep_anchor = Some(deep.local_a),
                Some(a) => assert_eq!(deep.local_a, a, "deep anchor must be stable"),
            }
        }
        assert_eq!(cache.points().len(), 4);
        let anchors: Vec<Vector3> = cache.points().iter().map(|p| p.local_a).collect();
        assert!(spread(&anchors) > 0.05, "the kept set must not collapse");
        let kept: Vec<_> = anchors.iter().map(|a| (a.x, a.z)).collect();
        for f in &frames[4..] {
            assert!(
                !kept.contains(&(f.position.x, f.position.z)),
                "late patch points are dropped by depth truncation"
            );
        }
        // The reference policy keeps absorbing the patch instead.
        let mut reference: Vec<ContactPoint> = Vec::new();
        for f in &frames {
            bullet_overflow_reference(&mut reference, *f);
        }
        let ref_pos: Vec<_> = reference.iter().map(|p| (p.position.x, p.position.z)).collect();
        assert!(
            ref_pos.iter().any(|p| !kept.contains(p)),
            "the reference keeps absorbing late patch points"
        );
    }

    /// Same evolution on an irregular (triangular) patch: after the first
    /// four frames the identity of the kept points is frozen (anchors never
    /// change again), the seed survives, and the triangle stays spread.
    #[test]
    fn overflow_evolution_on_a_triangular_patch() {
        let origin = Vector3::ZERO;
        let id = Quaternion::IDENTITY;
        let mut cache = PersistentManifold::new();
        let verts = [
            Vector3::new(0.0, 0.0, 0.0),  // deep seed (pen 0.5)
            Vector3::new(0.08, 0.0, 0.0), // triangle vertices
            Vector3::new(0.0, 0.0, 0.08),
            Vector3::new(0.06, 0.0, 0.07),
            // edge midpoints arrive later, all beyond the matching radius
            Vector3::new(0.04, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.04),
            Vector3::new(0.03, 0.0, 0.035),
        ];
        let mut seen: Vec<Vec<Vector3>> = Vec::new();
        for (n, v) in verts.iter().enumerate() {
            let pen = if n == 0 { 0.5 } else { 0.1 };
            let cp = ContactPoint::new(*v, Vector3::Y, pen);
            cache.capture(
                &ContactManifold::from_points(vec![cp]),
                origin,
                id,
                origin,
                id,
                CONTACT_BREAKING_THRESHOLD,
            );
            assert!(cache.points().len() <= 4);
            seen.push(cache.points().iter().map(|p| p.local_a).collect());
        }
        let frozen = &seen[3];
        for anchors in &seen[4..] {
            assert_eq!(anchors, frozen, "cache identity must stop churning");
        }
        assert_relative_eq!(cache.points()[0].local_a.y, 0.5, epsilon = 1e-6);
        assert!(spread(frozen) > 0.08, "the triangle must stay spread");
    }

    /// D2 (audit deviation): `capture` matches the FIRST cached point
    /// within the threshold radius, not the nearest one. Two cached points
    /// sit inside the radius of the incoming detection; the nearer one is
    /// second in the cache. The inherited impulse must come from the first.
    #[test]
    fn capture_matches_the_first_cached_point_not_the_nearest() {
        let origin = Vector3::ZERO;
        let id = Quaternion::IDENTITY;
        let mut cache = PersistentManifold::new();
        // Build the cache with the tight per-pair threshold so the two
        // seeds do not match each other, then match with a wide one.
        let spread_far = ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.01);
        let spread_near = ContactPoint::new(Vector3::new(0.06, 0.0, 0.0), Vector3::Y, 0.01);
        cache.capture(
            &ContactManifold::from_points(vec![spread_far]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        cache.capture(
            &ContactManifold::from_points(vec![spread_near]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        cache.point_mut(0).unwrap().normal_impulse = 5.0;
        cache.point_mut(1).unwrap().normal_impulse = 7.0;

        // The reference would pick the nearer second entry; ours must pick
        // the first and inherit its impulse.
        let incoming = ContactPoint::new(Vector3::new(0.045, 0.0, 0.0), Vector3::Y, 0.01);
        assert_eq!(
            nearest_cache_entry_reference(cache.points(), incoming.position, 0.1),
            Some(1),
            "the reference (Bullet) would match the nearer point"
        );
        cache.capture(
            &ContactManifold::from_points(vec![incoming]),
            origin,
            id,
            origin,
            id,
            0.1,
        );
        assert_eq!(cache.points().len(), 2);
        assert_relative_eq!(cache.points()[0].local_a.x, 0.045, epsilon = 1e-6);
        assert_relative_eq!(cache.points()[0].normal_impulse, 5.0);
    }

    /// The matching deviation is deterministic: it follows cache order, and
    /// the inherited impulse always comes from the point actually matched —
    /// never from a nearest-neighbor side effect or the eviction order.
    #[test]
    fn capture_matching_follows_cache_order_deterministically() {
        let origin = Vector3::ZERO;
        let id = Quaternion::IDENTITY;
        // The same two seeds, appended in the opposite order: now the FIRST
        // cache entry is the near one, so the incoming point inherits its
        // impulse instead.
        let mut cache = PersistentManifold::new();
        let near = ContactPoint::new(Vector3::new(0.06, 0.0, 0.0), Vector3::Y, 0.01);
        let far = ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.01);
        cache.capture(
            &ContactManifold::from_points(vec![near]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        cache.capture(
            &ContactManifold::from_points(vec![far]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        cache.point_mut(0).unwrap().normal_impulse = 7.0;
        cache.point_mut(1).unwrap().normal_impulse = 5.0;
        let incoming = ContactPoint::new(Vector3::new(0.045, 0.0, 0.0), Vector3::Y, 0.01);
        cache.capture(
            &ContactManifold::from_points(vec![incoming]),
            origin,
            id,
            origin,
            id,
            0.1,
        );
        assert_relative_eq!(cache.points()[0].normal_impulse, 7.0);
        // The unmatched neighbor keeps its own impulse untouched.
        assert_relative_eq!(cache.points()[1].normal_impulse, 5.0);
    }

    /// D2's extra guard: a position match with a disagreeing normal
    /// (dot < 0.9) does not inherit — it appends as a fresh contact.
    #[test]
    fn capture_requires_normal_agreement_to_match() {
        let origin = Vector3::ZERO;
        let id = Quaternion::IDENTITY;
        let mut cache = PersistentManifold::new();
        let base = ContactPoint::new(Vector3::ZERO, Vector3::Y, 0.01);
        cache.capture(
            &ContactManifold::from_points(vec![base]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        cache.point_mut(0).unwrap().normal_impulse = 3.0;

        // 25 degrees off the cached normal: dot ~ 0.906 >= 0.9 — matches
        // and inherits.
        let ang = 0.4363323; // 25 degrees
        let tilted = ContactPoint::new(
            Vector3::ZERO,
            Vector3::new(math::sin(ang), math::cos(ang), 0.0),
            0.01,
        );
        cache.capture(
            &ContactManifold::from_points(vec![tilted]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        assert_eq!(cache.points().len(), 1);
        assert_relative_eq!(cache.points()[0].normal_impulse, 3.0);

        // 30 degrees off: dot ~ 0.866 < 0.9 — appended as a fresh contact
        // starting from zero. (Fresh cache: after the match above the cache
        // normal is the 25-degree one, which would agree with almost
        // anything near it.)
        let mut cache = PersistentManifold::new();
        cache.capture(
            &ContactManifold::from_points(vec![base]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        let ang = bullet_math::PI / 6.0; // 30 degrees
        let tilted = ContactPoint::new(
            Vector3::ZERO,
            Vector3::new(math::sin(ang), math::cos(ang), 0.0),
            0.01,
        );
        cache.capture(
            &ContactManifold::from_points(vec![tilted]),
            origin,
            id,
            origin,
            id,
            CONTACT_BREAKING_THRESHOLD,
        );
        assert_eq!(cache.points().len(), 2);
        assert_relative_eq!(cache.points()[1].normal_impulse, 0.0);
    }

    /// Rotation-drift gate boundaries: the sum of per-body rotation angles
    /// is compared against 0.01 rad with a strict `>`; either body alone
    /// triggers it, and opposite rotations add up.
    #[test]
    fn rotation_gate_boundaries_and_body_sides() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        let rot = |angle: Real| Quaternion::from_rotation_z(angle);

        // Below the gate on either body alone: refresh keeps serving.
        assert!(!cache.needs_redetect(pos_a, rot(0.008), pos_b, Quaternion::IDENTITY));
        assert!(!cache.needs_redetect(pos_a, Quaternion::IDENTITY, pos_b, rot(0.008)));
        // Above it on either body alone: re-detection.
        assert!(cache.needs_redetect(pos_a, rot(0.012), pos_b, Quaternion::IDENTITY));
        assert!(cache.needs_redetect(pos_a, Quaternion::IDENTITY, pos_b, rot(0.012)));
        // Equal and opposite rotations add up (0.016 > 0.01), while smaller
        // opposite pairs stay inside the gate.
        assert!(cache.needs_redetect(pos_a, rot(0.008), pos_b, rot(-0.008)));
        assert!(!cache.needs_redetect(pos_a, rot(0.003), pos_b, rot(-0.003)));
    }

    /// The exact-threshold comparison is a strict `>`, but
    /// `rotation_angle` reconstructs the angle through acos — the f32/f64
    /// rounding of `from_rotation_z(0.01)` can land on either side of the
    /// gate, so the expectation is derived from the measured angle
    /// (self-consistency of the gate at the boundary).
    #[test]
    fn rotation_gate_at_the_threshold_tracks_the_measured_angle() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        let q = Quaternion::from_rotation_z(0.01);
        let measured = rotation_angle(q) + rotation_angle(Quaternion::IDENTITY);
        assert_eq!(
            cache.needs_redetect(pos_a, q, pos_b, Quaternion::IDENTITY),
            measured > REDRIFT_ANGLE,
        );
    }

    /// q / -q denote the same rotation: the only cover sensitivity is the
    /// gate reading (2·acos(w)), so a sign-flipped quaternion triggers one
    /// spurious re-detection. The refreshed contact geometry must be
    /// identical either way — the gate changes WHEN detection reruns, not
    /// WHAT the contact is.
    #[test]
    fn negated_quaternion_only_forces_a_redetect_not_new_geometry() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        let q = Quaternion::from_rotation_z(0.008);
        assert!(!cache.needs_redetect(pos_a, q, pos_b, Quaternion::IDENTITY));
        assert!(cache.needs_redetect(pos_a, -q, pos_b, Quaternion::IDENTITY));

        // Refreshing under the flipped quaternion yields the same contact
        // geometry as under the original one (same rotation, same anchors).
        let mut out_a = Vec::new();
        cache.refresh(pos_a, q, pos_b, Quaternion::IDENTITY, &mut out_a);
        let mut out_b = Vec::new();
        cache.refresh(pos_a, -q, pos_b, Quaternion::IDENTITY, &mut out_b);
        assert_eq!(out_a.len(), 1);
        assert_eq!(out_b.len(), 1);
        assert_relative_eq!(out_a[0].position.x, out_b[0].position.x, epsilon = 1e-6);
        assert_relative_eq!(out_a[0].position.y, out_b[0].position.y, epsilon = 1e-6);
        assert_relative_eq!(out_a[0].position.z, out_b[0].position.z, epsilon = 1e-6);
        assert_relative_eq!(out_a[0].penetration, out_b[0].penetration, epsilon = 1e-6);
    }

    /// A resting contact survives sub-threshold pose perturbations without
    /// churn: anchors stay bit-stable, the normal never changes, the
    /// inherited impulse rides along every frame, and only a separation
    /// past the breaking threshold kills the point. A drift just past the
    /// gate asks for re-detection but leaves the healthy point intact (the
    /// gate only decides WHICH path the pipeline runs).
    #[test]
    fn resting_contact_survives_sub_threshold_perturbations_without_churn() {
        let (manifold, pos_a, pos_b) = resting_contact(0.01);
        let mut cache = PersistentManifold::new();
        cache.capture(
            &manifold,
            pos_a,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            CONTACT_BREAKING_THRESHOLD,
        );
        cache.point_mut(0).unwrap().normal_impulse = 3.0;
        let anchor_a = cache.points()[0].local_a;
        let anchor_b = cache.points()[0].local_b;
        let normal = cache.points()[0].normal;

        // 20 frames of alternating sub-drift nudges (0.002 < 0.005).
        let mut out = Vec::new();
        for n in 0..20 {
            let dx = if n % 2 == 0 { 0.002 } else { -0.002 };
            let p = pos_a + Vector3::X * dx;
            assert!(!cache.needs_redetect(p, Quaternion::IDENTITY, pos_b, Quaternion::IDENTITY));
            cache.refresh(p, Quaternion::IDENTITY, pos_b, Quaternion::IDENTITY, &mut out);
            assert_eq!(out.len(), 1, "frame {n}: no churn expected");
            assert_eq!(cache.points()[0].local_a, anchor_a);
            assert_eq!(cache.points()[0].local_b, anchor_b);
            assert_eq!(cache.points()[0].normal, normal);
            assert_relative_eq!(out[0].normal_impulse, 3.0);
        }

        // Drift just past the gate: re-detection is asked for, but refresh
        // keeps the healthy point alive with continuous geometry.
        let p = pos_a + Vector3::X * 0.006;
        assert!(cache.needs_redetect(p, Quaternion::IDENTITY, pos_b, Quaternion::IDENTITY));
        cache.refresh(p, Quaternion::IDENTITY, pos_b, Quaternion::IDENTITY, &mut out);
        assert_eq!(out.len(), 1);
        assert_relative_eq!(out[0].penetration, 0.01, epsilon = 1e-6);

        // Separation past the breaking threshold is what actually kills
        // the point.
        cache.refresh(
            pos_a - Vector3::Y * 0.035,
            Quaternion::IDENTITY,
            pos_b,
            Quaternion::IDENTITY,
            &mut out,
        );
        assert!(out.is_empty() && cache.is_empty());
    }
}
