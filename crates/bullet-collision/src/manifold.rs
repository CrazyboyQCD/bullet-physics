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
}
