//! Soft-body builders — port of `btSoftBodyHelpers` (phase A: ropes;
//! patches/ellipsoids/meshes arrive with the cloth and volume phases).

use bullet_math::{Real, Vector3};

use super::body::SoftBody;

/// Bit flag for [`create_rope`]: pin the first node (Bullet's `fixeds & 1`).
pub const ROPE_FIX_START: u32 = 1;
/// Bit flag for [`create_rope`]: pin the last node (Bullet's `fixeds & 2`).
pub const ROPE_FIX_END: u32 = 2;

/// Create a rope of `res + 2` unit-mass nodes strung between `from` and
/// `to`, chained by structural links (Bullet's
/// `btSoftBodyHelpers::CreateRope`). `fixeds` combines
/// [`ROPE_FIX_START`]/[`ROPE_FIX_END`] to pin the ends (mass 0).
pub fn create_rope(from: Vector3, to: Vector3, res: usize, fixeds: u32) -> SoftBody {
    // Create nodes.
    let r = res + 2;
    let one: Real = 1.0;
    let mut positions = Vec::with_capacity(r);
    let masses = vec![one; r];
    for i in 0..r {
        let t = i as Real / (r - 1) as Real;
        positions.push(from.lerp(to, t));
    }
    let mut psb = SoftBody::new(&positions, &masses);
    if fixeds & ROPE_FIX_START != 0 {
        psb.set_mass(0, 0.0);
    }
    if fixeds & ROPE_FIX_END != 0 {
        psb.set_mass(r - 1, 0.0);
    }
    // Create links.
    for i in 1..r {
        psb.append_link(i - 1, i, None, false);
    }
    psb
}
