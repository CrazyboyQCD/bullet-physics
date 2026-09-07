//! Point-to-point (ball-socket) joint data (Bullet's `btPoint2PointConstraint`).

use super::*;

/// Parameters of a ball-socket joint (Bullet's `btPoint2PointConstraint`).
#[derive(Debug, Clone, PartialEq)]
pub struct Point2PointConstraint {
    /// Pivot point in body A's center-of-mass frame.
    pub pivot_a: Vector3,
    /// Pivot point in body B's center-of-mass frame.
    pub pivot_b: Vector3,
    /// Optional symmetric clamp on the impulse of each row (Bullet's
    /// `btConstraintSetting::m_impulseClamp`); `0.0` disables the clamp.
    pub impulse_clamp: Real,
    /// Optional ERP override (Bullet's `BT_P2P_FLAGS_ERP`); `None` uses the
    /// global ERP (0.2).
    pub erp: Option<Real>,
}

impl Point2PointConstraint {
    /// Clamp the impulse of every row to `[-clamp, clamp]` (Bullet's
    /// `btConstraintSetting::m_impulseClamp`); `0.0` disables the clamp.
    pub fn set_impulse_clamp(&mut self, clamp: Real) {
        self.impulse_clamp = clamp;
    }

    /// Override the position-correction factor for this joint (Bullet's
    /// `BT_CONSTRAINT_ERP`); `None` restores the global 0.2.
    pub fn set_erp(&mut self, erp: Option<Real>) {
        self.erp = erp;
    }
}

