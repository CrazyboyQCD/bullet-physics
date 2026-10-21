# bullet-inverse-dynamics

Inverse dynamics for tree-structured multibody systems — the Rust port of
Bullet's `BulletInverseDynamics` (`btInverseDynamics::MultiBodyTree`).

Computes generalized joint forces from (q, u, u̇) for trees of
fixed/revolute/prismatic/floating/spherical joints, plus the joint-space mass
matrix (composite rigid body algorithm), forward kinematics at three levels,
and per-body 3×N jacobians. `f32`/`f64` via feature, `no_std` compatible,
zero heap allocation after `finalize`.

Part of the `bullet-physics` workspace; depends only on `bullet-math`.
