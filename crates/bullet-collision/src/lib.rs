//! Collision detection: shapes, broad phase, narrow phase, and contact
//! manifolds. Equivalent to Bullet's `BulletCollision` library.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

pub mod broadphase;
pub mod manifold;
pub mod narrowphase;
pub mod raycast;
pub mod shapes;

pub use shapes::{
    BoxShape, Capsule, CollisionShape, Compound, CompoundChild, Cone, ConvexHull, Cylinder,
    Plane, ShapeType, Sphere,
};
