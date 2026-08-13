//! Collision detection: shapes, broad phase, narrow phase, and contact
//! manifolds. Equivalent to Bullet's `BulletCollision` library.

pub mod broadphase;
pub mod manifold;
pub mod narrowphase;
pub mod shapes;

pub use shapes::{BoxShape, CollisionShape, ShapeType, Sphere};
