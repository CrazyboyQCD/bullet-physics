//! Rigid body dynamics and constraint solving. Equivalent to Bullet's
//! `BulletDynamics` library.

pub mod body;
pub mod solver;
pub mod world;

pub use body::RigidBody;
pub use world::PhysicsWorld;
