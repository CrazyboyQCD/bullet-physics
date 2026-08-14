//! Rigid body dynamics and constraint solving. Equivalent to Bullet's
//! `BulletDynamics` library.

pub mod body;
pub mod character;
pub mod solver;
pub mod world;

pub use body::RigidBody;
pub use character::KinematicCharacterController;
pub use world::PhysicsWorld;
