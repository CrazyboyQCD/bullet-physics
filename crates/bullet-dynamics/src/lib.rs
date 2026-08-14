//! Rigid body dynamics and constraint solving. Equivalent to Bullet's
//! `BulletDynamics` library.

pub mod body;
pub mod character;
pub mod constraint;
pub mod solver;
pub mod vehicle;
pub mod world;

pub use body::RigidBody;
pub use character::KinematicCharacterController;
pub use constraint::{
    AngularLimit, Constraint, ConstraintKind, HingeConstraint, HingeMotor, JointFrame,
    Point2PointConstraint,
};
pub use vehicle::{RaycastVehicle, VehicleTuning, WheelInfo};
pub use world::PhysicsWorld;
