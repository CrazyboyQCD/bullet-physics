//! Rigid body dynamics and constraint solving. Equivalent to Bullet's
//! `BulletDynamics` library.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

pub mod body;
pub mod character;
pub mod constraint;
pub mod handles;
mod island;
pub mod multibody;
pub mod softbody;
pub mod solver;
pub mod timing;
pub mod vehicle;
pub mod world;

pub use body::{CollisionFlags, MotionState, RigidBody};
pub use character::KinematicCharacterController;
pub use constraint::{
    AngularLimit, ConeTwistConstraint, ConeTwistMotor, Constraint, ConstraintKind,
    FixedConstraint, Generic6DofConstraint, HingeConstraint, HingeMotor, JointFrame,
    Point2PointConstraint, RotationalLimitMotor, SliderConstraint, SliderMotor,
    TranslationalLimitMotor,
};
pub use handles::{BodyHandle, MultiBodyHandle, SoftBodyHandle};
pub use multibody::{MultiBody, MultiBodyConstraint};
pub use softbody::{
    create_ellipsoid, create_from_convex_hull, create_from_tri_mesh, create_patch, create_rope,
    patch_fixeds, AeroModel, AngularControl, AngularJointSpecs, AJoint, Cluster, DefaultAngularControl,
    Face, Impulse, ImplicitFn, JointSpecs, LinearJointSpecs, LJoint, Material, MsvcRand, Pose,
    SContact, Side, SoftBody, SoftCollision, SoftJoint, Tetra, ROPE_FIX_END, ROPE_FIX_START,
};
pub use vehicle::{RaycastVehicle, VehicleTuning, WheelInfo};
pub use timing::{StepStage, StepTimings, STEP_STAGES};
pub use world::{CollisionEvent, CollisionObject, PhysicsWorld};
