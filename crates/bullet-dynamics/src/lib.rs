//! Rigid body dynamics and constraint solving. Equivalent to Bullet's
//! `BulletDynamics` library.

pub mod body;
pub mod character;
pub mod constraint;
mod island;
pub mod multibody;
pub mod softbody;
pub mod solver;
pub mod vehicle;
pub mod world;

pub use body::RigidBody;
pub use character::KinematicCharacterController;
pub use constraint::{
    AngularLimit, ConeTwistConstraint, ConeTwistMotor, Constraint, ConstraintKind,
    FixedConstraint, Generic6DofConstraint, HingeConstraint, HingeMotor, JointFrame,
    Point2PointConstraint, RotationalLimitMotor, SliderConstraint, SliderMotor,
    TranslationalLimitMotor,
};
pub use multibody::{MultiBody, MultiBodyConstraint};
pub use softbody::{
    create_ellipsoid, create_from_convex_hull, create_from_tri_mesh, create_patch, create_rope,
    patch_fixeds, AeroModel, Cluster, Face, Impulse, Material, MsvcRand, Pose, SContact, Side,
    SoftBody, SoftCollision, Tetra, ROPE_FIX_END, ROPE_FIX_START,
};
pub use vehicle::{RaycastVehicle, VehicleTuning, WheelInfo};
pub use world::{PhysicsWorld, StepStage, StepTimings, STEP_STAGES};
