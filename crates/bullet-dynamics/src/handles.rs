//! Opaque typed handles for the bodies stored in a
//! [`PhysicsWorld`](crate::world::PhysicsWorld).
//!
//! [`PhysicsWorld::add_body`](crate::world::PhysicsWorld::add_body) and its
//! multibody/soft-body siblings return these instead of a bare `usize`, so
//! the three index spaces cannot be mixed up at the call site (a rigid
//! handle does not type-check as a soft-body lookup and vice versa). The
//! handles are small value types (`NonZeroU32`, 4 bytes) and `Option<H>`
//! occupies the same space as `H`.
//!
//! Identity semantics match the index-based status quo they replace: a
//! handle is the 0-based slot the body was added to. Rigid-body slots are
//! stable — `remove_body` never renumbers the others and never reuses the
//! slot (Bullet's pointer-identity analog; looking up a removed body
//! panics). Soft-body slots renumber on `remove_soft_body`, exactly like
//! the indices they replace. Handles do not carry a generation counter,
//! so a stale handle behaves like a stale index did: inert data for
//! removed rigid bodies, a different body for renumbered soft bodies.

use std::num::NonZeroU32;

macro_rules! world_handle {
    ($doc:expr, $name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        #[doc = $doc]
        pub struct $name(NonZeroU32);

        impl $name {
            /// The handle of the body at 0-based slot `index` (the value
            /// [`PhysicsWorld::add_body`](crate::world::PhysicsWorld::add_body)
            /// returned as a `usize` before the handle API). Intended for
            /// tests and literal migration; the normal way to obtain a
            /// handle is the return value of an `add_*` method.
            #[inline]
            pub const fn from_index(index: usize) -> Self {
                let raw = index as u32 + 1;
                match NonZeroU32::new(raw) {
                    Some(nz) => Self(nz),
                    None => panic!("body slot index out of u32 range"),
                }
            }

            /// The 0-based slot this handle wraps (for interop, debug
            /// output and tests; world lookups take the handle itself).
            #[inline]
            pub fn index(self) -> usize {
                (self.0.get() - 1) as usize
            }
        }
    };
}

world_handle!(
    "A rigid body in a [`PhysicsWorld`](crate::world::PhysicsWorld)
    (Bullet's `btRigidBody*` pointer).",
    BodyHandle
);
world_handle!(
    "An articulated body in a [`PhysicsWorld`](crate::world::PhysicsWorld)
    (Bullet's `btMultiBody*` pointer).",
    MultiBodyHandle
);
world_handle!(
    "A soft body in a [`PhysicsWorld`](crate::world::PhysicsWorld)
    (Bullet's `btSoftBody*` pointer).",
    SoftBodyHandle
);
