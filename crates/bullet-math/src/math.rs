//! Scalar math backend dispatch for [`Real`].
//!
//! The `f32`/`f64` transcendental methods (`sqrt`/`sin`/`acos`/...) exist in
//! `core` but lower to LLVM intrinsics that become calls to `sinf`/`acosf`/...
//! symbols, which do not link on `no_std` targets. Every engine call site for
//! those operations therefore goes through this module:
//!
//! - under the default `std` feature each function is an inlined pass-through
//!   to the standard method, bit-identical to the direct call it replaces;
//! - under the `libm` feature it routes to the pure-Rust `libm` crate instead
//!   (ulp-level differences from the platform CRT are expected there and only
//!   affect `libm` builds, never the default/std ledgers).
//!
//! `std` and `libm` are build-time alternatives; enabling both resolves to the
//! `std` branch here and inside glam alike. glam's own internal math is routed
//! independently via `glam/std` vs `glam/nostd-libm` (see `Cargo.toml`) and
//! needs no call-site changes.

use crate::Real;

/// Returns `sqrt(x)`.
#[inline]
#[cfg(feature = "std")]
pub fn sqrt(x: Real) -> Real {
    Real::sqrt(x)
}

/// Returns `sqrt(x)`.
#[inline]
#[cfg(not(feature = "std"))]
pub fn sqrt(x: Real) -> Real {
    #[cfg(not(feature = "f64"))]
    {
        libm::sqrtf(x)
    }
    #[cfg(feature = "f64")]
    {
        libm::sqrt(x)
    }
}

/// Returns `sin(x)`.
#[inline]
#[cfg(feature = "std")]
pub fn sin(x: Real) -> Real {
    Real::sin(x)
}

/// Returns `sin(x)`.
#[inline]
#[cfg(not(feature = "std"))]
pub fn sin(x: Real) -> Real {
    #[cfg(not(feature = "f64"))]
    {
        libm::sinf(x)
    }
    #[cfg(feature = "f64")]
    {
        libm::sin(x)
    }
}

/// Returns `cos(x)`.
#[inline]
#[cfg(feature = "std")]
pub fn cos(x: Real) -> Real {
    Real::cos(x)
}

/// Returns `cos(x)`.
#[inline]
#[cfg(not(feature = "std"))]
pub fn cos(x: Real) -> Real {
    #[cfg(not(feature = "f64"))]
    {
        libm::cosf(x)
    }
    #[cfg(feature = "f64")]
    {
        libm::cos(x)
    }
}

/// Returns the four-quadrant arctangent of `y` and `x`.
#[inline]
#[cfg(feature = "std")]
pub fn atan2(y: Real, x: Real) -> Real {
    Real::atan2(y, x)
}

/// Returns the four-quadrant arctangent of `y` and `x`.
#[inline]
#[cfg(not(feature = "std"))]
pub fn atan2(y: Real, x: Real) -> Real {
    #[cfg(not(feature = "f64"))]
    {
        libm::atan2f(y, x)
    }
    #[cfg(feature = "f64")]
    {
        libm::atan2(y, x)
    }
}

/// Returns `arcsin(x)`.
#[inline]
#[cfg(feature = "std")]
pub fn asin(x: Real) -> Real {
    Real::asin(x)
}

/// Returns `arcsin(x)`.
#[inline]
#[cfg(not(feature = "std"))]
pub fn asin(x: Real) -> Real {
    #[cfg(not(feature = "f64"))]
    {
        libm::asinf(x)
    }
    #[cfg(feature = "f64")]
    {
        libm::asin(x)
    }
}

/// Returns `arccos(x)`.
#[inline]
#[cfg(feature = "std")]
pub fn acos(x: Real) -> Real {
    Real::acos(x)
}

/// Returns `arccos(x)`.
#[inline]
#[cfg(not(feature = "std"))]
pub fn acos(x: Real) -> Real {
    #[cfg(not(feature = "f64"))]
    {
        libm::acosf(x)
    }
    #[cfg(feature = "f64")]
    {
        libm::acos(x)
    }
}

/// Returns `x` raised to `y`.
#[inline]
#[cfg(feature = "std")]
pub fn powf(x: Real, y: Real) -> Real {
    Real::powf(x, y)
}

/// Returns `x` raised to `y`.
#[inline]
#[cfg(not(feature = "std"))]
pub fn powf(x: Real, y: Real) -> Real {
    #[cfg(not(feature = "f64"))]
    {
        libm::powf(x, y)
    }
    #[cfg(feature = "f64")]
    {
        libm::pow(x, y)
    }
}
