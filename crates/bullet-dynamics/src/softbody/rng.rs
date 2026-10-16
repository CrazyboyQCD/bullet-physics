//! Bit-for-bit replay of the C runtime `rand()` sequence (design §8):
//! `state = state·214013 + 2531011 (mod 2³²)`, returning
//! `(state >> 16) & 0x7fff`.
//!
//! Portable by construction: plain integer arithmetic, no platform or toolchain
//! dependency, same output on every target. The sequence replayed is
//! specifically the one MSVC's CRT ships — other libc implementations use a
//! different generator — which is what the MSVC-built C++ probes draw from, so
//! this is what lets the soft-body helpers reproduce their numbers. The
//! deterministic helpers prefer the built-in LCG
//! ([`super::body::SoftBody::randomize_constraints`]).

/// Replay state of that C runtime `rand()` sequence (MSVC's LCG).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CrtRand {
    state: u32,
}

impl CrtRand {
    /// Equivalent of the default-seeded sequence (C requires `rand()` to
    /// behave as if `srand(1)` had been called).
    pub(crate) fn default_seeded() -> Self {
        Self { state: 1 }
    }

    /// The next `rand()` value, `0..=32767`.
    pub(crate) fn next_rand(&mut self) -> i32 {
        self.state = self
            .state
            .wrapping_mul(214013)
            .wrapping_add(2531011);
        ((self.state >> 16) & 0x7fff) as i32
    }
}
