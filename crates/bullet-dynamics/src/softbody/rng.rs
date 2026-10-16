//! Bit-for-bit replay of the C runtime `rand()` sequence MSVC ships (design
//! §8): `state = state·214013 + 2531011 (mod 2³²)`, returning
//! `(state >> 16) & 0x7fff`.
//!
//! Portable by construction: plain integer arithmetic, no platform or toolchain
//! dependency, same output on every target. It is what lets the soft-body
//! helpers reproduce the numbers of the MSVC-built C++ probes (clang's libc
//! ships a different generator); the deterministic helpers prefer the built-in
//! LCG ([`super::body::SoftBody::randomize_constraints`]).

/// Replay state of that C runtime `rand()` sequence.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MsvcRand {
    state: u32,
}

impl MsvcRand {
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
