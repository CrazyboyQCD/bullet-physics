//! MSVC `rand()` bit-for-bit replay (design §8): the C runtime generator
//! `state = state·214013 + 2531011 (mod 2³²)`, returning `(state >> 16) &
//! 0x7fff`. Only meaningful for MSVC-built probes (clang's libc uses a
//! different generator); the deterministic soft-body helpers prefer the
//! built-in LCG ([`super::body::SoftBody::randomize_constraints`]).

/// MSVC CRT `rand()` state.
#[derive(Clone, Copy, Debug)]
pub struct MsvcRand {
    state: u32,
}

impl MsvcRand {
    /// Equivalent of calling `srand(seed)` (MSVC sets the state directly).
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Equivalent of the default-seeded sequence (C requires `rand()` to
    /// behave as if `srand(1)` had been called).
    pub fn default_seeded() -> Self {
        Self { state: 1 }
    }

    /// The next `rand()` value, `0..=32767`.
    pub fn next_rand(&mut self) -> i32 {
        self.state = self
            .state
            .wrapping_mul(214013)
            .wrapping_add(2531011);
        ((self.state >> 16) & 0x7fff) as i32
    }

    /// A random `btScalar` in `[0, 1)` like `rand() / (RAND_MAX + 1)` (the
    /// `CreatePatch` perturbation divides by `RAND_MAX` itself, so callers
    /// that need that exact expression use [`Self::next_rand`] directly).
    pub fn next_unit(&mut self) -> f64 {
        self.next_rand() as f64 / 32768.0
    }
}
