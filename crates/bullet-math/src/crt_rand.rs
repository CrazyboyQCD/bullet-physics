//! Bit-for-bit replay of the C runtime `rand()` sequence:
//! `state = state·214013 + 2531011 (mod 2³²)`, returning
//! `(state >> 16) & 0x7fff`.
//!
//! Portable by construction: plain integer arithmetic, no platform or
//! toolchain dependency, same output on every target. The sequence replayed
//! is specifically the one MSVC's CRT ships — other libc implementations use
//! a different generator — which is what MSVC-built C++ probes draw from, so
//! this is what lets parity probes reproduce their random draws. Upstream's
//! own portable-random home is `LinearMath/btRandom.h`, which this module
//! maps to.

/// Replay state of that C runtime `rand()` sequence (MSVC's LCG).
#[derive(Clone, Copy, Debug)]
pub struct CrtRand {
    state: u32,
}

impl CrtRand {
    /// Equivalent of the default-seeded sequence (C requires `rand()` to
    /// behave as if `srand(1)` had been called).
    pub fn default_seeded() -> Self {
        Self { state: 1 }
    }

    /// Equivalent of `srand(seed)`.
    pub fn seeded(seed: u32) -> Self {
        Self { state: seed }
    }

    /// The next `rand()` value, `0..=32767`.
    pub fn next_rand(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(214013).wrapping_add(2531011);
        ((self.state >> 16) & 0x7fff) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::CrtRand;

    #[test]
    fn replays_the_msvc_crt_sequence() {
        let mut rng = CrtRand::default_seeded();
        let expected = [
            41, 18467, 6334, 26500, 19169, 15724, 11478, 29358, 26962, 24464,
        ];
        for want in expected {
            assert_eq!(rng.next_rand(), want);
        }
    }

    #[test]
    fn seeded_matches_srand_semantics() {
        let mut a = CrtRand::seeded(1806);
        let mut b = CrtRand { state: 0 };
        b.state = 1806;
        for _ in 0..8 {
            assert_eq!(a.next_rand(), b.next_rand());
        }
    }
}
