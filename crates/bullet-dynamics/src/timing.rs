//! Step timing instrumentation: per-stage wall-clock sampling for
//! [`crate::world::PhysicsWorld::step_simulation`]. The instrumentation
//! points mirror Bullet's `BT_PROFILE` spans in
//! `btDiscreteDynamicsWorld::stepSimulation`.
//!
//! The sampler never touches the numerical path, so timing cannot change
//! the simulation result. Under the default `std` feature the clock is the
//! standard library's monotonic `Instant`; `no_std` hosts inject their
//! platform clock through [`PhysicsWorld::set_step_clock`].

use core::time::Duration;

use crate::world::PhysicsWorld;

/// One instrumented span of [`PhysicsWorld::step_simulation`], in execution
/// order (the numbered stage comments inside the step; the sampling points
/// mirror Bullet's `BT_PROFILE` instrumentation in
/// `btDiscreteDynamicsWorld::stepSimulation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepStage {
    /// Gravity + external-force velocity integration (stages 1/1m).
    IntegrateVelocities,
    /// Predicted poses + broad-phase AABB refresh (stages 2/3/3m).
    UpdateAabbs,
    /// Broad-phase pair query (stage 4).
    Broadphase,
    /// Simulation islands + island sleep decisions (stage 4b).
    Islands,
    /// Narrow phase + joint row assembly (stages 5/5a).
    Narrowphase,
    /// Constraint solving + warm-start cache writeback (stages 6/6b/6bm).
    Solve,
    /// Deactivation, CCD clamping, position integration, force clearing
    /// (stages 6c/7/8/8m).
    IntegratePositions,
}

/// Every stage in execution order.
pub const STEP_STAGES: [StepStage; 7] = [
    StepStage::IntegrateVelocities,
    StepStage::UpdateAabbs,
    StepStage::Broadphase,
    StepStage::Islands,
    StepStage::Narrowphase,
    StepStage::Solve,
    StepStage::IntegratePositions,
];

/// Cumulative wall-clock durations per [`StepStage`], accumulated over every
/// step since the last reset. All zeros until enabled with
/// [`PhysicsWorld::enable_step_timing`]; while disabled the per-step sampling
/// cost is one branch per stage, while enabled it is ~8 clock reads. Timing
/// never touches the numerical path, so it cannot change the simulation
/// result.
#[derive(Clone, Debug)]
pub struct StepTimings {
    pub(crate) enabled: bool,
    stages: [Duration; 7],
}

impl Default for StepTimings {
    fn default() -> Self {
        Self {
            enabled: false,
            stages: [Duration::from_secs(0); 7],
        }
    }
}

impl StepTimings {
    /// Cumulative duration of one stage.
    pub fn stage(&self, stage: StepStage) -> Duration {
        self.stages[stage as usize]
    }

    /// Sum over all stages (the instrumented portion of every step).
    pub fn total(&self) -> Duration {
        self.stages.iter().copied().sum()
    }

    pub(crate) fn add(&mut self, stage: StepStage, span: Duration) {
        self.stages[stage as usize] += span;
    }
}

/// Stage-boundary sampler used inside `step_simulation`; a no-op unless
/// timing was enabled when the step started.
pub(crate) struct StepClock {
    enabled: bool,
    now: fn() -> Duration,
    last: Option<Duration>,
}

impl StepClock {
    pub(crate) fn new(enabled: bool, now: fn() -> Duration) -> Self {
        Self {
            enabled,
            now,
            last: enabled.then(now),
        }
    }

    /// Close the previous stage (accumulating the elapsed span under
    /// `stage`) and restart the clock for the next one. A non-monotonic
    /// injected clock makes the offending span saturate to zero: the
    /// sampler never panics and never produces a negative duration.
    #[inline]
    pub(crate) fn mark(&mut self, timings: &mut StepTimings, stage: StepStage) {
        if !self.enabled {
            return;
        }
        let now = (self.now)();
        if let Some(last) = self.last.replace(now) {
            timings.add(stage, now.saturating_sub(last));
        }
    }
}

/// Default clock source for step timings: the standard library's monotonic
/// `Instant`, measured process-relative from the first read. Only used under
/// the `std` feature.
#[cfg(feature = "std")]
pub(crate) fn default_step_clock() -> Duration {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    Instant::now().duration_since(*START.get_or_init(Instant::now))
}

/// Default clock source outside `std`: always zero. Hosts inject a real
/// monotonic clock via [`PhysicsWorld::set_step_clock`].
#[cfg(not(feature = "std"))]
pub(crate) fn default_step_clock() -> Duration {
    Duration::ZERO
}

impl PhysicsWorld {
    /// Start accumulating per-stage wall-clock timings in
    /// `step_simulation` (Bullet's quickprof instrumentation points).
    /// Off by default; enabling costs ~8 clock reads per step. The clock
    /// source defaults to the std monotonic `Instant` and can be replaced
    /// with [`PhysicsWorld::set_step_clock`] (without the `std` feature
    /// the default clock reads zero).
    pub fn enable_step_timing(&mut self) {
        self.step_timings.enabled = true;
    }

    /// Replace the clock source feeding [`PhysicsWorld::step_timings`].
    ///
    /// The default under the `std` feature is the standard library's
    /// monotonic `Instant` (process-relative); without `std` the default
    /// reads zero, so `no_std` hosts inject their platform's monotonic
    /// clock here.
    ///
    /// # Contract for injected clocks
    ///
    /// - Return the time elapsed since some arbitrary fixed point (a
    ///   monotonic tick count as a `Duration`); only differences between
    ///   successive reads are used, so the absolute origin does not
    ///   matter.
    /// - Be monotonic if accurate timings matter: a read that goes
    ///   backwards makes the offending stage span saturate to zero — the
    ///   sampler never panics and never produces a negative duration,
    ///   but cumulative totals then undercount.
    /// - Called only while step timing is enabled, roughly 8 times per
    ///   `step_simulation` (once per stage boundary); never called when
    ///   timing is disabled.
    /// - Swapping clocks mid-step is tolerated: the first span after the
    ///   switch may read as zero, and the next full step is exact again.
    pub fn set_step_clock(&mut self, now: fn() -> Duration) {
        self.step_clock_fn = now;
    }

    /// Cumulative per-stage durations over all steps since timing was
    /// enabled (zeros while disabled).
    pub fn step_timings(&self) -> &StepTimings {
        &self.step_timings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The injected step-timing clock: monotonic reads accumulate exact
    /// spans, a backwards read saturates to zero, and disabled timing
    /// never reads the clock at all.
    #[test]
    fn step_clock_injection_contract() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        static IDX: AtomicUsize = AtomicUsize::new(0);
        // First read seeds the sampler's start; the next 7 are stage
        // marks. The third mark goes backwards (115 -> 110).
        const TICKS: [u64; 8] = [100, 110, 115, 110, 130, 140, 150, 160];
        fn fake_clock() -> Duration {
            Duration::from_nanos(TICKS[IDX.fetch_add(1, Ordering::SeqCst)])
        }

        let mut world = PhysicsWorld::new();
        // Disabled timing: the clock is never read.
        world.set_step_clock(fake_clock);
        world.step_simulation(1.0 / 60.0);
        assert_eq!(IDX.load(Ordering::SeqCst), 0);
        assert_eq!(world.step_timings().total(), Duration::ZERO);

        // Enabled: spans accumulate; the backwards span clamps to zero.
        IDX.store(0, Ordering::SeqCst);
        world.enable_step_timing();
        world.step_simulation(1.0 / 60.0);
        let t = world.step_timings();
        assert_eq!(
            t.stage(StepStage::IntegrateVelocities),
            Duration::from_nanos(10)
        );
        assert_eq!(t.stage(StepStage::UpdateAabbs), Duration::from_nanos(5));
        assert_eq!(t.stage(StepStage::Broadphase), Duration::ZERO);
        assert_eq!(t.stage(StepStage::Islands), Duration::from_nanos(20));
        assert_eq!(t.stage(StepStage::Narrowphase), Duration::from_nanos(10));
        assert_eq!(t.stage(StepStage::Solve), Duration::from_nanos(10));
        assert_eq!(
            t.stage(StepStage::IntegratePositions),
            Duration::from_nanos(10)
        );
        assert_eq!(t.total(), Duration::from_nanos(65));

        // Swapping clocks mid-session is tolerated (fresh start point).
        IDX.store(0, Ordering::SeqCst);
        world.set_step_clock(fake_clock);
        world.step_simulation(1.0 / 60.0);
    }
}
