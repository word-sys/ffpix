//! Thread-safe resource and decompression-bomb budget accounting.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{BudgetKind, FfpError, FfpResult};

const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetLimits {
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
    pub max_memory_bytes: u64,
    pub max_frames: u32,
    pub max_metadata_bytes: u64,
    pub max_compression_ratio: u32,
    pub max_inflate_bytes: u64,
    pub max_recursion_depth: u32,
    pub max_output_bytes: Option<u64>,
}

impl BudgetLimits {
    #[must_use]
    pub const fn preset_default() -> Self {
        Self {
            max_width: 65_536,
            max_height: 65_536,
            max_pixels: 256_000_000,
            max_memory_bytes: 256 * MIB,
            max_frames: 10_000,
            max_metadata_bytes: 16 * MIB,
            max_compression_ratio: 100,
            max_inflate_bytes: 512 * MIB,
            max_recursion_depth: 64,
            max_output_bytes: None,
        }
    }

    #[must_use]
    pub const fn preset_strict() -> Self {
        Self {
            max_width: 16_384,
            max_height: 16_384,
            max_pixels: 50_000_000,
            max_memory_bytes: 64 * MIB,
            max_frames: 500,
            max_metadata_bytes: 2 * MIB,
            max_compression_ratio: 50,
            max_inflate_bytes: 128 * MIB,
            max_recursion_depth: 32,
            max_output_bytes: Some(128 * MIB),
        }
    }
}

impl Default for BudgetLimits {
    fn default() -> Self {
        Self::preset_default()
    }
}

#[derive(Debug)]
struct BudgetInner {
    limits: BudgetLimits,
    current_memory: AtomicU64,
    peak_memory: AtomicU64,
}

impl BudgetInner {
    fn try_reserve(&self, delta: u64) -> FfpResult<()> {
        let limit = self.limits.max_memory_bytes;
        let mut cur = self.current_memory.load(Ordering::Acquire);
        loop {
            let next = cur
                .checked_add(delta)
                .ok_or_else(|| FfpError::overflow("budget memory reservation overflow"))?;
            if next > limit {
                return Err(FfpError::budget_exceeded(
                    BudgetKind::MemoryBytes,
                    limit,
                    next,
                ));
            }
            match self.current_memory.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak_memory.fetch_max(next, Ordering::AcqRel);
                    return Ok(());
                }
                Err(observed) => cur = observed,
            }
        }
    }

    fn release(&self, bytes: u64) {
        if bytes > 0 {
            self.current_memory.fetch_sub(bytes, Ordering::AcqRel);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Budget {
    inner: Arc<BudgetInner>,
}

impl Budget {
    #[must_use]
    pub fn new(limits: BudgetLimits) -> Self {
        Self {
            inner: Arc::new(BudgetInner {
                limits,
                current_memory: AtomicU64::new(0),
                peak_memory: AtomicU64::new(0),
            }),
        }
    }

    #[must_use]
    pub fn preset_default() -> Self {
        Self::new(BudgetLimits::preset_default())
    }

    #[must_use]
    pub fn preset_strict() -> Self {
        Self::new(BudgetLimits::preset_strict())
    }

    #[must_use]
    pub fn limits(&self) -> BudgetLimits {
        self.inner.limits
    }

    #[must_use]
    pub fn current_memory_bytes(&self) -> u64 {
        self.inner.current_memory.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn peak_memory_bytes(&self) -> u64 {
        self.inner.peak_memory.load(Ordering::Acquire)
    }

    pub fn check_dimensions(&self, width: u32, height: u32) -> FfpResult<()> {
        if width == 0 || height == 0 {
            return Err(FfpError::invalid_input(format!(
                "image dimensions must be non-zero, got {width}x{height}"
            )));
        }
        let limits = self.inner.limits;
        if width > limits.max_width {
            return Err(FfpError::budget_exceeded(
                BudgetKind::Width,
                u64::from(limits.max_width),
                u64::from(width),
            ));
        }
        if height > limits.max_height {
            return Err(FfpError::budget_exceeded(
                BudgetKind::Height,
                u64::from(limits.max_height),
                u64::from(height),
            ));
        }
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| FfpError::overflow("pixel count overflow"))?;
        if pixels > limits.max_pixels {
            return Err(FfpError::budget_exceeded(
                BudgetKind::Pixels,
                limits.max_pixels,
                pixels,
            ));
        }
        Ok(())
    }

    pub fn check_frames(&self, frames: u32) -> FfpResult<()> {
        if frames == 0 {
            return Err(FfpError::invalid_input("frame count must be at least 1"));
        }
        let limit = self.inner.limits.max_frames;
        if frames > limit {
            return Err(FfpError::budget_exceeded(
                BudgetKind::Frames,
                u64::from(limit),
                u64::from(frames),
            ));
        }
        Ok(())
    }

    pub fn check_metadata_bytes(&self, bytes: u64) -> FfpResult<()> {
        let limit = self.inner.limits.max_metadata_bytes;
        if bytes > limit {
            return Err(FfpError::budget_exceeded(
                BudgetKind::MetadataBytes,
                limit,
                bytes,
            ));
        }
        Ok(())
    }

    pub fn check_inflate_bytes(&self, bytes: u64) -> FfpResult<()> {
        let limit = self.inner.limits.max_inflate_bytes;
        if bytes > limit {
            return Err(FfpError::budget_exceeded(
                BudgetKind::InflateBytes,
                limit,
                bytes,
            ));
        }
        Ok(())
    }

    pub fn check_recursion_depth(&self, depth: u32) -> FfpResult<()> {
        let limit = self.inner.limits.max_recursion_depth;
        if depth > limit {
            return Err(FfpError::budget_exceeded(
                BudgetKind::RecursionDepth,
                u64::from(limit),
                u64::from(depth),
            ));
        }
        Ok(())
    }

    pub fn check_output_bytes(&self, bytes: u64) -> FfpResult<()> {
        if let Some(limit) = self.inner.limits.max_output_bytes
            && bytes > limit
        {
            return Err(FfpError::budget_exceeded(
                BudgetKind::OutputBytes,
                limit,
                bytes,
            ));
        }
        Ok(())
    }

    /// Rejects streams exceeding `max_inflate_bytes` or `max_compression_ratio`
    /// once `bytes_out` surpasses `grace_bytes`.
    pub fn check_compression_ratio(
        &self,
        bytes_in: u64,
        bytes_out: u64,
        grace_bytes: u64,
    ) -> FfpResult<()> {
        self.check_inflate_bytes(bytes_out)?;
        if bytes_out <= grace_bytes {
            return Ok(());
        }
        let limit = u64::from(self.inner.limits.max_compression_ratio);
        let denom = bytes_in.max(1);
        let max_allowed = denom.saturating_mul(limit);
        if bytes_out > max_allowed {
            let ratio = (bytes_out / denom).max(limit.saturating_add(1));
            return Err(FfpError::budget_exceeded(
                BudgetKind::CompressionRatio,
                limit,
                ratio,
            ));
        }
        Ok(())
    }

    pub fn reserve_memory(&self, bytes: u64) -> FfpResult<MemoryReservation> {
        self.inner.try_reserve(bytes)?;
        Ok(MemoryReservation {
            inner: Arc::clone(&self.inner),
            bytes,
        })
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self::preset_default()
    }
}

#[derive(Debug)]
pub struct MemoryReservation {
    inner: Arc<BudgetInner>,
    bytes: u64,
}

impl MemoryReservation {
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn resize(&mut self, new_bytes: u64) -> FfpResult<()> {
        if new_bytes == self.bytes {
            return Ok(());
        }
        if new_bytes < self.bytes {
            self.inner.release(self.bytes - new_bytes);
            self.bytes = new_bytes;
            return Ok(());
        }
        let delta = new_bytes - self.bytes;
        self.inner.try_reserve(delta)?;
        self.bytes = new_bytes;
        Ok(())
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.inner.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatusCode;
    use std::thread;

    #[test]
    fn dimension_frame_metadata_and_recursion_checks() {
        let budget = Budget::preset_strict();
        assert!(budget.check_dimensions(1920, 1080).is_ok());
        assert_eq!(
            budget
                .check_dimensions(0, 100)
                .err()
                .map(|e| e.status_code()),
            Some(StatusCode::InvalidInput)
        );
        assert!(matches!(
            budget.check_dimensions(20_000, 100),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::Width,
                ..
            })
        ));
        assert!(matches!(
            budget.check_dimensions(100, 20_000),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::Height,
                ..
            })
        ));
        assert!(matches!(
            budget.check_dimensions(10_000, 10_000),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::Pixels,
                ..
            })
        ));

        assert!(budget.check_frames(1).is_ok());
        assert!(budget.check_frames(0).is_err());
        assert!(matches!(
            budget.check_frames(501),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::Frames,
                ..
            })
        ));

        assert!(budget.check_metadata_bytes(1024).is_ok());
        assert!(matches!(
            budget.check_metadata_bytes(10 * MIB),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::MetadataBytes,
                ..
            })
        ));

        assert!(budget.check_recursion_depth(32).is_ok());
        assert!(matches!(
            budget.check_recursion_depth(33),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::RecursionDepth,
                ..
            })
        ));

        assert!(matches!(
            budget.check_output_bytes(256 * MIB),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::OutputBytes,
                ..
            })
        ));
    }

    #[test]
    fn compression_ratio_and_inflate_rejection() {
        let budget = Budget::preset_default();
        let grace = 64 * 1024;

        // Small solid outputs within grace threshold pass regardless of ratio.
        assert!(budget.check_compression_ratio(32, 60_000, grace).is_ok());
        // Legitimate 20:1 ratio above grace passes.
        assert!(
            budget
                .check_compression_ratio(50_000, 1_000_000, grace)
                .is_ok()
        );
        // 1000:1 bomb above grace is rejected.
        assert!(matches!(
            budget.check_compression_ratio(1_000, 1_000_000, grace),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::CompressionRatio,
                ..
            })
        ));
        // Absolute inflate limit triggers even at 1:1 ratio.
        assert!(matches!(
            budget.check_compression_ratio(600 * MIB, 600 * MIB, grace),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::InflateBytes,
                ..
            })
        ));
    }

    #[test]
    fn raii_memory_reservation_peak_tracking_and_overflow_guard() {
        let limits = BudgetLimits {
            max_memory_bytes: 1000,
            ..BudgetLimits::preset_default()
        };
        let budget = Budget::new(limits);
        assert_eq!(budget.current_memory_bytes(), 0);
        assert_eq!(budget.peak_memory_bytes(), 0);

        {
            let res1 = budget.reserve_memory(400);
            assert!(res1.is_ok());
            assert_eq!(budget.current_memory_bytes(), 400);
            assert_eq!(budget.peak_memory_bytes(), 400);

            let mut res2 = match budget.reserve_memory(500) {
                Ok(r) => r,
                Err(_) => return,
            };
            assert_eq!(budget.current_memory_bytes(), 900);
            assert_eq!(budget.peak_memory_bytes(), 900);

            // Exceeding cap fails without mutating current or peak.
            assert!(matches!(
                budget.reserve_memory(101),
                Err(FfpError::BudgetExceeded {
                    kind: BudgetKind::MemoryBytes,
                    limit: 1000,
                    requested: 1001,
                })
            ));
            assert_eq!(budget.current_memory_bytes(), 900);
            assert_eq!(budget.peak_memory_bytes(), 900);

            // u64::MAX overflow guard.
            assert_eq!(
                budget
                    .reserve_memory(u64::MAX)
                    .err()
                    .map(|e| e.status_code()),
                Some(StatusCode::Overflow)
            );

            // Resize down and back up.
            assert!(res2.resize(200).is_ok());
            assert_eq!(res2.bytes(), 200);
            assert_eq!(budget.current_memory_bytes(), 600);
            assert_eq!(budget.peak_memory_bytes(), 900);

            assert!(res2.resize(600).is_ok());
            assert_eq!(budget.current_memory_bytes(), 1000);
            assert_eq!(budget.peak_memory_bytes(), 1000);
            assert!(res2.resize(601).is_err());
        }

        // Dropping all reservations restores current memory to zero while preserving peak.
        assert_eq!(budget.current_memory_bytes(), 0);
        assert_eq!(budget.peak_memory_bytes(), 1000);
    }

    #[test]
    fn multithreaded_contention_never_exceeds_limit_and_returns_to_zero() {
        let cap = 4096u64;
        let limits = BudgetLimits {
            max_memory_bytes: cap,
            ..BudgetLimits::preset_default()
        };
        let budget = Budget::new(limits);

        let mut handles = Vec::new();
        for tid in 0..8u64 {
            let b = budget.clone();
            handles.push(thread::spawn(move || {
                let mut succeeded = 0u64;
                for iter in 0..200u64 {
                    let size = 256 + ((tid * 64 + iter * 17) % 1024);
                    if let Ok(mut guard) = b.reserve_memory(size) {
                        succeeded += 1;
                        assert!(b.current_memory_bytes() <= cap);
                        let _ = guard.resize(size / 2);
                        assert!(b.current_memory_bytes() <= cap);
                        drop(guard);
                    }
                }
                succeeded
            }));
        }

        let mut total_succeeded = 0u64;
        for h in handles {
            if let Ok(count) = h.join() {
                total_succeeded += count;
            }
        }

        assert!(total_succeeded > 0);
        assert_eq!(budget.current_memory_bytes(), 0);
        assert!(budget.peak_memory_bytes() <= cap);
        assert!(budget.peak_memory_bytes() > 0);
    }
}
