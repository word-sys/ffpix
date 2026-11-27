//! Rolling-window scanline ring buffer for sequential pull sources.

use std::sync::{Arc, Mutex};

use ffpix_core::{
    Budget, FfpError, FfpResult, MemoryReservation, PixelFormat, Rect, Size, TileViewMut,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StripCacheStats {
    pub rows_produced: u64,
    pub rows_served: u64,
    pub window_rows: u32,
    pub frontier_start_y: u32,
    pub next_produce_y: u32,
    pub allocated_bytes: u64,
}

/// Bounded rolling-window scanline cache for `AccessPattern::Sequential` nodes.
///
/// Advances monotonically top-to-bottom across `image_size.h` scanlines in a
/// single ring buffer sized `O(width * window_rows * bpp)`, rejecting backward
/// pulls past the retired frontier.
#[derive(Debug)]
pub struct StripCache {
    image_size: Size,
    format: PixelFormat,
    row_bytes: usize,
    window_rows: u32,
    ring_rows: usize,
    ring: Vec<u8>,
    _reservation: MemoryReservation,
    next_produce_y: u32,
    rows_produced: u64,
    rows_served: u64,
}

impl StripCache {
    pub fn new(
        image_size: Size,
        format: PixelFormat,
        window_rows: u32,
        budget: &Budget,
    ) -> FfpResult<Self> {
        if image_size.is_empty() {
            return Err(FfpError::invalid_input(format!(
                "strip cache image dimensions must be non-zero, got {image_size}"
            )));
        }
        if window_rows == 0 {
            return Err(FfpError::invalid_argument(
                "strip cache window_rows must be at least 1",
            ));
        }

        let row_bytes = format.checked_row_bytes(image_size.w)?;
        let effective_rows = window_rows.min(image_size.h);
        let ring_rows = usize::try_from(effective_rows)
            .map_err(|_| FfpError::overflow("strip cache window_rows exceeds usize"))?;
        let slab_bytes = row_bytes
            .checked_mul(ring_rows)
            .ok_or_else(|| FfpError::overflow("strip cache ring buffer byte size overflow"))?;
        if slab_bytes > isize::MAX as usize {
            return Err(FfpError::overflow(
                "strip cache ring buffer exceeds isize::MAX",
            ));
        }

        let reservation = budget.reserve_memory(slab_bytes as u64)?;
        let ring = vec![0u8; slab_bytes];

        Ok(Self {
            image_size,
            format,
            row_bytes,
            window_rows: effective_rows,
            ring_rows,
            ring,
            _reservation: reservation,
            next_produce_y: 0,
            rows_produced: 0,
            rows_served: 0,
        })
    }

    #[must_use]
    pub const fn image_size(&self) -> Size {
        self.image_size
    }

    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    #[must_use]
    pub const fn window_rows(&self) -> u32 {
        self.window_rows
    }

    #[must_use]
    pub const fn next_produce_y(&self) -> u32 {
        self.next_produce_y
    }

    /// Earliest scanline index still retained in the ring buffer.
    #[must_use]
    pub const fn frontier_start_y(&self) -> u32 {
        self.next_produce_y.saturating_sub(self.window_rows)
    }

    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.ring.len() as u64
    }

    #[must_use]
    pub fn stats(&self) -> StripCacheStats {
        StripCacheStats {
            rows_produced: self.rows_produced,
            rows_served: self.rows_served,
            window_rows: self.window_rows,
            frontier_start_y: self.frontier_start_y(),
            next_produce_y: self.next_produce_y,
            allocated_bytes: self.allocated_bytes(),
        }
    }

    /// Fills `out` with pixels for `req`, advancing the sequential producer as
    /// needed and returning an error if `req` reaches behind the retired frontier.
    pub fn pull_rect<F>(
        &mut self,
        req: Rect,
        out: &mut TileViewMut<'_>,
        mut produce_strip: F,
    ) -> FfpResult<()>
    where
        F: FnMut(Rect, &mut TileViewMut<'_>) -> FfpResult<()>,
    {
        req.validate()?;
        if out.format() != self.format {
            return Err(FfpError::format_mismatch(format!(
                "strip cache format {} does not match output {}",
                self.format,
                out.format()
            )));
        }
        if out.rect().size() != req.size() {
            return Err(FfpError::dimension_mismatch(
                req.w,
                req.h,
                out.rect().w,
                out.rect().h,
            ));
        }
        if !self.image_size.to_rect().contains_rect(req) {
            return Err(FfpError::invalid_argument(format!(
                "strip cache request {req} outside image bounds {}",
                self.image_size
            )));
        }
        if req.is_empty() {
            return Ok(());
        }
        if req.h > self.window_rows {
            return Err(FfpError::invalid_argument(format!(
                "strip request height {} exceeds window_rows {}",
                req.h, self.window_rows
            )));
        }

        let req_end_y = req
            .bottom()
            .ok_or_else(|| FfpError::overflow("strip request bottom overflow"))?;

        // Check against the frontier that will remain once we advance to req_end_y.
        let projected_frontier = req_end_y
            .max(self.next_produce_y)
            .saturating_sub(self.window_rows);
        if req.y < projected_frontier {
            return Err(FfpError::invalid_argument(format!(
                "sequential backward seek to row {} past retired frontier {}",
                req.y, projected_frontier
            )));
        }

        while self.next_produce_y < req_end_y {
            let slot = (self.next_produce_y as usize) % self.ring_rows;
            let remaining = req_end_y - self.next_produce_y;
            let contiguous = remaining.min((self.ring_rows - slot) as u32);
            let byte_start = slot * self.row_bytes;
            let byte_len = (contiguous as usize) * self.row_bytes;
            let strip_rect = Rect::new(0, self.next_produce_y, self.image_size.w, contiguous);

            let slice = self
                .ring
                .get_mut(byte_start..byte_start + byte_len)
                .ok_or_else(|| FfpError::internal("strip cache ring slice out of bounds"))?;
            let mut strip_view =
                TileViewMut::from_slice(strip_rect, self.format, self.row_bytes, slice)?;
            produce_strip(strip_rect, &mut strip_view)?;

            self.next_produce_y += contiguous;
            self.rows_produced += u64::from(contiguous);
        }

        let bpp = self.format.bytes_per_pixel();
        let x_start = (req.x as usize) * bpp;
        let copy_bytes = (req.w as usize) * bpp;

        for ly in 0..req.h {
            let src_y = req.y + ly;
            let slot = (src_y as usize) % self.ring_rows;
            let row_offset = slot * self.row_bytes;
            let src_slice = self
                .ring
                .get(row_offset + x_start..row_offset + x_start + copy_bytes)
                .ok_or_else(|| FfpError::internal("strip cache read slice out of bounds"))?;
            out.row_mut(ly)?.copy_from_slice(src_slice);
        }
        self.rows_served += u64::from(req.h);

        Ok(())
    }
}

/// Thread-safe shared wrapper around [`StripCache`].
#[derive(Debug, Clone)]
pub struct SharedStripCache {
    inner: Arc<Mutex<StripCache>>,
}

impl SharedStripCache {
    pub fn new(
        image_size: Size,
        format: PixelFormat,
        window_rows: u32,
        budget: &Budget,
    ) -> FfpResult<Self> {
        let cache = StripCache::new(image_size, format, window_rows, budget)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(cache)),
        })
    }

    pub fn pull_rect<F>(
        &self,
        req: Rect,
        out: &mut TileViewMut<'_>,
        produce_strip: F,
    ) -> FfpResult<()>
    where
        F: FnMut(Rect, &mut TileViewMut<'_>) -> FfpResult<()>,
    {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| FfpError::internal("strip cache mutex poisoned"))?;
        guard.pull_rect(req, out, produce_strip)
    }

    #[must_use]
    pub fn stats(&self) -> StripCacheStats {
        self.inner.lock().map(|g| g.stats()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffpix_core::{
        BudgetLimits, StatusCode, SynthPattern, TileBuffer, eval_pattern_rgba8, fill_tile_pattern,
    };
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn streams_1024x4096_virtual_image_with_9_row_halo_below_512_kib() -> FfpResult<()> {
        let size = Size::new(1024, 4096);
        let window_rows = 9u32;
        let strip_h = 4u32;
        let halo_radius = 2u32;

        // Hard 512 KiB budget proves memory stays strictly bounded far below full 16 MiB image.
        let budget = Budget::new(BudgetLimits {
            max_memory_bytes: 512 * 1024,
            ..BudgetLimits::preset_default()
        });

        {
            let mut cache = StripCache::new(size, PixelFormat::RGBA8, window_rows, &budget)?;
            // 1024 * 4 * 9 = 36,864 bytes.
            assert_eq!(cache.allocated_bytes(), 36_864);

            let max_chunk_rows = AtomicU32::new(0);
            let pattern = SynthPattern::HorizontalGradient;

            // Scratch output buffer holds at most 8 rows (strip_h + 2*halo_radius = 8 <= 9).
            let mut y = 0u32;
            while y < size.h {
                let out_h = strip_h.min(size.h - y);
                let out_rect = Rect::new(0, y, size.w, out_h);
                let halo_rect = out_rect.expand_halo(0, halo_radius, 0, halo_radius, size);
                let mut halo_buf = TileBuffer::new_zeroed(halo_rect, PixelFormat::RGBA8, &budget)?;

                cache.pull_rect(halo_rect, &mut halo_buf.as_view_mut(), |strip, view| {
                    max_chunk_rows.fetch_max(strip.h, Ordering::Relaxed);
                    fill_tile_pattern(view, size, pattern)
                })?;

                // Spot-check top-left, middle, and right-edge pixels of the requested window.
                for ly in 0..halo_rect.h {
                    let row = halo_buf.row(ly)?;
                    let wy = halo_rect.y + ly;
                    for wx in [0u32, 511, 1023] {
                        let off = (wx as usize) * 4;
                        let expected = eval_pattern_rgba8(pattern, wx, wy, size.w, size.h);
                        assert_eq!(&row[off..off + 4], &expected);
                    }
                }

                y += out_h;
            }

            let stats = cache.stats();
            // Every row of the 4096-row image was decoded strictly once.
            assert_eq!(stats.rows_produced, u64::from(size.h));
            assert_eq!(stats.next_produce_y, size.h);
            assert!(max_chunk_rows.load(Ordering::Relaxed) <= window_rows);
            assert!(budget.peak_memory_bytes() < 512 * 1024);
        }

        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn rejects_backward_seeks_past_retired_frontier_and_validates_bounds() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let size = Size::new(64, 32);
        let mut cache = StripCache::new(size, PixelFormat::RGBA8, 8, &budget)?;

        let mut tile = TileBuffer::new_zeroed(Rect::new(0, 0, 64, 8), PixelFormat::RGBA8, &budget)?;
        cache.pull_rect(
            Rect::new(0, 0, 64, 8),
            &mut tile.as_view_mut(),
            |_, view| fill_tile_pattern(view, size, SynthPattern::VerticalGradient),
        )?;
        assert_eq!(cache.frontier_start_y(), 0);

        // Advance to row 16: window holds rows [8..16], frontier is 8.
        let mut tile2 =
            TileBuffer::new_zeroed(Rect::new(16, 12, 16, 4), PixelFormat::RGBA8, &budget)?;
        cache.pull_rect(
            Rect::new(16, 12, 16, 4),
            &mut tile2.as_view_mut(),
            |_, view| fill_tile_pattern(view, size, SynthPattern::VerticalGradient),
        )?;
        assert_eq!(cache.frontier_start_y(), 8);

        // Pulling row 8 still succeeds (inside active window).
        let mut sub_ok =
            TileBuffer::new_zeroed(Rect::new(0, 8, 32, 4), PixelFormat::RGBA8, &budget)?;
        assert!(
            cache
                .pull_rect(
                    Rect::new(0, 8, 32, 4),
                    &mut sub_ok.as_view_mut(),
                    |_, _| Ok(())
                )
                .is_ok()
        );
        // Sub-rect crop matches exact vertical gradient pixels without re-invoking producer.
        for ly in 0..4u32 {
            let expected = eval_pattern_rgba8(SynthPattern::VerticalGradient, 0, 8 + ly, 64, 32);
            assert_eq!(&sub_ok.row(ly)?[0..4], &expected);
        }

        // Pulling row 7 (before frontier 8) fails with structured InvalidArgument.
        let mut back = TileBuffer::new_zeroed(Rect::new(0, 7, 16, 2), PixelFormat::RGBA8, &budget)?;
        let err = cache.pull_rect(Rect::new(0, 7, 16, 2), &mut back.as_view_mut(), |_, _| {
            Ok(())
        });
        assert_eq!(
            err.err().map(|e| e.status_code()),
            Some(StatusCode::InvalidArgument)
        );

        // Request taller than window_rows fails cleanly.
        let mut tall =
            TileBuffer::new_zeroed(Rect::new(0, 10, 16, 9), PixelFormat::RGBA8, &budget)?;
        assert_eq!(
            cache
                .pull_rect(Rect::new(0, 10, 16, 9), &mut tall.as_view_mut(), |_, _| Ok(
                    ()
                ))
                .err()
                .map(|e| e.status_code()),
            Some(StatusCode::InvalidArgument)
        );

        // Invalid construction parameters rejected cleanly.
        assert!(StripCache::new(Size::new(0, 10), PixelFormat::RGBA8, 8, &budget).is_err());
        assert!(StripCache::new(size, PixelFormat::RGBA8, 0, &budget).is_err());
        assert!(StripCache::new(Size::new(u32::MAX, 10), PixelFormat::RGBA8, 8, &budget).is_err());

        drop(tile);
        drop(tile2);
        drop(sub_ok);
        drop(back);
        drop(tall);
        drop(cache);
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn shared_strip_cache_serves_overlapping_horizontal_tiles() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let size = Size::new(64, 32);
        let shared = SharedStripCache::new(size, PixelFormat::RGB8, 8, &budget)?;

        for y in (0..32u32).step_by(4) {
            for x in (0..64u32).step_by(32) {
                let rect = Rect::new(x, y, 32, 4);
                let mut out = TileBuffer::new_zeroed(rect, PixelFormat::RGB8, &budget)?;
                shared.pull_rect(rect, &mut out.as_view_mut(), |_, view| {
                    fill_tile_pattern(view, size, SynthPattern::DiagonalGradient)
                })?;
                let px = eval_pattern_rgba8(SynthPattern::DiagonalGradient, x, y, 64, 32);
                assert_eq!(&out.row(0)?[0..3], &px[0..3]);
            }
        }
        assert_eq!(shared.stats().rows_produced, 32);
        drop(shared);
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }
}
