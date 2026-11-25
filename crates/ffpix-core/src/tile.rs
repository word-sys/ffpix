//! Budget-tracked pixel tile buffers, zero-copy views, and reusable slab pooling.

use std::sync::{Arc, Mutex};

use crate::budget::{Budget, MemoryReservation};
use crate::error::{FfpError, FfpResult};
use crate::geom::Rect;
use crate::pixel::PixelFormat;

fn checked_min_slice_len(rect: Rect, format: PixelFormat, stride_bytes: usize) -> FfpResult<usize> {
    rect.validate()?;
    let row_bytes = format.checked_row_bytes(rect.w)?;
    if stride_bytes < row_bytes {
        return Err(FfpError::invalid_argument(format!(
            "stride_bytes ({stride_bytes}) is smaller than row width ({row_bytes})"
        )));
    }
    if rect.is_empty() {
        return Ok(0);
    }
    let last_row = usize::try_from(rect.h - 1)
        .map_err(|_| FfpError::overflow("tile height exceeds addressable usize"))?;
    let tail_offset = last_row
        .checked_mul(stride_bytes)
        .ok_or_else(|| FfpError::overflow("tile stride offset overflow"))?;
    let total = tail_offset
        .checked_add(row_bytes)
        .ok_or_else(|| FfpError::overflow("tile slice length overflow"))?;
    if total > isize::MAX as usize {
        return Err(FfpError::overflow("tile slice length exceeds isize::MAX"));
    }
    Ok(total)
}

fn checked_slab_bytes(rect: Rect, format: PixelFormat, stride_bytes: usize) -> FfpResult<usize> {
    rect.validate()?;
    let row_bytes = format.checked_row_bytes(rect.w)?;
    if stride_bytes < row_bytes {
        return Err(FfpError::invalid_argument(format!(
            "stride_bytes ({stride_bytes}) is smaller than row width ({row_bytes})"
        )));
    }
    if rect.is_empty() {
        return Ok(0);
    }
    let h = usize::try_from(rect.h)
        .map_err(|_| FfpError::overflow("tile height exceeds addressable usize"))?;
    let total = stride_bytes
        .checked_mul(h)
        .ok_or_else(|| FfpError::overflow("tile buffer byte size overflow"))?;
    if total > isize::MAX as usize {
        return Err(FfpError::overflow("tile buffer exceeds isize::MAX"));
    }
    Ok(total)
}

fn checked_sub_range(
    parent_rect: Rect,
    format: PixelFormat,
    stride_bytes: usize,
    sub_rect: Rect,
) -> FfpResult<(usize, usize)> {
    sub_rect.validate()?;
    if !parent_rect.contains_rect(sub_rect) {
        return Err(FfpError::invalid_argument(format!(
            "sub_view rect {sub_rect} is outside parent rect {parent_rect}"
        )));
    }
    if sub_rect.is_empty() {
        return Ok((0, 0));
    }
    let dx = usize::try_from(sub_rect.x - parent_rect.x)
        .map_err(|_| FfpError::overflow("sub_view x offset overflow"))?;
    let dy = usize::try_from(sub_rect.y - parent_rect.y)
        .map_err(|_| FfpError::overflow("sub_view y offset overflow"))?;
    let y_bytes = dy
        .checked_mul(stride_bytes)
        .ok_or_else(|| FfpError::overflow("sub_view row offset overflow"))?;
    let x_bytes = dx
        .checked_mul(format.bytes_per_pixel())
        .ok_or_else(|| FfpError::overflow("sub_view column offset overflow"))?;
    let start = y_bytes
        .checked_add(x_bytes)
        .ok_or_else(|| FfpError::overflow("sub_view start offset overflow"))?;
    let len = checked_min_slice_len(sub_rect, format, stride_bytes)?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| FfpError::overflow("sub_view end offset overflow"))?;
    Ok((start, end))
}

#[derive(Debug)]
pub struct TileBuffer {
    rect: Rect,
    format: PixelFormat,
    stride_bytes: usize,
    data: Vec<u8>,
    reservation: MemoryReservation,
}

impl TileBuffer {
    pub fn new_zeroed(rect: Rect, format: PixelFormat, budget: &Budget) -> FfpResult<Self> {
        let stride = format.checked_row_bytes(rect.w)?;
        Self::with_stride(rect, format, stride, budget)
    }

    pub fn with_stride(
        rect: Rect,
        format: PixelFormat,
        stride_bytes: usize,
        budget: &Budget,
    ) -> FfpResult<Self> {
        let total = checked_slab_bytes(rect, format, stride_bytes)?;
        let reservation = budget.reserve_memory(total as u64)?;
        let data = vec![0u8; total];
        Ok(Self {
            rect,
            format,
            stride_bytes,
            data,
            reservation,
        })
    }

    pub fn from_vec(
        rect: Rect,
        format: PixelFormat,
        stride_bytes: usize,
        data: Vec<u8>,
        budget: &Budget,
    ) -> FfpResult<Self> {
        let min_len = checked_min_slice_len(rect, format, stride_bytes)?;
        if data.len() < min_len {
            return Err(FfpError::invalid_argument(format!(
                "tile data length ({}) is smaller than required ({min_len})",
                data.len()
            )));
        }
        let reservation = budget.reserve_memory(data.len() as u64)?;
        Ok(Self {
            rect,
            format,
            stride_bytes,
            data,
            reservation,
        })
    }

    #[must_use]
    pub const fn rect(&self) -> Rect {
        self.rect
    }

    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    #[must_use]
    pub fn row_bytes(&self) -> usize {
        (self.rect.w as usize).saturating_mul(self.format.bytes_per_pixel())
    }

    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    #[must_use]
    pub const fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    #[must_use]
    pub fn as_view(&self) -> TileView<'_> {
        TileView {
            rect: self.rect,
            format: self.format,
            stride_bytes: self.stride_bytes,
            data: &self.data,
        }
    }

    #[must_use]
    pub fn as_view_mut(&mut self) -> TileViewMut<'_> {
        TileViewMut {
            rect: self.rect,
            format: self.format,
            stride_bytes: self.stride_bytes,
            data: &mut self.data,
        }
    }

    pub fn row(&self, local_y: u32) -> FfpResult<&[u8]> {
        self.as_view().row(local_y)
    }

    pub fn row_mut(&mut self, local_y: u32) -> FfpResult<&mut [u8]> {
        let row_bytes = self.format.checked_row_bytes(self.rect.w)?;
        let (start, end) = row_byte_range(self.rect.h, self.stride_bytes, row_bytes, local_y)?;
        self.data
            .get_mut(start..end)
            .ok_or_else(|| FfpError::internal("tile row slice out of bounds"))
    }

    pub fn fill_zero(&mut self) {
        self.as_view_mut().fill_zero();
    }
}

fn row_byte_range(
    height: u32,
    stride_bytes: usize,
    row_bytes: usize,
    local_y: u32,
) -> FfpResult<(usize, usize)> {
    if local_y >= height {
        return Err(FfpError::invalid_argument(format!(
            "row index {local_y} out of bounds for height {height}"
        )));
    }
    let y = usize::try_from(local_y).map_err(|_| FfpError::overflow("row index overflow"))?;
    let start = y
        .checked_mul(stride_bytes)
        .ok_or_else(|| FfpError::overflow("row offset overflow"))?;
    let end = start
        .checked_add(row_bytes)
        .ok_or_else(|| FfpError::overflow("row end overflow"))?;
    Ok((start, end))
}

#[derive(Debug, Clone, Copy)]
pub struct TileView<'a> {
    rect: Rect,
    format: PixelFormat,
    stride_bytes: usize,
    data: &'a [u8],
}

impl<'a> TileView<'a> {
    pub fn from_slice(
        rect: Rect,
        format: PixelFormat,
        stride_bytes: usize,
        data: &'a [u8],
    ) -> FfpResult<Self> {
        let min_len = checked_min_slice_len(rect, format, stride_bytes)?;
        if data.len() < min_len {
            return Err(FfpError::invalid_argument(format!(
                "tile view slice length ({}) is smaller than required ({min_len})",
                data.len()
            )));
        }
        Ok(Self {
            rect,
            format,
            stride_bytes,
            data,
        })
    }

    #[must_use]
    pub const fn rect(&self) -> Rect {
        self.rect
    }

    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    #[must_use]
    pub fn row_bytes(&self) -> usize {
        (self.rect.w as usize).saturating_mul(self.format.bytes_per_pixel())
    }

    pub fn row(&self, local_y: u32) -> FfpResult<&'a [u8]> {
        let row_bytes = self.format.checked_row_bytes(self.rect.w)?;
        let (start, end) = row_byte_range(self.rect.h, self.stride_bytes, row_bytes, local_y)?;
        self.data
            .get(start..end)
            .ok_or_else(|| FfpError::invalid_argument("tile row slice out of bounds"))
    }

    pub fn sub_view(&self, rect: Rect) -> FfpResult<TileView<'a>> {
        let (start, end) = checked_sub_range(self.rect, self.format, self.stride_bytes, rect)?;
        let slice = self
            .data
            .get(start..end)
            .ok_or_else(|| FfpError::invalid_argument("sub_view exceeds backing slice"))?;
        Ok(TileView {
            rect,
            format: self.format,
            stride_bytes: self.stride_bytes,
            data: slice,
        })
    }
}

#[derive(Debug)]
pub struct TileViewMut<'a> {
    rect: Rect,
    format: PixelFormat,
    stride_bytes: usize,
    data: &'a mut [u8],
}

impl<'a> TileViewMut<'a> {
    pub fn from_slice(
        rect: Rect,
        format: PixelFormat,
        stride_bytes: usize,
        data: &'a mut [u8],
    ) -> FfpResult<Self> {
        let min_len = checked_min_slice_len(rect, format, stride_bytes)?;
        if data.len() < min_len {
            return Err(FfpError::invalid_argument(format!(
                "tile view slice length ({}) is smaller than required ({min_len})",
                data.len()
            )));
        }
        Ok(Self {
            rect,
            format,
            stride_bytes,
            data,
        })
    }

    #[must_use]
    pub const fn rect(&self) -> Rect {
        self.rect
    }

    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    #[must_use]
    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    #[must_use]
    pub fn row_bytes(&self) -> usize {
        (self.rect.w as usize).saturating_mul(self.format.bytes_per_pixel())
    }

    #[must_use]
    pub fn as_view(&self) -> TileView<'_> {
        TileView {
            rect: self.rect,
            format: self.format,
            stride_bytes: self.stride_bytes,
            data: self.data,
        }
    }

    pub fn row(&self, local_y: u32) -> FfpResult<&[u8]> {
        self.as_view().row(local_y)
    }

    pub fn row_mut(&mut self, local_y: u32) -> FfpResult<&mut [u8]> {
        let row_bytes = self.format.checked_row_bytes(self.rect.w)?;
        let (start, end) = row_byte_range(self.rect.h, self.stride_bytes, row_bytes, local_y)?;
        self.data
            .get_mut(start..end)
            .ok_or_else(|| FfpError::invalid_argument("tile row slice out of bounds"))
    }

    pub fn sub_view(&self, rect: Rect) -> FfpResult<TileView<'_>> {
        self.as_view().sub_view(rect)
    }

    pub fn sub_view_mut(&mut self, rect: Rect) -> FfpResult<TileViewMut<'_>> {
        let (start, end) = checked_sub_range(self.rect, self.format, self.stride_bytes, rect)?;
        let slice = self
            .data
            .get_mut(start..end)
            .ok_or_else(|| FfpError::invalid_argument("sub_view_mut exceeds backing slice"))?;
        Ok(TileViewMut {
            rect,
            format: self.format,
            stride_bytes: self.stride_bytes,
            data: slice,
        })
    }

    pub fn copy_from(&mut self, src: &TileView<'_>) -> FfpResult<()> {
        if self.format != src.format {
            return Err(FfpError::format_mismatch(format!(
                "cannot copy {} into {}",
                src.format, self.format
            )));
        }
        if self.rect.size() != src.rect.size() {
            return Err(FfpError::dimension_mismatch(
                self.rect.w,
                self.rect.h,
                src.rect.w,
                src.rect.h,
            ));
        }
        for y in 0..self.rect.h {
            let src_row = src.row(y)?;
            let dst_row = self.row_mut(y)?;
            dst_row.copy_from_slice(src_row);
        }
        Ok(())
    }

    pub fn copy_rect_from(&mut self, src: &TileView<'_>, area: Rect) -> FfpResult<()> {
        let src_sub = src.sub_view(area)?;
        let mut dst_sub = self.sub_view_mut(area)?;
        dst_sub.copy_from(&src_sub)
    }

    /// Zeroes only the active pixels within `self.rect`, leaving stride padding untouched.
    pub fn fill_zero(&mut self) {
        let Ok(row_bytes) = self.format.checked_row_bytes(self.rect.w) else {
            return;
        };
        for y in 0..self.rect.h {
            if let Ok((start, end)) = row_byte_range(self.rect.h, self.stride_bytes, row_bytes, y)
                && let Some(row) = self.data.get_mut(start..end)
            {
                row.fill(0);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TilePoolStats {
    pub allocations: u64,
    pub reuses: u64,
    pub cached_slabs: usize,
    pub cached_bytes: u64,
}

#[derive(Debug)]
struct CachedSlab {
    data: Vec<u8>,
    reservation: MemoryReservation,
}

#[derive(Debug)]
struct PoolInner {
    slabs: Vec<CachedSlab>,
    cached_bytes: u64,
    max_cached_slabs: usize,
    max_cached_bytes: u64,
    allocations: u64,
    reuses: u64,
}

#[derive(Debug, Clone)]
pub struct TilePool {
    budget: Budget,
    inner: Arc<Mutex<PoolInner>>,
}

impl TilePool {
    #[must_use]
    pub fn new(budget: Budget, max_cached_slabs: usize, max_cached_bytes: u64) -> Self {
        Self {
            budget,
            inner: Arc::new(Mutex::new(PoolInner {
                slabs: Vec::new(),
                cached_bytes: 0,
                max_cached_slabs,
                max_cached_bytes,
                allocations: 0,
                reuses: 0,
            })),
        }
    }

    #[must_use]
    pub fn with_defaults(budget: Budget) -> Self {
        Self::new(budget, 16, 8 * 1024 * 1024)
    }

    #[must_use]
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    pub fn acquire(&self, rect: Rect, format: PixelFormat) -> FfpResult<TileBuffer> {
        let stride = format.checked_row_bytes(rect.w)?;
        self.acquire_with_stride(rect, format, stride)
    }

    pub fn acquire_with_stride(
        &self,
        rect: Rect,
        format: PixelFormat,
        stride_bytes: usize,
    ) -> FfpResult<TileBuffer> {
        let needed = checked_slab_bytes(rect, format, stride_bytes)?;

        if needed > 0
            && let Ok(mut guard) = self.inner.lock()
        {
            let max_acceptable = needed.saturating_mul(2).max(4096);
            let best_idx = guard
                .slabs
                .iter()
                .enumerate()
                .filter(|(_, s)| s.data.capacity() >= needed && s.data.capacity() <= max_acceptable)
                .min_by_key(|(_, s)| s.data.capacity())
                .map(|(i, _)| i);

            if let Some(idx) = best_idx {
                let mut slab = guard.slabs.swap_remove(idx);
                guard.cached_bytes = guard.cached_bytes.saturating_sub(slab.reservation.bytes());
                guard.reuses += 1;
                drop(guard);

                // Ensure active slice is zeroed so recycled slabs never expose prior pixel data.
                slab.data.clear();
                slab.data.resize(needed, 0);
                return Ok(TileBuffer {
                    rect,
                    format,
                    stride_bytes,
                    data: slab.data,
                    reservation: slab.reservation,
                });
            }
        }

        let tile = TileBuffer::with_stride(rect, format, stride_bytes, &self.budget)?;
        if let Ok(mut guard) = self.inner.lock() {
            guard.allocations += 1;
        }
        Ok(tile)
    }

    pub fn release(&self, tile: TileBuffer) {
        let TileBuffer {
            data, reservation, ..
        } = tile;
        let bytes = reservation.bytes();
        if data.capacity() == 0 || bytes == 0 {
            return;
        }
        if let Ok(mut guard) = self.inner.lock()
            && guard.slabs.len() < guard.max_cached_slabs
            && guard.cached_bytes.saturating_add(bytes) <= guard.max_cached_bytes
        {
            guard.cached_bytes = guard.cached_bytes.saturating_add(bytes);
            guard.slabs.push(CachedSlab { data, reservation });
        }
    }

    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.slabs.clear();
            guard.cached_bytes = 0;
        }
    }

    #[must_use]
    pub fn stats(&self) -> TilePoolStats {
        match self.inner.lock() {
            Ok(guard) => TilePoolStats {
                allocations: guard.allocations,
                reuses: guard.reuses,
                cached_slabs: guard.slabs.len(),
                cached_bytes: guard.cached_bytes,
            },
            Err(_) => TilePoolStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::BudgetLimits;
    use crate::error::BudgetKind;

    #[test]
    fn sub_view_cropping_and_nested_views_preserve_pixels() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let rect = Rect::new(10, 20, 8, 6);
        let mut buf = TileBuffer::new_zeroed(rect, PixelFormat::RGBA8, &budget)?;

        for y in 0..rect.h {
            let row = buf.row_mut(y)?;
            for x in 0..rect.w {
                let i = (x as usize) * 4;
                row[i] = (rect.x + x) as u8;
                row[i + 1] = (rect.y + y) as u8;
                row[i + 2] = 0xAA;
                row[i + 3] = 0xFF;
            }
        }

        let sub_rect = Rect::new(12, 21, 4, 3);
        let view = buf.as_view().sub_view(sub_rect)?;
        assert_eq!(view.rect(), sub_rect);
        assert_eq!(view.stride_bytes(), 32);
        assert_eq!(
            view.row(0)?,
            &[
                12, 21, 0xAA, 0xFF, 13, 21, 0xAA, 0xFF, 14, 21, 0xAA, 0xFF, 15, 21, 0xAA, 0xFF
            ]
        );
        assert!(view.row(3).is_err());

        // Nested sub-view reaching the exact bottom-right corner of the parent sub-view.
        let corner = view.sub_view(Rect::new(14, 22, 2, 2))?;
        assert_eq!(corner.row(1)?, &[14, 23, 0xAA, 0xFF, 15, 23, 0xAA, 0xFF]);

        // Out-of-bounds sub-view fails cleanly.
        assert!(buf.as_view().sub_view(Rect::new(15, 20, 4, 2)).is_err());
        assert!(buf.as_view().sub_view(Rect::new(0, 0, 2, 2)).is_err());
        Ok(())
    }

    #[test]
    fn padded_stride_fill_zero_and_copy_from_leave_padding_intact() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let rect = Rect::new(0, 0, 3, 2);
        // 3 pixels RGB8 = 9 bytes active, padded to 16 bytes stride.
        let mut dst = TileBuffer::with_stride(rect, PixelFormat::RGB8, 16, &budget)?;
        dst.data_mut().fill(0xCC);
        dst.fill_zero();

        assert_eq!(&dst.data()[0..9], &[0u8; 9]);
        assert_eq!(&dst.data()[9..16], &[0xCCu8; 7]);
        assert_eq!(&dst.data()[16..25], &[0u8; 9]);
        assert_eq!(&dst.data()[25..32], &[0xCCu8; 7]);

        let mut src =
            TileBuffer::new_zeroed(Rect::new(100, 100, 3, 2), PixelFormat::RGB8, &budget)?;
        for (idx, b) in src.data_mut().iter_mut().enumerate() {
            *b = (idx as u8) + 1;
        }

        dst.as_view_mut().copy_from(&src.as_view())?;
        assert_eq!(dst.row(0)?, src.row(0)?);
        assert_eq!(dst.row(1)?, src.row(1)?);
        assert_eq!(&dst.data()[9..16], &[0xCCu8; 7]);

        // World-coordinate sub-rect blit via copy_rect_from.
        let mut partial =
            TileBuffer::new_zeroed(Rect::new(101, 100, 2, 2), PixelFormat::RGB8, &budget)?;
        partial
            .as_view_mut()
            .copy_rect_from(&src.as_view(), Rect::new(101, 100, 2, 2))?;
        assert_eq!(partial.row(0)?, &src.row(0)?[3..9]);
        assert_eq!(partial.row(1)?, &src.row(1)?[3..9]);

        assert!(TileBuffer::with_stride(rect, PixelFormat::RGB8, 8, &budget).is_err());
        Ok(())
    }

    #[test]
    fn tile_pool_reuses_slabs_and_zeroes_recycled_buffers() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let pool = TilePool::new(budget.clone(), 4, 1024 * 1024);
        let rect = Rect::new(0, 0, 32, 32);

        for i in 0..10u32 {
            let mut tile = pool.acquire(rect, PixelFormat::RGBA8)?;
            assert!(tile.data().iter().all(|&b| b == 0));
            tile.data_mut().fill((i as u8) + 1);
            pool.release(tile);
        }

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 9);
        assert_eq!(stats.cached_slabs, 1);
        assert_eq!(budget.peak_memory_bytes(), 32 * 32 * 4);

        pool.clear();
        assert_eq!(pool.stats().cached_slabs, 0);
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn budget_exhaustion_returns_budget_exceeded() {
        let limits = BudgetLimits {
            max_memory_bytes: 1024,
            ..BudgetLimits::preset_default()
        };
        let budget = Budget::new(limits);

        let first = TileBuffer::new_zeroed(Rect::new(0, 0, 16, 16), PixelFormat::RGBA8, &budget);
        assert!(first.is_ok());
        assert_eq!(budget.current_memory_bytes(), 1024);

        let second = TileBuffer::new_zeroed(Rect::new(0, 0, 1, 1), PixelFormat::RGBA8, &budget);
        assert!(matches!(
            second,
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::MemoryBytes,
                limit: 1024,
                requested: 1028,
            })
        ));

        drop(first);
        assert_eq!(budget.current_memory_bytes(), 0);
    }
}
