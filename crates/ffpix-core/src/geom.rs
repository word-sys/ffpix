//! Checked 2D geometry and tile-grid iteration.

use std::fmt;

use crate::error::{FfpError, FfpResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Size {
    pub w: u32,
    pub h: u32,
}

impl Size {
    #[must_use]
    pub const fn new(w: u32, h: u32) -> Self {
        Self { w, h }
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }

    #[must_use]
    pub const fn area(self) -> Option<u64> {
        (self.w as u64).checked_mul(self.h as u64)
    }

    #[must_use]
    pub const fn to_rect(self) -> Rect {
        Rect {
            x: 0,
            y: 0,
            w: self.w,
            h: self.h,
        }
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.w, self.h)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

impl Point {
    #[must_use]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub fn translate_checked(self, dx: i64, dy: i64) -> Option<Self> {
        let nx = i64::from(self.x).checked_add(dx)?;
        let ny = i64::from(self.y).checked_add(dy)?;
        Some(Self {
            x: u32::try_from(nx).ok()?,
            y: u32::try_from(ny).ok()?,
        })
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const EMPTY: Self = Self {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };

    #[must_use]
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    #[must_use]
    pub const fn from_origin_size(origin: Point, size: Size) -> Self {
        Self {
            x: origin.x,
            y: origin.y,
            w: size.w,
            h: size.h,
        }
    }

    #[must_use]
    pub const fn from_bounds(left: u32, top: u32, right: u32, bottom: u32) -> Option<Self> {
        let Some(w) = right.checked_sub(left) else {
            return None;
        };
        let Some(h) = bottom.checked_sub(top) else {
            return None;
        };
        Some(Self {
            x: left,
            y: top,
            w,
            h,
        })
    }

    #[must_use]
    pub const fn origin(self) -> Point {
        Point {
            x: self.x,
            y: self.y,
        }
    }

    #[must_use]
    pub const fn size(self) -> Size {
        Size {
            w: self.w,
            h: self.h,
        }
    }

    #[must_use]
    pub const fn right(self) -> Option<u32> {
        self.x.checked_add(self.w)
    }

    #[must_use]
    pub const fn bottom(self) -> Option<u32> {
        self.y.checked_add(self.h)
    }

    #[must_use]
    pub const fn area(self) -> Option<u64> {
        (self.w as u64).checked_mul(self.h as u64)
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }

    pub fn validate(self) -> FfpResult<()> {
        if self.right().is_none() || self.bottom().is_none() {
            return Err(FfpError::overflow("rect right or bottom exceeds u32::MAX"));
        }
        Ok(())
    }

    #[inline]
    const fn right_u64(self) -> u64 {
        (self.x as u64) + (self.w as u64)
    }

    #[inline]
    const fn bottom_u64(self) -> u64 {
        (self.y as u64) + (self.h as u64)
    }

    #[must_use]
    pub const fn contains_point(self, p: Point) -> bool {
        if self.is_empty() {
            return false;
        }
        let px = p.x as u64;
        let py = p.y as u64;
        px >= self.x as u64
            && px < self.right_u64()
            && py >= self.y as u64
            && py < self.bottom_u64()
    }

    #[must_use]
    pub const fn contains_rect(self, other: Self) -> bool {
        (other.x as u64) >= (self.x as u64)
            && (other.y as u64) >= (self.y as u64)
            && other.right_u64() <= self.right_u64()
            && other.bottom_u64() <= self.bottom_u64()
    }

    #[must_use]
    pub fn intersect(self, other: Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::EMPTY;
        }
        let x0 = u64::from(self.x.max(other.x));
        let y0 = u64::from(self.y.max(other.y));
        let x1 = self.right_u64().min(other.right_u64());
        let y1 = self.bottom_u64().min(other.bottom_u64());

        let (Some(w64), Some(h64)) = (x1.checked_sub(x0), y1.checked_sub(y0)) else {
            return Self::EMPTY;
        };
        if w64 == 0 || h64 == 0 {
            return Self::EMPTY;
        }
        let (Ok(x), Ok(y), Ok(w), Ok(h)) = (
            u32::try_from(x0),
            u32::try_from(y0),
            u32::try_from(w64),
            u32::try_from(h64),
        ) else {
            return Self::EMPTY;
        };
        Self { x, y, w, h }
    }

    /// Returns the minimal bounding rectangle covering `self` and `other`,
    /// or `None` if the resulting right/bottom edge exceeds `u32::MAX`.
    #[must_use]
    pub fn union(self, other: Self) -> Option<Self> {
        if self.is_empty() {
            other.right()?;
            other.bottom()?;
            return Some(other);
        }
        if other.is_empty() {
            self.right()?;
            self.bottom()?;
            return Some(self);
        }
        let x0 = u64::from(self.x.min(other.x));
        let y0 = u64::from(self.y.min(other.y));
        let x1 = self.right_u64().max(other.right_u64());
        let y1 = self.bottom_u64().max(other.bottom_u64());

        let right = u32::try_from(x1).ok()?;
        let bottom = u32::try_from(y1).ok()?;
        let left = u32::try_from(x0).ok()?;
        let top = u32::try_from(y0).ok()?;
        Self::from_bounds(left, top, right, bottom)
    }

    #[must_use]
    pub fn clamp_to(self, bounds: Size) -> Self {
        self.intersect(bounds.to_rect())
    }

    /// Expands `self` by the given halo margins and clamps the result to `[0..bounds)`.
    #[must_use]
    pub fn expand_halo(self, left: u32, top: u32, right: u32, bottom: u32, bounds: Size) -> Self {
        if self.is_empty() || bounds.is_empty() {
            return Self::EMPTY;
        }
        let bw = u64::from(bounds.w);
        let bh = u64::from(bounds.h);

        let x0 = u64::from(self.x).saturating_sub(u64::from(left)).min(bw);
        let y0 = u64::from(self.y).saturating_sub(u64::from(top)).min(bh);
        let x1 = self.right_u64().saturating_add(u64::from(right)).min(bw);
        let y1 = self.bottom_u64().saturating_add(u64::from(bottom)).min(bh);

        let (Some(w64), Some(h64)) = (x1.checked_sub(x0), y1.checked_sub(y0)) else {
            return Self::EMPTY;
        };
        if w64 == 0 || h64 == 0 {
            return Self::EMPTY;
        }
        let (Ok(x), Ok(y), Ok(w), Ok(h)) = (
            u32::try_from(x0),
            u32::try_from(y0),
            u32::try_from(w64),
            u32::try_from(h64),
        ) else {
            return Self::EMPTY;
        };
        Self { x, y, w, h }
    }

    #[must_use]
    pub fn translate_checked(self, dx: i64, dy: i64) -> Option<Self> {
        let nx = i64::from(self.x).checked_add(dx)?;
        let ny = i64::from(self.y).checked_add(dy)?;
        let x = u32::try_from(nx).ok()?;
        let y = u32::try_from(ny).ok()?;
        x.checked_add(self.w)?;
        y.checked_add(self.h)?;
        Some(Self {
            x,
            y,
            w: self.w,
            h: self.h,
        })
    }
}

impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}) {}x{}", self.x, self.y, self.w, self.h)
    }
}

/// Row-major iterator partitioning `bounds` into non-overlapping tiles clipped to the edges.
#[derive(Debug, Clone)]
pub struct TileGridIter {
    bounds: Size,
    tile_w: u32,
    tile_h: u32,
    cur_x: u32,
    cur_y: u32,
    done: bool,
    remaining: u64,
}

impl TileGridIter {
    pub fn new(bounds: Size, tile_w: u32, tile_h: u32) -> FfpResult<Self> {
        if tile_w == 0 || tile_h == 0 {
            return Err(FfpError::invalid_argument(
                "tile dimensions must be non-zero",
            ));
        }
        if bounds.is_empty() {
            return Ok(Self {
                bounds,
                tile_w,
                tile_h,
                cur_x: 0,
                cur_y: 0,
                done: true,
                remaining: 0,
            });
        }
        let cols = div_ceil_u32(bounds.w, tile_w);
        let rows = div_ceil_u32(bounds.h, tile_h);
        let remaining = u64::from(cols)
            .checked_mul(u64::from(rows))
            .ok_or_else(|| FfpError::overflow("tile count overflow"))?;
        Ok(Self {
            bounds,
            tile_w,
            tile_h,
            cur_x: 0,
            cur_y: 0,
            done: false,
            remaining,
        })
    }

    #[must_use]
    pub const fn cols(&self) -> u32 {
        if self.bounds.is_empty() {
            0
        } else {
            div_ceil_u32(self.bounds.w, self.tile_w)
        }
    }

    #[must_use]
    pub const fn rows(&self) -> u32 {
        if self.bounds.is_empty() {
            0
        } else {
            div_ceil_u32(self.bounds.h, self.tile_h)
        }
    }

    #[must_use]
    pub const fn remaining_tiles(&self) -> u64 {
        self.remaining
    }
}

#[inline]
const fn div_ceil_u32(num: u32, den: u32) -> u32 {
    let q = num / den;
    let r = num % den;
    if r == 0 { q } else { q + 1 }
}

impl Iterator for TileGridIter {
    type Item = Rect;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let w = self.bounds.w.saturating_sub(self.cur_x).min(self.tile_w);
        let h = self.bounds.h.saturating_sub(self.cur_y).min(self.tile_h);
        let tile = Rect {
            x: self.cur_x,
            y: self.cur_y,
            w,
            h,
        };

        self.remaining = self.remaining.saturating_sub(1);

        match self.cur_x.checked_add(self.tile_w) {
            Some(next_x) if next_x < self.bounds.w => {
                self.cur_x = next_x;
            }
            _ => {
                self.cur_x = 0;
                match self.cur_y.checked_add(self.tile_h) {
                    Some(next_y) if next_y < self.bounds.h => {
                        self.cur_y = next_y;
                    }
                    _ => {
                        self.done = true;
                    }
                }
            }
        }

        Some(tile)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining) {
            Ok(n) => (n, Some(n)),
            Err(_) => (usize::MAX, None),
        }
    }
}

pub fn tile_grid(bounds: Size, tile_w: u32, tile_h: u32) -> FfpResult<TileGridIter> {
    TileGridIter::new(bounds, tile_w, tile_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_max_overflow_resistance() {
        let huge = Rect::new(u32::MAX, u32::MAX, 10, 10);
        assert_eq!(huge.right(), None);
        assert_eq!(huge.bottom(), None);
        assert!(huge.validate().is_err());

        let max_area = Rect::new(0, 0, u32::MAX, u32::MAX);
        assert_eq!(max_area.right(), Some(u32::MAX));
        assert_eq!(max_area.bottom(), Some(u32::MAX));
        assert_eq!(
            max_area.area(),
            Some(u64::from(u32::MAX) * u64::from(u32::MAX))
        );

        let a = Rect::new(u32::MAX - 5, 0, 10, 10);
        let b = Rect::new(u32::MAX - 2, 0, 10, 10);
        assert_eq!(a.union(b), None);

        let clamped = a.clamp_to(Size::new(u32::MAX, 20));
        assert_eq!(clamped, Rect::new(u32::MAX - 5, 0, 5, 10));

        let moved = Rect::new(u32::MAX - 10, 5, 10, 10);
        assert_eq!(moved.translate_checked(1, 0), None);
        assert_eq!(moved.translate_checked(-5, -6), None);
        assert_eq!(
            moved.translate_checked(-5, -5),
            Some(Rect::new(u32::MAX - 15, 0, 10, 10))
        );

        let p = Point::new(u32::MAX, 0);
        assert_eq!(p.translate_checked(1, 0), None);
        assert_eq!(p.translate_checked(0, -1), None);
        assert_eq!(
            p.translate_checked(-1, 5),
            Some(Point::new(u32::MAX - 1, 5))
        );
    }

    #[test]
    fn intersect_union_and_containment() {
        let a = Rect::new(10, 20, 30, 40);
        let b = Rect::new(25, 35, 30, 40);
        assert_eq!(a.intersect(b), Rect::new(25, 35, 15, 25));
        assert_eq!(a.union(b), Some(Rect::new(10, 20, 45, 55)));

        // Disjoint and edge-touching rectangles return EMPTY.
        let disjoint = Rect::new(100, 100, 10, 10);
        assert_eq!(a.intersect(disjoint), Rect::EMPTY);
        let touching = Rect::new(40, 20, 10, 40);
        assert_eq!(a.intersect(touching), Rect::EMPTY);

        assert_eq!(a.union(Rect::EMPTY), Some(a));
        assert_eq!(Rect::EMPTY.union(a), Some(a));

        assert!(a.contains_point(Point::new(10, 20)));
        assert!(a.contains_point(Point::new(39, 59)));
        assert!(!a.contains_point(Point::new(40, 20)));
        assert!(!a.contains_point(Point::new(10, 60)));
        assert!(!Rect::EMPTY.contains_point(Point::new(0, 0)));

        assert!(a.contains_rect(Rect::new(15, 25, 10, 10)));
        assert!(a.contains_rect(a));
        assert!(!a.contains_rect(b));
        assert!(!a.contains_rect(Rect::new(500, 500, 0, 0)));
    }

    #[test]
    fn halo_expansion_clamps_to_image_edges() {
        let bounds = Size::new(100, 80);

        let top_left = Rect::new(0, 0, 16, 16);
        assert_eq!(
            top_left.expand_halo(3, 4, 3, 4, bounds),
            Rect::new(0, 0, 19, 20)
        );

        let interior = Rect::new(20, 20, 10, 10);
        assert_eq!(
            interior.expand_halo(3, 2, 4, 5, bounds),
            Rect::new(17, 18, 17, 17)
        );

        let bottom_right = Rect::new(90, 70, 10, 10);
        assert_eq!(
            bottom_right.expand_halo(2, 2, 10, 10, bounds),
            Rect::new(88, 68, 12, 12)
        );

        let huge_halo = interior.expand_halo(u32::MAX, u32::MAX, u32::MAX, u32::MAX, bounds);
        assert_eq!(huge_halo, Rect::new(0, 0, 100, 80));

        assert_eq!(Rect::EMPTY.expand_halo(2, 2, 2, 2, bounds), Rect::EMPTY);
        assert_eq!(
            interior.expand_halo(2, 2, 2, 2, Size::new(0, 80)),
            Rect::EMPTY
        );
    }

    #[test]
    fn tile_grid_covers_non_multiple_dimensions_exactly() {
        assert!(tile_grid(Size::new(10, 10), 0, 16).is_err());
        assert!(tile_grid(Size::new(10, 10), 16, 0).is_err());

        let empty_grid = tile_grid(Size::new(0, 100), 16, 16);
        assert!(empty_grid.is_ok());
        if let Ok(iter) = empty_grid {
            assert_eq!(iter.remaining_tiles(), 0);
            assert_eq!(iter.count(), 0);
        }

        let bounds = Size::new(130, 75);
        let grid_res = tile_grid(bounds, 64, 32);
        assert!(grid_res.is_ok());
        if let Ok(iter) = grid_res {
            assert_eq!(iter.cols(), 3);
            assert_eq!(iter.rows(), 3);
            assert_eq!(iter.remaining_tiles(), 9);

            let tiles: Vec<Rect> = iter.collect();
            assert_eq!(tiles.len(), 9);

            // Verify row-major ordering, non-overlap, bounds containment, and exact pixel coverage.
            let mut total_area = 0u64;
            for (i, &t) in tiles.iter().enumerate() {
                assert!(!t.is_empty());
                assert!(bounds.to_rect().contains_rect(t));
                total_area += t.area().unwrap_or(0);
                for &other in &tiles[i + 1..] {
                    assert_eq!(t.intersect(other), Rect::EMPTY);
                }
            }
            assert_eq!(total_area, bounds.area().unwrap_or(0));

            // Check right-edge and bottom-edge partial tile dimensions explicitly.
            assert_eq!(tiles[0], Rect::new(0, 0, 64, 32));
            assert_eq!(tiles[2], Rect::new(128, 0, 2, 32));
            assert_eq!(tiles[6], Rect::new(0, 64, 64, 11));
            assert_eq!(tiles[8], Rect::new(128, 64, 2, 11));
        }

        // Single huge tile spanning near u32::MAX terminates cleanly without wraparound.
        let near_max = tile_grid(Size::new(u32::MAX, 1), u32::MAX - 10, 1);
        assert!(near_max.is_ok());
        if let Ok(iter) = near_max {
            let tiles: Vec<Rect> = iter.collect();
            assert_eq!(
                tiles,
                vec![
                    Rect::new(0, 0, u32::MAX - 10, 1),
                    Rect::new(u32::MAX - 10, 0, 10, 1),
                ]
            );
        }
    }
}
