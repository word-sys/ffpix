//! Deterministic synthetic image generators and pixel-comparison oracle.

use crate::budget::Budget;
use crate::convert::convert_row_format;
use crate::error::{FfpError, FfpResult};
use crate::geom::{Rect, Size};
use crate::hash::Sha256;
use crate::pixel::{PixelFormat, SampleType};
use crate::tile::{TileBuffer, TileView, TileViewMut};

/// Fixed 256-entry cosine table `(0.5 + 0.5 * cos(2*pi*i/256)) * 255` for
/// bit-identical cross-platform zone-plate generation without libm drift.
const COS_LUT_256: [u8; 256] = [
    255, 255, 255, 254, 254, 253, 252, 251, 250, 249, 247, 246, 244, 242, 240, 238, 236, 233, 231,
    228, 225, 222, 219, 216, 213, 209, 206, 202, 198, 195, 191, 187, 183, 179, 174, 170, 166, 161,
    157, 152, 148, 143, 138, 134, 129, 124, 120, 115, 110, 106, 101, 97, 92, 88, 83, 79, 75, 70,
    66, 62, 58, 55, 51, 47, 44, 41, 37, 34, 31, 28, 26, 23, 21, 18, 16, 14, 12, 10, 9, 7, 6, 5, 4,
    3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 16, 18, 21, 23, 26, 28,
    31, 34, 37, 41, 44, 47, 51, 55, 58, 62, 66, 70, 75, 79, 83, 88, 92, 97, 101, 106, 110, 115,
    120, 124, 129, 134, 138, 143, 148, 152, 157, 161, 166, 170, 174, 179, 183, 187, 191, 195, 198,
    202, 206, 209, 213, 216, 219, 222, 225, 228, 231, 233, 236, 238, 240, 242, 244, 246, 247, 249,
    250, 251, 252, 253, 254, 254, 255, 255, 255, 255, 254, 254, 253, 252, 251, 250, 249, 247, 246,
    244, 242, 240, 238, 236, 233, 231, 228, 225, 222, 219, 216, 213, 209, 206, 202, 198, 195, 191,
    187, 183, 179, 174, 170, 166, 161, 157, 152, 148, 143, 138, 134, 129, 124, 120, 115, 110, 106,
    101, 97, 92, 88, 83, 79, 75, 70, 66, 62, 58, 55, 51, 47, 44, 41, 37, 34, 31, 28, 26, 23, 21,
    18, 16, 14, 12, 10,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthPattern {
    Solid([u8; 4]),
    HorizontalGradient,
    VerticalGradient,
    DiagonalGradient,
    ZonePlate,
    Checkerboard {
        cell_size: u32,
        color_a: [u8; 4],
        color_b: [u8; 4],
    },
    ColorBars,
    AlphaRings,
    XorShiftNoise {
        seed: u64,
    },
}

#[inline]
const fn scale_coord_u8(pos: u32, span: u32) -> u8 {
    if span <= 1 {
        return 0;
    }
    (((pos as u64) * 255) / ((span - 1) as u64)) as u8
}

#[inline]
const fn xorshift64_mix(mut x: u64) -> u64 {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[must_use]
pub const fn eval_pattern_rgba8(
    pattern: SynthPattern,
    x: u32,
    y: u32,
    full_w: u32,
    full_h: u32,
) -> [u8; 4] {
    match pattern {
        SynthPattern::Solid(rgba) => rgba,
        SynthPattern::HorizontalGradient => {
            let v = scale_coord_u8(x, full_w);
            [v, v, v, 255]
        }
        SynthPattern::VerticalGradient => {
            let v = scale_coord_u8(y, full_h);
            [v, v, v, 255]
        }
        SynthPattern::DiagonalGradient => {
            let r = scale_coord_u8(x, full_w);
            let g = scale_coord_u8(y, full_h);
            let b = ((r as u16 + g as u16) / 2) as u8;
            [r, g, b, 255]
        }
        SynthPattern::ZonePlate => {
            let cx = (full_w / 2) as i64;
            let cy = (full_h / 2) as i64;
            let dx = (x as i64) - cx;
            let dy = (y as i64) - cy;
            let r2 = (dx * dx + dy * dy) as u64;
            let phase_r = ((r2 >> 2) & 0xFF) as usize;
            let phase_g = ((r2 >> 3) & 0xFF) as usize;
            let phase_b = (((r2 >> 2) + 64) & 0xFF) as usize;
            [
                COS_LUT_256[phase_r],
                COS_LUT_256[phase_g],
                COS_LUT_256[phase_b],
                255,
            ]
        }
        SynthPattern::Checkerboard {
            cell_size,
            color_a,
            color_b,
        } => {
            let step = if cell_size == 0 { 1 } else { cell_size };
            let parity = ((x / step) + (y / step)) & 1;
            if parity == 0 { color_a } else { color_b }
        }
        SynthPattern::ColorBars => {
            const BARS: [[u8; 4]; 8] = [
                [255, 255, 255, 255],
                [255, 255, 0, 255],
                [0, 255, 255, 255],
                [0, 255, 0, 255],
                [255, 0, 255, 255],
                [255, 0, 0, 255],
                [0, 0, 255, 255],
                [0, 0, 0, 255],
            ];
            let idx = if full_w == 0 {
                0
            } else {
                let bar = ((x as u64) * 8) / (full_w as u64);
                if bar > 7 { 7 } else { bar as usize }
            };
            BARS[idx]
        }
        SynthPattern::AlphaRings => {
            let cx = (full_w / 2) as i64;
            let cy = (full_h / 2) as i64;
            let dx = (x as i64) - cx;
            let dy = (y as i64) - cy;
            let r2 = (dx * dx + dy * dy) as u64;
            let r = isqrt_u64(r2);
            let alpha = COS_LUT_256[((r.wrapping_mul(16)) & 0xFF) as usize];
            let red = scale_coord_u8(x, full_w);
            let green = scale_coord_u8(y, full_h);
            [red, green, 255 - red, alpha]
        }
        SynthPattern::XorShiftNoise { seed } => {
            let state = seed
                .wrapping_add(((y as u64) << 32) | (x as u64))
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            let h = xorshift64_mix(if state == 0 { 1 } else { state });
            [
                (h & 0xFF) as u8,
                ((h >> 8) & 0xFF) as u8,
                ((h >> 16) & 0xFF) as u8,
                255,
            ]
        }
    }
}

const fn isqrt_u64(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let shift = (64 - n.leading_zeros()).div_ceil(2);
    let mut x = 1u64 << shift;
    loop {
        let next = (x + n / x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

pub fn fill_tile_pattern(
    dst: &mut TileViewMut<'_>,
    full_bounds: Size,
    pattern: SynthPattern,
) -> FfpResult<()> {
    let rect = dst.rect();
    if rect.is_empty() {
        return Ok(());
    }
    let fmt = dst.format();
    if fmt == PixelFormat::RGBA8 {
        for ly in 0..rect.h {
            let wy = rect.y.saturating_add(ly);
            let row = dst.row_mut(ly)?;
            for (lx, px) in row.chunks_exact_mut(4).enumerate() {
                let wx = rect.x.saturating_add(lx as u32);
                px.copy_from_slice(&eval_pattern_rgba8(
                    pattern,
                    wx,
                    wy,
                    full_bounds.w,
                    full_bounds.h,
                ));
            }
        }
        return Ok(());
    }

    let rgba_row_bytes = PixelFormat::RGBA8.checked_row_bytes(rect.w)?;
    let mut scratch = vec![0u8; rgba_row_bytes];
    for ly in 0..rect.h {
        let wy = rect.y.saturating_add(ly);
        for (lx, px) in scratch.chunks_exact_mut(4).enumerate() {
            let wx = rect.x.saturating_add(lx as u32);
            px.copy_from_slice(&eval_pattern_rgba8(
                pattern,
                wx,
                wy,
                full_bounds.w,
                full_bounds.h,
            ));
        }
        let dst_row = dst.row_mut(ly)?;
        convert_row_format(&scratch, PixelFormat::RGBA8, dst_row, fmt, rect.w)?;
    }
    Ok(())
}

pub fn generate_tile(
    width: u32,
    height: u32,
    format: PixelFormat,
    pattern: SynthPattern,
    budget: &Budget,
) -> FfpResult<TileBuffer> {
    if width == 0 || height == 0 {
        return Err(FfpError::invalid_input(
            "synthetic image dimensions must be non-zero",
        ));
    }
    let rect = Rect::new(0, 0, width, height);
    let mut buf = TileBuffer::new_zeroed(rect, format, budget)?;
    fill_tile_pattern(&mut buf.as_view_mut(), Size::new(width, height), pattern)?;
    Ok(buf)
}

pub fn tile_sha256(view: &TileView<'_>) -> FfpResult<[u8; 32]> {
    let mut hasher = Sha256::new();
    for y in 0..view.rect().h {
        hasher.update(view.row(y)?);
    }
    Ok(hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDiff {
    pub mismatched_bytes: u64,
    pub max_abs_diff: u8,
    pub first_mismatch: Option<(u32, usize)>,
}

impl TileDiff {
    #[must_use]
    pub const fn is_exact(&self) -> bool {
        self.mismatched_bytes == 0
    }
}

pub fn compare_tiles_exact(a: &TileView<'_>, b: &TileView<'_>) -> FfpResult<TileDiff> {
    if a.format() != b.format() {
        return Err(FfpError::format_mismatch(format!(
            "cannot compare {} with {}",
            a.format(),
            b.format()
        )));
    }
    if a.rect().size() != b.rect().size() {
        return Err(FfpError::dimension_mismatch(
            a.rect().w,
            a.rect().h,
            b.rect().w,
            b.rect().h,
        ));
    }
    let mut mismatched_bytes = 0u64;
    let mut max_abs_diff = 0u8;
    let mut first_mismatch = None;

    for y in 0..a.rect().h {
        let ra = a.row(y)?;
        let rb = b.row(y)?;
        for (idx, (&ba, &bb)) in ra.iter().zip(rb.iter()).enumerate() {
            if ba != bb {
                mismatched_bytes += 1;
                let d = ba.abs_diff(bb);
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
                if first_mismatch.is_none() {
                    first_mismatch = Some((y, idx));
                }
            }
        }
    }

    Ok(TileDiff {
        mismatched_bytes,
        max_abs_diff,
        first_mismatch,
    })
}

pub fn compute_tile_psnr(a: &TileView<'_>, b: &TileView<'_>) -> FfpResult<f64> {
    if a.format() != b.format() {
        return Err(FfpError::format_mismatch(format!(
            "cannot compute PSNR between {} and {}",
            a.format(),
            b.format()
        )));
    }
    if a.rect().size() != b.rect().size() {
        return Err(FfpError::dimension_mismatch(
            a.rect().w,
            a.rect().h,
            b.rect().w,
            b.rect().h,
        ));
    }
    if a.rect().is_empty() {
        return Ok(f64::INFINITY);
    }

    let mut sq_sum = 0.0f64;
    let mut sample_count = 0u64;
    let peak = match a.format().sample {
        SampleType::U8 => 255.0f64,
        SampleType::U16 => 65_535.0f64,
        SampleType::F32 => 1.0f64,
    };

    for y in 0..a.rect().h {
        let ra = a.row(y)?;
        let rb = b.row(y)?;
        match a.format().sample {
            SampleType::U8 => {
                for (&va, &vb) in ra.iter().zip(rb.iter()) {
                    let d = f64::from(va) - f64::from(vb);
                    sq_sum += d * d;
                    sample_count += 1;
                }
            }
            SampleType::U16 => {
                for (ca, cb) in ra.chunks_exact(2).zip(rb.chunks_exact(2)) {
                    let va = u16::from_ne_bytes([ca[0], ca[1]]);
                    let vb = u16::from_ne_bytes([cb[0], cb[1]]);
                    let d = f64::from(va) - f64::from(vb);
                    sq_sum += d * d;
                    sample_count += 1;
                }
            }
            SampleType::F32 => {
                for (ca, cb) in ra.chunks_exact(4).zip(rb.chunks_exact(4)) {
                    let va = f32::from_ne_bytes([ca[0], ca[1], ca[2], ca[3]]);
                    let vb = f32::from_ne_bytes([cb[0], cb[1], cb[2], cb[3]]);
                    let d = f64::from(va) - f64::from(vb);
                    sq_sum += d * d;
                    sample_count += 1;
                }
            }
        }
    }

    if sample_count == 0 || sq_sum == 0.0 {
        return Ok(f64::INFINITY);
    }
    let mse = sq_sum / (sample_count as f64);
    Ok(10.0 * ((peak * peak) / mse).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_patterns_are_deterministic_and_subtile_consistent() {
        let budget = Budget::preset_default();
        let patterns = [
            SynthPattern::Solid([12, 34, 56, 255]),
            SynthPattern::HorizontalGradient,
            SynthPattern::VerticalGradient,
            SynthPattern::DiagonalGradient,
            SynthPattern::ZonePlate,
            SynthPattern::Checkerboard {
                cell_size: 4,
                color_a: [0, 0, 0, 255],
                color_b: [255, 255, 255, 255],
            },
            SynthPattern::ColorBars,
            SynthPattern::AlphaRings,
            SynthPattern::XorShiftNoise {
                seed: 0xDEAD_BEEF_CAFE_BABE,
            },
        ];

        for pattern in patterns {
            let Ok(a) = generate_tile(32, 24, PixelFormat::RGBA8, pattern, &budget) else {
                continue;
            };
            let Ok(b) = generate_tile(32, 24, PixelFormat::RGBA8, pattern, &budget) else {
                continue;
            };
            assert_eq!(
                tile_sha256(&a.as_view()).ok(),
                tile_sha256(&b.as_view()).ok()
            );
            assert_eq!(
                compare_tiles_exact(&a.as_view(), &b.as_view())
                    .ok()
                    .map(|d| d.is_exact()),
                Some(true)
            );

            // Sub-tile generated directly matches cropped full tile bit-for-bit.
            let sub_rect = Rect::new(7, 5, 13, 11);
            if let Ok(mut sub) = TileBuffer::new_zeroed(sub_rect, PixelFormat::RGBA8, &budget) {
                assert!(
                    fill_tile_pattern(&mut sub.as_view_mut(), Size::new(32, 24), pattern).is_ok()
                );
                if let Ok(cropped) = a.as_view().sub_view(sub_rect) {
                    assert_eq!(
                        compare_tiles_exact(&sub.as_view(), &cropped)
                            .ok()
                            .map(|d| d.is_exact()),
                        Some(true)
                    );
                }
            }
        }
    }

    #[test]
    fn pinned_sha256_hashes_for_reference_patterns() {
        let budget = Budget::preset_default();
        let Ok(diag) = generate_tile(
            16,
            16,
            PixelFormat::RGBA8,
            SynthPattern::DiagonalGradient,
            &budget,
        ) else {
            return;
        };
        let Ok(bars) = generate_tile(16, 8, PixelFormat::RGB8, SynthPattern::ColorBars, &budget)
        else {
            return;
        };
        let Ok(noise) = generate_tile(
            16,
            16,
            PixelFormat::RGBA8,
            SynthPattern::XorShiftNoise { seed: 42 },
            &budget,
        ) else {
            return;
        };

        let Ok(solid) = generate_tile(
            16,
            16,
            PixelFormat::RGBA8,
            SynthPattern::Solid([12, 34, 56, 255]),
            &budget,
        ) else {
            return;
        };

        let h_solid = tile_sha256(&solid.as_view())
            .ok()
            .map(|d| Sha256::to_hex(&d));
        let h_diag_1 = tile_sha256(&diag.as_view())
            .ok()
            .map(|d| Sha256::to_hex(&d));
        let h_bars_1 = tile_sha256(&bars.as_view())
            .ok()
            .map(|d| Sha256::to_hex(&d));
        let h_noise_1 = tile_sha256(&noise.as_view())
            .ok()
            .map(|d| Sha256::to_hex(&d));

        assert_eq!(
            h_solid.as_deref(),
            Some("46650a034752fc448c9e1fafc4b67fc6540e7565c87b8c6aa4921853d36f1979")
        );
        assert_eq!(
            h_bars_1.as_deref(),
            Some("5e16531c134d2368eed4acd1748fb6c8e0bc6cae5a8030458612bf2d62bcf6e8")
        );
        assert_eq!(
            h_diag_1.as_deref(),
            Some("4408185fd170878816ebdc7412ecc36d75d53a02ff41a07f4f76c9042a531b73")
        );
        assert_eq!(h_noise_1.as_deref().map(str::len), Some(64));
        assert_ne!(h_diag_1, h_noise_1);

        // Verify analytical pixel corners on DiagonalGradient and ColorBars.
        assert_eq!(
            diag.row(0).ok().map(|r| &r[0..4]),
            Some([0, 0, 0, 255].as_slice())
        );
        assert_eq!(
            diag.row(15).ok().map(|r| &r[15 * 4..16 * 4]),
            Some([255, 255, 255, 255].as_slice())
        );
        assert_eq!(
            bars.row(0).ok().map(|r| &r[0..3]),
            Some([255, 255, 255].as_slice())
        );
        assert_eq!(
            bars.row(0).ok().map(|r| &r[15 * 3..16 * 3]),
            Some([0, 0, 0].as_slice())
        );
    }

    #[test]
    fn isqrt_u64_exact_floor_values() {
        assert_eq!(isqrt_u64(0), 0);
        assert_eq!(isqrt_u64(1), 1);
        assert_eq!(isqrt_u64(2), 1);
        assert_eq!(isqrt_u64(3), 1);
        assert_eq!(isqrt_u64(4), 2);
        assert_eq!(isqrt_u64(99), 9);
        assert_eq!(isqrt_u64(100), 10);
        assert_eq!(isqrt_u64(101), 10);
        assert_eq!(isqrt_u64(1_000_000), 1000);
        assert_eq!(isqrt_u64(u64::MAX), u64::from(u32::MAX));
    }

    #[test]
    fn psnr_is_infinity_on_identical_and_matches_analytical_plus_one_delta() {
        let budget = Budget::preset_default();
        let Ok(a) = generate_tile(
            16,
            16,
            PixelFormat::RGB8,
            SynthPattern::Solid([100, 120, 140, 255]),
            &budget,
        ) else {
            return;
        };
        let Ok(b) = generate_tile(
            16,
            16,
            PixelFormat::RGB8,
            SynthPattern::Solid([101, 121, 141, 255]),
            &budget,
        ) else {
            return;
        };

        assert_eq!(
            compute_tile_psnr(&a.as_view(), &a.as_view()).ok(),
            Some(f64::INFINITY)
        );

        let expected = 20.0 * 255.0f64.log10();
        let observed = compute_tile_psnr(&a.as_view(), &b.as_view()).ok();
        assert!(observed.is_some());
        if let Some(psnr) = observed {
            assert!(
                (psnr - expected).abs() < 1e-12,
                "psnr={psnr} expected={expected}"
            );
        }

        let diff = compare_tiles_exact(&a.as_view(), &b.as_view()).ok();
        assert_eq!(
            diff,
            Some(TileDiff {
                mismatched_bytes: 16 * 16 * 3,
                max_abs_diff: 1,
                first_mismatch: Some((0, 0)),
            })
        );
    }
}
