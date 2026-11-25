//! Safe endianness, fixed-point, sample type, and canonical scalar row conversions.

use crate::error::{FfpError, FfpResult};
use crate::pixel::{ChannelLayout, PixelFormat, SampleType};

#[must_use]
#[inline]
pub const fn u8_to_u16(v: u8) -> u16 {
    (v as u16) * 257
}

#[must_use]
#[inline]
pub const fn u16_to_u8(v: u16) -> u8 {
    (((v as u32) + 128) / 257) as u8
}

#[must_use]
#[inline]
pub fn u8_to_f32(v: u8) -> f32 {
    f32::from(v) / 255.0
}

#[must_use]
#[inline]
pub fn f32_to_u8(v: f32) -> u8 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    if v >= 1.0 {
        return 255;
    }
    (v * 255.0).round_ties_even() as u8
}

#[must_use]
#[inline]
pub fn u16_to_f32(v: u16) -> f32 {
    f32::from(v) / 65535.0
}

#[must_use]
#[inline]
pub fn f32_to_u16(v: f32) -> u16 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    if v >= 1.0 {
        return 65535;
    }
    (v * 65535.0).round_ties_even() as u16
}

/// Exact rounded `x / 255` for `x <= 65025` (`255 * 255`).
#[must_use]
#[inline]
pub const fn div255_round(x: u32) -> u8 {
    let clamped = if x > 65_025 { 65_025 } else { x };
    let t = clamped + 128;
    ((t + (t >> 8)) >> 8) as u8
}

/// Exact rounded `x / 65535` clamped to `u16`.
#[must_use]
#[inline]
pub const fn div65535_round(x: u64) -> u16 {
    let q = (x.saturating_add(32_768)) / 65_535;
    if q > 65_535 { 65_535 } else { q as u16 }
}

#[must_use]
#[inline]
pub const fn clamp_u8_i32(x: i32) -> u8 {
    if x <= 0 {
        0
    } else if x >= 255 {
        255
    } else {
        x as u8
    }
}

#[must_use]
#[inline]
pub const fn clamp_u16_i32(x: i32) -> u16 {
    if x <= 0 {
        0
    } else if x >= 65_535 {
        65_535
    } else {
        x as u16
    }
}

pub fn read_u16_le(buf: &[u8], offset: usize) -> FfpResult<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| FfpError::overflow("u16_le offset overflow"))?;
    let bytes: [u8; 2] = buf
        .get(offset..end)
        .ok_or_else(|| FfpError::unexpected_eof("truncated u16_le"))?
        .try_into()
        .map_err(|_| FfpError::internal("u16_le slice length mismatch"))?;
    Ok(u16::from_le_bytes(bytes))
}

pub fn read_u16_be(buf: &[u8], offset: usize) -> FfpResult<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| FfpError::overflow("u16_be offset overflow"))?;
    let bytes: [u8; 2] = buf
        .get(offset..end)
        .ok_or_else(|| FfpError::unexpected_eof("truncated u16_be"))?
        .try_into()
        .map_err(|_| FfpError::internal("u16_be slice length mismatch"))?;
    Ok(u16::from_be_bytes(bytes))
}

pub fn read_u32_le(buf: &[u8], offset: usize) -> FfpResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| FfpError::overflow("u32_le offset overflow"))?;
    let bytes: [u8; 4] = buf
        .get(offset..end)
        .ok_or_else(|| FfpError::unexpected_eof("truncated u32_le"))?
        .try_into()
        .map_err(|_| FfpError::internal("u32_le slice length mismatch"))?;
    Ok(u32::from_le_bytes(bytes))
}

pub fn read_u32_be(buf: &[u8], offset: usize) -> FfpResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| FfpError::overflow("u32_be offset overflow"))?;
    let bytes: [u8; 4] = buf
        .get(offset..end)
        .ok_or_else(|| FfpError::unexpected_eof("truncated u32_be"))?
        .try_into()
        .map_err(|_| FfpError::internal("u32_be slice length mismatch"))?;
    Ok(u32::from_be_bytes(bytes))
}

pub fn read_f32_le(buf: &[u8], offset: usize) -> FfpResult<f32> {
    Ok(f32::from_bits(read_u32_le(buf, offset)?))
}

pub fn read_f32_be(buf: &[u8], offset: usize) -> FfpResult<f32> {
    Ok(f32::from_bits(read_u32_be(buf, offset)?))
}

pub fn write_u16_le(buf: &mut [u8], offset: usize, val: u16) -> FfpResult<()> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| FfpError::overflow("u16_le write offset overflow"))?;
    let dst = buf
        .get_mut(offset..end)
        .ok_or_else(|| FfpError::invalid_argument("u16_le write out of bounds"))?;
    dst.copy_from_slice(&val.to_le_bytes());
    Ok(())
}

pub fn write_u16_be(buf: &mut [u8], offset: usize, val: u16) -> FfpResult<()> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| FfpError::overflow("u16_be write offset overflow"))?;
    let dst = buf
        .get_mut(offset..end)
        .ok_or_else(|| FfpError::invalid_argument("u16_be write out of bounds"))?;
    dst.copy_from_slice(&val.to_be_bytes());
    Ok(())
}

pub fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) -> FfpResult<()> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| FfpError::overflow("u32_le write offset overflow"))?;
    let dst = buf
        .get_mut(offset..end)
        .ok_or_else(|| FfpError::invalid_argument("u32_le write out of bounds"))?;
    dst.copy_from_slice(&val.to_le_bytes());
    Ok(())
}

pub fn write_u32_be(buf: &mut [u8], offset: usize, val: u32) -> FfpResult<()> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| FfpError::overflow("u32_be write offset overflow"))?;
    let dst = buf
        .get_mut(offset..end)
        .ok_or_else(|| FfpError::invalid_argument("u32_be write out of bounds"))?;
    dst.copy_from_slice(&val.to_be_bytes());
    Ok(())
}

fn is_basic_layout(layout: ChannelLayout) -> bool {
    matches!(
        layout,
        ChannelLayout::Gray | ChannelLayout::GrayAlpha | ChannelLayout::Rgb | ChannelLayout::Rgba
    )
}

#[inline]
fn read_sample_f32(bytes: &[u8], sample: SampleType) -> f32 {
    match sample {
        SampleType::U8 => u8_to_f32(bytes[0]),
        SampleType::U16 => u16_to_f32(u16::from_ne_bytes([bytes[0], bytes[1]])),
        SampleType::F32 => f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    }
}

#[inline]
fn write_sample_f32(dst: &mut [u8], sample: SampleType, val: f32) {
    match sample {
        SampleType::U8 => dst[0] = f32_to_u8(val),
        SampleType::U16 => dst[..2].copy_from_slice(&f32_to_u16(val).to_ne_bytes()),
        SampleType::F32 => {
            let safe = if val.is_nan() { 0.0 } else { val };
            dst[..4].copy_from_slice(&safe.to_ne_bytes());
        }
    }
}

#[inline]
fn rgb_to_luma(r: f32, g: f32, b: f32) -> f32 {
    // Rec.709 luma weights; neutral grays preserve exact channel value.
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Converts a single interleaved scanline between `Gray`/`GrayAlpha`/`Rgb`/`Rgba`
/// across `U8`/`U16`/`F32` sample types.
pub fn convert_row_format(
    src: &[u8],
    src_fmt: PixelFormat,
    dst: &mut [u8],
    dst_fmt: PixelFormat,
    width: u32,
) -> FfpResult<()> {
    if !is_basic_layout(src_fmt.layout) || !is_basic_layout(dst_fmt.layout) {
        return Err(FfpError::unsupported_feature(format!(
            "convert_row_format does not convert between {src_fmt} and {dst_fmt}"
        )));
    }
    let src_bytes = src_fmt.checked_row_bytes(width)?;
    let dst_bytes = dst_fmt.checked_row_bytes(width)?;
    if src.len() < src_bytes {
        return Err(FfpError::invalid_argument(format!(
            "source row too short: got {}, need {src_bytes}",
            src.len()
        )));
    }
    if dst.len() < dst_bytes {
        return Err(FfpError::invalid_argument(format!(
            "destination row too short: got {}, need {dst_bytes}",
            dst.len()
        )));
    }
    if src_fmt == dst_fmt {
        dst[..dst_bytes].copy_from_slice(&src[..src_bytes]);
        return Ok(());
    }

    let sbps = src_fmt.bytes_per_sample();
    let dbps = dst_fmt.bytes_per_sample();
    let sbpp = src_fmt.bytes_per_pixel();
    let dbpp = dst_fmt.bytes_per_pixel();

    let src_pixels = src[..src_bytes].chunks_exact(sbpp);
    let dst_pixels = dst[..dst_bytes].chunks_exact_mut(dbpp);

    for (sp, dp) in src_pixels.zip(dst_pixels) {
        let (r, g, b, a) = match src_fmt.layout {
            ChannelLayout::Gray => {
                let y = read_sample_f32(&sp[..sbps], src_fmt.sample);
                (y, y, y, 1.0)
            }
            ChannelLayout::GrayAlpha => {
                let y = read_sample_f32(&sp[..sbps], src_fmt.sample);
                let a = read_sample_f32(&sp[sbps..2 * sbps], src_fmt.sample);
                (y, y, y, a)
            }
            ChannelLayout::Rgb => {
                let r = read_sample_f32(&sp[..sbps], src_fmt.sample);
                let g = read_sample_f32(&sp[sbps..2 * sbps], src_fmt.sample);
                let b = read_sample_f32(&sp[2 * sbps..3 * sbps], src_fmt.sample);
                (r, g, b, 1.0)
            }
            ChannelLayout::Rgba => {
                let r = read_sample_f32(&sp[..sbps], src_fmt.sample);
                let g = read_sample_f32(&sp[sbps..2 * sbps], src_fmt.sample);
                let b = read_sample_f32(&sp[2 * sbps..3 * sbps], src_fmt.sample);
                let a = read_sample_f32(&sp[3 * sbps..4 * sbps], src_fmt.sample);
                (r, g, b, a)
            }
            _ => continue,
        };

        match dst_fmt.layout {
            ChannelLayout::Gray => {
                let y = if matches!(
                    src_fmt.layout,
                    ChannelLayout::Gray | ChannelLayout::GrayAlpha
                ) {
                    r
                } else {
                    rgb_to_luma(r, g, b)
                };
                write_sample_f32(&mut dp[..dbps], dst_fmt.sample, y);
            }
            ChannelLayout::GrayAlpha => {
                let y = if matches!(
                    src_fmt.layout,
                    ChannelLayout::Gray | ChannelLayout::GrayAlpha
                ) {
                    r
                } else {
                    rgb_to_luma(r, g, b)
                };
                write_sample_f32(&mut dp[..dbps], dst_fmt.sample, y);
                write_sample_f32(&mut dp[dbps..2 * dbps], dst_fmt.sample, a);
            }
            ChannelLayout::Rgb => {
                write_sample_f32(&mut dp[..dbps], dst_fmt.sample, r);
                write_sample_f32(&mut dp[dbps..2 * dbps], dst_fmt.sample, g);
                write_sample_f32(&mut dp[2 * dbps..3 * dbps], dst_fmt.sample, b);
            }
            ChannelLayout::Rgba => {
                write_sample_f32(&mut dp[..dbps], dst_fmt.sample, r);
                write_sample_f32(&mut dp[dbps..2 * dbps], dst_fmt.sample, g);
                write_sample_f32(&mut dp[2 * dbps..3 * dbps], dst_fmt.sample, b);
                write_sample_f32(&mut dp[3 * dbps..4 * dbps], dst_fmt.sample, a);
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_u8_round_trips() {
        for v in 0..=255u8 {
            assert_eq!(u16_to_u8(u8_to_u16(v)), v);
            assert_eq!(f32_to_u8(u8_to_f32(v)), v);
            assert_eq!(f32_to_u16(u8_to_f32(v)), u8_to_u16(v));
        }
    }

    #[test]
    fn exhaustive_u16_round_trips() {
        for v in 0..=65_535u16 {
            assert_eq!(f32_to_u16(u16_to_f32(v)), v);
        }
        assert_eq!(u16_to_u8(0), 0);
        assert_eq!(u16_to_u8(128), 0);
        assert_eq!(u16_to_u8(129), 1);
        assert_eq!(u16_to_u8(65_535), 255);
    }

    #[test]
    fn nan_infinity_and_out_of_range_clamping() {
        for &x in &[f32::NAN, f32::NEG_INFINITY, -10.0, -0.0, 0.0] {
            assert_eq!(f32_to_u8(x), 0);
            assert_eq!(f32_to_u16(x), 0);
        }
        for &x in &[f32::INFINITY, 100.0, 1.0] {
            assert_eq!(f32_to_u8(x), 255);
            assert_eq!(f32_to_u16(x), 65_535);
        }
    }

    #[test]
    fn fixed_point_div255_and_endian_helpers() {
        for a in 0..=255u32 {
            for b in 0..=255u32 {
                let prod = a * b;
                let exact = ((prod as f64) / 255.0).round() as u8;
                assert_eq!(div255_round(prod), exact, "a={a}, b={b}");
            }
        }
        assert_eq!(div65535_round(0), 0);
        assert_eq!(div65535_round(65_535 * 65_535), 65_535);
        assert_eq!(clamp_u8_i32(-50), 0);
        assert_eq!(clamp_u8_i32(300), 255);
        assert_eq!(clamp_u16_i32(-1), 0);
        assert_eq!(clamp_u16_i32(70_000), 65_535);

        let mut buf = [0u8; 8];
        assert!(write_u16_le(&mut buf, 0, 0x1234).is_ok());
        assert!(write_u16_be(&mut buf, 2, 0x1234).is_ok());
        assert!(write_u32_be(&mut buf, 4, 0xDEAD_BEEF).is_ok());
        assert_eq!(read_u16_le(&buf, 0).ok(), Some(0x1234));
        assert_eq!(read_u16_be(&buf, 2).ok(), Some(0x1234));
        assert_eq!(read_u32_be(&buf, 4).ok(), Some(0xDEAD_BEEF));
        assert!(read_u32_le(&buf, 6).is_err());
        assert!(read_u32_le(&buf, usize::MAX).is_err());
    }

    #[test]
    fn convert_row_format_covers_layouts_and_preserves_neutral_gray() {
        // RGBA8 -> RGBA16 -> RGBA32F -> RGBA8 exact round-trip.
        let src_rgba8 = [0u8, 64, 128, 255, 255, 192, 32, 0];
        let mut rgba16 = [0u8; 16];
        let mut rgba32f = [0u8; 32];
        let mut back = [0u8; 8];
        assert!(
            convert_row_format(
                &src_rgba8,
                PixelFormat::RGBA8,
                &mut rgba16,
                PixelFormat::RGBA16,
                2
            )
            .is_ok()
        );
        assert!(
            convert_row_format(
                &rgba16,
                PixelFormat::RGBA16,
                &mut rgba32f,
                PixelFormat::RGBA32F,
                2
            )
            .is_ok()
        );
        assert!(
            convert_row_format(
                &rgba32f,
                PixelFormat::RGBA32F,
                &mut back,
                PixelFormat::RGBA8,
                2
            )
            .is_ok()
        );
        assert_eq!(back, src_rgba8);

        // Gray8 -> RGB8 -> Gray8 is exact identity for all 256 gray levels.
        let gray_in: Vec<u8> = (0..=255u8).collect();
        let mut rgb = vec![0u8; 256 * 3];
        let mut gray_out = vec![0u8; 256];
        assert!(
            convert_row_format(
                &gray_in,
                PixelFormat::GRAY8,
                &mut rgb,
                PixelFormat::RGB8,
                256
            )
            .is_ok()
        );
        assert!(
            convert_row_format(
                &rgb,
                PixelFormat::RGB8,
                &mut gray_out,
                PixelFormat::GRAY8,
                256
            )
            .is_ok()
        );
        assert_eq!(gray_out, gray_in);

        // Unsupported color space layout rejected cleanly.
        let mut cmyk = [0u8; 8];
        assert!(
            convert_row_format(
                &src_rgba8,
                PixelFormat::RGBA8,
                &mut cmyk,
                PixelFormat::CMYK8,
                2
            )
            .is_err()
        );
    }
}
