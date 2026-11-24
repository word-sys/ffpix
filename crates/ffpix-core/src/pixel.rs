//! Pixel formats, channel layouts, alpha modes, EXIF orientation, and color space descriptors.

use std::fmt;

use crate::error::{FfpError, FfpResult};
use crate::geom::Size;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
    U8,
    U16,
    F32,
}

impl SampleType {
    pub const ALL: [Self; 3] = [Self::U8, Self::U16, Self::F32];

    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }

    #[must_use]
    pub const fn bits_per_sample(self) -> u8 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::F32 => 32,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32)
    }
}

impl fmt::Display for SampleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::F32 => "f32",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
    Cmyk,
    YCbCr,
    Lab,
    Xyz,
}

impl ChannelLayout {
    pub const ALL: [Self; 8] = [
        Self::Gray,
        Self::GrayAlpha,
        Self::Rgb,
        Self::Rgba,
        Self::Cmyk,
        Self::YCbCr,
        Self::Lab,
        Self::Xyz,
    ];

    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb | Self::YCbCr | Self::Lab | Self::Xyz => 3,
            Self::Rgba | Self::Cmyk => 4,
        }
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Gray => "gray",
            Self::GrayAlpha => "graya",
            Self::Rgb => "rgb",
            Self::Rgba => "rgba",
            Self::Cmyk => "cmyk",
            Self::YCbCr => "ycbcr",
            Self::Lab => "lab",
            Self::Xyz => "xyz",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    Opaque,
    Straight,
    Premultiplied,
}

impl AlphaMode {
    /// Layouts without an alpha channel require `Opaque`; layouts with an alpha
    /// channel require `Straight` or `Premultiplied`.
    pub fn validate_with_layout(self, layout: ChannelLayout) -> FfpResult<()> {
        match (layout.has_alpha(), self) {
            (false, Self::Opaque) | (true, Self::Straight | Self::Premultiplied) => Ok(()),
            (false, Self::Straight | Self::Premultiplied) => Err(FfpError::format_mismatch(
                format!("alpha mode {self} requires a channel layout with alpha, got {layout}"),
            )),
            (true, Self::Opaque) => Err(FfpError::format_mismatch(format!(
                "alpha mode opaque is inconsistent with alpha-bearing layout {layout}"
            ))),
        }
    }
}

impl fmt::Display for AlphaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Opaque => "opaque",
            Self::Straight => "straight",
            Self::Premultiplied => "premultiplied",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelFormat {
    pub layout: ChannelLayout,
    pub sample: SampleType,
}

impl PixelFormat {
    pub const GRAY8: Self = Self::new(ChannelLayout::Gray, SampleType::U8);
    pub const GRAY16: Self = Self::new(ChannelLayout::Gray, SampleType::U16);
    pub const GRAY32F: Self = Self::new(ChannelLayout::Gray, SampleType::F32);
    pub const GRAYA8: Self = Self::new(ChannelLayout::GrayAlpha, SampleType::U8);
    pub const GRAYA16: Self = Self::new(ChannelLayout::GrayAlpha, SampleType::U16);
    pub const GRAYA32F: Self = Self::new(ChannelLayout::GrayAlpha, SampleType::F32);
    pub const RGB8: Self = Self::new(ChannelLayout::Rgb, SampleType::U8);
    pub const RGB16: Self = Self::new(ChannelLayout::Rgb, SampleType::U16);
    pub const RGB32F: Self = Self::new(ChannelLayout::Rgb, SampleType::F32);
    pub const RGBA8: Self = Self::new(ChannelLayout::Rgba, SampleType::U8);
    pub const RGBA16: Self = Self::new(ChannelLayout::Rgba, SampleType::U16);
    pub const RGBA32F: Self = Self::new(ChannelLayout::Rgba, SampleType::F32);
    pub const CMYK8: Self = Self::new(ChannelLayout::Cmyk, SampleType::U8);
    pub const CMYK16: Self = Self::new(ChannelLayout::Cmyk, SampleType::U16);

    #[must_use]
    pub const fn new(layout: ChannelLayout, sample: SampleType) -> Self {
        Self { layout, sample }
    }

    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        self.sample.bytes_per_sample()
    }

    #[must_use]
    pub const fn channels(self) -> usize {
        self.layout.channels()
    }

    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_sample()
    }

    /// Computes tight row byte width, rejecting overflow above `isize::MAX`.
    pub fn checked_row_bytes(self, width: u32) -> FfpResult<usize> {
        let w = usize::try_from(width)
            .map_err(|_| FfpError::overflow("width exceeds addressable usize"))?;
        let bytes = w
            .checked_mul(self.bytes_per_pixel())
            .ok_or_else(|| FfpError::overflow("row byte size overflow"))?;
        if bytes > isize::MAX as usize {
            return Err(FfpError::overflow("row byte size exceeds isize::MAX"));
        }
        Ok(bytes)
    }

    /// Computes tight buffer byte size (`width * height * bytes_per_pixel`),
    /// rejecting any overflow above `isize::MAX` (Rust's allocation ceiling).
    pub fn checked_buffer_bytes(self, width: u32, height: u32) -> FfpResult<usize> {
        let row = self.checked_row_bytes(width)?;
        let h = usize::try_from(height)
            .map_err(|_| FfpError::overflow("height exceeds addressable usize"))?;
        let bytes = row
            .checked_mul(h)
            .ok_or_else(|| FfpError::overflow("buffer byte size overflow"))?;
        if bytes > isize::MAX as usize {
            return Err(FfpError::overflow("buffer byte size exceeds isize::MAX"));
        }
        Ok(bytes)
    }

    pub fn checked_buffer_bytes_u64(self, width: u32, height: u32) -> FfpResult<u64> {
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| FfpError::overflow("pixel count overflow"))?;
        pixels
            .checked_mul(self.bytes_per_pixel() as u64)
            .ok_or_else(|| FfpError::overflow("buffer byte size overflow"))
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.layout, self.sample)
    }
}

/// EXIF TIFF Orientation tag values (1..=8).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Orientation {
    #[default]
    Normal = 1,
    FlipH = 2,
    Rotate180 = 3,
    FlipV = 4,
    Transpose = 5,
    Rotate90 = 6,
    Transverse = 7,
    Rotate270 = 8,
}

impl Orientation {
    pub const ALL: [Self; 8] = [
        Self::Normal,
        Self::FlipH,
        Self::Rotate180,
        Self::FlipV,
        Self::Transpose,
        Self::Rotate90,
        Self::Transverse,
        Self::Rotate270,
    ];

    #[must_use]
    pub const fn from_exif_u16(val: u16) -> Option<Self> {
        match val {
            1 => Some(Self::Normal),
            2 => Some(Self::FlipH),
            3 => Some(Self::Rotate180),
            4 => Some(Self::FlipV),
            5 => Some(Self::Transpose),
            6 => Some(Self::Rotate90),
            7 => Some(Self::Transverse),
            8 => Some(Self::Rotate270),
            _ => None,
        }
    }

    #[must_use]
    pub const fn to_exif_u16(self) -> u16 {
        self as u16
    }

    #[must_use]
    pub const fn swaps_dimensions(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90 | Self::Transverse | Self::Rotate270
        )
    }

    #[must_use]
    pub const fn oriented_size(self, size: Size) -> Size {
        if self.swaps_dimensions() {
            Size::new(size.h, size.w)
        } else {
            size
        }
    }
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Normal => "normal",
            Self::FlipH => "flip-h",
            Self::Rotate180 => "rotate-180",
            Self::FlipV => "flip-v",
            Self::Transpose => "transpose",
            Self::Rotate90 => "rotate-90",
            Self::Transverse => "transverse",
            Self::Rotate270 => "rotate-270",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    #[default]
    Srgb,
    LinearSrgb,
    Rec709,
    DisplayP3,
    AdobeRgb,
    Rec2020,
    XyzD50,
    XyzD65,
    Lab,
    OkLab,
    Gray,
    LinearGray,
    Cmyk,
    Unknown,
}

impl ColorSpace {
    #[must_use]
    pub const fn is_linear(self) -> bool {
        matches!(
            self,
            Self::LinearSrgb | Self::LinearGray | Self::XyzD50 | Self::XyzD65
        )
    }
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Srgb => "srgb",
            Self::LinearSrgb => "linear-srgb",
            Self::Rec709 => "rec709",
            Self::DisplayP3 => "display-p3",
            Self::AdobeRgb => "adobe-rgb",
            Self::Rec2020 => "rec2020",
            Self::XyzD50 => "xyz-d50",
            Self::XyzD65 => "xyz-d65",
            Self::Lab => "lab",
            Self::OkLab => "oklab",
            Self::Gray => "gray",
            Self::LinearGray => "linear-gray",
            Self::Cmyk => "cmyk",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// ITU-T H.273 Coding-Independent Code Points (CICP).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cicp {
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub full_range: bool,
}

impl Cicp {
    pub const SRGB: Self = Self::new(1, 13, 0, true);
    pub const DISPLAY_P3: Self = Self::new(12, 13, 0, true);
    pub const REC2020_PQ: Self = Self::new(9, 16, 0, true);
    pub const REC2020_HLG: Self = Self::new(9, 18, 0, true);

    #[must_use]
    pub const fn new(primaries: u8, transfer: u8, matrix: u8, full_range: bool) -> Self {
        Self {
            primaries,
            transfer,
            matrix,
            full_range,
        }
    }

    pub fn from_bytes(bytes: [u8; 4]) -> FfpResult<Self> {
        let full_range = match bytes[3] {
            0 => false,
            1 => true,
            other => {
                return Err(FfpError::corrupt_data(format!(
                    "invalid CICP video full-range flag: {other}"
                )));
            }
        };
        Ok(Self {
            primaries: bytes[0],
            transfer: bytes[1],
            matrix: bytes[2],
            full_range,
        })
    }

    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        [
            self.primaries,
            self.transfer,
            self.matrix,
            if self.full_range { 1 } else { 0 },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatusCode;

    #[test]
    fn all_24_pixel_format_combinations_have_exact_byte_widths() {
        let mut count = 0usize;
        for layout in ChannelLayout::ALL {
            for sample in SampleType::ALL {
                let fmt = PixelFormat::new(layout, sample);
                let expected_bps = match sample {
                    SampleType::U8 => 1,
                    SampleType::U16 => 2,
                    SampleType::F32 => 4,
                };
                let expected_ch = match layout {
                    ChannelLayout::Gray => 1,
                    ChannelLayout::GrayAlpha => 2,
                    ChannelLayout::Rgb
                    | ChannelLayout::YCbCr
                    | ChannelLayout::Lab
                    | ChannelLayout::Xyz => 3,
                    ChannelLayout::Rgba | ChannelLayout::Cmyk => 4,
                };
                assert_eq!(fmt.bytes_per_sample(), expected_bps);
                assert_eq!(fmt.channels(), expected_ch);
                assert_eq!(fmt.bytes_per_pixel(), expected_bps * expected_ch);
                assert_eq!(
                    fmt.checked_row_bytes(10).ok(),
                    Some(10 * expected_bps * expected_ch)
                );
                assert_eq!(
                    fmt.checked_buffer_bytes(10, 5).ok(),
                    Some(50 * expected_bps * expected_ch)
                );
                count += 1;
            }
        }
        assert_eq!(count, 24);
    }

    #[test]
    fn checked_buffer_bytes_rejects_u32_max_overflow() {
        for layout in ChannelLayout::ALL {
            for sample in SampleType::ALL {
                let fmt = PixelFormat::new(layout, sample);
                let res = fmt.checked_buffer_bytes(u32::MAX, u32::MAX);
                assert!(res.is_err(), "expected overflow for {fmt}");
                if let Err(err) = res {
                    assert_eq!(err.status_code(), StatusCode::Overflow);
                }
            }
        }
        assert!(
            PixelFormat::RGBA8
                .checked_buffer_bytes_u64(u32::MAX, u32::MAX)
                .is_err()
        );
        assert_eq!(
            PixelFormat::RGBA8.checked_buffer_bytes(0, 100).ok(),
            Some(0)
        );
    }

    #[test]
    fn alpha_mode_layout_validation() {
        for layout in ChannelLayout::ALL {
            if layout.has_alpha() {
                assert!(AlphaMode::Opaque.validate_with_layout(layout).is_err());
                assert!(AlphaMode::Straight.validate_with_layout(layout).is_ok());
                assert!(
                    AlphaMode::Premultiplied
                        .validate_with_layout(layout)
                        .is_ok()
                );
            } else {
                assert!(AlphaMode::Opaque.validate_with_layout(layout).is_ok());
                assert!(AlphaMode::Straight.validate_with_layout(layout).is_err());
                assert!(
                    AlphaMode::Premultiplied
                        .validate_with_layout(layout)
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn exif_orientation_round_trips_and_dimension_swaps() {
        assert_eq!(Orientation::from_exif_u16(0), None);
        assert_eq!(Orientation::from_exif_u16(9), None);
        assert_eq!(Orientation::from_exif_u16(u16::MAX), None);

        let src = Size::new(800, 600);
        for (idx, orient) in Orientation::ALL.iter().copied().enumerate() {
            let tag = (idx + 1) as u16;
            assert_eq!(orient.to_exif_u16(), tag);
            assert_eq!(Orientation::from_exif_u16(tag), Some(orient));

            let expected_swap = tag >= 5;
            assert_eq!(orient.swaps_dimensions(), expected_swap);
            let oriented = orient.oriented_size(src);
            if expected_swap {
                assert_eq!(oriented, Size::new(600, 800));
            } else {
                assert_eq!(oriented, src);
            }
        }
    }

    #[test]
    fn cicp_bytes_round_trip_and_flag_validation() {
        let bytes = Cicp::REC2020_PQ.to_bytes();
        assert_eq!(bytes, [9, 16, 0, 1]);
        assert_eq!(Cicp::from_bytes(bytes).ok(), Some(Cicp::REC2020_PQ));
        assert!(Cicp::from_bytes([1, 13, 0, 2]).is_err());
    }
}
