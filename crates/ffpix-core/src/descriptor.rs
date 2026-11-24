//! Image descriptors, lightweight metadata payload carriers, and privacy sanitization policies.

use std::fmt;

use crate::error::{FfpError, FfpResult};
use crate::geom::{Rect, Size};
use crate::pixel::{AlphaMode, ChannelLayout, Cicp, ColorSpace, Orientation, PixelFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SanitizePolicy {
    KeepAll,
    #[default]
    KeepSafe,
    StripAll,
}

impl fmt::Display for SanitizePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::KeepAll => "keep-all",
            Self::KeepSafe => "keep-safe",
            Self::StripAll => "strip-all",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MetadataSummary {
    pub has_exif: bool,
    pub has_xmp: bool,
    pub has_iptc: bool,
    pub has_icc: bool,
    pub has_cicp: bool,
    pub total_bytes: u64,
}

impl MetadataSummary {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        !(self.has_exif || self.has_xmp || self.has_iptc || self.has_icc || self.has_cicp)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MetadataBundle {
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub iptc: Option<Vec<u8>>,
    pub icc: Option<Vec<u8>>,
    pub cicp: Option<Cicp>,
}

impl MetadataBundle {
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        let len_of = |opt: &Option<Vec<u8>>| opt.as_ref().map_or(0u64, |v| v.len() as u64);
        len_of(&self.exif)
            .saturating_add(len_of(&self.xmp))
            .saturating_add(len_of(&self.iptc))
            .saturating_add(len_of(&self.icc))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.exif.is_none()
            && self.xmp.is_none()
            && self.iptc.is_none()
            && self.icc.is_none()
            && self.cicp.is_none()
    }

    #[must_use]
    pub fn summary(&self) -> MetadataSummary {
        MetadataSummary {
            has_exif: self.exif.is_some(),
            has_xmp: self.xmp.is_some(),
            has_iptc: self.iptc.is_some(),
            has_icc: self.icc.is_some(),
            has_cicp: self.cicp.is_some(),
            total_bytes: self.total_bytes(),
        }
    }

    pub fn clear(&mut self) {
        self.exif = None;
        self.xmp = None;
        self.iptc = None;
        self.icc = None;
        self.cicp = None;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageDescriptor {
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub color_space: ColorSpace,
    pub cicp: Option<Cicp>,
    pub alpha_mode: AlphaMode,
    pub orientation: Orientation,
    pub tile_hint: Option<Size>,
    pub frame_count: u32,
    pub dpi: Option<(f32, f32)>,
    pub metadata_summary: MetadataSummary,
}

impl ImageDescriptor {
    #[must_use]
    pub const fn new(width: u32, height: u32, pixel_format: PixelFormat) -> Self {
        let color_space = match pixel_format.layout {
            ChannelLayout::Gray | ChannelLayout::GrayAlpha => ColorSpace::Gray,
            ChannelLayout::Cmyk => ColorSpace::Cmyk,
            ChannelLayout::Lab => ColorSpace::Lab,
            ChannelLayout::Xyz => ColorSpace::XyzD65,
            ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::YCbCr => ColorSpace::Srgb,
        };
        let alpha_mode = if pixel_format.layout.has_alpha() {
            AlphaMode::Straight
        } else {
            AlphaMode::Opaque
        };
        Self {
            width,
            height,
            pixel_format,
            color_space,
            cicp: None,
            alpha_mode,
            orientation: Orientation::Normal,
            tile_hint: None,
            frame_count: 1,
            dpi: None,
            metadata_summary: MetadataSummary {
                has_exif: false,
                has_xmp: false,
                has_iptc: false,
                has_icc: false,
                has_cicp: false,
                total_bytes: 0,
            },
        }
    }

    #[must_use]
    pub const fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    #[must_use]
    pub const fn oriented_size(&self) -> Size {
        self.orientation.oriented_size(self.size())
    }

    #[must_use]
    pub const fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }

    #[must_use]
    pub const fn pixel_count(&self) -> Option<u64> {
        self.size().area()
    }

    #[must_use]
    pub const fn is_animated(&self) -> bool {
        self.frame_count > 1
    }

    pub fn validate(&self) -> FfpResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(FfpError::invalid_input(format!(
                "image dimensions must be non-zero, got {}x{}",
                self.width, self.height
            )));
        }
        if self.frame_count == 0 {
            return Err(FfpError::invalid_input("frame_count must be at least 1"));
        }
        self.alpha_mode
            .validate_with_layout(self.pixel_format.layout)?;
        if let Some(tile) = self.tile_hint
            && tile.is_empty()
        {
            return Err(FfpError::invalid_argument(
                "tile_hint dimensions must be non-zero",
            ));
        }
        if let Some((dx, dy)) = self.dpi
            && (!dx.is_finite() || !dy.is_finite() || dx <= 0.0 || dy <= 0.0)
        {
            return Err(FfpError::invalid_argument(
                "dpi values must be finite and positive",
            ));
        }
        self.pixel_format.checked_row_bytes(self.width)?;
        self.pixel_format
            .checked_buffer_bytes_u64(self.width, self.height)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatusCode;

    #[test]
    fn valid_descriptor_defaults_and_orientation() {
        let mut desc = ImageDescriptor::new(1920, 1080, PixelFormat::RGBA8);
        assert_eq!(desc.color_space, ColorSpace::Srgb);
        assert_eq!(desc.alpha_mode, AlphaMode::Straight);
        assert_eq!(desc.frame_count, 1);
        assert!(!desc.is_animated());
        assert_eq!(desc.pixel_count(), Some(1920 * 1080));
        assert!(desc.validate().is_ok());

        desc.orientation = Orientation::Rotate90;
        desc.tile_hint = Some(Size::new(256, 256));
        desc.dpi = Some((300.0, 300.0));
        desc.frame_count = 4;
        assert!(desc.is_animated());
        assert_eq!(desc.oriented_size(), Size::new(1080, 1920));
        assert!(desc.validate().is_ok());

        let gray = ImageDescriptor::new(64, 64, PixelFormat::GRAY16);
        assert_eq!(gray.color_space, ColorSpace::Gray);
        assert_eq!(gray.alpha_mode, AlphaMode::Opaque);
        assert!(gray.validate().is_ok());
    }

    #[test]
    fn invalid_descriptors_fail_deterministically() {
        let zero_w = ImageDescriptor::new(0, 100, PixelFormat::RGB8);
        assert_eq!(
            zero_w.validate().err().map(|e| e.status_code()),
            Some(StatusCode::InvalidInput)
        );

        let zero_h = ImageDescriptor::new(100, 0, PixelFormat::RGB8);
        assert_eq!(
            zero_h.validate().err().map(|e| e.status_code()),
            Some(StatusCode::InvalidInput)
        );

        let mut zero_frames = ImageDescriptor::new(10, 10, PixelFormat::RGB8);
        zero_frames.frame_count = 0;
        assert_eq!(
            zero_frames.validate().err().map(|e| e.status_code()),
            Some(StatusCode::InvalidInput)
        );

        let mut bad_alpha = ImageDescriptor::new(10, 10, PixelFormat::RGBA8);
        bad_alpha.alpha_mode = AlphaMode::Opaque;
        assert_eq!(
            bad_alpha.validate().err().map(|e| e.status_code()),
            Some(StatusCode::FormatMismatch)
        );

        let mut bad_tile = ImageDescriptor::new(10, 10, PixelFormat::RGB8);
        bad_tile.tile_hint = Some(Size::new(0, 64));
        assert_eq!(
            bad_tile.validate().err().map(|e| e.status_code()),
            Some(StatusCode::InvalidArgument)
        );

        for bad_dpi in [
            (0.0, 72.0),
            (72.0, -1.0),
            (f32::NAN, 72.0),
            (f32::INFINITY, 72.0),
        ] {
            let mut d = ImageDescriptor::new(10, 10, PixelFormat::RGB8);
            d.dpi = Some(bad_dpi);
            assert_eq!(
                d.validate().err().map(|e| e.status_code()),
                Some(StatusCode::InvalidArgument)
            );
        }

        let overflow = ImageDescriptor::new(u32::MAX, u32::MAX, PixelFormat::RGBA8);
        assert_eq!(
            overflow.validate().err().map(|e| e.status_code()),
            Some(StatusCode::Overflow)
        );
    }

    #[test]
    fn metadata_bundle_byte_accounting_and_summary() {
        let mut bundle = MetadataBundle::default();
        assert!(bundle.is_empty());
        assert_eq!(bundle.total_bytes(), 0);
        assert!(bundle.summary().is_empty());

        bundle.exif = Some(vec![0u8; 128]);
        bundle.xmp = Some(vec![b'x'; 256]);
        bundle.iptc = Some(vec![1u8; 64]);
        bundle.icc = Some(vec![2u8; 512]);
        bundle.cicp = Some(Cicp::DISPLAY_P3);

        assert!(!bundle.is_empty());
        assert_eq!(bundle.total_bytes(), 128 + 256 + 64 + 512);

        let summary = bundle.summary();
        assert!(!summary.is_empty());
        assert!(summary.has_exif);
        assert!(summary.has_xmp);
        assert!(summary.has_iptc);
        assert!(summary.has_icc);
        assert!(summary.has_cicp);
        assert_eq!(summary.total_bytes, 960);

        bundle.clear();
        assert!(bundle.is_empty());
        assert_eq!(bundle.total_bytes(), 0);
    }

    #[test]
    fn sanitize_policy_defaults_to_keep_safe() {
        assert_eq!(SanitizePolicy::default(), SanitizePolicy::KeepSafe);
        assert_eq!(SanitizePolicy::KeepAll.to_string(), "keep-all");
        assert_eq!(SanitizePolicy::KeepSafe.to_string(), "keep-safe");
        assert_eq!(SanitizePolicy::StripAll.to_string(), "strip-all");
    }
}
