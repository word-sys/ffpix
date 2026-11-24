//! Structured error taxonomy and deterministic C ABI status codes for `ffpix`.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::io;

/// Convenient result alias using [`FfpError`].
pub type FfpResult<T> = Result<T, FfpError>;

/// Deterministic C99 ABI status codes.
///
/// `0` indicates success; all error variants map to unique, stable negative
/// `i32` values matching `ffp_status` in `include/ffpix.h`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatusCode {
    /// Operation completed successfully.
    Ok = 0,
    /// Underlying I/O operation failed.
    Io = -1,
    /// Input ended before the expected bytes could be read.
    UnexpectedEof = -2,
    /// Input stream or structure is invalid.
    InvalidInput = -3,
    /// Unrecognized or malformed format signature / magic bytes.
    InvalidMagic = -4,
    /// Image container or codec format is not supported.
    UnsupportedFormat = -5,
    /// Specific feature within a supported format is not supported.
    UnsupportedFeature = -6,
    /// Encoded payload or metadata is corrupted or violates specification.
    CorruptData = -7,
    /// Checked arithmetic overflow occurred while computing sizes or offsets.
    Overflow = -8,
    /// Configured safety or resource [`BudgetKind`] limit was exceeded.
    BudgetExceeded = -9,
    /// Cyclic reference detected in DAG graph or metadata/IFD structure.
    CycleDetected = -10,
    /// Caller supplied an invalid argument, range, or null pointer.
    InvalidArgument = -11,
    /// Image, tile, or buffer dimensions do not match expectations.
    DimensionMismatch = -12,
    /// Pixel format, channel layout, or sample type does not match expectations.
    FormatMismatch = -13,
    /// Unexpected internal engine invariant failure.
    Internal = -14,
}

impl StatusCode {
    /// Complete ordered list of all status codes (success plus every error).
    pub const ALL: [Self; 15] = [
        Self::Ok,
        Self::Io,
        Self::UnexpectedEof,
        Self::InvalidInput,
        Self::InvalidMagic,
        Self::UnsupportedFormat,
        Self::UnsupportedFeature,
        Self::CorruptData,
        Self::Overflow,
        Self::BudgetExceeded,
        Self::CycleDetected,
        Self::InvalidArgument,
        Self::DimensionMismatch,
        Self::FormatMismatch,
        Self::Internal,
    ];

    /// Returns the raw `i32` representation used across the C ABI.
    #[must_use]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// Converts a raw `i32` status code back into a [`StatusCode`].
    #[must_use]
    pub const fn from_i32(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Ok),
            -1 => Some(Self::Io),
            -2 => Some(Self::UnexpectedEof),
            -3 => Some(Self::InvalidInput),
            -4 => Some(Self::InvalidMagic),
            -5 => Some(Self::UnsupportedFormat),
            -6 => Some(Self::UnsupportedFeature),
            -7 => Some(Self::CorruptData),
            -8 => Some(Self::Overflow),
            -9 => Some(Self::BudgetExceeded),
            -10 => Some(Self::CycleDetected),
            -11 => Some(Self::InvalidArgument),
            -12 => Some(Self::DimensionMismatch),
            -13 => Some(Self::FormatMismatch),
            -14 => Some(Self::Internal),
            _ => None,
        }
    }

    /// Returns `true` if the status code represents success (`Ok`).
    #[must_use]
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Returns `true` if the status code represents an error.
    #[must_use]
    pub const fn is_err(self) -> bool {
        !self.is_ok()
    }

    /// Returns the canonical C ABI constant name for this status code.
    #[must_use]
    pub const fn c_name(self) -> &'static str {
        match self {
            Self::Ok => "FFP_OK",
            Self::Io => "FFP_ERR_IO",
            Self::UnexpectedEof => "FFP_ERR_UNEXPECTED_EOF",
            Self::InvalidInput => "FFP_ERR_INVALID_INPUT",
            Self::InvalidMagic => "FFP_ERR_INVALID_MAGIC",
            Self::UnsupportedFormat => "FFP_ERR_UNSUPPORTED_FORMAT",
            Self::UnsupportedFeature => "FFP_ERR_UNSUPPORTED_FEATURE",
            Self::CorruptData => "FFP_ERR_CORRUPT_DATA",
            Self::Overflow => "FFP_ERR_OVERFLOW",
            Self::BudgetExceeded => "FFP_ERR_BUDGET_EXCEEDED",
            Self::CycleDetected => "FFP_ERR_CYCLE_DETECTED",
            Self::InvalidArgument => "FFP_ERR_INVALID_ARGUMENT",
            Self::DimensionMismatch => "FFP_ERR_DIMENSION_MISMATCH",
            Self::FormatMismatch => "FFP_ERR_FORMAT_MISMATCH",
            Self::Internal => "FFP_ERR_INTERNAL",
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.c_name(), self.as_i32())
    }
}

/// Identifies which resource or safety budget cap was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetKind {
    /// Maximum allowed image width in pixels.
    Width,
    /// Maximum allowed image height in pixels.
    Height,
    /// Maximum allowed total pixel count (`width * height`).
    Pixels,
    /// Maximum allowed concurrent live memory bytes.
    MemoryBytes,
    /// Maximum allowed frame or page count in animated/multi-page formats.
    Frames,
    /// Maximum allowed ancillary metadata bytes (EXIF, XMP, IPTC, ICC).
    MetadataBytes,
    /// Maximum allowed decompressed-to-compressed expansion ratio.
    CompressionRatio,
    /// Maximum allowed decompressed stream output bytes.
    InflateBytes,
    /// Maximum allowed recursion or nesting depth (IFD chains, boxes, DAG).
    RecursionDepth,
    /// Maximum allowed bytes written to an output sink.
    OutputBytes,
}

impl BudgetKind {
    /// Returns a concise static label describing the exceeded budget kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Width => "max_width",
            Self::Height => "max_height",
            Self::Pixels => "max_pixels",
            Self::MemoryBytes => "max_memory_bytes",
            Self::Frames => "max_frames",
            Self::MetadataBytes => "max_metadata_bytes",
            Self::CompressionRatio => "max_compression_ratio",
            Self::InflateBytes => "max_inflate_bytes",
            Self::RecursionDepth => "max_recursion_depth",
            Self::OutputBytes => "max_output_bytes",
        }
    }
}

impl fmt::Display for BudgetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Unified error type returned by all `ffpix` crates.
#[derive(Debug)]
pub enum FfpError {
    /// Underlying I/O error (excluding `UnexpectedEof`, which maps to [`Self::UnexpectedEof`]).
    Io(io::Error),
    /// Stream reached EOF prematurely.
    UnexpectedEof(Cow<'static, str>),
    /// Input stream, header, or container structure is invalid.
    InvalidInput(Cow<'static, str>),
    /// Format magic bytes did not match any expected signature.
    InvalidMagic(Cow<'static, str>),
    /// Image format is recognized or unknown and not supported for this operation.
    UnsupportedFormat(Cow<'static, str>),
    /// Specific codec or pipeline feature is not supported.
    UnsupportedFeature(Cow<'static, str>),
    /// Encoded payload, checksum, or metadata structure is corrupted.
    CorruptData(Cow<'static, str>),
    /// Arithmetic overflow occurred during checked calculation.
    Overflow(Cow<'static, str>),
    /// A configured safety or resource budget limit was exceeded.
    BudgetExceeded {
        /// Which resource limit was breached.
        kind: BudgetKind,
        /// Configured upper bound.
        limit: u64,
        /// Requested or observed value that exceeded `limit`.
        requested: u64,
    },
    /// Cyclic reference detected in DAG graph or metadata/IFD structure.
    CycleDetected(Cow<'static, str>),
    /// Invalid argument passed to API, CLI, or FFI function.
    InvalidArgument(Cow<'static, str>),
    /// Observed image or tile dimensions did not match expected dimensions.
    DimensionMismatch {
        /// Expected width in pixels.
        expected_w: u32,
        /// Expected height in pixels.
        expected_h: u32,
        /// Observed width in pixels.
        actual_w: u32,
        /// Observed height in pixels.
        actual_h: u32,
    },
    /// Observed pixel format, layout, or sample type did not match expectations.
    FormatMismatch(Cow<'static, str>),
    /// Unexpected internal error or invariant failure.
    Internal(Cow<'static, str>),
}

impl FfpError {
    /// Returns the deterministic negative C ABI [`StatusCode`] for this error.
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        match self {
            Self::Io(_) => StatusCode::Io,
            Self::UnexpectedEof(_) => StatusCode::UnexpectedEof,
            Self::InvalidInput(_) => StatusCode::InvalidInput,
            Self::InvalidMagic(_) => StatusCode::InvalidMagic,
            Self::UnsupportedFormat(_) => StatusCode::UnsupportedFormat,
            Self::UnsupportedFeature(_) => StatusCode::UnsupportedFeature,
            Self::CorruptData(_) => StatusCode::CorruptData,
            Self::Overflow(_) => StatusCode::Overflow,
            Self::BudgetExceeded { .. } => StatusCode::BudgetExceeded,
            Self::CycleDetected(_) => StatusCode::CycleDetected,
            Self::InvalidArgument(_) => StatusCode::InvalidArgument,
            Self::DimensionMismatch { .. } => StatusCode::DimensionMismatch,
            Self::FormatMismatch(_) => StatusCode::FormatMismatch,
            Self::Internal(_) => StatusCode::Internal,
        }
    }

    /// Returns `true` if this error is [`Self::BudgetExceeded`].
    #[must_use]
    pub const fn is_budget_exceeded(&self) -> bool {
        matches!(self, Self::BudgetExceeded { .. })
    }

    /// Constructs an [`FfpError::UnexpectedEof`] error.
    #[must_use]
    pub fn unexpected_eof(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::UnexpectedEof(msg.into())
    }

    /// Constructs an [`FfpError::InvalidInput`] error.
    #[must_use]
    pub fn invalid_input(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Constructs an [`FfpError::InvalidMagic`] error.
    #[must_use]
    pub fn invalid_magic(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidMagic(msg.into())
    }

    /// Constructs an [`FfpError::UnsupportedFormat`] error.
    #[must_use]
    pub fn unsupported_format(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::UnsupportedFormat(msg.into())
    }

    /// Constructs an [`FfpError::UnsupportedFeature`] error.
    #[must_use]
    pub fn unsupported_feature(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::UnsupportedFeature(msg.into())
    }

    /// Constructs an [`FfpError::CorruptData`] error.
    #[must_use]
    pub fn corrupt_data(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::CorruptData(msg.into())
    }

    /// Constructs an [`FfpError::Overflow`] error.
    #[must_use]
    pub fn overflow(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::Overflow(msg.into())
    }

    /// Constructs an [`FfpError::BudgetExceeded`] error.
    #[must_use]
    pub const fn budget_exceeded(kind: BudgetKind, limit: u64, requested: u64) -> Self {
        Self::BudgetExceeded {
            kind,
            limit,
            requested,
        }
    }

    /// Constructs an [`FfpError::CycleDetected`] error.
    #[must_use]
    pub fn cycle_detected(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::CycleDetected(msg.into())
    }

    /// Constructs an [`FfpError::InvalidArgument`] error.
    #[must_use]
    pub fn invalid_argument(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    /// Constructs an [`FfpError::DimensionMismatch`] error.
    #[must_use]
    pub const fn dimension_mismatch(
        expected_w: u32,
        expected_h: u32,
        actual_w: u32,
        actual_h: u32,
    ) -> Self {
        Self::DimensionMismatch {
            expected_w,
            expected_h,
            actual_w,
            actual_h,
        }
    }

    /// Constructs an [`FfpError::FormatMismatch`] error.
    #[must_use]
    pub fn format_mismatch(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::FormatMismatch(msg.into())
    }

    /// Constructs an [`FfpError::Internal`] error.
    #[must_use]
    pub fn internal(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::Internal(msg.into())
    }
}

impl fmt::Display for FfpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::UnexpectedEof(msg) => write!(f, "unexpected end of input: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::InvalidMagic(msg) => write!(f, "invalid magic signature: {msg}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            Self::UnsupportedFeature(msg) => write!(f, "unsupported feature: {msg}"),
            Self::CorruptData(msg) => write!(f, "corrupt data: {msg}"),
            Self::Overflow(msg) => write!(f, "arithmetic overflow: {msg}"),
            Self::BudgetExceeded {
                kind,
                limit,
                requested,
            } => write!(
                f,
                "budget exceeded ({kind}): requested {requested}, limit {limit}"
            ),
            Self::CycleDetected(msg) => write!(f, "cycle detected: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::DimensionMismatch {
                expected_w,
                expected_h,
                actual_w,
                actual_h,
            } => write!(
                f,
                "dimension mismatch: expected {expected_w}x{expected_h}, got {actual_w}x{actual_h}"
            ),
            Self::FormatMismatch(msg) => write!(f, "format mismatch: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl Error for FfpError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for FfpError {
    fn from(err: io::Error) -> Self {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            Self::UnexpectedEof(Cow::Owned(err.to_string()))
        } else {
            Self::Io(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn sample_errors() -> Vec<FfpError> {
        vec![
            FfpError::Io(io::Error::new(io::ErrorKind::NotFound, "missing file")),
            FfpError::unexpected_eof("truncated header"),
            FfpError::invalid_input("zero-length payload"),
            FfpError::invalid_magic("expected PNG signature"),
            FfpError::unsupported_format("unknown container"),
            FfpError::unsupported_feature("arithmetic JPEG"),
            FfpError::corrupt_data("CRC-32 mismatch"),
            FfpError::overflow("width * height overflow"),
            FfpError::budget_exceeded(BudgetKind::MemoryBytes, 64, 128),
            FfpError::cycle_detected("IFD0 -> IFD0"),
            FfpError::invalid_argument("quality must be 1..=100"),
            FfpError::dimension_mismatch(100, 200, 64, 64),
            FfpError::format_mismatch("expected Rgba U8"),
            FfpError::internal("worker state missing"),
        ]
    }

    #[test]
    fn status_code_bijection_and_uniqueness() {
        assert_eq!(StatusCode::Ok.as_i32(), 0);
        assert!(StatusCode::Ok.is_ok());
        assert!(!StatusCode::Ok.is_err());
        assert_eq!(StatusCode::from_i32(0), Some(StatusCode::Ok));
        assert_eq!(StatusCode::from_i32(1), None);
        assert_eq!(StatusCode::from_i32(-999), None);

        let mut raw_codes = HashSet::new();
        for code in StatusCode::ALL {
            let raw = code.as_i32();
            assert!(raw_codes.insert(raw), "duplicate status code {raw}");
            assert_eq!(StatusCode::from_i32(raw), Some(code));
            if code != StatusCode::Ok {
                assert!(raw < 0, "error status code must be negative: {code}");
                assert!(code.is_err());
            }
            let rendered = code.to_string();
            assert!(rendered.starts_with("FFP_"));
            assert!(rendered.is_ascii());
        }

        let errors = sample_errors();
        let mut mapped_codes = HashSet::new();
        for err in &errors {
            let code = err.status_code();
            assert!(code.is_err());
            assert!(mapped_codes.insert(code), "duplicate mapping for {code}");
        }
        // Every non-Ok StatusCode must be covered bijectively by FfpError variants.
        assert_eq!(mapped_codes.len(), StatusCode::ALL.len() - 1);
    }

    #[test]
    fn display_output_is_clean_non_empty_ascii() {
        for err in sample_errors() {
            let msg = err.to_string();
            assert!(!msg.is_empty());
            assert!(msg.is_ascii(), "non-ASCII error display: {msg}");
            assert!(!msg.contains('\n'), "multiline error display: {msg}");
        }

        let budget_err = FfpError::budget_exceeded(BudgetKind::Pixels, 1000, 5000);
        assert!(budget_err.is_budget_exceeded());
        assert_eq!(
            budget_err.to_string(),
            "budget exceeded (max_pixels): requested 5000, limit 1000"
        );

        let dim_err = FfpError::dimension_mismatch(1920, 1080, 640, 480);
        assert_eq!(
            dim_err.to_string(),
            "dimension mismatch: expected 1920x1080, got 640x480"
        );
    }

    #[test]
    fn io_unexpected_eof_translates_to_unexpected_eof_variant() {
        let eof_err = io::Error::new(io::ErrorKind::UnexpectedEof, "failed to fill buffer");
        let ffp_err = FfpError::from(eof_err);
        assert_eq!(ffp_err.status_code(), StatusCode::UnexpectedEof);
        assert!(matches!(ffp_err, FfpError::UnexpectedEof(_)));
        assert!(ffp_err.source().is_none());

        let perm_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let ffp_io = FfpError::from(perm_err);
        assert_eq!(ffp_io.status_code(), StatusCode::Io);
        assert!(matches!(ffp_io, FfpError::Io(_)));
        assert!(ffp_io.source().is_some());
    }

    #[test]
    fn budget_kind_labels_are_unique_and_ascii() {
        let kinds = [
            BudgetKind::Width,
            BudgetKind::Height,
            BudgetKind::Pixels,
            BudgetKind::MemoryBytes,
            BudgetKind::Frames,
            BudgetKind::MetadataBytes,
            BudgetKind::CompressionRatio,
            BudgetKind::InflateBytes,
            BudgetKind::RecursionDepth,
            BudgetKind::OutputBytes,
        ];
        let mut labels = HashSet::new();
        for kind in kinds {
            let label = kind.to_string();
            assert!(!label.is_empty());
            assert!(label.is_ascii());
            assert!(labels.insert(label));
        }
    }
}
