//! Core types, geometry, pixel formats, budget accounting, streaming I/O, and
//! deterministic error taxonomy for the `ffpix` streaming image engine.
//!
//! This crate enforces `#![forbid(unsafe_code)]` and has zero third-party
//! dependencies.

#![forbid(unsafe_code)]

pub mod budget;
pub mod descriptor;
pub mod error;
pub mod geom;
pub mod pixel;
pub mod tile;

pub use budget::{Budget, BudgetLimits, MemoryReservation};
pub use descriptor::{ImageDescriptor, MetadataBundle, MetadataSummary, SanitizePolicy};
pub use error::{BudgetKind, FfpError, FfpResult, StatusCode};
pub use geom::{Point, Rect, Size, TileGridIter, tile_grid};
pub use pixel::{AlphaMode, ChannelLayout, Cicp, ColorSpace, Orientation, PixelFormat, SampleType};
pub use tile::{TileBuffer, TilePool, TilePoolStats, TileView, TileViewMut};

/// Major version component of the `ffpix` engine.
pub const FFPIX_VERSION_MAJOR: u32 = 0;

/// Minor version component of the `ffpix` engine.
pub const FFPIX_VERSION_MINOR: u32 = 1;

/// Patch version component of the `ffpix` engine.
pub const FFPIX_VERSION_PATCH: u32 = 0;

/// Compile-time crate version string matching `Cargo.toml`.
pub const FFPIX_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the semantic version components as `(major, minor, patch)`.
#[must_use]
pub const fn version_tuple() -> (u32, u32, u32) {
    (
        FFPIX_VERSION_MAJOR,
        FFPIX_VERSION_MINOR,
        FFPIX_VERSION_PATCH,
    )
}

/// Returns the static version string matching `CARGO_PKG_VERSION`.
#[must_use]
pub const fn version_str() -> &'static str {
    FFPIX_VERSION
}

/// Formats the numeric version constants as `"major.minor.patch"`.
#[must_use]
pub fn version_string() -> String {
    format!("{FFPIX_VERSION_MAJOR}.{FFPIX_VERSION_MINOR}.{FFPIX_VERSION_PATCH}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_matches_cargo_pkg_version() {
        assert_eq!(version_string(), env!("CARGO_PKG_VERSION"));
        assert_eq!(version_str(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn version_constants_match_cargo_metadata() {
        let major = env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>();
        let minor = env!("CARGO_PKG_VERSION_MINOR").parse::<u32>();
        let patch = env!("CARGO_PKG_VERSION_PATCH").parse::<u32>();

        assert_eq!(major, Ok(FFPIX_VERSION_MAJOR));
        assert_eq!(minor, Ok(FFPIX_VERSION_MINOR));
        assert_eq!(patch, Ok(FFPIX_VERSION_PATCH));
        assert_eq!(
            version_tuple(),
            (
                FFPIX_VERSION_MAJOR,
                FFPIX_VERSION_MINOR,
                FFPIX_VERSION_PATCH,
            )
        );
    }
}
