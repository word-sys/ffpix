#![forbid(unsafe_code)]

use std::io::Cursor;

use ffpix_core::{
    Adler32, AlphaMode, Budget, BudgetKind, BudgetLimits, ByteSink, ByteSource, Crc32, FfpError,
    FfpResult, ImageDescriptor, PeekReader, PixelFormat, Rect, SanitizePolicy, SequentialSource,
    Sha256, Size, SliceSource, StatusCode, SynthPattern, TileBuffer, TilePool, TileView, VecSink,
    compare_tiles_exact, compute_tile_psnr, convert_row_format, fill_tile_pattern, generate_tile,
    tile_grid, tile_sha256,
};

fn run_500_tile_sweep() -> FfpResult<([u8; 32], u32, u64, u64, u64)> {
    let bounds = Size::new(800, 640);
    let tile_w = 32u32;
    let tile_h = 32u32;
    let single_tile_bytes = u64::from(tile_w) * u64::from(tile_h) * 4;

    let limits = BudgetLimits {
        max_memory_bytes: single_tile_bytes * 2 - 1,
        ..BudgetLimits::preset_default()
    };
    let budget = Budget::new(limits);
    let pool = TilePool::new(budget.clone(), 2, single_tile_bytes);

    let mut sha = Sha256::new();
    let mut crc = Crc32::new();
    let mut count = 0u64;

    for tile_rect in tile_grid(bounds, tile_w, tile_h)? {
        let mut tile = pool.acquire(tile_rect, PixelFormat::RGBA8)?;
        fill_tile_pattern(&mut tile.as_view_mut(), bounds, SynthPattern::ZonePlate)?;
        for y in 0..tile_rect.h {
            let row = tile.row(y)?;
            sha.update(row);
            crc.update(row);
        }
        pool.release(tile);
        count += 1;
    }

    let stats = pool.stats();
    let peak = budget.peak_memory_bytes();
    pool.clear();
    assert_eq!(budget.current_memory_bytes(), 0);

    Ok((
        sha.finalize(),
        crc.finalize(),
        count,
        peak,
        stats.allocations,
    ))
}

#[test]
fn streaming_500_tile_sweep_stays_below_2x_single_tile_memory() -> FfpResult<()> {
    let single_tile_bytes = 32u64 * 32 * 4;
    let (sha1, crc1, count1, peak1, allocs1) = run_500_tile_sweep()?;
    let (sha2, crc2, count2, peak2, allocs2) = run_500_tile_sweep()?;

    assert_eq!(count1, 500);
    assert_eq!(count2, 500);
    assert_eq!(allocs1, 1);
    assert_eq!(allocs2, 1);
    assert!(peak1 < 2 * single_tile_bytes);
    assert_eq!(peak1, single_tile_bytes);
    assert_eq!(peak2, single_tile_bytes);
    assert_eq!(sha1, sha2);
    assert_eq!(crc1, crc2);
    Ok(())
}

#[test]
fn streaming_io_round_trip_with_peek_convert_and_checksums() -> FfpResult<()> {
    let budget = Budget::preset_default();
    let w = 64u32;
    let h = 48u32;
    let src = generate_tile(w, h, PixelFormat::RGBA8, SynthPattern::AlphaRings, &budget)?;

    let row16_len = PixelFormat::RGBA16.checked_row_bytes(w)?;
    let expected_total = 12 + (row16_len as u64) * u64::from(h) + 8;

    let mut sink = VecSink::with_limit(expected_total);
    sink.write_bytes(b"FFPX")?;
    sink.write_u32_be(w)?;
    sink.write_u32_be(h)?;

    let mut row16 = vec![0u8; row16_len];
    let mut crc_write = Crc32::new();
    let mut adler_write = Adler32::new();
    for y in 0..h {
        convert_row_format(
            src.row(y)?,
            PixelFormat::RGBA8,
            &mut row16,
            PixelFormat::RGBA16,
            w,
        )?;
        crc_write.update(&row16);
        adler_write.update(&row16);
        sink.write_bytes(&row16)?;
    }
    sink.write_u32_be(crc_write.finalize())?;
    sink.write_u32_be(adler_write.finalize())?;
    sink.flush()?;
    assert_eq!(sink.bytes_written(), expected_total);

    // Extra byte beyond configured limit fails cleanly.
    assert!(matches!(
        sink.write_u8(0),
        Err(FfpError::BudgetExceeded {
            kind: BudgetKind::OutputBytes,
            ..
        })
    ));

    let wire = sink.into_vec();
    let seq = SequentialSource::new(Cursor::new(wire));
    let mut reader = PeekReader::new(seq);
    assert!(!reader.is_seekable());
    assert_eq!(reader.peek_exact(4)?, b"FFPX");
    reader.consume(4)?;

    let rw = reader.read_u32_be()?;
    let rh = reader.read_u32_be()?;
    assert_eq!((rw, rh), (w, h));

    let mut dst = TileBuffer::new_zeroed(Rect::new(0, 0, rw, rh), PixelFormat::RGBA8, &budget)?;
    let mut crc_read = Crc32::new();
    let mut adler_read = Adler32::new();
    for y in 0..rh {
        reader.read_exact_checked(&mut row16)?;
        crc_read.update(&row16);
        adler_read.update(&row16);
        convert_row_format(
            &row16,
            PixelFormat::RGBA16,
            dst.row_mut(y)?,
            PixelFormat::RGBA8,
            rw,
        )?;
    }
    assert_eq!(reader.read_u32_be()?, crc_read.finalize());
    assert_eq!(reader.read_u32_be()?, adler_read.finalize());
    assert_eq!(
        reader.read_u8().err().map(|e| e.status_code()),
        Some(StatusCode::UnexpectedEof)
    );

    let diff = compare_tiles_exact(&src.as_view(), &dst.as_view())?;
    assert!(diff.is_exact());
    assert_eq!(
        compute_tile_psnr(&src.as_view(), &dst.as_view())?,
        f64::INFINITY
    );
    Ok(())
}

#[test]
fn sub_tile_halo_assembly_matches_monolithic_reference() -> FfpResult<()> {
    let budget = Budget::preset_default();
    let bounds = Size::new(100, 75);
    let reference = generate_tile(
        bounds.w,
        bounds.h,
        PixelFormat::RGBA8,
        SynthPattern::DiagonalGradient,
        &budget,
    )?;
    let mut assembled = TileBuffer::new_zeroed(bounds.to_rect(), PixelFormat::RGBA8, &budget)?;

    for tile_rect in tile_grid(bounds, 32, 32)? {
        let halo_rect = tile_rect.expand_halo(3, 3, 3, 3, bounds);
        let mut halo_buf = TileBuffer::new_zeroed(halo_rect, PixelFormat::RGBA8, &budget)?;
        fill_tile_pattern(
            &mut halo_buf.as_view_mut(),
            bounds,
            SynthPattern::DiagonalGradient,
        )?;
        assembled
            .as_view_mut()
            .copy_rect_from(&halo_buf.as_view(), tile_rect)?;
    }

    assert_eq!(
        tile_sha256(&reference.as_view())?,
        tile_sha256(&assembled.as_view())?
    );
    assert!(compare_tiles_exact(&reference.as_view(), &assembled.as_view())?.is_exact());
    Ok(())
}

#[test]
fn adversarial_malformed_dimensions_strides_and_budgets() {
    let budget = Budget::preset_strict();

    assert_eq!(
        ImageDescriptor::new(0, 10, PixelFormat::RGB8)
            .validate()
            .err()
            .map(|e| e.status_code()),
        Some(StatusCode::InvalidInput)
    );
    assert_eq!(
        ImageDescriptor::new(u32::MAX, u32::MAX, PixelFormat::RGBA32F)
            .validate()
            .err()
            .map(|e| e.status_code()),
        Some(StatusCode::Overflow)
    );

    let mut bad_desc = ImageDescriptor::new(16, 16, PixelFormat::RGB8);
    bad_desc.alpha_mode = AlphaMode::Premultiplied;
    assert_eq!(
        bad_desc.validate().err().map(|e| e.status_code()),
        Some(StatusCode::FormatMismatch)
    );
    assert_eq!(SanitizePolicy::default(), SanitizePolicy::KeepSafe);

    // Overflowing tile geometry and undersized strides fail cleanly with zero leaks.
    assert!(
        TileBuffer::new_zeroed(Rect::new(u32::MAX, 0, 4, 4), PixelFormat::RGBA8, &budget).is_err()
    );
    assert!(
        TileBuffer::new_zeroed(
            Rect::new(0, 0, u32::MAX, u32::MAX),
            PixelFormat::RGBA8,
            &budget
        )
        .is_err()
    );
    assert!(
        TileBuffer::with_stride(Rect::new(0, 0, 8, 8), PixelFormat::RGBA8, 16, &budget).is_err()
    );
    assert!(
        TileView::from_slice(Rect::new(0, 0, 4, 4), PixelFormat::RGBA8, 16, &[0u8; 32]).is_err()
    );

    // Strict budget rejects bomb ratio and excessive dimensions.
    assert!(matches!(
        budget.check_compression_ratio(128, 1024 * 1024, 64 * 1024),
        Err(FfpError::BudgetExceeded {
            kind: BudgetKind::CompressionRatio,
            ..
        })
    ));
    assert!(matches!(
        budget.check_dimensions(20_000, 20_000),
        Err(FfpError::BudgetExceeded {
            kind: BudgetKind::Width,
            ..
        })
    ));

    // PeekReader bounds & non-seekable error handling.
    let small = [1u8, 2, 3];
    if let Ok(mut pr) = PeekReader::with_max_lookahead(SliceSource::new(&small), 2) {
        assert!(pr.peek_up_to(3).is_err());
        assert!(pr.consume(1).is_err());
        assert_eq!(
            pr.peek_exact(3).err().map(|e| e.status_code()),
            Some(StatusCode::InvalidArgument)
        );
    }
    let mut seq = SequentialSource::new(Cursor::new(small));
    assert_eq!(
        seq.seek(0).err().map(|e| e.status_code()),
        Some(StatusCode::UnsupportedFeature)
    );

    assert_eq!(budget.current_memory_bytes(), 0);
}
