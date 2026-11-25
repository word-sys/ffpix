//! Streaming byte sources, bounded lookahead reader, and byte-limited sinks.

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{BudgetKind, FfpError, FfpResult};

pub const DEFAULT_MAX_LOOKAHEAD: usize = 64 * 1024;

pub trait ByteSource {
    fn read_bytes(&mut self, buf: &mut [u8]) -> FfpResult<usize>;
    fn is_seekable(&self) -> bool;
    fn seek(&mut self, pos: u64) -> FfpResult<u64>;
    fn position(&self) -> u64;
    fn len_hint(&self) -> Option<u64>;

    fn read_exact_bytes(&mut self, mut buf: &mut [u8]) -> FfpResult<()> {
        while !buf.is_empty() {
            let n = self.read_bytes(buf)?;
            if n == 0 {
                return Err(FfpError::unexpected_eof(
                    "source ended before expected bytes",
                ));
            }
            buf = &mut buf[n..];
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SliceSource<'a> {
    data: &'a [u8],
    pos: u64,
}

impl<'a> SliceSource<'a> {
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl ByteSource for SliceSource<'_> {
    fn read_bytes(&mut self, buf: &mut [u8]) -> FfpResult<usize> {
        let Ok(pos) = usize::try_from(self.pos) else {
            return Ok(0);
        };
        let Some(rem) = self.data.get(pos..) else {
            return Ok(0);
        };
        let n = rem.len().min(buf.len());
        buf[..n].copy_from_slice(&rem[..n]);
        self.pos = self.pos.saturating_add(n as u64);
        Ok(n)
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: u64) -> FfpResult<u64> {
        self.pos = pos;
        Ok(self.pos)
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn len_hint(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
}

#[derive(Debug)]
pub struct FileSource {
    reader: BufReader<File>,
    pos: u64,
    len: Option<u64>,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> FfpResult<Self> {
        let file = File::open(path)?;
        Ok(Self::from_file(file))
    }

    #[must_use]
    pub fn from_file(file: File) -> Self {
        let len = file.metadata().ok().map(|m| m.len());
        Self {
            reader: BufReader::new(file),
            pos: 0,
            len,
        }
    }
}

impl ByteSource for FileSource {
    fn read_bytes(&mut self, buf: &mut [u8]) -> FfpResult<usize> {
        loop {
            match self.reader.read(buf) {
                Ok(n) => {
                    self.pos = self.pos.saturating_add(n as u64);
                    return Ok(n);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(FfpError::from(e)),
            }
        }
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: u64) -> FfpResult<u64> {
        let new_pos = self.reader.seek(SeekFrom::Start(pos))?;
        self.pos = new_pos;
        Ok(new_pos)
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn len_hint(&self) -> Option<u64> {
        self.len
    }
}

pub struct SequentialSource<R: Read> {
    reader: R,
    pos: u64,
    len_hint: Option<u64>,
}

impl<R: Read> fmt::Debug for SequentialSource<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SequentialSource")
            .field("pos", &self.pos)
            .field("len_hint", &self.len_hint)
            .finish_non_exhaustive()
    }
}

impl<R: Read> SequentialSource<R> {
    #[must_use]
    pub const fn new(reader: R) -> Self {
        Self {
            reader,
            pos: 0,
            len_hint: None,
        }
    }

    #[must_use]
    pub const fn with_len_hint(reader: R, len_hint: u64) -> Self {
        Self {
            reader,
            pos: 0,
            len_hint: Some(len_hint),
        }
    }

    #[must_use]
    pub fn into_inner(self) -> R {
        self.reader
    }
}

impl<R: Read> ByteSource for SequentialSource<R> {
    fn read_bytes(&mut self, buf: &mut [u8]) -> FfpResult<usize> {
        loop {
            match self.reader.read(buf) {
                Ok(n) => {
                    self.pos = self.pos.saturating_add(n as u64);
                    return Ok(n);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(FfpError::from(e)),
            }
        }
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn seek(&mut self, _pos: u64) -> FfpResult<u64> {
        Err(FfpError::unsupported_feature(
            "sequential source does not support seek",
        ))
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn len_hint(&self) -> Option<u64> {
        self.len_hint
    }
}

/// Bounded lookahead reader over any [`ByteSource`] (sequential or seekable).
#[derive(Debug)]
pub struct PeekReader<S: ByteSource> {
    inner: S,
    buf: Vec<u8>,
    head: usize,
    max_lookahead: usize,
}

impl<S: ByteSource> PeekReader<S> {
    #[must_use]
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buf: Vec::new(),
            head: 0,
            max_lookahead: DEFAULT_MAX_LOOKAHEAD,
        }
    }

    pub fn with_max_lookahead(inner: S, max_lookahead: usize) -> FfpResult<Self> {
        if max_lookahead == 0 {
            return Err(FfpError::invalid_argument(
                "max_lookahead must be at least 1 byte",
            ));
        }
        Ok(Self {
            inner,
            buf: Vec::new(),
            head: 0,
            max_lookahead,
        })
    }

    #[must_use]
    pub const fn max_lookahead(&self) -> usize {
        self.max_lookahead
    }

    #[must_use]
    pub fn buffered_len(&self) -> usize {
        self.buf.len().saturating_sub(self.head)
    }

    #[must_use]
    pub fn position(&self) -> u64 {
        self.inner
            .position()
            .saturating_sub(self.buffered_len() as u64)
    }

    #[must_use]
    pub fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    #[must_use]
    pub fn len_hint(&self) -> Option<u64> {
        self.inner.len_hint()
    }

    pub fn seek(&mut self, pos: u64) -> FfpResult<u64> {
        self.buf.clear();
        self.head = 0;
        self.inner.seek(pos)
    }

    fn compact(&mut self) {
        if self.head == 0 {
            return;
        }
        let rem = self.buffered_len();
        if rem == 0 {
            self.buf.clear();
        } else {
            self.buf.copy_within(self.head.., 0);
            self.buf.truncate(rem);
        }
        self.head = 0;
    }

    pub fn peek_up_to(&mut self, n: usize) -> FfpResult<&[u8]> {
        if n > self.max_lookahead {
            return Err(FfpError::invalid_argument(format!(
                "peek length {n} exceeds max_lookahead {}",
                self.max_lookahead
            )));
        }
        if self.buffered_len() < n {
            self.compact();
            let mut tmp = [0u8; 4096];
            while self.buf.len() < n {
                let want = (n - self.buf.len()).min(tmp.len());
                let read = self.inner.read_bytes(&mut tmp[..want])?;
                if read == 0 {
                    break;
                }
                self.buf.extend_from_slice(&tmp[..read]);
            }
        }
        let avail = self.buffered_len().min(n);
        let end = self.head.saturating_add(avail);
        self.buf
            .get(self.head..end)
            .ok_or_else(|| FfpError::internal("peek buffer range out of bounds"))
    }

    pub fn peek_exact(&mut self, n: usize) -> FfpResult<&[u8]> {
        let slice = self.peek_up_to(n)?;
        if slice.len() < n {
            return Err(FfpError::unexpected_eof(format!(
                "expected {n} peek bytes, only {} available",
                slice.len()
            )));
        }
        Ok(slice)
    }

    pub fn consume(&mut self, n: usize) -> FfpResult<()> {
        if n > self.buffered_len() {
            return Err(FfpError::invalid_argument(format!(
                "cannot consume {n} bytes when only {} are buffered",
                self.buffered_len()
            )));
        }
        self.head += n;
        if self.head == self.buf.len() {
            self.buf.clear();
            self.head = 0;
        }
        Ok(())
    }

    pub fn read_bytes(&mut self, dst: &mut [u8]) -> FfpResult<usize> {
        if dst.is_empty() {
            return Ok(0);
        }
        let avail = self.buffered_len();
        if avail > 0 {
            let n = avail.min(dst.len());
            let end = self.head + n;
            let Some(src) = self.buf.get(self.head..end) else {
                return Err(FfpError::internal("peek buffer slice out of bounds"));
            };
            dst[..n].copy_from_slice(src);
            self.head = end;
            if self.head == self.buf.len() {
                self.buf.clear();
                self.head = 0;
            }
            return Ok(n);
        }
        self.inner.read_bytes(dst)
    }

    pub fn read_exact_checked(&mut self, mut dst: &mut [u8]) -> FfpResult<()> {
        while !dst.is_empty() {
            let n = self.read_bytes(dst)?;
            if n == 0 {
                return Err(FfpError::unexpected_eof(
                    "unexpected EOF while reading exact bytes",
                ));
            }
            dst = &mut dst[n..];
        }
        Ok(())
    }

    pub fn read_u8(&mut self) -> FfpResult<u8> {
        let mut b = [0u8; 1];
        self.read_exact_checked(&mut b)?;
        Ok(b[0])
    }

    pub fn read_u16_le(&mut self) -> FfpResult<u16> {
        let mut b = [0u8; 2];
        self.read_exact_checked(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    pub fn read_u16_be(&mut self) -> FfpResult<u16> {
        let mut b = [0u8; 2];
        self.read_exact_checked(&mut b)?;
        Ok(u16::from_be_bytes(b))
    }

    pub fn read_u32_le(&mut self) -> FfpResult<u32> {
        let mut b = [0u8; 4];
        self.read_exact_checked(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn read_u32_be(&mut self) -> FfpResult<u32> {
        let mut b = [0u8; 4];
        self.read_exact_checked(&mut b)?;
        Ok(u32::from_be_bytes(b))
    }

    pub fn read_u64_le(&mut self) -> FfpResult<u64> {
        let mut b = [0u8; 8];
        self.read_exact_checked(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn read_u64_be(&mut self) -> FfpResult<u64> {
        let mut b = [0u8; 8];
        self.read_exact_checked(&mut b)?;
        Ok(u64::from_be_bytes(b))
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: ByteSource> ByteSource for PeekReader<S> {
    fn read_bytes(&mut self, buf: &mut [u8]) -> FfpResult<usize> {
        PeekReader::read_bytes(self, buf)
    }

    fn is_seekable(&self) -> bool {
        PeekReader::is_seekable(self)
    }

    fn seek(&mut self, pos: u64) -> FfpResult<u64> {
        PeekReader::seek(self, pos)
    }

    fn position(&self) -> u64 {
        PeekReader::position(self)
    }

    fn len_hint(&self) -> Option<u64> {
        PeekReader::len_hint(self)
    }
}

pub trait ByteSink {
    fn write_bytes(&mut self, data: &[u8]) -> FfpResult<()>;
    fn flush(&mut self) -> FfpResult<()>;
    fn bytes_written(&self) -> u64;

    fn write_u8(&mut self, v: u8) -> FfpResult<()> {
        self.write_bytes(&[v])
    }

    fn write_u16_le(&mut self, v: u16) -> FfpResult<()> {
        self.write_bytes(&v.to_le_bytes())
    }

    fn write_u16_be(&mut self, v: u16) -> FfpResult<()> {
        self.write_bytes(&v.to_be_bytes())
    }

    fn write_u32_le(&mut self, v: u32) -> FfpResult<()> {
        self.write_bytes(&v.to_le_bytes())
    }

    fn write_u32_be(&mut self, v: u32) -> FfpResult<()> {
        self.write_bytes(&v.to_be_bytes())
    }

    fn write_u64_le(&mut self, v: u64) -> FfpResult<()> {
        self.write_bytes(&v.to_le_bytes())
    }

    fn write_u64_be(&mut self, v: u64) -> FfpResult<()> {
        self.write_bytes(&v.to_be_bytes())
    }
}

fn check_sink_limit(written: u64, delta: usize, max_bytes: Option<u64>) -> FfpResult<u64> {
    let next = written
        .checked_add(delta as u64)
        .ok_or_else(|| FfpError::overflow("output byte counter overflow"))?;
    if let Some(limit) = max_bytes
        && next > limit
    {
        return Err(FfpError::budget_exceeded(
            BudgetKind::OutputBytes,
            limit,
            next,
        ));
    }
    Ok(next)
}

#[derive(Debug, Clone, Default)]
pub struct VecSink {
    buf: Vec<u8>,
    max_bytes: Option<u64>,
}

impl VecSink {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buf: Vec::new(),
            max_bytes: None,
        }
    }

    #[must_use]
    pub const fn with_limit(max_bytes: u64) -> Self {
        Self {
            buf: Vec::new(),
            max_bytes: Some(max_bytes),
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

impl ByteSink for VecSink {
    fn write_bytes(&mut self, data: &[u8]) -> FfpResult<()> {
        let _ = check_sink_limit(self.buf.len() as u64, data.len(), self.max_bytes)?;
        self.buf.extend_from_slice(data);
        Ok(())
    }

    fn flush(&mut self) -> FfpResult<()> {
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.buf.len() as u64
    }
}

#[derive(Debug)]
pub struct FileSink {
    writer: BufWriter<File>,
    written: u64,
    max_bytes: Option<u64>,
}

impl FileSink {
    pub fn create(path: impl AsRef<Path>) -> FfpResult<Self> {
        let file = File::create(path)?;
        Ok(Self::from_file(file, None))
    }

    pub fn create_with_limit(path: impl AsRef<Path>, max_bytes: u64) -> FfpResult<Self> {
        let file = File::create(path)?;
        Ok(Self::from_file(file, Some(max_bytes)))
    }

    #[must_use]
    pub fn from_file(file: File, max_bytes: Option<u64>) -> Self {
        Self {
            writer: BufWriter::new(file),
            written: 0,
            max_bytes,
        }
    }
}

impl ByteSink for FileSink {
    fn write_bytes(&mut self, data: &[u8]) -> FfpResult<()> {
        let next = check_sink_limit(self.written, data.len(), self.max_bytes)?;
        self.writer.write_all(data)?;
        self.written = next;
        Ok(())
    }

    fn flush(&mut self) -> FfpResult<()> {
        self.writer.flush()?;
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.written
    }
}

pub struct WriterSink<W: Write> {
    writer: W,
    written: u64,
    max_bytes: Option<u64>,
}

impl<W: Write> fmt::Debug for WriterSink<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriterSink")
            .field("written", &self.written)
            .field("max_bytes", &self.max_bytes)
            .finish_non_exhaustive()
    }
}

impl<W: Write> WriterSink<W> {
    #[must_use]
    pub const fn new(writer: W) -> Self {
        Self {
            writer,
            written: 0,
            max_bytes: None,
        }
    }

    #[must_use]
    pub const fn with_limit(writer: W, max_bytes: u64) -> Self {
        Self {
            writer,
            written: 0,
            max_bytes: Some(max_bytes),
        }
    }

    #[must_use]
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: Write> ByteSink for WriterSink<W> {
    fn write_bytes(&mut self, data: &[u8]) -> FfpResult<()> {
        let next = check_sink_limit(self.written, data.len(), self.max_bytes)?;
        self.writer.write_all(data)?;
        self.written = next;
        Ok(())
    }

    fn flush(&mut self) -> FfpResult<()> {
        self.writer.flush()?;
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatusCode;
    use std::env;
    use std::fs;
    use std::process;

    struct ChunkedReader<'a> {
        data: &'a [u8],
        pos: usize,
        chunk: usize,
    }

    impl Read for ChunkedReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let rem = &self.data[self.pos..];
            if rem.is_empty() {
                return Ok(0);
            }
            let n = rem.len().min(buf.len()).min(self.chunk);
            buf[..n].copy_from_slice(&rem[..n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn sequential_peek_then_read_preserves_exact_stream() {
        let payload: Vec<u8> = (0..100u8).collect();
        let src = SequentialSource::new(ChunkedReader {
            data: &payload,
            pos: 0,
            chunk: 3,
        });
        assert!(!src.is_seekable());

        let mut reader = PeekReader::new(src);
        assert!(!reader.is_seekable());
        assert!(reader.seek(0).is_err());

        // Peek magic header without advancing logical position.
        assert_eq!(reader.peek_exact(8).ok(), Some(&payload[..8]));
        assert_eq!(reader.position(), 0);

        // Larger peek extends lookahead cleanly.
        assert_eq!(reader.peek_exact(16).ok(), Some(&payload[..16]));
        assert_eq!(reader.position(), 0);

        // Consume 4 bytes then peek again (forces compaction on subsequent growth).
        assert!(reader.consume(4).is_ok());
        assert_eq!(reader.position(), 4);
        assert_eq!(reader.peek_exact(20).ok(), Some(&payload[4..24]));

        // Read rest of the stream and verify exact match with payload[4..].
        let mut out = Vec::new();
        let mut chunk = [0u8; 7];
        loop {
            match reader.read_bytes(&mut chunk) {
                Ok(0) => break,
                Ok(n) => out.extend_from_slice(&chunk[..n]),
                Err(_) => break,
            }
        }
        assert_eq!(out, payload[4..]);
        assert_eq!(reader.position(), 100);
    }

    #[test]
    fn peek_lookahead_bounds_and_typed_readers_handle_eof() {
        let mut bytes = Vec::new();
        bytes.push(0x42);
        bytes.extend_from_slice(&0x1234u16.to_le_bytes());
        bytes.extend_from_slice(&0x1234u16.to_be_bytes());
        bytes.extend_from_slice(&0x89ABCDEFu32.to_le_bytes());
        bytes.extend_from_slice(&0x89ABCDEFu32.to_be_bytes());
        bytes.extend_from_slice(&0x0102030405060708u64.to_le_bytes());
        bytes.extend_from_slice(&0x0102030405060708u64.to_be_bytes());

        let Ok(mut r) = PeekReader::with_max_lookahead(SliceSource::new(&bytes), 8) else {
            return;
        };
        assert!(r.peek_up_to(9).is_err());
        assert_eq!(r.peek_up_to(8).ok(), Some(&bytes[..8]));

        assert_eq!(r.read_u8().ok(), Some(0x42));
        assert_eq!(r.read_u16_le().ok(), Some(0x1234));
        assert_eq!(r.read_u16_be().ok(), Some(0x1234));
        assert_eq!(r.read_u32_le().ok(), Some(0x89ABCDEF));
        assert_eq!(r.read_u32_be().ok(), Some(0x89ABCDEF));
        assert_eq!(r.read_u64_le().ok(), Some(0x0102030405060708));
        assert_eq!(r.read_u64_be().ok(), Some(0x0102030405060708));

        // Reading past EOF returns UnexpectedEof cleanly.
        assert_eq!(
            r.read_u8().err().map(|e| e.status_code()),
            Some(StatusCode::UnexpectedEof)
        );
        assert_eq!(
            r.peek_exact(1).err().map(|e| e.status_code()),
            Some(StatusCode::UnexpectedEof)
        );
        assert_eq!(r.peek_up_to(4).ok(), Some([].as_slice()));
    }

    #[test]
    fn seekable_file_source_and_file_sink_round_trip() {
        let mut path = env::temp_dir();
        path.push(format!("ffpix_core_io_test_{}.bin", process::id()));

        let write_res = (|| -> FfpResult<()> {
            let mut sink = FileSink::create_with_limit(&path, 128)?;
            sink.write_u32_be(0x8950_4E47)?;
            sink.write_u16_le(0xBEEF)?;
            sink.write_bytes(b"ffpix-stream")?;
            sink.flush()?;
            assert_eq!(sink.bytes_written(), 18);
            Ok(())
        })();
        assert!(write_res.is_ok());

        let read_res = (|| -> FfpResult<()> {
            let mut reader = PeekReader::new(FileSource::open(&path)?);
            assert!(reader.is_seekable());
            assert_eq!(reader.len_hint(), Some(18));
            assert_eq!(reader.peek_exact(4)?, &[0x89, 0x50, 0x4E, 0x47]);
            assert_eq!(reader.read_u32_be()?, 0x8950_4E47);
            assert_eq!(reader.read_u16_le()?, 0xBEEF);
            let mut label = [0u8; 12];
            reader.read_exact_checked(&mut label)?;
            assert_eq!(&label, b"ffpix-stream");

            // Seek back to offset 6 and re-read label.
            assert_eq!(reader.seek(6)?, 6);
            assert_eq!(reader.position(), 6);
            let mut again = [0u8; 12];
            reader.read_exact_checked(&mut again)?;
            assert_eq!(&again, b"ffpix-stream");
            Ok(())
        })();
        let _ = fs::remove_file(&path);
        assert!(read_res.is_ok());
    }

    #[test]
    fn sink_output_limit_enforced_before_write() {
        let mut vec_sink = VecSink::with_limit(10);
        assert!(vec_sink.write_bytes(b"123456").is_ok());
        assert_eq!(vec_sink.bytes_written(), 6);
        assert!(matches!(
            vec_sink.write_bytes(b"78901"),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::OutputBytes,
                limit: 10,
                requested: 11,
            })
        ));
        assert_eq!(vec_sink.as_slice(), b"123456");

        let mut writer_sink = WriterSink::with_limit(Vec::<u8>::new(), 4);
        assert!(writer_sink.write_u32_be(0x0102_0304).is_ok());
        assert!(writer_sink.write_u8(0x05).is_err());
        assert!(writer_sink.flush().is_ok());
        assert_eq!(writer_sink.into_inner(), vec![1, 2, 3, 4]);
    }
}
