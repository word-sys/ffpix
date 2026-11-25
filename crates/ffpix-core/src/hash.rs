//! Streaming SHA-256 (FIPS 180-4), slice-by-8 IEEE CRC-32, and RFC 1950 Adler-32.

const SHA256_IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bytes: u64,
}

impl Sha256 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: SHA256_IV,
            buf: [0u8; 64],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);

        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                compress_sha256(&mut self.state, &block);
                self.buf_len = 0;
            }
        }

        let mut chunks = data.chunks_exact(64);
        for chunk in &mut chunks {
            let mut block = [0u8; 64];
            block.copy_from_slice(chunk);
            compress_sha256(&mut self.state, &block);
        }

        let rem = chunks.remainder();
        if !rem.is_empty() {
            self.buf[..rem.len()].copy_from_slice(rem);
            self.buf_len = rem.len();
        }
    }

    #[must_use]
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let block = self.buf;
            compress_sha256(&mut self.state, &block);
            self.buf = [0u8; 64];
            self.buf_len = 0;
        } else {
            self.buf[self.buf_len..56].fill(0);
        }

        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        compress_sha256(&mut self.state, &block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    #[must_use]
    pub fn finalize_hex(self) -> String {
        Self::to_hex(&self.finalize())
    }

    #[must_use]
    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    #[must_use]
    pub fn to_hex(digest: &[u8; 32]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(64);
        for &b in digest {
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0x0F) as usize] as char);
        }
        s
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

fn compress_sha256(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

const CRC32_POLY: u32 = 0xEDB8_8320;

const fn build_crc32_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (c & 1) != 0 {
                c = (c >> 1) ^ CRC32_POLY;
            } else {
                c >>= 1;
            }
            bit += 1;
        }
        tables[0][i] = c;
        i += 1;
    }

    let mut t = 1usize;
    while t < 8 {
        let mut j = 0usize;
        while j < 256 {
            let prev = tables[t - 1][j];
            tables[t][j] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            j += 1;
        }
        t += 1;
    }
    tables
}

const CRC32_TABLES: [[u32; 256]; 8] = build_crc32_tables();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    #[must_use]
    pub const fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    pub fn reset(&mut self) {
        self.state = 0xFFFF_FFFF;
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        let mut chunks = data.chunks_exact(8);
        for c in &mut chunks {
            let lo = crc ^ u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let hi = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            crc = CRC32_TABLES[7][(lo & 0xFF) as usize]
                ^ CRC32_TABLES[6][((lo >> 8) & 0xFF) as usize]
                ^ CRC32_TABLES[5][((lo >> 16) & 0xFF) as usize]
                ^ CRC32_TABLES[4][(lo >> 24) as usize]
                ^ CRC32_TABLES[3][(hi & 0xFF) as usize]
                ^ CRC32_TABLES[2][((hi >> 8) & 0xFF) as usize]
                ^ CRC32_TABLES[1][((hi >> 16) & 0xFF) as usize]
                ^ CRC32_TABLES[0][(hi >> 24) as usize];
        }
        for &b in chunks.remainder() {
            crc = (crc >> 8) ^ CRC32_TABLES[0][((crc ^ u32::from(b)) & 0xFF) as usize];
        }
        self.state = crc;
    }

    #[must_use]
    pub const fn finalize(&self) -> u32 {
        !self.state
    }

    #[must_use]
    pub fn checksum(data: &[u8]) -> u32 {
        let mut c = Self::new();
        c.update(data);
        c.finalize()
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

const ADLER32_MOD: u32 = 65_521;
// Largest byte count guaranteed not to overflow u32 `s2` before modulo reduction.
const ADLER32_NMAX: usize = 5552;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Adler32 {
    s1: u32,
    s2: u32,
}

impl Adler32 {
    #[must_use]
    pub const fn new() -> Self {
        Self { s1: 1, s2: 0 }
    }

    pub fn reset(&mut self) {
        self.s1 = 1;
        self.s2 = 0;
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut s1 = self.s1;
        let mut s2 = self.s2;
        for chunk in data.chunks(ADLER32_NMAX) {
            for &b in chunk {
                s1 += u32::from(b);
                s2 += s1;
            }
            s1 %= ADLER32_MOD;
            s2 %= ADLER32_MOD;
        }
        self.s1 = s1;
        self.s2 = s2;
    }

    #[must_use]
    pub const fn finalize(&self) -> u32 {
        (self.s2 << 16) | self.s1
    }

    #[must_use]
    pub fn checksum(data: &[u8]) -> u32 {
        let mut a = Self::new();
        a.update(data);
        a.finalize()
    }
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            Sha256::to_hex(&Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            Sha256::to_hex(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 56-byte (448-bit) padding boundary forces an extra compression block.
        assert_eq!(
            Sha256::to_hex(&Sha256::digest(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );

        let mut million = Sha256::new();
        let block = [b'a'; 1000];
        for _ in 0..1000 {
            million.update(&block);
        }
        assert_eq!(
            million.finalize_hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn crc32_ieee_and_adler32_reference_vectors() {
        assert_eq!(Crc32::checksum(b""), 0x0000_0000);
        assert_eq!(Crc32::checksum(b"123456789"), 0xCBF4_3926);

        assert_eq!(Adler32::checksum(b""), 0x0000_0001);
        assert_eq!(Adler32::checksum(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn streaming_chunked_updates_match_single_shot_exactly() {
        let data: Vec<u8> = (0..20_000u32)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 3)) as u8)
            .collect();

        let ref_sha = Sha256::digest(&data);
        let ref_crc = Crc32::checksum(&data);
        let ref_adler = Adler32::checksum(&data);

        for step in [1usize, 3, 7, 8, 13, 63, 64, 65, 5551, 5552, 5553] {
            let mut sha = Sha256::new();
            let mut crc = Crc32::new();
            let mut adler = Adler32::new();
            for chunk in data.chunks(step) {
                sha.update(chunk);
                crc.update(chunk);
                adler.update(chunk);
            }
            assert_eq!(sha.finalize(), ref_sha, "sha256 mismatch at step {step}");
            assert_eq!(crc.finalize(), ref_crc, "crc32 mismatch at step {step}");
            assert_eq!(
                adler.finalize(),
                ref_adler,
                "adler32 mismatch at step {step}"
            );
        }
    }
}
