//! LZHUF as used by Winlink mail ("B2" format): LZSS with a 2048-byte window and adaptive Huffman
//! coding. A compressed message is a 2-byte CRC, the 4-byte unpacked length, then the bit stream.
//!
//! The algorithm is Haruyasu Yoshizaki's LZHUF; this decoder follows the wl2k-go implementation
//! (MIT licence, see NOTICE) and is checked against that project's test data.

const N: usize = 2048;
const F: usize = 60;
const THRESHOLD: usize = 2;
const N_CHAR: usize = 256 - THRESHOLD + F;
const T: usize = N_CHAR * 2 - 1;
const R: usize = T - 1;
const MAX_FREQ: u32 = 0x8000;

/// For the first byte of a position code: the upper 6 bits of the position, and the code length.
const fn position_tables() -> ([u8; 256], [u8; 256]) {
    // (how many first bytes, code length) for consecutive upper-position values
    let groups: [(usize, usize, u8); 6] = [(1, 32, 3), (3, 16, 4), (8, 8, 5), (12, 4, 6), (24, 2, 7), (16, 1, 8)];
    let (mut code, mut len) = ([0u8; 256], [0u8; 256]);
    let (mut i, mut value, mut g) = (0, 0u8, 0);
    while g < 6 {
        let mut v = 0;
        while v < groups[g].0 {
            let mut k = 0;
            while k < groups[g].1 {
                code[i] = value;
                len[i] = groups[g].2;
                i += 1;
                k += 1;
            }
            value += 1;
            v += 1;
        }
        g += 1;
    }
    (code, len)
}

const D_CODE: [u8; 256] = position_tables().0;
const D_LEN: [u8; 256] = position_tables().1;

struct Tree {
    freq: [u32; T + 1],
    prnt: [usize; T + N_CHAR],
    son: [usize; T],
}

impl Tree {
    fn new() -> Self {
        let mut t = Tree { freq: [0; T + 1], prnt: [0; T + N_CHAR], son: [0; T] };
        for i in 0..N_CHAR {
            t.freq[i] = 1;
            t.son[i] = i + T;
            t.prnt[i + T] = i;
        }
        let (mut i, mut j) = (0, N_CHAR);
        while j <= R {
            t.freq[j] = t.freq[i] + t.freq[i + 1];
            t.son[j] = i;
            t.prnt[i] = j;
            t.prnt[i + 1] = j;
            i += 2;
            j += 1;
        }
        t.freq[T] = 0xFFFF;
        t.prnt[R] = 0;
        t
    }

    fn rebuild(&mut self) {
        let mut j = 0;
        for i in 0..T {
            if self.son[i] >= T {
                self.freq[j] = (self.freq[i] + 1) / 2;
                self.son[j] = self.son[i];
                j += 1;
            }
        }
        let (mut i, mut j) = (0, N_CHAR);
        while j < T {
            let first = self.freq[i] + self.freq[i + 1];
            self.freq[j] = first;
            let mut k = j;
            while first < self.freq[k - 1] {
                k -= 1;
            }
            self.freq.copy_within(k..j, k + 1);
            self.freq[k] = first;
            self.son.copy_within(k..j, k + 1);
            self.son[k] = i;
            i += 2;
            j += 1;
        }
        for i in 0..T {
            let k = self.son[i];
            self.prnt[k] = i;
            if k < T {
                self.prnt[k + 1] = i;
            }
        }
    }

    fn update(&mut self, c: usize) {
        if self.freq[R] == MAX_FREQ {
            self.rebuild();
        }
        let mut c = self.prnt[c + T];
        loop {
            self.freq[c] += 1;
            if self.freq[c] <= self.freq[c + 1] || c + 2 >= T + 1 {
                c = self.prnt[c];
                if c == 0 {
                    break;
                }
                continue;
            }
            let (mut l, k) = (c + 1, self.freq[c]);
            while k > self.freq[l + 1] {
                l += 1;
            }
            self.freq[c] = self.freq[l];
            self.freq[l] = k;
            let i = self.son[c];
            self.prnt[i] = l;
            if i < T {
                self.prnt[i + 1] = l;
            }
            let j = self.son[l];
            self.son[l] = i;
            self.prnt[j] = c;
            if j < T {
                self.prnt[j + 1] = c;
            }
            self.son[c] = j;
            c = self.prnt[l];
            if c == 0 {
                break;
            }
        }
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Option<usize> {
        let b = *self.data.get(self.pos / 8)?;
        let v = (b >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Some(usize::from(v))
    }

    fn byte(&mut self) -> Option<usize> {
        let mut v = 0;
        for _ in 0..8 {
            v = (v << 1) | self.bit()?;
        }
        Some(v)
    }
}

/// CRC-16/XMODEM, as used for the B2 header.
pub fn crc16(data: &[u8]) -> u16 {
    let mut c: u16 = 0;
    for &b in data {
        c ^= u16::from(b) << 8;
        for _ in 0..8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x1021 } else { c << 1 };
        }
    }
    c
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    TooShort,
    Checksum,
    Truncated,
}

/// Unpack a B2-format message: CRC (2 bytes), length (4 bytes, little endian), LZHUF stream.
pub fn unpack_b2(data: &[u8]) -> Result<Vec<u8>, Error> {
    if data.len() < 6 {
        return Err(Error::TooShort);
    }
    if u16::from_le_bytes([data[0], data[1]]) != crc16(&data[2..]) {
        return Err(Error::Checksum);
    }
    let size = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;
    if size > 16 * 1024 * 1024 {
        return Err(Error::Checksum);
    }
    let mut tree = Tree::new();
    let mut bits = Bits { data: &data[6..], pos: 0 };
    let mut window = [0u8; N];
    window[..N - F].fill(b' ');
    let mut r = N - R;
    let mut out = Vec::with_capacity(size);
    while out.len() < size {
        // walk from the root to a leaf: the smaller child for a 0 bit, the larger for a 1
        let mut c = tree.son[R];
        while c < T {
            c = tree.son[c + bits.bit().ok_or(Error::Truncated)?];
        }
        c -= T;
        tree.update(c);
        if c < 256 {
            out.push(c as u8);
            window[r] = c as u8;
            r = (r + 1) & (N - 1);
            continue;
        }
        let first = bits.byte().ok_or(Error::Truncated)?;
        let mut i = first;
        for _ in 0..D_LEN[first] - 2 {
            i = (i << 1) + bits.bit().ok_or(Error::Truncated)?;
        }
        let position = usize::from(D_CODE[first]) << 6 | (i & 0x3F);
        let from = (r + N - position - 1) & (N - 1);
        for k in 0..c - 255 + THRESHOLD {
            if out.len() == size {
                break;
            }
            let b = window[(from + k) & (N - 1)];
            out.push(b);
            window[r] = b;
            r = (r + 1) & (N - 1);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_table_ends() {
        assert_eq!((D_CODE[0], D_LEN[0], D_CODE[31], D_CODE[32], D_LEN[32]), (0, 3, 0, 1, 4));
        assert_eq!((D_CODE[255], D_LEN[255], D_CODE[240], D_CODE[192], D_LEN[192]), (0x3F, 8, 0x30, 0x18, 7));
    }

    /// A file compressed by another implementation (test data of the wl2k-go project).
    #[test]
    fn reference_vector() {
        let plain = include_bytes!("../tests/data/gettysburg.txt");
        let packed = include_bytes!("../tests/data/gettysburg.txt.lzh");
        assert_eq!(unpack_b2(packed).as_deref(), Ok(&plain[..]));
    }

    #[test]
    fn damage_is_detected() {
        let mut packed = include_bytes!("../tests/data/gettysburg.txt.lzh").to_vec();
        packed[100] ^= 1;
        assert_eq!(unpack_b2(&packed), Err(Error::Checksum));
        assert_eq!(unpack_b2(&[1, 2, 3]), Err(Error::TooShort));
    }
}
