//! Tables of the tone-keyed levels: interleaver, turbo permutation, hop sequence and pilot symbols.
//!
//! All tables of one bandwidth mode come from a single generator stream, level after level.

use crate::vb::VbRnd;

pub const PILOTS: usize = 24;
const TABLE_SEED: f64 = 43.0;

/// Parameters of a tone-keyed level. The tone grid is `3000 / sps` Hz wide per bin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FskLevel {
    pub symbols: u32,
    pub bits: u32,
    /// puncture period of the turbo code: 1 = rate 1/3, 2 = rate 1/2
    pub puncture: u32,
    pub sps: u32,
    pub first: u32,
    pub last: u32,
    pub step: u32,
    /// information bits per frame
    pub k: u32,
}

impl FskLevel {
    /// coded bits per frame
    pub fn n(&self) -> u32 {
        (self.symbols - 1 - PILOTS as u32) * self.bits
    }

    pub fn bins(&self) -> u32 {
        self.last - self.first + 1
    }

    /// The bin of symbol `i` carrying tone value `t`.
    pub fn bin(&self, hop: u16, t: u32) -> u32 {
        self.first + (u32::from(hop) + self.step * t - self.first) % self.bins()
    }
}

const fn lv(symbols: u32, bits: u32, puncture: u32, sps: u32, first: u32, last: u32, step: u32, k: u32) -> FskLevel {
    FskLevel { symbols, bits, puncture, sps, first, last, step, k }
}

pub const BW2300: [FskLevel; 4] =
    [lv(125, 3, 1, 128, 33, 96, 2, 96), lv(229, 3, 1, 64, 17, 48, 2, 200), lv(229, 4, 2, 64, 17, 48, 2, 402), lv(396, 4, 2, 32, 9, 24, 1, 736)];
pub const BW500: [FskLevel; 3] = [lv(125, 3, 1, 128, 54, 74, 2, 96), lv(229, 3, 1, 64, 27, 37, 1, 200), lv(227, 3, 2, 64, 27, 37, 1, 297)];
pub const BW2750: [FskLevel; 4] =
    [lv(125, 3, 1, 128, 24, 103, 2, 96), lv(229, 3, 1, 64, 12, 51, 2, 200), lv(229, 4, 2, 64, 12, 51, 2, 402), lv(396, 4, 2, 32, 6, 25, 1, 736)];

#[derive(Clone, Debug)]
pub struct FskTables {
    pub level: FskLevel,
    /// transmitted bit `j` is coded bit `chan[j]`
    pub chan: Vec<u32>,
    /// input order of the second encoder
    pub turbo: Vec<u32>,
    /// per symbol: base bin of the tone hop
    pub hop: Vec<u16>,
    /// per symbol: the fixed tone value of a pilot symbol, or `None` for a data symbol
    pub pilot: Vec<Option<u32>>,
}

/// A random permutation of `0..n`, drawn by rejection.
pub fn permutation(r: &mut VbRnd, n: u32) -> Vec<u32> {
    let mut used = vec![false; n as usize];
    let mut out = Vec::with_capacity(n as usize);
    while out.len() < n as usize {
        let v = r.draw(n);
        if !std::mem::replace(&mut used[v as usize], true) {
            out.push(v);
        }
    }
    out
}

/// Tables for the tone-keyed levels of one bandwidth mode, in level order.
pub fn fsk_tables(levels: &[FskLevel]) -> Vec<FskTables> {
    let mut r = VbRnd::stream(TABLE_SEED);
    levels.iter().map(|&l| one_level(&mut r, l)).collect()
}

pub(crate) fn one_level(r: &mut VbRnd, l: FskLevel) -> FskTables {
    let chan = permutation(r, l.n());
    let turbo = permutation(r, l.k);
    let nsym = (l.symbols - 1) as usize;
    let m = if l.step > 1 { l.bins() / l.step + 1 } else { l.bins() };
    let hop = (0..nsym).map(|i| (l.first + u32::from(l.step > 1 && i % 2 == 0) + l.step * r.draw(m)) as u16).collect();
    let inc = (l.symbols - 2) as f32 / PILOTS as f32;
    let mut acc = 1.0f32;
    let rows: Vec<usize> = (0..PILOTS)
        .map(|_| {
            acc += inc;
            acc as usize
        })
        .collect();
    let scale = (1u32 << l.bits) / 8;
    let mut pilot = vec![None; nsym];
    for row in rows {
        pilot[row - 2] = Some(scale * r.draw(8));
    }
    FskTables { level: l, chan, turbo, hop, pilot }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv(vals: impl Iterator<Item = u32>) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        for v in vals {
            for b in v.to_le_bytes() {
                h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
            }
        }
        h
    }

    /// Hashes of the tables produced by the Python reference for the 2300 Hz mode.
    #[test]
    fn matches_reference_2300() {
        let want: [(u32, u32, u32, u32, usize, usize); 4] = [
            (0xd49e4595, 0x3d2cf095, 0xe452cfbf, 0x8cafe9c1, 300, 96),
            (0xdcbc07b9, 0x32551755, 0xa75f7d51, 0x28158e72, 612, 200),
            (0xa254d501, 0xdf510ac4, 0x65582005, 0x8591f1b9, 816, 402),
            (0x18637c95, 0xdba74b5d, 0xf73a44ca, 0x07e0eb7e, 1484, 736),
        ];
        for (t, w) in fsk_tables(&BW2300).iter().zip(want) {
            assert_eq!((t.chan.len(), t.turbo.len()), (w.4, w.5));
            assert_eq!(fnv(t.chan.iter().copied()), w.0, "interleaver");
            assert_eq!(fnv(t.turbo.iter().copied()), w.1, "turbo permutation");
            assert_eq!(fnv(t.hop.iter().map(|&h| u32::from(h))), w.2, "hop sequence");
            assert_eq!(fnv(t.pilot.iter().map(|p| p.unwrap_or(255))), w.3, "pilots");
            assert_eq!(t.pilot.iter().filter(|p| p.is_some()).count(), PILOTS);
        }
    }

    #[test]
    fn other_modes_are_consistent() {
        for levels in [&BW500[..], &BW2750[..]] {
            for t in fsk_tables(levels) {
                let l = t.level;
                let p = l.puncture;
                let coded = if p == 1 { 3 * l.k + 12 } else { l.k + l.k.div_ceil(p) + (l.k - p / 2 - 1) / p + 1 + 12 };
                assert_eq!(coded, l.n());
                // the hop base may run past the last bin; `bin()` wraps it back into range
                assert!(t.hop.iter().all(|&h| u32::from(h) >= l.first && u32::from(h) <= l.last + 2));
                assert!(t.hop.iter().all(|&h| (0..1u32 << l.bits).all(|v| (l.first..=l.last).contains(&l.bin(h, v)))));
            }
        }
    }
}
