//! Every level of the three bandwidth modes, and the tables each one is sent with.
//!
//! A mode has 17 records. The first 13 (500 Hz), 16 (2300 Hz) or 17 (2750 Hz) are the speed
//! levels; the rest are special frame types. All tables of a mode come from one generator
//! stream, record after record.

use crate::tables::{one_level, permutation, FskLevel, FskTables};
use crate::vb::VbRnd;

pub const RECORDS: usize = 17;
/// In the QAM levels every eighth symbol of a carrier is a pilot.
pub const PILOT_PERIOD: u32 = 8;
const TABLE_SEED: f64 = 43.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bandwidth {
    Hz500,
    Hz2300,
    Hz2750,
}

impl Bandwidth {
    pub const ALL: [Bandwidth; 3] = [Bandwidth::Hz2300, Bandwidth::Hz2750, Bandwidth::Hz500];

    pub const fn hz(self) -> u32 {
        match self {
            Bandwidth::Hz500 => 500,
            Bandwidth::Hz2300 => 2300,
            Bandwidth::Hz2750 => 2750,
        }
    }

    pub fn from_hz(hz: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.hz() == hz)
    }

    /// How many of the 17 records are speed levels.
    pub const fn speed_levels(self) -> usize {
        match self {
            Bandwidth::Hz500 => 13,
            Bandwidth::Hz2300 => 16,
            Bandwidth::Hz2750 => 17,
        }
    }

    pub fn levels(self) -> &'static [Level; RECORDS] {
        match self {
            Bandwidth::Hz500 => &BW500,
            Bandwidth::Hz2300 => &BW2300,
            Bandwidth::Hz2750 => &BW2750,
        }
    }
}

/// Pulse shape of the phase-keyed carriers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pulse {
    /// raised cosine, one symbol long
    Rc,
    /// root raised cosine reaching this many symbols to each side
    Rrc(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    /// one tone per symbol; a symbol is `16 * sps` samples at 48 kHz
    Fsk { sps: u32, first: u32, last: u32, step: u32 },
    /// a few carriers at the listed centre frequencies, 93.75 symbols/s, differential
    Psk { markers: u32, carriers: &'static [u32], pulse: Pulse },
    /// `nc` carriers from bin `first` of a 1024-point transform at 48 kHz, plus 128 samples of guard
    Ofdm { markers: u32, nc: u32, first: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    pub wave: Wave,
    /// symbols per frame, counting the reference symbol
    pub spf: u32,
    /// bits per tone or per cell
    pub bps: u32,
    /// puncture period of the turbo code (1 = rate 1/3, 2 = 1/2, 4 = 2/3, 8 = 4/5, 10 = 5/6)
    pub puncture: u32,
    /// information bits per frame
    pub k: u32,
}

impl Level {
    pub const fn is_qam(&self) -> bool {
        matches!(self.wave, Wave::Ofdm { .. }) && self.bps >= 4
    }

    pub const fn carriers(&self) -> u32 {
        match self.wave {
            Wave::Fsk { .. } => 1,
            Wave::Psk { carriers, .. } => carriers.len() as u32,
            Wave::Ofdm { nc, .. } => nc,
        }
    }

    pub const fn markers(&self) -> u32 {
        match self.wave {
            Wave::Fsk { .. } => crate::tables::PILOTS as u32,
            Wave::Psk { markers, .. } | Wave::Ofdm { markers, .. } => markers,
        }
    }

    /// coded bits per frame
    pub const fn n(&self) -> u32 {
        let cells = match self.wave {
            Wave::Fsk { .. } => self.spf - 1 - self.markers(),
            _ if self.is_qam() => self.carriers() * (self.spf - 1 - self.markers() - ((self.spf - 3) / PILOT_PERIOD + 1)),
            _ => self.carriers() * (self.spf - 2 - self.markers()),
        };
        cells * self.bps
    }

    pub const fn frame_bytes(&self) -> usize {
        (self.k / 8) as usize
    }

    /// samples per symbol at 48 kHz
    pub const fn symbol_samples(&self) -> usize {
        match self.wave {
            Wave::Fsk { sps, .. } => 16 * sps as usize,
            Wave::Psk { .. } => 512,
            Wave::Ofdm { .. } => 1152,
        }
    }

    /// random symbols sent before the first frame
    pub const fn lead_in(&self) -> usize {
        let want: usize = if self.bps < 4 { 3840 } else { 5760 }; // 80 or 120 ms
        want.div_ceil(self.symbol_samples())
    }

    /// seconds one frame takes on the air, without the lead-in
    pub fn frame_seconds(&self) -> f64 {
        ((self.spf - 1) as usize * self.symbol_samples()) as f64 / 48_000.0
    }

    /// a short description, e.g. "OFDM, 49 carriers, 16QAM"
    pub fn describe(&self) -> String {
        let cst = |bps| match bps {
            1 => "2-phase".to_string(),
            2 => "4-phase".to_string(),
            3 => "8-phase".to_string(),
            n => format!("{}QAM", 1u32 << n),
        };
        match self.wave {
            Wave::Fsk { .. } => format!("tone-keyed, {} tones", 1u32 << self.bps),
            Wave::Psk { carriers, .. } => format!("{} carriers, {}", carriers.len(), cst(self.bps)),
            Wave::Ofdm { nc, .. } => format!("OFDM, {nc} carriers, {}", cst(self.bps)),
        }
    }
}

const fn period(rate: u32) -> u32 {
    match rate {
        2 => 1,
        3 => 2,
        4 => 4,
        6 => 8,
        7 => 10,
        _ => panic!("unknown code rate"),
    }
}

const fn fsk(spf: u32, bps: u32, rate: u32, sps: u32, first: u32, last: u32, step: u32, k: u32) -> Level {
    Level { wave: Wave::Fsk { sps, first, last, step }, spf, bps, puncture: period(rate), k }
}

const fn psk(spf: u32, bps: u32, markers: u32, carriers: &'static [u32], pulse: Pulse, k: u32) -> Level {
    Level { wave: Wave::Psk { markers, carriers, pulse }, spf, bps, puncture: 2, k }
}

const fn ofdm(spf: u32, bps: u32, rate: u32, grid: (u32, u32, u32), k: u32) -> Level {
    Level { wave: Wave::Ofdm { markers: grid.0, nc: grid.1, first: grid.2 }, spf, bps, puncture: period(rate), k }
}

// OFDM grids as (markers per carrier, carriers, first bin)
const G500: (u32, u32, u32) = (4, 11, 27);
const G2300: (u32, u32, u32) = (1, 49, 8);
const G2750: (u32, u32, u32) = (1, 59, 3);
const SHORT: Level = ofdm(51, 5, 6, G500, 1704);
use Pulse::{Rc, Rrc};

static BW500: [Level; RECORDS] = [
    fsk(125, 3, 2, 128, 54, 74, 2, 96),
    fsk(229, 3, 2, 64, 27, 37, 1, 200),
    fsk(227, 3, 3, 64, 27, 37, 1, 297),
    psk(394, 1, 18, &[1350, 1650], Rc, 368),
    psk(395, 2, 18, &[1350, 1650], Rc, 744),
    psk(376, 2, 12, &[1320, 1500, 1680], Rrc(2), 1080),
    ofdm(150, 2, 3, G500, 1578),
    ofdm(150, 2, 4, G500, 2104),
    ofdm(150, 2, 6, G500, 2524),
    ofdm(150, 3, 4, G500, 3160),
    ofdm(150, 3, 6, G500, 3792),
    ofdm(155, 4, 7, G500, 4755),
    ofdm(155, 5, 6, G500, 5708),
    psk(451, 2, 18, &[1350, 1650], Rc, 856),
    fsk(93, 3, 2, 128, 54, 74, 2, 64),
    SHORT,
    SHORT,
];

static BW2300: [Level; RECORDS] = [
    fsk(125, 3, 2, 128, 33, 96, 2, 96),
    fsk(229, 3, 2, 64, 17, 48, 2, 200),
    fsk(229, 4, 3, 64, 17, 48, 2, 402),
    fsk(396, 4, 3, 32, 9, 24, 1, 736),
    psk(376, 2, 12, &[900, 1500, 2100], Rc, 1080),
    psk(365, 2, 9, &[750, 1250, 1750, 2250], Rc, 1410),
    psk(355, 2, 6, &[700, 1020, 1340, 1660, 1980, 2300], Rrc(2), 2076),
    psk(352, 2, 5, &[702, 930, 1158, 1386, 1614, 1842, 2070, 2298], Rrc(2), 2754),
    psk(349, 2, 4, &[555, 765, 975, 1185, 1395, 1605, 1815, 2025, 2235, 2445], Rrc(1), 3424),
    ofdm(150, 2, 3, G2300, 7197),
    ofdm(150, 2, 4, G2300, 9596),
    ofdm(150, 2, 6, G2300, 11512),
    ofdm(150, 3, 4, G2300, 14398),
    ofdm(150, 3, 6, G2300, 17276),
    ofdm(155, 4, 7, G2300, 21710),
    ofdm(155, 5, 6, G2300, 26056),
    SHORT,
];

static BW2750: [Level; RECORDS] = [
    fsk(125, 3, 2, 128, 24, 103, 2, 96),
    fsk(229, 3, 2, 64, 12, 51, 2, 200),
    fsk(229, 4, 3, 64, 12, 51, 2, 402),
    fsk(396, 4, 3, 32, 6, 25, 1, 736),
    psk(376, 2, 12, &[600, 1500, 2400], Rc, 1080),
    psk(365, 2, 9, &[510, 1170, 1830, 2490], Rc, 1410),
    psk(355, 2, 6, &[450, 870, 1290, 1710, 2130, 2550], Rrc(2), 2076),
    psk(352, 2, 5, &[380, 700, 1020, 1340, 1660, 1980, 2300, 2620], Rrc(2), 2754),
    psk(349, 2, 4, &[375, 625, 875, 1125, 1375, 1625, 1875, 2125, 2375, 2625], Rrc(1), 3424),
    psk(349, 2, 3, &[240, 450, 660, 870, 1080, 1290, 1500, 1710, 1920, 2130, 2340, 2550, 2760], Rrc(1), 4466),
    ofdm(150, 2, 3, G2750, 8667),
    ofdm(150, 2, 4, G2750, 11556),
    ofdm(150, 2, 6, G2750, 13864),
    ofdm(150, 3, 4, G2750, 17338),
    ofdm(150, 3, 6, G2750, 20804),
    ofdm(155, 4, 7, G2750, 26145),
    ofdm(155, 5, 6, G2750, 31376),
];

/// Tables of a phase-keyed or OFDM record.
#[derive(Clone, Debug)]
pub struct GridTables {
    pub level: Level,
    /// transmitted bit `j` is coded bit `chan[j]`
    pub chan: Vec<u32>,
    /// input order of the second encoder
    pub turbo: Vec<u32>,
    /// `marks[c][s - 2]`: symbol `s` (counted from 1) of carrier `c` is a marker
    pub marks: Vec<Vec<bool>>,
}

#[derive(Clone, Debug)]
pub enum Record {
    Fsk(FskTables),
    Grid(GridTables),
}

impl Record {
    pub fn chan(&self) -> &[u32] {
        match self {
            Record::Fsk(t) => &t.chan,
            Record::Grid(t) => &t.chan,
        }
    }

    pub fn turbo(&self) -> &[u32] {
        match self {
            Record::Fsk(t) => &t.turbo,
            Record::Grid(t) => &t.turbo,
        }
    }
}

/// The tables of all 17 records of a mode.
pub fn tables(bw: Bandwidth) -> Vec<Record> {
    let mut r = VbRnd::stream(TABLE_SEED);
    bw.levels()
        .iter()
        .map(|&l| match l.wave {
            Wave::Fsk { sps, first, last, step } => {
                Record::Fsk(one_level(&mut r, FskLevel { symbols: l.spf, bits: l.bps, puncture: l.puncture, sps, first, last, step, k: l.k }))
            }
            _ => {
                let chan = permutation(&mut r, l.n());
                let turbo = permutation(&mut r, l.k);
                let first = if let Wave::Ofdm { first, .. } = l.wave { first } else { 0 };
                let marks = (0..l.carriers())
                    .map(|c| {
                        let mut row = vec![false; (l.spf - 1) as usize];
                        let mut have = 0;
                        while have < l.markers() {
                            let v = if l.is_qam() { 3 + (first + c) % 2 + PILOT_PERIOD * r.draw((l.spf - 3) / PILOT_PERIOD) } else { 5 + r.draw(l.spf - 8) };
                            if !std::mem::replace(&mut row[(v - 2) as usize], true) {
                                have += 1;
                            }
                        }
                        row
                    })
                    .collect();
                Record::Grid(GridTables { level: l, chan, turbo, marks })
            }
        })
        .collect()
}

/// Whether symbol `s` (counted from 1) of the carrier in transform bin `bin` is a pilot (QAM levels).
pub const fn qam_pilot(bin: u32, s: u32) -> bool {
    s % PILOT_PERIOD == if bin % 2 == 0 { 2 } else { 3 }
}

/// The constellation of a phase-keyed or QAM level: point `v` carries the bits of `v`, first bit highest.
pub fn constellation(bps: u32) -> Vec<rustfft::num_complex::Complex64> {
    use rustfft::num_complex::Complex64 as C;
    let h = std::f64::consts::FRAC_1_SQRT_2;
    const Q16: [f64; 4] = [-0.7778175, -0.2592725, 0.7778175, 0.2592725];
    const Q32: [f64; 3] = [0.2121320, 0.6363961, 1.0606602];
    match bps {
        1 => vec![C::new(1.0, 0.0), C::new(-1.0, 0.0)],
        2 => vec![C::new(1.0, 0.0), C::new(0.0, 1.0), C::new(0.0, -1.0), C::new(-1.0, 0.0)],
        3 => vec![C::new(1.0, 0.0), C::new(h, h), C::new(-h, h), C::new(0.0, 1.0), C::new(h, -h), C::new(0.0, -1.0), C::new(-1.0, 0.0), C::new(-h, -h)],
        4 => (0..16).map(|v| C::new(Q16[v >> 2], Q16[v & 3])).collect(),
        5 => {
            let base: Vec<C> = [(0, 0), (1, 0), (1, 2), (2, 0), (0, 1), (1, 1), (0, 2), (2, 1)].iter().map(|&(i, q)| C::new(Q32[i], Q32[q])).collect();
            let mut out = base.clone();
            out.extend(base.iter().map(|z| z.conj()));
            out.extend(base.iter().map(|z| -z.conj()));
            out.extend(base.iter().map(|z| -z));
            out
        }
        _ => Vec::new(),
    }
}

/// Scrambling phasors of the phase-keyed levels: `[carrier][symbol]`.
pub fn psk_phasors() -> Vec<Vec<rustfft::num_complex::Complex64>> {
    let mut r = VbRnd::stream(292.0);
    let mut d = vec![vec![0u32; 471]; 13];
    for range in [0..411, 411..471] {
        for row in d.iter_mut() {
            for k in range.clone() {
                row[k] = r.rand_int(128);
            }
        }
    }
    to_phasors(d)
}

/// Scrambling phasors of the OFDM levels: `[bin][symbol]`.
pub fn ofdm_phasors() -> Vec<Vec<rustfft::num_complex::Complex64>> {
    let mut r = VbRnd::stream(292.0);
    let mut d = vec![vec![0u32; 469]; 128];
    for range in [0..409, 409..469] {
        for row in d.iter_mut() {
            for m in range.clone() {
                row[m] = r.rand_int(128);
            }
        }
    }
    to_phasors(d)
}

fn to_phasors(d: Vec<Vec<u32>>) -> Vec<Vec<rustfft::num_complex::Complex64>> {
    d.iter().map(|row| row.iter().map(|&v| rustfft::num_complex::Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * f64::from(v) / 128.0)).collect()).collect()
}

/// The whitening bits: information bit `i` is sent as `bit ^ whitening[i]`.
pub fn whitening(n: usize) -> Vec<u8> {
    let mut r = VbRnd::stream(292.0);
    (0..n).map(|_| r.rand_int(1) as u8).collect()
}

/// Frame bytes to the bits that are transmitted, in transmission order.
pub fn encode(level: &Level, rec: &Record, frame: &[u8]) -> Vec<u8> {
    let k = level.k as usize;
    let mut bits: Vec<u8> = frame.iter().flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1)).collect();
    bits.resize(k, 0);
    for (b, w) in bits.iter_mut().zip(whitening(k)) {
        *b ^= w;
    }
    let coded = crate::turbo::encode(&bits, rec.turbo(), level.puncture, level.n() as usize);
    rec.chan().iter().map(|&i| coded[i as usize]).collect()
}

/// Soft values in transmission order (positive = 0) to frame bytes.
pub fn decode(level: &Level, chan: &[u32], turbo: &[u32], soft: &[f32], iterations: usize) -> Vec<u8> {
    let k = level.k as usize;
    let mut coded = vec![0.0f32; chan.len()];
    for (j, &i) in chan.iter().enumerate() {
        coded[i as usize] = soft[j];
    }
    let mut bits = crate::turbo::decode(&coded, turbo, level.puncture, k, iterations);
    for (b, w) in bits.iter_mut().zip(whitening(k)) {
        *b ^= w;
    }
    bits.chunks_exact(8).map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | b)).collect()
}

/// Soft values (positive = 0) for the bits of one received point `y`, first bit highest, by the
/// nearest constellation points; `sig2` is the noise power.
pub fn soft_bits(y: rustfft::num_complex::Complex64, cst: &[rustfft::num_complex::Complex64], bps: u32, sig2: f64, out: &mut Vec<f32>) {
    for j in (0..bps).rev() {
        let (mut zero, mut one) = (f64::MAX, f64::MAX);
        for (v, p) in cst.iter().enumerate() {
            let d = (y - p).norm_sqr();
            if (v >> j) & 1 == 0 {
                zero = zero.min(d);
            } else {
                one = one.min(d);
            }
        }
        out.push(((one - zero) / sig2).clamp(-30.0, 30.0) as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_consistent() {
        for bw in Bandwidth::ALL {
            for (l, t) in bw.levels().iter().zip(tables(bw)) {
                let p = l.puncture;
                let coded = if p == 1 { 3 * l.k + 12 } else { l.k + l.k.div_ceil(p) + (l.k - p / 2 - 1) / p + 1 + 12 };
                assert!(coded <= l.n() && l.n() - coded <= 4, "{bw:?} {l:?}");
                assert_eq!(t.chan().len() as u32, l.n());
                if let Record::Grid(g) = &t {
                    assert!(g.marks.iter().all(|m| m.iter().filter(|&&x| x).count() as u32 == l.markers()));
                }
            }
        }
    }

    /// The transmitted bits of a fixed test frame, per record. The values were taken after the
    /// coding chain had been compared with outside test vectors for all 51 records
    /// (`tools/check_vectors.py`), so a change in tables, whitening or coding shows up here.
    #[test]
    fn coding_chain_is_unchanged() {
        let want: [(Bandwidth, [u32; RECORDS]); 3] = [
            (Bandwidth::Hz500, [0x3abca278, 0x3e5de71b, 0x504b7054, 0x332e35ee, 0x1cda473e, 0x39213244, 0x35a58439, 0x3ef54ec7, 0x20fe80b0, 0xb6f10d85, 0xd71f3b66, 0xf994a1e5, 0xdc56d4f1, 0x0105a3fc, 0x371c7fa2, 0x73339dbe, 0xb41500dc]),
            (Bandwidth::Hz2300, [0x3abca278, 0x3e5de71b, 0x70651c24, 0x4e11e480, 0xfbfde134, 0x3d542bbf, 0x92b785cd, 0xe1d9aa97, 0xbf86564d, 0xb663f80e, 0x3edccf3d, 0xc92d12ac, 0xa87e6d6c, 0x206d803c, 0x13030d27, 0xb5188fc5, 0xe7cdc12d]),
            (Bandwidth::Hz2750, [0x3abca278, 0x3e5de71b, 0x70651c24, 0x4e11e480, 0xfbfde134, 0x3d542bbf, 0x92b785cd, 0xe1d9aa97, 0xbf86564d, 0x57af4349, 0x61c39882, 0x2ce00d4c, 0x3be1e217, 0x16715c6a, 0xe9160916, 0x7c66de1b, 0x8956c0c6]),
        ];
        for (bw, hashes) in want {
            let t = tables(bw);
            for (i, l) in bw.levels().iter().enumerate() {
                let mut h: u32 = 0x811c_9dc5;
                for b in encode(l, &t[i], &crate::synth::test_frame(l, i as u32)) {
                    h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
                }
                assert_eq!(h, hashes[i], "{bw:?} record {}", i + 1);
            }
        }
    }
}
