//! VARA FM: an OFDM signal in the audio of an FM radio (carriers 46.875 Hz apart, symbols of 1024
//! samples plus 128 of guard at 48 kHz) and short tone bursts for acknowledgements.
//!
//! Two bandwidth modes, narrow and wide, with 17 records each. Record 0 is the connect frame
//! (callsigns, unscrambled table), records 1-13 the speed levels, 14 a short frame, 15 the start
//! frame of a turn, 16 a broadcast frame. All records except 0 and 16 are scrambled with a table
//! drawn from the called station's callsign.

use std::f64::consts::{PI, SQRT_2};

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::vb::VbRnd;

pub const RECORDS: usize = 17;
pub const BINS: usize = 128;
pub const ROWS: usize = 201;
pub const NFFT: usize = 1024;
pub const GUARD: usize = 128;
pub const SYMBOL: usize = NFFT + GUARD;
/// In the QAM levels the cells of a carrier at symbols `s` with `s % 9 == 2 + bin % 2` are pilots.
pub const PILOT_PERIOD: u32 = 9;
const SEED: f64 = 43.0;
const BROADCAST_SEED: f64 = 77.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Narrow,
    Wide,
}

impl Mode {
    pub const BOTH: [Mode; 2] = [Mode::Wide, Mode::Narrow];

    pub fn records(self) -> &'static [Level; RECORDS] {
        match self {
            Mode::Narrow => &NARROW,
            Mode::Wide => &WIDE,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Narrow => "narrow",
            Mode::Wide => "wide",
        }
    }
}

/// One record: where its carriers are, how it is coded, and its transmit level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    /// carriers, first bin, bins between carriers
    pub nc: u32,
    pub first: u32,
    pub step: u32,
    /// bits per cell: 2 = differential 4-phase, more = coherent QAM with pilots
    pub bps: u32,
    /// symbols per frame, counting the reference symbol
    pub spf: u32,
    pub markers: u32,
    /// information bits per frame
    pub k: u32,
    /// puncture period of the turbo code (2 = rate 1/2, 4 = 2/3, 6 = 3/4)
    pub puncture: u32,
    pub papr_db: f32,
    pub amp_norm: f32,
}

impl Level {
    pub const fn is_qam(&self) -> bool {
        self.bps > 2
    }

    pub const fn bin(&self, c: u32) -> usize {
        (self.first + self.step * c) as usize
    }

    pub const fn rows(&self) -> usize {
        (self.spf - 1) as usize
    }

    /// coded bits per frame
    pub const fn n(&self) -> u32 {
        if self.bps == 2 {
            self.nc * 2 * (self.spf - 2 - self.markers)
        } else {
            self.nc * self.bps * (self.spf - 1 - self.markers - ((self.spf - 3) / PILOT_PERIOD + 1))
        }
    }

    pub const fn frame_bytes(&self) -> usize {
        (self.k / 8) as usize
    }

    pub fn describe(&self) -> String {
        let cst = match self.bps {
            2 => "4-phase".to_string(),
            3 => "8-phase".to_string(),
            n => format!("{}QAM", 1u32 << n),
        };
        format!("OFDM, {} carriers, {cst}", self.nc)
    }
}

const fn lv(nc: u32, bps: u32, spf: u32, markers: u32, k: u32, rate: u32, papr_db: f32, amp_norm: f32) -> Level {
    let (first, step) = match nc {
        14 => (10, 4),
        29 => (10, 2),
        58 => (10, 1),
        _ => (6, 1),
    };
    let puncture = match rate {
        3 => 2,
        4 => 4,
        _ => 6,
    };
    Level { nc, first, step, bps, spf, markers, k, puncture, papr_db, amp_norm }
}

static NARROW: [Level; RECORDS] = [
    lv(14, 2, 29, 4, 316, 3, 8.6, 5.29),
    lv(14, 2, 200, 4, 2710, 3, 8.6, 5.29),
    lv(29, 2, 200, 2, 5678, 3, 9.3, 7.62),
    lv(58, 2, 200, 1, 11420, 3, 10.0, 10.77),
    lv(58, 2, 200, 1, 15226, 4, 10.0, 10.77),
    lv(58, 3, 201, 1, 20408, 4, 9.9, 9.4844),
    lv(58, 4, 201, 1, 27212, 4, 9.9, 9.05),
    lv(58, 5, 201, 1, 34018, 4, 9.9, 10.28),
    lv(58, 6, 201, 1, 40824, 4, 9.9, 11.2),
    lv(58, 6, 201, 1, 45927, 5, 9.9, 11.2),
    lv(58, 7, 201, 1, 53583, 5, 9.9, 11.24),
    lv(58, 8, 192, 1, 58455, 5, 9.9, 11.18),
    lv(58, 2, 50, 1, 2720, 3, 10.0, 10.77),
    lv(58, 2, 50, 1, 2720, 3, 10.0, 10.77),
    lv(14, 2, 83, 4, 1072, 3, 8.6, 5.29),
    lv(14, 2, 18, 6, 134, 3, 8.6, 5.29),
    lv(14, 2, 69, 4, 876, 3, 8.6, 5.29),
];

static WIDE: [Level; RECORDS] = [
    lv(14, 2, 29, 4, 316, 3, 8.6, 5.29),
    lv(14, 2, 200, 4, 2710, 3, 8.6, 5.29),
    lv(29, 2, 200, 2, 5678, 3, 9.3, 7.62),
    lv(58, 2, 200, 1, 11420, 3, 10.0, 10.77),
    lv(98, 2, 200, 1, 19300, 3, 10.3, 14.0),
    lv(98, 2, 200, 1, 25732, 4, 10.3, 14.0),
    lv(98, 3, 201, 1, 34488, 4, 10.3, 12.32448),
    lv(98, 4, 201, 1, 45986, 4, 10.2, 11.76),
    lv(116, 4, 201, 1, 54434, 4, 10.4, 12.79),
    lv(116, 5, 165, 1, 55672, 4, 10.3, 14.53),
    lv(116, 6, 138, 1, 55672, 4, 10.3, 15.83),
    lv(116, 6, 120, 1, 54279, 5, 10.3, 15.83),
    lv(116, 7, 102, 1, 53583, 5, 10.3, 15.9),
    lv(116, 8, 93, 1, 55671, 5, 10.3, 15.9),
    lv(14, 2, 83, 4, 1072, 3, 8.6, 5.29),
    lv(14, 2, 18, 6, 134, 3, 8.6, 5.29),
    lv(14, 2, 69, 4, 876, 3, 8.6, 5.29),
];

/// What a record is for.
pub fn record_name(rec: usize) -> &'static str {
    match rec {
        0 => "connect",
        14 => "short frame",
        15 => "start frame",
        16 => "broadcast",
        _ => "data",
    }
}

/// Records 0 (connect) and 16 (broadcast) use the unkeyed table.
pub const fn keyed(rec: usize) -> bool {
    rec != 0 && rec != 16
}

/// The tables of one record.
#[derive(Clone, Debug)]
pub struct Tables {
    pub level: Level,
    pub chan: Vec<u32>,
    pub turbo: Vec<u32>,
    /// `marks[c][s - 2]`: symbol `s` of carrier `c` is a marker
    pub marks: Vec<Vec<bool>>,
}

/// The tables of all 17 records of a mode, drawn from one generator stream (record 16 starts its own).
pub fn tables(mode: Mode) -> Vec<Tables> {
    let mut r = VbRnd::stream(SEED);
    mode.records()
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            if i == 16 {
                r = VbRnd::stream(BROADCAST_SEED);
            }
            let chan = crate::tables::permutation(&mut r, l.n());
            let turbo = crate::tables::permutation(&mut r, l.k);
            let gap = if l.spf < 50 { 0 } else { 2 };
            let marks = (0..l.nc)
                .map(|c| {
                    let mut chosen: Vec<u32> = Vec::new();
                    while chosen.len() < l.markers as usize {
                        let v = if l.is_qam() {
                            3 + (l.first + l.step * c) % 2 + PILOT_PERIOD * r.draw((l.spf - 3) / PILOT_PERIOD)
                        } else {
                            3 + r.draw(l.spf - 2)
                        };
                        if chosen.iter().all(|&w| v.abs_diff(w) > gap) {
                            chosen.push(v);
                        }
                    }
                    let mut row = vec![false; l.rows()];
                    for v in chosen {
                        row[(v - 2) as usize] = true;
                    }
                    row
                })
                .collect();
            Tables { level: l, chan, turbo, marks }
        })
        .collect()
}

pub const fn qam_pilot(bin: usize, s: usize) -> bool {
    s % PILOT_PERIOD as usize == 2 + bin % 2
}

/// The scrambling phasors, `[bin][row]`: unkeyed, or drawn from a 24-bit key.
pub fn phasors(key: Option<u32>) -> Vec<[Complex64; ROWS]> {
    let mut r = match key {
        None => VbRnd::stream(SEED),
        Some(k) => VbRnd::seeded(u64::from(k)),
    };
    (0..BINS).map(|_| std::array::from_fn(|_| Complex64::from_polar(1.0, PI * f64::from(r.rand_int(32)) / 15.0))).collect()
}

fn whitening(n: usize) -> Vec<u8> {
    let mut r = VbRnd::stream(SEED);
    (0..n).map(|_| r.rand_int(1) as u8).collect()
}

const fn ginv(mut n: u32) -> u32 {
    let mut s = 1;
    while s < 8 {
        n ^= n >> s;
        s <<= 1;
    }
    n
}

fn amp(p: f64, l: f64, x: i32) -> f64 {
    f64::from((p * f64::from(x) / (l * SQRT_2)) as f32)
}

fn signed(v: u32, si: u32, sq: u32, x: i32, y: i32, p: f64, l: f64) -> Complex64 {
    let sx = if v & si != 0 { -x } else { x };
    let sy = if v & sq != 0 { -y } else { y };
    Complex64::new(amp(p, l, sx), amp(p, l, sy))
}

fn square(bps: u32, p: f64, l: f64) -> Vec<Complex64> {
    let h = bps / 2;
    let m = (1 << h) - 1;
    let c = |g: u32| 2 * ginv(g) as i32 - m as i32;
    (0..1u32 << bps).map(|v| Complex64::new(amp(p, l, c(v >> h)), amp(p, l, c(v & m)))).collect()
}

/// The constellation of a record: point `v` carries the bits of `v`, first bit highest.
pub fn constellation(bps: u32) -> Vec<Complex64> {
    let c = Complex64::new;
    let bit = |v: u32, b: u32| (v >> b) & 1;
    match bps {
        2 => vec![c(1.0, 0.0), c(0.0, 1.0), c(0.0, -1.0), c(-1.0, 0.0)],
        3 => {
            let a = f64::from(std::f64::consts::FRAC_1_SQRT_2 as f32);
            [(-1, -1), (-1, 0), (0, -1), (-1, 1), (1, 0), (1, 1), (1, -1), (0, 1)].iter().map(|&(i, q): &(i32, i32)| c(a * f64::from(i), a * f64::from(q))).collect()
        }
        4 => square(4, 1.1, 3.0),
        5 => {
            let q = [(1, 1), (3, 1), (3, 5), (5, 1), (1, 3), (3, 3), (1, 5), (5, 3)];
            (0..32).map(|v: u32| signed(v, 16, 8, q[v as usize & 7].0, q[v as usize & 7].1, 1.5, 5.0)).collect()
        }
        6 => square(6, 1.6, 7.0),
        7 => (0..128)
            .map(|v| {
                let x = 2 * ginv(bit(v, 3) << 2 | bit(v, 2) << 1 | bit(v, 0)) as i32 + 1;
                let y = 2 * ginv(bit(v, 4) << 1 | bit(v, 1)) as i32 + 1;
                let (x, y) = if x >= 13 {
                    if y <= 3 {
                        (x - 8, y + 8)
                    } else {
                        (16 - x, 16 - y)
                    }
                } else {
                    (x, y)
                };
                signed(v, 64, 32, x, y, 1.8, 11.0)
            })
            .collect(),
        8 => (0..256)
            .map(|v| {
                let x = 2 * ginv(bit(v, 4) << 2 | bit(v, 2) << 1 | bit(v, 0)) as i32 + 1;
                let y = 2 * ginv(bit(v, 5) << 2 | bit(v, 3) << 1 | bit(v, 1)) as i32 + 1;
                signed(v, 128, 64, x, y, 1.7, 15.0)
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Frame bytes to transmitted bits.
pub fn encode(t: &Tables, frame: &[u8]) -> Vec<u8> {
    let k = t.level.k as usize;
    let mut bits: Vec<u8> = frame.iter().flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1)).collect();
    bits.resize(k, 0);
    for (b, w) in bits.iter_mut().zip(whitening(k)) {
        *b ^= w;
    }
    let coded = crate::turbo::encode(&bits, &t.turbo, t.level.puncture, t.level.n() as usize);
    t.chan.iter().map(|&i| coded[i as usize]).collect()
}

/// Soft values in transmission order (positive = 0) to frame bytes.
pub fn decode(t: &Tables, soft: &[f32]) -> Vec<u8> {
    let k = t.level.k as usize;
    let mut coded = vec![0.0f32; t.chan.len()];
    for (j, &i) in t.chan.iter().enumerate() {
        coded[i as usize] = soft[j];
    }
    let mut bits = crate::turbo::decode(&coded, &t.turbo, t.level.puncture, k, 8);
    for (b, w) in bits.iter_mut().zip(whitening(k)) {
        *b ^= w;
    }
    bits.chunks_exact(8).map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | b)).collect()
}

fn take(tx: &[u8], ptr: &mut usize, n: u32) -> usize {
    let mut v = 0;
    for _ in 0..n {
        v = 2 * v + usize::from(tx[*ptr]);
        *ptr += 1;
    }
    v
}

pub type Row = [Complex64; BINS];

/// The symbols of one frame, scrambled, as rows of 128 bins.
pub fn frame_rows(t: &Tables, ph: &[[Complex64; ROWS]], tx: &[u8]) -> Vec<Row> {
    let l = &t.level;
    let cst = constellation(l.bps);
    let one = Complex64::new(1.0, 0.0);
    let mut rows = vec![[Complex64::new(0.0, 0.0); BINS]; l.rows()];
    let mut ptr = 0;
    for c in 0..l.nc {
        let b = l.bin(c);
        let mut cell = one;
        for s in 2..=l.spf as usize {
            let marker = t.marks[c as usize][s - 2];
            if l.is_qam() {
                cell = if qam_pilot(b, s) {
                    one
                } else if marker {
                    if cell.norm() > 0.0 {
                        cell / cell.norm()
                    } else {
                        one
                    }
                } else {
                    cst[take(tx, &mut ptr, l.bps)]
                };
            } else if s > 2 {
                if marker {
                    cell /= cell.norm();
                } else {
                    cell *= cst[take(tx, &mut ptr, l.bps)];
                }
            }
            rows[s - 2][b] = cell * ph[b][s - 2].conj();
        }
    }
    assert_eq!(ptr, l.n() as usize);
    rows
}

fn fall(n: usize) -> Vec<f64> {
    (1..=n).map(|k| 0.5 * (1.0 + ((k as f64 - 0.5) * PI / n as f64).cos())).collect()
}

/// Rows of cells to audio: each symbol with its guard, 32 samples cross-faded, every sample times `scale`.
pub fn synth(rows: &[Row], scale: f64) -> Vec<f64> {
    let ifft = FftPlanner::new().plan_fft_inverse(NFFT);
    let window = fall(32);
    let mut out: Vec<f64> = Vec::with_capacity(rows.len() * SYMBOL);
    let mut saved = [0.0; 32];
    for row in rows {
        let mut buf = vec![Complex64::new(0.0, 0.0); NFFT];
        buf[..BINS].copy_from_slice(row);
        ifft.process(&mut buf);
        let x: Vec<f64> = buf.iter().map(|z| 2.0 * z.re * scale).collect();
        let start = out.len();
        out.extend_from_slice(&x[NFFT - GUARD..]);
        out.extend_from_slice(&x);
        for k in 0..32 {
            out[start + k] = window[31 - k] * out[start + k] + window[k] * saved[k];
        }
        saved.copy_from_slice(&x[..32]);
    }
    if let Some(tail) = out.len().checked_sub(32) {
        for (v, f) in out[tail..].iter_mut().zip(&window) {
            *v *= f;
        }
    }
    out
}

fn db_amp(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn gain(papr_db: f64, drive_db: f64) -> f64 {
    db_amp(-(papr_db + 2.0 + (10.0 - papr_db) / 1000.0 - drive_db))
}

/// The audio of one transmission of record `rec` (48 kHz, full scale 1): `lead` random symbols, then the
/// frames. `rand(m)` supplies the lead-in cells, an integer below `m`; `key` scrambles the keyed records.
pub fn transmission(mode: Mode, tabs: &[Tables], rec: usize, key: u32, frames: &[&[u8]], lead: usize, drive_db: f64, rand: &mut dyn FnMut(u32) -> u32) -> Vec<f64> {
    let t = &tabs[rec];
    let l = &mode.records()[rec];
    let cst = constellation(l.bps);
    let mut rows = vec![[Complex64::new(0.0, 0.0); BINS]; lead];
    for c in 0..l.nc {
        for row in rows.iter_mut() {
            row[l.bin(c)] = cst[rand(cst.len() as u32) as usize];
        }
    }
    let ph = phasors(keyed(rec).then_some(key));
    for f in frames {
        rows.extend(frame_rows(t, &ph, &encode(t, f)));
    }
    let g = gain(f64::from(l.papr_db), drive_db) / f64::from(l.amp_norm);
    synth(&rows, g).into_iter().map(|v| v.clamp(-1.0, 1.0)).collect()
}

// ------------------------------------------------------------------------------------------------
// Tone bursts (acknowledgements and the other short signals)

pub const TONE_SYMBOL: usize = 1024;
pub const CODE_ROWS: usize = 10;
pub const LEAD_ROWS: usize = 28;
pub type Tones = [u32; 3];

fn tone_rows(r: &mut VbRnd, n: usize, nblock: u32, block0: u32) -> Vec<Tones> {
    (0..n)
        .map(|_| {
            let mut bases: Vec<u32> = Vec::with_capacity(3);
            let mut tones = [0; 3];
            for t in tones.iter_mut() {
                let b = loop {
                    let b = r.rand_int(nblock - 1) * 7 + block0;
                    if !bases.contains(&b) {
                        break b;
                    }
                };
                bases.push(b);
                *t = b + r.rand_int(6);
            }
            tones
        })
        .collect()
}

/// The 28 rows a tone burst's lead-in is cut from (its last rows are sent).
pub fn lead_in() -> Vec<Tones> {
    let mut r = VbRnd::stream(SEED);
    for _ in 0..150_000 {
        r.advance();
    }
    tone_rows(&mut r, LEAD_ROWS, 3, 27)
}

/// The 24 codes of a link: 1-22 from the called station's callsign hash, 23-24 (connect replies)
/// from the caller's. Index = code number - 1.
pub fn codebook(called: u32, caller: u32) -> Vec<Vec<Tones>> {
    let mut r = VbRnd::seeded(u64::from(called) + 10);
    let mut out: Vec<Vec<Tones>> = (0..22).map(|_| tone_rows(&mut r, CODE_ROWS, 7, 13)).collect();
    let mut r = VbRnd::seeded(u64::from(caller) + 199);
    out.extend((0..2).map(|_| tone_rows(&mut r, CODE_ROWS, 7, 13)));
    out
}

pub fn code_name(code: usize) -> &'static str {
    match code {
        1 => "ACK, stay at this level",
        2 => "NACK",
        3 => "QRT (disconnect)",
        4 => "BREAK",
        5 => "ACK, up 2 levels",
        6 => "ACK, down a level",
        7 => "ACK, up 1 level",
        8 => "ACK, up 3 levels",
        9 => "ACK, up 6 levels",
        10 => "ACK, second frame only",
        11 => "ACK, first frame only",
        12 | 13 => "ACK request",
        14 => "GO AHEAD",
        20 => "IDLE",
        21 => "NO TRAFFIC",
        22 => "RELAY FAILED",
        23 => "CONNECT REPLY (narrow)",
        24 => "CONNECT REPLY (wide)",
        _ => "control code",
    }
}

/// The audio of a tone burst: the last `lead` lead-in rows, then the code's rows.
pub fn tone_audio(code: &[Tones], lead: usize, drive_db: f64) -> Vec<f64> {
    let sin: Vec<f64> = (0..TONE_SYMBOL).map(|n| (2.0 * PI * n as f64 / TONE_SYMBOL as f64).sin()).collect();
    let g = 2.0 * (1.0 / 2.45) * gain(7.45, drive_db);
    let all = lead_in();
    all[LEAD_ROWS - lead..]
        .iter()
        .chain(code)
        .flat_map(|t| {
            let sin = &sin;
            (0..TONE_SYMBOL).map(move |n| (t.iter().map(|&k| sin[(k as usize * n) % TONE_SYMBOL]).sum::<f64>() * g).clamp(-1.0, 1.0))
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Frame contents

/// The connect frame (record 0): who calls whom, through which relays, and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connect {
    pub kind: u8,
    pub source: String,
    pub via1: String,
    pub via2: String,
    pub target: String,
    pub registered: bool,
    pub wide: bool,
    pub compression: u8,
}

fn call_field(f: &[u8]) -> String {
    if f[0] == 0 {
        return String::new();
    }
    let base: String = f[..7].iter().map(|&b| b as char).collect::<String>().trim_end().to_string();
    match f[7] {
        0 => base,
        0x10 => base + "-T",
        0x11 => base + "-R",
        0x12 => base + "-X",
        0x13 => base + "-L",
        n => format!("{base}-{n}"),
    }
}

pub fn connect(f: &[u8]) -> Option<Connect> {
    if f.len() != 39 || !crate::crc::frame_ok(f) {
        return None;
    }
    let call = |i: usize| call_field(&f[1 + 8 * i..9 + 8 * i]);
    let source = call(0);
    if source.is_empty() || !source.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'/') {
        return None;
    }
    Some(Connect {
        kind: (f[0] >> 5) & 3,
        registered: f[0] & 4 != 0,
        wide: f[0] & 2 != 0,
        source,
        via1: call(1),
        via2: call(2),
        target: call(3),
        compression: (f[34] >> 1) & 3,
    })
}

/// A data, start, short or broadcast frame: payload and control byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Data {
    pub data: Vec<u8>,
    pub compressed: bool,
    pub last: bool,
    pub full: bool,
}

/// Read a frame whose CRC is correct. `relayed`: the link goes through relays (one more trailer byte).
pub fn data(rec: usize, f: &[u8], relayed: bool) -> Option<Data> {
    let over = match rec {
        0 => return None,
        16 => 3,
        15 => 5 + usize::from(relayed),
        _ => 4 + usize::from(relayed),
    };
    let p = f.len().checked_sub(over)?;
    let c = f[p];
    let full = c & 1 != 0;
    let n = if full { p } else { f[..p].iter().rposition(|&b| b == crate::frame::TERMINATOR)? };
    Some(Data { data: f[..n].to_vec(), compressed: c & 0x20 != 0, last: c & 2 != 0, full })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_consistent() {
        for mode in Mode::BOTH {
            for (l, t) in mode.records().iter().zip(tables(mode)) {
                let p = l.puncture;
                let coded = l.k + l.k.div_ceil(p) + (l.k - p / 2 - 1) / p + 1 + 12;
                assert!(coded <= l.n(), "{mode:?} {l:?}");
                assert_eq!(t.chan.len() as u32, l.n());
                assert!(t.marks.iter().all(|m| m.iter().filter(|&&x| x).count() as u32 == l.markers));
            }
        }
    }

    #[test]
    fn constellations_have_distinct_points() {
        for bps in 2..=8 {
            let c = constellation(bps);
            assert_eq!(c.len(), 1 << bps);
            for (i, a) in c.iter().enumerate() {
                assert!(c[i + 1..].iter().all(|b| (a - b).norm() > 1e-6), "bps {bps}");
            }
        }
    }
}
