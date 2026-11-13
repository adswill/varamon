//! Control bursts of the 2300 Hz mode: acknowledgements and other short signals between frames.
//!
//! A burst is one lead-in symbol followed by 31, 15 or 7 code symbols of 2048 samples (formats 0,
//! 1, 2; format 2 carries two tones per symbol). The code tones are drawn straight from the table
//! generator, one code after another, from a seed that depends on a station's callsign. The seed is
//! found from the signal itself: every generator state is tried against a burst's tones, and the
//! candidates of several bursts are intersected.

use std::collections::HashMap;
use std::f64::consts::PI;

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::audio::RATE;
use crate::fsk::analytic;
use crate::vb::VbRnd;

pub const SYMBOL: usize = 2048;
/// code symbols per format
pub const CODE_SYMBOLS: [usize; 3] = [31, 15, 7];
/// seed step of each format relative to format 0, as observed on air
const FORMAT_STEP: [u64; 3] = [0, 5, 2];
/// codes drawn in sequence from the session seed
const SESSION_CODES: u32 = 25;
const STATES: u32 = 1 << 24;
const SEED_LIMIT: u64 = (1 << 24) + 2048;

pub fn code_name(code: u32) -> &'static str {
    match code {
        1 => "ACK, stay at this level",
        2 => "NACK",
        3 => "QRT (disconnect)",
        5 => "ACK, up 2 levels",
        6 => "ACK, down a level",
        7 => "ACK, up 1 level",
        8 => "ACK, up 3 levels",
        9 => "ACK, up 4 levels",
        10 => "ACK, second frame only",
        11 => "ACK, first frame only",
        12 | 13 => "ACK request",
        14 => "GO AHEAD",
        15..=18 => "CONNECT REPLY",
        19 => "NACK, frame missed",
        21 => "NO TRAFFIC",
        _ => "control code",
    }
}

/// The tones of one code symbol: one or two bins (the second is 0 in single-tone formats).
pub type Tones = [u32; 2];

#[derive(Clone, Debug)]
pub struct Measured {
    /// 0, 1 or 2
    pub format: usize,
    pub tones: Vec<Tones>,
    /// share of the energy in the strongest tone(s), 0..1
    pub purity: f64,
}

/// Measure the code tones of a burst that starts near `start` seconds and lasts `seconds`.
pub fn measure(x: &[f64], start: f64, seconds: f64, offset_hz: f64) -> Option<Measured> {
    let nsym = (seconds * RATE as f64 / SYMBOL as f64).round() as usize;
    let format = match nsym {
        31..=33 => 0,
        15..=17 => 1,
        7..=9 => 2,
        _ => return None,
    };
    let total = CODE_SYMBOLS[format] + 1;
    let a = ((start * RATE as f64) as usize).saturating_sub(4096);
    let b = (a + (total + 4) * SYMBOL).min(x.len());
    if b <= a + total * SYMBOL {
        return None;
    }
    let mut z = analytic(&x[a..b]);
    for (n, v) in z.iter_mut().enumerate() {
        *v *= Complex64::from_polar(1.0, -2.0 * PI * offset_hz * n as f64 / RATE as f64);
    }
    let fft = FftPlanner::new().plan_fft_forward(SYMBOL);
    let mut buf = vec![Complex64::new(0.0, 0.0); SYMBOL];
    let two = format == 2;
    let mut best: Option<(f64, Vec<Tones>)> = None;
    let mut off = 0;
    while off + total * SYMBOL <= z.len() {
        let (mut sum, mut tones) = (0.0, Vec::with_capacity(total));
        for k in 0..total {
            buf.copy_from_slice(&z[off + k * SYMBOL..off + (k + 1) * SYMBOL]);
            fft.process(&mut buf);
            let p: Vec<f64> = buf[20..110].iter().map(|v| v.norm_sqr()).collect();
            let all: f64 = p.iter().sum::<f64>() + 1e-12;
            let first = (0..p.len()).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
            let second = (0..p.len()).filter(|&i| i != first).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
            if two {
                sum += (p[first] + p[second]) / all;
                tones.push([first.min(second) as u32 + 20, first.max(second) as u32 + 20]);
            } else {
                sum += p[first] / all;
                tones.push([first as u32 + 20, 0]);
            }
        }
        let purity = sum / total as f64;
        if best.as_ref().map_or(true, |b| purity > b.0) {
            best = Some((purity, tones));
        }
        off += 128;
    }
    let (purity, tones) = best?;
    Some(Measured { format, tones: tones[1..].to_vec(), purity })
}

/// The tones of one code, drawn from `r`.
pub fn draw_code(r: &mut VbRnd, format: usize) -> Vec<Tones> {
    let ntones = if format == 2 { 2 } else { 1 };
    (0..CODE_SYMBOLS[format])
        .map(|i| {
            let parity = u32::from(i % 2 == 0); // symbol numbers start at 2
            let mut bases = [0u32; 2];
            let mut n = 0;
            while n < ntones {
                let b = r.rand_int(4) * 14 + 29;
                if !bases[..n].contains(&b) {
                    bases[n] = b;
                    n += 1;
                }
            }
            let mut t = [0u32; 2];
            for i in 0..ntones {
                t[i] = bases[i] + parity + 2 * r.rand_int(6);
            }
            if ntones == 2 && t[0] > t[1] {
                t.swap(0, 1);
            }
            t
        })
        .collect()
}

/// Generator states from which a single-tone code would come out as `tones`.
fn find_states(tones: &[Tones]) -> Vec<u32> {
    let mut out = Vec::new();
    'state: for s in 0..STATES {
        let mut r = VbRnd::from_state(s);
        for (i, t) in tones.iter().enumerate() {
            let b = r.rand_int(4) * 14 + 29;
            if b + u32::from(i % 2 == 0) + 2 * r.rand_int(6) != t[0] {
                continue 'state;
            }
        }
        out.push(s);
    }
    out
}

/// For every generator state: the seed arguments that start a stream there.
struct SeedIndex {
    first: Vec<u32>,
    more: HashMap<u32, Vec<u32>>,
}

impl SeedIndex {
    fn new() -> Self {
        let mut first = vec![u32::MAX; STATES as usize];
        let mut more: HashMap<u32, Vec<u32>> = HashMap::new();
        for base in 0..32768u64 {
            // all seed arguments with the same remainder share the first two steps
            let mut r = VbRnd::new();
            r.reseed();
            r.randomize(base as f64);
            let mut x = base;
            while x < SEED_LIMIT {
                r.advance();
                let s = r.state() as usize;
                if first[s] == u32::MAX {
                    first[s] = x as u32;
                } else {
                    more.entry(s as u32).or_default().push(x as u32);
                }
                x += 32768;
            }
        }
        Self { first, more }
    }

    fn get(&self, state: u32) -> impl Iterator<Item = u32> + '_ {
        let f = self.first[state as usize];
        (f != u32::MAX).then_some(f).into_iter().chain(self.more.get(&state).into_iter().flatten().copied())
    }
}

/// A resolved key: the seed argument of format 0's code stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key(pub u32);

impl Key {
    /// The callsign hash behind the key, for the two possible roles of the sending station.
    pub fn hashes(&self) -> (u32, u32) {
        (self.0.wrapping_sub(60) & 0xFF_FFFF, self.0.wrapping_sub(288) & 0xFF_FFFF)
    }
}

/// Find the key of a station from its control bursts. Needs single-tone bursts (formats 0 and 1)
/// that differ from each other, so that only one seed explains them all.
pub fn resolve(bursts: &[Measured]) -> Option<Key> {
    let mut distinct: Vec<&Measured> = Vec::new();
    for b in bursts.iter().filter(|b| b.format < 2 && b.purity > 0.5) {
        if !distinct.iter().any(|d| d.format == b.format && d.tones == b.tones) {
            distinct.push(b);
        }
    }
    if distinct.len() < 2 {
        return None;
    }
    let index = SeedIndex::new();
    let mut votes: HashMap<u32, usize> = HashMap::new();
    for b in &distinct {
        let per = 2 * CODE_SYMBOLS[b.format];
        let mut seen = Vec::new();
        for s in find_states(&b.tones) {
            let mut r = VbRnd::from_state(s);
            for _ in 0..SESSION_CODES {
                for x in index.get(r.state()) {
                    let base = u64::from(x).wrapping_sub(FORMAT_STEP[b.format]) as u32;
                    if !seen.contains(&base) {
                        seen.push(base);
                    }
                }
                for _ in 0..per {
                    r.retreat();
                }
            }
        }
        for base in seen {
            *votes.entry(base).or_default() += 1;
        }
    }
    let (&key, &n) = votes.iter().max_by_key(|(_, &n)| n)?;
    (n == distinct.len() && votes.values().filter(|&&v| v == n).count() == 1).then_some(Key(key))
}

/// Which code a burst is, given the key: (code number, symbols that match, symbols in the code).
pub fn identify(b: &Measured, key: Key) -> Option<(u32, usize, usize)> {
    let mut r = VbRnd::seeded(u64::from(key.0) + FORMAT_STEP[b.format]);
    let total = CODE_SYMBOLS[b.format];
    let mut best: Option<(u32, usize)> = None;
    for code in 1..=SESSION_CODES {
        let want = draw_code(&mut r, b.format);
        let hits = want.iter().zip(&b.tones).filter(|(w, t)| w == t).count();
        if best.map_or(true, |x| hits > x.1) {
            best = Some((code, hits));
        }
    }
    let (code, hits) = best?;
    (hits * 10 >= total * 7).then_some((code, hits, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(key: u32, format: usize, code: u32) -> Measured {
        let mut r = VbRnd::seeded(u64::from(key) + FORMAT_STEP[format]);
        let mut tones = Vec::new();
        for _ in 0..code {
            tones = draw_code(&mut r, format);
        }
        Measured { format, tones, purity: 0.9 }
    }

    /// The five bursts of the off-air test recording: tones as measured.
    #[test]
    fn off_air_bursts() {
        let long: Vec<Tones> =
            [62, 35, 32, 75, 64, 87, 86, 79, 52, 81, 88, 39, 82, 69, 52, 33, 50, 49, 74, 85, 78, 55, 70, 53, 44, 81, 82, 53, 62, 0, 0].iter().map(|&t| [t, 0]).collect();
        let medium: Vec<Tones> = [48, 67, 76, 97, 50, 71, 66, 71, 90, 29, 84, 45, 76, 75, 92].iter().map(|&t| [t, 0]).collect();
        let key = Key(16_759_090);
        let got = code(key.0, 0, 6);
        assert_eq!(&got.tones[..29], &long[..29]);
        assert_eq!(code(key.0, 1, 1).tones, medium);
        let short_up: Vec<Tones> = vec![[40, 48], [47, 97], [52, 78], [37, 61], [54, 94], [51, 69], [54, 96]];
        assert_eq!(identify(&Measured { format: 2, tones: short_up, purity: 0.8 }, key), Some((7, 7, 7)));
        assert_eq!(key.hashes(), (0xffb8f6, 0xffb812));
    }

    /// The key is found again from two generated bursts, and the third is then identified.
    #[test]
    fn key_from_two_bursts() {
        let key = 0x12_3456 + 60;
        let found = resolve(&[code(key, 0, 6), code(key, 1, 1), code(key, 2, 7)]);
        assert_eq!(found, Some(Key(key)));
        assert_eq!(identify(&code(key, 2, 7), Key(key)), Some((7, 7, 7)));
        assert_eq!(identify(&code(key, 0, 14), Key(key)), Some((14, 31, 31)));
        // one burst alone does not pin the key
        assert_eq!(resolve(&[code(key, 1, 1), code(key, 1, 1)]), None);
    }
}
