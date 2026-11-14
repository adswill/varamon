//! Receiver for the OFDM levels: 11, 49 or 59 carriers 46.875 Hz apart, symbols of 1024 samples
//! plus 128 of guard. The lower levels are differential along time with marker cells; the QAM
//! levels are coherent with pilot cells.

use std::f64::consts::PI;

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::audio::RATE;
use crate::crc::frame_ok;
use crate::fsk::analytic;
use crate::levels::{constellation, decode, qam_pilot, soft_bits, GridTables, Wave};

const NFFT: usize = 1024;
const GUARD: usize = 128;
const SYMBOL: usize = NFFT + GUARD;
/// The transform window starts this far into a symbol: inside the guard, clear of the cross-fade.
const EARLY: usize = 96;

#[derive(Clone, Debug)]
pub struct Decoded {
    pub frame: Vec<u8>,
    pub crc_ok: bool,
    /// how well the markers or pilots agree, 0..1
    pub score: f64,
    /// seconds from the start of the slice to the frame's first symbol
    pub start: f64,
    /// tuning offset in Hz
    pub offset: f64,
}

/// A slice of audio cut into OFDM symbols: timing and tuning offset come from the guard interval,
/// so one front end serves every level that uses the same carriers.
pub struct Front {
    first: usize,
    nc: usize,
    /// `[symbol][carrier]`
    rows: Vec<Vec<Complex64>>,
    phase: usize,
    pub offset: f64,
    /// share of the signal that repeats after 1024 samples, 0..1
    pub guard: f64,
}

impl Front {
    pub fn new(x: &[f64], first: usize, nc: usize) -> Option<Self> {
        let z = analytic(x);
        if z.len() < NFFT + 4 * SYMBOL {
            return None;
        }
        // each symbol's last 128 samples are sent again at its start
        let n = z.len() - NFFT;
        let c: Vec<Complex64> = (0..n).map(|i| z[i] * z[i + NFFT].conj()).collect();
        let mut fold = vec![Complex64::new(0.0, 0.0); SYMBOL];
        let mut sum: Complex64 = c[..GUARD].iter().sum();
        let mut energy = 0.0;
        for i in 0..n - GUARD {
            fold[i % SYMBOL] += sum;
            sum += c[i + GUARD] - c[i];
            energy += z[i].norm_sqr();
        }
        let phase = (0..SYMBOL).max_by(|&a, &b| fold[a].norm().total_cmp(&fold[b].norm()))?;
        let guard = fold[phase].norm() / (energy * GUARD as f64 / SYMBOL as f64 + 1e-12);
        let offset = -fold[phase].arg() / (2.0 * PI * NFFT as f64 / RATE as f64);
        let w = -2.0 * PI * offset / RATE as f64;
        let fft = FftPlanner::new().plan_fft_forward(NFFT);
        let mut buf = vec![Complex64::new(0.0, 0.0); NFFT];
        let early: Vec<Complex64> =
            (0..nc).map(|bi| Complex64::from_polar(1.0 / NFFT as f64, 2.0 * PI * (first + bi) as f64 * (GUARD - EARLY) as f64 / NFFT as f64)).collect();
        let rows = (0..(z.len() - phase) / SYMBOL)
            .map(|q| {
                let at = phase + q * SYMBOL + EARLY;
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = z[at + i] * Complex64::from_polar(1.0, w * (at + i) as f64);
                }
                fft.process(&mut buf);
                (0..nc).map(|bi| buf[first + bi] * early[bi]).collect()
            })
            .collect();
        Some(Self { first, nc, rows, phase, offset, guard })
    }

    pub fn of(x: &[f64], g: &GridTables) -> Option<Self> {
        let Wave::Ofdm { nc, first, .. } = g.level.wave else { panic!("not an OFDM level") };
        Self::new(x, first as usize, nc as usize)
    }

    /// Symbols cut from the slice.
    pub fn symbols(&self) -> usize {
        self.rows.len()
    }

    /// The cell of carrier `bi` (counted from the first bin) in symbol `q`, timing and tuning corrected.
    pub fn cell_at(&self, q: usize, bi: usize) -> Complex64 {
        self.rows[q][bi]
    }

    /// Where symbol 0 starts, in samples from the start of the slice.
    pub fn phase(&self) -> usize {
        self.phase
    }

    pub fn fits(&self, g: &GridTables) -> bool {
        matches!(g.level.wave, Wave::Ofdm { nc, first, .. } if nc as usize == self.nc && first as usize == self.first)
    }

    fn cell(&self, ph: &[Vec<Complex64>], q0: usize, r: usize, bi: usize) -> Complex64 {
        self.rows[q0 + r][bi] * ph[self.first + bi][r]
    }

    /// How well a frame of this level fits with its first symbol at symbol `q0`.
    fn score(&self, g: &GridTables, ph: &[Vec<Complex64>], q0: usize) -> f64 {
        let l = &g.level;
        let nrows = (l.spf - 1) as usize;
        let (mut num, mut den) = (0.0, 1e-12);
        if l.is_qam() {
            // pilots of one carrier add up in phase when the channel holds still
            for bi in 0..self.nc {
                let mut acc = Complex64::new(0.0, 0.0);
                for r in 0..nrows {
                    if qam_pilot((self.first + bi) as u32, r as u32 + 2) {
                        let v = self.cell(ph, q0, r, bi);
                        acc += v;
                        den += v.norm();
                    }
                }
                num += acc.norm();
            }
        } else {
            let mut acc = Complex64::new(0.0, 0.0);
            for bi in 0..self.nc {
                for r in 1..nrows {
                    if g.marks[bi][r] {
                        let d = self.cell(ph, q0, r, bi) * self.cell(ph, q0, r - 1, bi).conj();
                        acc += d;
                        den += d.norm();
                    }
                }
            }
            num = acc.norm();
        }
        num / den
    }

    /// The best position of a frame of this level: (agreement, first symbol).
    pub fn find(&self, g: &GridTables, ph: &[Vec<Complex64>]) -> Option<(f64, usize)> {
        let nrows = (g.level.spf - 1) as usize;
        let (score, mut q0) = (0..(self.rows.len() + 1).checked_sub(nrows)?).map(|q0| (self.score(g, ph, q0), q0)).max_by(|a, b| a.0.total_cmp(&b.0))?;
        // a transmission can carry several frames back to back: start from the first of them
        while q0 >= nrows && self.score(g, ph, q0 - nrows) > 0.7 * score {
            q0 -= nrows;
        }
        Some((self.score(g, ph, q0), q0))
    }

    pub fn has_room(&self, g: &GridTables, q0: usize) -> bool {
        q0 + (g.level.spf - 1) as usize <= self.rows.len()
    }

    pub fn seconds(&self, q0: usize) -> f64 {
        (self.phase + q0 * SYMBOL) as f64 / RATE as f64
    }

    /// Decode the frame whose first symbol is symbol `q0`.
    pub fn decode(&self, g: &GridTables, ph: &[Vec<Complex64>], q0: usize) -> Decoded {
        let l = &g.level;
        let nrows = (l.spf - 1) as usize;
        let cst = constellation(l.bps);
        let mut points: Vec<Complex64> = Vec::with_capacity((l.n() / l.bps) as usize);
        for bi in 0..self.nc {
            let col: Vec<Complex64> = (0..nrows).map(|r| self.cell(ph, q0, r, bi)).collect();
            if l.is_qam() {
                let pilots: Vec<usize> = (0..nrows).filter(|&r| qam_pilot((self.first + bi) as u32, r as u32 + 2)).collect();
                let mut seg = 0;
                for r in 0..nrows {
                    let s = r as u32 + 2;
                    if qam_pilot((self.first + bi) as u32, s) || g.marks[bi][r] {
                        continue;
                    }
                    // channel by straight line between the pilots on either side
                    while seg + 1 < pilots.len() && pilots[seg + 1] <= r {
                        seg += 1;
                    }
                    let h = if r <= pilots[0] {
                        col[pilots[0]]
                    } else if seg + 1 >= pilots.len() {
                        col[pilots[seg]]
                    } else {
                        let (a, b) = (pilots[seg], pilots[seg + 1]);
                        let f = (r - a) as f64 / (b - a) as f64;
                        col[a] * (1.0 - f) + col[b] * f
                    };
                    points.push(col[r] / h);
                }
            } else {
                for r in 1..nrows {
                    if !g.marks[bi][r] {
                        points.push(col[r] * col[r - 1].conj() / (col[r - 1].norm_sqr() + 1e-18));
                    }
                }
            }
        }
        // noise power from how far the points sit from the nearest constellation point
        let spread = points.iter().map(|y| cst.iter().map(|p| (y - p).norm_sqr()).fold(f64::MAX, f64::min)).sum::<f64>() / points.len() as f64;
        let sig2 = spread.max(0.003);
        let mut soft = Vec::with_capacity(l.n() as usize);
        for &y in &points {
            soft_bits(y, &cst, l.bps, sig2, &mut soft);
        }
        let frame = decode(l, &g.chan, &g.turbo, &soft, 8);
        Decoded { crc_ok: frame_ok(&frame), frame, score: self.score(g, ph, q0), start: self.seconds(q0), offset: self.offset }
    }
}

/// Try to decode one frame of the level from the slice `x` (48 kHz).
pub fn try_level(x: &[f64], g: &GridTables, ph: &[Vec<Complex64>]) -> Option<Decoded> {
    let front = Front::of(x, g)?;
    let (_, q0) = front.find(g, ph)?;
    Some(front.decode(g, ph, q0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::{ofdm_phasors, tables, Bandwidth, Record};
    use crate::synth::{on_air, test_frame, transmission};

    #[test]
    fn round_trip_every_ofdm_level() {
        let ph = ofdm_phasors();
        for bw in Bandwidth::ALL {
            let t = tables(bw);
            for (i, l) in bw.levels().iter().enumerate() {
                let Record::Grid(g) = &t[i] else { continue };
                if !matches!(l.wave, Wave::Ofdm { .. }) {
                    continue;
                }
                let frame = test_frame(l, i as u32);
                let mut n = 7u32;
                let mut rand = |m: u32| {
                    n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (n >> 16) % m
                };
                let x = on_air(&transmission(l, &t[i], &[&frame], &mut rand), 0.3, 0.4, 3.0, 0.01, 9 + i as u64);
                let d = try_level(&x, g, &ph).expect("no frame found");
                assert!(d.crc_ok, "{bw:?} record {}: score {:.2}, offset {:.1}", i + 1, d.score, d.offset);
                assert_eq!(d.frame, frame);
                assert!((d.offset - 3.0).abs() < 0.5, "offset {:.2}", d.offset);
            }
        }
    }
}
