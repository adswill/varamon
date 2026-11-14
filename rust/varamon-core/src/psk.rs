//! Receiver for the phase-keyed levels: 2 to 13 carriers, 93.75 symbols/s, differential 2- or
//! 4-phase. Each carrier is mixed down and matched-filtered; the frame is found by its marker
//! symbols (a marker repeats the previous symbol's phase), which also give the tuning offset.

use std::f64::consts::PI;

use rustfft::num_complex::Complex64;

use crate::audio::RATE;
use crate::crc::frame_ok;
use crate::levels::{decode, GridTables, Pulse, Wave};
use crate::synth::stagger;

/// Samples between matched-filter outputs (3000 per second).
const STEP: usize = 16;
/// Filter outputs per symbol.
const SYMBOL: usize = 32;

#[derive(Clone, Debug)]
pub struct Decoded {
    pub frame: Vec<u8>,
    pub crc_ok: bool,
    /// how well the markers agree, 0..1
    pub score: f64,
    /// seconds from the start of the slice to the frame's reference symbol
    pub start: f64,
    /// tuning offset in Hz
    pub offset: f64,
}

/// The pulse at 48 kHz, as the receiver's matched filter.
fn matched(p: Pulse) -> Vec<f64> {
    let half: i32 = match p {
        Pulse::Rc => 256,
        Pulse::Rrc(span) => 512 * span as i32,
    };
    let h: Vec<f64> = (-half..=half)
        .map(|n| match p {
            Pulse::Rc => {
                let x = f64::from(n) / 256.0;
                if n == 0 {
                    1.0
                } else if n.abs() == 128 {
                    0.5
                } else {
                    (PI * x).sin() * (PI * x).cos() / (x * PI * (1.0 - 4.0 * x * x))
                }
            }
            Pulse::Rrc(_) => {
                let x = f64::from(n) / 512.0;
                if n.abs() == 128 {
                    1.0 / 2f64.sqrt() * ((1.0 + 2.0 / PI) * (PI / 4.0).sin() + (1.0 - 2.0 / PI) * (PI / 4.0).cos())
                } else {
                    4.0 * (2.0 * PI * x).cos() / (PI * (1.0 - 16.0 * x * x))
                }
            }
        })
        .collect();
    let e: f64 = h.iter().map(|v| v * v).sum();
    h.iter().map(|v| v / e).collect()
}

/// Carrier `fc` mixed to zero and matched-filtered, one output every 16 samples.
fn carrier(x: &[f64], fc: f64, h: &[f64]) -> Vec<Complex64> {
    let w = -2.0 * PI * fc / RATE as f64;
    let bb: Vec<Complex64> = x.iter().enumerate().map(|(n, &v)| Complex64::from_polar(v, w * n as f64)).collect();
    if bb.len() < h.len() {
        return Vec::new();
    }
    (0..(bb.len() - h.len()) / STEP + 1)
        .map(|m| {
            let s = &bb[m * STEP..m * STEP + h.len()];
            let (mut re, mut im) = (0.0, 0.0);
            for (v, &t) in s.iter().zip(h) {
                re += v.re * t;
                im += v.im * t;
            }
            Complex64::new(re, im)
        })
        .collect()
}

/// A slice of audio prepared for one phase-keyed level: every carrier filtered once.
pub struct Front<'a> {
    g: &'a GridTables,
    ph: &'a [Vec<Complex64>],
    y: Vec<Vec<Complex64>>,
    nsym: usize,
    half: usize,
    offset: f64,
}

impl<'a> Front<'a> {
    /// `offset` is a tuning offset in Hz to take out before filtering.
    pub fn new(x: &[f64], g: &'a GridTables, ph: &'a [Vec<Complex64>], offset: f64) -> Self {
        let Wave::Psk { carriers, pulse, .. } = g.level.wave else { panic!("not a phase-keyed level") };
        let h = matched(pulse);
        let y = carriers.iter().map(|&fc| carrier(x, f64::from(fc) + offset, &h)).collect();
        Self { g, ph, y, nsym: (g.level.spf - 1) as usize, half: h.len() / 2, offset }
    }

    /// Last filter index a frame can start at.
    fn last_start(&self) -> Option<usize> {
        let nc = self.y.len();
        let need = (self.nsym - 1) * SYMBOL + (0..nc).map(|c| stagger(c, nc)).max().unwrap_or(0) + 1;
        self.y.iter().map(|v| v.len()).min()?.checked_sub(need)
    }

    /// Marker sum at filter index `m0`: (agreement 0..1, mean rotation per symbol in radians).
    fn markers(&self, m0: usize) -> (f64, f64) {
        let nc = self.y.len();
        let (mut acc, mut tot) = (Complex64::new(0.0, 0.0), 1e-12);
        for c in 0..nc {
            let at = |s: usize| self.y[c][m0 + stagger(c, nc) + s * SYMBOL] * self.ph[c][s];
            for s in 2..self.nsym {
                if self.g.marks[c][s - 2] {
                    let d = at(s) * at(s - 1).conj();
                    acc += d;
                    tot += d.norm();
                }
            }
        }
        (acc.norm() / tot, acc.arg())
    }

    /// The best frame start between `lo` and `hi` seconds: (agreement, filter index, rotation).
    pub fn find(&self, lo: f64, hi: f64) -> Option<(f64, usize, f64)> {
        let last = self.last_start()?;
        let to_m = |t: f64| ((t * RATE as f64) as usize).saturating_sub(self.half) / STEP;
        let mut best: Option<(f64, usize, f64)> = None;
        for m0 in to_m(lo)..=to_m(hi).min(last) {
            let (score, theta) = self.markers(m0);
            if best.map_or(true, |b| score > b.0) {
                best = Some((score, m0, theta));
            }
        }
        best
    }

    /// Seconds from the start of the slice to the middle of the reference symbol at filter index `m0`.
    pub fn seconds(&self, m0: usize) -> f64 {
        (m0 * STEP + self.half) as f64 / RATE as f64
    }

    /// Decode the frame whose reference symbol sits at filter index `m0`.
    pub fn decode(&self, m0: usize, score: f64, theta: f64) -> Decoded {
        let l = &self.g.level;
        let nc = self.y.len();
        let rot = Complex64::from_polar(1.0, -theta);
        let mut soft = Vec::with_capacity(l.n() as usize);
        for c in 0..nc {
            let at = |s: usize| self.y[c][m0 + stagger(c, nc) + s * SYMBOL] * self.ph[c][s];
            let d: Vec<Complex64> = (1..self.nsym).map(|s| at(s) * at(s - 1).conj() * rot).collect();
            let mean = d.iter().map(|v| v.norm()).sum::<f64>() / d.len() as f64 + 1e-12;
            for s in 1..self.nsym {
                if s >= 2 && self.g.marks[c][s - 2] {
                    continue;
                }
                let v = d[s - 1] / mean * 3.0;
                if l.bps == 1 {
                    soft.push(v.re as f32);
                } else {
                    soft.push((v.re + v.im) as f32);
                    soft.push((v.re - v.im) as f32);
                }
            }
        }
        let frame = decode(l, &self.g.chan, &self.g.turbo, &soft, 8);
        Decoded { crc_ok: frame_ok(&frame), frame, score, start: self.seconds(m0), offset: self.offset + theta / (2.0 * PI) * 93.75 }
    }
}

/// Try to decode one frame of the level from the slice `x` (48 kHz), the frame starting between
/// `t_lo` and `t_hi` seconds.
pub fn try_level(x: &[f64], g: &GridTables, ph: &[Vec<Complex64>], t_lo: f64, t_hi: f64) -> Option<Decoded> {
    let front = Front::new(x, g, ph, 0.0);
    let (score, m0, theta) = front.find(t_lo, t_hi)?;
    let offset = theta / (2.0 * PI) * 93.75;
    if offset.abs() < 1.5 {
        return Some(front.decode(m0, score, theta));
    }
    // a second pass with the tuning offset taken out before the filters
    let t = front.seconds(m0);
    let front = Front::new(x, g, ph, offset);
    let (score, m0, theta) = front.find((t - 0.02).max(0.0), t + 0.02)?;
    Some(front.decode(m0, score, theta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::{psk_phasors, tables, Bandwidth, Record};
    use crate::synth::{on_air, test_frame, transmission};

    #[test]
    fn round_trip_every_phase_keyed_level() {
        let ph = psk_phasors();
        for bw in Bandwidth::ALL {
            let t = tables(bw);
            for (i, l) in bw.levels().iter().enumerate() {
                let Record::Grid(g) = &t[i] else { continue };
                if !matches!(l.wave, Wave::Psk { .. }) {
                    continue;
                }
                let frame = test_frame(l, i as u32);
                let mut n = 0u32;
                let mut rand = |m: u32| {
                    n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (n >> 16) % m
                };
                let x = on_air(&transmission(l, &t[i], &[&frame], &mut rand), 0.3, 0.4, 7.0, 0.05, 5 + i as u64);
                let d = try_level(&x, g, &ph, 0.0, 1.0).expect("no frame found");
                assert!(d.crc_ok, "{bw:?} record {}: score {:.2}, offset {:.1}", i + 1, d.score, d.offset);
                assert_eq!(d.frame, frame);
                assert!((d.offset - 7.0).abs() < 1.0, "offset {:.1}", d.offset);
            }
        }
    }
}
