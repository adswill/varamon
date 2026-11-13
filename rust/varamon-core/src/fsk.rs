//! Receiver for the tone-keyed levels: finds a frame by its pilot symbols (searching timing and
//! tuning offset), reads soft bits from the tone energies and runs the turbo decoder.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::audio::RATE;
use crate::crc::frame_ok;
use crate::tables::FskTables;
use crate::turbo;
use crate::vb::VbRnd;

const GRAY: [u32; 16] = [0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10];
const WHITENING_SEED: f64 = 292.0;

#[derive(Clone, Debug)]
pub struct Decoded {
    pub frame: Vec<u8>,
    pub crc_ok: bool,
    /// share of the pilot symbols' energy found at the expected tones, 0..1
    pub score: f64,
    /// seconds from the start of the slice to the first frame symbol
    pub start: f64,
    /// tuning offset in Hz
    pub offset: f64,
}

/// The analytic signal of `x` (negative frequencies removed).
pub fn analytic(x: &[f64]) -> Vec<Complex64> {
    let n = x.len();
    let mut planner = FftPlanner::new();
    let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut buf);
    for (i, v) in buf.iter_mut().enumerate() {
        if i == 0 || (n % 2 == 0 && i == n / 2) {
        } else if i < n.div_ceil(2) {
            *v *= 2.0;
        } else {
            *v = Complex64::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    let scale = 1.0 / n as f64;
    buf.iter_mut().for_each(|v| *v *= scale);
    buf
}

fn shifted(z: &[Complex64], df: f64) -> Vec<Complex64> {
    z.iter().enumerate().map(|(n, &v)| v * Complex64::from_polar(1.0, -2.0 * PI * df * n as f64 / RATE as f64)).collect()
}

fn power(fft: &Arc<dyn Fft<f64>>, z: &[Complex64], start: usize, n: usize, bins: usize, buf: &mut [Complex64]) -> Vec<f64> {
    buf.copy_from_slice(&z[start..start + n]);
    fft.process(buf);
    buf[..bins].iter().map(|v| v.norm_sqr()).collect()
}

/// Try to decode one frame of the level described by `t` from the slice `x` (48 kHz), with the
/// frame starting between `t_lo` and `t_hi` seconds.
pub fn try_level(x: &[f64], t: &FskTables, t_lo: f64, t_hi: f64) -> Option<Decoded> {
    let l = t.level;
    let n = 16 * l.sps as usize;
    let nsym = (l.symbols - 1) as usize;
    let (first, nb) = (l.first as usize, l.bins() as usize);
    let bins = first + nb + 2;
    let hop = n / 8;
    let z = analytic(x);
    let fft = FftPlanner::new().plan_fft_forward(n);
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    let pilots: Vec<(usize, usize)> = t.pilot.iter().enumerate().filter_map(|(i, p)| p.map(|v| (i, l.bin(t.hop[i], v) as usize))).collect();
    let base = (t_lo * RATE as f64 / hop as f64) as usize;
    let span = ((t_hi - t_lo) * RATE as f64 / hop as f64) as usize;
    let mut best: Option<(f64, f64, usize)> = None;
    let mut df = -90.0;
    while df <= 90.0 {
        let zz = shifted(&z, df);
        let frames_avail = (zz.len().saturating_sub(n)) / hop;
        let want = (base + span + nsym * 8 + 8).min(frames_avail);
        if want > base {
            let s: Vec<Vec<f64>> = (base..want).map(|q| power(&fft, &zz, q * hop, n, bins, &mut buf)).collect();
            for q0 in 0..span {
                if q0 + pilots.last().map_or(0, |p| p.0) * 8 >= s.len() {
                    break;
                }
                let (mut num, mut den) = (0.0, 1e-9);
                for &(i, b) in &pilots {
                    let row = &s[q0 + i * 8];
                    num += row[b];
                    den += row[first..first + nb].iter().sum::<f64>();
                }
                if best.map_or(true, |b| num / den > b.0) {
                    best = Some((num / den, df, (base + q0) * hop));
                }
            }
        }
        df += 6.0;
    }
    let (score, df, start) = best?;
    if start + nsym * n > z.len() {
        return None;
    }
    let zz = shifted(&z, df);
    let levels = 1usize << l.bits;
    let mut llr = Vec::with_capacity(l.n() as usize);
    for i in 0..nsym {
        if t.pilot[i].is_some() {
            continue;
        }
        let p = power(&fft, &zz, start + i * n, n, n.min(bins + 64), &mut buf);
        let mut e: Vec<f64> = (0..levels).map(|v| p[l.bin(t.hop[i], GRAY[v]) as usize]).collect();
        let mean = e.iter().sum::<f64>() / levels as f64 + 1e-12;
        e.iter_mut().for_each(|v| *v /= mean);
        for b in (0..l.bits).rev() {
            let (mut zero, mut one) = (0.0f64, 0.0f64);
            for (v, &ev) in e.iter().enumerate() {
                if (v >> b) & 1 == 0 {
                    zero = zero.max(ev);
                } else {
                    one = one.max(ev);
                }
            }
            llr.push(((zero + 1e-9).ln() - (one + 1e-9).ln()) as f32);
        }
    }
    let mut coded = vec![0.0f32; t.chan.len()];
    for (j, &i) in t.chan.iter().enumerate() {
        coded[i as usize] = 2.0 * llr[j];
    }
    let k = l.k as usize;
    let mut bits = turbo::decode(&coded, &t.turbo, l.puncture, k, 10);
    let mut white = VbRnd::stream(WHITENING_SEED);
    bits.iter_mut().for_each(|b| *b ^= white.rand_int(1) as u8);
    let frame: Vec<u8> = bits.chunks_exact(8).map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | b)).collect();
    Some(Decoded { crc_ok: frame_ok(&frame), frame, score, start: start as f64 / RATE as f64, offset: df })
}

/// Frame length in seconds, without the lead-in.
pub fn frame_seconds(t: &FskTables) -> f64 {
    ((t.level.symbols - 1) * 16 * t.level.sps) as f64 / RATE as f64
}
