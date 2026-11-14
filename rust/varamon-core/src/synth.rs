//! Test-signal generator: the audio of a transmission, built from the same tables the receiver
//! uses. It exists so the receivers can be tested (self-test, unit tests, comparison with outside
//! reference audio). The program has no way to key a transmitter.

use std::f64::consts::PI;

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::levels::{constellation, encode, ofdm_phasors, psk_phasors, qam_pilot, GridTables, Level, Pulse, Record, Wave};
use crate::tables::FskTables;

const GRAY: [u32; 16] = [0, 1, 3, 2, 7, 6, 4, 5, 15, 14, 12, 13, 8, 9, 11, 10];

fn take(tx: &[u8], ptr: &mut usize, n: u32) -> usize {
    let mut v = 0;
    for _ in 0..n {
        v = 2 * v + usize::from(tx[*ptr]);
        *ptr += 1;
    }
    v
}

/// A window falling from 1 to 0 over `n` samples.
fn fall(n: usize) -> Vec<f64> {
    (1..=n).map(|k| 0.5 * (1.0 + ((k as f64 - 0.5) * PI / n as f64).cos())).collect()
}

/// The pulse of the phase-keyed carriers at 3000 samples/s (32 samples per symbol).
pub fn pulse_taps(p: Pulse) -> Vec<f64> {
    match p {
        Pulse::Rc => (-16..=16)
            .map(|n: i32| {
                let x = f64::from(n) / 16.0;
                if n == 0 {
                    1.0
                } else if n.abs() == 8 {
                    0.5
                } else {
                    (PI * x).sin() * (PI * x).cos() / (x * PI * (1.0 - 4.0 * x * x))
                }
            })
            .collect(),
        Pulse::Rrc(span) => {
            let half = 32 * span as i32;
            (-half..=half)
                .map(|n| {
                    let x = f64::from(n) / 32.0;
                    if n == 0 {
                        4.0 / PI
                    } else if n.abs() == 8 {
                        1.0 / 2f64.sqrt() * ((1.0 + 2.0 / PI) * (PI / 4.0).sin() + (1.0 - 2.0 / PI) * (PI / 4.0).cos())
                    } else {
                        4.0 * (2.0 * PI * x).cos() / (PI * (1.0 - 16.0 * x * x))
                    }
                })
                .collect()
        }
    }
}

/// The smoothing filter that takes the 3000 samples/s carriers to 48 kHz.
fn interp_taps() -> Vec<f64> {
    let f2 = 125.0 / 48000.0;
    (0..32)
        .map(|i| {
            let m = f64::from(i) - 16.0;
            let h = if m == 0.0 { 2.0 * f2 } else { (2.0 * PI * f2 * m).sin() / (PI * m) };
            h * (0.54 - 0.46 * (2.0 * PI * f64::from(i) / 32.0).cos())
        })
        .collect()
}

/// Symbol timing offset of carrier `c`, in samples at 3000 samples/s.
pub const fn stagger(c: usize, nc: usize) -> usize {
    if nc == 3 {
        [0, 11, 21][c]
    } else if c % 2 == 1 {
        16
    } else {
        0
    }
}

fn scatter(x: &[Complex64], stride: usize, offset: usize, h: &[f64], len: usize) -> Vec<Complex64> {
    let mut y = vec![Complex64::new(0.0, 0.0); len.max((x.len() - 1) * stride + offset + h.len())];
    for (i, &v) in x.iter().enumerate() {
        if v.norm_sqr() == 0.0 {
            continue;
        }
        for (j, &t) in h.iter().enumerate() {
            y[offset + i * stride + j] += v * t;
        }
    }
    y.truncate(len);
    y
}

fn fsk(t: &FskTables, txs: &[Vec<u8>], lead: usize, rand: &mut dyn FnMut(u32) -> u32) -> Vec<f64> {
    let l = t.level;
    let mut bins: Vec<u32> = Vec::new();
    for _ in txs {
        bins = (0..lead).map(|i| l.bin(t.hop[i + 1], rand(1 << l.bits))).collect();
    }
    for tx in txs {
        let mut ptr = 0;
        for i in 0..(l.symbols - 1) as usize {
            let v = match t.pilot[i] {
                Some(v) => v,
                None => GRAY[take(tx, &mut ptr, l.bits)],
            };
            bins.push(l.bin(t.hop[i], v));
        }
    }
    let n = 16 * l.sps as usize;
    let mut x: Vec<f64> = bins.iter().flat_map(|&k| (0..n).map(move |i| 2.0 * (2.0 * PI * f64::from(k) * i as f64 / n as f64).sin())).collect();
    let tail = x.len() - n / 32;
    for (v, f) in x[tail..].iter_mut().zip(fall(n / 32)) {
        *v *= f;
    }
    x
}

fn psk(g: &GridTables, txs: &[Vec<u8>], lead: usize, rand: &mut dyn FnMut(u32) -> u32) -> Vec<f64> {
    let l = &g.level;
    let Wave::Psk { carriers, pulse, .. } = l.wave else { unreachable!() };
    let cst = constellation(l.bps);
    let m = cst.len() as u32;
    let nc = carriers.len();
    let nsym = (l.spf - 1) as usize;
    let ph = psk_phasors();
    let mut hp = pulse_taps(pulse);
    *hp.last_mut().unwrap() = 0.0;
    let hi = interp_taps();
    let refs: Vec<Vec<Complex64>> = txs.iter().map(|_| (0..nc).map(|_| cst[rand(m) as usize]).collect()).collect();
    let leads: Vec<Vec<Complex64>> = (0..lead).map(|_| (0..nc).map(|_| cst[rand(m) as usize]).collect()).collect();
    let mut total: Vec<f64> = Vec::new();
    for (c, &fc) in carriers.iter().enumerate() {
        let mut x: Vec<Complex64> = leads.iter().enumerate().map(|(k, v)| v[c] * ph[c][k].conj()).collect();
        for (tx, d0) in txs.iter().zip(&refs) {
            let (mut d, mut ptr) = (d0[c], 0);
            // the bits of one frame run carrier after carrier
            let before: usize = (0..c).map(|cc| g.marks[cc].iter().take(nsym - 1).filter(|&&v| !v).count() * l.bps as usize).sum();
            ptr += before;
            x.push(d * ph[c][0].conj());
            for s in 1..nsym {
                if !(s >= 2 && g.marks[c][s - 2]) {
                    d *= cst[take(tx, &mut ptr, l.bps)];
                }
                x.push(d * ph[c][s].conj());
            }
        }
        x.push(Complex64::new(0.0, 0.0));
        let delay = stagger(c, nc);
        let len3 = x.len() * 32 + delay + hp.len() - 1;
        let y3 = scatter(&x, 32, delay, &hp, len3);
        let y = scatter(&y3, 16, 0, &hi, y3.len() * 16 + hi.len() - 1);
        if y.len() > total.len() {
            total.resize(y.len(), 0.0);
        }
        let step = 2.0 * PI * f64::from(fc) / 48000.0;
        for (i, v) in y.iter().enumerate() {
            let p = step * (i + 1) as f64;
            total[i] += v.re * p.cos() - v.im * p.sin();
        }
    }
    total
}

/// The cells of one OFDM frame as `[symbol][carrier]`, before scrambling.
pub fn ofdm_cells(g: &GridTables, tx: &[u8]) -> Vec<Vec<Complex64>> {
    let l = &g.level;
    let Wave::Ofdm { nc, first, .. } = l.wave else { unreachable!() };
    let cst = constellation(l.bps);
    let one = Complex64::new(1.0, 0.0);
    let mut rows = vec![vec![one; nc as usize]; (l.spf - 1) as usize];
    let mut ptr = 0;
    for bi in 0..nc as usize {
        let mut cell = one;
        for s in 2..=l.spf as usize {
            let marker = g.marks[bi][s - 2];
            if l.is_qam() {
                cell = if qam_pilot(first + bi as u32, s as u32) {
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
            rows[s - 2][bi] = cell;
        }
    }
    assert_eq!(ptr, l.n() as usize);
    rows
}

fn ofdm(g: &GridTables, txs: &[Vec<u8>], lead: usize, rand: &mut dyn FnMut(u32) -> u32) -> Vec<f64> {
    let l = &g.level;
    let Wave::Ofdm { nc, first, .. } = l.wave else { unreachable!() };
    let (nc, first) = (nc as usize, first as usize);
    let cst = constellation(l.bps);
    let ph = ofdm_phasors();
    let mut draw = || {
        let mut rows = vec![vec![Complex64::new(0.0, 0.0); nc]; lead];
        for bi in 0..nc {
            for row in rows.iter_mut() {
                row[bi] = cst[rand(cst.len() as u32) as usize];
            }
        }
        rows
    };
    let mut sets: Vec<_> = (0..if txs.len() > 1 { 3 } else { 1 }).map(|_| draw()).collect();
    let mut rows = sets.swap_remove(sets.len() / 2);
    for tx in txs {
        for (r, row) in ofdm_cells(g, tx).into_iter().enumerate() {
            rows.push(row.iter().enumerate().map(|(bi, &v)| v * ph[first + bi][r].conj()).collect());
        }
    }
    let ifft = FftPlanner::new().plan_fft_inverse(1024);
    let window = fall(32);
    let mut out: Vec<f64> = Vec::with_capacity(rows.len() * 1152);
    let mut saved = [0.0; 32];
    for row in rows {
        let mut buf = vec![Complex64::new(0.0, 0.0); 1024];
        buf[first..first + nc].copy_from_slice(&row);
        ifft.process(&mut buf);
        let x: Vec<f64> = buf.iter().map(|z| 2.0 * z.re).collect();
        let start = out.len();
        out.extend_from_slice(&x[1024 - 128..]);
        out.extend_from_slice(&x);
        for k in 0..32 {
            out[start + k] = window[31 - k] * out[start + k] + window[k] * saved[k];
        }
        saved.copy_from_slice(&x[..32]);
    }
    let tail = out.len() - 32;
    for (v, f) in out[tail..].iter_mut().zip(&window) {
        *v *= f;
    }
    out
}

/// The audio (48 kHz, arbitrary scale) of one transmission carrying `frames` back to back.
/// `rand(m)` supplies the random symbols of the lead-in: an integer below `m`.
pub fn transmission(level: &Level, rec: &Record, frames: &[&[u8]], rand: &mut dyn FnMut(u32) -> u32) -> Vec<f64> {
    let txs: Vec<Vec<u8>> = frames.iter().map(|f| encode(level, rec, f)).collect();
    let lead = level.lead_in();
    match rec {
        Record::Fsk(t) => fsk(t, &txs, lead, rand),
        Record::Grid(g) if matches!(level.wave, Wave::Psk { .. }) => psk(g, &txs, lead, rand),
        Record::Grid(g) => ofdm(g, &txs, lead, rand),
    }
}

/// A frame of the level's size with a correct CRC, filled from a simple counter-based pattern.
pub fn test_frame(level: &Level, seed: u32) -> Vec<u8> {
    let n = level.frame_bytes();
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    let mut f: Vec<u8> = (0..n - 2)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (s >> 16) as u8
        })
        .collect();
    let crc = crate::crc::crc16(&f);
    f.extend(crc.to_be_bytes());
    f
}

/// Normalise to a peak of `peak`, add `silence` seconds before and after, a tuning offset and white noise.
pub fn on_air(x: &[f64], peak: f64, silence: f64, offset_hz: f64, noise: f64, seed: u64) -> Vec<f64> {
    let pad = (silence * 48_000.0) as usize;
    let top = x.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-12);
    let mut y = vec![0.0; pad];
    y.extend(x.iter().map(|v| v / top));
    y.extend(std::iter::repeat(0.0).take(pad));
    if offset_hz != 0.0 {
        let z = crate::fsk::analytic(&y);
        y = z.iter().enumerate().map(|(n, v)| (v * Complex64::from_polar(1.0, 2.0 * PI * offset_hz * n as f64 / 48_000.0)).re).collect();
    }
    let mut s = seed | 1;
    let mut uniform = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    y.iter().map(|v| peak * (v + noise * (-2.0 * uniform().max(1e-12).ln()).sqrt() * (2.0 * PI * uniform()).cos())).collect()
}
