//! Reading recordings: 16-bit PCM WAV, any sample rate, brought to 48 kHz mono.

use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;

pub const RATE: usize = 48_000;

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// Read a WAV file. Returns the first channel as samples in the int16 range, and the sample rate.
pub fn read_wav(path: &Path) -> io::Result<(Vec<f64>, usize)> {
    let raw = fs::read(path)?;
    if raw.len() < 12 || &raw[0..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
        return Err(bad("not a WAV file"));
    }
    let u16_at = |p: usize| u16::from_le_bytes([raw[p], raw[p + 1]]) as usize;
    let u32_at = |p: usize| u32::from_le_bytes([raw[p], raw[p + 1], raw[p + 2], raw[p + 3]]) as usize;
    let (mut pos, mut fmt) = (12usize, None);
    while pos + 8 <= raw.len() {
        let (tag, size) = (&raw[pos..pos + 4], u32_at(pos + 4));
        let body = pos + 8;
        if tag == b"fmt " && body + 16 <= raw.len() {
            fmt = Some((u16_at(body), u16_at(body + 2), u32_at(body + 4), u16_at(body + 14)));
        } else if tag == b"data" {
            let (kind, channels, rate, bits) = fmt.ok_or_else(|| bad("WAV data before format"))?;
            if !(kind == 1 || kind == 0xFFFE) || bits != 16 || channels == 0 {
                return Err(bad("16-bit PCM WAV expected"));
            }
            // recorders stopped abruptly leave a wrong size: take what is in the file
            let end = if size == 0 || body + size > raw.len() { raw.len() } else { body + size };
            let frame = 2 * channels;
            let out = raw[body..end].chunks_exact(frame).map(|c| f64::from(i16::from_le_bytes([c[0], c[1]]))).collect();
            return Ok((out, rate));
        }
        pos = body + size + (size & 1);
    }
    Err(bad("WAV file has no audio data"))
}

/// Band-limited resampling to 48 kHz (windowed sinc, 16 taps; a low-pass first when decimating).
pub fn to_48k(x: &[f64], rate: usize) -> Vec<f64> {
    if rate == RATE {
        return x.to_vec();
    }
    let filtered;
    let x = if rate > RATE {
        filtered = low_pass(x, 0.45 * RATE as f64 / rate as f64);
        &filtered[..]
    } else {
        x
    };
    let n_out = x.len() * RATE / rate;
    let step = rate as f64 / RATE as f64;
    (0..n_out)
        .map(|j| {
            let t = j as f64 * step;
            let i = t.floor() as isize;
            let frac = t - i as f64;
            let mut acc = 0.0;
            for k in -7..=8isize {
                let idx = (i + k).clamp(0, x.len() as isize - 1) as usize;
                let u = frac - k as f64;
                let sinc = if u.abs() < 1e-12 { 1.0 } else { (PI * u).sin() / (PI * u) };
                acc += x[idx] * sinc * (0.5 + 0.5 * (PI * u / 8.0).cos());
            }
            acc
        })
        .collect()
}

fn low_pass(x: &[f64], cutoff: f64) -> Vec<f64> {
    let n = ((4.0 / cutoff) as usize) | 1;
    let mid = (n - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..n)
        .map(|i| {
            let k = i as f64 - mid;
            let sinc = if k == 0.0 { 1.0 } else { (2.0 * PI * cutoff * k).sin() / (2.0 * PI * cutoff * k) };
            2.0 * cutoff * sinc * (0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        })
        .collect();
    let sum: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= sum);
    let half = n / 2;
    (0..x.len())
        .map(|i| {
            let mut acc = 0.0;
            for (j, &c) in h.iter().enumerate() {
                let idx = i as isize + half as isize - j as isize;
                if idx >= 0 && (idx as usize) < x.len() {
                    acc += c * x[idx as usize];
                }
            }
            acc
        })
        .collect()
}

/// Read a recording and return it at 48 kHz.
pub fn load(path: &Path) -> io::Result<Vec<f64>> {
    let (x, rate) = read_wav(path)?;
    Ok(to_48k(&x, rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_a_tone() {
        let rate = 44_100;
        let x: Vec<f64> = (0..rate).map(|n| (2.0 * PI * 1000.0 * n as f64 / rate as f64).sin()).collect();
        let y = to_48k(&x, rate);
        assert_eq!(y.len(), RATE);
        let err = (2000..46_000).map(|n| (y[n] - (2.0 * PI * 1000.0 * n as f64 / RATE as f64).sin()).abs()).fold(0.0, f64::max);
        assert!(err < 0.01, "largest error {err}");
    }
}
