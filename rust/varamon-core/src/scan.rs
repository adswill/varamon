//! Scanner: finds transmissions in a recording and decodes each one.
//!
//! This Rust port covers the tone-keyed levels of the 500, 2300 and 2750 Hz modes so far.

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::audio::RATE;
use crate::fsk::{frame_seconds, try_level};
use crate::tables::{fsk_tables, FskTables, BW2300, BW2750, BW500};

pub struct Tables {
    pub modes: Vec<(u32, Vec<FskTables>)>,
}

impl Tables {
    pub fn new() -> Self {
        Self { modes: vec![(2300, fsk_tables(&BW2300)), (2750, fsk_tables(&BW2750)), (500, fsk_tables(&BW500))] }
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Found {
    /// start of the transmission in seconds
    pub time: f64,
    pub seconds: f64,
    pub mode: u32,
    pub level: u32,
    pub frame: Vec<u8>,
    pub crc_ok: bool,
    pub score: f64,
    pub offset: f64,
    /// signal level of the transmission in dB (relative)
    pub level_db: f64,
    /// which station sent it: 0, 1, ... in order of first appearance
    pub station: usize,
}

/// Transmissions as (first sample, last sample), found by energy between 300 and 2900 Hz.
pub fn segments(x: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = x.len();
    if n < RATE {
        return Vec::new();
    }
    let mut planner = FftPlanner::new();
    let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut buf);
    for (i, v) in buf.iter_mut().enumerate() {
        let f = i.min(n - i) as f64 * RATE as f64 / n as f64;
        if !(300.0..=2900.0).contains(&f) {
            *v = Complex64::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    const BLOCK: usize = 480;
    let e: Vec<f64> = buf.chunks_exact(BLOCK).map(|c| (c.iter().map(|v| (v.re / n as f64).powi(2)).sum::<f64>() / BLOCK as f64).sqrt()).collect();
    let mut sorted = e.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let threshold = (sorted[sorted.len() / 10] * 3.0).max(sorted[sorted.len() - 1] * 0.08);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, &v) in e.iter().enumerate() {
        match (v > threshold, start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                runs.push((a, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(a) = start {
        runs.push((a, e.len()));
    }
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in runs {
        match merged.last_mut() {
            Some(last) if a - last.1 < 45 => last.1 = b, // bridge fades shorter than 0.45 s
            _ => merged.push((a, b)),
        }
    }
    merged
        .into_iter()
        .filter(|(a, b)| (b - a) * BLOCK >= 3 * RATE)
        .map(|(a, b)| {
            // level from the loudest half of the blocks, so fades do not pull it down
            let mut v: Vec<f64> = e[a..b].to_vec();
            v.sort_by(|p, q| q.total_cmp(p));
            let top = &v[..v.len().div_ceil(2)];
            let rms = (top.iter().map(|x| x * x).sum::<f64>() / top.len() as f64).sqrt();
            (a * BLOCK, b * BLOCK, 20.0 * rms.max(1e-9).log10())
        })
        .collect()
}

/// Decode one transmission: try every tone-keyed level of every mode that fits its length.
fn decode_segment(s: &[f64], tables: &Tables) -> Option<(u32, u32, crate::fsk::Decoded)> {
    let len = s.len() as f64 / RATE as f64;
    let mut best: Option<(u32, u32, crate::fsk::Decoded)> = None;
    for (mode, levels) in &tables.modes {
        for (i, t) in levels.iter().enumerate() {
            let flen = frame_seconds(t);
            if len < flen - 0.2 {
                continue;
            }
            if let Some(d) = try_level(s, t, 0.0, (len - flen).clamp(0.05, 2.9)) {
                if d.crc_ok {
                    return Some((*mode, i as u32 + 1, d));
                }
                if best.as_ref().map_or(true, |b| d.score > b.2.score) {
                    best = Some((*mode, i as u32 + 1, d));
                }
            }
        }
    }
    best
}

/// Scan a recording at 48 kHz. `report` is called for each transmission as it is decoded.
pub fn scan(x: &[f64], tables: &Tables, mut report: impl FnMut(&Found)) -> Vec<Found> {
    let mut out = Vec::new();
    let mut stations: Vec<(f64, f64, usize)> = Vec::new(); // level, tuning offset, frames seen
    for (a, b, level_db) in segments(x) {
        // 2 s margin before and 0.7 s after: a faded start or end is common off air
        let lo = a.saturating_sub(2 * RATE);
        let hi = (b + RATE * 7 / 10).min(x.len());
        if let Some((mode, level, d)) = decode_segment(&x[lo..hi], tables) {
            // the same station comes back at about the same level and tuning
            let station = match stations.iter().position(|s| (s.0 - level_db).abs() <= 2.5 && (s.1 - d.offset).abs() <= 6.0) {
                Some(i) => {
                    let s = &mut stations[i];
                    s.0 = (s.0 * s.2 as f64 + level_db) / (s.2 + 1) as f64;
                    s.2 += 1;
                    i
                }
                None => {
                    stations.push((level_db, d.offset, 1));
                    stations.len() - 1
                }
            };
            let f = Found {
                time: a as f64 / RATE as f64,
                seconds: (b - a) as f64 / RATE as f64,
                mode,
                level,
                frame: d.frame,
                crc_ok: d.crc_ok,
                score: d.score,
                offset: d.offset,
                level_db,
                station,
            };
            report(&f);
            out.push(f);
        }
    }
    out
}

/// The payloads of one station joined in time order. Exact repeats of the previous frame (a
/// retransmission after a missed acknowledgement) are dropped.
pub fn stream_of(found: &[Found], station: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut prev: Option<&[u8]> = None;
    for f in found.iter().filter(|f| f.crc_ok && f.station == station) {
        if prev == Some(&f.frame[..]) {
            continue;
        }
        prev = Some(&f.frame[..]);
        if let Some(data) = crate::frame::parse(&f.frame).and_then(|p| p.data) {
            out.extend(data);
        }
    }
    out
}
