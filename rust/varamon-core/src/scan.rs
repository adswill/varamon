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
    /// where the frame itself starts and ends, in seconds
    pub frame_start: f64,
    pub frame_end: f64,
}

/// A short burst between frames: an acknowledgement or other control signal (not decoded).
#[derive(Clone, Debug)]
pub struct Control {
    pub time: f64,
    pub seconds: f64,
    pub level_db: f64,
    pub station: usize,
    /// burst format 0, 1 or 2 (31, 15 or 7 code symbols), if the length fits one
    pub format: Option<usize>,
    /// (code number, matching symbols, symbols in the code), once the station's key is known
    pub code: Option<(u32, usize, usize)>,
}

/// Everything heard, in time order.
#[derive(Clone, Debug)]
pub enum Event {
    Frame(Found),
    Control(Control),
}

impl Event {
    pub fn time(&self) -> f64 {
        match self {
            Event::Frame(f) => f.frame_start,
            Event::Control(c) => c.time,
        }
    }
    fn end(&self) -> f64 {
        match self {
            Event::Frame(f) => f.frame_end,
            Event::Control(c) => c.time + c.seconds,
        }
    }
    fn level(&self) -> f64 {
        match self {
            Event::Frame(f) => f.level_db,
            Event::Control(c) => c.level_db,
        }
    }
    fn set_station(&mut self, s: usize) {
        match self {
            Event::Frame(f) => f.station = s,
            Event::Control(c) => c.station = s,
        }
    }
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
fn decode_segment(s: &[f64], tables: &Tables) -> Option<(u32, u32, crate::fsk::Decoded, f64)> {
    let len = s.len() as f64 / RATE as f64;
    let mut best: Option<(u32, u32, crate::fsk::Decoded, f64)> = None;
    for (mode, levels) in &tables.modes {
        for (i, t) in levels.iter().enumerate() {
            let flen = frame_seconds(t);
            if len < flen - 0.2 {
                continue;
            }
            if let Some(d) = try_level(s, t, 0.0, (len - flen).clamp(0.05, 2.9)) {
                if d.crc_ok {
                    return Some((*mode, i as u32 + 1, d, flen));
                }
                if best.as_ref().map_or(true, |b| d.score > b.2.score) {
                    best = Some((*mode, i as u32 + 1, d, flen));
                }
            }
        }
    }
    best
}

/// Every burst of at least 0.25 s as (start, end, level in dB), cut at gaps of 0.08 s or more.
fn bursts(x: &[f64]) -> Vec<(f64, f64, f64)> {
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
    let db: Vec<f64> =
        buf.chunks_exact(BLOCK).map(|c| 10.0 * (c.iter().map(|v| (v.re / n as f64).powi(2)).sum::<f64>() / BLOCK as f64).max(1e-18).log10()).collect();
    let mut sorted = db.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let threshold = sorted[sorted.len() / 20] + 8.0;
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, &v) in db.iter().chain(std::iter::once(&f64::MIN)).enumerate() {
        match (v > threshold, start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                match out.last_mut() {
                    Some(last) if a - last.1 < 8 => last.1 = i,
                    _ => out.push((a, i)),
                }
                start = None;
            }
            _ => {}
        }
    }
    out.into_iter()
        .filter(|(a, b)| b - a >= 25)
        .map(|(a, b)| {
            let mut v = db[a..b].to_vec();
            v.sort_by(|p, q| q.total_cmp(p));
            let top = &v[..v.len().div_ceil(2)];
            (a as f64 * 0.01, b as f64 * 0.01, top.iter().sum::<f64>() / top.len() as f64)
        })
        .collect()
}

/// Work out who sent what from the turn-taking. A link is half duplex: after a station stops, the
/// other one answers within a fraction of a second. So a burst that follows the previous one
/// closely comes from the other station; after a longer silence the same station is sending
/// again (its acknowledgement was not heard here), unless the level clearly says otherwise.
fn assign_stations(events: &mut [Event]) {
    const TURNAROUND: f64 = 0.6; // seconds; replies come after about 0.2 s
    const NEW_LINK: f64 = 20.0;
    const CLEAR_DB: f64 = 6.0;
    let mut pair = [0usize, 1]; // labels of the two stations of the current link
    let mut next_label = 2;
    let mut cur = 0usize; // index into `pair`
    let mut level = [f64::NAN; 2];
    let mut prev_end = f64::NEG_INFINITY;
    for (i, ev) in events.iter_mut().enumerate() {
        let gap = ev.time() - prev_end;
        let lv = ev.level();
        if i > 0 {
            if gap > NEW_LINK {
                pair = [next_label, next_label + 1];
                next_label += 2;
                cur = 0;
                level = [f64::NAN; 2];
            } else if gap < TURNAROUND {
                cur ^= 1;
            } else {
                let other = cur ^ 1;
                let far_from_cur = !level[cur].is_nan() && (lv - level[cur]).abs() >= CLEAR_DB;
                let near_other = level[other].is_nan() || (lv - level[other]).abs() < (lv - level[cur]).abs();
                if far_from_cur && near_other {
                    cur = other;
                }
            }
        }
        level[cur] = if level[cur].is_nan() { lv } else { 0.7 * level[cur] + 0.3 * lv };
        ev.set_station(pair[cur]);
        prev_end = ev.end();
    }
}

/// Scan a recording at 48 kHz. Returns every frame and control burst in time order, with the
/// sending station worked out from the turn-taking.
pub fn scan(x: &[f64], tables: &Tables) -> Scan {
    let mut events = Vec::new();
    for (a, b, level_db) in segments(x) {
        // 2 s margin before and 0.7 s after: a faded start or end is common off air
        let lo = a.saturating_sub(2 * RATE);
        let hi = (b + RATE * 7 / 10).min(x.len());
        if let Some((mode, level, d, flen)) = decode_segment(&x[lo..hi], tables) {
            let frame_start = lo as f64 / RATE as f64 + d.start;
            events.push(Event::Frame(Found {
                time: a as f64 / RATE as f64,
                seconds: (b - a) as f64 / RATE as f64,
                mode,
                level,
                frame: d.frame,
                crc_ok: d.crc_ok,
                score: d.score,
                offset: d.offset,
                level_db,
                station: 0,
                frame_start,
                frame_end: frame_start + flen,
            }));
        }
    }
    // short bursts that are not part of a frame: acknowledgements and other control signals
    let spans: Vec<(f64, f64)> = events.iter().map(|e| (e.time() - 0.1, e.end() + 0.05)).collect();
    for (a, b, level_db) in bursts(x) {
        let (mut a, mut b) = (a, b);
        // cut away the part covered by a frame
        for &(s, e) in &spans {
            if a < e && b > s {
                if a >= s && b <= e {
                    b = a;
                } else if a < s {
                    b = b.min(s);
                } else {
                    a = a.max(e);
                }
            }
        }
        if b - a >= 0.25 && b - a < 3.0 {
            events.push(Event::Control(Control { time: a, seconds: b - a, level_db, station: 0, format: None, code: None }));
        }
    }
    events.sort_by(|p, q| p.time().total_cmp(&q.time()));
    assign_stations(&mut events);
    let keys = decode_control(x, &mut events);
    Scan { events, keys }
}

/// The result of a scan.
pub struct Scan {
    pub events: Vec<Event>,
    /// for each station whose control bursts could be resolved: (station, key)
    pub keys: Vec<(usize, crate::ctrl::Key)>,
}

/// Measure every control burst, find each station's key from its own bursts and name the codes.
fn decode_control(x: &[f64], events: &mut [Event]) -> Vec<(usize, crate::ctrl::Key)> {
    use crate::ctrl;
    let mut offsets: Vec<f64> = events.iter().filter_map(|e| if let Event::Frame(f) = e { f.crc_ok.then_some(f.offset) } else { None }).collect();
    offsets.sort_by(|a, b| a.total_cmp(b));
    let offset = offsets.get(offsets.len() / 2).copied().unwrap_or(0.0);
    let mut measured: Vec<(usize, usize, ctrl::Measured)> = Vec::new(); // (event index, station, tones)
    for (i, e) in events.iter_mut().enumerate() {
        if let Event::Control(c) = e {
            if let Some(m) = ctrl::measure(x, c.time, c.seconds, offset) {
                c.format = Some(m.format);
                measured.push((i, c.station, m));
            }
        }
    }
    let mut keys = Vec::new();
    let mut stations: Vec<usize> = measured.iter().map(|m| m.1).collect();
    stations.sort_unstable();
    stations.dedup();
    for st in stations {
        let own: Vec<ctrl::Measured> = measured.iter().filter(|m| m.1 == st).map(|m| m.2.clone()).collect();
        if let Some(key) = ctrl::resolve(&own) {
            keys.push((st, key));
            for (i, _, m) in measured.iter().filter(|m| m.1 == st) {
                if let Event::Control(c) = &mut events[*i] {
                    c.code = ctrl::identify(m, key);
                }
            }
        }
    }
    keys
}

/// The frames among the events.
pub fn frames(events: &[Event]) -> Vec<&Found> {
    events.iter().filter_map(|e| if let Event::Frame(f) = e { Some(f) } else { None }).collect()
}

/// The payloads of one station joined in time order. Exact repeats of the previous frame (a
/// retransmission after a missed acknowledgement) are dropped.
pub fn stream_of(found: &[&Found], station: usize) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl(time: f64, seconds: f64, level_db: f64) -> Event {
        Event::Control(Control { time, seconds, level_db, station: 9, format: None, code: None })
    }

    fn stations(events: &[Event]) -> Vec<usize> {
        events.iter().map(|e| if let Event::Control(c) = e { c.station } else { 9 }).collect()
    }

    #[test]
    fn quick_replies_alternate() {
        // data, quick reply, data, quick reply
        let mut e = vec![ctl(0.0, 5.0, 80.0), ctl(5.2, 0.7, 70.0), ctl(6.1, 5.0, 79.0), ctl(11.3, 0.3, 71.0)];
        assign_stations(&mut e);
        assert_eq!(stations(&e), [0, 1, 0, 1]);
    }

    #[test]
    fn unheard_reply_keeps_the_same_station() {
        // the other station's replies are too weak to be heard: longer silences, same level
        let mut e = vec![ctl(0.0, 5.0, 80.0), ctl(6.8, 5.0, 78.0), ctl(13.6, 5.0, 79.0)];
        assign_stations(&mut e);
        assert_eq!(stations(&e), [0, 0, 0]);
    }

    #[test]
    fn clear_level_change_after_a_silence_switches() {
        let mut e = vec![ctl(0.0, 5.0, 74.0), ctl(6.0, 5.0, 74.0), ctl(12.0, 5.0, 66.0), ctl(18.0, 5.0, 74.0)];
        assign_stations(&mut e);
        assert_eq!(stations(&e), [0, 0, 1, 0]);
    }

    #[test]
    fn long_silence_starts_a_new_pair() {
        let mut e = vec![ctl(0.0, 5.0, 80.0), ctl(5.2, 0.7, 70.0), ctl(60.0, 5.0, 75.0), ctl(65.2, 0.7, 60.0)];
        assign_stations(&mut e);
        assert_eq!(stations(&e), [0, 1, 2, 3]);
    }
}
