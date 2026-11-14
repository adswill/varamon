//! Scanner: finds transmissions in a recording and decodes each one, whatever its bandwidth mode
//! and level.

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::audio::RATE;
use crate::levels::{self, Bandwidth, GridTables, Record, Wave};
use crate::{fsk, ofdm, psk};

/// Everything the receivers need, generated once.
pub struct Tables {
    pub modes: Vec<(Bandwidth, Vec<Record>)>,
    pub psk: Vec<Vec<Complex64>>,
    pub ofdm: Vec<Vec<Complex64>>,
}

impl Tables {
    pub fn new() -> Self {
        Self { modes: Bandwidth::ALL.iter().map(|&b| (b, levels::tables(b))).collect(), psk: levels::psk_phasors(), ofdm: levels::ofdm_phasors() }
    }

    fn mode(&self, bw: Bandwidth) -> &[Record] {
        &self.modes.iter().find(|m| m.0 == bw).expect("every mode has tables").1
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
    /// 0 for the first frame of a transmission, 1 for the one sent right behind it, ...
    pub part: usize,
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
    /// what the burst is when the key does not explain it: a connect request, BREAK, IDLE, CQ ...
    pub other: Option<String>,
    /// the code number if the burst announces unconnected traffic: 48 = CQ, 45, 47, 58 = a message
    pub announces: Option<u32>,
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

/// One decoded (or attempted) frame inside a slice of audio.
struct Hit {
    mode: u32,
    level: u32,
    frame: Vec<u8>,
    crc_ok: bool,
    score: f64,
    offset: f64,
    /// seconds from the start of the slice
    start: f64,
    seconds: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Fsk,
    Psk,
    Ofdm,
}

/// What kind of signal a transmission is: one tone at a time, OFDM (every symbol's end repeats at
/// its start) or a few phase-keyed carriers.
fn kind_of(s: &[f64]) -> Kind {
    let mut planner = FftPlanner::new();
    let mut share = 0.0f64;
    for (n, lo, hi) in [(1024usize, 6usize, 64usize), (512, 3, 32)] {
        let fft = planner.plan_fft_forward(n);
        let window: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect();
        let mut ratios: Vec<f64> = s
            .chunks_exact(n)
            .map(|c| {
                let mut buf: Vec<Complex64> = c.iter().zip(&window).map(|(&v, &w)| Complex64::new(v * w, 0.0)).collect();
                fft.process(&mut buf);
                let p: Vec<f64> = buf[lo..hi].iter().map(|v| v.norm_sqr()).collect();
                p.iter().fold(0.0f64, |a, &v| a.max(v)) / (p.iter().sum::<f64>() + 1e-12)
            })
            .collect();
        if ratios.is_empty() {
            continue;
        }
        ratios.sort_by(|a, b| a.total_cmp(b));
        share = share.max(ratios[ratios.len() / 2]);
    }
    if share > 0.3 {
        return Kind::Fsk;
    }
    match ofdm::Front::new(s, 8, 49) {
        Some(f) if f.guard > 0.35 => Kind::Ofdm,
        _ => Kind::Psk,
    }
}

/// The bandwidth modes to try, most likely first: the 500 Hz mode keeps to 1250-1750 Hz.
fn modes_in_order(s: &[f64]) -> Vec<Bandwidth> {
    let n = s.len();
    let mut buf: Vec<Complex64> = s.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut buf);
    let (mut inside, mut all) = (0.0, 1e-12);
    for (i, v) in buf.iter().enumerate().take(n / 2) {
        let f = i as f64 * RATE as f64 / n as f64;
        if (300.0..=2900.0).contains(&f) {
            all += v.norm_sqr();
            if (1150.0..=1850.0).contains(&f) {
                inside += v.norm_sqr();
            }
        }
    }
    if inside / all > 0.9 {
        vec![Bandwidth::Hz500, Bandwidth::Hz2300, Bandwidth::Hz2750]
    } else {
        vec![Bandwidth::Hz2300, Bandwidth::Hz2750, Bandwidth::Hz500]
    }
}

fn try_fsk(s: &[f64], tables: &Tables, modes: &[Bandwidth]) -> Vec<Hit> {
    let len = s.len() as f64 / RATE as f64;
    let mut best: Option<Hit> = None;
    for &bw in modes {
        for (i, rec) in tables.mode(bw).iter().enumerate() {
            let Record::Fsk(t) = rec else { continue };
            let flen = fsk::frame_seconds(t);
            if len < flen - 0.2 {
                continue;
            }
            let hit = |d: fsk::Decoded| Hit { mode: bw.hz(), level: i as u32 + 1, frame: d.frame, crc_ok: d.crc_ok, score: d.score, offset: d.offset, start: d.start, seconds: flen };
            let Some(d) = fsk::try_level(s, t, 0.0, (len - flen).clamp(0.05, 2.9)) else { continue };
            if d.crc_ok {
                // more frames may sit right before and right behind
                let mut out = vec![hit(d)];
                while out[0].start > flen {
                    let prev = out[0].start - flen;
                    match fsk::try_level(s, t, prev - 0.03, prev + 0.03) {
                        Some(d) if d.crc_ok => out.insert(0, hit(d)),
                        _ => break,
                    }
                }
                loop {
                    let next = out.last().map_or(0.0, |h| h.start + h.seconds);
                    if next + flen > len + 0.05 {
                        break;
                    }
                    match fsk::try_level(s, t, (next - 0.03).max(0.0), next + 0.03) {
                        Some(d) if d.crc_ok => out.push(hit(d)),
                        _ => break,
                    }
                }
                return out;
            }
            if best.as_ref().map_or(true, |b| d.score > b.score) {
                best = Some(hit(d));
            }
        }
    }
    best.into_iter().collect()
}

fn grids<'a>(tables: &'a Tables, bw: Bandwidth, psk: bool) -> impl Iterator<Item = (usize, &'a GridTables)> {
    tables.mode(bw).iter().enumerate().filter_map(move |(i, rec)| match rec {
        Record::Grid(g) if matches!(g.level.wave, Wave::Psk { .. }) == psk => Some((i, g)),
        _ => None,
    })
}

/// Average power spectrum of a slice in bins of 11.7 Hz (4096-point transforms, half overlapped).
fn spectrum(s: &[f64]) -> Vec<f64> {
    const N: usize = 4096;
    let fft = FftPlanner::new().plan_fft_forward(N);
    let window: Vec<f64> = (0..N).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos()).collect();
    let mut p = vec![0.0; N / 8]; // up to 6 kHz
    let mut at = 0;
    while at + N <= s.len() {
        let mut buf: Vec<Complex64> = s[at..at + N].iter().zip(&window).map(|(&v, &w)| Complex64::new(v * w, 0.0)).collect();
        fft.process(&mut buf);
        for (acc, v) in p.iter_mut().zip(&buf) {
            *acc += v.norm_sqr();
        }
        at += N / 2;
    }
    p
}

/// How well the spectrum fits a set of carriers, each about 100 Hz wide: 0..1.
fn carrier_fit(p: &[f64], carriers: &[u32]) -> f64 {
    let bin = RATE as f64 / 4096.0;
    let mut mask = vec![false; p.len()];
    for &fc in carriers {
        let (lo, hi) = (((f64::from(fc) - 50.0) / bin).round() as usize, ((f64::from(fc) + 50.0) / bin).round() as usize);
        mask[lo..=hi.min(p.len() - 1)].iter_mut().for_each(|m| *m = true);
    }
    // only the audio passband counts
    let band = (300.0 / bin) as usize..(2900.0 / bin) as usize;
    let inside: f64 = band.clone().filter(|&i| mask[i]).map(|i| p[i]).sum();
    let count = band.clone().filter(|&i| mask[i]).count() as f64;
    let total: f64 = band.map(|i| p[i] * p[i]).sum::<f64>().sqrt() + 1e-12;
    inside / (count.sqrt() * total)
}

fn try_psk(s: &[f64], tables: &Tables, modes: &[Bandwidth]) -> Vec<Hit> {
    let len = s.len() as f64 / RATE as f64;
    let mut best: Option<Hit> = None;
    // the 500 Hz mode is tried apart from the two wide ones
    let groups: Vec<Vec<Bandwidth>> = if modes[0] == Bandwidth::Hz500 { vec![vec![modes[0]], modes[1..].to_vec()] } else { vec![modes[..2].to_vec(), modes[2..].to_vec()] };
    for group in groups {
        let mut cands: Vec<(f64, Bandwidth, usize, &GridTables, f64)> = Vec::new();
        // the carriers show in the spectrum: only the records whose carrier set fits best get the
        // full search, which filters every carrier
        let p = spectrum(s);
        let mut fits: Vec<(f64, Bandwidth, usize, &GridTables)> = Vec::new();
        for &bw in &group {
            for (i, g) in grids(tables, bw, true) {
                if let Wave::Psk { carriers, .. } = g.level.wave {
                    if len >= g.level.frame_seconds() {
                        fits.push((carrier_fit(&p, carriers), bw, i, g));
                    }
                }
            }
        }
        fits.sort_by(|a, b| b.0.total_cmp(&a.0));
        // records that share a carrier set tie; keep every record of the four best sets
        let cut = fits.iter().map(|f| f.0).fold(Vec::<f64>::new(), |mut v, x| {
            if v.last().map_or(true, |&l| (l - x).abs() > 1e-9) {
                v.push(x);
            }
            v
        });
        let least = cut.get(3).or(cut.last()).copied().unwrap_or(0.0);
        for &(fit, bw, i, g) in &fits {
            if fit < least {
                continue;
            }
            {
                let flen = g.level.frame_seconds();
                let front = psk::Front::new(s, g, &tables.psk, 0.0);
                if let Some((score, m0, _)) = front.find(0.0, len - flen) {
                    cands.push((score, bw, i, g, front.seconds(m0)));
                }
            }
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, bw, i, g, t) in cands.iter().take(2) {
            let flen = g.level.frame_seconds();
            let hit = |d: psk::Decoded| Hit { mode: bw.hz(), level: i as u32 + 1, frame: d.frame, crc_ok: d.crc_ok, score: d.score, offset: d.offset, start: d.start, seconds: flen };
            let Some(d) = psk::try_level(s, g, &tables.psk, (t - 0.01).max(0.0), t + 0.01) else { continue };
            if d.crc_ok {
                let mut out = vec![hit(d)];
                while out[0].start > flen + 0.004 {
                    let prev = out[0].start - flen;
                    match psk::try_level(s, g, &tables.psk, prev - 0.004, prev + 0.004) {
                        Some(d) if d.crc_ok => out.insert(0, hit(d)),
                        _ => break,
                    }
                }
                loop {
                    let next = out.last().map_or(0.0, |h| h.start + h.seconds);
                    if next + flen > len {
                        break;
                    }
                    match psk::try_level(s, g, &tables.psk, next - 0.004, next + 0.004) {
                        Some(d) if d.crc_ok => out.push(hit(d)),
                        _ => break,
                    }
                }
                return out;
            }
            if best.as_ref().map_or(true, |b| d.score > b.score) {
                best = Some(hit(d));
            }
        }
    }
    best.into_iter().collect()
}

fn try_ofdm(s: &[f64], tables: &Tables, modes: &[Bandwidth]) -> Vec<Hit> {
    let mut best: Option<Hit> = None;
    for &bw in modes {
        let mut front: Option<ofdm::Front> = None;
        let mut cands: Vec<(f64, usize, &GridTables, usize)> = Vec::new();
        for (i, g) in grids(tables, bw, false) {
            // record 17 of the 2300 Hz mode uses the narrow grid: it needs its own front end
            if !front.as_ref().is_some_and(|f| f.fits(g)) {
                front = ofdm::Front::of(s, g);
            }
            let Some(f) = &front else { continue };
            if let Some((score, q0)) = f.find(g, &tables.ofdm) {
                cands.push((score, i, g, q0));
            }
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, i, g, q0) in cands.iter().take(4) {
            let Some(f) = ofdm::Front::of(s, g) else { continue };
            let flen = g.level.frame_seconds();
            let hit = |d: ofdm::Decoded| Hit { mode: bw.hz(), level: i as u32 + 1, frame: d.frame, crc_ok: d.crc_ok, score: d.score, offset: d.offset, start: d.start, seconds: flen };
            let d = f.decode(g, &tables.ofdm, q0);
            if d.crc_ok {
                let mut out = vec![hit(d)];
                let rows = (g.level.spf - 1) as usize;
                let mut q = q0 + rows;
                while f.has_room(g, q) {
                    let d = f.decode(g, &tables.ofdm, q);
                    if !d.crc_ok {
                        break;
                    }
                    out.push(hit(d));
                    q += rows;
                }
                return out;
            }
            if best.as_ref().map_or(true, |b| d.score > b.score) {
                best = Some(hit(d));
            }
        }
    }
    best.into_iter().collect()
}

/// Decode one transmission: work out the kind of signal, then try the levels of that kind in
/// every bandwidth mode. If nothing decodes, the other kinds get a try as well. Returns every
/// frame of the transmission, or the best failed attempt.
fn decode_segment(s: &[f64], tables: &Tables) -> Vec<Hit> {
    let modes = modes_in_order(s);
    let order = match kind_of(s) {
        Kind::Fsk => [Kind::Fsk, Kind::Psk, Kind::Ofdm],
        Kind::Psk => [Kind::Psk, Kind::Ofdm, Kind::Fsk],
        Kind::Ofdm => [Kind::Ofdm, Kind::Psk, Kind::Fsk],
    };
    let mut failed: Option<Hit> = None;
    for kind in order {
        let hits = match kind {
            Kind::Fsk => try_fsk(s, tables, &modes),
            Kind::Psk => try_psk(s, tables, &modes),
            Kind::Ofdm => try_ofdm(s, tables, &modes),
        };
        if hits.first().is_some_and(|h| h.crc_ok) {
            return hits;
        }
        if failed.is_none() {
            failed = hits.into_iter().next();
        }
    }
    failed.into_iter().collect()
}

/// Every burst of at least 0.25 s as (start, end, level in dB), cut at gaps of 0.08 s or more.
fn bursts(x: &[f64]) -> (Vec<(f64, f64)>, Vec<f64>) {
    let n = x.len();
    if n < RATE {
        return (Vec::new(), Vec::new());
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
    let list = out.into_iter().filter(|(a, b)| b - a >= 25).map(|(a, b)| (a as f64 * 0.01, b as f64 * 0.01)).collect();
    (list, db)
}

/// Level of the stretch from `a` to `b` seconds: the mean of its louder half, in dB.
fn level_of(db: &[f64], a: f64, b: f64) -> f64 {
    let (i, j) = (((a * 100.0) as usize).min(db.len()), ((b * 100.0) as usize).min(db.len()));
    if j <= i {
        return f64::NEG_INFINITY;
    }
    let mut v = db[i..j].to_vec();
    v.sort_by(|p, q| q.total_cmp(p));
    let top = &v[..v.len().div_ceil(2)];
    top.iter().sum::<f64>() / top.len() as f64
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
        let follows = matches!(ev, Event::Frame(f) if f.part > 0); // sent right behind the previous frame
        if i > 0 && !follows {
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
    scan_with(x, tables, &[])
}

/// As `scan`, with callsigns the listener already knows: connect requests to them, and BREAK or
/// IDLE of links they started, are then named even if no header frame was heard.
pub fn scan_with(x: &[f64], tables: &Tables, calls: &[String]) -> Scan {
    scan_live(x, tables, &mut Memory { calls: calls.to_vec(), ..Memory::default() })
}

/// What is carried from one stretch of audio to the next when a receiver is followed live:
/// callsigns heard, keys found, and control bursts not yet explained.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub calls: Vec<String>,
    pub keys: Vec<crate::ctrl::Key>,
    pub pending: Vec<crate::ctrl::Measured>,
}

/// As `scan`, for one stretch of a live signal: uses and updates what earlier stretches gave.
pub fn scan_live(x: &[f64], tables: &Tables, memory: &mut Memory) -> Scan {
    let mut events = Vec::new();
    let (list, db) = bursts(x);
    let push = |events: &mut Vec<Event>, lo: usize, a: usize, b: usize, level_db: f64, hits: Vec<Hit>| {
        for (part, h) in hits.into_iter().enumerate() {
            let frame_start = lo as f64 / RATE as f64 + h.start;
            events.push(Event::Frame(Found {
                time: a as f64 / RATE as f64,
                seconds: (b - a) as f64 / RATE as f64,
                mode: h.mode,
                level: h.level,
                frame: h.frame,
                crc_ok: h.crc_ok,
                score: h.score,
                offset: h.offset,
                level_db,
                station: 0,
                frame_start,
                frame_end: frame_start + h.seconds,
                part,
            }));
        }
    };
    // first pass: every unbroken stretch of signal long enough to be a frame. Stations take turns
    // with pauses of a fraction of a second, so each stretch is one transmission.
    for &(ta, tb) in list.iter().filter(|(a, b)| b - a >= 2.5) {
        let (a, b) = ((ta * RATE as f64) as usize, ((tb * RATE as f64) as usize).min(x.len()));
        let lo = a.saturating_sub(RATE * 3 / 10);
        let hi = (b + RATE * 3 / 10).min(x.len());
        let hits = decode_segment(&x[lo..hi], tables);
        if hits.first().is_some_and(|h| h.crc_ok) {
            push(&mut events, lo, a, b, level_of(&db, ta, tb), hits);
        }
    }
    // second pass: what is left once fades shorter than 0.45 s are bridged, for frames with a
    // faded start, end or middle
    for (a, b, level_db) in segments(x) {
        let mut free = vec![(a, b)];
        for e in &events {
            let (s, t) = ((e.time() * RATE as f64) as usize, (e.end() * RATE as f64) as usize);
            free = free.into_iter().flat_map(|(p, q)| if s >= q || t <= p { vec![(p, q)] } else { vec![(p, s.max(p)), (t.min(q), q)] }).filter(|(p, q)| q > p).collect();
        }
        for (a, b) in free.into_iter().filter(|(p, q)| q - p >= 3 * RATE) {
            // a run of short bursts with pauses between them is control traffic, not a faded frame
            let (ta, tb) = (a as f64 / RATE as f64, b as f64 / RATE as f64);
            if list.iter().filter(|(s, e)| e.min(tb) - s.max(ta) > 0.5 * (e - s)).all(|(s, e)| e - s <= 1.8) {
                continue;
            }
            // 2 s margin before and 0.7 s after: a faded start or end is common off air
            let lo = a.saturating_sub(2 * RATE);
            let hi = (b + RATE * 7 / 10).min(x.len());
            let hits = decode_segment(&x[lo..hi], tables);
            push(&mut events, lo, a, b, level_db, hits);
        }
    }
    events.sort_by(|p, q| p.time().total_cmp(&q.time()));
    // short bursts that are not part of a frame: acknowledgements and other control signals
    let spans: Vec<(f64, f64)> = events.iter().map(|e| (e.time() - 0.1, e.end() + 0.05)).collect();
    // anything more than 25 dB below the strongest frame is noise, not a station
    let floor = events.iter().map(|e| e.level()).fold(f64::NEG_INFINITY, f64::max) - 25.0;
    for (a, b) in list {
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
        let level_db = level_of(&db, a, b);
        if b - a >= 0.25 && b - a < 3.0 && level_db >= floor {
            events.push(Event::Control(Control { time: a, seconds: b - a, level_db, station: 0, format: None, code: None, other: None, announces: None }));
        }
    }
    events.sort_by(|p, q| p.time().total_cmp(&q.time()));
    assign_stations(&mut events);
    let keys = decode_control(x, &mut events, memory);
    Scan { events, keys }
}

/// The result of a scan.
pub struct Scan {
    pub events: Vec<Event>,
    /// for each station whose control bursts could be resolved: (station, key)
    pub keys: Vec<(usize, crate::ctrl::Key)>,
}

/// Measure every control burst, find each station's key from its own bursts and name the codes.
/// Bursts the key does not explain are tested as connect requests, BREAK, IDLE and the fixed codes.
fn decode_control(x: &[f64], events: &mut [Event], memory: &mut Memory) -> Vec<(usize, crate::ctrl::Key)> {
    use crate::ctrl;
    let good: Vec<&Found> = events.iter().filter_map(|e| if let Event::Frame(f) = e { f.crc_ok.then_some(f) } else { None }).collect();
    let mut offsets: Vec<f64> = good.iter().map(|f| f.offset).collect();
    offsets.sort_by(|a, b| a.total_cmp(b));
    let offset = offsets.get(offsets.len() / 2).copied().unwrap_or(0.0);
    // the link's bandwidth mode: the one most frames were decoded in
    let bw = Bandwidth::ALL.into_iter().max_by_key(|b| good.iter().filter(|f| f.mode == b.hz()).count()).unwrap_or(Bandwidth::Hz2300);
    let bw = if good.is_empty() { Bandwidth::Hz2300 } else { bw };
    // callsigns: the ones given or heard before, and those in this stretch's header frames
    for h in good.iter().filter_map(|f| crate::frame::header(&f.frame)) {
        if !memory.calls.contains(&h.call) {
            memory.calls.push(h.call);
        }
    }
    let heard: Vec<(String, u32)> = memory.calls.iter().map(|c| (c.to_uppercase(), crate::frame::call_hash(&c.to_uppercase()))).collect();
    let frame_ends: Vec<f64> = events.iter().filter_map(|e| if let Event::Frame(f) = e { Some(f.frame_end) } else { None }).collect();
    let mut measured: Vec<(usize, usize, ctrl::Measured)> = Vec::new(); // (event index, station, tones)
    for (i, e) in events.iter_mut().enumerate() {
        if let Event::Control(c) = e {
            if let Some(m) = ctrl::measure(x, c.time, c.seconds, offset) {
                c.format = Some(m.format);
                measured.push((i, c.station, m));
            }
        }
    }
    // keys: the ones known already, and any that the unexplained bursts (old and new) agree on
    let explained = |m: &ctrl::Measured, keys: &[ctrl::Key]| keys.iter().filter_map(|&k| ctrl::identify(m, k, bw).map(|c| (c, k))).max_by_key(|x| x.0 .1);
    let mut open: Vec<ctrl::Measured> = memory.pending.clone();
    open.extend(measured.iter().map(|m| m.2.clone()).filter(|m| explained(m, &memory.keys).is_none()));
    for key in ctrl::resolve_all(&open, bw) {
        if !memory.keys.contains(&key) {
            memory.keys.push(key);
        }
    }
    let mut keys: Vec<(usize, ctrl::Key)> = Vec::new();
    for (i, st, m) in &measured {
        if let Some((code, key)) = explained(m, &memory.keys) {
            if let Event::Control(c) = &mut events[*i] {
                c.code = Some(code);
            }
            if !keys.iter().any(|k| k.1 == key) {
                keys.push((*st, key));
            }
        }
    }
    // callsign hashes worth testing: the ones heard, and both readings of every key
    let mut known: Vec<u32> = heard.iter().map(|h| h.1).collect();
    for key in &memory.keys {
        let (a, b) = key.hashes();
        known.extend([a, b]);
    }
    known.sort_unstable();
    known.dedup();
    let name = |h: u32| heard.iter().find(|c| c.1 == h).map_or(format!("the station with callsign hash {h:06x}"), |c| c.0.clone());
    let mut seen_requests: Vec<Vec<ctrl::Tones>> = Vec::new();
    for (i, _, m) in &measured {
        let Event::Control(c) = &mut events[*i] else { continue };
        if c.code.is_some() || m.purity < 0.5 {
            continue;
        }
        if let Some((what, hits, total)) = ctrl::identify_other(m, &known, bw) {
            let text = match what {
                ctrl::Other::Request { called, bw } => format!("CONNECT REQUEST to {}, {} Hz mode", name(called), bw.hz()),
                ctrl::Other::Connect { code, caller, side } => {
                    format!("{}, sent by the station that {} (link started by {})", ctrl::code_name(code), if side == 0 { "made the call" } else { "answered" }, name(caller))
                }
                ctrl::Other::Fixed { code } => {
                    c.announces = Some(code);
                    ctrl::code_name(code).to_string()
                }
            };
            c.other = Some(format!("{text} ({hits} of {total} symbols)"));
            continue;
        }
        // a long burst that does not follow a frame and is heard more than once is a call
        let after_frame = frame_ends.iter().any(|&e| c.time > e - 0.1 && c.time < e + 1.0);
        if m.format == 0 && !after_frame {
            let repeated = measured.iter().filter(|o| o.2.format == 0 && o.2.tones == m.tones).count() >= 2;
            if repeated {
                if !seen_requests.contains(&m.tones) {
                    seen_requests.push(m.tones.clone());
                }
                if let Some((called, req)) = ctrl::request_target(m) {
                    c.other = Some(format!("repeated call, read as a CONNECT REQUEST to {}, {} Hz mode", name(called), req.hz()));
                }
            }
        }
    }
    // single-tone bursts nothing explains yet wait for a second one from the same station
    memory.pending.retain(|m| explained(m, &memory.keys).is_none());
    for (i, _, m) in &measured {
        let Event::Control(c) = &events[*i] else { continue };
        if c.code.is_none() && c.other.is_none() && m.format < 2 && m.purity > 0.5 && !memory.pending.iter().any(|p| p.format == m.format && p.tones == m.tones) {
            memory.pending.push(m.clone());
        }
    }
    let extra = memory.pending.len().saturating_sub(12);
    memory.pending.drain(..extra);
    keys
}

/// The frames among the events.
pub fn frames(events: &[Event]) -> Vec<&Found> {
    events.iter().filter_map(|e| if let Event::Frame(f) = e { Some(f) } else { None }).collect()
}

/// The payloads of one station in time order, as runs of (compressed flag, frame payloads).
/// Exact repeats of the previous frame (a retransmission after a missed acknowledgement) are
/// dropped.
pub fn stream_of(found: &[&Found], station: usize) -> Vec<(bool, Vec<Vec<u8>>)> {
    let mut out: Vec<(bool, Vec<Vec<u8>>)> = Vec::new();
    let mut prev: Option<&[u8]> = None;
    for f in found.iter().filter(|f| f.crc_ok && f.station == station) {
        if prev == Some(&f.frame[..]) {
            continue;
        }
        prev = Some(&f.frame[..]);
        if crate::frame::header(&f.frame).is_some() {
            continue; // the header frame carries the callsign, not application data
        }
        let Some(p) = crate::frame::parse(&f.frame) else { continue };
        let Some(data) = p.data else { continue };
        if data.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.0 == p.compressed => last.1.push(data),
            _ => out.push((p.compressed, vec![data])),
        }
    }
    out
}

/// Undo the modem's compression on a run of compressed frame payloads. Tries the "files" mode on
/// the joined stream, then the text mode frame by frame. Returns the bytes and the mode's name.
pub fn unpack(frames: &[Vec<u8>]) -> Option<(Vec<u8>, &'static str)> {
    let joined: Vec<u8> = frames.concat();
    if let Some((out, clean)) = crate::frame::unpack_files(&joined) {
        if clean || !out.is_empty() {
            return Some((out, "files compression"));
        }
    }
    let mut out = Vec::new();
    for f in frames {
        out.extend(crate::textcomp::decompress(f)?);
    }
    crate::frame::is_text(&out).then_some((out, "text compression"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl(time: f64, seconds: f64, level_db: f64) -> Event {
        Event::Control(Control { time, seconds, level_db, station: 9, format: None, code: None, other: None, announces: None })
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

    /// A whole link from generated audio: two calls, the reply, header frames, data,
    /// acknowledgements, IDLE, BREAK and GO AHEAD. Every burst must be named and the callsigns shown.
    #[test]
    fn link_start_is_read() {
        use crate::ctrl::{self, Key};
        use crate::frame::call_hash;
        use crate::synth;
        let tables = Tables::new();
        let bw = Bandwidth::Hz2300;
        let (caller, called) = (call_hash("N0CALL"), call_hash("N1CALL"));
        let key = |side: usize| Key { base: called + 10 + ctrl::SIDE[side], steps: [0, 1, 2] };
        let code = |side: usize, f: usize, c: u32| ctrl::session_book(key(side), f, bw)[c as usize].clone();
        let unhex = |h: &str| (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect::<Vec<u8>>();
        let frame = |level: usize, f: &[u8], amp: f64| {
            let mut n = 5u32;
            let mut rand = |m: u32| {
                n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (n >> 16) % m
            };
            let x = synth::transmission(&bw.levels()[level - 1], &tables.mode(bw)[level - 1], &[f], &mut rand);
            let top = x.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            x.iter().map(|v| v / top * amp).collect::<Vec<f64>>()
        };
        let burst = |f: usize, lead: usize, tones: Vec<ctrl::Tones>, amp: f64| ctrl::audio(f, lead, &tones).iter().map(|v| v * amp).collect::<Vec<f64>>();
        let header_caller = unhex("39b0c130c00000800183df5c"); // N0CALL, level 1
        let mut header_called = crate::frame::pack_call("N1CALL").to_vec();
        header_called.extend([0x00, 0x80, 0x02, 0x83]);
        let crc = crate::crc::crc16(&header_called);
        header_called.extend(crc.to_be_bytes());
        assert_eq!(crate::frame::header(&header_called).map(|h| h.call), Some("N1CALL".to_string()));
        // (pause before in seconds, audio); the caller is louder than the called station
        let (a, b) = (0.5, 0.2);
        let parts: Vec<(f64, Vec<f64>)> = vec![
            (1.0, burst(0, 10, ctrl::request_code(called, 0, bw), a)),
            (3.0, burst(0, 1, ctrl::request_code(called, 0, bw), a)),
            (0.25, burst(1, 8, code(1, 1, 16), b)),
            (0.25, frame(1, &header_caller, a)),
            (0.25, burst(0, 1, code(1, 0, 1), b)),
            (0.25, frame(1, &header_caller, a)),
            (0.25, burst(1, 1, code(1, 1, 7), b)),
            (0.25, burst(0, 1, ctrl::connect_codes(caller, 0, 0, bw)[1].clone(), a)),
            (0.25, burst(0, 1, ctrl::connect_codes(caller, 1, 0, bw)[0].clone(), b)),
            (0.25, burst(0, 1, code(0, 0, 14), a)),
            (0.25, frame(1, &header_called, b)),
            (0.25, burst(0, 1, code(0, 0, 1), a)),
            (0.25, frame(1, &header_called, b)),
            (0.25, burst(1, 1, code(0, 1, 5), a)),
        ];
        let mut x: Vec<f64> = Vec::new();
        for (pause, audio) in &parts {
            x.extend(std::iter::repeat(0.0).take((pause * RATE as f64) as usize));
            x.extend(audio);
        }
        x.extend(std::iter::repeat(0.0).take(RATE));
        let x = synth::on_air(&x, 0.5, 0.0, 3.0, 0.01, 77);
        let result = scan(&x, &tables);
        let lines: Vec<String> = result
            .events
            .iter()
            .map(|e| match e {
                Event::Frame(f) => format!("{} frame {}", f.station, crate::frame::header(&f.frame).map_or("?".to_string(), |h| h.call)),
                Event::Control(c) => format!("{} {}", c.station, c.other.clone().unwrap_or_else(|| c.code.map_or("unknown".to_string(), |k| format!("code {}", k.0)))),
            })
            .collect();
        let want = [
            "0 CONNECT REQUEST to N1CALL, 2300 Hz mode",
            "0 CONNECT REQUEST to N1CALL, 2300 Hz mode",
            "1 code 16",
            "0 frame N0CALL",
            "1 code 1",
            "0 frame N0CALL",
            "1 code 7",
            "0 IDLE, sent by the station that made the call (link started by N0CALL)",
            "1 BREAK, sent by the station that answered (link started by N0CALL)",
            "0 code 14",
            "1 frame N1CALL",
            "0 code 1",
            "1 frame N1CALL",
            "0 code 5",
        ];
        assert_eq!(lines.len(), want.len(), "{lines:#?}");
        for (got, want) in lines.iter().zip(want) {
            assert!(got.starts_with(want), "got {got:?}, want {want:?}\nall: {lines:#?}");
        }
        assert_eq!(result.keys.len(), 2);
    }
}
