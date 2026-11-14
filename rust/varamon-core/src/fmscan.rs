//! Scanner for VARA FM: finds each transmission in a recording, tells tone bursts from OFDM
//! frames, works out the record, the mode and the key, decodes, and checks the CRC.
//!
//! The data frames of a link are scrambled with a table drawn from the called station's callsign.
//! The callsign comes from the connect frame when it was heard, or from `--call`. When neither is
//! there, the same number is found from the link's tone bursts, whose codes are drawn from it too.

use std::collections::HashMap;

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::audio::RATE;
use crate::fm::{self, Mode, Tables, Tones, ROWS};
use crate::frame::call_hash;
use crate::levels::soft_bits;
use crate::ofdm::Front;

type Phasors = Vec<[Complex64; ROWS]>;

/// Tables of both modes and the scrambling tables of the keys tried so far.
pub struct FmTables {
    pub narrow: Vec<Tables>,
    pub wide: Vec<Tables>,
    phasors: HashMap<Option<u32>, Phasors>,
}

impl FmTables {
    pub fn new() -> Self {
        Self { narrow: fm::tables(Mode::Narrow), wide: fm::tables(Mode::Wide), phasors: HashMap::new() }
    }

    fn tables(&self, mode: Mode) -> &[Tables] {
        match mode {
            Mode::Narrow => &self.narrow,
            Mode::Wide => &self.wide,
        }
    }

    fn phasors(&mut self, key: Option<u32>) -> &Phasors {
        self.phasors.entry(key).or_insert_with(|| fm::phasors(key))
    }
}

impl Default for FmTables {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct FmFrame {
    pub time: f64,
    pub seconds: f64,
    pub mode: Mode,
    pub rec: usize,
    pub frame: Vec<u8>,
    pub crc_ok: bool,
    /// how well the markers or pilots agree, 0..1
    pub score: f64,
    pub station: usize,
    /// 0 for the first frame of a transmission, 1 for the next, ...
    pub part: usize,
    /// where the transmission began (its lead-in comes before the frame)
    pub began: f64,
}

#[derive(Clone, Debug)]
pub struct ToneBurst {
    pub time: f64,
    pub seconds: f64,
    /// the tones of each symbol, lowest first
    pub rows: Vec<Tones>,
    /// (code number 1-24, symbols that match out of 10)
    pub code: Option<(usize, usize)>,
    pub station: usize,
}

#[derive(Clone, Debug)]
pub enum FmEvent {
    Frame(FmFrame),
    Tone(ToneBurst),
}

impl FmEvent {
    pub fn time(&self) -> f64 {
        match self {
            FmEvent::Frame(f) => f.time,
            FmEvent::Tone(t) => t.time,
        }
    }
    fn end(&self) -> f64 {
        match self {
            FmEvent::Frame(f) => f.time + f.seconds,
            FmEvent::Tone(t) => t.time + t.seconds,
        }
    }
    fn set_station(&mut self, s: usize) {
        match self {
            FmEvent::Frame(f) => f.station = s,
            FmEvent::Tone(t) => t.station = s,
        }
    }
}

/// What is known about the link.
#[derive(Clone, Debug, Default)]
pub struct Link {
    /// callsigns given or heard
    pub calls: Vec<String>,
    /// hash of the called station's callsign: the key of the data frames and tone codes 1-22
    pub called: Option<u32>,
    /// hash of the calling station's callsign: tone codes 23-24
    pub caller: Option<u32>,
    pub mode: Option<Mode>,
    pub relayed: bool,
    /// how the key was found
    pub key_from: &'static str,
    /// keys that tone bursts allow, best first, while the data frames have not settled it
    pub candidates: Vec<u32>,
}

impl Link {
    fn keys(&self) -> Vec<u32> {
        let mut k: Vec<u32> = self.called.into_iter().collect();
        for &c in &self.candidates {
            if !k.contains(&c) {
                k.push(c);
            }
        }
        for c in &self.calls {
            let h = call_hash(c);
            if !k.contains(&h) {
                k.push(h);
            }
        }
        k
    }
}

pub struct FmScan {
    pub events: Vec<FmEvent>,
    pub link: Link,
}

/// Transmissions as (first sample, last sample): signal between 150 Hz and 6 kHz, cut at pauses of 10 ms.
fn bursts(x: &[f64]) -> Vec<(usize, usize)> {
    let n = x.len();
    if n < RATE / 2 {
        return Vec::new();
    }
    let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(n).process(&mut buf);
    for (i, v) in buf.iter_mut().enumerate() {
        let f = i.min(n - i) as f64 * RATE as f64 / n as f64;
        if !(150.0..=6000.0).contains(&f) {
            *v = Complex64::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    const BLOCK: usize = 240;
    let db: Vec<f64> = buf.chunks_exact(BLOCK).map(|c| 10.0 * (c.iter().map(|v| (v.re / n as f64).powi(2)).sum::<f64>() / BLOCK as f64).max(1e-20).log10()).collect();
    let mut sorted = db.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let top = sorted[sorted.len() * 99 / 100];
    let threshold = (sorted[sorted.len() / 10] + 10.0).max(top - 30.0);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, &v) in db.iter().chain(std::iter::once(&f64::MIN)).enumerate() {
        match (v > threshold, start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                match runs.last_mut() {
                    Some(last) if a - last.1 <= 2 => last.1 = i,
                    _ => runs.push((a, i)),
                }
                start = None;
            }
            _ => {}
        }
    }
    runs.into_iter().filter(|(a, b)| b - a >= 30).map(|(a, b)| (a * BLOCK, b * BLOCK)).collect()
}

/// The three strongest tones of each 1024-sample symbol, at the best symbol timing, and how much of
/// the energy they hold (0..1).
pub fn tones(s: &[f64]) -> (f64, Vec<Tones>) {
    const N: usize = fm::TONE_SYMBOL;
    if s.len() < 2 * N {
        return (0.0, Vec::new());
    }
    let fft = FftPlanner::new().plan_fft_forward(N);
    let mut buf = vec![Complex64::new(0.0, 0.0); N];
    let mut measure = |off: usize| -> (f64, Vec<(f64, Tones)>) {
        let (mut share, mut weight, mut rows) = (0.0, 1e-30, Vec::new());
        let mut at = off;
        while at + N <= s.len() {
            for (b, &v) in buf.iter_mut().zip(&s[at..at + N]) {
                *b = Complex64::new(v, 0.0);
            }
            fft.process(&mut buf);
            let p: Vec<f64> = buf[..130].iter().map(|v| v.norm_sqr()).collect();
            let all: f64 = p.iter().sum::<f64>() + 1e-30;
            let mut idx: Vec<usize> = (0..p.len()).collect();
            idx.sort_by(|&i, &j| p[j].total_cmp(&p[i]));
            let mut t = [idx[0] as u32, idx[1] as u32, idx[2] as u32];
            t.sort_unstable();
            share += (p[idx[0]] + p[idx[1]] + p[idx[2]]) / all * all;
            weight += all;
            rows.push((all, t));
            at += N;
        }
        (share / weight, rows)
    };
    let mut best = (0.0, Vec::new(), 0);
    for off in (0..N).step_by(32) {
        let (sh, rows) = measure(off);
        if sh > best.0 {
            best = (sh, rows, off);
        }
    }
    let centre = best.2;
    for off in (centre.saturating_sub(30)..(centre + 31).min(N)).step_by(2) {
        let (sh, rows) = measure(off);
        if sh > best.0 {
            best = (sh, rows, off);
        }
    }
    // symbols at the edges that are partly silence are not part of the burst
    let mut e: Vec<f64> = best.1.iter().map(|r| r.0).collect();
    e.sort_by(|a, b| a.total_cmp(b));
    let typical = e[e.len() / 2];
    let keep = |r: &&(f64, Tones)| r.0 > 0.5 * typical;
    let first = best.1.iter().position(|r| keep(&r)).unwrap_or(0);
    let last = best.1.iter().rposition(|r| keep(&r)).unwrap_or(0);
    (best.0, best.1[first..=last].iter().map(|r| r.1).collect())
}

fn sorted(t: &Tones) -> Tones {
    let mut t = *t;
    t.sort_unstable();
    t
}

/// The code of a tone burst in a link's codebook: (code number, symbols matched).
fn match_code(rows: &[Tones], book: &[Vec<Tones>]) -> Option<(usize, usize)> {
    if rows.len() < fm::CODE_ROWS {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    // the burst may end with a partial symbol: try the last ten rows and the ten before the last
    for end in [rows.len(), rows.len() - 1] {
        if end < fm::CODE_ROWS {
            continue;
        }
        let tail = &rows[end - fm::CODE_ROWS..end];
        for (i, code) in book.iter().enumerate() {
            let hits = code.iter().zip(tail).filter(|(c, t)| sorted(c) == **t).count();
            if best.is_none_or(|b| hits > b.1) {
                best = Some((i + 1, hits));
            }
        }
    }
    best.filter(|b| b.1 >= 7)
}

/// Find the callsign hashes that could be behind a tone burst whose link is not known: every
/// generator state is tried against the burst's rows, and each state found is walked back to the
/// seeds it can have started from. One burst fits several seeds; the data frames decide (only the
/// true key makes their CRC come out right). Returns (hash, true if it is the caller's, draws walked
/// back), nearest first: code 1, the most common acknowledgement, sits right at its seed.
pub fn blind_keys(rows: &[Tones]) -> Vec<(u32, bool, usize)> {
    if rows.len() < fm::CODE_ROWS {
        return Vec::new();
    }
    let code: Vec<Tones> = rows[rows.len() - fm::CODE_ROWS..].to_vec();
    let first = code[0];
    let next = |s: &mut u32| -> f32 {
        *s = s.wrapping_mul(0x00FD_43FD).wrapping_add(0x00C3_9EC3) & 0x00FF_FFFF;
        *s as f32 / 16_777_216.0
    };
    let row = |s: &mut u32| -> Tones {
        let mut bases = [u32::MAX; 3];
        let mut t = [0u32; 3];
        for i in 0..3 {
            let b = loop {
                let b = (next(s) * 7.0) as u32 * 7 + 13;
                if !bases[..i].contains(&b) {
                    break b;
                }
            };
            bases[i] = b;
            t[i] = b + (next(s) * 7.0) as u32;
        }
        t.sort_unstable();
        t
    };
    let mut starts = Vec::new();
    for s0 in 0..(1u32 << 24) {
        let mut s = s0;
        if row(&mut s) != first {
            continue;
        }
        let hits = 1 + code[1..].iter().filter(|want| row(&mut s) == **want).count();
        if hits >= 8 {
            starts.push(s0);
        }
    }
    let index = crate::ctrl::seed_index();
    let fits = |c: &Vec<Tones>| c.iter().zip(&code).filter(|(a, b)| sorted(a) == **b).count() >= 8;
    let mut out: Vec<(u32, bool, usize)> = Vec::new();
    for s0 in starts {
        let mut r = crate::vb::VbRnd::from_state(s0);
        // a code is at most 22 codes into its stream, a code about 70 draws
        for back in 0..1600 {
            for x in index.get(r.state()) {
                if let Some(called) = x.checked_sub(10).filter(|&h| h < 1 << 24) {
                    if fm::codebook(called, 0)[..22].iter().any(fits) && !out.iter().any(|o| o.0 == called && !o.1) {
                        out.push((called, false, back));
                    }
                }
                if let Some(caller) = x.checked_sub(199).filter(|&h| h < 1 << 24) {
                    if back < 160 && fm::codebook(0, caller)[22..].iter().any(fits) && !out.iter().any(|o| o.0 == caller && o.1) {
                        out.push((caller, true, back));
                    }
                }
            }
            r.retreat();
        }
    }
    out.sort_by_key(|o| o.2);
    out
}

/// How well a frame of these tables fits with its first symbol at symbol `q0`.
fn score(front: &Front, t: &Tables, ph: &Phasors, q0: usize) -> f64 {
    let l = &t.level;
    let cell = |r: usize, c: u32| front.cell_at(q0 + r, l.bin(c)) * ph[l.bin(c)][r];
    let (mut num, mut den) = (0.0, 1e-12);
    if l.is_qam() {
        for c in 0..l.nc {
            let mut acc = Complex64::new(0.0, 0.0);
            for r in 0..l.rows() {
                if fm::qam_pilot(l.bin(c), r + 2) {
                    let v = cell(r, c);
                    acc += v;
                    den += v.norm();
                }
            }
            num += acc.norm();
        }
    } else {
        let mut acc = Complex64::new(0.0, 0.0);
        for c in 0..l.nc {
            for r in 1..l.rows() {
                if t.marks[c as usize][r] {
                    let d = cell(r, c) * cell(r - 1, c).conj();
                    acc += d;
                    den += d.norm();
                }
            }
        }
        num = acc.norm();
    }
    num / den
}

/// Decode the frame of these tables whose first symbol is symbol `q0`.
fn decode_at(front: &Front, t: &Tables, ph: &Phasors, q0: usize) -> Vec<u8> {
    let l = &t.level;
    let cst = fm::constellation(l.bps);
    let mut points: Vec<Complex64> = Vec::with_capacity((l.n() / l.bps) as usize);
    for c in 0..l.nc {
        let b = l.bin(c);
        let col: Vec<Complex64> = (0..l.rows()).map(|r| front.cell_at(q0 + r, b) * ph[b][r]).collect();
        if l.is_qam() {
            let pilots: Vec<usize> = (0..l.rows()).filter(|&r| fm::qam_pilot(b, r + 2)).collect();
            let mut seg = 0;
            for r in 0..l.rows() {
                if fm::qam_pilot(b, r + 2) || t.marks[c as usize][r] {
                    continue;
                }
                while seg + 1 < pilots.len() && pilots[seg + 1] <= r {
                    seg += 1;
                }
                let h = if r <= pilots[0] {
                    col[pilots[0]]
                } else if seg + 1 >= pilots.len() {
                    col[pilots[seg]]
                } else {
                    let (a, z) = (pilots[seg], pilots[seg + 1]);
                    let f = (r - a) as f64 / (z - a) as f64;
                    col[a] * (1.0 - f) + col[z] * f
                };
                points.push(col[r] / h);
            }
        } else {
            for r in 1..l.rows() {
                if !t.marks[c as usize][r] {
                    points.push(col[r] * col[r - 1].conj() / (col[r - 1].norm_sqr() + 1e-18));
                }
            }
        }
    }
    let spread = points.iter().map(|y| cst.iter().map(|p| (y - p).norm_sqr()).fold(f64::MAX, f64::min)).sum::<f64>() / points.len().max(1) as f64;
    let mut soft = Vec::with_capacity(l.n() as usize);
    for &y in &points {
        soft_bits(y, &cst, l.bps, spread.max(0.002), &mut soft);
    }
    fm::decode(t, &soft)
}

struct Hit {
    key: Option<u32>,
    mode: Mode,
    rec: usize,
    frame: Vec<u8>,
    crc_ok: bool,
    score: f64,
    start: f64,
    seconds: f64,
}

/// Decode an OFDM transmission: try every record of the possible modes with every known key, decode
/// the best fits, and follow a decoded frame to the frames sent right before and after it.
fn decode_ofdm(s: &[f64], tabs: &mut FmTables, link: &Link) -> Vec<Hit> {
    let Some(front) = Front::new(s, 0, fm::BINS) else { return Vec::new() };
    let modes: Vec<Mode> = link.mode.map_or(Mode::BOTH.to_vec(), |m| vec![m]);
    let keys = link.keys();
    // mean power of every bin: a record whose carriers miss part of the signal, or cover empty bins,
    // is not the one sent (the QAM levels share their pilot pattern, so that alone cannot tell them apart)
    let power: Vec<f64> = (0..fm::BINS).map(|b| (0..front.symbols()).map(|q| front.cell_at(q, b).norm_sqr()).sum::<f64>()).collect();
    let total: f64 = power[3..125].iter().sum::<f64>() + 1e-30;
    let mean_in = |l: &fm::Level| (0..l.nc).map(|c| power[l.bin(c)]).sum::<f64>() / l.nc as f64;
    let fit = |l: &fm::Level| {
        let inside = (0..l.nc).map(|c| power[l.bin(c)]).sum::<f64>() / total;
        // the carriers between a sparse record's carriers must be empty
        let gaps = if l.step > 1 { (l.first as usize..l.bin(l.nc - 1)).filter(|b| (b - l.first as usize) % l.step as usize != 0).map(|b| power[b]).sum::<f64>() / ((l.nc - 1) * (l.step - 1)) as f64 } else { 0.0 };
        inside.min(1.0) * (1.0 - (gaps / (mean_in(l) + 1e-30)).min(1.0))
    };
    let mut cands: Vec<(f64, Mode, usize, Option<u32>, usize)> = Vec::new();
    for &mode in &modes {
        for rec in 0..fm::RECORDS {
            // records 0-3 are the same in both modes
            if modes.len() == 2 && mode == Mode::Narrow && rec <= 3 {
                continue;
            }
            let rows = mode.records()[rec].rows();
            if front.symbols() < rows {
                continue;
            }
            let ks: Vec<Option<u32>> = if fm::keyed(rec) { keys.iter().map(|&k| Some(k)).collect() } else { vec![None] };
            for key in ks {
                tabs.phasors(key);
                let t = &tabs.tables(mode)[rec];
                let ph = &tabs.phasors[&key];
                let f = fit(&t.level);
                if f < 0.5 {
                    continue;
                }
                let best = (0..=front.symbols() - rows).map(|q0| (score(&front, t, ph, q0), q0)).max_by(|a, b| a.0.total_cmp(&b.0));
                if let Some((sc, q0)) = best {
                    cands.push((sc * f, mode, rec, key, q0));
                }
            }
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    let sym = fm::SYMBOL as f64 / RATE as f64;
    let mut failed: Option<Hit> = None;
    // every candidate close to the best is decoded in turn: records that share carriers and pilots
    // differ only in their constellation and code, which only decoding tells
    let top = cands.first().map_or(0.0, |c| c.0);
    for &(sc, mode, rec, key, q0) in cands.iter().take_while(|c| c.0 >= 0.8 * top).take(16) {
        let t = &tabs.tables(mode)[rec];
        let ph = &tabs.phasors[&key];
        let rows = t.level.rows();
        let at = |q: usize| (front.phase() + q * fm::SYMBOL) as f64 / RATE as f64;
        let hit = |q: usize, frame: Vec<u8>| {
            let ok = crate::crc::frame_ok(&frame);
            Hit { key, mode, rec, frame, crc_ok: ok, score: score(&front, t, ph, q), start: at(q), seconds: rows as f64 * sym }
        };
        let first = hit(q0, decode_at(&front, t, ph, q0));
        if !first.crc_ok {
            if failed.as_ref().is_none_or(|f| sc > f.score) {
                failed = Some(first);
            }
            continue;
        }
        let mut out = vec![first];
        let mut q = q0;
        while q >= rows {
            let h = hit(q - rows, decode_at(&front, t, ph, q - rows));
            if !h.crc_ok {
                break;
            }
            q -= rows;
            out.insert(0, h);
        }
        let mut q = q0 + rows;
        while q + rows <= front.symbols() {
            let h = hit(q, decode_at(&front, t, ph, q));
            if !h.crc_ok {
                break;
            }
            out.push(h);
            q += rows;
        }
        return out;
    }
    failed.into_iter().collect()
}

/// Scan a recording at 48 kHz. `calls` are callsigns the listener knows.
pub fn scan(x: &[f64], tabs: &mut FmTables, calls: &[String]) -> FmScan {
    let debug = std::env::var_os("VARAMON_DEBUG").is_some();
    let mut link = Link { calls: calls.iter().map(|c| c.to_uppercase()).collect(), ..Link::default() };
    let pieces: Vec<(usize, usize)> = bursts(x);
    // first: which pieces are tone bursts, and their tones
    let mut kinds: Vec<Option<Vec<Tones>>> = Vec::new();
    for &(a, b) in &pieces {
        let lo = a.saturating_sub(RATE / 50);
        let hi = (b + RATE / 50).min(x.len());
        let (purity, rows) = tones(&x[lo..hi]);
        kinds.push((purity > 0.6 && rows.len() >= fm::CODE_ROWS).then_some(rows));
    }
    let mut events: Vec<FmEvent> = Vec::new();
    let mut undecoded: Vec<usize> = Vec::new();
    if debug {
        for (i, &(a, b)) in pieces.iter().enumerate() {
            eprintln!("piece {i}: {:.2}-{:.2} s {}", a as f64 / RATE as f64, b as f64 / RATE as f64, if kinds[i].is_some() { "tones" } else { "ofdm" });
        }
    }
    for pass in 0..2 {
        if pass == 1 && (undecoded.is_empty() || (link.called.is_none() && link.candidates.is_empty())) {
            break;
        }
        for (i, &(a, b)) in pieces.iter().enumerate() {
            if kinds[i].is_some() || (pass == 1 && !undecoded.contains(&i)) {
                continue;
            }
            let lo = a.saturating_sub(RATE / 20);
            let hi = (b + RATE / 20).min(x.len());
            let hits = decode_ofdm(&x[lo..hi], tabs, &link);
            if pass == 1 {
                if !hits.first().is_some_and(|h| h.crc_ok) {
                    continue;
                }
                events.retain(|e| !matches!(e, FmEvent::Frame(f) if (f.time - lo as f64 / RATE as f64).abs() < (b - lo) as f64 / RATE as f64 && f.time >= lo as f64 / RATE as f64 && !f.crc_ok));
            }
            let ok = hits.first().is_some_and(|h| h.crc_ok);
            if !ok && pass == 0 {
                undecoded.push(i);
            }
            for (part, h) in hits.into_iter().enumerate() {
                if h.rec == 0 && h.crc_ok {
                    if let Some(c) = fm::connect(&h.frame) {
                        for call in [&c.source, &c.target, &c.via1, &c.via2] {
                            if !call.is_empty() && !link.calls.contains(call) {
                                link.calls.push(call.clone());
                            }
                        }
                        link.called = Some(call_hash(&c.target));
                        link.caller = Some(call_hash(&c.source));
                        link.mode = Some(if c.wide { Mode::Wide } else { Mode::Narrow });
                        link.relayed = !c.via1.is_empty() || !c.via2.is_empty();
                        link.key_from = "the connect frame";
                    }
                } else if h.crc_ok && fm::keyed(h.rec) {
                    if link.mode.is_none() && h.rec >= 4 {
                        link.mode = Some(h.mode);
                    }
                    if link.called.is_none() && h.key.is_some_and(|k| link.candidates.contains(&k)) {
                        // a data frame decoded with a key found from the tone bursts: that settles it
                        link.called = h.key;
                        link.candidates.clear();
                        link.key_from = "the tone bursts (callsign not heard)";
                    }
                }
                events.push(FmEvent::Frame(FmFrame {
                    time: lo as f64 / RATE as f64 + h.start,
                    seconds: h.seconds,
                    mode: h.mode,
                    rec: h.rec,
                    frame: h.frame,
                    crc_ok: h.crc_ok,
                    score: h.score,
                    station: 0,
                    part,
                    began: a as f64 / RATE as f64,
                }));
            }
        }
        if pass == 0 {
            // the tone bursts: named from the link's codebook, or used to find the key
            let tone_rows: Vec<&Vec<Tones>> = kinds.iter().flatten().collect();
            if link.called.is_none() && !undecoded.is_empty() {
                // each distinct burst gives a few candidates; a key that several distinct bursts allow,
                // and then the one nearest its seed, is tried first
                let mut distinct: Vec<&Vec<Tones>> = Vec::new();
                for rows in &tone_rows {
                    let tail = &rows[rows.len() - fm::CODE_ROWS..];
                    if !distinct.iter().any(|d| &d[d.len() - fm::CODE_ROWS..] == tail) {
                        distinct.push(rows);
                    }
                }
                let mut votes: HashMap<u32, (usize, usize)> = HashMap::new();
                for rows in distinct.iter().take(6) {
                    for (h, is_caller, back) in blind_keys(rows) {
                        if is_caller {
                            link.caller.get_or_insert(h);
                            continue;
                        }
                        let v = votes.entry(h).or_insert((0, usize::MAX));
                        v.0 += 1;
                        v.1 = v.1.min(back);
                    }
                }
                let mut c: Vec<(u32, (usize, usize))> = votes.into_iter().collect();
                c.sort_by_key(|(h, (n, back))| (std::cmp::Reverse(*n), *back, *h));
                link.candidates = c.into_iter().take(24).map(|x| x.0).collect();
                if debug {
                    eprintln!("candidates {:?}, undecoded {:?}", link.candidates, undecoded);
                }
            }
        }
    }
    let books: Vec<Vec<Vec<Tones>>> = {
        let calls: Vec<u32> = link.keys();
        let callers: Vec<u32> = link.caller.into_iter().chain(calls.iter().copied()).collect();
        let mut v = Vec::new();
        for &d in &calls {
            for &u in &callers {
                v.push(fm::codebook(d, u));
            }
        }
        v
    };
    for (i, &(a, b)) in pieces.iter().enumerate() {
        let Some(rows) = &kinds[i] else { continue };
        let code = books.iter().filter_map(|book| match_code(rows, book)).max_by_key(|c| c.1);
        events.push(FmEvent::Tone(ToneBurst { time: a as f64 / RATE as f64, seconds: (b - a) as f64 / RATE as f64, rows: rows.clone(), code, station: 0 }));
    }
    events.sort_by(|p, q| p.time().total_cmp(&q.time()));
    assign_stations(&mut events);
    FmScan { events, link }
}

/// Half duplex: a transmission that follows the previous one within 0.6 s is the other station's;
/// after a longer pause the same station is sending again. Frames sent right behind each other are
/// one transmission.
fn assign_stations(events: &mut [FmEvent]) {
    let mut cur = 0usize;
    let mut prev_end = f64::NEG_INFINITY;
    for (i, ev) in events.iter_mut().enumerate() {
        let follows = matches!(ev, FmEvent::Frame(f) if f.part > 0);
        let began = match ev {
            FmEvent::Frame(f) => f.began,
            FmEvent::Tone(t) => t.time,
        };
        if i > 0 && !follows && began - prev_end < 0.6 {
            cur ^= 1;
        }
        ev.set_station(cur);
        prev_end = ev.end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u32) -> impl FnMut(u32) -> u32 {
        let mut n = seed;
        move |m: u32| {
            n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (n >> 16) % m
        }
    }

    fn crc_frame(mut f: Vec<u8>) -> Vec<u8> {
        let c = crate::crc::crc16(&f);
        f.extend(c.to_be_bytes());
        f
    }

    fn connect_frame(source: &str, target: &str, wide: bool) -> Vec<u8> {
        let field = |c: &str| {
            let (base, ssid) = c.split_once('-').unwrap_or((c, "0"));
            let mut f = [b' '; 8];
            f[..base.len()].copy_from_slice(base.as_bytes());
            f[7] = ssid.parse().unwrap();
            f
        };
        let mut f = vec![4 | (u8::from(wide) << 1)];
        f.extend(field(source));
        f.extend([0u8; 8]);
        f.extend([0u8; 8]);
        f.extend(field(target));
        f.extend([0, 0x08, 0x07, 0]);
        crc_frame(f)
    }

    fn data_frame(len: usize, text: &[u8]) -> Vec<u8> {
        let room = len - 4;
        let mut f = text.to_vec();
        f.push(crate::frame::TERMINATOR);
        f.resize(room, 0);
        f.extend([0x02, 0x00]);
        crc_frame(f)
    }

    /// A whole link from generated audio: connect, the reply tone, data frames at several levels
    /// (4-phase and QAM), acknowledgements. Everything must decode, with and without the connect frame.
    #[test]
    fn generated_link() {
        let mut tabs = FmTables::new();
        let (caller, called) = ("N0CALL", "N1CALL-2");
        let (hc, hd) = (call_hash(caller), call_hash(called));
        let book = fm::codebook(hd, hc);
        let mut x: Vec<f64> = vec![0.0; RATE / 2];
        let push = |audio: Vec<f64>, x: &mut Vec<f64>| {
            x.extend(audio.iter().map(|v| v * 0.8));
            x.extend(std::iter::repeat(0.0).take(RATE / 5));
        };
        let narrow = fm::Mode::Narrow;
        push(fm::transmission(narrow, &tabs.narrow, 0, 0, &[&connect_frame(caller, called, false)], 21, 0.0, &mut lcg(1)), &mut x);
        push(fm::tone_audio(&book[22], 8, 0.0), &mut x);
        let mut sent = Vec::new();
        for (i, rec) in [15usize, 3, 6, 8].into_iter().enumerate() {
            let len = narrow.records()[rec].frame_bytes();
            let text = format!("message {i} at record {rec}");
            let f = data_frame(len, text.as_bytes());
            push(fm::transmission(narrow, &tabs.narrow, rec, hd, &[&f], 1, 0.0, &mut lcg(10 + i as u32)), &mut x);
            push(fm::tone_audio(&book[0], 2, 0.0), &mut x);
            sent.push((rec, f));
        }
        let x = crate::synth::on_air(&x, 0.5, 0.2, 0.0, 0.003, 7);
        for blind in [false, true] {
            let mut y = x.clone();
            if blind {
                // cut the connect frame away: the key has to come from the tone bursts
                let cut = RATE / 2 + 21 * fm::SYMBOL + 28 * fm::SYMBOL + RATE / 5;
                y.drain(..cut);
            }
            let r = scan(&y, &mut tabs, &[]);
            let frames: Vec<&FmFrame> = r.events.iter().filter_map(|e| if let FmEvent::Frame(f) = e { Some(f) } else { None }).collect();
            let got: Vec<(usize, Vec<u8>)> = frames.iter().filter(|f| f.crc_ok && f.rec != 0).map(|f| (f.rec, f.frame.clone())).collect();
            assert_eq!(got, sent, "blind {blind}: {:?}", frames.iter().map(|f| (f.rec, f.crc_ok, f.score)).collect::<Vec<_>>());
            let codes: Vec<usize> = r.events.iter().filter_map(|e| if let FmEvent::Tone(t) = e { t.code.map(|c| c.0) } else { None }).collect();
            if blind {
                // the reply is named too; without the connect frame narrow and wide reply cannot be told apart
                assert!(matches!(codes[0], 23 | 24) && codes[1..] == [1, 1, 1, 1], "{codes:?}");
                assert_eq!(r.link.called, Some(hd));
            } else {
                assert_eq!(codes, [23, 1, 1, 1, 1]);
                assert_eq!(r.link.calls, [caller.to_string(), called.to_string()]);
            }
        }
    }
}
