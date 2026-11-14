//! Control bursts: acknowledgements, connect requests and other short signals between frames.
//!
//! A burst is a short lead-in followed by 31, 15 or 7 code symbols of 2048 samples (formats 0,
//! 1, 2; format 2 carries two tones per symbol). There are 58 codes. Their tones are drawn
//! straight from the table generator, one code after another, from a seed that depends on the
//! callsign of the station that was called; a few codes start a stream of their own. The seed is
//! found from the signal itself: every generator state is tried against a burst's tones, and the
//! candidates of several bursts are intersected.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::OnceLock;

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::audio::RATE;
use crate::fsk::analytic;
use crate::levels::Bandwidth;
use crate::vb::VbRnd;

pub const SYMBOL: usize = 2048;
/// code symbols per format
pub const CODE_SYMBOLS: [usize; 3] = [31, 15, 7];
pub const CODES: u32 = 58;
/// Seed step of each format relative to format 0: the published rule, and the one seen in an
/// older off-air recording.
const FORMAT_STEPS: [[u64; 3]; 2] = [[0, 1, 2], [0, 5, 2]];
const STATES: u32 = 1 << 24;
const SEED_LIMIT: u64 = (1 << 24) + 2048;
/// What a station adds to the seed for its own bursts: the one that made the call, the one that answered.
pub const SIDE: [u32; 2] = [50, 278];
/// The same for the BREAK and IDLE codes, whose seed is the caller's callsign.
pub const CONNECT_SIDE: [u32; 2] = [850, 1078];

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
        15..=18 => "CONNECT REPLY (2300 Hz)",
        19 => "NACK, frame missed",
        21 => "NO TRAFFIC",
        22..=25 => "CONNECT REPLY (500 Hz)",
        26..=30 => "CONNECT REQUEST (2300 Hz)",
        31..=35 => "CONNECT REQUEST (500 Hz)",
        36..=40 => "CONNECT REQUEST (2750 Hz)",
        41..=42 | 44 => "CONNECT REPLY (2750 Hz)",
        43 => "REJECT",
        45 | 47 | 58 => "unconnected frame follows",
        46 => "QRT (disconnect, no identification)",
        48 => "CQ",
        49 => "BREAK",
        50 => "IDLE",
        51..=57 => "START LEVEL",
        _ => "control code",
    }
}

/// Which tone set a code uses: 1 (500 Hz), 2 (2300 Hz) or 3 (2750 Hz). Most codes use the set of
/// the link's bandwidth mode; connect requests and replies use the one of the mode they name.
pub fn family(code: u32, bw: Bandwidth) -> u32 {
    let of = |b: Bandwidth| match b {
        Bandwidth::Hz500 => 1,
        Bandwidth::Hz2300 => 2,
        Bandwidth::Hz2750 => 3,
    };
    match code {
        15..=18 | 26..=30 | 36..=40 => 2,
        22..=25 | 31..=35 | 45 | 47 | 48 | 58 => 1,
        41..=44 => 3,
        _ => of(bw),
    }
}

/// Tone set `v`: (bases to choose from minus one, lowest base bin). Bases are 14 bins apart.
const fn comb(v: u32) -> (u32, u32) {
    match v {
        1 => (1, 50),
        2 => (4, 29),
        _ => (5, 22),
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
/// A burst is a lead-in of one to ten symbols followed by the code, so the code is read from the
/// burst's last symbols.
pub fn measure(x: &[f64], start: f64, seconds: f64, offset_hz: f64) -> Option<Measured> {
    let nsym = (seconds * RATE as f64 / SYMBOL as f64).round() as usize;
    let format = match nsym {
        30..=34 | 39..=43 => 0,
        14..=18 | 21..=25 => 1,
        6..=12 => 2,
        _ => return None,
    };
    let len = CODE_SYMBOLS[format];
    let a = ((start * RATE as f64) as usize).saturating_sub(4096);
    let b = (a + (nsym + 5) * SYMBOL).min(x.len());
    if b <= a + (len + 1) * SYMBOL {
        return None;
    }
    let mut z = analytic(&x[a..b]);
    for (n, v) in z.iter_mut().enumerate() {
        *v *= Complex64::from_polar(1.0, -2.0 * PI * offset_hz * n as f64 / RATE as f64);
    }
    let fft = FftPlanner::new().plan_fft_forward(SYMBOL);
    let mut buf = vec![Complex64::new(0.0, 0.0); SYMBOL];
    let two = format == 2;
    // per symbol: (energy in the band, share of it in the strongest tone or two, the tones)
    let mut best: Option<(f64, Vec<(f64, f64, Tones)>)> = None;
    let mut off = 0;
    while off < SYMBOL && off + (len + 1) * SYMBOL <= z.len() {
        let rows: Vec<(f64, f64, Tones)> = (0..(z.len() - off) / SYMBOL)
            .map(|k| {
                buf.copy_from_slice(&z[off + k * SYMBOL..off + (k + 1) * SYMBOL]);
                fft.process(&mut buf);
                let p: Vec<f64> = buf[20..110].iter().map(|v| v.norm_sqr()).collect();
                let all: f64 = p.iter().sum::<f64>() + 1e-12;
                let first = (0..p.len()).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
                let second = (0..p.len()).filter(|&i| i != first).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap();
                if two {
                    (all, (p[first] + p[second]) / all, [first.min(second) as u32 + 20, first.max(second) as u32 + 20])
                } else {
                    (all, p[first] / all, [first as u32 + 20, 0])
                }
            })
            .collect();
        let score: f64 = rows.iter().map(|r| r.0 * r.1).sum();
        if best.as_ref().map_or(true, |b| score > b.0) {
            best = Some((score, rows));
        }
        off += 128;
    }
    let (_, rows) = best?;
    // which symbols belong to the burst: those with at least a quarter of the typical energy
    let mut e: Vec<f64> = rows.iter().map(|r| r.0).collect();
    e.sort_by(|a, b| b.total_cmp(a));
    let typical = e[(len / 2).min(e.len() - 1)];
    let last = rows.iter().rposition(|r| r.0 > 0.25 * typical)?;
    let first = (last + 1).checked_sub(len)?;
    let code = &rows[first..=last];
    let purity = code.iter().map(|r| r.1).sum::<f64>() / len as f64;
    Some(Measured { format, tones: code.iter().map(|r| r.2).collect(), purity })
}

/// The lead-in symbols of each format: fixed tones, the same for every station.
pub fn lead_in(format: usize, symbols: usize) -> Vec<Tones> {
    let mut r = VbRnd::new();
    let mut out = Vec::new();
    for f in 0..=format {
        out.clear();
        for _ in 0..100 {
            let mut t = [0u32; 2];
            let mut have = 0;
            while have < if f == 2 { 2 } else { 1 } {
                let x = r.rand_int(34) + 50;
                if !t[..have].contains(&x) {
                    t[have] = x;
                    have += 1;
                }
            }
            out.push(t);
        }
    }
    out.truncate(symbols);
    out
}

/// Test audio of a burst: lead-in symbols, then the code's tones (48 kHz, peak about 1).
pub fn audio(format: usize, lead: usize, code: &[Tones]) -> Vec<f64> {
    let n = if format == 2 { 2 } else { 1 };
    lead_in(format, lead)
        .iter()
        .chain(code)
        .flat_map(|t| (0..SYMBOL).map(move |i| t[..n].iter().map(|&k| (2.0 * PI * f64::from(k) * i as f64 / SYMBOL as f64).sin()).sum::<f64>() / n as f64))
        .collect()
}

/// The tones of one code in tone set `v`, drawn from `r`.
pub fn draw_code(r: &mut VbRnd, format: usize, v: u32) -> Vec<Tones> {
    let ntones = if format == 2 { 2 } else { 1 };
    let (n, off) = comb(v);
    (0..CODE_SYMBOLS[format])
        .map(|i| {
            let parity = u32::from(i % 2 == 0 && v != 1); // symbol numbers start at 2
            let mut bases = [0u32; 2];
            let mut have = 0;
            while have < ntones {
                let b = r.rand_int(n) * 14 + off;
                if !bases[..have].contains(&b) {
                    bases[have] = b;
                    have += 1;
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

/// Generator states from which a single-tone code of tone set `v` would come out as `tones`.
fn find_states(tones: &[Tones], v: u32) -> Vec<u32> {
    let (n, off) = comb(v);
    let mut out = Vec::new();
    'state: for s in 0..STATES {
        let mut r = VbRnd::from_state(s);
        for (i, t) in tones.iter().enumerate() {
            let b = r.rand_int(n) * 14 + off;
            if b + u32::from(i % 2 == 0 && v != 1) + 2 * r.rand_int(6) != t[0] {
                continue 'state;
            }
        }
        out.push(s);
    }
    out
}

/// For every generator state: the seed arguments that start a stream there.
pub(crate) struct SeedIndex {
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

    pub(crate) fn get(&self, state: u32) -> impl Iterator<Item = u32> + '_ {
        let f = self.first[state as usize];
        (f != u32::MAX).then_some(f).into_iter().chain(self.more.get(&state).into_iter().flatten().copied())
    }
}

pub(crate) fn seed_index() -> &'static SeedIndex {
    static INDEX: OnceLock<SeedIndex> = OnceLock::new();
    INDEX.get_or_init(SeedIndex::new)
}

/// A resolved key: the seed argument of format 0's code stream, and the seed step of each format.
/// The seed argument is the called station's callsign hash plus 10 plus the sender's side value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub base: u32,
    pub steps: [u64; 3],
}

impl Key {
    /// The callsign hash of the station that was called, if the sender of the bursts made the
    /// call, and if it answered it.
    pub fn hashes(&self) -> (u32, u32) {
        (self.base.wrapping_sub(10 + SIDE[0]) & 0xFF_FFFF, self.base.wrapping_sub(10 + SIDE[1]) & 0xFF_FFFF)
    }
}

/// The codes of a link that can be worked out from the key alone, for one burst format: index =
/// code number; codes that depend on something else (connect requests, BREAK, IDLE, the
/// unconnected ones) are left empty.
pub fn session_book(key: Key, format: usize, bw: Bandwidth) -> Vec<Vec<Tones>> {
    let step = key.steps[format];
    let base = u64::from(key.base);
    let mut r = VbRnd::seeded(base + step);
    let mut book = vec![Vec::new()];
    for c in 1..=CODES {
        match c {
            41 => r = VbRnd::seeded(base - 10 + 300 + step),
            51 => r = VbRnd::seeded(base - 10 + 1500 + step),
            _ => {}
        }
        // code 45 is drawn here with the link's own tone set (it is replaced by a fixed code afterwards)
        let code = draw_code(&mut r, format, family(if c == 45 { 1 } else { c }, bw).max(if c == 45 { family(1, bw) } else { 0 }));
        book.push(if matches!(c, 26..=40 | 45 | 47..=50 | 58) { Vec::new() } else { code });
    }
    book
}

fn request_offset(bw: Bandwidth) -> u64 {
    match bw {
        Bandwidth::Hz500 => 7,
        Bandwidth::Hz2300 => 0,
        Bandwidth::Hz2750 => 8,
    }
}

/// The connect request for the station whose callsign hash is `called`, asking for mode `bw`.
pub fn request_code(called: u32, format: usize, bw: Bandwidth) -> Vec<Tones> {
    let mut r = VbRnd::seeded(u64::from(called) + format as u64 + u64::from(SIDE[0]) + request_offset(bw));
    draw_code(&mut r, format, if bw == Bandwidth::Hz500 { 1 } else { 2 })
}

/// The BREAK and IDLE codes of a link whose caller has callsign hash `caller`; `side` is 0 if the
/// caller sends them, 1 if the answerer does.
pub fn connect_codes(caller: u32, side: usize, format: usize, bw: Bandwidth) -> [Vec<Tones>; 2] {
    let mut r = VbRnd::seeded(format as u64 + u64::from(caller) + u64::from(CONNECT_SIDE[side]));
    let v = family(49, bw);
    [draw_code(&mut r, format, v), draw_code(&mut r, format, v)]
}

/// The codes that are the same for everyone: (code number, tones) for CQ and the three
/// "unconnected frame follows" codes, as sent by side 0 or 1.
pub fn fixed_codes(side: usize, format: usize) -> Vec<(u32, Vec<Tones>)> {
    let fo = format as u64 + u64::from(SIDE[side]);
    [(45u32, 0x02D2_77B0u64), (47, 0xCD_80A4), (48, 0xA3_CE37), (58, 0x49_E730)].iter().map(|&(c, seed)| (c, draw_code(&mut VbRnd::seeded(seed + fo), format, 1))).collect()
}

fn matches(want: &[Tones], got: &[Tones]) -> usize {
    want.iter().zip(got).filter(|(w, t)| w == t).count()
}

fn close_enough(hits: usize, total: usize) -> bool {
    hits * 10 >= total * 7
}

/// Find the key of a station from its control bursts. Needs single-tone bursts (formats 0 and 1)
/// that differ from each other and come from the same seed. `bw` is the link's mode.
pub fn resolve(bursts: &[Measured], bw: Bandwidth) -> Option<Key> {
    match resolve_all(bursts, bw)[..] {
        [key] => Some(key),
        _ => None,
    }
}

/// Every key that at least two different single-tone bursts agree on. Bursts of several stations
/// may be mixed: each station's bursts agree only among themselves.
pub fn resolve_all(bursts: &[Measured], bw: Bandwidth) -> Vec<Key> {
    let mut distinct: Vec<&Measured> = Vec::new();
    for b in bursts.iter().filter(|b| b.format < 2 && b.purity > 0.5) {
        if !distinct.iter().any(|d| d.format == b.format && d.tones == b.tones) {
            distinct.push(b);
        }
    }
    if distinct.len() < 2 {
        return Vec::new();
    }
    let index = seed_index();
    // every seed argument from which one of the first 25 codes of a stream comes out as the burst
    let starts: Vec<Vec<u32>> = distinct
        .iter()
        .map(|b| {
            let per = 2 * CODE_SYMBOLS[b.format];
            let mut seen = Vec::new();
            for s in find_states(&b.tones, family(1, bw)) {
                let mut r = VbRnd::from_state(s);
                for _ in 0..25 {
                    for x in index.get(r.state()) {
                        if !seen.contains(&x) {
                            seen.push(x);
                        }
                    }
                    for _ in 0..per {
                        r.retreat();
                    }
                }
            }
            seen
        })
        .collect();
    // candidates: (bursts that agree, key)
    let mut cands: Vec<(Vec<usize>, Key)> = Vec::new();
    for steps in FORMAT_STEPS {
        let mut votes: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, (b, seen)) in distinct.iter().zip(&starts).enumerate() {
            let mut bases: Vec<u32> = seen.iter().map(|&x| u64::from(x).wrapping_sub(steps[b.format]) as u32).collect();
            bases.sort_unstable();
            bases.dedup();
            for base in bases {
                votes.entry(base).or_default().push(i);
            }
        }
        // bursts from other streams (connect requests, BREAK, IDLE) vote for nothing in common, so
        // two bursts that agree are enough; a chance agreement is one in millions
        cands.extend(votes.into_iter().filter(|(_, v)| v.len() >= 2).map(|(base, v)| (v, Key { base, steps })));
    }
    // The same bursts also agree on shifted copies of their stream (a seed 32768 lower starts
    // one step later) and, if they share a format, under either step rule. The real key is the
    // one most bursts agree on; among equals, the published step rule and the highest seed.
    cands.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then((a.1.steps != FORMAT_STEPS[0]).cmp(&(b.1.steps != FORMAT_STEPS[0]))).then(b.1.base.cmp(&a.1.base)));
    let mut keys: Vec<(Vec<usize>, Key)> = Vec::new();
    for (voters, key) in cands {
        if !keys.iter().any(|(v, _)| voters.iter().all(|i| v.contains(i))) {
            keys.push((voters, key));
        }
    }
    keys.into_iter().map(|k| k.1).collect()
}

/// Which code a burst is, given the key: (code number, symbols that match, symbols in the code).
pub fn identify(b: &Measured, key: Key, bw: Bandwidth) -> Option<(u32, usize, usize)> {
    let total = CODE_SYMBOLS[b.format];
    let book = session_book(key, b.format, bw);
    let (code, hits) = (1..=CODES).map(|c| (c, matches(&book[c as usize], &b.tones))).max_by_key(|x| x.1)?;
    close_enough(hits, total).then_some((code, hits, total))
}

/// What a burst is when the link's key does not explain it: a connect request, BREAK, IDLE, CQ
/// or the announcement of an unconnected frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Other {
    /// a call to the station with this callsign hash, asking for this mode
    Request { called: u32, bw: Bandwidth },
    /// BREAK (49) or IDLE (50) of a link whose caller has this callsign hash; side 0 = sent by the caller
    Connect { code: u32, caller: u32, side: usize },
    /// CQ or "unconnected frame follows"
    Fixed { code: u32 },
}

/// Test a burst against the codes that the callsign hashes in `known` would give: connect
/// requests to them, and BREAK or IDLE of links they started. Also against the fixed codes.
pub fn identify_other(b: &Measured, known: &[u32], bw: Bandwidth) -> Option<(Other, usize, usize)> {
    let total = CODE_SYMBOLS[b.format];
    let ok = |want: &[Tones]| close_enough(matches(want, &b.tones), total).then(|| matches(want, &b.tones));
    for side in 0..2 {
        for (code, tones) in fixed_codes(side, b.format) {
            if let Some(hits) = ok(&tones) {
                return Some((Other::Fixed { code }, hits, total));
            }
        }
    }
    for &h in known {
        for req in [Bandwidth::Hz2300, Bandwidth::Hz500, Bandwidth::Hz2750] {
            if let Some(hits) = ok(&request_code(h, b.format, req)) {
                return Some((Other::Request { called: h, bw: req }, hits, total));
            }
        }
        for side in 0..2 {
            let pair = connect_codes(h, side, b.format, bw);
            for (i, tones) in pair.iter().enumerate() {
                if let Some(hits) = ok(tones) {
                    return Some((Other::Connect { code: 49 + i as u32, caller: h, side }, hits, total));
                }
            }
        }
    }
    None
}

/// For a single-tone burst that is taken to be a connect request: the callsign hash of the
/// station being called and the mode asked for. A request is the first code of its own stream,
/// so the hash follows from the generator state behind the tones. `None` if no state fits.
pub fn request_target(b: &Measured) -> Option<(u32, Bandwidth)> {
    if b.format == 2 {
        return None;
    }
    // requests for the 500 Hz mode use its narrow tone set, the others the 2300 Hz one
    let narrow = b.tones.iter().all(|t| (50..=76).contains(&t[0]));
    let v = if narrow { 1 } else { 2 };
    let states = find_states(&b.tones, v);
    let [state] = states[..] else { return None };
    let x = seed_index().get(state).next()?;
    let bw = if narrow { Bandwidth::Hz500 } else { Bandwidth::Hz2300 };
    let called = u64::from(x).checked_sub(b.format as u64 + u64::from(SIDE[0]) + request_offset(bw))?;
    Some((called as u32 & 0xFF_FFFF, bw))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLD: [u64; 3] = [0, 5, 2];

    fn code(key: Key, format: usize, code: u32) -> Measured {
        Measured { format, tones: session_book(key, format, Bandwidth::Hz2300)[code as usize].clone(), purity: 0.9 }
    }

    /// The five bursts of the off-air test recording: tones as measured.
    #[test]
    fn off_air_bursts() {
        let long: Vec<Tones> =
            [62, 35, 32, 75, 64, 87, 86, 79, 52, 81, 88, 39, 82, 69, 52, 33, 50, 49, 74, 85, 78, 55, 70, 53, 44, 81, 82, 53, 62, 0, 0].iter().map(|&t| [t, 0]).collect();
        let medium: Vec<Tones> = [48, 67, 76, 97, 50, 71, 66, 71, 90, 29, 84, 45, 76, 75, 92].iter().map(|&t| [t, 0]).collect();
        let key = Key { base: 16_759_090, steps: OLD };
        let got = code(key, 0, 6);
        assert_eq!(&got.tones[..29], &long[..29]);
        assert_eq!(code(key, 1, 1).tones, medium);
        let short_up: Vec<Tones> = vec![[40, 48], [47, 97], [52, 78], [37, 61], [54, 94], [51, 69], [54, 96]];
        assert_eq!(identify(&Measured { format: 2, tones: short_up, purity: 0.8 }, key, Bandwidth::Hz2300), Some((7, 7, 7)));
        assert_eq!(key.hashes(), (0xffb8f6, 0xffb812));
    }

    /// The key is found again from two generated bursts, with either seed step rule.
    #[test]
    fn key_from_two_bursts() {
        for steps in FORMAT_STEPS {
            let key = Key { base: 0x12_3456 + 60, steps };
            let found = resolve(&[code(key, 0, 6), code(key, 1, 1), code(key, 2, 7)], Bandwidth::Hz2300);
            assert_eq!(found, Some(key));
            assert_eq!(identify(&code(key, 2, 7), key, Bandwidth::Hz2300), Some((7, 7, 7)));
            assert_eq!(identify(&code(key, 0, 14), key, Bandwidth::Hz2300), Some((14, 31, 31)));
            assert_eq!(identify(&code(key, 1, 43), key, Bandwidth::Hz2300), Some((43, 15, 15)));
            assert_eq!(identify(&code(key, 0, 53), key, Bandwidth::Hz2300), Some((53, 31, 31)));
        }
        // one burst alone does not pin the key
        let key = Key { base: 0x12_3456 + 60, steps: FORMAT_STEPS[0] };
        assert_eq!(resolve(&[code(key, 1, 1), code(key, 1, 1)], Bandwidth::Hz2300), None);
    }

    /// A connect request gives back the callsign hash of the station being called.
    #[test]
    fn request_names_the_called_station() {
        let called = crate::frame::call_hash("N1CALL");
        assert_eq!(called, 0xd8a3d7);
        for bw in [Bandwidth::Hz2300, Bandwidth::Hz500] {
            let b = Measured { format: 0, tones: request_code(called, 0, bw), purity: 0.9 };
            assert_eq!(request_target(&b), Some((called, bw)));
            assert_eq!(identify_other(&b, &[1, called], Bandwidth::Hz2300).map(|x| x.0), Some(Other::Request { called, bw }));
        }
        let caller = crate::frame::call_hash("N0CALL");
        let idle = Measured { format: 0, tones: connect_codes(caller, 1, 0, Bandwidth::Hz2300)[1].clone(), purity: 0.9 };
        assert_eq!(identify_other(&idle, &[called, caller], Bandwidth::Hz2300).map(|x| x.0), Some(Other::Connect { code: 50, caller, side: 1 }));
    }
}
