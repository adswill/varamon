//! The modem's text compression: a dictionary coder with 8 states. Each compressed frame starts
//! with a 3-bit state number, followed by variable-length codes. A code stands for a phrase (often
//! a single byte) and may switch to another state; some states have an escape code that only
//! switches. The last bit is repeated in inverted form to fill the frame.
//!
//! The table in `textcomp.dict` comes from the xara project (MIT licence, see NOTICE).

use std::collections::HashMap;
use std::sync::OnceLock;

const MAX_CODE: usize = 10;
pub const START_STATE: usize = 1;

struct Entry {
    phrase: Vec<u8>,
    code: Vec<bool>,
    next: usize,
}

#[derive(Clone, Copy)]
enum Hit {
    Entry(usize),
    Escape(usize),
}

struct State {
    start: Vec<bool>,
    entries: Vec<Entry>,
    escape: Option<(Vec<bool>, usize)>,
    decode: HashMap<(usize, u64), Hit>,
}

fn bits_of(s: &str) -> Vec<bool> {
    s.bytes().map(|b| b == b'1').collect()
}

fn key(code: &[bool]) -> (usize, u64) {
    (code.len(), code.iter().fold(0u64, |a, &b| a << 1 | u64::from(b)))
}

fn tables() -> &'static Vec<State> {
    static T: OnceLock<Vec<State>> = OnceLock::new();
    T.get_or_init(|| {
        let mut states: Vec<State> = Vec::new();
        for line in include_str!("textcomp.dict").lines() {
            let f: Vec<&str> = line.split(' ').collect();
            if f[0] == "S" {
                let escape = (f[3] != "-").then(|| (bits_of(f[3]), f[4].parse().unwrap()));
                states.push(State { start: bits_of(f[2]), entries: Vec::new(), escape, decode: HashMap::new() });
            } else {
                let phrase = if f[1] == "-" { Vec::new() } else { (0..f[1].len()).step_by(2).map(|i| u8::from_str_radix(&f[1][i..i + 2], 16).unwrap()).collect() };
                states.last_mut().unwrap().entries.push(Entry { phrase, code: bits_of(f[2]), next: f[3].parse().unwrap() });
            }
        }
        for st in &mut states {
            for (i, e) in st.entries.iter().enumerate() {
                st.decode.entry(key(&e.code)).or_insert(Hit::Entry(i));
            }
            if let Some((c, next)) = &st.escape {
                st.decode.entry(key(c)).or_insert(Hit::Escape(*next));
            }
        }
        states
    })
}

fn unpack(bytes: &[u8]) -> Vec<bool> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1 == 1)).collect()
}

/// Unpack the payload of one compressed frame. `None` if the bits do not parse.
pub fn decompress(payload: &[u8]) -> Option<Vec<u8>> {
    let t = tables();
    let mut b = unpack(payload);
    let pad = *b.last()?;
    while b.last() == Some(&pad) {
        b.pop();
    }
    if b.len() < 3 {
        return None;
    }
    let mut state = key(&b[..3]).1 as usize + 1;
    let (mut i, mut out) = (3, Vec::new());
    while i < b.len() {
        let st = &t[state - 1];
        let (len, hit) = (1..=MAX_CODE.min(b.len() - i)).find_map(|l| st.decode.get(&key(&b[i..i + l])).map(|h| (l, *h)))?;
        i += len;
        match hit {
            Hit::Entry(e) => {
                out.extend(&st.entries[e].phrase);
                if st.entries[e].next != 0 {
                    state = st.entries[e].next;
                }
            }
            Hit::Escape(next) => state = next,
        }
    }
    Some(out)
}

/// Pack as much of `data` as fits into `cap` bytes, starting in `state`.
/// Returns (frame payload, bytes of `data` used, state for the next frame).
pub fn compress(data: &[u8], cap: usize, state: usize) -> (Vec<u8>, usize, usize) {
    let t = tables();
    let mut state = if (1..=t.len()).contains(&state) { state } else { START_STATE };
    let mut out = t[state - 1].start.clone();
    let (mut consumed, mut pos) = (0, 0);
    let mut last: Option<(usize, usize, usize)> = None;
    while pos < data.len() {
        let st = &t[state - 1];
        let hit = st.entries.iter().find(|e| !e.phrase.is_empty() && data[pos..].starts_with(&e.phrase));
        let (code, used, next) = match (hit, &st.escape) {
            (Some(e), _) => (&e.code, e.phrase.len(), e.next),
            (None, Some((c, n))) => (c, 0, *n),
            (None, None) => {
                let e = st.entries.iter().find(|e| e.phrase == [data[pos]]).expect("state without escape lacks a byte");
                (&e.code, 1, e.next)
            }
        };
        let full = out.len() + code.len() > cap * 8;
        if !full {
            consumed += used;
            out.extend(code);
            last = Some((code.len(), used, state));
        }
        if next != 0 {
            state = next;
        }
        if full {
            break;
        }
        pos += used;
    }
    let pad = |b: &mut Vec<bool>| {
        let p = !*b.last().unwrap_or(&true);
        let n = if b.len() % 8 == 0 { 8 } else { 8 - b.len() % 8 };
        b.extend(std::iter::repeat(p).take(n));
    };
    pad(&mut out);
    if out.len() > cap * 8 {
        if let Some((len, used, st)) = last {
            out.truncate(out.len() - 8 - len);
            consumed -= used;
            state = st;
            pad(&mut out);
        }
    }
    let bytes = out.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |a, (i, &b)| a | (u8::from(b) << (7 - i)))).collect();
    (bytes, consumed, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_loads() {
        let t = tables();
        assert_eq!(t.len(), 8);
        assert!(t.iter().all(|s| s.start.len() == 3 && !s.entries.is_empty()));
    }

    /// A mail-session opening, packed frame by frame and unpacked again.
    #[test]
    fn round_trip_in_frames() {
        let text = b"[WL2K-5.0-B2FWIHJM$]\r;PQ: 12345678\rFC EM TESTMSG00001 310 190 0\rF> 4D\rHello, this is a test of the text mode.\r";
        for room in [22usize, 47, 89] {
            let (mut pos, mut state, mut out, mut packed) = (0, START_STATE, Vec::new(), 0);
            while pos < text.len() {
                let (bytes, used, next) = compress(&text[pos..], room, state);
                assert!(bytes.len() <= room && used > 0, "room {room}");
                out.extend(decompress(&bytes).expect("frame unpacks"));
                packed += bytes.len();
                pos += used;
                state = next;
            }
            assert_eq!(out, text, "room {room}");
            assert!(packed < text.len(), "{packed} bytes for {} of text", text.len());
        }
    }

    #[test]
    fn rubbish_is_rejected_or_harmless() {
        assert!(decompress(&[]).is_none());
        assert!(decompress(&[0xFF]).is_none());
    }
}
