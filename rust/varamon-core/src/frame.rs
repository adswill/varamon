//! What is inside a decoded frame, and joining the frames of one station into a stream.
//!
//! A frame is: payload field, an optional "next level" byte (only in a station's last frame of a
//! turn), a control byte, and the CRC. If the payload does not fill its field it ends with a
//! terminator byte.

pub const TERMINATOR: u8 = 0x14;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    /// the payload bytes, or `None` if the field is neither full nor properly terminated
    pub data: Option<Vec<u8>>,
    /// which acknowledgement format the sender asks for (0-2)
    pub reply: u8,
    /// the payload went through the modem's text compression
    pub compressed: bool,
    /// how much more the sender has queued (0-6)
    pub hint: u8,
    /// last frame of this station's turn
    pub last: bool,
    /// the payload fills the whole field
    pub full: bool,
    pub next_level: Option<u8>,
}

/// Split a frame whose CRC has already been checked.
pub fn parse(frame: &[u8]) -> Option<Parsed> {
    let at = frame.len().checked_sub(3)?;
    let c = frame[at];
    let (last, full) = (c & 2 != 0, c & 1 != 0);
    let end = at.checked_sub(usize::from(last))?;
    let field = &frame[..end];
    let data = if full {
        Some(field.to_vec())
    } else {
        // the terminator is followed by one marker byte and zero fill
        field.iter().rposition(|&b| b == TERMINATOR).and_then(|i| field[i + 1..].iter().skip(1).all(|&b| b == 0).then(|| field[..i].to_vec()))
    };
    Some(Parsed { data, reply: c >> 6, compressed: c & 0x20 != 0, hint: (c >> 2) & 7, last, full, next_level: last.then(|| frame[end]) })
}

/// True if the bytes read as text: printable ASCII plus line breaks and tabs, at least 4 characters.
pub fn is_text(data: &[u8]) -> bool {
    data.len() >= 4 && data.iter().all(|&b| (0x20..0x7F).contains(&b) || matches!(b, b'\r' | b'\n' | b'\t'))
}

/// Text for display: line breaks shown as `\n`.
pub fn text_of(data: &[u8]) -> String {
    let mut out = String::new();
    let mut prev_cr = false;
    for &b in data {
        match b {
            b'\r' => out.push_str("\\n"),
            b'\n' if prev_cr => {}
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push(' '),
            _ => out.push(b as char),
        }
        prev_cr = b == b'\r';
    }
    out
}

/// The longest runs of text inside binary data (at least `min` characters), for a quick look.
pub fn text_runs(data: &[u8], min: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut run = Vec::new();
    for &b in data.iter().chain(std::iter::once(&0)) {
        if (0x20..0x7F).contains(&b) {
            run.push(b);
        } else {
            if run.len() >= min {
                out.push(String::from_utf8_lossy(&run).into_owned());
            }
            run.clear();
        }
    }
    out
}

/// Undo the modem's "files" compression: the stream is a series of chunks, each with a 2-byte
/// header (top bit: the body is DEFLATE data; low 15 bits: body length). Returns the unpacked
/// bytes and whether the stream ended cleanly at a chunk boundary.
pub fn unpack_files(stream: &[u8]) -> Option<(Vec<u8>, bool)> {
    const MAX_INFLATED: usize = 50_000;
    let mut out = Vec::new();
    let mut pos = 0;
    let mut chunks = 0;
    while pos + 2 <= stream.len() {
        let len = usize::from(stream[pos] & 0x7F) << 8 | usize::from(stream[pos + 1]);
        let deflated = stream[pos] & 0x80 != 0;
        if len == 0 || pos + 2 + len > stream.len() {
            break;
        }
        let body = &stream[pos + 2..pos + 2 + len];
        if deflated {
            out.extend(miniz_oxide::inflate::decompress_to_vec_with_limit(body, MAX_INFLATED).ok()?);
        } else {
            out.extend_from_slice(body);
        }
        pos += 2 + len;
        chunks += 1;
    }
    (chunks > 0).then_some((out, pos == stream.len()))
}

/// Lines that are commands of a Winlink mail session, if the text contains any.
pub fn winlink_commands(text: &[u8]) -> usize {
    text.split(|&b| b == b'\r' || b == b'\n')
        .filter(|l| {
            l.starts_with(b"[WL2K-") || l.starts_with(b"[RMS ") || l.starts_with(b";PQ:") || l.starts_with(b";PR:") || l.starts_with(b";FW:")
                || l.starts_with(b"FC EM ") || l.starts_with(b"FS ") || l.starts_with(b"FF") || l.starts_with(b"FQ") || l.starts_with(b"F> ")
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crc::crc16;

    fn build(len: usize, data: &[u8], ctrl: u8, next_level: Option<u8>) -> Vec<u8> {
        let room = len - 3 - usize::from(next_level.is_some());
        let full = data.len() >= room;
        let mut out = data[..data.len().min(room)].to_vec();
        if !full {
            out.push(TERMINATOR);
            if out.len() < room {
                out.push(0x5A);
            }
            out.resize(room, 0);
        }
        out.extend(next_level);
        out.push((ctrl & !3) | (u8::from(next_level.is_some()) << 1) | u8::from(full));
        out.extend(crc16(&out).to_be_bytes());
        out
    }

    #[test]
    fn full_and_short_payloads() {
        let p = parse(&build(25, b"0123456789012345678901", 0x40, None)).unwrap();
        assert_eq!((p.data.as_deref(), p.full, p.last, p.reply), (Some(&b"0123456789012345678901"[..]), true, false, 1));
        let p = parse(&build(25, b"CQ CQ DE TEST", 0x00, Some(3))).unwrap();
        assert_eq!((p.data.as_deref(), p.full, p.last, p.next_level), (Some(&b"CQ CQ DE TEST"[..]), false, true, Some(3)));
        let p = parse(&build(12, b"", 0x00, None)).unwrap();
        assert_eq!(p.data.as_deref(), Some(&b""[..]));
    }

    #[test]
    fn files_compression() {
        let text = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let z = miniz_oxide::deflate::compress_to_vec(text, 6);
        let mut stream = vec![0x80 | (z.len() >> 8) as u8, z.len() as u8];
        stream.extend(&z);
        stream.extend([0x00, 0x03, b'a', b'b', b'c']); // a second chunk, stored
        let (out, clean) = unpack_files(&stream).unwrap();
        assert!(clean);
        assert_eq!(out, [&text[..], b"abc"].concat());
        // a stream cut in the middle of a chunk gives what is complete so far
        let (part, clean) = unpack_files(&stream[..stream.len() - 2]).unwrap();
        assert_eq!((part.len(), clean), (text.len(), false));
        assert!(unpack_files(&[0x12]).is_none());
    }

    #[test]
    fn winlink_lines() {
        assert_eq!(winlink_commands(b"[WL2K-5.0-B2FWIHJM$]\r;PQ: 12345678\rFC EM ABCDEFGHIJKL 100 80 0\rF> 3A\r"), 4);
        assert_eq!(winlink_commands(b"HELLO DE TEST\r"), 0);
    }

    #[test]
    fn text_detection() {
        assert!(is_text(b"HELLO DE TEST\r\n"));
        assert!(!is_text(&[0x79, 0x62, 0x34, 0xd9, 0x7a]));
        assert_eq!(text_of(b"A\r\nB\tC"), "A\\nB C");
        assert_eq!(text_runs(b"\x01\x02CALLSIGN\x00ab\x05", 4), vec!["CALLSIGN".to_string()]);
    }
}

/// The header frame a station sends when a link starts: its callsign and what it offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub call: String,
    /// compression the station offers: 0 none, 1 text, 2 files
    pub compression: u8,
    /// the level the station proposes for the other direction
    pub offer: u8,
    pub registered: bool,
}

/// A callsign from its packed form: eight 6-bit codes in six bytes (seven characters, then the suffix).
pub fn unpack_call(b: &[u8]) -> Option<String> {
    let v = b.get(..6)?.iter().fold(0u64, |v, &x| (v << 8) | u64::from(x));
    let mut base = String::new();
    let mut ended = false;
    for i in (1..8).rev() {
        match ((v >> (6 * i)) & 63) as u8 {
            0 => ended = true,
            _ if ended => return None, // a character after the end of the callsign
            c @ 1..=26 => base.push((b'A' + c - 1) as char),
            c @ 27..=36 => base.push((b'0' + c - 27) as char),
            _ => return None,
        }
    }
    if base.len() < 3 {
        return None;
    }
    Some(match (v & 63) as u8 {
        0 => base,
        0x10 => base + "-T",
        0x11 => base + "-R",
        0x12 => base + "-X",
        0x13 => base + "-L",
        n if n < 0x10 => format!("{base}-{n}"),
        _ => return None,
    })
}

/// Read a frame as a header frame. `None` if it does not have the shape of one.
pub fn header(frame: &[u8]) -> Option<Header> {
    let p = parse(frame)?;
    let data = p.data?;
    if !p.last || p.compressed || data.len() != 8 || data[6] != 0 || data[7] & 0x80 == 0 {
        return None;
    }
    Some(Header { call: unpack_call(&data)?, compression: (data[7] >> 1) & 3, offer: p.next_level?, registered: data[7] & 1 != 0 })
}

/// The 24-bit number a callsign is turned into for the control bursts.
pub fn call_hash(call: &str) -> u32 {
    let b = call.as_bytes();
    if b.is_empty() {
        return 0;
    }
    let c1 = u32::from(crate::crc::crc16(b));
    let c2 = u32::from(crate::crc::crc16(&b[b.len().saturating_sub(3)..]));
    ((c2 >> 8) << 16) | c1
}

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn header_frame_gives_the_callsign() {
        // header frames of levels 1 and 2 as sent on the air: callsign N0CALL
        let unhex = |h: &str| (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect::<Vec<u8>>();
        let h = header(&unhex("39b0c130c00000800183df5c")).expect("level 1 header");
        assert_eq!((h.call.as_str(), h.offer, h.compression), ("N0CALL", 1, 0));
        let h = header(&unhex("39b0c130c0000080140900000000000000000000000282f5b2")).expect("level 2 header");
        assert_eq!((h.call.as_str(), h.offer, h.compression), ("N0CALL", 2, 0));
        assert_eq!(call_hash("N0CALL"), 0xD80986);
        assert_eq!(unpack_call(&[0x04, 0x27, 0x03, 0x10, 0x00, 0x01]).as_deref(), Some("AB1CD-1"));
    }

    #[test]
    fn data_frames_are_not_headers() {
        let mut f = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0x03];
        let crc = crate::crc::crc16(&f);
        f.extend(crc.to_be_bytes());
        assert_eq!(header(&f), None);
    }
}
