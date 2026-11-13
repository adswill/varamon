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
    fn text_detection() {
        assert!(is_text(b"HELLO DE TEST\r\n"));
        assert!(!is_text(&[0x79, 0x62, 0x34, 0xd9, 0x7a]));
        assert_eq!(text_of(b"A\r\nB\tC"), "A\\nB C");
        assert_eq!(text_runs(b"\x01\x02CALLSIGN\x00ab\x05", 4), vec!["CALLSIGN".to_string()]);
    }
}
