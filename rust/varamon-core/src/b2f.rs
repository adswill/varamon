//! Winlink mail transfers (B2F). After the text commands, a message travels as binary blocks:
//! a header block (0x01, length, title, NUL, offset, NUL), data blocks (0x02, length, data; a
//! length byte of 0 means 256) and an end block (0x04, checksum). The data is an LZHUF "B2"
//! stream holding the message: header lines, an empty line, then the body.

use crate::lzhuf;

#[derive(Clone, Debug)]
pub struct Transfer {
    /// where the transfer sits in the stream
    pub start: usize,
    pub end: usize,
    pub title: String,
    pub offset: usize,
    /// the compressed message
    pub packed: Vec<u8>,
    /// the end block was seen and its checksum fits; `None` if the stream ends before it
    pub checksum_ok: Option<bool>,
}

/// Find the transfers in a stream of application bytes.
pub fn find_transfers(stream: &[u8]) -> Vec<Transfer> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 2 < stream.len() {
        if stream[i] == 0x01 {
            if let Some(t) = parse_at(stream, i) {
                i = t.end;
                out.push(t);
                continue;
            }
        }
        i += 1;
    }
    out
}

fn parse_at(s: &[u8], start: usize) -> Option<Transfer> {
    let len = usize::from(*s.get(start + 1)?);
    let head = s.get(start + 2..start + 2 + len)?;
    // title NUL offset NUL
    let mut parts = head.split(|&b| b == 0);
    let title = parts.next()?;
    let offset = parts.next()?;
    if parts.next() != Some(&[][..]) || parts.next().is_some() || title.is_empty() || title.len() > 80 {
        return None;
    }
    if offset.is_empty() || offset.len() > 6 || !offset.iter().all(u8::is_ascii_digit) || !title.iter().all(|&b| (0x20..0x7F).contains(&b)) {
        return None;
    }
    let mut pos = start + 2 + len;
    let mut packed = Vec::new();
    let mut checksum_ok = None;
    loop {
        match s.get(pos) {
            Some(0x02) => {
                let n = match *s.get(pos + 1)? {
                    0 => 256,
                    n => usize::from(n),
                };
                match s.get(pos + 2..pos + 2 + n) {
                    Some(d) => packed.extend_from_slice(d),
                    None => {
                        // the stream ends inside this block: keep what there is
                        packed.extend_from_slice(&s[(pos + 2).min(s.len())..]);
                        pos = s.len();
                        break;
                    }
                }
                pos += 2 + n;
            }
            Some(0x04) => {
                let sum = packed.iter().fold(0u8, |a, &b| a.wrapping_add(b));
                checksum_ok = Some(sum.wrapping_add(*s.get(pos + 1)?) == 0);
                pos += 2;
                break;
            }
            None => break,
            Some(_) => return None,
        }
    }
    if packed.is_empty() {
        return None;
    }
    Some(Transfer {
        start,
        end: pos,
        title: String::from_utf8_lossy(title).into_owned(),
        offset: std::str::from_utf8(offset).ok()?.parse().ok()?,
        packed,
        checksum_ok,
    })
}

#[derive(Clone, Debug)]
pub struct Message {
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// size of the whole unpacked message
    pub size: usize,
}

impl Message {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|h| h.0.eq_ignore_ascii_case(name)).map(|h| h.1.as_str())
    }
}

/// Unpack a transfer's data into the message.
pub fn message(t: &Transfer) -> Result<Message, lzhuf::Error> {
    let raw = lzhuf::unpack_b2(&t.packed)?;
    // a mail message starts with "Mid:" header lines and an empty line; anything else is all body
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").filter(|_| raw.starts_with(b"Mid:"));
    let (head, body) = match split {
        Some(i) => (&raw[..i], &raw[i + 4..]),
        None => (&[][..], &raw[..]),
    };
    let headers = String::from_utf8_lossy(head)
        .split("\r\n")
        .filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
        .collect();
    Ok(Message { headers, body: body.to_vec(), size: raw.len() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap(title: &str, packed: &[u8]) -> Vec<u8> {
        let mut s = vec![0x01, (title.len() + 3) as u8];
        s.extend(title.as_bytes());
        s.extend([0, b'0', 0]);
        for c in packed.chunks(250) {
            s.extend([0x02, c.len() as u8]);
            s.extend(c);
        }
        let sum = packed.iter().fold(0u8, |a, &b| a.wrapping_add(b));
        s.extend([0x04, sum.wrapping_neg()]);
        s
    }

    #[test]
    fn transfer_in_a_session() {
        let packed = include_bytes!("../tests/data/gettysburg.txt.lzh");
        let mut stream = b"FC EM ABCDEFGHIJKL 1548 861 0\rF> 2B\r".to_vec();
        let at = stream.len();
        stream.extend(wrap("Test message", packed));
        stream.extend(b"FF\r");
        let found = find_transfers(&stream);
        assert_eq!(found.len(), 1);
        let t = &found[0];
        assert_eq!((t.start, t.end, t.title.as_str(), t.offset, t.checksum_ok), (at, stream.len() - 3, "Test message", 0, Some(true)));
        assert_eq!(t.packed, packed);
        assert_eq!(lzhuf::unpack_b2(&t.packed).unwrap(), include_bytes!("../tests/data/gettysburg.txt"));
        // cut short: reported as incomplete, and the unpacker refuses it
        let cut = find_transfers(&stream[..at + 300]);
        assert_eq!((cut.len(), cut[0].checksum_ok), (1, None));
        assert!(lzhuf::unpack_b2(&cut[0].packed).is_err());
    }

    #[test]
    fn ordinary_bytes_are_not_a_transfer() {
        assert!(find_transfers(b"\x01\x05hello world, nothing here").is_empty());
        assert!(find_transfers(&[0x79, 0x62, 0x34, 0xd9, 0x01, 0x02, 0x03]).is_empty());
    }
}
