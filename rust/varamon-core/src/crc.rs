//! Frame check: CRC-16 with polynomial 0x1021, start value 0xFFFF, result inverted, sent high byte first.

pub fn crc16(data: &[u8]) -> u16 {
    let mut c: u16 = 0xFFFF;
    for &b in data {
        c ^= u16::from(b) << 8;
        for _ in 0..8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x1021 } else { c << 1 };
        }
    }
    !c
}

/// True if the last two bytes of `frame` are the CRC of the rest.
pub fn frame_ok(frame: &[u8]) -> bool {
    frame.len() > 2 && {
        let (body, tail) = frame.split_at(frame.len() - 2);
        crc16(body) == u16::from_be_bytes([tail[0], tail[1]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_value() {
        assert_eq!(crc16(b"123456789"), 0xD64E);
    }

    #[test]
    fn frame_check() {
        let mut f = b"hello".to_vec();
        f.extend(crc16(b"hello").to_be_bytes());
        assert!(frame_ok(&f));
        f[0] ^= 1;
        assert!(!frame_ok(&f));
    }
}
