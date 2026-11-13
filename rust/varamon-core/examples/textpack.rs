//! Test helper: pack text with the modem's text compression into frame payloads of a given size.
//! usage: textpack ROOM "text with \r for line ends"   -> one hex line per frame payload

use varamon_core::textcomp::{compress, START_STATE};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let room: usize = args[1].parse().expect("room in bytes");
    let text = args[2].replace("\\r", "\r").into_bytes();
    let (mut pos, mut state) = (0, START_STATE);
    while pos < text.len() {
        let (bytes, used, next) = compress(&text[pos..], room, state);
        println!("{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>());
        pos += used;
        state = next;
    }
}
