//! Developer tool: prints what this crate produces, for comparison with outside test vectors.
//!   vectool encode BW 0 < lines of "LEVEL FRAMEHEX"      transmitted bits, packed MSB first, as hex
//!   vectool audio BW LEVEL OUT.f32 DRAWS FRAMEHEX...    the transmission as raw 32-bit floats; DRAWS = lead-in draws, comma separated
use varamon_core::levels::{encode, tables, Bandwidth};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bw = Bandwidth::from_hz(a[2].parse().unwrap()).unwrap();
    let rec: usize = a[3].parse().unwrap();
    let t = tables(bw);
    match a[1].as_str() {
        "encode" => {
            for line in std::io::stdin().lines() {
                let line = line.unwrap();
                let mut it = line.split_whitespace();
                let (lv, hex): (usize, &str) = (it.next().unwrap().parse().unwrap(), it.next().unwrap());
                let bits = encode(&bw.levels()[lv - 1], &t[lv - 1], &unhex(hex));
                let out: String = bits.chunks(8).map(|c| format!("{:02x}", c.iter().enumerate().fold(0u8, |a, (i, &b)| a | (b << (7 - i))))).collect();
                println!("{lv} {out}");
            }
            let _ = rec;
        }
        "audio" => {
            let draws: Vec<u32> = a[5].split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
            let frames: Vec<Vec<u8>> = a[6..].iter().map(|h| unhex(h)).collect();
            let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            let mut it = draws.into_iter();
            let mut rand = |m: u32| {
                let v = it.next().expect("ran out of draws");
                assert!(v < m);
                v
            };
            let x = varamon_core::synth::transmission(&bw.levels()[rec - 1], &t[rec - 1], &refs, &mut rand);
            assert!(it.next().is_none(), "unused draws");
            let bytes: Vec<u8> = x.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect();
            std::fs::write(&a[4], bytes).unwrap();
        }
        _ => panic!("unknown command"),
    }
}
