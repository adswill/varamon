//! Command line: decode recordings.

use std::path::Path;
use std::process::ExitCode;

use varamon_core::{audio, scan};

const USAGE: &str = "usage: varamon [--hex] RECORDING.wav ...

Receive-only monitor for VARA HF signals. Recordings are 16-bit WAV of the receiver's audio in USB.

  --hex       also print the bytes of each decoded frame
  --version   print the version
  --help      print this text";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|a| a == "--version") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let hex = args.iter().any(|a| a == "--hex");
    let files: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if files.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let tables = scan::Tables::new();
    let mut status = ExitCode::SUCCESS;
    for path in files {
        let x = match audio::load(Path::new(path)) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("{path}: {e}");
                status = ExitCode::FAILURE;
                continue;
            }
        };
        let name = Path::new(path).file_name().map_or(path.as_str().into(), |n| n.to_string_lossy());
        println!("{name}: {:.1} s", x.len() as f64 / audio::RATE as f64);
        let found = scan::scan(&x, &tables, |f| {
            let c = if f.frame.len() >= 3 { f.frame[f.frame.len() - 3] } else { 0 };
            println!(
                "{:8.2} s  {:5.2} s  fsk  {:4} Hz level {:2}  {:4} bytes  CRC {:<5}  control {:02x} (reply {}, compressed {}, last {}, full {})  score {:.2}  offset {:+.0} Hz",
                f.time,
                f.seconds,
                f.mode,
                f.level,
                f.frame.len(),
                if f.crc_ok { "OK" } else { "wrong" },
                c,
                c >> 6,
                (c >> 5) & 1,
                (c >> 1) & 1,
                c & 1,
                f.score,
                f.offset
            );
            if hex && f.crc_ok {
                println!("            {}", f.frame.iter().map(|b| format!("{b:02x}")).collect::<String>());
            }
        });
        println!("{} transmissions, {} decoded with correct CRC\n", found.len(), found.iter().filter(|f| f.crc_ok).count());
    }
    status
}
