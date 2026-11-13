//! Command line: decode recordings.

use std::path::Path;
use std::process::ExitCode;

use varamon_core::{audio, frame, scan};

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
        let events = scan::scan(&x, &tables);
        let label = |s: usize| (b'A' + (s % 26) as u8) as char;
        for ev in &events {
            let f = match ev {
                scan::Event::Control(c) => {
                    println!("{:8.2} s  {:5.2} s  station {}  control burst ({:.0} dB), not decoded", c.time, c.seconds, label(c.station), c.level_db);
                    continue;
                }
                scan::Event::Frame(f) => f,
            };
            let head = format!("{:8.2} s  {:5.2} s  station {}  {:4} Hz L{:<2} ", f.frame_start, f.frame_end - f.frame_start, label(f.station), f.mode, f.level);
            if !f.crc_ok {
                println!("{head}not decoded (CRC failed, pattern match {:.2})", f.score);
                continue;
            }
            let Some(p) = frame::parse(&f.frame) else { continue };
            let flags = [(p.last, "last"), (p.compressed, "compressed")].iter().filter(|x| x.0).map(|x| x.1).collect::<Vec<_>>().join(", ");
            let content = match &p.data {
                None => "payload not terminated".to_string(),
                Some(d) if d.is_empty() => "empty".to_string(),
                Some(d) if frame::is_text(d) && !p.compressed => format!("text  \"{}\"", frame::text_of(d)),
                Some(d) => format!("data  {} bytes", d.len()),
            };
            println!("{head}{content}{}", if flags.is_empty() { String::new() } else { format!("  [{flags}]") });
            if hex {
                println!("            {}", f.frame.iter().map(|b| format!("{b:02x}")).collect::<String>());
            }
        }
        let found = scan::frames(&events);
        let stations = found.iter().map(|f| f.station + 1).max().unwrap_or(0);
        for s in 0..stations {
            let data = scan::stream_of(&found, s);
            let frames = found.iter().filter(|f| f.crc_ok && f.station == s).count();
            if frames == 0 {
                continue;
            }
            let name = label(s);
            if data.is_empty() {
                println!("station {name}: {frames} frames, no payload");
            } else if frame::is_text(&data) {
                println!("station {name}: {frames} frames, {} bytes of text:\n  {}", data.len(), frame::text_of(&data).replace("\\n", "\n  "));
            } else {
                let runs = frame::text_runs(&data, 8);
                println!("station {name}: {frames} frames, {} bytes, not plain text (compressed or binary){}", data.len(), if runs.is_empty() { String::new() } else { format!("; readable pieces: {}", runs.join(" | ")) });
            }
        }
        println!("{} frames, {} decoded with correct CRC; {} control bursts\n", found.len(), found.iter().filter(|f| f.crc_ok).count(), events.len() - found.len());
    }
    status
}
