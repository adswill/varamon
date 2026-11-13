//! Command line: decode recordings.

use std::path::Path;
use std::process::ExitCode;

use varamon_core::{audio, ctrl, frame, scan};

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
        let result = scan::scan(&x, &tables);
        let events = &result.events;
        let label = |s: usize| (b'A' + (s % 26) as u8) as char;
        for ev in events {
            let f = match ev {
                scan::Event::Control(c) => {
                    let what = match (c.code, c.format) {
                        (Some((code, hits, total)), _) => format!("{} (code {code}, {hits} of {total} symbols)", ctrl::code_name(code)),
                        (None, Some(_)) => "control burst, key not found".to_string(),
                        (None, None) => "short burst, not a control format".to_string(),
                    };
                    println!("{:8.2} s  {:5.2} s  station {}  {what}", c.time, c.seconds, label(c.station));
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
        let found = scan::frames(events);
        for (st, key) in &result.keys {
            let (caller, answerer) = key.hashes();
            println!("station {} key found: callsign hash {caller:06x} if it made the call, {answerer:06x} if it answered", label(*st));
        }
        let stations = found.iter().map(|f| f.station + 1).max().unwrap_or(0);
        for s in 0..stations {
            let frames = found.iter().filter(|f| f.crc_ok && f.station == s).count();
            if frames == 0 {
                continue;
            }
            let name = label(s);
            let runs = scan::stream_of(&found, s);
            if runs.iter().all(|r| r.1.is_empty()) {
                println!("station {name}: {frames} frames, no payload");
                continue;
            }
            println!("station {name}: {frames} frames");
            for (compressed, data) in runs.iter().filter(|r| !r.1.is_empty()) {
                let show = |what: &str, d: &[u8]| {
                    if frame::is_text(d) {
                        let wl = frame::winlink_commands(d);
                        let note = if wl > 0 { format!(", {wl} Winlink session commands") } else { String::new() };
                        println!("  {what}: {} bytes of text{note}:\n    {}", d.len(), frame::text_of(d).replace("\\n", "\n    "));
                    } else {
                        let pieces = frame::text_runs(d, 8);
                        let extra = if pieces.is_empty() { String::new() } else { format!("; readable pieces: {}", pieces.join(" | ")) };
                        println!("  {what}: {} bytes, not plain text{extra}", d.len());
                    }
                };
                if !compressed {
                    show("stream", data);
                } else {
                    match frame::unpack_files(data) {
                        Some((out, clean)) => show(&format!("unpacked from {} compressed bytes{}", data.len(), if clean { "" } else { " (stream cut short)" }), &out),
                        None => println!("  compressed stream: {} bytes, not unpacked (text-dictionary compression, or the start is missing)", data.len()),
                    }
                }
            }
        }
        println!("{} frames, {} decoded with correct CRC; {} control bursts\n", found.len(), found.iter().filter(|f| f.crc_ok).count(), events.len() - found.len());
    }
    status
}
