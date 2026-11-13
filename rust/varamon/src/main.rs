//! Command line: decode recordings.

use std::path::Path;
use std::process::ExitCode;

use varamon_core::{audio, b2f, ctrl, frame, scan};

const USAGE: &str = "usage: varamon [--hex] RECORDING.wav ...

Receive-only monitor for VARA HF signals. Recordings are 16-bit WAV of the receiver's audio in USB.

  --hex       also print the bytes of each decoded frame
  --mail      show the headers and text of mail messages (off by default: they are other people's mail)
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
    let show_mail = args.iter().any(|a| a == "--mail");
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
                Some(d) if p.compressed => match varamon_core::textcomp::decompress(d).filter(|t| frame::is_text(t)) {
                    Some(t) => format!("text  \"{}\"  ({} bytes packed)", frame::text_of(&t), d.len()),
                    None => format!("data  {} bytes", d.len()),
                },
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
            if runs.is_empty() {
                println!("station {name}: {frames} frames, no payload");
                continue;
            }
            // the application stream: every run, with the modem's own compression undone
            let mut app: Vec<u8> = Vec::new();
            let mut notes: Vec<String> = Vec::new();
            for (compressed, frames_in_run) in &runs {
                let raw: Vec<u8> = frames_in_run.concat();
                if !compressed {
                    app.extend(raw);
                } else if let Some((out, mode)) = scan::unpack(frames_in_run) {
                    notes.push(format!("{} bytes unpacked to {} ({mode})", raw.len(), out.len()));
                    app.extend(out);
                } else {
                    notes.push(format!("{} compressed bytes could not be unpacked", raw.len()));
                    app.extend(raw);
                }
            }
            println!("station {name}: {frames} frames, {} bytes{}", app.len(), if notes.is_empty() { String::new() } else { format!("  [{}]", notes.join("; ")) });
            let show = |d: &[u8]| {
                if d.is_empty() {
                } else if frame::is_text(d) || d.iter().all(|&b| (0x20..0x7F).contains(&b) || b == b'\r' || b == b'\n') {
                    let wl = frame::winlink_commands(d);
                    let note = if wl > 0 { format!(", {wl} Winlink session commands") } else { String::new() };
                    println!("  text, {} bytes{note}:\n    {}", d.len(), frame::text_of(d).replace("\\n", "\n    "));
                } else {
                    let pieces = frame::text_runs(d, 8);
                    let extra = if pieces.is_empty() { String::new() } else { format!("; readable pieces: {}", pieces.join(" | ")) };
                    println!("  {} bytes, not plain text{extra}", d.len());
                }
            };
            let mut at = 0;
            for t in b2f::find_transfers(&app) {
                show(&app[at..t.start]);
                at = t.end;
                let state = match t.checksum_ok {
                    Some(true) => "block checksum correct",
                    Some(false) => "block checksum WRONG",
                    None => "incomplete",
                };
                match b2f::message(&t) {
                    Ok(m) => {
                        println!("  Winlink message: {} bytes packed, {} bytes unpacked, {state}, {} header lines, body {} bytes", t.packed.len(), m.size, m.headers.len(), m.body.len());
                        if show_mail {
                            println!("    Title: {}", t.title);
                            for (k, v) in &m.headers {
                                println!("    {k}: {v}");
                            }
                            println!("    ----\n    {}", String::from_utf8_lossy(&m.body).replace("\r\n", "\n").replace('\r', "\n").replace('\n', "\n    "));
                        }
                    }
                    Err(e) => println!("  Winlink message: {} bytes packed, {state}, not unpacked ({e:?})", t.packed.len()),
                }
            }
            show(&app[at..]);
        }
        println!("{} frames, {} decoded with correct CRC; {} control bursts\n", found.len(), found.iter().filter(|f| f.crc_ok).count(), events.len() - found.len());
    }
    status
}
