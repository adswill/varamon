//! Command line: decode recordings.

use std::path::Path;
use std::process::ExitCode;

use varamon_core::levels::Bandwidth;
use varamon_core::{audio, b2f, ctrl, frame, scan, synth};

const USAGE: &str = "usage: varamon [--hex] [--mail] [--call CALLSIGN] RECORDING.wav ...

Receive-only monitor for VARA HF signals. Recordings are 16-bit WAV of the receiver's audio in USB.

  --hex       also print the bytes of each decoded frame
  --mail      show the headers and text of mail messages (off by default: they are other people's mail)
  --call CALLSIGN   a callsign you know (repeatable): calls to it and links it started are then named
  --selftest  generate a test signal for every level of every bandwidth mode and decode it
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
    if args.iter().any(|a| a == "--selftest") {
        return if selftest() { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    let hex = args.iter().any(|a| a == "--hex");
    let show_mail = args.iter().any(|a| a == "--mail");
    let calls: Vec<String> = args.windows(2).filter(|w| w[0] == "--call").map(|w| w[1].clone()).collect();
    let files: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--call")).map(|x| x.1).collect();
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
        let result = scan::scan_with(&x, &tables, &calls);
        let events = &result.events;
        let label = |s: usize| (b'A' + (s % 26) as u8) as char;
        for ev in events {
            let f = match ev {
                scan::Event::Control(c) => {
                    let what = match (c.code, c.format) {
                        _ if c.other.is_some() => c.other.clone().unwrap_or_default(),
                        (Some((code, hits, total)), _) => format!("{} (code {code}, {hits} of {total} symbols)", ctrl::code_name(code)),
                        (None, Some(_)) => "control burst, key not found".to_string(),
                        (None, None) => "short burst, not a control format".to_string(),
                    };
                    println!("{:8.2} s  {:5.2} s  station {}  {what}", c.time, c.seconds, label(c.station));
                    continue;
                }
                scan::Event::Frame(f) => f,
            };
            let what = Bandwidth::from_hz(f.mode).map(|b| b.levels()[f.level as usize - 1].describe()).unwrap_or_default();
            let head = format!("{:8.2} s  {:5.2} s  station {}  {:4} Hz L{:<2} ({what})  ", f.frame_start, f.frame_end - f.frame_start, label(f.station), f.mode, f.level);
            if !f.crc_ok {
                println!("{head}not decoded (CRC failed, pattern match {:.2})", f.score);
                continue;
            }
            if let Some(h) = frame::header(&f.frame) {
                let comp = ["no compression", "text compression", "files compression", "compression 3"][h.compression as usize];
                println!("{head}link start: callsign {}, {comp}, proposes level {}  (callsign hash {:06x})", h.call, h.offer, frame::call_hash(&h.call));
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
            // a callsign heard in a header frame may be this station's
            let mut heard: Vec<String> = found.iter().filter(|f| f.crc_ok).filter_map(|f| frame::header(&f.frame)).map(|h| h.call).collect();
            heard.extend(calls.iter().map(|c| c.to_uppercase()));
            let who = heard.iter().find_map(|c| {
                let h = frame::call_hash(c);
                (h == caller).then(|| format!("made the call to {c}")).or_else(|| (h == answerer).then(|| format!("is {c}, the station that was called")))
            });
            match who {
                Some(w) => println!("station {} {w}", label(*st)),
                None => println!("station {} key found: the called station's callsign hash is {caller:06x} if station {} made the call, {answerer:06x} if it was the one called", label(*st), label(*st)),
            }
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

/// Generate every record of every bandwidth mode with noise and a tuning offset, run each
/// through the scanner and compare the frames. Returns true if all come back exactly.
fn selftest() -> bool {
    let tables = scan::Tables::new();
    let mut all_ok = true;
    for bw in [Bandwidth::Hz500, Bandwidth::Hz2300, Bandwidth::Hz2750] {
        let recs = &tables.modes.iter().find(|m| m.0 == bw).expect("mode").1;
        let mut good = 0;
        for (i, level) in bw.levels().iter().enumerate() {
            // short records are sent three at a time, every fourth record two at a time
            let count = if level.frame_seconds() < 3.0 { 3 } else if i % 4 == 3 { 2 } else { 1 };
            let frames: Vec<Vec<u8>> = (0..count).map(|j| synth::test_frame(level, (i * 7 + j) as u32)).collect();
            let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
            let mut n = 99 + i as u32;
            let mut rand = |m: u32| {
                n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (n >> 16) % m
            };
            let offset = if i % 2 == 0 { 4.0 } else { -6.0 };
            let x = synth::on_air(&synth::transmission(level, &recs[i], &refs, &mut rand), 0.3, 1.0, offset, 0.03, 1000 + i as u64);
            let result = scan::scan(&x, &tables);
            let got: Vec<&scan::Found> = scan::frames(&result.events);
            let ok = got.len() == count && got.iter().zip(&frames).all(|(g, f)| g.crc_ok && g.mode == bw.hz() && g.level as usize == i + 1 && &g.frame == f);
            if ok {
                good += 1;
            } else {
                all_ok = false;
                let seen: Vec<String> = got.iter().map(|g| format!("{} Hz L{} {}", g.mode, g.level, if g.crc_ok { "CRC ok" } else { "CRC failed" })).collect();
                println!("  {} Hz record {:2} ({}, {} frame{}): FAILED, scanner saw [{}]", bw.hz(), i + 1, level.describe(), count, if count == 1 { "" } else { "s" }, seen.join(", "));
            }
        }
        println!("{} Hz mode: {good} of {} records decoded", bw.hz(), bw.levels().len());
    }
    all_ok
}
