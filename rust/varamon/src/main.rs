//! Command line: decode recordings.

mod live;

use std::path::Path;
use std::process::ExitCode;

use varamon_core::levels::Bandwidth;
use varamon_core::{audio, b2f, ctrl, frame, scan, synth};

const USAGE: &str = "usage: varamon [--hex] [--mail] [--call CALLSIGN] RECORDING.wav ...
       varamon --live [NAME]
       varamon --stdin RATE

Receive-only monitor for VARA HF signals. Recordings are 16-bit WAV of the receiver's audio in USB.

  --hex       also print the bytes of each decoded frame
  --mail      show the headers and text of mail messages (off by default: they are other people's mail)
  --call CALLSIGN   a callsign you know (repeatable): calls to it and links it started are then named
  --live [NAME]     listen to a sound input (the default one, or the first whose name contains NAME)
  --devices   list the sound inputs
  --stdin RATE      listen to raw audio on standard input: 16-bit, one channel, RATE samples a second
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
    let value_of = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).filter(|v| !v.starts_with("--"));
    if args.iter().any(|a| a == "--stdin") {
        let Some(rate) = value_of("--stdin").and_then(|r| r.parse::<usize>().ok()).filter(|r| (8000..=384_000).contains(r)) else {
            eprintln!("--stdin needs the sample rate, for example --stdin 48000");
            return ExitCode::from(2);
        };
        return match live::from_stdin(rate, &calls, hex) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("standard input: {e}");
                ExitCode::FAILURE
            }
        };
    }
    #[cfg(feature = "sound")]
    {
        if args.iter().any(|a| a == "--devices") {
            for name in live::list() {
                println!("{name}");
            }
            return ExitCode::SUCCESS;
        }
        if args.iter().any(|a| a == "--live") {
            return match live::listen(value_of("--live").map(|s| s.as_str()), &calls, hex) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::FAILURE
                }
            };
        }
    }
    #[cfg(not(feature = "sound"))]
    if args.iter().any(|a| a == "--live" || a == "--devices") {
        eprintln!("this build has no sound input; pipe audio in with --stdin RATE");
        return ExitCode::from(2);
    }
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
        let mut printer = Printer::new(hex, false);
        printer.print(events);
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

/// Prints events one line each. Keeps what must be remembered between events: the fragments of an
/// unconnected message.
struct Printer {
    hex: bool,
    /// live mode: times are counted from the start of listening and stations are not labelled
    live: bool,
    /// seconds before the audio the events' times refer to
    base: f64,
    fragments: Vec<Vec<u8>>,
    announced: Option<f64>,
}

fn label(s: usize) -> char {
    (b'A' + (s % 26) as u8) as char
}

impl Printer {
    fn new(hex: bool, live: bool) -> Self {
        Self { hex, live, base: 0.0, fragments: Vec::new(), announced: None }
    }

    fn when(&self, t: f64) -> String {
        let t = self.base + t;
        if self.live {
            format!("{:02}:{:02}:{:04.1}", (t / 3600.0) as u32, (t / 60.0) as u32 % 60, t % 60.0)
        } else {
            format!("{t:8.2} s")
        }
    }

    fn who(&self, station: usize) -> String {
        if self.live {
            String::new()
        } else {
            format!("station {}  ", label(station))
        }
    }

    fn print(&mut self, events: &[scan::Event]) {
        // frames that follow an announcement burst are unconnected traffic: their fragments are joined
        for ev in events {
            if let (Some(t), scan::Event::Frame(f)) = (self.announced, ev) {
                if self.base + f.frame_start - t > 12.0 {
                    self.announced = None;
                    self.fragments.clear();
                }
            }
            let f = match ev {
                scan::Event::Control(c) => {
                    let what = match (c.code, c.format) {
                        _ if c.other.is_some() => c.other.clone().unwrap_or_default(),
                        (Some((code, hits, total)), _) => format!("{} (code {code}, {hits} of {total} symbols)", ctrl::code_name(code)),
                        (None, Some(_)) => "control burst, key not found".to_string(),
                        (None, None) => "short burst, not a control format".to_string(),
                    };
                    println!("{}  {:5.2} s  {}{what}", self.when(c.time), c.seconds, self.who(c.station));
                    if c.announces.is_some() {
                        self.announced = Some(self.base + c.time + c.seconds);
                        self.fragments.clear();
                    }
                    continue;
                }
                scan::Event::Frame(f) => f,
            };
            let what = Bandwidth::from_hz(f.mode).map(|b| b.levels()[f.level as usize - 1].describe()).unwrap_or_default();
            let head = format!("{}  {:5.2} s  {}{:4} Hz L{:<2} ({what})  ", self.when(f.frame_start), f.frame_end - f.frame_start, self.who(f.station), f.mode, f.level);
            if !f.crc_ok {
                println!("{head}not decoded (CRC failed, pattern match {:.2})", f.score);
                continue;
            }
            if f.mode == 500 && f.level == 15 {
                match frame::cq(&f.frame) {
                    Some((call, mode)) => println!("{head}CQ from {call}{}", mode.map_or(String::new(), |m| format!(", listening in the {m} Hz mode"))),
                    None => println!("{head}CQ frame, callsign not readable"),
                }
                continue;
            }
            if let (Some(_), Some(data)) = (self.announced, frame::parse(&f.frame).and_then(|p| p.data)) {
                self.announced = Some(self.base + f.frame_end);
                self.fragments.push(data);
                let more = self.fragments.last().and_then(|d| d.first()).is_some_and(|b| b & 0x80 != 0);
                match frame::unconnected(&self.fragments) {
                    Some(frame::Unconnected::Text(t)) => println!("{head}unconnected message, {} fragment(s): \"{}\"", self.fragments.len(), frame::text_of(&t)),
                    Some(frame::Unconnected::Packet(p)) => println!("{head}unconnected AX.25 packet, {} fragment(s), {} bytes packed (not unpacked)", self.fragments.len(), p.len()),
                    None if more => println!("{head}unconnected message, fragment {}", self.fragments.len()),
                    None => println!("{head}unconnected fragment that does not fit the others"),
                }
                if !more {
                    self.announced = None;
                    self.fragments.clear();
                }
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
            if self.hex {
                println!("            {}", f.frame.iter().map(|b| format!("{b:02x}")).collect::<String>());
            }
        }
    }
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
