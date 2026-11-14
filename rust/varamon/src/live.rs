//! Live listening: audio arrives in blocks, is cut at pauses and each stretch is scanned.

use std::io::Read;

use varamon_core::{audio, scan};

use crate::Printer;

/// Collects live audio and scans it stretch by stretch.
pub struct Listener {
    rate: usize,
    buf: Vec<f64>,
    /// samples (at the input rate) that came before `buf`
    before: usize,
    checked: usize,
    floor: f64,
    tables: scan::Tables,
    memory: scan::Memory,
    printer: Printer,
}

/// Loudness of each twentieth of a second.
fn blocks(x: &[f64], rate: usize) -> Vec<f64> {
    x.chunks_exact(rate / 20).map(|c| (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt()).collect()
}

impl Listener {
    pub fn new(rate: usize, calls: &[String], hex: bool) -> Self {
        Self { rate, buf: Vec::new(), before: 0, checked: 0, floor: f64::MAX, tables: scan::Tables::new(), memory: scan::Memory { calls: calls.to_vec(), ..Default::default() }, printer: Printer::new(hex, true) }
    }

    /// Add audio. A stretch is scanned once the signal has paused for 0.8 s, or after 25 s.
    pub fn feed(&mut self, samples: &[f64]) {
        self.buf.extend_from_slice(samples);
        if self.buf.len() - self.checked < self.rate / 2 {
            return;
        }
        self.checked = self.buf.len();
        let e = blocks(&self.buf, self.rate);
        if e.len() < 24 {
            return;
        }
        // the noise level: the quietest twentieth of a second heard lately. It may creep up (the
        // band gets noisier) but drops at once when something quieter is heard.
        let quietest = e.iter().fold(f64::MAX, |a, &v| a.min(v));
        self.floor = (self.floor * 1.02).min(quietest).max(1e-7);
        let floor = self.floor;
        let active: Vec<bool> = e.iter().map(|&v| v > 2.0 * floor).collect();
        let any = active.iter().filter(|&&a| a).count() >= 6;
        let paused = active[active.len() - 16..].iter().all(|&a| !a);
        let block = self.rate / 20;
        if any && paused {
            self.scan(self.buf.len(), self.rate / 2);
        } else if !any && self.buf.len() > 6 * self.rate {
            // nothing but noise: keep the last two seconds
            self.drop_to(2 * self.rate);
        } else if self.buf.len() > 12 * self.rate {
            // stations answering each other leave only short pauses: cut in the middle of the
            // last pause of 0.2 s or more, so no transmission is split. Without any pause the
            // stretch is scanned whole once it reaches 25 s.
            let mut cut = None;
            let mut run = 0;
            for (i, &a) in active.iter().enumerate() {
                if a {
                    if run >= 4 && i > run {
                        cut = Some((i - run / 2) * block);
                    }
                    run = 0;
                } else {
                    run += 1;
                }
            }
            match cut {
                Some(at) if at > 3 * self.rate => self.scan(at, self.rate / 10),
                _ if self.buf.len() > 25 * self.rate => self.scan(self.buf.len(), 0),
                _ => {}
            }
        }
    }

    /// Scan whatever is left when the input ends.
    pub fn finish(&mut self) {
        if self.buf.len() > self.rate {
            self.scan(self.buf.len(), 0);
        }
    }

    fn drop_to(&mut self, keep: usize) {
        let cut = self.buf.len().saturating_sub(keep);
        self.buf.drain(..cut);
        self.before += cut;
        self.checked = self.buf.len();
    }

    /// Scan the first `upto` samples, then forget them except for the last `keep`.
    fn scan(&mut self, upto: usize, keep: usize) {
        if std::env::var_os("VARAMON_DEBUG").is_some() {
            eprintln!("scan {:.1} s to {:.1} s", self.before as f64 / self.rate as f64, (self.before + upto) as f64 / self.rate as f64);
        }
        let x = audio::to_48k(&self.buf[..upto], self.rate);
        let result = scan::scan_live(&x, &self.tables, &mut self.memory);
        self.printer.base = self.before as f64 / self.rate as f64;
        self.printer.print(&result.events);
        let cut = upto.saturating_sub(keep);
        self.buf.drain(..cut);
        self.before += cut;
        self.checked = self.buf.len();
    }
}

/// Listen to raw audio on standard input: 16-bit little-endian, one channel, `rate` samples a second.
pub fn from_stdin(rate: usize, calls: &[String], hex: bool) -> std::io::Result<()> {
    let mut listener = Listener::new(rate, calls, hex);
    let mut input = std::io::stdin().lock();
    let mut bytes = vec![0u8; rate / 5 * 2];
    loop {
        let mut got = 0;
        while got < bytes.len() {
            match input.read(&mut bytes[got..])? {
                0 => break,
                n => got += n,
            }
        }
        let samples: Vec<f64> = bytes[..got - got % 2].chunks_exact(2).map(|b| f64::from(i16::from_le_bytes([b[0], b[1]]))).collect();
        listener.feed(&samples);
        if got < bytes.len() {
            listener.finish();
            return Ok(());
        }
    }
}

#[cfg(feature = "sound")]
mod device {
    use std::sync::mpsc;

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    use super::Listener;

    /// The names of the sound inputs.
    pub fn list() -> Vec<String> {
        cpal::default_host().input_devices().map(|d| d.filter_map(|x| x.name().ok()).collect()).unwrap_or_default()
    }

    /// Listen to a sound input until the program is stopped. `name` picks the first input whose
    /// name contains it; without it the system's default input is used.
    pub fn listen(name: Option<&str>, calls: &[String], hex: bool) -> Result<(), String> {
        let host = cpal::default_host();
        let device = match name {
            Some(n) => host
                .input_devices()
                .map_err(|e| e.to_string())?
                .find(|d| d.name().is_ok_and(|x| x.to_lowercase().contains(&n.to_lowercase())))
                .ok_or_else(|| format!("no sound input with \"{n}\" in its name; --devices lists them"))?,
            None => host.default_input_device().ok_or("no default sound input")?,
        };
        let config = device.default_input_config().map_err(|e| e.to_string())?;
        let (rate, channels) = (config.sample_rate().0 as usize, config.channels() as usize);
        let (tx, rx) = mpsc::channel::<Vec<f64>>();
        let fail = |e| eprintln!("sound input: {e}");
        // the first channel only: a receiver's audio is the same on both
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let tx = tx.clone();
                device.build_input_stream(&config.into(), move |d: &[f32], _: &_| drop(tx.send(d.iter().step_by(channels).map(|&v| f64::from(v) * 32768.0).collect())), fail, None)
            }
            cpal::SampleFormat::I16 => {
                let tx = tx.clone();
                device.build_input_stream(&config.into(), move |d: &[i16], _: &_| drop(tx.send(d.iter().step_by(channels).map(|&v| f64::from(v)).collect())), fail, None)
            }
            cpal::SampleFormat::U16 => {
                let tx = tx.clone();
                device.build_input_stream(&config.into(), move |d: &[u16], _: &_| drop(tx.send(d.iter().step_by(channels).map(|&v| f64::from(v) - 32768.0).collect())), fail, None)
            }
            other => return Err(format!("sample format {other:?} of this input is not supported")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        println!("listening to \"{}\" at {rate} Hz; stop with Ctrl-C", device.name().unwrap_or_default());
        let mut listener = Listener::new(rate, calls, hex);
        for block in rx {
            listener.feed(&block);
        }
        Ok(())
    }
}

#[cfg(feature = "sound")]
pub use device::{list, listen};
