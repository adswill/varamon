//! The pseudo-random generator the waveform's tables are drawn from (Visual Basic's `Rnd`).

const MASK: u32 = 0x00FF_FFFF;

#[derive(Clone, Debug)]
pub struct VbRnd {
    state: u32,
}

impl Default for VbRnd {
    fn default() -> Self {
        Self::new()
    }
}

impl VbRnd {
    pub fn new() -> Self {
        Self { state: 0x05_0000 }
    }

    /// `Rnd(-1)` followed by `Randomize(seed)`: the start of every table stream.
    pub fn stream(seed: f64) -> Self {
        let mut r = Self::new();
        r.reseed();
        r.randomize(seed);
        r
    }

    pub fn state(&self) -> u32 {
        self.state
    }

    /// `Rnd(-1)`
    pub fn reseed(&mut self) {
        let b = (-1.0f32).to_bits();
        self.state = b.wrapping_add(b >> 24) & MASK;
        self.advance();
    }

    pub fn randomize(&mut self, seed: f64) {
        let hi = (seed.to_bits() >> 32) as u32;
        let mid = ((hi & 0xFFFF) ^ (hi >> 16)) & 0xFFFF;
        self.state = (self.state & 0xFF00_00FF & MASK) | (mid << 8);
    }

    pub fn advance(&mut self) {
        self.state = self.state.wrapping_mul(0x00FD_43FD).wrapping_add(0x00C3_9EC3) & MASK;
    }

    /// An integer in `0..m`.
    pub fn draw(&mut self, m: u32) -> u32 {
        self.advance();
        (self.state as f32 / 16_777_216.0 * m as f32) as u32
    }

    /// An integer in `0..=n`.
    pub fn rand_int(&mut self, n: u32) -> u32 {
        self.draw(n + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_states() {
        let mut r = VbRnd::new();
        r.advance();
        assert_eq!(r.state(), 0xB49EC3);
        let mut r = VbRnd::new();
        r.reseed();
        assert_eq!(r.state(), 0x395886);
        assert_eq!(VbRnd::stream(43.0).state(), 0xC04586);
    }

    #[test]
    fn known_draws() {
        let mut r = VbRnd::stream(292.0);
        let first: Vec<u32> = (0..4).map(|_| r.rand_int(128)).collect();
        assert_eq!(first, [78, 83, 81, 38]);
        let mut r = VbRnd::stream(292.0);
        let bits: String = (0..32).map(|_| char::from(b'0' + r.rand_int(1) as u8)).collect();
        assert_eq!(bits, "11101010111111100001100000101110");
    }
}
