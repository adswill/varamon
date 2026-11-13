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

    pub fn from_state(state: u32) -> Self {
        Self { state: state & MASK }
    }

    /// The state after `Rnd(-1)`, `Randomize(x mod 32768)` and `x div 32768 + 1` steps: how the
    /// control codebooks are seeded.
    pub fn seeded(x: u64) -> Self {
        let mut r = Self::new();
        r.reseed();
        r.randomize((x % 32768) as f64);
        for _ in 0..=x / 32768 {
            r.advance();
        }
        r
    }

    /// One step backwards.
    pub fn retreat(&mut self) {
        const MUL_INV: u32 = {
            let mut x: u32 = 0x00FD_43FD;
            let mut i = 0;
            while i < 5 {
                x = x.wrapping_mul(2u32.wrapping_sub(0x00FD_43FDu32.wrapping_mul(x)));
                i += 1;
            }
            x & MASK
        };
        self.state = self.state.wrapping_sub(0x00C3_9EC3).wrapping_mul(MUL_INV) & MASK;
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
    fn retreat_undoes_advance() {
        let mut r = VbRnd::from_state(0x123456);
        r.advance();
        r.retreat();
        assert_eq!(r.state(), 0x123456);
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
