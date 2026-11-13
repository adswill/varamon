//! The turbo code: two identical 8-state recursive encoders, the second fed through a permutation.
//! Systematic bits are always sent; the parities are punctured with period `p`; each encoder is
//! terminated with three tail steps. Decoding is iterative max-log-MAP on soft values (positive = 0).

const STATES: usize = 8;
const NEG: f32 = -1.0e9;

/// For state `s` and input `u`: (next state, parity bit).
const fn step(s: usize, u: usize) -> (usize, usize) {
    let (r0, r1, r2) = (s >> 2, (s >> 1) & 1, s & 1);
    let a = u ^ r0 ^ r1;
    ((r1 << 2) | (r2 << 1) | a, a ^ r2 ^ r0)
}

/// The input that drives the encoder towards state 0.
const fn tail_input(s: usize) -> usize {
    (s >> 2) ^ ((s >> 1) & 1)
}

fn rsc(bits: impl Iterator<Item = u8>) -> (Vec<u8>, Vec<u8>) {
    let (mut x, mut p, mut s) = (Vec::new(), Vec::new(), 0usize);
    let mut push = |u: usize, s: &mut usize| {
        let (ns, par) = step(*s, u);
        x.push(u as u8);
        p.push(par as u8);
        *s = ns;
    };
    for u in bits {
        push(usize::from(u), &mut s);
    }
    for _ in 0..3 {
        let u = tail_input(s);
        push(u, &mut s);
    }
    (x, p)
}

fn has_p1(t: usize, p: usize) -> bool {
    p == 1 || t % p == 0
}

fn has_p2(t: usize, p: usize) -> bool {
    p == 1 || t % p == p / 2
}

/// Encode `info` into `n` coded bits (zero padded).
pub fn encode(info: &[u8], perm: &[u32], p: u32, n: usize) -> Vec<u8> {
    let (k, p) = (info.len(), p as usize);
    let (x1, p1) = rsc(info.iter().copied());
    let (x2, p2) = rsc(perm.iter().map(|&i| info[i as usize]));
    let mut out = Vec::with_capacity(n);
    for t in 0..k {
        out.push(x1[t]);
        if has_p1(t, p) {
            out.push(p1[t]);
        }
        if has_p2(t, p) {
            out.push(p2[t]);
        }
    }
    for t in k..k + 3 {
        out.extend([x1[t], p1[t], x2[t], p2[t]]);
    }
    assert!(out.len() <= n, "coded length {} exceeds {n}", out.len());
    out.resize(n, 0);
    out
}

/// One soft-in soft-out pass. Returns (extrinsic, posterior) for the `k` information steps.
fn siso(ls: &[f32], lp: &[f32], la: &[f32], tail_s: &[f32; 3], tail_p: &[f32; 3]) -> (Vec<f32>, Vec<f32>) {
    let k = ls.len();
    let total = k + 3;
    // branch metric for step t, state s, input u
    let gamma = |t: usize, s: usize, u: usize| -> f32 {
        let (_, par) = step(s, u);
        let sign = |b: usize| if b == 0 { 0.5 } else { -0.5 };
        if t < k {
            sign(u) * (ls[t] + la[t]) + sign(par) * lp[t]
        } else if u == tail_input(s) {
            sign(u) * tail_s[t - k] + sign(par) * tail_p[t - k]
        } else {
            NEG
        }
    };
    let mut alpha = vec![[NEG; STATES]; total + 1];
    alpha[0][0] = 0.0;
    for t in 0..total {
        let mut next = [NEG; STATES];
        for s in 0..STATES {
            for u in 0..2 {
                let v = alpha[t][s] + gamma(t, s, u);
                let ns = step(s, u).0;
                if v > next[ns] {
                    next[ns] = v;
                }
            }
        }
        let m = next.iter().cloned().fold(NEG, f32::max);
        alpha[t + 1] = next.map(|v| v - m);
    }
    let mut beta = vec![[NEG; STATES]; total + 1];
    beta[total][0] = 0.0;
    for t in (0..total).rev() {
        let mut cur = [NEG; STATES];
        for s in 0..STATES {
            for u in 0..2 {
                let v = gamma(t, s, u) + beta[t + 1][step(s, u).0];
                if v > cur[s] {
                    cur[s] = v;
                }
            }
        }
        let m = cur.iter().cloned().fold(NEG, f32::max);
        beta[t] = cur.map(|v| v - m);
    }
    let mut ext = Vec::with_capacity(k);
    let mut post = Vec::with_capacity(k);
    for t in 0..k {
        let mut best = [NEG; 2];
        for s in 0..STATES {
            for u in 0..2 {
                let v = alpha[t][s] + gamma(t, s, u) + beta[t + 1][step(s, u).0];
                if v > best[u] {
                    best[u] = v;
                }
            }
        }
        let l = best[0] - best[1];
        post.push(l);
        ext.push(l - ls[t] - la[t]);
    }
    (ext, post)
}

/// Decode `k` information bits from soft values in coded order (after de-interleaving).
pub fn decode(llr: &[f32], perm: &[u32], p: u32, k: usize, iterations: usize) -> Vec<u8> {
    let p = p as usize;
    let (mut ls, mut l1, mut l2) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    let mut pos = 0;
    for t in 0..k {
        ls[t] = llr[pos];
        pos += 1;
        if has_p1(t, p) {
            l1[t] = llr[pos];
            pos += 1;
        }
        if has_p2(t, p) {
            l2[t] = llr[pos];
            pos += 1;
        }
    }
    let tail = |i: usize| -> [f32; 3] { [llr[pos + i], llr[pos + 4 + i], llr[pos + 8 + i]] };
    let (ts1, tp1, ts2, tp2) = (tail(0), tail(1), tail(2), tail(3));
    let ls2: Vec<f32> = perm.iter().map(|&i| ls[i as usize]).collect();
    let mut la = vec![0.0f32; k];
    let mut post2 = vec![0.0f32; k];
    for _ in 0..iterations {
        let (e1, _) = siso(&ls, &l1, &la, &ts1, &tp1);
        let la2: Vec<f32> = perm.iter().map(|&i| e1[i as usize]).collect();
        let (e2, post) = siso(&ls2, &l2, &la2, &ts2, &tp2);
        for (j, &i) in perm.iter().enumerate() {
            la[i as usize] = e2[j];
        }
        post2 = post;
    }
    let mut out = vec![0u8; k];
    for (j, &i) in perm.iter().enumerate() {
        out[i as usize] = u8::from(post2[j] < 0.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{fsk_tables, BW2300};

    fn fnv(vals: &[u8]) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        for &v in vals {
            for b in u32::from(v).to_le_bytes() {
                h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
            }
        }
        h
    }

    fn test_bits(k: usize) -> Vec<u8> {
        let mut x: u32 = 12345;
        (0..k)
            .map(|_| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7FFF_FFFF;
                ((x >> 16) & 1) as u8
            })
            .collect()
    }

    /// The encoder output matches the Python reference (level 2 and level 3 tables).
    #[test]
    fn encoder_matches_reference() {
        let t = fsk_tables(&BW2300);
        let info = test_bits(200);
        let c = encode(&info, &t[1].turbo, 1, 612);
        assert_eq!((c.len(), fnv(&c)), (612, 0xf6e3c625));
        let mut info3 = info.clone();
        info3.extend(&info);
        info3.extend([0, 1]);
        assert_eq!(fnv(&encode(&info3, &t[2].turbo, 2, 816)), 0xf4605864);
    }

    /// Clean and noisy round trips at both code rates.
    #[test]
    fn round_trip_with_noise() {
        let t = fsk_tables(&BW2300);
        let mut seed: u32 = 99;
        let mut noise = move || {
            // sum of 12 uniforms: close to a unit normal
            let mut s = 0.0f32;
            for _ in 0..12 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                s += (seed >> 8) as f32 / 16_777_216.0;
            }
            s - 6.0
        };
        for (lvl, p, n, sigma) in [(1usize, 1u32, 612usize, 0.9f32), (2, 2, 816, 0.7)] {
            let k = t[lvl].level.k as usize;
            let info = test_bits(k);
            let coded = encode(&info, &t[lvl].turbo, p, n);
            let llr: Vec<f32> = coded.iter().map(|&b| 2.0 * ((1.0 - 2.0 * f32::from(b)) + sigma * noise()) / (sigma * sigma)).collect();
            let raw_errors = llr.iter().zip(&coded).filter(|(l, &b)| (**l < 0.0) != (b == 1)).count();
            assert!(raw_errors > n / 20, "the test should start from a noisy signal");
            assert_eq!(decode(&llr, &t[lvl].turbo, p, k, 8), info, "level {} with {raw_errors} raw bit errors", lvl + 1);
        }
    }
}
