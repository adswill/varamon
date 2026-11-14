#!/usr/bin/env python3
"""Compares the test-signal generator with outside reference audio (samples of real transmissions).

  python3 tools/check_audio.py VECTORS_DIR

VECTORS_DIR holds audio_2300.json (every 97th sample of each reference transmission) and
xara_audio_draws.json (the frames and lead-in symbols of each). They are not part of this repository.
Prints, per case, how far the generated audio is from the reference after fitting one gain, in dB.
"""
import json, os, struct, subprocess, sys, tempfile, math
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tool = os.path.join(here, "rust", "target", "release", "examples", "vectool")
V = sys.argv[1]
refs = json.load(open(os.path.join(V, "audio_2300.json")))["content"]
cases = json.load(open(os.path.join(V, "xara_audio_draws.json")))["content"]
KIND = {1: "fsk", 2: "fsk", 3: "fsk", 4: "fsk", 5: "psk", 6: "psk", 7: "psk", 8: "psk", 9: "psk"}
LEAD = {1: 2, 2: 4, 3: 6, 4: 12}
NC = {5: 3, 6: 4, 7: 6, 8: 8, 9: 10}
def order(level, nframes, old):
    kind = KIND.get(level, "ofdm")
    if kind == "fsk":
        return old * nframes
    if kind == "psk":
        nc, lead = NC[level], 8
        block = lead + nframes
        return [old[c * block + lead + f] for f in range(nframes) for c in range(nc)] + [old[c * block + k] for k in range(lead) for c in range(nc)]
    nc, lead = 49, (4 if level < 15 else 5)
    one = [old[r * nc + b] for b in range(nc) for r in range(lead)]
    return one * (3 if nframes > 1 else 1)
worst = -999
for name, case in cases.items():
    level, frames = case["level"], case["frames_hex"]
    draws = order(level, len(frames), case["draws"])
    with tempfile.NamedTemporaryFile(suffix=".f32") as tmp:
        subprocess.run([tool, "audio", "2300", str(level), tmp.name, ",".join(map(str, draws))] + frames, check=True)
        raw = open(tmp.name, "rb").read()
    x = struct.unpack("<%df" % (len(raw) // 4), raw)
    ref = refs[name]
    fp, every, n = ref["fingerprint_int16"], ref["fingerprint_every"], ref["n_samples"]
    kind = KIND.get(level, "ofdm")
    sym = {1: 2048, 2: 1024, 3: 1024, 4: 512}.get(level, 0)
    skip, tail = (LEAD[level] * sym, sym) if kind == "fsk" else ((8 * 512 + 2560, 4096) if kind == "psk" else (240, 1152))
    pts = [(x[j * every], fp[j]) for j in range(len(fp)) if skip <= j * every < len(x) - tail]
    g = sum(a * b for a, b in pts) / sum(a * a for a, _ in pts)
    err = sum((b - g * a) ** 2 for a, b in pts) / sum(b * b for _, b in pts)
    db = 10 * math.log10(err + 1e-30)
    worst = max(worst, db)
    print("%-16s level %2d  %d frame(s)  samples %7d (reference %7d)  difference %6.1f dB" % (name, level, len(frames), len(x), n, db))
print("worst case %.1f dB" % worst)
