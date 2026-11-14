#!/usr/bin/env python3
"""Compares the VARA FM generator with outside reference audio, sample for sample.

  python3 tools/check_fm.py VECTORS_DIR

VECTORS_DIR/fm/fm_tx.json holds, for transmissions of every FM record and for tone bursts, the
frames sent, the random lead-in cells, and the SHA-256 of the resulting 16-bit audio. Not part of
this repository. A matching SHA-256 means the frame coding, scrambling, constellation, symbol
shaping and level are all the same.
"""
import hashlib, json, os, subprocess, sys, tempfile
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tool = os.path.join(here, "rust", "target", "release", "examples", "vectool")
V = json.load(open(os.path.join(sys.argv[1], "fm", "fm_tx.json")))
V = V.get("content", V)
NC = {"NARROW": [14, 14, 29, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 14, 14, 14],
      "WIDE": [14, 14, 29, 58, 98, 98, 98, 98, 116, 116, 116, 116, 116, 116, 14, 14, 14]}
ok = bad = 0
def run(args, want, name):
    global ok, bad
    with tempfile.NamedTemporaryFile(suffix=".i16") as t:
        subprocess.run([tool] + args[:3] + [t.name] + args[3:], check=True)
        raw = open(t.name, "rb").read()
    good = hashlib.sha256(raw).hexdigest() == want["sha256_int16le"] and len(raw) // 2 == want["n_samples"]
    ok += good; bad += not good
    if not good:
        print("DIFFERENT", name, len(raw) // 2, want["n_samples"])
for group in ("levels", "l4"):
    for c in V[group]:
        draws = c.get("draws") or []
        lead = len(draws) // NC[c["mode"]][c["record"]]
        frames = c["frames_hex"] if isinstance(c["frames_hex"], list) else json.loads(c["frames_hex"].replace("'", '"'))
        run(["fmaudio", c["mode"].lower(), str(c["record"]), str(c["key"]), str(lead), ",".join(map(str, draws))] + frames, c, "%s %s record %d" % (group, c["mode"], c["record"]))
for c in V["ctrl"]:
    run(["fmtone", "0", str(c["code"]), c["upstream"], c["downstream"], str(c["lead"])], c, "tone code %d lead %d" % (c["code"], c["lead"]))
print("%d transmissions identical to the reference, %d different" % (ok, bad))
sys.exit(1 if bad else 0)
