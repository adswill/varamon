#!/usr/bin/env python3
"""Compares this project's coding chain with outside test vectors (frame bytes -> transmitted bits).

  python3 tools/check_vectors.py VECTORS_DIR

VECTORS_DIR holds frames_500.json, frames_2300.json, frames_2750.json and vara_frames.json: lists of
{mode, level, frame_hex, tx_hex}. They are not part of this repository.
"""
import json, os, subprocess, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tool = os.path.join(here, "rust", "target", "release", "examples", "vectool")
bad = total = 0
for name in ("frames_500.json", "frames_2300.json", "frames_2750.json", "vara_frames.json"):
    frames = json.load(open(os.path.join(sys.argv[1], name)))["content"]["frames"]
    for mode in sorted({f["mode"] for f in frames}):
        mine = [f for f in frames if f["mode"] == mode]
        out = subprocess.run([tool, "encode", str(mode), "0"], input="".join("%d %s\n" % (f["level"], f["frame_hex"]) for f in mine), capture_output=True, text=True, check=True).stdout.split("\n")
        for f, line in zip(mine, out):
            total += 1
            got = line.split()[1]
            if got != f["tx_hex"]:
                bad += 1
                print("DIFFERENT %s mode %d level %d %s" % (name, mode, f["level"], f["kind"]))
print("%d frames compared, %d different" % (total, bad))
sys.exit(1 if bad else 0)
