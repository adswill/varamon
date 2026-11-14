#!/usr/bin/env python3
"""Compares the control-code tables with outside test vectors.

  python3 tools/check_codebooks.py VECTORS_DIR

VECTORS_DIR holds ctrl_codebooks_500.json, ctrl_codebooks_2300.json and ctrl_codebooks_2750.json:
for a station in a link (role, called station's hash, caller's hash), the tones of all 58 codes in
the three burst formats, for both directions. They are not part of this repository.
"""
import json, os, subprocess, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tool = os.path.join(here, "rust", "target", "release", "examples", "vectool")
SIDE = [50, 278]
def norm(t):
    return " ".join("+".join(map(str, sorted(map(int, x.split("+"))))) for x in t.split())
total = bad = skipped = 0
for bw in (500, 2300, 2750):
    for book in json.load(open(os.path.join(sys.argv[1], "ctrl_codebooks_%d.json" % bw)))["content"]["codebooks"]:
        if book["phase"] == "listening":
            continue
        seed, role = int(book["seed"], 16), book["role"]
        caller = int(book["connect_seed"], 16) if book["connect_seed"] else None
        # side0 = what the other station sends, side1 = what this station sends; role 1 = it made the call
        for side_name, sender in (("side0", 1 - role), ("side1", role)):
            sender_side = 0 if sender == 1 else 1          # 0 = the caller sends
            for f in range(3):
                args = [tool, "book", str(bw), "0", str(seed + 10 + SIDE[sender_side]), str(f)] + ([str(caller)] if caller is not None else [])
                mine = dict(l.split(" ", 1) for l in subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip().split("\n"))
                theirs = book["codes"][side_name]["fmt%d" % f]
                for code, tones in theirs.items():
                    c = int(code)
                    if c in (49, 50):
                        key = "conn%d_%d" % (sender_side, c)
                    elif c == 45:
                        skipped += 1          # not a stable value in the vectors
                        continue
                    elif c in (47, 48, 58):
                        key = "fixed%d_%d" % (sender_side, c)
                    elif 26 <= c <= 40:
                        skipped += 1          # depend on callsigns the vectors do not state per code
                        continue
                    else:
                        key = code
                    if key not in mine:
                        skipped += 1
                        continue
                    total += 1
                    if norm(mine[key]) != norm(tones):
                        bad += 1
                        if bad <= 12:
                            print("DIFFERENT %d Hz %s %s %s fmt%d code %s" % (bw, book["station"], book["phase"], side_name, f, code))
print("%d codes compared, %d different, %d not comparable" % (total, bad, skipped))
sys.exit(1 if bad else 0)
