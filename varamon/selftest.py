"""Self-test on generated signals: one frame each of levels 2, 6, 11 and 15, scanned and compared."""
from . import scan


def run(verbose=True):
    x, sent = scan.mix()
    found = scan.scan(x, scan.tables(), report=(print if verbose else (lambda *a: None)))
    good = sum(any(q["ok"] and q["frame"] == fr and q["level"] == lv for q in found) for lv, fr in sent)
    if verbose:
        print("generated frames recovered: %d of %d" % (good, len(sent)))
    return 0 if good == len(sent) else 1
