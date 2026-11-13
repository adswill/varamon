"""Command line: decode recordings, or open the web page when no file is given."""
import argparse
import os
import sys

from . import __version__


def main(argv=None):
    p = argparse.ArgumentParser(prog="varamon", description="receive-only monitor for VARA HF signals")
    p.add_argument("files", nargs="*", help="16-bit WAV recordings of the receiver audio (USB)")
    p.add_argument("--hex", action="store_true", help="print the bytes of each decoded frame")
    p.add_argument("--ui", action="store_true", help="open the web page (default when no file is given)")
    p.add_argument("--port", type=int, default=8792)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--selftest", action="store_true", help="decode generated test signals and exit")
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)
    if a.selftest:
        from .selftest import run
        return run()
    if a.ui or not a.files:
        from .web import serve
        return serve(a.port, not a.no_browser)
    from . import hffsk, scan
    tables = scan.tables()
    status = 0
    for path in a.files:
        try:
            x = hffsk.load48(path)
        except (OSError, ValueError, EOFError) as e:
            print("%s: %s" % (path, e), file=sys.stderr)
            status = 1
            continue
        print("%s: %.1f s" % (os.path.basename(path), len(x) / 48000))
        found = scan.scan(x, tables)
        if a.hex:
            for q in found:
                if q["ok"]:
                    print("%8.2f s  %s" % (q["t"], q["frame"].hex()))
        print("%d transmissions, %d decoded with correct CRC\n" % (len(found), sum(q["ok"] for q in found)))
    return status


if __name__ == "__main__":
    sys.exit(main())
