"""Rig check before a session: python3 -m camsync.preflight [--seconds 30]

Checks that every named camera is connected, settings were accepted, and that all
cameras stream at full rate without losing frames. Exit code 1 if anything fails.
"""
import argparse
import os
import sys
import time

from . import config
from .rig import Rig

MIN_FPS = 118.0


def main():
    p = argparse.ArgumentParser(description="Check the camera rig before a session.")
    p.add_argument("--config", default=config.DEFAULT_PATH)
    p.add_argument("--seconds", type=float, default=30, help="streaming test length")
    args = p.parse_args()
    levels = []

    def report(level, msg):
        levels.append(level)
        print(f"[{level:4}] {msg}")

    print("== configuration")
    if os.path.exists(args.config):
        report("PASS", f"loaded {args.config}")
    else:
        report("WARN", "no rig.json: cameras are unnamed (run: python3 -m camsync.assign)")
    cfg = config.load(args.config)

    print("== cameras")
    cams, missing, unassigned = config.resolve(cfg)
    if not cams:
        report("FAIL", "no cameras found")
        return finish(levels)
    for name in missing:
        report("FAIL", f"{name}: nothing connected on its port {cfg['cameras'][name]}")
    for info in cams:
        if info.id in cfg["cameras"]:
            report("PASS", f"{info.id}: found on port {info.port} ({info.path})")
    for info in unassigned:
        report("WARN" if cfg["cameras"] else "INFO",
               f"camera on port {info.port} has no name (run: python3 -m camsync.assign)")
    for info in cams:
        if "." in info.port:
            report("INFO", f"{info.id} is behind a USB hub (port {info.port}); if it loses frames, "
                           "move it to a direct port")

    print("== settings")
    try:
        rig = Rig(cfg, cams, queue=1024)  # keep every set; it is counted, not displayed
    except Exception as e:
        report("FAIL", f"could not open cameras: {e}")
        return finish(levels)
    for cid in rig.ids:
        ctl = rig.current[cid]
        bad = rig.mismatched.get(cid)
        if bad:
            report("WARN", f"{cid}: camera did not accept {bad} (wanted, actual)")
        else:
            report("PASS", f"{cid}: all settings applied (exposure "
                           f"{ctl['exposure_time_absolute'] / 10:.1f} ms, gain {ctl['gain']})")

    print(f"== streaming test ({args.seconds:.0f} s, 1280x720 at 120 fps)")
    sets = complete = 0
    max_offset = 0.0
    with rig:
        deadline = time.monotonic() + 5
        while not rig.synced and time.monotonic() < deadline:
            rig.poll_sets()
            time.sleep(0.1)
        if not rig.synced:
            locked = [c for c in rig.ids if rig.clocks[c].warm]
            report("FAIL", f"camera clocks did not lock within 5 s (locked: {locked or 'none'})")
            return finish(levels)
        for st in rig.stats.values():
            st.reset()
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            time.sleep(0.2)
            for s in rig.poll_sets():
                sets += 1
                if s.complete:
                    complete += 1
                    max_offset = max(max_offset, *(abs(s.offset(c)) for c in rig.ids))
        elapsed = time.monotonic() - start
    for cid in rig.ids:
        st = rig.stats[cid]
        fps = st.total_frames / elapsed
        msg = f"{cid}: {fps:.1f} fps, lost {st.total_lost} frames ({st.total_loss:.2%})"
        if st.total_lost == 0 and fps >= MIN_FPS:
            report("PASS", msg)
        elif st.total_loss < 0.001 and fps >= MIN_FPS:
            report("WARN", msg)
        else:
            report("FAIL", msg + " - check cooling, cable and port")
    if sets:
        report("PASS" if complete / sets > 0.999 else "WARN",
               f"synchronized sets: {complete}/{sets} complete, "
               f"largest nearest-frame offset {max_offset * 1e3:.2f} ms")
    return finish(levels)


def finish(levels):
    worst = "FAIL" if "FAIL" in levels else "WARN" if "WARN" in levels else "PASS"
    print(f"\nRESULT: {worst}")
    sys.exit(1 if worst == "FAIL" else 0)


if __name__ == "__main__":
    main()
