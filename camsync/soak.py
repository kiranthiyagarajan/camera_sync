"""Long stability test without video: python3 -m camsync.soak [--minutes 10] [--every 10]

Prints frame rate and loss per camera at a fixed interval, logs every second to
logs/soak-*.csv, and ends with frames lost per minute (heat problems show up as
loss that grows over time). Ctrl+C stops early and still prints the summary.
"""
import argparse
import time

from . import config
from .rig import Rig
from .stats import CsvLogger


def main():
    p = argparse.ArgumentParser(description="Long-running stability test for the camera rig.")
    p.add_argument("--config", default=config.DEFAULT_PATH)
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--every", type=float, default=10, help="seconds between status lines")
    args = p.parse_args()

    cfg = config.load(args.config)
    rig = Rig(cfg, queue=1024)  # keep every set; it is counted, not displayed
    logger = CsvLogger("soak")
    last_set, max_offset = None, 0.0
    print(f"soak test: {len(rig.ids)} cameras ({', '.join(f'{c}=port {rig.ports[c]}' for c in rig.ids)}), "
          f"{args.minutes:g} min, log {logger.path}")
    with rig:
        # Count from the moment every clock has locked, so startup doesn't skew the numbers.
        deadline = time.monotonic() + 5
        while not rig.synced and time.monotonic() < deadline:
            rig.poll_sets()
            time.sleep(0.1)
        for st in rig.stats.values():
            st.reset()
        start = time.monotonic()
        next_print, next_log = start + args.every, start + 1
        try:
            while time.monotonic() - start < args.minutes * 60:
                time.sleep(0.2)
                now = time.monotonic()
                for st in rig.stats.values():
                    st.tick(now)
                for s in rig.poll_sets():
                    if s.complete:
                        last_set = s
                        max_offset = max(max_offset, *(abs(s.offset(c)) for c in rig.ids))
                while now >= next_log:
                    offs = {c: last_set.offset(c) for c in rig.ids} if last_set else {}
                    logger.log(rig, offs)
                    next_log += 1
                if now >= next_print:
                    k = int(args.every)
                    parts = []
                    for cid in rig.ids:
                        fps, loss = rig.stats[cid].window(k)
                        parts.append(f"{cid} {fps:5.1f} fps {loss:6.2%}")
                    el = int(now - start)
                    print(f"[{el // 60:02d}:{el % 60:02d}] " + " | ".join(parts), flush=True)
                    next_print += args.every
        except KeyboardInterrupt:
            print("stopped early")
        elapsed = time.monotonic() - start
    logger.close()

    print(f"\n== summary ({elapsed / 60:.1f} min)")
    for cid in rig.ids:
        st = rig.stats[cid]
        secs = st.seconds(st.history)
        per_min = {}
        if secs:
            first = secs[0][0]
            for sec, _, lost in secs:
                per_min[(sec - first) // 60] = per_min.get((sec - first) // 60, 0) + lost
        verdict = "OK" if st.total_lost == 0 else "LOSING FRAMES"
        print(f"{cid} (port {rig.ports[cid]}): {st.total_frames} frames, lost {st.total_lost} "
              f"({st.total_loss:.2%}) {verdict}")
        print(f"    lost per minute: {[per_min[m] for m in sorted(per_min)]}")
    if last_set:
        print(f"largest nearest-frame offset between cameras: {max_offset * 1e3:.2f} ms")
    print(f"log: {logger.path}")


if __name__ == "__main__":
    main()
