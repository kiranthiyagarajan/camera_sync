# camera_sync — 4-camera motion capture for a soft robot

Goal: real-time pose of a ~300 mm soft robot (4 segments incl. tip, 4–5 reflective
markers per segment on rigid PLA disks) from 4 synchronized IR cameras; pose output
will later feed a robot control model (not chosen yet). Repo:
https://github.com/kiranthiyagarajan/camera_sync (public, branch main).

## Working with this user
- Propose a plan and wait for an explicit go-ahead before building features. Don't add
  unrequested robustness or edge-case handling (USB disconnect/reconnect handling was
  built and then reverted at their request). Report problems and ask before fixing.
- Run long camera tests in the background and report a short summary (they asked to
  keep token use low). Tables + 2–4 bullet interpretation work well.
- They rearrange cameras/hubs between messages without saying so: **re-check the USB
  topology before every test** (snippet below).
- They prefer the GUI: `python3 -m camsync.app` (app menu: "Camera Rig").
- Claude has no sudo. Anything needing sudo (apt installs, uvcvideo params) must be
  run by the user **in a separate terminal**; the `!` prefix can't prompt for passwords.
- Commits: author "Kiran Thiyagarajan", GitHub no-reply email (already in git config);
  end messages with the Co-Authored-By trailer. Push after each feature.

## Hardware facts (measured)
- Cameras: "openaicam" 32e6:9251, USB 2.0 UVC, **MJPEG only**, 1280x720/800x600/640x480 at
  120 or 60 fps. **All four report the same serial (5423534724)**: identify cameras by
  USB port path (3-1, 3-3, 3-4, 3-6.2…), never by serial or /dev/videoN (it changes on
  every replug). Mono sensor but sends YUV422 JPEGs with empty chroma, so decode in grayscale.
- Real rates: "120 fps" runs at 118.55 fps (period 8.435 ms); "60 fps" at 60.0–60.1.
- Exposure (`exposure_time_absolute`) is in 100 µs units; max 83 at 120 fps and 165 at
  60 fps. Auto exposure (aperture priority) can lower the fps, so always use manual.
- Each camera at 120 fps reserves ~16 MB/s of isochronous bandwidth (2x1024 B per
  microframe), so **at most 2 cameras per shared 480 Mbit/s link** (any hub).
- Laptop: Dell G15 5520, i7-12700H, Ubuntu 22.04, OpenCV 4.5.4 (has aruco/ChArUco,
  GTK3 GUI), numpy; python3-gi (GTK3) available, **no tkinter**, scipy may still be
  missing (needed for bundle adjustment: `sudo apt install python3-scipy`).
- Ports: 3 USB-A (3-1, 3-3, 3-4, all USB 3) on the PCH controller 00:14.0, and 1 USB-C
  whose USB 2 lines also appear on the PCH as port 3-6.
- Hubs: the **Anker 552 USB-C dock fails** with cameras (error -71, corrupt frames). The
  **Prolific USB 2.0 hub (067b:2586), slot 2, in USB-C** has been flawless (keyboard/mouse
  receivers can share it).

## Known problems / findings
- **Frame loss on ports 3-3/3-4: fixed (2026-10-21) by changing the cable.** The user
  then streamed 2 h with no drops, and a 30 s check on 2026-10-21 lost 0 frames on all four
  cameras (cam4 on 3-4). Before the fix, frames were lost after ~1–2 min and kept getting
  worse. An earlier swap test had seemed to show the problem following the port; in
  hindsight the cable was bad. If drops come back, suspect cables first. Soak reports
  corrupt vs missing frames when `uvcvideo nodrop=1`
  (`echo 1 | sudo tee /sys/module/uvcvideo/parameters/nodrop`, resets on reboot).
- Camera heat also matters: ventilating a camera cut its drops. Short (≤60 s) tests
  miss these problems, so soak for 3–10 min.
- Inter-camera offset comes from each camera's own crystal (2–95 ppm drift, changing
  with temperature), not from cable length or hub vs direct. Nearest-frame skew ≤ half
  a frame (~4.2 ms at 120 fps); interpolating to a common time gives < 1 ms.

## How the code handles cameras (camsync/)
- `v4l2.py`: stdlib-only V4L2 mmap capture (ctypes ioctls); kernel timestamps
  (CLOCK_MONOTONIC, start of exposure) + sequence numbers; set/get controls.
- `clock.py`: per-camera linear clock model. Raw timestamps jitter ±2 ms (USB
  quantization, so consecutive intervals can be 13.7 ms with no frame lost): number
  frames sequentially until a validated 32-frame fit locks (residuals < 3 ms, period
  within 3% of nominal), then index from the fitted line. Warm after 128 frames.
  Stall gaps count as lost frames.
- `sync.py`: Synchronizer groups frames by the reference camera's times; FrameSet has
  the nearest frame per camera, offsets, and interpolation brackets/weights.
- `rig.py`: one capture thread polling all cameras; applies `rig.json` controls with
  read-back + retry (the camera occasionally rejects a control once, -71); holds sets
  until every clock is warm; per-camera stats; `latest` frame per camera; `queue`
  size (use 1024 when counting sets, small for live display).
- `config.py` (rig.json: camera names by port, format/fps, controls, ChArUco board
  7x5, 35 mm squares, 25 mm markers, DICT_6X6_50), `stats.py`, `imaging.py` (decode,
  auto exposure marker/board modes, focus score, cover-detector), `plots.py`,
  `sensors.py` (hwmon temps), `geometry.py`, `board.py`, `calib.py` (intrinsics,
  extrinsics init + bundle adjustment, z-up world from the board lying flat in the
  first capture, triangulated board check).
- Tools (also buttons in `app.py`): `rigview` (live dashboard), `assign` (name cameras
  by covering them), `calibrate intrinsics|extrinsics`, `mask` (static reflection
  mask → calib/masks/<cam>.png, used by marker auto exposure and rigview `k`), `preflight`, `soak [--graph]`,
  `config --fps 60|120`. Usage for the user is in SETUP.md. Tests:
  `python3 -m unittest discover tests` (bundle adjustment test skipped without scipy).

## Gotchas learned the hard way
- Cameras are exclusive: if a tool fails with "Device or resource busy", find the
  holder (often rigview launched from the app):
  `for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q /dev/video && tr '\0' ' ' < $p/cmdline; echo; done`
- List cameras by port (use if/then; an `[ ] && …` loop returns non-zero and breaks `&&` chains):
  `for n in /sys/class/video4linux/video*; do if [ "$(cat $n/index)" = 0 ] && grep -q openaicam $n/name; then echo -n "$(basename $n):$(basename $(dirname $(readlink -f $n/device))) "; fi; done`
  plus `lsusb -t` and `journalctl -k --since …` for plug events / -71 errors.
- `pkill -f <pattern>` kills the shell running the command if the pattern appears in
  it; kill by PID instead.
- With v4l2-ctl `--verbose` output, number frames by `seq`, not by rounding intervals.
- OpenCV GTK build: `WND_PROP_VISIBLE` is unsupported (always -1); detect a closed
  window with `getWindowProperty(win, WND_PROP_AUTOSIZE) < 0`.
- PyGObject: `gi.require_version("Gdk", "3.0")` as well as Gtk, or Gdk 4 gets loaded.
- In rigview/soak the GUI tools can be screenshot-tested: `rigview --duration N
  --snapshot file.png`, `app --screenshot file.png`, soak saves `logs/soak-*.png`.

## Status and next steps
Done: capture + sync core, rig tools, desktop app, 60 fps option, guided ChArUco
calibration (math tested on a simulated 500 mm cube; **not yet tried with the real
board**). The rig: 500 mm cube, cameras on 4 corners at different heights, 24-LED IR
ring + IR filter per camera. Before calibrating: measure the printed square, check the
board is visible through the IR filter (inkjet may vanish), focus with rigview `f`.

Agreed mocap plan (build one step at a time, each with a go-ahead):
1. ✅ Calibration tool + fps setting.
2. 2D marker detection (threshold, blob filter, intensity-weighted sub-pixel
   centroids, undistort points), static reflection/LED-ring mask (diagonal cameras see
   each other's IR rings), recording/replay of raw MJPEG + timestamps.
3. Correspondence + triangulation (epipolar matching to start, then tracking by
   projecting predicted 3D positions; ≥2 views per marker), live 3D view, accuracy tests.
4. Labeling via unique marker-distance patterns per segment disk (≥10 mm differences),
   per-segment rigid fit (Kabsch) → segment poses.
5. Piecewise-constant-curvature backbone + tip pose; real-time output (Python API +
   UDP stream) and recording for training the control model.
Lenses are wide (fx ≈ 505–550 px, ~100° horizontal field of view): ≈0.8 mm/pixel at 0.4 m, so
6–8 mm markers ≈ 8–10 px. Expect ~0.1–0.3 mm precision.

Calibration notes (2026-10-21): the board is now 7x5, DICT_6X6_50. Saved intrinsics were
made with the old 10x7 board, and fx/fy differ by up to 15%, so redo them. At floor
distance the 7x5 board isn't detected (markers ~10–12 px, dim), so hold it at ~25–35 cm.
The default exposure (8.3 ms, gain 63) suits calibration; for tracking use 1–2 ms.
cam1/cam3 see ~15 bright reflections (cube rails, opposite IR rings), so a mask is needed.
