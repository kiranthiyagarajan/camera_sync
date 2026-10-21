# Rig setup checklist

Run everything from the project folder: `cd ~/camera_sync`

**Easiest: open the "Camera Rig" app** from the app menu (Super key, type "Camera Rig";
right-click it > Add to Favorites to pin it to the dock). It has a button for every
tool below, shows which cameras are connected and calibrated, sets the frame rate and
board size, and shows each tool's output. Stop = Ctrl+C for the running tool.
To add it to the menu on another computer: `python3 -m camsync.app --install`.
The commands below do the same from a terminal.

## 1. Connect
- Cameras straight into the laptop's 3 USB-A ports, 4th camera on the Prolific hub **slot 2**
  in the USB-C port. Keyboard/mouse receivers on the hub too; nothing else on it.
- The Anker dock does not work for cameras. Don't copy files to/from storage on the hub while
  capturing (it takes bandwidth from the camera).
- Keep cameras ventilated: at 120 fps they lose frames once they heat up.

## Frame rate
120 fps (default) or 60 fps. 60 fps halves the CPU load and the cameras run cooler;
for a slow robot it is plenty. Set it for every tool:
```
python3 -m camsync.config --fps 60      # or --fps 120; prints the current rig.json
```
Each tool also takes `--fps 60` to override it for one run. The longest exposure is
8.3 ms at 120 fps and 16.5 ms at 60 fps.

## 2. Name the cameras (once, or after re-plugging into different ports)
```
python3 -m camsync.assign
```
Cover each camera when asked (or shine a light into it); `s` saves to `rig.json`.
Names are tied to USB ports, so **label each cable and its port**.

## 3. Focus and exposure
```
python3 -m camsync.rigview
```
| Key | Action |
|---|---|
| `1`-`4` / `0` | select a camera / all cameras |
| `f` | focus aid: turn each lens until its score reaches its peak (green bar) |
| `m` | auto exposure for **markers**: have markers in view of every camera (IR lights on) |
| `b` | auto exposure for the **calibration board** |
| `e`/`E`, `g`/`G` | exposure / gain down / up by hand |
| `o` | show clipped (saturated) pixels in red; markers should not be red |
| `h` | brightness histograms |
| `s` | **save** exposure and gain to `rig.json` (used by every tool from then on) |
| `c` or space | capture a synchronized set of images to `captures/` (hold the board still) |
| `r` | reset statistics |
| `q` | quit |

On each feed: frame rate, frame loss (last 10 s and total), exposure, gain, sync
offset to the first camera. Red border = losing more than 1% of frames, orange = any loss.
Graphs at the bottom show frame rate and loss over the last 5 minutes. Every
second is logged to `logs/rigview-*.csv`.

Marker exposure is capped at 2 ms to limit motion blur. If auto exposure warns that a
lot of the image is bright, look for reflections (shiny surfaces, lights in view).

## 4. Calibrate (after mounting and focusing; redo if a camera moves or is refocused)
Needs `sudo apt install -y python3-scipy` for the final refinement.

Board: ChArUco 7 x 5 squares, 25 mm squares, 18 mm markers, DICT_6X6_50 (in
`rig.json` under `"board"`; change it in the app under Settings, then Save board).
Measure a printed square with a ruler and correct the sizes if the printer scaled it.
Lens and position calibration may use different boards: switch the board setting
before step 2. The position board's square size sets the world scale. Laser prints stay visible through
the IR filter; inkjet often does not. Check in rigview first.

1. Exposure for the board: `rigview`, press `b`, wait for OK, press `s`.
2. Lenses:
   ```
   python3 -m camsync.calibrate intrinsics
   ```
   Move the board slowly through every camera's view (near, far, tilted, into the
   corners). Views are captured automatically when the board is held still; aim for
   25 views and an all-green coverage map per camera, then press `k`.
   Good: rms below 0.5 px.
3. Camera positions:
   ```
   python3 -m camsync.calibrate extrinsics
   ```
   First capture (`c`): board **flat in the centre of the cube**, seen by all 4 cameras.
   That defines the world frame (origin at the board's first inner corner, z up).
   Then 10-20 more captures in different positions and tilts, each seen by at least
   2 cameras. Press `k`: it shows each camera's position and error, plus a board check
   (triangulated corner spacing vs. the true 35 mm). Good: well under 0.5 mm.

Results: `calib/intrinsics/<camera>.json`, `calib/extrinsics.json`.

Each camera-position capture is saved to `calib/captures/<session>/` as you take it.
If the computation fails, redo it without the cameras:
`python3 -m camsync.calibrate extrinsics --recompute calib/captures/<session>`.

## 4b. Reflection mask (after calibration; redo if a camera moves)
Static bright spots (shiny cube rails, other cameras' IR rings) would look like markers.
1. Put a few markers in view of every camera; in the Live dashboard press `m` (marker
   auto exposure, max 2 ms), then `s` to save. Close the dashboard.
2. Remove or cover all markers (and the calibration board), IR rings on.
3. App: "3. Reflection mask" (or `python3 -m camsync.mask`). It records 3 s, masks every
   pixel that got bright plus a 4 px margin, and saves `calib/masks/<camera>.png` and
   `calib/masks/preview.png` (masked areas red). A few % masked is normal.
4. Markers back in view: press `m` again. Marker auto exposure now ignores masked
   pixels. Press `k` in the dashboard to show the mask.
Matte black tape on shiny rails, or hoods on the IR rings, shrinks the mask.

## 4c. Track the robot (proof of concept)
App: "Track robot (proof of concept)" (or `python3 -m camsync.track --exposure 5.5 --gain 63`).
Shows each camera with detected blobs (green circles) and triangulated markers (yellow =
seen, red = predicted but not detected) and the pose drawn over it (disk rims, backbone,
base axes; `o` toggles). Below: a 3D view of cameras, markers, disk rims and backbone (tip
in red; left-drag rotates, wheel or right-drag zooms, `v` resets) and the bend angle of
each segment and base to tip (`~` = estimated from disk centres because a disk has fewer
than 3 markers, so its normal is unknown). `r` records disk centres and directions to
`logs/track-*.csv`. Needs lens + position calibration made with the same lens files:
redo "2. Camera positions" after every lens calibration.

## 5. Check before each session
```
python3 -m camsync.preflight        # ~35 s; RESULT: PASS / WARN / FAIL
```

## 6. Long stability test
```
python3 -m camsync.soak --minutes 10    # Ctrl+C stops early with a summary
```
Frames lost per minute that grow over time = overheating.

## 7. Stepper motors (ESP32 + 3 DRV8825)
Firmware: `firmware/stepper_control/stepper_control.ino`. The ESP32 shows up as
`/dev/ttyUSB0` (CP210x). Once per computer: `sudo usermod -aG dialout $USER`, then log
out and back in (otherwise "Permission denied" on the port).
```
python3 -m camsync.motors --flash     # compile + upload (hold BOOT if it says "Connecting...")
python3 -m camsync.motors             # interactive: type 1, +, -, 0, p, x, ?  (Ctrl+] quits)
python3 -m camsync.motors 1 + + p     # send keys, print the replies
```
From Python: `m = Motors(); m.send("1+p")` returns the reply lines. Opening the port
doesn't reset the ESP32; positions are software counts and reset with the board.
Only one program can hold the port: close the Arduino IDE serial monitor first.

## Tests
```
python3 -m unittest discover tests
```
