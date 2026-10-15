# Rig setup checklist

Run everything from the project folder: `cd ~/camera_sync`

## 1. Connect
- Cameras straight into the laptop's 3 USB-A ports, 4th camera on the Prolific hub **slot 2**
  in the USB-C port. Keyboard/mouse receivers on the hub too; nothing else on it.
- The Anker dock does not work for cameras. Don't copy files to/from storage on the hub while
  capturing (it takes bandwidth from the camera).
- Keep cameras ventilated: at 120 fps they lose frames once they heat up.

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

## 4. Check before each session
```
python3 -m camsync.preflight        # ~35 s; RESULT: PASS / WARN / FAIL
```

## 5. Long stability test
```
python3 -m camsync.soak --minutes 10    # Ctrl+C stops early with a summary
```
Frames lost per minute that grow over time = overheating.

## Tests
```
python3 -m unittest discover tests
```
