"""Rig configuration (rig.json): camera names by USB port, video format, camera settings.

The cameras all report the same serial number, so a camera is identified by the
USB port it is plugged into. Calibration and settings are keyed by camera name.
"""
import json
import os

from .v4l2 import find_cameras

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(PROJECT_DIR, "rig.json")
CALIB_DIR = os.path.join(PROJECT_DIR, "calib")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
CAPTURE_DIR = os.path.join(PROJECT_DIR, "captures")

# Applied to every camera, in this order (auto_exposure must be manual before the
# exposure time can be set).
BASE_CONTROLS = {
    "auto_exposure": 1,              # manual; auto exposure can lower the frame rate
    "exposure_time_absolute": 50,    # 100 µs units -> 5 ms; must fit in a frame (max_exposure)
    "gain": 0,
    "power_line_frequency": 0,       # anti-flicker off so it can't alter frame timing
    "backlight_compensation": 0,
    "sharpness": 0,                  # sharpening puts halos around markers, biasing centroids
    "gamma": 100,                    # linear response, so brightness ~ exposure x gain
    "contrast": 30,
    "brightness": 0,
    "white_balance_automatic": 0,    # mono sensor; keep the pipeline from changing anything
    "saturation": 0,
}

FPS_CHOICES = (120, 60)


def max_exposure(fps):
    """Longest exposure (100 µs units) that still fits in one frame at this frame rate:
    8.3 ms at 120 fps (measured period 8.44 ms), 16.5 ms at 60 fps (measured 16.64 ms)."""
    return {120: 83, 60: 165}[fps]


def min_fps(fps):
    """Lowest acceptable measured frame rate (the cameras run ~1.2% under nominal)."""
    return fps * 0.98


def default_config():
    return {
        "format": {"width": 1280, "height": 720, "fps": 120},
        "cameras": {},    # name -> USB port, e.g. {"cam1": "3-1"}
        "controls": {},   # "all" or camera name -> {control: value}
        # ChArUco calibration board; measure the printed square/marker size with a ruler.
        "board": {"squares_x": 10, "squares_y": 7, "square_mm": 35.0, "marker_mm": 25.0,
                  "dictionary": "DICT_5X5_100"},
    }


def load(path=DEFAULT_PATH):
    cfg = default_config()
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for key in cfg:
            if key in data:
                cfg[key] = data[key]
    return cfg


def save(cfg, path=DEFAULT_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def set_fps(cfg, fps):
    if fps not in FPS_CHOICES:
        raise ValueError(f"fps must be one of {FPS_CHOICES}")
    cfg["format"]["fps"] = fps


def controls_for(cfg, cam):
    """Controls for one camera: base settings, then "all" overrides, then per-camera ones."""
    out = dict(BASE_CONTROLS)
    out.update(cfg["controls"].get("all", {}))
    out.update(cfg["controls"].get(cam, {}))
    return out


def resolve(cfg, infos=None):
    """Name the connected cameras from the config.

    Returns (cameras, missing, unassigned): cameras is a list of CameraInfo whose id
    is the configured name (or the port, if the port is not in the config), named
    cameras first; missing lists configured names whose port has no camera.
    """
    infos = find_cameras() if infos is None else infos
    by_port = {port: name for name, port in cfg["cameras"].items()}
    named, unassigned = [], []
    for info in infos:
        if info.port in by_port:
            info.id = by_port[info.port]
            named.append(info)
        else:
            info.id = info.port
            unassigned.append(info)
    found = {i.id for i in named}
    missing = sorted(n for n in cfg["cameras"] if n not in found)
    return (sorted(named, key=lambda i: i.id) + sorted(unassigned, key=lambda i: i.port),
            missing, unassigned)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Show or change rig.json settings.")
    p.add_argument("--config", default=DEFAULT_PATH)
    p.add_argument("--fps", type=int, choices=FPS_CHOICES, help="set the frame rate for all tools")
    args = p.parse_args()
    cfg = load(args.config)
    if args.fps:
        set_fps(cfg, args.fps)
        save(cfg, args.config)
        print(f"frame rate set to {args.fps} fps in {args.config}")
    print(json.dumps(cfg, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
