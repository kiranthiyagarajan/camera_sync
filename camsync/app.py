"""Desktop app for the rig tools, so nothing needs the command line.

  python3 -m camsync.app             open the app
  python3 -m camsync.app --install   add "Camera Rig" to the desktop's app menu

Each tool runs as its own process (cameras can only be used by one at a time), with
its output shown in the app. Stop sends Ctrl+C, so tools still save and summarise.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading

import cv2
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

from . import calib, config  # noqa: E402
from .board import DICTIONARIES  # noqa: E402

DESKTOP_FILE = os.path.expanduser("~/.local/share/applications/camera-rig.desktop")


class App(Gtk.Window):
    def __init__(self):
        super().__init__(title="Camera Rig")
        self.set_default_size(1150, 760)
        self.proc = None
        self.tool_buttons = []
        self.motor_busy = False
        self.connect("delete-event", self.on_close)

        outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        for side in ("start", "end", "top", "bottom"):
            getattr(outer, f"set_margin_{side}")(12)
        self.add(outer)
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        left.set_size_request(320, -1)
        outer.pack_start(left, False, False, 0)
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        outer.pack_start(right, True, True, 0)
        self.build_tools(left)
        self.build_status(right)
        self.refresh()
        GLib.timeout_add_seconds(3, self.refresh)

    # ---- layout ----------------------------------------------------------------
    @staticmethod
    def heading(box, title):
        lbl = Gtk.Label(xalign=0)
        lbl.set_markup(f"<b>{title}</b>")
        lbl.set_margin_top(8)
        box.pack_start(lbl, False, False, 0)

    def tool_button(self, box, label, tooltip, module, args=lambda: [], pack=True):
        b = Gtk.Button(label=label)
        b.set_tooltip_text(tooltip)
        b.connect("clicked", lambda _b: self.run_tool(label, module, args()))
        if pack:
            box.pack_start(b, False, False, 0)
        self.tool_buttons.append(b)
        return b

    def build_tools(self, left):
        self.heading(left, "Setup")
        self.tool_button(left, "Live dashboard",
                         "All feeds with frame rate, loss, exposure, focus aid and capture "
                         "(keys are shown at the bottom of its window)", "rigview")
        self.tool_button(left, "Name cameras",
                         "Cover each camera when asked to name it cam1-cam4 by USB port", "assign")

        self.heading(left, "Calibration")
        self.tool_button(left, "1. Lens calibration",
                         "Move the ChArUco board through each camera's view; views are captured "
                         "automatically. Press k in its window to compute and save.",
                         "calibrate", lambda: ["intrinsics"])
        self.tool_button(left, "2. Camera positions",
                         "First capture: board flat in the centre of the cube. Then more board "
                         "positions (c each), k to compute and save.",
                         "calibrate", lambda: ["extrinsics"])
        self.tool_button(left, "3. Reflection mask (about 5 s)",
                         "Set the tracking exposure first (m in the Live dashboard), then remove "
                         "all markers: bright spots seen now (rails, IR rings) are masked",
                         "mask")

        self.heading(left, "Tracking")
        self.tool_button(left, "Track robot (proof of concept)",
                         "Live markers, 3D pose view and bend angles at 5.5 ms exposure, gain 63. "
                         "Drag rotates the 3D view, o toggles the camera overlay, "
                         "r records disk poses to logs/track-*.csv, q quits.",
                         "track", lambda: ["--exposure", "5.5", "--gain", "63"])

        self.heading(left, "Robot motors")
        self.motor_status = Gtk.Label(label="Motors: not checked", xalign=0)
        left.pack_start(self.motor_status, False, False, 0)
        row = Gtk.Box(spacing=6)
        for label, tip, keys in (
                ("Release", "Disable all motors so the robot can be moved by hand", "x"),
                ("Set home", "Make the current pose home (0, 0, 0) and hold it", "He"),
                ("Hold", "Enable all motors at their current position", "e"),
                ("Go home", "Move all motors back to home together", None)):
            b = Gtk.Button(label=label)
            b.set_tooltip_text(tip)
            b.connect("clicked", lambda _b, k=keys: self.motor_command(k))
            row.pack_start(b, True, True, 0)
        left.pack_start(row, False, False, 0)
        jog = Gtk.Grid(column_spacing=6, row_spacing=4)
        jog.attach(Gtk.Label(label="Jog step", xalign=0), 0, 0, 1, 1)
        self.jog_step = Gtk.ComboBoxText()
        for d in ("1", "5", "18"):
            self.jog_step.append(d, f"{d} deg")
        self.jog_step.set_active_id("5")
        jog.attach(self.jog_step, 1, 0, 2, 1)
        for i in range(3):
            jog.attach(Gtk.Label(label=f"Motor {i + 1}", xalign=0), 0, i + 1, 1, 1)
            for col, sign in ((1, -1), (2, 1)):
                b = Gtk.Button(label="−" if sign < 0 else "+")
                b.set_tooltip_text("Turn this motor only, by the jog step (the others hold)")
                b.connect("clicked", lambda _b, i=i, sg=sign: self.motor_jog(i, sg))
                jog.attach(b, col, i + 1, 1, 1)
        left.pack_start(jog, False, False, 0)

        self.heading(left, "Checks")
        self.tool_button(left, "Preflight check (about 35 s)",
                         "Cameras present, settings accepted, 30 s streaming test", "preflight")
        row = Gtk.Box(spacing=6)
        self.soak_minutes = Gtk.SpinButton.new_with_range(1, 120, 1)
        self.soak_minutes.set_value(10)
        self.soak_graph = Gtk.CheckButton(label="graph")
        self.soak_graph.set_active(True)
        b = self.tool_button(row, "Soak test", "Long stability test: frame rate and loss over time",
                             "soak", self.soak_args, pack=False)
        row.pack_start(b, True, True, 0)
        row.pack_start(self.soak_minutes, False, False, 0)
        row.pack_start(Gtk.Label(label="min"), False, False, 0)
        row.pack_start(self.soak_graph, False, False, 0)
        left.pack_start(row, False, False, 0)

        self.heading(left, "Settings")
        cfg = config.load()
        grid = Gtk.Grid(column_spacing=8, row_spacing=6)
        grid.attach(Gtk.Label(label="Frame rate", xalign=0), 0, 0, 1, 1)
        self.fps = Gtk.ComboBoxText()
        for f in config.FPS_CHOICES:
            self.fps.append(str(f), f"{f} fps")
        self.fps.set_active_id(str(cfg["format"]["fps"]))
        self.fps.connect("changed", self.on_fps)
        grid.attach(self.fps, 1, 0, 1, 1)
        board = cfg["board"]
        self.square = Gtk.SpinButton.new_with_range(1, 300, 0.01)
        self.square.set_value(board["square_mm"])
        self.marker = Gtk.SpinButton.new_with_range(1, 300, 0.01)
        self.marker.set_value(board["marker_mm"])
        self.dictionary = Gtk.ComboBoxText()
        for d in DICTIONARIES:
            self.dictionary.append(d, d)
        self.dictionary.set_active_id(board["dictionary"])
        self.squares_x = Gtk.SpinButton.new_with_range(3, 30, 1)
        self.squares_x.set_value(board["squares_x"])
        self.squares_y = Gtk.SpinButton.new_with_range(3, 30, 1)
        self.squares_y.set_value(board["squares_y"])
        self.squares = Gtk.Box(spacing=4)
        self.squares.pack_start(self.squares_x, False, False, 0)
        self.squares.pack_start(Gtk.Label(label="x"), False, False, 0)
        self.squares.pack_start(self.squares_y, False, False, 0)
        self.squares.set_tooltip_text("Columns x rows of chessboard squares (count squares, not corners)")
        for r, (name, w) in enumerate([("Squares", self.squares), ("Square (mm)", self.square),
                                       ("Marker (mm)", self.marker), ("Dictionary", self.dictionary)], 1):
            grid.attach(Gtk.Label(label=name, xalign=0), 0, r, 1, 1)
            grid.attach(w, 1, r, 1, 1)
        save = Gtk.Button(label="Save board")
        save.set_tooltip_text("Measure a printed square with a ruler: printers often scale slightly")
        save.connect("clicked", self.on_save_board)
        grid.attach(save, 1, 5, 1, 1)
        left.pack_start(grid, False, False, 0)

        self.heading(left, "Folders")
        folders = Gtk.Box(spacing=6)
        for label, path in (("Logs", config.LOG_DIR), ("Captures", config.CAPTURE_DIR),
                            ("Calibration", config.CALIB_DIR)):
            b = Gtk.Button(label=label)
            b.connect("clicked", lambda _b, p=path: self.open_path(p, folder=True))
            folders.pack_start(b, True, True, 0)
        left.pack_start(folders, False, False, 0)
        guide = Gtk.Button(label="Setup guide")
        guide.connect("clicked", lambda _b: self.open_path(os.path.join(config.PROJECT_DIR, "SETUP.md")))
        left.pack_start(guide, False, False, 0)

        self.stop_button = Gtk.Button(label="Stop running tool")
        self.stop_button.get_style_context().add_class("destructive-action")
        self.stop_button.set_sensitive(False)
        self.stop_button.connect("clicked", lambda _b: self.stop())
        left.pack_end(self.stop_button, False, False, 0)

    def build_status(self, right):
        self.running = Gtk.Label(xalign=0)
        right.pack_start(self.running, False, False, 0)
        # name, port, device, lens, position
        self.cams = Gtk.ListStore(str, str, str, str, str)
        view = Gtk.TreeView(model=self.cams)
        for i, title in enumerate(["Camera", "USB port", "Device", "Lens calibration",
                                   "Position calibration"]):
            col = Gtk.TreeViewColumn(title, Gtk.CellRendererText(), text=i)
            col.set_expand(i >= 3)
            view.append_column(col)
        right.pack_start(view, False, False, 0)
        self.calib_summary = Gtk.Label(xalign=0)
        self.calib_summary.set_line_wrap(True)
        right.pack_start(self.calib_summary, False, False, 0)
        self.heading(right, "Output")
        self.log_view = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        self.log_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        scroll = Gtk.ScrolledWindow()
        scroll.add(self.log_view)
        right.pack_start(scroll, True, True, 0)
        clear = Gtk.Button(label="Clear output")
        clear.connect("clicked", lambda _b: self.log_view.get_buffer().set_text(""))
        right.pack_start(clear, False, False, 0)

    # ---- status ----------------------------------------------------------------
    def refresh(self):
        cfg = config.load()
        cams, missing, _ = config.resolve(cfg)
        ext = calib.load_extrinsics()
        ext_cams = ext[1]["cameras"] if ext else {}
        self.cams.clear()
        for info in cams:
            intr = calib.load_intrinsics(info.id)
            lens = f"done ({intr[2]['rms_px']:.2f} px)" if intr else "not done"
            e = ext_cams.get(info.id)
            pos = "not done" if not e else (f"done ({e['rms_px']:.2f} px)" if e.get("rms_px")
                                            else "done")
            name = info.id if info.id in cfg["cameras"] else f"{info.id} (unnamed)"
            self.cams.append([name, info.port, info.path, lens, pos])
        for name in missing:
            self.cams.append([name, cfg["cameras"][name], "NOT CONNECTED", "", ""])
        if ext:
            d = ext[1]
            chk = d.get("board_check_mm")
            check = f", board check {chk['mean']:.3f} mm mean error" if chk else ""
            refined = "" if d.get("refined_by_bundle_adjustment") else " (unrefined: install python3-scipy)"
            self.calib_summary.set_text(f"Camera positions calibrated {d['created']} from "
                                        f"{d['captures']} captures{check}{refined}.")
        else:
            self.calib_summary.set_text("Camera positions not calibrated yet.")
        if not cams:
            self.calib_summary.set_text("No cameras connected.")
        return True

    # ---- motors ------------------------------------------------------------------
    def motor_jog(self, i, sign):
        self.motor_command(("jog", i, sign * float(self.jog_step.get_active_id())))

    def motor_command(self, keys):
        """Send keys (None = go home, ("jog", motor, degrees) = turn one motor) to the
        ESP32 in a thread, then show its position. Clicks while busy are ignored."""
        if self.motor_busy:
            return
        self.motor_busy = True
        self.motor_status.set_text("Motors: working...")

        def work():
            from .motors import Motors
            try:
                m = Motors()
                try:
                    if keys is None:
                        m.goto(0, 0, 0)
                    elif keys[0] == "jog":
                        _, i, deg = keys
                        pos, home = m.position()
                        if not home:  # G needs a home: use this pose until Set home
                            m.send("H")
                            pos = (0.0, 0.0, 0.0)
                        target = list(pos)
                        target[i] += deg
                        m.goto(*target, limit=360)
                    else:
                        m.send(keys)
                    (a1, a2, a3), home = m.position()
                    held = "holding" if "Motors enabled (holding)." in m.send("p") else "released"
                finally:
                    m.close()
                text = (f"Motors: {a1:.1f}, {a2:.1f}, {a3:.1f} deg from home, {held}"
                        if home else "Motors: home not set")
            except (Exception, SystemExit) as e:
                text = f"Motors: {e}"
            GLib.idle_add(self.motor_status.set_text, text)
            self.motor_busy = False
        threading.Thread(target=work, daemon=True).start()

    # ---- running tools -----------------------------------------------------------
    def soak_args(self):
        args = ["--minutes", str(int(self.soak_minutes.get_value())), "--every", "30"]
        return args + (["--graph"] if self.soak_graph.get_active() else [])

    def run_tool(self, label, module, args):
        if self.proc:
            return
        cmd = [sys.executable, "-u", "-m", f"camsync.{module}", *args]
        self.log(f"\n=== {label} ===\n")
        try:
            self.proc = subprocess.Popen(cmd, cwd=config.PROJECT_DIR, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            self.log(f"could not start: {e}\n")
            return
        self.set_running(label)
        threading.Thread(target=self.read_output, args=(self.proc,), daemon=True).start()

    def read_output(self, proc):
        for line in proc.stdout:
            GLib.idle_add(self.log, line)
        code = proc.wait()
        GLib.idle_add(self.finished, code)

    def finished(self, code):
        self.proc = None
        result = "finished" if code == 0 else f"finished with problems (exit code {code})"
        self.log(f"=== {result} ===\n")
        self.set_running(None)
        self.refresh()

    def set_running(self, label):
        for b in self.tool_buttons:
            b.set_sensitive(label is None)
        self.stop_button.set_sensitive(label is not None)
        self.running.set_markup(f"<b>Running:</b> {GLib.markup_escape_text(label)}" if label
                                else "Ready. Only one tool can use the cameras at a time.")

    def stop(self):
        proc = self.proc
        if proc is None:
            return
        proc.send_signal(signal.SIGINT)  # like Ctrl+C: tools clean up and print summaries

        def force():
            if proc.poll() is None:
                proc.terminate()
            return False
        GLib.timeout_add(5000, force)

    def log(self, text):
        buf = self.log_view.get_buffer()
        buf.insert(buf.get_end_iter(), text)
        self.log_view.scroll_to_mark(buf.get_insert(), 0, False, 0, 0)
        buf.place_cursor(buf.get_end_iter())
        return False

    # ---- settings ----------------------------------------------------------------
    def on_fps(self, combo):
        cfg = config.load()
        config.set_fps(cfg, int(combo.get_active_id()))
        config.save(cfg)
        self.log(f"frame rate set to {combo.get_active_id()} fps\n")

    def on_save_board(self, _b):
        cfg = config.load()
        cfg["board"].update(squares_x=self.squares_x.get_value_as_int(),
                            squares_y=self.squares_y.get_value_as_int(),
                            square_mm=round(self.square.get_value(), 2),
                            marker_mm=round(self.marker.get_value(), 2),
                            dictionary=self.dictionary.get_active_id())
        if cfg["board"]["marker_mm"] >= cfg["board"]["square_mm"]:
            self.log("board not saved: the marker must be smaller than the square\n")
            return
        b = cfg["board"]
        needed = b["squares_x"] * b["squares_y"] // 2
        available = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b["dictionary"])).bytesList.shape[0]
        if needed > available:
            self.log(f"board not saved: {b['squares_x']} x {b['squares_y']} needs {needed} markers but "
                     f"{b['dictionary']} has only {available}\n")
            return
        config.save(cfg)
        self.log(f"board saved: {json.dumps(cfg['board'])}\n")

    def open_path(self, path, folder=False):
        if folder:
            os.makedirs(path, exist_ok=True)
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_close(self, *_):
        if self.proc:
            self.stop()
        Gtk.main_quit()
        return False


def install():
    os.makedirs(os.path.dirname(DESKTOP_FILE), exist_ok=True)
    with open(DESKTOP_FILE, "w") as f:
        f.write("[Desktop Entry]\nType=Application\nName=Camera Rig\n"
                "Comment=Camera sync, calibration and motion capture tools\n"
                f"Exec={sys.executable} -m camsync.app\nPath={config.PROJECT_DIR}\n"
                "Icon=camera-video\nTerminal=false\nCategories=Science;Utility;\n")
    os.chmod(DESKTOP_FILE, 0o755)
    print(f"installed {DESKTOP_FILE}: find 'Camera Rig' in the app menu "
          "(right-click it > Add to Favorites to pin it to the dock)")


def main():
    p = argparse.ArgumentParser(description="Desktop app for the camera rig tools.")
    p.add_argument("--install", action="store_true", help="add Camera Rig to the app menu")
    p.add_argument("--screenshot", help=argparse.SUPPRESS)  # testing: save the window and quit
    args = p.parse_args()
    if args.install:
        install()
        return
    app = App()
    app.set_running(None)
    app.show_all()
    if args.screenshot:
        def shot():
            win = app.get_window()
            pb = Gdk.pixbuf_get_from_window(win, 0, 0, win.get_width(), win.get_height())
            pb.savev(args.screenshot, "png", [], [])
            Gtk.main_quit()
            return False
        GLib.timeout_add(1500, shot)
    Gtk.main()


if __name__ == "__main__":
    main()
