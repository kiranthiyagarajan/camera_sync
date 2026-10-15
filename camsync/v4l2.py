"""Minimal V4L2 mmap capture for UVC cameras, using only the standard library.

Gives raw MJPEG frames with the kernel's capture timestamp (CLOCK_MONOTONIC,
start of exposure) and the driver's sequence number.
"""
import ctypes
import fcntl
import glob
import mmap
import os
from dataclasses import dataclass

c_u8, c_u32, c_s32 = ctypes.c_uint8, ctypes.c_uint32, ctypes.c_int32

BUF_TYPE_VIDEO_CAPTURE = 1
MEMORY_MMAP = 1
FIELD_NONE = 1
BUF_FLAG_ERROR = 0x40
BUF_FLAG_TIMESTAMP_MASK = 0xE000
BUF_FLAG_TIMESTAMP_MONOTONIC = 0x2000
PIX_FMT_MJPEG = int.from_bytes(b"MJPG", "little")

# Control IDs (see `v4l2-ctl --list-ctrls`).
CONTROLS = {
    "brightness": 0x00980900,
    "contrast": 0x00980901,
    "saturation": 0x00980902,
    "hue": 0x00980903,
    "white_balance_automatic": 0x0098090C,
    "gamma": 0x00980910,
    "gain": 0x00980913,
    "power_line_frequency": 0x00980918,
    "white_balance_temperature": 0x0098091A,
    "sharpness": 0x0098091B,
    "backlight_compensation": 0x0098091C,
    "auto_exposure": 0x009A0901,
    "exposure_time_absolute": 0x009A0902,
}


class timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [("type", c_u32), ("flags", c_u32), ("frames", c_u8), ("seconds", c_u8),
                ("minutes", c_u8), ("hours", c_u8), ("userbits", c_u8 * 4)]


class _buffer_m(ctypes.Union):
    _fields_ = [("offset", c_u32), ("userptr", ctypes.c_ulong),
                ("planes", ctypes.c_void_p), ("fd", c_s32)]


class v4l2_buffer(ctypes.Structure):
    _fields_ = [("index", c_u32), ("type", c_u32), ("bytesused", c_u32), ("flags", c_u32),
                ("field", c_u32), ("timestamp", timeval), ("timecode", v4l2_timecode),
                ("sequence", c_u32), ("memory", c_u32), ("m", _buffer_m), ("length", c_u32),
                ("reserved2", c_u32), ("request_fd", c_s32)]


class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [("count", c_u32), ("type", c_u32), ("memory", c_u32),
                ("capabilities", c_u32), ("flags", c_u8), ("reserved", c_u8 * 3)]


class v4l2_pix_format(ctypes.Structure):
    _fields_ = [(name, c_u32) for name in (
        "width", "height", "pixelformat", "field", "bytesperline", "sizeimage", "colorspace",
        "priv", "flags", "ycbcr_enc", "quantization", "xfer_func")]


class _format_u(ctypes.Union):
    # The kernel union holds pointers, so it is 8-byte aligned.
    _fields_ = [("pix", v4l2_pix_format), ("raw_data", c_u8 * 200), ("_align", ctypes.c_void_p)]


class v4l2_format(ctypes.Structure):
    _fields_ = [("type", c_u32), ("fmt", _format_u)]


class v4l2_fract(ctypes.Structure):
    _fields_ = [("numerator", c_u32), ("denominator", c_u32)]


class v4l2_captureparm(ctypes.Structure):
    _fields_ = [("capability", c_u32), ("capturemode", c_u32), ("timeperframe", v4l2_fract),
                ("extendedmode", c_u32), ("readbuffers", c_u32), ("reserved", c_u32 * 4)]


class _parm_u(ctypes.Union):
    _fields_ = [("capture", v4l2_captureparm), ("raw_data", c_u8 * 200)]


class v4l2_streamparm(ctypes.Structure):
    _fields_ = [("type", c_u32), ("parm", _parm_u)]


class v4l2_control(ctypes.Structure):
    _fields_ = [("id", c_u32), ("value", c_s32)]


assert ctypes.sizeof(v4l2_buffer) == 88 and ctypes.sizeof(v4l2_format) == 208
assert ctypes.sizeof(v4l2_streamparm) == 204 and ctypes.sizeof(v4l2_requestbuffers) == 20


def _ioc(direction, nr, struct):
    return (direction << 30) | (ctypes.sizeof(struct) << 16) | (ord("V") << 8) | nr


_W, _RW = 1, 3
VIDIOC_S_FMT = _ioc(_RW, 5, v4l2_format)
VIDIOC_REQBUFS = _ioc(_RW, 8, v4l2_requestbuffers)
VIDIOC_QUERYBUF = _ioc(_RW, 9, v4l2_buffer)
VIDIOC_QBUF = _ioc(_RW, 15, v4l2_buffer)
VIDIOC_DQBUF = _ioc(_RW, 17, v4l2_buffer)
VIDIOC_STREAMON = _ioc(_W, 18, ctypes.c_int)
VIDIOC_STREAMOFF = _ioc(_W, 19, ctypes.c_int)
VIDIOC_S_PARM = _ioc(_RW, 22, v4l2_streamparm)
VIDIOC_G_CTRL = _ioc(_RW, 27, v4l2_control)
VIDIOC_S_CTRL = _ioc(_RW, 28, v4l2_control)


@dataclass
class CameraInfo:
    path: str      # /dev/videoN
    serial: str    # USB serial number ("" if the camera has none)
    port: str      # USB port path, e.g. "3-1"
    id: str = ""   # stable name used by the rig: serial if unique, else port


def find_cameras(name="openaicam"):
    """Capture nodes of every connected camera whose name contains `name`, in stable order."""
    found = []
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        def read(rel):
            try:
                with open(os.path.join(node, rel)) as f:
                    return f.read().strip()
            except OSError:
                return ""
        # Each UVC camera exposes two nodes; index 0 is the one that streams video.
        if name not in read("name") or read("index") != "0":
            continue
        usb_dev = os.path.dirname(os.path.realpath(os.path.join(node, "device")))
        found.append(CameraInfo("/dev/" + os.path.basename(node), read("device/../serial"),
                                os.path.basename(usb_dev)))
    serials = [c.serial for c in found]
    unique = all(serials) and len(set(serials)) == len(serials)
    for c in found:
        c.id = c.serial if unique else c.port
    return sorted(found, key=lambda c: c.id)


class Camera:
    def __init__(self, path, width=1280, height=720, fps=120, nbuf=4):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        self.maps = []
        try:
            self._setup(width, height, fps, nbuf)
        except Exception:
            self.close()
            raise

    def _setup(self, width, height, fps, nbuf):
        fmt = v4l2_format(type=BUF_TYPE_VIDEO_CAPTURE)
        fmt.fmt.pix.width, fmt.fmt.pix.height = width, height
        fmt.fmt.pix.pixelformat, fmt.fmt.pix.field = PIX_FMT_MJPEG, FIELD_NONE
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        if (fmt.fmt.pix.width, fmt.fmt.pix.height) != (width, height):
            raise ValueError(f"{self.path}: {width}x{height} not supported "
                             f"(got {fmt.fmt.pix.width}x{fmt.fmt.pix.height})")
        self.width, self.height = width, height

        parm = v4l2_streamparm(type=BUF_TYPE_VIDEO_CAPTURE)
        parm.parm.capture.timeperframe = v4l2_fract(1, fps)
        fcntl.ioctl(self.fd, VIDIOC_S_PARM, parm)
        tpf = parm.parm.capture.timeperframe
        self.nominal_period = tpf.numerator / tpf.denominator

        req = v4l2_requestbuffers(count=nbuf, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
        for i in range(req.count):
            buf = v4l2_buffer(index=i, type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf)
            self.maps.append(mmap.mmap(self.fd, buf.length, mmap.MAP_SHARED,
                                       mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset))
            fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)

    def set_control(self, name, value):
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, v4l2_control(CONTROLS[name], value))

    def get_control(self, name):
        ctrl = v4l2_control(CONTROLS[name], 0)
        fcntl.ioctl(self.fd, VIDIOC_G_CTRL, ctrl)
        return ctrl.value

    def fileno(self):
        return self.fd

    def start(self):
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_int(BUF_TYPE_VIDEO_CAPTURE))

    def stop(self):
        fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_int(BUF_TYPE_VIDEO_CAPTURE))

    def read(self):
        """Dequeue one frame: (jpeg_bytes, sequence, t_kernel) or None if none is ready.

        jpeg_bytes is None when the driver flagged the frame as corrupt.
        """
        buf = v4l2_buffer(type=BUF_TYPE_VIDEO_CAPTURE, memory=MEMORY_MMAP)
        try:
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
        except BlockingIOError:
            return None
        if buf.flags & BUF_FLAG_TIMESTAMP_MASK != BUF_FLAG_TIMESTAMP_MONOTONIC:
            raise RuntimeError(f"{self.path}: timestamps are not CLOCK_MONOTONIC")
        data = None if buf.flags & BUF_FLAG_ERROR else self.maps[buf.index][:buf.bytesused]
        t = buf.timestamp.tv_sec + buf.timestamp.tv_usec * 1e-6
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
        return data, buf.sequence, t

    def close(self):
        for m in self.maps:
            m.close()
        self.maps = []
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
