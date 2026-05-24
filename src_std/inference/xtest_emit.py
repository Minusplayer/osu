"""XTest-based emitter for run_bot.

osu! lazer runs under XWayland (DISPLAY=:0), so libXtst's XTestFakeMotionEvent
and XTestFakeKeyEvent inject events directly into the XWayland server. Unlike
uinput, this does NOT create a new input device, so:
  - the user's real mouse keeps working
  - osu! lazer's raw-input grab doesn't lock onto a virtual absolute device
  - cursor doesn't snap to a corner on device enumeration

Takes ABSOLUTE SCREEN pixel coordinates. Press is KEY_Z by default (osu! K1).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

_NIX_LIBXTST = "/nix/store/v0h458qw43r3jx0vfcxc42v0dgm8rzdh-libxtst-1.2.5/lib/libXtst.so.6"


def _load_libxtst() -> ctypes.CDLL:
    override = os.environ.get("LIBXTST_PATH")
    if override:
        return ctypes.CDLL(override)
    found = ctypes.util.find_library("Xtst")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            pass
    if os.path.exists(_NIX_LIBXTST):
        return ctypes.CDLL(_NIX_LIBXTST)
    raise RuntimeError("libXtst not found; set LIBXTST_PATH or LD_LIBRARY_PATH")


_XTST = _load_libxtst()
_XTST.XTestFakeMotionEvent.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong,
]
_XTST.XTestFakeMotionEvent.restype = ctypes.c_int
_XTST.XTestFakeKeyEvent.argtypes = [
    ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong,
]
_XTST.XTestFakeKeyEvent.restype = ctypes.c_int


class XTestEmitter:
    def __init__(self, press_key: str = "z"):
        from src_std.capture.x11_input import _LIB as XLIB, X11
        self._XLIB = XLIB
        # Add XFlush binding if not present.
        if not hasattr(XLIB, "_xflush_wired"):
            XLIB.XFlush.argtypes = [ctypes.c_void_p]
            XLIB.XFlush.restype = ctypes.c_int
            XLIB._xflush_wired = True  # type: ignore[attr-defined]
        self._x11 = X11()
        self._dpy = self._x11._dpy
        self._screen = XLIB.XDefaultScreen(self._dpy)
        self._keycode = self._x11.keycode_for(press_key)
        self._key_down = False

    def move(self, screen_x: int, screen_y: int) -> None:
        _XTST.XTestFakeMotionEvent(self._dpy, self._screen,
                                   int(screen_x), int(screen_y), 0)
        self._XLIB.XFlush(self._dpy)

    def set_key(self, pressed: bool) -> None:
        if pressed == self._key_down:
            return
        _XTST.XTestFakeKeyEvent(self._dpy, self._keycode,
                                1 if pressed else 0, 0)
        self._XLIB.XFlush(self._dpy)
        self._key_down = pressed

    def release_all(self) -> None:
        if self._key_down:
            self.set_key(False)

    def close(self) -> None:
        try:
            self.release_all()
        finally:
            self._x11.close()
