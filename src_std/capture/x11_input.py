"""X11 input polling via libX11 ctypes.

osu! lazer runs under XWayland on niri (DISPLAY=:0), so X11 APIs see the
real cursor and keyboard state at sub-20 us per query.

Two probes:
- pointer(): cursor screen coords + button mask
- keymap(): full 32-byte bitmap of every keycode's pressed state

Both are blocking calls but so fast that the 30 Hz capture loop barely
notices them.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass

# Nix-store fallback: ctypes.util.find_library doesn't see nix-isolated libs
# unless LD_LIBRARY_PATH is set. The shell.nix overlay should expose libX11
# via xorg.libX11; if find_library still misses it we hit a known path.
_NIX_LIBX11 = "/nix/store/5m91jqg1526jzsahrgmd37k4ml3nc5l4-libx11-1.8.13/lib/libX11.so.6"


def _load_libx11() -> ctypes.CDLL:
    override = os.environ.get("LIBX11_PATH")
    if override:
        return ctypes.CDLL(override)
    found = ctypes.util.find_library("X11")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            pass
    if os.path.exists(_NIX_LIBX11):
        return ctypes.CDLL(_NIX_LIBX11)
    raise RuntimeError(
        "libX11 not found. Set LIBX11_PATH env var or ensure xorg.libX11 is in "
        "LD_LIBRARY_PATH (shell.nix should add it)."
    )


_LIB = _load_libx11()

_LIB.XOpenDisplay.argtypes = [ctypes.c_char_p]
_LIB.XOpenDisplay.restype = ctypes.c_void_p
_LIB.XCloseDisplay.argtypes = [ctypes.c_void_p]
_LIB.XCloseDisplay.restype = ctypes.c_int
_LIB.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
_LIB.XDefaultRootWindow.restype = ctypes.c_ulong
_LIB.XQueryPointer.argtypes = [
    ctypes.c_void_p, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_uint),
]
_LIB.XQueryPointer.restype = ctypes.c_int
_KEYMAP_T = ctypes.c_char * 32
_LIB.XQueryKeymap.argtypes = [ctypes.c_void_p, _KEYMAP_T]
_LIB.XQueryKeymap.restype = ctypes.c_int
_LIB.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_LIB.XKeysymToKeycode.restype = ctypes.c_ubyte
_LIB.XStringToKeysym.argtypes = [ctypes.c_char_p]
_LIB.XStringToKeysym.restype = ctypes.c_ulong
_LIB.XDefaultScreen.argtypes = [ctypes.c_void_p]
_LIB.XDefaultScreen.restype = ctypes.c_int
_LIB.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.XDisplayWidth.restype = ctypes.c_int
_LIB.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.XDisplayHeight.restype = ctypes.c_int


@dataclass(frozen=True)
class Pointer:
    x: int
    y: int
    button_mask: int


class X11:
    def __init__(self, display: str | None = None):
        name = (display or os.environ.get("DISPLAY") or ":0").encode()
        self._dpy = _LIB.XOpenDisplay(name)
        if not self._dpy:
            raise RuntimeError(f"XOpenDisplay({name!r}) failed")
        self._root = _LIB.XDefaultRootWindow(self._dpy)
        # Reused buffers to keep per-call overhead minimal.
        self._r_root = ctypes.c_ulong()
        self._r_child = ctypes.c_ulong()
        self._r_x = ctypes.c_int()
        self._r_y = ctypes.c_int()
        self._w_x = ctypes.c_int()
        self._w_y = ctypes.c_int()
        self._mask = ctypes.c_uint()
        self._keymap = _KEYMAP_T()

    def close(self) -> None:
        if self._dpy:
            _LIB.XCloseDisplay(self._dpy)
            self._dpy = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def pointer(self) -> Pointer:
        _LIB.XQueryPointer(
            self._dpy, self._root,
            ctypes.byref(self._r_root), ctypes.byref(self._r_child),
            ctypes.byref(self._r_x), ctypes.byref(self._r_y),
            ctypes.byref(self._w_x), ctypes.byref(self._w_y),
            ctypes.byref(self._mask),
        )
        return Pointer(self._r_x.value, self._r_y.value, self._mask.value)

    def keycode_for(self, keysym_name: str) -> int:
        """Resolve e.g. 'z' or 'Z' or 'space' to an X11 keycode (0-255)."""
        ks = _LIB.XStringToKeysym(keysym_name.encode())
        if ks == 0:
            raise ValueError(f"unknown keysym name: {keysym_name!r}")
        kc = _LIB.XKeysymToKeycode(self._dpy, ks)
        if kc == 0:
            raise RuntimeError(f"no keycode for keysym {keysym_name!r}")
        return int(kc)

    def screen_size(self) -> tuple[int, int]:
        """Return (width_px, height_px) of the default screen."""
        scr = _LIB.XDefaultScreen(self._dpy)
        return int(_LIB.XDisplayWidth(self._dpy, scr)), int(_LIB.XDisplayHeight(self._dpy, scr))

    def keys_pressed(self, keycodes: list[int]) -> list[bool]:
        """Return per-keycode pressed state from a single XQueryKeymap call."""
        _LIB.XQueryKeymap(self._dpy, self._keymap)
        out = []
        for kc in keycodes:
            byte = self._keymap[kc >> 3]
            b = byte[0] if isinstance(byte, bytes) else byte
            out.append(bool(b & (1 << (kc & 7))))
        return out
