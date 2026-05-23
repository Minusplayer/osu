"""uinput emitter for osu! standard: absolute cursor + left/right click.

Phase-1 uses mouse_mode=True: one uinput device drives both aim (EV_ABS X/Y)
and tap (BTN_LEFT). osu! reads the OS cursor in absolute mode; the desktop
scales 0..UABS_MAX to the active screen resolution. Coordinates passed to
move() are osu! playfield pixels (0..512, 0..384).
"""

import time

from evdev import UInput, AbsInfo, ecodes as e

PLAYFIELD_W = 512
PLAYFIELD_H = 384
UABS_MAX = 32767


class StdEmitter:
    """Drives cursor + click via uinput."""

    def __init__(self, mouse_mode: bool = True,
                 tap_keys=("z", "x"),
                 device_name: str = "aiosu-std-virtual"):
        self.mouse_mode = mouse_mode

        caps = {}
        if mouse_mode:
            caps[e.EV_ABS] = [
                (e.ABS_X, AbsInfo(0, 0, UABS_MAX, 0, 0, 0)),
                (e.ABS_Y, AbsInfo(0, 0, UABS_MAX, 0, 0, 0)),
            ]
            caps[e.EV_KEY] = [e.BTN_LEFT, e.BTN_RIGHT]
        else:
            caps[e.EV_KEY] = [getattr(e, f"KEY_{k.upper()}") for k in tap_keys]

        self.ui = UInput(caps, name=device_name)
        self.tap_keys = [getattr(e, f"KEY_{k.upper()}") for k in tap_keys]
        self._cur_click = False
        time.sleep(0.1)

    def move(self, x_px: float, y_px: float):
        if not self.mouse_mode:
            return
        x_px = max(0.0, min(PLAYFIELD_W, x_px))
        y_px = max(0.0, min(PLAYFIELD_H, y_px))
        ax = int(x_px / PLAYFIELD_W * UABS_MAX)
        ay = int(y_px / PLAYFIELD_H * UABS_MAX)
        self.ui.write(e.EV_ABS, e.ABS_X, ax)
        self.ui.write(e.EV_ABS, e.ABS_Y, ay)

    def set_click(self, pressed: bool):
        if pressed == self._cur_click:
            return
        if self.mouse_mode:
            self.ui.write(e.EV_KEY, e.BTN_LEFT, 1 if pressed else 0)
        else:
            key = self.tap_keys[0]
            self.ui.write(e.EV_KEY, key, 1 if pressed else 0)
        self._cur_click = pressed

    def flush(self):
        self.ui.syn()

    def release_all(self):
        if self._cur_click:
            self.set_click(False)
            self.flush()

    def close(self):
        try:
            self.release_all()
        finally:
            self.ui.close()
