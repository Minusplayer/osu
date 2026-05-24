"""uinput-based emitter for run_bot.

Takes ABSOLUTE SCREEN pixel coordinates (not playfield) and normalizes to
UABS_MAX using the actual screen resolution so the cursor lands inside the
osu! window region rather than at the screen corner.

Press is emitted as KEY_Z by default (osu! standard K1). Uses evdev UInput;
requires /dev/uinput access (user in `input` group).
"""

from __future__ import annotations

import time

from evdev import UInput, AbsInfo, ecodes as e

UABS_MAX = 32767


class UinputEmitter:
    def __init__(self, screen_w: int, screen_h: int,
                 press_key: str = "z",
                 device_name: str = "aiosu-bot-virtual"):
        if screen_w <= 0 or screen_h <= 0:
            raise ValueError(f"bad screen size {screen_w}x{screen_h}")
        self.screen_w = screen_w
        self.screen_h = screen_h

        self._press_key = getattr(e, f"KEY_{press_key.upper()}")
        caps = {
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(0, 0, UABS_MAX, 0, 0, 0)),
                (e.ABS_Y, AbsInfo(0, 0, UABS_MAX, 0, 0, 0)),
            ],
            e.EV_KEY: [e.BTN_LEFT, self._press_key],
        }
        self.ui = UInput(caps, name=device_name)
        self._key_down = False
        # Prime ABS_X/Y at center so libinput/osu! don't see ABS=0 (which
        # libinput interprets as a corner) when the device is first picked up.
        # We emit a stream of centered events with delays so the compositor
        # has time to enumerate the device and read the initial state.
        cx = UABS_MAX // 2
        cy = UABS_MAX // 2
        for _ in range(5):
            self.ui.write(e.EV_ABS, e.ABS_X, cx)
            self.ui.write(e.EV_ABS, e.ABS_Y, cy)
            self.ui.syn()
            time.sleep(0.1)

    def move(self, screen_x: int, screen_y: int) -> None:
        sx = max(0, min(self.screen_w - 1, int(screen_x)))
        sy = max(0, min(self.screen_h - 1, int(screen_y)))
        ax = int(sx / (self.screen_w - 1) * UABS_MAX)
        ay = int(sy / (self.screen_h - 1) * UABS_MAX)
        self.ui.write(e.EV_ABS, e.ABS_X, ax)
        self.ui.write(e.EV_ABS, e.ABS_Y, ay)
        self.ui.syn()

    def set_key(self, pressed: bool) -> None:
        if pressed == self._key_down:
            return
        self.ui.write(e.EV_KEY, self._press_key, 1 if pressed else 0)
        self.ui.syn()
        self._key_down = pressed

    def release_all(self) -> None:
        if self._key_down:
            self.set_key(False)

    def close(self) -> None:
        try:
            self.release_all()
        finally:
            self.ui.close()
