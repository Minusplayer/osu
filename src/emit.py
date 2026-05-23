"""Virtual keyboard for osu!mania bot output (Linux uinput).

Independent of model + game — exercise standalone via the CLI self-test
to verify perms and timing before wiring into inference.
"""

import sys
import time
from typing import Sequence

from evdev import UInput, ecodes as e


# Default mania keybinds (osu! defaults).
#   4K → D F J K
#   7K → S D F Space J K L
DEFAULT_4K = (e.KEY_D, e.KEY_F, e.KEY_J, e.KEY_K)
DEFAULT_7K = (e.KEY_S, e.KEY_D, e.KEY_F, e.KEY_SPACE,
              e.KEY_J, e.KEY_K, e.KEY_L)


class ManiaEmitter:
    """Wraps a uinput virtual keyboard for K-column mania output.

    Tracks held columns so callers pass target keystates and the emitter
    computes the press/release diff.
    """

    def __init__(self, key_codes: Sequence[int] = DEFAULT_4K,
                 name: str = "aiosu-mania"):
        self.key_codes = tuple(key_codes)
        self.keys = len(self.key_codes)
        self.held = [False] * self.keys
        self.ui = UInput(
            {e.EV_KEY: list(self.key_codes)},
            name=name,
            vendor=0x1209, product=0xA105, version=1,
        )
        # uinput devices need a brief settling time before the host accepts input
        time.sleep(0.1)

    def set_keystate(self, target: Sequence[bool]) -> None:
        """Press/release keys to match `target`. len(target) must equal self.keys."""
        if len(target) != self.keys:
            raise ValueError(
                f"target length {len(target)} != configured keys {self.keys}")
        emitted = False
        for col in range(self.keys):
            want = bool(target[col])
            if want != self.held[col]:
                self.ui.write(e.EV_KEY, self.key_codes[col], 1 if want else 0)
                self.held[col] = want
                emitted = True
        if emitted:
            self.ui.syn()

    def release_all(self) -> None:
        self.set_keystate([False] * self.keys)

    def close(self) -> None:
        self.release_all()
        self.ui.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _self_test():
    """Type 'dfjk' with brief presses, then a 4-key chord. Focus a text field first."""
    print("opening uinput device...")
    try:
        emitter = ManiaEmitter()
    except PermissionError as ex:
        print(f"\nERROR: cannot open /dev/uinput ({ex})")
        print("Fix:  sudo usermod -aG input $USER  (then re-login)")
        print("and a udev rule for /dev/uinput, e.g.:")
        print('  KERNEL=="uinput", GROUP="input", MODE="0660", '
              'OPTIONS+="static_node=uinput"')
        sys.exit(1)

    print("focus a text field — typing in 3 seconds...")
    time.sleep(3.0)

    print("singles:")
    for col in range(emitter.keys):
        state = [False] * emitter.keys
        state[col] = True
        emitter.set_keystate(state)
        time.sleep(0.06)
        emitter.set_keystate([False] * emitter.keys)
        time.sleep(0.06)

    print("chord:")
    emitter.set_keystate([True] * emitter.keys)
    time.sleep(0.08)
    emitter.set_keystate([False] * emitter.keys)

    emitter.close()
    print("done — you should have seen 'dfjk' + a 4-key chord (default keybinds).")


if __name__ == "__main__":
    _self_test()
