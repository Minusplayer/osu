"""Live deploy: drive real osu! from a trained checkpoint.

Sim is advanced in lockstep with wall clock; its internal state builds the
observation each tick (reward is ignored). Each tick: build obs from sim
state → forward model → emit cursor + click.

For mouse_mode=True (default), the OS cursor is hijacked — focus osu! before
the countdown ends.
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src_std.emit_std import StdEmitter
from src_std.model_std import ObjectTokenPolicy
from src_std.parse_std import parse_beatmap_std
from src_std.sim import OsuStdEnv, build_map_tensors


def find_keyboard_devices(want_codes):
    import evdev
    devs = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        caps = dev.capabilities().get(evdev.ecodes.EV_KEY, [])
        if all(c in caps for c in want_codes):
            devs.append(dev)
        else:
            dev.close()
    if not devs:
        raise RuntimeError("no input device with requested hotkeys found")
    return devs


class HotkeyListener:

    def __init__(self, start_code, stop_code):
        self.start_code = start_code
        self.stop_code = stop_code
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.devices = find_keyboard_devices([start_code, stop_code])
        print(f"hotkey listeners attached to {len(self.devices)} device(s)")
        self._threads = []

    def start(self):
        for dev in self.devices:
            t = threading.Thread(target=self._read, args=(dev,), daemon=True)
            t.start()
            self._threads.append(t)

    def _read(self, dev):
        import evdev
        for ev in dev.read_loop():
            if ev.type != evdev.ecodes.EV_KEY or ev.value != 1:
                continue
            if ev.code == self.start_code:
                self.start_event.set()
            elif ev.code == self.stop_code:
                self.stop_event.set()
                return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("ckpt")
    p.add_argument("--dt-ms", type=int, default=4)
    p.add_argument("--start-delay", type=int, default=3)
    p.add_argument("--offset-ms", type=int, default=0,
                   help="positive = act earlier; tune for system latency")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--hotkeys", action="store_true")
    p.add_argument("--start-key", default="up")
    p.add_argument("--stop-key", default="down")
    p.add_argument("--keyboard-mode", action="store_true",
                   help="emit Z taps instead of EV_ABS mouse (no aim)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    model = ObjectTokenPolicy(**model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded ckpt @ iter {ckpt.get('iter','?')} "
          f"R/env={ckpt.get('reward_per_env','?')}")

    bm = parse_beatmap_std(args.map)
    if bm.mode != 0:
        sys.exit(f"map mode={bm.mode}, expected 0")
    mp = build_map_tensors(bm, device=device)
    env = OsuStdEnv(mp, batch_size=1, dt_ms=args.dt_ms, device=device)
    print(f"map: {bm.title}  objects={len(bm.notes)}  ticks={env.n_ticks}")

    emitter = StdEmitter(mouse_mode=not args.keyboard_mode)

    hotkeys = None
    if args.hotkeys:
        import evdev
        sc = getattr(evdev.ecodes, f"KEY_{args.start_key.upper()}")
        kc = getattr(evdev.ecodes, f"KEY_{args.stop_key.upper()}")
        hotkeys = HotkeyListener(sc, kc)
        hotkeys.start()
        print(f"focus osu!, press {args.start_key.upper()} to start, "
              f"{args.stop_key.upper()} to stop")
        hotkeys.start_event.wait()
        print("go.")
    else:
        print(f"starting in {args.start_delay}s — focus osu! now")
        for s in range(args.start_delay, 0, -1):
            print(f"  {s}...", flush=True)
            time.sleep(1)

    obj_obs, cur_obs = env.reset()
    t0 = time.perf_counter()
    overruns = 0

    try:
        for tick in range(env.n_ticks):
            if hotkeys and hotkeys.stop_event.is_set():
                print("\nstop hotkey pressed")
                break

            target_wall = t0 + (tick * args.dt_ms - args.offset_ms) / 1000.0
            now = time.perf_counter()
            if target_wall > now:
                time.sleep(target_wall - now)
            elif now - target_wall > args.dt_ms / 1000.0:
                overruns += 1

            with torch.no_grad():
                dxy, click, *_ = model.act(obj_obs, cur_obs,
                                            deterministic=args.deterministic)

            obj_obs, cur_obs, _, done, _ = env.step(dxy, click)
            emitter.move(env.cursor[0, 0].item(), env.cursor[0, 1].item())
            emitter.set_click(bool(click[0].item()))
            emitter.flush()

            if done.any():
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        emitter.release_all()
        emitter.close()
        if overruns:
            print(f"warning: {overruns} ticks ran more than 1 dt late")
        print("emitter closed")


if __name__ == "__main__":
    main()
