"""Phase 5b/c/d: drive the trained model against a clock.

Modes:
  dry   — sweep every tick, log predictions, optionally diff vs a replay
  live  — wall-clock loop, send keys via uinput

Usage:
  python src/infer.py dry  <map.osu> <ckpt.pt> [--replay <r.osr>]
  python src/infer.py live <map.osu> <ckpt.pt> [--start-delay 3] [--threshold 0.5]
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_note_grid
from model import ManiaTransformer
from parse import parse_beatmap, parse_replay


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    keys = ckpt["keys"]
    lookahead_ticks = ckpt["lookahead_ticks"]
    model_kwargs = ckpt.get("model_kwargs",
                            {"keys": keys, "lookahead_ticks": lookahead_ticks})
    model = ManiaTransformer(**model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, keys, lookahead_ticks


def build_grid(map_path: str, keys: int, dt_ms: int, lookahead_ticks: int):
    bm = parse_beatmap(map_path)
    if bm.keys != keys:
        raise ValueError(f"map has {bm.keys} keys, checkpoint trained on {keys}")
    last_note = max(n.t_end for n in bm.notes)
    total_ms = last_note + lookahead_ticks * dt_ms + 1000
    grid = build_note_grid(bm.notes, keys, dt_ms, total_ms)
    first = min(n.t_start for n in bm.notes) // dt_ms
    last = last_note // dt_ms
    return bm, grid, first, last


def find_keyboard_devices(want_codes):
    """Scan /dev/input for ALL devices exposing the given key codes."""
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
    """Reads kernel input from every matching device; sets events on start/stop."""

    def __init__(self, start_code, stop_code):
        self.start_code = start_code
        self.stop_code = stop_code
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.devices = find_keyboard_devices([start_code, stop_code])
        print(f"hotkey listeners attached to {len(self.devices)} device(s):")
        for d in self.devices:
            print(f"  {d.path}  {d.name!r}")
        self._threads = []

    def start(self):
        import evdev
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


@torch.no_grad()
def predict_batch(model, states_np: np.ndarray, device, threshold: float = 0.5):
    x = torch.from_numpy(states_np).to(device)
    logits = model(x)
    if logits.dim() == 3:
        logits = logits[:, 0]   # multi-target model: use the now-tick
    probs = torch.sigmoid(logits)
    return (probs > threshold).cpu().numpy().astype(np.uint8), probs.cpu().numpy()


def cmd_dry(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, keys, lookahead = load_model(args.ckpt, device)
    print(f"model: keys={keys}, lookahead_ticks={lookahead}")

    bm, grid, first, last = build_grid(args.map, keys, args.dt_ms, lookahead)
    n_ticks = last - first + 1
    print(f"map: {len(bm.notes)} notes, ticks [{first}, {last}] ({n_ticks} active)")

    states = np.stack([grid[t:t + lookahead] for t in range(first, last + 1)])
    preds = []
    bs = 512
    for i in range(0, len(states), bs):
        p, _ = predict_batch(model, states[i:i + bs], device, args.threshold)
        preds.append(p)
    preds = np.concatenate(preds)

    print(f"\npredicted keystate for {len(preds)} ticks")
    held_frac = preds.mean(axis=0)
    print(f"  press fraction per col: {[f'{x*100:.1f}%' for x in held_frac]}")

    if args.replay:
        from dataset import build_action_grid
        events = parse_replay(args.replay)
        total_ms = (last + lookahead) * args.dt_ms + 1000
        truth = build_action_grid(events, keys, args.dt_ms, total_ms)
        truth = truth[first:last + 1].astype(np.uint8)
        match = (preds == truth).mean()
        per_col = (preds == truth).mean(axis=0)
        print(f"\nvs replay:")
        print(f"  overall agreement: {match*100:.2f}%")
        print(f"  per-col agreement: {[f'{x*100:.1f}%' for x in per_col]}")
        diffs = np.where(preds != truth)
        if len(diffs[0]):
            first_diff = diffs[0][0]
            t_ms = (first + first_diff) * args.dt_ms
            print(f"  first disagreement: tick {first + first_diff} (~{t_ms}ms)")
            print(f"    pred={preds[first_diff].tolist()} truth={truth[first_diff].tolist()}")


def cmd_live(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, keys, lookahead = load_model(args.ckpt, device)
    bm, grid, first, last = build_grid(args.map, keys, args.dt_ms, lookahead)
    print(f"model: keys={keys}, lookahead_ticks={lookahead}")
    print(f"map: {len(bm.notes)} notes, ticks [{first}, {last}]")

    from emit import ManiaEmitter

    precomputed = None
    if args.precompute:
        print("\nprecomputing keystates...")
        t_pre = time.perf_counter()
        states = np.stack([grid[t:t + lookahead] for t in range(first, last + 1)])
        chunks = []
        bs = 512
        for i in range(0, len(states), bs):
            p, _ = predict_batch(model, states[i:i + bs], device, args.threshold)
            chunks.append(p)
        precomputed = np.concatenate(chunks)
        print(f"  {len(precomputed)} ticks baked in {time.perf_counter()-t_pre:.2f}s")

    emitter = ManiaEmitter()

    hotkeys = None
    if args.hotkeys:
        import evdev
        start_code = getattr(evdev.ecodes, f"KEY_{args.start_key.upper()}")
        stop_code = getattr(evdev.ecodes, f"KEY_{args.stop_key.upper()}")
        hotkeys = HotkeyListener(start_code, stop_code)
        hotkeys.start()
        print(f"\nfocus osu!, then press {args.start_key.upper()} to start, "
              f"{args.stop_key.upper()} to stop")
        hotkeys.start_event.wait()
        print("go.")
    else:
        print(f"\nstarting in {args.start_delay}s — focus osu! window now")
        for s in range(args.start_delay, 0, -1):
            print(f"  {s}...", flush=True)
            time.sleep(1)

    # t0 = audio time 0 of the song. Press the hotkey when the song starts;
    # the loop then idles through any lead-in until the first note tick.
    t0 = time.perf_counter()
    overruns = 0
    zeros = np.zeros(keys, dtype=np.uint8)

    try:
        for tick in range(0, last + 1):
            if hotkeys and hotkeys.stop_event.is_set():
                print("\nstop hotkey pressed")
                break

            target_wall = t0 + (tick * args.dt_ms - args.offset_ms) / 1000.0
            now = time.perf_counter()
            if target_wall > now:
                time.sleep(target_wall - now)
            elif tick >= first and now - target_wall > args.dt_ms / 1000.0:
                overruns += 1

            if tick < first:
                target = zeros
            elif precomputed is not None:
                target = precomputed[tick - first]
            else:
                state = grid[tick:tick + lookahead][None]
                with torch.no_grad():
                    logits = model(torch.from_numpy(state).to(device))
                    if logits.dim() == 3:
                        logits = logits[:, 0]
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                target = (probs > args.threshold).astype(np.uint8)
            emitter.set_keystate(target)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        emitter.release_all()
        emitter.close()
        if overruns:
            print(f"warning: {overruns} ticks ran more than 1 dt late")
        print("emitter closed")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pd = sub.add_parser("dry", help="sweep ticks, optionally diff vs replay")
    pd.add_argument("map")
    pd.add_argument("ckpt")
    pd.add_argument("--replay", default=None)
    pd.add_argument("--dt-ms", type=int, default=4)
    pd.add_argument("--threshold", type=float, default=0.5)
    pd.set_defaults(func=cmd_dry)

    pl = sub.add_parser("live", help="real-time playback via uinput")
    pl.add_argument("map")
    pl.add_argument("ckpt")
    pl.add_argument("--dt-ms", type=int, default=4)
    pl.add_argument("--threshold", type=float, default=0.5)
    pl.add_argument("--start-delay", type=int, default=3)
    pl.add_argument("--offset-ms", type=int, default=0,
                    help="positive = press earlier; tune to compensate latency")
    pl.add_argument("--precompute", action="store_true",
                    help="batch-infer all keystates before playback (no GPU work in loop)")
    pl.add_argument("--hotkeys", action="store_true",
                    help="wait for start-key instead of countdown; stop-key aborts")
    pl.add_argument("--start-key", default="up",
                    help="hotkey to begin playback (default: up)")
    pl.add_argument("--stop-key", default="down",
                    help="hotkey to abort playback (default: down)")
    pl.set_defaults(func=cmd_live)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
