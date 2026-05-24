"""Phase 7a — live inference loop.

Runs the trained BC policy at 30 Hz against osu! lazer:

    grim(playfield region) -> 80x60 grayscale stack of 4 frames
                              + map_ctx slice from cached parsed beatmap
                              + state_vec from previous frame's model output
        -> BCPolicy forward -> cursor (px in 512x384) + press logit

Output emission (cursor mousemove + key state) goes via ydotool when
``--emit`` is set; without ``--emit`` the loop runs "dry" and just logs
what it WOULD do (used for 7a infrastructure validation).

The loop idles when tosu reports a non-play state and resumes immediately
when a map begins. Beatmap parsing is cached by md5 — re-parse only on map
change.

Run:
    nix-shell --run "./.venv/bin/python -m src_std.inference.run_bot \\
        --ckpt runs/bc/20260523T193501/best.pt"
"""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from src_std.capture.self_capture import (  # noqa: E402
    GrimGrabber, downsample_to_gray, load_config,
)
from src_std.capture.tosu_client import TosuClient  # noqa: E402
from src_std.capture.x11_input import X11  # noqa: E402
from src_std.inference.uinput_emit import UinputEmitter  # noqa: E402
from src_std.inference.xtest_emit import XTestEmitter  # noqa: E402
from src_std.data.dataset import (  # noqa: E402
    STACK_LEN, FRAME_W, FRAME_H, K_TOKENS, F_OBJ, F_STATE,
    PLAYFIELD_W, PLAYFIELD_H,
    _build_object_table, _build_map_ctx,
)
from src_std.eval_replay import load_model  # noqa: E402
from src_std.parse_std import parse_beatmap_std  # noqa: E402

log = logging.getLogger("run_bot")


def _parse_region(region: str) -> tuple[int, int, int, int]:
    """'316,60 1280x960' -> (316, 60, 1280, 960)."""
    head, dims = region.split(" ")
    x, y = head.split(",")
    w, h = dims.lower().split("x")
    return int(x), int(y), int(w), int(h)


def _playfield_to_screen(cx_pf: float, cy_pf: float,
                         region: tuple[int, int, int, int]) -> tuple[int, int]:
    """osu! internal 512x384 -> absolute screen pixel within region."""
    rx, ry, rw, rh = region
    sx = rx + (cx_pf / PLAYFIELD_W) * rw
    sy = ry + (cy_pf / PLAYFIELD_H) * rh
    return int(round(sx)), int(round(sy))


# Linux input keycodes (see /usr/include/linux/input-event-codes.h)
KEYCODE_Z = 44
KEYCODE_X = 45


class YdotoolEmitter:
    """Spawns ydotool subprocesses for cursor + key state.

    Requires ``ydotoold`` running with a writable /tmp/.ydotool_socket.
    For Phase 7a we only drive K1 (Z by default); K2 isn't a separate
    model output (single press head).
    """

    def __init__(self, ydotool_path: str = "ydotool",
                 keycode: int = KEYCODE_Z):
        self._path = ydotool_path
        self._keycode = keycode
        self._key_down = False

    def move(self, screen_x: int, screen_y: int) -> None:
        subprocess.run(
            [self._path, "mousemove", "--absolute",
             "-x", str(screen_x), "-y", str(screen_y)],
            check=False, capture_output=True,
        )

    def set_key(self, pressed: bool) -> None:
        if pressed == self._key_down:
            return
        state = "1" if pressed else "0"
        subprocess.run(
            [self._path, "key", f"{self._keycode}:{state}"],
            check=False, capture_output=True,
        )
        self._key_down = pressed

    def release_all(self) -> None:
        if self._key_down:
            self.set_key(False)


class DryEmitter:
    """No-op emitter used when --emit is not set."""

    def move(self, screen_x: int, screen_y: int) -> None:
        pass

    def set_key(self, pressed: bool) -> None:
        pass

    def release_all(self) -> None:
        pass


class BeatmapCache:
    """Re-parses the .osu file only when tosu reports a new md5."""

    def __init__(self):
        self.md5: str = ""
        self.obj_table: np.ndarray | None = None
        self.total_dur_ms: float = 0.0

    def update(self, md5: str, songs_folder: str, map_file: str) -> bool:
        if md5 == self.md5 and self.obj_table is not None:
            return False
        if not (md5 and songs_folder and map_file):
            return False
        full = Path(songs_folder) / map_file
        try:
            bm = parse_beatmap_std(full)
        except Exception as e:
            log.warning("beatmap parse failed for %s: %s", full, e)
            return False
        self.md5 = md5
        self.obj_table = _build_object_table(bm)
        self.total_dur_ms = float(bm.notes[-1].t_end) if bm.notes else 0.0
        log.info("beatmap cached: %s  notes=%d  dur=%.1fs",
                 full.name, len(bm.notes), self.total_dur_ms / 1000.0)
        return True


def _build_inputs(stack: deque,
                  map_ctx_row: np.ndarray,
                  state_vec_row: np.ndarray,
                  device: str):
    frames = np.stack(list(stack), axis=0).astype(np.float32) / 255.0
    return (
        torch.from_numpy(frames).unsqueeze(0).to(device),
        torch.from_numpy(map_ctx_row).unsqueeze(0).to(device),
        torch.from_numpy(state_vec_row).unsqueeze(0).to(device),
    )


def _zero_map_ctx() -> np.ndarray:
    return np.zeros((K_TOKENS, F_OBJ), dtype=np.float32)


def _maybe_sleep(tick_start: float, target_dt: float) -> None:
    remain = target_dt - (time.monotonic() - tick_start)
    if remain > 0:
        time.sleep(remain)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to best.pt from a training run.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emit", action="store_true",
                    help="Drive ydotool. Without this, runs dry and only logs.")
    ap.add_argument("--ydotool", default="ydotool",
                    help="(legacy) Path to ydotool binary; not used by default.")
    ap.add_argument("--emit-backend", default="xtest",
                    choices=["xtest", "uinput", "ydotool"],
                    help="Input emission backend when --emit is set. "
                         "xtest = libXtst into XWayland (does not create a "
                         "new device; user's real mouse keeps working).")
    ap.add_argument("--press-thresh", type=float, default=0.5,
                    help="Sigmoid threshold for key press.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Exit after N frames during play (0 = unlimited). "
                         "Useful for 7a smoke validation.")
    ap.add_argument("--start-delay-s", type=float, default=1.0,
                    help="Grace period after entering play state before "
                         "the bot starts emitting input. Avoids interrupting "
                         "osu!'s loading screen.")
    args = ap.parse_args()

    cfg = load_config()
    region = _parse_region(cfg["capture"]["region"])
    target_hz = int(cfg["capture"]["frame_hz"])
    target_dt = 1.0 / target_hz
    grim = GrimGrabber(region=cfg["capture"]["region"],
                       grim_path=cfg["capture"]["grim_path"])

    log.info("loading model from %s on %s", args.ckpt, args.device)
    net = load_model(args.ckpt, args.device)

    if args.emit:
        if args.emit_backend == "ydotool":
            emitter = YdotoolEmitter(args.ydotool)
            log.info("emitter: ydotool (LIVE — bot will move cursor + press keys)")
        elif args.emit_backend == "uinput":
            with X11() as x11:
                screen_w, screen_h = x11.screen_size()
            emitter = UinputEmitter(screen_w, screen_h)
            log.info("emitter: uinput (LIVE — screen=%dx%d, K1=KEY_Z)",
                     screen_w, screen_h)
        else:
            emitter = XTestEmitter()
            log.info("emitter: xtest (LIVE — libXtst into XWayland, K1=Z)")
    else:
        emitter = DryEmitter()
        log.info("emitter: dry-run (no input emitted; use --emit to drive osu!)")

    tosu = TosuClient(precise_url=cfg["tosu"]["precise_url"],
                      v2_url=cfg["tosu"]["v2_url"])
    tosu.start()
    log.info("waiting for tosu v2 first message...")
    while True:
        snap = tosu.snapshot()
        if snap.v2_recv_mono_ns > 0:
            break
        time.sleep(0.1)
    log.info("tosu connected. initial state=%r mode=%d",
             snap.state_name, snap.map_mode_number)

    bm_cache = BeatmapCache()
    play_states = set(cfg["session"]["gameplay_state_names"])

    stop_requested = False

    def _on_signal(signum, _frame):
        nonlocal stop_requested
        log.info("signal %d received; stopping", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    stack: deque = deque(maxlen=STACK_LEN)
    prev_cx_pf = 0.0
    prev_cy_pf = 0.0
    prev_press = 0.0

    frames_this_session = 0
    last_report = time.monotonic()
    frames_since_report = 0
    in_play = False
    play_entered_mono: float = 0.0

    try:
        while not stop_requested:
            tick_start = time.monotonic()
            snap = tosu.snapshot()
            now_playing = (snap.state_name in play_states
                           and not snap.game_paused
                           and snap.map_mode_number == 0)

            if not now_playing:
                if in_play:
                    log.info("exited play state — releasing keys, resetting")
                    emitter.release_all()
                    stack.clear()
                    prev_cx_pf = prev_cy_pf = prev_press = 0.0
                    frames_this_session = 0
                in_play = False
                time.sleep(0.1)
                continue

            if not in_play:
                log.info("entering play state map_md5=%s map_id=%d "
                         "(grace=%.1fs)",
                         snap.map_md5[:8], snap.map_id, args.start_delay_s)
                in_play = True
                play_entered_mono = time.monotonic()

            bm_cache.update(snap.map_md5, snap.songs_folder, snap.map_file)

            rgb = grim.grab()
            gray = downsample_to_gray(rgb, (FRAME_H, FRAME_W))
            stack.append(gray)
            if len(stack) < STACK_LEN:
                _maybe_sleep(tick_start, target_dt)
                continue

            map_t_ms = max(0, int(snap.current_time_ms))
            if bm_cache.obj_table is not None and bm_cache.obj_table.shape[0] > 0:
                mc = _build_map_ctx(bm_cache.obj_table,
                                    np.array([map_t_ms], dtype=np.int64))[0]
            else:
                mc = _zero_map_ctx()

            sv = np.zeros(F_STATE, dtype=np.float32)
            denom = max(1.0, bm_cache.total_dur_ms)
            sv[0] = float(np.clip(map_t_ms / denom, 0.0, 1.0))
            sv[1] = float(np.clip(prev_cx_pf / PLAYFIELD_W, 0.0, 1.0))
            sv[2] = float(np.clip(prev_cy_pf / PLAYFIELD_H, 0.0, 1.0))
            sv[3] = float(prev_press)

            frames_t, mc_t, sv_t = _build_inputs(stack, mc, sv, args.device)
            with torch.no_grad():
                cursor_xy, press_logit = net(frames_t, mc_t, sv_t)
            cx_pf = float(cursor_xy[0, 0].item())
            cy_pf = float(cursor_xy[0, 1].item())
            press_prob = float(torch.sigmoid(press_logit[0]).item())
            press = 1.0 if press_prob >= args.press_thresh else 0.0

            screen_x, screen_y = _playfield_to_screen(cx_pf, cy_pf, region)
            in_grace = (time.monotonic() - play_entered_mono) < args.start_delay_s
            if not in_grace:
                emitter.move(screen_x, screen_y)
                emitter.set_key(press > 0.5)
            else:
                emitter.set_key(False)

            prev_cx_pf, prev_cy_pf, prev_press = cx_pf, cy_pf, press
            frames_this_session += 1
            frames_since_report += 1

            now = time.monotonic()
            if now - last_report >= 1.0:
                hz = frames_since_report / (now - last_report)
                log.info("hz=%.1f  map_t=%dms  pf=(%.0f,%.0f)  scr=(%d,%d)  "
                         "press=%.2f(%s)  emit=%s",
                         hz, map_t_ms, cx_pf, cy_pf, screen_x, screen_y,
                         press_prob, "DOWN" if press > 0.5 else "up",
                         ("grace" if in_grace else "live") if args.emit else "dry")
                last_report = now
                frames_since_report = 0

            if args.max_frames and frames_this_session >= args.max_frames:
                log.info("reached --max-frames=%d, exiting", args.max_frames)
                break

            _maybe_sleep(tick_start, target_dt)

    finally:
        emitter.release_all()
        log.info("run_bot exiting cleanly")


if __name__ == "__main__":
    main()
