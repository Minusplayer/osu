"""Self-capture loop for osu!std behavioral cloning.

Records, at 30 Hz while osu! is in the gameplay state, one .npz per session
containing per-frame:

    t_ns           int64    monotonic-ns timestamp (relative to session start)
    frame          uint8    (96, 96) grayscale, downsampled from the playfield
                            screen region
    cursor_x       int16    raw screen X (still in screen coords; the playfield
                            transform happens later in dataset prep)
    cursor_y       int16    raw screen Y
    k1             bool     osu!std K1 (default: Z) pressed at frame time
    k2             bool     osu!std K2 (default: X) pressed at frame time
    map_time_ms    int32    latest tosu precise currentTime (-1 if no message
                            received yet)
    map_age_ns     int32    age of the map_time_ms snapshot at frame time
                            (how stale the precise message was — useful for
                            judging alignment quality)
    game_paused    bool     latest tosu game.paused state

Metadata is stored under the "meta" key in the .npz as a JSON-encoded dict:
map md5/id/set/path, capture region, frame size, key bindings, tosu poll rates,
session start ISO timestamp, achieved Hz.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src_std.capture.tosu_client import TosuClient  # noqa: E402
from src_std.capture.x11_input import X11  # noqa: E402

log = logging.getLogger("self_capture")


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "training_config.json"

DEFAULT_CONFIG = {
    "capture": {
        "region": "316,60 1280x960",
        "frame_hz": 30,
        "frame_size": [96, 96],
        "grim_path": "grim",
    },
    "keys": {
        "k1": "z",
        "k2": "x",
    },
    "tosu": {
        "precise_url": "ws://127.0.0.1:24050/websocket/v2/precise",
        "v2_url": "ws://127.0.0.1:24050/websocket/v2",
        "precise_poll_ms": 10,
        "common_poll_ms": 50,
    },
    "session": {
        "out_dir": "data/sessions",
        "gameplay_state_names": ["play"],
        "require_mode_std": True,
    },
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    with path.open("w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    log.info("wrote default config to %s", path)
    return DEFAULT_CONFIG


class GrimGrabber:
    """Run `grim -t ppm -g REGION -` and parse the resulting raw RGB.

    PPM is ~30x faster than PNG because we skip zlib compression; the bytes
    just round-trip through a pipe and into a numpy view.
    """

    def __init__(self, region: str, grim_path: str = "grim"):
        resolved = _resolve_grim(grim_path)
        self._cmd = [resolved, "-t", "ppm", "-g", region, "-"]

    def grab(self) -> np.ndarray:
        proc = subprocess.run(self._cmd, capture_output=True, check=True)
        return _parse_ppm(proc.stdout)


# Known nix-store path as a fallback when grim isn't on PATH (claude's shell
# doesn't inherit the nix-shell PATH that adds it).
_GRIM_NIX_FALLBACK = "/nix/store/8s5p1if67gzz6ymdksbn28b41v1mf06l-grim-1.5.0/bin/grim"


def _resolve_grim(path: str) -> str:
    # Absolute path given — trust it.
    if path.startswith("/"):
        return path
    found = shutil.which(path)
    if found:
        return found
    # Fall back to a glob of /nix/store for any grim-*/bin/grim.
    import glob
    for p in glob.glob("/nix/store/*-grim-*/bin/grim"):
        return p
    if Path(_GRIM_NIX_FALLBACK).exists():
        return _GRIM_NIX_FALLBACK
    raise FileNotFoundError(
        f"grim not found via PATH or /nix/store glob. Set capture.grim_path in "
        f"training_config.json to an absolute path."
    )


def _parse_ppm(data: bytes) -> np.ndarray:
    nl1 = data.index(b"\n")
    if data[:nl1] != b"P6":
        raise ValueError(f"unexpected PPM magic: {data[:nl1]!r}")
    nl2 = data.index(b"\n", nl1 + 1)
    w, h = (int(x) for x in data[nl1 + 1 : nl2].split())
    nl3 = data.index(b"\n", nl2 + 1)
    pix = np.frombuffer(data[nl3 + 1 :], dtype=np.uint8)
    if pix.size != w * h * 3:
        raise ValueError(f"PPM payload size {pix.size} != {w}*{h}*3")
    return pix.reshape(h, w, 3)


def downsample_to_gray(rgb: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """RGB H×W×3 uint8 -> out_size grayscale uint8 (bilinear)."""
    img = Image.fromarray(rgb, mode="RGB").convert("L")
    img = img.resize(out_size[::-1], Image.BILINEAR)  # PIL takes (W, H)
    return np.asarray(img, dtype=np.uint8)


@dataclass
class FrameRow:
    t_ns: int
    frame: np.ndarray  # (H, W) uint8
    cursor_x: int
    cursor_y: int
    k1: bool
    k2: bool
    map_time_ms: int
    map_age_ns: int
    game_paused: bool


class SessionWriter:
    """Accumulates frames in memory and writes a single .npz on close."""

    def __init__(self, out_dir: Path, meta: dict):
        self.out_dir = out_dir
        self.meta = dict(meta)
        self.rows: list[FrameRow] = []
        self.dropped = 0
        self.started_iso = datetime.now(timezone.utc).isoformat()

    def append(self, row: FrameRow) -> None:
        self.rows.append(row)

    def note_drop(self, n: int = 1) -> None:
        self.dropped += n

    def close(self) -> Path | None:
        if not self.rows:
            log.info("session had 0 frames; nothing to write")
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        t_ns = np.array([r.t_ns for r in self.rows], dtype=np.int64)
        frame = np.stack([r.frame for r in self.rows]).astype(np.uint8)
        cursor_x = np.array([r.cursor_x for r in self.rows], dtype=np.int16)
        cursor_y = np.array([r.cursor_y for r in self.rows], dtype=np.int16)
        k1 = np.array([r.k1 for r in self.rows], dtype=bool)
        k2 = np.array([r.k2 for r in self.rows], dtype=bool)
        map_time_ms = np.array([r.map_time_ms for r in self.rows], dtype=np.int32)
        map_age_ns = np.array([r.map_age_ns for r in self.rows], dtype=np.int32)
        game_paused = np.array([r.game_paused for r in self.rows], dtype=bool)

        total_s = (self.rows[-1].t_ns - self.rows[0].t_ns) / 1e9 if len(self.rows) > 1 else 0.0
        achieved_hz = (len(self.rows) - 1) / total_s if total_s > 0 else 0.0

        meta = dict(self.meta)
        meta.update({
            "started_iso": self.started_iso,
            "frames": len(self.rows),
            "duration_s": total_s,
            "achieved_hz": achieved_hz,
            "dropped_frames": self.dropped,
        })

        safe_iso = self.started_iso.replace(":", "").split(".")[0]
        map_id = meta.get("map_id", 0)
        out = self.out_dir / f"session_{safe_iso}_map{map_id}.npz"
        np.savez_compressed(
            out,
            t_ns=t_ns,
            frame=frame,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            k1=k1,
            k2=k2,
            map_time_ms=map_time_ms,
            map_age_ns=map_age_ns,
            game_paused=game_paused,
            meta=np.array(json.dumps(meta), dtype=object),
        )
        return out


def _initial_session_meta(snap, cfg, kc_k1, kc_k2) -> dict:
    cap = cfg["capture"]
    tosu = cfg["tosu"]
    keys = cfg["keys"]
    songs = snap.songs_folder
    map_file = snap.map_file
    abs_path = str(Path(songs) / map_file) if (songs and map_file) else ""
    return {
        "map_md5": snap.map_md5,
        "map_id": snap.map_id,
        "map_set": snap.map_set,
        "map_mode_number": snap.map_mode_number,
        "map_file_rel": map_file,
        "map_file_abs": abs_path,
        "songs_folder": songs,
        "region": cap["region"],
        "frame_hz": int(cap["frame_hz"]),
        "frame_size": list(cap["frame_size"]),
        "k1_name": keys["k1"],
        "k2_name": keys["k2"],
        "k1_keycode": kc_k1,
        "k2_keycode": kc_k2,
        "tosu_precise_poll_ms": int(tosu["precise_poll_ms"]),
        "tosu_common_poll_ms": int(tosu["common_poll_ms"]),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Capture a single session then exit.")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="Hard cap on a single session in seconds (0 = no cap).")
    args = parser.parse_args()

    cfg = load_config()
    cap_cfg = cfg["capture"]
    keys_cfg = cfg["keys"]
    tosu_cfg = cfg["tosu"]
    sess_cfg = cfg["session"]

    target_hz = int(cap_cfg["frame_hz"])
    target_dt = 1.0 / target_hz
    out_size = tuple(cap_cfg["frame_size"])

    grim = GrimGrabber(region=cap_cfg["region"], grim_path=cap_cfg["grim_path"])
    x11 = X11()
    kc_k1 = x11.keycode_for(keys_cfg["k1"])
    kc_k2 = x11.keycode_for(keys_cfg["k2"])
    log.info("X11 keycodes: k1(%s)=%d  k2(%s)=%d",
             keys_cfg["k1"], kc_k1, keys_cfg["k2"], kc_k2)

    tosu = TosuClient(precise_url=tosu_cfg["precise_url"], v2_url=tosu_cfg["v2_url"])
    tosu.start()
    log.info("waiting for tosu v2 first message...")
    while True:
        snap = tosu.snapshot()
        if snap.v2_recv_mono_ns > 0:
            break
        time.sleep(0.1)
    log.info("tosu connected. initial state=%r mode=%d paused=%s",
             snap.state_name, snap.map_mode_number, snap.game_paused)

    gameplay_states = set(sess_cfg["gameplay_state_names"])
    require_std = bool(sess_cfg["require_mode_std"])
    out_dir = (REPO_ROOT / sess_cfg["out_dir"]).resolve()

    stop_requested = False

    def _on_signal(signum, frame):
        nonlocal stop_requested
        log.info("signal %d received; stopping after current session", signum)
        stop_requested = True
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop_requested:
            log.info("idle: waiting for gameplay state %s (current=%r)...",
                     sorted(gameplay_states), tosu.snapshot().state_name)
            while not stop_requested:
                s = tosu.snapshot()
                if s.state_name in gameplay_states:
                    if not require_std or s.map_mode_number == 0:
                        break
                time.sleep(0.05)
            if stop_requested:
                break

            session_meta = _initial_session_meta(tosu.snapshot(), cfg, kc_k1, kc_k2)
            log.info("=== STARTING SESSION ===  map_id=%s md5=%s mode=%d",
                     session_meta.get("map_id"),
                     (session_meta["map_md5"] or "")[:8],
                     session_meta["map_mode_number"])
            writer = SessionWriter(out_dir=out_dir, meta=session_meta)
            sess_start_mono = time.monotonic_ns()
            sess_start_perf = time.perf_counter()
            tick = 0
            last_log_t = sess_start_perf

            while not stop_requested:
                target_t = sess_start_perf + (tick + 1) * target_dt

                snap = tosu.snapshot()
                if snap.state_name not in gameplay_states:
                    log.info("state transitioned to %r — ending session", snap.state_name)
                    break
                if args.max_seconds and (time.perf_counter() - sess_start_perf) >= args.max_seconds:
                    log.info("hit --max-seconds=%g — ending session", args.max_seconds)
                    break

                # Read inputs + tosu snapshot RIGHT BEFORE grim so the cursor
                # read aligns with the screen state visible during the capture
                # and map_age is naturally bounded by the tosu Precise poll
                # interval (instead of being polluted by grim's 14ms grab).
                ptr = x11.pointer()
                pressed = x11.keys_pressed([kc_k1, kc_k2])
                snap_pre = tosu.snapshot()
                t_ns_pre = time.monotonic_ns()

                try:
                    rgb = grim.grab()
                except subprocess.CalledProcessError as e:
                    log.warning("grim failed (rc=%d) — dropping frame", e.returncode)
                    writer.note_drop()
                    sleep = target_t - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    tick += 1
                    continue
                gray96 = downsample_to_gray(rgb, out_size)

                if snap_pre.precise_recv_mono_ns > 0:
                    map_age = t_ns_pre - snap_pre.precise_recv_mono_ns
                    map_time_ms = snap_pre.current_time_ms
                else:
                    map_age = -1
                    map_time_ms = -1
                map_age_i32 = max(min(map_age, 2_147_483_647), -2_147_483_648)

                writer.append(FrameRow(
                    t_ns=t_ns_pre - sess_start_mono,
                    frame=gray96,
                    cursor_x=ptr.x,
                    cursor_y=ptr.y,
                    k1=pressed[0],
                    k2=pressed[1],
                    map_time_ms=int(map_time_ms),
                    map_age_ns=int(map_age_i32),
                    game_paused=snap_pre.game_paused,
                ))

                if time.perf_counter() - last_log_t >= 5.0:
                    elapsed = time.perf_counter() - sess_start_perf
                    log.info("[%6.1fs] frames=%d  hz=%.1f  drops=%d  map_t=%d ms",
                             elapsed, len(writer.rows), len(writer.rows) / elapsed,
                             writer.dropped, map_time_ms)
                    last_log_t = time.perf_counter()

                # Schedule keeping.
                now = time.perf_counter()
                if now >= target_t + target_dt:
                    behind = int((now - target_t) / target_dt)
                    log.warning("schedule slip: %d ticks behind", behind + 1)
                    writer.note_drop(behind)
                    tick += behind
                else:
                    sleep = target_t - now
                    if sleep > 0:
                        time.sleep(sleep)
                tick += 1

            out = writer.close()
            if out:
                log.info("session written: %s  (%d frames, %d drops)",
                         out, len(writer.rows), writer.dropped)
            else:
                log.info("session ended with no frames")
            if args.once:
                break
    finally:
        x11.close()
        tosu.stop()


if __name__ == "__main__":
    main()
