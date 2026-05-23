"""Phase 4/6 — osu!std behavioral-cloning dataset built from session .npz.

One sample = (state, action):

  state  : dict with
             frames    : (4, 60, 80) float32 in [0,1] — stacked grayscale
                         frames at 30 Hz. Stored 96×96 captures are resized
                         to 80×60 (W×H) via PIL.BILINEAR at build time to
                         match the model input (and the original 4:3 spec).
                         Frame 3 is "current"; 0..2 are the three preceding
                         frames within the SAME contiguous play segment.
             map_ctx   : (K=12, F_obj=8) float32 — next K upcoming hit
                         objects relative to the current frame's map_time.
                         Per token: [dt_norm, x_norm, y_norm, dur_norm,
                         is_circle, is_slider, is_spinner, present].
                         Padded slots have present=0; the transformer
                         masks attention over them.
             state_vec : (F_state=4,) float32 — [map_time_norm,
                         prev_cursor_x_norm, prev_cursor_y_norm, prev_press].
                         "prev_*" come from the frame at center-1 (i.e.
                         t_now-33ms). map_time_norm = map_time_ms / map
                         total duration (clipped to [0,1]).
  action : dict with
             cursor_pf : (2,) float32  — playfield coords in [0,512] × [0,384]
             press     : float32 (0 or 1) — derived from beatmap, not from
                         captured k1/k2 (captured keys may be all-zero on
                         Relax mod)

Filtering (per Phase 3 findings):

  - drop frames where ``game_paused``
  - drop frames where ``map_time_ms < 0`` (osu! lead-in pre-roll)
  - segment on map-time back-jumps or gaps > ``GAP_MS`` (default 2 s);
    frame stacks must not span a segment boundary
  - require at least ``STACK_LEN`` frames in a segment to emit any sample

Press label is True when ``map_time_ms`` falls within any hit object's
active window expanded by ``LABEL_WINDOW_MS`` on each side:

  CIRCLE  : [t_start - W, t_start + W]
  SLIDER  : [t_start - W, t_end   + W]
  SPINNER : [t_start - W, t_end   + W]

``LABEL_WINDOW_MS`` defaults to 60 ms (~1.5 × the σ=46 ms clock residual
measured in Phase 3). Tighter windows mislabel hit frames as no-press
due to alignment jitter.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src_std.parse_std import (  # noqa: E402
    CIRCLE, SLIDER, SPINNER, parse_beatmap_std,
)

REGION_X, REGION_Y, REGION_W, REGION_H = 316, 60, 1280, 960

STACK_LEN = 4
LABEL_WINDOW_MS = 60
GAP_MS = 2000   # any map-time gap larger than this is a discontinuity

# Model input — 80x60 (W x H), 4:3 playfield aspect. Stored captures are
# 96x96; we resize at build time with PIL.BILINEAR to match the capture
# pipeline's downsample interp.
FRAME_W = 80
FRAME_H = 60

# Map-context tokens.
K_TOKENS = 12
F_OBJ = 8
DT_HORIZON_MS = 2000.0   # objects beyond this are still emitted with dt=1.0
DUR_HORIZON_MS = 4000.0
PLAYFIELD_W = 512.0
PLAYFIELD_H = 384.0

# State vector.
F_STATE = 4


@dataclass
class BuildStats:
    """Counts produced while building a session's per-frame mask."""
    total_frames: int = 0
    dropped_paused: int = 0
    dropped_negative_t: int = 0
    dropped_seg_too_short: int = 0
    dropped_stack_warmup: int = 0   # frames at segment start with no full history
    kept_frames: int = 0
    samples: int = 0
    press_positive: int = 0
    n_segments: int = 0
    map_title: str = ""
    map_id: int = 0
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["press_positive_rate"] = (
            self.press_positive / self.samples if self.samples else None
        )
        return d


def screen_to_playfield(sx: np.ndarray, sy: np.ndarray):
    px = (sx.astype(np.float32) - REGION_X) / REGION_W * 512.0
    py = (sy.astype(np.float32) - REGION_Y) / REGION_H * 384.0
    return px, py


def _press_mask_from_beatmap(bm, map_t_ms: np.ndarray,
                             window_ms: int) -> np.ndarray:
    """For each frame's map_time, True if any hit-object window contains it.

    Build padded intervals, merge overlaps, then binary-search per frame.
    """
    intervals: list[tuple[int, int]] = []
    for n in bm.notes:
        if n.type == CIRCLE:
            a, b = n.t_start, n.t_start
        else:  # SLIDER, SPINNER — hold for the full duration
            a, b = n.t_start, n.t_end
        intervals.append((a - window_ms, b + window_ms))
    if not intervals:
        return np.zeros_like(map_t_ms, dtype=bool)

    intervals.sort()
    merged_starts = [intervals[0][0]]
    merged_ends = [intervals[0][1]]
    for s, e in intervals[1:]:
        if s <= merged_ends[-1]:
            if e > merged_ends[-1]:
                merged_ends[-1] = e
        else:
            merged_starts.append(s)
            merged_ends.append(e)
    ms = np.array(merged_starts, dtype=np.int64)
    me = np.array(merged_ends, dtype=np.int64)

    t = map_t_ms.astype(np.int64)
    idx = np.searchsorted(ms, t, side="right") - 1
    out = np.zeros_like(map_t_ms, dtype=bool)
    valid = idx >= 0
    out[valid] = me[idx[valid]] >= t[valid]
    return out


def _resize_stack_to_80x60(frames_96: np.ndarray) -> np.ndarray:
    """Resize (M, 96, 96) uint8 -> (M, 60, 80) uint8 via PIL.BILINEAR.

    Matches the capture pipeline's interp (PIL.Image.resize, BILINEAR),
    so the model sees the same low-frequency content the capture produced.
    """
    if frames_96.shape[0] == 0:
        return np.empty((0, FRAME_H, FRAME_W), dtype=np.uint8)
    out = np.empty((frames_96.shape[0], FRAME_H, FRAME_W), dtype=np.uint8)
    for i, f in enumerate(frames_96):
        im = Image.fromarray(f, mode="L").resize(
            (FRAME_W, FRAME_H), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)
    return out


def _build_object_table(bm) -> np.ndarray:
    """Pack beatmap objects into a (N, F_obj-1) array (no `present` flag).

    Columns (in 'raw' units, not normalized):
      [0] t_start_ms (int as float)
      [1] x_px
      [2] y_px
      [3] duration_ms (t_end - t_start; 0 for circles)
      [4] is_circle
      [5] is_slider
      [6] is_spinner
    """
    rows = np.zeros((len(bm.notes), F_OBJ - 1), dtype=np.float32)
    for i, n in enumerate(bm.notes):
        rows[i, 0] = float(n.t_start)
        rows[i, 1] = float(n.x)
        rows[i, 2] = float(n.y)
        rows[i, 3] = float(max(0, n.t_end - n.t_start))
        rows[i, 4] = 1.0 if n.type == CIRCLE else 0.0
        rows[i, 5] = 1.0 if n.type == SLIDER else 0.0
        rows[i, 6] = 1.0 if n.type == SPINNER else 0.0
    return rows


def _build_map_ctx(obj_table: np.ndarray,
                   map_t_ms: np.ndarray) -> np.ndarray:
    """For each frame's map_time, encode the next K upcoming hit objects.

    Returns (M, K, F_obj) float32. The 8th feature is `present` (1 if real
    token, 0 if padding past end-of-map).
    """
    M = map_t_ms.shape[0]
    out = np.zeros((M, K_TOKENS, F_OBJ), dtype=np.float32)
    if obj_table.shape[0] == 0:
        return out
    starts = obj_table[:, 0]
    # For each frame, first object with t_start >= map_time.
    idx = np.searchsorted(starts, map_t_ms.astype(np.float64), side="left")
    for f in range(M):
        i0 = int(idx[f])
        n_take = min(K_TOKENS, obj_table.shape[0] - i0)
        if n_take <= 0:
            continue
        toks = obj_table[i0:i0 + n_take]   # (n_take, 7)
        dt = (toks[:, 0] - float(map_t_ms[f])) / DT_HORIZON_MS
        out[f, :n_take, 0] = np.clip(dt, 0.0, 1.0)
        out[f, :n_take, 1] = toks[:, 1] / PLAYFIELD_W
        out[f, :n_take, 2] = toks[:, 2] / PLAYFIELD_H
        out[f, :n_take, 3] = np.clip(toks[:, 3] / DUR_HORIZON_MS, 0.0, 1.0)
        out[f, :n_take, 4:7] = toks[:, 4:7]
        out[f, :n_take, 7] = 1.0  # present
    return out


def _build_state_vec(map_t_ms: np.ndarray,
                     cursor_pf: np.ndarray,
                     press: np.ndarray,
                     total_dur_ms: float) -> np.ndarray:
    """Build (M, 4) state vector: [map_time_norm, prev_cx_norm, prev_cy_norm,
    prev_press]. 'prev' means the frame at index i-1 (i.e. ~33ms earlier at
    30 Hz); for i=0 we use the current frame as a self-reference."""
    M = map_t_ms.shape[0]
    out = np.zeros((M, F_STATE), dtype=np.float32)
    denom = max(1.0, float(total_dur_ms))
    out[:, 0] = np.clip(map_t_ms.astype(np.float32) / denom, 0.0, 1.0)
    prev_idx = np.maximum(np.arange(M) - 1, 0)
    out[:, 1] = cursor_pf[prev_idx, 0] / PLAYFIELD_W
    out[:, 2] = cursor_pf[prev_idx, 1] / PLAYFIELD_H
    out[:, 3] = press[prev_idx]
    return out


def _segment_indices(map_t_ms: np.ndarray, gap_ms: int) -> list[tuple[int, int]]:
    """Split a frame index range on back-jumps or forward gaps > ``gap_ms``."""
    n = len(map_t_ms)
    if n == 0:
        return []
    diffs = np.diff(map_t_ms.astype(np.int64))
    breaks = np.where((diffs < 0) | (diffs > gap_ms))[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [n]])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


@dataclass
class _SessionTensors:
    """Per-session arrays after filtering. Indexable by sample index."""
    npz_path: Path
    frames: np.ndarray            # (M, FRAME_H, FRAME_W) uint8 (60x80)
    cursor_pf: np.ndarray         # (M, 2) float32 (playfield px)
    press: np.ndarray             # (M,) float32 (0 or 1)
    map_ctx: np.ndarray           # (M, K, F_obj) float32
    state_vec: np.ndarray         # (M, F_state) float32
    sample_centers: np.ndarray    # (S,) int — indices into the above
    stats: BuildStats = field(default_factory=BuildStats)


def _build_one_session(npz_path: Path,
                       label_window_ms: int,
                       gap_ms: int) -> _SessionTensors:
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))

    total = len(d["map_time_ms"])
    map_t = d["map_time_ms"]
    paused = d["game_paused"]

    keep = ~paused & (map_t >= 0)
    stats = BuildStats(
        total_frames=total,
        dropped_paused=int(paused.sum()),
        dropped_negative_t=int(((map_t < 0) & ~paused).sum()),
        map_id=int(meta.get("map_id", 0)),
        duration_s=float(meta.get("duration_s", 0.0)),
    )

    kept_idx = np.where(keep)[0]
    if len(kept_idx) == 0:
        return _SessionTensors(
            npz_path=npz_path,
            frames=np.empty((0, FRAME_H, FRAME_W), dtype=np.uint8),
            cursor_pf=np.empty((0, 2), dtype=np.float32),
            press=np.empty((0,), dtype=np.float32),
            map_ctx=np.empty((0, K_TOKENS, F_OBJ), dtype=np.float32),
            state_vec=np.empty((0, F_STATE), dtype=np.float32),
            sample_centers=np.empty((0,), dtype=np.int64),
            stats=stats,
        )

    frames_raw = d["frame"][kept_idx]
    frames = _resize_stack_to_80x60(frames_raw)
    cx = d["cursor_x"][kept_idx]
    cy = d["cursor_y"][kept_idx]
    map_t_kept = map_t[kept_idx]
    pf_x, pf_y = screen_to_playfield(cx, cy)
    cursor_pf = np.stack([pf_x, pf_y], axis=1).astype(np.float32)

    bm = parse_beatmap_std(Path(meta["map_file_abs"]))
    stats.map_title = bm.title
    press = _press_mask_from_beatmap(
        bm, map_t_kept, label_window_ms).astype(np.float32)

    obj_table = _build_object_table(bm)
    map_ctx = _build_map_ctx(obj_table, map_t_kept)
    total_dur_ms = (bm.notes[-1].t_end if bm.notes else 0)
    state_vec = _build_state_vec(map_t_kept, cursor_pf, press,
                                 total_dur_ms=float(total_dur_ms))

    # Segment the kept frames; emit one sample per frame that has 3 prior
    # frames inside the same segment.
    segs = _segment_indices(map_t_kept, gap_ms=gap_ms)
    stats.n_segments = len(segs)
    centers: list[int] = []
    for s, e in segs:
        if e - s < STACK_LEN:
            stats.dropped_seg_too_short += (e - s)
            continue
        centers.extend(range(s + STACK_LEN - 1, e))
        stats.dropped_stack_warmup += (STACK_LEN - 1)

    stats.kept_frames = int(len(kept_idx))
    stats.samples = len(centers)
    if centers:
        cc = np.array(centers, dtype=np.int64)
        stats.press_positive = int(press[cc].sum())
    else:
        cc = np.empty((0,), dtype=np.int64)

    return _SessionTensors(
        npz_path=npz_path,
        frames=frames,
        cursor_pf=cursor_pf,
        press=press,
        map_ctx=map_ctx,
        state_vec=state_vec,
        sample_centers=cc,
        stats=stats,
    )


class OsuStdDataset(Dataset):
    """Concatenation of per-session sample tables.

    __getitem__(i) returns:
      state  : dict {
                 "frames":    (4, FRAME_H, FRAME_W) float32 in [0,1],
                 "map_ctx":   (K, F_obj) float32,
                 "state_vec": (F_state,) float32,
               }
      action : dict {
                 "cursor_pf": (2,) float32,
                 "press":     scalar float32,
               }
    """

    def __init__(self, npz_paths: Iterable[Path],
                 label_window_ms: int = LABEL_WINDOW_MS,
                 gap_ms: int = GAP_MS):
        self.sessions: list[_SessionTensors] = []
        for p in npz_paths:
            self.sessions.append(_build_one_session(
                Path(p), label_window_ms=label_window_ms, gap_ms=gap_ms))
        lens = [s.sample_centers.shape[0] for s in self.sessions]
        self._cumlen = np.cumsum([0] + lens)

    def __len__(self) -> int:
        return int(self._cumlen[-1])

    def __getitem__(self, idx: int):
        i = int(np.searchsorted(self._cumlen[1:], idx, side="right"))
        local = idx - int(self._cumlen[i])
        sess = self.sessions[i]
        center = int(sess.sample_centers[local])
        stack = sess.frames[center - STACK_LEN + 1: center + 1]
        state = {
            "frames": torch.from_numpy(stack.astype(np.float32) / 255.0),
            "map_ctx": torch.from_numpy(sess.map_ctx[center]),
            "state_vec": torch.from_numpy(sess.state_vec[center]),
        }
        action = {
            "cursor_pf": torch.from_numpy(sess.cursor_pf[center]),
            "press": torch.tensor(sess.press[center], dtype=torch.float32),
        }
        return state, action

    def stats(self) -> list[dict]:
        return [s.stats.as_dict() for s in self.sessions]

    def aggregate_stats(self) -> dict:
        agg = BuildStats()
        for s in self.sessions:
            agg.total_frames += s.stats.total_frames
            agg.dropped_paused += s.stats.dropped_paused
            agg.dropped_negative_t += s.stats.dropped_negative_t
            agg.dropped_seg_too_short += s.stats.dropped_seg_too_short
            agg.dropped_stack_warmup += s.stats.dropped_stack_warmup
            agg.kept_frames += s.stats.kept_frames
            agg.samples += s.stats.samples
            agg.press_positive += s.stats.press_positive
            agg.n_segments += s.stats.n_segments
        d = agg.as_dict()
        d["n_sessions"] = len(self.sessions)
        return d
