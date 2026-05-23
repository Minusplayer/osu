"""Build (state, action) training pairs from a beatmap + replay."""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from parse import Beatmap, Note, parse_beatmap, parse_replay


# Feature indices in the state tensor's last dim.
F_TAP, F_HOLD_HEAD, F_HOLD_BODY, F_HOLD_TAIL, F_TIME_TO_PRESS = 0, 1, 2, 3, 4
NUM_FEATURES = 5

# Saturation horizon for the continuous "ticks until next press" feature.
# Anything beyond this is clipped to 1.0 (i.e. "press is far away or absent").
TIME_TO_PRESS_HORIZON_TICKS = 200


def build_note_grid(notes: List[Note], keys: int, dt_ms: int,
                    total_ms: int) -> np.ndarray:
    """Rasterize notes into a (T, K, F) grid at dt_ms resolution.

    Features (last dim):
      [0] F_TAP            — 1.0 at the tick a tap note hits
      [1] F_HOLD_HEAD      — 1.0 at the tick a hold starts
      [2] F_HOLD_BODY      — 1.0 between head and tail of a hold
      [3] F_HOLD_TAIL      — 1.0 at the tick a hold ends
      [4] F_TIME_TO_PRESS  — normalized ticks until next press in this column
                              (0 = press now, 1 = far/none)
    """
    n_ticks = total_ms // dt_ms + 1
    grid = np.zeros((n_ticks, keys, NUM_FEATURES), dtype=np.float32)
    for note in notes:
        i0 = note.t_start // dt_ms
        if i0 >= n_ticks:
            continue
        if note.is_hold:
            grid[i0, note.column, F_HOLD_HEAD] = 1.0
            i1 = min(note.t_end // dt_ms, n_ticks - 1)
            if i1 > i0:
                grid[i0 + 1:i1, note.column, F_HOLD_BODY] = 1.0
                grid[i1, note.column, F_HOLD_TAIL] = 1.0
        else:
            grid[i0, note.column, F_TAP] = 1.0

    H = TIME_TO_PRESS_HORIZON_TICKS
    for k in range(keys):
        press_ticks = sorted({
            n.t_start // dt_ms for n in notes
            if n.column == k and (n.t_start // dt_ms) < n_ticks
        })
        if not press_ticks:
            grid[:, k, F_TIME_TO_PRESS] = 1.0
            continue
        pi = 0
        for t in range(n_ticks):
            while pi < len(press_ticks) and press_ticks[pi] < t:
                pi += 1
            d = (press_ticks[pi] - t) if pi < len(press_ticks) else H
            grid[t, k, F_TIME_TO_PRESS] = min(d / H, 1.0)
    return grid


def build_action_grid(events: List[Tuple[int, int]], keys: int,
                      dt_ms: int, total_ms: int) -> np.ndarray:
    """Forward-fill keystate over time at dt_ms resolution."""
    n_ticks = total_ms // dt_ms + 1
    actions = np.zeros((n_ticks, keys), dtype=np.float32)
    current = 0
    ev_idx = 0
    n_events = len(events)
    for tick in range(n_ticks):
        t = tick * dt_ms
        while ev_idx < n_events and events[ev_idx][0] <= t:
            current = events[ev_idx][1]
            ev_idx += 1
        if current:
            for k in range(keys):
                if current & (1 << k):
                    actions[tick, k] = 1.0
    return actions


class ManiaDataset(Dataset):
    """One sample = (lookahead window of notes, target keystate at tick t).

    state[t]  : (T_lookahead, K, F)  — what's coming in the next lookahead_ms
    action[t] : (K,)                 — keys the human held at tick t
    """

    def __init__(self, map_path, replay_path, dt_ms: int = 10,
                 lookahead_ms: int = 800):
        bm = parse_beatmap(map_path)
        events = parse_replay(replay_path)

        if not bm.notes:
            raise ValueError("beatmap has no notes")

        self.keys = bm.keys
        self.dt_ms = dt_ms
        self.lookahead_ticks = lookahead_ms // dt_ms

        last_note = max(n.t_end for n in bm.notes) + lookahead_ms
        last_event = events[-1][0] if events else 0
        total_ms = max(last_note, last_event)

        self.note_grid = build_note_grid(bm.notes, bm.keys, dt_ms, total_ms)
        self.action_grid = build_action_grid(events, bm.keys, dt_ms, total_ms)

        first = min(n.t_start for n in bm.notes) // dt_ms
        last = max(n.t_end for n in bm.notes) // dt_ms
        self.start_tick = max(0, first - self.lookahead_ticks)
        self.end_tick = min(last + 1,
                            len(self.note_grid) - self.lookahead_ticks)

    def __len__(self) -> int:
        return self.end_tick - self.start_tick

    def __getitem__(self, idx: int):
        t = self.start_tick + idx
        state = self.note_grid[t:t + self.lookahead_ticks]   # (T, K, F)
        action = self.action_grid[t]                         # (K,)
        return torch.from_numpy(state), torch.from_numpy(action)


class MultiMapDataset(Dataset):
    """Concatenation of ManiaDataset across (map, replay) pairs."""

    def __init__(self, pairs, dt_ms: int = 10, lookahead_ms: int = 800,
                 verbose: bool = True):
        if not pairs:
            raise ValueError("no (map, replay) pairs provided")
        self.subsets = []
        self.keys = None
        for map_path, replay_path in pairs:
            try:
                ds = ManiaDataset(map_path, replay_path, dt_ms, lookahead_ms)
            except Exception as ex:
                if verbose:
                    print(f"  skip {Path(map_path).name}: {ex}")
                continue
            if self.keys is None:
                self.keys = ds.keys
            elif ds.keys != self.keys:
                if verbose:
                    print(f"  skip {Path(map_path).name}: "
                          f"keys={ds.keys} != expected {self.keys}")
                continue
            self.subsets.append(ds)
            if verbose:
                print(f"  + {Path(map_path).name}: {len(ds)} samples")
        if not self.subsets:
            raise ValueError("no usable subsets after filtering")
        self.lookahead_ticks = self.subsets[0].lookahead_ticks
        self.dt_ms = dt_ms
        lens = [len(s) for s in self.subsets]
        self.cumlen = np.cumsum([0] + lens)

    def __len__(self) -> int:
        return int(self.cumlen[-1])

    def __getitem__(self, idx: int):
        i = int(np.searchsorted(self.cumlen[1:], idx, side="right"))
        return self.subsets[i][idx - int(self.cumlen[i])]


class GPUMultiMapDataset:
    """Like MultiMapDataset but keeps all grids on GPU. Batches are
    gathered with vectorized indexing — no DataLoader, no workers, no
    per-sample Python. Use `.iter_batches(batch_size, shuffle)`.
    """

    def __init__(self, pairs, dt_ms: int = 10, lookahead_ms: int = 800,
                 target_ticks: int = 1, sample_stride: int = 1,
                 device=None, verbose: bool = True):
        if not pairs:
            raise ValueError("no (map, replay) pairs provided")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dt_ms = dt_ms
        self.lookahead_ticks = lookahead_ms // dt_ms
        self.target_ticks = target_ticks
        self.keys = None

        grids, actions, sample_starts = [], [], []
        offset = 0
        for map_path, replay_path in pairs:
            try:
                ds = ManiaDataset(map_path, replay_path, dt_ms, lookahead_ms)
            except Exception as ex:
                if verbose:
                    print(f"  skip {Path(map_path).name}: {ex}")
                continue
            if self.keys is None:
                self.keys = ds.keys
            elif ds.keys != self.keys:
                if verbose:
                    print(f"  skip {Path(map_path).name}: "
                          f"keys={ds.keys} != expected {self.keys}")
                continue
            g = torch.from_numpy(ds.note_grid).to(device)
            a = torch.from_numpy(ds.action_grid).to(device)
            # ensure t + target_ticks <= len(action_grid)
            max_start = a.shape[0] - target_ticks
            end = min(ds.end_tick, max_start)
            if end <= ds.start_tick:
                if verbose:
                    print(f"  skip {Path(map_path).name}: too short for target_ticks={target_ticks}")
                continue
            grids.append(g)
            actions.append(a)
            starts = torch.arange(ds.start_tick, end, sample_stride,
                                  dtype=torch.long, device=device) + offset
            sample_starts.append(starts)
            offset += g.shape[0]
            if verbose:
                print(f"  + {Path(map_path).name}: {end-ds.start_tick} samples")

        if not grids:
            raise ValueError("no usable subsets after filtering")
        self.flat_grid = torch.cat(grids, dim=0)
        self.flat_action = torch.cat(actions, dim=0)
        self.sample_starts = torch.cat(sample_starts)
        self._arange = torch.arange(self.lookahead_ticks, dtype=torch.long,
                                    device=device)
        self._target_arange = torch.arange(target_ticks, dtype=torch.long,
                                           device=device)

    def __len__(self) -> int:
        return int(self.sample_starts.shape[0])

    def iter_batches(self, batch_size: int, shuffle: bool = True,
                     subsample: float = 1.0, symmetric: bool = False):
        """If subsample<1, draw a fresh random subset each call (each epoch).
        If symmetric, with p=0.5 per batch reverse columns (4K symmetry 0↔3, 1↔2).
        """
        n = len(self)
        if shuffle:
            perm = torch.randperm(n, device=self.device)
        else:
            perm = torch.arange(n, device=self.device)
        if 0.0 < subsample < 1.0:
            keep = max(1, int(n * subsample))
            perm = perm[:keep]
            n = keep
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            t_starts = self.sample_starts[idx]
            window_idx = t_starts.unsqueeze(1) + self._arange.unsqueeze(0)
            state = self.flat_grid[window_idx]                 # (B, L, K, F)
            if self.target_ticks == 1:
                action = self.flat_action[t_starts]            # (B, K)
            else:
                tgt_idx = t_starts.unsqueeze(1) + self._target_arange.unsqueeze(0)
                action = self.flat_action[tgt_idx]             # (B, n_tgt, K)
            if symmetric and torch.rand(1, device=self.device).item() < 0.5:
                state = state.flip(dims=(-2,))     # flip K
                action = action.flip(dims=(-1,))   # flip K (last dim)
            yield state, action

    def num_batches(self, batch_size: int) -> int:
        return (len(self) + batch_size - 1) // batch_size


def pair_replays_with_maps(replay_dir, map_root):
    """Match .osr files to .osu files by beatmap MD5 hash."""
    import hashlib
    from osrparse import Replay

    osu_by_hash = {}
    for osu in Path(map_root).rglob("*.osu"):
        h = hashlib.md5(osu.read_bytes()).hexdigest()
        osu_by_hash[h] = osu

    pairs, skipped = [], 0
    for osr in Path(replay_dir).rglob("*.osr"):
        try:
            r = Replay.from_path(str(osr))
        except Exception:
            skipped += 1
            continue
        m = osu_by_hash.get(r.beatmap_hash)
        if m is None:
            skipped += 1
            continue
        pairs.append((m, osr))
    return pairs, skipped


def split_pairs_by_map(pairs, val_frac: float = 0.1, seed: int = 42):
    """Hold out entire maps for validation — never see same map in train+val."""
    import random
    rng = random.Random(seed)
    by_map = {}
    for m, r in pairs:
        by_map.setdefault(str(m), []).append((m, r))
    maps = list(by_map.keys())
    rng.shuffle(maps)
    n_val = max(1, int(len(maps) * val_frac))
    val_maps = set(maps[:n_val])
    train, val = [], []
    for m, r in pairs:
        (val if str(m) in val_maps else train).append((m, r))
    return train, val


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python src/dataset.py <map.osu> <replay.osr>")
        sys.exit(1)

    ds = ManiaDataset(sys.argv[1], sys.argv[2])
    print(f"dataset: {len(ds)} samples")
    print(f"  keys:      {ds.keys}")
    print(f"  dt:        {ds.dt_ms} ms")
    print(f"  lookahead: {ds.lookahead_ticks} ticks "
          f"({ds.lookahead_ticks * ds.dt_ms} ms)")

    state, action = ds[0]
    print(f"\nfirst sample:")
    print(f"  state shape:  {tuple(state.shape)}  dtype={state.dtype}")
    print(f"  action shape: {tuple(action.shape)} dtype={action.dtype}")

    sub = min(len(ds), 5000)
    acts = torch.stack([ds[i][1] for i in range(sub)])
    press_frac = acts.mean(dim=0).tolist()
    any_pressed = (acts.sum(dim=1) > 0).float().mean().item()
    print(f"\nclass balance over first {sub} ticks:")
    for k, p in enumerate(press_frac):
        print(f"  col {k}: {p * 100:5.2f}% held")
    print(f"  any key held: {any_pressed * 100:.2f}% of ticks")

    for i in range(len(ds)):
        s, a = ds[i]
        if s.sum() > 0 and a.sum() > 0:
            cols_with_notes = (s.sum(dim=(0, 2)) > 0).nonzero().squeeze(-1).tolist()
            print(f"\nfirst non-trivial sample at idx {i} "
                  f"(tick {ds.start_tick + i}, t≈{(ds.start_tick + i) * ds.dt_ms} ms):")
            print(f"  cols with upcoming notes in window: {cols_with_notes}")
            print(f"  current action: {a.tolist()}")
            break
