"""Curriculum gate: stage progression by moving-average FC rate.

Stages (per the plan):
  1: easy   — advance at 90% FC over last 100 rollouts
  2: medium — advance at 70% FC over last 100
  3: hard   — stop at 50% FC

True osu! star-rating is ~500 LOC of math; we use a density+AR+CS proxy
instead. Bands can be retuned once we have ground-truth SR per map.
"""

import random
from collections import deque
from pathlib import Path

from .parse_std import parse_beatmap_std


def difficulty_proxy(bm) -> float:
    """Coarse difficulty score in roughly star-rating units (1..7)."""
    if len(bm.notes) < 2:
        return 0.0
    span_s = max((bm.notes[-1].t_end - bm.notes[0].t_start) / 1000.0, 1.0)
    density = len(bm.notes) / span_s
    return 0.30 * density + 0.20 * bm.ar + 0.20 * bm.cs


STAGE_BANDS = [
    (1, 0.0, 2.5),
    (2, 2.5, 4.5),
    (3, 4.5, 8.0),
]


def stage_for(score: float) -> int:
    for s, lo, hi in STAGE_BANDS:
        if lo <= score < hi:
            return s
    return STAGE_BANDS[-1][0]


class MapPool:

    def __init__(self, root: str):
        self.root = Path(root)
        self.by_stage = {s: [] for s, _, _ in STAGE_BANDS}
        self.scores = {}
        for p in self.root.rglob("*.osu"):
            try:
                bm = parse_beatmap_std(p)
            except Exception:
                continue
            if bm.mode != 0 or len(bm.notes) < 2:
                continue
            sc = difficulty_proxy(bm)
            self.by_stage[stage_for(sc)].append(str(p))
            self.scores[str(p)] = sc

    def __len__(self):
        return sum(len(v) for v in self.by_stage.values())

    def sample(self, stage: int) -> str:
        pool = self.by_stage.get(stage) or []
        if not pool:
            for s in range(stage, 0, -1):
                if self.by_stage.get(s):
                    pool = self.by_stage[s]
                    break
        if not pool:
            raise RuntimeError("map pool is empty for all stages")
        return random.choice(pool)

    def summary(self):
        return {s: len(v) for s, v in self.by_stage.items()}


class Curriculum:

    GATES = {1: 0.90, 2: 0.70, 3: 0.50}
    WINDOW = 100

    def __init__(self, start_stage: int = 1):
        self.stage = start_stage
        self.recent_fc = {s: deque(maxlen=self.WINDOW) for s in self.GATES}

    def record(self, fc_flags) -> None:
        for f in fc_flags:
            self.recent_fc[self.stage].append(bool(f))

    def fc_rate(self) -> float:
        d = self.recent_fc[self.stage]
        return sum(d) / max(len(d), 1)

    def maybe_advance(self) -> bool:
        d = self.recent_fc[self.stage]
        if len(d) < self.WINDOW:
            return False
        if self.fc_rate() >= self.GATES[self.stage] and self.stage < max(self.GATES):
            self.stage += 1
            return True
        return False

    def should_stop(self) -> bool:
        if self.stage != max(self.GATES):
            return False
        d = self.recent_fc[self.stage]
        return len(d) >= self.WINDOW and self.fc_rate() >= self.GATES[self.stage]


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/beatmaps_std"
    pool = MapPool(root)
    print(f"{len(pool)} maps in {root}")
    print(f"by stage: {pool.summary()}")
