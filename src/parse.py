"""Parse osu!mania beatmaps (.osu) and replays (.osr) into aligned timelines."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from osrparse import Replay, GameMode


@dataclass
class Note:
    column: int      # 0..K-1
    t_start: int     # ms since map start
    t_end: int       # ms; == t_start for taps
    is_hold: bool


@dataclass
class Beatmap:
    keys: int        # number of mania columns (CircleSize)
    notes: List[Note]
    title: str


def parse_beatmap(path: Union[Path, str]) -> Beatmap:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    section = None
    keys = 4
    title = path.stem
    mode = None
    notes: List[Note] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue

        if section == "General":
            if line.startswith("Mode:"):
                mode = int(line.split(":", 1)[1].strip())
        elif section == "Metadata":
            if line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
        elif section == "Difficulty":
            if line.startswith("CircleSize:"):
                keys = int(float(line.split(":", 1)[1].strip()))
        elif section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 5:
                continue
            x = int(parts[0])
            t = int(parts[2])
            typ = int(parts[3])
            col = min(int(x * keys / 512), keys - 1)
            is_hold = bool(typ & 128)
            if is_hold and len(parts) > 5:
                end = int(parts[5].split(":")[0])
            else:
                end = t
            notes.append(Note(column=col, t_start=t, t_end=end, is_hold=is_hold))

    if mode != 3:
        raise ValueError(f"Not a mania beatmap (Mode={mode})")

    notes.sort(key=lambda n: n.t_start)
    return Beatmap(keys=keys, notes=notes, title=title)


def parse_replay(path: Union[Path, str]) -> List[Tuple[int, int]]:
    """Return [(time_ms, keystate_bitmask), ...] from a mania .osr."""
    rep = Replay.from_path(str(path))
    if rep.mode != GameMode.MANIA:
        raise ValueError(f"Not a mania replay (mode={rep.mode})")

    events: List[Tuple[int, int]] = []
    t = 0
    for ev in rep.replay_data:
        # osrparse may include an RNG-seed sentinel with time_delta = -12345 — skip.
        if ev.time_delta == -12345:
            continue
        t += int(ev.time_delta)
        # osrparse v7: ReplayEventMania.keys is an IntFlag bitmask of held columns
        events.append((t, int(ev.keys)))
    return events


def diff_keystates(events: List[Tuple[int, int]]):
    """Convert (time, keystate) stream → (time, column, +1=press / -1=release)."""
    transitions = []
    prev = 0
    for t, state in events:
        changed = state ^ prev
        if changed:
            col = 0
            c = changed
            while c:
                if c & 1:
                    transitions.append((t, col, 1 if state & (1 << col) else -1))
                c >>= 1
                col += 1
        prev = state
    return transitions


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python src/parse.py <map.osu> <replay.osr>")
        sys.exit(1)

    bm = parse_beatmap(sys.argv[1])
    print(f"Beatmap: {bm.title}")
    print(f"  keys:  {bm.keys}")
    print(f"  notes: {len(bm.notes)}")
    print(f"  first 5: {bm.notes[:5]}")

    events = parse_replay(sys.argv[2])
    transitions = diff_keystates(events)
    presses = [tr for tr in transitions if tr[2] == 1]
    print(f"Replay:")
    print(f"  events:  {len(events)}")
    print(f"  presses: {len(presses)}")
    print(f"  ratio notes/presses: {len(bm.notes)}/{len(presses)} "
          f"= {len(bm.notes)/max(1, len(presses)):.3f}")
