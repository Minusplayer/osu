"""Visual sanity check: overlay parsed notes + replay presses on the timeline."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from parse import parse_beatmap, parse_replay, diff_keystates


def main(map_path: str, replay_path: str, out: str = "data/alignment.png",
         window_ms: int = 30_000):
    bm = parse_beatmap(map_path)
    events = parse_replay(replay_path)
    transitions = diff_keystates(events)
    presses = [(t, c) for t, c, d in transitions if d == 1 and c < bm.keys]

    note_x = [n.t_start for n in bm.notes if n.t_start <= window_ms]
    note_y = [n.column for n in bm.notes if n.t_start <= window_ms]
    press_x = [t for t, c in presses if t <= window_ms]
    press_y = [c for t, c in presses if t <= window_ms]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.scatter(note_x, note_y, s=80, marker="o", facecolors="none",
               edgecolors="tab:blue", label="notes (chart)")
    ax.scatter(press_x, press_y, s=30, marker="v",
               color="tab:red", alpha=0.7, label="presses (replay)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("column")
    ax.set_yticks(range(bm.keys))
    ax.set_title(f"{bm.title} — first {window_ms/1000:.0f}s "
                 f"(notes={len(note_x)} presses={len(press_x)})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python src/plot_alignment.py <map.osu> <replay.osr> [out.png] [window_ms]")
        sys.exit(1)
    args = sys.argv[1:]
    main(*args[:2],
         out=args[2] if len(args) > 2 else "data/alignment.png",
         window_ms=int(args[3]) if len(args) > 3 else 30_000)
