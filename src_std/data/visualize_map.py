"""Render a parsed .osu beatmap on the 512x384 playfield as a PNG.

Phase 1 sanity check: compare against lazer's renderer.

Circles: blue dots sized by CS radius.
Sliders: orange curve from the parsed 32-sample path, head (green) / tail (red) marked.
Spinners: magenta ring at playfield center.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src_std.parse_std import (
    parse_beatmap_std, CIRCLE, SLIDER, SPINNER,
    circle_radius_px, PLAYFIELD_W, PLAYFIELD_H,
)


def render(bm, out_path):
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=100)
    ax.set_xlim(0, PLAYFIELD_W)
    ax.set_ylim(PLAYFIELD_H, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    ax.set_title(f"{bm.title}  CS={bm.cs} AR={bm.ar} OD={bm.od}  "
                 f"circles={sum(1 for n in bm.notes if n.type==CIRCLE)} "
                 f"sliders={sum(1 for n in bm.notes if n.type==SLIDER)} "
                 f"spinners={sum(1 for n in bm.notes if n.type==SPINNER)}",
                 fontsize=9)

    r = circle_radius_px(bm.cs)
    ax.add_patch(mpatches.Rectangle((0, 0), PLAYFIELD_W, PLAYFIELD_H,
                                    fill=False, edgecolor="#444", linewidth=1))

    for n in bm.notes:
        if n.type == CIRCLE:
            ax.add_patch(mpatches.Circle((n.x, n.y), r, fill=False,
                                         edgecolor="#5aa9ff", linewidth=0.6, alpha=0.7))
            ax.plot(n.x, n.y, ".", color="#5aa9ff", markersize=2)
        elif n.type == SLIDER:
            curve = n.slider_curve
            if curve:
                xs = [p[0] for p in curve]
                ys = [p[1] for p in curve]
                ax.plot(xs, ys, "-", color="#ff9a3c", linewidth=1.2, alpha=0.8)
                ax.plot(xs[0], ys[0], "o", color="#3cff6e", markersize=3)
                ax.plot(xs[-1], ys[-1], "o", color="#ff3c3c", markersize=3)
                ax.add_patch(mpatches.Circle((n.x, n.y), r, fill=False,
                                             edgecolor="#ff9a3c", linewidth=0.5, alpha=0.5))
        elif n.type == SPINNER:
            ax.add_patch(mpatches.Circle((PLAYFIELD_W / 2, PLAYFIELD_H / 2), 100,
                                         fill=False, edgecolor="#ff4ade", linewidth=2))

    legend_elems = [
        mpatches.Patch(color="#5aa9ff", label="circle"),
        mpatches.Patch(color="#ff9a3c", label="slider path"),
        mpatches.Patch(color="#3cff6e", label="slider head"),
        mpatches.Patch(color="#ff3c3c", label="slider tail"),
        mpatches.Patch(color="#ff4ade", label="spinner"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7,
              facecolor="#222", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor="#1a1a1a")
    plt.close(fig)


def main():
    if len(sys.argv) < 3:
        print("usage: python -m src_std.data.visualize_map <map.osu> <out.png>")
        sys.exit(1)
    bm = parse_beatmap_std(sys.argv[1])
    if bm.mode != 0:
        print(f"WARNING: mode={bm.mode}, not std")
    render(bm, sys.argv[2])
    print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
