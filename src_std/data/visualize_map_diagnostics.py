"""Phase 1 diagnostic renders: time-windowed slice + per-curve-type slider zoo.

Goal: make slider curve shapes legible for visual comparison against lazer's editor.
Full-map overlays are too dense; this pulls out individual sliders by curve type.
"""
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src_std.parse_std import (
    parse_beatmap_std, CIRCLE, SLIDER, SPINNER,
    circle_radius_px, PLAYFIELD_W, PLAYFIELD_H,
)


def render_time_window(bm, t0, t1, out_path):
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=100)
    ax.set_xlim(0, PLAYFIELD_W)
    ax.set_ylim(PLAYFIELD_H, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    r = circle_radius_px(bm.cs)
    ax.add_patch(mpatches.Rectangle((0, 0), PLAYFIELD_W, PLAYFIELD_H,
                                    fill=False, edgecolor="#444", linewidth=1))
    in_window = [n for n in bm.notes if t0 <= n.t_start <= t1]
    for i, n in enumerate(in_window):
        label = f"{i}"
        if n.type == CIRCLE:
            ax.add_patch(mpatches.Circle((n.x, n.y), r, fill=False,
                                         edgecolor="#5aa9ff", linewidth=1.2))
            ax.text(n.x, n.y, label, color="white", fontsize=6, ha="center", va="center")
        elif n.type == SLIDER:
            curve = n.slider_curve
            if curve:
                xs = [p[0] for p in curve]; ys = [p[1] for p in curve]
                ax.plot(xs, ys, "-", color="#ff9a3c", linewidth=2.0)
                ax.plot(xs[0], ys[0], "o", color="#3cff6e", markersize=5)
                ax.plot(xs[-1], ys[-1], "o", color="#ff3c3c", markersize=5)
                ax.text(xs[0], ys[0] - 8, label, color="white", fontsize=6, ha="center")
        elif n.type == SPINNER:
            ax.add_patch(mpatches.Circle((PLAYFIELD_W / 2, PLAYFIELD_H / 2), 100,
                                         fill=False, edgecolor="#ff4ade", linewidth=2))

    ax.set_title(f"{bm.title}: t={t0}-{t1}ms  ({len(in_window)} objects)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor="#1a1a1a")
    plt.close(fig)


def render_slider_zoo(bm, raw_lines, out_path):
    by_type = defaultdict(list)
    in_hits = False
    note_idx = 0
    for line in raw_lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_hits = (s == "[HitObjects]")
            continue
        if not in_hits or "," not in s:
            continue
        parts = s.split(",")
        if len(parts) < 5:
            continue
        try:
            tb = int(parts[3])
        except ValueError:
            continue
        if tb & 1 or tb & 8:
            note_idx += 1
            continue
        if tb & 2:
            if len(parts) < 8:
                note_idx += 1
                continue
            curve_spec = parts[5]
            ctype = curve_spec.split("|", 1)[0]
            anchor_count = len(curve_spec.split("|")) - 1
            if note_idx < len(bm.notes) and bm.notes[note_idx].type == SLIDER:
                by_type[(ctype, anchor_count)].append(bm.notes[note_idx])
            note_idx += 1

    picks = []
    for (ctype, ac), notes_list in sorted(by_type.items()):
        if len(picks) >= 16:
            break
        picks.append((ctype, ac, notes_list[0]))
    for (ctype, ac), notes_list in by_type.items():
        if ctype == "B" and ac >= 4 and len(picks) < 16:
            picks.append((ctype, ac, notes_list[0]))
            break

    n = len(picks)
    if n == 0:
        return
    cols = 4
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=100)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    r = circle_radius_px(bm.cs)
    for ax, (ctype, ac, note) in zip(axes, picks):
        ax.set_facecolor("#1a1a1a")
        xs = [p[0] for p in note.slider_curve]
        ys = [p[1] for p in note.slider_curve]
        pad = 30
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymax + pad, ymin - pad)
        ax.set_aspect("equal")
        ax.plot(xs, ys, "-o", color="#ff9a3c", markersize=2, linewidth=1.5)
        ax.plot(xs[0], ys[0], "o", color="#3cff6e", markersize=8)
        ax.plot(xs[-1], ys[-1], "o", color="#ff3c3c", markersize=8)
        ax.add_patch(mpatches.Circle((note.x, note.y), r, fill=False,
                                     edgecolor="#ff9a3c", linewidth=0.5, alpha=0.5))
        ax.set_title(f"type={ctype}  anchors={ac}  len={note.slider_length:.0f}px",
                     fontsize=8, color="white")
        ax.tick_params(colors="white", labelsize=6)
    for ax in axes[len(picks):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor="#1a1a1a")
    plt.close(fig)


def stack_report(bm):
    n = len(bm.notes)
    stacks = 0
    for i in range(n):
        for j in range(i + 1, min(i + 8, n)):
            if abs(bm.notes[i].x - bm.notes[j].x) < 3 and abs(bm.notes[i].y - bm.notes[j].y) < 3:
                stacks += 1
                break
    return stacks


def main():
    if len(sys.argv) < 3:
        print("usage: python -m src_std.data.visualize_map_diagnostics <map.osu> <out_dir>")
        sys.exit(1)
    map_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    bm = parse_beatmap_std(map_path)
    raw_lines = Path(map_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    if bm.notes:
        t0 = bm.notes[0].t_start
        render_time_window(bm, t0, t0 + 10_000, out_dir / "window_first10s.png")
        mid = bm.notes[len(bm.notes) // 2].t_start
        render_time_window(bm, mid, mid + 5_000, out_dir / "window_mid5s.png")

    render_slider_zoo(bm, raw_lines, out_dir / "slider_zoo.png")

    n_stacks = stack_report(bm)
    print(f"{bm.title}: {len(bm.notes)} objects, ~{n_stacks} candidate raw-coord stacks")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
