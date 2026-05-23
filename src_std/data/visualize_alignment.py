"""Phase 3: alignment visualization for a captured session.

Validates that the four captured streams line up in time and space against
the parsed beatmap:

1. Spatial — cursor trajectory drawn on the 512x384 playfield, with hit
   objects overlaid. Output as 3 time-windowed panels so trajectories
   don't smear into a single blob.

2. Per-circle error — at each circle's hit time we look up where the
   cursor was (in playfield coords) and plot Δx, Δy. If this is centered
   on zero and well within the CS radius, our screen->playfield transform
   is correct.

3. Cursor over time — cursor_x and cursor_y traces with vertical lines at
   each circle's t_start, so eyeballable that the cursor moves toward
   each object.

4. Clock alignment — map_time_ms vs wall-clock t_ns. Should be a straight
   line of slope 1 ms / 1e6 ns.

5. Frame samples — six captured frames (downsampled 96x96) at known hit
   times, with the parsed object's expected screen position annotated.

Usage:
    python -m src_std.data.visualize_alignment <session.npz>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src_std.parse_std import (  # noqa: E402
    CIRCLE, SLIDER, SPINNER, parse_beatmap_std, circle_radius_px,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# The captured region 316,60 1280x960 is the rendered playfield rectangle on
# screen. osu! playfield logical coords are 512x384 in a 4:3 box; both regions
# are 4:3 so a linear map suffices.
REGION_X, REGION_Y, REGION_W, REGION_H = 316, 60, 1280, 960


def screen_to_playfield(sx, sy):
    px = (np.asarray(sx, dtype=np.float64) - REGION_X) / REGION_W * 512.0
    py = (np.asarray(sy, dtype=np.float64) - REGION_Y) / REGION_H * 384.0
    return px, py


def playfield_to_screen_in_region(px, py):
    sx = np.asarray(px, dtype=np.float64) / 512.0 * REGION_W
    sy = np.asarray(py, dtype=np.float64) / 384.0 * REGION_H
    return sx, sy


def load_session(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    bm = parse_beatmap_std(Path(meta["map_file_abs"]))
    return d, meta, bm


def plot_spatial_windows(d, bm, out_path: Path, n_windows: int = 3):
    mask = ~d["game_paused"]
    map_t = d["map_time_ms"][mask]
    cx, cy = d["cursor_x"][mask], d["cursor_y"][mask]
    valid = map_t >= 0
    map_t = map_t[valid]; cx = cx[valid]; cy = cy[valid]
    if len(map_t) == 0:
        return

    pf_x, pf_y = screen_to_playfield(cx, cy)
    t_lo, t_hi = map_t.min(), map_t.max()
    edges = np.linspace(t_lo, t_hi, n_windows + 1)

    fig, axes = plt.subplots(1, n_windows, figsize=(6 * n_windows, 5.5))
    if n_windows == 1:
        axes = [axes]
    r = circle_radius_px(bm.cs)

    for i, ax in enumerate(axes):
        lo, hi = edges[i], edges[i + 1]
        in_window = (map_t >= lo) & (map_t < hi)
        for n in bm.notes:
            if not (lo <= n.t_start <= hi):
                continue
            if n.type == CIRCLE:
                ax.add_patch(Circle((n.x, n.y), r, fc="none", ec="#1f77b4", lw=0.8, alpha=0.7))
            elif n.type == SLIDER:
                if n.slider_curve:
                    pts = np.array(n.slider_curve)
                    ax.plot(pts[:, 0], pts[:, 1], color="#ff7f0e", lw=1.2, alpha=0.6)
                ax.add_patch(Circle((n.x, n.y), r, fc="none", ec="#2ca02c", lw=0.7, alpha=0.6))
            elif n.type == SPINNER:
                ax.add_patch(Circle((n.x, n.y), 30, fc="none", ec="#9467bd", lw=0.8, alpha=0.5))

        if in_window.any():
            t_local = map_t[in_window]
            sc = ax.scatter(pf_x[in_window], pf_y[in_window],
                            c=t_local, s=2, cmap="viridis", alpha=0.6)
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="map_t (ms)")

        ax.set_xlim(-20, 532)
        ax.set_ylim(404, -20)
        ax.set_aspect("equal")
        ax.set_title(f"Window {i+1}/{n_windows}:  {lo/1000:.1f}s – {hi/1000:.1f}s")
        ax.set_xlabel("playfield x")
        ax.set_ylabel("playfield y")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Cursor trajectory vs. hit objects  ({bm.title!r})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_circle_error(d, bm, out_path: Path):
    mask = ~d["game_paused"]
    map_t = d["map_time_ms"][mask].astype(np.int64)
    cx = d["cursor_x"][mask].astype(np.float32)
    cy = d["cursor_y"][mask].astype(np.float32)
    valid = map_t >= 0
    map_t = map_t[valid]; cx = cx[valid]; cy = cy[valid]
    if len(map_t) == 0:
        return
    sort_idx = np.argsort(map_t)
    map_t = map_t[sort_idx]; cx = cx[sort_idx]; cy = cy[sort_idx]

    circles = [n for n in bm.notes if n.type == CIRCLE
               and map_t.min() <= n.t_start <= map_t.max()]
    if not circles:
        return

    pf_x, pf_y = screen_to_playfield(cx, cy)
    dx, dy, ts, dists = [], [], [], []
    for c in circles:
        idx = int(np.searchsorted(map_t, c.t_start))
        idx = min(max(idx, 0), len(map_t) - 1)
        dx.append(pf_x[idx] - c.x)
        dy.append(pf_y[idx] - c.y)
        ts.append(c.t_start)
        dists.append((pf_x[idx] - c.x) ** 2 + (pf_y[idx] - c.y) ** 2)
    dx = np.array(dx); dy = np.array(dy); ts = np.array(ts)
    err_mag = np.sqrt(np.array(dists))
    r = circle_radius_px(bm.cs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].axhline(0, color="black", lw=0.5)
    axes[0, 0].axhspan(-r, r, color="#1f77b4", alpha=0.1, label=f"±CS radius ({r:.1f}px)")
    axes[0, 0].scatter(ts / 1000, dx, s=8, alpha=0.5)
    axes[0, 0].set_xlabel("hit time (s)")
    axes[0, 0].set_ylabel("Δx = cursor − circle (playfield px)")
    axes[0, 0].set_title("Δx over time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axhline(0, color="black", lw=0.5)
    axes[0, 1].axhspan(-r, r, color="#1f77b4", alpha=0.1)
    axes[0, 1].scatter(ts / 1000, dy, s=8, alpha=0.5, color="#ff7f0e")
    axes[0, 1].set_xlabel("hit time (s)")
    axes[0, 1].set_ylabel("Δy (playfield px)")
    axes[0, 1].set_title("Δy over time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].axhline(0, color="black", lw=0.5)
    axes[1, 0].axvline(0, color="black", lw=0.5)
    axes[1, 0].add_patch(Circle((0, 0), r, fc="none", ec="#1f77b4", lw=1.5,
                                label=f"CS radius ({r:.1f}px)"))
    axes[1, 0].scatter(dx, dy, s=10, alpha=0.4)
    axes[1, 0].scatter([dx.mean()], [dy.mean()], marker="x", color="red", s=200, lw=3,
                       label=f"mean ({dx.mean():+.1f}, {dy.mean():+.1f})")
    axes[1, 0].set_xlabel("Δx (px)")
    axes[1, 0].set_ylabel("Δy (px)")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_xlim(-300, 300)
    axes[1, 0].set_ylim(300, -300)
    axes[1, 0].set_title("2D miss distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(err_mag, bins=40, color="#2ca02c", alpha=0.7, edgecolor="black")
    axes[1, 1].axvline(r, color="red", lw=2,
                       label=f"CS radius ({r:.1f}px)")
    axes[1, 1].set_xlabel("|cursor − circle| at hit time (playfield px)")
    axes[1, 1].set_ylabel("# circles")
    axes[1, 1].set_title(f"Miss-distance histogram"
                         f"  (median={np.median(err_mag):.1f}, mean={err_mag.mean():.1f},"
                         f" within-CS={(err_mag < r).mean()*100:.0f}%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Cursor alignment vs. circles at hit time  "
                 f"(n={len(circles)}, cs={bm.cs})", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cursor_over_time(d, bm, out_path: Path):
    mask = ~d["game_paused"]
    map_t = d["map_time_ms"][mask].astype(np.int64)
    cx = d["cursor_x"][mask].astype(np.float32)
    cy = d["cursor_y"][mask].astype(np.float32)
    valid = map_t >= 0
    map_t = map_t[valid]; cx = cx[valid]; cy = cy[valid]
    if len(map_t) == 0:
        return
    pf_x, pf_y = screen_to_playfield(cx, cy)

    circles = [n for n in bm.notes if n.type == CIRCLE
               and map_t.min() <= n.t_start <= map_t.max()]
    sliders_starts = [n.t_start for n in bm.notes if n.type == SLIDER
                      and map_t.min() <= n.t_start <= map_t.max()]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for ax, vals, label, c in [
        (axes[0], pf_x, "cursor playfield X", "#1f77b4"),
        (axes[1], pf_y, "cursor playfield Y", "#ff7f0e"),
    ]:
        ax.plot(map_t / 1000, vals, color=c, lw=0.7, alpha=0.8)
        for cobj in circles:
            ax.axvline(cobj.t_start / 1000, color="green", alpha=0.15, lw=0.5)
            target = cobj.x if label.endswith("X") else cobj.y
            ax.plot(cobj.t_start / 1000, target, "o", color="green", ms=4, alpha=0.7)
        for st in sliders_starts:
            ax.axvline(st / 1000, color="orange", alpha=0.1, lw=0.4)
        ax.set_ylabel(label)
        ax.set_xlim(map_t.min() / 1000, map_t.max() / 1000)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("map_time (s)")
    axes[0].set_title(
        "Cursor over time (line) with circle hit times (green dots = expected target)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _segment_indices(map_t_ms, min_len: int = 30):
    # Split on backwards jumps (restart) or gaps > 2s (pause/seek).
    if len(map_t_ms) == 0:
        return []
    diffs = np.diff(map_t_ms)
    breaks = np.where((diffs < 0) | (diffs > 2000))[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(map_t_ms)]])
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def plot_clock_alignment(d, out_path: Path):
    mask = ~d["game_paused"]
    t_ns = d["t_ns"][mask]
    map_t = d["map_time_ms"][mask]
    valid = map_t >= 0
    t_ns = t_ns[valid]
    map_t = map_t[valid]
    if len(t_ns) < 2:
        return
    t_wall_s = (t_ns - t_ns[0]) / 1e9
    t_map_s = map_t / 1000.0
    segs = _segment_indices(map_t)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(t_wall_s, t_map_s, ".", ms=2, alpha=0.4, color="gray", label="all data")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(segs), 1)))
    all_resid = []
    for i, (s, e) in enumerate(segs):
        xs, ys = t_wall_s[s:e], t_map_s[s:e]
        p = np.polyfit(xs, ys, 1)
        fit = np.polyval(p, xs)
        resid_ms = (ys - fit) * 1000.0
        all_resid.append(resid_ms)
        axes[0].plot(xs, fit, "-", lw=1.2, color=colors[i],
                     label=f"seg {i+1}: slope={p[0]:.4f}, σ={resid_ms.std():.1f}ms (n={e-s})")
        axes[1].plot(xs, resid_ms, ".", ms=2, alpha=0.6, color=colors[i])

    axes[0].set_xlabel("wall_clock (s, monotonic)")
    axes[0].set_ylabel("map_time (s)")
    axes[0].set_title(f"Clock alignment — {len(segs)} contiguous play segment(s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color="red", lw=0.7)
    if all_resid:
        all_r = np.concatenate(all_resid)
        axes[1].set_title(f"Per-segment residual  (combined σ={all_r.std():.2f} ms)")
    axes[1].set_xlabel("wall_clock (s)")
    axes[1].set_ylabel("residual (ms) = map_time − fit")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_frame_samples(d, bm, out_path: Path, n_samples: int = 6):
    mask = ~d["game_paused"]
    map_t_full = d["map_time_ms"]
    valid_idx_all = np.where(mask & (map_t_full >= 0))[0]
    if len(valid_idx_all) == 0:
        return
    valid_map_t = map_t_full[valid_idx_all]

    circles = [n for n in bm.notes if n.type == CIRCLE
               and valid_map_t.min() <= n.t_start <= valid_map_t.max()]
    if not circles:
        return
    chosen = [circles[int(k)] for k in np.linspace(0, len(circles) - 1, n_samples)]

    H, W = d["frame"].shape[1:]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, c in zip(axes.flat, chosen):
        idx = int(np.searchsorted(valid_map_t, c.t_start))
        idx = min(max(idx, 0), len(valid_map_t) - 1)
        frame_idx = int(valid_idx_all[idx])
        frame = d["frame"][frame_idx]
        ax.imshow(frame, cmap="gray", interpolation="nearest")

        sx, sy = playfield_to_screen_in_region(c.x, c.y)
        fx = float(sx) / REGION_W * W
        fy = float(sy) / REGION_H * H
        r = circle_radius_px(bm.cs) / 512.0 * W
        ax.add_patch(Circle((fx, fy), r, fc="none", ec="lime", lw=1.5))

        cur_sx = float(d["cursor_x"][frame_idx]) - REGION_X
        cur_sy = float(d["cursor_y"][frame_idx]) - REGION_Y
        cur_fx = cur_sx / REGION_W * W
        cur_fy = cur_sy / REGION_H * H
        ax.plot(cur_fx, cur_fy, "x", color="red", ms=10, mew=2)

        ax.set_title(f"t={c.t_start}ms  circle=({c.x:.0f},{c.y:.0f})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Captured frames at circle-hit moments  "
                 "(green = expected circle, red × = cursor)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_all(npz_path: Path) -> dict:
    d, meta, bm = load_session(npz_path)
    out_dir = REPO_ROOT / "notebooks" / "phase3" / npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_spatial_windows(d, bm, out_dir / "01_spatial_windows.png")
    plot_circle_error(d, bm, out_dir / "02_circle_error.png")
    plot_cursor_over_time(d, bm, out_dir / "03_cursor_over_time.png")
    plot_clock_alignment(d, out_dir / "04_clock_alignment.png")
    plot_frame_samples(d, bm, out_dir / "05_frame_samples.png")

    mask = ~d["game_paused"]
    valid = mask & (d["map_time_ms"] >= 0)
    map_t = d["map_time_ms"][valid].astype(np.int64)
    cx = d["cursor_x"][valid].astype(np.float32)
    cy = d["cursor_y"][valid].astype(np.float32)
    sort_idx = np.argsort(map_t)
    map_t_s = map_t[sort_idx]
    pf_x, pf_y = screen_to_playfield(cx[sort_idx], cy[sort_idx])
    circles = [n for n in bm.notes if n.type == CIRCLE
               and map_t_s.min() <= n.t_start <= map_t_s.max()]
    errs, dxs, dys = [], [], []
    for c in circles:
        idx = int(np.searchsorted(map_t_s, c.t_start))
        idx = min(max(idx, 0), len(map_t_s) - 1)
        dxs.append(pf_x[idx] - c.x)
        dys.append(pf_y[idx] - c.y)
        errs.append((pf_x[idx] - c.x) ** 2 + (pf_y[idx] - c.y) ** 2)
    errs = np.sqrt(np.array(errs)) if errs else np.array([])
    r = circle_radius_px(bm.cs)

    t_ns_v = d["t_ns"][valid]
    t_wall_s = (t_ns_v - t_ns_v[0]) / 1e9
    t_map_s = d["map_time_ms"][valid] / 1000.0
    segs = _segment_indices(d["map_time_ms"][valid])
    seg_fits = []
    all_resid = []
    for s, e in segs:
        xs, ys = t_wall_s[s:e], t_map_s[s:e]
        pp = np.polyfit(xs, ys, 1)
        rr = (ys - np.polyval(pp, xs)) * 1000.0
        all_resid.append(rr)
        seg_fits.append({
            "n": int(e - s),
            "wall_start_s": float(xs[0]),
            "wall_end_s": float(xs[-1]),
            "map_start_s": float(ys[0]),
            "map_end_s": float(ys[-1]),
            "slope_map_per_wall": float(pp[0]),
            "intercept_s": float(pp[1]),
            "residual_std_ms": float(rr.std()),
            "residual_max_ms": float(np.abs(rr).max()),
        })
    combined_resid = np.concatenate(all_resid) if all_resid else np.array([])

    summary = {
        "session": npz_path.name,
        "map_title": bm.title,
        "map_id": meta.get("map_id"),
        "cs": bm.cs,
        "cs_radius_px": float(r),
        "n_frames": int(valid.sum()),
        "frames_paused": int(d["game_paused"].sum()),
        "circles_in_range": len(circles),
        "circle_err_px": {
            "mean_dx": float(np.mean(dxs)) if dxs else None,
            "median_dx": float(np.median(dxs)) if dxs else None,
            "mean_dy": float(np.mean(dys)) if dys else None,
            "median_dy": float(np.median(dys)) if dys else None,
            "mean_dist": float(errs.mean()) if errs.size else None,
            "median_dist": float(np.median(errs)) if errs.size else None,
            "p90_dist": float(np.percentile(errs, 90)) if errs.size else None,
            "within_cs_radius_pct": float((errs < r).mean() * 100) if errs.size else None,
        },
        "clock_fit": {
            "n_segments": len(seg_fits),
            "segments": seg_fits,
            "combined_residual_std_ms": float(combined_resid.std()) if combined_resid.size else None,
        },
        "out_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, help=".npz produced by self_capture")
    args = parser.parse_args()
    summary = run_all(args.session)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
