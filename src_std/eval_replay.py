"""Phase 6c — offline replay evaluation for the BC model.

Loads a checkpoint produced by ``src_std/train_bc.py`` and runs the model
frame-by-frame across one session, then renders:

  - timeline: cursor_x, cursor_y, press for GT vs predicted over time
  - playfield strip: N evenly-spaced snapshots showing cursor positions
    (GT dot vs predicted dot) and the active/upcoming hit objects

Usage:
    python -m src_std.eval_replay \\
        --ckpt runs/bc/<utc>/best.pt \\
        --session data/sessions/<file>.npz \\
        --out runs/bc/<utc>/eval_replay

We currently have only one session (Phase 4 smoke), so the default session
*is* the training session. That's intentional for this first eval: we want
to see whether the model can even imitate a sequence it has been exposed
to before we worry about generalization to held-out maps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src_std.data.dataset import OsuStdDataset, PLAYFIELD_W, PLAYFIELD_H  # noqa: E402
from src_std.model_bc import BCModelConfig, BCPolicy  # noqa: E402
from src_std.parse_std import (  # noqa: E402
    CIRCLE, SLIDER, SPINNER, parse_beatmap_std,
)


def load_model(ckpt_path: Path, device: str) -> BCPolicy:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = BCModelConfig(**ckpt["model_cfg"])
    net = BCPolicy(cfg).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    return net


@torch.no_grad()
def run_replay(net: BCPolicy, ds: OsuStdDataset, device: str,
               session_idx: int = 0):
    """Predict cursor_xy/press for every sample in ``session_idx``."""
    sess = ds.sessions[session_idx]
    base = int(np.cumsum([0] + [s.sample_centers.shape[0]
                                for s in ds.sessions])[session_idx])
    n = sess.sample_centers.shape[0]
    gt_xy = np.zeros((n, 2), dtype=np.float32)
    gt_press = np.zeros(n, dtype=np.float32)
    pred_xy = np.zeros((n, 2), dtype=np.float32)
    pred_press_prob = np.zeros(n, dtype=np.float32)
    BATCH = 64
    for i0 in range(0, n, BATCH):
        i1 = min(i0 + BATCH, n)
        states = []
        for i in range(i0, i1):
            s, _ = ds[base + i]
            states.append(s)
        frames = torch.stack([s["frames"] for s in states]).to(device)
        mctx = torch.stack([s["map_ctx"] for s in states]).to(device)
        svec = torch.stack([s["state_vec"] for s in states]).to(device)
        cursor, logit = net(frames, mctx, svec)
        pred_xy[i0:i1] = cursor.cpu().numpy()
        pred_press_prob[i0:i1] = torch.sigmoid(logit).cpu().numpy()
        for i in range(i0, i1):
            c = int(sess.sample_centers[i])
            gt_xy[i] = sess.cursor_pf[c]
            gt_press[i] = sess.press[c]
    return {
        "kept_centers": sess.sample_centers.copy(),
        "gt_xy": gt_xy,
        "gt_press": gt_press,
        "pred_xy": pred_xy,
        "pred_press_prob": pred_press_prob,
    }


def compute_metrics(out: dict, press_thresh: float = 0.5) -> dict:
    gt_xy = out["gt_xy"]
    pred_xy = out["pred_xy"]
    gt_press = out["gt_press"]
    pred_press = (out["pred_press_prob"] > press_thresh).astype(np.float32)
    mae_px = float(np.linalg.norm(gt_xy - pred_xy, axis=1).mean())
    if gt_xy.shape[0] >= 2:
        cx = float(np.corrcoef(gt_xy[:, 0], pred_xy[:, 0])[0, 1])
        cy = float(np.corrcoef(gt_xy[:, 1], pred_xy[:, 1])[0, 1])
    else:
        cx = cy = float("nan")
    return {
        "n_frames": int(gt_xy.shape[0]),
        "cursor_mae_px": mae_px,
        "cursor_corr_x": cx,
        "cursor_corr_y": cy,
        "press_acc": float((gt_press == pred_press).mean()),
        "press_pred_pos_rate": float(pred_press.mean()),
        "press_gt_pos_rate": float(gt_press.mean()),
    }


def _active_objects(bm, t_ms: int, window_ms: int = 600):
    out = []
    for n in bm.notes:
        if n.t_end + window_ms < t_ms:
            continue
        if n.t_start - window_ms > t_ms:
            break
        out.append(n)
    return out


def render(out: dict, ds: OsuStdDataset, session_idx: int,
           bm, png_path: Path, n_strip: int = 8):
    n = out["gt_xy"].shape[0]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, n_strip, hspace=0.5, wspace=0.15)
    x_axis = np.arange(n)

    ax_x = fig.add_subplot(gs[0, :])
    ax_x.plot(x_axis, out["gt_xy"][:, 0], color="#1f77b4", lw=1, label="GT")
    ax_x.plot(x_axis, out["pred_xy"][:, 0], color="#d62728", lw=1, alpha=0.8,
              label="Pred")
    ax_x.set_ylabel("cursor x (pf px)")
    ax_x.set_ylim(0, PLAYFIELD_W)
    ax_x.legend(loc="upper right", fontsize=8)
    ax_x.set_title("Replay eval — GT vs predicted")

    ax_y = fig.add_subplot(gs[1, :], sharex=ax_x)
    ax_y.plot(x_axis, out["gt_xy"][:, 1], color="#1f77b4", lw=1)
    ax_y.plot(x_axis, out["pred_xy"][:, 1], color="#d62728", lw=1, alpha=0.8)
    ax_y.set_ylabel("cursor y (pf px)")
    ax_y.set_ylim(0, PLAYFIELD_H)

    ax_p = fig.add_subplot(gs[2, :], sharex=ax_x)
    ax_p.fill_between(x_axis, 0, out["gt_press"],
                      color="#1f77b4", alpha=0.4, label="GT press")
    ax_p.plot(x_axis, out["pred_press_prob"], color="#d62728", lw=1,
              label="Pred prob")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.set_ylabel("press")
    ax_p.set_xlabel("sample idx (30 Hz)")
    ax_p.legend(loc="upper right", fontsize=8)

    sess = ds.sessions[session_idx]
    snap_idx = np.linspace(0, n - 1, n_strip).astype(int)
    raw = np.load(sess.npz_path, allow_pickle=True)
    map_t = raw["map_time_ms"]
    paused = raw["game_paused"]
    keep_mask = (~paused) & (map_t >= 0)
    kept_map_t = map_t[keep_mask]

    for col, si in enumerate(snap_idx):
        center = int(sess.sample_centers[si])
        ax = fig.add_subplot(gs[3, col])
        ax.add_patch(patches.Rectangle(
            (0, 0), PLAYFIELD_W, PLAYFIELD_H,
            linewidth=1, edgecolor="black", facecolor="#f6f6f6"))
        t_ms = int(kept_map_t[center])
        for n_obj in _active_objects(bm, t_ms, window_ms=400):
            color = {CIRCLE: "#aaaaaa",
                     SLIDER: "#888888",
                     SPINNER: "#cccccc"}[n_obj.type]
            ax.add_patch(patches.Circle(
                (n_obj.x, n_obj.y), 20, color=color, alpha=0.5))
            if n_obj.type == SLIDER and n_obj.slider_curve:
                pts = np.array(n_obj.slider_curve)
                ax.plot(pts[:, 0], pts[:, 1], color="#555555",
                        lw=1, alpha=0.4)
        ax.plot(out["gt_xy"][si, 0], out["gt_xy"][si, 1],
                "o", color="#1f77b4", ms=8, label="GT")
        ax.plot(out["pred_xy"][si, 0], out["pred_xy"][si, 1],
                "x", color="#d62728", ms=8, mew=2, label="Pred")
        gp = int(out["gt_press"][si])
        pp = int(out["pred_press_prob"][si] > 0.5)
        ax.set_title(f"t={t_ms/1000:.1f}s\nGT k={gp} | Pred k={pp}",
                     fontsize=8)
        ax.set_xlim(-20, PLAYFIELD_W + 20)
        ax.set_ylim(PLAYFIELD_H + 20, -20)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.legend(loc="lower left", fontsize=7)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--session", type=Path, default=None,
                    help="Session .npz to replay. Default: first under "
                         "data/sessions/.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path prefix (no extension). Writes .png + .json.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.session is None:
        candidates = sorted(Path("data/sessions").glob("*.npz"))
        if not candidates:
            print("no sessions found", file=sys.stderr)
            sys.exit(2)
        args.session = candidates[0]

    print(f"loading checkpoint {args.ckpt}")
    net = load_model(args.ckpt, args.device)
    print(f"replaying session {args.session.name}")

    ds = OsuStdDataset([args.session])
    out = run_replay(net, ds, args.device, session_idx=0)

    metrics = compute_metrics(out)
    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    meta = json.loads(str(np.load(args.session, allow_pickle=True)["meta"]))
    bm = parse_beatmap_std(Path(meta["map_file_abs"]))
    png_path = args.out.with_suffix(".png")
    json_path = args.out.with_suffix(".json")
    render(out, ds, session_idx=0, bm=bm, png_path=png_path)
    with json_path.open("w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "session": str(args.session),
            "metrics": metrics,
        }, f, indent=2)
    print(f"wrote {png_path} and {json_path}")


if __name__ == "__main__":
    main()
