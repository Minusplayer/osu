"""Phase 6 — behavioral cloning trainer for osu!std.

Distinct from ``src_std/train.py`` (PPO trainer for ``model_std``); this
script trains the BCPolicy from ``src_std/model_bc.py``.

Usage:
    # Phase 6a smoke (3 epochs on existing session, print every step):
    python -m src_std.train_bc --smoke

    # Phase 6b full run (uses configs/training_bc.json schedule):
    python -m src_std.train_bc

Reads ``configs/training_bc.json`` for model + loss + optim + schedule.

Outputs (under --out-dir, default ``runs/bc/<UTC YYYYMMDDTHHMMSS>``):
    config.json            — frozen copy of the config that produced this run
    train_log.jsonl        — one JSON line per epoch (train/val metrics)
    best.pt                — checkpoint with lowest val cursor MAE (px)
    last.pt                — final-epoch checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src_std.data.dataset import OsuStdDataset  # noqa: E402
from src_std.model_bc import BCModelConfig, BCPolicy, count_params  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training_bc.json"


def collate(batch):
    states, actions = zip(*batch)
    return (
        {
            "frames": torch.stack([s["frames"] for s in states]),
            "map_ctx": torch.stack([s["map_ctx"] for s in states]),
            "state_vec": torch.stack([s["state_vec"] for s in states]),
        },
        {
            "cursor_pf": torch.stack([a["cursor_pf"] for a in actions]),
            "press": torch.stack([a["press"] for a in actions]),
        },
    )


def _split_indices_by_segment(ds: OsuStdDataset, val_frac: float, seed: int = 0):
    """Split sample indices keeping each beatmap-segment intact.

    Phase 4 segments are contiguous in the per-session sample order. We split
    by *segment* rather than per-sample to prevent leakage between adjacent
    frames (frames within a segment are 33ms apart and share neighbors).
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    base = 0
    for sess in ds.sessions:
        centers = sess.sample_centers
        n = len(centers)
        if n == 0:
            continue
        seg_breaks = [0]
        for i in range(1, n):
            if int(centers[i]) - int(centers[i - 1]) != 1:
                seg_breaks.append(i)
        seg_breaks.append(n)
        spans = list(zip(seg_breaks[:-1], seg_breaks[1:]))
        order = rng.permutation(len(spans))
        target_val = int(round(val_frac * n))
        val_count = 0
        val_spans: set[int] = set()
        for k in order:
            s, e = spans[int(k)]
            if val_count + (e - s) <= max(1, int(target_val * 1.5)):
                val_spans.add(int(k))
                val_count += (e - s)
            if val_count >= target_val:
                break
        for k, (s, e) in enumerate(spans):
            for j in range(s, e):
                (val if k in val_spans else train).append(base + j)
        base += n
    return train, val


def cursor_loss_fn(pred_xy: torch.Tensor,
                   tgt_xy: torch.Tensor,
                   scale_px: float) -> torch.Tensor:
    return F.smooth_l1_loss(pred_xy / scale_px, tgt_xy / scale_px)


def press_loss_fn(logit: torch.Tensor,
                  tgt: torch.Tensor,
                  pos_weight: float) -> torch.Tensor:
    pw = torch.tensor(pos_weight, device=logit.device, dtype=logit.dtype)
    return F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pw)


def evaluate(net: BCPolicy, loader: DataLoader, device: str, cfg_loss: dict):
    net.eval()
    n_seen = 0
    sum_cl = 0.0
    sum_pl = 0.0
    sum_cursor_px = 0.0
    n_press_correct = 0
    sum_press_pos_pred = 0.0
    with torch.no_grad():
        for state, action in loader:
            B = state["frames"].shape[0]
            frames = state["frames"].to(device, non_blocking=True)
            mctx = state["map_ctx"].to(device, non_blocking=True)
            svec = state["state_vec"].to(device, non_blocking=True)
            tgt_xy = action["cursor_pf"].to(device, non_blocking=True)
            tgt_p = action["press"].to(device, non_blocking=True)
            pred_xy, logit = net(frames, mctx, svec)
            cl = cursor_loss_fn(pred_xy, tgt_xy, cfg_loss["cursor_scale_px"])
            pl = press_loss_fn(logit, tgt_p, cfg_loss["press_pos_weight"])
            sum_cl += cl.item() * B
            sum_pl += pl.item() * B
            sum_cursor_px += (pred_xy - tgt_xy).pow(2).sum(dim=-1).sqrt().sum().item()
            pred_p = (torch.sigmoid(logit) > 0.5).float()
            n_press_correct += (pred_p == tgt_p).sum().item()
            sum_press_pos_pred += pred_p.sum().item()
            n_seen += B
    return {
        "cursor_loss": sum_cl / max(1, n_seen),
        "press_loss": sum_pl / max(1, n_seen),
        "cursor_mae_px": sum_cursor_px / max(1, n_seen),
        "press_acc": n_press_correct / max(1, n_seen),
        "press_pred_pos_rate": sum_press_pos_pred / max(1, n_seen),
        "n": n_seen,
    }


def make_lr_lambda(warmup_steps: int, total_steps: int, decay: str):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if decay == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    return lr_lambda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--sessions-dir", type=Path, default=Path("data/sessions"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run: 3 epochs, prints every step, no checkpointing.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with args.config.open() as f:
        cfg = json.load(f)

    paths = sorted(args.sessions_dir.glob("*.npz"))
    if not paths:
        print(f"no .npz under {args.sessions_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"loading {len(paths)} session(s) from {args.sessions_dir}...")
    ds = OsuStdDataset(
        paths,
        label_window_ms=cfg["data"]["label_window_ms"],
        gap_ms=cfg["data"]["gap_ms"],
    )
    print(f"  dataset: {len(ds)} samples")

    train_idx, val_idx = _split_indices_by_segment(
        ds, val_frac=cfg["data"]["val_frac"], seed=0)
    print(f"  split: train={len(train_idx)} val={len(val_idx)}")

    num_workers = cfg["schedule"].get("num_workers", 4)
    pin = args.device == "cuda"
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=cfg["schedule"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=cfg["schedule"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin,
    )

    model_cfg = BCModelConfig(**cfg["model"]["config"])
    net = BCPolicy(model_cfg).to(args.device)
    print(f"  model: {count_params(net):,} params on {args.device}")

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    epochs = 3 if args.smoke else cfg["schedule"]["epochs"]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = cfg["schedule"]["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(warmup, total_steps, cfg["schedule"]["lr_decay"]))

    out_dir = args.out_dir
    if out_dir is None and not args.smoke:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_dir = Path("runs/bc") / ts
    log_f = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "config.json").open("w") as f:
            json.dump(cfg, f, indent=2)
        log_f = (out_dir / "train_log.jsonl").open("w")

    best_mae = float("inf")
    step = 0
    for ep in range(epochs):
        net.train()
        ep_start = time.time()
        sum_cl = 0.0
        sum_pl = 0.0
        n_seen = 0
        for state, action in train_loader:
            B = state["frames"].shape[0]
            frames = state["frames"].to(args.device, non_blocking=True)
            mctx = state["map_ctx"].to(args.device, non_blocking=True)
            svec = state["state_vec"].to(args.device, non_blocking=True)
            tgt_xy = action["cursor_pf"].to(args.device, non_blocking=True)
            tgt_p = action["press"].to(args.device, non_blocking=True)

            pred_xy, logit = net(frames, mctx, svec)
            cl = cursor_loss_fn(pred_xy, tgt_xy, cfg["loss"]["cursor_scale_px"])
            pl = press_loss_fn(logit, tgt_p, cfg["loss"]["press_pos_weight"])
            loss = (cfg["loss"]["cursor_weight"] * cl
                    + cfg["loss"]["press_weight"] * pl)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            sched.step()

            sum_cl += cl.item() * B
            sum_pl += pl.item() * B
            n_seen += B
            step += 1
            if args.smoke:
                print(f"  ep{ep} step{step:>4}  "
                      f"cl={cl.item():.4f}  pl={pl.item():.4f}  "
                      f"loss={loss.item():.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}")

        val_metrics = evaluate(net, val_loader, args.device, cfg["loss"])
        val_loss = (cfg["loss"]["cursor_weight"] * val_metrics["cursor_loss"]
                    + cfg["loss"]["press_weight"] * val_metrics["press_loss"])
        train_cl = sum_cl / max(1, n_seen)
        train_pl = sum_pl / max(1, n_seen)
        dt = time.time() - ep_start
        print(f"epoch {ep:>2}/{epochs}  "
              f"train_cl={train_cl:.4f} train_pl={train_pl:.4f}  "
              f"val_cl={val_metrics['cursor_loss']:.4f} "
              f"val_pl={val_metrics['press_loss']:.4f}  "
              f"cursor_mae_px={val_metrics['cursor_mae_px']:.1f}  "
              f"press_acc={val_metrics['press_acc']:.3f}  "
              f"({dt:.1f}s)")
        rec = {
            "epoch": ep,
            "step": step,
            "lr": sched.get_last_lr()[0],
            "train_cursor_loss": train_cl,
            "train_press_loss": train_pl,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_secs": dt,
        }
        if log_f is not None:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
        mae = val_metrics["cursor_mae_px"]
        if out_dir is not None and mae < best_mae:
            best_mae = mae
            torch.save({
                "epoch": ep,
                "step": step,
                "model_cfg": model_cfg.__dict__,
                "state_dict": net.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }, out_dir / "best.pt")

    if out_dir is not None:
        torch.save({
            "epoch": epochs - 1,
            "step": step,
            "model_cfg": model_cfg.__dict__,
            "state_dict": net.state_dict(),
        }, out_dir / "last.pt")
        if log_f is not None:
            log_f.close()
        print(f"wrote checkpoints + log to {out_dir}")


if __name__ == "__main__":
    main()
