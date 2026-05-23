"""Phase 6: train on multiple (map, replay) pairs with held-out validation maps.

Auto-pairs replays in <replay-dir> to beatmaps in <map-root> by MD5 hash,
splits maps into train/val, and trains with early-stop on val loss.
"""

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

# NixOS: Triton's libcuda_dirs() shells out to /sbin/ldconfig which doesn't
# exist. Setting this env var (read first by Triton) bypasses that probe.
os.environ.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (GPUMultiMapDataset, MultiMapDataset, NUM_FEATURES,
                     pair_replays_with_maps, split_pairs_by_map)
from model import ManiaTransformer


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    @torch.no_grad()
    def swap_in(self, model: nn.Module):
        sd = model.state_dict()
        self._backup = {k: sd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            sd[k].copy_(v)

    @torch.no_grad()
    def swap_out(self, model: nn.Module):
        if self._backup is None:
            return
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None


def epoch_pass(model, batches, total_samples, loss_fn, device,
               opt=None, amp_dtype=None, ema=None):
    is_train = opt is not None
    model.train(is_train)
    total_loss_t = torch.zeros((), device=device)
    correct_t = torch.zeros((), device=device)
    total_t = torch.zeros((), device=device)
    use_amp = amp_dtype is not None and device.type == "cuda"
    for state, action in batches:
        if state.device != device:
            state = state.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            ctx = (torch.amp.autocast("cuda", dtype=amp_dtype)
                   if use_amp else torch.amp.autocast("cuda", enabled=False))
            with ctx:
                logits = model(state)
                loss = loss_fn(logits, action)
            if is_train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update(model._orig_mod
                               if hasattr(model, "_orig_mod") else model)
        total_loss_t += loss.detach() * state.size(0)
        pred = (logits > 0)
        correct_t += (pred == action.bool()).sum()
        total_t += action.numel()
    total_loss = total_loss_t.item()
    correct = correct_t.item()
    total = total_t.item()
    return total_loss / total_samples, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", default="data/replays")
    p.add_argument("--maps", default="data/beatmaps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"],
                   help="autocast dtype (bf16 default; needs no GradScaler)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5,
                   help="early-stop epochs of no val-loss improvement")
    p.add_argument("--save", default="checkpoints/multi.pt")
    p.add_argument("--gpu-dataset", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="keep grids on GPU + vectorized batch gather (faster)")
    p.add_argument("--subsample", type=float, default=0.25,
                   help="fraction of ticks sampled per train epoch (1.0 = all)")
    p.add_argument("--sample-stride", type=int, default=1,
                   help="emit every Nth valid start tick (1 = every tick)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="torch.compile(model, mode='reduce-overhead')")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dim-ff", type=int, default=1024)
    p.add_argument("--n-targets", type=int, default=1,
                   help="number of future ticks to predict (multi-target head)")
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="4K key-symmetry augmentation (flip cols 0<->3, 1<->2)")
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                   help="track EMA of weights; eval/save use EMA params")
    p.add_argument("--ema-decay", type=float, default=0.999)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    print(f"pairing replays in {args.replays} with maps in {args.maps}...")
    pairs, skipped = pair_replays_with_maps(args.replays, args.maps)
    print(f"  paired: {len(pairs)}    unmatched: {skipped}")
    if not pairs:
        sys.exit("no pairs found — check replay/map paths")

    train_pairs, val_pairs = split_pairs_by_map(pairs, args.val_frac)
    n_train_maps = len({str(m) for m, _ in train_pairs})
    n_val_maps = len({str(m) for m, _ in val_pairs})
    print(f"  train: {len(train_pairs)} replays / {n_train_maps} maps")
    print(f"  val:   {len(val_pairs)} replays / {n_val_maps} maps\n")

    if args.gpu_dataset:
        print("building train dataset (GPU-resident)...")
        train_ds = GPUMultiMapDataset(train_pairs, device=device,
                                      target_ticks=args.n_targets,
                                      sample_stride=args.sample_stride)
        print(f"  -> {len(train_ds)} samples\n")
        print("building val dataset (GPU-resident)...")
        val_ds = GPUMultiMapDataset(val_pairs, device=device,
                                    target_ticks=args.n_targets)
        print(f"  -> {len(val_ds)} samples\n")
        nw = 0
        train_loader = val_loader = None
    else:
        print("building train dataset...")
        train_ds = MultiMapDataset(train_pairs)
        print(f"  -> {len(train_ds)} samples\n")
        print("building val dataset...")
        val_ds = MultiMapDataset(val_pairs)
        print(f"  -> {len(val_ds)} samples\n")
        nw = args.num_workers
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=(device.type == "cuda"),
                                  persistent_workers=(nw > 0))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=nw, pin_memory=(device.type == "cuda"),
                                persistent_workers=(nw > 0))

    model_kwargs = dict(
        keys=train_ds.keys,
        n_features=NUM_FEATURES,
        lookahead_ticks=train_ds.lookahead_ticks,
        n_targets=args.n_targets,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    )
    model = ManiaTransformer(**model_kwargs).to(device)
    ema = EMA(model, decay=args.ema_decay) if args.ema else None
    if args.compile:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model, mode="reduce-overhead")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, fused=True)
    loss_fn = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params, dropout={args.dropout}")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "off": None}[args.amp]
    print(f"batch={args.batch_size} workers={nw} amp={args.amp} "
          f"gpu_dataset={args.gpu_dataset}\n")

    def train_batches():
        if args.gpu_dataset:
            return train_ds.iter_batches(args.batch_size, shuffle=True,
                                         subsample=args.subsample,
                                         symmetric=args.symmetric)
        return train_loader

    def val_batches():
        if args.gpu_dataset:
            return val_ds.iter_batches(args.batch_size, shuffle=False)
        return val_loader

    n_train_per_epoch = (int(len(train_ds) * args.subsample)
                         if args.gpu_dataset and 0 < args.subsample < 1
                         else len(train_ds))
    best_val, best_epoch, stale = float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = epoch_pass(model, train_batches(), n_train_per_epoch,
                            loss_fn, device, opt, amp_dtype, ema=ema)
        # Evaluate (and save) with EMA weights swapped in.
        sd_src = model._orig_mod if hasattr(model, "_orig_mod") else model
        if ema is not None:
            ema.swap_in(sd_src)
        vl, va = epoch_pass(model, val_batches(), len(val_ds),
                            loss_fn, device, amp_dtype=amp_dtype)
        marker = ""
        if vl < best_val:
            best_val, best_epoch, stale = vl, epoch, 0
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": sd_src.state_dict(),
                        "keys": train_ds.keys,
                        "lookahead_ticks": train_ds.lookahead_ticks,
                        "model_kwargs": model_kwargs,
                        "dropout": args.dropout,
                        "epoch": epoch, "val_loss": vl},
                       args.save)
            marker = "  [saved]"
        else:
            stale += 1
        if ema is not None:
            ema.swap_out(sd_src)
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train: loss={tl:.4f} acc={ta*100:5.2f}%  "
              f"val: loss={vl:.4f} acc={va*100:5.2f}%  "
              f"t={time.time()-t0:.0f}s{marker}")
        if stale >= args.patience:
            print(f"\nearly stop: val loss didn't improve for {args.patience} epochs")
            break

    print(f"\nbest: epoch {best_epoch}, val loss {best_val:.4f}")
    print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
