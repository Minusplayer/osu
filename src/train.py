"""Phase 3 sanity check: overfit a tiny Transformer on ONE (map, replay) pair.

If loss does not drop near zero on a single replay, something is wrong with
parsing/alignment/state construction. Fix that before scaling.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ManiaDataset
from model import ManiaTransformer


def main(map_path: str, replay_path: str, *,
         epochs: int = 20, batch_size: int = 256, lr: float = 1e-3,
         save_path: str = "checkpoints/overfit.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = ManiaDataset(map_path, replay_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model = ManiaTransformer(
        keys=ds.keys, lookahead_ticks=ds.lookahead_ticks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model:   {n_params/1e6:.2f}M params")
    print(f"dataset: {len(ds)} samples, {len(loader)} batches/epoch")
    print()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for state, action in loader:
            state = state.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)

            logits = model(state)
            loss = loss_fn(logits, action)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * state.size(0)
            pred = (logits > 0).float()
            correct += (pred == action).float().sum().item()
            total += action.numel()

        epoch_loss /= len(ds)
        acc = correct / total
        print(f"epoch {epoch:3d}/{epochs}  "
              f"loss={epoch_loss:.5f}  acc={acc*100:6.2f}%  "
              f"t={time.time()-t0:.1f}s")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "keys": ds.keys,
                "lookahead_ticks": ds.lookahead_ticks},
               save_path)
    print(f"\nsaved → {save_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python src/train.py <map.osu> <replay.osr> [epochs]")
        sys.exit(1)
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    main(sys.argv[1], sys.argv[2], epochs=epochs)
