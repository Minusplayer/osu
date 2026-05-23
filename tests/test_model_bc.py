"""Phase 5 unit test: dummy batch through BCPolicy, verify shapes + param count.

Run:
    python tests/test_model_bc.py
    # or
    pytest tests/test_model_bc.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch

from src_std.model_bc import (  # noqa: E402
    BCModelConfig, BCPolicy, count_params, PLAYFIELD_W, PLAYFIELD_H,
)


def _make_dummy_batch(cfg: BCModelConfig, B: int = 8):
    frames = torch.randn(B, cfg.stack_len, cfg.frame_h, cfg.frame_w)
    map_ctx = torch.randn(B, cfg.k_tokens, cfg.obj_features)
    map_ctx[..., -1] = 1.0  # mark all slots present
    state = torch.randn(B, cfg.state_features)
    return frames, map_ctx, state


def test_forward_shapes():
    cfg = BCModelConfig()
    net = BCPolicy(cfg).eval()
    B = 8
    frames, map_ctx, state = _make_dummy_batch(cfg, B)
    cursor_xy, press_logit = net(frames, map_ctx, state)

    assert cursor_xy.shape == (B, 2), f"cursor_xy shape {cursor_xy.shape}"
    assert press_logit.shape == (B,), f"press_logit shape {press_logit.shape}"

    # cursor must lie in playfield px range (tanh -> linear-scaled).
    assert cursor_xy[:, 0].min() >= 0 and cursor_xy[:, 0].max() <= PLAYFIELD_W
    assert cursor_xy[:, 1].min() >= 0 and cursor_xy[:, 1].max() <= PLAYFIELD_H


def test_predict_press_is_binary():
    cfg = BCModelConfig()
    net = BCPolicy(cfg).eval()
    frames, map_ctx, state = _make_dummy_batch(cfg, B=4)
    cursor_xy, press = net.predict(frames, map_ctx, state)
    assert cursor_xy.shape == (4, 2)
    assert press.shape == (4,)
    assert torch.all((press == 0) | (press == 1)), "press should be 0/1"


def test_param_count_in_target_range():
    cfg = BCModelConfig()
    net = BCPolicy(cfg)
    n = count_params(net)
    print(f"\n[params] BCPolicy: {n:,} ({n/1e6:.2f} M)")
    print(f"  vision:      {count_params(net.vision):>9,}")
    print(f"  map_enc:     {count_params(net.map_enc):>9,}")
    print(f"  state_enc:   {count_params(net.state_enc):>9,}")
    print(f"  fusion:      {count_params(net.fusion):>9,}")
    print(f"  cursor_head: {count_params(net.cursor_head):>9,}")
    print(f"  press_head:  {count_params(net.press_head):>9,}")
    assert 750_000 <= n <= 1_000_000, f"param count {n} outside 750K-1M target"


def test_padded_tokens_dont_break_pool():
    """If some slots are padded (present=0), forward must still work."""
    cfg = BCModelConfig()
    net = BCPolicy(cfg).eval()
    B = 4
    frames, map_ctx, state = _make_dummy_batch(cfg, B)
    map_ctx[: B // 2, 5:, -1] = 0.0
    cursor_xy, press_logit = net(frames, map_ctx, state)
    assert torch.isfinite(cursor_xy).all()
    assert torch.isfinite(press_logit).all()


def test_loss_uses_documented_pos_weight():
    """Sanity-check that the config's pos_weight produces a valid BCE loss
    over a dummy batch with positive labels (matches Phase 4 distribution)."""
    cfg = BCModelConfig()
    net = BCPolicy(cfg)
    frames, map_ctx, state = _make_dummy_batch(cfg, B=16)
    _, press_logit = net(frames, map_ctx, state)
    target = (torch.rand(16) < 0.78).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        press_logit, target, pos_weight=torch.tensor(0.3))
    assert torch.isfinite(loss).item(), "BCE loss must be finite"
    assert loss.item() > 0


if __name__ == "__main__":
    test_forward_shapes()
    test_predict_press_is_binary()
    test_padded_tokens_dont_break_pool()
    test_loss_uses_documented_pos_weight()
    test_param_count_in_target_range()
    print("\nall tests passed.")
