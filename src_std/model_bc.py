"""Phase 5 behavioral-cloning policy for osu! standard.

Multi-modal fusion model:

  frames    (B, 4, H, W)             4 stacked grayscale frames at 30 Hz.
                                     H = 60, W = 80 (4:3 playfield aspect).
                                     Captures are stored at 96x96 and
                                     resized to 80x60 at build/load time.

  map_ctx   (B, K=12, F_obj=8)       Upcoming K hit objects as feature
                                     tokens. F_obj layout:
                                       [0] dt_norm    (t_to_object / 2000ms, clipped to [0,1])
                                       [1] x_norm     (x / 512)
                                       [2] y_norm     (y / 384)
                                       [3] dur_norm   ((t_end-t_start) / 4000, clipped)
                                       [4] is_circle  one-hot
                                       [5] is_slider  one-hot
                                       [6] is_spinner one-hot
                                       [7] present    1 if real, 0 if padded slot
                                     Padded slots get present=0; the
                                     transformer masks attention over them.

  state_vec (B, F_state=4)           [map_time_norm, prev_cursor_x_norm,
                                     prev_cursor_y_norm, prev_press]

Outputs:

  cursor_xy   (B, 2)   cursor position in playfield coords (px). The
                       model emits [-1, 1] via tanh, then scales to
                       [0, 512] × [0, 384] so downstream code stays in
                       playfield px (same convention as the dataset).

  press_logit (B,)     BCE logit; train with pos_weight (~0.3, see
                       training_config_bc.json).

Param-count target: 750K - 1M. Measured ~830K with defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


PLAYFIELD_W = 512.0
PLAYFIELD_H = 384.0


@dataclass
class BCModelConfig:
    frame_h: int = 60
    frame_w: int = 80
    stack_len: int = 4
    # Map tokens
    k_tokens: int = 12
    obj_features: int = 8
    map_d_model: int = 96
    map_layers: int = 2
    map_heads: int = 6
    map_ff: int = 192
    # State MLP
    state_features: int = 4
    state_hidden: int = 32
    # Vision CNN
    vision_out_dim: int = 384
    # Fusion
    fusion_hidden: int = 256
    fusion_out: int = 128
    # Heads
    cursor_dim: int = 2
    press_dim: int = 1
    dropout: float = 0.1


class _VisionCNN(nn.Module):
    """Atari-style strided CNN, sized for 96x96 stacked frames."""

    def __init__(self, cfg: BCModelConfig):
        super().__init__()
        self.cfg = cfg
        self.body = nn.Sequential(
            nn.Conv2d(cfg.stack_len, 32, kernel_size=7, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Compute flatten dim by tracing a dummy input through self.body.
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.stack_len, cfg.frame_h, cfg.frame_w)
            n_flat = self.body(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, cfg.vision_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(frames))


class _MapTransformer(nn.Module):
    """Token-level transformer over upcoming K hit objects."""

    def __init__(self, cfg: BCModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.obj_features, cfg.map_d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, cfg.k_tokens, cfg.map_d_model) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.map_d_model,
            nhead=cfg.map_heads,
            dim_feedforward=cfg.map_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.map_layers)
        self.out = nn.Linear(cfg.map_d_model, cfg.fusion_hidden // 2)

    def forward(self, map_ctx: torch.Tensor) -> torch.Tensor:
        # map_ctx: (B, K, F_obj). Last feature is `present` (1=real, 0=pad).
        present = map_ctx[..., -1] > 0.5      # (B, K) bool
        # If a row has ZERO present tokens (e.g. end of map), attention
        # softmax over no keys -> NaN. Unmask slot 0 so attention has at
        # least one key; pooling still ignores it because `mask` stays 0.
        all_padded = ~present.any(dim=1)      # (B,) bool
        key_padding_mask = ~present.clone()
        key_padding_mask[all_padded, 0] = False
        x = self.proj(map_ctx) + self.pos_emb
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mask = present.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h * mask).sum(dim=1) / denom
        # All-padded rows: force pooled=0 (NaN*0=NaN otherwise).
        pooled = torch.where(all_padded.unsqueeze(-1),
                             torch.zeros_like(pooled), pooled)
        return self.out(pooled)


class _StateMLP(nn.Module):
    def __init__(self, cfg: BCModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_features, cfg.state_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.state_hidden, cfg.state_hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec)


class BCPolicy(nn.Module):
    """Vision + map + state → cursor (regression) + press (logit)."""

    def __init__(self, cfg: BCModelConfig | None = None):
        super().__init__()
        cfg = cfg or BCModelConfig()
        self.cfg = cfg
        self.vision = _VisionCNN(cfg)
        self.map_enc = _MapTransformer(cfg)
        self.state_enc = _StateMLP(cfg)

        fused_in = cfg.vision_out_dim + (cfg.fusion_hidden // 2) + cfg.state_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_out),
            nn.ReLU(inplace=True),
        )

        self.cursor_head = nn.Linear(cfg.fusion_out, cfg.cursor_dim)
        self.press_head = nn.Linear(cfg.fusion_out, cfg.press_dim)

    def forward(self,
                frames: torch.Tensor,
                map_ctx: torch.Tensor,
                state_vec: torch.Tensor):
        v = self.vision(frames)
        m = self.map_enc(map_ctx)
        s = self.state_enc(state_vec)
        h = self.fusion(torch.cat([v, m, s], dim=-1))
        c = torch.tanh(self.cursor_head(h))   # (B, 2) in [-1, 1]
        cursor_xy = torch.stack([
            (c[..., 0] + 1.0) * 0.5 * PLAYFIELD_W,
            (c[..., 1] + 1.0) * 0.5 * PLAYFIELD_H,
        ], dim=-1)
        press_logit = self.press_head(h).squeeze(-1)
        return cursor_xy, press_logit

    @torch.no_grad()
    def predict(self, frames, map_ctx, state_vec, press_thresh: float = 0.5):
        cursor_xy, press_logit = self.forward(frames, map_ctx, state_vec)
        press = (torch.sigmoid(press_logit) > press_thresh).float()
        return cursor_xy, press


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _smoke():
    cfg = BCModelConfig()
    net = BCPolicy(cfg)
    B = 8
    frames = torch.randn(B, cfg.stack_len, cfg.frame_h, cfg.frame_w)
    map_ctx = torch.randn(B, cfg.k_tokens, cfg.obj_features)
    map_ctx[..., -1] = 1.0
    state = torch.randn(B, cfg.state_features)
    cur, lg = net(frames, map_ctx, state)
    print(f"cursor_xy : {tuple(cur.shape)}  "
          f"range=({cur.min().item():.1f}, {cur.max().item():.1f})")
    print(f"press_logit: {tuple(lg.shape)}")
    print(f"params: {count_params(net):,}  ({count_params(net)/1e6:.2f} M)")
    breakdown = {
        "vision": count_params(net.vision),
        "map_enc": count_params(net.map_enc),
        "state_enc": count_params(net.state_enc),
        "fusion": count_params(net.fusion),
        "cursor_head": count_params(net.cursor_head),
        "press_head": count_params(net.press_head),
    }
    print("breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:14s} {v:>9,}  ({v/1e6:.3f} M)")


if __name__ == "__main__":
    _smoke()
