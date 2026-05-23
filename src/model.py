"""Tiny Transformer for osu!mania behavioral cloning."""

import torch
import torch.nn as nn


class ManiaTransformer(nn.Module):
    """Maps a lookahead window of notes → next-tick keystate logits.

    Input  : (B, T, K, F)   T=lookahead_ticks, K=columns, F=note features
    Output : (B, K)         per-column press logits (sigmoid → probability)
    """

    def __init__(self,
                 keys: int = 4,
                 n_features: int = 5,
                 lookahead_ticks: int = 200,
                 n_targets: int = 1,
                 d_model: int = 128,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dim_ff: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.keys = keys
        self.n_features = n_features
        self.lookahead_ticks = lookahead_ticks
        self.n_targets = n_targets
        input_dim = keys * n_features

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, lookahead_ticks, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, keys)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B, T, K, F = state.shape
        x = state.reshape(B, T, K * F)
        x = self.input_proj(x) + self.pos_emb[:, :T]
        x = self.encoder(x)
        if self.n_targets == 1:
            return self.head(x[:, 0])
        return self.head(x[:, :self.n_targets])
