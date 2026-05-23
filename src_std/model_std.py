"""Object-token transformer policy for osu! standard.

Inputs:
  obj_feat    : (B, K=32, OBJ_FEATURES=11)   next-K hit object tokens
  cursor_feat : (B, CURSOR_FEATURES=6)       current cursor state

Architecture:
  - Project obj_feat → d_model, cursor_feat → d_model
  - Prepend cursor as the [CLS] token of the sequence
  - 4 transformer encoder layers (d=128, 4 heads, ff=256)
  - Read [CLS] for all three heads:
      mu     (B, 2)  : continuous cursor delta in [-1, 1] (tanh)
      log_std(2,)   : learned, per-dim
      logit  (B, 1) : Bernoulli click logit
      value  (B, 1) : V(s)
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Independent, Normal


class ObjectTokenPolicy(nn.Module):

    def __init__(self,
                 obj_features: int = 11,
                 cursor_features: int = 6,
                 k_tokens: int = 32,
                 d_model: int = 128,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 dim_ff: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.obj_features = obj_features
        self.cursor_features = cursor_features
        self.k_tokens = k_tokens
        self.d_model = d_model

        self.obj_proj = nn.Linear(obj_features, d_model)
        self.cursor_proj = nn.Linear(cursor_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, k_tokens + 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mu_head = nn.Linear(d_model, 2)
        self.click_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)
        self.log_std = nn.Parameter(torch.full((2,), -1.2))

    def encode(self, obj_feat, cursor_feat):
        obj_tok = self.obj_proj(obj_feat)
        cur_tok = self.cursor_proj(cursor_feat).unsqueeze(1)
        x = torch.cat([cur_tok, obj_tok], dim=1)
        x = x + self.pos_emb[:, : x.size(1)]
        return self.encoder(x)

    def forward(self, obj_feat, cursor_feat):
        h = self.encode(obj_feat, cursor_feat)
        cls = h[:, 0]
        mu = torch.tanh(self.mu_head(cls))
        click_logit = self.click_head(cls).squeeze(-1)
        value = self.value_head(cls).squeeze(-1)
        log_std = self.log_std.clamp(-5.0, 1.0)
        return mu, log_std, click_logit, value

    def act(self, obj_feat, cursor_feat, deterministic: bool = False):
        mu, log_std, click_logit, value = self.forward(obj_feat, cursor_feat)
        std = log_std.exp()
        dist_xy = Independent(Normal(mu, std.expand_as(mu)), 1)
        dist_clk = Bernoulli(logits=click_logit)
        if deterministic:
            dxy = mu
            click = (click_logit > 0).float()
        else:
            dxy = dist_xy.rsample()
            click = dist_clk.sample()
        log_p = dist_xy.log_prob(dxy) + dist_clk.log_prob(click)
        return dxy, click.bool(), log_p, value, mu, log_std, click_logit

    def evaluate(self, obj_feat, cursor_feat, dxy, click):
        mu, log_std, click_logit, value = self.forward(obj_feat, cursor_feat)
        std = log_std.exp()
        dist_xy = Independent(Normal(mu, std.expand_as(mu)), 1)
        dist_clk = Bernoulli(logits=click_logit)
        log_p = dist_xy.log_prob(dxy) + dist_clk.log_prob(click.float())
        ent = dist_xy.entropy() + dist_clk.entropy()
        return log_p, ent, value


def _smoke_test():
    net = ObjectTokenPolicy()
    obj = torch.randn(8, 32, 11)
    cur = torch.randn(8, 6)
    mu, log_std, lg, v = net(obj, cur)
    print(f"mu={tuple(mu.shape)} log_std={tuple(log_std.shape)} "
          f"click={tuple(lg.shape)} value={tuple(v.shape)}")
    dxy, click, log_p, value, *_ = net.act(obj, cur)
    print(f"sampled dxy={tuple(dxy.shape)} click={tuple(click.shape)} "
          f"log_p={tuple(log_p.shape)}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params/1e6:.2f} M")


if __name__ == "__main__":
    _smoke_test()
