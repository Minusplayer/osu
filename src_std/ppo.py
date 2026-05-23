"""PPO trainer for the osu!std env. Cleanrl-style.

One PPOTrainer wraps a model + an env. It collects T steps of rollouts
across the env's batch (B parallel envs in one OsuStdEnv), then runs
n_epochs of minibatch updates with clipped surrogate loss + value loss +
entropy bonus.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    n_epochs: int = 4
    n_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    advantage_norm: bool = True


class PPOTrainer:

    def __init__(self, model, env, cfg: PPOConfig = None, device=None):
        self.model = model
        self.env = env
        self.cfg = cfg or PPOConfig()
        self.device = device or next(model.parameters()).device
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr,
                                     weight_decay=0.0,
                                     fused=(self.device.type == "cuda"))
        self._reset_obs()

    def _reset_obs(self):
        obs_obj, obs_cur = self.env.reset()
        self._obs_obj = obs_obj
        self._obs_cur = obs_cur

    @torch.no_grad()
    def collect_rollout(self):
        T, B = self.cfg.rollout_steps, self.env.B
        dev = self.device

        obj_buf = torch.zeros(T, B, *self._obs_obj.shape[1:], device=dev)
        cur_buf = torch.zeros(T, B, *self._obs_cur.shape[1:], device=dev)
        dxy_buf = torch.zeros(T, B, 2, device=dev)
        click_buf = torch.zeros(T, B, dtype=torch.bool, device=dev)
        logp_buf = torch.zeros(T, B, device=dev)
        val_buf = torch.zeros(T, B, device=dev)
        rew_buf = torch.zeros(T, B, device=dev)
        done_buf = torch.zeros(T, B, device=dev)

        for t in range(T):
            obj_buf[t] = self._obs_obj
            cur_buf[t] = self._obs_cur
            dxy, click, log_p, value, *_ = self.model.act(self._obs_obj, self._obs_cur)
            dxy_buf[t] = dxy
            click_buf[t] = click
            logp_buf[t] = log_p
            val_buf[t] = value

            obs_obj, obs_cur, reward, done, info = self.env.step(dxy, click)
            rew_buf[t] = reward
            done_buf[t] = done.float()
            if done.all():
                obs_obj, obs_cur = self.env.reset()
            self._obs_obj = obs_obj
            self._obs_cur = obs_cur

        with torch.no_grad():
            _, _, _, last_val = self.model(self._obs_obj, self._obs_cur)

        adv_buf = torch.zeros_like(rew_buf)
        gae = torch.zeros(B, device=dev)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_buf[t + 1]
            mask = 1.0 - done_buf[t]
            delta = rew_buf[t] + self.cfg.gamma * next_val * mask - val_buf[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            adv_buf[t] = gae
        ret_buf = adv_buf + val_buf

        return {
            "obj": obj_buf.reshape(T * B, *obj_buf.shape[2:]),
            "cur": cur_buf.reshape(T * B, *cur_buf.shape[2:]),
            "dxy": dxy_buf.reshape(T * B, 2),
            "click": click_buf.reshape(T * B),
            "logp_old": logp_buf.reshape(T * B),
            "value_old": val_buf.reshape(T * B),
            "adv": adv_buf.reshape(T * B),
            "ret": ret_buf.reshape(T * B),
            "reward_mean": rew_buf.mean().item(),
            "reward_sum_per_env": rew_buf.sum(dim=0).mean().item(),
        }

    def update(self, batch):
        cfg = self.cfg
        N = batch["obj"].size(0)
        mb_size = max(N // cfg.n_minibatches, 1)

        adv = batch["adv"]
        if cfg.advantage_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = torch.randperm(N, device=batch["obj"].device)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        nu = 0
        for _ in range(cfg.n_epochs):
            for start in range(0, N, mb_size):
                mb = idx[start:start + mb_size]
                log_p, ent, value = self.model.evaluate(
                    batch["obj"][mb], batch["cur"][mb],
                    batch["dxy"][mb], batch["click"][mb],
                )
                ratio = (log_p - batch["logp_old"][mb]).exp()
                surr1 = ratio * adv[mb]
                surr2 = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv[mb]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(value, batch["ret"][mb])
                ent_mean = ent.mean()
                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_mean

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = (batch["logp_old"][mb] - log_p).mean().item()
                    clipfrac = ((ratio - 1).abs() > cfg.clip_coef).float().mean().item()
                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["ent"] += ent_mean.item()
                stats["kl"] += kl
                stats["clipfrac"] += clipfrac
                nu += 1
        for k in stats:
            stats[k] /= max(nu, 1)
        return stats

    def iterate(self):
        batch = self.collect_rollout()
        upd = self.update(batch)
        return {
            "reward_mean": batch["reward_mean"],
            "reward_per_env": batch["reward_sum_per_env"],
            **upd,
        }
