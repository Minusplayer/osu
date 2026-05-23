"""Headless osu!std env, batched in torch.

One OsuStdEnv instance simulates B parallel rollouts of the same map.
The training driver advances all envs in lockstep at `dt_ms` resolution.

Action per tick (per env):
  dxy : float[B, 2]   in [-1, 1], scaled to cursor px/tick
  clk : bool[B]       sampled click state

Observation per tick:
  obj_feat:    float[B, K, 11]  next K=32 objects, each with
    [dx, dy, dt_norm, type_circ, type_slid, type_spin,
     new_combo, slider_active, slider_dx, slider_dy, slider_progress]
  cursor_feat: float[B, 6]
    [x/W, y/H, vx, vy, click_state, ticks_since_click/250]

Reward shaping is dense (per-tick aim distance + slider follow + hit/miss).
"""

import math
from dataclasses import dataclass

import torch

PLAYFIELD_W = 512.0
PLAYFIELD_H = 384.0
MAX_DIST = math.hypot(PLAYFIELD_W, PLAYFIELD_H)

K_TOKENS = 32
OBJ_FEATURES = 11
CURSOR_FEATURES = 6

# action speed: tanh action [-1,1] → cursor delta in px per tick.
# At 4 ms ticks, MAX_DXY_PX=2.5 → 625 px/s, faster than the playfield diagonal.
MAX_DXY_PX = 2.5


@dataclass
class MapTensors:
    """Pre-baked tensors describing one map. Shared (read-only) across the batch."""
    pos: torch.Tensor              # (N, 2) px
    t_start: torch.Tensor          # (N,) ms
    t_end: torch.Tensor            # (N,) ms
    type: torch.Tensor             # (N,) long: 0=circ, 1=slid, 2=spin
    new_combo: torch.Tensor        # (N,) bool
    slider_curve: torch.Tensor     # (N, SLIDER_SAMPLES, 2) px (zeros for non-sliders)
    slider_length: torch.Tensor    # (N,) px (0 for non-sliders)
    cs: float
    ar: float
    od: float
    n_objects: int

    def to(self, device):
        return MapTensors(
            pos=self.pos.to(device),
            t_start=self.t_start.to(device),
            t_end=self.t_end.to(device),
            type=self.type.to(device),
            new_combo=self.new_combo.to(device),
            slider_curve=self.slider_curve.to(device),
            slider_length=self.slider_length.to(device),
            cs=self.cs, ar=self.ar, od=self.od,
            n_objects=self.n_objects,
        )


def build_map_tensors(beatmap, device=torch.device("cpu")) -> MapTensors:
    """Convert parse_std.Beatmap into batched-ready tensors."""
    from .parse_std import SLIDER_SAMPLES

    n = len(beatmap.notes)
    pos = torch.zeros(n, 2)
    t_start = torch.zeros(n, dtype=torch.long)
    t_end = torch.zeros(n, dtype=torch.long)
    typ = torch.zeros(n, dtype=torch.long)
    nc = torch.zeros(n, dtype=torch.bool)
    curve = torch.zeros(n, SLIDER_SAMPLES, 2)
    slen = torch.zeros(n)

    for i, ho in enumerate(beatmap.notes):
        pos[i, 0] = ho.x
        pos[i, 1] = ho.y
        t_start[i] = ho.t_start
        t_end[i] = ho.t_end
        typ[i] = ho.type
        nc[i] = ho.new_combo
        if ho.type == 1 and ho.slider_curve:
            for j, (cx, cy) in enumerate(ho.slider_curve[:SLIDER_SAMPLES]):
                curve[i, j, 0] = cx
                curve[i, j, 1] = cy
            slen[i] = ho.slider_length

    return MapTensors(
        pos=pos.to(device), t_start=t_start.to(device), t_end=t_end.to(device),
        type=typ.to(device), new_combo=nc.to(device),
        slider_curve=curve.to(device), slider_length=slen.to(device),
        cs=beatmap.cs, ar=beatmap.ar, od=beatmap.od, n_objects=n,
    )


class OsuStdEnv:
    """Batched osu!std env. One instance, B parallel rollouts of the same map."""

    def __init__(self, mp: MapTensors, batch_size: int = 32, dt_ms: int = 4,
                 device=torch.device("cpu")):
        self.mp = mp.to(device)
        self.B = batch_size
        self.dt_ms = dt_ms
        self.device = device

        from .parse_std import (circle_radius_px, follow_radius_px,
                                hit_windows_ms, preempt_ms)
        self.circle_r = circle_radius_px(mp.cs)
        self.follow_r = follow_radius_px(mp.cs)
        hw = hit_windows_ms(mp.od)
        self.hw_300 = hw[300]
        self.hw_100 = hw[100]
        self.hw_50 = hw[50]
        self.preempt_ms = preempt_ms(mp.ar)
        self.miss_window_ms = self.hw_50

        self.t0_ms = int(mp.t_start[0].item() - self.preempt_ms)
        self.t_end_ms = int(mp.t_end[-1].item() + 1000)
        self.n_ticks = (self.t_end_ms - self.t0_ms) // dt_ms + 1

        self.reset()

    @torch.no_grad()
    def reset(self):
        B, dev = self.B, self.device
        self.cursor = torch.tensor([PLAYFIELD_W / 2, PLAYFIELD_H / 2], device=dev
                                   ).repeat(B, 1).float()
        self.vel = torch.zeros(B, 2, device=dev)
        self.click = torch.zeros(B, dtype=torch.bool, device=dev)
        self.prev_click = torch.zeros(B, dtype=torch.bool, device=dev)
        self.ticks_since_click = torch.full((B,), 1000, dtype=torch.long, device=dev)

        self.tick = 0
        self.score = torch.zeros(B, device=dev)
        self.combo = torch.zeros(B, dtype=torch.long, device=dev)
        self.hits = torch.zeros(B, dtype=torch.long, device=dev)
        self.misses = torch.zeros(B, dtype=torch.long, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

        self.obj_status = torch.zeros(self.B, self.mp.n_objects,
                                      dtype=torch.uint8, device=dev)
        self.next_obj = torch.zeros(B, dtype=torch.long, device=dev)
        self.active_slider = torch.full((B,), -1, dtype=torch.long, device=dev)

        return self._observe()

    @property
    def t_ms(self):
        return self.t0_ms + self.tick * self.dt_ms

    @torch.no_grad()
    def step(self, dxy: torch.Tensor, clk: torch.Tensor):
        """dxy: (B,2) in [-1,1]; clk: (B,) bool.

        Returns: (obs_objects, obs_cursor, reward, done, info)
        """
        B = self.B

        delta = dxy.clamp(-1.0, 1.0) * MAX_DXY_PX
        self.vel = delta
        self.cursor = self.cursor + delta
        self.cursor[:, 0].clamp_(0.0, PLAYFIELD_W)
        self.cursor[:, 1].clamp_(0.0, PLAYFIELD_H)

        self.prev_click = self.click.clone()
        self.click = clk.bool()
        click_edge = self.click & (~self.prev_click)
        self.ticks_since_click = torch.where(click_edge,
                                             torch.zeros_like(self.ticks_since_click),
                                             self.ticks_since_click + 1)

        t = self.t_ms
        reward = torch.zeros(B, device=self.device)

        # ---- aim shaping ----
        next_idx = self.next_obj.clamp(max=self.mp.n_objects - 1)
        next_pos = self.mp.pos[next_idx]
        dist = torch.linalg.norm(self.cursor - next_pos, dim=-1)
        active = self.next_obj < self.mp.n_objects
        reward = reward - 0.005 * (dist / MAX_DIST) * active.float()

        # ---- click resolution ----
        obj_t = torch.where(active, self.mp.t_start.gather(0, next_idx),
                            torch.full_like(next_idx, 10**9, dtype=torch.long))
        in_window = (t - obj_t).abs() <= int(self.hw_50)
        obj_typ = self.mp.type.gather(0, next_idx)
        is_circ_or_slid = (obj_typ == 0) | (obj_typ == 1)
        attempt = click_edge & active & is_circ_or_slid & in_window
        within = dist <= self.circle_r
        success = attempt & within
        err = (t - obj_t).abs().float()
        v300 = (err <= self.hw_300).float() * 1.0
        v100 = ((err <= self.hw_100) & (err > self.hw_300)).float() * 0.75
        v50 = ((err <= self.hw_50) & (err > self.hw_100)).float() * 0.5
        hit_value = v300 + v100 + v50
        reward = reward + success.float() * hit_value

        if success.any():
            idx_succ = success.nonzero(as_tuple=True)[0]
            obj_to_mark = next_idx[idx_succ]
            self.obj_status[idx_succ, obj_to_mark] = 1
            self.hits[idx_succ] += 1
            self.combo[idx_succ] += 1
            self.score[idx_succ] += hit_value[idx_succ]
            slid_mask = (self.mp.type[obj_to_mark] == 1)
            slid_idx = idx_succ[slid_mask]
            if slid_idx.numel() > 0:
                self.active_slider[slid_idx] = obj_to_mark[slid_mask]
            circ_mask = (self.mp.type[obj_to_mark] == 0)
            circ_idx = idx_succ[circ_mask]
            if circ_idx.numel() > 0:
                self.next_obj[circ_idx] += 1

        bad_click = click_edge & ~success & (self.active_slider < 0)
        reward = reward - 0.05 * bad_click.float()

        # ---- miss ----
        too_late = active & ((t - obj_t) > int(self.miss_window_ms))
        cur_status = self.obj_status.gather(1, next_idx.unsqueeze(1)).squeeze(1)
        miss_now = too_late & is_circ_or_slid & (cur_status == 0)
        if miss_now.any():
            idx_m = miss_now.nonzero(as_tuple=True)[0]
            self.obj_status[idx_m, next_idx[idx_m]] = 2
            self.misses[idx_m] += 1
            self.combo[idx_m] = 0
            reward[idx_m] -= 1.0
            self.next_obj[idx_m] += 1

        # ---- slider follow ----
        sliding = self.active_slider >= 0
        if sliding.any():
            sli_idx = sliding.nonzero(as_tuple=True)[0]
            obj_i = self.active_slider[sli_idx]
            s_t0 = self.mp.t_start[obj_i].float()
            s_t1 = self.mp.t_end[obj_i].float()
            prog = ((t - s_t0) / (s_t1 - s_t0).clamp(min=1.0)).clamp(0.0, 1.0)
            n_samp = self.mp.slider_curve.size(1)
            sample_i = (prog * (n_samp - 1)).long().clamp(0, n_samp - 1)
            tgt = self.mp.slider_curve[obj_i, sample_i]
            d = torch.linalg.norm(self.cursor[sli_idx] - tgt, dim=-1)
            inside = d <= self.follow_r
            r = torch.where(inside,
                            torch.full_like(d, 0.10),
                            torch.full_like(d, -0.10))
            reward[sli_idx] += r
            done_slide = t >= self.mp.t_end[obj_i]
            if done_slide.any():
                end_idx = sli_idx[done_slide]
                end_obj = self.active_slider[end_idx]
                still_in = inside[done_slide]
                bonus = still_in.float() * 0.5
                self.score[end_idx] += bonus
                reward[end_idx] += bonus
                self.active_slider[end_idx] = -1
                self.next_obj[end_idx] = end_obj + 1

        # ---- spinner ----
        next_is_spin = active & (obj_typ == 2)
        obj_t_end = self.mp.t_end.gather(0, next_idx)
        spin_active = next_is_spin & (t >= obj_t) & (t <= obj_t_end)
        if spin_active.any():
            sp_idx = spin_active.nonzero(as_tuple=True)[0]
            speed = torch.linalg.norm(self.vel[sp_idx], dim=-1)
            r = (0.05 * (speed / 1.25)).clamp(max=0.05)
            reward[sp_idx] += r
        spin_end = next_is_spin & (t > obj_t_end)
        if spin_end.any():
            end_idx = spin_end.nonzero(as_tuple=True)[0]
            self.obj_status[end_idx, next_idx[end_idx]] = 1
            self.next_obj[end_idx] += 1
            self.hits[end_idx] += 1

        # ---- advance ----
        self.tick += 1
        if self.t_ms >= self.t_end_ms or self.next_obj.min().item() >= self.mp.n_objects:
            self.done = torch.ones_like(self.done)

        obs_obj, obs_cur = self._observe()
        info = {
            "score": self.score.clone(),
            "combo": self.combo.clone(),
            "hits": self.hits.clone(),
            "misses": self.misses.clone(),
            "fc": (self.misses == 0) & (self.next_obj >= self.mp.n_objects),
        }
        return obs_obj, obs_cur, reward, self.done.clone(), info

    @torch.no_grad()
    def _observe(self):
        B, K = self.B, K_TOKENS
        N = self.mp.n_objects
        dev = self.device

        offsets = torch.arange(K, device=dev).unsqueeze(0)
        idx = (self.next_obj.unsqueeze(1) + offsets).clamp(max=N - 1)
        valid = (self.next_obj.unsqueeze(1) + offsets) < N

        pos = self.mp.pos[idx]
        t_s = self.mp.t_start[idx].float()
        typ = self.mp.type[idx]
        nc = self.mp.new_combo[idx].float()

        dx = (pos[..., 0] - self.cursor[:, 0].unsqueeze(1)) / PLAYFIELD_W
        dy = (pos[..., 1] - self.cursor[:, 1].unsqueeze(1)) / PLAYFIELD_H
        dt_norm = (t_s - self.t_ms) / self.preempt_ms
        type_c = (typ == 0).float()
        type_s = (typ == 1).float()
        type_sp = (typ == 2).float()

        sliding = self.active_slider >= 0
        slider_dx = torch.zeros(B, K, device=dev)
        slider_dy = torch.zeros(B, K, device=dev)
        slider_prog = torch.zeros(B, K, device=dev)
        slider_act = torch.zeros(B, K, device=dev)
        if sliding.any():
            sli = sliding.nonzero(as_tuple=True)[0]
            obj_i = self.active_slider[sli]
            s_t0 = self.mp.t_start[obj_i].float()
            s_t1 = self.mp.t_end[obj_i].float()
            prog = ((self.t_ms - s_t0) / (s_t1 - s_t0).clamp(min=1.0)).clamp(0.0, 1.0)
            n_samp = self.mp.slider_curve.size(1)
            sample_i = (prog * (n_samp - 1)).long().clamp(0, n_samp - 1)
            tgt = self.mp.slider_curve[obj_i, sample_i]
            slider_dx[sli, 0] = (tgt[:, 0] - self.cursor[sli, 0]) / PLAYFIELD_W
            slider_dy[sli, 0] = (tgt[:, 1] - self.cursor[sli, 1]) / PLAYFIELD_H
            slider_prog[sli, 0] = prog
            slider_act[sli, 0] = 1.0

        obj_feat = torch.stack([
            dx, dy, dt_norm, type_c, type_s, type_sp, nc,
            slider_act, slider_dx, slider_dy, slider_prog,
        ], dim=-1)
        obj_feat = obj_feat * valid.unsqueeze(-1).float()

        cursor_feat = torch.stack([
            self.cursor[:, 0] / PLAYFIELD_W,
            self.cursor[:, 1] / PLAYFIELD_H,
            self.vel[:, 0] / MAX_DXY_PX,
            self.vel[:, 1] / MAX_DXY_PX,
            self.click.float(),
            (self.ticks_since_click.float() / 250.0).clamp(max=1.0),
        ], dim=-1)

        return obj_feat, cursor_feat


def _perfect_run(map_path: str):
    """Drive a perfect (teleport-aim) player; print final score for sanity."""
    from .parse_std import parse_beatmap_std
    bm = parse_beatmap_std(map_path)
    mp = build_map_tensors(bm)
    env = OsuStdEnv(mp, batch_size=1, dt_ms=4)

    total_r = 0.0
    while not env.done.any():
        idx = env.next_obj.clamp(max=mp.n_objects - 1)
        target = env.mp.pos[idx]
        d = (target - env.cursor) / MAX_DXY_PX
        d = d.clamp(-1.0, 1.0)
        t_obj = env.mp.t_start[idx].item()
        click = torch.tensor([abs(env.t_ms - t_obj) <= env.hw_300], dtype=torch.bool)
        _, _, r, _, _ = env.step(d, click)
        total_r += r.item()
    print(f"perfect run: hits={env.hits.item()}/{mp.n_objects}, "
          f"misses={env.misses.item()}, total_reward={total_r:.2f}, "
          f"score={env.score.item():.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src_std.sim <map.osu>")
        sys.exit(1)
    _perfect_run(sys.argv[1])
