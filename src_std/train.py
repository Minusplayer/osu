"""PPO training entrypoint for osu! standard.

Stage-0 is single-map overfit, for verifying the pipeline. The curriculum
expansion to a map pool lives in curriculum.py (driven by the same trainer).
"""

import argparse
import os
import sys
import time
from pathlib import Path

# NixOS: Triton's libcuda_dirs() shells out to /sbin/ldconfig which doesn't
# exist. Setting this env var (read first by Triton) bypasses that probe.
os.environ.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src_std.model_std import ObjectTokenPolicy
from src_std.parse_std import parse_beatmap_std
from src_std.ppo import PPOConfig, PPOTrainer
from src_std.sim import OsuStdEnv, build_map_tensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map", help="path to a .osu file (mode-0)")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch-envs", type=int, default=32)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--dt-ms", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--save", default="checkpoints/std.pt")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"parsing map: {args.map}")
    bm = parse_beatmap_std(args.map)
    if bm.mode != 0:
        sys.exit(f"map mode={bm.mode}, expected 0 (std)")
    print(f"  title: {bm.title}")
    print(f"  CS={bm.cs} AR={bm.ar} OD={bm.od}")
    print(f"  objects: {len(bm.notes)}")

    mp = build_map_tensors(bm, device=device)
    env = OsuStdEnv(mp, batch_size=args.batch_envs, dt_ms=args.dt_ms, device=device)
    print(f"  episode ticks: {env.n_ticks} ({env.t0_ms}..{env.t_end_ms} ms)")

    model_kwargs = dict(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dim_ff=args.dim_ff, dropout=args.dropout,
    )
    model = ObjectTokenPolicy(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f} M params")

    cfg = PPOConfig(
        rollout_steps=args.rollout_steps, n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_coef=args.clip, value_coef=args.vf_coef, entropy_coef=args.ent_coef,
        lr=args.lr,
    )
    trainer = PPOTrainer(model, env, cfg)

    best_reward = float("-inf")
    t0 = time.time()
    for it in range(1, args.iters + 1):
        stats = trainer.iterate()
        rpe = stats["reward_per_env"]
        if it % args.log_every == 0:
            print(f"iter {it:5d}  R/env={rpe:+8.2f}  "
                  f"pg={stats['pg_loss']:+.3f} v={stats['v_loss']:.3f} "
                  f"ent={stats['ent']:.3f} kl={stats['kl']:+.4f} "
                  f"clip={stats['clipfrac']*100:5.1f}%  "
                  f"t={time.time()-t0:.0f}s", flush=True)
        save_now = (it % args.save_every == 0) or rpe > best_reward
        if rpe > best_reward:
            best_reward = rpe
        if save_now:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "model_kwargs": model_kwargs,
                "cfg": cfg.__dict__,
                "iter": it,
                "reward_per_env": rpe,
                "best_reward": best_reward,
                "map_path": str(Path(args.map).resolve()),
            }, args.save)

    print(f"\ndone. best reward/env = {best_reward:.2f}")
    print(f"checkpoint -> {args.save}")


if __name__ == "__main__":
    main()
