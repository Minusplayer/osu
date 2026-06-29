# ai-osu
Very very action and sloppily written code. I wrote half of it then Also had claude do stuff since I was lazy. 
the actual modern version is cleaner and less laggy

Vision-based osu!std bot. Trains a neural net to play osu! Standard from screen pixels + beatmap context, with input injected via X11 / evdev.

**Status:** active research. Phase 4 dataset built, behavior-cloning training works end-to-end. Phase 7 (live inference) in progress. See `notebooks/` for phase artifacts.

There is also a legacy osu!mania pipeline in `src/` — kept as a reference for the training loop. The active code is in `src_std/`.

## Layout

```
src_std/             active osu!std code
  parse_std.py         .osu beatmap parser (Phase 1, verified)
  data/                dataset build + visualization (Phase 4)
  capture/             screen + input capture (X11, tosu websocket)
  model_bc.py          behavior-cloning model
  train_bc.py          BC training loop
  eval_replay.py       replay-based eval
  model_std.py         (scaffold) RL model
  ppo.py / sim.py      (scaffold) PPO + simulator for self-play
src/                 legacy osu!mania pipeline (reference only)
configs/             training configs
tests/               pytest suite
notebooks/           per-phase outputs (alignment plots, eval reports)
data/                gitignored — beatmaps, replays, captures
checkpoints/, runs/  gitignored — model artifacts + TensorBoard logs
```

## Setup

Requires NixOS (or a system with the same package set) and an NVIDIA GPU for training.

```bash
nix-shell                # provisions Python 3.12 + creates .venv with torch + osrparse
cp .env.example .env     # then fill in your osu! OAuth credentials
```

The `shell.nix` overlay installs `torch` and `osrparse` via pip because the current nixpkgs `python312Packages.torch` has a broken eval. `LD_LIBRARY_PATH` is set so the pip torch wheel finds CUDA via `/run/opengl-driver/lib`.

If torch import fails after a nixpkgs update: `rm -rf .venv && nix-shell`.

## Data acquisition

Beatmaps and replays are not redistributed. Fetch them yourself:

```bash
# requires OSU_CLIENT_ID / OSU_CLIENT_SECRET in .env
./.venv/bin/python src/fetch_replays.py
```

Live capture during training/inference uses [tosu](https://github.com/KotRikD/tosu) for game state (websocket on `localhost:24050`) and `grim` / X11 for the playfield. The capture region is hardcoded to `316,60 1280x960` (4:3 corner-calibrated) — adjust in `src_std/capture/` for your resolution.

## Running

Call modules via the venv python so PYTHONPATH resolves correctly:

```bash
./.venv/bin/python -m src_std.train_bc --config configs/training_bc.json
./.venv/bin/python -m src_std.eval_replay <replay.osr>
./.venv/bin/python -m pytest tests/
```

## Notes

- Research code. Interfaces are unstable; phases get rewritten.
- Use at your own risk — running input-injection bots against ranked osu! servers will get your account restricted. Intended for local research, offline replay eval, and unranked practice.
- osu! beatmap and replay data belong to their authors and to ppy. Don't redistribute fetched data.

## License

MIT — see [LICENSE](LICENSE).
