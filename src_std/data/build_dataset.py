"""CLI: build OsuStdDataset over all session .npz files and dump build stats.

Usage:
    python -m src_std.data.build_dataset [--sessions-dir data/sessions]
                                         [--out data/dataset_stats.json]
                                         [--label-window-ms 60]
                                         [--gap-ms 2000]

This does not materialize a flat tensor file; it instantiates the in-memory
dataset (so it ALSO validates that every session is loadable + parses
without error) and writes a stats report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src_std.data.dataset import (  # noqa: E402
    OsuStdDataset, LABEL_WINDOW_MS, GAP_MS, STACK_LEN,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", type=Path, default=Path("data/sessions"))
    ap.add_argument("--out", type=Path, default=Path("data/dataset_stats.json"))
    ap.add_argument("--label-window-ms", type=int, default=LABEL_WINDOW_MS)
    ap.add_argument("--gap-ms", type=int, default=GAP_MS)
    args = ap.parse_args()

    paths = sorted(args.sessions_dir.glob("*.npz"))
    if not paths:
        print(f"no .npz under {args.sessions_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"building dataset over {len(paths)} session(s)...")
    ds = OsuStdDataset(
        paths,
        label_window_ms=args.label_window_ms,
        gap_ms=args.gap_ms,
    )

    per_session = []
    for p, s in zip(paths, ds.stats()):
        s = {"file": p.name, **s}
        per_session.append(s)
        rate = s["press_positive_rate"]
        rate_str = "N/A" if rate is None else f"{rate*100:5.1f}%"
        print(f"  {p.name}")
        print(f"    map: {s['map_title']!r}  (id {s['map_id']})  "
              f"dur={s['duration_s']:.1f}s  segs={s['n_segments']}")
        print(f"    frames: total={s['total_frames']}  paused={s['dropped_paused']}  "
              f"neg_t={s['dropped_negative_t']}  short={s['dropped_seg_too_short']}  "
              f"warmup={s['dropped_stack_warmup']}  kept={s['kept_frames']}")
        print(f"    samples: {s['samples']}  press_rate={rate_str}")

    agg = ds.aggregate_stats()
    print()
    print("AGGREGATE:")
    print(f"  sessions={agg['n_sessions']}  total_frames={agg['total_frames']}  "
          f"kept={agg['kept_frames']}  samples={agg['samples']}")
    rate = agg["press_positive_rate"]
    rate_str = "N/A" if rate is None else f"{rate*100:.1f}%"
    print(f"  press_positive_rate={rate_str}  segments={agg['n_segments']}")
    print(f"  dropped: paused={agg['dropped_paused']} neg_t={agg['dropped_negative_t']} "
          f"short={agg['dropped_seg_too_short']} warmup={agg['dropped_stack_warmup']}")

    report = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "sessions_dir": str(args.sessions_dir),
        "knobs": {
            "label_window_ms": args.label_window_ms,
            "gap_ms": args.gap_ms,
            "stack_len": STACK_LEN,
        },
        "n_samples": len(ds),
        "per_session": per_session,
        "aggregate": agg,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
