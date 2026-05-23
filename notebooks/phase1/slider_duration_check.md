# Phase 1: Slider Duration Spot-Check

Date: 2026-05-23

Goal: verify `src_std/parse_std.py`'s `_slider_duration_ms()` + `_sv_at()` against
manual calculation across varying timing contexts. Slider geometry was visually
verified earlier in Phase 1; this fills the gap on duration math.

## Formula

```
velocity_px_per_beat = 100 × slider_multiplier_global × sv_mult_inherited
beats_per_pass       = length / velocity_px_per_beat
duration_ms          = beats_per_pass × beat_length × repeats
```

Parser rounds via `int(round(...))`, so sub-1 ms diffs vs. float manual calc are
expected and correct.

## Samples

### 1. Of Our Time — first slider, inherited SV = 0.6×
Map hash: `ddfb49...102a9be`

| field | value |
|---|---|
| t_start | 270 ms |
| length | 60.00 px |
| repeats | 1 |
| active beat_length | 294.1176 ms (~204 BPM) |
| sv_mult_inh | 0.6000 |
| slider_multiplier (global) | 2.0 |

```
manual = 60 / (2.0 × 100 × 0.6) × 294.1176 × 1 = 147.0588 ms
parser = 147 ms
diff   = −0.06 ms  → MATCH (rounding)
```

### 2. Of Our Time — mid-song slider, inherited SV = 0.7×
Same map.

| field | value |
|---|---|
| t_start | 5858 ms |
| length | 105.00 px |
| repeats | 1 |
| active beat_length | 294.1176 ms |
| sv_mult_inh | 0.7000 |
| slider_multiplier (global) | 2.0 |

```
manual = 105 / (2.0 × 100 × 0.7) × 294.1176 × 1 = 220.5882 ms
parser = 221 ms
diff   = +0.41 ms  → MATCH (rounding)
```

### 3. Vespera Stella — slider AFTER mid-map BPM change
Map hash: `4b9f68...7faca3`. This map has 3 uninherited timing points
(175 → 160 → 150 BPM). The slider below lives in the 160 BPM section, so
`_sv_at()` must track through two uninherited transitions and pick up
the new `beat_length`.

| field | value |
|---|---|
| t_start | 118258 ms |
| length | 190.00 px |
| repeats | 1 |
| active beat_length | 375.0000 ms (= 160 BPM, post-change) |
| sv_mult_inh | 2.0000 (uninh reset SV, then a 2× inh fired) |
| slider_multiplier (global) | 1.9 |

```
manual = 190 / (1.9 × 100 × 2.0) × 375.0 × 1 = 187.5000 ms
parser = 188 ms
diff   = +0.50 ms  → MATCH (rounding)
```

## Verdict

All 3 sliders match manual calculation within sub-1 ms rounding. Coverage:

- Non-unity inherited SV (0.6×, 0.7×, 2.0×) — sv lookup correct.
- Mid-map uninherited BPM change (175 → 160) — `_sv_at` correctly switches
  `beat_length` after crossing the uninherited timing point boundary.
- Different global `slider_multiplier` values (2.0 and 1.9) correctly applied.
- Single-repeat sliders only; multi-repeat math is `× repeats` and extends trivially.

Not tested:
- `repeats > 1` directly.
- Inherited points that change SV *during* a slider's lifetime (osu! semantics:
  only the SV at slider start applies — parser looks up at `t = n.t_start`,
  which is correct behavior).
