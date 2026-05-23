"""osu! standard (mode 0) .osu parser.

Reads circles / sliders / spinners from a .osu chart and resamples slider
curves into fixed-length point arrays the sim can consume directly.

Curve types supported: L (linear), P (perfect circle), B (bezier),
C (catmull, treated as bezier — rare in modern maps).

Coordinate space: osu! playfield is 512x384 px; this module returns raw
pixel coords. The sim normalizes to [0,1] downstream.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

CIRCLE, SLIDER, SPINNER = 0, 1, 2

SLIDER_SAMPLES = 32
PLAYFIELD_W = 512
PLAYFIELD_H = 384


@dataclass
class TimingPoint:
    t_ms: int
    beat_length: float          # ms per beat (uninherited) or -slider_velocity_pct (inherited)
    uninherited: bool


@dataclass
class HitObject:
    x: float
    y: float
    t_start: int
    t_end: int
    type: int                   # CIRCLE / SLIDER / SPINNER
    new_combo: bool
    slider_curve: list = field(default_factory=list)   # SLIDER_SAMPLES (x, y) points along the path
    slider_repeats: int = 1
    slider_length: float = 0.0  # pixels


@dataclass
class Beatmap:
    title: str
    mode: int
    cs: float
    ar: float
    od: float
    hp: float
    slider_multiplier: float
    slider_tick_rate: float
    timing_points: list
    notes: list


# ---------------- slider curve math ----------------

def _linear(points):
    return list(points)


def _bezier_segment(ctrl, n_samples):
    """Evaluate a bezier curve with given control points at n_samples uniform t."""
    out = []
    m = len(ctrl) - 1
    for i in range(n_samples):
        t = i / max(1, n_samples - 1)
        # de Casteljau
        pts = [list(p) for p in ctrl]
        for r in range(1, m + 1):
            for j in range(m - r + 1):
                pts[j][0] = (1 - t) * pts[j][0] + t * pts[j + 1][0]
                pts[j][1] = (1 - t) * pts[j][1] + t * pts[j + 1][1]
        out.append((pts[0][0], pts[0][1]))
    return out


def _bezier_path(points, n_samples):
    """osu! splits bezier control points at repeated anchors into sub-segments."""
    segments = []
    seg = [points[0]]
    for i in range(1, len(points)):
        if points[i] == points[i - 1]:
            if len(seg) >= 2:
                segments.append(seg)
            seg = [points[i]]
        else:
            seg.append(points[i])
    if len(seg) >= 2:
        segments.append(seg)
    if not segments:
        return [points[0]] * n_samples
    # sample each segment proportional to its straight-line length
    seg_lens = []
    for s in segments:
        L = 0.0
        for j in range(1, len(s)):
            L += math.hypot(s[j][0] - s[j - 1][0], s[j][1] - s[j - 1][1])
        seg_lens.append(max(L, 1e-6))
    total = sum(seg_lens)
    pts = []
    remaining = n_samples
    for i, s in enumerate(segments):
        if i == len(segments) - 1:
            k = remaining
        else:
            k = max(1, int(round(n_samples * seg_lens[i] / total)))
            k = min(k, remaining - (len(segments) - i - 1))
        pts.extend(_bezier_segment(s, k))
        remaining -= k
    return pts[:n_samples]


def _perfect_circle(p0, p1, p2, n_samples):
    """Arc through three non-collinear points."""
    ax, ay = p0; bx, by = p1; cx, cy = p2
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return _linear([p0, p2])  # degenerate → straight line
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    cx0, cy0 = ux, uy
    r = math.hypot(ax - cx0, ay - cy0)
    a0 = math.atan2(ay - cy0, ax - cx0)
    a1 = math.atan2(by - cy0, bx - cx0)
    a2 = math.atan2(cy - cy0, cx - cx0)

    def norm(a):
        while a - a0 > math.pi: a -= 2 * math.pi
        while a - a0 < -math.pi: a += 2 * math.pi
        return a
    a1n, a2n = norm(a1), norm(a2)
    # ensure a1 lies between a0 and a2 (choose sweep direction through middle anchor)
    if not (min(a0, a2n) <= a1n <= max(a0, a2n)):
        a2n += -2 * math.pi if a2n > a0 else 2 * math.pi
    out = []
    for i in range(n_samples):
        t = i / max(1, n_samples - 1)
        a = a0 + (a2n - a0) * t
        out.append((cx0 + r * math.cos(a), cy0 + r * math.sin(a)))
    return out


def _resample_to_length(pts, target_length, n_samples):
    """Trim/extend a polyline to target_length px, then resample uniformly."""
    if len(pts) < 2:
        return [pts[0] if pts else (0.0, 0.0)] * n_samples
    seg_lens = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                for i in range(len(pts) - 1)]
    total = sum(seg_lens)
    if total < 1e-6:
        return [pts[0]] * n_samples
    # cumulative arc-length
    cum = [0.0]
    for L in seg_lens:
        cum.append(cum[-1] + L)
    L_target = min(target_length, total) if target_length > 0 else total
    out = []
    for i in range(n_samples):
        s = L_target * i / max(1, n_samples - 1)
        # find segment containing s
        lo, hi = 0, len(cum) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        seg_L = cum[lo + 1] - cum[lo]
        u = (s - cum[lo]) / seg_L if seg_L > 1e-6 else 0.0
        x = pts[lo][0] + u * (pts[lo + 1][0] - pts[lo][0])
        y = pts[lo][1] + u * (pts[lo + 1][1] - pts[lo][1])
        out.append((x, y))
    return out


def slider_path(start_x, start_y, curve_type, anchors, length, _src_path=None):
    """Return SLIDER_SAMPLES (x,y) points along the slider, trimmed to `length` px."""
    if curve_type == "C":
        raise NotImplementedError(
            f"Catmull (C) slider curve encountered in {_src_path or '<unknown>'}. "
            f"Catmull is deprecated and not implemented to avoid silently producing "
            f"wrong paths (previous code fell through to bezier math). Implement "
            f"properly if real maps in the dataset use it."
        )
    ctrl = [(start_x, start_y)] + anchors
    dense_n = max(SLIDER_SAMPLES * 4, 128)
    if curve_type == "L":
        raw = []
        for i in range(len(ctrl) - 1):
            for j in range(dense_n // max(1, len(ctrl) - 1)):
                t = j / max(1, dense_n // max(1, len(ctrl) - 1))
                x = ctrl[i][0] + t * (ctrl[i + 1][0] - ctrl[i][0])
                y = ctrl[i][1] + t * (ctrl[i + 1][1] - ctrl[i][1])
                raw.append((x, y))
        raw.append(ctrl[-1])
    elif curve_type == "P" and len(ctrl) == 3:
        raw = _perfect_circle(ctrl[0], ctrl[1], ctrl[2], dense_n)
    else:   # B (bezier), or P with !=3 anchors (degenerate → bezier fallback)
        raw = _bezier_path(ctrl, dense_n)
    return _resample_to_length(raw, length, SLIDER_SAMPLES)


# ---------------- difficulty helpers ----------------

def circle_radius_px(cs: float) -> float:
    """osu!std circle radius in playfield pixels."""
    return 54.4 - 4.48 * cs


def follow_radius_px(cs: float) -> float:
    """Slider follow-circle radius (~2.4x circle radius during sliding)."""
    return circle_radius_px(cs) * 2.4


def hit_windows_ms(od: float) -> dict:
    """Return {300, 100, 50} hit windows in ms (half-width)."""
    return {
        300: 80 - 6 * od,
        100: 140 - 8 * od,
        50: 200 - 10 * od,
    }


def preempt_ms(ar: float) -> float:
    """How far before t_start the object becomes visible."""
    if ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    if ar > 5:
        return 1200 - 750 * (ar - 5) / 5
    return 1200


# ---------------- parser ----------------

def _slider_duration_ms(length, beat_length_uninherited, sv_mult_inherited,
                       slider_multiplier, repeats):
    """One repeat duration = length / (slider_multiplier * 100 * sv_mult_inherited) beats."""
    velocity = 100.0 * slider_multiplier * sv_mult_inherited
    one_pass_beats = length / velocity
    return int(round(one_pass_beats * beat_length_uninherited * repeats))


def _sv_at(timing_points, t_ms):
    """Returns (uninherited_beat_length, inherited_slider_velocity_multiplier) active at t_ms."""
    last_uninh = 500.0  # default 120 BPM if map starts before first uninh point
    last_inh_mult = 1.0
    for tp in timing_points:
        if tp.t_ms > t_ms:
            break
        if tp.uninherited:
            last_uninh = tp.beat_length
            last_inh_mult = 1.0   # uninherited point resets SV
        else:
            # beat_length is -100 / sv_mult_pct  →  sv_mult = -100 / beat_length
            last_inh_mult = -100.0 / tp.beat_length if tp.beat_length < 0 else 1.0
    return last_uninh, last_inh_mult


def parse_beatmap_std(path) -> Beatmap:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    section = None
    title = ""
    mode = 0
    cs = ar = od = hp = 5.0
    sm = 1.4
    str_ = 1.0
    timing_points = []
    notes = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "General":
            if line.startswith("Mode:"):
                mode = int(line.split(":", 1)[1].strip())
        elif section == "Metadata":
            if line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
        elif section == "Difficulty":
            k, _, v = line.partition(":")
            v = v.strip()
            try:
                fv = float(v)
            except ValueError:
                continue
            if k == "CircleSize": cs = fv
            elif k == "ApproachRate": ar = fv
            elif k == "OverallDifficulty": od = fv
            elif k == "HPDrainRate": hp = fv
            elif k == "SliderMultiplier": sm = fv
            elif k == "SliderTickRate": str_ = fv
        elif section == "TimingPoints":
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                t = int(round(float(parts[0])))
                bl = float(parts[1])
            except ValueError:
                continue
            uninh = True if len(parts) < 7 else (parts[6].strip() == "1")
            timing_points.append(TimingPoint(t, bl, uninh))
        elif section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
                t = int(parts[2]); type_bits = int(parts[3])
            except ValueError:
                continue
            new_combo = bool(type_bits & 4)
            if type_bits & 1:               # circle
                notes.append(HitObject(x, y, t, t, CIRCLE, new_combo))
            elif type_bits & 2:             # slider
                if len(parts) < 8:
                    continue
                curve_spec = parts[5]
                try:
                    repeats = int(parts[6])
                    length = float(parts[7])
                except ValueError:
                    continue
                ctype, *anchor_strs = curve_spec.split("|")
                anchors = []
                for a in anchor_strs:
                    if ":" in a:
                        ax, ay = a.split(":")
                        anchors.append((float(ax), float(ay)))
                path_pts = slider_path(x, y, ctype, anchors, length, _src_path=str(path))
                bl, sv_mult = _sv_at(timing_points, t)
                dur = _slider_duration_ms(length, bl, sv_mult, sm, repeats)
                notes.append(HitObject(x, y, t, t + dur, SLIDER, new_combo,
                                       slider_curve=path_pts,
                                       slider_repeats=repeats,
                                       slider_length=length))
            elif type_bits & 8:             # spinner
                if len(parts) < 6:
                    continue
                try:
                    end_t = int(parts[5])
                except ValueError:
                    continue
                notes.append(HitObject(256.0, 192.0, t, end_t, SPINNER, new_combo))

    notes.sort(key=lambda n: n.t_start)
    return Beatmap(title=title, mode=mode, cs=cs, ar=ar, od=od, hp=hp,
                   slider_multiplier=sm, slider_tick_rate=str_,
                   timing_points=timing_points, notes=notes)


def main():
    if len(sys.argv) < 2:
        print("usage: python -m src_std.parse_std <map.osu>")
        sys.exit(1)
    bm = parse_beatmap_std(sys.argv[1])
    if bm.mode != 0:
        print(f"warning: map mode={bm.mode} (std is 0)")
    print(f"title: {bm.title}")
    print(f"CS={bm.cs} AR={bm.ar} OD={bm.od} HP={bm.hp}")
    print(f"circle_r={circle_radius_px(bm.cs):.1f}px  follow_r={follow_radius_px(bm.cs):.1f}px")
    print(f"hit windows (ms): {hit_windows_ms(bm.od)}")
    print(f"preempt: {preempt_ms(bm.ar):.0f} ms")
    n_circ = sum(1 for n in bm.notes if n.type == CIRCLE)
    n_slid = sum(1 for n in bm.notes if n.type == SLIDER)
    n_spin = sum(1 for n in bm.notes if n.type == SPINNER)
    print(f"objects: {len(bm.notes)} ({n_circ} circles, {n_slid} sliders, {n_spin} spinners)")
    if bm.notes:
        first = bm.notes[0]
        last = bm.notes[-1]
        print(f"first @ {first.t_start}ms ({['CIRC','SLID','SPIN'][first.type]})")
        print(f"last  @ {last.t_end}ms")


if __name__ == "__main__":
    main()
