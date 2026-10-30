"""Beat-to-beat interval series (RR for ventricular beats, PP for atrial beats).

Every interval is kept and labelled. An interval is *valid* (usable as a normal-to-normal, NN,
interval for heart rate variability) only when none of these flags applies:

``gap``       the interval spans a dropout, an unanalyzable stretch or a gap between files
``beat``      one of its two beats was marked unreliable by the detector
``range``     outside the physiological range of the species preset
``artifact``  differs from the median of its neighbours by more than the artifact threshold
              (default 20 %), which catches missed beats, extra beats and ectopic beats

Invalid intervals are excluded from HRV, never silently altered. Pauses are reported from all
intervals except those flagged ``gap`` (a pause must not be confused with a telemetry dropout).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

FLAGS = ("gap", "beat", "range", "artifact")


@dataclass
class IntervalSeries:
    kind: str              # "RR" or "PP"
    t: np.ndarray          # time (s from recording start) of the beat that ends the interval
    ms: np.ndarray         # interval length (ms)
    valid: np.ndarray      # bool
    flag: np.ndarray       # str: "" or one of FLAGS

    def __len__(self):
        return len(self.ms)

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    def window(self, t0: float, t1: float) -> "IntervalSeries":
        m = (self.t >= t0) & (self.t < t1)
        return IntervalSeries(self.kind, self.t[m], self.ms[m], self.valid[m], self.flag[m])

    def counts(self) -> dict:
        return {f: int(np.sum(self.flag == f)) for f in FLAGS}


def _overlaps(a: np.ndarray, b: np.ndarray, segs: Sequence[Tuple[float, float]]) -> np.ndarray:
    """True where the interval (a, b) overlaps any segment."""
    out = np.zeros(len(a), bool)
    if not len(segs) or not len(a):
        return out
    s = np.array(sorted(segs), dtype=float)
    # for each interval, the last segment starting before b
    j = np.searchsorted(s[:, 0], b, side="left") - 1
    ok = j >= 0
    # merge overlapping segments' running max end to make the test exact
    run_end = np.maximum.accumulate(s[:, 1])
    out[ok] = run_end[j[ok]] > a[ok]
    return out


def local_median(ms: np.ndarray, usable: np.ndarray, half: int = 5,
                 run: Optional[np.ndarray] = None) -> np.ndarray:
    """Median of up to ``half`` usable neighbours on each side, excluding the interval itself.
    ``run``: integer label of the uninterrupted stretch each interval belongs to; neighbours from
    another stretch (beyond a dropout or a gap between files) are not used."""
    n = len(ms)
    v = np.where(usable, ms, np.nan).astype(float)
    pad = np.concatenate([np.full(half, np.nan), v, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1).copy()
    win[:, half] = np.nan
    if run is not None:
        r = np.concatenate([np.full(half, -1), np.asarray(run), np.full(half, -1)])
        rw = np.lib.stride_tricks.sliding_window_view(r, 2 * half + 1)
        win[rw != np.asarray(run)[:, None]] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=1)
    return med[:n]


def build_intervals(times_s: np.ndarray, *, kind: str = "RR",
                    bad_segments: Sequence[Tuple[float, float]] = (),
                    beat_ok: Optional[np.ndarray] = None,
                    range_ms: Tuple[float, float] = (0.0, np.inf),
                    artifact_threshold: Optional[float] = 0.2,
                    neighbours: int = 5) -> IntervalSeries:
    """Build and label the interval series from sorted beat times (seconds)."""
    t = np.asarray(times_s, dtype=float)
    if t.size < 2:
        e = np.zeros(0)
        return IntervalSeries(kind, e, e, np.zeros(0, bool), np.zeros(0, dtype="<U8"))
    a, b = t[:-1], t[1:]
    ms = (b - a) * 1000.0
    flag = np.full(len(ms), "", dtype="<U8")
    gap = _overlaps(a, b, bad_segments)
    flag[gap] = "gap"
    if beat_ok is not None:
        ok = np.asarray(beat_ok, bool)
        bad_beat = ~(ok[:-1] & ok[1:])
        flag[(flag == "") & bad_beat] = "beat"
    lo, hi = range_ms
    flag[(flag == "") & ((ms < lo) | (ms > hi))] = "range"
    if artifact_threshold:
        usable = flag == ""
        run = np.cumsum(gap)          # a new stretch begins at every interval across a gap
        med = local_median(ms, usable, neighbours, run)
        dev = np.abs(ms - med) / med
        art = usable & np.isfinite(med) & (dev > artifact_threshold)
        flag[art] = "artifact"
    return IntervalSeries(kind, b.copy(), ms, flag == "", flag)


def find_pauses(series: IntervalSeries, threshold_ms: Optional[float]) -> List[dict]:
    """Intervals longer than ``threshold_ms`` (default: 2 x median valid interval)."""
    if len(series) == 0:
        return []
    if threshold_ms is None:
        med = np.median(series.ms[series.valid]) if series.n_valid else np.median(series.ms)
        threshold_ms = 2.0 * med
    m = (series.ms > threshold_ms) & (series.flag != "gap")
    return [dict(kind=series.kind, start_s=float(t - ms / 1000.0), end_s=float(t), duration_ms=float(ms),
                 flag=str(f)) for t, ms, f in zip(series.t[m], series.ms[m], series.flag[m])]


def merge_segments(segs: Sequence[Tuple[float, float]], pad: float = 0.0) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for a, b in sorted((float(a) - pad, float(b) + pad) for a, b in segs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def mask_to_segments(mask: np.ndarray, fs: float, offset_s: float = 0.0) -> List[Tuple[float, float]]:
    """Runs of True in a boolean sample mask -> [(start_s, end_s)]."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(offset_s + a / fs, offset_s + b / fs) for a, b in zip(s, e)]
