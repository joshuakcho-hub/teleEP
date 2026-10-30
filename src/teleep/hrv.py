"""Heart rate variability (HRV) from an :class:`~teleep.rr.IntervalSeries`.

Only valid (NN) intervals are used. Successive-difference measures (RMSSD, SDSD, pNNx, Poincare
SD1) use only pairs of consecutive valid intervals, so an excluded interval never creates a
false difference.

Time domain      mean/median NN, SDNN, RMSSD, SDSD, pNNx, CV, heart rate, HRV triangular index
Frequency domain Welch periodogram of the NN series resampled by natural cubic spline (default), or the
                 Lomb-Scargle periodogram of the unevenly sampled series; band powers in ms^2,
                 normalized units, LF/HF ratio, peak frequencies
Nonlinear        Poincare SD1/SD2, sample entropy, detrended fluctuation analysis (alpha1, alpha2)
Long-term        SDANN and SDNN index over fixed epochs (default 5 min)

References: Task Force of the ESC and NASPE. Heart rate variability: standards of measurement,
physiological interpretation and clinical use. Circulation 1996;93:1043-1065. Shaffer F,
Ginsberg JP. An overview of heart rate variability metrics and norms. Front Public Health
2017;5:258. Peng CK et al. Chaos 1995;5:82-87 (DFA). Richman JS, Moorman JR. Am J Physiol
2000;278:H2039-H2049 (sample entropy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sg
from scipy.interpolate import CubicSpline

from .rr import IntervalSeries

_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class HRVSettings:
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"VLF": (0.0033, 0.04), "LF": (0.04, 0.15), "HF": (0.15, 0.4)})
    pnn_ms: Tuple[float, ...] = (50.0,)
    spectral_method: str = "welch"     # "welch" or "lomb"
    interp_hz: float = 4.0
    welch_window_s: float = 64.0
    nonlinear: bool = True
    sampen_m: int = 2
    sampen_r: float = 0.2              # fraction of SD
    max_sampen_n: int = 5000


def _nan_dict(keys):
    return {k: float("nan") for k in keys}


# ------------------------------------------------------------------------- time domain
def time_domain(s: IntervalSeries, pnn_ms: Sequence[float] = (50.0,)) -> dict:
    v = s.ms[s.valid]
    n = len(s.ms)
    out = dict(n_intervals=int(n), n_nn=int(len(v)),
               valid_fraction=float(len(v) / n) if n else float("nan"))
    # rate statistics over every interval that is not across a gap and not ending on an
    # unreliable beat (includes ectopic / blocked-beat intervals, unlike the NN statistics)
    u = s.ms[(s.flag != "gap") & (s.flag != "beat")]
    out.update(n_rate_intervals=int(len(u)),
               mean_interval_ms=float(u.mean()) if len(u) else float("nan"),
               median_interval_ms=float(np.median(u)) if len(u) else float("nan"),
               rate_from_median_bpm=float(60000.0 / np.median(u)) if len(u) else float("nan"))
    if len(v) < 3:
        out.update(_nan_dict(["mean_nn_ms", "median_nn_ms", "sdnn_ms", "min_nn_ms", "max_nn_ms",
                              "cv_nn_pct", "mean_hr_bpm", "sd_hr_bpm", "min_hr_bpm", "max_hr_bpm",
                              "rmssd_ms", "sdsd_ms", "hrv_triangular_index"]
                             + [f"pnn{_fmt(x)}_pct" for x in pnn_ms]))
        return out
    hr = 60000.0 / v
    pair = s.valid[1:] & s.valid[:-1]
    d = (s.ms[1:] - s.ms[:-1])[pair]
    out.update(
        mean_nn_ms=float(v.mean()), median_nn_ms=float(np.median(v)), sdnn_ms=float(v.std(ddof=1)),
        min_nn_ms=float(v.min()), max_nn_ms=float(v.max()),
        cv_nn_pct=float(100 * v.std(ddof=1) / v.mean()),
        mean_hr_bpm=float(hr.mean()), sd_hr_bpm=float(hr.std(ddof=1)),
        min_hr_bpm=float(hr.min()), max_hr_bpm=float(hr.max()),
        rmssd_ms=float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan"),
        sdsd_ms=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        hrv_triangular_index=_tri_index(v))
    for x in pnn_ms:
        out[f"pnn{_fmt(x)}_pct"] = float(100 * np.mean(np.abs(d) > x)) if len(d) else float("nan")
    return out


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x).replace(".", "p")


def _tri_index(v: np.ndarray, bin_ms: float = 1000 / 128) -> float:
    if len(v) < 20:
        return float("nan")
    edges = np.arange(v.min(), v.max() + 2 * bin_ms, bin_ms)
    h, _ = np.histogram(v, edges)
    return float(len(v) / h.max()) if h.max() else float("nan")


# ------------------------------------------------------------------------- frequency domain
def spectrum(s: IntervalSeries, settings: HRVSettings) -> Tuple[np.ndarray, np.ndarray]:
    """Power spectral density of the NN series: (frequency Hz, PSD ms^2/Hz)."""
    tv, rv = s.t[s.valid], s.ms[s.valid]
    if len(tv) < 10 or tv[-1] - tv[0] < 10:
        return np.zeros(0), np.zeros(0)
    if settings.spectral_method == "lomb":
        T = tv[-1] - tv[0]
        y = sg.detrend(rv, type="linear")
        fmax = max(b[1] for b in settings.bands.values())
        f = np.linspace(1.0 / T, fmax * 1.05, 2000)
        p = sg.lombscargle(tv, y, 2 * np.pi * f)
        return f, 2.0 * p * T / len(tv)
    fs = settings.interp_hz
    grid = np.arange(tv[0], tv[-1], 1.0 / fs)
    if len(grid) < 16:
        return np.zeros(0), np.zeros(0)
    _, keep = np.unique(tv, return_index=True)
    y = CubicSpline(tv[keep], rv[keep], bc_type="natural")(grid)
    y = sg.detrend(y, type="linear")
    nper = int(min(len(y), max(16, round(settings.welch_window_s * fs))))
    f, p = sg.welch(y, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                    detrend="constant", scaling="density")
    return f, p


def band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    """Integral of the PSD from ``lo`` to ``hi`` (trapezoidal rule), with the PSD interpolated
    linearly at the band edges. Adjacent bands therefore add up exactly to their combined range,
    whatever the frequency resolution. Band edges outside the computed spectrum are clipped."""
    lo, hi = max(lo, float(f[0])), min(hi, float(f[-1]))
    if hi <= lo:
        return float("nan")
    m = (f > lo) & (f < hi)
    ff = np.concatenate(([lo], f[m], [hi]))
    pp = np.concatenate(([np.interp(lo, f, p)], p[m], [np.interp(hi, f, p)]))
    return float(_trapz(pp, ff))


def frequency_domain(s: IntervalSeries, settings: HRVSettings, f=None, p=None) -> dict:
    if f is None:
        f, p = spectrum(s, settings)
    names = list(settings.bands)
    keys = ([f"{b.lower()}_power_ms2" for b in names] + [f"{b.lower()}_peak_hz" for b in names]
            + ["total_power_ms2", "lf_hf_ratio", "lf_nu", "hf_nu"])
    if len(f) == 0:
        return _nan_dict(keys)
    out = {}
    duration = s.t[s.valid][-1] - s.t[s.valid][0]
    for b in names:
        lo, hi = settings.bands[b]
        m = (f >= lo) & (f < hi)
        resolvable = duration >= 1.0 / max(lo, 1e-9) or b.upper() != "VLF"
        if m.sum() >= 1 and resolvable:
            out[f"{b.lower()}_power_ms2"] = band_power(f, p, lo, hi)
            out[f"{b.lower()}_peak_hz"] = float(f[m][np.argmax(p[m])])
        else:
            out[f"{b.lower()}_power_ms2"] = float("nan")
            out[f"{b.lower()}_peak_hz"] = float("nan")
    powers = [out[f"{b.lower()}_power_ms2"] for b in names]
    out["total_power_ms2"] = float(np.nansum(powers)) if any(np.isfinite(powers)) else float("nan")
    lf, hf = out.get("lf_power_ms2", float("nan")), out.get("hf_power_ms2", float("nan"))
    out["lf_hf_ratio"] = float(lf / hf) if hf and np.isfinite(hf) and hf > 0 else float("nan")
    tot = lf + hf
    out["lf_nu"] = float(100 * lf / tot) if tot and np.isfinite(tot) and tot > 0 else float("nan")
    out["hf_nu"] = float(100 * hf / tot) if tot and np.isfinite(tot) and tot > 0 else float("nan")
    return out


# ------------------------------------------------------------------------- nonlinear
def poincare(s: IntervalSeries) -> dict:
    pair = s.valid[1:] & s.valid[:-1]
    x, y = s.ms[:-1][pair], s.ms[1:][pair]
    if len(x) < 3:
        return _nan_dict(["sd1_ms", "sd2_ms", "sd1_sd2_ratio", "ellipse_area_ms2"])
    sd1 = float(np.std((y - x) / math.sqrt(2), ddof=1))
    sd2 = float(np.std((y + x) / math.sqrt(2), ddof=1))
    return dict(sd1_ms=sd1, sd2_ms=sd2, sd1_sd2_ratio=float(sd1 / sd2) if sd2 > 0 else float("nan"),
                ellipse_area_ms2=float(math.pi * sd1 * sd2))


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2, max_n: int = 5000) -> float:
    """Sample entropy SampEn(m, r*SD) (Richman and Moorman 2000)."""
    x = np.asarray(x, float)
    if len(x) > max_n:
        x = x[:max_n]
    n = len(x)
    if n < m + 10:
        return float("nan")
    tol = r * np.std(x, ddof=1)
    if tol <= 0:
        return float("nan")

    def count(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        c = 0
        for i in range(0, len(emb) - 1, 512):
            blk = emb[i:i + 512]
            d = np.max(np.abs(blk[:, None, :] - emb[None, :, :]), axis=2)
            idx = np.arange(i, i + len(blk))[:, None]
            c += int(np.sum((d <= tol) & (np.arange(len(emb))[None, :] > idx)))
        return c

    b, a = count(m), count(m + 1)
    if a == 0 or b == 0:
        return float("nan")
    return float(-math.log(a / b))


def dfa(x: np.ndarray, scales: Sequence[int]) -> float:
    """Detrended fluctuation analysis exponent over the given box sizes (beats)."""
    x = np.asarray(x, float)
    scales = [int(sc) for sc in scales if 4 <= sc <= len(x) // 4]
    if len(scales) < 3:
        return float("nan")
    y = np.cumsum(x - x.mean())
    fl = []
    for sc in scales:
        nb = len(y) // sc
        seg = y[: nb * sc].reshape(nb, sc)
        tt = np.arange(sc)
        coef = np.polyfit(tt, seg.T, 1)
        trend = np.outer(coef[0], tt) + coef[1][:, None]
        fl.append(np.sqrt(np.mean((seg - trend) ** 2)))
    fl = np.array(fl)
    ok = fl > 0
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(np.array(scales)[ok]), np.log(fl[ok]), 1)[0])


def nonlinear(s: IntervalSeries, settings: HRVSettings) -> dict:
    out = poincare(s)
    v = s.ms[s.valid]
    out["sampen"] = sample_entropy(v, settings.sampen_m, settings.sampen_r, settings.max_sampen_n)
    out["dfa_alpha1"] = dfa(v, range(4, 17))
    out["dfa_alpha2"] = dfa(v, [16, 20, 24, 28, 32, 40, 48, 56, 64])
    return out


def hrv_all(s: IntervalSeries, settings: HRVSettings) -> dict:
    out = time_domain(s, settings.pnn_ms)
    out.update(frequency_domain(s, settings))
    if settings.nonlinear:
        out.update(nonlinear(s, settings))
    else:
        out.update(poincare(s))
    return out


# ------------------------------------------------------------------------- epochs
def epoch_bounds(duration_s: float, epoch_s: float, start: Optional[datetime] = None,
                 align_to_clock: bool = True) -> List[Tuple[float, float]]:
    """Epoch limits (s from recording start). With ``align_to_clock`` the boundaries fall on
    multiples of ``epoch_s`` of the wall clock (e.g. 10:00, 10:05 ...)."""
    first = 0.0
    if align_to_clock and start is not None:
        midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
        into = (start - midnight).total_seconds() % epoch_s
        first = (epoch_s - into) % epoch_s
    out = []
    if first > 0:
        out.append((0.0, first))
    t = first
    while t < duration_s - 1e-9:
        out.append((t, min(duration_s, t + epoch_s)))
        t += epoch_s
    return out


def epoch_table(s: IntervalSeries, duration_s: float, settings: HRVSettings, epoch_s: float = 300.0,
                start: Optional[datetime] = None, min_coverage: float = 0.8,
                align_to_clock: bool = True, min_valid_fraction: float = 0.7,
                clock: Optional[Callable[[float], datetime]] = None) -> List[dict]:
    """HRV per epoch. An epoch is analyzed when (1) analyzable intervals (not across a dropout,
    gap or unreliable beat) cover at least ``min_coverage`` of a full epoch and (2) at least
    ``min_valid_fraction`` of its intervals are valid NN intervals. Otherwise only counts are
    reported and ``analyzed`` is False. Isolated excluded intervals (ectopic beats, P waves hidden
    in a QRS complex) are bridged by the spline used for the spectrum. ``clock`` converts seconds
    from the start to local time (default: ``start`` + seconds)."""
    if clock is None and start is not None:
        clock = lambda t: start + timedelta(seconds=float(t))
    rows = []
    for t0, t1 in epoch_bounds(duration_s, epoch_s, start, align_to_clock):
        if t1 - t0 < min_coverage * epoch_s:      # partial epoch that can never qualify
            continue
        w = s.window(t0, t1)
        usable = (w.flag != "gap") & (w.flag != "beat")
        coverage = float(w.ms[usable].sum() / 1000.0 / epoch_s) if len(w) else 0.0
        vf = w.n_valid / len(w) if len(w) else 0.0
        row = dict(series=s.kind, epoch_start_s=round(t0, 3), epoch_end_s=round(t1, 3))
        if clock is not None:
            row["epoch_start"] = clock(t0).isoformat(sep=" ", timespec="seconds")
        row["coverage"] = round(coverage, 4)
        row["analyzed"] = bool(coverage >= min_coverage and vf >= min_valid_fraction and w.n_valid >= 10)
        if row["analyzed"]:
            row.update(hrv_all(w, settings))
        else:
            row.update(dict(n_intervals=len(w), n_nn=w.n_valid))
        rows.append(row)
    return rows


SUMMARY_SPECTRAL = ("vlf_power_ms2", "lf_power_ms2", "hf_power_ms2", "total_power_ms2",
                    "lf_hf_ratio", "lf_nu", "hf_nu", "sd1_ms", "sd2_ms", "sampen", "dfa_alpha1")


def recording_summary(s: IntervalSeries, epochs: List[dict], settings: HRVSettings) -> dict:
    """Whole-recording HRV: time domain over all NN intervals; SDANN / SDNN index from epochs;
    spectral and nonlinear measures as the median over analyzed epochs (spectra of a long,
    non-stationary recording are not meaningful as one estimate)."""
    out = time_domain(s, settings.pnn_ms)
    out.update(poincare(s))
    ana = [e for e in epochs if e.get("analyzed")]
    out["n_epochs"] = len(epochs)
    out["n_epochs_analyzed"] = len(ana)
    if len(ana) >= 2:
        out["sdann_ms"] = float(np.std([e["mean_nn_ms"] for e in ana], ddof=1))
        out["sdnn_index_ms"] = float(np.mean([e["sdnn_ms"] for e in ana]))
    else:
        out["sdann_ms"] = out["sdnn_index_ms"] = float("nan")
    for k in SUMMARY_SPECTRAL:
        vals = [e.get(k, float("nan")) for e in ana]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        out[f"epoch_median_{k}"] = float(np.median(vals)) if vals else float("nan")
    return out


def hourly_table(epochs: List[dict], start: Optional[datetime],
                 clock: Optional[Callable[[float], datetime]] = None) -> List[dict]:
    """Average of analyzed epochs per clock hour."""
    if clock is None:
        if start is None:
            return []
        clock = lambda t: start + timedelta(seconds=float(t))
    groups: Dict[str, List[dict]] = {}
    for e in epochs:
        if not e.get("analyzed"):
            continue
        t = clock(e["epoch_start_s"])
        groups.setdefault(t.strftime("%Y-%m-%d %H:00"), []).append(e)
    keys = ["mean_hr_bpm", "mean_nn_ms", "sdnn_ms", "rmssd_ms", "lf_power_ms2", "hf_power_ms2",
            "lf_hf_ratio", "sd1_ms", "sd2_ms"]
    rows = []
    for hour, es in sorted(groups.items()):
        r = dict(series=es[0]["series"], hour=hour, n_epochs=len(es))
        for k in keys:
            v = [e.get(k) for e in es if e.get(k) is not None and np.isfinite(e.get(k))]
            r[k] = float(np.mean(v)) if v else float("nan")
        rows.append(r)
    return rows
