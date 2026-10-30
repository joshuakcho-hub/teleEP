"""Digital filters for telemetry ECG.

All filters are zero-phase (applied forward and backward), so they do not shift the timing of
R peaks or P waves. Missing samples (NaN, e.g. telemetry dropout) are bridged by linear
interpolation before filtering and set back to NaN afterwards, so a dropout cannot spread
filter ringing into neighbouring good signal beyond a short margin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sg
from scipy.ndimage import median_filter


def fill_missing(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN runs. Returns (filled copy, boolean mask of missing samples)."""
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    if not bad.any():
        return x.copy(), bad
    y = x.copy()
    good = np.flatnonzero(~bad)
    if good.size == 0:
        return np.zeros_like(x), bad
    y[bad] = np.interp(np.flatnonzero(bad), good, x[good])
    return y, bad


def _restore(y: np.ndarray, bad: np.ndarray, fs: float, margin_s: float = 0.0) -> np.ndarray:
    if bad.any():
        if margin_s > 0:
            k = max(1, int(round(margin_s * fs)))
            bad = np.convolve(bad.astype(np.int8), np.ones(2 * k + 1, np.int8), mode="same") > 0
        y = y.copy()
        y[bad] = np.nan
    return y


def _check(fs: float, f: float, name: str) -> None:
    if not 0 < f < fs / 2:
        raise ValueError(f"{name} cut-off {f} Hz must be between 0 and the Nyquist frequency {fs / 2} Hz")


def butter_sos(fs: float, low: Optional[float] = None, high: Optional[float] = None, order: int = 4):
    """Butterworth design: ``low`` = high-pass cut-off, ``high`` = low-pass cut-off (Hz)."""
    if low and high:
        _check(fs, low, "high-pass"), _check(fs, high, "low-pass")
        return sg.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if low:
        _check(fs, low, "high-pass")
        return sg.butter(order, low, btype="highpass", fs=fs, output="sos")
    if high:
        _check(fs, high, "low-pass")
        return sg.butter(order, high, btype="lowpass", fs=fs, output="sos")
    return None


def _apply_sos(x: np.ndarray, sos, fs: float) -> np.ndarray:
    y, bad = fill_missing(x)
    if y.size < 3 * (2 * len(sos) + 1):
        return _restore(y, bad, fs)
    y = sg.sosfiltfilt(sos, y, padtype="odd", padlen=min(y.size - 1, 3 * (2 * len(sos) + 1)))
    return _restore(y, bad, fs)


def bandpass(x, fs, low, high, order: int = 4):
    """Zero-phase Butterworth band-pass between ``low`` and ``high`` Hz."""
    return _apply_sos(x, butter_sos(fs, low, high, order), fs)


def highpass(x, fs, cutoff, order: int = 4):
    return _apply_sos(x, butter_sos(fs, low=cutoff, order=order), fs)


def lowpass(x, fs, cutoff, order: int = 4):
    return _apply_sos(x, butter_sos(fs, high=cutoff, order=order), fs)


def notch(x, fs, freq: float = 60.0, q: float = 30.0, harmonics: int = 1):
    """Zero-phase notch at ``freq`` (and its first ``harmonics`` multiples below Nyquist)."""
    y, bad = fill_missing(x)
    for k in range(1, harmonics + 1):
        f0 = freq * k
        if f0 >= fs / 2:
            break
        b, a = sg.iirnotch(f0, q, fs=fs)
        if y.size > 3 * max(len(a), len(b)):
            y = sg.filtfilt(b, a, y)
    return _restore(y, bad, fs)


def median_baseline(x, fs, short_s: float = 0.2, long_s: float = 0.6):
    """Remove baseline wander with a two-stage running median (QRS-preserving)."""
    y, bad = fill_missing(x)
    k1 = max(3, int(short_s * fs) | 1)
    k2 = max(3, int(long_s * fs) | 1)
    b = median_filter(median_filter(y, size=k1, mode="nearest"), size=k2, mode="nearest")
    return _restore(y - b, bad, fs)


@dataclass
class FilterSpec:
    """A chain of display / export filters. ``None`` disables a stage."""

    highpass_hz: Optional[float] = 0.5
    lowpass_hz: Optional[float] = 40.0
    notch_hz: Optional[float] = None       # 50 or 60 (mains)
    notch_harmonics: int = 1
    median_baseline: bool = False
    order: int = 4

    def describe(self) -> str:
        parts = []
        if self.median_baseline:
            parts.append("median baseline removal")
        if self.highpass_hz:
            parts.append(f"HP {self.highpass_hz:g} Hz")
        if self.lowpass_hz:
            parts.append(f"LP {self.lowpass_hz:g} Hz")
        if self.notch_hz:
            parts.append(f"notch {self.notch_hz:g} Hz")
        return ", ".join(parts) if parts else "none"

    def as_dict(self) -> dict:
        return asdict(self)


def apply_filters(x: np.ndarray, fs: float, spec: Optional[FilterSpec]) -> np.ndarray:
    """Apply a :class:`FilterSpec` to a 1-D signal (NaN-tolerant, zero phase)."""
    y = np.asarray(x, dtype=np.float64)
    if spec is None:
        return y.copy()
    if spec.median_baseline:
        y = median_baseline(y, fs)
    hp = spec.highpass_hz if spec.highpass_hz else None
    lp = spec.lowpass_hz if spec.lowpass_hz and spec.lowpass_hz < fs / 2 else None
    sos = butter_sos(fs, hp, lp, spec.order)
    if sos is not None:
        y = _apply_sos(y, sos, fs)
    if spec.notch_hz:
        y = notch(y, fs, spec.notch_hz, harmonics=spec.notch_harmonics)
    return y
