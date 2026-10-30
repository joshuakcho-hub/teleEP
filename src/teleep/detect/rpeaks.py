"""Generic R-peak detector for single- or multi-lead telemetry ECG.

Pan-Tompkins style, adapted to telemetry:

1. Each lead is band-passed (zero phase, species QRS band) and scaled by its own robust noise
   level, so leads are combined in proportion to their signal-to-noise ratio. Dropouts (NaN)
   contribute nothing.
2. Feature: sum over leads of the squared slope, integrated over one QRS width.
3. Adaptive dual threshold on the feature peaks, with running signal and noise levels,
   a refractory period, T-wave rejection (a peak soon after a beat whose slope is less than half
   that of the beat) and search-back for missed beats when an interval exceeds 1.66 x the
   running RR.
4. Fiducial point: the extreme of the dominant lead (largest QRS signal-to-noise) inside the QRS.
5. Beat quality: normalized cross-correlation (NCC) of every beat with the median beat.

Reference: Pan J, Tompkins WJ. A real-time QRS detection algorithm. IEEE Trans Biomed Eng
1985;32:230-236.

This detector assumes a dominant QRS. For large animals with conduction disease, where P waves
must be separated from small escape QRS complexes, use :mod:`teleep.detect.pigecg`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import signal as sg

from ..filters import bandpass, fill_missing


def _mad(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    return 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12


def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def detect_rpeaks(x, fs: float, *, band=(5.0, 40.0), qrs_width_ms: float = 80.0,
                  min_rr_ms: float = 220.0, lead_weights: Optional[Sequence[float]] = None,
                  threshold_fraction: float = 0.25, polarity: Optional[int] = None) -> dict:
    """Detect R peaks.

    Parameters
    ----------
    x : array (n_samples,) or (n_leads, n_samples), physical units, NaN for missing samples
    fs : sample rate (Hz)
    band : QRS band (Hz); the upper edge is limited to 0.45 x fs
    qrs_width_ms : integration window and fiducial search window
    min_rr_ms : refractory period
    lead_weights : optional manual weights per lead (default: automatic, by signal-to-noise)
    threshold_fraction : detection threshold between noise and signal level (Pan-Tompkins 0.25)
    polarity : +1 / -1 to force the R-peak sign on the dominant lead (default: automatic)

    Returns
    -------
    dict with ``peaks`` (sample indices), ``ncc`` (similarity to median beat), ``dominant_lead``,
    ``polarity``, ``missing`` (bool mask of samples where no lead had data), ``feature``.
    """
    X = np.atleast_2d(np.asarray(x, dtype=np.float64))
    nl, n = X.shape
    lo, hi = band
    hi = min(hi, 0.45 * fs)
    lo = min(lo, hi / 2)
    qrs = max(3, int(round(qrs_width_ms / 1000 * fs)))
    min_rr = max(1, int(round(min_rr_ms / 1000 * fs)))

    Z = np.zeros_like(X)
    miss = np.zeros((nl, n), bool)
    snr = np.zeros(nl)
    for l in range(nl):
        filled, bad = fill_missing(X[l])
        miss[l] = bad
        if bad.all():
            continue
        y = bandpass(filled, fs, lo, hi, order=3)
        y[bad] = 0.0
        noise = _mad(y[~bad])
        Z[l] = y / noise
        amp = np.percentile(np.abs(Z[l][~bad]), 99.5)
        snr[l] = amp
    if lead_weights is not None:
        w = np.asarray(lead_weights, float)
    else:
        w = np.ones(nl)
    w = w / (w.sum() + 1e-12)
    all_missing = miss.all(axis=0)

    slope = np.gradient(Z, axis=1)
    e = np.sum(w[:, None] * slope ** 2, axis=0)
    f = np.convolve(e, np.ones(qrs) / qrs, mode="same")
    f[all_missing] = 0.0

    cand, _ = sg.find_peaks(f, distance=max(1, qrs // 2))
    cand = cand[~all_missing[cand]]
    peaks = _adaptive(cand, f, slope, w, fs, min_rr, threshold_fraction, all_missing)

    # dominant lead and polarity from the median beat
    dom = int(np.argmax(snr * (w > 0))) if nl > 1 else 0
    half = qrs
    yd = Z[dom]
    if polarity is None:
        seg = [yd[p - half:p + half] for p in peaks if p - half >= 0 and p + half <= n]
        med = np.median(np.array(seg), axis=0) if seg else np.zeros(2 * half)
        polarity = 1 if med.max() >= -med.min() else -1
    fid = []
    for p in peaks:
        a, b = max(0, p - half), min(n, p + half)
        fid.append(a + int(np.argmax(polarity * yd[a:b])))
    fid = np.unique(np.array(fid, dtype=np.int64))
    if fid.size > 1:   # enforce refractory after snapping, keep the larger feature
        keep = [fid[0]]
        for p in fid[1:]:
            if p - keep[-1] < min_rr:
                if f[p] > f[keep[-1]]:
                    keep[-1] = p
            else:
                keep.append(p)
        fid = np.array(keep, dtype=np.int64)
    ncc = beat_similarity(Z, fid, int(0.4 * qrs + 0.5 * qrs), qrs + qrs // 2)
    return dict(peaks=fid, ncc=ncc, dominant_lead=dom, polarity=int(polarity),
                missing=all_missing, feature=f, lead_snr=snr)


def _adaptive(cand, f, slope, w, fs, min_rr, frac, missing):
    if cand.size == 0:
        return np.zeros(0, dtype=np.int64)
    h = f[cand]
    # learning phase: first 10 s of candidates (or all if shorter)
    n_learn = np.searchsorted(cand, cand[0] + int(10 * fs))
    learn = h[:max(n_learn, min(len(h), 20))]
    spk = np.percentile(learn, 95) if learn.size else h.max()
    npk = np.median(learn) if learn.size else 0.0
    thr1 = npk + frac * (spk - npk)
    t_wave = int(1.6 * min_rr)
    slope_mag = np.sqrt(np.sum(w[:, None] * slope ** 2, axis=0))

    def max_slope(p):
        a, b = max(0, p - min_rr // 4), min(len(f), p + min_rr // 4)
        return slope_mag[a:b].max() if b > a else 0.0

    beats = []
    rr_hist = []
    last = -10 ** 12
    last_slope = None
    i = 0
    # cumulative count of missing samples, to know whether an interval spans a dropout
    cmiss = np.concatenate(([0], np.cumsum(missing)))
    while i < len(cand):
        p, hp = cand[i], h[i]
        # search-back
        if beats and rr_hist:
            rr_avg = np.mean(rr_hist[-8:])
            if p - last > 1.66 * rr_avg and cmiss[p] - cmiss[last] == 0:
                lo_i = np.searchsorted(cand, last + min_rr)
                hi_i = np.searchsorted(cand, p - min_rr)
                if hi_i > lo_i:
                    j = lo_i + int(np.argmax(h[lo_i:hi_i]))
                    if h[j] > 0.5 * thr1:
                        q = cand[j]
                        beats.append(q)
                        rr_hist.append(q - last)
                        last, last_slope = q, max_slope(q)
                        spk = 0.25 * h[j] + 0.75 * spk
                        thr1 = npk + frac * (spk - npk)
                        continue
        if hp > thr1 and p - last >= min_rr:
            s = max_slope(p)
            if beats and p - last < t_wave and last_slope is not None and s < 0.5 * last_slope:
                npk = 0.125 * hp + 0.875 * npk          # T wave
            else:
                if beats and cmiss[p] - cmiss[last] == 0:
                    rr_hist.append(p - last)
                beats.append(p)
                last, last_slope = p, s
                spk = 0.125 * hp + 0.875 * spk
        elif hp > thr1 and beats and p - last < min_rr and hp > h[np.searchsorted(cand, last)]:
            # a larger peak inside the refractory period replaces the previous beat
            beats[-1] = p
            last, last_slope = p, max_slope(p)
        else:
            npk = 0.125 * hp + 0.875 * npk
        thr1 = npk + frac * (spk - npk)
        i += 1
    return np.array(sorted(set(beats)), dtype=np.int64)


def beat_similarity(Z, peaks, pre, post):
    """NCC of every beat with the median beat (all leads concatenated)."""
    Z = np.atleast_2d(Z)
    n = Z.shape[1]
    ok = (peaks - pre >= 0) & (peaks + post <= n)
    out = np.full(len(peaks), np.nan)
    if ok.sum() < 3:
        return out
    W = np.stack([Z[:, p - pre:p + post].ravel() for p in peaks[ok]])
    W = W - W.mean(axis=1, keepdims=True)
    t = np.median(W, axis=0)
    t = t - t.mean()
    den = np.linalg.norm(W, axis=1) * np.linalg.norm(t) + 1e-12
    out[ok] = (W @ t) / den
    return out
