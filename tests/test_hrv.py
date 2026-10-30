import numpy as np
import pytest

from teleep.hrv import (HRVSettings, dfa, epoch_bounds, frequency_domain, poincare, sample_entropy,
                        time_domain)
from teleep.rr import build_intervals, find_pauses


def series_from_rr(rr_ms, t0=0.0):
    t = np.concatenate([[t0], t0 + np.cumsum(rr_ms) / 1000.0])
    return build_intervals(t, artifact_threshold=None)


def test_time_domain_matches_definitions():
    rr = np.array([800, 810, 790, 805, 795, 820, 780, 800, 815, 785], float)
    s = series_from_rr(rr)
    td = time_domain(s, (50, 12))
    d = np.diff(s.ms)
    assert td["mean_nn_ms"] == pytest.approx(rr.mean())
    assert td["sdnn_ms"] == pytest.approx(rr.std(ddof=1))
    assert td["rmssd_ms"] == pytest.approx(np.sqrt(np.mean(d ** 2)))
    assert td["pnn12_pct"] == pytest.approx(100 * np.mean(np.abs(d) > 12))
    assert td["mean_hr_bpm"] == pytest.approx(np.mean(60000 / rr))


def test_successive_differences_skip_excluded_intervals():
    rr = np.full(40, 800.0)
    rr[20] = 1600.0                       # missed beat
    t = np.concatenate([[0.0], np.cumsum(rr) / 1000.0])
    s = build_intervals(t, artifact_threshold=0.2)
    assert s.flag[20] == "artifact" and s.valid.sum() == 39
    td = time_domain(s)
    assert td["rmssd_ms"] == pytest.approx(0.0, abs=1e-6)
    assert td["sdnn_ms"] == pytest.approx(0.0, abs=1e-6)


def test_gap_flag_and_pauses():
    t = np.arange(0, 60, 0.8)
    t = np.concatenate([t[t < 20], t[t > 23]])         # 3-s hole
    s = build_intervals(t, bad_segments=[(20.5, 22.5)], artifact_threshold=0.2)
    assert (s.flag == "gap").sum() == 1
    assert find_pauses(s, 2000) == []                   # the hole is a dropout, not a pause
    s2 = build_intervals(t, artifact_threshold=0.2)
    p = find_pauses(s2, 2000)
    assert len(p) == 1 and p[0]["duration_ms"] > 3000


@pytest.mark.parametrize("method", ["welch", "lomb"])
def test_band_power_of_known_modulation(method):
    # RR modulated at 0.3 Hz with amplitude 20 ms -> HF power ~ 20^2 / 2 = 200 ms^2
    t, rr = [0.0], []
    while t[-1] < 600:
        r = 800 + 20 * np.sin(2 * np.pi * 0.3 * t[-1])
        rr.append(r)
        t.append(t[-1] + r / 1000)
    s = build_intervals(np.array(t), artifact_threshold=None)
    st = HRVSettings(bands={"LF": (0.04, 0.15), "HF": (0.15, 0.4)}, spectral_method=method)
    fd = frequency_domain(s, st)
    assert fd["hf_power_ms2"] == pytest.approx(200, rel=0.15)
    assert fd["lf_power_ms2"] < 0.05 * fd["hf_power_ms2"]
    assert fd["hf_peak_hz"] == pytest.approx(0.3, abs=0.02)


def test_poincare_dfa_sampen():
    rng = np.random.default_rng(0)
    rr = 800 + rng.normal(0, 20, 2000)
    s = series_from_rr(rr)
    p = poincare(s)
    assert p["sd1_ms"] == pytest.approx(20, rel=0.1) and p["sd2_ms"] == pytest.approx(20, rel=0.1)
    assert dfa(rr, range(4, 17)) == pytest.approx(0.5, abs=0.1)        # white noise
    assert dfa(np.cumsum(rng.normal(size=2000)), range(4, 17)) > 1.3    # random walk
    reg = 800 + 20 * np.sin(np.arange(600) * 0.5)
    assert sample_entropy(reg) < sample_entropy(rr[:600])


def test_epoch_bounds_align_to_clock():
    from datetime import datetime
    b = epoch_bounds(1000, 300, datetime(2024, 1, 1, 10, 3, 0))
    assert b[0] == (0.0, 120.0) and b[1] == (120.0, 420.0)


def test_artifact_median_does_not_reach_across_a_gap():
    # the rate doubles during a dropout: the first intervals after it must not be compared
    # with the intervals before it
    before = np.arange(0, 10.01, 1.0)
    after = 20.0 + np.arange(0, 10.01, 0.5)
    t = np.concatenate([before, after])
    s = build_intervals(t, bad_segments=[(10.5, 19.5)], artifact_threshold=0.2)
    assert s.counts()["artifact"] == 0 and s.counts()["gap"] == 1


def test_band_power_is_additive_across_adjacent_bands():
    from teleep.hrv import band_power
    f = np.arange(0, 2.0, 0.0371)           # coarse resolution; band edges fall between bins
    p = 1.0 + np.sin(3 * f) ** 2
    lf, hf = band_power(f, p, 0.04, 0.4), band_power(f, p, 0.4, 1.0)
    whole = band_power(f, p, 0.04, 1.0)
    assert lf + hf == pytest.approx(whole, rel=1e-12)
    fine = np.linspace(0.04, 1.0, 100001)
    exact = np.trapz(np.interp(fine, f, p), fine) if hasattr(np, "trapz") else None
    if exact is not None:
        assert whole == pytest.approx(exact, rel=1e-3)
