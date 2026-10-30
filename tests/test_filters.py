import numpy as np

from teleep.filters import FilterSpec, apply_filters, bandpass, fill_missing, median_baseline, notch


def _rms(x):
    return float(np.sqrt(np.nanmean(x ** 2)))


def test_bandpass_passes_in_band_and_removes_out_of_band():
    fs = 500
    t = np.arange(0, 20, 1 / fs)
    inband = np.sin(2 * np.pi * 10 * t)
    drift = 2 * np.sin(2 * np.pi * 0.05 * t)
    hum = 0.5 * np.sin(2 * np.pi * 120 * t)
    y = bandpass(inband + drift + hum, fs, 1.0, 40.0)
    core = slice(2 * fs, -2 * fs)
    assert abs(_rms(y[core]) - _rms(inband[core])) < 0.02
    assert _rms(y[core] - inband[core]) < 0.03


def test_zero_phase_keeps_peak_timing():
    fs = 500
    x = np.zeros(5000)
    x[2500] = 1.0
    y = bandpass(x, fs, 5, 40)
    assert int(np.argmax(np.abs(y))) == 2500


def test_missing_samples_are_preserved_and_bridged():
    fs = 250
    x = np.sin(2 * np.pi * 3 * np.arange(0, 10, 1 / fs))
    x[1000:1100] = np.nan
    f, bad = fill_missing(x)
    assert bad.sum() == 100 and np.isfinite(f).all()
    y = bandpass(x, fs, 0.5, 40)
    assert np.isnan(y[1000:1100]).all() and np.isfinite(y[:1000]).all()


def test_notch_and_chain():
    fs = 1000
    t = np.arange(0, 10, 1 / fs)
    x = np.sin(2 * np.pi * 7 * t) + 0.3 * np.sin(2 * np.pi * 60 * t)
    y = notch(x, fs, 60)
    assert _rms(y[fs:-fs] - np.sin(2 * np.pi * 7 * t)[fs:-fs]) < 0.02
    z = apply_filters(x, fs, FilterSpec(highpass_hz=0.5, lowpass_hz=40, notch_hz=60))
    assert np.isfinite(z).all()


def test_median_baseline_removes_wander_keeps_spikes():
    fs = 500
    t = np.arange(0, 10, 1 / fs)
    x = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    x[::500] += 1.0
    y = median_baseline(x, fs)
    idx = np.arange(len(x))
    interior = (idx > fs) & (idx < len(x) - fs) & (idx % 500 != 0)
    assert np.max(np.abs(y[interior])) < 0.06
    assert np.all(y[::500][1:-1] > 0.9)
