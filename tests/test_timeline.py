"""Conversion layout and time line: several subjects, daylight saving time, file lengths,
auxiliary channels, and robustness of the analysis."""
import glob
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pytest

from teleep import analysis
from teleep.analysis import AnalysisSettings, analyze
from teleep.convert import _chunks, convert_experiment
from teleep.io.edf import EdfFile
from teleep.io.ponemah import MISSING, TICKS_PER_SECOND, open_experiment
from teleep.synthetic import write_synthetic_experiment
from teleep.validate import best_lead_totals, compare_with_ponemah

QUIET = dict(log=lambda m: None)


def test_two_subjects_get_their_own_folders_and_are_not_mixed(tmp_path):
    exp = tmp_path / "exp"
    write_synthetic_experiment(str(exp), duration_s=120, subject="PIG1", seed=1)
    other = tmp_path / "other"
    write_synthetic_experiment(str(other), duration_s=120, subject="PIG2", seed=2)
    for f in glob.glob(str(other / "*PIG2*")):
        shutil.copy(f, exp)
    out = tmp_path / "edf"
    files = convert_experiment(str(exp), str(out), chunk_minutes=None, **QUIET)
    assert sorted(os.path.basename(os.path.dirname(f)) for f in files) == ["PIG1", "PIG2"]
    with pytest.raises(ValueError, match="more than one subject"):
        analyze(files, AnalysisSettings(species="pig", detect_p_waves=False), **QUIET)
    with pytest.raises(FileNotFoundError, match="PIG1, PIG2"):
        analyze([str(out)], AnalysisSettings(species="pig"), **QUIET)
    res = analyze([str(out / "PIG1")], AnalysisSettings(species="pig", detect_p_waves=False), **QUIET)
    assert len(res.files) == 1


@pytest.fixture(scope="module")
def dst_recording(tmp_path_factory):
    """40 minutes across the end of daylight saving time in New York (2023-11-05 06:00 UTC,
    when clocks went back from 02:00 EDT to 01:00 EST)."""
    root = tmp_path_factory.mktemp("dst")
    truth = write_synthetic_experiment(str(root / "exp"), duration_s=2400, seed=5,
                                       start_utc=datetime(2023, 11, 5, 5, 45, tzinfo=timezone.utc))
    files = convert_experiment(str(root / "exp"), str(root / "edf"), chunk_minutes=10, **QUIET)
    return root, truth, files


def test_files_keep_their_local_offset_across_dst(dst_recording):
    _, _, files = dst_recording
    starts = [(EdfFile(f).start, EdfFile(f).utc_offset.total_seconds() / 3600) for f in files]
    assert starts == [(datetime(2023, 11, 5, 1, 45), -4), (datetime(2023, 11, 5, 1, 55), -4),
                      (datetime(2023, 11, 5, 1, 5), -5), (datetime(2023, 11, 5, 1, 15), -5)]


def test_analysis_places_files_in_utc_across_dst(dst_recording):
    root, truth, files = dst_recording
    res = analyze(files, AnalysisSettings(species="pig", detect_p_waves=False), **QUIET)
    assert [f["offset_s"] for f in res.files] == [0, 600, 1200, 1800]
    assert res.duration_s == pytest.approx(2400)
    assert not [b for b in res.bad_segments if b[2] == "gap between files"]
    assert res.start_utc == datetime(2023, 11, 5, 5, 45)
    # local clock times come from the file that contains them
    assert res.datetime_at(1300) == datetime(2023, 11, 5, 1, 6, 40)
    assert res.datetime_at(500) == datetime(2023, 11, 5, 1, 53, 20)
    rows = compare_with_ponemah(res, str(root / "exp"))
    tot = best_lead_totals(rows)["ECG"]
    assert tot["matched"] > 0.99 * tot["ponemah"]


def test_short_final_piece_joins_the_previous_file():
    s = TICKS_PER_SECOND
    assert _chunks(0, 21 * 60 * s, 600) == [(0, 600 * s), (600 * s, 1260 * s)]
    assert _chunks(0, 25 * 60 * s, 600) == [(0, 600 * s), (600 * s, 1200 * s), (1200 * s, 1500 * s)]
    assert _chunks(0, 100 * s, None) == [(0, 100 * s)]


def test_partial_second_is_kept_and_max_seconds_is_total(tmp_path):
    exp = tmp_path / "exp"
    write_synthetic_experiment(str(exp), duration_s=600.3, gap_after_s=300, gap_s=60, seed=3)
    files = convert_experiment(str(exp), str(tmp_path / "all"), chunk_minutes=None, **QUIET)
    assert len(files) == 2
    e = EdfFile(files[1])
    assert e.duration == pytest.approx(301)                     # 300.3 s padded to whole records
    assert np.isnan(e.read("ECG")[-int(0.7 * 500):]).all()     # padding reads as missing
    files = convert_experiment(str(exp), str(tmp_path / "cut"), chunk_minutes=None, max_seconds=400, **QUIET)
    assert [EdfFile(f).duration for f in files] == [pytest.approx(300), pytest.approx(100)]


def test_auxiliary_channels_keep_values_at_the_bottom_of_their_range(tmp_path):
    exp = tmp_path / "exp"
    write_synthetic_experiment(str(exp), duration_s=300, seed=6)
    rec = open_experiment(str(exp))[0]
    raw = rec.read_raw("Activity")
    assert (raw == -32256).any()                                # some seconds with zero counts
    f = convert_experiment(str(exp), str(tmp_path / "edf"), chunk_minutes=None, **QUIET)[0]
    e = EdfFile(f)
    act = e.read("Activity")
    assert np.isfinite(act).all()
    np.testing.assert_allclose(act, rec.read("Activity"), atol=1e-2)
    assert act[raw == -32256] == pytest.approx(0.0, abs=1e-2)
    assert e.signal("Activity").digital_min == MISSING


def test_detector_failure_in_one_chunk_does_not_stop_the_analysis(dst_recording, monkeypatch):
    _, _, files = dst_recording
    real = analysis.pigecg.run
    calls = {"n": 0}

    def flaky(x, fs, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated failure")
        return real(x, fs, **kw)

    monkeypatch.setattr(analysis.pigecg, "run", flaky)
    st = AnalysisSettings(species="pig", detect_p_waves=False, chunk_s=300)
    res = analyze(files[:1], st, **QUIET)
    err = [b for b in res.bad_segments if b[2] == "detector error"]
    assert err == [(300.0, 600.0, "detector error")]
    assert len(res.beats["time_s"]) > 250 and res.files[0]["unanalyzable_s"] >= 300
    assert res.files[0]["unanalyzable_s"] <= res.files[0]["duration_s"]
