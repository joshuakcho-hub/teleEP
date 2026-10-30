from datetime import datetime, timezone

import numpy as np
import pytest

from teleep.convert import convert_experiment
from teleep.io.edf import EdfFile
from teleep.io.ponemah import (ECG_MV_PER_COUNT, MISSING, TICKS_PER_SECOND, open_experiment,
                               ticks_to_utc, utc_to_ticks)
from teleep.synthetic import write_synthetic_experiment


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pnm")
    truth = write_synthetic_experiment(str(folder), duration_s=120, gap_after_s=60, gap_s=30, seed=2)
    return str(folder), truth


def test_ticks_roundtrip():
    dt = datetime(2023, 8, 6, 19, 59, 59, 123456, tzinfo=timezone.utc)
    assert ticks_to_utc(utc_to_ticks(dt)) == dt


def test_channels_rates_labels_and_calibration(experiment):
    folder, _ = experiment
    rec = open_experiment(folder)[0]
    assert rec.subject == "SYN1"
    assert [c.label for c in rec.channels] == ["ECG", "ECG1", "ECG2", "Temperature", "Activity"]
    assert [c.sample_rate for c in rec.channels] == [500, 500, 500, 10, 1]
    ecg = rec.channel("ECG")
    assert ecg.calibration == "ecg-validated" and ecg.gain == pytest.approx(ECG_MV_PER_COUNT)
    temp = rec.read("Temperature")
    assert 38.4 < np.nanmedian(temp) < 38.8            # header calibration 20..45 degC
    assert rec.timezone_name == "Eastern Standard Time"


def test_gap_detected_as_two_segments(experiment):
    folder, truth = experiment
    rec = open_experiment(folder)[0]
    assert len(rec.segments) == 2
    gap = (rec.segments[1].start - rec.segments[0].end) / TICKS_PER_SECOND
    assert gap == pytest.approx(30.0)
    assert sum(s.duration_s for s in rec.segments) == pytest.approx(120.0)


def test_missing_samples_and_alignment(experiment):
    folder, truth = experiment
    rec = open_experiment(folder)[0]
    seg = rec.segments[0]
    raw = rec.read_raw("ECG", seg.start, seg.end)
    assert len(raw) == int(seg.duration_s * 500)
    assert (raw == MISSING).sum() == 1000          # the synthetic 2-s dropout
    x = rec.read("ECG", seg.start, seg.end)
    assert np.isnan(x).sum() == 1000
    # a window that straddles the gap is filled with MISSING where no data exist
    w = rec.read_raw("ECG", seg.end - 5 * TICKS_PER_SECOND, rec.segments[1].start + 5 * TICKS_PER_SECOND)
    assert len(w) == 40 * 500 and (w[2500:2500 + 15000] == MISSING).all()


def test_local_time_conversion(experiment):
    folder, truth = experiment
    rec = open_experiment(folder)[0]
    # 2024-03-12 14:00 UTC is 10:00 in New York (daylight saving time)
    assert rec.to_local(rec.segments[0].start) == datetime(2024, 3, 12, 10, 0, 0)


def test_ponemah_marks(experiment):
    folder, truth = experiment
    rec = open_experiment(folder)[0]
    beats = rec.ponemah_beats()
    assert np.array_equal(beats["ECG"], truth["qrs_ticks"])


def test_convert_writes_lossless_annotated_edf(experiment, tmp_path):
    folder, truth = experiment
    files = convert_experiment(folder, str(tmp_path), chunk_minutes=0.5, log=lambda m: None)
    assert len(files) == 4                         # two 60-s segments in 30-s files
    e = EdfFile(files[0])
    assert e.start == datetime(2024, 3, 12, 10, 0, 0)
    assert "UTC-04:00" in e.recording
    assert e.labels == ["ECG", "ECG1", "ECG2", "Temperature", "Activity"]
    rec = open_experiment(folder)[0]
    seg = rec.segments[0]
    raw = rec.read_raw("ECG1", seg.start, seg.start + 30 * TICKS_PER_SECOND)
    d = e.read_digital("ECG1")
    ok = raw != MISSING
    assert np.array_equal(d[ok], raw[ok])
    assert np.isnan(e.read("ECG1")[~ok]).all()
    anns = [t for f in files for _, _, t in EdfFile(f).annotations]
    assert any(t.startswith("Signal dropout ECG1") for t in anns)
    e2 = EdfFile(files[2])
    gap = (e2.start - e.start).total_seconds()
    assert gap == pytest.approx(90.0)              # second segment starts after the 30-s gap


def test_convert_utc_option(experiment, tmp_path):
    folder, _ = experiment
    files = convert_experiment(folder, str(tmp_path), chunk_minutes=None, time_reference="utc",
                               ecg_only=True, log=lambda m: None)
    e = EdfFile(files[0])
    assert e.start == datetime(2024, 3, 12, 14, 0, 0)
    assert e.labels == ["ECG", "ECG1", "ECG2"]
