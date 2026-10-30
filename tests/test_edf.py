from datetime import datetime

import numpy as np
import pytest

from teleep.io.edf import DROPOUT_TEXT, EdfFile, EdfSignal, write_edf


def test_digital_roundtrip_is_lossless(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(-32256, 32256, 5000).astype(np.int32)
    sig = EdfSignal("ECG", raw, 500.0, "mV", physical_min=-10.0, physical_max=32255 * 10 / 32256,
                    digital_min=-32256, digital_max=32255, digital=True)
    p = write_edf(str(tmp_path / "a.edf"), [sig], start=datetime(2024, 1, 2, 3, 4, 5))
    e = EdfFile(p, rail_as_missing=False)
    assert e.start == datetime(2024, 1, 2, 3, 4, 5)
    assert e.duration == 10.0
    assert np.array_equal(e.read_digital("ECG"), raw)
    np.testing.assert_allclose(e.read("ECG"), raw * 10 / 32256, atol=1e-6)


def test_physical_roundtrip_mixed_rates_and_annotations(tmp_path):
    t = np.arange(0, 20, 1 / 250)
    ecg = np.sin(2 * np.pi * 1.3 * t)
    temp = 38 + 0.01 * np.arange(200)
    anns = [(1.25, 0.0, "marker"), (12345.678901 / 1000, 2.5, f"{DROPOUT_TEXT} ECG")]
    p = write_edf(str(tmp_path / "b.edf"), [EdfSignal("ECG", ecg, 250.0), EdfSignal("Temp", temp, 10.0, "degC")],
                  start=datetime(2023, 8, 6, 15, 59, 59), annotations=anns)
    e = EdfFile(p)
    assert e.edf_type == "EDF+C"
    assert e.labels == ["ECG", "Temp"]
    x = e.read("ECG", missing_as_nan=False)
    q = (x.max() - x.min()) / 65535
    assert np.max(np.abs(x - ecg)) <= q
    assert np.allclose(e.read("Temp", missing_as_nan=False), temp, atol=1e-3)
    onsets = [a[0] for a in e.annotations]
    assert onsets[0] == pytest.approx(1.25)
    assert onsets[1] == pytest.approx(12.345679, abs=1e-6)
    assert e.dropouts() == [(pytest.approx(12.345679, abs=1e-6), pytest.approx(14.845679, abs=1e-6), f"{DROPOUT_TEXT} ECG")]


def test_window_reading_matches_full_read(tmp_path):
    x = np.arange(10_000, dtype=float) / 100.0
    p = write_edf(str(tmp_path / "c.edf"), [EdfSignal("S", x, 100.0, physical_min=0, physical_max=100)])
    e = EdfFile(p)
    full = e.read("S")
    np.testing.assert_allclose(e.read("S", 12.34, 5.0), full[1234:1734])


def test_rail_values_become_nan_for_ponemah_files(tmp_path):
    d = np.zeros(1000, dtype=np.int32)
    d[100:200] = -32256
    sig = EdfSignal("ECG", d, 100.0, physical_min=-10, physical_max=9.99969, digital_min=-32256,
                    digital_max=32255, digital=True)
    p = write_edf(str(tmp_path / "d.edf"), [sig], equipment="DSI_Ponemah")
    e = EdfFile(p)
    assert e.rail_as_missing
    y = e.read("ECG")
    assert np.isnan(y[100:200]).all() and np.isfinite(y[:100]).all()


def test_readable_by_pyedflib(tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    t = np.arange(0, 10, 1 / 500)
    p = write_edf(str(tmp_path / "e.edf"), [EdfSignal("ECG", np.sin(t), 500.0)],
                  annotations=[(2.0, 0.0, "hello")])
    f = pyedflib.EdfReader(p)
    try:
        assert f.getSignalLabels() == ["ECG"]
        assert f.getSampleFrequency(0) == 500
        np.testing.assert_allclose(f.readSignal(0), np.sin(t), atol=1e-4)
        assert list(f.readAnnotations()[2]) == ["hello"]
    finally:
        f.close()


def test_duplicate_labels_are_made_unique(tmp_path):
    # files from the earlier Ponemah2EDF converter call all three leads "ECG"
    sigs = [EdfSignal("ECG", np.full(500, float(k)), 100.0, physical_min=-5, physical_max=5) for k in range(3)]
    e = EdfFile(write_edf(str(tmp_path / "dup.edf"), sigs))
    assert e.labels == ["ECG", "ECG_2", "ECG_3"]
    x, fs, labels = e.read_matrix(e.ecg_labels())
    np.testing.assert_allclose(x.mean(axis=1), [0, 1, 2], atol=1e-3)


def test_fractional_start_time_and_annotations(tmp_path):
    from datetime import timedelta
    start = datetime(2024, 3, 12, 10, 0, 0, 250000)
    p = write_edf(str(tmp_path / "f.edf"), [EdfSignal("ECG", np.zeros(1000), 100.0)], start=start,
                  annotations=[(2.0, 0.5, "x")], recording_info="TeleEP_1 UTC-04:00")
    e = EdfFile(p)
    assert e.start == start and e.start_fraction == pytest.approx(0.25)
    assert e.annotations == [(pytest.approx(2.0), pytest.approx(0.5), "x")]
    assert e.utc_offset == timedelta(hours=-4)
    assert e.start_utc == start + timedelta(hours=4)
    # a fraction of a second forces EDF+, because plain EDF cannot store it
    q = write_edf(str(tmp_path / "g.edf"), [EdfSignal("ECG", np.zeros(1000), 100.0)], start=start)
    assert EdfFile(q).edf_type == "EDF+C" and EdfFile(q).start == start


def test_utc_offset_parsing():
    from datetime import timedelta
    from teleep.io.edf import format_utc_offset, parse_utc_offset
    assert parse_utc_offset("Startdate X TeleEP_1.0 UTC seg1/1") == timedelta(0)
    assert parse_utc_offset("TeleEP_1.0 UTC+05:30 seg1/1") == timedelta(hours=5, minutes=30)
    assert parse_utc_offset("no zone") is None
    for h in (-4, 0, 9):
        assert parse_utc_offset(format_utc_offset(timedelta(hours=h))) == timedelta(hours=h)


def test_partial_last_record_is_padded(tmp_path):
    x = np.arange(1050, dtype=np.int32)
    sig = EdfSignal("ECG", x, 100.0, physical_min=-10, physical_max=10, digital_min=-32768,
                    digital_max=32767, digital=True)
    e = EdfFile(write_edf(str(tmp_path / "p.edf"), [sig], pad_last_record=True,
                          equipment="DSI_Ponemah", force_edfplus=True))
    assert e.n_records == 11
    y = e.read_digital("ECG")
    assert np.array_equal(y[:1050], x) and (y[1050:] == -32768).all()
    assert np.isnan(e.read("ECG")[1050:]).all()
    # less than one record: padded to one record instead of failing
    short = EdfSignal("ECG", x[:30], 100.0, physical_min=-10, physical_max=10, digital=True)
    assert EdfFile(write_edf(str(tmp_path / "s.edf"), [short], pad_last_record=True)).n_records == 1


def test_header_month_is_independent_of_locale(tmp_path):
    p = write_edf(str(tmp_path / "m.edf"), [EdfSignal("ECG", np.zeros(100), 100.0)],
                  start=datetime(2023, 8, 6, 15, 59, 59))
    with open(p, "rb") as f:
        h = f.read(256)
    assert b"Startdate 06-AUG-2023" in h and h[168:184] == b"06.08.2315.59.59"
