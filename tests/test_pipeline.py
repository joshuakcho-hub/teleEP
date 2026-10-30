"""End to end: synthetic Ponemah experiment -> EDF -> analysis -> export, and the CLI."""
import csv
import json
import os

import numpy as np
import pytest

from teleep.analysis import AnalysisSettings, analyze
from teleep.cli import main
from teleep.convert import convert_experiment
from teleep.export import export_all
from teleep.synthetic import write_synthetic_experiment
from teleep.validate import compare_with_ponemah


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    root = tmp_path_factory.mktemp("rec")
    exp = root / "exp"
    truth = write_synthetic_experiment(str(exp), duration_s=600, seed=4)
    edf = root / "edf"
    convert_experiment(str(exp), str(edf), chunk_minutes=5, log=lambda m: None)
    return root, truth


def test_analysis_recovers_rates_and_beats(recording):
    root, truth = recording
    res = analyze([str(root / "edf")], AnalysisSettings(species="pig"), log=lambda m: None)
    assert len(res.files) == 2 and res.duration_s == pytest.approx(600)
    q = res.beats["time_s"]
    true_q = truth["qrs_s"]
    near = np.array([np.min(np.abs(q - t)) for t in true_q if not 195 < t < 205])
    assert np.mean(near < 0.05) > 0.995
    true_v = np.mean(60000 / np.diff(truth["qrs_s"] * 1000))
    true_a = np.mean(60000 / np.diff(truth["p_s"] * 1000))
    assert res.summary["RR"]["mean_hr_bpm"] == pytest.approx(true_v, rel=0.01)
    assert res.summary["PP"]["mean_hr_bpm"] == pytest.approx(true_a, rel=0.02)
    # beats are continuous across the file boundary at 300 s
    assert np.any((q > 299) & (q < 301.5))
    rows = compare_with_ponemah(res, str(root / "exp"))
    tot = sum(r["matched"] for r in rows)
    assert tot / sum(r["ponemah"] for r in rows) > 0.99


def test_export_files(recording, tmp_path):
    root, _ = recording
    res = analyze([str(root / "edf")], AnalysisSettings(species="pig", detector="generic"), log=lambda m: None)
    paths = export_all(res, str(tmp_path), log=lambda m: None)
    for k in ("beats", "hrv_summary", "hrv_epochs", "files", "summary_json", "xlsx", "figure"):
        assert os.path.exists(paths[k]), k
    with open(paths["beats"]) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(res.beats["time_s"])
    js = json.load(open(paths["summary_json"]))
    assert js["summary"]["RR"]["n_nn"] > 500
    marks = json.load(open(os.path.join(paths["viewer_marks"], os.listdir(paths["viewer_marks"])[0])))
    assert marks["format"] == "teleep-marks-1" and len(marks["qrs"]) > 250


def test_cli_demo_and_info(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", str(out), "--minutes", "4", "--chunk-minutes", "2"]) == 0
    assert (out / "results" / "hrv_summary.csv").exists()
    edf = sorted((out / "edf").glob("*.edf"))[0]
    assert main(["info", str(edf)]) == 0
    assert "ECG" in capsys.readouterr().out
    assert main(["export-signal", str(edf), "-o", str(tmp_path / "s.csv"), "--start", "1", "--duration", "2",
                 "--hp", "0.5", "--lp", "40"]) == 0
    lines = open(tmp_path / "s.csv").read().splitlines()
    assert len(lines) == 1 + 1000 and lines[0].startswith("time_s,datetime,ECG")
