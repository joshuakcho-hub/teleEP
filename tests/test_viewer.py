"""The HTML viewer computes HRV in JavaScript. Check that it agrees with the Python package.

Runs only when Playwright and its Chromium browser are installed:
    pip install playwright && python -m playwright install chromium
"""
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from teleep.analysis import AnalysisSettings, analyze
from teleep.convert import convert_experiment
from teleep.export import export_all
from teleep.synthetic import write_synthetic_experiment
from teleep.viewer.launch import viewer_file

KEYS = ["n_nn", "mean_nn_ms", "sdnn_ms", "rmssd_ms", "hf_power_ms2", "lf_power_ms2", "sd1_ms", "sd2_ms",
        "sampen", "dfa_alpha1", "mean_hr_bpm"]


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as ex:   # browser not installed
            pytest.skip(f"Chromium not available: {ex}")
        yield b
        b.close()


def test_viewer_hrv_matches_python(tmp_path, browser):
    # a start time with a fraction of a second checks that both read it from the EDF+ time-keeping
    # annotation and align epochs to the clock in the same way
    write_synthetic_experiment(str(tmp_path / "exp"), duration_s=900, seed=7,
                               start_utc=datetime(2024, 3, 12, 14, 2, 3, 370000, tzinfo=timezone.utc))
    files = convert_experiment(str(tmp_path / "exp"), str(tmp_path / "edf"), chunk_minutes=None, log=lambda m: None)
    res = analyze(files, AnalysisSettings(species="pig", align_epochs_to_clock=True), log=lambda m: None)
    out = tmp_path / "res"
    export_all(res, str(out), xlsx=False, figure=False, log=lambda m: None)
    marks = next((out / "viewer_marks").glob("*.json"))

    page = browser.new_page(accept_downloads=True)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(Path(viewer_file()).as_uri())
    page.set_input_files("#fEdf", files[0])
    page.wait_for_function("document.getElementById('msg').textContent.startsWith('Loaded')")
    page.set_input_files("#fMarks", str(marks))
    page.wait_for_function("document.getElementById('msg').textContent.startsWith('Imported')")
    page.click("#bExport")
    with page.expect_download() as d:
        page.click("#menu button[data-x=epochs]")
    js_csv = tmp_path / "viewer_epochs.csv"
    d.value.save_as(str(js_csv))
    assert not errors

    py = {(r["series"], round(r["epoch_start_s"], 1)): r for r in res.epochs if r.get("analyzed")}
    js = list(csv.DictReader(open(js_csv)))
    compared = 0
    for r in js:
        key = (r["series"], round(float(r["epoch_start_s"]), 1))
        if r["analyzed"] != "1" or key not in py:
            continue
        for k in KEYS:
            assert float(r[k]) == pytest.approx(py[key][k], rel=1e-4, abs=1e-6), (key, k)
        compared += 1
    assert compared >= 2
    assert res.start.microsecond == 370000
