import numpy as np
import pytest

from teleep.detect import pigecg
from teleep.detect.rpeaks import detect_rpeaks
from teleep.synthetic import synthetic_ecg
from teleep.validate import score

DROP = [(49.5, 52.5)]


@pytest.mark.parametrize("rhythm", ["sinus", "complete_block"])
def test_generic_detector_finds_every_qrs(rhythm):
    s = synthetic_ecg(240, 500, rhythm=rhythm, seed=5)
    r = detect_rpeaks(s["signals"], 500, band=(5, 40), qrs_width_ms=80, min_rr_ms=220)
    sc = score(r["peaks"] / 500, s["qrs"], 0.04, exclude=DROP)
    assert sc["sensitivity"] >= 0.995 and sc["ppv"] >= 0.995
    # fiducial point lies on the R peak (within 6 ms)
    d = [np.min(np.abs(r["peaks"] / 500 - q)) for q in s["qrs"] if not 49 < q < 53]
    assert np.median(d) < 0.006


def test_generic_detector_single_lead():
    s = synthetic_ecg(120, 500, rhythm="sinus", seed=6)
    r = detect_rpeaks(s["signals"][1], 500)
    sc = score(r["peaks"] / 500, s["qrs"], 0.04, exclude=DROP)
    assert sc["sensitivity"] >= 0.99 and sc["ppv"] >= 0.99


def test_pigecg_separates_p_and_qrs_in_complete_block():
    s = synthetic_ecg(300, 500, rhythm="complete_block", seed=3)
    R = pigecg.run(s["signals"], 500, verbose=False)
    q = score(R["qrs"] / 500, s["qrs"], 0.05, exclude=DROP)
    p = score(R["p"] / 500, s["p"], 0.06, exclude=DROP)
    assert q["sensitivity"] >= 0.995 and q["ppv"] >= 0.995
    # P waves coinciding with a QRS complex cannot be seen; the rest must be found without false marks
    assert p["sensitivity"] >= 0.85 and p["ppv"] >= 0.98
    assert not R["quality"][int(50.5 * 500)]          # the dropout is flagged unanalyzable
