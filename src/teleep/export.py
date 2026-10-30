"""Write analysis results to CSV, Excel (.xlsx) and JSON, plus marks files for the viewer."""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import __version__

EXCEL_MAX_ROWS = 1_048_000

COLUMN_GUIDE: Dict[str, str] = {
    "beat": "Beat number (0-based) in the recording",
    "time_s": "Time in seconds from the start of the recording (first file)",
    "datetime": "Wall-clock time (local time of the acquisition computer unless converted with --utc)",
    "file": "EDF file that contains the beat",
    "rr_ms": "Interval from the previous QRS to this QRS (ms)",
    "rr_flag": "Why the interval is excluded from HRV: gap (dropout / unanalyzable / file gap), beat "
               "(unreliable detection), range (outside the species range), artifact (differs from "
               "neighbours by more than the artifact threshold); empty = valid NN interval",
    "hr_bpm": "Instantaneous heart rate 60000 / rr_ms",
    "confidence": "Detector confidence that this is a QRS complex (0-1)",
    "template_ncc": "Similarity of the beat to its QRS template (normalized cross-correlation, -1 to 1)",
    "morphology": "QRS morphology class (pigecg: k-means cluster of the multi-lead QRS shape)",
    "good": "1 = detection reliable and signal analyzable; 0 = excluded from HRV",
    "pr_ms": "Time from the closest preceding P wave (50-450 ms window) to the QRS; descriptive only",
    "p": "P-wave number", "pp_ms": "Interval from the previous P wave (ms)",
    "pp_flag": "As rr_flag, for the PP interval",
    "series": "RR = ventricular (QRS-to-QRS) intervals; PP = atrial (P-to-P) intervals",
    "epoch_start": "Epoch start (wall clock)",
    "coverage": "Fraction of the epoch covered by analyzable intervals (not across a dropout, gap or unreliable beat)",
    "analyzed": "1 when coverage >= 0.8 and valid_fraction >= 0.7 (defaults); HRV is computed only then",
    "n_intervals": "Intervals in the window", "n_nn": "Valid (NN) intervals used for HRV",
    "valid_fraction": "n_nn / n_intervals",
    "n_rate_intervals": "Intervals not crossing a gap and not ending on an unreliable beat",
    "mean_interval_ms": "Mean of those intervals (includes ectopic or blocked-beat intervals)",
    "median_interval_ms": "Median of those intervals", "rate_from_median_bpm": "60000 / median_interval_ms",
    "mean_nn_ms": "Mean NN interval", "median_nn_ms": "Median NN interval",
    "sdnn_ms": "Standard deviation of NN intervals", "min_nn_ms": "Shortest NN interval",
    "max_nn_ms": "Longest NN interval", "cv_nn_pct": "Coefficient of variation of NN (SDNN / mean x 100)",
    "mean_hr_bpm": "Mean instantaneous heart (or atrial) rate over NN intervals",
    "sd_hr_bpm": "SD of instantaneous rate", "min_hr_bpm": "Minimum instantaneous rate",
    "max_hr_bpm": "Maximum instantaneous rate",
    "rmssd_ms": "Root mean square of successive NN differences (consecutive valid pairs only)",
    "sdsd_ms": "SD of successive NN differences",
    "pnn50_pct": "Percent of successive NN differences > 50 ms (pNNx: > x ms)",
    "hrv_triangular_index": "NN count / height of the NN histogram (bin 7.8125 ms)",
    "vlf_power_ms2": "Very-low-frequency power (requires windows longer than 1 / lower band edge)",
    "lf_power_ms2": "Low-frequency power", "hf_power_ms2": "High-frequency power",
    "total_power_ms2": "Sum of the band powers that could be resolved",
    "lf_hf_ratio": "LF / HF", "lf_nu": "LF / (LF + HF) x 100", "hf_nu": "HF / (LF + HF) x 100",
    "lf_peak_hz": "Frequency of the largest LF spectral peak", "hf_peak_hz": "Frequency of the largest HF peak",
    "sd1_ms": "Poincare SD1 (short-term variability)", "sd2_ms": "Poincare SD2 (long-term variability)",
    "sd1_sd2_ratio": "SD1 / SD2", "ellipse_area_ms2": "Poincare ellipse area (pi x SD1 x SD2)",
    "sampen": "Sample entropy (m = 2, r = 0.2 x SD)",
    "dfa_alpha1": "Detrended fluctuation analysis, short-term exponent (4-16 beats)",
    "dfa_alpha2": "Detrended fluctuation analysis, long-term exponent (16-64 beats)",
    "sdann_ms": "SD of the mean NN of all analyzed epochs",
    "sdnn_index_ms": "Mean of the SDNN of all analyzed epochs",
    "start_s": "Start (s from recording start)", "end_s": "End (s from recording start)",
    "duration_ms": "Pause length", "reason": "Why the stretch is excluded",
}


EMPTY_HEADERS = {
    "pauses": ["series", "start_s", "end_s", "start", "duration_ms", "flag"],
    "signal_quality": ["start_s", "end_s", "duration_s", "start", "reason"],
    "hrv_hourly": ["series", "hour", "n_epochs", "mean_hr_bpm", "mean_nn_ms", "sdnn_ms", "rmssd_ms",
                   "lf_power_ms2", "hf_power_ms2", "lf_hf_ratio", "sd1_ms", "sd2_ms"],
    "hrv_epochs": ["series", "epoch_start_s", "epoch_end_s", "epoch_start", "coverage", "analyzed"],
}


def _fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(v):
            return ""
        return round(float(v), nd)
    return v


def _write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    return path


def _dicts_to_rows(dicts: List[dict]):
    keys: List[str] = []
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    return keys, [[d.get(k) for k in keys] for d in dicts]


def _rnd(v, nd: int, missing=""):
    """Rounded number, or ``missing`` for NaN / infinity."""
    v = float(v)
    return round(v, nd) if math.isfinite(v) else missing


def beat_table(res) -> tuple:
    b = res.beats
    n = len(b["time_s"])
    rr = np.full(n, np.nan)
    flag = np.full(n, "", dtype=object)
    if n > 1:
        rr[1:] = res.rr.ms
        flag[1:] = res.rr.flag
        flag[0] = "first"
    hr = 60000.0 / rr
    names = [f["file"] for f in res.files]
    header = ["beat", "time_s", "datetime", "file", "rr_ms", "rr_flag", "hr_bpm", "confidence",
              "template_ncc", "morphology", "good"]
    has_pr = "pr_ms" in b
    if has_pr:
        header.append("pr_ms")

    def rows():
        for i in range(n):
            t = float(b["time_s"][i])
            r = [i, round(t, 4), res.datetime_at(t).isoformat(sep=" ", timespec="milliseconds"),
                 names[int(b["file_index"][i])], _rnd(rr[i], 1), flag[i], _rnd(hr[i], 2),
                 _rnd(b["confidence"][i], 3), _rnd(b["ncc"][i], 3), int(b["morphology"][i]),
                 int(bool(b["good"][i]))]
            if has_pr:
                v = b["pr_ms"][i]
                r.append(round(float(v), 1) if np.isfinite(v) else "")
            yield r
    return header, rows


def pwave_table(res) -> tuple:
    p = res.pwaves
    n = len(p["time_s"])
    pp = np.full(n, np.nan)
    flag = np.full(n, "", dtype=object)
    if n > 1:
        pp[1:] = res.pp.ms
        flag[1:] = res.pp.flag
        flag[0] = "first"
    names = [f["file"] for f in res.files]
    header = ["p", "time_s", "datetime", "file", "pp_ms", "pp_flag", "rate_per_min", "template_ncc", "good"]

    def rows():
        for i in range(n):
            t = float(p["time_s"][i])
            yield [i, round(t, 4), res.datetime_at(t).isoformat(sep=" ", timespec="milliseconds"),
                   names[int(p["file_index"][i])], round(pp[i], 1) if np.isfinite(pp[i]) else "", flag[i],
                   round(60000 / pp[i], 2) if np.isfinite(pp[i]) else "", round(float(p["ncc"][i]), 3),
                   int(bool(p["good"][i]))]
    return header, rows


def summary_rows(res) -> List[dict]:
    rows = []
    for series, s in res.summary.items():
        d = dict(series=series, recording_start=res.start.isoformat(sep=" "),
                 duration_h=round(res.duration_s / 3600, 4))
        d.update(s)
        rows.append(d)
    return rows


def quality_rows(res) -> List[dict]:
    return [dict(start_s=round(a, 3), end_s=round(b, 3), duration_s=round(b - a, 3),
                 start=res.datetime_at(a).isoformat(sep=" ", timespec="milliseconds"), reason=w)
            for a, b, w in sorted(res.bad_segments)]


def pause_rows(res) -> List[dict]:
    return [dict(series=p["kind"], start_s=round(p["start_s"], 3), end_s=round(p["end_s"], 3),
                 start=res.datetime_at(p["start_s"]).isoformat(sep=" ", timespec="milliseconds"),
                 duration_ms=round(p["duration_ms"], 1), flag=p["flag"]) for p in res.pauses]


def settings_rows(res) -> List[List]:
    rows = [["software", res.software], ["analyzed", datetime.now().isoformat(sep=" ", timespec="seconds")],
            ["recording_start", res.start.isoformat(sep=" ")],
            ["recording_start_utc", res.start_utc.isoformat(sep=" ") if res.start_utc else "not stated in the files"],
            ["duration_h", round(res.duration_s / 3600, 4)]]
    for k, v in res.settings.items():
        rows.append([k, json.dumps(v) if isinstance(v, (dict, list, tuple)) else v])
    for k, v in (res.av or {}).items():
        rows.append([f"av_{k}", v])
    return rows


def write_marks(res, out_dir: str) -> List[str]:
    """One marks file per EDF (times relative to that file) for the TeleEP viewer."""
    os.makedirs(out_dir, exist_ok=True)
    out = []
    b, p = res.beats, res.pwaves
    for i, f in enumerate(res.files):
        off = f["offset_s"]
        m = b["file_index"] == i
        qt = b["time_s"][m] - off
        d = dict(format="teleep-marks-1", software=res.software, file=f["file"], start=f["start"],
                 fs=f["fs"], qrs=[round(float(v), 4) for v in qt],
                 qrs_morph=[int(v) for v in b["morphology"][m]],
                 qrs_conf=[_rnd(v, 3, None) for v in b["confidence"][m]],
                 qrs_good=[int(bool(v)) for v in b["good"][m]],
                 p=[], t_end=[None] * int(m.sum()),
                 bad=[[round(a - off, 3), round(bb - off, 3)] for a, bb, w in res.bad_segments
                      if w != "unanalyzable P" and bb > off and a < off + f["duration_s"]],
                 bad_p=[[round(a - off, 3), round(bb - off, 3)] for a, bb, w in res.bad_segments
                        if w == "unanalyzable P" and bb > off and a < off + f["duration_s"]])
        if p is not None:
            mp = p["file_index"] == i
            d["p"] = [round(float(v), 4) for v in p["time_s"][mp] - off]
            d["p_good"] = [int(bool(v)) for v in p["good"][mp]]
        path = os.path.join(out_dir, os.path.splitext(f["file"])[0] + ".marks.json")
        with open(path, "w") as fh:
            json.dump(d, fh)
        out.append(path)
    return out


def _clean_json(o):
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean_json(o.tolist())
    if isinstance(o, datetime):
        return o.isoformat(sep=" ")
    return o


def export_all(res, out_dir: str, xlsx: bool = True, marks: bool = True, figure: bool = True,
               log=print) -> Dict[str, str]:
    """Write every table. Returns {name: path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    h, rows = beat_table(res)
    paths["beats"] = _write_csv(os.path.join(out_dir, "beats.csv"), h, rows())
    if res.pwaves is not None:
        h, rows = pwave_table(res)
        paths["pwaves"] = _write_csv(os.path.join(out_dir, "pwaves.csv"), h, rows())
    for name, data in (("hrv_summary", summary_rows(res)), ("hrv_epochs", res.epochs),
                       ("hrv_hourly", res.hourly), ("pauses", pause_rows(res)),
                       ("signal_quality", quality_rows(res)), ("files", res.files)):
        keys, rws = _dicts_to_rows(data) if data else (EMPTY_HEADERS.get(name, ["(none)"]), [])
        paths[name] = _write_csv(os.path.join(out_dir, f"{name}.csv"), keys, rws)
    js = dict(software=res.software, recording_start=res.start, recording_start_utc=res.start_utc,
              duration_s=res.duration_s,
              settings=res.settings, files=res.files, summary=res.summary, av_association=res.av,
              rr_flag_counts=res.rr.counts(), pp_flag_counts=res.pp.counts() if res.pp is not None else None,
              n_pauses=len(res.pauses))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(_clean_json(js), fh, indent=1)
    paths["summary_json"] = os.path.join(out_dir, "summary.json")
    if marks:
        write_marks(res, os.path.join(out_dir, "viewer_marks"))
        paths["viewer_marks"] = os.path.join(out_dir, "viewer_marks")
    if xlsx:
        try:
            paths["xlsx"] = write_xlsx(res, os.path.join(out_dir, "results.xlsx"))
        except ImportError:
            log("openpyxl is not installed: Excel export skipped (pip install openpyxl)")
    if figure:
        try:
            from .report import overview_figure
            paths["figure"] = overview_figure(res, os.path.join(out_dir, "overview.png"))
        except Exception as ex:   # a figure must never break the export
            log(f"overview figure failed: {ex}")
    log(f"Results written to {out_dir}")
    return paths


def write_xlsx(res, path: str) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook(write_only=True)
    bold = Font(bold=True)

    def sheet(title, header, rows, widths=None):
        ws = wb.create_sheet(title)
        if widths:
            for i, w in enumerate(widths):
                ws.column_dimensions[chr(65 + i) if i < 26 else "A"].width = w
        from openpyxl.cell import WriteOnlyCell
        hc = []
        for h in header:
            c = WriteOnlyCell(ws, value=h)
            c.font = bold
            hc.append(c)
        ws.append(hc)
        n = 0
        for r in rows:
            if n >= EXCEL_MAX_ROWS:
                ws.append(["(truncated: see the CSV file for all rows)"])
                break
            ws.append([_fmt(v) for v in r])
            n += 1

    srows = summary_rows(res)
    keys = list(srows[0]) if srows else []
    # summary transposed: one row per metric, one column per series
    sheet("Summary", ["metric"] + [r["series"] for r in srows],
          ([k] + [r.get(k) for r in srows] for k in keys if k != "series"), [28, 16, 16])
    sheet("Settings", ["parameter", "value"], settings_rows(res), [28, 60])
    k, r = _dicts_to_rows(res.files)
    sheet("Files", k, r)
    if res.epochs:
        k, r = _dicts_to_rows(res.epochs)
        sheet("HRV epochs", k, r)
    if res.hourly:
        k, r = _dicts_to_rows(res.hourly)
        sheet("HRV hourly", k, r)
    pr = pause_rows(res)
    k, r = _dicts_to_rows(pr) if pr else (EMPTY_HEADERS["pauses"], [])
    sheet("Pauses", k, r)
    h, rows = beat_table(res)
    sheet("Beats", h, rows())
    if res.pwaves is not None:
        h, rows = pwave_table(res)
        sheet("P waves", h, rows())
    qr = quality_rows(res)
    k, r = _dicts_to_rows(qr) if qr else (["(none)"], [])
    sheet("Signal quality", k, r)
    sheet("Column guide", ["column", "meaning"], sorted(COLUMN_GUIDE.items()), [24, 110])
    wb.save(path)
    return path


def export_signal_csv(path_edf: str, out_csv: str, start_s: float = 0.0, duration_s: Optional[float] = None,
                      keys: Optional[Sequence[str]] = None, filter_spec=None) -> str:
    """Export (optionally filtered) samples of an EDF time window to CSV: time_s, datetime, one column
    per signal. Signals must share a sample rate."""
    from .filters import apply_filters
    from .io.edf import EdfFile

    e = EdfFile(path_edf)
    keys = list(keys) if keys else (e.ecg_labels() or e.labels)
    x, fs, labels = e.read_matrix(keys, start_s, duration_s)
    if filter_spec is not None:
        x = np.vstack([apply_filters(v, fs, filter_spec) for v in x])
    t = start_s + np.arange(x.shape[1]) / fs
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        unit = e.signal(keys[0]).unit
        w.writerow(["time_s", "datetime"] + [f"{l} ({unit})" for l in labels])
        for i in range(x.shape[1]):
            dt = (e.start + timedelta(seconds=float(t[i]))).isoformat(sep=" ", timespec="milliseconds")
            w.writerow([f"{t[i]:.4f}", dt] + ["" if not np.isfinite(v) else f"{v:.6f}" for v in x[:, i]])
    return out_csv
