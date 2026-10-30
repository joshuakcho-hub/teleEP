"""End-to-end analysis of telemetry ECG stored as EDF.

    from teleep.analysis import AnalysisSettings, analyze
    result = analyze(["P10_seg01_part001.edf", ...], AnalysisSettings(species="pig"))

Consecutive files (for example the hourly files written by the converter) are treated as one
recording of one subject: beats from all files are placed on a common time axis, RR intervals
are computed across file boundaries when the files are contiguous, and gaps between files are
excluded. Files written by TeleEP state their UTC offset, so they are placed in UTC and a change
to or from daylight saving time does not shift the time axis. Long files are processed in chunks
with overlap, so memory use stays bounded.
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .detect import pigecg
from .detect.rpeaks import detect_rpeaks
from .filters import FilterSpec, apply_filters
from .hrv import HRVSettings, epoch_table, hourly_table, recording_summary
from .io.edf import EdfFile
from .rr import IntervalSeries, build_intervals, find_pauses, mask_to_segments, merge_segments
from .species import get_preset

Log = Callable[[str], None]
PIGECG_MIN_PP_MS = 180.0       # refractory period between P waves (pigecg)
GAP_TOLERANCE_S = 0.1          # larger gaps between consecutive files are excluded
UNANALYZABLE_QRS = ("unanalyzable QRS", "dropout", "no signal", "detector error")


@dataclass
class AnalysisSettings:
    """All parameters of an analysis run. ``None`` means "use the species preset"."""

    species: str = "pig"
    detector: Optional[str] = None            # "pigecg" | "generic"
    leads: Optional[List[str]] = None         # ECG signal labels to use (default: all ECG signals)
    prefilter: Optional[FilterSpec] = None    # optional filter applied before detection
    detect_p_waves: bool = True               # pigecg only
    min_rr_ms: Optional[float] = None
    qrs_band_hz: Optional[Tuple[float, float]] = None
    rr_range_ms: Optional[Tuple[float, float]] = None
    pp_range_ms: Optional[Tuple[float, float]] = (150.0, 3000.0)
    artifact_threshold: Optional[float] = 0.2
    min_confidence: float = 0.5               # pigecg beats below this are marked unreliable
    pause_ms: Optional[float] = None
    epoch_s: float = 300.0
    align_epochs_to_clock: bool = True
    min_epoch_coverage: float = 0.8
    min_valid_fraction: float = 0.7
    spectral_method: str = "welch"
    bands: Optional[Dict[str, Tuple[float, float]]] = None
    pnn_ms: Optional[Tuple[float, ...]] = None
    nonlinear: bool = True
    chunk_s: float = 3600.0
    overlap_s: float = 10.0
    use_trained_model: bool = True            # pigecg model from EP Annotator, if present

    def resolved(self) -> dict:
        p = get_preset(self.species)
        d = asdict(self)
        d["detector"] = self.detector or p.detector
        d["min_rr_ms"] = self.min_rr_ms or p.min_rr_ms
        d["qrs_band_hz"] = tuple(self.qrs_band_hz or p.qrs_band_hz)
        d["rr_range_ms"] = tuple(self.rr_range_ms or p.rr_range_ms)
        d["pause_ms"] = self.pause_ms if self.pause_ms is not None else p.pause_ms
        d["bands"] = dict(self.bands or p.bands)
        d["pnn_ms"] = tuple(self.pnn_ms or p.pnn_ms)
        d["qrs_width_ms"] = p.qrs_width_ms
        d["interp_hz"] = p.interp_hz
        d["welch_window_s"] = p.welch_window_s
        d["band_source"] = p.band_source if self.bands is None else "user"
        d["prefilter"] = self.prefilter.as_dict() if self.prefilter else None
        return d

    def hrv_settings(self) -> HRVSettings:
        r = self.resolved()
        return HRVSettings(bands=r["bands"], pnn_ms=r["pnn_ms"], spectral_method=self.spectral_method,
                           interp_hz=r["interp_hz"], welch_window_s=r["welch_window_s"],
                           nonlinear=self.nonlinear)


@dataclass
class AnalysisResult:
    settings: dict
    start: datetime
    duration_s: float
    files: List[dict]
    beats: Dict[str, np.ndarray]
    pwaves: Optional[Dict[str, np.ndarray]]
    bad_segments: List[Tuple[float, float, str]]
    rr: IntervalSeries
    pp: Optional[IntervalSeries]
    epochs: List[dict]
    summary: Dict[str, dict]
    hourly: List[dict]
    pauses: List[dict]
    av: dict = field(default_factory=dict)
    software: str = f"TeleEP {__version__}"
    start_utc: Optional[datetime] = None
    # (offset in s from the recording start, local start time) of every file: local clock times
    # are computed from the file that contains them, which keeps them correct across a change
    # to or from daylight saving time
    clock_marks: List[Tuple[float, datetime]] = field(default_factory=list)

    def datetime_at(self, t_s) -> datetime:
        """Local date and time of a point ``t_s`` seconds after the recording start."""
        t = float(t_s)
        if self.clock_marks:
            i = bisect.bisect_right([o for o, _ in self.clock_marks], t) - 1
            if i >= 0:
                off, st = self.clock_marks[i]
                return st + timedelta(seconds=t - off)
        return self.start + timedelta(seconds=t)


# ----------------------------------------------------------------------------- per file
def _load_model(edf_path: str):
    p = os.path.join(os.path.dirname(edf_path), "annotations", "pigecg_model.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def detect_file(path: str, cfg: dict) -> dict:
    """Run the detector on one EDF file (chunked). Times are seconds from the file start."""
    t_start = time.time()
    e = EdfFile(path)
    leads = cfg["leads"] or e.ecg_labels()
    if not leads:
        raise ValueError(f"{os.path.basename(path)}: no ECG signal found (signals: {e.labels})")
    fs = e.signal(leads[0]).sample_rate
    dur = e.duration
    chunk, ov = cfg["chunk_s"], cfg["overlap_s"]
    model = _load_model(path) if (cfg["use_trained_model"] and cfg["detector"] == "pigecg") else None
    prefilter = FilterSpec(**cfg["prefilter"]) if cfg.get("prefilter") else None

    q_t, q_conf, q_ncc, q_morph, q_good = [], [], [], [], []
    p_t, p_ncc, p_good = [], [], []
    bad: List[Tuple[float, float, str]] = []
    errors: List[str] = []
    n_missing = 0.0
    a = 0.0
    while a < dur - 1e-9:
        b = min(dur, a + chunk)
        ra, rb = max(0.0, a - ov), min(dur, b + ov)
        x, _, _ = e.read_matrix(leads, ra, rb - ra)
        if prefilter is not None:
            x = np.vstack([apply_filters(v, fs, prefilter) for v in x])
        miss_all = ~np.isfinite(x).any(axis=0)
        n_missing += float(np.sum(miss_all[int((a - ra) * fs):int((b - ra) * fs)])) / fs
        if np.isfinite(x).sum() < fs * 5:
            bad.append((a, b, "no signal"))
            a = b
            continue
        try:
            c = _detect_chunk(x, fs, ra, a, b, cfg, model)
        except Exception as ex:   # a failure in one chunk must not stop a long analysis
            errors.append(f"{a:.0f}-{b:.0f} s: {type(ex).__name__}: {ex}")
            c = dict(bad=[(a, b, "detector error")])
        if "q_t" in c:
            q_t.append(c["q_t"]); q_conf.append(c["q_conf"]); q_ncc.append(c["q_ncc"])
            q_morph.append(c["q_morph"]); q_good.append(c["q_good"])
        if "p_t" in c:
            p_t.append(c["p_t"]); p_ncc.append(c["p_ncc"]); p_good.append(c["p_good"])
        bad += c["bad"]
        for s0, s1 in mask_to_segments(miss_all, fs, ra):
            bad.append((max(s0, a), min(s1, b), "dropout"))
        a = b
    bad = [(s0, s1, why) for s0, s1, why in bad if s1 > s0]
    cat = lambda L, dt=float: np.concatenate(L) if L else np.zeros(0, dt)
    out = dict(path=path, file=os.path.basename(path), start=e.start, start_utc=e.start_utc,
               subject=subject_of(e), duration=dur, fs=fs, leads=list(leads),
               q_t=cat(q_t), q_conf=cat(q_conf), q_ncc=cat(q_ncc), q_morph=cat(q_morph, int),
               q_good=cat(q_good, bool), p_t=cat(p_t), p_ncc=cat(p_ncc), p_good=cat(p_good, bool),
               bad=bad, errors=errors, dropout_s=round(n_missing, 2),
               elapsed_s=round(time.time() - t_start, 1))
    # de-duplicate beats and P waves detected twice where chunks overlap
    _dedup(out, ("q_t", "q_conf", "q_ncc", "q_morph", "q_good"), cfg["min_rr_ms"] / 2000.0)
    _dedup(out, ("p_t", "p_ncc", "p_good"), PIGECG_MIN_PP_MS / 2000.0)
    return out


def subject_of(e: EdfFile) -> str:
    """Subject identifier from the EDF patient field (first word, the EDF+ patient code)."""
    words = (e.patient or "").split()
    return words[0] if words and words[0].upper() != "X" else ""


def _dedup(out: dict, keys, min_sep_s: float) -> None:
    t = out[keys[0]]
    if len(t) < 2:
        return
    o = np.argsort(t, kind="stable")
    for k in keys:
        out[k] = out[k][o]
    dup = np.concatenate(([False], np.diff(out[keys[0]]) < min_sep_s))
    for k in keys:
        out[k] = out[k][~dup]


def _detect_chunk(x, fs, ra, a, b, cfg, model) -> dict:
    """Detect beats in one chunk read from ``ra`` s; keep those in [a, b). All results are
    returned together, so that a failure leaves nothing half-recorded."""
    keep = lambda t: (t >= a) & (t < b)
    out = dict(bad=[])
    if cfg["detector"] == "pigecg":
        R = pigecg.run(x, fs, verbose=False, model=model, min_rr_ms=cfg["min_rr_ms"],
                       min_pp_ms=PIGECG_MIN_PP_MS)
        tq = R["qrs"] / fs + ra
        k = keep(tq)
        conf = np.asarray(R["qrs_conf"], float)
        out.update(q_t=tq[k], q_conf=conf[k], q_ncc=np.asarray(R["qrs_ncc"], float)[k],
                   q_morph=np.asarray(R["morph"])[k],
                   q_good=(np.asarray(R["qrs_good"], bool) & (conf >= cfg["min_confidence"]))[k])
        if cfg["detect_p_waves"] and len(R["p"]):
            tp = R["p"] / fs + ra
            kp = keep(tp)
            out.update(p_t=tp[kp], p_ncc=np.asarray(R["p_ncc"], float)[kp],
                       p_good=np.asarray(R["p_good"], bool)[kp])
        for s0, s1 in mask_to_segments(~R["quality"], fs, ra):
            out["bad"].append((max(s0, a), min(s1, b), "unanalyzable QRS"))
        if cfg["detect_p_waves"]:
            for s0, s1 in mask_to_segments(~R["quality_p"], fs, ra):
                out["bad"].append((max(s0, a), min(s1, b), "unanalyzable P"))
    else:
        R = detect_rpeaks(x, fs, band=cfg["qrs_band_hz"], qrs_width_ms=cfg["qrs_width_ms"],
                          min_rr_ms=cfg["min_rr_ms"])
        tq = R["peaks"] / fs + ra
        k = keep(tq)
        ncc = np.asarray(R["ncc"], float)
        # NaN: similarity not computable (beat at the edge of the signal or fewer than three
        # beats); such beats are not penalized
        good = ~(ncc < cfg["min_confidence"])
        out.update(q_t=tq[k], q_conf=ncc[k], q_ncc=ncc[k], q_morph=np.zeros(int(k.sum()), int),
                   q_good=good[k])
    return out


# ----------------------------------------------------------------------------- recording
def find_edf_files(inputs: Sequence[str]) -> List[str]:
    files: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*.edf")) + glob.glob(os.path.join(p, "*.EDF"))
        elif os.path.exists(p):
            files.append(p)
        else:
            files += glob.glob(p)
    return sorted(set(files))


def analyze(inputs: Sequence[str], settings: Optional[AnalysisSettings] = None, workers: int = 1,
            log: Log = print) -> AnalysisResult:
    """Analyze one or more EDF files as one recording (files sorted by start time)."""
    settings = settings or AnalysisSettings()
    cfg = settings.resolved()
    paths = find_edf_files(inputs)
    if not paths:
        subs = sorted({os.path.relpath(os.path.dirname(f), d) for d in inputs if os.path.isdir(d)
                       for f in glob.glob(os.path.join(d, "**", "*.[eE][dD][fF]"), recursive=True)})
        hint = (f" Subfolders with EDF files: {', '.join(subs)}. Analyze one subject folder at a time."
                if subs else "")
        raise FileNotFoundError(f"No EDF files found in {', '.join(map(str, inputs))}.{hint}")
    starts, subjects = [], {}
    for p in paths:
        with EdfFile(p) as e:
            utc = e.start_utc
            starts.append((utc if utc is not None else e.start, p))
            subjects.setdefault(subject_of(e), []).append(os.path.basename(p))
    named = {k: v for k, v in subjects.items() if k}
    if len(named) > 1:
        raise ValueError(
            "The input contains recordings of more than one subject ("
            + ", ".join(f"{k}: {len(v)} file(s)" for k, v in sorted(named.items()))
            + "). Analyze each subject separately; the converter writes one subfolder per subject.")
    paths = [p for _, p in sorted(starts)]
    log(f"Analyzing {len(paths)} file(s) | species {settings.species} | detector {cfg['detector']}")

    results: List[dict] = []
    if workers and workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(detect_file, paths, [cfg] * len(paths)):
                log(_file_line(r))
                results.append(r)
    else:
        for p in paths:
            r = detect_file(p, cfg)
            log(_file_line(r))
            results.append(r)
    return assemble(results, settings, log)


def _file_line(r: dict) -> str:
    rate = ""
    if r.get("errors"):
        rate = f" | DETECTOR ERROR in {len(r['errors'])} chunk(s): {r['errors'][0]}"
    if len(r["q_t"]) > 2:
        rate = f" | ventricular {60 / np.median(np.diff(r['q_t'])):.0f} bpm"
    if len(r["p_t"]) > 2:
        rate += f" | atrial {60 / np.median(np.diff(r['p_t'])):.0f} bpm"
    return (f"  {r['file']}: {len(r['q_t'])} QRS, {len(r['p_t'])} P{rate} | dropout {r['dropout_s']:g} s "
            f"| {r['elapsed_s']} s")


def assemble(results: List[dict], settings: AnalysisSettings, log: Log = print) -> AnalysisResult:
    cfg = settings.resolved()
    # place files in UTC when every file states its UTC offset (files written by TeleEP)
    use_utc = all(r.get("start_utc") is not None for r in results)
    key = (lambda r: r["start_utc"]) if use_utc else (lambda r: r["start"])
    results = sorted(results, key=key)
    rec_start = results[0]["start"]
    rec_start_utc = results[0]["start_utc"] if use_utc else None
    offsets = [(key(r) - key(results[0])).total_seconds() for r in results]
    files, bad, clock_marks = [], [], []
    Q = {k: [] for k in ("time_s", "file_index", "confidence", "ncc", "morphology", "good")}
    P = {k: [] for k in ("time_s", "file_index", "ncc", "good")}
    prev_end = None
    for i, r in enumerate(results):
        off = offsets[i]
        if prev_end is not None and off > prev_end + GAP_TOLERANCE_S:
            bad.append((prev_end, off, "gap between files"))
        if prev_end is not None and off < prev_end - 1.0:
            log(f"  warning: {r['file']} overlaps the previous file by {prev_end - off:.1f} s")
        prev_end = max(prev_end or 0.0, off + r["duration"])
        clock_marks.append((off, r["start"]))
        # a file's padded end (last data record completed with missing samples) must not mark
        # the beginning of the next file as unanalyzable
        limit = offsets[i + 1] - off if i + 1 < len(results) else np.inf
        r_bad = [(a, min(b, limit), why) for a, b, why in r["bad"] if a < limit]
        Q["time_s"].append(r["q_t"] + off); Q["file_index"].append(np.full(len(r["q_t"]), i))
        Q["confidence"].append(r["q_conf"]); Q["ncc"].append(r["q_ncc"]); Q["morphology"].append(r["q_morph"])
        Q["good"].append(r["q_good"])
        P["time_s"].append(r["p_t"] + off); P["file_index"].append(np.full(len(r["p_t"]), i))
        P["ncc"].append(r["p_ncc"]); P["good"].append(r["p_good"])
        bad += [(a + off, b + off, why) for a, b, why in r_bad]
        n_unq = sum(b - a for a, b in merge_segments(
            [(a, b) for a, b, why in r["bad"] if why in UNANALYZABLE_QRS]))
        files.append(dict(file=r["file"], start=r["start"].isoformat(sep=" "), offset_s=round(off, 3),
                          duration_s=r["duration"], fs=r["fs"], leads=",".join(r["leads"]),
                          n_qrs=int(len(r["q_t"])), n_p=int(len(r["p_t"])), dropout_s=r["dropout_s"],
                          unanalyzable_s=round(min(n_unq, r["duration"]), 1), elapsed_s=r["elapsed_s"]))
    beats = {k: np.concatenate(v) if v else np.zeros(0) for k, v in Q.items()}
    pw = {k: np.concatenate(v) if v else np.zeros(0) for k, v in P.items()}
    o = np.argsort(beats["time_s"], kind="stable")
    beats = {k: v[o] for k, v in beats.items()}
    beats["good"] = beats["good"].astype(bool)
    duration = prev_end or 0.0

    q_bad = merge_segments([(a, b) for a, b, why in bad if why != "unanalyzable P"])
    p_bad = merge_segments([(a, b) for a, b, why in bad])
    rr = build_intervals(beats["time_s"], kind="RR", bad_segments=q_bad, beat_ok=beats["good"],
                         range_ms=cfg["rr_range_ms"], artifact_threshold=settings.artifact_threshold)
    hs = settings.hrv_settings()
    clock = _clock(clock_marks, rec_start)
    epochs = epoch_table(rr, duration, hs, settings.epoch_s, rec_start, settings.min_epoch_coverage,
                         settings.align_epochs_to_clock, settings.min_valid_fraction, clock=clock)
    summary = {"RR": recording_summary(rr, epochs, hs)}
    hourly = hourly_table(epochs, rec_start, clock=clock)
    pauses = find_pauses(rr, cfg["pause_ms"])

    pp = None
    pwaves = None
    av: dict = {}
    if cfg["detector"] == "pigecg" and settings.detect_p_waves and len(pw["time_s"]) > 2:
        o = np.argsort(pw["time_s"], kind="stable")
        pwaves = {k: v[o] for k, v in pw.items()}
        pwaves["good"] = pwaves["good"].astype(bool)
        pp = build_intervals(pwaves["time_s"], kind="PP", bad_segments=p_bad, beat_ok=pwaves["good"],
                             range_ms=settings.pp_range_ms or (0, np.inf),
                             artifact_threshold=settings.artifact_threshold)
        pe = epoch_table(pp, duration, hs, settings.epoch_s, rec_start, settings.min_epoch_coverage,
                         settings.align_epochs_to_clock, settings.min_valid_fraction, clock=clock)
        epochs += pe
        summary["PP"] = recording_summary(pp, pe, hs)
        hourly += hourly_table(pe, rec_start, clock=clock)
        pauses += find_pauses(pp, cfg["pause_ms"])
        av = av_association(beats["time_s"], pwaves["time_s"])
        beats["pr_ms"] = av.pop("pr_ms")
    res = AnalysisResult(settings=cfg, start=rec_start, duration_s=duration, files=files, beats=beats,
                         pwaves=pwaves, bad_segments=[(a, b, w) for a, b, w in bad], rr=rr, pp=pp,
                         epochs=epochs, summary=summary, hourly=hourly, pauses=pauses, av=av,
                         start_utc=rec_start_utc, clock_marks=clock_marks)
    s = summary["RR"]
    log(f"Recording: {duration / 3600:.2f} h | {len(beats['time_s'])} QRS | NN {s['n_nn']} "
        f"({100 * s['valid_fraction']:.1f} % of RR) | mean HR {s['mean_hr_bpm']:.1f} bpm | "
        f"SDNN {s['sdnn_ms']:.1f} ms | RMSSD {s['rmssd_ms']:.1f} ms | pauses {len(pauses)}")
    if pp is not None:
        sp = summary["PP"]
        log(f"Atrial: {len(pwaves['time_s'])} P | mean atrial rate {sp['mean_hr_bpm']:.1f} /min | "
            f"P:QRS ratio {av.get('p_to_qrs_ratio', float('nan')):.2f} | "
            f"QRS preceded by P (60-300 ms) {100 * av.get('frac_qrs_with_p_60_300ms', float('nan')):.0f} %")
    return res


def _clock(marks: List[Tuple[float, datetime]], start: datetime):
    offs = [o for o, _ in marks]

    def at(t: float) -> datetime:
        i = bisect.bisect_right(offs, t) - 1
        if i < 0:
            return start + timedelta(seconds=float(t))
        return marks[i][1] + timedelta(seconds=float(t) - marks[i][0])
    return at


def av_association(q: np.ndarray, p: np.ndarray, pr_min=50.0, pr_max=450.0) -> dict:
    """Descriptive atrioventricular statistics. PR = time from the closest preceding P wave
    (within 50-450 ms) to each QRS. This is not a claim that the P wave was conducted."""
    pr = np.full(len(q), np.nan)
    if len(p) and len(q):
        j = np.searchsorted(p, q) - 1
        ok = j >= 0
        d = (q[ok] - p[j[ok]]) * 1000
        v = np.where((d >= pr_min) & (d <= pr_max), d, np.nan)
        pr[ok] = v
    in_win = (pr >= 60) & (pr <= 300)
    pr_ok = pr[in_win]
    jumps = np.abs(np.diff(pr_ok)) if len(pr_ok) > 10 else np.zeros(0)
    return dict(
        n_qrs=int(len(q)), n_p=int(len(p)),
        p_to_qrs_ratio=float(len(p) / len(q)) if len(q) else float("nan"),
        frac_qrs_with_p_60_300ms=float(np.mean(in_win)) if len(q) else float("nan"),
        pr_median_ms=float(np.median(pr_ok)) if len(pr_ok) else float("nan"),
        pr_iqr_ms=float(np.subtract(*np.percentile(pr_ok, [75, 25]))) if len(pr_ok) > 5 else float("nan"),
        frac_pr_change_gt40ms=float(np.mean(jumps > 40)) if len(jumps) else float("nan"),
        pr_ms=pr)
