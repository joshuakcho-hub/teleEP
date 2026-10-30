"""Compare TeleEP detections with a reference: Ponemah's own beat marks or user annotations.

Sensitivity (Se) = TP / (TP + FN), positive predictivity (PPV) = TP / (TP + FP). A detection
counts as a true positive when it lies within ``tol_s`` of a reference mark. Only signal that
TeleEP considers analyzable is scored, so dropouts and noise do not count against either side.
"""
from __future__ import annotations

import csv
import os
from datetime import timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io.ponemah import TICKS_PER_SECOND, PonemahRecording, open_experiment, utc_to_ticks


def match(det: np.ndarray, ref: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """(detection matched?, reference matched?) with nearest-neighbour matching in time."""
    det, ref = np.asarray(det, float), np.asarray(ref, float)
    if len(det) == 0 or len(ref) == 0:
        return np.zeros(len(det), bool), np.zeros(len(ref), bool)
    j = np.clip(np.searchsorted(ref, det), 1, len(ref) - 1)
    near = np.minimum(np.abs(det - ref[j - 1]), np.abs(det - ref[j]))
    dm = near <= tol
    i = np.clip(np.searchsorted(det, ref), 1, len(det) - 1)
    near_r = np.minimum(np.abs(ref - det[i - 1]), np.abs(ref - det[i]))
    rm = near_r <= tol
    return dm, rm


def in_segments(t: np.ndarray, segs: Sequence[Tuple[float, float]]) -> np.ndarray:
    t = np.asarray(t, float)
    m = np.zeros(len(t), bool)
    if not len(segs):
        return m
    s = np.array(sorted(segs), float)
    j = np.searchsorted(s[:, 0], t, side="right") - 1
    ok = j >= 0
    m[ok] = t[ok] < np.maximum.accumulate(s[:, 1])[j[ok]]
    return m


def score(det, ref, tol: float, exclude: Sequence[Tuple[float, float]] = (),
          include: Optional[Sequence[Tuple[float, float]]] = None) -> dict:
    det, ref = np.sort(np.asarray(det, float)), np.sort(np.asarray(ref, float))
    if include is not None:
        det, ref = det[in_segments(det, include)], ref[in_segments(ref, include)]
    if len(exclude):
        det, ref = det[~in_segments(det, exclude)], ref[~in_segments(ref, exclude)]
    dm, rm = match(det, ref, tol)
    tp, fp, fn = int(dm.sum()), int((~dm).sum()), int((~rm).sum())
    return dict(n_detected=len(det), n_reference=len(ref), tp=tp, fp=fp, fn=fn,
                sensitivity=tp / max(1, tp + fn), ppv=tp / max(1, tp + fp),
                fp_times=det[~dm], fn_times=ref[~rm])


def compare_with_ponemah(result, experiment: str, tol_s: float = 0.12,
                         out_dir: Optional[str] = None) -> List[dict]:
    """Agreement of TeleEP QRS detections with Ponemah's beat marks, per hour and per lead.

    ``result`` is an :class:`~teleep.analysis.AnalysisResult`; ``experiment`` a Ponemah folder.
    """
    recs: List[PonemahRecording] = open_experiment(experiment) if isinstance(experiment, str) else [experiment]
    excl = [(a, b) for a, b, why in result.bad_segments if why != "unanalyzable P"]
    q = result.beats["time_s"]
    rows = []
    dis = []
    start_utc = getattr(result, "start_utc", None)
    for rec in recs:
        if start_utc is not None:
            # both time axes in UTC: exact, including across daylight saving time changes
            t0_ticks = utc_to_ticks(start_utc.replace(tzinfo=timezone.utc))
            to_s = lambda ticks: (np.asarray(ticks, np.int64) - t0_ticks) / TICKS_PER_SECOND
        else:
            # files without a stated UTC offset: assume they use the experiment's local clock
            to_s = lambda ticks: np.array([(rec.to_local(int(tk)) - result.start).total_seconds()
                                           for tk in ticks])
        for lab, ticks in rec.ponemah_beats().items():
            t_ref = to_s(ticks)
            t_ref = t_ref[(t_ref >= 0) & (t_ref <= result.duration_s)]
            hours = np.arange(0, result.duration_s + 3600, 3600)
            for h0 in hours[:-1]:
                inc = [(h0, min(h0 + 3600, result.duration_s))]
                s = score(q, t_ref, tol_s, excl, inc)
                if s["n_reference"] < 50:
                    continue
                rows.append(dict(subject=rec.subject, ponemah_lead=lab,
                                 hour_start=result.datetime_at(h0).isoformat(sep=" ", timespec="seconds"),
                                 teleep=s["n_detected"], ponemah=s["n_reference"], matched=s["tp"],
                                 teleep_only=s["fp"], ponemah_only=s["fn"],
                                 agreement_teleep=round(s["ppv"], 5), agreement_ponemah=round(s["sensitivity"], 5)))
                dis += [(float(t), lab, "teleep_only") for t in s["fp_times"]]
                dis += [(float(t), lab, "ponemah_only") for t in s["fn_times"]]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if rows:
            with open(os.path.join(out_dir, "ponemah_agreement.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        with open(os.path.join(out_dir, "ponemah_disagreements.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "datetime", "ponemah_lead", "category"])
            for t, lab, cat in sorted(dis):
                w.writerow([f"{t:.3f}", result.datetime_at(t).isoformat(sep=" ", timespec="milliseconds"), lab, cat])
    return rows


def best_lead_totals(rows: List[dict]) -> Dict[str, dict]:
    """Totals per Ponemah lead (useful to pick the lead on which Ponemah worked best)."""
    out: Dict[str, dict] = {}
    for r in rows:
        d = out.setdefault(r["ponemah_lead"], dict(matched=0, teleep_only=0, ponemah_only=0, teleep=0, ponemah=0))
        for k in d:
            d[k] += r[k]
    return out
