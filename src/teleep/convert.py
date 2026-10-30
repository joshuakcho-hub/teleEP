"""Convert DSI Ponemah experiments to EDF / EDF+.

* Each channel is stored losslessly: the 16-bit Ponemah counts are written as the EDF digital
  values, and the EDF physical range reproduces Ponemah's calibration exactly.
* Telemetry dropouts (raw -32768) are written as the digital minimum and listed as EDF+
  annotations ("Signal dropout <channel>"), so no software mistakes them for real signal.
* Acquisition gaps are recovered from the block time stamps in the .PnmWav file. One set of
  files is written per uninterrupted segment, each with its true start time (to the
  microsecond, kept in the EDF+ time-keeping annotation).
* Start times are local (the acquisition computer's time zone, stored in the file) unless
  ``time_reference='utc'``. The UTC offset is written in the EDF recording field
  (for example "UTC-04:00"), so files can be placed on one time line across daylight saving
  time changes.
* An experiment with several subjects is written to one subfolder per subject.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from . import __version__
from .io.edf import DROPOUT_TEXT, EdfSignal, format_utc_offset, write_edf
from .io.ponemah import (MISSING, TICKS_PER_SECOND, ChannelInfo, PonemahRecording, Segment,
                         is_experiment_folder, missing_runs, open_experiment, ticks_to_utc,
                         warn_unknown_timezone)

Log = Callable[[str], None]


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s)).strip("_") or "subject"


def _select(rec: PonemahRecording, channels: Optional[Sequence[str]], ecg_only: bool) -> List[ChannelInfo]:
    chans = rec.channels
    if ecg_only:
        chans = [c for c in chans if c.is_ecg]
    if channels:
        want = {c.lower() for c in channels}
        chans = [c for c in chans if c.label.lower() in want or str(c.channel_id) in want]
    return [c for c in chans if c.sample_rate > 0]


def _edf_signal(ch: ChannelInfo, raw: np.ndarray) -> EdfSignal:
    d = raw.astype(np.int32)
    dmin, pmin = ch.digital_min, ch.physical_min
    if not ch.is_ecg and dmin > MISSING:
        # Auxiliary channels (temperature, battery, activity) can legitimately sit at the bottom
        # of the header range (an activity count of 0 is the raw value -32256). The digital range
        # is therefore extended down to -32768 with the same calibration, so that only telemetry
        # dropout (raw -32768) is read back as missing.
        pmin = ch.physical_min - (dmin - MISSING) * ch.gain
        dmin = MISSING
    d[raw == MISSING] = dmin
    return EdfSignal(label=ch.label, data=d, sample_rate=ch.sample_rate, unit=ch.unit,
                     physical_min=pmin, physical_max=ch.physical_max,
                     digital_min=dmin, digital_max=ch.digital_max, digital=True,
                     transducer="DSI telemetry implant",
                     prefilter="none (raw Ponemah samples)")


def _dropout_annotations(ch: ChannelInfo, raw: np.ndarray) -> List[tuple]:
    fs = ch.sample_rate
    return [(a / fs, (b - a) / fs, f"{DROPOUT_TEXT} {ch.label}") for a, b in missing_runs(raw)]


def _chunks(start: int, end: int, chunk_s: Optional[float], min_last_fraction: float = 0.25):
    """Split [start, end) ticks into files of ``chunk_s`` seconds. A final piece shorter than
    ``min_last_fraction`` of a file is appended to the previous file instead of standing alone."""
    if not chunk_s or chunk_s <= 0:
        return [(start, end)]
    step = int(round(chunk_s * TICKS_PER_SECOND))
    out = []
    t = start
    while t < end:
        out.append([t, min(end, t + step)])
        t += step
    if len(out) > 1 and (out[-1][1] - out[-1][0]) < min_last_fraction * step:
        last = out.pop()
        out[-1][1] = last[1]
    return [tuple(x) for x in out]


def convert_recording(rec: PonemahRecording, out_dir: str, *, channels=None, ecg_only=False,
                      chunk_minutes: Optional[float] = 60, max_seconds: Optional[float] = None,
                      time_reference: str = "local", annotate_dropouts: bool = True,
                      export_ponemah_marks: bool = False, log: Log = print) -> List[str]:
    """Convert one subject recording; returns the list of EDF paths written.

    One set of files is written per acquisition segment, split into files of ``chunk_minutes``.
    ``max_seconds`` limits the total duration converted (across all segments).
    """
    os.makedirs(out_dir, exist_ok=True)
    warn_unknown_timezone(rec)
    chans = _select(rec, channels, ecg_only)
    if not chans:
        log(f"  {rec.subject}: no matching channels, skipped")
        return []
    if not rec.segments:
        log(f"  {rec.subject}: no data, skipped")
        return []
    use_local = time_reference == "local" and rec.tzinfo is not None
    clock = (lambda tk: rec.to_local(tk)) if use_local else (lambda tk: ticks_to_utc(tk).replace(tzinfo=None))
    tzlabel = "local" if use_local else "UTC"
    subj = _safe(rec.subject)
    written: List[str] = []
    files_meta = []
    chunk_s = chunk_minutes * 60 if chunk_minutes else None

    # segments, cut to the total duration requested
    segs = []
    budget = int(max_seconds * TICKS_PER_SECOND) if max_seconds else None
    for seg in rec.segments:
        end = seg.end
        if budget is not None:
            if budget <= 0:
                break
            end = min(end, seg.start + budget)
            budget -= end - seg.start
        segs.append((seg.start, end))

    for si, (s0, s1) in enumerate(segs):
        ranges = _chunks(s0, s1, chunk_s)
        for ci, (t0, t1) in enumerate(ranges):
            sigs, anns, n_drop = [], [], {}
            for c in chans:
                raw = rec.read_raw(c, t0, t1)
                if annotate_dropouts and c.is_ecg:
                    anns += _dropout_annotations(c, raw)
                if c.is_ecg:
                    n_drop[c.label] = round(float(np.sum(raw == MISSING)) / c.sample_rate, 2)
                if raw.size:
                    sigs.append(_edf_signal(c, raw))
            if not sigs:
                continue
            start = clock(t0)
            fn = f"{subj}_seg{si + 1:02d}_part{ci + 1:03d}_{start:%Y%m%d_%H%M%S}.edf"
            path = os.path.join(out_dir, fn)
            tz = format_utc_offset(rec.utc_offset(t0)) if use_local else "UTC"
            write_edf(path, sigs, start=start, patient_code=subj, patient_name=rec.subject,
                      equipment="DSI_Ponemah",
                      recording_info=f"TeleEP_{__version__} {tz} seg{si + 1}/{len(segs)} part{ci + 1}/{len(ranges)}",
                      annotations=anns, force_edfplus=True, pad_last_record=True)
            written.append(path)
            files_meta.append(dict(file=fn, segment=si + 1, part=ci + 1, start=start.isoformat(),
                                   utc_offset=tz, start_utc=ticks_to_utc(t0).isoformat(),
                                   duration_s=round((t1 - t0) / TICKS_PER_SECOND, 3),
                                   dropout_s=n_drop))
            drop_txt = ", ".join(f"{k} {v:g} s" for k, v in n_drop.items() if v) or "none"
            log(f"  wrote {fn}  (dropout: {drop_txt})")

    first = clock(segs[0][0]) if segs else None
    stem = f"{subj}_{first:%Y%m%d_%H%M%S}" if first else subj
    meta = dict(software=f"TeleEP {__version__}", converted=datetime.now().isoformat(timespec="seconds"),
                source=rec.summary(), time_reference=tzlabel, files=files_meta)
    with open(os.path.join(out_dir, f"{stem}_conversion.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    if export_ponemah_marks:
        export_marks_csv(rec, os.path.join(out_dir, f"{stem}_ponemah_beats.csv"), clock)
        log(f"  wrote {stem}_ponemah_beats.csv")
    return written


def export_marks_csv(rec: PonemahRecording, path: str, clock=None) -> str:
    """Ponemah's own beat marks (from .PnmMarks) as CSV: channel, UTC time, local time."""
    clock = clock or rec.to_local
    beats = rec.ponemah_beats()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "time_utc", "time_local", "ticks"])
        for lab, ticks in beats.items():
            for tk in ticks.tolist():
                w.writerow([lab, ticks_to_utc(tk).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            clock(tk).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], tk])
    return path


def convert_experiment(folder: str, out_dir: str, *, subjects: Optional[Iterable[str]] = None,
                       log: Log = print, **kw) -> List[str]:
    """Convert every subject recording in a Ponemah experiment folder."""
    recs = open_experiment(folder)
    if subjects:
        keep = {s.lower() for s in subjects}
        recs = [r for r in recs if r.subject.lower() in keep]
    log(f"Experiment: {folder}  ({len(recs)} recording(s))")
    # several subjects: one subfolder per subject, so that each folder can be analyzed as one animal
    per_subject = len({_safe(r.subject) for r in recs}) > 1
    out: List[str] = []
    for r in recs:
        segs = ", ".join(f"{r.to_local(s.start):%Y-%m-%d %H:%M:%S} ({s.duration_s / 3600:.2f} h)" for s in r.segments)
        log(f"Subject {r.subject}: {len(r.channels)} channel(s) "
            f"[{', '.join(f'{c.label} {c.sample_rate:g} Hz' for c in r.channels)}]; segments: {segs}")
        dest = os.path.join(out_dir, _safe(r.subject)) if per_subject else out_dir
        out += convert_recording(r, dest, log=log, **kw)
    log(f"Done. {len(out)} EDF file(s) in {out_dir}" + (" (one subfolder per subject)" if per_subject else ""))
    return out


def convert_batch(parent: str, out_dir: str, log: Log = print, **kw) -> List[str]:
    """Convert every experiment folder found directly under ``parent``."""
    folders = sorted(os.path.join(parent, d) for d in os.listdir(parent)
                     if is_experiment_folder(os.path.join(parent, d)))
    if is_experiment_folder(parent):
        folders.insert(0, parent)
    log(f"Batch: {len(folders)} experiment folder(s) under {parent}")
    out: List[str] = []
    for d in folders:
        out += convert_experiment(d, os.path.join(out_dir, _safe(os.path.basename(d))), log=log, **kw)
    return out
