"""Command-line interface: ``teleep <command> ...`` (or ``python -m teleep``).

Commands
  info           describe a Ponemah experiment folder or an EDF file
  convert        Ponemah experiment -> EDF / EDF+
  analyze        detect beats, RR / PP intervals and HRV in EDF files; write CSV / Excel / JSON
  export-signal  export (filtered) samples of an EDF time window to CSV
  view           open the interactive viewer
  annotate       open the EP Annotator (review and correct beat marks)
  train          learn from saved annotator corrections
  demo           create a synthetic Ponemah experiment and run the whole pipeline on it
  gui            open the desktop application
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__


def _filter_spec(a):
    from .filters import FilterSpec
    if a.hp is None and a.lp is None and a.notch is None and not getattr(a, "baseline", False):
        return None
    return FilterSpec(highpass_hz=a.hp, lowpass_hz=a.lp, notch_hz=a.notch,
                      median_baseline=getattr(a, "baseline", False))


def _add_filter_args(p, defaults=(None, None)):
    p.add_argument("--hp", type=float, default=defaults[0], help="high-pass cut-off (Hz)")
    p.add_argument("--lp", type=float, default=defaults[1], help="low-pass cut-off (Hz)")
    p.add_argument("--notch", type=float, default=None, help="mains notch frequency (50 or 60 Hz)")
    p.add_argument("--baseline", action="store_true", help="median baseline-wander removal")


def cmd_info(a):
    import json
    from .io.edf import EdfFile
    from .io.ponemah import is_experiment_folder, open_experiment
    for path in a.paths:
        if os.path.isdir(path) and is_experiment_folder(path):
            for r in open_experiment(path):
                print(json.dumps(r.summary(), indent=1))
        elif path.lower().endswith(".edf"):
            e = EdfFile(path)
            drop = sum(b - a_ for a_, b, _ in e.dropouts())
            print(f"{os.path.basename(path)}  start {e.start}  duration {e.duration:.1f} s  "
                  f"({e.edf_type or 'EDF'})")
            print(f"  patient: {e.patient}\n  recording: {e.recording}")
            for s in e.data_signals:
                print(f"  {s.index:2d} {s.label:<16} {s.sample_rate:g} Hz  {s.unit:<6} "
                      f"phys [{s.physical_min:g}, {s.physical_max:g}]  dig [{s.digital_min}, {s.digital_max}]")
            print(f"  annotations: {len(e.annotations)} (dropout marks: {len(e.dropouts())}, {drop:.1f} s)")
        else:
            print(f"{path}: not a Ponemah experiment folder or EDF file")
    return 0


def cmd_convert(a):
    from .convert import convert_batch, convert_experiment
    kw = dict(channels=a.channels, ecg_only=a.ecg_only, chunk_minutes=a.chunk_minutes or None,
              max_seconds=a.max_seconds, time_reference="utc" if a.utc else "local",
              subjects=a.subjects, annotate_dropouts=not a.no_dropout_annotations, export_ponemah_marks=a.ponemah_marks)
    if a.batch:
        convert_batch(a.folder, a.out, **kw)
    else:
        convert_experiment(a.folder, a.out, **kw)
    return 0


def _parse_bands(items):
    if not items:
        return None
    bands = {}
    for name, lo, hi in items:
        bands[name.upper()] = (float(lo), float(hi))
    return bands


def cmd_analyze(a):
    from .analysis import AnalysisSettings, analyze
    from .export import export_all
    st = AnalysisSettings(
        species=a.species, detector=a.detector, leads=a.leads, prefilter=_filter_spec(a),
        detect_p_waves=not a.no_p, min_rr_ms=a.min_rr_ms,
        artifact_threshold=None if str(a.artifact).lower() in ("off", "none", "0") else float(a.artifact),
        min_confidence=a.min_confidence, pause_ms=a.pause_ms, epoch_s=a.epoch_min * 60,
        min_epoch_coverage=a.min_coverage, min_valid_fraction=a.min_valid, spectral_method=a.spectral, bands=_parse_bands(a.band),
        pnn_ms=tuple(a.pnn) if a.pnn else None, nonlinear=not a.no_nonlinear,
        align_epochs_to_clock=not a.no_clock_align)
    res = analyze(a.inputs, st, workers=a.workers)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.inputs[0])) if not os.path.isdir(a.inputs[0])
                                else os.path.abspath(a.inputs[0]), "teleep_results")
    export_all(res, out, xlsx=not a.no_xlsx, figure=not a.no_figure)
    if a.compare_ponemah:
        from .validate import best_lead_totals, compare_with_ponemah
        rows = compare_with_ponemah(res, a.compare_ponemah, out_dir=out)
        for lead, d in best_lead_totals(rows).items():
            print(f"Ponemah {lead}: matched {d['matched']}, TeleEP-only {d['teleep_only']}, "
                  f"Ponemah-only {d['ponemah_only']}")
    return 0


def cmd_export_signal(a):
    from .export import export_signal_csv
    export_signal_csv(a.edf, a.out, a.start, a.duration, a.leads, _filter_spec(a))
    print(f"wrote {a.out}")
    return 0


def viewer_path() -> str:
    return os.path.join(os.path.dirname(__file__), "viewer", "TeleEP_Viewer.html")


def cmd_view(a):
    from .viewer.launch import open_viewer
    open_viewer(native=a.window)
    return 0


def cmd_annotate(a):
    from .annotator.app import serve
    serve(a.folder, port=a.port, open_browser=not a.no_browser)
    return 0


def cmd_train(a):
    from .annotator.train import train
    train(a.folder, fit=not a.no_fit)
    return 0


def cmd_demo(a):
    from .analysis import AnalysisSettings, analyze
    from .convert import convert_experiment
    from .export import export_all
    from .synthetic import write_synthetic_experiment
    exp = os.path.join(a.out, "ponemah_experiment")
    edf = os.path.join(a.out, "edf")
    res_dir = os.path.join(a.out, "results")
    write_synthetic_experiment(exp, duration_s=a.minutes * 60, seed=a.seed)
    print(f"Synthetic Ponemah experiment written to {exp}")
    convert_experiment(exp, edf, chunk_minutes=a.chunk_minutes)
    res = analyze([edf], AnalysisSettings(species="pig"))
    export_all(res, res_dir)
    return 0


def cmd_gui(a):
    from .gui.app import main as gui_main
    gui_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="teleep", description="TeleEP: telemetry ECG conversion and analysis")
    p.add_argument("--version", action="version", version=f"TeleEP {__version__}")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("info", help="describe a Ponemah folder or EDF file")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("convert", help="convert a Ponemah experiment to EDF")
    s.add_argument("folder", help="experiment folder (.PnmWav + .PnmExp + .PnmMarks)")
    s.add_argument("-o", "--out", default="edf_out")
    s.add_argument("--batch", action="store_true", help="convert every experiment folder under FOLDER")
    s.add_argument("--subjects", nargs="*", help="subjects to convert (default: all)")
    s.add_argument("--channels", nargs="*", help="channel labels to export (default: all)")
    s.add_argument("--ecg-only", action="store_true")
    s.add_argument("--chunk-minutes", type=float, default=60, help="split into files of N minutes (0 = one file per segment)")
    s.add_argument("--max-seconds", type=float, default=None, help="convert only the first N seconds of each subject")
    s.add_argument("--utc", action="store_true", help="write UTC instead of local time")
    s.add_argument("--no-dropout-annotations", action="store_true")
    s.add_argument("--ponemah-marks", action="store_true", help="also export Ponemah's beat marks to CSV")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("analyze", help="detect beats and compute RR / PP intervals and HRV")
    s.add_argument("inputs", nargs="+", help="EDF files or folders")
    s.add_argument("-o", "--out", default=None, help="results folder (default: <input>/teleep_results)")
    s.add_argument("--species", default="pig", choices=["pig", "dog", "human", "rat", "mouse"])
    s.add_argument("--detector", choices=["pigecg", "generic"], default=None)
    s.add_argument("--leads", nargs="*", help="ECG signal labels to use (default: all ECG signals)")
    _add_filter_args(s)
    s.add_argument("--no-p", action="store_true", help="skip P-wave detection")
    s.add_argument("--min-rr-ms", type=float, default=None, help="refractory period (ms)")
    s.add_argument("--min-confidence", type=float, default=0.5)
    s.add_argument("--artifact", default="0.2", help="artifact threshold (fraction) or 'off'")
    s.add_argument("--pause-ms", type=float, default=None)
    s.add_argument("--epoch-min", type=float, default=5.0)
    s.add_argument("--min-coverage", type=float, default=0.8,
                   help="fraction of an epoch that must be analyzable signal (default 0.8)")
    s.add_argument("--min-valid", type=float, default=0.7,
                   help="fraction of an epoch's intervals that must be valid NN intervals (default 0.7)")
    s.add_argument("--no-clock-align", action="store_true", help="epochs start at the recording start")
    s.add_argument("--spectral", choices=["welch", "lomb"], default="welch")
    s.add_argument("--band", nargs=3, action="append", metavar=("NAME", "LOW", "HIGH"),
                   help="HRV band, e.g. --band LF 0.04 0.15 (repeat for each band)")
    s.add_argument("--pnn", type=float, nargs="*", help="pNNx thresholds in ms")
    s.add_argument("--no-nonlinear", action="store_true")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--no-xlsx", action="store_true")
    s.add_argument("--no-figure", action="store_true")
    s.add_argument("--compare-ponemah", metavar="FOLDER", help="compare QRS detections with Ponemah's marks")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("export-signal", help="export an EDF time window to CSV")
    s.add_argument("edf")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--start", type=float, default=0.0, help="start (s from file start)")
    s.add_argument("--duration", type=float, default=60.0, help="duration (s)")
    s.add_argument("--leads", nargs="*")
    _add_filter_args(s)
    s.set_defaults(func=cmd_export_signal)

    s = sub.add_parser("view", help="open the interactive viewer")
    s.add_argument("--window", action="store_true", help="native window (requires pywebview)")
    s.set_defaults(func=cmd_view)

    s = sub.add_parser("annotate", help="review and correct beat marks")
    s.add_argument("folder")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_annotate)

    s = sub.add_parser("train", help="learn from annotator corrections")
    s.add_argument("folder")
    s.add_argument("--no-fit", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("demo", help="synthetic experiment -> EDF -> analysis")
    s.add_argument("out")
    s.add_argument("--minutes", type=float, default=20)
    s.add_argument("--chunk-minutes", type=float, default=10)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("gui", help="desktop application")
    s.set_defaults(func=cmd_gui)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "func", None):
        p.print_help()
        return 0
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
