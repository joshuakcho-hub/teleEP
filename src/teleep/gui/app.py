"""TeleEP desktop application (tkinter): convert, analyze, view.

Start with ``teleep gui`` or ``teleep-gui`` (or the packaged app). Long tasks run in a
background thread; progress is shown in the log at the bottom of each tab.
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .. import __version__

PAD = dict(padx=8, pady=4)


def _open_folder(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class LogBox(ttk.Frame):
    def __init__(self, master, height=10):
        super().__init__(master)
        self.q: "queue.Queue[str]" = queue.Queue()
        self.text = tk.Text(self, height=height, wrap="word", relief="flat", borderwidth=1)
        sb = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.after(100, self._drain)

    def write(self, msg: str) -> None:
        self.q.put(str(msg))

    def _drain(self):
        while not self.q.empty():
            self.text.insert("end", self.q.get() + "\n")
            self.text.see("end")
        self.after(100, self._drain)


def _row(parent, label, var, browse=None, extra=()):
    """Label, entry and buttons on one line; ``extra`` = [(text, command)] buttons after Browse."""
    f = ttk.Frame(parent)
    f.pack(fill="x", **PAD)
    ttk.Label(f, text=label, width=30).pack(side="left")
    for text, cmd in reversed(list(extra)):
        ttk.Button(f, text=text, command=cmd).pack(side="right", padx=(4, 0))
    if browse:
        ttk.Button(f, text="Browse…", command=browse).pack(side="right", padx=(4, 0))
    ttk.Entry(f, textvariable=var, width=40).pack(side="left", fill="x", expand=True)
    return f


class ConvertTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.exp = tk.StringVar()
        self.out = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "TeleEP_EDF"))
        self.chunk = tk.StringVar(value="60")
        self.maxs = tk.StringVar(value="")
        self.utc = tk.BooleanVar(value=False)
        self.marks = tk.BooleanVar(value=False)
        self.batch = tk.BooleanVar(value=False)
        ttk.Label(self, text="Convert a DSI Ponemah experiment (.PnmWav, .PnmExp, .PnmMarks) to EDF+. "
                             "Signals are stored losslessly; telemetry dropouts are annotated.",
                  wraplength=900).pack(anchor="w", **PAD)
        _row(self, "Experiment folder", self.exp, self._pick_exp)
        bar = ttk.Frame(self)
        bar.pack(fill="x", **PAD)
        ttk.Button(bar, text="Load channels", command=self._load).pack(side="left")
        ttk.Checkbutton(bar, text="Batch: convert every experiment folder inside this folder",
                        variable=self.batch).pack(side="left", padx=12)
        lf = ttk.LabelFrame(self, text="Channels (none selected = all)")
        lf.pack(fill="both", expand=True, **PAD)
        self.chans = tk.Listbox(lf, selectmode="multiple", exportselection=False, height=7)
        self.chans.pack(fill="both", expand=True, padx=6, pady=6)
        opt = ttk.LabelFrame(self, text="Options")
        opt.pack(fill="x", **PAD)
        r = ttk.Frame(opt)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text="File length (minutes, 0 = one file per segment)").pack(side="left", padx=6)
        ttk.Entry(r, textvariable=self.chunk, width=8).pack(side="left")
        ttk.Label(r, text="Convert only the first N seconds").pack(side="left", padx=12)
        ttk.Entry(r, textvariable=self.maxs, width=10).pack(side="left")
        r2 = ttk.Frame(opt)
        r2.pack(fill="x", pady=3)
        ttk.Checkbutton(r2, text="Write UTC times (default: local time of the acquisition computer)",
                        variable=self.utc).pack(side="left", padx=6)
        ttk.Checkbutton(r2, text="Also export Ponemah's beat marks (CSV)", variable=self.marks).pack(side="left", padx=12)
        _row(self, "Output folder", self.out, lambda: self._pick(self.out))
        b = ttk.Frame(self)
        b.pack(fill="x", **PAD)
        self.btn = ttk.Button(b, text="Convert to EDF", command=self._run)
        self.btn.pack(side="left")
        ttk.Button(b, text="Open output folder", command=lambda: _open_folder(self.out.get())).pack(side="left", padx=6)
        self.prog = ttk.Progressbar(b, mode="indeterminate")
        self.prog.pack(side="left", fill="x", expand=True, padx=8)
        self.log = LogBox(self, 9)
        self.log.pack(fill="both", expand=True, **PAD)

    def _pick_exp(self):
        d = filedialog.askdirectory(title="Ponemah experiment folder")
        if d:
            self.exp.set(d)

    def _pick(self, var):
        d = filedialog.askdirectory(title="Choose a folder")
        if d:
            var.set(d)

    def _load(self):
        from ..io.ponemah import is_experiment_folder, open_experiment
        try:
            path = self.exp.get().strip()
            if not is_experiment_folder(path):
                raise ValueError("This folder has no .PnmWav file.")
            self.chans.delete(0, "end")
            seen = []
            for r in open_experiment(path):
                for c in r.channels:
                    if c.label not in seen:
                        seen.append(c.label)
                        self.chans.insert("end", f"{c.label}   ({c.sample_rate:g} Hz, {c.unit}, {c.calibration})")
                segs = "; ".join(f"{r.to_local(s.start):%Y-%m-%d %H:%M:%S} for {s.duration_s / 3600:.2f} h" for s in r.segments)
                self.log.write(f"Subject {r.subject}: time zone {r.timezone_name}; segments: {segs}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _run(self):
        from ..convert import convert_batch, convert_experiment
        path, out = self.exp.get().strip(), self.out.get().strip()
        if not path:
            messagebox.showwarning("Experiment folder", "Choose an experiment folder first.")
            return
        sel = [self.chans.get(i).split("   (")[0] for i in self.chans.curselection()]
        try:
            chunk = float(self.chunk.get() or 0)
            maxs = float(self.maxs.get()) if self.maxs.get().strip() else None
        except ValueError:
            messagebox.showerror("Options", "File length and N seconds must be numbers.")
            return
        kw = dict(channels=sel or None, chunk_minutes=chunk or None, max_seconds=maxs,
                  time_reference="utc" if self.utc.get() else "local", export_ponemah_marks=self.marks.get())
        self.btn.config(state="disabled")
        self.prog.start(12)

        def work():
            try:
                if self.batch.get():
                    files = convert_batch(path, out, log=self.log.write, **kw)
                else:
                    files = convert_experiment(path, out, log=self.log.write, **kw)
                self.log.write(f"Finished: {len(files)} EDF file(s) in {out}")
            except Exception as e:
                self.log.write(f"ERROR: {e}")
            finally:
                self.after(0, lambda: (self.prog.stop(), self.btn.config(state="normal")))
        threading.Thread(target=work, daemon=True).start()


class AnalyzeTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.inp = tk.StringVar()
        self.out = tk.StringVar()
        self.species = tk.StringVar(value="pig")
        self.detector = tk.StringVar(value="species default")
        self.leads = tk.StringVar(value="")
        self.pw = tk.BooleanVar(value=True)
        self.art = tk.StringVar(value="20")
        self.epoch = tk.StringVar(value="5")
        self.cov = tk.StringVar(value="80")
        self.spec = tk.StringVar(value="welch")
        self.workers = tk.StringVar(value=str(max(1, min(4, (os.cpu_count() or 2) - 1))))
        self.xlsx = tk.BooleanVar(value=True)
        self.fig = tk.BooleanVar(value=True)
        self.pnm = tk.StringVar(value="")
        ttk.Label(self, text="Detect beats in EDF files, compute RR (ventricular) and PP (atrial) intervals and heart "
                             "rate variability, and export CSV, Excel and JSON tables plus an overview figure. "
                             "Files in one folder are analyzed as one continuous recording.",
                  wraplength=900).pack(anchor="w", **PAD)
        _row(self, "EDF folder (or a single file)", self.inp, self._pick_folder, [("File…", self._pick_file)])
        g = ttk.LabelFrame(self, text="Settings")
        g.pack(fill="x", **PAD)
        r = ttk.Frame(g)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text="Species").pack(side="left", padx=6)
        ttk.Combobox(r, textvariable=self.species, values=["pig", "dog", "human", "rat", "mouse"], width=8,
                     state="readonly").pack(side="left")
        ttk.Label(r, text="Detector").pack(side="left", padx=(14, 6))
        ttk.Combobox(r, textvariable=self.detector, values=["species default", "pigecg", "generic"], width=15,
                     state="readonly").pack(side="left")
        ttk.Label(r, text="ECG leads (blank = all)").pack(side="left", padx=(14, 6))
        ttk.Entry(r, textvariable=self.leads, width=18).pack(side="left")
        ttk.Checkbutton(r, text="P waves (pigecg)", variable=self.pw).pack(side="left", padx=14)
        r = ttk.Frame(g)
        r.pack(fill="x", pady=3)
        for lab, var, w in (("Artifact threshold (%)", self.art, 6), ("Epoch (min)", self.epoch, 6),
                            ("Min. epoch coverage (%)", self.cov, 6), ("Parallel workers", self.workers, 4)):
            ttk.Label(r, text=lab).pack(side="left", padx=(6, 4))
            ttk.Entry(r, textvariable=var, width=w).pack(side="left", padx=(0, 10))
        r = ttk.Frame(g)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text="Spectrum").pack(side="left", padx=(6, 4))
        ttk.Combobox(r, textvariable=self.spec, values=["welch", "lomb"], width=7, state="readonly").pack(side="left", padx=(0, 12))
        ttk.Checkbutton(r, text="Excel workbook", variable=self.xlsx).pack(side="left", padx=6)
        ttk.Checkbutton(r, text="Overview figure", variable=self.fig).pack(side="left", padx=6)
        _row(self, "Compare with Ponemah (optional)", self.pnm, lambda: self._pick(self.pnm))
        _row(self, "Results folder", self.out, lambda: self._pick(self.out))
        b = ttk.Frame(self)
        b.pack(fill="x", **PAD)
        self.btn = ttk.Button(b, text="Analyze", command=self._run)
        self.btn.pack(side="left")
        ttk.Button(b, text="Open results folder", command=lambda: _open_folder(self.out.get())).pack(side="left", padx=6)
        self.prog = ttk.Progressbar(b, mode="indeterminate")
        self.prog.pack(side="left", fill="x", expand=True, padx=8)
        self.log = LogBox(self, 10)
        self.log.pack(fill="both", expand=True, **PAD)

    def _pick(self, var):
        d = filedialog.askdirectory(title="Choose a folder")
        if d:
            var.set(d)

    def _pick_folder(self):
        d = filedialog.askdirectory(title="Folder with EDF files")
        if d:
            self.inp.set(d)
            if not self.out.get():
                self.out.set(os.path.join(d, "teleep_results"))

    def _pick_file(self):
        f = filedialog.askopenfilename(title="EDF file", filetypes=[("EDF", "*.edf *.EDF"), ("All", "*.*")])
        if f:
            self.inp.set(f)
            if not self.out.get():
                self.out.set(os.path.join(os.path.dirname(f), "teleep_results"))

    def _run(self):
        from ..analysis import AnalysisSettings, analyze
        from ..export import export_all
        inp = self.inp.get().strip()
        if not inp:
            messagebox.showwarning("Input", "Choose an EDF folder or file first.")
            return
        out = self.out.get().strip() or os.path.join(inp if os.path.isdir(inp) else os.path.dirname(inp), "teleep_results")
        self.out.set(out)
        try:
            art = float(self.art.get() or 0) / 100
            st = AnalysisSettings(
                species=self.species.get(),
                detector=None if self.detector.get() == "species default" else self.detector.get(),
                leads=[s.strip() for s in self.leads.get().split(",") if s.strip()] or None,
                detect_p_waves=self.pw.get(), artifact_threshold=art or None,
                epoch_s=float(self.epoch.get()) * 60, min_epoch_coverage=float(self.cov.get()) / 100,
                spectral_method=self.spec.get())
            workers = int(self.workers.get() or 1)
        except ValueError:
            messagebox.showerror("Settings", "Check that the numeric settings are numbers.")
            return
        self.btn.config(state="disabled")
        self.prog.start(12)

        def work():
            try:
                res = analyze([inp], st, workers=workers, log=self.log.write)
                export_all(res, out, xlsx=self.xlsx.get(), figure=self.fig.get(), log=self.log.write)
                if self.pnm.get().strip():
                    from ..validate import best_lead_totals, compare_with_ponemah
                    rows = compare_with_ponemah(res, self.pnm.get().strip(), out_dir=out)
                    for lead, d in best_lead_totals(rows).items():
                        self.log.write(f"Ponemah {lead}: matched {d['matched']}, TeleEP only {d['teleep_only']}, "
                                       f"Ponemah only {d['ponemah_only']}")
                self.log.write(f"Finished. Results in {out}")
            except Exception as e:
                import traceback
                self.log.write(f"ERROR: {e}\n{traceback.format_exc()}")
            finally:
                self.after(0, lambda: (self.prog.stop(), self.btn.config(state="normal")))
        threading.Thread(target=work, daemon=True).start()


class ViewTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        ttk.Label(self, text="TeleEP Viewer", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", **PAD)
        ttk.Label(self, wraplength=900, text=(
            "Opens in your web browser and works entirely offline. Open an EDF file to scroll through the "
            "signals with filtering (high-pass, low-pass, mains notch), detect R peaks, review and correct "
            "beats, inspect the tachogram, compute HRV for the whole file, the visible window or a selected "
            "range, and export beats, intervals, HRV tables, signal segments and images. Results of "
            "'Analyze' can be loaded with 'Import marks' (files in results/viewer_marks).")).pack(anchor="w", **PAD)
        ttk.Button(self, text="Open the viewer", command=self._open).pack(anchor="w", **PAD)
        ttk.Separator(self).pack(fill="x", pady=10)
        ttk.Label(self, text="EP Annotator", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", **PAD)
        ttk.Label(self, wraplength=900, text=(
            "Review the multi-lead P/QRS detector beat by beat, correct marks, and save corrections that "
            "measure the detector's accuracy and train it (teleep train). Choose the folder with the EDF files.")).pack(anchor="w", **PAD)
        ttk.Button(self, text="Start the annotator…", command=self._annot).pack(anchor="w", **PAD)

    def _open(self):
        from ..viewer.launch import open_viewer
        open_viewer()

    def _annot(self):
        d = filedialog.askdirectory(title="Folder with EDF files")
        if not d:
            return
        from ..annotator.app import serve
        threading.Thread(target=serve, args=(d, 8765, True), daemon=True).start()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"TeleEP {__version__}: telemetry ECG conversion and analysis")
        self.geometry("1000x760")
        self.minsize(820, 620)
        try:
            ttk.Style(self).theme_use("clam" if sys.platform.startswith("linux") else ttk.Style(self).theme_use())
        except tk.TclError:
            pass
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        nb.add(ConvertTab(nb), text="  1. Convert  ")
        nb.add(AnalyzeTab(nb), text="  2. Analyze  ")
        nb.add(ViewTab(nb), text="  3. View and annotate  ")
        ttk.Label(self, text=f"TeleEP {__version__} · Joshua Kyungjoon Cho and Hee Cheol Cho · BSD-3-Clause",
                  foreground="#52514e").pack(anchor="e", padx=10, pady=(0, 6))


def main():
    multiprocessing.freeze_support()
    App().mainloop()


if __name__ == "__main__":
    main()
