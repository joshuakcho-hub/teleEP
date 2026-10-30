"""Overview figure of an analysis: rate trend, short-term variability, Poincare plot, spectrum."""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from .hrv import spectrum

SERIES_COLOR = {"RR": "#2a78d6", "PP": "#eb6834"}
SERIES_NAME = {"RR": "Ventricular (RR)", "PP": "Atrial (PP)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _binned_rate(s, bin_s=60.0):
    if s.n_valid < 2:
        return np.zeros(0), np.zeros(0)
    t, ms = s.t[s.valid], s.ms[s.valid]
    edges = np.arange(0, t.max() + bin_s, bin_s)
    idx = np.digitize(t, edges) - 1
    out_t, out_r = [], []
    for k in np.unique(idx):
        v = ms[idx == k]
        if len(v) >= 3:
            out_t.append(edges[k] + bin_s / 2)
            out_r.append(60000.0 / np.median(v))
    return np.array(out_t), np.array(out_r)


def overview_figure(res, path: str, dpi: int = 110) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if f in avail), "sans-serif")
    plt.rcParams.update({"font.family": family, "font.size": 10})
    series = [s for s in (res.rr, res.pp) if s is not None and len(s)]
    to_dt = lambda t: [res.start + timedelta(seconds=float(v)) for v in t]

    fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1.3], hspace=0.45, wspace=0.22)
    ax1, ax2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    ax3, ax4 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    for ax in (ax1, ax2, ax3, ax4):
        _style(ax)

    # 1. rate trend (1-min medians of valid intervals); excluded stretches shaded
    for a, b, why in res.bad_segments:
        if why in ("dropout", "gap between files", "no signal") and b - a > 1:
            ax1.axvspan(*to_dt([a, b]), color="#d9d8d3", linewidth=0)
    for s in series:
        t, r = _binned_rate(s)
        if len(t):
            t, r = np.insert(t, np.flatnonzero(np.diff(t) > 180) + 1, np.nan), np.insert(r, np.flatnonzero(np.diff(t) > 180) + 1, np.nan)
            ax1.plot([res.start + timedelta(seconds=float(v if np.isfinite(v) else 0)) for v in t], r,
                     color=SERIES_COLOR[s.kind], lw=1.6, label=SERIES_NAME[s.kind])
    ax1.set_ylabel("Rate (per min)", color=INK2)
    ax1.set_title("Rate, 1-minute medians of valid intervals (grey: dropout or gap)", loc="left",
                  fontsize=11, color=INK)
    loc = AutoDateLocator()
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(ConciseDateFormatter(loc))

    # 2. RMSSD per epoch
    ep_len = res.settings["epoch_s"]
    for kind in [s.kind for s in series]:
        ep = [e for e in res.epochs if e["series"] == kind and e.get("analyzed")]
        if ep:
            t = np.array([e["epoch_start_s"] for e in ep], float)
            y = np.array([e["rmssd_ms"] for e in ep], float)
            # break the line where epochs are missing so no trend is drawn across them
            brk = np.flatnonzero(np.diff(t) > 1.5 * ep_len) + 1
            t, y = np.insert(t, brk, np.nan), np.insert(y, brk, np.nan)
            dt = [res.start + timedelta(seconds=float(v)) if np.isfinite(v) else None for v in t]
            dt = [d if d is not None else dt[i - 1] for i, d in enumerate(dt)]
            ax2.plot(dt, y, "-", color=SERIES_COLOR[kind], lw=1.4)
            ax2.plot(to_dt(t[np.isfinite(t)]), y[np.isfinite(y)], "o", color=SERIES_COLOR[kind], ms=3.5)
    ax2.set_ylabel("RMSSD (ms)", color=INK2)
    ax2.set_title(f"RMSSD per {res.settings['epoch_s'] / 60:g}-minute epoch (analyzed epochs only)",
                  loc="left", fontsize=11, color=INK)
    ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_formatter(ConciseDateFormatter(loc))

    # 3. Poincare
    for s in series:
        pair = s.valid[1:] & s.valid[:-1]
        x, y = s.ms[:-1][pair], s.ms[1:][pair]
        if len(x) > 20000:
            k = np.random.default_rng(0).choice(len(x), 20000, replace=False)
            x, y = x[k], y[k]
        ax3.scatter(x, y, s=3, color=SERIES_COLOR[s.kind], alpha=0.25, linewidths=0,
                    label=SERIES_NAME[s.kind])
    lim = ax3.get_xlim()
    ax3.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax3.set_xlabel("Interval n (ms)", color=INK2)
    ax3.set_ylabel("Interval n+1 (ms)", color=INK2)
    ax3.set_title("Poincare plot (valid consecutive pairs)", loc="left", fontsize=11, color=INK)

    # 4. median spectrum over analyzed epochs
    hs = None
    try:
        from .analysis import AnalysisSettings
        hs = AnalysisSettings(species=res.settings["species"], bands=res.settings["bands"],
                              spectral_method=res.settings["spectral_method"]).hrv_settings()
    except Exception:
        pass
    if hs is not None:
        bands = hs.bands
        cols = {"VLF": "#f0efec", "LF": "#e3edf9", "HF": "#fdeee7"}
        for name, (lo, hi) in bands.items():
            if name.upper() != "VLF":
                ax4.axvspan(lo, hi, color=cols.get(name.upper(), "#f0efec"), linewidth=0)
                ax4.text((lo + hi) / 2, 1.0, name, transform=ax4.get_xaxis_transform(), ha="center",
                         va="bottom", fontsize=9, color=INK2)
        fmax = max(b[1] for b in bands.values())
        for s in series:
            ep = [e for e in res.epochs if e["series"] == s.kind and e.get("analyzed")]
            grid = np.linspace(0, fmax * 1.1, 400)
            spectra = []
            for e in ep[:200]:
                f, p = spectrum(s.window(e["epoch_start_s"], e["epoch_end_s"]), hs)
                if len(f):
                    spectra.append(np.interp(grid, f, p))
            if spectra:
                ax4.semilogy(grid[1:], np.median(spectra, axis=0)[1:], color=SERIES_COLOR[s.kind], lw=1.8,
                             label=SERIES_NAME[s.kind])
        ax4.set_xlim(0, fmax * 1.1)
    ax4.set_xlabel("Frequency (Hz)", color=INK2)
    ax4.set_ylabel("PSD (ms²/Hz)", color=INK2)
    ax4.set_title("Median power spectrum of analyzed epochs", loc="left", fontsize=11, color=INK, pad=14)

    if len(series) > 1:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=SERIES_COLOR[s.kind], lw=2.5) for s in series]
        leg = fig.legend(handles, [SERIES_NAME[s.kind] for s in series], loc="upper left", ncol=len(series),
                         frameon=False, fontsize=10, bbox_to_anchor=(0.055, 0.955))
        for t in leg.get_texts():
            t.set_color(INK)
    s = res.summary["RR"]
    fig.suptitle(f"{res.files[0]['file'].split('_seg')[0]}  |  {res.start:%Y-%m-%d %H:%M}  |  "
                 f"{res.duration_s / 3600:.1f} h  |  {len(res.beats['time_s'])} QRS  |  "
                 f"mean HR {s['mean_hr_bpm']:.0f} bpm  |  SDNN {s['sdnn_ms']:.0f} ms  |  "
                 f"RMSSD {s['rmssd_ms']:.0f} ms", x=0.06, ha="left", fontsize=12, color=INK, y=0.985)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
