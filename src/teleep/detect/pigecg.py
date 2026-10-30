"""
pigecg: P / QRS / T detection for subcutaneous porcine telemetry ECG (DSI Ponemah -> EDF)
=========================================================================================
Designed for pigs with induced conduction disease (AV block, sinus node dysfunction):

* QRS and P waves are detected INDEPENDENTLY (no assumed 1:1 coupling), so AV
  dissociation, 2:1 block, Wenckebach, sinus pauses and escape rhythms are handled.
* Multi-lead fusion: leads are noise-weighted and combined into one energy signal for
  candidate detection; multi-lead templates are used for classification.
* Candidate deflections are separated into QRS / P / noise using
    - log energy in the 5-40 Hz band (relative to the confident QRS population),
    - high-frequency ratio (15-45 Hz energy / 5-40 Hz energy): QRS is sharp, P is smooth,
    - normalized cross-correlation (NCC) with multi-lead QRS templates (k-means, so
      conducted and escape morphologies get their own template).
* An iterative, rhythm-aware gap-filling pass recovers QRS complexes when the QRS
  amplitude drops (posture / lead-vector changes), using the local RR and local energy.
* Signal-quality mask: flat-line (dropout) and high-noise segments are flagged as
  unanalyzable; detections inside them are reported but flagged.
* P waves hidden in the QRS-T are recovered by subtracting the average QRS-T of each
  beat (per morphology, least-squares scaled) and matched-filtering the residual with a
  multi-lead P template. P detections inside the QRS-T window must be consistent with
  the local PP rhythm.
* T waves are delineated on the per-beat 15-Hz low-passed signal, seeded from the
  average beat; T end by the tangent method.

Dependencies: numpy, scipy.

Origin: written for the Cho laboratory (Johns Hopkins) pig telemetry studies and validated on an
18.6-hour recording of a pig with complete atrioventricular block (see docs/validation.md).
TeleEP changes: missing samples (NaN, telemetry dropout) are accepted and excluded from the lead
weighting and flagged as unanalyzable; refractory periods are parameters of ``run``.
"""
import numpy as np
from scipy import signal as sg
from scipy.ndimage import uniform_filter1d, median_filter, minimum_filter1d
from scipy.signal import argrelextrema

# ----------------------------------------------------------------------------- helpers

def _odd(n):
    n = int(n); return n if n % 2 else n + 1

def moving_avg(v, w):
    return uniform_filter1d(v, size=max(1, int(w)), mode='nearest')

def baseline_remove(v, fs):
    """Two-stage median baseline (200 ms, 600 ms)."""
    b = median_filter(v, size=_odd(0.2 * fs), mode='nearest')
    b = median_filter(b, size=_odd(0.6 * fs), mode='nearest')
    return v - b

def bandpass(v, fs, lo, hi, order=3):
    sos = sg.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sg.sosfiltfilt(sos, v)

def lowpass(v, fs, hi, order=3):
    sos = sg.butter(order, hi, btype='low', fs=fs, output='sos')
    return sg.sosfiltfilt(sos, v)

def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))

def local_median(times, values, ref_times, half_n=8):
    """Median of `values` over the half_n neighbours (by index) around each ref time."""
    idx = np.searchsorted(times, ref_times)
    out = np.empty(len(ref_times))
    for k, i in enumerate(idx):
        s, e = max(0, i - half_n), min(len(values), i + half_n)
        out[k] = np.median(values[s:e]) if e > s else np.nan
    return out

# ----------------------------------------------------------------------------- preprocessing

class Prep:
    def __init__(self, x, fs, missing=None):
        x = np.atleast_2d(np.asarray(x, float))
        # per-lead missing-sample mask (dropout); missing samples must already be filled (e.g. 0)
        self.missing = np.zeros(x.shape, bool) if missing is None else np.atleast_2d(missing)
        self.fs = fs; self.n = x.shape[1]; self.nl = x.shape[0]
        self.raw = x
        self.xb = np.array([baseline_remove(v, fs) for v in x])          # baseline-free, full band
        self.xf = np.array([bandpass(v, fs, 5, 40) for v in self.xb])      # deflection band
        self.xh = np.array([bandpass(v, fs, 15, 45) for v in self.xb])     # sharpness band
        self.xl = np.array([lowpass(v, fs, 15) for v in self.xb])          # smooth, for T & P delineation
        self.noise = np.array([mad(v) for v in self.xf]) + 1e-9
        # time-varying lead weights: local noise = running median of |xf| (2 s), robust to beats
        ln = np.array([median_filter(np.abs(v[::5]), size=_odd(2 * fs / 5), mode='nearest') for v in self.xf])
        ln = np.repeat(ln, 5, axis=1)[:, :self.n] + 1e-6
        wt = 1 / ln ** 2
        if self.missing.any():                                              # a lead in dropout gets no weight
            wt[self.missing] = 0.0
            none = wt.sum(axis=0) == 0
            wt[:, none] = 1.0
        wt /= wt.sum(axis=0, keepdims=True)
        self.wt = np.array([moving_avg(v, 1.0 * fs) for v in wt])
        w = 1 / self.noise ** 2; self.w = w / w.sum()                        # global lead weights
        self.energy = np.sum(self.wt * self.xf ** 2, axis=0)
        self.energy_s = moving_avg(self.energy, 0.03 * fs)
        self.energy_h = moving_avg(np.sum(self.wt * self.xh ** 2, axis=0), 0.03 * fs)
        # local noise floor: 1-s running median of the energy (dominated by non-beat samples)
        self.floor = median_filter(self.energy_s[::10], size=_odd(fs / 10), mode='nearest')
        self.floor = np.repeat(self.floor, 10)[:self.n]
        self.quality = self._quality_mask()

    def _quality_mask(self):
        """True where the signal is analyzable. Flags flat-line dropouts and noise bursts."""
        fs = self.fs
        flat = np.ones(self.n, bool)
        for l in range(self.nl):
            d = np.abs(np.diff(self.raw[l], prepend=self.raw[l][0]))
            # flat: no change for >= 200 ms
            run = moving_avg((d == 0).astype(float), 0.1 * fs) > 0.98
            flat &= run
        bad = flat | self.missing.all(axis=0)
        bad = minimum_filter1d((~bad).astype(np.uint8), size=_odd(1.2 * fs)) == 0
        return ~bad

# ----------------------------------------------------------------------------- candidates & features

def find_candidates(prep, min_sep_ms=60):
    fs = prep.fs
    pk, _ = sg.find_peaks(prep.energy_s, distance=int(min_sep_ms / 1000 * fs), height=3 * prep.floor + 1e-12)
    return pk

def peak_features(prep, pk):
    loge = np.log10(prep.energy_s[pk] + 1e-15)
    ratio = prep.energy_h[pk] / (prep.energy_s[pk] + 1e-15)
    snr = np.log10(prep.energy_s[pk] / (prep.floor[pk] + 1e-15))
    return loge, ratio, snr

# ----------------------------------------------------------------------------- templates

def extract(sig, centers, pre, post):
    n = sig.shape[1]; out = []; keep = []
    for c in centers:
        ok = c - pre >= 0 and c + post <= n
        keep.append(ok)
        if ok: out.append(sig[:, c - pre:c + post])
    return np.array(out), np.array(keep)

def kmeans_templates(W, kmax=3, min_frac=0.03, iters=30, seed=0, gain=0.02):
    nb = W.shape[0]
    Z = W.reshape(nb, -1); Z = Z - Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    rng = np.random.default_rng(seed)
    best = None
    for k in range(1, kmax + 1):
        cen = Z[rng.choice(nb, k, replace=False)]
        for _ in range(iters):
            sim = Z @ cen.T; lab = sim.argmax(axis=1)
            new = np.array([Z[lab == j].mean(axis=0) if np.any(lab == j) else cen[j] for j in range(k)])
            new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-12
            if np.allclose(new, cen): break
            cen = new
        sim = Z @ cen.T; lab = sim.argmax(axis=1)
        frac = np.bincount(lab, minlength=k) / nb
        if k > 1 and frac.min() < min_frac: break
        score = sim.max(axis=1).mean()
        if best is None or score > best[0] + gain: best = (score, lab.copy(), k)
        else: break
    _, lab, k = best
    temps = np.array([np.median(W[lab == j], axis=0) for j in range(k)])
    return lab, temps

def match_template(sig, centers, temp, pre, post):
    out = np.full(len(centers), -1.0); n = sig.shape[1]
    t = temp - temp.mean(); tn = np.linalg.norm(t) + 1e-12
    for i, c in enumerate(centers):
        if c - pre >= 0 and c + post <= n:
            w = sig[:, c - pre:c + post]; w = w - w.mean()
            out[i] = float((w * t).sum() / (np.linalg.norm(w) * tn + 1e-12))
    return out

def _refractory(idx, min_sep, score):
    idx = np.asarray(idx)
    if len(idx) == 0: return idx
    order = np.argsort(-score); keep = np.zeros(len(idx), bool); chosen = []
    for o in order:
        if all(abs(idx[o] - c) >= min_sep for c in chosen):
            keep[o] = True; chosen.append(idx[o])
    return np.sort(idx[keep])

def candidate_features(erel, best, ratio, snr, pst):
    """Feature matrix used by the supervised QRS-vs-other model (train.py)."""
    lg = np.log(np.clip(pst, 1e-4, 1 - 1e-4) / (1 - np.clip(pst, 1e-4, 1 - 1e-4)))
    return np.column_stack([erel, best, ratio, snr, lg / 10.0, np.ones(len(erel))])

def _valley_threshold(v, default=-0.6):
    """Threshold between the top mode (QRS, near 0) and the next lower mode (P) of log rel energy."""
    v = v[(v > -2.5) & (v < 1.0)]
    if len(v) < 50: return default
    grid = np.linspace(-2.5, 1.0, 351); bw = 0.08
    kde = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2).sum(axis=1)
    mx = argrelextrema(kde, np.greater, order=5)[0]
    mx = mx[kde[mx] > 0.05 * kde.max()]
    hi = mx[grid[mx] > -0.7]
    if len(hi) == 0: return default
    top = hi[np.argmax(kde[hi])]
    lower = mx[mx < top]
    if len(lower) == 0: return default
    nxt = lower[-1]
    valley = nxt + int(np.argmin(kde[nxt:top]))
    if kde[valley] > 0.6 * min(kde[nxt], kde[top]): return default
    return float(grid[valley])

# ----------------------------------------------------------------------------- QRS detection

def detect_qrs(prep, pk, loge, ratio, snr, min_rr_ms=220, verbose=False, model=None):
    fs = prep.fs; min_rr = int(min_rr_ms / 1000 * fs)
    pre, post = int(0.06 * fs), int(0.08 * fs)
    good = prep.quality[pk]
    # --- 1. seeds: top 5 % energy candidates in good signal -> energy reference
    seed = pk[good & (loge >= np.percentile(loge[good], 95))]
    seed = _refractory(seed, min_rr, prep.energy_s[seed])
    e_ref = np.median(prep.energy_s[seed])
    erel = np.log10(prep.energy_s[pk] / e_ref)
    # noise masks relative to the QRS energy
    typ = np.median(prep.floor)
    bad_q = (prep.floor > 0.025 * e_ref) | (prep.floor > 10 * typ) | ~prep.quality
    bad_p = (prep.floor > 0.015 * e_ref) | (prep.floor > 8 * typ) | ~prep.quality
    dil = lambda m: minimum_filter1d((~m).astype(np.uint8), size=_odd(1.0 * fs)) == 0
    bad_q = dil(bad_q); bad_p = dil(bad_p)
    good = ~bad_q[pk]
    # --- 2. energy valley between QRS mode and P mode (first template only to gate NCC>0.4)
    W, keep = extract(prep.xf, seed, pre, post); seed = seed[keep]
    lab, temps = kmeans_templates(W, kmax=2)
    sc = np.array([match_template(prep.xf, pk, t, pre, post) for t in temps]).T
    best = sc.max(axis=1)
    thr = _valley_threshold(erel[good & (best > 0.4)])
    # --- 3. sharpness statistics: QRS class = above threshold, P class = P mode below it
    hi = good & (erel > thr); lo = good & (erel < thr) & (erel > thr - 1.2)
    r_q = np.median(ratio[hi]); r_p = np.median(ratio[lo]) if lo.sum() > 20 else 0.4 * r_q
    r_cut = r_p + 0.5 * (r_q - r_p)
    # --- 4. templates from ALL high-energy candidates (so changed morphologies get a template)
    W, keep = extract(prep.xf, pk[hi], pre, post)
    lab, temps = kmeans_templates(W, kmax=4, min_frac=0.01)
    sc = np.array([match_template(prep.xf, pk, t, pre, post) for t in temps]).T
    best = sc.max(axis=1); morph = sc.argmax(axis=1)
    # --- 5. QRS vs P posterior from class-conditional Gaussians in (log rel energy, NCC, sharpness)
    dq0 = np.abs(pk[:, None] - pk[hi][None, :]).min(axis=1) if hi.sum() else np.full(len(pk), 1e9)
    cq = good & (erel > thr + 0.1) & (best > 0.3)
    cp = good & (erel < thr - 0.2) & (erel > thr - 1.4) & (best > 0.3) & (dq0 > int(0.12 * fs))
    def _gauss(X):
        mu = np.median(X, axis=0); C = np.cov((X - mu).T) + np.diag([0.02, 0.005, 0.005]); Ci = np.linalg.inv(C)
        return lambda Y: -0.5 * np.einsum('ij,jk,ik->i', Y - mu, Ci, Y - mu) - 0.5 * np.log(np.linalg.det(C))
    if cq.sum() > 30 and cp.sum() > 30:
        # P class: one Gaussian on (erel, best NCC, ratio); QRS class: one Gaussian per morphology
        lp = _gauss(np.column_stack([erel, best, ratio])[cp])(np.column_stack([erel, best, ratio]))
        lq = np.full(len(pk), -1e9)
        for j in range(len(temps)):
            mj = cq & (morph == j)
            if mj.sum() < 20: continue
            Fj = np.column_stack([erel, sc[:, j], ratio])
            lq = np.maximum(lq, _gauss(Fj[mj])(Fj) + np.log(mj.sum() / cq.sum()))
        llr = lq - lp + np.log(cq.sum() / cp.sum())
        pst = 1 / (1 + np.exp(-np.clip(llr, -50, 50)))
    else:
        pst = (erel > thr).astype(float)
    # hard gate: a QRS must look at least somewhat like a QRS template unless it is clearly energetic
    plaus = (erel > thr - 0.5) & ((best > 0.5) | (erel > thr + 0.2)) & (ratio > 0.5 * r_p)
    pst = np.where(plaus, pst, 0.0)
    pst_unsup = pst.copy()
    if model is not None and 'w' in model:
        # supervised refinement learned from user corrections (train.py): logistic model on candidate features
        Fm = candidate_features(erel, best, ratio, snr, pst_unsup)
        z = Fm @ np.asarray(model['w']) + model['b']
        pst = np.where(plaus | (erel > thr - 0.9), 1 / (1 + np.exp(-np.clip(z, -50, 50))), 0.0)
    qrs0 = (pst > 0.5) | (erel > thr + 0.2)
    qrs0 &= ~((ratio < r_p + 0.3 * (r_q - r_p)) & (best < 0.5) & (erel < thr + 0.5))   # smooth & unlike: T/artefact
    if verbose: print(f'  seeds {len(seed)}  thr={thr:.2f}  morphs {len(temps)} {np.bincount(lab)}  ratio QRS {r_q:.2f} P {r_p:.2f}  badQ {bad_q.mean():.3f} badP {bad_p.mean():.3f}  classes Q {cq.sum()} P {cp.sum()}')
    sel = np.where(qrs0)[0]
    sel = sel[np.isin(pk[sel], _refractory(pk[sel], min_rr, erel[sel] + 0.5 * ratio[sel] / r_q))]
    # --- 6. iterative rhythm-aware gap filling (adaptive to local energy and local RR)
    e = prep.energy_s[pk]
    is_q = np.zeros(len(pk), bool); is_q[sel] = True
    for it in range(60):
        qi = np.where(is_q)[0]
        if len(qi) < 3: break
        qt = pk[qi]; rr = np.diff(qt)
        rr_loc = local_median(qt[1:], rr, qt, 6)
        e_loc = local_median(qt, e[qi], qt, 6)
        added = 0
        for i in range(len(qi) - 1):
            gap = qt[i + 1] - qt[i]
            if gap < 1.45 * rr_loc[i]: continue
            lo_, hi_ = qt[i] + min_rr, qt[i + 1] - min_rr
            m = (pk > lo_) & (pk < hi_) & ~is_q & (snr > 0.7)
            m &= (ratio > r_cut) | ((ratio > r_p + 0.3 * (r_q - r_p)) & (best > 0.7))
            m &= e > 0.05 * min(e_loc[i], e_loc[i + 1])
            if not np.any(m): continue
            cand = np.where(m)[0]
            exp_pos = qt[i] + rr_loc[i]
            rhythm = np.exp(-0.5 * ((pk[cand] - exp_pos) / (0.2 * rr_loc[i])) ** 2)
            s = (np.clip((ratio[cand] - r_p) / (r_q - r_p + 1e-9), 0, 1.5) + 0.3 * np.clip(best[cand], 0, 1)
                 + np.clip(1 + np.log10(e[cand] / e_loc[i]), 0, 1) + 1.0 * rhythm)
            j = cand[np.argmax(s)]
            is_q[j] = True; added += 1
        if added == 0: break
    # --- 7. remove implausible extras: (a) beat creating two short RRs that is weak / smooth,
    #        (b) of two beats closer than 0.5 x local RR, drop the weaker one if it is sub-threshold
    qi = np.where(is_q)[0]; qt = pk[qi]
    qscore = erel + 0.5 * (ratio - r_p) / (r_q - r_p + 1e-9)
    for rep in range(3):
        rr = np.diff(qt); drop = np.zeros(len(qt), bool)
        for i in range(1, len(qt) - 1):
            loc = np.median(rr[max(0, i - 8):i + 8])
            if rr[i - 1] < 0.6 * loc and rr[i] < 0.6 * loc:
                k = qi[i]
                if (erel[k] < thr) and (best[k] < 0.8 or ratio[k] < r_cut): drop[i] = True
        for i in range(len(qt) - 1):
            loc = np.median(rr[max(0, i - 8):i + 8])
            if rr[i] < 0.5 * loc:
                a, b = qi[i], qi[i + 1]
                weak, strong = (a, b) if qscore[a] < qscore[b] else (b, a)
                if (erel[weak] < thr) and (best[weak] < 0.8 or ratio[weak] < r_cut) and qscore[weak] < qscore[strong] - 0.3:
                    drop[i if weak == a else i + 1] = True
        qi = qi[~drop]; qt = pk[qi]
        if not drop.any(): break
    is_q = np.zeros(len(pk), bool); is_q[qi] = True
    # --- 8. fiducial: extreme of the dominant lead (full band)
    dom = int(np.argmax(prep.w * np.abs(temps[0]).max(axis=1)))
    sign = np.sign(temps[0][dom][np.argmax(np.abs(temps[0][dom]))])
    snap = int(0.04 * fs); fid = []
    for c in pk[is_q]:
        s, e_ = max(0, c - snap), min(prep.n, c + snap + 1)
        fid.append(s + int(np.argmax(prep.xb[dom, s:e_] * sign)))
    fid = np.array(sorted(set(fid)))
    fid = _refractory(fid, min_rr, prep.energy_s[fid])
    sc = np.array([match_template(prep.xf, fid, t, pre, post) for t in temps]).T
    return dict(qrs=fid, morph=sc.argmax(axis=1), ncc=sc.max(axis=1), templates=temps, dom_lead=dom, sign=sign,
                pre=pre, post=post, thr=thr, e_ref=e_ref, erel_all=erel, best_all=best, ratio_cut=r_cut,
                r_q=r_q, r_p=r_p, qrs_cand_mask=is_q, bad_q=bad_q, bad_p=bad_p, posterior=pst, posterior_unsup=pst_unsup)

# ----------------------------------------------------------------------------- QRS-T subtraction & P detection

def average_beats(prep, qrs, morph, pre_ms=150, post_ms=450):
    fs = prep.fs; pre, post = int(pre_ms / 1000 * fs), int(post_ms / 1000 * fs)
    k = morph.max() + 1; avg = []
    for j in range(k):
        W, keep = extract(prep.xb, qrs[morph == j], pre, post)
        avg.append(np.median(W, axis=0) if len(W) >= 3 else np.zeros((prep.nl, pre + post)))
    return np.array(avg), pre, post

def subtract_qrst(prep, qrs, morph, avg, pre, post):
    """Residual = xb minus the average QRS-T of each beat (per lead least-squares gain)."""
    res = prep.xb.copy(); n = prep.n; fs = prep.fs
    taper = np.ones(pre + post); tl = int(0.02 * fs)
    taper[:tl] = np.linspace(0, 1, tl); taper[-tl:] = np.linspace(1, 0, tl)
    core = slice(pre - int(0.04 * fs), pre + int(0.06 * fs))   # gain fitted on the QRS core only
    for i, c in enumerate(qrs):
        s, e_ = c - pre, c + post
        if s < 0 or e_ > n: continue
        t = avg[morph[i]]
        if i + 1 < len(qrs):
            lim = qrs[i + 1] - int(0.06 * fs) - s
            if lim < pre + post:
                t = t.copy(); t[:, max(lim, 0):] = 0
        seg = res[:, s:e_]
        for l in range(prep.nl):
            g = (seg[l, core] @ t[l, core]) / (t[l, core] @ t[l, core] + 1e-12)
            g = np.clip(g, 0.4, 1.8)
            res[l, s:e_] = seg[l] - g * t[l] * taper

    # second pass: also remove residual QRS spikes (hard blank ±30 ms around fiducial, linear interp)
    b = int(0.03 * fs)
    for c in qrs:
        s, e_ = max(0, c - b), min(n, c + b)
        for l in range(prep.nl):
            res[l, s:e_] = np.linspace(res[l, s], res[l, e_ - 1], e_ - s)
    return res

def detect_p(prep, res, qrs, pk, erel, best, thr, bad_p, min_pp_ms=180, verbose=False):
    fs = prep.fs
    pre, post = int(0.05 * fs), int(0.05 * fs)
    # P seeds: candidates in the P energy mode, well away from any QRS, in good signal
    dq = np.abs(pk[:, None] - qrs[None, :]).min(axis=1) if len(qrs) else np.full(len(pk), 1e9)
    seeds = pk[~bad_p[pk] & (dq > int(0.15 * fs)) & (erel > thr - 1.1) & (erel < thr) & (best > 0.3)]
    if len(seeds) < 20:
        return dict(p=np.array([], int), ncc=np.array([]), amp=np.array([]), template=None, mf=None)
    resf = np.array([bandpass(v, fs, 2, 30) for v in res])
    W, keep = extract(resf, seeds, pre, post)
    lab, temps = kmeans_templates(W, kmax=2, min_frac=0.2, gain=0.05)
    ptemp = temps[np.argmax(np.bincount(lab))]
    if verbose: print(f'  P seeds {len(seeds)}  P morphs {len(temps)}')
    t = ptemp - ptemp.mean(axis=1, keepdims=True); L = pre + post
    mf = np.zeros(prep.n)
    for l in range(prep.nl):
        mf += prep.w[l] * np.correlate(resf[l], t[l], mode='same')
    loc = np.sqrt(np.maximum(0, moving_avg(np.sum(prep.w[:, None] * resf ** 2, axis=0), L) * L)) + 1e-9
    tn = np.sqrt(np.sum(prep.w[:, None] * t ** 2)) + 1e-9
    nccf = mf / (loc * tn)
    amp_tr = mf / (tn ** 2)                       # projection gain (1 = template amplitude)
    cand, _ = sg.find_peaks(nccf, distance=int(min_pp_ms / 1000 * fs), height=0.5)
    # local amplitude reference: median projection at confident seeds within ±60 s
    ref = local_median(seeds, amp_tr[seeds], cand, 40)
    ok = (amp_tr[cand] > 0.3 * ref) & (nccf[cand] > 0.55)
    p = cand[ok]; pn = nccf[p]; pa = amp_tr[p]
    # exclude P coincident with a QRS (< 50 ms): QRS residuals
    if len(qrs):
        dq = np.abs(p[:, None] - qrs[None, :]).min(axis=1)
        m = dq > int(0.05 * fs); p, pn, pa = p[m], pn[m], pa[m]
    # rhythm consistency for P inside the QRS-T window (50-350 ms after a QRS):
    # require the PP intervals to be consistent with the local PP rhythm, or a strong match
    if len(qrs) and len(p) > 10:
        j = np.searchsorted(qrs, p) - 1
        after = np.where(j >= 0, p - qrs[np.clip(j, 0, len(qrs) - 1)], 1e9)
        inwin = (after > 0) & (after < int(0.35 * fs))
        pp = np.diff(p)
        keep = np.ones(len(p), bool)
        for i in np.where(inwin)[0]:
            if i == 0 or i == len(p) - 1: continue
            loc = np.median(pp[max(0, i - 6):i + 6])
            d_prev, d_next = p[i] - p[i - 1], p[i + 1] - p[i]
            fits = (abs(d_prev - loc) < 0.25 * loc) or (abs(d_next - loc) < 0.25 * loc) or \
                   (abs(d_prev + d_next - 2 * loc) < 0.3 * loc and abs(d_prev - d_next) < 0.5 * loc)
            strong = pn[i] > 0.8 and pa[i] > 0.6 * ref[cand.searchsorted(p[i])] if False else pn[i] > 0.85
            if not (fits or strong): keep[i] = False
        p, pn, pa = p[keep], pn[keep], pa[keep]
    return dict(p=p, ncc=pn, amp=pa, template=ptemp, mf=nccf, pre=pre, post=post, resf=resf, seeds=seeds)

# ----------------------------------------------------------------------------- T wave

def delineate_t(prep, qrs, morph, avg, pre, post, min_t_uv=8.0):
    """T peak / T end per beat.
    1. Per QRS morphology, the median beat (15-Hz low-passed, dominant lead) gives a template T peak and
       T end (tangent method). The T is 'measurable' only if its amplitude exceeds min_t_uv and 3x the
       baseline noise of the median beat.
    2. Per beat, T end is refined within +/-40 ms of the template offset if that beat's own T deflection
       is above 3x its local noise; otherwise the template offset is used and t_from_template=True.
    Returns tpeak, tend (sample indices, -1 if unmeasurable), t_from_template (bool), t_amp_uv per morphology."""
    fs = prep.fs; n = prep.n; xl = prep.xl
    dom = int(np.argmax(prep.w * np.abs(avg[0]).max(axis=1)))
    rr_med = np.median(np.diff(qrs)) if len(qrs) > 2 else int(fs)
    templ = []
    for j in range(len(avg)):
        W, _ = extract(xl, qrs[morph == j], pre, post)
        if len(W) < 5: templ.append(None); continue
        med_all = np.median(W, axis=0)
        # lead for T: the one with the largest T deflection (80-450 ms after the fiducial) relative to its baseline noise
        amps, snrs = [], []
        for l in range(prep.nl):
            bs = med_all[l][pre - int(0.12 * fs):pre - int(0.05 * fs)]
            sgm = med_all[l][pre + int(0.08 * fs):min(len(med_all[l]), pre + int(0.45 * fs))] - np.median(bs)
            amps.append(np.abs(sgm).max() if len(sgm) else 0); snrs.append(amps[-1] / (np.std(bs) + 1e-6))
        amps, snrs = np.array(amps), np.array(snrs)
        okl = np.where(snrs > 3)[0]
        tl = int(okl[np.argmax(amps[okl])]) if len(okl) else int(np.argmax(snrs))
        a = med_all[tl]
        base_seg = a[pre - int(0.12 * fs):pre - int(0.05 * fs)]
        base = np.median(base_seg); noise = np.std(base_seg) + 1e-6
        # QRS end: first point after +30 ms where |a-base| stays < 25 % of QRS amplitude for 20 ms
        amp_q = np.abs(a[pre] - base)
        j_pt = pre + int(0.04 * fs)
        for k in range(pre + int(0.03 * fs), pre + int(0.12 * fs)):
            if np.all(np.abs(a[k:k + int(0.02 * fs)] - base) < 0.25 * amp_q): j_pt = k; break
        w_end = int(min(len(a), pre + min(0.45 * fs, 0.6 * rr_med)))
        seg = a[j_pt + int(0.02 * fs):w_end] - base
        if len(seg) < int(0.05 * fs): templ.append(None); continue
        tp = j_pt + int(0.02 * fs) + int(np.argmax(np.abs(seg)))
        t_amp = a[tp] - base
        if abs(t_amp) * 1000 < min_t_uv or abs(t_amp) < 3 * noise: templ.append(dict(ok=False, amp_uv=abs(t_amp) * 1000, lead=tl)); continue
        e_ = min(len(a) - 1, tp + int(0.20 * fs))
        d = np.gradient(a[tp:e_]); sgn = np.sign(t_amp)
        m = tp + int(np.argmin(sgn * d)); slope = d[m - tp]
        te = int(np.clip(m + (base - a[m]) / slope, tp + 2, e_)) if abs(slope) > 1e-9 else -1
        if te > 0 and (te - pre) < int(0.15 * fs): te = -1        # QT < 150 ms is not a T wave
        templ.append(dict(ok=te > 0, tp=tp - pre, te=te - pre, amp_uv=abs(t_amp) * 1000, sign=sgn, lead=tl, base_off=(pre - int(0.12 * fs), pre - int(0.05 * fs))))
    tpk = np.full(len(qrs), -1); tend = np.full(len(qrs), -1); from_t = np.zeros(len(qrs), bool)
    for i, c in enumerate(qrs):
        T = templ[morph[i]]
        if T is None or not T['ok']: continue
        dom = T['lead']
        nxt = qrs[i + 1] if i + 1 < len(qrs) else n
        if c + T['te'] >= min(n - 1, nxt - int(0.03 * fs)):
            continue
        b0, b1 = c - int(0.12 * fs), c - int(0.05 * fs)
        if b0 < 0: continue
        base = np.median(xl[dom, b0:b1]); noise = np.std(xl[dom, b0:b1]) + 1e-6
        lo, hi = c + T['tp'] - int(0.04 * fs), c + T['tp'] + int(0.04 * fs)
        seg = (xl[dom, lo:hi] - base) * T['sign']
        if seg.max() > 3 * noise:                      # this beat's T is visible: refine
            tp = lo + int(np.argmax(seg))
            e_ = min(n - 1, tp + int(0.20 * fs), nxt - int(0.03 * fs))
            d = np.gradient(xl[dom, tp:e_]) if e_ - tp > 4 else None
            if d is not None:
                m = tp + int(np.argmin(T['sign'] * d)); slope = d[m - tp]
                if abs(slope) > 1e-9:
                    te = int(np.clip(m + (base - xl[dom, m]) / slope, tp + 2, e_))
                    # accept only if within 60 ms of template end; otherwise fall back
                    if abs(te - (c + T['te'])) <= int(0.06 * fs):
                        tpk[i] = tp; tend[i] = te; continue
        tpk[i] = c + T['tp']; tend[i] = c + T['te']; from_t[i] = True
    return dict(tpeak=tpk, tend=tend, dom_lead=dom, t_from_template=from_t,
                t_template=[(None if T is None else dict(ok=T['ok'], amp_uv=round(T['amp_uv'], 1), lead=int(T.get('lead', dom)),
                             qt_ms=(None if not T['ok'] else round(T['te'] / fs * 1000, 1)))) for T in templ])

# ----------------------------------------------------------------------------- association

def associate(qrs, p, fs, pr_min_ms=50, pr_max_ms=450):
    pr = np.full(len(qrs), np.nan); pidx = np.full(len(qrs), -1)
    if len(p) == 0: return pr, pidx
    j = np.searchsorted(p, qrs) - 1
    for i, q in enumerate(qrs):
        if j[i] >= 0:
            d = (q - p[j[i]]) / fs * 1000
            if pr_min_ms <= d <= pr_max_ms: pr[i] = d; pidx[i] = j[i]
    return pr, pidx

# ----------------------------------------------------------------------------- driver

def p_grid_fit(c, p, fs, tol=0.15):
    """Does time c (samples) fall on the local P-wave grid? Returns (fits, local_pp)."""
    win = int(3 * fs)
    near = p[(p > c - win) & (p < c + win) & (np.abs(p - c) > int(0.06 * fs))]
    if len(near) < 3: return False, np.nan
    pp = np.median(np.diff(near))
    if not np.isfinite(pp) or pp <= 0: return False, np.nan
    before = near[near < c]; after = near[near > c]
    ok = []
    if len(before):
        ph = ((c - before[-1]) / pp) % 1.0; ok.append(min(ph, 1 - ph) < tol)
    if len(after):
        ph = ((after[0] - c) / pp) % 1.0; ok.append(min(ph, 1 - ph) < tol)
    return (len(ok) > 0 and all(ok)), pp

def resolve_ambiguous(prep, pk, q, P, fs, lo=0.03, hi=0.97, verbose=False):
    """Candidates with intermediate QRS posterior: decide by rhythm context.
    On the P grid -> P; off the P grid (and RR-plausible) -> QRS. Returns updated qrs, morph, ncc, p, flags."""
    qrs, p = q['qrs'], P['p']
    pst = q['posterior']; erel = q['erel_all']
    amb = np.where((pst > lo) & (pst < hi) & (erel > q['thr'] - 0.7))[0]
    add_q, drop_p, review = [], [], []
    for i in amb:
        c = pk[i]
        near_q = np.abs(qrs - c).min() if len(qrs) else 1e9
        fits, pp = p_grid_fit(c, p, fs)
        is_q_now = near_q < int(0.06 * fs)
        if fits and is_q_now:
            # currently QRS but sits on the P grid -> demote only if weak
            if pst[i] < 0.6: review.append((c, 'QRS-on-P-grid'))
        elif (not fits) and (not is_q_now) and near_q > int(0.22 * fs) and np.isfinite(pp) and pst[i] > 0.15 \
                and q['best_all'][i] > 0.5 and erel[i] > q['thr'] - 0.5:
            add_q.append(c); review.append((c, 'off-P-grid->QRS'))
            dp = np.abs(p - c); drop_p += list(np.where(dp < int(0.06 * fs))[0])
        elif not is_q_now and pst[i] > 0.3:
            review.append((c, 'uncertain'))
    if verbose: print(f'  ambiguous {len(amb)}: promoted {len(add_q)}, review {len(review)}')
    if add_q:
        dom, sign, snap = q['dom_lead'], q['sign'], int(0.04 * fs)
        fid = []
        for c in add_q:
            s, e_ = max(0, c - snap), min(prep.n, c + snap + 1)
            fid.append(s + int(np.argmax(prep.xb[dom, s:e_] * sign)))
        qrs = np.sort(np.concatenate([qrs, fid]))
    if drop_p:
        p = np.delete(p, np.unique(drop_p))
    return qrs, p, review

def run(x, fs, verbose=True, model=None, min_rr_ms=220, min_pp_ms=180):
    """Detect QRS, P and T in a (leads x samples) array. NaN marks missing samples."""
    x = np.atleast_2d(np.asarray(x, float))
    missing = ~np.isfinite(x)
    if missing.any():
        x = np.where(missing, 0.0, x)
    prep = Prep(x, fs, missing=missing)
    pk = find_candidates(prep)
    loge, ratio, snr = peak_features(prep, pk)
    q = detect_qrs(prep, pk, loge, ratio, snr, min_rr_ms=min_rr_ms, verbose=verbose, model=model)
    qrs, morph = q['qrs'], q['morph']
    avg, pre, post = average_beats(prep, qrs, morph)
    res = subtract_qrst(prep, qrs, morph, avg, pre, post)
    P = detect_p(prep, res, qrs, pk, q['erel_all'], q['best_all'], q['thr'], q['bad_q'], min_pp_ms=min_pp_ms, verbose=verbose)
    # rhythm-context resolution of ambiguous deflections, then redo P on the updated QRS set
    qrs2, p2, review = resolve_ambiguous(prep, pk, q, P, fs, verbose=verbose)
    if len(qrs2) != len(qrs):
        sc = np.array([match_template(prep.xf, qrs2, t, q['pre'], q['post']) for t in q['templates']]).T
        qrs, morph = qrs2, sc.argmax(axis=1); q['ncc'] = sc.max(axis=1)
        avg, pre, post = average_beats(prep, qrs, morph)
        res = subtract_qrst(prep, qrs, morph, avg, pre, post)
        P = detect_p(prep, res, qrs, pk, q['erel_all'], q['best_all'], q['thr'], q['bad_q'], min_pp_ms=min_pp_ms, verbose=False)
    T = delineate_t(prep, qrs, morph, avg, pre, post)
    pr, pidx = associate(qrs, P['p'], fs)
    # per-beat QRS confidence = posterior of the nearest candidate
    ci = np.clip(np.searchsorted(pk, qrs), 0, len(pk) - 1)
    ci2 = np.clip(ci - 1, 0, len(pk) - 1)
    near = np.where(np.abs(pk[ci] - qrs) <= np.abs(pk[ci2] - qrs), ci, ci2)
    qconf = q['posterior'][near]
    return dict(prep=prep, cand=pk, loge=loge, ratio=ratio, snr=snr, qrs=qrs, morph=morph, qrs_ncc=q['ncc'],
                qrs_conf=qconf, qrs_templates=q['templates'], avg=avg, avg_pre=pre, residual=res, p=P['p'],
                p_ncc=P['ncc'], p_amp=P.get('amp'), p_info=P, tpeak=T['tpeak'], tend=T['tend'], pr_ms=pr,
                p_of_qrs=pidx, dom_lead=q['dom_lead'], qrs_info=q, t_from_template=T['t_from_template'], t_template=T['t_template'], quality=~q['bad_q'], quality_p=~q['bad_p'],
                qrs_good=~q['bad_q'][qrs], p_good=~q['bad_p'][P['p']] if len(P['p']) else np.array([], bool),
                review=review)
