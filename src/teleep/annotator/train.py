#!/usr/bin/env python3
"""
train.py: learn from the corrections saved by EP Annotator.

    teleep train "<folder with EDF files>"            # report + fit model
    teleep train "<folder>" --no-fit                  # report only (how good is the detector on reviewed data?)

What it does
1. For every <folder>/annotations/<edf>.corrections.json, take the REVIEWED stretches: inside them the
   corrected QRS and P lists are treated as ground truth.
2. Runs the detector (without any learned model) on those EDFs, and scores it on the reviewed stretches:
   QRS / P sensitivity, positive predictivity, and a list of every remaining error with its time, so the
   error classes can be studied (that is the part used to improve the algorithm itself).
3. Labels every candidate deflection inside the reviewed stretches (QRS / P / neither) and fits a
   logistic model on the candidate features -> annotations/pigecg_model.json. app.py and teleep analyze
   pick that model up automatically (QRS-vs-other posterior); the report shows the before/after effect
   with leave-one-file-out cross-validation so the number is honest.
"""
import argparse, glob, json, os, sys
import numpy as np
from ..detect import pigecg
from .app import read_edf

TOL = 0.08   # s; a detection within 80 ms of a true mark counts as that mark


def in_intervals(t, iv):
    t = np.asarray(t); m = np.zeros(len(t), bool)
    for a, b in iv: m |= (t >= a) & (t <= b)
    return m

def match(det, truth, tol=TOL):
    """Return (det_matched mask, truth_matched mask)."""
    det, truth = np.asarray(det), np.asarray(truth)
    if len(det) == 0 or len(truth) == 0: return np.zeros(len(det), bool), np.zeros(len(truth), bool)
    d = np.abs(det[:, None] - truth[None, :])
    return d.min(axis=1) < tol, d.min(axis=0) < tol

def score(name, det, truth, iv):
    det = np.asarray(det)[in_intervals(det, iv)]; truth = np.asarray(truth)[in_intervals(truth, iv)]
    dm, tm = match(det, truth)
    tp = int(dm.sum()); fp = int((~dm).sum()); fn = int((~tm).sum())
    se = tp / max(1, tp + fn); ppv = tp / max(1, tp + fp)
    return dict(name=name, n_true=int(len(truth)), tp=tp, fp=fp, fn=fn, sensitivity=se, ppv=ppv,
                fp_times=det[~dm].round(3).tolist(), fn_times=truth[~tm].round(3).tolist())

def fit_logistic(X, y, l2=1e-2, iters=500):
    """Plain L2 logistic regression by Newton iterations (X already has a bias column)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w; p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (p - y) + l2 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g); w -= step
        if np.abs(step).max() < 1e-6: break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder'); ap.add_argument('--no-fit', action='store_true')
    a = ap.parse_args()
    train(a.folder, fit=not a.no_fit)


def train(folder, fit=True):
    folder = os.path.abspath(folder)
    files = sorted(glob.glob(os.path.join(folder, 'annotations', '*.corrections.json')))
    if not files: print('no corrections found in', os.path.join(folder, 'annotations')); return
    per_file = []; Xs = []; ys = []; groups = []
    total_reviewed = 0
    for cf in files:
        C = json.load(open(cf)); iv = C.get('reviewed', [])
        if not iv: print(f'{C["file"]}: no reviewed stretches, skipped'); continue
        edf = os.path.join(folder, C['file'])
        if not os.path.exists(edf): print('missing EDF', edf); continue
        e = read_edf(edf); x, fs = e['sigs'], e['fs']
        R = pigecg.run(x, fs, verbose=False, model=None)
        q = R['qrs_info']
        det_q = R['qrs'] / fs; det_p = R['p'] / fs
        sq = score('QRS', det_q, C['qrs'], iv); sp = score('P', det_p, C['p'], iv)
        rev_s = sum(b - a for a, b in iv); total_reviewed += rev_s
        per_file.append(dict(file=C['file'], reviewed_s=rev_s, qrs=sq, p=sp,
                             n_user_edits=len([l for l in C.get('log', []) if l['op'] != 'review'])))
        # candidate labels inside reviewed stretches
        ct = R['cand'] / fs; m = in_intervals(ct, iv)
        isq, _ = match(ct[m], np.asarray(C['qrs']))
        isp, _ = match(ct[m], np.asarray(C['p']))
        F = pigecg.candidate_features(q['erel_all'], q['best_all'], R['ratio'], R['snr'], q.get('posterior_unsup', q['posterior']))[m]
        Xs.append(F); ys.append(isq.astype(float)); groups.append(np.full(m.sum(), len(per_file) - 1))
        print(f"{C['file']}: reviewed {rev_s:.0f} s · QRS Se {sq['sensitivity']:.4f} PPV {sq['ppv']:.4f} (FN {sq['fn']}, FP {sq['fp']} of {sq['n_true']}) · "
              f"P Se {sp['sensitivity']:.4f} PPV {sp['ppv']:.4f} (FN {sp['fn']}, FP {sp['fp']} of {sp['n_true']})")
    if not per_file: return
    rep = dict(total_reviewed_s=total_reviewed, files=per_file)
    # pooled
    for k in ('qrs', 'p'):
        tp = sum(f[k]['tp'] for f in per_file); fp = sum(f[k]['fp'] for f in per_file); fn = sum(f[k]['fn'] for f in per_file)
        rep[k + '_pooled'] = dict(tp=tp, fp=fp, fn=fn, sensitivity=tp / max(1, tp + fn), ppv=tp / max(1, tp + fp))
        print(f"POOLED {k.upper()}: Se {rep[k + '_pooled']['sensitivity']:.4f}  PPV {rep[k + '_pooled']['ppv']:.4f}  (TP {tp} FN {fn} FP {fp}) over {total_reviewed:.0f} s reviewed")
    if fit and Xs:
        X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(groups)
        if y.sum() < 30 or (1 - y).sum() < 30:
            print('not enough labelled candidates yet to fit a model (need >= 30 QRS and >= 30 non-QRS in reviewed stretches)')
        else:
            # leave-one-file-out estimate of candidate-level accuracy, unsupervised vs supervised
            base = (X[:, 4] * 10 > 0).astype(float)           # unsupervised posterior > 0.5
            cv_pred = np.zeros(len(y))
            for gi in np.unique(g):
                tr = g != gi
                if tr.sum() < 50 or (~tr).sum() == 0: cv_pred[~tr] = base[~tr]; continue
                w = fit_logistic(X[tr], y[tr]); cv_pred[~tr] = (X[~tr] @ w > 0)
            err_b = np.mean(base != y); err_s = np.mean(cv_pred != y) if len(np.unique(g)) > 1 else None
            w = fit_logistic(X, y)
            model = dict(w=w.round(6).tolist(), b=0.0, features=['erel', 'ncc', 'ratio', 'snr', 'logit_unsup/10', 'bias'],
                         n_candidates=int(len(y)), n_qrs=int(y.sum()), trained_on=[f['file'] for f in per_file],
                         candidate_error_unsupervised=float(err_b), candidate_error_supervised_LOFO=err_s)
            json.dump(model, open(os.path.join(folder, 'annotations', 'pigecg_model.json'), 'w'), indent=1)
            print(f"model fitted on {len(y)} candidates ({int(y.sum())} QRS): candidate-level error unsupervised {err_b:.4f}"
                  + (f" -> supervised (leave-one-file-out) {err_s:.4f}" if err_s is not None else " (need >= 2 annotated files for a cross-validated number)"))
            print('saved', os.path.join(folder, 'annotations', 'pigecg_model.json'), '(the annotator and teleep analyze will use it from now on)')
            rep['model'] = model
    json.dump(rep, open(os.path.join(folder, 'annotations', 'training_report.json'), 'w'), indent=1)
    print('report:', os.path.join(folder, 'annotations', 'training_report.json'))


if __name__ == '__main__':
    main()
