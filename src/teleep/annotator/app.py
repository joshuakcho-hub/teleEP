#!/usr/bin/env python3
"""
EP Annotator: standalone P / QRS / T detection + correction tool for the pig telemetry EDFs.

    teleep annotate "<folder with EDF files>"        # opens http://127.0.0.1:8765 in your browser
    teleep annotate "<folder>" --port 8800 --no-browser

* Runs the pigecg detector on the EDF you pick (cached in <folder>/pigecg_results/<edf>/).
* Shows all leads with QRS (colour = morphology), P and T-end marks.
* You correct the marks (add / delete / relabel P<->QRS) and mark stretches as REVIEWED.
* Corrections are saved to <folder>/annotations/<edf>.corrections.json and are the
  training data for train.py.

Only numpy + scipy are needed (pywebview is optional for a native window).
"""
import argparse, json, os, sys, threading, webbrowser, glob, time, datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import numpy as np

from ..detect import pigecg
from ..io.edf import EdfFile

HERE = os.path.dirname(os.path.abspath(__file__))


def read_edf(path):
    """ECG leads of an EDF file (dropout samples as NaN)."""
    e = EdfFile(path)
    x, fs, labels = e.read_matrix(e.ecg_labels() or e.labels)
    return dict(sigs=x, fs=fs, labels=labels, start=e.start)

STATE = dict(folder=None, edf=None, R=None, e=None, cand=None)
LOCK = threading.Lock()


def results_dir(folder, stem):
    d = os.path.join(folder, 'pigecg_results', stem); os.makedirs(d, exist_ok=True); return d

def ann_path(folder, stem):
    d = os.path.join(folder, 'annotations'); os.makedirs(d, exist_ok=True)
    return os.path.join(d, stem + '.corrections.json')

def load_model(folder):
    p = os.path.join(folder, 'annotations', 'pigecg_model.json')
    return json.load(open(p)) if os.path.exists(p) else None


def analyze(folder, edf_name, force=False, leads=None):
    """Run (or load cached) detection for one EDF; returns the marks payload."""
    path = os.path.join(folder, edf_name); stem = os.path.splitext(edf_name)[0]
    e = read_edf(path); x, fs = e['sigs'], e['fs']
    t0 = time.time()
    sel = list(range(x.shape[0])) if not leads else [int(v) for v in leads.split(',') if v.strip() != '']
    sel = [i for i in sel if 0 <= i < x.shape[0]] or list(range(x.shape[0]))
    R = pigecg.run(x[sel], fs, verbose=False, model=load_model(folder))
    # keep the full 3-lead signal for display (detection used only `sel`)
    R['prep_display'] = pigecg.Prep(np.nan_to_num(x), fs, missing=~np.isfinite(x)) if len(sel) != x.shape[0] else R['prep']
    prep = R['prep']
    q = R['qrs_info']
    seg = lambda mask: [[float(a / fs), float(b / fs)] for a, b in _segments(mask)]
    marks = dict(
        file=edf_name, fs=fs, n=int(prep.n), duration=float(prep.n / fs), start=e['start'].isoformat(),
        labels=e['labels'], leads_used=sel,
        qrs=[round(float(v / fs), 4) for v in R['qrs']], qrs_morph=[int(m) for m in R['morph']],
        qrs_conf=[round(float(c), 3) for c in R['qrs_conf']],
        p=[round(float(v / fs), 4) for v in R['p']], p_ncc=[round(float(c), 3) for c in R['p_ncc']],
        t_end=[(round(float(v / fs), 4) if v > 0 else None) for v in R['tend']],
        t_peak=[(round(float(v / fs), 4) if v > 0 else None) for v in R['tpeak']],
        bad=seg(R['quality']), bad_p=seg(R['quality_p']),
        review=[[round(float(c / fs), 3), why] for c, why in R['review']],
        # every candidate deflection with its features (so user-added marks snap to real deflections
        # and train.py can learn from them)
        cand=[round(float(v / fs), 4) for v in R['cand']],
        cand_feat=np.column_stack([q['erel_all'], q['best_all'], R['ratio'], R['snr'], q['posterior'],
                                   q.get('posterior_unsup', q['posterior'])]).round(3).tolist(),
        elapsed=round(time.time() - t0, 1), n_morph=int(R['morph'].max() + 1) if len(R['qrs']) else 0,
        t_template=R['t_template'])
    json.dump(marks, open(os.path.join(results_dir(folder, stem), 'marks.json' if len(sel) == x.shape[0] else f'marks_leads{"".join(map(str, sel))}.json'), 'w'))
    with LOCK:
        STATE.update(edf=edf_name, R=R, e=e)
    return marks

def _segments(mask):
    m = ~mask; d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def avgbeat(pre_ms=150, post_ms=450):
    """Median beat per QRS morphology (all display leads, 15-Hz low-passed) with the template T peak / T end,
    plus the median P wave. Lets the user see T waves that are invisible on single beats."""
    R, e = STATE['R'], STATE['e']; fs = e['fs']; P_ = R.get('prep_display', R['prep'])
    pre, post = int(pre_ms / 1000 * fs), int(post_ms / 1000 * fs)
    out = dict(t=(np.arange(-pre, post) / fs * 1000).round(1).tolist(), morphs=[], p=None, fs=fs)
    for j in range(int(R['morph'].max()) + 1 if len(R['qrs']) else 0):
        idx = R['qrs'][R['morph'] == j]
        W, _ = pigecg.extract(P_.xl, idx, pre, post)
        if len(W) < 3: continue
        med = np.median(W, axis=0)
        T = R['t_template'][j] if j < len(R['t_template']) else None
        # per-beat T stats for this morphology
        tp = R['tpeak'][R['morph'] == j]; te = R['tend'][R['morph'] == j]; ok = te > 0
        out['morphs'].append(dict(morph=j, n=int(len(W)), y=med.round(5).tolist(),
            t_ok=bool(T and T.get('ok')), t_amp_uv=(T or {}).get('amp_uv'),
            qt_ms=(T or {}).get('qt_ms'),
            tpeak_ms=(float(np.median((tp[ok] - idx[ok]) / fs * 1000)) if ok.any() else None),
            tend_ms=(float(np.median((te[ok] - idx[ok]) / fs * 1000)) if ok.any() else None),
            frac_measured=float(ok.mean()) if len(ok) else 0.0))
    if len(R['p']) > 10:
        Wp, _ = pigecg.extract(P_.xl, R['p'], int(0.1 * fs), int(0.2 * fs))
        out['p'] = dict(t=(np.arange(-int(0.1 * fs), int(0.2 * fs)) / fs * 1000).round(1).tolist(), y=np.median(Wp, axis=0).round(5).tolist(), n=int(len(Wp)))
    return out

def segment(t0, dur, display='base'):
    """Waveform window for the UI. display: raw | base (baseline removed) | band (5-40 Hz)."""
    R, e = STATE['R'], STATE['e']
    fs = e['fs']; P_ = R.get('prep_display', R['prep']); n = P_.n
    a = max(0, int(t0 * fs)); b = min(n, int((t0 + dur) * fs))
    src = {'raw': P_.raw, 'base': P_.xb, 'band': P_.xf}[display]
    y = src[:, a:b]
    # decimate for long windows so the payload stays small
    step = max(1, (b - a) // 6000)
    if step > 1:
        m = (b - a) // step * step
        yy = y[:, :m].reshape(y.shape[0], -1, step)
        # min/max envelope preserves spikes
        y = np.stack([yy.min(axis=2), yy.max(axis=2)], axis=2).reshape(y.shape[0], -1)
        t = (a + np.repeat(np.arange(0, m, step), 2) + step / 2) / fs
    else:
        t = np.arange(a, b) / fs
    return dict(t0=float(a / fs), fs=fs, step=step, t=t.round(4).tolist(), y=y.round(5).tolist())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        folder = STATE['folder']
        try:
            if u.path in ('/', '/index.html'):
                body = open(os.path.join(HERE, 'annotator.html'), 'rb').read()
                self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            elif u.path == '/files':
                files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(folder, '*.edf')) + glob.glob(os.path.join(folder, '*.EDF')))
                done = {os.path.basename(f).replace('.corrections.json', '') for f in glob.glob(os.path.join(folder, 'annotations', '*.corrections.json'))}
                self._json(dict(folder=folder, files=files, annotated=sorted(done), model=load_model(folder) is not None))
            elif u.path == '/analyze':
                self._json(analyze(folder, qs['file'], force=qs.get('force') == '1', leads=qs.get('leads')))
            elif u.path == '/avgbeat':
                self._json(avgbeat())
            elif u.path == '/segment':
                self._json(segment(float(qs['t0']), float(qs['dur']), qs.get('display', 'base')))
            elif u.path == '/corrections':
                p = ann_path(folder, os.path.splitext(qs['file'])[0])
                self._json(json.load(open(p)) if os.path.exists(p) else {})
            else:
                self._json(dict(error='not found'), 404)
        except Exception as ex:
            import traceback; traceback.print_exc()
            self._json(dict(error=str(ex)), 500)

    def do_POST(self):
        u = urlparse(self.path); ln = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(ln) or b'{}')
        try:
            if u.path == '/folder':
                f = os.path.abspath(os.path.expanduser(data.get('path', '')))
                if not os.path.isdir(f): self._json(dict(error=f'not a folder: {f}'), 400); return
                STATE['folder'] = f
                self._json(dict(ok=True, folder=f, n_edf=len(glob.glob(os.path.join(f, '*.edf')) + glob.glob(os.path.join(f, '*.EDF')))))
            elif u.path == '/save':
                stem = os.path.splitext(data['file'])[0]
                data['saved'] = datetime.datetime.now().isoformat(timespec='seconds')
                json.dump(data, open(ann_path(STATE['folder'], stem), 'w'), indent=0)
                self._json(dict(ok=True, path=ann_path(STATE['folder'], stem)))
            else:
                self._json(dict(error='not found'), 404)
        except Exception as ex:
            import traceback; traceback.print_exc()
            self._json(dict(error=str(ex)), 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', nargs='?', default=None, help='folder with EDF files')
    ap.add_argument('--port', type=int, default=8765); ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    serve(a.folder, a.port, not a.no_browser)


def serve(folder=None, port=8765, open_browser=True):
    if folder is None:
        try:
            import tkinter as tk; from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); folder = filedialog.askdirectory(title='Folder with EDF files')
        except Exception:
            print('usage: app.py <folder with EDF files>'); sys.exit(1)
    STATE['folder'] = os.path.abspath(folder)
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    url = f'http://127.0.0.1:{port}/'
    print(f'EP Annotator serving {STATE["folder"]} at {url}  (Ctrl-C to quit)')
    if open_browser:
        try:
            import webview  # optional native window
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            webview.create_window('EP Annotator', url, width=1500, height=900); webview.start(); return
        except ImportError:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
