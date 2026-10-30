"""Pure-Python EDF / EDF+ writer and reader (numpy only).

Reference: Kemp B, Olivan J. European data format 'plus' (EDF+), an EDF alike standard
format for the exchange of physiological data. Clin Neurophysiol 2003;114:1755-61.
Specification: https://www.edfplus.info

Writer
    Signals are given either as digital integers (lossless, e.g. original ADC counts) or as
    physical floats (scaled to 16 bits). EDF+ annotations are written when supplied.
Reader
    Memory-mapped, so long recordings can be read window by window. EDF+ annotations
    (including TeleEP's "Signal dropout" marks) are parsed. Samples at the digital minimum of
    files produced from Ponemah are returned as NaN (telemetry dropout).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

Annotation = Tuple[float, float, str]   # onset (s), duration (s, 0 if none), text
DROPOUT_TEXT = "Signal dropout"
# month abbreviations required by EDF+ (independent of the computer's language setting)
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_UTC_RE = re.compile(r"\bUTC(?:([+-])(\d{1,2}):?(\d{2}))?(?![\w:+-])")


def format_utc_offset(offset: Optional[timedelta]) -> str:
    """'UTC', 'UTC-04:00' or 'UTC+05:30' (the form TeleEP writes in the recording field)."""
    if offset is None or offset == timedelta(0):
        return "UTC"
    sec = int(round(offset.total_seconds()))
    sign = "+" if sec >= 0 else "-"
    sec = abs(sec)
    return f"UTC{sign}{sec // 3600:02d}:{sec % 3600 // 60:02d}"


def parse_utc_offset(text: str) -> Optional[timedelta]:
    """UTC offset written by TeleEP in an EDF recording field, or None if absent."""
    m = _UTC_RE.search(text or "")
    if not m:
        return None
    if not m.group(1):
        return timedelta(0)
    sec = int(m.group(2)) * 3600 + int(m.group(3)) * 60
    return timedelta(seconds=sec if m.group(1) == "+" else -sec)


# =============================================================================== writer
@dataclass
class EdfSignal:
    """One signal to be written.

    ``data`` holds digital integers when ``digital=True`` (stored as is, clipped to the
    digital range) or physical values otherwise (scaled to the digital range).
    NaN in physical data is written as the digital minimum.
    """

    label: str
    data: np.ndarray
    sample_rate: float
    unit: str = "mV"
    physical_min: Optional[float] = None
    physical_max: Optional[float] = None
    digital_min: int = -32768
    digital_max: int = 32767
    digital: bool = False
    prefilter: str = ""
    transducer: str = ""

    def __post_init__(self):
        self.data = np.asarray(self.data).ravel()


def _record_duration(rates: Sequence[float]) -> float:
    """Shortest record duration (s) that holds a whole number of samples of every signal."""
    if all(float(r) == int(r) for r in rates):
        return 1.0
    fr = [Fraction(r).limit_denominator(1_000_000) for r in rates]
    den = 1
    for f in fr:
        den = den * f.denominator // math.gcd(den, f.denominator)
    num = 0
    for f in fr:
        num = math.gcd(num, f.numerator)
    return float(Fraction(den, num)) if num else 1.0


def _field(value, width: int) -> bytes:
    b = str(value).encode("ascii", "replace")[:width]
    return b + b" " * (width - len(b))


def _num8(x: float) -> str:
    """Number formatted to fit EDF's 8-character field with maximum precision."""
    x = float(x)
    if x.is_integer() and -9_999_999 <= x <= 99_999_999:
        return str(int(x))
    for dec in range(7, -1, -1):
        s = f"{x:.{dec}f}"
        if len(s) <= 8:
            return s
    raise ValueError(f"value {x} does not fit an EDF numeric field")


def _clean(s: str) -> str:
    return "".join(ch if 32 < ord(ch) < 127 else "_" for ch in str(s or ""))


def _fmt_onset(t: float) -> str:
    s = f"{abs(t):.6f}".rstrip("0").rstrip(".")
    return ("+" if t >= 0 else "-") + (s or "0")


def _tal(onset: float, duration: Optional[float], text: Optional[str]) -> bytes:
    s = _fmt_onset(onset).encode("ascii")
    if duration is not None and duration > 0:
        s += b"\x15" + f"{duration:.6f}".rstrip("0").rstrip(".").encode("ascii")
    s += b"\x14"
    if text is not None:
        s += str(text).encode("utf-8", "replace") + b"\x14"
    else:
        s += b"\x14"
    return s + b"\x00"


def _ranges(sig: EdfSignal):
    if sig.digital:
        if sig.physical_min is None or sig.physical_max is None:
            raise ValueError(f"{sig.label}: physical range required for digital data")
        return float(sig.physical_min), float(sig.physical_max), int(sig.digital_min), int(sig.digital_max)
    d = sig.data[np.isfinite(sig.data)] if sig.data.size else sig.data
    pmin = float(np.min(d)) if sig.physical_min is None and d.size else sig.physical_min
    pmax = float(np.max(d)) if sig.physical_max is None and d.size else sig.physical_max
    pmin = -1.0 if pmin is None else pmin
    pmax = 1.0 if pmax is None else pmax
    if pmin == pmax:
        pmin, pmax = pmin - 1.0, pmax + 1.0
    # round outward so the 8-character fields still enclose the data
    pmin = float(_num8(math.floor(pmin * 1e4) / 1e4)) if abs(pmin) < 1e3 else math.floor(pmin)
    pmax = float(_num8(math.ceil(pmax * 1e4) / 1e4)) if abs(pmax) < 1e3 else math.ceil(pmax)
    return pmin, pmax, int(sig.digital_min), int(sig.digital_max)


def _to_digital(sig: EdfSignal, pmin, pmax, dmin, dmax) -> np.ndarray:
    if sig.digital:
        d = np.asarray(sig.data, dtype=np.int64)
    else:
        x = np.asarray(sig.data, dtype=np.float64)
        d = np.round((x - pmin) * (dmax - dmin) / (pmax - pmin) + dmin)
        d[~np.isfinite(x)] = dmin
    return np.clip(d, dmin, dmax).astype("<i2")


def write_edf(path: str, signals: Sequence[EdfSignal], start: Optional[datetime] = None,
              patient_code: str = "", patient_name: str = "", recording_info: str = "",
              equipment: str = "", annotations: Optional[Sequence[Annotation]] = None,
              force_edfplus: bool = False, pad_last_record: bool = False) -> str:
    """Write an EDF (or EDF+C when annotations are given) file. Returns ``path``.

    Annotation onsets are in seconds from the first sample. EDF headers store the start time
    to the second; in EDF+ files the fraction of a second is kept in the time-keeping
    annotation of the first data record, as the EDF+ specification provides.
    A start time with a fraction of a second therefore always produces an EDF+ file.
    ``pad_last_record``: complete a final partial data record with the digital minimum (read
    back as missing in files from Ponemah) instead of dropping those samples.
    """
    signals = list(signals)
    if not signals:
        raise ValueError("no signals to write")
    rates = [s.sample_rate for s in signals]
    rec_dur = _record_duration(rates)
    spr = [int(round(r * rec_dur)) for r in rates]
    if pad_last_record:
        n_rec = max(int(math.ceil(s.data.size / k)) for s, k in zip(signals, spr) if k)
    else:
        n_rec = min(int(s.data.size // k) for s, k in zip(signals, spr) if k)
    if n_rec == 0:
        raise ValueError(f"not enough samples for one {rec_dur:g}-s data record")
    ranges = [_ranges(s) for s in signals]
    dig = []
    for s, r, k in zip(signals, ranges, spr):
        d = _to_digital(s, *r)[:n_rec * k]
        if d.size < n_rec * k:
            d = np.concatenate([d, np.full(n_rec * k - d.size, r[2], dtype=d.dtype)])
        dig.append(d)

    st = start or datetime(1985, 1, 1)
    frac = st.microsecond / 1e6
    # plain EDF cannot store a fraction of a second, so such files are written as EDF+
    plus = bool(annotations) or force_edfplus or frac > 0
    st = st.replace(microsecond=0)
    ann_bytes = 0
    ann_recs: List[bytearray] = []
    if plus:
        ann_recs = [bytearray(_tal(frac + r * rec_dur, None, None)) for r in range(n_rec)]
        for onset, dur, text in sorted(annotations or [], key=lambda a: a[0]):
            r = min(max(int(onset // rec_dur), 0), n_rec - 1)
            ann_recs[r] += _tal(frac + onset, dur, text)
        ann_bytes = max(len(a) for a in ann_recs)
        ann_bytes += ann_bytes % 2
        ann_bytes = max(ann_bytes, 120)
    ns = len(signals) + (1 if plus else 0)
    labels = [s.label for s in signals] + (["EDF Annotations"] if plus else [])
    units = [s.unit for s in signals] + ([""] if plus else [])
    pmins = [_num8(r[0]) for r in ranges] + (["-1"] if plus else [])
    pmaxs = [_num8(r[1]) for r in ranges] + (["1"] if plus else [])
    dmins = [str(r[2]) for r in ranges] + (["-32768"] if plus else [])
    dmaxs = [str(r[3]) for r in ranges] + (["32767"] if plus else [])
    prefs = [s.prefilter for s in signals] + ([""] if plus else [])
    trans = [s.transducer for s in signals] + ([""] if plus else [])
    sprs = spr + ([ann_bytes // 2] if plus else [])

    pid = " ".join([_clean(patient_code) or "X", "X", "X", _clean(patient_name) or "X"])
    rid = f"Startdate {st.day:02d}-{MONTHS[st.month - 1]}-{st.year:04d} X X " + (_clean(equipment) or "X")
    if recording_info:
        rid += " " + "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in recording_info)

    rec_samples = sum(sprs)
    buf = np.zeros((n_rec, rec_samples), dtype="<i2")
    off = 0
    for d, k in zip(dig, spr):
        buf[:, off:off + k] = d.reshape(n_rec, k)
        off += k
    if plus:
        ab = np.zeros((n_rec, ann_bytes), dtype=np.uint8)
        for r, a in enumerate(ann_recs):
            ab[r, :len(a)] = np.frombuffer(bytes(a), dtype=np.uint8)
        buf[:, off:] = ab.view("<i2")

    with open(path, "wb") as f:
        f.write(_field("0", 8))
        f.write(_field(pid, 80))
        f.write(_field(rid, 80))
        f.write(_field(f"{st.day:02d}.{st.month:02d}.{st.year % 100:02d}", 8))
        f.write(_field(f"{st.hour:02d}.{st.minute:02d}.{st.second:02d}", 8))
        f.write(_field(256 * (ns + 1), 8))
        f.write(_field("EDF+C" if plus else "", 44))
        f.write(_field(n_rec, 8))
        f.write(_field(_num8(rec_dur), 8))
        f.write(_field(ns, 4))
        for vals, w in ((labels, 16), (trans, 80), (units, 8), (pmins, 8), (pmaxs, 8),
                        (dmins, 8), (dmaxs, 8), (prefs, 80), (sprs, 8), ([""] * ns, 32)):
            for v in vals:
                f.write(_field(v, w))
        f.write(buf.tobytes())
    return path


# =============================================================================== reader
@dataclass
class EdfSignalInfo:
    index: int
    label: str
    unit: str
    sample_rate: float
    samples_per_record: int
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    prefilter: str
    transducer: str
    offset: int          # offset (in samples) inside one data record
    n_samples: int

    @property
    def is_annotation(self) -> bool:
        return self.label.replace(" ", "").lower() == "edfannotations"

    @property
    def gain(self) -> float:
        return (self.physical_max - self.physical_min) / (self.digital_max - self.digital_min)


@dataclass
class EdfFile:
    """Read-only, memory-mapped EDF / EDF+ file.

    ``rail_as_missing``: treat samples equal to the digital minimum as missing (NaN). ``None``
    (default) enables this automatically for files converted from Ponemah, where the digital
    minimum marks telemetry dropout and is never reached by real signal.
    """

    path: str
    rail_as_missing: Optional[bool] = None
    start: datetime = field(init=False)
    n_records: int = field(init=False)
    record_duration: float = field(init=False)
    signals: List[EdfSignalInfo] = field(init=False)
    patient: str = field(init=False)
    recording: str = field(init=False)
    edf_type: str = field(init=False)
    annotations: List[Annotation] = field(init=False)

    def __post_init__(self):
        with open(self.path, "rb") as f:
            h = f.read(256)
            if len(h) < 256:
                raise ValueError(f"{self.path}: not an EDF file")
            ns = int(h[252:256])
            sh = f.read(256 * ns)
        a = lambda o, l: h[o:o + l].decode("ascii", "replace").strip()
        self.patient, self.recording = a(8, 80), a(88, 80)
        d, t = a(168, 8), a(176, 8)
        try:
            dd, mm, yy = (int(v) for v in d.split("."))
            HH, MM, SS = (int(v) for v in t.split("."))
            year = 1900 + yy if yy >= 85 else 2000 + yy
            m = re.search(r"Startdate \d\d-[A-Za-z]{3}-(\d{4})", self.recording)
            if m:
                year = int(m.group(1))
            self.start = datetime(year, mm, dd, HH, MM, SS)
        except Exception:
            self.start = datetime(1985, 1, 1)
        header_bytes = int(a(184, 8))
        self.edf_type = a(192, 44)
        self.n_records = int(a(236, 8))
        self.record_duration = float(a(244, 8))

        def fld(off, width):
            base = off * ns
            return [sh[base + i * width: base + (i + 1) * width].decode("ascii", "replace").strip()
                    for i in range(ns)]
        labels, trans, units = fld(0, 16), fld(16, 80), fld(96, 8)
        pmin, pmax = fld(104, 8), fld(112, 8)
        dmin, dmax = fld(120, 8), fld(128, 8)
        pref, spr = fld(136, 80), fld(216, 8)
        spr_i = [int(v) for v in spr]
        rec_samples = sum(spr_i)
        file_bytes = os.path.getsize(self.path) - header_bytes
        max_rec = file_bytes // (2 * rec_samples) if rec_samples else 0
        if self.n_records < 0 or self.n_records > max_rec:
            self.n_records = max_rec
        self._header_bytes = header_bytes
        self._rec_samples = rec_samples
        # duplicate labels (e.g. three signals called "ECG" in files from older converters) are
        # made unique so that every signal can be addressed by label: ECG, ECG_2, ECG_3
        seen = {}
        for i, lab in enumerate(labels):
            if lab.replace(" ", "").lower() == "edfannotations":
                continue
            seen[lab] = seen.get(lab, 0) + 1
            if seen[lab] > 1:
                labels[i] = f"{lab}_{seen[lab]}"
        self.signals = []
        off = 0
        for i in range(ns):
            fs = spr_i[i] / self.record_duration if self.record_duration else 0.0
            self.signals.append(EdfSignalInfo(
                i, labels[i], units[i], fs, spr_i[i], float(pmin[i]), float(pmax[i]),
                int(dmin[i]), int(dmax[i]), pref[i], trans[i], off, spr_i[i] * self.n_records))
            off += spr_i[i]
        self._mm = np.memmap(self.path, dtype="<i2", mode="r", offset=header_bytes,
                             shape=(self.n_records, rec_samples)) if self.n_records else None
        self._first_tal: Optional[float] = None
        self.annotations = self._read_annotations()
        # EDF+: the time-keeping annotation of the first record holds the fraction of a second
        # by which the first sample follows the header start time
        self.start_fraction = 0.0
        if self._first_tal is not None and 0.0 < self._first_tal < 1.0:
            self.start_fraction = self._first_tal
            self.start = self.start + timedelta(seconds=self._first_tal)
            self.annotations = [(o - self._first_tal, d, t) for o, d, t in self.annotations]
        if self.rail_as_missing is None:
            self.rail_as_missing = ("ponemah" in self.recording.lower()
                                    or any(t.startswith(DROPOUT_TEXT) for _, _, t in self.annotations))

    # ---------------------------------------------------------------- properties
    @property
    def data_signals(self) -> List[EdfSignalInfo]:
        return [s for s in self.signals if not s.is_annotation]

    @property
    def labels(self) -> List[str]:
        return [s.label for s in self.data_signals]

    @property
    def duration(self) -> float:
        return self.n_records * self.record_duration

    @property
    def utc_offset(self) -> Optional[timedelta]:
        """Offset of ``start`` from UTC when the file states it (TeleEP files do), else None."""
        return parse_utc_offset(self.recording)

    @property
    def start_utc(self) -> Optional[datetime]:
        """Start time in UTC (naive datetime), or None when the time zone is not stated."""
        off = self.utc_offset
        return None if off is None else self.start - off

    def signal(self, key: Union[int, str]) -> EdfSignalInfo:
        if isinstance(key, int):
            return self.data_signals[key]
        for s in self.data_signals:
            if s.label == key:
                return s
        for s in self.data_signals:
            if s.label.lower() == str(key).lower():
                return s
        raise KeyError(f"no signal {key!r}; available: {self.labels}")

    # ---------------------------------------------------------------- data
    def read_digital(self, key, start_s: float = 0.0, duration_s: Optional[float] = None) -> np.ndarray:
        s = self.signal(key)
        if self._mm is None:
            return np.zeros(0, dtype=np.int16)
        n_tot = s.n_samples
        a = max(0, int(round(start_s * s.sample_rate)))
        b = n_tot if duration_s is None else min(n_tot, a + int(round(duration_s * s.sample_rate)))
        if b <= a:
            return np.zeros(0, dtype=np.int16)
        k = s.samples_per_record
        r0, r1 = a // k, (b - 1) // k + 1
        block = np.asarray(self._mm[r0:r1, s.offset:s.offset + k]).reshape(-1)
        return block[a - r0 * k: b - r0 * k]

    def read(self, key, start_s: float = 0.0, duration_s: Optional[float] = None,
             missing_as_nan: Optional[bool] = None) -> np.ndarray:
        """Physical values (float64). Dropout samples become NaN (see ``rail_as_missing``)."""
        s = self.signal(key)
        d = self.read_digital(key, start_s, duration_s)
        y = (d.astype(np.float64) - s.digital_min) * s.gain + s.physical_min
        use = self.rail_as_missing if missing_as_nan is None else missing_as_nan
        if use:
            y[d <= s.digital_min] = np.nan
        return y

    def read_matrix(self, keys: Optional[Sequence] = None, start_s: float = 0.0,
                    duration_s: Optional[float] = None) -> Tuple[np.ndarray, float, List[str]]:
        """Several signals with a common sample rate as a (n_signals, n_samples) array."""
        sigs = [self.signal(k) for k in keys] if keys else self.data_signals
        rates = {s.sample_rate for s in sigs}
        if len(rates) != 1:
            raise ValueError(f"signals have different sample rates: {sorted(rates)}")
        arrs = [self.read(s.label if s.label else s.index, start_s, duration_s) for s in sigs]
        n = min(len(v) for v in arrs)
        return np.vstack([v[:n] for v in arrs]), rates.pop(), [s.label for s in sigs]

    def ecg_labels(self) -> List[str]:
        """Labels of ECG-like signals (label contains 'ECG' or unit is a voltage at >= 100 Hz)."""
        out = [s.label for s in self.data_signals if "ecg" in s.label.lower()]
        if not out:
            out = [s.label for s in self.data_signals
                   if s.unit.lower() in ("mv", "uv", "v") and s.sample_rate >= 100]
        return out

    def dropouts(self) -> List[Tuple[float, float, str]]:
        """Dropout intervals from EDF+ annotations: [(start_s, end_s, text)]."""
        return [(o, o + d, t) for o, d, t in self.annotations if t.startswith(DROPOUT_TEXT)]

    # ---------------------------------------------------------------- annotations
    def _read_annotations(self) -> List[Annotation]:
        out: List[Annotation] = []
        ann = [s for s in self.signals if s.is_annotation]
        if not ann or self._mm is None:
            return out
        for s in ann:
            raw = np.asarray(self._mm[:, s.offset:s.offset + s.samples_per_record]).tobytes()
            rec_len = 2 * s.samples_per_record
            for r in range(self.n_records):
                chunk = raw[r * rec_len:(r + 1) * rec_len]
                first = True
                for tal in chunk.split(b"\x00"):
                    if not tal or tal[:1] not in (b"+", b"-"):
                        continue
                    parts = tal.split(b"\x14")
                    head = parts[0].split(b"\x15")
                    try:
                        onset = float(head[0])
                        dur = float(head[1]) if len(head) > 1 and head[1] else 0.0
                    except ValueError:
                        continue
                    if first and r == 0 and s is ann[0]:
                        self._first_tal = onset
                    first = False
                    for txt in parts[1:]:
                        if txt:
                            out.append((onset, dur, txt.decode("utf-8", "replace")))
        out.sort(key=lambda a: a[0])
        return out

    def close(self):
        self._mm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_edf(path: str, keys: Optional[Sequence] = None) -> dict:
    """Convenience: read all (or selected) equal-rate signals of an EDF file into memory."""
    e = EdfFile(path)
    if keys is None:
        keys = e.ecg_labels() or e.labels
    x, fs, labels = e.read_matrix(keys)
    return dict(signals=x, fs=fs, labels=labels, start=e.start, annotations=e.annotations,
                dropouts=e.dropouts(), duration=e.duration)


def start_plus(start: datetime, seconds: float) -> datetime:
    return start + timedelta(seconds=float(seconds))
