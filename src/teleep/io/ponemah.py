"""Read DSI Ponemah 6.x experiments in pure Python (no DSI software, no .NET).

A Ponemah experiment folder contains, per subject recording:

``*.PnmWav``
    Binary waveform file in a tag-length-value (TLV) layout. It holds a channel
    header section (sample period, digital and physical ranges per channel) and
    then interleaved data blocks. Every data block carries its own start and end
    time, so gaps in acquisition can be recovered exactly.
``*.PnmExp``
    SQLite database with the experiment configuration (channel labels and units).
``*.PnmMarks``
    SQLite database with Ponemah's own beat marks and bad-data marks.

The format details used here were established by inspecting Ponemah 6.x files
and are documented in ``docs/ponemah-format.md``.

Calibration
-----------
ECG channels (unit ``mV``)
    millivolts = raw count x 10 / 32256. This factor was validated against
    Ponemah's own exported values (agreement to rounding precision).
Other channels (temperature, battery, activity ...)
    Linear map from the digital range to the physical range stored in the
    channel header of the .PnmWav file (for example -32256..32255 -> 20..45 degC).
Missing samples
    The raw value -32768 marks samples the receiver did not get (telemetry
    dropout). They are returned as NaN by :meth:`PonemahRecording.read`.

Times in the .PnmWav file are UTC (.NET ticks of 100 ns since 0001-01-01). The
Windows time-zone name stored in the file is used to convert to local time.
"""
from __future__ import annotations

import glob
import os
import sqlite3
import struct
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

TICKS_PER_SECOND = 10_000_000
DOTNET_EPOCH = datetime(1, 1, 1)
MISSING = -32768
ECG_MV_PER_COUNT = 10.0 / 32256.0
MAGIC = b"\xafUU\xaa"

# TLV tags (uint16) observed in Ponemah 6.x .PnmWav files
TAG_CHANNEL_ID = 5045      # uint64, sets the "current" channel
TAG_TICKS_PER_SAMPLE = 5012  # uint64, sample period in 100-ns ticks (channel header)
TAG_DIGITAL_MIN = 5025     # int16
TAG_DIGITAL_MAX = 5026     # int16
TAG_PHYSICAL_MIN = 5029    # float32
TAG_PHYSICAL_MAX = 5030    # float32
TAG_BLOCK_START = 5013     # int64 ticks (UTC)
TAG_BLOCK_END = 5014       # int64 ticks (UTC)
TAG_DATA = 5015            # int16 little-endian samples
TAG_SUBJECT = 5041         # utf-8
TAG_TIMEZONE = 5042        # utf-8, Windows time-zone id

# Windows time-zone id -> IANA name (the zones most likely to be found in lab data)
WINDOWS_TZ = {
    "Eastern Standard Time": "America/New_York",
    "US Eastern Standard Time": "America/Indiana/Indianapolis",
    "Central Standard Time": "America/Chicago",
    "Canada Central Standard Time": "America/Regina",
    "Mountain Standard Time": "America/Denver",
    "US Mountain Standard Time": "America/Phoenix",
    "Pacific Standard Time": "America/Los_Angeles",
    "Alaskan Standard Time": "America/Anchorage",
    "Hawaiian Standard Time": "Pacific/Honolulu",
    "Atlantic Standard Time": "America/Halifax",
    "Newfoundland Standard Time": "America/St_Johns",
    "Central Standard Time (Mexico)": "America/Mexico_City",
    "SA Pacific Standard Time": "America/Bogota",
    "E. South America Standard Time": "America/Sao_Paulo",
    "UTC": "UTC",
    "Coordinated Universal Time": "UTC",
    "GMT Standard Time": "Europe/London",
    "Greenwich Standard Time": "Atlantic/Reykjavik",
    "W. Europe Standard Time": "Europe/Berlin",
    "Romance Standard Time": "Europe/Paris",
    "Central Europe Standard Time": "Europe/Budapest",
    "Central European Standard Time": "Europe/Warsaw",
    "E. Europe Standard Time": "Europe/Chisinau",
    "FLE Standard Time": "Europe/Kiev",
    "GTB Standard Time": "Europe/Bucharest",
    "Russian Standard Time": "Europe/Moscow",
    "Israel Standard Time": "Asia/Jerusalem",
    "India Standard Time": "Asia/Kolkata",
    "China Standard Time": "Asia/Shanghai",
    "Taipei Standard Time": "Asia/Taipei",
    "Tokyo Standard Time": "Asia/Tokyo",
    "Korea Standard Time": "Asia/Seoul",
    "Singapore Standard Time": "Asia/Singapore",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "E. Australia Standard Time": "Australia/Brisbane",
    "Cen. Australia Standard Time": "Australia/Adelaide",
    "W. Australia Standard Time": "Australia/Perth",
    "New Zealand Standard Time": "Pacific/Auckland",
}

# Ponemah truncates labels to 8 characters in the configuration database.
_LABEL_FIXES = {"temperat": "Temperature", "activity": "Activity", "battery": "Battery"}
_UNIT_FIXES = {"celsius": "degC", "counts": "count", "volts": "V", "days": "days"}


def ticks_to_utc(ticks: int) -> datetime:
    """Convert .NET ticks (100 ns since 0001-01-01, UTC) to a timezone-aware datetime."""
    return (DOTNET_EPOCH + timedelta(microseconds=ticks // 10)).replace(tzinfo=timezone.utc)


def utc_to_ticks(dt: datetime) -> int:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    delta = dt - DOTNET_EPOCH
    return (delta.days * 86400 + delta.seconds) * TICKS_PER_SECOND + delta.microseconds * 10


def resolve_timezone(name: Optional[str]):
    """Return a tzinfo for a Windows time-zone id (or IANA name); None if unknown."""
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
    except ImportError:  # pragma: no cover
        return None
    iana = WINDOWS_TZ.get(name.strip(), name.strip())
    try:
        return ZoneInfo(iana)
    except Exception:
        return None


@dataclass
class Block:
    """One data block of one channel in the .PnmWav file."""

    offset: int   # byte offset of the first sample
    n: int        # number of samples
    start: int    # ticks (UTC) of the first sample
    end: int      # ticks (UTC) one sample period after the last sample


@dataclass
class ChannelInfo:
    channel_id: int
    label: str
    unit: str
    ticks_per_sample: int
    digital_min: int = -32256
    digital_max: int = 32255
    physical_min: float = -10.0
    physical_max: float = 10.0
    calibration: str = "header"   # "ecg-validated" | "header" | "raw-counts"
    blocks: List[Block] = field(default_factory=list)

    @property
    def sample_rate(self) -> float:
        return TICKS_PER_SECOND / self.ticks_per_sample if self.ticks_per_sample else 0.0

    @property
    def n_samples(self) -> int:
        return sum(b.n for b in self.blocks)

    @property
    def is_ecg(self) -> bool:
        return self.unit.lower() == "mv"

    @property
    def gain(self) -> float:
        return (self.physical_max - self.physical_min) / (self.digital_max - self.digital_min)

    def to_physical(self, raw: np.ndarray) -> np.ndarray:
        """Raw counts -> physical units; MISSING -> NaN."""
        raw = np.asarray(raw)
        y = (raw.astype(np.float64) - self.digital_min) * self.gain + self.physical_min
        y[raw == MISSING] = np.nan
        return y


@dataclass
class Segment:
    """A stretch of uninterrupted acquisition (ticks, UTC)."""

    start: int
    end: int

    @property
    def duration_s(self) -> float:
        return (self.end - self.start) / TICKS_PER_SECOND


def find_recordings(folder: str) -> Tuple[List[str], Optional[str], List[str]]:
    """Return (.PnmWav paths, .PnmExp path, .PnmMarks paths) found in an experiment folder."""
    def g(ext):
        out = set()
        for pat in (f"*{ext}", f"*{ext.lower()}", f"*{ext.upper()}"):
            out.update(glob.glob(os.path.join(folder, pat)))
        return sorted(out)

    exp = g(".PnmExp")
    return g(".PnmWav"), (exp[0] if exp else None), g(".PnmMarks")


def is_experiment_folder(folder: str) -> bool:
    return os.path.isdir(folder) and bool(find_recordings(folder)[0])


def _read_labels(exp_path: Optional[str]) -> Dict[int, Tuple[str, str]]:
    """{config channel number: (label, unit)} from the CHANNEL section of the .PnmExp database."""
    out: Dict[int, Tuple[str, str]] = {}
    if not exp_path or not os.path.exists(exp_path):
        return out
    con = sqlite3.connect(f"file:{exp_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT Metadata, Payload FROM ConfigInfo WHERE ConfigSection='CHANNEL'").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    for meta, payload in rows:
        try:
            num = int(meta)
        except (TypeError, ValueError):
            continue
        if payload is None:
            continue
        p = payload if isinstance(payload, str) else payload.decode("utf-8", "ignore")
        lines = p.split("\r\n")
        if len(lines) > 4:
            out[num] = (lines[3].strip(), lines[4].strip())
    return out


def _rates_from_marks(marks_paths: Iterable[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for mp in marks_paths:
        if not os.path.exists(mp):
            continue
        con = sqlite3.connect(f"file:{mp}?mode=ro", uri=True)
        try:
            for subj_ch, tps in con.execute(
                    "SELECT SubjectChannelId, TicksPerSample FROM ChannelContextInfo"):
                if tps:
                    out[int(subj_ch)] = int(tps)
        except sqlite3.Error:
            pass
        finally:
            con.close()
    return out


def _clean_label(label: str) -> str:
    lab = (label or "").strip()
    return _LABEL_FIXES.get(lab.lower(), lab)


def _clean_unit(unit: str) -> str:
    u = (unit or "").strip()
    return _UNIT_FIXES.get(u.lower(), u)


def _unique_labels(channels: List[ChannelInfo]) -> None:
    """Make labels unique. Same label + same unit -> Ponemah numbering (ECG, ECG1, ECG2);
    same label + different units -> label + unit (Battery V, Battery days)."""
    groups: Dict[str, List[ChannelInfo]] = {}
    for c in channels:
        groups.setdefault(c.label, []).append(c)
    for lab, chs in groups.items():
        if len(chs) < 2:
            continue
        units = {c.unit for c in chs}
        if len(units) == len(chs):
            for c in chs:
                c.label = f"{lab} {c.unit}".strip()
        else:
            for i, c in enumerate(chs):
                c.label = lab if i == 0 else f"{lab}{i}"


class PonemahRecording:
    """One subject recording: a .PnmWav file plus the experiment's .PnmExp / .PnmMarks.

    Parameters
    ----------
    wav_path : path to the .PnmWav file
    exp_path : optional path to the .PnmExp configuration database (channel labels and units)
    marks_paths : optional .PnmMarks databases (fallback sample rates; Ponemah beat marks)
    gap_tolerance_s : blocks whose start differs from the previous block's end by more than
        this are treated as a new acquisition segment
    """

    def __init__(self, wav_path: str, exp_path: Optional[str] = None,
                 marks_paths: Optional[List[str]] = None, gap_tolerance_s: float = 0.5):
        self.wav_path = wav_path
        self.exp_path = exp_path
        self.marks_paths = list(marks_paths or [])
        self.gap_tolerance_s = gap_tolerance_s
        with open(wav_path, "rb") as f:
            if f.read(4) != MAGIC:
                raise ValueError(f"{os.path.basename(wav_path)} is not a Ponemah .PnmWav file")
        self.subject: str = ""
        self.timezone_name: Optional[str] = None
        self.channels: List[ChannelInfo] = []
        self._scan()
        self.tzinfo = resolve_timezone(self.timezone_name)
        self.segments: List[Segment] = self._segments()

    # ------------------------------------------------------------------ scanning
    def _scan(self) -> None:
        hdr: Dict[int, dict] = {}
        blocks: Dict[int, List[Block]] = {}
        cur: Optional[int] = None
        bstart = bend = None
        size_total = os.path.getsize(self.wav_path)
        with open(self.wav_path, "rb") as f:
            f.seek(4)
            pos = 4
            while pos + 6 <= size_total:
                tb = f.read(6)
                if len(tb) < 6:
                    break
                tag, size = struct.unpack("<HI", tb)
                pos += 6
                if tag == TAG_DATA:
                    if cur is not None and bstart is not None:
                        n = size // 2
                        blocks.setdefault(cur, []).append(
                            Block(pos, n, bstart, bend if bend is not None else bstart))
                    f.seek(size, 1)
                    pos += size
                    bstart = bend = None
                    continue
                if size > 1 << 20:      # unknown large record: skip
                    f.seek(size, 1)
                    pos += size
                    continue
                v = f.read(size)
                pos += size
                if tag == TAG_CHANNEL_ID and size == 8:
                    cur = struct.unpack("<Q", v)[0]
                    hdr.setdefault(cur, {})
                elif tag == TAG_BLOCK_START and size == 8:
                    bstart = struct.unpack("<q", v)[0]
                elif tag == TAG_BLOCK_END and size == 8:
                    bend = struct.unpack("<q", v)[0]
                elif tag == TAG_SUBJECT:
                    self.subject = v.decode("utf-8", "ignore").strip("\x00").strip()
                elif tag == TAG_TIMEZONE:
                    self.timezone_name = v.decode("utf-8", "ignore").strip("\x00").strip()
                elif cur is not None and cur in hdr and cur not in blocks:
                    # channel header section
                    if tag == TAG_TICKS_PER_SAMPLE and size == 8:
                        hdr[cur]["tps"] = struct.unpack("<Q", v)[0]
                    elif tag == TAG_DIGITAL_MIN and size == 2:
                        hdr[cur]["dmin"] = struct.unpack("<h", v)[0]
                    elif tag == TAG_DIGITAL_MAX and size == 2:
                        hdr[cur]["dmax"] = struct.unpack("<h", v)[0]
                    elif tag == TAG_PHYSICAL_MIN and size == 4:
                        hdr[cur]["pmin"] = struct.unpack("<f", v)[0]
                    elif tag == TAG_PHYSICAL_MAX and size == 4:
                        hdr[cur]["pmax"] = struct.unpack("<f", v)[0]

        labels = _read_labels(self.exp_path)
        marks_tps = _rates_from_marks(self.marks_paths) if self.marks_paths else {}
        chans: List[ChannelInfo] = []
        for cid in sorted(blocks):
            h = hdr.get(cid, {})
            bl = blocks[cid]
            tps = h.get("tps") or marks_tps.get(cid)
            if not tps:
                # derive from block timing
                spans = [(b.end - b.start, b.n) for b in bl if b.n]
                tps = int(round(sum(s for s, _ in spans) / max(1, sum(n for _, n in spans))))
            label, unit = labels.get(cid - 1, (f"CH{cid}", ""))
            label, unit = _clean_label(label) or f"CH{cid}", _clean_unit(unit)
            dmin, dmax = int(h.get("dmin", -32256)), int(h.get("dmax", 32255))
            ch = ChannelInfo(cid, label, unit, int(tps), dmin, dmax, blocks=bl)
            if unit.lower() == "mv":
                ch.digital_min, ch.digital_max = -32256, 32255
                ch.physical_min = -32256 * ECG_MV_PER_COUNT
                ch.physical_max = 32255 * ECG_MV_PER_COUNT
                ch.calibration = "ecg-validated"
            elif "pmin" in h and "pmax" in h and h["pmax"] > h["pmin"]:
                ch.physical_min, ch.physical_max = float(h["pmin"]), float(h["pmax"])
                ch.calibration = "header"
            else:
                ch.physical_min, ch.physical_max = float(dmin), float(dmax)
                ch.unit = ch.unit or "count"
                ch.calibration = "raw-counts"
            chans.append(ch)
        _unique_labels(chans)
        self.channels = chans
        self.subject = self.subject or os.path.basename(self.wav_path).split(".")[0]

    def _channel_runs(self, ch: ChannelInfo) -> List[Segment]:
        tol = int(self.gap_tolerance_s * TICKS_PER_SECOND)
        runs: List[Segment] = []
        for b in ch.blocks:
            if b.n == 0:
                continue
            end = b.start + b.n * ch.ticks_per_sample
            if runs and abs(b.start - runs[-1].end) <= tol:
                runs[-1].end = max(runs[-1].end, end)
            else:
                runs.append(Segment(b.start, end))
        return runs

    def _segments(self) -> List[Segment]:
        """Acquisition segments defined by the fastest (waveform) channels."""
        if not self.channels:
            return []
        fastest = max(c.sample_rate for c in self.channels)
        ref = [c for c in self.channels if c.sample_rate == fastest]
        runs = sorted((r for c in ref for r in self._channel_runs(c)), key=lambda s: s.start)
        tol = int(self.gap_tolerance_s * TICKS_PER_SECOND)
        merged: List[Segment] = []
        for r in runs:
            if merged and r.start <= merged[-1].end + tol:
                merged[-1].end = max(merged[-1].end, r.end)
            else:
                merged.append(Segment(r.start, r.end))
        return merged

    # ------------------------------------------------------------------ public API
    @property
    def start_utc(self) -> Optional[datetime]:
        return ticks_to_utc(self.segments[0].start) if self.segments else None

    def to_local(self, ticks: int) -> datetime:
        """Ticks -> naive local datetime (local clock of the acquisition computer)."""
        utc = ticks_to_utc(ticks)
        if self.tzinfo is None:
            return utc.replace(tzinfo=None)
        return utc.astimezone(self.tzinfo).replace(tzinfo=None)

    def utc_offset(self, ticks: int) -> Optional[timedelta]:
        if self.tzinfo is None:
            return None
        return ticks_to_utc(ticks).astimezone(self.tzinfo).utcoffset()

    def channel(self, key) -> ChannelInfo:
        for c in self.channels:
            if c.channel_id == key or c.label == key:
                return c
        for c in self.channels:
            if isinstance(key, str) and c.label.lower() == key.lower():
                return c
        raise KeyError(f"no channel {key!r}; available: {[c.label for c in self.channels]}")

    def read_raw(self, ch, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """Raw int16 counts of one channel between two tick times (UTC).

        Positions without data are filled with MISSING (-32768), so the returned length is always
        round((end - start) / ticks_per_sample) and sample i is at time start + i * period.
        """
        if not isinstance(ch, ChannelInfo):
            ch = self.channel(ch)
        tps = ch.ticks_per_sample
        if start is None:
            start = self.segments[0].start if self.segments else ch.blocks[0].start
        if end is None:
            end = self.segments[-1].end if self.segments else ch.blocks[-1].end
        n = max(0, int(round((end - start) / tps)))
        out = np.full(n, MISSING, dtype=np.int16)
        if n == 0:
            return out
        with open(self.wav_path, "rb") as f:
            for b in ch.blocks:
                if b.n == 0 or b.start + b.n * tps <= start or b.start >= end:
                    continue
                # output index of the block's first sample, rounded to the nearest sample so
                # that block time stamps with sub-sample jitter neither skip nor shift samples
                shift = int(round((b.start - start) / tps))
                i0 = max(0, -shift)
                i1 = min(b.n, n - shift)
                if i1 <= i0:
                    continue
                f.seek(b.offset + 2 * i0)
                out[shift + i0:shift + i1] = np.fromfile(f, dtype="<i2", count=i1 - i0)
        return out

    def read(self, ch, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """Calibrated samples (float64, physical units) with NaN where data are missing."""
        if not isinstance(ch, ChannelInfo):
            ch = self.channel(ch)
        return ch.to_physical(self.read_raw(ch, start, end))

    # ------------------------------------------------------------------ Ponemah marks
    def ponemah_beats(self) -> Dict[str, np.ndarray]:
        """Ponemah's own beat (cycle) marks per ECG channel: {label: UTC ticks array}."""
        out: Dict[str, np.ndarray] = {}
        by_id = {c.channel_id: c for c in self.channels}
        for mp in self.marks_paths:
            con = sqlite3.connect(f"file:{mp}?mode=ro", uri=True)
            try:
                ctx = con.execute(
                    "SELECT ChannelContextId, SubjectChannelId FROM ChannelContextInfo").fetchall()
                for ctx_id, subj_ch in ctx:
                    ch = by_id.get(int(subj_ch))
                    if ch is None or not ch.is_ecg:
                        continue
                    rows = con.execute("SELECT CycleTime FROM Cycle WHERE ChannelContextId=? "
                                       "ORDER BY CycleTime", (ctx_id,)).fetchall()
                    if rows:
                        out[ch.label] = np.fromiter((r[0] for r in rows), dtype=np.int64)
            except sqlite3.Error:
                pass
            finally:
                con.close()
        return out

    def ponemah_bad_data(self) -> List[Tuple[str, int, int]]:
        """Ponemah bad-data marks: [(channel label, start ticks, end ticks)]."""
        out = []
        by_id = {c.channel_id: c for c in self.channels}
        for mp in self.marks_paths:
            con = sqlite3.connect(f"file:{mp}?mode=ro", uri=True)
            try:
                for subj_ch, s, e in con.execute(
                        "SELECT SubjectChannelId, StartTime, EndTime FROM BadDataMark ORDER BY StartTime"):
                    ch = by_id.get(int(subj_ch))
                    out.append((ch.label if ch else f"CH{subj_ch}", int(s), int(e)))
            except sqlite3.Error:
                pass
            finally:
                con.close()
        return out

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        segs = [dict(start_utc=ticks_to_utc(s.start).isoformat(),
                     start_local=self.to_local(s.start).isoformat(),
                     duration_s=round(s.duration_s, 3)) for s in self.segments]
        return dict(
            file=os.path.basename(self.wav_path), subject=self.subject,
            timezone=self.timezone_name, segments=segs,
            channels=[dict(id=c.channel_id, label=c.label, unit=c.unit,
                           sample_rate_hz=c.sample_rate, n_samples=c.n_samples,
                           digital_range=[c.digital_min, c.digital_max],
                           physical_range=[round(c.physical_min, 6), round(c.physical_max, 6)],
                           calibration=c.calibration) for c in self.channels])


def open_experiment(folder: str, gap_tolerance_s: float = 0.5) -> List[PonemahRecording]:
    """Open every subject recording (.PnmWav) in a Ponemah experiment folder."""
    wavs, exp, marks = find_recordings(folder)
    if not wavs:
        raise FileNotFoundError(f"No .PnmWav file found in {folder}")
    recs = []
    for w in wavs:
        # pair each .PnmWav with the .PnmMarks of the same subject when names allow it
        stem_parts = os.path.basename(w).split(".")
        subj = stem_parts[1] if len(stem_parts) > 2 else None
        mine = [m for m in marks if subj and f".{subj}." in os.path.basename(m)] or marks
        recs.append(PonemahRecording(w, exp, mine, gap_tolerance_s=gap_tolerance_s))
    return recs


def missing_runs(raw: np.ndarray, min_samples: int = 1) -> List[Tuple[int, int]]:
    """[(start, end)) sample index runs where raw == MISSING."""
    m = np.asarray(raw) == MISSING
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (e - s) >= min_samples
    return list(zip(s[keep].tolist(), e[keep].tolist()))


def warn_unknown_timezone(rec: PonemahRecording) -> None:
    if rec.timezone_name and rec.tzinfo is None:
        warnings.warn(f"Time zone '{rec.timezone_name}' is not recognized; times are written as UTC.")
