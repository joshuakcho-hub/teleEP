"""Synthetic telemetry ECG with known beat times, and a synthetic Ponemah experiment writer.

Used by the test suite and by ``teleep demo`` so that the whole pipeline can be exercised
without real (unpublished) animal data. The waveform is a sum of Gaussian P, QRS and T waves;
it is not a physiological model, only a realistic test signal.
"""
from __future__ import annotations

import os
import sqlite3
import struct
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np

from .io.ponemah import MAGIC, MISSING, ECG_MV_PER_COUNT, TICKS_PER_SECOND, utc_to_ticks


def _beat_times(duration_s, rate_bpm, rng, rsa_frac=0.03, lf_frac=0.02, jitter_ms=4.0, t0=0.2, slow_frac=0.05):
    times, t = [], t0
    mean_rr = 60.0 / rate_bpm
    while t < duration_s:
        times.append(t)
        rr = mean_rr * (1 + rsa_frac * np.sin(2 * np.pi * 0.25 * t) + lf_frac * np.sin(2 * np.pi * 0.1 * t)
                        + slow_frac * np.sin(2 * np.pi * t / 2400.0))
        t += rr + rng.normal(0, jitter_ms / 1000)
    return np.array(times)


def synthetic_ecg(duration_s: float = 120.0, fs: float = 500.0, atrial_rate: float = 140.0,
                  ventricular_rate: float = 60.0, rhythm: str = "complete_block", n_leads: int = 3,
                  noise_mv: float = 0.01, dropouts=((50.0, 2.0),), seed: int = 0) -> Dict[str, np.ndarray]:
    """Return dict(signals (leads x samples, mV, NaN in dropouts), fs, qrs (s), p (s)).

    rhythm: ``"sinus"`` (every P conducted, PR 120 ms) or ``"complete_block"`` (independent
    atrial and ventricular escape rhythms, as in complete atrioventricular block).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    p = _beat_times(duration_s, atrial_rate, rng)
    if rhythm == "sinus":
        q = p + 0.12
        q = q[q < duration_s - 0.5]
    else:
        q = _beat_times(duration_s, ventricular_rate, rng, rsa_frac=0.01, lf_frac=0.01, jitter_ms=6.0, t0=0.37,
                        slow_frac=0.02)
    q = q[(q > 0.3) & (q < duration_s - 0.5)]
    p = p[(p > 0.1) & (p < duration_s - 0.3)]
    gains = np.array([[1.0, -0.6, 0.8], [0.7, 0.9, 0.5], [-0.8, 0.5, 1.1]])[:n_leads]
    X = np.zeros((n_leads, n))

    def add(times, amp, center_ms, sigma_ms, lead_gain):
        w = int(6 * sigma_ms / 1000 * fs) + 1
        k = np.arange(-w, w + 1)
        shape = amp * np.exp(-0.5 * (k / (sigma_ms / 1000 * fs)) ** 2)
        for tt in times:
            c = int(round((tt + center_ms / 1000) * fs))
            a, b = max(0, c - w), min(n, c + w + 1)
            if b > a:
                for l in range(n_leads):
                    X[l, a:b] += lead_gain[l] * shape[a - (c - w):b - (c - w)]

    add(p, 0.15, 0, 10, gains[:, 0])                    # P wave
    add(q, -0.10, -14, 5, gains[:, 0])                  # Q
    add(q, 1.00, 0, 7, gains[:, 0])                     # R
    add(q, -0.30, 16, 7, gains[:, 0])                   # S
    add(q, 0.15, 250, 40, gains[:, 1])                  # T
    X += 0.05 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6))   # baseline wander
    X += rng.normal(0, noise_mv, X.shape)
    for start, dur in dropouts or ():
        a, b = int(start * fs), int((start + dur) * fs)
        X[:, a:b] = np.nan
    return dict(signals=X, fs=fs, qrs=q, p=p)


# ------------------------------------------------------------------------- Ponemah writer
def _tlv(tag: int, payload: bytes) -> bytes:
    return struct.pack("<HI", tag, len(payload)) + payload


def write_synthetic_experiment(folder: str, duration_s: float = 600.0, fs: float = 500.0,
                               subject: str = "SYN1", start_utc: Optional[datetime] = None,
                               gap_after_s: Optional[float] = None, gap_s: float = 60.0,
                               seed: int = 1, rhythm: str = "complete_block") -> dict:
    """Write a small Ponemah-like experiment (.PnmWav, .PnmExp, .PnmMarks) with 3 ECG leads,
    temperature (10 Hz) and activity (1 Hz). Returns the ground truth (beat times in UTC ticks)."""
    os.makedirs(folder, exist_ok=True)
    start_utc = start_utc or datetime(2024, 3, 12, 14, 0, 0, tzinfo=timezone.utc)
    t0 = utc_to_ticks(start_utc)
    ecg = synthetic_ecg(duration_s, fs, rhythm=rhythm, seed=seed,
                        dropouts=((min(50.0, duration_s / 3), 2.0),))
    X = ecg["signals"]
    raw_ecg = np.where(np.isfinite(X), np.clip(np.round(np.nan_to_num(X) / ECG_MV_PER_COUNT), -32256, 32255),
                       MISSING).astype("<i2")
    n_aux10 = int(duration_s * 10)
    temp = 38.5 + 0.2 * np.sin(np.arange(n_aux10) / n_aux10 * np.pi)
    raw_temp = np.round((temp - 20.0) / 25.0 * 64511 - 32256).astype("<i2")
    act = np.random.default_rng(seed).poisson(3, int(duration_s)).astype(float)
    raw_act = np.round(act / 250000.0 * 64511 - 32256).astype("<i2")

    chans = [  # id, tps, data, pmin, pmax, label, unit
        (2, int(TICKS_PER_SECOND / fs), raw_ecg[0], -14.0, 14.0, "ECG", "mV"),
        (3, int(TICKS_PER_SECOND / fs), raw_ecg[1], -14.0, 14.0, "ECG", "mV"),
        (4, int(TICKS_PER_SECOND / fs), raw_ecg[2], -14.0, 14.0, "ECG", "mV"),
        (5, 1_000_000, raw_temp, 20.0, 45.0, "Temperat", "Celsius"),
        (7, 10_000_000, raw_act, 0.0, 250000.0, "Activity", "Counts"),
    ]
    stamp = start_utc.strftime("%Y%m%d%H%M%S")
    wav = os.path.join(folder, f"SYN.{subject}.{stamp}.PnmWav")
    gap_ticks = int(gap_s * TICKS_PER_SECOND) if gap_after_s else 0
    with open(wav, "wb") as f:
        f.write(MAGIC)
        f.write(_tlv(5041, subject.encode()))
        f.write(_tlv(5042, b"Eastern Standard Time"))
        for cid, tps, _, pmin, pmax, _, _ in chans:
            f.write(_tlv(5045, struct.pack("<Q", cid)))
            f.write(_tlv(5012, struct.pack("<Q", tps)))
            f.write(_tlv(5025, struct.pack("<h", -32256)))
            f.write(_tlv(5026, struct.pack("<h", 32255)))
            f.write(_tlv(5029, struct.pack("<f", pmin)))
            f.write(_tlv(5030, struct.pack("<f", pmax)))
        # interleaved data blocks of 4 s
        block_s = 4.0
        nblk = int(np.ceil(duration_s / block_s))
        for k in range(nblk):
            for cid, tps, data, *_ in chans:
                rate = TICKS_PER_SECOND / tps
                a, b = int(round(k * block_s * rate)), int(round(min(duration_s, (k + 1) * block_s) * rate))
                if b <= a:
                    continue
                st = t0 + a * tps
                if gap_after_s and k * block_s >= np.ceil(gap_after_s / block_s) * block_s:
                    st += gap_ticks
                f.write(_tlv(5045, struct.pack("<Q", cid)))
                f.write(_tlv(5013, struct.pack("<q", st)))
                f.write(_tlv(5014, struct.pack("<q", st + (b - a) * tps)))
                f.write(_tlv(5015, data[a:b].tobytes()))
    # configuration database
    con = sqlite3.connect(os.path.join(folder, "SYN.PnmExp"))
    con.execute("CREATE TABLE ConfigInfo (ConfigSection TEXT, Metadata TEXT, Payload BLOB)")
    for cid, _, _, _, _, label, unit in chans:
        payload = f"7 50 4\r\n1 \r\n1 1\r\n{label}\r\n{unit}\r\n0\r\n1\r\n-32256 32255 0 0 0 0 0 0\r\n0\r\n"
        con.execute("INSERT INTO ConfigInfo VALUES ('CHANNEL', ?, ?)", (str(cid - 1), payload))
    con.commit()
    con.close()
    # marks database with "Ponemah" beat marks = true QRS times on lead 2
    def to_ticks(ts):
        tk = t0 + np.round(np.asarray(ts) * TICKS_PER_SECOND).astype(np.int64)
        if gap_after_s:
            cut = np.ceil(gap_after_s / block_s) * block_s
            tk = np.where(np.asarray(ts) >= cut, tk + gap_ticks, tk)
        return tk
    q_ticks = to_ticks(ecg["qrs"])
    con = sqlite3.connect(os.path.join(folder, f"SYN.1.{subject}.PnmMarks"))
    con.executescript("""
        CREATE TABLE ChannelContextInfo (ChannelContextId INTEGER PRIMARY KEY AUTOINCREMENT,
            SubjectChannelId LONG NOT NULL, TicksPerSample LONG NOT NULL, AlgoMarkVersion INTEGER NOT NULL,
            CycleMarkIndex INTEGER NOT NULL, AlgorithmId INTEGER NOT NULL);
        CREATE TABLE Cycle (ChannelContextId INTEGER NOT NULL, CycleTime LONG NOT NULL, Marks BLOB);
        CREATE TABLE BadDataMark (BadDataMarkId INTEGER PRIMARY KEY AUTOINCREMENT, SubjectChannelId LONG,
            StartTime LONG, EndTime LONG);
    """)
    for cid, tps, *_ in chans:
        con.execute("INSERT INTO ChannelContextInfo (SubjectChannelId, TicksPerSample, AlgoMarkVersion, "
                    "CycleMarkIndex, AlgorithmId) VALUES (?, ?, 2, 1, 7)", (cid, tps))
    con.executemany("INSERT INTO Cycle VALUES (1, ?, ?)", [(int(v), b"") for v in q_ticks])   # context 1 = ECG
    con.commit()
    con.close()
    return dict(wav=wav, qrs_ticks=q_ticks, p_ticks=to_ticks(ecg["p"]), qrs_s=ecg["qrs"], p_s=ecg["p"],
                start_utc=start_utc, fs=fs, duration_s=duration_s)
