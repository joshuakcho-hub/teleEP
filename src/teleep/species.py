"""Species presets: detection time constants, physiological RR limits and HRV frequency bands.

Every value can be overridden in :class:`teleep.analysis.AnalysisSettings`. The HRV bands are:

human  Task Force of the ESC and NASPE (1996). Circulation 93:1043-1065.
dog    Behar JA et al. (2018) Front Physiol 9:1001 (Gaussian-mixture fit to canine data).
mouse  Behar JA et al. (2018) Front Physiol 9:1001 (Gaussian-mixture fit to murine data).
rat    Aubert AE et al. (1999), as cited in Behar et al. (2018).
pig    No reference bands exist. Bands are estimated with the heart-rate scaling law of
       Behar et al. (2018) at a heart rate of 90 bpm:
       VLF/LF = 0.0037 * HR^0.58, LF/HF = 0.0017 * HR^1.01, HF upper = 0.0128 * HR^0.86.
       Use ``bands_from_heart_rate`` to recompute them for animals with a different rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

Band = Tuple[float, float]


@dataclass(frozen=True)
class SpeciesPreset:
    name: str
    typical_hr_bpm: float
    detector: str                     # "pigecg" (multi-lead P/QRS/T) or "generic" (R peaks only)
    qrs_band_hz: Tuple[float, float]  # band used to find QRS complexes
    qrs_width_ms: float
    min_rr_ms: float                  # refractory period between two QRS complexes
    rr_range_ms: Tuple[float, float]  # intervals outside are excluded from HRV
    pause_ms: Optional[float]         # RR above this is reported as a pause (None: 2 x median RR)
    bands: Dict[str, Band] = field(default_factory=dict)
    pnn_ms: Tuple[float, ...] = (50.0,)
    interp_hz: float = 4.0            # resampling rate of the RR series for spectral analysis
    welch_window_s: float = 64.0      # Welch segment length
    recommended_fs_hz: float = 250.0
    band_source: str = ""


def bands_from_heart_rate(hr_bpm: float) -> Dict[str, Band]:
    """HRV bands from mean heart rate with the scaling law of Behar et al. (2018)."""
    vlf_lf = 0.0037 * hr_bpm ** 0.58
    lf_hf = 0.0017 * hr_bpm ** 1.01
    hf_up = 0.0128 * hr_bpm ** 0.86
    return {"VLF": (0.0033, round(vlf_lf, 4)), "LF": (round(vlf_lf, 4), round(lf_hf, 4)),
            "HF": (round(lf_hf, 4), round(hf_up, 4))}


PRESETS: Dict[str, SpeciesPreset] = {
    "pig": SpeciesPreset(
        name="pig", typical_hr_bpm=90, detector="pigecg", qrs_band_hz=(5, 40), qrs_width_ms=80,
        min_rr_ms=220, rr_range_ms=(240, 3000), pause_ms=2000,
        bands={"VLF": (0.0033, 0.05), "LF": (0.05, 0.16), "HF": (0.16, 0.61)},
        pnn_ms=(50.0,), interp_hz=4.0, welch_window_s=64.0, recommended_fs_hz=250,
        band_source="Behar et al. 2018 scaling law at 90 bpm (no published porcine reference bands)"),
    "human": SpeciesPreset(
        name="human", typical_hr_bpm=75, detector="generic", qrs_band_hz=(5, 30), qrs_width_ms=100,
        min_rr_ms=250, rr_range_ms=(300, 2000), pause_ms=2000,
        bands={"VLF": (0.0033, 0.04), "LF": (0.04, 0.15), "HF": (0.15, 0.4)},
        pnn_ms=(50.0,), interp_hz=4.0, welch_window_s=64.0, recommended_fs_hz=250,
        band_source="Task Force of the ESC and NASPE 1996"),
    "dog": SpeciesPreset(
        name="dog", typical_hr_bpm=110, detector="generic", qrs_band_hz=(5, 40), qrs_width_ms=60,
        min_rr_ms=200, rr_range_ms=(200, 3000), pause_ms=2000,
        bands={"VLF": (0.0033, 0.067), "LF": (0.067, 0.235), "HF": (0.235, 0.877)},
        pnn_ms=(50.0,), interp_hz=4.0, welch_window_s=64.0, recommended_fs_hz=250,
        band_source="Behar et al. 2018 (canine fit)"),
    "rat": SpeciesPreset(
        name="rat", typical_hr_bpm=350, detector="generic", qrs_band_hz=(10, 100), qrs_width_ms=25,
        min_rr_ms=100, rr_range_ms=(100, 500), pause_ms=None,
        bands={"VLF": (0.0033, 0.19), "LF": (0.19, 0.74), "HF": (0.78, 2.5)},
        pnn_ms=(50.0, 6.0), interp_hz=10.0, welch_window_s=25.6, recommended_fs_hz=1000,
        band_source="Aubert et al. 1999 (cited in Behar et al. 2018)"),
    "mouse": SpeciesPreset(
        name="mouse", typical_hr_bpm=550, detector="generic", qrs_band_hz=(10, 150), qrs_width_ms=15,
        min_rr_ms=60, rr_range_ms=(60, 400), pause_ms=None,
        bands={"VLF": (0.0056, 0.152), "LF": (0.152, 1.24), "HF": (1.24, 3.471)},
        pnn_ms=(6.0,), interp_hz=20.0, welch_window_s=12.8, recommended_fs_hz=1000,
        band_source="Behar et al. 2018 (murine fit)"),
}


def get_preset(name: str) -> SpeciesPreset:
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown species '{name}'; choose one of {sorted(PRESETS)}") from None


def with_overrides(p: SpeciesPreset, **kw) -> SpeciesPreset:
    kw = {k: v for k, v in kw.items() if v is not None}
    return replace(p, **kw)
