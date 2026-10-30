"""Convert a Ponemah experiment and analyze it from Python.

    python scripting.py <Ponemah experiment folder> <output folder>
"""
import os
import sys

from teleep.analysis import AnalysisSettings, analyze
from teleep.convert import convert_experiment
from teleep.export import export_all
from teleep.filters import FilterSpec
from teleep.species import bands_from_heart_rate

experiment, out = sys.argv[1], sys.argv[2]

# 1. Ponemah -> hourly EDF+ files (local time; dropouts annotated)
edf_dir = os.path.join(out, "edf")
convert_experiment(experiment, edf_dir, chunk_minutes=60)

# 2. Analysis settings: pig preset, 60-Hz notch before detection, 2-minute epochs,
#    HRV bands scaled to a resting heart rate of 110 bpm
settings = AnalysisSettings(
    species="pig",
    prefilter=FilterSpec(highpass_hz=None, lowpass_hz=None, notch_hz=60),
    epoch_s=120,
    bands=bands_from_heart_rate(110),
)
res = analyze([edf_dir], settings)

# 3. Results are plain Python objects and numpy arrays
rr = res.rr                                # interval series: rr.t, rr.ms, rr.valid, rr.flag
print("valid RR intervals:", rr.n_valid, "of", len(rr))
print("RMSSD (whole recording): %.1f ms" % res.summary["RR"]["rmssd_ms"])
for e in res.epochs[:5]:
    if e["analyzed"]:
        print(e["series"], e["epoch_start"], "HF %.1f ms2" % e["hf_power_ms2"])

# 4. Write every table, the Excel workbook and the overview figure
export_all(res, os.path.join(out, "results"))
