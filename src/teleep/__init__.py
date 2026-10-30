"""TeleEP: analysis of implantable telemetry ECG.

Modules
-------
io.ponemah   Read DSI Ponemah experiments (.PnmWav, .PnmExp, .PnmMarks) without DSI software.
io.edf       Write and read EDF / EDF+ files.
convert      Convert Ponemah experiments to EDF.
filters      Zero-phase digital filters that tolerate signal dropouts.
detect       R-peak detection (generic) and P/QRS/T detection for conduction disease (pigecg).
rr           RR (and PP) interval series, artifact flagging, pauses.
hrv          Heart rate variability: time domain, frequency domain, nonlinear.
analysis     End-to-end analysis of one or more EDF files.
export       CSV, Excel and JSON export.
"""

__version__ = "1.0.0"
