# Changelog

## 1.0.0 (2026-10-30)

First public release. Combines the laboratory's earlier tools (Ponemah2EDF converter, EDF Viewer,
pigecg detector and EP Annotator) into one installable package.

### Conversion
* Acquisition gaps are recovered from the per-block time stamps of the `.PnmWav` file (previously all
  blocks were joined, which shifts times after a gap).
* Telemetry dropouts (raw value −32768) are annotated in the EDF+ file and read back as missing values
  (previously they appeared as −10 mV signal).
* EDF start times are local wall-clock time by default, from the time zone stored in the file
  (previously UTC without indication); `--utc` keeps UTC.
* Sample rates are read from the `.PnmWav` header; the `.PnmMarks` file is no longer required.
* Temperature, battery and activity channels are calibrated from the file header (previously raw counts).
* Ponemah's beat marks can be exported (`--ponemah-marks`).
* ECG samples are identical to those written by Ponemah2EDF.
* Start times are kept to the microsecond (EDF+ time-keeping annotation) and the UTC offset of each file
  is recorded, so files are placed correctly across daylight saving time changes.
* Experiments with several subjects are written to one subfolder per subject.
* A short final piece is added to the previous file; a final fraction of a second is no longer dropped.
* Activity, temperature and battery values at the bottom of their range are no longer mistaken for
  dropout (previously an activity count of zero was read back as missing).
* EDF headers no longer depend on the language setting of the computer.

### Analysis
* New end-to-end analysis (`teleep analyze`) treating consecutive files as one recording, placed on a
  UTC time axis; files from different subjects are refused.
* pigecg detector integrated unchanged except for dropout handling; agreement with Ponemah's marks on the
  validation recording is the same as before.
* New generic multi-lead R-peak detector with species presets (pig, dog, human, rat, mouse).
* RR and PP interval series with explicit exclusion flags; pauses.
* HRV: time domain, Welch and Lomb-Scargle spectra, Poincaré, sample entropy, DFA, epochs, hourly summary.
* Export to CSV, Excel, JSON and an overview figure; filtered signal export.
* Comparison with Ponemah beat marks (`--compare-ponemah`).

### Viewer
* Butterworth filters, whole-file R-peak detection, beat editing with undo, tachogram, HRV for any range
  (identical to the Python results), multiple exports, light and dark themes.
