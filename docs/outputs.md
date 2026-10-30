# Output files

`teleep analyze` writes the files below into the results folder. Times are given twice: `time_s` in
seconds from the start of the recording (the start of the first EDF file) and `datetime` as wall-clock
time. Empty cells mean "not applicable" or "could not be computed". The Excel workbook `results.xlsx`
holds the same tables, a Settings sheet and a Column guide sheet.

## beats.csv: one row per QRS complex

| Column | Meaning |
|---|---|
| `beat` | beat number, starting at 0 |
| `time_s`, `datetime` | time of the R peak (fiducial point) |
| `file` | EDF file that contains the beat |
| `rr_ms` | interval from the previous QRS complex |
| `rr_flag` | empty = valid NN interval; otherwise `gap`, `beat`, `range` or `artifact` (see below); `first` for the first beat |
| `hr_bpm` | instantaneous rate, 60000 / `rr_ms` |
| `confidence` | probability (0 to 1) that the deflection is a QRS complex (pigecg); similarity to the median beat (generic) |
| `template_ncc` | normalized cross-correlation with the QRS template (−1 to 1) |
| `morphology` | QRS morphology class (pigecg k-means cluster; 0 for the generic detector) |
| `good` | 1 = reliable detection in analyzable signal; 0 = excluded from HRV |
| `pr_ms` | time from the closest preceding P wave within 50 to 450 ms (pigecg); descriptive, not proof of conduction |

## pwaves.csv: one row per P wave (pigecg)

| Column | Meaning |
|---|---|
| `p`, `time_s`, `datetime`, `file` | P-wave number and time (peak of the matched P template) |
| `pp_ms`, `pp_flag` | interval from the previous P wave and its exclusion flag |
| `rate_per_min` | instantaneous atrial rate |
| `template_ncc` | match with the learned P-wave template |
| `good` | 1 = P wave in signal that is analyzable for P waves |

## Interval flags

| Flag | The interval is excluded from HRV because |
|---|---|
| `gap` | it overlaps a telemetry dropout, an unanalyzable stretch or a gap between files |
| `beat` | one of its beats is unreliable (`good` = 0) |
| `range` | it is outside the physiological range of the species preset |
| `artifact` | it differs by more than the artifact threshold (default 20 %) from the median of its neighbours: missed beat, extra beat, ectopic beat, or a P wave hidden in a QRS complex |

## hrv_epochs.csv: one row per epoch and series

`series` is `RR` (ventricular) or `PP` (atrial). `epoch_start` is the wall-clock start of the epoch.

| Column | Meaning |
|---|---|
| `coverage` | fraction of the epoch covered by analyzable intervals (not across a dropout, gap or unreliable beat) |
| `analyzed` | 1 when `coverage` ≥ 0.8 and `valid_fraction` ≥ 0.7 (defaults); HRV columns are filled only then |
| `n_intervals`, `n_nn`, `valid_fraction` | intervals in the epoch, valid NN intervals, and their ratio |
| `n_rate_intervals`, `mean_interval_ms`, `median_interval_ms`, `rate_from_median_bpm` | statistics of all intervals that are not across a gap and not at an unreliable beat (they include ectopic or blocked-beat intervals) |
| `mean_nn_ms`, `median_nn_ms`, `min_nn_ms`, `max_nn_ms` | NN interval statistics |
| `sdnn_ms` | standard deviation of NN intervals |
| `cv_nn_pct` | SDNN / mean NN × 100 |
| `mean_hr_bpm`, `sd_hr_bpm`, `min_hr_bpm`, `max_hr_bpm` | instantaneous rate statistics over NN intervals (per minute; atrial rate for PP) |
| `rmssd_ms` | root mean square of successive differences of consecutive valid NN intervals |
| `sdsd_ms` | standard deviation of those successive differences |
| `pnn50_pct` (and `pnnX_pct`) | percent of successive differences larger than 50 (X) ms |
| `hrv_triangular_index` | number of NN intervals / height of their histogram (1/128-s bins) |
| `vlf_power_ms2`, `lf_power_ms2`, `hf_power_ms2` | band powers (ms²); VLF is empty when the epoch is too short to resolve it |
| `vlf_peak_hz`, `lf_peak_hz`, `hf_peak_hz` | frequency of the largest spectral value in each band |
| `total_power_ms2` | sum of the band powers that could be resolved |
| `lf_hf_ratio`, `lf_nu`, `hf_nu` | LF/HF; LF and HF as percent of LF + HF |
| `sd1_ms`, `sd2_ms`, `sd1_sd2_ratio`, `ellipse_area_ms2` | Poincaré plot descriptors |
| `sampen` | sample entropy (m = 2, r = 0.2 × SD) |
| `dfa_alpha1`, `dfa_alpha2` | detrended fluctuation analysis exponents (4 to 16 and 16 to 64 beats) |

## hrv_hourly.csv

Mean of the analyzed epochs in each clock hour, per series: `n_epochs`, `mean_hr_bpm`, `mean_nn_ms`,
`sdnn_ms`, `rmssd_ms`, `lf_power_ms2`, `hf_power_ms2`, `lf_hf_ratio`, `sd1_ms`, `sd2_ms`.

## hrv_summary.csv and summary.json

One row per series for the whole recording: the time-domain and Poincaré columns above computed over all
NN intervals, plus `n_epochs`, `n_epochs_analyzed`, `sdann_ms` (SD of epoch mean NN), `sdnn_index_ms`
(mean of epoch SDNN) and `epoch_median_*` (median of a spectral or nonlinear measure over analyzed
epochs).

`summary.json` contains the same numbers plus all settings, the per-file table, the number of intervals
with each flag, and the atrioventricular statistics:

| Key | Meaning |
|---|---|
| `p_to_qrs_ratio` | number of P waves / number of QRS complexes |
| `frac_qrs_with_p_60_300ms` | fraction of QRS complexes with a P wave 60 to 300 ms before them |
| `pr_median_ms`, `pr_iqr_ms` | median and interquartile range of those PR values |
| `frac_pr_change_gt40ms` | fraction of consecutive beats whose PR changes by more than 40 ms (high in atrioventricular dissociation) |

## pauses.csv

Intervals longer than the pause threshold (default 2 s for pig, dog and human; twice the median RR for
rat and mouse), for RR and PP: `series`, `start_s`, `end_s`, `start` (wall clock), `duration_ms` and
`flag`. A pause whose flag is `artifact` or `beat` may be a missed beat and should be checked in the
viewer. Intervals across dropouts are never listed.

## signal_quality.csv

Stretches excluded from analysis: `start_s`, `end_s`, `duration_s`, `start` and `reason`, which is one of
`dropout` (no lead had data), `unanalyzable QRS` (flat line or heavy noise), `unanalyzable P` (noise too
high for P waves, QRS still usable), `gap between files` or `no signal`.

## files.csv

Per EDF file: `start`, `offset_s` (seconds from the recording start), `duration_s`, `fs`, `leads` used,
`n_qrs`, `n_p`, `dropout_s`, `unanalyzable_s` and `elapsed_s` (processing time).

## viewer_marks/<file>.marks.json

Marks for the TeleEP Viewer, with times relative to the start of that EDF file: `qrs`, `qrs_conf`,
`qrs_good`, `qrs_morph`, `p`, `p_good`, `bad` (excluded stretches) and `bad_p`. The viewer writes the same
format when you export corrected marks.

## Conversion outputs

`teleep convert` writes, next to the EDF files, `<subject>_<start>_conversion.json` (source file,
channels with calibration, segments, and for each EDF file its local and UTC start, UTC offset and
dropout seconds) and, with `--ponemah-marks`, `<subject>_<start>_ponemah_beats.csv` (Ponemah's beat
marks with UTC and local times). When the experiment contains several subjects, each subject's files
are written to a subfolder named after the subject.
