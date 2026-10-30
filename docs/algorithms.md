# Algorithms

This document describes every processing step in the order in which it is applied. Parameter values
are the defaults; all of them can be changed on the command line, in the desktop application or in
`teleep.analysis.AnalysisSettings`.

## 1. Conversion from Ponemah

* Samples are read from the data blocks of the `.PnmWav` file. Each block carries its own start and end
  time, so consecutive blocks whose times do not join (tolerance 0.5 s) start a new acquisition segment.
  Each block is placed at the sample position nearest to its time stamp. One set of EDF files is written
  per segment, each with its true start time.
* Files are cut at the chosen length (60 minutes by default). A final piece shorter than a quarter of
  that length is added to the previous file rather than written on its own. A final fraction of a
  second is kept by completing the last EDF data record with missing samples.
* An experiment folder that contains several subjects is written to one subfolder per subject.
* ECG is calibrated as millivolts = counts × 10 / 32256. Other channels are calibrated linearly from the
  digital range (−32256 to 32255) to the physical range stored in the channel header of the `.PnmWav`
  file (for example 20 to 45 °C for temperature).
* The count −32768 marks a sample the receiver did not get (telemetry dropout). It is written as the EDF
  digital minimum and every ECG dropout run is listed as an EDF+ annotation "Signal dropout <channel>".
  When TeleEP reads the file back, these samples become missing values (NaN). For temperature, battery
  and activity channels the EDF digital range is extended down to −32768 with the same calibration,
  because these channels can legitimately sit at the bottom of their range (an activity count of zero
  is the count −32256); only a true dropout is then read back as missing.
* Times in the `.PnmWav` file are UTC. They are converted to the local time of the acquisition computer
  using the Windows time-zone name stored in the file (daylight saving time included). The EDF header
  holds the start time to the second; the fraction of a second is stored in the first EDF+ time-keeping
  annotation, as the EDF+ specification provides. The UTC offset of each file (for example
  "UTC-04:00") is written into the EDF+ recording field.

## 2. Filters (`teleep.filters`)

| Filter | Implementation |
|---|---|
| High-pass, low-pass, band-pass | Butterworth, 4th order by default, second-order sections, applied forward and backward (zero phase, doubled effective order) |
| Notch | second-order IIR notch, Q = 30, forward and backward |
| Baseline removal | running median of 200 ms followed by a running median of 600 ms, subtracted from the signal |

Missing samples are replaced by linear interpolation before filtering and set back to NaN afterwards.
Filters change only what is displayed or exported unless `--hp/--lp/--notch` is given to `analyze`, in
which case the filtered signal is also used for detection.

## 3. Beat detection

### 3.1 pigecg (default for pigs)

Designed for subcutaneous multi-lead telemetry in pigs with conduction disease. QRS complexes and P waves
are detected independently, so atrioventricular dissociation, 2:1 block, Wenckebach periodicity, sinus
pauses and escape rhythms are all handled.

1. **Preprocessing.** Two-stage median baseline removal; a 5-40 Hz band for deflections and a 15-45 Hz band
   for sharpness. Leads are combined with time-varying weights proportional to the inverse local noise,
   so a noisy lead is down-weighted automatically; a lead in dropout gets no weight.
2. **Candidates.** Every peak of the combined energy that exceeds three times the local noise floor.
3. **QRS versus P.** The distribution of candidate energies is bimodal (QRS energy is about ten times P
   energy); the valley gives a first separation. Multi-lead QRS templates are learned by k-means, so
   conducted, junctional and ventricular morphologies each get a template. Each candidate is scored with
   class-conditional Gaussian models of energy, template correlation and sharpness, giving a
   probability (`confidence`) that it is a QRS complex.
4. **Rhythm-aware gap filling.** When an RR interval exceeds 1.45 times the local RR, the most QRS-like
   candidate inside it is accepted with a relaxed energy criterion. This recovers beats when the QRS
   amplitude falls three- to five-fold after a change of posture.
5. **P waves.** The average QRS-T of each morphology is subtracted from every beat. The residual is matched
   against a learned multi-lead P template. P waves inside the QRS-T window must fit the local PP rhythm.
6. **Ambiguous deflections.** A small narrow deflection can be a low-amplitude junctional QRS or a P wave.
   It is classified by rhythm: on the P-wave grid it is a P wave, otherwise a QRS complex.
7. **T waves.** The T peak and T end (tangent method) are found on the median beat of each morphology and
   refined beat by beat when the T wave is above the noise.
8. **Signal quality.** Flat lines, dropouts and noise bursts are marked as unanalyzable for QRS and,
   with a stricter limit, for P waves.

Beats with confidence below 0.5, or inside unanalyzable signal, are marked unreliable (`good = 0`) and the
intervals they bound are excluded from HRV. If an EP Annotator model (`annotations/pigecg_model.json`)
exists in the EDF folder, its logistic correction of the QRS probability is applied.

### 3.2 generic (default for other species)

A multi-lead form of the Pan-Tompkins detector (Pan and Tompkins 1985):

1. Each lead is band-passed (species QRS band) and divided by its robust noise level, so leads contribute
   in proportion to their signal-to-noise ratio.
2. Feature: the sum over leads of the squared slope, integrated over one QRS width.
3. Adaptive signal and noise levels with threshold = noise + 0.25 × (signal − noise); refractory period
   (species value); a peak within 1.6 refractory periods of a beat whose slope is less than half that of the
   beat is treated as a T wave; when an interval exceeds 1.66 times the running mean RR, the largest
   peak above half the threshold is accepted (search-back).
4. The fiducial point is the extreme of the lead with the highest signal-to-noise ratio, with automatic
   polarity.
5. Each beat's normalized cross-correlation with the median beat is reported; beats below 0.5 are marked
   unreliable. A beat whose correlation cannot be computed (at the very edge of the signal) is not
   marked unreliable.

The viewer contains a JavaScript implementation of the same detector.

## 4. Interval series and exclusion rules (`teleep.rr`)

**One recording from several files.** All files given to one analysis must come from one subject (the
first word of the EDF patient field); otherwise the analysis stops with a message. Files written by
TeleEP state their UTC offset, so they are ordered and placed on one time axis in UTC; a change to or
from daylight saving time therefore neither reorders files nor opens a false gap, and local clock
times in the results are taken from the file that contains them. Files without a stated offset are
placed by their local start times. A space of more than 0.1 s between the end of one file and the
start of the next is excluded as a gap between files. Long files are processed in one-hour chunks
with 10 s of overlap; beats and P waves found twice in an overlap are counted once. If the detector
fails on a chunk, that hour is marked "detector error" in `signal_quality.csv`, the error is shown in
the log, and the analysis continues.

RR intervals are formed from consecutive QRS complexes and PP intervals from consecutive P waves. Every
interval is reported; it is valid (an NN interval) only if none of the following flags applies, checked
in this order:

| Flag | Rule |
|---|---|
| `gap` | the interval overlaps a dropout, an unanalyzable stretch or a gap between files |
| `beat` | one of its two beats is marked unreliable |
| `range` | outside the species range (pig RR 240 to 3000 ms; PP 150 to 3000 ms) |
| `artifact` | differs by more than 20 % from the median of up to five valid neighbours on each side; neighbours on the far side of a `gap` interval are not used, because the rate may have changed during the gap |

Pauses are intervals longer than the pause threshold (pig, dog, human 2 s; rat and mouse 2 × median RR).
Intervals flagged `gap` are never reported as pauses.

## 5. Heart rate variability (`teleep.hrv`)

Definitions follow the Task Force of the European Society of Cardiology and the North American Society of
Pacing and Electrophysiology (1996) and Shaffer and Ginsberg (2017).

**Time domain.** Mean, median, minimum and maximum NN; SDNN (sample standard deviation); CV = SDNN / mean NN;
mean, SD, minimum and maximum instantaneous rate (60000 / NN). RMSSD, SDSD and pNNx use only pairs of
consecutive valid intervals, so an excluded interval never produces a false difference. HRV triangular index =
number of NN intervals / height of the histogram with 1/128-s bins. Rate statistics over all intervals that
are not across a gap and not at an unreliable beat (`mean_interval_ms`, `rate_from_median_bpm`) are also
reported, because in conduction disease they describe the rhythm better than NN statistics.

**Frequency domain.**

* *Welch (default).* The NN series is resampled at 4 Hz (pig, human, dog; 10 Hz rat; 20 Hz mouse) by natural
  cubic spline interpolation, linearly detrended, and analyzed with Welch's method (Hann window of 64 s for
  large animals, 25.6 s rat, 12.8 s mouse; 50 % overlap; power spectral density in ms²/Hz). Excluded intervals
  are bridged by the spline.
* *Lomb-Scargle.* The periodogram of the unevenly sampled, linearly detrended NN series, scaled so that the
  integral of the one-sided spectrum equals the variance.

Band powers are trapezoidal integrals of the spectrum from the lower to the upper band edge, with the
spectrum interpolated linearly at the edges, so adjacent bands add up exactly to their combined range
whatever the frequency resolution. A band is reported only if the window is long enough to resolve its
lower edge (VLF therefore needs more than about 5 minutes for pigs).
LF/HF, LF and HF in normalized units (percent of LF + HF) and the peak frequency of each band are reported.

**Nonlinear.** Poincaré SD1 and SD2 from consecutive valid pairs; sample entropy with m = 2 and
r = 0.2 × SD (Richman and Moorman 2000); detrended fluctuation analysis with short-term exponent α1 over
4 to 16 beats and long-term α2 over 16 to 64 beats (Peng et al. 1995).

**Epochs.** By default 5-minute epochs aligned to the wall clock (10:00, 10:05 ...). An epoch is analyzed
when analyzable intervals cover at least 80 % of it and at least 70 % of its intervals are valid. Partial
epochs that cannot reach the coverage are omitted.

**Whole recording.** Time-domain and Poincaré measures over all NN intervals; SDANN (SD of epoch mean NN)
and SDNN index (mean of epoch SDNN); spectral and nonlinear measures as the median over analyzed epochs,
because a single spectrum of a long, non-stationary recording is not meaningful.

**Atrioventricular statistics** (pigecg only). For each QRS, the time from the closest preceding P wave within
50 to 450 ms (`pr_ms`); the fraction of QRS complexes preceded by a P wave at 60 to 300 ms; median and
interquartile range of PR; the fraction of consecutive beats whose PR changes by more than 40 ms; and the
ratio of P waves to QRS complexes. These are descriptive and do not by themselves establish conduction.

## References

* Task Force of the European Society of Cardiology and the North American Society of Pacing and
  Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation and
  clinical use. *Circulation* 1996;93:1043-1065.
* Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. *Front Public Health*
  2017;5:258.
* Behar JA, Rosenberg AA, Shemla O, Murphy KR, Koren G, Billman GE, Yaniv Y. A universal scaling relation for
  defining power spectral bands in mammalian heart rate variability analysis. *Front Physiol* 2018;9:1001.
* Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Trans Biomed Eng* 1985;32:230-236.
* Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy.
  *Am J Physiol Heart Circ Physiol* 2000;278:H2039-H2049.
* Peng CK, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling exponents and crossover phenomena in
  nonstationary heartbeat time series. *Chaos* 1995;5:82-87.
* Kemp B, Olivan J. European data format 'plus' (EDF+), an EDF alike standard format for the exchange of
  physiological data. *Clin Neurophysiol* 2003;114:1755-1761.
