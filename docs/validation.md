# Validation

## Development recording

* Subject P10, a pig with complete atrioventricular block (Cho laboratory, Johns Hopkins University).
* DSI implantable telemetry, three subcutaneous ECG leads at 500 Hz, temperature, battery and activity
  channels; Ponemah 6.x; 18.64 hours of continuous recording (2023-08-06 15:59:59 to 2023-08-07 10:38:28,
  local time).
* Atrial rate 130 to 160 per minute; ventricular escape rate 60 to 71 per minute with junctional and
  ventricular escape morphologies; 2 to 45 seconds of telemetry dropout in 5 of the 19 hours.

The recording is not distributed with TeleEP. The automated tests use synthetic recordings
(`teleep.synthetic`).

## Conversion

| Check | Result |
|---|---|
| ECG samples compared with Ponemah's exported values (earlier development of Ponemah2EDF) | agreement to 4.5 × 10⁻⁷ mV, i.e. rounding of the exported values |
| TeleEP 1.0 EDF compared with the earlier Ponemah2EDF EDF, all non-dropout samples of an hourly file | ECG identical (maximum difference 0) |
| Dropout samples | 19,314 to 22,500 per lead in the worst hour, all annotated and read back as missing |
| Start time | 15:59:59 local (EDT, recorded as UTC-04:00), matching the time stamps of Ponemah's own Excel export; the earlier converter wrote 19:59:59 (UTC) without saying so |
| Reading by third-party software | files open in pyedflib with correct signals, rates and annotations (automated test) |
| Speed | 18.6 hours of 7 channels converted in 3.6 s |

## QRS detection compared with Ponemah

`teleep analyze conv --compare-ponemah "<experiment>"` on all 19 hourly files, counting only signal that
TeleEP classifies as analyzable, with a matching tolerance of ±120 ms. Ponemah's beat marks were made
separately on each lead.

| Ponemah lead | Matched | Ponemah only | TeleEP only |
|---|---|---|---|
| ECG2 (Ponemah's best lead) | 67,806 | 176 (0.26 % of Ponemah marks) | 1,409 (2.0 % of TeleEP beats) |
| ECG1 | 60,497 | 177 | 8,718 |
| ECG | 3,179 | 54 | 36,875 (Ponemah detected few beats on this lead) |

These figures are within three beats of those obtained with the original pigecg detector before it was
integrated into TeleEP (67,809, 176 and 1,408), so the integration did not change detection.

During the original development of pigecg, the disagreements on the best lead were reviewed visually. The
TeleEP-only beats were mostly small, narrow junctional escape beats that Ponemah skipped. Of the
Ponemah-only marks, about 80 % were Ponemah detections on P waves and about 20 % were small junctional beats
that pigecg also missed (about 35 beats, 0.05 % of all beats). In 54 random 8-second strips (about 480 QRS
complexes and 950 P waves in analyzable signal), one QRS complex was missed, none was falsely detected, and
one was ambiguous; no false P waves were found and about 1 to 2 % of P waves were missed, almost all within
baseline-wander artifacts or hidden in the QRS-T complex. This review was not blinded and was not performed
by an independent expert; the EP Annotator (below) is provided for that purpose.

## Analysis of the development recording

| Measure | Ventricular (RR) | Atrial (PP) |
|---|---|---|
| Beats detected | 75,769 QRS | 139,658 P |
| Intervals valid for HRV | 57,589 of 75,768 (76.0 %) | 99,692 of 139,657 (71.4 %) |
| Mean rate (NN) | 64.7 per minute | 147.1 per minute |
| SDNN / RMSSD | 65.9 / 16.0 ms | 39.1 / 16.3 ms |
| 5-minute epochs analyzed | 155 of 223 | 131 of 223 |
| Pauses > 2 s | 2 | 15 |

Excluded RR intervals were mainly in unanalyzable signal (8,206) and in irregular stretches with alternating
RR (9,120 flagged as artifact). Most excluded PP intervals contained a P wave hidden in a QRS complex
(15,391 artifact) or fell in signal too noisy for P waves (24,573). Processing took 13 to 20 s per hour
of three-lead data on one processor core.

## Synthetic recordings (automated tests)

| Test | Result |
|---|---|
| generic detector, sinus rhythm and complete block | three leads: sensitivity and positive predictivity ≥ 99.5 %, R peak located within 6 ms; single lead: ≥ 99 % |
| pigecg, complete block | QRS sensitivity and positive predictivity ≥ 99.5 %; P waves: positive predictivity ≥ 98 %, sensitivity ≥ 85 % (the remainder coincide with QRS complexes) |
| End-to-end pipeline (synthetic Ponemah experiment to EDF to analysis) | > 99.5 % of true QRS complexes found; mean ventricular and atrial rates within 1 % and 2 %; > 99 % agreement with the synthetic "Ponemah" marks |
| Time line | 40 minutes recorded across the end of daylight saving time: files placed without gap or overlap, local times correct on both sides of the change, > 99 % agreement with the "Ponemah" marks; start times with a fraction of a second kept exactly |
| Conversion layout | two subjects in one experiment written to separate folders, and refused when analyzed together; a short final piece joined to the previous file; activity counts of zero kept as values rather than dropout |
| Robustness | a detector failure in one chunk is recorded as "detector error" and the analysis continues |
| HRV | time-domain definitions; band power of a known 0.3-Hz modulation within 15 % (Welch and Lomb-Scargle); adjacent band powers add up exactly; DFA of white noise 0.5 ± 0.1; the artifact rule does not compare intervals across a gap |
| Viewer | HRV computed in the browser agrees with the Python package within 0.01 % for every analyzed epoch |

Run the tests with `pytest` (see CONTRIBUTING.md).

## Validating on your own recordings

1. **Against Ponemah.** `teleep analyze <EDF folder> --compare-ponemah <experiment folder>` writes
   `ponemah_agreement.csv` (per hour and lead) and `ponemah_disagreements.csv` (the time of every
   disagreement), which can be reviewed in the viewer.
2. **Against expert review.** Start the EP Annotator (`teleep annotate <EDF folder>`), review stretches of
   8 seconds spread across the recording (press R for each stretch that is fully correct, including those
   that needed no change), and save. `teleep train <EDF folder>` then reports the detector's sensitivity and
   positive predictivity for QRS complexes and P waves on the reviewed stretches, lists every remaining
   error, and, with two or more annotated files, estimates the benefit of the learned correction by
   leave-one-file-out cross-validation.
