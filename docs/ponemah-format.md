# Ponemah 6.x file structure

This is the structure of DSI Ponemah 6.x telemetry files as used by `teleep.io.ponemah`. It was
established by inspecting recordings and checking the decoded values against Ponemah's own exports. It
is not an official specification; tags that are not listed here are skipped.

## Experiment folder

A Ponemah experiment folder contains, for an experiment named `08062023 P10` with subject `P10`:

| File | Content |
|---|---|
| `08062023 P10.PnmExp` | SQLite database: experiment configuration |
| `08062023 P10.P10.20230806195959.PnmWav` | binary waveform file, one per subject recording (time stamp in UTC) |
| `08062023 P10.1.P10.PnmMarks` | SQLite database: Ponemah's beat marks and bad-data marks for the subject |
| `*.PnmResults`, `*.PnmExpLog` | derived parameters and logs (not used) |

## .PnmWav

The file starts with the 4-byte signature `AF 55 55 AA`, followed by records of the form

```
uint16 tag | uint32 length | <length> bytes of value
```

all little-endian. The "current channel" is set by tag 5045 and applies to the records that follow.

| Tag | Type | Meaning |
|---|---|---|
| 5041 | UTF-8 | subject name |
| 5042 | UTF-8 | Windows time-zone name of the acquisition computer, e.g. `Eastern Standard Time` |
| 5045 | uint64 | channel id (sets the current channel) |
| 5012 | uint64 | sample period in 100-ns ticks (channel header): 20000 = 500 Hz, 1000000 = 10 Hz, 10000000 = 1 Hz |
| 5025, 5026 | int16 | digital minimum and maximum (−32256, 32255) |
| 5029, 5030 | float32 | physical minimum and maximum of the channel (for example 20 and 45 for temperature in °C) |
| 5013, 5014 | int64 | start and end time of the following data block, in .NET ticks (100 ns since 0001-01-01, UTC) |
| 5015 | int16[] | data block: samples of the current channel |

The file has a header section with one group of channel-header records per channel, followed by data
blocks of all channels interleaved in time. Within a channel, consecutive blocks join without a gap during
continuous acquisition; a jump in the block start time marks an acquisition gap.

Sample value −32768 marks a sample that the telemetry receiver did not get (dropout). In the development
recording, real signal never reached the digital limits.

### Calibration

* ECG (unit mV): millivolts = counts × 10 / 32256. This factor reproduces Ponemah's exported ECG values to
  rounding precision. The physical range in the channel header of ECG channels (±14) is not used.
* Other channels: physical = pmin + (counts − dmin) × (pmax − pmin) / (dmax − dmin), with the values of
  tags 5025, 5026, 5029 and 5030. On the development recording this gives a body temperature of 38.0 °C,
  a battery voltage of 3.62 V and a battery life of 80 days, which are plausible, but these channels were
  not compared with a Ponemah export.

## .PnmExp

SQLite database. Table `ConfigInfo(ConfigSection, Metadata, Payload)`. Rows with
`ConfigSection = 'CHANNEL'` describe the channels; `Metadata` is the channel configuration number and
`Payload` is text with CR LF line breaks, in which line 4 is the label and line 5 the unit, for example

```
7 50 4 / 1 / 1 1 / ECG / mV / 0 / 1 / -32256 32255 2743 0 0 0 0 0 / 5334
```

Channel id *k* in the `.PnmWav` file corresponds to configuration number *k* − 1. Labels are truncated to
8 characters (`Temperat`, `Battery `); TeleEP restores common labels and makes duplicate labels unique the
way Ponemah does in its exports (`ECG`, `ECG1`, `ECG2`), or by unit when units differ (`Battery V`,
`Battery days`).

## .PnmMarks

SQLite database.

| Table | Use |
|---|---|
| `ChannelContextInfo(ChannelContextId, SubjectChannelId, TicksPerSample, ...)` | maps a mark context to a channel id and its sample period |
| `Cycle(ChannelContextId, CycleTime, Marks)` | one row per detected beat; `CycleTime` in .NET ticks (UTC); `Marks` holds seven int32 values that appear to be sample offsets of fiducial points (0x80000000 = not marked); TeleEP does not use them |
| `BadDataMark(SubjectChannelId, StartTime, EndTime)` | stretches marked as bad data in Ponemah |

TeleEP reads `Cycle` times for comparison with its own detections (`teleep analyze --compare-ponemah`) and
for export (`teleep convert --ponemah-marks`).
