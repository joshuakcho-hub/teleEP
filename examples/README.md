# Examples

* `teleep demo <folder>` creates a synthetic Ponemah experiment (20 minutes of a pig-like recording with
  complete atrioventricular block and one telemetry dropout), converts it to EDF and analyzes it. Open one
  of the EDF files in the viewer (`teleep view`) and import the matching file from
  `results/viewer_marks` to explore the interface.
* `scripting.py` shows the Python interface: conversion, analysis with custom settings, and access to the
  results.

```bash
teleep demo teleep_demo
python examples/scripting.py teleep_demo/ponemah_experiment scripted_results
```
