# Changelog

All notable public research-tool changes will be documented in this file.

## 1.0.5-paper - 2026-05-10

- Reframed the repository as the Suicide Prediction Temporal Validation Toolkit.
- Added public README language aligned with the revised manuscript framing.
- Added `CITATION.cff` for GitHub citation support.
- Added explicit safety and use boundaries.
- Added website-ready card copy for `julianwaterslynch.com`.
- Preserved the existing one-shot replication and `verify` workflow.
- Treated SHAP output as an optional verifier artifact unless `--shap` is run.

## Pre-release

- Built the 2015-2023 NSDUH pipeline.
- Added temporal train-year by test-year validation outputs.
- Added rolling-window, calibration, threshold, and supplemental comparator
  artifacts.
- Added verification checks for paper-critical metrics.
