# Workplace Suicide Ideation — Reproducible Pipeline (2015–2023)

This repository is the canonical, one-shot reproducible codebase for the workplace suicidal-ideation paper.
It downloads NSDUH public-use data, runs the full 9-year analysis (2015–2023), writes all paper artifacts, and verifies headline metrics.

## One-shot run

```bash
# from repo root
chmod +x bootstrap.sh code/run_si.sh
./bootstrap.sh
```

`bootstrap.sh` will:
1. Create a Python 3.12 virtual environment (`.venv`) via `uv`.
2. Install pinned dependencies from `requirements.lock` (fallback to `code/requirements.txt`).
3. Run the full pipeline for years 2015–2023.
4. Run `verify` checks.

## Manual run

```bash
# full 9-year run
./code/run_si.sh 2015 2016 2017 2018 2019 2020 2021 2022 2023

# quick artifact/metric verification only (no retraining)
./code/run_si.sh verify
```

If no years are passed, `run_si.sh` defaults to `2015–2023`.

## Expected headline metrics

Verification should reproduce values within tolerance:
- Longitudinal analytic sample (sum of 9 diagonal test-year Ns): 176,957
- 2020 full model AUC: ~0.872
- Same-year AUC mean: ~0.750
- Cross-year AUC mean: ~0.688
- Same-vs-cross gap: ~0.062

Additional derived/rolling metrics are in `outputs/merged/derived_metrics.json` and `outputs/merged/rolling_window_results.json`.

## Key outputs

- `data/temporal_results.json` — train/test matrix + core metrics
- `data/checksums.txt` — SHA-256 checksums of downloaded inputs
- `outputs/metadata.json` — runtime provenance (python/packages/platform/git)
- `outputs/roc_overlay_2020.png` — 2020 ROC figure
- `outputs/shap_values.png` — SHAP figure
- `outputs/merged/` — manuscript-ready figures/tables/derived metrics

## Reproducibility notes

- Determinism knobs are set in `code/run_si.sh` (`OMP_NUM_THREADS=1`, fixed seed).
- Environment is pinned in `requirements.lock`.
- `verify` mode checks file presence and core metric tolerance without retraining.

## Repository scope

Use this repository as the single source of truth for replication. Legacy folders in the parent manuscript workspace are archival and not canonical for code execution.

## Links

- GitHub: https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline
- OSF project: https://osf.io/mj2kr/

## License

MIT (see `LICENSE`).
