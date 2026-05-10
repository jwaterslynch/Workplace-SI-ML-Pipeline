# Suicide Prediction Temporal Validation Toolkit

Reproducible machine-learning pipeline for evaluating suicidal-ideation
prediction across nine years of NSDUH public-use survey data.

This is the public research-tool release behind the paper:

> Machine-Learning Prediction of Suicidal Ideation in Employed U.S. Adults:
> Temporal Validation and Model Maintenance Across Nine Years of Survey Data

The current GitHub URL is retained for continuity from the original replication
project. The public-facing tool identity is broader: this is a
temporal-validation and model-maintenance workbench for high-stakes
mental-health prediction research.

## What This Tool Does

The pipeline:

- Downloads NSDUH public-use data for 2015-2023.
- Constructs the employed-adult analytic sample used in the paper.
- Trains and evaluates suicidal-ideation prediction models across all
  train-year and test-year combinations.
- Produces a 9-by-9 temporal validation matrix.
- Compares same-year, cross-year, and rolling-window training strategies.
- Writes manuscript-ready tables, figures, predictions, calibration outputs,
  and provenance metadata.
- Verifies headline metrics against locked tolerances.

The core contribution is not a deployable individual risk score. It is a
transparent way to inspect temporal transportability, model drift, and
maintenance rules in a high-stakes prediction setting.

## What This Tool Is Not

This repository is not:

- A clinical diagnostic system.
- A suicide-risk assessment service.
- An individual-level screening product.
- An employer, HR, or workplace surveillance tool.
- A substitute for clinician assessment, crisis support, consent, local
  validation, legal review, or ethics review.

The outputs are intended for research, replication, auditing, teaching, and
methods development. Any applied use of similar models would require a separate
governance pathway and independent validation in the intended setting.

## Quick Start

From a local checkout:

```bash
chmod +x bootstrap.sh code/run_si.sh
./bootstrap.sh
```

`bootstrap.sh` will:

1. Create a Python 3.12 virtual environment with `uv`.
2. Install pinned dependencies from `requirements.lock`, with fallback to
   `code/requirements.txt`.
3. Run the full 2015-2023 pipeline.
4. Run verification checks.

For a fast artifact and metric check without retraining:

```bash
./code/run_si.sh verify
```

To rerun selected years:

```bash
./code/run_si.sh 2019 2020 2021 2022 2023
```

If no years are supplied, `run_si.sh` defaults to the full 2015-2023 window.

## Expected Headline Metrics

`verify` checks that the paper-critical artifacts exist and that headline
metrics remain within tolerance:

| Quantity | Expected value |
|---|---:|
| Longitudinal analytic sample | 176,957 |
| 2020 full model AUC | approximately 0.872 |
| Same-year AUC mean | approximately 0.750 |
| Cross-year AUC mean | approximately 0.688 |
| Same-year vs cross-year gap | approximately 0.062 |

Derived and rolling-window metrics are written to:

- `outputs/merged/derived_metrics.json`
- `outputs/merged/rolling_window_results.json`

## Key Outputs

| Path | Purpose |
|---|---|
| `data/temporal_results.json` | Train-year by test-year validation matrix and core metrics |
| `data/checksums.txt` | SHA-256 checksums for downloaded input data |
| `outputs/metadata.json` | Runtime provenance: Python, packages, platform, git state |
| `outputs/merged/Table_3_AUC_Matrix.csv` | Manuscript-ready temporal AUC matrix |
| `outputs/merged/Figure_1_AUC_Heatmap.png` | Cross-year AUC heatmap |
| `outputs/merged/Table_4_Training_Strategy.csv` | Single-year vs rolling-window comparison |
| `outputs/fairness_table3.csv` | Group-level threshold and performance audit output |
| `outputs/appendix_stats/elastic_net_coefs.csv` | Penalized-logistic comparator coefficients |
| `outputs/appendix_stats/nri_idi.csv` | Incremental performance statistics |

See `outputs/merged/README.md` for the publication figure and table guide.

## Reproducibility

- Dependency state is pinned in `requirements.lock`.
- The runner sets deterministic single-threading knobs for BLAS/OMP libraries.
- A fixed random seed is used for model runs.
- `verify` checks artifact presence and headline metric tolerances without
  retraining.
- Raw NSDUH public-use files are downloaded locally and excluded from git.

The repository is designed so a reviewer can reproduce the paper-critical
analysis from a clean checkout, subject to network access for public-use data
downloads.

## Research Use And Citation

If you use this repository in academic work, cite both the software release and
the associated paper or working paper.

```bibtex
@software{waterslynch_suicide_prediction_temporal_validation_2026,
  title = {Suicide Prediction Temporal Validation Toolkit},
  author = {Waters-Lynch, Julian},
  year = {2026},
  url = {https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline},
  version = {1.0.5-paper}
}
```

The repository also includes `CITATION.cff`, which GitHub can use to generate a
software citation.

## Website Copy

Website-ready card copy for `julianwaterslynch.com` is in:

- `docs/WEBSITE_CARD.md`

## Links

- GitHub: https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline
- OSF project: https://osf.io/mj2kr/
- Author profile: https://julianwaterslynch.com/tools.html

## License

MIT. See `LICENSE`.
