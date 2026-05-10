# Suicide Prediction Temporal Validation Toolkit

This OSF component accompanies the GitHub repository and manuscript for:

> Machine-Learning Prediction of Suicidal Ideation in Employed U.S. Adults:
> Temporal Validation and Model Maintenance Across Nine Years of Survey Data

The project provides a reproducible pipeline for evaluating temporal
transportability and model-maintenance strategies in suicidal-ideation
prediction using 2015-2023 NSDUH public-use survey data.

## Repository

GitHub: https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline

## Purpose

The repository is a research replication and validation workbench. It downloads
public-use NSDUH data, recreates the employed-adult analytic sample, trains and
tests models across all train-year and test-year combinations, and writes the
figures, tables, predictions, calibration outputs, and provenance metadata used
for the paper.

## Scope Boundary

This is not a clinical risk assessment product, diagnostic system, individual
screening tool, employer surveillance tool, or HR analytics product. It is a
research tool for inspecting temporal validation, model drift, and maintenance
choices in a high-stakes prediction context.

## Reproduction Commands

From the repository root:

```bash
chmod +x bootstrap.sh code/run_si.sh
./bootstrap.sh
```

For verification without retraining:

```bash
./code/run_si.sh verify
```

Expected verification targets include:

- Longitudinal analytic sample: 176,957
- 2020 full model AUC: approximately 0.872
- Same-year AUC mean: approximately 0.750
- Cross-year AUC mean: approximately 0.688
- Same-year vs cross-year gap: approximately 0.062

## Key Artifacts

- `data/temporal_results.json`: temporal validation matrix and core metrics
- `data/checksums.txt`: input-data checksums
- `outputs/metadata.json`: runtime provenance
- `outputs/merged/`: manuscript-ready figures and tables
- `outputs/appendix_stats/`: supplemental comparator and robustness outputs

## Citation

Use the `CITATION.cff` file in the GitHub repository to cite the software
release. If using the results or research design, cite the associated paper or
working paper as well.
