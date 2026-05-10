# Contributing

This repository supports a high-stakes mental-health prediction paper, so
contributions should preserve reproducibility, traceability, and the stated use
boundaries.

## Good Contributions

- Reproducibility improvements that make clean reruns easier.
- Documentation that clarifies data sources, outputs, or validation logic.
- Robustness checks that are clearly separated from locked paper outputs.
- Tests or verification checks that catch metric drift or missing artifacts.
- Small refactors that preserve the public runner interface.

## Out Of Scope

- Individual risk-scoring interfaces.
- Clinical triage recommendations.
- Employer, HR, or workplace monitoring workflows.
- Interfaces that invite use as a diagnostic or screening product.
- Changes that silently alter paper-critical outputs without documenting why.

## Before Opening A Pull Request

Run:

```bash
./code/run_si.sh verify
```

For changes that touch model logic, features, data construction, or output
generation, run the full pipeline:

```bash
./bootstrap.sh
```

Then document:

- The exact command run.
- Whether verification passed.
- Any expected changes to headline metrics.
- Any new files added under `data/` or `outputs/`.

## Data Handling

Raw NSDUH public-use data are downloaded locally and should not be committed.
Commit only code, documentation, aggregate artifacts, and paper-support outputs
that are intended for public release.

## Release Discipline

If a change affects public use, update:

- `README.md`
- `CITATION.cff`
- `CHANGELOG.md`
- `docs/WEBSITE_CARD.md`, if website copy changes
