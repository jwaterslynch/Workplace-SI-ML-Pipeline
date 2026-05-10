# Website Card Draft

Copy for the Research Tools page on `julianwaterslynch.com`.

## Card Label

Research Tool - Open source

## Product Name

Suicide Prediction Temporal Validation Toolkit

## Short Line

A reproducible validation workbench for suicide-risk prediction drift and model
maintenance.

## Product Description

An open-source Python pipeline for evaluating temporal transportability in
suicidal-ideation prediction. The toolkit downloads NSDUH public-use data,
reconstructs the 176,957-respondent employed-adult analytic sample, trains and
tests models across all 2015-2023 train-year and test-year combinations, and generates the
manuscript figures, tables, calibration outputs, threshold audits, and
provenance metadata.

Built from the machine-learning suicide-prediction paper. The tool is meant for
replication, methods development, and model-governance research, with an
independent reproduction-audit protocol included. It is not a clinical
diagnostic system, individual risk scorer, or employer screening product.

## Status

Open-source research release.

## Tags

Open source - Python - Temporal validation - Model drift - Reproducible
pipeline - Mental-health methods - MIT license

## Links

- GitHub: `https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline`
- OSF: `https://osf.io/mj2kr/`
- Related paper anchor: `index.html#paper-suicidal-ideation`

## Suggested HTML

```html
<article class="tool-card" id="suicide-prediction-temporal-validation" style="--product-accent:#3f5e66">
  <div class="tool-heading">
    <span class="tool-mark" aria-hidden="true">Sv</span>
    <div>
      <h2>Suicide Prediction Temporal Validation Toolkit</h2>
      <p class="tagline">A reproducible validation workbench for suicide-risk prediction drift and model maintenance.</p>
    </div>
  </div>
  <p>
    An open-source Python pipeline for evaluating temporal transportability in
    suicidal-ideation prediction. The toolkit reconstructs the 176,957-respondent
    employed-adult NSDUH analytic sample, tests models across all 2015-2023
    train-year and test-year combinations, and generates paper-ready validation
    artifacts.
  </p>
  <p>
    Built from the machine-learning suicide-prediction paper. It is a
    replication and governance research tool with an independent audit protocol,
    not a clinical diagnostic system, individual risk scorer, or employer
    screening product.
  </p>
  <div class="meta-row">
    <span class="chip live">Open source</span>
    <span class="chip">Python</span>
    <span class="chip">Temporal validation</span>
    <span class="chip">176,957 respondents</span>
    <span class="chip">Model drift</span>
    <span class="chip">MIT license</span>
  </div>
  <div class="actions">
    <a class="btn btn-primary" href="https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline" target="_blank" rel="noreferrer">Open GitHub repo -></a>
    <a class="btn btn-secondary" href="index.html#paper-suicidal-ideation">Related paper</a>
    <a class="btn-link" href="https://osf.io/mj2kr/" target="_blank" rel="me noreferrer">OSF project -></a>
  </div>
</article>
```
