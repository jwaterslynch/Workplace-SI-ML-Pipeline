# Safety And Use Boundaries

This repository studies predictive model performance in a high-stakes
mental-health domain. It should be read as a temporal-validation and
model-maintenance workbench, not as a deployable suicide-risk tool.

## Appropriate Uses

- Replicating the manuscript analysis.
- Auditing temporal transportability across survey years.
- Studying model drift, maintenance windows, and validation design.
- Teaching reproducible high-stakes prediction workflows.
- Developing safer methods for evaluating mental-health prediction models.

## Inappropriate Uses

- Scoring identifiable individuals.
- Screening employees, applicants, students, patients, customers, or platform
  users.
- Making clinical, employment, insurance, or legal decisions.
- Replacing clinician assessment or crisis response.
- Presenting outputs as diagnosis, prognosis, or treatment guidance.

## Why This Boundary Matters

The pipeline uses public-use survey data and retrospective validation. It does
not establish clinical utility, prospective safety, informed-consent procedures,
implementation readiness, or fairness under real-world decision constraints.
Even strong retrospective discrimination would not be enough to justify
deployment.

## Minimum Governance For Any Future Applied Work

Any future applied system inspired by this research would need, at minimum:

- Prospective validation in the intended setting.
- Human-subjects ethics review.
- Explicit consent and privacy controls.
- Clinician-mediated interpretation.
- False-positive and false-negative harm analysis.
- Local calibration, monitoring, and drift review.
- Legal review for the deployment context.
- Clear non-punitive support pathways.

Those requirements are outside the scope of this repository.
