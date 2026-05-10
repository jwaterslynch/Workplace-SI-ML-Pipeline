# Claude Code Reproduction Audit Protocol

This document is a handoff brief for an independent Claude Code audit of the
suicide prediction paper pipeline and associated open-source reference model.

The goal is not to trust prior reports. The goal is to clone fresh repositories,
run the analyses independently, compare outputs against locked expectations,
and write a clear audit report that states what reproduced, what did not, and
what remaining caveats matter.

## Executive Instruction To Claude Code

You are auditing two linked repositories:

1. `https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline`
2. `https://github.com/jwaterslynch/suicidal-ideation-reference-model`

Work in a fresh directory on the external hard drive if available. Do not place
large NSDUH data files on the internal laptop drive. Do not commit or push
anything. Record exact commands, commits, package versions, file hashes, and
metric comparisons.

At the end, produce a Markdown audit report with:

- Reproduction status: pass, partial pass, or fail.
- Exact commit hashes used.
- Exact commands run.
- Whether the 2015-2023 paper pipeline reproduced.
- Whether the reference-model artifact rebuilt exactly.
- Whether the 2024 fresh-data validation reproduced.
- Any warnings, caveats, or unresolved risks.

## Expected Public Commits

At the time this protocol was written:

- Paper pipeline `main`: use the latest public `main` and record the actual
  commit. It should include `a6f8850` (`Make SHAP optional in verifier`) and
  `d9a9a17` (`Document temporal validation toolkit release`). This protocol was
  initially added in `a7df0ef`.
- Reference model `v0.1.2`: `34f80df` (`Add NSDUH 2024 validation release`)
- Reference-model artifact source commit: `d2554b6` (`v1.0.4-paper`)

If the remote has moved, record the new commit and say so explicitly. Do not
silently mix results from different commits.

## External-Drive Workspace

Use an external-drive run root. Example:

```bash
export RUN_ROOT="/Volumes/Jules Hardrive/Workspace_Offload/Research/claude_code_si_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"
cd "$RUN_ROOT"
```

If that external path does not exist, stop and ask where large files should go.

## Step 1: Fresh Clones

```bash
cd "$RUN_ROOT"
git clone https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline.git
git clone https://github.com/jwaterslynch/suicidal-ideation-reference-model.git

git -C Workplace-SI-ML-Pipeline log --oneline --decorate -3
git -C suicidal-ideation-reference-model log --oneline --decorate -3
git -C suicidal-ideation-reference-model checkout v0.1.2
```

Record:

```bash
git -C Workplace-SI-ML-Pipeline rev-parse HEAD
git -C suicidal-ideation-reference-model rev-parse HEAD
```

## Step 2: Verify Checked-In Paper Artifacts

This checks whether the paper pipeline repository is internally coherent before
any rerun.

```bash
cd "$RUN_ROOT/Workplace-SI-ML-Pipeline"
./code/run_si.sh verify 2>&1 | tee "$RUN_ROOT/logs/paper_verify_preexisting.log"
```

Expected result:

- Verification passes.
- If `outputs/shap_values.png` is absent, that should be reported only as an
  optional artifact note, not a failure.

## Step 3: Full 2015-2023 Paper Reproduction

Preferred independent mode: let the fresh clone download data into the external
drive. This may use several GB and can take time.

```bash
cd "$RUN_ROOT/Workplace-SI-ML-Pipeline"
/opt/homebrew/bin/python3.12 -m venv .venv || python3.12 -m venv .venv

./code/run_si.sh 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
  2>&1 | tee "$RUN_ROOT/logs/full_paper_reproduction.log"

./code/run_si.sh verify \
  2>&1 | tee "$RUN_ROOT/logs/paper_verify_after_full_run.log"
```

If a fresh download is impractical but an existing external NSDUH data folder is
available, use it only as a documented pragmatic mode:

```bash
cd "$RUN_ROOT/Workplace-SI-ML-Pipeline"
rm -rf data
ln -s "/Volumes/Jules Hardrive/Workspace_Offload/Research/suicidal_ideation_pipeline/data" data
./code/run_si.sh 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
  2>&1 | tee "$RUN_ROOT/logs/full_paper_reproduction_existing_data.log"
```

If using the symlink mode, state that the analysis was independently rerun but
did not independently redownload all 2015-2023 raw files.

### Expected 2015-2023 Values

Read `data/temporal_results.json` and verify:

- `2020_full.auc`: approximately `0.8721268238243411`
- `2020_full.n_total`: `3738`
- `2020_full.n_positive`: `206`
- Sum of diagonal `basic[*][*].n_total`: `176957`
- Same-year AUC mean: approximately `0.7498939813937959`
- Cross-year AUC mean: approximately `0.6876727308762675`
- Same-year minus cross-year gap: approximately `0.06222125051752847`
- Basic matrix years: `2015` through `2023`

Use this helper:

```bash
python - <<'PY'
import json
from pathlib import Path

j = json.load(open("data/temporal_results.json"))
years = [str(y) for y in range(2015, 2024)]
basic = j["basic"]
diag, off = [], []
diag_n_total = 0
for tr in years:
    for te in years:
        cell = basic[tr][te]
        if tr == te:
            diag.append(float(cell["auc"]))
            diag_n_total += int(cell["n_total"])
        else:
            off.append(float(cell["auc"]))

print("2020_full_auc", j["2020_full"]["auc"])
print("2020_full_n_total", j["2020_full"]["n_total"])
print("2020_full_n_positive", j["2020_full"]["n_positive"])
print("diag_n_total", diag_n_total)
print("same_year_auc_mean", sum(diag) / len(diag))
print("cross_year_auc_mean", sum(off) / len(off))
print("gap", sum(diag) / len(diag) - sum(off) / len(off))
PY
```

Acceptance:

- Values should match to at least 3 decimals for AUC summaries.
- Counts should match exactly.
- If rerunning from the same dependency family, exact JSON equality with the
  checked-in `data/temporal_results.json` is ideal but not required if all
  locked metrics and counts match.

## Step 4: Reference Model Install, Tests, And Wheel

```bash
cd "$RUN_ROOT/suicidal-ideation-reference-model"
uv run --extra dev pytest -q 2>&1 | tee "$RUN_ROOT/logs/reference_model_tests.log"
uv build 2>&1 | tee "$RUN_ROOT/logs/reference_model_build.log"
```

Then install the wheel in a fresh environment and score the example CSV:

```bash
cd "$RUN_ROOT/suicidal-ideation-reference-model"
/opt/homebrew/bin/python3.12 -m venv "$RUN_ROOT/wheel_check_venv" || python3.12 -m venv "$RUN_ROOT/wheel_check_venv"
"$RUN_ROOT/wheel_check_venv/bin/pip" install --upgrade pip
"$RUN_ROOT/wheel_check_venv/bin/pip" install dist/suicidal_ideation_reference_model-*.whl
"$RUN_ROOT/wheel_check_venv/bin/si-risk-score" examples/example_input.csv \
  --output "$RUN_ROOT/outputs/example_predictions_from_wheel.csv" \
  --threshold 0.17
cat "$RUN_ROOT/outputs/example_predictions_from_wheel.csv"
```

Expected example probabilities:

- `0.0137352268`
- `0.0762623748`
- `0.3254862831`
- `0.2287566224`

Small final-digit differences are acceptable; row flags at threshold `0.17`
should be `0, 0, 1, 1`.

## Step 5: Rebuild Reference Model Artifact From Paper Pipeline

This tests whether the packaged reference model can be regenerated from the
paper-pipeline repository.

The reference-model artifact records paper-pipeline source commit `d2554b6`.
Use a separate source clone checked out to that commit so metadata, metrics, and
predictions are all compared against the same source state.

```bash
cd "$RUN_ROOT"
git clone https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline.git \
  Workplace-SI-ML-Pipeline-artifact-source
git -C Workplace-SI-ML-Pipeline-artifact-source checkout d2554b65ccfb8097761336fa73654beece06c646
rm -rf Workplace-SI-ML-Pipeline-artifact-source/data
ln -s "$RUN_ROOT/Workplace-SI-ML-Pipeline/data" \
  "$RUN_ROOT/Workplace-SI-ML-Pipeline-artifact-source/data"
```

```bash
cd "$RUN_ROOT/suicidal-ideation-reference-model"
mkdir -p "$RUN_ROOT/outputs/artifact_rebuild"

uv run --extra dev --with requests --with urllib3 --with matplotlib \
  python scripts/build_reference_model.py \
  --source-repo "$RUN_ROOT/Workplace-SI-ML-Pipeline-artifact-source" \
  --output-dir "$RUN_ROOT/outputs/artifact_rebuild" \
  2>&1 | tee "$RUN_ROOT/logs/artifact_rebuild.log"
```

Compare rebuilt metadata and predictions against the packaged artifact:

```bash
cd "$RUN_ROOT/suicidal-ideation-reference-model"
python - <<'PY'
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from suicidal_ideation_reference_model.model import load_reference_model, predict_dataframe

run_root = Path(__import__("os").environ["RUN_ROOT"])
rebuilt_path = run_root / "outputs/artifact_rebuild/si_xgb_full_2020_v0_1_1.joblib"
packaged = load_reference_model()
rebuilt = joblib.load(rebuilt_path)

metrics = [
    ("test_auc", ["metrics", "test_auc"]),
    ("test_brier", ["metrics", "test_brier"]),
    ("f1_threshold_from_train", ["metrics", "f1_threshold_from_train"]),
    ("high_specificity_reference.threshold", ["metrics", "high_specificity_reference", "threshold"]),
    ("high_specificity_reference.sensitivity", ["metrics", "high_specificity_reference", "sensitivity"]),
    ("high_specificity_reference.specificity", ["metrics", "high_specificity_reference", "specificity"]),
    ("high_specificity_reference.ppv", ["metrics", "high_specificity_reference", "ppv"]),
    ("high_specificity_reference.npv", ["metrics", "high_specificity_reference", "npv"]),
    ("high_specificity_reference.tp", ["metrics", "high_specificity_reference", "tp"]),
    ("high_specificity_reference.fp", ["metrics", "high_specificity_reference", "fp"]),
    ("high_specificity_reference.fn", ["metrics", "high_specificity_reference", "fn"]),
    ("high_specificity_reference.tn", ["metrics", "high_specificity_reference", "tn"]),
]

def get(d, path):
    for p in path:
        d = d[p]
    return d

rows = []
all_match = True
for name, path in metrics:
    a = get(packaged["metadata"], path)
    b = get(rebuilt["metadata"], path)
    diff = abs(float(a) - float(b)) if isinstance(a, (int, float)) else None
    match = a == b or (diff is not None and diff <= 1e-12)
    rows.append({"metric": name, "packaged": a, "rebuilt": b, "abs_diff": diff, "match": match})
    all_match = all_match and match

example = pd.read_csv("examples/example_input.csv")
packaged_probs = predict_dataframe(example, bundle=packaged, threshold=0.17)["si_probability"].to_numpy()
rebuilt_probs = predict_dataframe(example, bundle=rebuilt, threshold=0.17)["si_probability"].to_numpy()
max_prob_diff = float(np.max(np.abs(packaged_probs - rebuilt_probs)))

report = {
    "packaged_model_id": packaged["metadata"].get("model_id"),
    "rebuilt_model_id": rebuilt["metadata"].get("model_id"),
    "packaged_source_commit": packaged["metadata"].get("source_commit"),
    "rebuilt_source_commit": rebuilt["metadata"].get("source_commit"),
    "metric_comparison": rows,
    "metric_all_match": all_match,
    "example_probability_max_abs_diff": max_prob_diff,
    "example_predictions_match": max_prob_diff <= 1e-12,
}
out = run_root / "outputs/artifact_rebuild/artifact_rebuild_comparison_report.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
```

Expected artifact rebuild:

- `test_auc`: `0.8721151414529021`
- `test_brier`: `0.04383519039812847`
- high-specificity threshold: `0.17000000000000004`
- high-specificity sensitivity: `0.529126213592233`
- high-specificity specificity: `0.9283691959229898`
- high-specificity counts: `TP=109`, `FP=253`, `FN=97`, `TN=3279`
- packaged and rebuilt example predictions should match with max absolute
  difference `<= 1e-12`.

## Step 6: NSDUH 2024 Fresh-Data Validation

Run the external validation workflow from the reference-model repository.

```bash
cd "$RUN_ROOT/suicidal-ideation-reference-model"
uv run --extra validation python validation/validate_nsduh_2024.py --download \
  2>&1 | tee "$RUN_ROOT/logs/reference_model_2024_validation.log"
```

Expected headline output:

- `n = 20,588`
- prevalence = `6.276%`
- AUC = `0.830`
- Brier = `0.0513`

Parse the JSON:

```bash
python - <<'PY'
import json
j = json.load(open("validation/results/nsduh_2024_validation_report.json"))
m = j["metrics"]
print("raw_rows", j["raw_rows"])
print("employed_rows", j["employed_rows"])
print("valid_outcome_rows", j["employed_rows_with_valid_outcome"])
print("n", m["n"])
print("positives", m["positives"])
print("prevalence", m["prevalence"])
print("auc", m["auc"])
print("auprc", m["auprc"])
print("brier", m["brier"])
print("mean_probability", m["mean_probability"])
print("calibration_intercept", m["calibration"]["intercept"])
print("calibration_slope", m["calibration"]["slope"])
print("threshold_0.17", m["threshold_0.17"])
print("weighted_prevalence", m["weighted_prevalence"])
print("weighted_flag_rate", m["weighted_flag_rate_at_default_threshold"])
print("source_columns", j["source_columns"])
PY
```

Expected detailed values:

- `raw_rows`: `58633`
- `employed_rows`: `20781`
- `employed_rows_with_valid_outcome`: `20588`
- `positives`: `1292`
- `prevalence`: `0.06275500291431903`
- `auc`: `0.8296844996508683`
- `auprc`: `0.30430184543249783`
- `brier`: `0.05125843170310194`
- `mean_probability`: approximately `0.0795067153651706`
- calibration intercept: approximately `-0.341655585590565`
- calibration slope: approximately `0.971616265914281`
- threshold `0.17`: `TP=931`, `FP=3320`, `FN=361`, `TN=15976`
- threshold `0.17`: sensitivity `0.7205882352941176`
- threshold `0.17`: specificity `0.8279436152570481`
- threshold `0.17`: PPV `0.21900729240178782`
- threshold `0.17`: NPV `0.9779029197527086`
- weighted prevalence: approximately `0.0499489521535034`
- weighted flag rate: approximately `0.15959849028633`
- `sexual_orientation`: `not_available_in_2024_public_use_file`

Acceptance:

- Counts should match exactly.
- Metrics should match to at least 6 significant figures.
- Tiny floating-point differences in the 14th-15th significant figure are not
  meaningful.

## Step 7: Data And Hash Checks

Record sizes and hashes of large files used or downloaded:

```bash
cd "$RUN_ROOT"
find . -type f \( -name "*.parquet" -o -name "*.tab" -o -name "*.zip" \) -print0 |
  xargs -0 shasum -a 256 > "$RUN_ROOT/outputs/raw_data_sha256.txt"

du -sh "$RUN_ROOT" "$RUN_ROOT/Workplace-SI-ML-Pipeline" "$RUN_ROOT/suicidal-ideation-reference-model"
df -h "$(dirname "$RUN_ROOT")"
```

For the 2024 validation raw parquet files, expected hashes from the v0.1.2
independent reproduction were:

- `nsduh_2024_data.parquet`:
  `95ad20cb919186c304c8b442aa060b279ad61dc29bf252bcb43dfe8274b56e86`
- `nsduh_2024_data_dictionary.parquet`:
  `ecbecdbf9be2794c5c82ac4b1171e203f393ae2a8a7b6e0d1b44bcd830dedc77`

If hashes differ, investigate whether SAMHSA/Baselight published a revised
public-use file. Do not assume equivalence.

## Known Non-Failures

These are expected and should not be treated as failures by themselves:

- `MLPClassifier` convergence warnings after 500 `lbfgs` iterations.
- scikit-learn future warnings about logistic regression arguments.
- `outputs/shap_values.png` absent during default verification; SHAP is optional
  unless `--shap` is explicitly run.
- JSON reports differing only in final floating-point digits.
- 2024 validation shows `sexual_orientation` unavailable in the public-use file;
  `lgbtq` is therefore imputed in that validation. This is a caveat, not a
  failed run.

## Real Red Flags

Treat these as substantive problems:

- `./code/run_si.sh verify` fails for required artifacts or metric drift.
- 2015-2023 diagonal total N is not `176957`.
- 2020 full-model AUC differs materially from `0.872`.
- The artifact rebuild does not match packaged metrics and example predictions.
- 2024 validation counts differ from `n=20588`, `positives=1292`, unless the
  data source hash also changed and the report explains why.
- Raw data or large public-use files are accidentally added to git.
- Any repo language implies deployment readiness, employer screening, diagnosis,
  or individual risk triage.

## Final Audit Report Template

Use this structure for the final report:

```markdown
# Independent Reproduction Audit

## Executive Summary

Pass / partial pass / fail.

## Environment

- Machine:
- OS:
- Python:
- External run root:
- Paper pipeline commit:
- Reference model commit/tag:

## Commands Run

List exact commands.

## Paper Pipeline 2015-2023

- Verification before rerun:
- Full rerun status:
- Verification after rerun:
- 2020 full AUC:
- Same-year AUC mean:
- Cross-year AUC mean:
- Diagonal total N:

## Reference Model

- Tests:
- Wheel build:
- CLI example:
- Artifact rebuild exact match:
- Max example-probability difference:

## NSDUH 2024 Fresh Validation

- Raw data hashes:
- n:
- positives:
- prevalence:
- AUC:
- Brier:
- calibration intercept/slope:
- threshold 0.17 operating point:

## Caveats

Include the 2024 LGBTQ unavailable/imputed caveat, survey-weight variance caveat,
and deployment-boundary caveat.

## Verdict

State whether the reproduction is strong enough to cite publicly and what, if
anything, should be fixed before the next release.
```

## Bottom Line

A strong audit outcome is:

1. The paper pipeline verifies from a fresh clone.
2. The full 2015-2023 rerun reproduces the temporal matrix and headline metrics.
3. The reference model installs, tests, builds, and scores deterministically.
4. The packaged reference model rebuilds exactly from the paper pipeline.
5. The 2024 fresh-data validation reproduces the previously reported metrics.
6. The auditor explicitly preserves the boundary: research validation tool, not
   deployable clinical or employer screening software.
