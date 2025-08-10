#!/usr/bin/env bash
# run_si.sh — one-shot runner & verifier for the Suicidal-Ideation-Pipeline
# Usage:
#   ./run_si.sh 2020 2021 2022 2023     # full run for these years
#   ./run_si.sh                         # defaults to 2020–2023
#   ./run_si.sh verify                  # verify outputs only, no re-run

set -euo pipefail
trap 'echo -e "\n❌ run_si.sh failed on line $LINENO"; exit 1' ERR

# Always run from the repo directory (the folder that contains this script)
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Quick path: verification only (no pipeline re-run)
if [[ "${1-}" == "verify" ]]; then
python - <<'PY'
from pathlib import Path
import json, sys

ok = True
msgs = []

def check_file(p):
    global ok
    if not Path(p).is_file():
        ok = False
        msgs.append(f"  MISSING → {p}")

expected = [
    "data/temporal_results.json",
    "outputs/fairness_table3.csv",
    "outputs/high_spec_metrics.json",
    "outputs/roc_overlay_2020.png",
    "outputs/roc_curves_data.csv",
    "outputs/mlp_model.joblib",
    "outputs/both_model_predictions_2020.csv",
    "outputs/calibration_mlp_2020.json",
    "outputs/appendix_stats/nri_idi.csv",
    "outputs/shap_values.png",
]
for p in expected: check_file(p)

# Metric tolerances (aligned with paper)
try:
    j = json.load(open("data/temporal_results.json"))
    auc = float(j["2020_full"]["auc"])
    if abs(auc - 0.8721) > 0.01:
        ok = False
        msgs.append(f"  AUC drift: got {auc:.4f}, expected ~0.8721 (±0.01)")
except Exception as e:
    ok = False
    msgs.append(f"  Could not parse data/temporal_results.json: {e}")

# Fairness table should have header + rows
try:
    nlines = sum(1 for _ in open("outputs/fairness_table3.csv", "r", encoding="utf-8"))
    if nlines < 3:
        ok = False
        msgs.append(f"  fairness_table3.csv has too few rows ({nlines})")
except Exception as e:
    ok = False
    msgs.append(f"  Could not read outputs/fairness_table3.csv: {e}")

print("\n──────── Verification Summary ────────")
if ok:
    print("✅ PASS — key artifacts present and metrics within tolerance.")
else:
    print("❌ FAIL — see details below:")
    for m in msgs:
        print(m)
    sys.exit(1)
PY
  exit 0
fi

# Create and activate a local venv if needed
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# Upgrade build tools and install deps
python -m pip install -U pip wheel
python -m pip install -r code/requirements.txt

# Freeze the exact environment for reproducibility
python -m pip freeze > requirements.lock

# Repro/Determinism knobs (keep training single-threaded across BLAS/OMP libs)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

# Years to run; default to full paper set
years=("$@")
if [ ${#years[@]} -eq 0 ]; then years=(2020 2021 2022 2023); fi

# Build the command (string for logs + array for safe execution)
PIPELINE_CMD_STR="python code/workplace_si_ml_pipeline.py --years ${years[*]} --download --no-plots --seed 42 --save-both-preds --high-spec 0.93 --table3 --nri-idi --export-roc --export-calibration-mlp --compare-spec 0.90 --calibration-stats"
PIPELINE_CMD=(python code/workplace_si_ml_pipeline.py --years "${years[@]}" --download --no-plots --seed 42 --save-both-preds --high-spec 0.93 --table3 --nri-idi --export-roc --export-calibration-mlp --compare-spec 0.90 --calibration-stats)

echo "▶️  ${PIPELINE_CMD_STR}"
mkdir -p outputs/appendix_stats data

# Run the pipeline
time "${PIPELINE_CMD[@]}"

# --- Compute data checksums (for reproducibility) ---
python - <<'PY'
import hashlib, pathlib

root = pathlib.Path("data")
if root.exists():
    # Collect substantial files under data/ (skip tiny files and our own checksum file)
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.name != "checksums.txt" and not p.name.endswith(".json"):
            if p.stat().st_size < 1024:
                continue
            paths.append(p)

    def sha256(fp):
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    new = {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
    out = root / "checksums.txt"

    # Compare with previous if it exists
    changed = []
    if out.exists():
        old = {}
        with open(out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # support "hash␠␠path" or "hash␠path"
                parts = line.split("  ", 1)
                if len(parts) == 1:
                    parts = line.split(" ", 1)
                if len(parts) == 2:
                    digest, rel = parts
                    old[rel] = digest
        for rel, dg in new.items():
            if old.get(rel) and old[rel] != dg:
                changed.append(rel)

    with open(out, "w", encoding="utf-8") as f:
        for rel, dg in new.items():
            f.write(f"{dg}  {rel}\n")

    if changed:
        print(f"⚠️  Checksums changed for {len(changed)} files in data/: " + ', '.join(changed[:5]) + (' …' if len(changed) > 5 else ''))
    else:
        print(f"✓ Wrote {out} with {len(new)} entries")
else:
    print('ℹ️ data/ directory not present; skipping checksums.')
PY

# --- Write lightweight provenance metadata (append JSONL) ---
PIPELINE_CMD="${PIPELINE_CMD_STR}" python - <<'PY'
import json, sys, os, platform, subprocess, datetime, pathlib
meta = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "cmd": os.environ.get("PIPELINE_CMD",""),
}
# Best-effort git commit (may not be a repo)
try:
    meta["git_commit"] = subprocess.check_output(
        ["git","rev-parse","--short","HEAD"],
        stderr=subprocess.DEVNULL, text=True
    ).strip()
except Exception:
    meta["git_commit"] = "NA"
# Include a short pip freeze snapshot
try:
    req = subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True)
    meta["pip_freeze"] = req.splitlines()
except Exception:
    meta["pip_freeze"] = []

pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)
with open("outputs/metadata.json","a") as f:
    f.write(json.dumps(meta) + "\n")
print("✓ Wrote outputs/metadata.json")
PY

# --- Built-in sanity checks (PASS/FAIL summary) ---
python - <<'PY'
from pathlib import Path
import json, sys

ok = True
msgs = []

def check_file(p):
    global ok
    if not Path(p).is_file():
        ok = False
        msgs.append(f"  MISSING → {p}")

expected = [
    "data/temporal_results.json",
    "outputs/fairness_table3.csv",
    "outputs/high_spec_metrics.json",
    "outputs/roc_overlay_2020.png",
    "outputs/roc_curves_data.csv",
    "outputs/mlp_model.joblib",
    "outputs/both_model_predictions_2020.csv",
    "outputs/calibration_mlp_2020.json",
    "outputs/appendix_stats/nri_idi.csv",
    "outputs/shap_values.png",
]
for p in expected: check_file(p)

# Metric tolerances (aligned with paper)
try:
    j = json.load(open("data/temporal_results.json"))
    auc = float(j["2020_full"]["auc"])
    if abs(auc - 0.8721) > 0.01:
        ok = False
        msgs.append(f"  AUC drift: got {auc:.4f}, expected ~0.8721 (±0.01)")
except Exception as e:
    ok = False
    msgs.append(f"  Could not parse data/temporal_results.json: {e}")

# Fairness table should have header + rows
try:
    nlines = sum(1 for _ in open("outputs/fairness_table3.csv", "r", encoding="utf-8"))
    if nlines < 3:
        ok = False
        msgs.append(f"  fairness_table3.csv has too few rows ({nlines})")
except Exception as e:
    ok = False
    msgs.append(f"  Could not read outputs/fairness_table3.csv: {e}")

print("\n──────── Verification Summary ────────")
if ok:
    print("✅ PASS — key artifacts present and metrics within tolerance.")
else:
    print("❌ FAIL — see details below:")
    for m in msgs:
        print(m)
    sys.exit(1)
PY

echo "✅ Done."
