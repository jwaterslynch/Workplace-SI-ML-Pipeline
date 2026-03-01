#!/usr/bin/env python3
"""
rolling_window_analysis.py — Reproduce rolling 3-year window vs single-year training.

Pools raw NSDUH data from 3 consecutive years, retrains XGBoost on the pooled
sample, and tests on the following year. Compares against single-year training.

Uses IDENTICAL model configuration and variable resolution to the main pipeline:
  - Features: male, age, veteran, drug_use, work_hours (basic 5-predictor set)
  - XGBoost: max_depth=3, n_estimators=150, scale_pos_weight=auto, seed=42
  - Imputation: median (SimpleImputer)
  - Scaling: StandardScaler
  - Employment filter: WRKSTATWK2 == 1
  - Outcome: SUICTHNK recoded via _bin (1→1, 2→0, skip→NaN)
  - Veteran: MILSTAT {2,3}→1, {85,94,97,98}→NaN, else→0
  - Drug use: ILLYR (already 0/1)
  - Age: CATAGE midpoints

Memory-efficient: reads raw .tab files in chunks, filters immediately.

Usage:
    python rolling_window_analysis.py
"""

import gc
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASIC_FEATURES = ["male", "age", "veteran", "drug_use", "work_hours"]
YEARS = list(range(2015, 2024))
CHUNK_SIZE = 20_000
SKIP_CODES = {85, 89, 94, 97, 98, 99}

# Column resolution — matches pipeline's validate_year_data() resolution order
COLUMN_MAP = {
    "employment": ["WRKSTATWK2"],
    "suicide":    ["SUICTHNK", "MHSUITHK"],  # pipeline uses SUICTHNK first
    "sex":        ["IRSEX"],
    "age":        ["CATAGE", "AGE2"],
    "military":   ["MILSTAT", "SERVICE"],      # pipeline uses MILSTAT first
    "drug_use":   ["ILLYR"],
    "work_hours": ["WRKDHRSWK2"],
}


def resolve_columns(all_cols: list) -> dict:
    """Find actual NSDUH column names for our standard variables."""
    col_set = set(all_cols)
    resolved = {}
    for std_name, candidates in COLUMN_MAP.items():
        hit = next((c for c in candidates if c in col_set), None)
        if hit:
            resolved[std_name] = hit
    return resolved


def _bin(series: pd.Series) -> pd.Series:
    """Pipeline's binary recoding: skip codes → NaN, {1→1, 2→0}, keep only 0/1."""
    out = pd.to_numeric(series, errors="coerce")
    out = out.where(~out.isin(SKIP_CODES), np.nan).replace({1: 1, 2: 0})
    return out.where(out.isin([0, 1]), np.nan)


def recode_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply pipeline-matching recoding to a chunk already renamed to standard names."""
    chunk = chunk.copy()

    # Employment filter
    if "employment" in chunk.columns:
        chunk = chunk[chunk["employment"] == 1].copy()

    # Outcome: apply _bin (handles both SUICTHNK 1/2 and MHSUITHK 0/1)
    if "suicide" in chunk.columns:
        chunk["suicide"] = _bin(chunk["suicide"])
        chunk = chunk[chunk["suicide"].isin([0, 1])].copy()

    if len(chunk) == 0:
        return chunk

    # Recode sex → male (IRSEX: 1=male, 2=female)
    if "sex" in chunk.columns:
        chunk["male"] = (chunk["sex"] == 1).astype(np.float32)
        chunk = chunk.drop(columns=["sex"])

    # Recode age (CATAGE: 1=12-17, 2=18-25, 3=26-34, 4=35-49, 5=50-64, 6=65+)
    if "age" in chunk.columns:
        if chunk["age"].max() <= 6:
            age_map = {1: 14.5, 2: 21.5, 3: 30, 4: 42, 5: 57, 6: 72.5}
            chunk["age"] = chunk["age"].map(age_map).astype(np.float32)
        else:
            age2_map = {
                1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17,
                7: 18, 8: 19, 9: 20, 10: 21,
                11: 23, 12: 27.5, 13: 32, 14: 37, 15: 42, 16: 52, 17: 62,
            }
            chunk["age"] = chunk["age"].map(age2_map).astype(np.float32)

    # Recode veteran from MILSTAT: {2,3}→1, {85,94,97,98}→NaN, else→0
    if "military" in chunk.columns:
        chunk["veteran"] = np.where(chunk["military"].isin([2, 3]), 1.0, 0.0)
        chunk.loc[chunk["military"].isin({85, 94, 97, 98}), "veteran"] = np.nan
        chunk = chunk.drop(columns=["military"])

    # Drug_use: apply _bin (ILLYR is already 0/1, _bin is a no-op; safe for other codings)
    if "drug_use" in chunk.columns:
        chunk["drug_use"] = _bin(chunk["drug_use"])

    # work_hours: skip codes → NaN
    if "work_hours" in chunk.columns:
        chunk.loc[chunk["work_hours"].isin(SKIP_CODES), "work_hours"] = np.nan

    # Keep only features + suicide
    keep = [c for c in BASIC_FEATURES if c in chunk.columns] + ["suicide"]
    return chunk[[c for c in keep if c in chunk.columns]].copy()


def load_year(year: int) -> pd.DataFrame:
    """Load one NSDUH year using chunked reading. Returns cleaned df (~20K rows)."""
    tab_path = DATA_DIR / f"NSDUH_{year}.tab"
    if not tab_path.exists():
        raise FileNotFoundError(f"Missing: {tab_path}")

    header_cols = pd.read_csv(tab_path, sep="\t", nrows=0).columns.tolist()
    col_map = resolve_columns(header_cols)
    use_cols = list(col_map.values())
    rename = {v: k for k, v in col_map.items()}

    print(f"    {year}: resolved columns = {col_map}", flush=True)

    chunks = []
    reader = pd.read_csv(tab_path, sep="\t", usecols=use_cols,
                         chunksize=CHUNK_SIZE, low_memory=True)
    for chunk in reader:
        chunk = chunk.rename(columns=rename)
        cleaned = recode_chunk(chunk)
        if len(cleaned) > 0:
            chunks.append(cleaned)
        del chunk, cleaned

    gc.collect()
    if not chunks:
        raise ValueError(f"No valid data for {year}")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def train_and_evaluate(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Train XGBoost on df_train, evaluate on df_test."""
    feats = [c for c in BASIC_FEATURES if c in df_train.columns and not df_train[c].isna().all()]

    X_tr = df_train[feats].values.astype(np.float32)
    y_tr = df_train["suicide"].values.astype(np.float32)
    X_te = df_test[feats].values.astype(np.float32)
    y_te = df_test["suicide"].values.astype(np.float32)

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_tr_proc = scl.fit_transform(imp.fit_transform(X_tr))
    X_te_proc = scl.transform(imp.transform(X_te))

    spw = float((len(y_tr) - y_tr.sum()) / y_tr.sum())
    mdl = XGBClassifier(
        max_depth=3, n_estimators=150,
        scale_pos_weight=spw, random_state=42,
        n_jobs=1,
    )
    mdl.fit(X_tr_proc, y_tr)

    p = mdl.predict_proba(X_te_proc)[:, 1]
    auc = roc_auc_score(y_te, p)

    del X_tr_proc, X_te_proc, mdl
    gc.collect()

    return {
        "auc": float(auc),
        "n_train": int(len(y_tr)),
        "n_train_pos": int(y_tr.sum()),
        "n_test": int(len(y_te)),
        "n_test_pos": int(y_te.sum()),
    }


def main():
    print("=" * 70)
    print("ROLLING WINDOW ANALYSIS: 3-Year Pooled vs Single-Year Training")
    print("=" * 70, flush=True)

    # Phase 1: Load all years
    print("\nPhase 1: Loading data (chunked reading)...", flush=True)
    data = {}
    for yr in YEARS:
        df = load_year(yr)
        data[yr] = df
        print(f"  {yr}: N={len(df):,}, positives={int(df['suicide'].sum()):,}, "
              f"cols={list(df.columns)}", flush=True)
        gc.collect()

    # Verify Ns match pipeline
    pipeline_ns = {2015: 20519, 2016: 20455, 2017: 20676, 2018: 20974,
                   2019: 20579, 2020: 12458, 2021: 20367, 2022: 21119, 2023: 19810}
    print("\nN verification vs pipeline:")
    all_match = True
    for yr in YEARS:
        expected = pipeline_ns[yr]
        actual = len(data[yr])
        match = "OK" if actual == expected else f"MISMATCH (expected {expected}, got {actual})"
        if actual != expected:
            all_match = False
        print(f"  {yr}: {actual:,} {match}")
    print(flush=True)

    if not all_match:
        print("  WARNING: Some Ns don't match. Proceeding anyway.\n")

    # Phase 2: Rolling 3-year vs single-year
    print(f"Phase 2: Training models...", flush=True)
    test_years = list(range(2018, 2024))
    results = {"test_years": [], "rolling_3year": [], "single_year": [],
               "rolling_details": [], "single_details": []}

    print(f"\n{'Test':>6}  {'Single-Year':>14}  {'Rolling 3-Year':>14}  {'Diff':>8}")
    print(f"{'-'*55}", flush=True)

    for test_yr in test_years:
        df_test = data[test_yr]

        # Single-year: train on t-1
        single_res = train_and_evaluate(data[test_yr - 1], df_test)

        # Rolling 3-year: pool t-3, t-2, t-1
        rolling_train = pd.concat(
            [data[test_yr - 3], data[test_yr - 2], data[test_yr - 1]],
            ignore_index=True
        )
        rolling_res = train_and_evaluate(rolling_train, df_test)
        del rolling_train
        gc.collect()

        diff = rolling_res["auc"] - single_res["auc"]
        results["test_years"].append(test_yr)
        results["single_year"].append(single_res["auc"])
        results["rolling_3year"].append(rolling_res["auc"])
        results["single_details"].append(single_res)
        results["rolling_details"].append(rolling_res)

        print(f"{test_yr:>6}  {single_res['auc']:>14.4f}  {rolling_res['auc']:>14.4f}  {diff:>+8.4f}", flush=True)

    # Summary
    single = np.array(results["single_year"])
    rolling = np.array(results["rolling_3year"])
    diff_arr = rolling - single
    t_stat, p_val = stats.ttest_rel(rolling, single)

    print(f"{'-'*55}")
    print(f"{'Mean':>6}  {np.mean(single):>14.4f}  {np.mean(rolling):>14.4f}  {np.mean(diff_arr):>+8.4f}")
    print(f"{'SD':>6}  {np.std(single,ddof=1):>14.4f}  {np.std(rolling,ddof=1):>14.4f}")
    print(f"\nPaired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"All improvements positive? {all(d > 0 for d in diff_arr)}")

    # Compare against documented values
    doc_rolling = [0.725, 0.707, 0.705, 0.711, 0.731, 0.717]
    doc_single  = [0.712, 0.692, 0.687, 0.687, 0.709, 0.712]

    print(f"\n{'='*70}")
    print("COMPARISON WITH ANALYSIS_RESULTS.md VALUES")
    print(f"{'='*70}")
    print(f"\n{'Test':>6}  {'Doc Single':>12}  {'Run Single':>12}  {'Doc Roll':>12}  {'Run Roll':>12}")
    print(f"{'-'*70}")
    for i, yr in enumerate(test_years):
        ds, rs = doc_single[i], results["single_year"][i]
        dr, rr = doc_rolling[i], results["rolling_3year"][i]
        s_flag = "OK" if abs(ds - rs) < 0.003 else f"D={rs-ds:+.4f}"
        r_flag = "OK" if abs(dr - rr) < 0.003 else f"D={rr-dr:+.4f}"
        print(f"{yr:>6}  {ds:>12.3f}  {rs:>12.4f}  {dr:>12.3f}  {rr:>12.4f}  {s_flag:>10} {r_flag:>10}")

    # Save
    output = {
        "description": "Rolling 3-year window vs single-year training comparison",
        "method": "Pool raw data from 3 years, retrain XGBoost, test on next year",
        "model": "XGBoost(max_depth=3, n_estimators=150, scale_pos_weight=auto, seed=42)",
        "features": BASIC_FEATURES,
        "test_years": results["test_years"],
        "single_year_auc": results["single_year"],
        "rolling_3year_auc": results["rolling_3year"],
        "differences": [float(d) for d in diff_arr],
        "summary": {
            "mean_single": float(np.mean(single)),
            "mean_rolling": float(np.mean(rolling)),
            "mean_difference": float(np.mean(diff_arr)),
            "paired_t": float(t_stat),
            "p_value": float(p_val),
        },
        "single_year_details": results["single_details"],
        "rolling_3year_details": results["rolling_details"],
        "documented_values": {
            "source": "ANALYSIS_RESULTS.md",
            "rolling_3year_auc": doc_rolling,
            "single_year_auc": doc_single,
        },
        "per_year_n": {str(yr): len(data[yr]) for yr in YEARS},
    }

    out_path = OUTPUT_DIR / "rolling_window_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
