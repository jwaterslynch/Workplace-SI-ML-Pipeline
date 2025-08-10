#!/usr/bin/env python3
"""
workplace_si_ml_pipeline.py  –  Temporal analysis of suicide‑ideation prediction
────────────────────────────────────────────────────────────────────────
•  Downloads / extracts NSDUH public‑use files (2008‑2023)
•  Aligns variable names across survey waves
•  Runs two‑tier temporal‑stability analysis (Full vs Basic feature sets)
•  Replicates the original SPSS MLP pipeline (5 progressive models)
•  Compares original MLP vs modern XGBoost on the same 2020 sample
•  Provides diagnostic utilities to track sample‑loss & threshold effects
"""

# ======================================================================
# Imports
# ======================================================================
import argparse
import json
import pathlib
import zipfile

import numpy as np
# ------------------------------------------------------------------
# Back‑compat shim for NumPy ≥ 2.0
# Some downstream libraries (e.g. statsmodels, patsy) still reference
# the deprecated alias `np.bool8`, which was removed in NumPy 2.x.
# Defining it here keeps those libraries working without having to
# downgrade NumPy.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
import requests
import urllib3
import re  # regex helpers for flexible variable‑detection
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import joblib  # needed for saving trained models

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    auc,
    brier_score_loss,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier, plot_importance
# Optional SHAP import (graceful fallback if unavailable)
try:
    import shap  # type: ignore
except Exception:
    shap = None


# ── Global random-seed (overridden by --seed) ───────────────────────
SEED: int = 42
# Default directory where all artefacts will be written
OUTPUTS_DIR: pathlib.Path = pathlib.Path("./outputs")

# ------------------------------------------------------------------
# Defaults for Bayesian hierarchical logistic (Appendix A)
BAYES_DRAWS: int = 2000   # posterior draws
BAYES_TUNE:  int = 1000   # warm-up / tuning steps
FORCE_BAYES: bool = False # re-fit even if cached artefacts exist

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ======================================================================
# Download catalogue
# ======================================================================
# ------------------------------------------------------------------
# Add fallback URLs for older survey waves (2008‑2019) without
# clobbering the explicit 2020‑2023 entries above.
NSDUH_URLS = {
    2020: "https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-2020-DS0001-bndl-data-tsv_v1.zip",
    2021: "https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-2021-DS0001-bndl-data-tsv_v4.zip",
    2022: "https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-2022-DS0001-bndl-data-tsv_v2.zip",
    2023: "https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-2023-DS0001-bndl-data-tsv_v1.zip",
}
for yr in range(2008, 2021):
    NSDUH_URLS.setdefault(
        yr,
        f"https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-{yr}-DS0001-bndl-data-tsv.zip",
    )

# ======================================================================
# Global flags
# ======================================================================
# >>> PATCH: default threshold‑sweep grid
THR_GRID_DEFAULT = (0.0, 1.0, 0.05)
EMPLOYMENT_FILTER: bool = True  # default = keep employed only
SHOW_PLOTS: bool = True   # can be disabled with --no-plots

# ----------------------------------------------------------------------
CORE_VARIABLES = {
    "suicide": [
        "SUICTHNK", "MHSUITHK", "MHSUITRY",
        "SUICTHNK1", "SUIPLANYR", "SUIPLANYR1", "SUITRYYR", "SUITRYYR1",
        "IRSUICTHNK", "IRSUIPLANYR", "IRSUITRYYR",
    ],
    "k6_score": [
        # pre‑COVID names
        "K6SCMON", "K6SCMAX",
        # 2022+ 12‑month name
        "K6SCYR",
        # occasional summaries / alternates
        "K6SCYR2", "K6SUM", "K6SC_12M", "K6SC_TOTAL",
        "KSSLR6MON", "KSSLR6MONED", "KSSLR6YR", "KSSLR6YRED", "KSSLR6MAX",
    ],
    "employment": [
        "WRKSTATWK2", "WRKSTATWK", "IRWRKSTAT", "IRWRKSTAT2",
        "WRKSTAT", "WRKSTAT2", "IRWRKST", "WRKST"
    ],
    "work_hours": [
        "WRKDHRSWK2", "WRKHRSUS2", "WRKHRSJOB2",
        "WRKDHRSWK", "WRKHRSUS", "WRKHRSJOB",
        "WRKHRSUS", "WRKHRSJOB"
    ],
    "sex": ["IRSEX"],
    "age": ["CATAGE", "AGE2"],
    "marital": ["IRMARIT"],
    "education": [
        "EDUHIGHCAT", "EDUHIGHCAT2",
        "IREDUHIGHST", "IREDUHIGHST2",
        "EDUHIGHST", "EDUHIGHST2", "EDUHIGHESTLVL"
    ],
    "income": ["INCOME", "IRFAMIN3"],
    "health_insurance": ["IRINSUR4", "ANYHLTI2"],
    "drug_use": [
        "ILLYR", "IRILLYR",
        "ANYILLYR", "ANYILLYR2",
        "ANYIMP", "ANYIMP2",
        "ANYDRUGYR", "ANYILLICIT", "ANYILLYR3"
    ],
    "mental_health_tx": [
        # 2020 public‑use (all‑time help received)
        "TXEVRRCVD2",
        # 2018‑2019 / 2021 telephone short‑form
        "AMHTXRC3", "AMHTXRC2",
        # 2022+ “received mental‑health tx past 12 m”
        "TXYRRCVD", "TXYRRCVD2", "TXEVRRCVD3",
        # PATCH: add new aliases introduced 2023+
        "TXYRRCVD", "TXYRRCVD1", "TXYRRCVD2", "TXRCVDYR", "TXRCVDYR1",
        "TXRCVDYR2",
        "TXRCVDYR3",
        "TXYRRCVD",
        "TXYRRCVD1",
        "TXYRRCVD2",
        "MHTXRCVD",
        "MHTXRCVDYR",
        "MHCAREYR",
        "MHCAREYR1",
        "MHCAREYR2",
        "MHCARE",
        "MHTXRCVDYR",
        "MHTXRCVDYR1",
        "MHTXRCVDYR2",
        "MHTRTPY",
        "MHTRTOTHPY",
    ],
    "military": ["MILSTAT", "SERVICE"],
    "sexual_orientation": [
        "SEXIDENT", "SEXATRACT",
        # PATCH: add new aliases introduced 2023+
        "IRSEXIDENT", "SEXIDENT1", "SEXIDENT2",
    ],
    "overall_health": ["HEALTH", "HEALTH2"],
    "sick_days": ["WRKSICKMO", "WRKSICKDY"],
    "criminal": ["BOOKED"],
    "survey_design": ["VESTR", "VEREP", "ANALWT_C"],
}

SKIP_CODES = {85, 89, 94, 97, 98, 99}

# ======================================================================
# Utility: download & extract
# ======================================================================


def download_year(year: int, data_dir: pathlib.Path) -> pathlib.Path | None:
    if year not in NSDUH_URLS:
        print(f"❌ No URL available for {year}")
        return None

    zip_path = data_dir / f"NSDUH-{year}.zip"
    if zip_path.exists():
        print(f"✓ {year} archive already present")
        return zip_path

    url = NSDUH_URLS[year]
    print(f"📥 Downloading {year} …")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except requests.exceptions.SSLError:
        r = requests.get(url, stream=True, verify=False, timeout=30)
        r.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"✓ Downloaded to {zip_path.name}")
    return zip_path


def extract_tab_file(zip_path: pathlib.Path, data_dir: pathlib.Path) -> pathlib.Path:
    year = int(zip_path.stem.split("-")[1])
    tab_path = data_dir / f"NSDUH_{year}.tab"
    if tab_path.exists():
        return tab_path

    print(f"📦 Extracting {year} …")
    with zipfile.ZipFile(zip_path) as zf:
        # pick the largest file >1 MB that looks like a data table
        cands = [
            (f, zf.getinfo(f).file_size)
            for f in zf.namelist()
            if not f.startswith("__MACOSX")
            and any(pat in f.lower() for pat in (".tab", ".tsv", ".txt", "ds0001"))
            and zf.getinfo(f).file_size > 1_000_000
        ]
        if not cands:
            raise ValueError(f"No data file in {zip_path.name}")
        sel = max(cands, key=lambda t: t[1])[0]

        with open(tab_path, "wb") as out, zf.open(sel) as src:
            while True:
                chunk = src.read(1_048_576)
                if not chunk:
                    break
                out.write(chunk)
    print(f"✓ Extracted to {tab_path.name}")
    return tab_path


# ======================================================================
# Validation & variable alignment
# ======================================================================


def validate_year_data(df: pd.DataFrame, yr: int) -> dict:
    print("\n" + "=" * 60 + f"\nVALIDATING {yr} DATA\n" + "=" * 60)
    print(f"Shape: {df.shape[0]:,} × {df.shape[1]:,}")

    found: dict[str, str] = {}
    missing: list[str] = []
    for std, alts in CORE_VARIABLES.items():
        # ------------------------------------------------------------------
        # Special‑case flexible matches for K‑6 distress score and
        # mental‑health‑treatment variables, which SAMHSA rename often.
        # ------------------------------------------------------------------
        if std == "k6_score":
            # Accept any column whose label starts with “K6SC” or "KSSLR6" (case‑insensitive),
            # e.g., K6SCMON, K6SCYR, K6SCYR2, KSSLR6MON, KSSLR6YR, etc.
            hit = next((c for c in df.columns if re.match(r"(K6SC|KSSLR6)", c, re.I)), None)
        elif std == "mental_health_tx":
            # Accept a broad family of mental‑health‑treatment variables, which
            # SAMHSA keep renaming (TXEVRRCVD*, TXYRRCVD*, TXRCVDYR*, MHCAREYR,
            # AMHTXRC*, etc.), and new MHT*RTPY style variables.
            hit = next(
                (
                    c
                    for c in df.columns
                    if re.match(
                        r"(TX.*RCVD|TX.*CVD|TXRCVDYR\d*|TXYRRCVD\d*|AMHTXR?C\d*|MHTXRCVD\d*|MHTXRCVDYR\d*|MHCAREYR\d*|MHT.*RT.*PY)",
                        c,
                        re.I,
                    )
                ),
                None,
            )
        elif std == "sexual_orientation":
            # Accept original SEXIDENT or imputed/edited counterparts (e.g., SEXIDENT1/2, IRSEXIDENT)
            hit = next((c for c in df.columns if re.match(r"(IR)?SEXIDENT\d*", c, re.I)), None)
        elif std == "employment":
            # IRWRKSTAT, WRKSTAT, WRKSTATWK, WRKSTATWK2, etc., plus pre-2015 aliases
            hit = next(
                (c for c in df.columns
                 if re.match(r"((IR)?WRKSTAT\w*|(IR)?WRKST\w*|EMPSTAT\w*)", c, re.I)),
                None,
            )
        elif std == "work_hours":
            # WRKHRSUS, WRKHRSJOB, WRKDHRSWK, WRKDHRSWK2 … plus early aliases
            hit = next((c for c in df.columns
                        if re.match(r"(WRK.*HR.*WK\d*|WRKHRS(US|JOB)\w*|WRKHRS\w*)", c, re.I)), None)
        elif std == "education":
            # Pre‑2015 waves label education as IREDUHIGHST (adult) or
            # variations such as EDUHIGHST, EDUHIGHST2, EDUHIGHESTLVL, etc.
            # Accept any column that contains EDU+HIGH or the specific older
            # IR prefix patterns.
            hit = next(
                (
                    c
                    for c in df.columns
                    if re.match(
                        r"((IR)?EDU.*HIGH\w*|IREDUHIGHST\w*|EDUHIGHST\w*|EDUHIGHEST\w*)",
                        c,
                        re.I,
                    )
                ),
                None,
            )
        elif std == "drug_use":
            # ANYILLYR / ANYILLICIT plus legacy ILLYR / IRILLYR flags
            hit = next(
                (
                    c
                    for c in df.columns
                    if re.match(
                        r"(ILLYR|IRILLYR|ANYILLYR\w*|ANYDRUGYR\w*|ANYIMP\w*|ANYILLIC\w*)",
                        c,
                        re.I,
                    )
                ),
                None,
            )
        else:
            hit = next((a for a in alts if a in df.columns), None)
        (found.__setitem__(std, hit) if hit else missing.append(std))

    print(f"✓ Found {len(found)}/{len(CORE_VARIABLES)} core variables")
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")

    if "suicide" in found:
        v = found["suicide"]
        vc = df[v].value_counts().sort_index().head(10)
        print(f"\n{v} distribution:\n{vc}")
        if {1, 2}.issubset(vc.index):
            prev = vc[1] / (vc[1] + vc[2])
            print(f"Suicide‑ideation prevalence: {prev:.1%}")

    if "employment" in found:
        emp = (df[found["employment"]] == 1).sum()
        print(f"Employed individuals: {emp:,} ({emp/len(df):.1%})")
    return found


def align_variables_across_years(found_by_year: dict) -> dict:
    print("\n" + "=" * 60 + "\nALIGNING VARIABLES ACROSS YEARS\n" + "=" * 60)
    choice: dict[str, str] = {}
    # ------------------------------------------------------------------
    # Choose one canonical column per logical variable – except for the
    # K‑6 distress score, whose label changes (K6SCMON → K6SCYR) after
    # the COVID‑era short‑form.  For k6_score we therefore keep a
    # *year‑specific* mapping so every wave can use whichever K‑6 column
    # is actually present.
    # ------------------------------------------------------------------
    per_year_map: dict[int, dict[str, str]] = {}       # yr → {std: col}

    for std, alts in CORE_VARIABLES.items():
        seen: dict[str, list[int]] = {}
        for yr, hits in found_by_year.items():
            if std in hits:
                seen.setdefault(hits[std], []).append(yr)

        if not seen:
            print(f"⚠️  {std}: absent everywhere")
            continue

        # ---------- special‑case the K‑6 score and mental‑health‑treatment flag ----------
        if std in ("k6_score", "mental_health_tx"):
            for col, yrs in seen.items():
                for y in yrs:
                    per_year_map.setdefault(y, {})[std] = col
            total = sum(len(v) for v in seen.values())
            print(f"✓ {std}: per‑year mapping written for "
                  f"{total}/{len(found_by_year)} years")
            continue

        # ---------- default rule for all other variables --------------
        best = max(seen.items(), key=lambda kv: (len(kv[1]), max(kv[1])))[0]
        choice[std] = best
        print(f"✓ {std}: using '{best}' ({len(seen[best])}/{len(found_by_year)})")

    # merge the per‑year K‑6 overrides into the main mapping dict
    if per_year_map:
        choice["__per_year__"] = per_year_map

    return choice


# ======================================================================
# Cleaning helpers
# ======================================================================


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    def _bin(col: pd.Series) -> pd.Series:
        out = pd.to_numeric(col, errors="coerce")
        out = out.where(~out.isin(SKIP_CODES), np.nan).replace({1: 1, 2: 0})
        return out.where(out.isin([0, 1]), np.nan)

    for b in ("suicide", "drug_use", "mental_health_tx", "criminal"):
        if b in df.columns:
            df[b] = _bin(df[b])

    num = [
        "k6_score",
        "work_hours",
        "sick_days",
        "income",
        "education",
        "age",
        "overall_health",
    ]
    for v in num:
        if v in df.columns:
            df.loc[df[v].isin(SKIP_CODES), v] = np.nan

    if "sex" in df.columns:
        df["male"] = (df["sex"] == 1).astype(int)

    if "marital" in df.columns:
        df["married"] = (df["marital"] == 1).astype(float)
        df.loc[df["marital"].isin(SKIP_CODES), "married"] = np.nan

    if "sexual_orientation" in df.columns:
        df["lgbtq"] = df["sexual_orientation"].replace({1: 0, 2: 1, 3: 1})
        df.loc[df["sexual_orientation"].isin(SKIP_CODES), "lgbtq"] = np.nan

    if "military" in df.columns:
        df["veteran"] = np.where(df["military"].isin([2, 3]), 1, 0)
        df.loc[df["military"].isin({85, 94, 97, 98}), "veteran"] = np.nan

    # make sure every downstream feature exists
    for c in (
        "male",
        "married",
        "lgbtq",
        "veteran",
        "drug_use",
        "work_hours",
        "k6_score",
        "mental_health_tx",
        "sexual_orientation",
    ):
        if c not in df.columns:
            df[c] = np.nan

    # ------------------------------------------------------------------
    # Fallback: if the explicit employment flag is absent in older waves,
    # infer employment from non‑missing work‑hours (> 0 → employed).
    # ------------------------------------------------------------------
    if "employment" not in df.columns or df["employment"].isna().all():
        if "work_hours" in df.columns:
            df["employment"] = np.where(
                df["work_hours"].notna() & (df["work_hours"] > 0), 1, np.nan
            )

    return df


# ======================================================================
# Load / clean one year
# ======================================================================


def load_and_clean_year(
    yr: int, data_dir: pathlib.Path, mapping: dict[str, str]
) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"NSDUH_{yr}.tab", sep="\t", low_memory=False)
    # ----------------------------------------------
    # Select the correct source column for each std
    # ----------------------------------------------
    out_cols: dict[str, pd.Series] = {}

    for std in CORE_VARIABLES:
        if std == "__per_year__":
            continue

        # 1. year‑specific override?
        chosen = mapping.get("__per_year__", {}).get(yr, {}).get(std)

        # 2. global winner?
        if not chosen:
            chosen = mapping.get(std)

        # 3. if still absent in this wave, scan all known aliases
        if (chosen is None) or (chosen not in df.columns):
            chosen = next((a for a in CORE_VARIABLES[std] if a in df.columns), None)

        # 4. keep column if found
        if chosen and chosen in df.columns:
            out_cols[std] = df[chosen]

    out = pd.DataFrame(out_cols)
    out["year"] = yr
    out = clean_data(out)
    # ensure every canonical column is present
    for std in CORE_VARIABLES:
        if std not in out.columns:
            out[std] = np.nan
    return out


# ======================================================================
# Temporal‑stability analysis
# ======================================================================


def plot_feature_gain(model, feature_names, title="XGBoost gain"):
    """
    Draw a gain‑based feature‑importance bar plot.

    Handles three cases transparently:
      • raw XGBClassifier / Booster
      • CalibratedClassifierCV wrapper (unwraps the base estimator)
      • unfitted / incompatible objects → gracefully skip

    The plot is suppressed outright when SHOW_PLOTS is False.
    """
    if not SHOW_PLOTS:
        return

    # ── Calibrated wrapper?  Peel off the underlying XGB model ─────────
    if isinstance(model, CalibratedClassifierCV):
        calibrator = model.calibrated_classifiers_[0]
        if hasattr(calibrator, "base_estimator"):
            model = calibrator.base_estimator
        elif hasattr(calibrator, "estimator"):
            model = calibrator.estimator
        else:
            raise AttributeError("Calibrator has no estimator attribute")

    # ── Attempt to plot; fall back silently on failure ─────────────────
    try:
        ax = plot_importance(
            model,
            importance_type="gain",
            max_num_features=min(12, len(feature_names)),
            show_values=False,
            height=0.4,
        )
        ax.set_title(title)
        ax.figure.tight_layout()
        plt.show()
    except ValueError:
        # Happens if the underlying estimator is not a fitted XGBClassifier
        print("⚠️  Skipping feature‑importance plot – model is not a fitted XGBClassifier.")


def _train_test_single_year(df, feats, label=""):
    feats = [f for f in feats if f in df.columns and not df[f].isna().all()]
    print(f"{label}: using {len(feats)} predictors → {feats}")

    if "employment" in df.columns and EMPLOYMENT_FILTER:
        df = df[df["employment"] == 1].copy()
    if len(feats) < 3:
        return {}

    X = df[feats]
    y = df["suicide"]
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    X, y = X.loc[y.isin([0, 1])], y.loc[y.isin([0, 1])]
    if y.sum() < 10:
        return {}

    # --- calibrated XGBoost ------------------------------------------
    xgb_base = XGBClassifier(
        max_depth=3,
        n_estimators=150,
        scale_pos_weight=(len(y) - y.sum()) / y.sum(),
        random_state=42,
    )
    # 5‑fold Platt‑scaling (sigmoid) calibration
    mdl = CalibratedClassifierCV(
        estimator=xgb_base,
        method="sigmoid",
        cv=5,
    )
    cv_auc = cross_val_score(
        mdl,
        SimpleImputer(strategy="median").fit_transform(X),
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
    )
    print(f"   5‑fold CV AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    mdl.fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)

    # threshold by F1 on training fold
    proba_tr = mdl.predict_proba(scl.transform(imp.transform(X_tr)))[:, 1]
    pr, rc, thr = precision_recall_curve(y_tr, proba_tr)
    f1 = (2 * pr * rc) / np.where(pr + rc == 0, 1, pr + rc)
    best_thr = thr[np.argmax(f1)]

    proba = mdl.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]
    pred = (proba >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    if SHOW_PLOTS:
        plot_feature_gain(mdl, feats, title=f"Feature gain – {label}")

    return {
        "auc": roc_auc_score(y_te, proba),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "n_positive": int(y_te.sum()),
        "n_total": len(y_te),
    }


def run_temporal_analysis(data_dict: dict) -> dict:
    print("\n" + "=" * 60 + "\nTEMPORAL STABILITY ANALYSIS\n" + "=" * 60)
    full_tpl = [
        "k6_score",
        "male",
        "age",
        "married",
        "lgbtq",
        "veteran",
        "drug_use",
        "mental_health_tx",
        "work_hours",
    ]
    basic_tpl = ["male", "age", "veteran", "drug_use", "work_hours"]

    def avail(template, df):
        return [c for c in template if c in df.columns and not df[c].isna().all()]

    print("\nChecking variable availability per year:")
    for yr, df in data_dict.items():
        miss = [c for c in full_tpl if c not in avail(full_tpl, df)]
        print(f"{yr}: {len(full_tpl)-len(miss)}/{len(full_tpl)} available "
              f"(missing: {', '.join(miss) if miss else 'none'})")

    res: dict = {}

    # ── FULL model (2020 only) ─────────────────────────────────────────
    if 2020 in data_dict:
        fcols = avail(full_tpl, data_dict[2020])
        if "k6_score" in fcols and len(fcols) >= 3:
            print("\n--- FULL MODEL (K6 present, train/test on 2020) ---")
            res["2020_full"] = _train_test_single_year(
                data_dict[2020], fcols, label="2020 Full Model"
            )

    # ── BASIC model (no K6, train × test grid) ────────────────────────
    print("\n--- BASIC MODEL (No K6, all years) ---")
    basic: dict[int, dict] = {}

    for tr_yr in sorted(data_dict):
        df_tr = data_dict[tr_yr]
        if EMPLOYMENT_FILTER and "employment" in df_tr.columns:
            df_tr = df_tr[df_tr["employment"] == 1].copy()
        feats = avail(basic_tpl, df_tr)
        if len(feats) < 3:
            continue

        X_tr, y_tr = df_tr[feats], df_tr["suicide"]
        mask = ~y_tr.isna()
        X_tr, y_tr = X_tr.loc[mask], y_tr.loc[mask]
        X_tr, y_tr = X_tr.loc[y_tr.isin([0, 1])], y_tr.loc[y_tr.isin([0, 1])]
        if y_tr.sum() < 10:
            continue

        imp, scl = SimpleImputer(strategy="median"), StandardScaler()
        mdl = XGBClassifier(
            max_depth=3,
            n_estimators=150,
            scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
            random_state=42,
        )
        mdl.fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)

        # F1‑threshold
        p_tr = mdl.predict_proba(scl.transform(imp.transform(X_tr)))[:, 1]
        pr, rc, thr = precision_recall_curve(y_tr, p_tr)
        f1 = (2 * pr * rc) / np.where(pr + rc == 0, 1, pr + rc)
        thr_best = thr[np.argmax(f1)]

        yearly: dict = {}
        for te_yr, df_te in data_dict.items():
            df_sub = (
                df_te[df_te["employment"] == 1].copy()
                if EMPLOYMENT_FILTER and "employment" in df_te.columns
                else df_te
            )
            X_te = df_sub.reindex(columns=feats)
            y_te = df_sub["suicide"]
            mask = ~y_te.isna()
            X_te, y_te = X_te.loc[mask], y_te.loc[mask]
            if y_te.sum() + (y_te == 0).sum() == 0:
                continue

            p = mdl.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]
            pred = (p >= thr_best).astype(int)
            auc_v = roc_auc_score(y_te, p)
            tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
            yearly[te_yr] = {
                "auc": auc_v,
                "sensitivity": tp / (tp + fn) if tp + fn else 0,
                "specificity": tn / (tn + fp) if tn + fp else 0,
                "n_positive": int(y_te.sum()),
                "n_total": len(y_te),
            }
        basic[tr_yr] = yearly
    res["basic"] = basic

    # ── SAME‑YEAR “upper‑bound” check ────────────────────────────────
    print("\n--- SAME‑YEAR OVER‑FIT CHECK (baseline feature set) ---")
    within: dict[int, dict] = {}
    for yr, df in data_dict.items():
        within[yr] = _train_test_single_year(
            df, basic_tpl, label=f"{yr} intra‑year"
        )
    res["within_year"] = within
    return res


# ======================================================================
# Original‑paper replication & comparison
# ======================================================================


def replicate_original_models(df_2020: pd.DataFrame) -> dict:
    if EMPLOYMENT_FILTER and "employment" in df_2020.columns:
        df = df_2020[df_2020["employment"] == 1].copy()
    else:
        df = df_2020.copy()

    y = df["suicide"]
    MODELS = {
        "Model_1_Basic": dict(
            feats=["income", "work_hours", "male", "education", "age"],
            hidden=3,
            desc="Basic demographics",
        ),
        "Model_2_Work": dict(
            feats=[
                "income",
                "work_hours",
                "male",
                "education",
                "age",
                "sick_days",
                "health_insurance",
            ],
            hidden=6,
            desc="Work context factors",
        ),
        "Model_3_Risk": dict(
            feats=[
                "income",
                "work_hours",
                "male",
                "education",
                "age",
                "sick_days",
                "health_insurance",
                "drug_use",
                "criminal",
                "mental_health_tx",
                "veteran",
                "lgbtq",
                "married",
            ],
            hidden=6,
            desc="Individual risk factors",
        ),
        "Model_4_Psych": dict(
            feats=[
                "income",
                "work_hours",
                "male",
                "education",
                "age",
                "sick_days",
                "health_insurance",
                "drug_use",
                "criminal",
                "mental_health_tx",
                "veteran",
                "lgbtq",
                "married",
                "k6_score",
                "overall_health",
            ],
            hidden=5,
            desc="Psychological indicators",
        ),
    }

    res: dict = {}
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()

    for nm, cfg in MODELS.items():
        feats = [f for f in cfg["feats"] if f in df.columns]
        if len(feats) < 3:
            continue

        X = df[feats]
        mask = ~y.isna()
        X, y_clean = X.loc[mask], y.loc[mask]
        if y_clean.sum() < 10:
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_clean, test_size=0.30, stratify=y_clean, random_state=42
        )
        X_tr = scl.fit_transform(imp.fit_transform(X_tr))
        X_te = scl.transform(imp.transform(X_te))

        mlp = MLPClassifier(
            hidden_layer_sizes=(cfg["hidden"],),
            activation="tanh",
            solver="lbfgs",
            max_iter=500,   # bumped from 200
            random_state=42,
        )
        mlp.fit(X_tr, y_tr)

        proba = mlp.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.10).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        res[nm] = dict(
            features=feats,
            auc=roc_auc_score(y_te, proba),
            sensitivity=tp / (tp + fn),
            specificity=tn / (tn + fp),
        )

    # step‑wise logistic → Model 5
    if "Model_4_Psych" in res:
        feats = res["Model_4_Psych"]["features"]
        X = df[feats]
        mask = ~y.isna()
        X, y_clean = X.loc[mask], y.loc[mask]
        X_std = scl.fit_transform(imp.fit_transform(X))
        lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_std, y_clean)
        z = lr.coef_[0] / (np.sqrt((X_std**2).sum(axis=0)) + 1e-9)
        sig = [f for f, zv in zip(feats, z) if abs(zv) > 1.96]
        if len(sig) >= 3:
            X_sig = df[sig].loc[mask]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sig,
                y_clean,
                test_size=0.30,
                stratify=y_clean,
                random_state=42,
            )
            X_tr = scl.fit_transform(imp.fit_transform(X_tr))
            X_te = scl.transform(imp.transform(X_te))
            mlp = MLPClassifier(
                hidden_layer_sizes=(4,),
                activation="tanh",
                solver="lbfgs",
                max_iter=500,   # bumped from 200
                random_state=42,
            ).fit(X_tr, y_tr)
            proba = mlp.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.10).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
            res["Model_5_Optimized"] = dict(
                features=sig,
                auc=roc_auc_score(y_te, proba),
                sensitivity=tp / (tp + fn),
                specificity=tn / (tn + fp),
            )
    return res


def compare_with_xgboost(df_2020: pd.DataFrame, original: dict) -> dict:
    data = (
        df_2020[df_2020["employment"] == 1].copy()
        if EMPLOYMENT_FILTER and "employment" in df_2020.columns
        else df_2020.copy()
    )
    feats = [
        f
        for f in [
            "k6_score",
            "male",
            "age",
            "married",
            "lgbtq",
            "veteran",
            "drug_use",
            "mental_health_tx",
            "work_hours",
        ]
        if f in data.columns
    ]
    X = data[feats]
    y = data["suicide"]
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    pos_w = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    xgb = XGBClassifier(
        max_depth=3, n_estimators=150, scale_pos_weight=pos_w, random_state=42
    )
    xgb.fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)

    proba_tr = xgb.predict_proba(scl.transform(imp.transform(X_tr)))[:, 1]
    pr, rc, thr = precision_recall_curve(y_tr, proba_tr)
    f1 = (2 * pr * rc) / np.where(pr + rc == 0, 1, pr + rc)
    thr_best = thr[np.argmax(f1)]

    proba = xgb.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]
    pred = (proba >= thr_best).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    metrics = dict(
        auc=roc_auc_score(y_te, proba),
        sensitivity=tp / (tp + fn),
        specificity=tn / (tn + fp),
    )

    print("\n" + "─" * 52)
    print("Original MLP models vs XGBoost")
    print("─" * 52)
    print(f"{'Model':<22} {'AUC':>6} {'Sens':>7} {'Spec':>7}")
    for m, r in original.items():
        print(
            f"{m:<22} {r['auc']:>6.3f} {r['sensitivity']:>6.1%} {r['specificity']:>6.1%}"
        )
    print(
        f"{'XGBoost (ours)':<22} {metrics['auc']:>6.3f} "
        f"{metrics['sensitivity']:>6.1%} {metrics['specificity']:>6.1%}"
    )
    return metrics


# ======================================================================
# Bootstrap, fixed‑specificity comparison & fairness audit
# ======================================================================


def bootstrap_confidence_intervals(df_2020: pd.DataFrame, n_boot: int = 5000):
    """Bootstrap 95% CIs for the 2020 full XGBoost model."""
    print(f"\n🔄 Bootstrapping XGBoost ({n_boot:,} resamples)…")
    feats = [
        "k6_score", "male", "age", "married", "lgbtq",
        "veteran", "drug_use", "mental_health_tx", "work_hours"
    ]
    df = df_2020.copy()
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]

    X = df[feats]
    y = df["suicide"]
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    imp, scl = SimpleImputer(strategy="median"), StandardScaler()

    def _one_resample(seed: int):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X), len(X))
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_b, y_b, test_size=0.30, stratify=y_b, random_state=seed
        )
        mdl = XGBClassifier(
            max_depth=3,
            n_estimators=150,
            scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1),
            random_state=seed,
        )
        mdl.fit(
            scl.fit_transform(imp.fit_transform(X_tr)),
            y_tr,
        )
        proba_tr = mdl.predict_proba(scl.transform(imp.transform(X_tr)))[:, 1]
        pr, rc, thr = precision_recall_curve(y_tr, proba_tr)
        f1 = (2 * pr * rc) / np.where(pr + rc == 0, 1, pr + rc)
        thr_best = thr[np.argmax(f1)]

        proba = mdl.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]
        pred = (proba >= thr_best).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

        ppv = tp / (tp + fp) if tp + fp else 0
        npv = tn / (tn + fn) if tn + fn else 0

        return (roc_auc_score(y_te, proba),
                tp / (tp + fn) if tp + fn else 0,
                tn / (tn + fp) if tn + fp else 0,
                ppv, npv)

    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_one_resample)(s) for s in range(n_boot)
    )
    aucs, sens, specs, ppvs, npvs = map(np.array, zip(*results))

    ci = lambda arr: np.percentile(arr, [2.5, 97.5])

    print("✔️  Bootstrap complete")
    print(
        f"\n📊 95% CIs over {n_boot:,} resamples\n"
        f"  AUC : {np.mean(aucs):.3f} [{ci(aucs)[0]:.3f}, {ci(aucs)[1]:.3f}]\n"
        f"  Sens: {np.mean(sens):.1%} [{ci(sens)[0]:.1%}, {ci(sens)[1]:.1%}]\n"
        f"  Spec: {np.mean(specs):.1%} [{ci(specs)[0]:.1%}, {ci(specs)[1]:.1%}]\n"
        f"  PPV : {np.mean(ppvs):.1%} [{ci(ppvs)[0]:.1%}, {ci(ppvs)[1]:.1%}]\n"
        f"  NPV : {np.mean(npvs):.1%} [{ci(npvs)[0]:.1%}, {ci(npvs)[1]:.1%}]"
    )


# ---- MLP bootstrap CI function
def bootstrap_confidence_intervals_mlp(df_2020: pd.DataFrame, n_boot: int = 1000):
    """Bootstrap 95% CIs for the MLP-4 model."""
    print(f"\n🔄 Bootstrapping MLP-4 ({n_boot:,} resamples)…")

    feats = [
        "income", "work_hours", "male", "education", "age",
        "sick_days", "health_insurance", "drug_use", "criminal",
        "mental_health_tx", "veteran", "lgbtq", "married",
        "k6_score", "overall_health"
    ]

    df = df_2020.copy()
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]

    feats = [f for f in feats if f in df.columns]
    X = df[feats]
    y = df["suicide"]
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    imp, scl = SimpleImputer(strategy="median"), StandardScaler()

    def _one_resample(seed: int):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X), len(X))
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_b, y_b, test_size=0.30, stratify=y_b, random_state=seed
        )

        mlp = MLPClassifier(
            hidden_layer_sizes=(5,),
            activation="tanh",
            solver="lbfgs",
            max_iter=500,   # bumped from 300
            random_state=seed,
        )
        mlp.fit(
            scl.fit_transform(imp.fit_transform(X_tr)),
            y_tr,
        )

        proba = mlp.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]
        pred = (proba >= 0.10).astype(int)   # Fixed 0.10 threshold
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

        ppv = tp / (tp + fp) if tp + fp else 0
        npv = tn / (tn + fn) if tn + fn else 0

        return (roc_auc_score(y_te, proba),
                tp / (tp + fn) if tp + fn else 0,
                tn / (tn + fp) if tn + fp else 0,
                ppv, npv)

    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_one_resample)(s) for s in range(n_boot)
    )
    aucs, sens, specs, ppvs, npvs = map(np.array, zip(*results))

    ci = lambda arr: np.percentile(arr, [2.5, 97.5])

    print("✔️  MLP Bootstrap complete")
    print(
        f"\n📊 95% CIs over {n_boot:,} resamples\n"
        f"  AUC : {np.mean(aucs):.3f} [{ci(aucs)[0]:.3f}, {ci(aucs)[1]:.3f}]\n"
        f"  Sens: {np.mean(sens):.1%} [{ci(sens)[0]:.1%}, {ci(sens)[1]:.1%}]\n"
        f"  Spec: {np.mean(specs):.1%} [{ci(specs)[0]:.1%}, {ci(specs)[1]:.1%}]\n"
        f"  PPV : {np.mean(ppvs):.1%} [{ci(ppvs)[0]:.1%}, {ci(ppvs)[1]:.1%}]\n"
        f"  NPV : {np.mean(npvs):.1%} [{ci(npvs)[0]:.1%}, {ci(npvs)[1]:.1%}]"
    )


# >>> Calibration & threshold helpers
from sklearn.calibration import calibration_curve

def plot_calibration_curve(y_true, y_prob, title="Calibration – 2020 XGB"):
    """Reliability curve with 10 equal‑frequency bins."""
    if not SHOW_PLOTS:
        return
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], ls="--", lw=1)
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed prevalence")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def sweep_thresholds(y_true, y_prob, start=0.0, end=1.0, step=0.05):
    """Yield (thr, sens, spec) across a threshold grid."""
    for thr in np.arange(start, end + 1e-12, step):
        pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        yield thr, sens, spec


def compare_at_fixed_specificity(df_2020: pd.DataFrame, target_spec: float = 0.90):
    """Compare MLP‑Model‑4 vs XGBoost at a user‑defined specificity."""
    print(f"\n⚖️  Comparing models at {target_spec:.0%} specificity…")
    df = df_2020.copy()
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]

    base_feats = [
        "income", "work_hours", "male", "education", "age",
        "sick_days", "health_insurance", "drug_use", "criminal",
        "mental_health_tx", "veteran", "lgbtq", "married",
        "k6_score", "overall_health"
    ]
    feats = [f for f in base_feats if f in df.columns]
    X = df[feats]
    y = df["suicide"]
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    X_tr_p = scl.fit_transform(imp.fit_transform(X_tr))
    X_te_p = scl.transform(imp.transform(X_te))

    # ----- train XGB -----
    xgb = XGBClassifier(
        max_depth=3, n_estimators=150,
        scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1),
        random_state=42,
    ).fit(X_tr_p, y_tr)

    # ----- train MLP (Model‑4 style) -----
    mlp = MLPClassifier(
        hidden_layer_sizes=(5,),
        activation="tanh",
        solver="lbfgs",
        max_iter=500,   # bumped from 300
        random_state=42,
    ).fit(X_tr_p, y_tr)

    results = {}
    for name, model in {"XGBoost": xgb, "MLP‑4": mlp}.items():
        proba = model.predict_proba(X_te_p)[:, 1]
        # iterate over thresholds and choose the one whose specificity is closest
        best_thr = 0.50
        best_diff = 1.0
        for t in np.linspace(0.0, 1.0, 1001):
            pred_tmp = (proba >= t).astype(int)
            tn_tmp, fp_tmp, _, _ = confusion_matrix(y_te, pred_tmp, labels=[0, 1]).ravel()
            spec_tmp = tn_tmp / (tn_tmp + fp_tmp) if (tn_tmp + fp_tmp) else 0
            diff = abs(spec_tmp - target_spec)
            if diff < best_diff:
                best_diff = diff
                best_thr = t
        thr = best_thr
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if tn + fp else 0
        sens = tp / (tp + fn) if tp + fn else 0
        auc_v = roc_auc_score(y_te, proba)
        results[name] = (auc_v, sens, spec, thr)

    print(f"\n{'Model':<10}  AUC   Sens   Spec   Thr")
    for m, (a, s, p, t) in results.items():
        print(f"{m:<10} {a:5.3f} {s:6.1%} {p:6.1%} {t:5.2f}")


#
# >>> Wrappers for optional analyses
def run_calibration(df, save_preds=False):
    feats = [
        "k6_score", "male", "age", "married", "lgbtq",
        "veteran", "drug_use", "mental_health_tx", "work_hours"
    ]
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]
    X = df[feats]; y = df["suicide"]
    m = ~y.isna(); X, y = X[m], y[m]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    mdl = XGBClassifier(
        max_depth=3,
        n_estimators=150,
        scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
        random_state=42,
    )
    # 5‑fold Platt‑scaling (sigmoid) calibration for FULL Model
    clf = CalibratedClassifierCV(
        estimator=mdl,          # scikit‑learn ≥ 1.3
        method="sigmoid",
        cv=5
    )
    clf.fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)
    proba_te = clf.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]
    plot_calibration_curve(y_te, proba_te)
    if save_preds:
        out = pd.DataFrame({"y_true": y_te, "p_hat": proba_te})
        out.to_csv("holdout_predictions_2020.csv", index=False)
        print("✓ Saved hold‑out predictions to holdout_predictions_2020.csv")


# ---- Save both model predictions helper
def save_both_model_predictions(df_2020: pd.DataFrame):
    """Save predictions from both MLP-4 and XGBoost models."""
    print("\n💾 Saving predictions from both models...")

    df = df_2020.copy()
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]

    xgb_feats = [
        "k6_score", "male", "age", "married", "lgbtq",
        "veteran", "drug_use", "mental_health_tx", "work_hours"
    ]

    mlp_feats = [
        "income", "work_hours", "male", "education", "age",
        "sick_days", "health_insurance", "drug_use", "criminal",
        "mental_health_tx", "veteran", "lgbtq", "married",
        "k6_score", "overall_health"
    ]
    mlp_feats = [f for f in mlp_feats if f in df.columns]

    X_xgb = df[xgb_feats]
    X_mlp = df[mlp_feats]
    y = df["suicide"]

    mask_xgb = ~y.isna()
    # Evaluate XGBoost on the full hold‑out set; MLP keeps its own completeness filter internally
    common_mask = mask_xgb

    X_xgb = X_xgb.loc[common_mask]
    X_mlp = X_mlp.loc[common_mask]
    y = y.loc[common_mask]

    X_xgb_tr, X_xgb_te, X_mlp_tr, X_mlp_te, y_tr, y_te = train_test_split(
        X_xgb, X_mlp, y, test_size=0.30, stratify=y, random_state=SEED
    )

    imp, scl = SimpleImputer(strategy="median"), StandardScaler()

    # --- XGBoost with Platt‑scaling calibration -----------------------
    xgb_base = XGBClassifier(
        max_depth=3,
        n_estimators=150,
        scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
        random_state=SEED,
    )
    # 5‑fold Platt‑scaling (sigmoid) calibration
    xgb = CalibratedClassifierCV(
        estimator=xgb_base,
        method="sigmoid",
        cv=5,
    )
    xgb.fit(scl.fit_transform(imp.fit_transform(X_xgb_tr)), y_tr)
    proba_xgb = xgb.predict_proba(scl.transform(imp.transform(X_xgb_te)))[:, 1]

    mlp = MLPClassifier(
        hidden_layer_sizes=(5,),
        activation="tanh",
        solver="lbfgs",
        max_iter=500,   # bumped from 300
        random_state=SEED,
    )
    mlp.fit(scl.fit_transform(imp.fit_transform(X_mlp_tr)), y_tr)
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp, out_dir / "mlp_model.joblib")
    # Save the exact hold‑out feature matrix used for ROC / calibration exports
    X_mlp_te.to_csv(out_dir / "X_hold.csv")
    print(f"✓ Saved MLP model to {out_dir / 'mlp_model.joblib'}")
    proba_mlp = mlp.predict_proba(scl.transform(imp.transform(X_mlp_te)))[:, 1]

    # ---------------------------------------------------------------
    # Include demographic variables for fairness‑audit subgroups
    # Keep both the original binary flags *and* a human‑readable band for age
    demo_cols = [
        "male",                 # 1 = Male, 0 = Female
        "sex",                  # raw IRSEX value, if still present
        "age",                  # NSDUH categorical age code
        "veteran",
        "lgbtq",
        "income",
        "education",
        "married",
    ]
    # Keep only the columns that actually exist in the dataframe
    demo_cols = [c for c in demo_cols if c in df.columns]

    demo_df = df.loc[X_xgb_te.index, demo_cols].reset_index(drop=True)

    # Add a convenient three‑level age band (matches Table 3 buckets)
    if "age" in demo_df.columns:
        demo_df["age_band"] = pd.cut(
            demo_df["age"],
            bins=[0, 2, 5, 100],         # IRAGE codes: 1‑2=18‑25, 3‑5=26‑49, 6+=50+
            labels=["18‑25", "26‑49", "50+"],
            include_lowest=True,
        )

    out = pd.concat(
        [
            demo_df,  # demographic features first
            pd.DataFrame(
                {
                    "y_true": y_te.values,
                    "p_hat_xgb": proba_xgb,
                    "p_hat_mlp": proba_mlp,
                }
            ),
        ],
        axis=1,
    )
    out.to_csv(out_dir / "both_model_predictions_2020.csv", index=False)
    print(f"✓ Saved predictions to {out_dir / 'both_model_predictions_2020.csv'}")

    from sklearn.metrics import roc_curve
    fpr_xgb, tpr_xgb, _ = roc_curve(y_te, proba_xgb)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_te, proba_mlp)

    # Pad shorter arrays with NaNs so all columns have equal length
    max_len = max(len(fpr_xgb), len(fpr_mlp))
    pad = lambda arr: np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

    roc_df = pd.DataFrame({
        "fpr_xgb": pad(fpr_xgb),
        "tpr_xgb": pad(tpr_xgb),
        "fpr_mlp": pad(fpr_mlp),
        "tpr_mlp": pad(tpr_mlp),
    })
    roc_df.to_csv(out_dir / "roc_curves_data.csv", index=False)
    print(f"✓ Saved ROC curve data to {out_dir / 'roc_curves_data.csv'}")


def run_threshold_sweep(df, start, end, step):
    feats = [
        "k6_score", "male", "age", "married", "lgbtq",
        "veteran", "drug_use", "mental_health_tx", "work_hours"
    ]
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]
    X = df[feats]; y = df["suicide"]
    m = ~y.isna(); X, y = X[m], y[m]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    mdl = XGBClassifier(
        max_depth=3,
        n_estimators=150,
        scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
        random_state=42,
    ).fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)
    proba_te = mdl.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]

    print("\nThr │ Sens │ Spec")
    print("────┼──────┼──────")
    for thr, sn, sp in sweep_thresholds(y_te, proba_te, start, end, step):
        print(f"{thr:4.2f} {sn:6.1%} {sp:6.1%}")

# ------------------------------------------------------------------
# High-specificity operating-point helper
def high_spec_metrics(
    pred_path: str = str(OUTPUTS_DIR / "both_model_predictions_2020.csv"),
    spec_target: float = 0.93,
    search_lo: float = 0.05,
    search_hi: float = 0.90,
    step: float = 0.01,
):
    """
    Scan thresholds and report the one whose specificity is closest to
    `spec_target` on the 2020 hold-out predictions produced by
    `--save-both-preds` (expects *calibrated* p_hat_xgb).
    Results are printed and also written to high_spec_metrics.json.
    """
    import json as _json, numpy as _np, pandas as _pd
    from pathlib import Path
    from sklearn.metrics import confusion_matrix as _cm

    path = Path(pred_path)
    if not path.exists():
        print(f"❌  {path} not found – run the script once with '--save-both-preds'")
        return

    df = _pd.read_csv(path)
    if {"y_true", "p_hat_xgb"} - set(df.columns):
        print("❌  CSV missing y_true / p_hat_xgb columns"); return

    best, gap = None, 1.0
    for thr in _np.arange(search_lo, search_hi + 1e-12, step):
        y_pred = (df.p_hat_xgb >= thr).astype(int)
        tn, fp, fn, tp = _cm(df.y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0
        diff = abs(spec - spec_target)
        if diff < gap:
            best, gap = (thr, tp, fp, fn, tn, spec), diff

    if best is None:
        print("⚠️  No threshold matched"); return

    thr, tp, fp, fn, tn, spec = best
    sens = tp / (tp + fn) if (tp + fn) else 0
    ppv  = tp / (tp + fp) if (tp + fp) else 0
    npv  = tn / (tn + fn) if (tn + fn) else 0

    print("\n🎯  High-specificity operating point")
    print(f"Threshold {thr:.2f}  |  Sens {sens:.3%}  Spec {spec:.3%}  "
          f"PPV {ppv:.3%}  NPV {npv:.3%}  "
          f"(TP={tp}, FP={fp}, FN={fn}, TN={tn}, N={len(df)})")

    # ---- build a fully JSON‑serialisable dict ----
    raw_items = {
        "threshold": thr,
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
        "npv": npv,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(df),
    }
    out_dict = {
        k: (float(v) if isinstance(v, (np.floating, np.float64))
            else int(v) if isinstance(v, (np.integer, np.int64))
            else v)
        for k, v in raw_items.items()
    }
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "high_spec_metrics.json", "w") as fp:
        _json.dump(out_dict, fp, indent=2)
    print(f"✓  Saved → {out_dir/'high_spec_metrics.json'}")

# ------------------------------------------------------------------
# Fairness Table 3 helper (sex + age bands, calibrated XGB, thr 0.18)
def fairness_table3(
    pred_path: str = str(OUTPUTS_DIR / "both_model_predictions_2020.csv"),
    threshold: float = 0.18,
):
    """
    Compute the subgroup metrics used in manuscript Table 3
    (Female / Male and 18–25 / 26–49 / 50+ age bands) from the
    calibrated XGBoost predictions saved by --save-both-preds.

    Results are printed to screen and also saved to
    ``fairness_table3.csv`` for easy copy‑paste.
    """
    import pandas as _pd, numpy as _np
    from pathlib import Path
    from sklearn.metrics import roc_auc_score as _auc, confusion_matrix as _cm

    path = Path(pred_path)
    if not path.exists():
        print(f"❌  {path} not found – run the script once with '--save-both-preds'")
        return

    df = _pd.read_csv(path)
    req = {"y_true", "p_hat_xgb", "male", "age"}
    if req - set(df.columns):
        print(f"❌  Predictions CSV is missing {req - set(df.columns)}"); return

    df["xgb_pred"] = (df["p_hat_xgb"] >= threshold).astype(int)

    def _metrics(g):
        if len(g) == 0:
            return _pd.Series({"n": 0,
                               "Prevalence %": _np.nan,
                               "AUC": _np.nan,
                               "Sens %": _np.nan,
                               "Spec %": _np.nan})
        y    = g["y_true"]
        p    = g["p_hat_xgb"]
        yhat = g["xgb_pred"]
        # ensure we always get a 2×2 matrix even if a group has only one class
        tn, fp, fn, tp = _cm(y, yhat, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else _np.nan
        spec = tn / (tn + fp) if (tn + fp) else _np.nan
        return _pd.Series({
            "n": len(g),
            "Prevalence %": 100 * y.mean(),
            "AUC": _auc(y, p),
            "Sens %": 100 * sens,
            "Spec %": 100 * spec,
        })

    # Sex sub‑groups (0 = Female, 1 = Male)
    grouper_sex = df.groupby(df["male"].map({0: "Female", 1: "Male"}))
    try:
        sex_tbl = grouper_sex.apply(_metrics, include_groups=False)
    except TypeError:
        # pandas < 2.2 does not support include_groups
        sex_tbl = grouper_sex.apply(_metrics)

    # Age bands: CATAGE codes → 1 = 18‑25, 2 = 18‑25, 3 = 26‑49, 4 = 26‑49, 5 = 50+, 6 = 50+
    age_map = {
        1: "18–25",
        2: "18–25",   # code 2 is also in the 18–25 bracket
        3: "26–49",
        4: "26–49",
        5: "50+",
        6: "50+",
    }
    df["age_band"] = df["age"].map(age_map)
    grouper_age = df.groupby(df["age_band"])
    try:
        age_tbl = grouper_age.apply(_metrics, include_groups=False)
    except TypeError:
        age_tbl = grouper_age.apply(_metrics)

    # guarantee that all three age bands are present
    for band in ["18–25", "26–49", "50+"]:
        if band not in age_tbl.index:
            age_tbl.loc[band] = {
                "n": 0,
                "Prevalence %": _np.nan,
                "AUC": _np.nan,
                "Sens %": _np.nan,
                "Spec %": _np.nan,
            }
    age_tbl = age_tbl.loc[["18–25", "26–49", "50+"], :]

    tbl = _pd.concat([sex_tbl, age_tbl]).round(1)

    print("\n📋  Fairness metrics (threshold {:.2f})".format(threshold))
    print(tbl.to_string())

    tbl.to_csv(OUTPUTS_DIR / "fairness_table3.csv")
    print(f"✓  Saved → {OUTPUTS_DIR / 'fairness_table3.csv'}")

# ------------------------------------------------------------------
# NRI / IDI helper (continuous re-classification metrics)
def compute_nri_idi(
    pred_path: str = str(OUTPUTS_DIR / "both_model_predictions_2020.csv"),
    out_dir: pathlib.Path = OUTPUTS_DIR / "appendix_stats",
):
    """
    Compute continuous Net Re-classification Improvement (NRI) and
    Integrated Discrimination Improvement (IDI) comparing calibrated
    XGBoost (new) vs MLP-4 (reference) on the 2020 hold-out set.

    Requires the CSV produced by --save-both-preds.
    Saves results to outputs/appendix_stats/nri_idi.csv
    """
    import pandas as pd, numpy as np
    from pathlib import Path
    from sklearn.metrics import roc_auc_score

    path = Path(pred_path)
    if not path.exists():
        print(f"❌  {path} not found – run once with '--save-both-preds'")
        return

    df = pd.read_csv(path)
    req = {"y_true", "p_hat_xgb", "p_hat_mlp"}
    if req - set(df.columns):
        print(f"❌  Predictions CSV missing columns: {req - set(df.columns)}")
        return

    y   = df.y_true.values
    p_x = df.p_hat_xgb.values      # new model
    p_m = df.p_hat_mlp.values      # reference model

    # --- continuous NRI (Pencina et al., 2008) -----------------
    up_new   = (p_x >  p_m) & (y == 1)
    down_new = (p_x <  p_m) & (y == 1)
    up_old   = (p_x <  p_m) & (y == 0)
    down_old = (p_x >  p_m) & (y == 0)
    nri = ((up_new.sum()  - down_new.sum()) / max((y == 1).sum(), 1) +
           (down_old.sum() - up_old.sum()) / max((y == 0).sum(), 1))

    # --- IDI ----------------------------------------------------
    idi = ((p_x[y == 1].mean() - p_x[y == 0].mean()) -
           (p_m[y == 1].mean() - p_m[y == 0].mean()))

    out_df = pd.DataFrame(
        {"metric": ["AUC_new", "AUC_old", "NRI", "IDI"],
         "value" : [roc_auc_score(y, p_x),
                    roc_auc_score(y, p_m),
                    nri, idi]}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nri_idi.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✓ NRI/IDI metrics saved → {out_path}")


# Extra evaluation helpers (prospective, calibration‑stats, SHAP, decision‑curve)
def prospective_holdout(data_dict: dict[int, pd.DataFrame], first_test_year: int = 2020):
    """Train on years < first_test_year, test on first_test_year+ (basic feature set)."""
    print(f"\n📅 Prospective evaluation (train ≤ {first_test_year-1}, test ≥ {first_test_year})")
    train_yrs = [y for y in data_dict if y < first_test_year]
    test_yrs  = [y for y in data_dict if y >= first_test_year]
    if not train_yrs or not test_yrs:
        print("⚠️  Need both pre‑ and post‑ years for prospective split"); return
    feats = ["male", "age", "veteran", "drug_use", "work_hours"]
    train_df = pd.concat([data_dict[y] for y in train_yrs], ignore_index=True)
    test_df  = pd.concat([data_dict[y] for y in test_yrs ], ignore_index=True)
    if EMPLOYMENT_FILTER and "employment" in train_df.columns:
        train_df = train_df[train_df["employment"] == 1]
        test_df  = test_df [test_df ["employment"] == 1]
    X_tr, y_tr = train_df[feats], train_df["suicide"]
    X_te, y_te = test_df [feats], test_df ["suicide"]
    m1, m2 = ~y_tr.isna(), ~y_te.isna()
    X_tr, y_tr = X_tr.loc[m1], y_tr.loc[m1]
    X_te, y_te = X_te.loc[m2], y_te.loc[m2]
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    mdl = XGBClassifier(max_depth=3, n_estimators=150,
                        scale_pos_weight=(len(y_tr)-y_tr.sum())/max(y_tr.sum(),1),
                        random_state=42)
    mdl.fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)
    proba = mdl.predict_proba(scl.transform(imp.transform(X_te)))[:,1]
    auc_v = roc_auc_score(y_te, proba)
    print(f"Prospective AUC = {auc_v:.3f}  (N={len(y_te):,})")

def calibration_stats(y_true, y_prob):
    """Return Brier score, calibration‑in‑the‑large (intercept) and slope."""
    eps = 1e-12
    bs = brier_score_loss(y_true, y_prob)
    lr = LogisticRegression(
        fit_intercept=True,
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        n_jobs=1,
        random_state=42,
    ).fit(np.log((y_prob + eps) / (1 - y_prob + eps)).reshape(-1, 1), y_true)
    return bs, lr.intercept_[0], lr.coef_[0][0]

def run_calibration_stats(df):
    feats = ["k6_score","male","age","married","lgbtq",
             "veteran","drug_use","mental_health_tx","work_hours"]
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]
    X, y = df[feats], df["suicide"]
    m = ~y.isna(); X, y = X[m], y[m]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    xgb_base = XGBClassifier(
        max_depth=3,
        n_estimators=150,
        scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
        random_state=42,
    )
    mdl = CalibratedClassifierCV(
        estimator=xgb_base,
        method="sigmoid",
        cv=5,
    ).fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)
    proba = mdl.predict_proba(scl.transform(imp.transform(X_te)))[:,1]
    bs, cint, cslope = calibration_stats(y_te, proba)
    print(f"\n📏 Calibration statistics\n  • Brier score      : {bs:.3f}"
          f"\n  • Calibration int. : {cint:+.3f}\n  • Calibration slope: {cslope:.3f}")

def shap_summary(df):
    if shap is None:
        print("⚠️  shap library not installed – skip"); return
    feats = ["k6_score","male","age","married","lgbtq",
             "veteran","drug_use","mental_health_tx","work_hours"]
    d = df[df["employment"] == 1] if EMPLOYMENT_FILTER and "employment" in df.columns else df
    X, y = d[feats], d["suicide"]
    m = ~y.isna(); X, y = X[m], y[m]
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    mdl = XGBClassifier(max_depth=3, n_estimators=150,
                        scale_pos_weight=(len(y)-y.sum())/y.sum(),
                        random_state=42).fit(
                            scl.fit_transform(imp.fit_transform(X)), y)
    explainer = shap.Explainer(mdl, scl.transform(imp.transform(X)))
    sv = explainer(scl.transform(imp.transform(X)))
    shap.summary_plot(sv, X, show=SHOW_PLOTS)

def decision_curve(y_true, y_prob, label="XGB"):
    thresh = np.linspace(0.01,0.99,99)
    net = []
    for t in thresh:
        pred = (y_prob>=t).astype(int)
        tp = ((pred==1)&(y_true==1)).sum()
        fp = ((pred==1)&(y_true==0)).sum()
        net.append(tp/len(y_true) - fp/len(y_true)*(t/(1-t)))
    if SHOW_PLOTS:
        plt.figure()
        plt.plot(thresh, net)
        plt.xlabel("Threshold"); plt.ylabel("Net benefit")
        plt.title(f"Decision curve – {label}")
        plt.tight_layout(); plt.show()


def fairness_audit(df_all: dict):
    """Report subgroup AUC / Sens / Spec for the basic cross‑year model."""
    print("\n🌐 Fairness audit (basic model features)")
    feats = ["male", "age", "veteran", "drug_use", "work_hours"]

    if 2020 not in df_all:
        print("⚠️  Need 2020 data for audit."); return

    train_df = pd.concat([df for yr, df in df_all.items() if yr != 2020], ignore_index=True)
    test_df = df_all[2020]

    if EMPLOYMENT_FILTER and "employment" in train_df.columns:
        train_df = train_df[train_df["employment"] == 1]
        test_df = test_df[test_df["employment"] == 1]

    X_tr, y_tr = train_df[feats], train_df["suicide"]
    X_te, y_te = test_df[feats], test_df["suicide"]

    m1 = ~y_tr.isna()
    X_tr = X_tr.loc[m1]
    y_tr = y_tr.loc[m1]
    m2 = ~y_te.isna()
    X_te = X_te.loc[m2]
    y_te = y_te.loc[m2]

    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    X_tr_p = scl.fit_transform(imp.fit_transform(X_tr))
    X_te_p = scl.transform(imp.transform(X_te))

    pos_w = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    mdl = XGBClassifier(max_depth=3, n_estimators=150, scale_pos_weight=pos_w, random_state=42)
    mdl.fit(X_tr_p, y_tr)
    proba_all = mdl.predict_proba(X_te_p)[:, 1]

    subgroup_defs = {
        "Female": ("male", 0),
        "Male": ("male", 1),
        "Veteran": ("veteran", 1),
        "Civilians": ("veteran", 0),
        "LGBTQ": ("lgbtq", 1),
        "Non‑LGBTQ": ("lgbtq", 0),
        "Age18‑25": ("age", lambda s: s <= 2),
        "Age26‑49": ("age", lambda s: (s >= 3) & (s <= 5)),
        "Age50+": ("age", lambda s: s >= 6),
        "LowInc": ("income", lambda s: s <= s.quantile(0.25)),
        "HighInc": ("income", lambda s: s >= s.quantile(0.75)),
    }

    print(f"\n{'Group':<12}  AUC   Sens   Spec    N")
    for label, (col, val) in subgroup_defs.items():
        # build mask in the same index space as y_te / proba_all
        col_data = test_df.loc[y_te.index, col]
        idx = val(col_data) if callable(val) else (col_data == val)
        if idx.sum() < 20:  # skip tiny groups
            continue
        y_g = y_te[idx]
        p_g = proba_all[idx]
        pred_g = (p_g >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_g, pred_g, labels=[0, 1]).ravel()
        auc_v = roc_auc_score(y_g, p_g)
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        print(f"{label:<12} {auc_v:5.3f} {sens:6.1%} {spec:6.1%} {len(y_g):5d}")



# ======================================================================
# Appendix A – classical statistical checks (optional)
# ======================================================================

def run_appendix_stats(df: pd.DataFrame):
    """
    Supplementary robustness analyses on a multi-year dataframe
        • survey-weighted logistic regression (+ key interactions)
        • elastic-net logistic (cross-validated)
        • Bayesian hierarchical logistic regression (random intercept = year)
        • GAM with pairwise spline interactions
        • dominance / relative-importance analysis

    Results are written to  outputs/appendix_stats/
    """
    # --- Lazy import of optional appendix dependencies -------------------
    try:
        import patsy
        import statsmodels.api as sm
        from sklearn.linear_model import LogisticRegressionCV
        from pygam import LogisticGAM, s, te
        from dominance_analysis import Dominance
    except Exception as e:
        # Any missing or incompatible library aborts the appendix gracefully
        print("⚠️  Skipping appendix‑stats – required libraries missing or incompatible:", e)
        return

    out_dir = OUTPUTS_DIR / "appendix_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    # cached summary path (used below)
    path_summary = out_dir / "bayes_hier_logit_summary.csv"


    # keep employed, non‑missing outcome
    df = df.copy()
    if EMPLOYMENT_FILTER and "employment" in df.columns:
        df = df[df["employment"] == 1]
    df = df.loc[~df["suicide"].isna()].copy()

    # model formula with two theory‑guided interactions
    formula = ("suicide ~ male + age + veteran + drug_use + work_hours + "
               "k6_score + married + lgbtq + mental_health_tx + "
               "drug_use:mental_health_tx + age:work_hours")

    y, X = patsy.dmatrices(formula, df, return_type="dataframe")

    # -------------------------------------------------- Survey‑weighted logit
    if {"VESTR", "VEREP", "ANALWT_C"} <= set(df.columns):
        design = sm.survey.SurveyDesign(
            strata=df["VESTR"], cluster=df["VEREP"], weights=df["ANALWT_C"]
        )
        swl = sm.survey.SurveyModel(design, X, y.squeeze()).fit()
        swl.summary().tables[1].to_csv(out_dir / "sw_logistic.csv")
        print("✓ Survey‑weighted logistic saved")
    else:
        print("⚠️  Survey design variables absent – skipping weighted logit")

    # ------------------------------------------- Elastic‑net (cross‑val logit)
    if LogisticRegressionCV is not None:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        imp, scl = SimpleImputer(strategy="median"), StandardScaler()
        X_std = scl.fit_transform(imp.fit_transform(X))
        enet = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.5],
            cv=5,
            scoring="roc_auc",
            max_iter=10000,
            random_state=SEED,
        ).fit(X_std, y.values.ravel())
        pd.Series(enet.coef_[0], index=X.columns).to_csv(
            out_dir / "elastic_net_coefs.csv"
        )
        print("✓ Elastic‑net coefficients saved")

        # ------------------------------------------------- Bayesian hierarchical logit
        try:
            import bambi as bmb
            import arviz as az
        except Exception as e:
            print("⚠️  Skipping Bayesian hierarchical logit – bambi/arviz missing or incompatible:", e)
        else:
            if "year" in df.columns and df["year"].nunique() > 1:
                # ----------------------------------------------------------------
                # 1) Respect cache unless the user set --force-bayes
                # 2) Use user‑adjustable draws / tune counts
                # ----------------------------------------------------------------
                if path_summary.exists() and not FORCE_BAYES:
                    print("✓ Bayesian hierarchical summary already present – skipping re‑fit "
                          "(use --force-bayes to recompute)")
                else:
                    # ----- Bayesian hierarchical logistic (complete‑case only) -----
                    df_bayes = df.dropna().copy()
                    if len(df_bayes) < 1_000:
                        print("⚠️  Skipping Bayesian hierarchical logit – fewer than "
                              f"1 000 complete cases after dropping missing values "
                              f"({len(df_bayes):,} remaining).")
                    else:
                        bayes_formula = formula + " + (1|year)"
                        print(f"⏳ Fitting Bayesian hierarchical logistic on "
                              f"{len(df_bayes):,} complete cases…")
                        model = bmb.Model(
                            bayes_formula,
                            df_bayes,
                            family="bernoulli",
                            priors={"Intercept": bmb.Prior('Normal', mu=0, sigma=5)},
                        )
                        idata = model.fit(
                            draws=BAYES_DRAWS,
                            tune=BAYES_TUNE,
                            chains=4,
                            cores=1,
                            random_seed=SEED,
                        )
                        # save both the readable summary *and* the full posterior
                        az.summary(idata).to_csv(path_summary)
                        idata.to_netcdf(out_dir / "bayes_hier_logit_trace.nc")
                        print("✓ Bayesian hierarchical summary & trace saved")

    # ---------------------------------------------------------------  GAM
    if LogisticGAM is not None:
        try:
            gam = LogisticGAM(
                s(0) + s(1) + s(2) + s(3) + s(4) + s(5) +
                te(3, 5) + te(1, 4)
            ).fit(X.values, y.values.ravel())
            # Only save if the coefficient vector length matches the feature list
            if len(gam.coef_) == len(X.columns):
                pd.DataFrame(
                    {"feature": X.columns, "beta": gam.coef_}
                ).to_csv(out_dir / "gam_coefs.csv", index=False)
                print("✓ GAM coefficients saved")
            else:
                print(
                    "⚠️  Skipping GAM – coef length mismatch "
                    f"({len(gam.coef_)} vs {len(X.columns)})"
                )
        except Exception as e:
            print("⚠️  Skipping GAM – error during fitting:", e)

    # ------------------------------------------------- Dominance analysis
    if Dominance is not None:
        dom = Dominance(data=pd.concat([y, X], axis=1), target="suicide")
        inc_r2 = dom.incremental_rsquare()

        # ── clean up stray *.txt that the library drops in CWD ─────────
        import pathlib as _pl, shutil
        txt_dump = _pl.Path("dominance_r2.txt")
        if txt_dump.exists():
            shutil.move(txt_dump, out_dir / "dominance_r2.txt")

        # Handle both return‑types
        if isinstance(inc_r2, dict):
            (pd.DataFrame.from_dict(inc_r2,
                                    orient="index",
                                    columns=["incremental_r2"])
             .to_csv(out_dir / "dominance_r2.csv"))
        else:
            inc_r2.to_csv(out_dir / "dominance_r2.csv")

        print("✓ Dominance R² table saved")

    print("📁 Appendix‑stats artefacts written to", out_dir)

# ======================================================================
# Diagnostic helpers
# ======================================================================


def investigate_sample_discrepancy(df_2020: pd.DataFrame):
    print("\n" + "=" * 60 + "\nSAMPLE SIZE INVESTIGATION\n" + "=" * 60)
    df = (
        df_2020[df_2020["employment"] == 1].copy()
        if EMPLOYMENT_FILTER and "employment" in df_2020.columns
        else df_2020.copy()
    )
    print(f"After employment filter: {len(df):,}")
    df = df[~df["suicide"].isna()]
    print(f"After dropping missing target: {len(df):,}")

    mdl_vars = {
        "Model_1": ["income", "work_hours", "male", "education", "age"],
        "Model_2": ["income", "work_hours", "male", "education", "age", "sick_days", "health_insurance"],
        "Model_3": [
            "income",
            "work_hours",
            "male",
            "education",
            "age",
            "sick_days",
            "health_insurance",
            "drug_use",
            "criminal",
            "mental_health_tx",
            "veteran",
            "lgbtq",
            "married",
        ],
        "Model_4": [
            "income",
            "work_hours",
            "male",
            "education",
            "age",
            "sick_days",
            "health_insurance",
            "drug_use",
            "criminal",
            "mental_health_tx",
            "veteran",
            "lgbtq",
            "married",
            "k6_score",
            "overall_health",
        ],
    }

    for nm, vars_ in mdl_vars.items():
        avail = [v for v in vars_ if v in df.columns]
        comp = df[avail + ["suicide"]].dropna()
        print(f"{nm}: complete cases = {len(comp):,}")
        if len(avail) < len(vars_):
            print(f"   missing vars: {[v for v in vars_ if v not in avail]}")


def investigate_variable_definitions(df_2020: pd.DataFrame):
    key = [
        "suicide",
        "k6_score",
        "male",
        "age",
        "drug_use",
        "mental_health_tx",
        "veteran",
        "lgbtq",
        "married",
    ]
    for v in key:
        if v not in df_2020.columns:
            continue
        col = df_2020[v]
        print(f"\n{v.upper()} – missing {col.isna().mean():.1%}")
        print(col.value_counts(dropna=False).head(10))


def investigate_threshold_sensitivity(df_2020: pd.DataFrame, target=0.184):
    feats = [
        "income",
        "work_hours",
        "male",
        "education",
        "age",
        "sick_days",
        "health_insurance",
        "drug_use",
        "criminal",
        "mental_health_tx",
        "veteran",
        "lgbtq",
        "married",
        "k6_score",
        "overall_health",
    ]
    df = (
        df_2020[df_2020["employment"] == 1].copy()
        if EMPLOYMENT_FILTER and "employment" in df_2020.columns
        else df_2020.copy()
    )
    X = df[[f for f in feats if f in df.columns]]
    y = df["suicide"]
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    imp, scl = SimpleImputer(strategy="median"), StandardScaler()
    X_tr = scl.fit_transform(imp.fit_transform(X_tr))
    X_te = scl.transform(imp.transform(X_te))

    mlp = MLPClassifier(
        hidden_layer_sizes=(5,), activation="tanh", solver="lbfgs", max_iter=500, random_state=42
    ).fit(X_tr, y_tr)
    proba = mlp.predict_proba(X_te)[:, 1]

    print("\nThreshold │ Sens │ Spec")
    print("──────────┼──────┼──────")
    for thr in np.linspace(0.01, 0.99, 10):
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        flag = " ← target" if abs(sens - target) < 0.03 else ""
        print(f"  {thr:4.2f} │ {sens:4.1%} │ {spec:4.1%}{flag}")


def run_full_investigation(df_2020: pd.DataFrame):
    investigate_sample_discrepancy(df_2020)
    investigate_variable_definitions(df_2020)
    investigate_threshold_sensitivity(df_2020)


# ======================================================================
# Pretty print final results
# ======================================================================


def print_temporal_results(res: dict):
    line = "=" * 60
    print(f"\n{line}\nTEMPORAL STABILITY RESULTS\n{line}")

    if res.get("2020_full"):
        r = res["2020_full"]
        print(
            f"\nFULL MODEL (2020 only, includes K6)\n"
            f"  AUC = {r['auc']:.3f} | Sens = {r['sensitivity']:.1%} | "
            f"Spec = {r['specificity']:.1%} | "
            f"N = {r['n_total']:,}, Pos = {r['n_positive']:,}"
        )

    basic = res.get("basic", {})
    if basic:
        trains = sorted(basic)
        tests = sorted({t for yr in basic for t in basic[yr]})
        print("\nBASIC MODEL – AUC (train → test)")
        print("Train\\Test".ljust(12) + "".join(f"{t:>8}" for t in tests))
        for tr in trains:
            row = f"{tr:>12}"
            for te in tests:
                cell = basic[tr].get(te)
                row += ("  *" if tr == te else "   ") + (
                    f"{cell['auc']:.3f}" if cell else "-- "
                )
            print(row)

        diag = [basic[y][y]["auc"] for y in trains if y in basic[y]]
        offd = [
            c["auc"]
            for tr in trains
            for te, c in basic[tr].items()
            if te != tr
        ]
        if diag:
            print(
                f"\nMean same‑year AUC   : {np.mean(diag):.3f} ± {np.std(diag):.3f}"
            )
        if offd:
            print(
                f"Mean cross‑year AUC : {np.mean(offd):.3f} ± {np.std(offd):.3f}"
            )

    intra = res.get("within_year", {})
    if intra:
        print("\nSAME‑YEAR OVER‑FIT (baseline features)")
        for yr in sorted(intra):
            r = intra[yr]
            if not r:
                continue
            print(
                f"  {yr}: AUC={r['auc']:.3f} | "
                f"Sens={r['sensitivity']:.1%} | "
                f"Spec={r['specificity']:.1%} | "
                f"Pos = {r['n_positive']}/{r['n_total']}"
            )


# ======================================================================
# Main entry
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi‑year NSDUH analysis")
    parser.add_argument(
        "--appendix-stats",
        action="store_true",
        help="Run supplementary classical statistical analyses (Appendix A)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023],
        help="Years to analyse",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("./data"),
        help="Directory for downloaded data",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download missing archives"
    )
    parser.add_argument(
        "--investigate",
        action="store_true",
        help="Run diagnostic investigation on 2020 sample",
    )
    parser.add_argument(
        "--no-employment-filter",
        action="store_true",
        help="Skip the ‘currently‑employed only’ filter (default = applied)",
    )
    parser.add_argument("--bootstrap", type=int,
                        help="Number of bootstrap resamples for CI estimation")
    parser.add_argument("--compare-spec", type=float,
                        help="Fix specificity and compare sensitivities (e.g., 0.90)")
    parser.add_argument("--fairness-audit", action="store_true",
                        help="Run subgroup fairness audit")
    parser.add_argument("--calibration", action="store_true",
                        help="Draw calibration curve for 2020 full model")
    parser.add_argument("--sweep-thresholds", nargs=3, type=float,
                        metavar=("START", "END", "STEP"),
                        help="Print Sens/Spec over a threshold grid")
    parser.add_argument("--save-preds", action="store_true",
                        help="Save 2020 hold‑out predictions to CSV")
    parser.add_argument("--no-plots", action="store_true",
                        help="Suppress all matplotlib pop‑ups (sets headless backend)")
    # NEW – reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data splits and model init")
    parser.add_argument("--prospective", action="store_true",
                        help="Train on ≤ (min test‑year‑1) and evaluate on later years")
    parser.add_argument("--calibration-stats", action="store_true",
                        help="Print numeric calibration statistics for 2020 model")
    parser.add_argument("--shap", action="store_true",
                        help="Show SHAP global importance for 2020 model (requires shap)")
    parser.add_argument("--decision-curve", action="store_true",
                        help="Plot decision‑curve for 2020 model")
    parser.add_argument("--bootstrap-mlp", type=int,
                        help="Number of bootstrap resamples for MLP CI estimation")
    parser.add_argument("--save-both-preds", action="store_true",
                        help="Save predictions from both MLP and XGBoost models")
    parser.add_argument("--nri-idi", action="store_true",
                        help="Compute continuous NRI and IDI metrics (needs --save-both-preds run first)")
    # ---- new Bayesian control switches ---------------------------------
    parser.add_argument("--bayes-draws", type=int, default=2000,
                        help="Posterior draws for Bayesian hierarchical logistic (Appendix A)")
    parser.add_argument("--bayes-tune", type=int, default=1000,
                        help="Tuning steps for Bayesian hierarchical logistic (Appendix A)")
    parser.add_argument("--force-bayes", action="store_true",
                        help="Force re-fit of Bayesian hierarchical logistic even if cached")
    parser.add_argument(
        "--table3",
        action="store_true",
        help="Compute Table 3 fairness metrics from saved predictions",
    )
    parser.add_argument(
        "--high-spec", nargs="?", const=0.93, type=float, metavar="SPEC",
        help="After --save-both-preds, scan thresholds and report the "
             "operating point whose specificity is closest to SPEC "
             "(default 0.93 = 93 %).",
    )

    parser.add_argument(
        '--export-roc',
        action='store_true',
        help='Generate ROC overlay (PNG + JSON) for XGB and MLP on the 2020 hold‑out set.'
    )
    parser.add_argument(
        '--export-calibration-mlp',
        action='store_true',
        help='Compute and save calibration statistics (Brier, intercept, slope) for MLP on the 2020 hold‑out set.'
    )

        # Escape bare '%' in help texts *after* all arguments are registered.
    # Keep argparse placeholders like %(default)s intact.
    for _act in parser._actions:
        if getattr(_act, "help", None):
            _act.help = re.sub(r'%(?!\()', '%%', _act.help)

    # Parse args and prepare output directory
    args = parser.parse_args()
    args.data_dir.mkdir(exist_ok=True)

        # Update module-level configuration without 'global' (avoids 3.12 annotated-name issue)
    _g = globals()
    _g["SEED"] = args.seed
    _g["BAYES_DRAWS"] = args.bayes_draws
    _g["BAYES_TUNE"] = args.bayes_tune
    _g["FORCE_BAYES"] = args.force_bayes

        # Toggle employment filter
    _g["EMPLOYMENT_FILTER"] = not args.no_employment_filter

    # handle plot suppression
    global SHOW_PLOTS
    SHOW_PLOTS = not args.no_plots
    if not SHOW_PLOTS:
        import matplotlib
        matplotlib.use("Agg")

    # ------------------------------------------------------------------
    # Download / extract (if requested)
    if args.download:
        for yr in args.years:
            z = download_year(yr, args.data_dir)
            if z:
                extract_tab_file(z, args.data_dir)

    # ------------------------------------------------------------------
    # Validate & align variables
    raw: dict[int, pd.DataFrame] = {}
    found: dict[int, dict] = {}
    for yr in args.years:
        tp = args.data_dir / f"NSDUH_{yr}.tab"
        if not tp.exists():
            # ── Auto‑download and extract if the .tab file is absent ──
            print(f"↪️  {yr} .tab file not found — fetching archive…")
            zip_path = download_year(yr, args.data_dir)
            if zip_path:
                tp = extract_tab_file(zip_path, args.data_dir)
            if not tp.exists():
                print(f"❌ Unable to obtain data for {yr} – skipping this year.")
                continue
        df = pd.read_csv(tp, sep="\t", low_memory=False)
        raw[yr] = df
        found[yr] = validate_year_data(df, yr)

    mapping = align_variables_across_years(found)

    # --- Save the final variable mapping so it’s easy to inspect ---
    try:
        (args.data_dir / "variable_mapping.json").write_text(
            json.dumps(mapping, indent=2)
        )
    except Exception as e:
        print(f"⚠️  Could not write variable_mapping.json: {e}")

    # ------------------------------------------------------------------
    # Load cleaned frames
    clean: dict[int, pd.DataFrame] = {
        yr: load_and_clean_year(yr, args.data_dir, mapping) for yr in raw
    }

    # ------------------------------------------------------------------
    # Main temporal analysis
    results = run_temporal_analysis(clean)
    print_temporal_results(results)

    # ------------------------------------------------------------------
    # Replicate original paper & compare
    if 2020 in clean:
        mlp_res = replicate_original_models(clean[2020])
        xgb_res = compare_with_xgboost(clean[2020], mlp_res)
        with open(args.data_dir / "method_comparison.json", "w") as fh:
            json.dump({"mlp": mlp_res, "xgb": xgb_res}, fh, indent=2)

    # ------------------------------------------------------------------
    # Optional new analyses
    if args.bootstrap and 2020 in clean:
        bootstrap_confidence_intervals(clean[2020], args.bootstrap)

    if args.bootstrap_mlp and 2020 in clean:
        bootstrap_confidence_intervals_mlp(clean[2020], args.bootstrap_mlp)

    if args.save_both_preds and 2020 in clean:
        save_both_model_predictions(clean[2020])

    if args.nri_idi:
        compute_nri_idi()

    if args.compare_spec and 2020 in clean:
        compare_at_fixed_specificity(clean[2020], args.compare_spec)

    if args.fairness_audit:
        fairness_audit(clean)

    if args.calibration and 2020 in clean:
        run_calibration(clean[2020], save_preds=args.save_preds)

    if args.sweep_thresholds and 2020 in clean:
        start, end, step = args.sweep_thresholds
        run_threshold_sweep(clean[2020], start, end, step)
    
    if args.high_spec is not None:
        high_spec_metrics(spec_target=args.high_spec)

    if args.table3:
        fairness_table3()

    # Prospective hold‑out across years
    if args.prospective:
        prospective_holdout(clean, first_test_year=min(y for y in args.years if y>=2020))

    # Calibration‑stats
    if args.calibration_stats and 2020 in clean:
        run_calibration_stats(clean[2020])

    # SHAP summary
    if args.shap and 2020 in clean:
        shap_summary(clean[2020])

    # Decision‑curve
    if args.decision_curve and 2020 in clean:
        feats = ["k6_score","male","age","married","lgbtq",
                 "veteran","drug_use","mental_health_tx","work_hours"]
        df = clean[2020]
        if EMPLOYMENT_FILTER and "employment" in df.columns:
            df = df[df["employment"] == 1]
        X, y = df[feats], df["suicide"]
        m = ~y.isna(); X, y = X[m], y[m]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        imp, scl = SimpleImputer(strategy="median"), StandardScaler()
        mdl = XGBClassifier(
            max_depth=3,
            n_estimators=150,
            scale_pos_weight=(len(y_tr) - y_tr.sum()) / y_tr.sum(),
            random_state=42,
        ).fit(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)
        proba_te = mdl.predict_proba(scl.transform(imp.transform(X_te)))[:, 1]
        decision_curve(y_te, proba_te)

    # Appendix classical statistics
    if args.appendix_stats:
        combined_df = pd.concat(clean.values(), ignore_index=True)
        run_appendix_stats(combined_df)

    # ------------------------------------------------------------------
    # Optional diagnostics
    if args.investigate and 2020 in clean:
        run_full_investigation(clean[2020])

    # save temporal results
    with open(args.data_dir / "temporal_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n📊 Results saved to {args.data_dir / 'temporal_results.json'}")

    # ------------------------------------------------------------------
    # Optional artefact exports requested via CLI
    # ------------------------------------------------------------------
    if args.export_roc:
        export_roc_overlay()

    if args.export_calibration_mlp:
        export_calibration_mlp()


#######################################################################
# ----------------------------------------------------------------------
# Extra utilities for manuscript artefacts (ROC overlay & calibration)
# ----------------------------------------------------------------------
def export_roc_overlay():
    """
    Generates an ROC overlay for XGB and MLP (9‑predictor versions) on the
    2020 hold‑out set, then writes:
        • roc_overlay_2020.png
        • roc_coords_2020.json
    Uses the already‑saved predictions in both_model_predictions_2020.csv.
    """
    import pandas as pd, json, matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.metrics import roc_curve, auc

    path = OUTPUTS_DIR / "both_model_predictions_2020.csv"
    if not path.exists():
        raise FileNotFoundError(
            "both_model_predictions_2020.csv not found – run the script once with '--save-both-preds' first."
        )

    df = pd.read_csv(path)
    if {"y_true", "p_hat_xgb", "p_hat_mlp"} - set(df.columns):
        raise ValueError("Predictions file is missing required columns.")

    # ── Plot ROC curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, col in [
        ("XGB (9‑predictor)", "p_hat_xgb"),
        ("MLP (9‑predictor)", "p_hat_mlp"),
    ]:
        fpr, tpr, _ = roc_curve(df["y_true"], df[col])
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC {auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, label="Chance")
    ax.set_xlabel("1 – Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC curves – 2020 hold‑out")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "roc_overlay_2020.png", dpi=300)

    # ── Save raw coordinates for reproducibility ─────────────────────
    from sklearn.metrics import roc_curve as _roc
    roc_dict = {
        "xgb": _roc(df["y_true"], df["p_hat_xgb"]),
        "mlp": _roc(df["y_true"], df["p_hat_mlp"]),
    }
    with open(OUTPUTS_DIR / "roc_coords_2020.json", "w") as fp:
        json.dump({k: [list(arr) for arr in v] for k, v in roc_dict.items()}, fp)

    print("✓ Wrote roc_overlay_2020.png and roc_coords_2020.json")


def export_calibration_mlp():
    """
    Computes the Brier score plus calibration intercept & slope for the
    MLP model on the 2020 hold‑out set, then writes them to
    calibration_mlp_2020.json.
    """
    import pandas as pd, json
    from pathlib import Path
    from sklearn.metrics import brier_score_loss
    from sklearn.linear_model import LogisticRegression

    path = OUTPUTS_DIR / "both_model_predictions_2020.csv"
    if not path.exists():
        raise FileNotFoundError(
            "both_model_predictions_2020.csv not found – run the script once with '--save-both-preds' first."
        )

    df = pd.read_csv(path)
    if {"y_true", "p_hat_mlp"} - set(df.columns):
        raise ValueError("Predictions file is missing required columns.")

    y = df["y_true"].values
    p = df["p_hat_mlp"].values

    brier = round(brier_score_loss(y, p), 3)
    cal   = LogisticRegression(fit_intercept=True, solver="lbfgs").fit(p.reshape(-1, 1), y)
    stats = {
        "brier": brier,
        "intercept": round(float(cal.intercept_[0]), 3),
        "slope": round(float(cal.coef_[0][0]), 3),
    }
    with open(OUTPUTS_DIR / "calibration_mlp_2020.json", "w") as fp:
        json.dump(stats, fp, indent=2)

    print("✓ Wrote calibration_mlp_2020.json →", stats)


if __name__ == "__main__":
    main()