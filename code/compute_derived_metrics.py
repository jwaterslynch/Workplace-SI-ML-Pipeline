#!/usr/bin/env python3
"""
compute_derived_metrics.py — Reproduce all derived metrics from temporal_results.json

This script takes the 9×9 AUC grid produced by the pipeline and computes:
1. Same-year vs cross-year summary statistics
2. Distance-decay regression
3. Pre-COVID vs Post-COVID transport analysis
4. Rolling 3-year window vs single-year comparison
5. Prevalence trend analysis

NOTE: The rolling window comparison uses DOCUMENTED values from the actual
rolling window analysis (which pools raw data from 3 years and retrains).
These cannot be derived from the grid alone — the grid tests individual
year-trained models, while rolling window pools data before training.
The values here match ANALYSIS_RESULTS.md exactly.

Usage:
    python compute_derived_metrics.py
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "temporal_results.json"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "merged" / "derived_metrics.json"

with open(DATA_PATH) as f:
    data = json.load(f)

years = [str(y) for y in range(2015, 2024)]
n_years = len(years)

# Build the AUC matrix
auc_matrix = np.zeros((n_years, n_years))
n_matrix = np.zeros((n_years, n_years), dtype=int)
npos_matrix = np.zeros((n_years, n_years), dtype=int)

for i, train_yr in enumerate(years):
    for j, test_yr in enumerate(years):
        auc_matrix[i, j] = data['basic'][train_yr][test_yr]['auc']
        n_matrix[i, j] = data['basic'][train_yr][test_yr]['n_total']
        npos_matrix[i, j] = data['basic'][train_yr][test_yr]['n_positive']

results = {}

# ============================================================
# 1. Same-year vs cross-year
# ============================================================
diag = np.diag(auc_matrix)
off_diag = auc_matrix[~np.eye(n_years, dtype=bool)]

results['same_year'] = {
    'mean': float(np.mean(diag)),
    'sd': float(np.std(diag, ddof=1)),
    'min': float(np.min(diag)),
    'max': float(np.max(diag)),
    'values': {yr: float(diag[i]) for i, yr in enumerate(years)}
}
results['cross_year'] = {
    'mean': float(np.mean(off_diag)),
    'sd': float(np.std(off_diag, ddof=1)),
    'min': float(np.min(off_diag)),
    'max': float(np.max(off_diag)),
}
results['gap'] = float(np.mean(diag) - np.mean(off_diag))

print("=" * 60)
print("1. SAME-YEAR vs CROSS-YEAR")
print("=" * 60)
print(f"  Same-year:  M={results['same_year']['mean']:.4f}, "
      f"SD={results['same_year']['sd']:.4f}, "
      f"Range=[{results['same_year']['min']:.4f}, {results['same_year']['max']:.4f}]")
print(f"  Cross-year: M={results['cross_year']['mean']:.4f}, "
      f"SD={results['cross_year']['sd']:.4f}, "
      f"Range=[{results['cross_year']['min']:.4f}, {results['cross_year']['max']:.4f}]")
print(f"  Gap: {results['gap']:.4f}")

# ============================================================
# 2. Distance decay
# ============================================================
distances = []
aucs_for_dist = []
for i in range(n_years):
    for j in range(n_years):
        if i != j:
            distances.append(abs(i - j))
            aucs_for_dist.append(auc_matrix[i, j])

distances = np.array(distances)
aucs_for_dist = np.array(aucs_for_dist)

slope, intercept, r_value, p_value, se = stats.linregress(distances, aucs_for_dist)
results['distance_decay'] = {
    'slope': float(slope),
    'intercept': float(intercept),
    'r_squared': float(r_value ** 2),
    'p_value': float(p_value),
    'se': float(se),
}

# Mean AUC by distance
unique_dists = sorted(set(distances))
for d in unique_dists:
    mask = distances == d
    m = float(np.mean(aucs_for_dist[mask]))
    results['distance_decay'][f'mean_at_dist_{int(d)}'] = m

print(f"\n{'=' * 60}")
print("2. DISTANCE DECAY")
print("=" * 60)
print(f"  Slope: {slope:.4f}/year")
print(f"  p-value: {p_value:.3f}")
print(f"  R²: {r_value**2:.3f}")
for d in unique_dists:
    mask = distances == d
    print(f"  Distance {int(d)}: M={np.mean(aucs_for_dist[mask]):.4f}")

# ============================================================
# 3. Pre-COVID vs Post-COVID transport
# ============================================================
pre_years = [0, 1, 2, 3, 4]   # 2015-2019
post_years = [5, 6, 7, 8]     # 2020-2023

pre_to_post = []
for i in pre_years:
    for j in post_years:
        pre_to_post.append(auc_matrix[i, j])

post_to_pre = []
for i in post_years:
    for j in pre_years:
        post_to_pre.append(auc_matrix[i, j])

pre_to_post = np.array(pre_to_post)
post_to_pre = np.array(post_to_pre)

t_stat, p_val = stats.ttest_ind(pre_to_post, post_to_pre)

results['covid_transport'] = {
    'pre_to_post_mean': float(np.mean(pre_to_post)),
    'pre_to_post_sd': float(np.std(pre_to_post, ddof=1)),
    'post_to_pre_mean': float(np.mean(post_to_pre)),
    'post_to_pre_sd': float(np.std(post_to_pre, ddof=1)),
    'difference': float(np.mean(pre_to_post) - np.mean(post_to_pre)),
    't_statistic': float(t_stat),
    'p_value': float(p_val),
}

print(f"\n{'=' * 60}")
print("3. COVID TRANSPORT")
print("=" * 60)
print(f"  Pre→Post: M={np.mean(pre_to_post):.4f}, SD={np.std(pre_to_post, ddof=1):.4f}")
print(f"  Post→Pre: M={np.mean(post_to_pre):.4f}, SD={np.std(post_to_pre, ddof=1):.4f}")
print(f"  Difference: {np.mean(pre_to_post) - np.mean(post_to_pre):.4f}")
print(f"  t = {t_stat:.2f}, p = {p_val:.3f}")

# ============================================================
# 4. Rolling 3-year window vs single-year
# ============================================================
# ACTUAL retrained values from ANALYSIS_RESULTS.md
# These are from pooling 3 years of raw data and retraining
# (NOT derived from the grid)
rolling_data = {
    'test_years': [2018, 2019, 2020, 2021, 2022, 2023],
    'rolling_3year_auc': [0.725, 0.707, 0.705, 0.711, 0.731, 0.717],
    'single_year_auc': [0.712, 0.692, 0.687, 0.687, 0.709, 0.712],
}

rolling = np.array(rolling_data['rolling_3year_auc'])
single = np.array(rolling_data['single_year_auc'])
diff = rolling - single

t_stat_rw, p_val_rw = stats.ttest_rel(rolling, single)

results['rolling_window'] = {
    'test_years': rolling_data['test_years'],
    'rolling_3year_auc': rolling_data['rolling_3year_auc'],
    'single_year_auc': rolling_data['single_year_auc'],
    'differences': [float(d) for d in diff],
    'mean_rolling': float(np.mean(rolling)),
    'sd_rolling': float(np.std(rolling, ddof=1)),
    'mean_single': float(np.mean(single)),
    'sd_single': float(np.std(single, ddof=1)),
    'mean_difference': float(np.mean(diff)),
    'sd_difference': float(np.std(diff, ddof=1)),
    't_statistic': float(t_stat_rw),
    'p_value': float(p_val_rw),
    'source': 'ANALYSIS_RESULTS.md (retrained on pooled data, not grid-derived)',
}

print(f"\n{'=' * 60}")
print("4. ROLLING WINDOW (from ANALYSIS_RESULTS.md)")
print("=" * 60)
print(f"  Rolling 3-year: M={np.mean(rolling):.4f}, SD={np.std(rolling, ddof=1):.4f}")
print(f"  Single-year:    M={np.mean(single):.4f}, SD={np.std(single, ddof=1):.4f}")
print(f"  Difference:     M=+{np.mean(diff):.4f}, SD={np.std(diff, ddof=1):.4f}")
print(f"  Paired t: t={t_stat_rw:.2f}, p={p_val_rw:.3f}")
for i, yr in enumerate(rolling_data['test_years']):
    print(f"    {yr}: single={single[i]:.3f}, rolling={rolling[i]:.3f}, diff=+{diff[i]:.3f}")

# ============================================================
# 5. Prevalence trend
# ============================================================
prev = []
ns = []
for i, yr in enumerate(years):
    n = n_matrix[i, i]  # same-year N
    npos = npos_matrix[i, i]
    p = npos / n * 100
    prev.append(p)
    ns.append(n)

years_num = np.arange(2015, 2024)
slope_prev, intercept_prev, r_prev, p_prev, se_prev = stats.linregress(years_num, prev)

results['prevalence'] = {
    'by_year': {str(y): {'n': int(ns[i]), 'n_positive': int(npos_matrix[i, i]),
                         'prevalence_pct': round(prev[i], 2)} for i, y in enumerate(range(2015, 2024))},
    'total_n': int(sum(ns)),
    'slope_pp_per_year': float(slope_prev),
    'r_squared': float(r_prev ** 2),
    'p_value': float(p_prev),
    'relative_change_pct': float((prev[-1] - prev[0]) / prev[0] * 100),
}

print(f"\n{'=' * 60}")
print("5. PREVALENCE TREND")
print("=" * 60)
for i, yr in enumerate(range(2015, 2024)):
    print(f"  {yr}: N={ns[i]:>6,}  positives={npos_matrix[i,i]:>5,}  prev={prev[i]:.2f}%")
print(f"  Total N: {sum(ns):,}")
print(f"  Slope: +{slope_prev:.3f} pp/year")
print(f"  R² = {r_prev**2:.3f}, p = {p_prev:.4f}")
print(f"  Relative change: +{(prev[-1] - prev[0])/prev[0]*100:.1f}%")

# ============================================================
# 6. 2020 Full Model
# ============================================================
results['2020_full_model'] = {
    'auc': data['2020_full']['auc'],
    'n_total': data['2020_full']['n_total'],
    'n_positive': data['2020_full']['n_positive'],
    'note': '30% hold-out from 2020 wave, 9-predictor model including K6',
}

print(f"\n{'=' * 60}")
print("6. 2020 FULL MODEL")
print("=" * 60)
print(f"  AUC: {data['2020_full']['auc']:.4f}")
print(f"  N (hold-out): {data['2020_full']['n_total']:,}")
print(f"  Positives: {data['2020_full']['n_positive']:,}")

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print(f"All derived metrics saved to: {OUTPUT_PATH}")
print("=" * 60)
