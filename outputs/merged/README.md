# Suicidal Ideation Prediction Paper - Publication Figures and Tables

## Overview
This directory contains all publication-ready figures and data tables for the merged suicidal ideation prediction paper (2015-2023).

## Generated Files

### Figures (PNG and PDF versions, 300 DPI)

#### Figure 1: Cross-Year Model Transport AUC Heatmap
- **Files**: `Figure_1_AUC_Heatmap.png` (428 KB), `Figure_1_AUC_Heatmap.pdf` (23 KB)
- **Description**: 9×9 heatmap showing AUC values for all training year × test year combinations
- **Key features**:
  - Diagonal cells (same-year models) highlighted with blue border
  - Color scale: 0.65-0.77 (RdYlGn colormap)
  - All 81 cells annotated with AUC values (3 decimal places)
  - Dimensions: 8×7 inches, 300 DPI

#### Figure 2: Same-Year vs Cross-Year AUC Distribution
- **Files**: `Figure_2_AUC_Distribution.png` (201 KB), `Figure_2_AUC_Distribution.pdf` (28 KB)
- **Description**: Violin plots with individual data points comparing same-year vs cross-year AUCs
- **Key features**:
  - Left plot: Same-year AUCs (n=9, mean=0.7499)
  - Right plot: Cross-year AUCs (n=72, mean=0.6877)
  - Performance gap annotated: 0.0622 (8.3% degradation)
  - Red diamond markers show means
  - Dimensions: 6×5 inches, 300 DPI

#### Figure 3: Single-Year vs Rolling 3-Year Training
- **Files**: `Figure_3_Training_Strategy.png` (146 KB), `Figure_3_Training_Strategy.pdf` (23 KB)
- **Description**: Grouped bar chart comparing single-year training vs rolling 3-year window strategy
- **Key features**:
  - Blue bars: Single-year training (train on immediately preceding year)
  - Coral bars: Rolling 3-year training (pool 3 preceding years)
  - Test years: 2016-2023 (8 bars)
  - Mean improvement shown in annotation box
  - Dimensions: 7×5 inches, 300 DPI

#### Figure 4: SI Prevalence Trend 2015-2023
- **Files**: `Figure_4_Prevalence_Trend.png` (157 KB), `Figure_4_Prevalence_Trend.pdf` (28 KB)
- **Description**: Line plot of SI prevalence percentage over time with linear trend
- **Key features**:
  - Blue line: Observed annual prevalence
  - Red dashed line: Linear regression trend
  - Gray shaded region: COVID-19 period (2020)
  - Slope annotation: +0.2873 pp/year
  - Y-axis: 0-8% prevalence
  - Dimensions: 7×4.5 inches, 300 DPI

### Data Tables (CSV format)

#### Table 1: Sample Characteristics by Year
- **File**: `Table_1_Sample_Characteristics.csv`
- **Columns**: Year, N (total sample), SI+ Cases, Prevalence %, Sensitivity, Specificity, AUC
- **Rows**: 9 (one per year, 2015-2023)
- **Key statistics**:
  - Total N across all years: 176,957
  - Total SI+ cases: 9,751
  - Prevalence range: 4.21% (2016) to 6.57% (2023)
  - 2020 sample size reduced due to COVID-19

#### Table 2: AUC Summary Statistics
- **File**: `Table_2_AUC_Summary.csv`
- **Rows**: Mean, SD, Min, Max, Range
- **Columns**: Same-Year (n=9), Cross-Year (n=72), Difference
- **Key findings**:
  - Same-year AUCs more stable (SD=0.0109)
  - Cross-year AUCs show more variability (SD=0.0159)
  - Gap between same-year and cross-year: 0.0622

#### Table 3: AUC Matrix
- **File**: `Table_3_AUC_Matrix.csv`
- **Dimensions**: 9×9 (training year × test year)
- **Content**: All 81 AUC values from the complete cross-year evaluation
- **Format**: Training years as rows, test years as columns
- **Range**: 0.6520 (Train 2016 → Test 2015) to 0.7641 (Train 2022 → Test 2022)

#### Table 4: Training Strategy Comparison
- **File**: `Table_4_Training_Strategy.csv`
- **Columns**: Test Year, Single-Year AUC, Rolling 3-Year AUC, Improvement
- **Rows**: 8 (test years 2016-2023; 2015 excluded as no prior year data)
- **Key finding**: Mean improvement = -0.0034 (rolling 3-year slightly worse on average)

## Key Findings Summary

### 1. Study Population
- **Period**: 2015-2023 (9 years)
- **Total patients**: 176,957
- **Total SI+ cases**: 9,751 (5.51% overall prevalence)
- **Annual average**: 19,662 patients/year
- **COVID-19 impact**: 2020 sample size reduced to 60% of normal

### 2. Model Performance
- **Same-year AUC**: 0.7499 ± 0.0109 (range: 0.7309-0.7641)
- **Cross-year AUC**: 0.6877 ± 0.0159 (range: 0.6520-0.7204)
- **Performance degradation**: 0.0622 AUC points (8.3% relative drop)
- **Best single-year model**: 2022 (AUC=0.7641)
- **Worst single-year model**: 2016 (AUC=0.7309)

### 3. Temporal Trends
- **SI prevalence trend**: +0.2873 percentage points/year
  - 2015: 4.30%
  - 2023: 6.57%
  - Overall increase: +2.27 percentage points
- **Prevalence range**: 4.21% (2016) to 6.57% (2023)

### 4. Cross-Year Generalization
- **81 total training/testing combinations** (9 same-year + 72 cross-year)
- **Lowest cross-year AUC**: 0.6520 (Train 2016 → Test 2015)
- **Highest cross-year AUC**: 0.7204 (Train 2022 → Test 2023)
- **Implication**: Model performance is highly dependent on training year

## Style and Quality Standards

All figures follow academic publication standards:
- **Resolution**: 300 DPI (suitable for print and digital)
- **Format**: PNG (raster, high quality) + PDF (vector, scalable)
- **Font**: Arial, 11-12pt
- **Style**: seaborn-v0_8-whitegrid with minimal chartjunk
- **Color**: Publication-grade colormaps (RdYlGn, steelblue, coral)
- **Dimensions**: Professional aspect ratios for journal publication

## Usage Instructions

1. **For manuscript submission**: Use PDF versions for text embedding
2. **For presentations**: Use PNG versions at native 300 DPI
3. **For supplementary material**: Include both PNG and PDF
4. **For data analysis**: CSV files are ready for import into any statistical software
5. **For figure captions**: See below for recommended caption text

## Recommended Figure Captions

**Figure 1 (AUC Heatmap)**: Cross-Year Model Transport: AUC by Training and Test Year
Cross-validation results showing model performance (AUC) for all training year × test year combinations (2015-2023). Blue bordered cells indicate same-year (within-year) training/testing. Note the systematic performance degradation when models trained on one year are applied to another year, with mean cross-year AUC = 0.688 compared to mean same-year AUC = 0.750.

**Figure 2 (Distribution)**: Comparison of Same-Year vs Cross-Year Model Performance
Distribution of AUC values for same-year models (n=9, left) versus cross-year models (n=72, right). Individual observations shown as dots, violin plots indicate density distribution, red diamonds show means. The 0.062-point gap demonstrates substantial temporal drift in model performance across years.

**Figure 3 (Training Strategy)**: Single-Year vs Rolling 3-Year Training Strategies
Comparison of predictive performance when using single-year training (immediately preceding year) versus rolling 3-year window training strategies. Error bars would show variability if available; values annotated above bars.

**Figure 4 (Prevalence Trend)**: Suicidal Ideation Prevalence Trend (2015-2023)
Annual SI prevalence (%) over the study period with linear trend line (slope = +0.287 pp/year). Gray shaded region indicates COVID-19 pandemic period (2020). The observed upward trend suggests increasing population-level SI across the study period.

## Dataset Information

- **Source data**: `/data/temporal_results.json`
- **Generated by**: `generate_figures.py`
- **Generated on**: 2026-02-17
- **Analysis period**: 2015-2023
- **Data format**: Cross-year AUC matrices with sample characteristics

## Technical Notes

1. All figures generated using Python 3 with matplotlib, seaborn, pandas, and scipy
2. 300 DPI resolution ensures publication quality for both print and digital media
3. PDF versions use vector graphics for perfect scaling in manuscripts
4. CSV tables are UTF-8 encoded with standard comma delimiters
5. All statistical calculations use standard formulas (mean, SD with N-1 denominator)

---
**Generated**: February 17, 2026  
**Script**: generate_figures.py  
**Contact**: See manuscript for corresponding author details
