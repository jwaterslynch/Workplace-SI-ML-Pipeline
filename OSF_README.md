from xgboost import XGBClassifier, plot_importance

# Optional SHAP import (used only when --shap is passed)
try:
    import shap
except Exception:
    shap = None


def calibration_stats(y_true, y_prob):
    eps = 1e-12
    lr = LogisticRegression(
        fit_intercept=True,
        penalty="none",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=1,
        random_state=42,
    ).fit(np.log((y_prob + eps) / (1 - y_prob + eps)).reshape(-1, 1), y_true)
    y_pred = lr.predict_proba(np.log((y_prob + eps) / (1 - y_prob + eps)).reshape(-1, 1))[:, 1]
    brier = brier_score_loss(y_true, y_prob)
    return {
        "brier": brier,
        "intercept": lr.intercept_[0],
        "slope": lr.coef_[0][0],
    }


def export_calibration_mlp():
    y, p = ...  # existing code to get true labels and predicted probabilities
    brier = round(brier_score_loss(y, p), 3)
    eps = 1e-12
    logit_p = np.log((p + eps) / (1 - p + eps)).reshape(-1, 1)
    cal = LogisticRegression(fit_intercept=True, solver="lbfgs", penalty="none", max_iter=1000).fit(logit_p, y)
    stats = {
        "brier": brier,
        "intercept": round(float(cal.intercept_[0]), 3),
        "slope": round(float(cal.coef_[0][0]), 3),
    }
    # existing code to save stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=DATA_DIR,
        help="Directory for raw NSDUH data files",
    )
    parser.add_argument(
        "--outputs-dir",
        type=pathlib.Path,
        default=OUTPUTS_DIR,
        help="Send all artefacts to this folder (default ./outputs)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download NSDUH data files",
    )
    # other arguments...

    args = parser.parse_args()
    args.data_dir.mkdir(exist_ok=True)

    # Allow overriding the outputs directory from CLI
    global OUTPUTS_DIR
    OUTPUTS_DIR = args.outputs_dir
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # rest of main() code


def shap_summary(model, X, feature_names):
    if shap is None:
        print("SHAP is not installed. Skipping SHAP summary.")
        return
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, features=X, feature_names=feature_names)