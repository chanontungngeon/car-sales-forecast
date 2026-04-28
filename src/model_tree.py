"""
model_tree.py — Gradient Boosting Forecasting (XGBoost + LightGBM)
====================================================================
Trains two tree-based regression models on the feature-engineered dataset
and evaluates them on 2024 data.  Produces three diagnostic plots:
  • feature_importance.png   — which inputs drove XGBoost's decisions
  • actual_vs_predicted.png  — scatter of true vs. forecast values
  • residuals.png            — distribution of forecast errors

Why gradient boosting for time series?
    Tree models have two key advantages over classical methods like ARIMA:
    1. They are MULTIVARIATE — they can use brand, region, promotion,
       competitor discounts, and all engineered lag features simultaneously.
       ARIMA can only see the historical values of the target series.
    2. They automatically capture non-linear relationships and interactions
       (e.g. "promotion effect is 20% larger in December than in September").

    Their main disadvantage: they cannot extrapolate beyond the range
    of training data, so long-horizon forecasts require careful lag
    construction to avoid "feeding the model its own errors".

XGBoost vs LightGBM:
    Both implement gradient boosted decision trees but differ in algorithm:
    • XGBoost: level-wise tree growth. More regularisation options.
      Typically slower but historically the gold standard on tabular data.
    • LightGBM: leaf-wise tree growth. Much faster on large datasets.
      Slightly more prone to overfitting on small datasets but rarely
      worse in practice.
    We train both and let metrics decide the winner.

TimeSeriesSplit — preventing data leakage in cross-validation:
    Standard k-fold CV shuffles data randomly, which for time series means
    training on future data to predict the past — completely invalid.
    TimeSeriesSplit(n_splits=5) produces 5 folds where each validation set
    is always strictly after its training set:
        Fold 1: train [0..16], val [17..33]
        Fold 2: train [0..33], val [34..50]
        ... and so on (expanding window)
    This gives an honest estimate of out-of-sample performance.

Run standalone:  python src/model_tree.py
Or import:       from src.model_tree import run_trees
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — renders to files, no GUI required
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prepend src/ to sys.path so data_prep imports work when running as a script.
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import (
    load_raw, make_tree_features,
    FEATURE_COLS, TARGET_COL, train_val_test_split,
)

# ── Directory constants ────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "output"
PLOTS = OUT / "plots"


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Excludes zeros from actual to prevent division-by-zero.
    Returns a percentage (e.g. 7.4 means the model is off by 7.4% on average).
    """
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _evaluate(
    name: str, model, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[np.ndarray, dict]:
    """
    Generate predictions, compute three metrics, and return both.

    This is a private helper (prefix _) called once for XGBoost and once
    for LightGBM — keeps run_trees() clean and avoids repeating metric logic.

    Parameters
    ----------
    name   : Display name for the model (used in print output).
    model  : Fitted sklearn-compatible estimator with a .predict() method.
    X_test : Feature matrix for the test set.
    y_test : True sales values for the test set.

    Returns
    -------
    predictions : np.ndarray  — clipped to >= 0
    metrics     : dict        — {"model", "MAE", "RMSE", "MAPE"}
    """
    # np.clip(pred, 0, None) sets a lower bound of 0.
    # Tree models can predict slightly negative values when the target
    # distribution extends near zero and the leaf average dips below 0.
    pred     = np.clip(model.predict(X_test), 0, None)
    mae_val  = mean_absolute_error(y_test, pred)
    rmse_val = float(np.sqrt(mean_squared_error(y_test, pred)))
    mape_val = mape(y_test.values, pred)
    print(f"[{name:8s}] MAE={mae_val:.0f}  RMSE={rmse_val:.0f}  MAPE={mape_val:.1f}%")
    return pred, {
        "model": name,
        "MAE":   round(mae_val,  1),
        "RMSE":  round(rmse_val, 1),
        "MAPE":  round(mape_val, 2),
    }


def run_trees() -> list[dict]:
    """
    Train XGBoost and LightGBM, evaluate on 2024, save artifacts and plots.

    Training strategy:
        1. Feature-engineer the full dataset.
        2. Split: train 2018–2022, val 2023, test 2024.
        3. For final training, concatenate train + val (train on 6 years,
           leaving 2024 completely unseen). This is standard practice —
           we use the validation year for design decisions (hyperparameters)
           but include it in the final fit to give the model maximum history.
        4. Evaluate on 2024 test set only.
        5. Save models, produce three diagnostic plots.

    Returns
    -------
    list[dict] : One metrics dict per model, in [XGBoost, LightGBM] order.
    """
    # Deferred imports — xgboost and lightgbm are only needed when this
    # function actually runs, not when the module is first imported.
    from xgboost  import XGBRegressor
    from lightgbm import LGBMRegressor

    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    # ── Load and engineer features ─────────────────────────────────────────────
    df = load_raw()
    df = make_tree_features(df)   # adds lags, rolling stats, calendar columns

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = train_val_test_split(df)

    # Combine train + val for the final model fit.
    # This gives 6 years of data (2018–2023) to train on, while keeping
    # 2024 (test_df) strictly held out for evaluation.
    fit_df = pd.concat([train_df, val_df])

    X_fit  = fit_df[FEATURE_COLS];   y_fit  = fit_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS];  y_test = test_df[TARGET_COL]

    # ── TimeSeriesSplit note ───────────────────────────────────────────────────
    # In a full hyperparameter search you would pass TimeSeriesSplit(n_splits=5)
    # into GridSearchCV or cross_val_score to get honest fold-level CV scores.
    # Here we use a manual year-based split for simplicity, which achieves the
    # same principle: validation data is always strictly after training data.

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("[XGBoost]  Training ...")
    xgb = XGBRegressor(
        n_estimators=300,        # number of trees (boosting rounds)
                                 # 300 balances accuracy and training time;
                                 # more trees risk diminishing returns
        learning_rate=0.05,      # "shrinkage" — each tree contributes only 5%
                                 # of its prediction. Low lr + more trees
                                 # generalises better than high lr + few trees
        max_depth=5,             # maximum depth of each tree.
                                 # 5 is moderate — deep enough to capture
                                 # interactions (brand × season × promo)
                                 # without memorising training data
        subsample=0.8,           # each tree is trained on a random 80% of rows
                                 # (row-level bagging) — reduces variance
        colsample_bytree=0.8,    # each tree sees a random 80% of features
                                 # (column-level bagging, like Random Forest)
                                 # prevents over-reliance on any single feature
        random_state=42,         # reproducible results
        n_jobs=-1,               # use all CPU cores
        verbosity=0,             # suppress XGBoost's own training output
    )
    xgb.fit(X_fit, y_fit)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("[LightGBM] Training ...")
    lgb = LGBMRegressor(
        n_estimators=300,        # same number of rounds as XGBoost for fair comparison
        learning_rate=0.05,      # identical shrinkage
        max_depth=5,             # same depth cap
        subsample=0.8,           # same row bagging (called "bagging_fraction" internally)
        colsample_bytree=0.8,    # same column bagging ("feature_fraction" internally)
        random_state=42,
        n_jobs=-1,
        verbose=-1,              # suppress LightGBM training output (-1 = silent)
    )
    lgb.fit(X_fit, y_fit)

    # ── Evaluate both models on the unseen 2024 test set ──────────────────────
    preds_xgb, metrics_xgb = _evaluate("XGBoost",  xgb, X_test, y_test)
    preds_lgb, metrics_lgb = _evaluate("LightGBM", lgb, X_test, y_test)

    # ── Persist models ────────────────────────────────────────────────────────
    # Saved as pickle files so the Streamlit dashboard can load them
    # without retraining. In production these would be uploaded to S3.
    with open(OUT / "model_xgboost.pkl",  "wb") as f: pickle.dump(xgb, f)
    with open(OUT / "model_lightgbm.pkl", "wb") as f: pickle.dump(lgb, f)

    # ── Plot 1: Feature Importance ─────────────────────────────────────────────
    # xgb.feature_importances_ returns "gain" importance by default in XGBoost —
    # the total improvement in prediction accuracy (reduction in split criterion)
    # brought by each feature across all trees and all splits.
    # Lag_12 and rolling_mean_6 typically rank highest because annual
    # seasonality is the dominant signal in this dataset.
    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Highlight the top-ranked feature in red; all others in blue.
    colors = [
        "#c81934" if importance[c] == importance.max() else "#3a68a8"
        for c in importance.index
    ]
    importance.plot.barh(ax=ax, color=colors)
    ax.set_title("XGBoost Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score (Gain)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / "feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Actual vs Predicted Scatter ───────────────────────────────────
    # A well-calibrated model produces points clustering tightly along
    # the diagonal (y = x line).  Points above the diagonal = over-prediction;
    # below = under-prediction.  A systematic bias would appear as all points
    # shifted to one side.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, preds, name, color in zip(
        axes,
        [preds_xgb, preds_lgb],
        ["XGBoost", "LightGBM"],
        ["#c81934", "#3a68a8"],
    ):
        ax.scatter(y_test, preds, alpha=0.35, color=color, s=14)

        # Extend the perfect-prediction line from 0 to slightly above the
        # largest observed value so it spans the full plot area.
        lim = max(y_test.max(), preds.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=1,
                label="Perfect fit (y=x)")

        ax.set_title(f"{name} — Actual vs Predicted (2024)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / "actual_vs_predicted.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Residual Distribution ─────────────────────────────────────────
    # Residual = actual − predicted.  A good model produces residuals that are:
    #   • centred at 0 (no systematic bias)
    #   • approximately symmetric (errors equally likely in both directions)
    #   • bell-shaped (most errors small, few large)
    # A left/right skew would indicate the model consistently over- or under-predicts.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, preds, name, color in zip(
        axes,
        [preds_xgb, preds_lgb],
        ["XGBoost", "LightGBM"],
        ["#c81934", "#3a68a8"],
    ):
        residuals = y_test.values - preds
        ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor="white")

        # Vertical line at 0 — shows whether the distribution is centred
        # (symmetric around zero = unbiased model).
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name} Residuals", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / "residuals.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(f"[Trees] Plots saved → {PLOTS}")

    # Return metrics in list form so main.py can extend the all_metrics list.
    return [metrics_xgb, metrics_lgb]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_trees()
    for r in results:
        print(r)
