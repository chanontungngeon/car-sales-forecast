"""
model_arima.py — Seasonal ARIMA Forecasting
============================================
Trains a Seasonal ARIMA model on total aggregated monthly car sales
(all brands, all regions combined) and evaluates it on 2024.

What is ARIMA?
    ARIMA = AutoRegressive Integrated Moving Average.
    It models a time series using three components:
      AR(p) — the value at time t depends on the previous p values
      I(d)  — differencing d times to make the series stationary
      MA(q) — the value at time t depends on the previous q forecast errors

    Seasonal ARIMA (SARIMA) adds a second set of (P, D, Q) parameters
    operating at the seasonal lag m. Here m=12 because we have monthly
    data with an annual seasonal cycle.

    ARIMA is a classical statistical model — it is interpretable,
    has well-understood assumptions, and provides confidence intervals.
    Its main limitation here is that it is UNIVARIATE: it cannot use
    brand, region, price, or promotion as predictors.

Why auto_arima?
    Selecting (p, d, q, P, D, Q) manually requires inspecting ACF/PACF
    plots and iterating. auto_arima automates this by searching over a
    parameter grid and selecting the combination that minimises AIC
    (Akaike Information Criterion — a measure of model fit penalised
    for complexity, to avoid overfitting).

Run standalone:  python src/model_arima.py
Or import:       from src.model_arima import run_arima
"""

import sys
import pickle
import numpy as np
import pandas as pd

# matplotlib.use("Agg") must be called BEFORE importing pyplot.
# "Agg" is a non-interactive backend — it renders plots to files instead
# of trying to open a GUI window. Required in headless environments like
# servers, Docker containers, or CI pipelines where no display is available.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Import path fix ────────────────────────────────────────────────────────────
# When running as `python src/model_arima.py`, Python's working directory
# is the project root but sys.path doesn't include `src/`.
# We prepend it explicitly so `from data_prep import ...` resolves correctly.
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_raw, aggregate_monthly

# ── Directory constants ────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent   # project root
OUT   = ROOT / "output"               # model artifacts and metrics
PLOTS = OUT / "plots"                  # saved chart images


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error — the most intuitive accuracy metric.

    MAPE = mean( |actual - predicted| / actual ) × 100

    A MAPE of 8% means the model is off by 8% on average — easy to
    communicate to non-technical stakeholders ("within 8%").

    WHY mask zeros?
        If actual sales are 0, division is undefined (ZeroDivisionError).
        In our simulated data the floor is 10, so this is a safety guard
        for real-world data where zero sales do occur.

    Parameters
    ----------
    actual    : True observed values.
    predicted : Model forecasts.

    Returns
    -------
    float : MAPE as a percentage (e.g. 8.3 means 8.3%).
    """
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def run_arima() -> dict:
    """
    Full ARIMA training and evaluation pipeline.

    Steps:
        1. Load and aggregate data to a monthly total series.
        2. Split: train on 2018–2023, test on 2024.
        3. Fit auto_arima to find the best SARIMA(p,d,q)(P,D,Q,12) order.
        4. Forecast 12 months (all of 2024).
        5. Compute MAE, RMSE, MAPE on the test set.
        6. Save the fitted model as output/model_arima.pkl.
        7. Generate and save the forecast plot.

    Note: pmdarima is imported INSIDE this function (not at module level).
    This lets model_arima.py be imported without crashing if pmdarima
    is not installed — the error only appears when run_arima() is actually called.

    Returns
    -------
    dict : {"model": "ARIMA", "MAE": float, "RMSE": float, "MAPE": float}
    """
    # Deferred import — allows the module to load even if pmdarima is missing.
    from pmdarima import auto_arima

    # Ensure output directories exist before writing anything.
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    # ── Load and aggregate data ────────────────────────────────────────────────
    df      = load_raw()
    monthly = aggregate_monthly(df)   # collapses to 84 monthly rows (2018–2024)

    # ── Train / test split ────────────────────────────────────────────────────
    # ARIMA needs all training data as a contiguous series — we filter
    # by date rather than year to keep the datetime index intact.
    train = monthly[monthly["date"] <  "2024-01-01"]   # 72 months (2018–2023)
    test  = monthly[monthly["date"] >= "2024-01-01"]   # 12 months (Jan–Dec 2024)

    # ── Fit SARIMA via auto_arima ──────────────────────────────────────────────
    print("[ARIMA] Fitting auto_arima (seasonal, m=12) — may take ~30 s ...")
    model = auto_arima(
        train["sales"],
        seasonal=True,           # enable seasonal component (P, D, Q, m)
        m=12,                    # seasonal period: 12 months = annual cycle
        stepwise=True,           # use stepwise search (faster than exhaustive grid)
                                 # stepwise tries only statistically-motivated orders
                                 # rather than all combinations, saving ~10× time
        suppress_warnings=True,  # hide convergence warnings from underlying SARIMAX
        error_action="ignore",   # skip parameter combos that fail to converge
        information_criterion="aic",  # AIC balances fit quality vs model complexity
                                      # (AIC preferred over BIC for forecasting tasks)
    )
    print(f"[ARIMA] Best order: {model.order}  seasonal: {model.seasonal_order}")
    # model.order = (p, d, q)
    # model.seasonal_order = (P, D, Q, 12)

    # ── Forecast ──────────────────────────────────────────────────────────────
    # Predict 12 steps ahead (one for each month in the test set).
    forecast = model.predict(n_periods=len(test))

    # Clip to 0: ARIMA can sometimes predict negative values if the
    # series shows high volatility near zero. Sales cannot be negative.
    forecast = np.clip(forecast, 0, None)

    # ── Evaluation metrics ────────────────────────────────────────────────────
    # Three complementary metrics:
    #   MAE  — average absolute error in raw units (easy to interpret)
    #   RMSE — root-mean-squared error; penalises large individual errors more
    #           than MAE does (sensitive to outliers)
    #   MAPE — percentage error; scale-independent for comparing across models
    mae_val  = mean_absolute_error(test["sales"], forecast)
    rmse_val = float(np.sqrt(mean_squared_error(test["sales"], forecast)))
    mape_val = mape(test["sales"].values, forecast)
    print(f"[ARIMA] MAE={mae_val:.0f}  RMSE={rmse_val:.0f}  MAPE={mape_val:.1f}%")

    # ── Persist model ─────────────────────────────────────────────────────────
    # pickle serialises the fitted pmdarima model object to disk so it can
    # be loaded later by the Streamlit app without retraining.
    with open(OUT / "model_arima.pkl", "wb") as f:
        pickle.dump(model, f)

    # ── Forecast visualisation ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot the full historical series in navy (brand colour).
    ax.plot(
        monthly["date"], monthly["sales"],
        label="Actual", color="#1b2a4a", linewidth=2,
    )

    # Overlay the 2024 forecast in red dashed line with circle markers.
    # test["date"].values converts to numpy array to align with numpy forecast array.
    ax.plot(
        test["date"].values, forecast,
        label="ARIMA Forecast", color="#c81934",
        linewidth=2, linestyle="--", marker="o", markersize=4,
    )

    # Vertical dashed line marks where training data ends — visually
    # separates "what the model saw" from "what it is predicting".
    ax.axvline(
        pd.Timestamp("2024-01-01"),
        color="gray", linestyle=":", alpha=0.7, label="Train/Test split",
    )

    ax.set_title("ARIMA — Total Monthly Sales Forecast vs Actual (2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    ax.legend()
    ax.grid(alpha=0.3)   # light grid improves readability without dominating
    plt.tight_layout()   # prevents axis labels from being clipped

    # dpi=130: higher than default (72/100) for sharper images in reports.
    # bbox_inches="tight": prevent title/labels from being cut off.
    fig.savefig(PLOTS / "arima_forecast.png", dpi=130, bbox_inches="tight")
    plt.close()  # release memory — important in batch runs
    print(f"[ARIMA] Plot saved → {PLOTS / 'arima_forecast.png'}")

    # Return a metrics dict in a consistent format so main.py can compare
    # all models in a unified table.
    return {
        "model": "ARIMA",
        "MAE":   round(mae_val,  1),
        "RMSE":  round(rmse_val, 1),
        "MAPE":  round(mape_val, 2),
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = run_arima()
    print(metrics)
