"""
data_prep.py — Feature Engineering (shared utility module)
===========================================================
This module is imported by BOTH model_arima.py and model_tree.py.
It provides three things:

1.  load_raw()           — load and sort the raw CSV
2.  aggregate_monthly()  — collapse brand×region into total monthly series
                           (used by ARIMA, which is univariate)
3.  make_tree_features() — add lag, rolling, and calendar features
                           (used by XGBoost / LightGBM)

WHY separate this into its own module?
    Both models need the same data loading and splitting logic. Keeping it
    here avoids copy-paste bugs: if we change a feature definition, it
    updates for both models simultaneously.

Key concept — feature engineering for time series:
    Unlike tabular ML, you cannot use future data to predict the past.
    All engineered features MUST be derived from values that would be
    known at prediction time (i.e. shifted at least 1 period into the past).
    This is enforced throughout via .shift(1) before any rolling window.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Default data location ──────────────────────────────────────────────────────
# Resolves relative to THIS file's directory, so imports work regardless
# of the caller's working directory.
DATA_PATH = Path(__file__).parent.parent / "data" / "car_sales.csv"

# ── Feature column registry ────────────────────────────────────────────────────
# All tree-model input features listed in one place so both model_tree.py
# and streamlit_app.py always use the identical column set — no drift between
# training and inference.
FEATURE_COLS = [
    # ── Categorical encodings (integer codes) ──────────────────────────────
    "brand_code",       # which car brand (label-encoded 0–4)
    "region_code",      # which region (label-encoded 0–4)
    "comp_disc_code",   # competitor discount intensity (label-encoded 0–2)

    # ── Business context features ──────────────────────────────────────────
    "price_avg",        # average transaction price this month
    "promotion",        # own promotion flag (0/1)
    "is_holiday_month", # major holiday cluster (0/1)

    # ── Calendar features ──────────────────────────────────────────────────
    # These encode temporal position so the model can learn patterns like
    # "December always peaks" without needing lag_12 alone to capture it.
    "month",            # 1–12: within-year position (raw seasonality signal)
    "quarter",          # 1–4: coarser seasonal grouping
    "year",             # 2018–2024: captures long-term trend direction
    "is_year_end",      # binary flag for December (Q4 push, bonus spending)
    "is_q1",            # binary flag for Jan–Mar (post-Songkran recovery)

    # ── Lag features (autoregressive) ─────────────────────────────────────
    # Lags capture the autocorrelation structure of sales (past sales
    # predict future sales). Each lag is within the same brand×region group
    # so we don't bleed Toyota-Bangkok into Honda-Chiang Mai.
    "lag_1",            # previous month's sales (short-term momentum)
    "lag_3",            # 3 months ago (quarterly rhythm)
    "lag_12",           # same month last year (annual seasonal memory)

    # ── Rolling statistics (trend smoothing) ──────────────────────────────
    # Rolling windows applied AFTER a 1-period shift to avoid leakage.
    # A 3-month rolling mean smooths noise and reveals the recent trend.
    # A 6-month mean captures a slower, medium-term trend signal.
    # Rolling std gives a volatility estimate — useful when sales are
    # erratic (high std → model should be less confident in lags).
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
]

# The column we are trying to predict.
TARGET_COL = "sales"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw car sales CSV and sort it chronologically within each group.

    Sorting by [brand, region, date] ensures that lag and rolling
    operations in make_tree_features() process each group in time order —
    without this, shifted values would be meaningless.

    Parameters
    ----------
    path : Path
        Path to the CSV file (defaults to data/car_sales.csv).

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with 'date' parsed as datetime.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    # Sort so each brand-region time series is contiguous and in order.
    # reset_index(drop=True) gives clean 0-based indexing after sort.
    df = df.sort_values(["brand", "region", "date"]).reset_index(drop=True)
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum sales across all brands and regions to produce a single
    total-market monthly time series.

    WHY this aggregation?
        ARIMA is a univariate model — it can only model one series at a time.
        We collapse the 25 brand×region combinations into a single total
        so ARIMA can find the market-level trend and seasonality.
        The tree models don't need this because they take brand/region
        as input features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with one row per brand-region-month.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'sales'] — one row per month.
    """
    monthly = (
        df.groupby("date")["sales"]
        .sum()
        .reset_index()
        .sort_values("date")       # ensure chronological order for ARIMA
        .reset_index(drop=True)
    )
    return monthly


def make_tree_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag, rolling-window, and calendar features to the raw dataset.

    All time-based features are constructed to be leak-free:
    - Lags are shifted at least 1 period forward (shift(1) means we only
      see values that were already observed before the current row).
    - Rolling statistics are computed on already-shifted data.

    At prediction time (real deployment), the caller must supply the
    same feature values from genuinely historical observations — the
    training process cannot enforce this, so it must be a workflow
    discipline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from load_raw().

    Returns
    -------
    pd.DataFrame
        Original columns plus all FEATURE_COLS. Rows with NaN lags
        (first 12 months of each brand-region group) are dropped.
    """
    # Work on a copy so we never mutate the caller's DataFrame.
    df = df.copy().sort_values(["brand", "region", "date"]).reset_index(drop=True)

    # ── Categorical encoding ───────────────────────────────────────────────────
    # pd.Categorical assigns an integer code to each unique string value.
    # This is "label encoding" — simpler than one-hot here because tree
    # models (XGBoost, LightGBM) can learn non-linear splits on integers.
    # One-hot encoding would create many sparse columns and is usually
    # unnecessary for gradient boosting.
    df["brand_code"]     = pd.Categorical(df["brand"]).codes
    df["region_code"]    = pd.Categorical(df["region"]).codes
    df["comp_disc_code"] = pd.Categorical(df["competitor_discount"]).codes

    # ── Calendar features ──────────────────────────────────────────────────────
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter
    df["year"]        = df["date"].dt.year
    df["is_year_end"] = (df["month"] == 12).astype(int)
    df["is_q1"]       = (df["quarter"] == 1).astype(int)

    # ── Lag and rolling features ───────────────────────────────────────────────
    # groupby().transform() applies the function independently within each
    # brand-region group and aligns results back to the original row index —
    # this is the key operation that keeps Toyota-Bangkok lags separate
    # from Honda-Chiang Mai lags.
    grp = df.groupby(["brand", "region"])["sales"]

    # Lag 1: what did this brand sell in this region last month?
    # shift(1) moves each value one row forward within the group, so
    # row N gets the value from row N-1 (the prior month).
    df["lag_1"]  = grp.shift(1)

    # Lag 3: three months ago (captures quarterly cycle, e.g., Q4 effect
    # visible 3 months later as Q1 recovery or hangover).
    df["lag_3"]  = grp.shift(3)

    # Lag 12: same calendar month in the prior year (the single strongest
    # predictor for seasonal data — December 2023 predicts December 2024).
    df["lag_12"] = grp.shift(12)

    # Rolling mean 3 months: we shift(1) FIRST so the window only includes
    # [t-3, t-2, t-1], never the current month t.
    # min_periods=1 allows the window to compute even when fewer than 3
    # observations exist (start of each group's history).
    df["rolling_mean_3"] = grp.shift(1).transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # Rolling mean 6 months: same logic, wider window → smoother trend.
    df["rolling_mean_6"] = grp.shift(1).transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )

    # Rolling std 3 months: measures recent volatility.
    # min_periods=2 because std requires at least 2 points; fillna(0) for
    # the first row of each group where only 1 observation is available.
    df["rolling_std_3"] = grp.shift(1).transform(
        lambda x: x.rolling(3, min_periods=2).std().fillna(0)
    )

    # Drop rows where lag_1, lag_3, or lag_12 are NaN.
    # These NaNs occur in the first 12 rows of each brand-region group
    # (we don't have 12 prior months of history at the start of 2018).
    # We only enforce the lag_12 threshold — lag_1 and lag_3 would also
    # be NaN in those early rows, but lag_12 is the binding constraint.
    df = df.dropna(subset=["lag_1", "lag_3", "lag_12"]).reset_index(drop=True)
    return df


def train_val_test_split(df: pd.DataFrame, test_year: int = 2024, val_year: int = 2023):
    """
    Split the feature-engineered dataset by year — no shuffling.

    WHY year-based splits (not random)?
        Random splitting would leak future data into training:
        a model trained on a random sample including 2024 rows could
        "see" 2024 lags and perform unrealistically well on the 2024 test set.
        Time-based splits are mandatory for honest time series evaluation.

    Split design:
        Train  2018–2022  (5 years of history — used to fit the model)
        Val    2023       (1 year hold-out — used for hyperparameter tuning)
        Test   2024       (1 year hold-out — final, untouched evaluation)

    In run_trees() we combine Train + Val for the final fit, using Val only
    conceptually for hyperparameter guidance during design.

    Parameters
    ----------
    df         : DataFrame with a 'year' column (added by make_tree_features).
    test_year  : The year held out for final evaluation.
    val_year   : The year held out for validation.

    Returns
    -------
    (train, val, test) : three DataFrames
    """
    train = df[df["year"] <  val_year].copy()   # strictly before validation year
    val   = df[df["year"] == val_year].copy()
    test  = df[df["year"] == test_year].copy()
    return train, val, test
