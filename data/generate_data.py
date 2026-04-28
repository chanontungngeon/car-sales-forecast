"""
generate_data.py — Simulated Car Sales Dataset
================================================
Creates a realistic monthly car sales dataset for Thailand from Jan 2018
to Dec 2024 and saves it to data/car_sales.csv.

WHY simulate data?
    Real dealership data is proprietary. Simulation lets us control the
    ground truth (known trend, seasonality, promo effect) so we can later
    verify that our models actually learned what we put in.

Data model (multiplicative):
    sales = base × trend × season × promo_boost × holiday_boost × noise

    Multiplicative is used (not additive) because each factor scales the
    others proportionally — a promotion in December (already high season)
    produces more extra units than the same promotion in September.

Run:  python data/generate_data.py
Out:  data/car_sales.csv  (2,100 rows: 5 brands × 5 regions × 84 months)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Output path ────────────────────────────────────────────────────────────────
# Path(__file__).parent resolves to the directory containing THIS file
# (i.e. the data/ folder), so the CSV always lands next to the script
# regardless of which directory the user runs the command from.
OUT_PATH = Path(__file__).parent / "car_sales.csv"

# ── Market configuration ───────────────────────────────────────────────────────
BRANDS  = ["Toyota", "Honda", "Isuzu", "Ford", "Mazda"]
REGIONS = ["Bangkok", "Central", "North", "Northeast", "South"]

# Monthly unit sales baseline per brand at an average region.
# Toyota leads because it dominates the Thai pickup-truck and sedan market.
# These are BEFORE regional, seasonal, and trend multipliers are applied.
BASE_SALES = {
    "Toyota": 500,  # market leader — largest dealer network
    "Honda":  350,  # strong in passenger cars (City, Civic)
    "Isuzu":  400,  # dominates pickup trucks (D-Max) — high rural demand
    "Ford":   200,  # smaller footprint; mainly Ranger pickup
    "Mazda":  150,  # premium positioning; lower volume
}

# How much each region scales the brand baseline.
# Bangkok is 1.4× average: largest population + highest income = most dealerships.
# Northeast is 0.8×: more rural, lower average income, fewer showrooms.
REGION_FACTOR = {
    "Bangkok":   1.4,
    "Central":   1.1,
    "North":     0.9,
    "Northeast": 0.8,
    "South":     0.8,
}

# Seasonal multiplier by calendar month (average month = 1.0).
# December (1.22): year-end bonuses, fleet renewals, dealer push for quotas.
# March/April (1.10/1.08): Songkran stimulus campaigns.
# February (0.88): shortest month + post-January fatigue in spending.
# September (0.88): low historically — mid-year budget exhaustion.
SEASONAL = {
    1: 1.05, 2: 0.88, 3: 1.10, 4: 1.08, 5: 0.95, 6: 0.92,
    7: 0.90, 8: 0.93, 9: 0.88, 10: 0.97, 11: 1.02, 12: 1.22,
}

# Months that receive the is_holiday_month flag (separate from seasonal index).
# We model holidays as an additional feature so the tree models can learn it
# independently of the seasonal pattern.
HOLIDAY_MONTHS = {3, 4, 12}

# Base sticker prices in Thai Baht per brand.
# These are jittered with 3% noise in the simulation to mimic monthly
# dealer-level pricing variation (discounts, trim-mix shifts).
PRICE_BASE = {
    "Toyota": 750_000,
    "Honda":  680_000,
    "Isuzu":  820_000,   # pickups tend to be pricier
    "Ford":   700_000,
    "Mazda":  690_000,
}


def generate(seed: int = 42) -> pd.DataFrame:
    """
    Generate and save the simulated car sales dataset.

    Parameters
    ----------
    seed : int
        NumPy random seed. Fix to 42 so the dataset is reproducible —
        every run produces the same CSV, which is important for debugging
        and fair model comparison.

    Returns
    -------
    pd.DataFrame
        The full 2,100-row dataset (also written to OUT_PATH as CSV).
    """
    # Fix the random seed FIRST before any stochastic calls.
    np.random.seed(seed)

    # pd.date_range with freq="MS" (Month Start) gives one date per month,
    # always on the 1st — e.g. 2018-01-01, 2018-02-01, …, 2024-12-01.
    dates = pd.date_range("2018-01", "2024-12", freq="MS")
    rows: list[dict] = []

    for date in dates:
        m = date.month  # calendar month (1–12) for seasonal lookup

        # ── Trend ──────────────────────────────────────────────────────────
        # Linear 2% annual growth representing Thailand's expanding middle
        # class and first-car buyers. year - 2018 gives 0 in the first year
        # (trend = 1.00) and 6 in the last (trend = 1.12 = +12% over 6 years).
        trend = 1 + 0.02 * (date.year - 2018)

        season = SEASONAL[m]  # look up the month's seasonal multiplier

        for brand in BRANDS:
            for region in REGIONS:

                # ── Base volume ────────────────────────────────────────────
                # Combine brand baseline with regional scaling.
                base = BASE_SALES[brand] * REGION_FACTOR[region]

                # ── Promotion flag ─────────────────────────────────────────
                # Each brand-region-month independently has a 25% chance of
                # running a promotion (finance rate cut, cashback, free accessory).
                # int() converts True/False → 1/0 for the CSV column.
                promotion = int(np.random.rand() < 0.25)

                # ── Holiday flag ───────────────────────────────────────────
                # 1 if this month is a major public holiday cluster, else 0.
                holiday = int(m in HOLIDAY_MONTHS)

                # ── Noise ──────────────────────────────────────────────────
                # Multiplicative Gaussian noise (mean=1, std=0.06) simulates
                # real-world variance: supply disruptions, weather events,
                # competitor launches, etc. 6% std dev is calibrated to
                # produce natural-looking fluctuations without overwhelming
                # the signal.
                noise = np.random.normal(1.0, 0.06)

                # ── Final sales calculation ────────────────────────────────
                # Multiply all factors together. Each factor is independent
                # and scales proportionally (multiplicative model).
                # +15% for promotion, +12% for holiday month.
                sales = int(
                    base * trend * season
                    * (1 + 0.15 * promotion)
                    * (1 + 0.12 * holiday)
                    * noise
                )

                # Floor at 10 units: even the slowest brand/region/month
                # records at least a handful of sales (floor prevents
                # nonsensical zero or negative values from extreme noise draws).
                sales = max(sales, 10)

                # ── Price simulation ───────────────────────────────────────
                # Average transaction price varies ±3% around the brand base
                # (dealer discounts, accessory bundles, trim-mix variation).
                price_avg = int(PRICE_BASE[brand] * np.random.normal(1.0, 0.03))

                # ── Competitor discount ────────────────────────────────────
                # Categorical variable representing the prevailing intensity
                # of competitor promotions that month.
                # 50% low / 35% medium / 15% high reflects typical market
                # conditions (aggressive discounting is relatively rare).
                competitor_discount = np.random.choice(
                    ["low", "medium", "high"], p=[0.50, 0.35, 0.15]
                )

                rows.append({
                    "date":                date,
                    "brand":               brand,
                    "region":              region,
                    "sales":               sales,           # target variable
                    "price_avg":           price_avg,
                    "promotion":           promotion,
                    "is_holiday_month":    holiday,
                    "competitor_discount": competitor_discount,
                })

    df = pd.DataFrame(rows)

    # Save without the row index (index=False) — cleaner CSV with no
    # unnamed column when re-loaded.
    df.to_csv(OUT_PATH, index=False)
    print(f"[generate_data] Saved {len(df):,} rows → {OUT_PATH}")

    # Quick sanity check: print total sales per brand so we can verify
    # Toyota is the leader and Mazda has the lowest volume.
    print(df.groupby("brand")["sales"].sum().sort_values(ascending=False).to_string())
    return df


# ── Entry point ────────────────────────────────────────────────────────────────
# Only runs when called directly: `python data/generate_data.py`
# When imported by main.py, this block is skipped.
if __name__ == "__main__":
    generate()
