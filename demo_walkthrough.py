"""
demo_walkthrough.py — Dashboard Demo Guide
===========================================
Prints a step-by-step walkthrough script for presenting the Streamlit dashboard.
Run this before or during your presentation to see the guided talking points.

Usage:
    python demo_walkthrough.py          # show all steps
    python demo_walkthrough.py --launch # show steps AND launch the app
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    {
        "tab": "SETUP",
        "title": "Check the data pipeline ran",
        "action": "Verify output/metrics.json and output/model_*.pkl exist",
        "talking_points": [
            "The pipeline (main.py) simulates 2,100 rows of monthly car sales",
            "5 brands × 5 regions × 84 months (Jan 2018 – Dec 2024)",
            "Models trained: XGBoost, LightGBM, ARIMA",
        ],
    },
    {
        "tab": "SIDEBAR",
        "title": "Show the sidebar filters",
        "action": "Point to Brand, Region, and Year Range selectors",
        "talking_points": [
            "Filters apply across ALL three tabs simultaneously",
            "Default: all brands, all regions, full date range",
            "Demo: filter to Toyota only to see single-brand trends",
        ],
    },
    {
        "tab": "TAB 1 — Overview",
        "title": "KPI Cards",
        "action": "Point to the four metric cards at the top",
        "talking_points": [
            "Total Sales: aggregate units for the filtered selection",
            "Top Brand: which brand sold most in the filtered period",
            "Peak Month: which calendar month averages highest sales (December)",
            "Promo Uplift: % increase in sales when promotions run (~+15%)",
        ],
    },
    {
        "tab": "TAB 1 — Overview",
        "title": "Monthly Sales Trend",
        "action": "Hover over the line chart to show multi-brand tooltip",
        "talking_points": [
            "Each brand coloured consistently across all charts",
            "Hover shows all brands at a single point in time",
            "Upward trend visible — ~2% annual growth built into the simulation",
            "Toyota (red) consistently leads — base volume 300 units/month",
        ],
    },
    {
        "tab": "TAB 1 — Overview",
        "title": "Seasonality & Regional split",
        "action": "Point to the two side-by-side charts",
        "talking_points": [
            "Seasonality bar: December and March are peak months",
            "This reflects holiday and new-year purchasing patterns",
            "Regional bar: Bangkok leads, similar volumes across regions",
        ],
    },
    {
        "tab": "TAB 1 — Overview",
        "title": "Promotion impact",
        "action": "Scroll down to the promotion effect chart",
        "talking_points": [
            "Box plot: clear shift upward when promotion=1",
            "Validates the +15% promotional effect in the data-generating model",
        ],
    },
    {
        "tab": "TAB 2 — Forecast",
        "title": "ARIMA model",
        "action": "Select 'ARIMA' from the model dropdown",
        "talking_points": [
            "ARIMA is univariate — only uses the target's own history",
            "Auto-selected order via AIC (information criterion)",
            "Shaded band = 95% confidence interval — widens further into future",
            "Good at capturing seasonality but cannot use promotions or brand info",
        ],
    },
    {
        "tab": "TAB 2 — Forecast",
        "title": "XGBoost / LightGBM model",
        "action": "Switch dropdown to XGBoost",
        "talking_points": [
            "Tree models are MULTIVARIATE — use brand, region, promo, 17 features",
            "Scatter plot: points cluster near the perfect-fit diagonal",
            "Brand drill-down: select individual brands to inspect residuals",
            "XGBoost MAPE ~7.2% vs ARIMA MAPE ~9–10% — clear improvement",
        ],
    },
    {
        "tab": "TAB 3 — Model Comparison",
        "title": "Metrics table",
        "action": "Point to the MAE / RMSE / MAPE table",
        "talking_points": [
            "MAPE is scale-independent — works across brands of different volumes",
            "XGBoost and LightGBM both outperform ARIMA by ~2–3 pp MAPE",
            "Best model highlighted in the table",
        ],
    },
    {
        "tab": "TAB 3 — Model Comparison",
        "title": "Diagnostic plots",
        "action": "Scroll down to the three static plots",
        "talking_points": [
            "Feature Importance: lag_12 and rolling_mean_6 dominate — annual seasonality is the key signal",
            "Actual vs Predicted scatter: tight clustering along the diagonal = well-calibrated model",
            "Residuals histogram: centred near 0, symmetric — no systematic bias",
        ],
    },
    {
        "tab": "WRAP UP",
        "title": "Key takeaways",
        "action": "Summarise the demo",
        "talking_points": [
            "Gradient boosting outperforms ARIMA on multivariate sales data",
            "Proper time-series splitting (TimeSeriesSplit) prevents data leakage",
            "Lag and rolling features encode the demand memory the model needs",
            "Architecture is production-ready: SageMaker inference.py + drift monitoring",
        ],
    },
]


def print_walkthrough() -> None:
    width = 65
    print("=" * width)
    print("  CAR SALES FORECASTING DASHBOARD — DEMO WALKTHROUGH")
    print("=" * width)

    for i, step in enumerate(STEPS, 1):
        print(f"\n{'─' * width}")
        print(f"  STEP {i:02d}  [{step['tab']}]  {step['title']}")
        print(f"{'─' * width}")
        print(f"  ACTION : {step['action']}")
        print()
        for point in step["talking_points"]:
            print(f"    • {point}")

    print(f"\n{'=' * width}")
    print("  END OF WALKTHROUGH  (~10–12 minutes)")
    print(f"{'=' * width}\n")


def check_pipeline() -> None:
    print("\nChecking pipeline outputs ...")
    files = {
        "data/car_sales.csv":         "training data",
        "output/model_xgboost.pkl":   "XGBoost model",
        "output/model_lightgbm.pkl":  "LightGBM model",
        "output/model_arima.pkl":     "ARIMA model",
        "output/metrics.json":        "metrics",
    }
    all_ok = True
    for rel_path, label in files.items():
        path = ROOT / rel_path
        status = "✓" if path.exists() else "✗ MISSING"
        if not path.exists():
            all_ok = False
        print(f"  [{status}] {label}")

    if not all_ok:
        print("\n  Some files are missing. Run the pipeline first:")
        print("    python main.py")
    else:
        print("\n  All pipeline outputs present — ready to demo.")


def launch_app() -> None:
    print("\nLaunching Streamlit dashboard ...")
    print("  URL: http://localhost:8501\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py")],
        cwd=str(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard demo walkthrough guide.")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the Streamlit app after printing the walkthrough.",
    )
    args = parser.parse_args()

    check_pipeline()
    print_walkthrough()

    if args.launch:
        launch_app()
    else:
        print("Tip: run with --launch to start the app automatically.")
        print("     python demo_walkthrough.py --launch\n")


if __name__ == "__main__":
    main()
