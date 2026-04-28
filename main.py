"""
main.py — Full Pipeline Orchestrator
=====================================
Runs the three pipeline steps in sequence:
    Step 1 → generate_data.py   : simulate and save car_sales.csv
    Step 2 → model_arima.py     : fit ARIMA, save model + forecast plot
    Step 3 → model_tree.py      : fit XGBoost + LightGBM, save models + plots

After all models are trained, prints a comparison table and writes
output/metrics.json so the Streamlit dashboard can display results.

Design principle — orchestration vs execution:
    This file does NOT contain any ML logic.  It is a thin coordinator:
    it handles CLI flags, calls the execution modules in the right order,
    catches optional-dependency errors gracefully, and summarises results.
    All the actual work lives in the specialised src/ modules.

Usage:
    python main.py              # run all three steps
    python main.py --skip-data  # reuse existing car_sales.csv
    python main.py --skip-arima # skip ARIMA (faster iteration, trees only)
    python main.py --skip-data --skip-arima  # trees only on existing data
"""

import sys
import json
import argparse
from pathlib import Path

# ── Import path setup ──────────────────────────────────────────────────────────
# Prepend both source directories so the modules inside src/ and data/
# can be imported by name (e.g. `from model_arima import run_arima`).
# This is necessary because main.py lives at the project root, not inside
# src/ or data/.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

OUT   = ROOT / "output"
PLOTS = OUT / "plots"


# ── Printing utilities ─────────────────────────────────────────────────────────

def banner(text: str) -> None:
    """Print a prominent section header to make console output scannable."""
    print(f"\n{'='*55}")
    print(f"  {text}")
    print(f"{'='*55}")


def step(n: int, total: int, text: str) -> None:
    """Print a numbered step indicator, e.g. [2/3] Training ARIMA model."""
    print(f"\n[{n}/{total}] {text}")
    print("-" * 45)


def print_table(metrics: list[dict]) -> None:
    """
    Print a formatted comparison table of all model metrics.

    Annotates the best model (lowest MAPE) with a "← best" marker so it
    is immediately visible without the reader having to scan numbers.

    MAPE is chosen as the ranking criterion because it is scale-independent
    (works across brands of different volumes) and intuitive for non-technical
    stakeholders ("off by X percent on average").
    """
    print(f"\n{'Model':<12}  {'MAE':>8}  {'RMSE':>8}  {'MAPE':>7}")
    print("-" * 45)
    best_mape = min(m["MAPE"] for m in metrics)
    for m in metrics:
        marker = " ← best" if m["MAPE"] == best_mape else ""
        # Comma formatting (,) on MAE/RMSE makes large numbers readable
        # (e.g. 12,450 instead of 12450).
        print(f"{m['model']:<12}  {m['MAE']:>8,.0f}  {m['RMSE']:>8,.0f}  {m['MAPE']:>6.1f}%{marker}")


def main() -> None:
    """
    Parse CLI arguments and run the selected pipeline steps.

    The --skip-* flags exist for fast iteration during development:
    - Data generation is deterministic (fixed seed), so after the first run
      there is no need to regenerate unless you change generate_data.py.
    - ARIMA is the slowest step (~30 s) due to auto_arima's parameter search,
      so --skip-arima lets you iterate on the tree models in ~5 s.
    """
    parser = argparse.ArgumentParser(
        description="Run the car sales forecasting pipeline."
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation and use the existing data/car_sales.csv.",
    )
    parser.add_argument(
        "--skip-arima",
        action="store_true",
        help="Skip ARIMA model training (requires pmdarima; ~30 s).",
    )
    args = parser.parse_args()

    banner("Car Sales Demand Forecasting Pipeline")

    # Ensure output directories exist before any step tries to write there.
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    # Calculate total step count dynamically so the [n/total] indicators
    # are always accurate regardless of which flags are set.
    total  = 3 - int(args.skip_data) - int(args.skip_arima)
    step_n = 0  # incremented before each step

    # ── Step 1: Data generation ────────────────────────────────────────────────
    if not args.skip_data:
        step_n += 1
        step(step_n, total, "Generating simulated car sales data")
        from generate_data import generate   # import here so sys.path is ready
        generate()
    else:
        print("\n[--skip-data] Using existing data/car_sales.csv")

    all_metrics: list[dict] = []

    # ── Step 2: ARIMA ──────────────────────────────────────────────────────────
    if not args.skip_arima:
        step_n += 1
        step(step_n, total, "Training ARIMA model")
        try:
            from model_arima import run_arima
            arima_metrics = run_arima()
            all_metrics.append(arima_metrics)
        except ImportError:
            # pmdarima is optional — the pipeline degrades gracefully if missing.
            # The user still gets tree-model results and can install pmdarima later.
            print("[ARIMA] pmdarima not installed — skipping.")
            print("        Install with: pip install pmdarima")
    else:
        print("\n[--skip-arima] Skipping ARIMA")

    # ── Step 3: Tree-based models ──────────────────────────────────────────────
    # This step always runs (no --skip flag) because tree models are the
    # primary contribution of this demo and are fast to train (~10 s).
    step_n += 1
    step(step_n, total, "Training XGBoost & LightGBM models")
    try:
        from model_tree import run_trees
        tree_metrics = run_trees()
        all_metrics.extend(tree_metrics)   # extend adds both dicts to the list
    except ImportError as e:
        print(f"[Trees] Missing package — {e}")
        print("        Install with: pip install xgboost lightgbm")

    # ── Results & artefact saving ──────────────────────────────────────────────
    if all_metrics:
        banner("Model Comparison Results")
        print_table(all_metrics)

        # Persist metrics to JSON so the Streamlit dashboard can load them
        # without rerunning the training pipeline.
        out_path = OUT / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(all_metrics, f, indent=2)   # indent=2 → human-readable file
        print(f"\nMetrics saved  → {out_path}")
        print(f"Plots saved    → {PLOTS}/")

    # Final instructions — guide the user to the next step.
    print("\nDone! Launch the dashboard with:")
    print("  streamlit run app/streamlit_app.py\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
