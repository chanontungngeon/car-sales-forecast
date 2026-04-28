"""
monitor.py — Data Drift & Performance Monitoring
=================================================
Compares a reference dataset (training data) against a current dataset
(new production data) to detect whether the model needs retraining.

Two types of drift to watch:

    Data Drift (input drift / covariate shift):
        The distribution of input features changes over time.
        Example: average car prices rise 15% due to supply-chain costs,
        so all price_avg values shift upward.  The model was trained on
        lower prices and may mis-predict.

    Concept Drift (model staleness):
        The relationship between inputs and the target changes.
        Example: economic recession causes promotions to be less effective
        (the 15% promo uplift drops to 5%).  Even if input distributions
        look the same, the model's learned relationships are now wrong.
        We detect concept drift indirectly via rising MAPE on new data.

Detection methods used here:
    1. Mean shift (standard deviations): simple, interpretable.
       Alert if the current mean drifts more than 2 std devs from training.
    2. Symmetric KL divergence: measures how much two distributions differ.
       Captures shape changes (skew, variance shifts) that mean-shift misses.
    3. Evidently AI (optional, richer): generates an HTML report with
       per-feature drift scores, statistical tests, and visualisations.

Retraining policy:
    MAPE < 10%   → model is healthy, keep it
    MAPE 10–15%  → watch closely, investigate root cause
    MAPE > 15%   → trigger automatic or manual retraining

Usage:
    python monitoring/monitor.py --ref data/car_sales.csv --cur data/new_data.csv
    python monitoring/monitor.py --cur data/new_data.csv --mape 12.5
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Import path fix ────────────────────────────────────────────────────────────
# This script lives in monitoring/, one level down from the project root.
# We need ROOT / "src" in sys.path to import data_prep.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_prep import load_raw

# ── Alert thresholds ───────────────────────────────────────────────────────────
# These are the decision boundaries that trigger action.
# Keeping them in one dict makes it easy to tune without hunting through code.
THRESHOLDS = {
    "mape_warn":     10.0,  # % — flag for investigation
    "mape_retrain":  15.0,  # % — trigger retraining workflow
    "feature_shift":  2.0,  # std devs — flag a feature as drifted
    "latency_ms":   500,    # ms — SageMaker endpoint P95 latency limit
}


# ══════════════════════════════════════════════════════════════════════════════
# Drift detection — manual (no external library required)
# ══════════════════════════════════════════════════════════════════════════════

def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """
    Compute the symmetric KL divergence between two numeric distributions.

    Standard KL divergence KL(P||Q) is asymmetric: KL(P||Q) ≠ KL(Q||P).
    The symmetric version averages both directions, making "reference vs
    current" and "current vs reference" give the same score — important
    when neither distribution is the "true" one.

    Formula:
        sym_KL(P, Q) = 0.5 * [KL(P||Q) + KL(Q||P)]
                     = 0.5 * Σ [p·log(p/q) + q·log(q/p)]

    Score interpretation:
        ~0.0  → distributions are nearly identical
        0.1+  → noticeable divergence
        1.0+  → substantially different distributions

    Why histogram-based?
        We approximate each distribution by binning it into 20 equal-width
        bins.  This is fast and works without assuming a parametric form
        (no assumption of normality).

    Parameters
    ----------
    p, q : np.ndarray
        Two arrays of numeric samples (e.g. training vs current sales).
    bins : int
        Number of histogram bins (20 is a practical default for ~100-10k samples).

    Returns
    -------
    float : Symmetric KL divergence score.
    """
    # Use the same bin edges for both distributions so we compare the same
    # intervals.  np.histogram returns (counts, edges); we discard counts
    # for p and reuse its edges for q.
    p_hist, edges = np.histogram(p, bins=bins, density=True)
    q_hist, _     = np.histogram(q, bins=edges, density=True)

    # Add a small epsilon to every bin to prevent log(0) = -infinity,
    # which would make the divergence undefined even if only one bin is empty.
    p_hist = p_hist + 1e-8
    q_hist = q_hist + 1e-8

    # Symmetric KL: average of KL(P||Q) and KL(Q||P).
    return float(
        0.5 * np.sum(p_hist * np.log(p_hist / q_hist) + q_hist * np.log(q_hist / p_hist))
    )


def check_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """
    Compare feature distributions between reference and current data.

    Runs two tests per numeric column:
        1. Mean shift in standard deviations — catches location drift.
        2. Symmetric KL divergence — catches shape / variance drift.

    A column is flagged as drifted if its mean shift exceeds the
    THRESHOLDS["feature_shift"] value (default: 2 std devs).

    Parameters
    ----------
    ref : pd.DataFrame — reference (training) dataset.
    cur : pd.DataFrame — current (production / new) dataset.

    Returns
    -------
    dict : Per-column drift report with ref_mean, cur_mean, shift, KL, drifted.
    """
    numeric_cols = ["sales", "price_avg"]
    results = {}

    for col in numeric_cols:
        if col not in ref.columns or col not in cur.columns:
            continue   # skip columns not present in both datasets

        ref_vals = ref[col].dropna().values
        cur_vals = cur[col].dropna().values

        # ── Mean shift ────────────────────────────────────────────────────────
        # Normalize the raw mean difference by the training std dev so the
        # threshold is on a comparable scale regardless of the column's
        # absolute magnitude (price_avg ~700,000 vs sales ~300).
        # +1e-8 prevents division by zero if ref has zero variance.
        mean_shift_std = abs(cur_vals.mean() - ref_vals.mean()) / (ref_vals.std() + 1e-8)

        kl_div  = kl_divergence(ref_vals, cur_vals)
        drifted = mean_shift_std > THRESHOLDS["feature_shift"]

        results[col] = {
            "ref_mean":       round(float(ref_vals.mean()), 2),
            "cur_mean":       round(float(cur_vals.mean()), 2),
            "mean_shift_std": round(mean_shift_std, 3),
            "kl_divergence":  round(kl_div, 4),
            "drifted":        drifted,
        }

        status = "DRIFT DETECTED" if drifted else "OK"
        print(f"  [{status}] {col}: mean shift = {mean_shift_std:.2f} std devs | KL = {kl_div:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Performance degradation check
# ══════════════════════════════════════════════════════════════════════════════

def check_mape(mape_val: float) -> str:
    """
    Compare current MAPE against retraining thresholds and return an action.

    This function implements the retraining policy.  In an automated MLOps
    pipeline, the returned action string would trigger downstream workflows:
        "KEEP"    → no action
        "WATCH"   → create a Jira ticket / Slack alert
        "RETRAIN" → trigger a SageMaker Training Job or Airflow DAG

    Parameters
    ----------
    mape_val : float — current model MAPE on recent data (as a percentage).

    Returns
    -------
    str : One of "KEEP", "WATCH", or "RETRAIN".
    """
    if mape_val < THRESHOLDS["mape_warn"]:
        action = "KEEP"
        msg    = "Model is healthy."
    elif mape_val < THRESHOLDS["mape_retrain"]:
        action = "WATCH"
        msg    = "Performance degrading — monitor closely and investigate root cause."
    else:
        action = "RETRAIN"
        msg    = "MAPE exceeded threshold — trigger retraining pipeline."

    print(f"  [MAPE={mape_val:.1f}%] → {action}: {msg}")
    return action


# ══════════════════════════════════════════════════════════════════════════════
# Evidently AI report (richer alternative to manual checks)
# ══════════════════════════════════════════════════════════════════════════════

def run_evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, out_path: Path) -> None:
    """
    Generate an Evidently drift report as an interactive HTML file.

    Evidently is a dedicated ML monitoring library that runs statistical
    tests (KS test, chi-squared, etc.) per feature and produces a
    self-contained HTML dashboard with drift scores and visualisations.
    It is more thorough than our manual KL-divergence check, but requires
    an extra dependency (pip install evidently).

    We wrap this in try/except so the rest of the monitoring script works
    even when Evidently is not installed — the manual drift checks above
    always run as a fallback.

    Parameters
    ----------
    ref      : pd.DataFrame — reference (training) data.
    cur      : pd.DataFrame — current (production) data.
    out_path : Path         — where to save the HTML report.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        # DataDriftPreset automatically selects the appropriate statistical
        # test for each column type (KS test for continuous, chi² for categorical).
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        report.save_html(str(out_path))
        print(f"  Evidently report saved → {out_path}")

    except ImportError:
        # Evidently is listed in requirements.txt as optional — don't crash.
        print("  [Evidently not installed] Run: pip install evidently")
    except Exception as e:
        # Catch any other Evidently error (e.g. schema mismatch) so it
        # doesn't abort the overall monitoring run.
        print(f"  [Evidently error] {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main — CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Parse CLI arguments and run the selected monitoring checks.

    Outputs:
        • Console report (always)
        • output/drift_report.html (if --cur is provided and Evidently installed)
        • output/monitor_summary.json (if any checks were run)
    """
    parser = argparse.ArgumentParser(
        description="Monitor car sales model for data drift and performance degradation."
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=ROOT / "data" / "car_sales.csv",
        help="Path to the reference (training) dataset CSV.",
    )
    parser.add_argument(
        "--cur",
        type=Path,
        default=None,
        help="Path to the current (new) dataset CSV for drift comparison.",
    )
    parser.add_argument(
        "--mape",
        type=float,
        default=None,
        help="Current model MAPE on new data (%). Triggers retrain advice if supplied.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Model Monitoring Report")
    print("=" * 50)

    # Always load and summarise the reference dataset.
    ref = load_raw(args.ref)
    print(
        f"\nReference data: {len(ref):,} rows  "
        f"({ref['date'].min().date()} – {ref['date'].max().date()})"
    )

    summary = {}

    # ── Drift check (only if --cur is provided) ────────────────────────────────
    if args.cur is not None and args.cur.exists():
        cur = load_raw(args.cur)
        print(
            f"Current data:   {len(cur):,} rows  "
            f"({cur['date'].min().date()} – {cur['date'].max().date()})"
        )

        print("\n── Data Drift Check ──────────────────────────────────")
        drift_results = check_drift(ref, cur)
        summary["drift"] = drift_results

        print("\n── Evidently Report ──────────────────────────────────")
        report_path = ROOT / "output" / "drift_report.html"
        run_evidently_report(ref, cur, report_path)
    else:
        print("\n[--cur not provided] Skipping drift check.")

    # ── Performance check (only if --mape is provided) ────────────────────────
    if args.mape is not None:
        print("\n── Performance Check ─────────────────────────────────")
        action = check_mape(args.mape)
        summary["mape_action"] = action

    # ── Save summary JSON ─────────────────────────────────────────────────────
    if summary:
        out = ROOT / "output" / "monitor_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved → {out}")

    # ── Print configured thresholds ───────────────────────────────────────────
    # Always shown so the operator can verify the current policy at a glance.
    print("\nConfigured alert thresholds:")
    for k, v in THRESHOLDS.items():
        unit = "%" if "mape" in k else ("std devs" if "shift" in k else "ms")
        print(f"  {k:<22} {v} {unit}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
