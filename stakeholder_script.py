"""
stakeholder_script.py — Spoken Presentation Script
====================================================
A word-for-word script for a 15-minute stakeholder presentation of the
Car Sales Demand Forecasting project.

Sections
--------
  0  Opening hook                    ~1 min
  1  Problem & business context      ~2 min
  2  Data & modelling approach       ~2 min
  3  Model results                   ~2 min
  4  Dashboard walkthrough           ~5 min  (live demo)
  5  Model maintenance               ~1.5 min
  6  Business value & conclusion     ~1 min
  Q  Q&A preparation                 (reference only)

Usage
-----
  python stakeholder_script.py              # print full script
  python stakeholder_script.py --section 4  # print one section only
  python stakeholder_script.py --qa         # print Q&A prep only
  python stakeholder_script.py --launch     # print script + open dashboard
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────────────
# Presentation content
# Each section: title, duration, stage_note (what to DO), script (what to SAY)
# ──────────────────────────────────────────────────────────────────────────────

SECTIONS = [
    {
        "id": 0,
        "title": "Opening Hook",
        "duration": "~1 min",
        "stage": "Stand at the front. Dashboard is open on Tab 1. Do NOT start by introducing yourself — open with the hook.",
        "script": """
Good [morning / afternoon].

I want to start with a question: if a dealership manager is sitting down on the first
of the month to decide how many cars to order, what number does she use?

Historically, the answer has been gut feel, last year's number, or a spreadsheet
with a single trend line.  The problem with all three is that they ignore the
interactions that actually drive demand — promotions running in the same month,
which region the stock is going to, how the competitor's discount changed.

This project builds and demonstrates a system that replaces that gut feel with
a machine learning forecast accurate to within 7 percent.  Let me show you how.
        """,
    },
    {
        "id": 1,
        "title": "Problem & Business Context",
        "duration": "~2 min",
        "stage": "Stay standing. No dashboard interaction yet. This is context-setting.",
        "script": """
The business problem is demand forecasting for a multi-brand automotive group
operating across five regions in Thailand.

Why does this matter financially?

  Over-ordering: Excess inventory ties up working capital.  A car sitting on
  the lot for 90 days costs roughly 1.5% of its value in financing alone.
  At an average price of 750,000 baht, that is 11,000 baht per unit, per month.

  Under-ordering: Stockouts mean lost sales and lost margin — there is no
  second chance once a customer walks out and buys from a competitor.

A forecast that is even 3 percentage points more accurate across 2,100
brand-region-month combinations translates directly into fewer overstocked
vehicles and fewer missed sales.

The secondary challenge is that simple methods break down at scale.  ARIMA,
the classical time-series model, can only see the historical sales of a single
series.  It cannot know that a promotion is running, or that the Bangkok region
historically outperforms the Northeast by 75%.  We need a multivariate model.
        """,
    },
    {
        "id": 2,
        "title": "Data & Modelling Approach",
        "duration": "~2 min",
        "stage": "Click to Tab 1, Overview. Let them see the data while you speak.",
        "script": """
[Switch to Tab 1 — Overview & EDA]

The dataset covers January 2018 through December 2024 — 84 months —
across five brands: Toyota, Honda, Isuzu, Ford, and Mazda, and five
regions, giving us 2,100 brand-region-month observations.

The data follows a multiplicative demand model.  Sales in any month equals
a brand-region baseline, scaled by a long-run trend, a seasonal factor,
a promotional multiplier, and noise.

[Point to the seasonality chart]

You can see the seasonal pattern clearly here — December and March are peak
months, driven by year-end fleet renewals and new-model launches.

[Point to the promotion bar chart]

And here is the promotion effect.  Months with an active promotion average
around 15% more units than non-promotion months.  Any model that cannot
account for this will systematically under-predict during promotional periods.

We built and compared three models.  ARIMA, as our classical baseline.
Then XGBoost and LightGBM — gradient-boosted tree models — as our primary
approach, because tree models can consume all 17 features simultaneously and
capture non-linear interactions, such as the fact that a December promotion
is worth far more than an August promotion.
        """,
    },
    {
        "id": 3,
        "title": "Model Results",
        "duration": "~2 min",
        "stage": "Click to Tab 3 — Model Comparison. Stay here for the numbers.",
        "script": """
[Switch to Tab 3 — Model Comparison]

Let me give you the headline numbers.

ARIMA achieved a MAPE — that is, mean absolute percentage error — of 2.2%
on the aggregated total-market series.  That looks impressive.

XGBoost achieved a MAPE of 7.2%, and LightGBM matched at 7.2%.

Those numbers seem to say ARIMA wins, but there is a critical caveat:
ARIMA operates on one smooth series — all brands, all regions, summed.
It cannot tell you how many Toyotas to order for Bangkok versus how many
Isuzus to order for the Northeast.  That granularity requires XGBoost.

When we compare apples to apples — the granular brand-region level —
XGBoost's 7% error is the relevant benchmark.  It is off by about 28 units
per brand-region-month on a base of around 350 units.  That is operationally
precise enough to drive ordering decisions.

[Point to the feature importance chart]

The feature importance chart tells us what the model learned.
Lag 12 — sales from the same month last year — is the strongest predictor.
Annual seasonality is the dominant signal.  Rolling mean over six months
and the one-month lag come next, capturing short-term momentum.
Promotion is visible in the middle of the chart — confirming the model
has learned the uplift we saw in the EDA.
        """,
    },
    {
        "id": 4,
        "title": "Dashboard Walkthrough",
        "duration": "~5 min",
        "stage": "This is the live demo. Move through tabs 1 → 2 → 3 → (4 is next section). Keep interactions brief — hover, filter once, move on.",
        "script": """
[Switch to Tab 1 — Overview & EDA]

Let me walk you through the dashboard itself, because this is what a business
user would interact with day-to-day — no Python, no model code.

The sidebar on the left has three filters: brand, region, and year range.
These apply across every tab simultaneously.

[In the sidebar, deselect all brands, then re-select Toyota only]

If I filter to Toyota only, every chart on this page updates instantly —
the trend line, the seasonality profile, the regional split.

[Point to the regional bar chart]

Bangkok leads, as expected.  But the North and Northeast are not far behind
for Toyota, because they sell a large proportion of pickups to agricultural
customers — a segment Toyota dominates with the Hilux.

[Switch to Tab 2 — Forecast]

Tab 2 is the forecasting view.  I can choose ARIMA to see total-market
confidence intervals — the shaded band is the 95% confidence interval,
which widens as we project further into the future.

[Switch to XGBoost in the radio button]

Or I can switch to XGBoost, which shows actual 2024 performance versus
the model's predictions.  The green line is what actually happened.
The red dashed line is what the model forecast.  They track closely.

[Select "Toyota" from the brand dropdown]

I can drill down to individual brands.  Here is Toyota for 2024.
The model captures the December spike and the mid-year trough correctly.

[Switch sidebar back to All brands]

[Mention without clicking Tab 3]

Tab 3, which we already reviewed, has the side-by-side metric comparison
and the diagnostic plots — the residual histogram shows the errors are
symmetric and centred at zero, which tells us there is no systematic bias.
        """,
    },
    {
        "id": 5,
        "title": "Model Maintenance",
        "duration": "~1.5 min",
        "stage": "Switch to Tab 4 — Model Maintenance.",
        "script": """
[Switch to Tab 4 — Model Maintenance]

Building a model is only half the job.  The other half is keeping it honest
over time.  This tab answers the question every deployment eventually faces:
how do I know when my model has stopped working?

[Point to the MAPE trend chart]

The solid blue line shows the model's quarterly MAPE through 2024 — healthy,
well below the 10% watch threshold.

The dashed orange line is a simulation of what happens if we never retrain.
As market conditions drift — prices change, promotional effectiveness shifts,
new competitors enter — the MAPE climbs through the watch zone and eventually
crosses the 15% retrain threshold.

[Point to the drift histograms]

Below that, we are comparing the feature distributions from the training
period, 2018 to 2022, against the most recent data from 2024.  Where the
distributions overlap closely, the model is scoring data that looks like
what it was trained on.  A large divergence is an early warning signal.

[Point to the workflow cards]

And at the bottom, the six-step retraining workflow: detect the degradation,
collect the new data, retrain, validate, redeploy, reset the baseline.
In a production setup this loop can be largely automated with a scheduled
pipeline on AWS SageMaker or Apache Airflow.
        """,
    },
    {
        "id": 6,
        "title": "Business Value & Conclusion",
        "duration": "~1 min",
        "stage": "Close the dashboard or step away from the screen. Direct eye contact with the audience for the close.",
        "script": """
To summarise.

We replaced a univariate statistical model with a multivariate gradient
boosting system that is granular enough to drive actual ordering decisions
at the brand-region level.  It captures promotions, seasonality, regional
demand patterns, and year-over-year momentum simultaneously.

The pipeline is not a one-off analysis — it is a maintainable system.
The monitoring tab tells an operations team when to act before customers
notice the model has gone stale.

The practical implication is straightforward: a dealership group ordering
2,100 brand-region combinations per month with 7% forecast error instead of,
say, 15% intuition-based error, carries meaningfully less inventory risk,
fewer stockouts, and better cash conversion.

Thank you.  I am happy to take questions.
        """,
    },
]

QA_PREP = [
    {
        "q": "Why use simulated data instead of real dealership data?",
        "a": """Real dealership transaction data is proprietary and under NDA.
Simulation gives us something better for a methodology demonstration: known
ground truth.  We encoded a 15% promotion uplift, a 2% annual trend, and a
December peak.  We can then verify the model actually recovered those parameters
— which it did.  The same pipeline runs unchanged on real data when available.""",
    },
    {
        "q": "ARIMA had lower MAPE than XGBoost — doesn't that mean ARIMA is better?",
        "a": """No, and this is the most important methodological distinction.
ARIMA runs on the aggregate total-market series: one smooth number per month.
XGBoost runs on 2,100 individual brand-region series.  Aggregation removes
variance, so ARIMA's 2% MAPE on a smooth line is not comparable to XGBoost's
7% MAPE on granular series.  For operational ordering decisions, you need the
granular forecast.  ARIMA cannot tell you Toyota Bangkok versus Ford Northeast.""",
    },
    {
        "q": "How would you handle a new brand or new region that wasn't in the training data?",
        "a": """Tree models cannot extrapolate to categories they have never seen — this
is a known limitation.  The practical fix is a cold-start rule: for the first
6–12 months of a new brand/region, fall back to a simple rule-based baseline
(e.g. regional average for the brand tier) while collecting enough data for
the model to learn that series.  After 12 observations you have enough to
include it in the next retraining run.""",
    },
    {
        "q": "What is the cost of a wrong forecast? How do you quantify the business impact?",
        "a": """There are two asymmetric costs.  Over-forecasting causes overstock:
financing cost is roughly 1.5% per month of the vehicle's value, so a 750,000
baht car sitting unsold for a month costs ~11,000 baht.  Under-forecasting
causes stockouts: the lost margin on a mid-size sedan is roughly 30,000–50,000
baht.  Because stockout costs dominate, you should slightly bias forecasts
upward for high-margin models during peak season — this is a business calibration
layer on top of the model.""",
    },
    {
        "q": "Why XGBoost over a neural network or transformer for time series?",
        "a": """Three reasons.  First, tabular data with engineered features (lags, rolling
means) is where gradient boosting consistently matches or beats deep learning
— it is well documented in the Kaggle literature.  Second, interpretability:
feature importance from XGBoost is immediately actionable for a business
audience.  Third, training time: XGBoost trains in under 10 seconds here,
versus minutes to hours for a transformer, with no GPU required.  For this
data size and structure, XGBoost is the right tool.""",
    },
    {
        "q": "How production-ready is this, really?",
        "a": """The current version is a portfolio-grade proof of concept.  The inference
script (inference.py) implements the four-function SageMaker protocol and can
be deployed to a real endpoint with two commands.  The monitoring module runs
independently and outputs a JSON summary that could trigger an Airflow DAG.
To go live, you would need: real data pipeline replacing the CSV, IAM role
setup, and a CI/CD step to promote the model after validation passes.""",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

W = 68  # console width


def hr(char: str = "─") -> str:
    return char * W


def print_section(s: dict) -> None:
    print(f"\n{hr('═')}")
    print(f"  SECTION {s['id']}  ·  {s['title'].upper()}  ·  {s['duration']}")
    print(hr("═"))
    print(f"\n  [STAGE] {s['stage']}\n")
    print(hr())
    # Clean up indentation from the triple-quoted strings
    for line in s["script"].strip().split("\n"):
        print(line)
    print()


def print_qa() -> None:
    print(f"\n{hr('═')}")
    print("  ANTICIPATED Q&A — PREPARATION NOTES")
    print(hr("═"))
    for i, item in enumerate(QA_PREP, 1):
        print(f"\n  Q{i}: {item['q']}")
        print(f"  {hr('·')}")
        for line in item["a"].strip().split("\n"):
            print(f"  {line}")
    print()


def print_timing_overview() -> None:
    print(f"\n{hr('═')}")
    print("  PRESENTATION OVERVIEW  (target 15 min + 5 min Q&A)")
    print(hr("═"))
    for s in SECTIONS:
        bar = "█" * (int(s["duration"].replace("~", "").replace(" min", "").replace(".5", "")) * 2)
        print(f"  Sec {s['id']}  {s['duration']:<8}  {bar}  {s['title']}")
    print(f"\n  Q&A   ~5 min  {'█' * 10}  Open questions")
    print()


def launch_dashboard() -> None:
    print(f"\n{'─' * W}")
    print("  Launching dashboard at http://localhost:8501 ...")
    print(f"{'─' * W}\n")
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         str(ROOT / "app" / "streamlit_app.py"), "--server.headless", "true"],
        cwd=str(ROOT),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stakeholder presentation script.")
    parser.add_argument("--section", type=int, default=None,
                        help="Print one section only (0–6).")
    parser.add_argument("--qa",      action="store_true",
                        help="Print Q&A prep only.")
    parser.add_argument("--launch",  action="store_true",
                        help="Also launch the Streamlit dashboard.")
    args = parser.parse_args()

    if args.qa:
        print_qa()
        return

    if args.section is not None:
        matches = [s for s in SECTIONS if s["id"] == args.section]
        if not matches:
            print(f"Section {args.section} not found. Valid: 0–6.")
            sys.exit(1)
        print_section(matches[0])
        return

    # Full script
    print_timing_overview()
    for s in SECTIONS:
        print_section(s)
    print_qa()

    print(f"\n{hr('═')}")
    print("  END OF SCRIPT  ·  Remember to practise once out loud before presenting.")
    print(f"{hr('═')}\n")

    if args.launch:
        launch_dashboard()
    else:
        print("  Tip: add --launch to open the dashboard automatically.")
        print("       python stakeholder_script.py --launch\n")


if __name__ == "__main__":
    main()
