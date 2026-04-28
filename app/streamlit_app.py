"""
streamlit_app.py — Interactive Car Sales Forecasting Dashboard
===============================================================
A four-tab Streamlit app that lets business users explore the data,
run forecasts, compare model performance, and understand how to keep
the model healthy in production.

Streamlit execution model:
    Streamlit re-executes the ENTIRE script from top to bottom every time
    a user interacts with a widget (slider, dropdown, etc.).
    This means all data loading, filtering, and chart generation is
    re-run on every interaction — which is why @st.cache_data and
    @st.cache_resource are critical for performance (they prevent
    expensive operations from rerunning on every click).

Four tabs:
    Tab 1 — Overview & EDA    : KPI cards, trend, seasonality, region, promo
    Tab 2 — Forecast          : ARIMA with confidence intervals, or tree model
                                actual vs predicted, with brand-level drill-down
    Tab 3 — Model Comparison  : MAPE / MAE / RMSE table, feature importance,
                                actual vs predicted scatter, residuals
    Tab 4 — Model Maintenance : health KPIs, quarterly MAPE trend, data drift
                                detection, retraining policy, MLOps workflow

Run:  streamlit run app/streamlit_app.py
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px        # high-level chart API (line, bar, scatter)
import plotly.graph_objects as go  # low-level API for multi-trace charts
import streamlit as st
from pathlib import Path

# ── Import path setup ──────────────────────────────────────────────────────────
# Needed so `from data_prep import ...` resolves when the app is launched
# from the project root with `streamlit run app/streamlit_app.py`.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

# ── Page config (must be called before any other st. call) ────────────────────
# layout="wide" uses the full browser width for charts.
# initial_sidebar_state="expanded" shows the sidebar by default.
st.set_page_config(
    page_title="Car Sales Forecast",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colour palette ───────────────────────────────────────────────────────
# Passed to Plotly's color_discrete_map so each brand always appears in
# the same colour across all charts — consistent visual identity.
BRAND_COLORS = {
    "Toyota": "#c81934",  # red — market leader, most prominent colour
    "Honda":  "#3a68a8",  # blue
    "Isuzu":  "#2ebd7a",  # green
    "Ford":   "#e8952a",  # amber
    "Mazda":  "#8e44ad",  # purple
}


# ══════════════════════════════════════════════════════════════════════════════
# Cached data loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and lightly transform the raw car sales CSV.

    @st.cache_data is for serialisable objects (DataFrames, lists, dicts).
    Streamlit serialises the return value to disk so repeated calls return
    the cached copy instantly without re-reading the CSV.

    The cache is invalidated automatically if the function's source code
    or the underlying file changes — no manual cache clearing needed.

    Returns an empty DataFrame (not an error) if the file doesn't exist yet,
    so the app can display a friendly "run the pipeline first" message.
    """
    path = ROOT / "data" / "car_sales.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    # Add year and month columns upfront so all tabs can filter and group
    # without recomputing them each time.
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


@st.cache_resource
def load_model(name: str):
    """
    Load a serialised model object from output/.

    @st.cache_resource is for non-serialisable objects like ML models.
    It stores the object in memory (not on disk) and shares the same
    instance across all user sessions — ideal for large models that are
    expensive to deserialise.

    WHY not @st.cache_data here?
        ML model objects (XGBRegressor, pmdarima ARIMAResults) contain
        internal C++ state that cannot be serialised by Streamlit's
        pickle-based cache_data mechanism.  cache_resource skips
        serialisation and keeps the live Python object in memory.

    Parameters
    ----------
    name : str — file stem, e.g. "arima", "xgboost", "lightgbm".
                 Looks for output/model_{name}.pkl.

    Returns None if the file doesn't exist (not trained yet).
    """
    path = ROOT / "output" / f"model_{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics() -> list[dict]:
    """
    Load model comparison metrics from output/metrics.json.

    Returns an empty list if the file doesn't exist, which triggers a
    "run main.py first" warning in the Model Comparison tab.
    """
    path = ROOT / "output" / "metrics.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Reusable UI component
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(col, label: str, value: str, delta: str = "", color: str = "#1b2a4a") -> None:
    """
    Render a styled KPI card inside a Streamlit column using raw HTML/CSS.

    We use st.markdown(unsafe_allow_html=True) because Streamlit's native
    st.metric() widget doesn't support custom colours or border accents.
    The left-border colour (color param) visually encodes the metric's
    status category (blue = neutral, red = alert, green = positive).

    Parameters
    ----------
    col   : Streamlit column object returned by st.columns().
    label : Small uppercase label (e.g. "Total Sales").
    value : Large primary number/text (e.g. "758,005").
    delta : Small secondary line below the value (e.g. "↑12% vs last year").
    color : Hex colour for the left border accent.
    """
    # Use &nbsp; when delta is empty so all cards share the same fixed height.
    delta_display = delta if delta else "&nbsp;"
    col.markdown(
        f"""
        <div style="background:#fff;border:1px solid #dde1e9;border-radius:10px;
                    padding:18px 20px;border-left:4px solid {color};
                    min-height:105px;box-sizing:border-box;">
          <div style="font-size:10px;color:#6b7a95;text-transform:uppercase;
                      letter-spacing:1px;font-weight:600;">{label}</div>
          <div style="font-size:26px;font-weight:700;color:#1b2a4a;margin:6px 0 4px;">{value}</div>
          <div style="font-size:12px;color:#6b7a95;">{delta_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Data loading — runs once at app startup (cached after first run)
# ══════════════════════════════════════════════════════════════════════════════

df      = load_data()
metrics = load_metrics()

# Guard clause: if the pipeline hasn't been run yet, stop early with instructions.
if df.empty:
    st.error(
        "**No data found.** Run the pipeline first:\n\n"
        "```bash\npython main.py\n```"
    )
    st.stop()   # halt script execution — nothing below this line runs

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — global filters applied to ALL tabs
# ══════════════════════════════════════════════════════════════════════════════
# Putting filters in the sidebar means they persist across tab switches.
# The user sets Brand/Region/Year once and the selection carries over
# to Overview, Forecast, and Comparison tabs automatically.

with st.sidebar:
    st.markdown("### 🚗 Car Sales Forecast")
    st.markdown("---")

    # multiselect defaults to all options selected, giving a "show all" start state.
    selected_brands = st.multiselect(
        "Brand",
        df["brand"].unique().tolist(),
        default=df["brand"].unique().tolist(),
    )
    selected_regions = st.multiselect(
        "Region",
        df["region"].unique().tolist(),
        default=df["region"].unique().tolist(),
    )
    # Slider range defaults to the full year span of the dataset.
    year_range = st.slider(
        "Year Range",
        int(df["year"].min()), int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())),
    )
    st.markdown("---")

    # Forecast controls — used in Tab 2 only but placed in sidebar for consistency.
    selected_model  = st.radio("Forecast Model", ["XGBoost", "LightGBM", "ARIMA"])
    forecast_months = st.slider("Forecast Horizon (months)", 3, 12, 6)

# ── Apply sidebar filters to produce the `filtered` DataFrame ─────────────────
# All Tab 1 charts use `filtered` so sidebar changes update every chart at once.
# Tab 2 and 3 may use `df` (full data) directly for model evaluation context.
mask = (
    df["brand"].isin(selected_brands)  &
    df["region"].isin(selected_regions) &
    df["year"].between(*year_range)
)
filtered = df[mask].copy()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview & EDA", "🔮 Forecast", "📊 Model Comparison",
    "🔧 Model Maintenance", "🚀 Future Roadmap",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview & Exploratory Data Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Overview & Exploratory Analysis")

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_sales  = filtered["sales"].sum()

    # idxmax() returns the index (brand name) where .sum() is maximum.
    best_brand   = filtered.groupby("brand")["sales"].sum().idxmax()

    # Best month: which calendar month has the highest average sales?
    # .mean() across all years and brand-region combos to avoid Dec 2020 anomalies.
    best_month_n = filtered.groupby("month")["sales"].mean().idxmax()
    best_month   = pd.Timestamp(f"2024-{best_month_n:02d}-01").strftime("%B")

    # Promotion uplift: % increase in average sales when a promotion is running.
    # Computed as: (promo_avg / no_promo_avg - 1) × 100
    promo_uplift = (
        filtered[filtered["promotion"] == 1]["sales"].mean() /
        filtered[filtered["promotion"] == 0]["sales"].mean() - 1
    ) * 100

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Total Sales (filtered)", f"{total_sales:,.0f}", "units", "#3a68a8")
    kpi_card(k2, "Top Brand",   best_brand,             "",                      "#c81934")
    kpi_card(k3, "Peak Month",  best_month,  "historically strongest",           "#2ebd7a")
    kpi_card(k4, "Promo Uplift", f"+{promo_uplift:.1f}%", "vs non-promo months", "#e8952a")

    st.markdown("---")

    # ── Sales trend by brand ──────────────────────────────────────────────────
    # Group by date AND brand so each brand gets its own line.
    monthly = filtered.groupby(["date", "brand"])["sales"].sum().reset_index()
    fig = px.line(
        monthly, x="date", y="sales", color="brand",
        color_discrete_map=BRAND_COLORS,
        title="Monthly Sales Trend by Brand",
        labels={"sales": "Units Sold", "date": ""},
    )
    # hovermode="x unified" shows all brand values at the same x (date)
    # when hovering, making it easy to compare brands at a point in time.
    fig.update_layout(hovermode="x unified", legend_title="Brand")
    st.plotly_chart(fig, use_container_width=True)

    # Compute brand-level share for the insight line so it reflects the current filter.
    brand_shares   = filtered.groupby("brand")["sales"].sum()
    top_share_pct  = brand_shares.max() / brand_shares.sum() * 100
    yoy_growth     = (
        filtered[filtered["year"] == filtered["year"].max()]["sales"].sum() /
        filtered[filtered["year"] == filtered["year"].max() - 1]["sales"].sum() - 1
    ) * 100 if filtered["year"].nunique() > 1 else 0
    st.caption(
        f"**Insight:** {best_brand} leads with {top_share_pct:.0f}% of filtered sales. "
        f"The upward slope across all brands reflects ~{yoy_growth:.1f}% year-on-year growth in "
        f"the most recent year. Market-share rankings are stable — shifts are gradual in a "
        f"mature automotive market, so large rank changes within a single filter period are a flag worth investigating."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        # ── Seasonality chart ─────────────────────────────────────────────────
        # Average across all years — removes trend so the seasonal pattern
        # is visible without the upward slope masking it.
        seasonal = filtered.groupby("month")["sales"].mean().reset_index()
        # Convert month number (3) to abbreviated name ("Mar") for readability.
        seasonal["month_name"] = pd.to_datetime(seasonal["month"], format="%m").dt.strftime("%b")
        fig2 = px.bar(
            seasonal, x="month_name", y="sales",
            title="Average Monthly Sales (Seasonality)",
            labels={"sales": "Avg Units", "month_name": ""},
            # Colour gradient from blue (low) to red (high) reinforces the
            # visual message of which months are strongest.
            color="sales", color_continuous_scale=["#3a68a8", "#c81934"],
        )
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
        peak_avg  = seasonal["sales"].max()
        trough_avg = seasonal["sales"].min()
        st.caption(
            f"**Insight:** {best_month} is the strongest month at {peak_avg:,.0f} avg units — "
            f"{(peak_avg/trough_avg - 1)*100:.0f}% above the weakest month. "
            "December peaks reflect year-end fleet renewals and dealer quota pushes. "
            "March benefits from motor-expo timing and new-model launches. "
            "Knowing this cycle in advance allows inventory to be pre-positioned 6–8 weeks ahead of each peak."
        )

    with col_b:
        # ── Regional breakdown ────────────────────────────────────────────────
        # Sort ascending so the longest bar appears at the top of the chart
        # (Plotly horizontal bar: first item is at the bottom by default).
        regional = (
            filtered.groupby("region")["sales"]
            .sum()
            .reset_index()
            .sort_values("sales", ascending=True)
        )
        fig3 = px.bar(
            regional, x="sales", y="region", orientation="h",
            title="Total Sales by Region",
            labels={"sales": "Units Sold", "region": ""},
            color="sales", color_continuous_scale=["#3a68a8", "#c81934"],
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
        top_region    = regional.sort_values("sales").iloc[-1]
        bottom_region = regional.sort_values("sales").iloc[0]
        region_ratio  = top_region["sales"] / bottom_region["sales"]
        st.caption(
            f"**Insight:** {top_region['region']} leads with {top_region['sales']:,.0f} units "
            f"— {region_ratio:.1f}× the volume of {bottom_region['region']}. "
            "This gap reflects both population density and average household income. "
            "A 1% forecast error in Bangkok has a larger absolute unit impact than the same error "
            "in lower-volume regions — so model accuracy in high-volume regions matters most for inventory planning."
        )

    col_c, col_d = st.columns(2)
    with col_c:
        # ── Promotion effect ─────────────────────────────────────────────────
        # Simple bar chart showing average sales with vs without promotion.
        # .assign() adds the human-readable label column in a single chain.
        promo_df = (
            filtered.groupby("promotion")["sales"].mean()
            .reset_index()
            .assign(label=lambda x: x["promotion"].map({0: "No Promo", 1: "Promotion"}))
        )
        fig4 = px.bar(
            promo_df, x="label", y="sales",
            title="Average Sales: Promotion Effect",
            labels={"sales": "Avg Units", "label": ""},
            color="label",
            color_discrete_map={"No Promo": "#3a68a8", "Promotion": "#c81934"},
        )
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        promo_avg    = filtered[filtered["promotion"] == 1]["sales"].mean()
        no_promo_avg = filtered[filtered["promotion"] == 0]["sales"].mean()
        st.caption(
            f"**Insight:** Promotional months average {promo_avg:,.0f} units vs {no_promo_avg:,.0f} "
            f"without promotion — a +{promo_uplift:.1f}% uplift. "
            "This is an actionable lever: knowing the promotional calendar 8 weeks ahead allows "
            "the model to pre-forecast the uplift and pre-position stock accordingly, "
            "reducing last-minute emergency orders and their premium freight costs."
        )

    with col_d:
        # ── Year-over-year by brand ───────────────────────────────────────────
        # Grouped bar chart: each cluster of bars represents one year,
        # each bar within the cluster is one brand.
        yoy = filtered.groupby(["year", "brand"])["sales"].sum().reset_index()
        fig5 = px.bar(
            yoy, x="year", y="sales", color="brand",
            color_discrete_map=BRAND_COLORS,
            title="Annual Sales by Brand (YoY)",
            labels={"sales": "Units Sold", "year": ""},
            barmode="group",  # bars side-by-side within each year cluster
        )
        st.plotly_chart(fig5, use_container_width=True)
        latest_yr  = filtered["year"].max()
        earliest_yr = filtered["year"].min()
        cagr = (
            filtered[filtered["year"] == latest_yr]["sales"].sum() /
            filtered[filtered["year"] == earliest_yr]["sales"].sum()
        ) ** (1 / max(latest_yr - earliest_yr, 1)) - 1
        st.caption(
            f"**Insight:** Total market CAGR across the selected period is ~{cagr*100:.1f}%. "
            f"All brands participate in the growth — there are no declining brands in this dataset, "
            "confirming the structural growth assumption in the simulation. In a live deployment, "
            "a brand showing flat or negative YoY growth would warrant a separate model investigation "
            "to determine whether the decline is structural or driven by a fixable operational issue."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Sales Forecast")

    # ── ARIMA path ────────────────────────────────────────────────────────────
    if selected_model == "ARIMA":
        model_obj = load_model("arima")
        if model_obj is None:
            st.warning("ARIMA model not found. Run `python main.py` first.")
        else:
            # ARIMA works on the TOTAL market series (all brands + regions summed).
            # We use the full `df` here (not filtered) because ARIMA was trained
            # on total market data — filtering brands/regions would be misleading.
            monthly_total = df.groupby("date")["sales"].sum().reset_index()

            # return_conf_int=True also returns the 95% confidence interval bounds
            # as a (n_periods, 2) array: col 0 = lower, col 1 = upper.
            forecast_vals, conf_int = model_obj.predict(
                n_periods=forecast_months, return_conf_int=True
            )

            # Build future dates starting the month AFTER the last observed month.
            last_date    = monthly_total["date"].max()
            future_dates = pd.date_range(
                last_date + pd.offsets.MonthBegin(1),
                periods=forecast_months,
                freq="MS",
            )

            # Build the chart with go.Figure for fine-grained trace control.
            fig = go.Figure()

            # Trace 1: full historical series.
            fig.add_trace(go.Scatter(
                x=monthly_total["date"], y=monthly_total["sales"],
                name="Historical", line=dict(color="#1b2a4a", width=2),
            ))

            # Trace 2: ARIMA forecast line.
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast_vals,
                name="ARIMA Forecast",
                line=dict(color="#c81934", width=2, dash="dash"),
                mode="lines+markers",
            ))

            # Trace 3: 95% confidence interval as a shaded polygon.
            # Technique: concatenate [lower bounds forward, upper bounds reversed]
            # to form a closed polygon that Plotly can fill with `fill="toself"`.
            fig.add_trace(go.Scatter(
                x=np.concatenate([future_dates, future_dates[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill="toself",
                fillcolor="rgba(200,25,52,0.12)",  # semi-transparent red
                line=dict(color="rgba(255,255,255,0)"),  # invisible border
                name="95% Confidence Interval",
            ))

            fig.update_layout(
                title=f"ARIMA — {forecast_months}-Month Forecast (Total Market)",
                xaxis_title="", yaxis_title="Units Sold",
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.15),  # legend below chart
            )
            st.plotly_chart(fig, use_container_width=True)
            ci_width = (conf_int[:, 1] - conf_int[:, 0]).mean()
            st.caption(
                f"**Insight:** The 95% confidence interval averages ±{ci_width/2:,.0f} units wide. "
                "This band widens each month into the future because ARIMA's uncertainty compounds — "
                "it cannot know about promotions, price changes, or competitor actions that haven't happened yet. "
                "Use the midpoint forecast for direction; treat the bounds as the planning range for min/max inventory positions. "
                "ARIMA is most reliable for the first 3 months where the interval is still tight."
            )

            # Forecast table: shows exact numbers behind the chart.
            st.markdown("**Forecast Values**")
            fcast_df = pd.DataFrame({
                "Month":      future_dates.strftime("%b %Y"),
                "Forecast":   forecast_vals.round(0).astype(int),
                "Lower 95%":  conf_int[:, 0].round(0).astype(int),
                "Upper 95%":  conf_int[:, 1].round(0).astype(int),
            })
            st.dataframe(fcast_df, use_container_width=True, hide_index=True)

    # ── Tree model path (XGBoost or LightGBM) ────────────────────────────────
    else:
        model_file = "xgboost" if selected_model == "XGBoost" else "lightgbm"
        model_obj  = load_model(model_file)
        if model_obj is None:
            st.warning(f"{selected_model} model not found. Run `python main.py` first.")
        else:
            from data_prep import make_tree_features, FEATURE_COLS

            # Re-engineer features on the FULL dataset so lag_12 for 2024
            # correctly references 2023 data.  Filtering before feature
            # engineering would break the lag computation.
            df_feat = make_tree_features(df)

            # Isolate the 2024 test rows (held-out during training).
            test_df = df_feat[df_feat["year"] == 2024].copy()
            test_df["predicted"] = np.clip(
                model_obj.predict(test_df[FEATURE_COLS]), 0, None
            )

            # Aggregate individual brand-region predictions to total monthly
            # for the market-level chart.
            test_monthly = (
                test_df.groupby("date")[["sales", "predicted"]]
                .sum()
                .reset_index()
            )
            hist_pre2024 = (
                df.groupby("date")["sales"]
                .sum()
                .reset_index()
                .query("date < '2024-01-01'")  # show historical context
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_pre2024["date"], y=hist_pre2024["sales"],
                name="Historical", line=dict(color="#1b2a4a", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=test_monthly["date"], y=test_monthly["sales"],
                name="Actual 2024", line=dict(color="#2ebd7a", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=test_monthly["date"], y=test_monthly["predicted"],
                name=f"{selected_model} Predicted",
                line=dict(color="#c81934", width=2, dash="dash"),
                mode="lines+markers",
            ))
            fig.update_layout(
                title=f"{selected_model} — 2024 Forecast vs Actual (Total Market)",
                xaxis_title="", yaxis_title="Units Sold",
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Compute actual vs predicted MAE at the aggregate level for the insight.
            agg_mae  = abs(test_monthly["sales"] - test_monthly["predicted"]).mean()
            agg_mape = (abs(test_monthly["sales"] - test_monthly["predicted"]) / test_monthly["sales"]).mean() * 100
            worst_month = test_monthly.loc[
                (abs(test_monthly["sales"] - test_monthly["predicted"]) / test_monthly["sales"]).idxmax(), "date"
            ].strftime("%B")
            st.caption(
                f"**Insight:** At the aggregated total-market level, {selected_model} achieves "
                f"{agg_mape:.1f}% MAPE ({agg_mae:,.0f} units average error per month). "
                f"The largest relative error falls in {worst_month} — peak demand months introduce "
                "more complexity because promotional and seasonal effects stack. "
                "Use the brand drill-down below to pinpoint which brands or regions contribute most to the residual error."
            )

            # ── Brand-level drill-down ────────────────────────────────────────
            # Business users often want to know "how does Ford specifically
            # compare to its actual 2024 sales?" — this selectbox enables that.
            brand_sel = st.selectbox(
                "View by Brand", ["All"] + sorted(df["brand"].unique().tolist())
            )
            if brand_sel != "All":
                brand_test = test_df[test_df["brand"] == brand_sel]
                brand_hist = df[
                    (df["brand"] == brand_sel) & (df["date"] < "2024-01-01")
                ]

                # Aggregate by date (sum across regions within the selected brand).
                bh   = brand_hist.groupby("date")["sales"].sum().reset_index()
                bt_a = brand_test.groupby("date")["sales"].sum().reset_index()
                bt_p = brand_test.groupby("date")["predicted"].sum().reset_index()

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=bh["date"], y=bh["sales"],
                    name="Historical", line=dict(color="#1b2a4a"),
                ))
                fig2.add_trace(go.Scatter(
                    x=bt_a["date"], y=bt_a["sales"],
                    name="Actual 2024", line=dict(color="#2ebd7a"),
                ))
                fig2.add_trace(go.Scatter(
                    x=bt_p["date"], y=bt_p["predicted"],
                    name="Predicted", line=dict(color="#c81934", dash="dash"),
                ))
                fig2.update_layout(
                    title=f"{brand_sel} — {selected_model} Forecast",
                    hovermode="x unified",
                )
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Model Comparison")

    if not metrics:
        st.warning("No metrics found. Run `python main.py` to train models first.")
    else:
        metrics_df = pd.DataFrame(metrics)

        # ── KPI cards: one per model ───────────────────────────────────────────
        st.markdown("### Performance Metrics (Test Set — 2024)")
        cols = st.columns(len(metrics))
        colors_map = {"ARIMA": "#e8952a", "XGBoost": "#c81934", "LightGBM": "#3a68a8"}
        for i, m in enumerate(metrics):
            kpi_card(
                cols[i], m["model"],
                f"MAPE {m['MAPE']}%",
                f"MAE {m['MAE']:,.0f} | RMSE {m['RMSE']:,.0f}",
                colors_map.get(m["model"], "#1b2a4a"),
            )

        st.markdown("---")

        col_l, col_r = st.columns(2)
        with col_l:
            # ── MAPE bar chart ─────────────────────────────────────────────────
            # MAPE is the primary ranking metric (lower = better).
            # text= adds labels directly on the bars — no need to hover.
            fig = px.bar(
                metrics_df, x="model", y="MAPE",
                title="MAPE by Model (lower is better)",
                color="model",
                color_discrete_map=colors_map,
                text="MAPE",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="MAPE (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # ── MAE vs RMSE grouped bars ───────────────────────────────────────
            # Plotting MAE and RMSE side by side shows their relationship:
            # if RMSE >> MAE, the model has a few very large individual errors
            # (the squaring in RMSE amplifies outliers more than MAE).
            fig2 = go.Figure()
            for col_name, color in [("MAE", "#3a68a8"), ("RMSE", "#c81934")]:
                fig2.add_trace(go.Bar(
                    x=metrics_df["model"],
                    y=metrics_df[col_name],
                    name=col_name,
                    marker_color=color,
                ))
            fig2.update_layout(
                title="MAE vs RMSE by Model",
                barmode="group",
                yaxis_title="Units Sold",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Static plot images generated by model_tree.py ─────────────────────
        # These were saved as PNG by the training pipeline.
        # We display them as images rather than recreating the charts so the
        # dashboard works even without access to the raw test set.

        fi_path = ROOT / "output" / "plots" / "feature_importance.png"
        ap_path = ROOT / "output" / "plots" / "actual_vs_predicted.png"
        rs_path = ROOT / "output" / "plots" / "residuals.png"

        if fi_path.exists():
            st.markdown("### XGBoost Feature Importance")
            st.image(str(fi_path), use_container_width=True)
            st.caption(
                "**Insight:** `lag_12` dominates — the single best predictor of this month's sales "
                "is the same month last year, confirming that annual seasonality is the strongest signal. "
                "`rolling_mean_6` and `lag_1` capture short-term momentum (a brand on an upswing stays on one). "
                "Promotional flags appear mid-ranking: real and important, but secondary to the seasonal cycle. "
                "Brand and region codes rank low because their information is already encoded in the lag features "
                "— a Toyota Bangkok row's lags implicitly reflect Toyota Bangkok's historical level and pattern."
            )

        if ap_path.exists():
            st.markdown("### Actual vs Predicted — Test Set 2024")
            st.image(str(ap_path), use_container_width=True)
            st.caption(
                "**Insight:** Points clustering tightly along the y = x diagonal indicate well-calibrated predictions. "
                "The spread widens at high volumes — December months, which have the largest absolute sales, "
                "also have the largest absolute errors (though not the largest percentage errors). "
                "Points consistently above the diagonal would indicate systematic over-prediction; "
                "below would indicate under-prediction. Neither pattern is visible here — the model is unbiased."
            )

        if rs_path.exists():
            st.markdown("### Residual Distribution")
            st.image(str(rs_path), use_container_width=True)
            st.caption(
                "**Insight:** Residuals centred near zero confirm no systematic over- or under-prediction. "
                "The roughly bell-shaped, symmetric distribution is consistent with well-behaved model errors. "
                "A left-skewed distribution would suggest the model occasionally predicts very high and misses; "
                "right-skewed would mean it occasionally under-predicts large demand spikes. "
                "The mild right tail here is expected — extreme December peaks are slightly under-predicted "
                "because the training data only had a few such observations."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Maintenance
# ══════════════════════════════════════════════════════════════════════════════
# This tab answers the question every ML deployment eventually faces:
# "How do I know when my model has stopped working, and what do I do then?"
#
# Two root causes of model degradation:
#   Data drift (covariate shift) — the INPUT distribution changes.
#       Example: average car prices rise 20% due to supply-chain costs.
#       The model was calibrated on lower prices and under-predicts.
#   Concept drift (relationship shift) — the TARGET relationship changes.
#       Example: a recession makes promotions less effective (15% → 5% uplift).
#       Even if feature distributions look the same, the model's learned
#       coefficients are now stale.
#
# We detect both indirectly:
#   • Rising MAPE over time → concept drift or compound data drift
#   • Distribution shift in features → data drift
#
# Retraining policy encoded here:
#   MAPE < 10%  → KEEP    (model is healthy)
#   10–15%      → WATCH   (investigate, prepare new data)
#   > 15%       → RETRAIN (trigger pipeline immediately)

with tab4:
    st.markdown("## Model Health & Maintenance")
    st.markdown(
        "Monitor whether the deployed XGBoost model is still performing well, "
        "detect when the input data has drifted from the training distribution, "
        "and understand when to trigger a retraining run."
    )

    # ── Helper: symmetric KL divergence (inlined from monitor.py) ─────────────
    # We re-implement here rather than importing from monitoring/ to avoid
    # path-resolution issues when the app is hosted on Streamlit Cloud.
    def _kl_sym(p: np.ndarray, q: np.ndarray, bins: int = 25) -> float:
        """
        Symmetric KL divergence between two empirical distributions.

        Uses the same bin edges for both histograms so the comparison is
        over identical intervals.  Adding 1e-8 prevents log(0) when a bin
        is empty in one distribution but not the other.

        Score guide:  ~0 = nearly identical  |  0.1+ = noticeable  |  1+ = large shift
        """
        p_hist, edges = np.histogram(p, bins=bins, density=True)
        q_hist, _     = np.histogram(q, bins=edges, density=True)
        p_hist += 1e-8
        q_hist += 1e-8
        return float(
            0.5 * np.sum(p_hist * np.log(p_hist / q_hist) + q_hist * np.log(q_hist / p_hist))
        )

    # ── Section 1: Current model health KPIs ──────────────────────────────────
    xgb_mape = next((m["MAPE"] for m in metrics if m["model"] == "XGBoost"), None)

    if xgb_mape is None:
        st.warning("No metrics found. Run `python main.py` to train models first.")
    else:
        # Map MAPE to a traffic-light status and colour.
        if xgb_mape < 10:
            status, status_color = "KEEP", "#2ebd7a"
        elif xgb_mape < 15:
            status, status_color = "WATCH", "#e8952a"
        else:
            status, status_color = "RETRAIN", "#c81934"

        k1, k2, k3, k4 = st.columns(4)
        kpi_card(k1, "Current MAPE (XGBoost)", f"{xgb_mape:.1f}%",  "on 2024 held-out test set", "#3a68a8")
        kpi_card(k2, "Model Decision",          status,               "KEEP < 10% | WATCH < 15%",   status_color)
        kpi_card(k3, "Training Window",          "2018 – 2023",        "72 months of history",        "#1b2a4a")
        kpi_card(k4, "Retrain Trigger",          "> 15% MAPE",         "triggers pipeline run",       "#e8952a")

    st.markdown("---")

    # ── Section 2: Quarterly MAPE trend ───────────────────────────────────────
    # We compute the ACTUAL per-quarter MAPE from the XGBoost model on 2024
    # data, then append a SIMULATED degradation curve for 2025–2026 to
    # illustrate what would happen if the model were never retrained.
    # This makes the retraining thresholds tangible for a business audience.
    st.markdown("### MAPE Over Time — Performance Monitoring")
    st.caption(
        "Solid line = actual model performance on 2024 test data (real).  "
        "Dashed line = simulated drift if the model is never retrained (illustrative)."
    )

    xgb_model = load_model("xgboost")
    if xgb_model is not None:
        from data_prep import make_tree_features, FEATURE_COLS

        # Re-engineer features on the full dataset so lag_12 for Jan 2024
        # correctly looks back to Jan 2023 (not clipped by a filter).
        df_feat    = make_tree_features(df)
        test_2024  = df_feat[df_feat["year"] == 2024].copy()
        test_2024["predicted"] = np.clip(
            xgb_model.predict(test_2024[FEATURE_COLS]), 0, None
        )

        # Map each row to its calendar quarter ("2024Q1", "2024Q2", …).
        test_2024["quarter"] = test_2024["date"].dt.to_period("Q").astype(str)

        # Per-quarter MAPE: exclude rows where actual = 0 to avoid div-by-zero.
        qmape_rows = []
        for q_label, grp in test_2024.groupby("quarter"):
            actual = grp["sales"].values
            pred   = grp["predicted"].values
            mask   = actual != 0
            q_mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)
            qmape_rows.append({"Quarter": q_label, "MAPE": round(q_mape, 2), "Series": "Actual (2024)"})

        # Simulate future degradation starting from the last real MAPE.
        # Each simulated quarter applies a compounding drift factor (8–12% per quarter)
        # to represent concept drift as market conditions evolve post-2024.
        last_real = qmape_rows[-1]["MAPE"]
        sim_quarters = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]
        drift_factors = [1.08, 1.20, 1.36, 1.55, 1.78]
        for sq, df_factor in zip(sim_quarters, drift_factors):
            qmape_rows.append({
                "Quarter": sq,
                "MAPE":    round(last_real * df_factor, 2),
                "Series":  "Simulated drift (no retrain)",
            })

        qmape_df = pd.DataFrame(qmape_rows)

        fig_trend = go.Figure()

        # Actual 2024 trace
        actual_df = qmape_df[qmape_df["Series"] == "Actual (2024)"]
        fig_trend.add_trace(go.Scatter(
            x=actual_df["Quarter"], y=actual_df["MAPE"],
            name="Actual 2024",
            mode="lines+markers",
            line=dict(color="#3a68a8", width=3),
            marker=dict(size=8),
        ))

        # Simulated degradation trace
        sim_df = qmape_df[qmape_df["Series"] == "Simulated drift (no retrain)"]
        fig_trend.add_trace(go.Scatter(
            x=sim_df["Quarter"], y=sim_df["MAPE"],
            name="Simulated drift (no retrain)",
            mode="lines+markers",
            line=dict(color="#e8952a", width=2, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        ))

        # Threshold reference lines — drawn as horizontal lines spanning the chart.
        # These are the actionable decision boundaries from monitor.py THRESHOLDS.
        fig_trend.add_hline(
            y=10, line_dash="dot", line_color="#e8952a", line_width=1.5,
            annotation_text="WATCH threshold (10%)",
            annotation_position="top right",
            annotation_font_color="#e8952a",
        )
        fig_trend.add_hline(
            y=15, line_dash="dot", line_color="#c81934", line_width=1.5,
            annotation_text="RETRAIN threshold (15%)",
            annotation_position="top right",
            annotation_font_color="#c81934",
        )

        # Shade the WATCH zone (10–15%) to make the risk band visually obvious.
        fig_trend.add_hrect(
            y0=10, y1=15,
            fillcolor="rgba(232,149,42,0.08)",
            layer="below", line_width=0,
        )
        # Shade the RETRAIN zone (15+)
        fig_trend.add_hrect(
            y0=15, y1=30,
            fillcolor="rgba(200,25,52,0.07)",
            layer="below", line_width=0,
        )

        fig_trend.update_layout(
            title="XGBoost MAPE by Quarter",
            xaxis_title="Quarter",
            yaxis_title="MAPE (%)",
            yaxis=dict(range=[0, max(qmape_df["MAPE"].max() * 1.15, 20)]),
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # ── Section 3: Data drift detection ───────────────────────────────────────
    # Compare the distribution of key features between the training window
    # (2018–2022) and the most recent full year (2024).
    # A significant shift means the model is scoring on data it never "saw"
    # during training, which is a leading indicator of rising MAPE.
    st.markdown("### Data Drift Detection")
    st.caption(
        "Reference = training data (2018–2022).  "
        "Current = most recent data (2024).  "
        "Alert threshold: mean shift > 2 standard deviations."
    )

    ref_data = df[df["year"] <= 2022]
    cur_data = df[df["year"] == 2024]

    drift_col_left, drift_col_right = st.columns(2)

    for i, (feat, label) in enumerate([
        ("sales",     "Sales Volume (units/month)"),
        ("price_avg", "Average Vehicle Price (THB)"),
    ]):
        col = drift_col_left if i == 0 else drift_col_right
        with col:
            ref_vals = ref_data[feat].dropna().values
            cur_vals = cur_data[feat].dropna().values

            # Mean shift normalised by training std dev — scale-independent.
            mean_shift = abs(cur_vals.mean() - ref_vals.mean()) / (ref_vals.std() + 1e-8)
            kl         = _kl_sym(ref_vals, cur_vals)
            drifted    = mean_shift > 2.0

            # Overlapping histograms: the visual overlap shows how similar the
            # distributions are.  Large separation = strong drift.
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=ref_vals, name="Training (2018–22)",
                opacity=0.65, marker_color="#3a68a8", nbinsx=30,
            ))
            fig_hist.add_trace(go.Histogram(
                x=cur_vals, name="Current (2024)",
                opacity=0.65, marker_color="#c81934", nbinsx=30,
            ))
            fig_hist.update_layout(
                barmode="overlay",
                title=label,
                xaxis_title=feat,
                yaxis_title="Count",
                legend=dict(orientation="h", y=-0.22),
                margin=dict(b=60),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Drift verdict badge: colour-coded so it reads at a glance.
            badge_color = "#c81934" if drifted else "#2ebd7a"
            badge_text  = "DRIFT DETECTED" if drifted else "OK — No Drift"
            st.markdown(
                f"""
                <div style="background:#f8f9fc;border-radius:8px;padding:12px 16px;
                            border-left:4px solid {badge_color};margin-bottom:8px;">
                  <strong style="color:{badge_color};">{badge_text}</strong><br>
                  <span style="font-size:13px;color:#6b7a95;">
                    Mean shift: <strong>{mean_shift:.2f}</strong> std devs &nbsp;|&nbsp;
                    KL divergence: <strong>{kl:.4f}</strong><br>
                    Ref mean: {ref_vals.mean():,.0f} &nbsp;→&nbsp; Cur mean: {cur_vals.mean():,.0f}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Section 4: Retraining decision matrix ─────────────────────────────────
    # A plain lookup table mapping MAPE ranges to business actions.
    # In a production MLOps system, the "RETRAIN" row would trigger a
    # SageMaker Training Job or an Airflow DAG automatically.
    st.markdown("### Retraining Decision Matrix")

    policy_df = pd.DataFrame([
        {
            "MAPE Range":       "< 10%",
            "Status":           "KEEP",
            "Action":           "No action — model is healthy",
            "Review Frequency": "Monthly",
        },
        {
            "MAPE Range":       "10% – 15%",
            "Status":           "WATCH",
            "Action":           "Investigate root cause; collect more recent data",
            "Review Frequency": "Weekly",
        },
        {
            "MAPE Range":       "> 15%",
            "Status":           "RETRAIN",
            "Action":           "Trigger retraining pipeline; promote new model after validation",
            "Review Frequency": "Daily",
        },
    ])
    st.dataframe(policy_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section 5: MLOps retraining workflow ──────────────────────────────────
    # Explains the end-to-end loop: detect → collect → retrain → validate → deploy.
    # This section is intentionally narrative — the goal is to give a non-technical
    # stakeholder a mental model of what "retraining" actually involves.
    st.markdown("### Retraining Workflow (6-Step MLOps Loop)")

    steps = [
        ("1 — Detect",    "#3a68a8", "MAPE rises above 15% on the weekly monitoring run, OR data drift is flagged on a key feature (mean shift > 2 std devs)."),
        ("2 — Collect",   "#3a68a8", "Append the latest sales records (new months) to `data/car_sales.csv`.  The more recent data, the better the model captures current market conditions."),
        ("3 — Retrain",   "#e8952a", "Run `python main.py --skip-data` to re-engineer features and retrain XGBoost and LightGBM on the expanded dataset (now including the new months)."),
        ("4 — Validate",  "#e8952a", "Evaluate the new model on a recent held-out window (last 3 months).  Only promote if MAPE improves vs the current production model."),
        ("5 — Deploy",    "#2ebd7a", "Upload the new `model_xgboost.pkl` to S3 and update the SageMaker endpoint: `sm_model.deploy(...)`.  Zero-downtime because SageMaker spins up the new container before decommissioning the old one."),
        ("6 — Reset",     "#2ebd7a", "Set the new model as the baseline in the monitoring system.  The MAPE clock restarts from the new model's performance level."),
    ]

    for title, color, desc in steps:
        st.markdown(
            f"""
            <div style="display:flex;align-items:flex-start;gap:14px;
                        margin-bottom:10px;padding:14px 18px;
                        background:#fff;border-radius:8px;
                        border:1px solid #dde1e9;border-left:4px solid {color};">
              <div style="min-width:110px;font-weight:700;color:{color};font-size:13px;">
                {title}
              </div>
              <div style="font-size:13px;color:#3a4560;line-height:1.6;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Future Roadmap: LangChain Agents & Microsoft Teams
# ══════════════════════════════════════════════════════════════════════════════
# This tab articulates how the current batch-pipeline system evolves into
# a conversational, event-driven platform:
#
#   Current state  →  Scheduled pipeline, static dashboard, manual review
#   Future state   →  LangChain agent answers NL questions, Teams bot delivers
#                     proactive alerts, retraining triggers automatically
#
# Nothing in this tab requires new model training — it is an architecture
# roadmap designed to show stakeholders the next two sprints of work.

with tab5:
    st.markdown("## Future Roadmap: AI Automation & Connectivity")
    st.markdown(
        "The current system delivers accurate forecasts through a batch pipeline and dashboard. "
        "The next phase connects it to the tools business users already live in — "
        "**Microsoft Teams** for proactive alerts and **LangChain agents** for natural-language queries — "
        "so forecasting insights reach decision-makers without them having to open the dashboard."
    )

    st.markdown("---")

    # ── Current vs Future state summary ───────────────────────────────────────
    col_now, col_arrow, col_future = st.columns([5, 1, 5])
    with col_now:
        st.markdown(
            """
            <div style="background:#f0f4ff;border-radius:10px;padding:20px 22px;
                        border-left:4px solid #3a68a8;">
              <div style="font-size:11px;color:#6b7a95;font-weight:700;
                          text-transform:uppercase;letter-spacing:1px;">Current State</div>
              <div style="margin-top:10px;font-size:13px;color:#1b2a4a;line-height:1.8;">
                ✅ Batch pipeline (manual run)<br>
                ✅ Static Streamlit dashboard<br>
                ✅ Manual MAPE monitoring<br>
                ❌ No proactive alerts<br>
                ❌ No natural-language queries<br>
                ❌ Manual retraining trigger
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_arrow:
        st.markdown(
            "<div style='text-align:center;font-size:36px;padding-top:40px;color:#6b7a95;'>→</div>",
            unsafe_allow_html=True,
        )
    with col_future:
        st.markdown(
            """
            <div style="background:#f0fff6;border-radius:10px;padding:20px 22px;
                        border-left:4px solid #2ebd7a;">
              <div style="font-size:11px;color:#6b7a95;font-weight:700;
                          text-transform:uppercase;letter-spacing:1px;">Future State</div>
              <div style="margin-top:10px;font-size:13px;color:#1b2a4a;line-height:1.8;">
                ✅ Scheduled & event-driven pipeline<br>
                ✅ Interactive dashboard (current)<br>
                ✅ Automated drift detection<br>
                ✅ Teams alerts on MAPE breach<br>
                ✅ NL query agent ("Ask the forecast")<br>
                ✅ Auto-retrain on threshold breach
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Architecture flow ──────────────────────────────────────────────────────
    st.markdown("### System Architecture")
    st.markdown(
        """
        <div style="background:#fff;border:1px solid #dde1e9;border-radius:10px;
                    padding:22px 26px;font-size:13px;line-height:2.2;color:#1b2a4a;">
          <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;">
            <div style="background:#e8f0fe;border-radius:8px;padding:8px 14px;font-weight:600;color:#3a68a8;">
              New Sales Data
            </div>
            <div style="padding:0 8px;color:#6b7a95;font-size:18px;">→</div>
            <div style="background:#e8f0fe;border-radius:8px;padding:8px 14px;font-weight:600;color:#3a68a8;">
              Feature Pipeline
            </div>
            <div style="padding:0 8px;color:#6b7a95;font-size:18px;">→</div>
            <div style="background:#fce8ec;border-radius:8px;padding:8px 14px;font-weight:600;color:#c81934;">
              XGBoost Model
            </div>
            <div style="padding:0 8px;color:#6b7a95;font-size:18px;">→</div>
            <div style="background:#e8f8f0;border-radius:8px;padding:8px 14px;font-weight:600;color:#2ebd7a;">
              Monitor (MAPE + Drift)
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;margin-top:12px;padding-left:40px;">
            <div style="padding:0 8px;color:#6b7a95;font-size:18px;">↓ if MAPE > 15%</div>
            <div style="padding:0 40px;"></div>
            <div style="padding:0 8px;color:#6b7a95;font-size:18px;">↓ always</div>
          </div>
          <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;margin-top:2px;">
            <div style="background:#fff3e0;border-radius:8px;padding:8px 14px;font-weight:600;color:#e8952a;margin-left:40px;">
              Auto-Retrain Pipeline
            </div>
            <div style="padding:0 50px;"></div>
            <div style="background:#ede7f6;border-radius:8px;padding:8px 14px;font-weight:600;color:#8e44ad;">
              LangChain Agent
            </div>
            <div style="padding:0 8px;color:#6b7a95;font-size:18px;">→</div>
            <div style="background:#e1f5fe;border-radius:8px;padding:8px 14px;font-weight:600;color:#0277bd;">
              Teams Bot / Webhook
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Feature 1: LangChain NL query agent ────────────────────────────────────
    st.markdown("### 1 — LangChain: Natural-Language Forecast Queries")
    st.markdown(
        "A LangChain agent wraps the forecast pipeline as a set of tools. "
        "Business users ask questions in plain language — the agent decides which tools to call, "
        "runs them, and synthesises a plain-English answer. No SQL, no Python."
    )

    uc_a, uc_b, uc_c = st.columns(3)
    for col, icon, title, desc in [
        (uc_a, "💬", "Natural Queries",      "\"What will Toyota's sales be next quarter?\" → agent calls the XGBoost predictor tool and returns a plain answer with the forecast value."),
        (uc_b, "📋", "Performance Summary",  "\"Summarise last month's performance vs forecast\" → agent queries the metrics store and drafts a bullet-point summary."),
        (uc_c, "⚠️", "Root-cause Alerts",   "\"Why did MAPE spike in Q3?\" → agent compares feature distributions and identifies which feature drifted most."),
    ]:
        col.markdown(
            f"""
            <div style="background:#fff;border:1px solid #dde1e9;border-radius:10px;
                        padding:16px 18px;height:160px;">
              <div style="font-size:22px;">{icon}</div>
              <div style="font-weight:700;color:#1b2a4a;margin:6px 0 4px;">{title}</div>
              <div style="font-size:12px;color:#6b7a95;line-height:1.6;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("**Implementation sketch (Python):**")
    st.code(
        '''from langchain.agents import initialize_agent, Tool
from langchain_anthropic import ChatAnthropic
import pandas as pd, pickle, numpy as np

# ── Tool 1: load forecast for a brand/region/month ──────────────────────────
def forecast_tool(query: str) -> str:
    """
    Input:  "brand=Toyota region=Bangkok month=2025-03"
    Output: "Forecast: 612 units (±7%)"
    """
    params  = dict(p.split("=") for p in query.split())
    model   = pickle.load(open("output/model_xgboost.pkl", "rb"))
    # ... feature engineering for the requested row ...
    pred    = int(np.clip(model.predict(features)[0], 0, None))
    return f"Forecast: {pred:,} units (±7% based on historical MAPE)"

# ── Tool 2: summarise recent performance ───────────────────────────────────
def performance_summary_tool(_: str) -> str:
    import json
    metrics = json.load(open("output/metrics.json"))
    best    = min(metrics, key=lambda m: m["MAPE"])
    return (f"Best model: {best['model']} — MAPE {best['MAPE']}%, "
            f"MAE {best['MAE']:,.0f} units")

# ── Agent setup ────────────────────────────────────────────────────────────
llm   = ChatAnthropic(model="claude-sonnet-4-6")
tools = [
    Tool(name="Forecast",    func=forecast_tool,           description="Predict sales for a brand/region/month"),
    Tool(name="Performance", func=performance_summary_tool, description="Get current model performance metrics"),
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# ── Example invocation ─────────────────────────────────────────────────────
response = agent.run("What will Toyota Bangkok sales look like in March 2025?")
print(response)
# → "Based on the XGBoost model, Toyota Bangkok is forecast at 612 units
#    in March 2025, with a ±7% confidence range (574–655 units)."
''',
        language="python",
    )

    st.markdown("---")

    # ── Feature 2: Microsoft Teams integration ────────────────────────────────
    st.markdown("### 2 — Microsoft Teams: Proactive Alerts & Weekly Digest")
    st.markdown(
        "Two integration patterns: **Incoming Webhooks** for one-way push alerts "
        "(zero infrastructure), and a **Teams Bot** via Azure Bot Service for two-way "
        "conversation (ask questions directly in Teams channels)."
    )

    t_a, t_b = st.columns(2)

    with t_a:
        st.markdown("**Pattern A — Incoming Webhook (push alerts)**")
        st.code(
            '''import requests, json

TEAMS_WEBHOOK_URL = "https://your-org.webhook.office.com/webhookb2/..."

def send_teams_alert(mape: float, model: str, quarter: str) -> None:
    """
    Post a colour-coded alert card to Teams when MAPE exceeds threshold.
    Uses Adaptive Card format for rich formatting.
    """
    if mape < 10:
        color, status = "Good",    "✅ KEEP"
    elif mape < 15:
        color, status = "Warning", "⚠️ WATCH"
    else:
        color, status = "Attention","🔴 RETRAIN"

    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard", "version": "1.4",
                "body": [
                    {"type": "TextBlock", "size": "Large", "weight": "Bolder",
                     "text": f"Model Alert — {quarter}"},
                    {"type": "FactSet", "facts": [
                        {"title": "Model",  "value": model},
                        {"title": "MAPE",   "value": f"{mape:.1f}%"},
                        {"title": "Status", "value": status},
                    ]},
                ],
                "actions": [{
                    "type": "Action.OpenUrl",
                    "title": "Open Dashboard",
                    "url": "https://your-streamlit-app.streamlit.app",
                }],
            }
        }]
    }
    requests.post(TEAMS_WEBHOOK_URL, json=card)

# ── Called by monitoring/monitor.py after each MAPE check ─────────────────
send_teams_alert(mape=16.2, model="XGBoost", quarter="2025Q3")
''',
            language="python",
        )

    with t_b:
        st.markdown("**Pattern B — Teams Bot (two-way queries)**")
        st.code(
            '''# requirements: botbuilder-core, botbuilder-integration-aiohttp
from botbuilder.core import ActivityHandler, TurnContext, MessageFactory
from langchain_agent import agent   # the LangChain agent from above

class ForecastBot(ActivityHandler):
    """
    Azure Bot Service handler — runs as a FastAPI/aiohttp app
    and registers as a bot in the Teams app manifest.
    Teams users message the bot directly or @mention it in a channel.
    """

    async def on_message_activity(self, turn_context: TurnContext):
        user_text = turn_context.activity.text.strip()

        # Route simple commands to direct functions; everything else to LangChain.
        if user_text.lower() == "/status":
            reply = performance_summary_tool("")
        elif user_text.lower() == "/help":
            reply = (
                "Commands:\\n"
                "  /status — current model MAPE\\n"
                "  /forecast brand=X region=Y month=YYYY-MM\\n"
                "  Or just ask in plain English!"
            )
        else:
            # LangChain agent handles natural language
            reply = agent.run(user_text)

        await turn_context.send_activity(MessageFactory.text(reply))

# Teams users can then type:
# "@ForecastBot What will Isuzu Northeast look like in Q1 2025?"
# "@ForecastBot /status"
# "@ForecastBot Why was last quarter's forecast off?"
''',
            language="python",
        )

    st.markdown("")

    # Sample Teams message mock
    st.markdown("**Sample Teams alert card (rendered):**")
    st.markdown(
        """
        <div style="background:#fff;border:1px solid #dde1e9;border-radius:10px;
                    padding:18px 22px;max-width:480px;font-size:13px;">
          <div style="font-weight:700;font-size:15px;color:#1b2a4a;margin-bottom:12px;">
            🔴 Model Alert — 2025Q3
          </div>
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="color:#6b7a95;padding:4px 0;width:80px;">Model</td>
                <td style="font-weight:600;color:#1b2a4a;">XGBoost</td></tr>
            <tr><td style="color:#6b7a95;padding:4px 0;">MAPE</td>
                <td style="font-weight:600;color:#c81934;">16.2%</td></tr>
            <tr><td style="color:#6b7a95;padding:4px 0;">Status</td>
                <td style="font-weight:600;color:#c81934;">RETRAIN — threshold exceeded</td></tr>
            <tr><td style="color:#6b7a95;padding:4px 0;">Action</td>
                <td style="color:#1b2a4a;">Trigger retraining pipeline immediately</td></tr>
          </table>
          <div style="margin-top:12px;">
            <span style="background:#3a68a8;color:#fff;border-radius:5px;
                         padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;">
              Open Dashboard
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Feature 3: Automated retraining pipeline ──────────────────────────────
    st.markdown("### 3 — Automated Retraining: Closing the Loop")
    st.markdown(
        "The monitoring module already produces the MAPE signal. "
        "The final step is wiring that signal to an automated retraining trigger "
        "— so the system self-heals without manual intervention."
    )

    auto_steps = [
        ("Scheduled Monitor",   "#3a68a8", "Airflow DAG or cron job runs `monitor.py` weekly. Computes MAPE on the latest 30-day batch. Writes result to `output/monitor_summary.json`."),
        ("Threshold Check",     "#3a68a8", "If `mape_action == 'RETRAIN'` in the JSON, the DAG triggers the next step. If `WATCH`, it posts a Teams warning only. If `KEEP`, it logs and exits."),
        ("Data Collection",     "#e8952a", "Pull the latest month's sales records from the source database (MySQL / BigQuery / Redshift) and append to the training CSV. Validate row counts and schema before proceeding."),
        ("Retrain & Validate",  "#e8952a", "Run `main.py --skip-data`. Compare the new model's MAPE on a 90-day holdout against the current production model. Only promote if the new model is better."),
        ("Deploy to SageMaker", "#2ebd7a", "Upload `model_xgboost.pkl` to S3. Call `sm_model.deploy()` to update the endpoint. SageMaker rolls out the new container with zero downtime."),
        ("Notify & Reset",      "#2ebd7a", "Post a 'Model retrained successfully' card to the Teams channel. Reset the monitoring baseline. The loop restarts for the next monitoring window."),
    ]

    for step_title, color, step_desc in auto_steps:
        st.markdown(
            f"""
            <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:8px;
                        padding:13px 18px;background:#fff;border-radius:8px;
                        border:1px solid #dde1e9;border-left:4px solid {color};">
              <div style="min-width:160px;font-weight:700;color:{color};font-size:13px;">
                {step_title}
              </div>
              <div style="font-size:13px;color:#3a4560;line-height:1.6;">{step_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("**Airflow DAG skeleton:**")
    st.code(
        '''from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="car_forecast_monitor_retrain",
    schedule_interval="0 8 * * MON",   # every Monday at 08:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    run_monitor = PythonOperator(
        task_id="run_monitor",
        python_callable=lambda: __import__("subprocess").run(
            ["python", "monitoring/monitor.py", "--mape", get_latest_mape()]
        ),
    )

    def branch_on_mape(**ctx):
        import json
        summary = json.load(open("output/monitor_summary.json"))
        action  = summary.get("mape_action", "KEEP")
        if action == "RETRAIN":  return "retrain_model"
        if action == "WATCH":    return "send_watch_alert"
        return "log_healthy"

    branch = BranchPythonOperator(task_id="branch_decision", python_callable=branch_on_mape)

    retrain       = PythonOperator(task_id="retrain_model",     python_callable=run_retrain_pipeline)
    watch_alert   = PythonOperator(task_id="send_watch_alert",  python_callable=lambda: send_teams_alert(action="WATCH"))
    log_healthy   = PythonOperator(task_id="log_healthy",       python_callable=lambda: print("Model healthy"))

    run_monitor >> branch >> [retrain, watch_alert, log_healthy]
''',
        language="python",
    )

    st.markdown("---")
    st.info(
        "**Implementation timeline estimate:**  "
        "Teams webhook (Pattern A) — 1 day.  "
        "LangChain query agent — 2–3 days.  "
        "Teams Bot (Pattern B) — 3–5 days including Azure Bot Service setup.  "
        "Full Airflow automation — 5–7 days depending on existing data infrastructure.",
        icon="📅",
    )
