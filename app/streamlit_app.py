"""
streamlit_app.py — Interactive Car Sales Forecasting Dashboard
===============================================================
A three-tab Streamlit app that lets business users explore the data,
run forecasts, and compare model performance — all without touching code.

Streamlit execution model:
    Streamlit re-executes the ENTIRE script from top to bottom every time
    a user interacts with a widget (slider, dropdown, etc.).
    This means all data loading, filtering, and chart generation is
    re-run on every interaction — which is why @st.cache_data and
    @st.cache_resource are critical for performance (they prevent
    expensive operations from rerunning on every click).

Three tabs:
    Tab 1 — Overview & EDA    : KPI cards, trend, seasonality, region, promo
    Tab 2 — Forecast          : ARIMA with confidence intervals, or tree model
                                actual vs predicted, with brand-level drill-down
    Tab 3 — Model Comparison  : MAPE / MAE / RMSE table, feature importance,
                                actual vs predicted scatter, residuals

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
tab1, tab2, tab3 = st.tabs(["📈 Overview & EDA", "🔮 Forecast", "📊 Model Comparison"])


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
    st.plotly_chart(fig, use_container_width=True)  # fills the full column width

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
        fig2.update_layout(coloraxis_showscale=False)  # hide the colour legend bar
        st.plotly_chart(fig2, use_container_width=True)

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

        if ap_path.exists():
            st.markdown("### Actual vs Predicted — Test Set 2024")
            st.image(str(ap_path), use_container_width=True)

        if rs_path.exists():
            st.markdown("### Residual Distribution")
            st.image(str(rs_path), use_container_width=True)
