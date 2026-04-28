# Car Sales Demand Forecasting

End-to-end ML pipeline forecasting monthly car sales in Thailand using ARIMA and XGBoost/LightGBM. Includes a Streamlit dashboard, SageMaker deployment script, and drift monitoring.

---

## Problem Statement

Forecast monthly car unit sales per brand and region using 7 years of simulated historical data. The pipeline covers data generation, feature engineering, model training, evaluation, and an interactive dashboard for business users.

---

## Project Structure

```
project-forecast/
├── data/
│   ├── generate_data.py    # Simulate car sales data
│   └── car_sales.csv       # Generated dataset (2018–2024)
├── src/
│   ├── data_prep.py        # Feature engineering (shared)
│   ├── model_arima.py      # ARIMA model
│   └── model_tree.py       # XGBoost + LightGBM models
├── app/
│   └── streamlit_app.py    # Interactive 3-tab dashboard
├── monitoring/
│   └── monitor.py          # Data drift & performance monitoring
├── output/
│   ├── plots/              # Generated charts
│   ├── metrics.json        # Model comparison results
│   └── model_*.pkl         # Saved model artifacts
├── main.py                 # Run full pipeline
├── inference.py            # AWS SageMaker entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This runs all steps in order:
1. Generates `data/car_sales.csv` (2,100 rows — 5 brands × 5 regions × 84 months)
2. Fits ARIMA on aggregated monthly totals
3. Trains XGBoost and LightGBM with TimeSeriesSplit CV
4. Saves metrics to `output/metrics.json` and plots to `output/plots/`

```bash
# Skip data generation if already created
python main.py --skip-data

# Skip ARIMA (faster, tree models only)
python main.py --skip-arima
```

### 3. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Data Description

Simulated monthly car sales (Jan 2018 – Dec 2024) with realistic patterns:

| Column | Description |
|--------|-------------|
| `date` | Month start date |
| `brand` | Toyota / Honda / Isuzu / Ford / Mazda |
| `region` | Bangkok / Central / North / Northeast / South |
| `sales` | Units sold (target variable) |
| `price_avg` | Average selling price (THB) |
| `promotion` | Promotion active (0/1) |
| `is_holiday_month` | Long holiday month — Mar, Apr, Dec (0/1) |
| `competitor_discount` | Competitor discount level (low/medium/high) |

**Built-in patterns:** ~2% annual trend growth, monthly seasonality (Dec peak, Sep trough), 15% promotion uplift, 12% holiday uplift, 6% random noise.

---

## Models

### ARIMA (via pmdarima auto_arima)
- Operates on **total aggregated** monthly sales
- Seasonal ARIMA with m=12 (annual cycle)
- Train: 2018–2023, Test: 2024
- Outputs: forecast + 95% confidence interval

### XGBoost & LightGBM
- Operates on **brand × region** level (granular)
- Feature set: lag-1, lag-3, lag-12, rolling means, calendar, promotion, price
- Split: Train 2018–2022 | Validate 2023 | Test 2024
- Uses `TimeSeriesSplit(n_splits=5)` — no data leakage

---

## Results

After running `python main.py`, check `output/metrics.json`:

```json
[
  {"model": "ARIMA",    "MAE": ..., "RMSE": ..., "MAPE": ...},
  {"model": "XGBoost",  "MAE": ..., "RMSE": ..., "MAPE": ...},
  {"model": "LightGBM", "MAE": ..., "RMSE": ..., "MAPE": ...}
]
```

Tree-based models typically outperform ARIMA here because they leverage brand/region/promotion features that ARIMA cannot use.

---

## Dashboard (Streamlit)

**Tab 1 — Overview & EDA**
- Sales trend by brand (interactive Plotly)
- Seasonality bar chart, regional breakdown
- Promotion uplift and YoY comparison

**Tab 2 — Forecast**
- Select model, brand, and forecast horizon (3–12 months)
- ARIMA shows confidence intervals
- Tree models show 2024 actual vs predicted

**Tab 3 — Model Comparison**
- MAPE / MAE / RMSE side-by-side
- Feature importance chart (XGBoost)
- Actual vs Predicted and residual plots

---

## Monitoring

```bash
# Check drift between training data and new data
python monitoring/monitor.py --ref data/car_sales.csv --cur data/new_data.csv

# Also report current MAPE to get a retrain recommendation
python monitoring/monitor.py --cur data/new_data.csv --mape 12.5
```

| MAPE | Action |
|------|--------|
| < 10% | Keep model |
| 10–15% | Watch closely |
| > 15% | Trigger retraining |

---

## AWS SageMaker Deployment

```python
import boto3, sagemaker
from sagemaker.sklearn import SKLearnModel

# Package model artifacts
import tarfile, joblib
joblib.dump(xgb_model, "model_xgboost.pkl")
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model_xgboost.pkl")

# Upload to S3
boto3.client("s3").upload_file("model.tar.gz", "my-bucket", "car-forecast/model.tar.gz")

# Deploy
sm_model = SKLearnModel(
    model_data="s3://my-bucket/car-forecast/model.tar.gz",
    role="arn:aws:iam::XXXX:role/SageMakerRole",
    entry_point="inference.py",
    framework_version="1.2-1",
)
predictor = sm_model.deploy(initial_instance_count=1, instance_type="ml.m5.large")
```

For demo/portfolio — use **Serverless Inference** (pay per call, no idle cost).

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy |
| Models | scikit-learn, xgboost, lightgbm, pmdarima |
| Dashboard | streamlit, plotly |
| Monitoring | evidently, mlflow |
| Deployment | boto3, sagemaker |
