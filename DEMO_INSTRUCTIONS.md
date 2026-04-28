# Car Sales Demand Forecasting — Demo Project Plan

## Project Overview
A demo project simulating car sales forecasting using both statistical (ARIMA) and tree-based (XGBoost/LightGBM) approaches. Designed to showcase end-to-end ML workflow for interview/portfolio purposes.

---

## Project Structure
```
car_sales_forecast/
│
├── data/
│   └── generate_data.py        # Simulated car sales data generator
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py            # Feature engineering & preprocessing
│   ├── model_arima.py          # ARIMA model training & forecasting
│   └── model_tree.py           # XGBoost/LightGBM model training & forecasting
│
├── outputs/
│   ├── plots/                  # Saved charts and visualizations
│   └── metrics.json            # Model comparison results
│
├── app/
│   └── streamlit_app.py        # Interactive dashboard
│
├── monitoring/
│   └── monitor.py              # Drift monitoring
│
├── main.py                     # Run full pipeline
├── inference.py                # SageMaker entry point
├── requirements.txt
└── README.md
```

---

## Step 1: Simulate Car Sales Data

**File:** `data/generate_data.py`

### Data Spec
- **Period:** January 2018 – December 2024 (monthly)
- **Columns:**

| Column | Description |
|--------|-------------|
| `date` | Monthly timestamp |
| `sales` | Number of cars sold (target variable) |
| `brand` | Car brand (Toyota, Honda, Isuzu, Ford, Mazda) |
| `region` | Region (Bangkok, Central, North, Northeast, South) |
| `price_avg` | Average selling price (THB) |
| `promotion` | Promotion active this month (0/1) |
| `is_holiday_month` | Long holiday month like Dec, Apr (0/1) |
| `competitor_discount` | Competitor discount level (low/medium/high) |

### Simulation Rules (make data realistic)
- Add **upward trend** ~2% per year (car market growth)
- Add **yearly seasonality** (peak in Dec, Mar-Apr; dip in Feb, Sep)
- Add **random noise** for natural variation
- Promotions → boost sales 10-20%
- Holiday months → boost sales 15%

---

## Step 2: Exploratory Data Analysis (EDA)

**File:** `notebooks/eda.ipynb`

### Tasks
- [ ] Plot overall sales trend over time
- [ ] Decompose: Trend + Seasonality + Residual
- [ ] Plot sales by brand and region
- [ ] Correlation heatmap (features vs sales)
- [ ] Check stationarity with ADF test (for ARIMA)
- [ ] Identify top seasons and anomalies

---

## Step 3: Feature Engineering

**File:** `src/data_prep.py`

### For ARIMA
- Check & apply differencing if non-stationary
- Select p, d, q using ACF/PACF plots

### For Tree-Based Models
Create time-based features from `date`:
```python
df['lag_1']          = df['sales'].shift(1)
df['lag_3']          = df['sales'].shift(3)
df['lag_12']         = df['sales'].shift(12)   # Same month last year
df['rolling_mean_3'] = df['sales'].rolling(3).mean()
df['rolling_mean_6'] = df['sales'].rolling(6).mean()
df['month']          = df['date'].dt.month
df['quarter']        = df['date'].dt.quarter
df['year']           = df['date'].dt.year
df['is_year_end']    = (df['month'] == 12).astype(int)
```

---

## Step 4: Model 1 — ARIMA

**File:** `src/model_arima.py`

### Steps
1. Check stationarity → apply differencing if needed
2. Plot ACF / PACF → select p, d, q
3. Use `auto_arima` (pmdarima) to find best parameters automatically
4. Train on data up to 2023, forecast 2024
5. Plot forecast vs actual
6. Evaluate with MAE, RMSE, MAPE

```python
from pmdarima import auto_arima

model = auto_arima(train_sales, seasonal=True, m=12,
                   stepwise=True, suppress_warnings=True)
forecast = model.predict(n_periods=12)
```

---

## Step 5: Model 2 — Tree-Based (XGBoost)

**File:** `src/model_tree.py`

### Steps
1. Use engineered features from Step 3
2. Split: Train = 2018–2022, Validate = 2023, Test = 2024
3. Train XGBoost and LightGBM
4. Tune with cross-validation (TimeSeriesSplit — no data leakage!)
5. Plot feature importance
6. Evaluate with MAE, RMSE, MAPE

```python
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

tscv = TimeSeriesSplit(n_splits=5)
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)
```

> **Important:** Always use `TimeSeriesSplit` — never shuffle time series data!

---

## Step 6: Compare Models

**File:** `main.py` / `outputs/metrics.json`

### Comparison Table (target output)
| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | ... | ... | ...% |
| XGBoost | ... | ... | ...% |
| LightGBM | ... | ... | ...% |

### Visualizations to Generate
- [ ] Actual vs Predicted (all models on same chart)
- [ ] Residual plot
- [ ] Feature importance (XGBoost)
- [ ] Forecast for next 6 months (2025 H1)

---

## Step 7: README.md

Write a clean README covering:
- Problem statement
- Data description
- How to run (`python main.py`)
- Model results summary
- Key findings (which model wins and why)

---

## Step 8: Streamlit Dashboard

**File:** `app/streamlit_app.py`

### Page 1: Overview & EDA
- Sales trend line chart (interactive, Plotly)
- Filter by brand and region (sidebar dropdowns)
- Seasonality chart — average sales by month
- KPI cards: Total Sales, Best Month, Best Brand

### Page 2: Forecast
- User picks: Model / Brand / Forecast horizon (3, 6, 12 months)
- Show forecast chart vs historical
- Show confidence interval (for ARIMA)
- Show MAPE of selected model

### Page 3: Model Comparison
- Side-by-side metrics table (MAE, RMSE, MAPE)
- Actual vs Predicted chart — all models overlaid
- Feature importance bar chart (XGBoost)
- Residual distribution plot

```python
import streamlit as st
import plotly.express as px
import pandas as pd, joblib

st.set_page_config(page_title="Car Sales Forecast", layout="wide")

selected_brand  = st.sidebar.selectbox("Brand", ["All", "Toyota", "Honda", "Isuzu"])
selected_model  = st.sidebar.radio("Model", ["ARIMA", "XGBoost", "LightGBM"])
forecast_months = st.sidebar.slider("Forecast Horizon (months)", 3, 12, 6)

df    = pd.read_csv("data/car_sales.csv", parse_dates=["date"])
model = joblib.load(f"outputs/model_{selected_model.lower()}.pkl")

tab1, tab2, tab3 = st.tabs(["Overview", "Forecast", "Model Comparison"])

with tab1:
    fig = px.line(df, x="date", y="sales", color="brand")
    st.plotly_chart(fig, use_container_width=True)
```

```bash
pip install streamlit plotly
streamlit run app/streamlit_app.py
```

---

## Step 9: Deploy to AWS SageMaker

### 9.1 Prepare Model Artifacts
```python
import joblib, tarfile
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model.pkl")
    tar.add("scaler.pkl")
```

### 9.2 Write inference.py
```python
import joblib, json, numpy as np

def model_fn(model_dir):
    return {"model":  joblib.load(f"{model_dir}/model.pkl"),
            "scaler": joblib.load(f"{model_dir}/scaler.pkl")}

def input_fn(request_body, content_type):
    return np.array(json.loads(request_body)["features"])

def predict_fn(input_data, model_dict):
    scaled = model_dict["scaler"].transform(input_data)
    return model_dict["model"].predict(scaled)

def output_fn(prediction, accept):
    return json.dumps({"prediction": prediction.tolist()})
```

### 9.3 Upload to S3 & Deploy
```python
import boto3, sagemaker
from sagemaker.sklearn import SKLearnModel

s3 = boto3.client("s3")
s3.upload_file("model.tar.gz", "my-bucket", "car-forecast/model.tar.gz")

sm_model = SKLearnModel(
    model_data        = "s3://my-bucket/car-forecast/model.tar.gz",
    role              = "arn:aws:iam::XXXX:role/SageMakerRole",
    entry_point       = "inference.py",
    framework_version = "1.2-1"
)
predictor = sm_model.deploy(initial_instance_count=1, instance_type="ml.m5.large")
```

### 9.4 Deployment Options
| Option | Use Case | Cost |
|--------|----------|------|
| **Real-time Endpoint** | Live app, instant prediction | Higher (always on) |
| **Serverless Inference** | Infrequent / demo | Low (pay per call) |
| **Batch Transform** | Predict on large CSV | Medium |
| **Async Inference** | Large input, can wait | Low-Medium |

> For demo/portfolio — use **Serverless Inference**

---

## Step 10: Maintenance & Monitoring

### 10.1 Two Types of Drift
- **Data Drift** — input distribution changes (e.g. avg price shifts)
- **Concept Drift** — relationship between features and target changes

### 10.2 Monitoring with Evidently AI
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
report.run(reference_data=train_df, current_data=new_df)
report.save_html("monitoring/drift_report.html")
```

### 10.3 Alert Thresholds
| Metric | Alert When | Action |
|--------|-----------|--------|
| MAPE on new data | > 15% | Investigate & retrain |
| Feature mean shift | > 2 std dev | Check data pipeline |
| Endpoint latency | > 500ms | Scale up instance |
| Error rate (5xx) | > 1% | Check logs immediately |

### 10.4 Retraining Strategy
```
MAPE < 10%   → Keep model
MAPE 10-15%  → Watch closely
MAPE > 15%   → Trigger retrain automatically
```

### 10.5 MLflow Tracking
```python
import mlflow
mlflow.set_experiment("car_sales_forecast")
with mlflow.start_run(run_name="xgboost_v2"):
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("mape", mape_score)
    mlflow.sklearn.log_model(model, "xgboost_model")
```

---

## Full Tech Stack
```
pandas, numpy, scikit-learn, xgboost, lightgbm, pmdarima
streamlit, plotly
boto3, sagemaker
evidently
mlflow
```

---

## Interview Talking Points

> "I built an end-to-end forecasting pipeline for car sales — from data simulation and model training, to a Streamlit dashboard for business users. For deployment, I packaged the model with inference.py and deployed it as a SageMaker endpoint. Post-deployment, I set up Model Monitor to track data drift weekly and automated retraining when MAPE exceeded 15%, so the model stays accurate as market conditions change."
