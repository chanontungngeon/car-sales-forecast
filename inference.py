"""
inference.py — AWS SageMaker Inference Entry Point
====================================================
SageMaker uses a strict four-function protocol to serve ML models.
When a prediction request arrives at the endpoint, SageMaker calls
these four functions in order:

    1. model_fn(model_dir)
           Called ONCE at container startup.
           Loads the model from disk and returns it.
           The return value is cached and passed to every subsequent call
           to predict_fn — you pay the disk I/O cost only at startup.

    2. input_fn(request_body, content_type)
           Called on EVERY request.
           Deserialises the raw HTTP request body into the format
           your model expects (here: a numpy array of feature rows).

    3. predict_fn(input_data, model)
           Called on EVERY request.
           Runs the actual prediction and returns raw output.

    4. output_fn(prediction, accept)
           Called on EVERY request.
           Serialises the prediction into the HTTP response body
           (here: a JSON string).

Why separate these four functions?
    SageMaker can swap out input_fn and output_fn independently for
    different content types (JSON, CSV, protobuf) without changing your
    model logic. This design follows the Single Responsibility Principle.

How to deploy:
    # 1. Package model artifacts
    import joblib, tarfile
    joblib.dump(xgb_model, "model_xgboost.pkl")
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("model_xgboost.pkl")

    # 2. Upload to S3
    import boto3
    boto3.client("s3").upload_file(
        "model.tar.gz", "my-bucket", "car-forecast/model.tar.gz"
    )

    # 3. Deploy endpoint
    from sagemaker.sklearn import SKLearnModel
    sm_model = SKLearnModel(
        model_data="s3://my-bucket/car-forecast/model.tar.gz",
        role="arn:aws:iam::XXXX:role/SageMakerRole",
        entry_point="inference.py",
        framework_version="1.2-1",
    )
    predictor = sm_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        # For demo/portfolio use Serverless (pay-per-call, no idle cost):
        # serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=2048)
    )

Invoke example (feature order must match FEATURE_COLS in data_prep.py):
    payload = {
        "features": [
            # brand_code, region_code, comp_disc_code,
            # price_avg, promotion, is_holiday_month,
            # month, quarter, year, is_year_end, is_q1,
            # lag_1, lag_3, lag_12,
            # rolling_mean_3, rolling_mean_6, rolling_std_3
            [0, 0, 1, 750000, 1, 0, 3, 1, 2025, 0, 1, 480, 460, 430, 465, 462, 12.5]
        ]
    }
    result = predictor.predict(payload)
    # → {"prediction": [512.0], "units": "car_units_sold"}
"""

import os
import json
import pickle
import numpy as np


def model_fn(model_dir: str) -> dict:
    """
    Load the trained XGBoost model (and optional scaler) from disk.

    SageMaker extracts model.tar.gz to /opt/ml/model/ and passes that
    path as model_dir.  We load whatever artifacts are present there.

    The scaler is optional — we trained XGBoost on raw feature values
    (tree models don't require feature scaling).  However, if a future
    version of the pipeline adds a scaler (e.g. for a neural network),
    model_fn can load it here without changing predict_fn's interface.

    Parameters
    ----------
    model_dir : str
        Directory where SageMaker has extracted the model.tar.gz archive.
        Locally this might be "output/" for testing.

    Returns
    -------
    dict : {"model": <XGBRegressor>, "scaler": <StandardScaler or None>}
    """
    model_path  = os.path.join(model_dir, "model_xgboost.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Load the XGBoost model — required; raise FileNotFoundError if missing.
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the scaler only if it exists — optional for tree models.
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return {"model": model, "scaler": scaler}


def input_fn(request_body: str, content_type: str = "application/json") -> np.ndarray:
    """
    Deserialise the raw HTTP request body into a numpy feature matrix.

    Expected JSON format:
        {"features": [[f1, f2, ..., f17], [f1, f2, ..., f17], ...]}

    The outer list supports batch requests (multiple rows in one call),
    which reduces per-request network overhead for bulk scoring.

    Parameters
    ----------
    request_body : str
        Raw HTTP request body as a string.
    content_type : str
        MIME type sent by the caller.  We only support "application/json".
        Raising ValueError here causes SageMaker to return HTTP 400.

    Returns
    -------
    np.ndarray : shape (n_rows, n_features), dtype float64
    """
    if content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: '{content_type}'. "
            "Send requests with Content-Type: application/json."
        )

    payload = json.loads(request_body)

    # Validate the expected key early so the error message is clear.
    if "features" not in payload:
        raise ValueError(
            "Request JSON must contain a 'features' key with a 2D list of "
            "feature rows, e.g. {\"features\": [[f1, f2, ...], ...]}."
        )

    # Convert to float64 numpy array.
    # dtype=float converts strings to numbers if the caller serialised them
    # as strings, and promotes integers to float for consistency with
    # how the model was trained.
    return np.array(payload["features"], dtype=float)


def predict_fn(input_data: np.ndarray, model_dict: dict) -> np.ndarray:
    """
    Run inference on the deserialised feature matrix.

    Applies an optional scaler before prediction.  Tree models (XGBoost,
    LightGBM) do not need feature scaling because splits are based on
    rank-ordering of values, not distances.  The scaler hook is included
    here to make the inference script reusable if the model changes.

    Parameters
    ----------
    input_data  : np.ndarray — shape (n_rows, n_features), from input_fn.
    model_dict  : dict       — {"model": ..., "scaler": ...}, from model_fn.

    Returns
    -------
    np.ndarray : Predicted sales units, clipped to >= 0.
    """
    # Apply scaling only when a scaler was found during model_fn.
    if model_dict["scaler"] is not None:
        input_data = model_dict["scaler"].transform(input_data)

    predictions = model_dict["model"].predict(input_data)

    # Sales cannot be negative.  Tree models occasionally predict tiny
    # negative values when trained data near the floor (min_sales=10)
    # creates leaf averages that dip below zero.
    predictions = np.clip(predictions, 0, None)
    return predictions


def output_fn(prediction: np.ndarray, accept: str = "application/json") -> str:
    """
    Serialise the prediction array into an HTTP response body.

    .tolist() converts the numpy array to a plain Python list, which is
    JSON-serialisable.  numpy arrays are NOT directly JSON-serialisable —
    json.dumps(np.array([1.0])) would raise a TypeError.

    The "units" field is metadata for the API consumer — it clarifies
    what the number means without requiring separate documentation.

    Parameters
    ----------
    prediction : np.ndarray — output from predict_fn.
    accept     : str        — response MIME type requested by the caller.

    Returns
    -------
    str : JSON-encoded response body.
    """
    return json.dumps({
        "prediction": prediction.tolist(),   # [512.0, 340.0, ...]
        "units": "car_units_sold",           # human-readable label for consumers
    })
