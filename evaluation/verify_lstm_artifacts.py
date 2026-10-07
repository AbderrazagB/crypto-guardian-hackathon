"""Reload the saved LSTM/scaler and check held-out predictions against the run."""

import hashlib
import json
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import joblib
import keras
import numpy as np
import pandas as pd
import torch


def main():
    root = Path(__file__).resolve().parent
    directory = root / "runs" / "lstm"
    if not torch.cuda.is_available():
        raise RuntimeError("The artifact replay must use the requested CUDA GPU.")
    torch.set_num_threads(4)
    payload = json.loads((root / "run-assets" / "lstm-history" / "ETH-USD.json").read_text())
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
    frame = pd.DataFrame({key.title(): values for key, values in quote.items()}, index=dates)
    frame = frame.loc[(frame.index >= "2018-01-01") & (frame.index < "2026-01-01")]
    features = json.loads((directory / "features.json").read_text())
    frame = frame[features].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    scaler = joblib.load(directory / "training_scaler.pkl")
    training = frame.loc[frame.index < "2023-01-01"]
    np.testing.assert_allclose(scaler.data_min_, training.min().to_numpy())
    np.testing.assert_allclose(scaler.data_max_, training.max().to_numpy())
    scaled = scaler.transform(frame.to_numpy()).astype("float32")
    X, actual, baseline, target_dates = [], [], [], []
    for end in range(30, len(frame)-7+1):
        if frame.index[end] < pd.Timestamp("2025-01-01"):
            continue
        X.append(scaled[end-30:end])
        actual.append(frame["Close"].iloc[end:end+7].to_numpy())
        baseline.append([frame["Close"].iloc[end-1]]*7)
        target_dates.append(frame.index[end:end+7].to_numpy())
    recorded = np.load(directory / "held_out_predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(target_dates, recorded["target_dates"])
    np.testing.assert_allclose(actual, recorded["actual"], rtol=0, atol=0.001)
    np.testing.assert_allclose(baseline, recorded["baseline"], rtol=0, atol=0)
    model = keras.models.load_model(directory / "eth_seven_day_lstm.keras", compile=False)
    assert model.weights[0].value.device.type == "cuda"
    predicted_scaled = model.predict(np.asarray(X), batch_size=256, verbose=0)
    index = features.index("Close")
    predicted = (predicted_scaled-scaler.min_[index])/scaler.scale_[index]
    np.testing.assert_allclose(predicted, recorded["predicted"], rtol=0, atol=0.005)
    summary = {
        "artifact_reload_passed": True, "test_windows": len(X),
        "max_absolute_replay_forecast_difference_usd": float(np.max(np.abs(predicted-recorded["predicted"]))),
        "max_actual_price_rounding_difference_usd": float(np.max(np.abs(actual-recorded["actual"]))),
        "training_only_scaler_verified": True,
        "device": str(model.weights[0].value.device),
        "saved_model_sha256": hashlib.sha256((directory / "eth_seven_day_lstm.keras").read_bytes()).hexdigest(),
    }
    (directory / "artifact_verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
