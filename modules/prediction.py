"""Saved-model loading and inference. No model is trained during app startup."""
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd
try:
    import streamlit as st
except ImportError:  # Allows unit tests of model helpers without the UI dependency installed.
    class _StreamlitStub:
        @staticmethod
        def cache_resource(func):
            return func
    st = _StreamlitStub()
from modules.data_processing import build_prediction_frame, MODEL_FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "trained_model.joblib"

@st.cache_resource
def load_model(model_path: str = str(MODEL_PATH)):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            "models/trained_model.joblib is missing. The uploaded midterm notebook contains training code, "
            "but no fitted model artifact was supplied. Add the original fitted pipeline, or explicitly run "
            "the documented reproducibility script after instructor approval."
        )
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise TypeError("The saved artifact does not expose predict(). Expected the fitted sklearn pipeline.")
    if hasattr(model, "feature_names_in_"):
        found = list(model.feature_names_in_)
        if found != MODEL_FEATURES:
            raise ValueError(f"Model feature schema mismatch. Expected {MODEL_FEATURES}; model reports {found}.")
    return model

def predict_one(model, values: dict) -> dict:
    row = build_prediction_frame(values)
    pred = model.predict(row)[0]
    result = {
        "prediction": int(pred) if str(pred).strip() in {"0","1"} or isinstance(pred, (int, bool)) else str(pred),
        "prediction_label": "Predicted SLA breach" if str(pred) == "1" else ("No breach predicted" if str(pred) == "0" else str(pred)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": getattr(model, "model_version", "Not recorded in artifact"),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            result["breach_probability"] = float(probabilities[classes.index(1)])
        result["class_probabilities"] = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    return result
