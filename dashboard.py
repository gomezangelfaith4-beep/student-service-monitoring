"""Dashboard calculations that distinguish observed and predicted breaches."""
import pandas as pd

def dashboard_kpis(frame: pd.DataFrame, predictions=None) -> dict:
    total = int(len(frame))
    completed = int((frame["Status"].astype(str).str.lower() == "completed").sum()) if "Status" in frame else None
    observed = frame["SLA_Breached"].dropna() if "SLA_Breached" in frame else pd.Series(dtype=float)
    observed_rate = float(observed.mean() * 100) if len(observed) else None
    pending = int((frame["Status"].astype(str).str.lower() != "completed").sum()) if "Status" in frame else None
    predicted_breaches = None
    if predictions is not None and "Predicted_SLA_Breach" in predictions:
        predicted_breaches = int((predictions["Predicted_SLA_Breach"].astype(str) == "1").sum())
    return {"total_requests": total, "completed_requests": completed, "pending_or_noncompleted": pending,
            "labeled_requests": int(len(observed)), "observed_breach_rate_percent": observed_rate,
            "predicted_breaches": predicted_breaches}

def review_queue(frame: pd.DataFrame, predictions=None) -> pd.DataFrame:
    """Return a minimal queue view; do not expose satisfaction or other unnecessary fields."""
    cols = [c for c in ["Request_ID","Service_Type","Assigned_Office","Queue_Length","Queue_Category","Priority","Status"] if c in frame]
    result = frame[cols].copy()
    if predictions is not None and len(predictions) == len(result):
        for c in ["Predicted_SLA_Breach","Predicted_Breach_Probability","Review_Status"]:
            if c in predictions: result[c] = predictions[c].values
    elif "SLA_Breached" in frame:
        result["Review_Status"] = frame["SLA_Breached"].map({1:"Historical breach (observed)",0:"Historical no breach (observed)"})
    return result
