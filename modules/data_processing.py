"""CSV loading, validation, and safe feature preparation."""
from __future__ import annotations
from io import BytesIO, StringIO
import pandas as pd

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score", "Submission_Weekday",
    "Queue_Category", "SLA_Breached",
]
MODEL_FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Queue_Category",
    "Required_Documents_Complete", "SLA_Days", "Submission_Weekday",
]
CATEGORICAL_FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Category", "Required_Documents_Complete",
    "Submission_Weekday",
]
NUMERIC_FEATURES = ["Queue_Length", "SLA_Days"]

def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Read CSV bytes and raise actionable ValueError messages."""
    if not contents or not contents.strip():
        raise ValueError("The uploaded file is empty. Please choose a CSV with a header and data rows.")
    try:
        frame = pd.read_csv(BytesIO(contents))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Could not read this CSV. Check its encoding, commas, quotes, and row lengths. Details: {exc}") from exc
    if frame.empty:
        raise ValueError("The CSV contains headers but no data rows.")
    if len(frame.columns) == 0:
        raise ValueError("No columns were found in the CSV.")
    return frame

def validate_dataset(frame: pd.DataFrame) -> dict:
    """Return validation findings without modifying the supplied DataFrame."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    numeric_expected = ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score", "SLA_Breached"]
    numeric_issues = {}
    for col in numeric_expected:
        if col in frame.columns:
            parsed = pd.to_numeric(frame[col], errors="coerce")
            numeric_issues[col] = int((frame[col].notna() & parsed.isna()).sum())
    return {
        "missing_columns": missing_columns,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_request_ids": int(frame["Request_ID"].duplicated().sum()) if "Request_ID" in frame else None,
        "numeric_type_issues": {k:v for k,v in numeric_issues.items() if v},
        "valid_schema": not missing_columns,
    }

def prepare_analysis_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a safe analysis copy; original uploaded frame remains unchanged."""
    clean = frame.copy(deep=True)
    if "Submission_Date" in clean:
        clean["Submission_Date"] = pd.to_datetime(clean["Submission_Date"], errors="coerce")
    for col in ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score", "SLA_Breached"]:
        if col in clean:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    if "Queue_Length" in clean and "Queue_Category" not in clean:
        clean["Queue_Category"] = clean["Queue_Length"].map(
            lambda x: "Low" if pd.notna(x) and x <= 5 else ("Medium" if pd.notna(x) and x <= 15 else ("High" if pd.notna(x) else None))
        )
    if "Submission_Date" in clean and "Submission_Weekday" not in clean:
        clean["Submission_Weekday"] = clean["Submission_Date"].dt.day_name()
    return clean

def build_prediction_frame(values: dict) -> pd.DataFrame:
    """Validate and order one submission-time request exactly as the midterm model expects."""
    missing = [f for f in MODEL_FEATURES if f not in values]
    if missing:
        raise ValueError("Missing model input(s): " + ", ".join(missing))
    row = {f: values[f] for f in MODEL_FEATURES}
    for col in NUMERIC_FEATURES:
        try:
            row[col] = float(row[col])
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be a number.")
        if pd.isna(row[col]) or row[col] < 0:
            raise ValueError(f"{col} must be a non-negative number.")
    for col in CATEGORICAL_FEATURES:
        if pd.isna(row[col]) or not str(row[col]).strip():
            raise ValueError(f"{col} is required.")
        row[col] = str(row[col])
    return pd.DataFrame([row], columns=MODEL_FEATURES)
