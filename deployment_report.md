# Deployment report and reflection

## Source reuse
The package preserves the submitted midterm notebook, exercise instructions, and report under `source_files/`. The app follows the original target (`SLA_Breached`), ten submission-time features, Decision Tree pipeline description, and leakage exclusions.

## Changes made
- Added Streamlit navigation for Home, Dataset, Data Analysis, Prediction, Dashboard, and System Design.
- Added CSV validation, missing/duplicate summaries, Plotly charts, dashboard calculations, a review queue, requirements, actor descriptions, and diagrams.
- Masked Request_ID values in the Dataset preview.
- Added tests and beginner-friendly run/deploy instructions.

## Critical limitation
No fitted `trained_model.joblib` was present among the uploaded files. The prediction page intentionally does not return predictions until that original artifact is supplied. Thus the five fictional cases are input examples only, not prediction results. The metrics displayed in the README are those stated in the uploaded midterm report and were not recomputed by this app.

## Testing status
Executed in the available build environment: `pytest -q` returned **12 passed**. These are unit tests for CSV parsing/validation, input schema, dashboard calculations, and missing-model handling. Streamlit UI/browser navigation was not exercised because Streamlit was not installed in the build environment. Model consistency, actual inference, screenshots of successful model predictions, and deployed URL remain unverified until the original artifact and external deployment are available.

## Reflection
The application turns the midterm analysis into a navigable decision-support prototype. Its main strengths are transparent data checks and the separation of observed outcomes from predicted risk. Its main limits are missing fitted model artifact, no authentication or persistent review logging, and no production data connection. Human review remains necessary.
