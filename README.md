# Student Performance Monitoring and Academic Support Dashboard

A Streamlit decision-support dashboard for exploring student academic performance using synthetic demonstration data.

## Important data notice
The included `data/student_performance.csv` contains 500 generated synthetic records. It is NOT actual school data. Replace it only with authorized, anonymized school records after obtaining required permission.

## Features
- Overview KPIs and risk distribution
- Grade, attendance, and failed-subject analysis
- Priority student list and suggested interventions
- Decision Tree and Random Forest classification with a stratified hold-out test set
- K-Means clustering and silhouette score
- CSV and PDF report downloads
- Sidebar filters for school year, year level, section, and risk

## Run locally (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## GitHub / Streamlit Cloud
Push this folder's contents to its own GitHub repository. In Streamlit Community Cloud, choose the repository, branch `main`, and main file path `app.py`.

## Modeling note
The classification target is the synthetic current `risk_level`. Predictors use prior-period features only. Current grades, current attendance, and other current outcome fields are excluded from the model inputs to reduce target leakage. Metrics are demonstration results and should not be interpreted as validated school outcomes.
