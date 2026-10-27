# Student Service Request Monitoring System

BSIS 3B · Section B · Systems Analysis and Design  
Prototype based on the uploaded midterm notebook, cleaned CSV, and examination report.

## Important source-artifact status

The original notebook and report describe a Decision Tree pipeline for target `SLA_Breached`, with ten submission-time features. **No fitted `trained_model.joblib` file was supplied in the uploaded materials.** The app therefore does not pretend to load a model and disables inference until the original fitted pipeline is added. It does not retrain on page refresh. Five fictional input examples are included, but their predictions are intentionally not shown until a valid model is supplied.

The notebook contains model-training code and reported metrics, but rerunning that code would create a newly fitted artifact rather than prove that it is the exact originally fitted artifact. Obtain the original pipeline from the midterm environment if available. If it cannot be recovered, ask the instructor whether reproducibly rebuilding it from the original notebook is acceptable.

## Project structure

```text
student-service-monitoring/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── exploratory_analysis.py
│   ├── prediction.py
│   └── dashboard.py
├── models/
│   └── [add original fitted trained_model.joblib here]
├── assets/
│   ├── use_case_diagram.png
│   └── use_case_diagram.dot
├── data/
│   └── sample_data.csv
├── tests/
│   └── test_app.py
├── source_files/
│   ├── original_midterm_notebook.ipynb
│   ├── FinalExer1-B.docx
│   └── midterm_report.pdf
└── reports/
    ├── analytical_process_flow.mmd
    └── deployment_report.md
```

## Model contract from the midterm notebook

Target: `SLA_Breached`, created for completed records with known `Processing_Days` and `SLA_Days`; 327 labeled records are reported in the exam PDF.

Predictors in exact order:
1. `Service_Type`
2. `Requester_Type`
3. `Requested_Channel`
4. `Priority`
5. `Assigned_Office`
6. `Queue_Length`
7. `Queue_Category`
8. `Required_Documents_Complete`
9. `SLA_Days`
10. `Submission_Weekday`

The notebook excludes `Processing_Days`, `Status`, `Reopened`, and `Satisfaction_Score` from model inputs to avoid leakage. Categorical features use most-frequent imputation and one-hot encoding (`handle_unknown="ignore"`); numeric features use median imputation. The classifier is `DecisionTreeClassifier(random_state=42, max_depth=5)` inside an sklearn `Pipeline`.

Reported results from the submitted midterm report (not re-evaluated by this Streamlit package):
- Accuracy: 0.682
- Breach precision: 0.651
- Breach recall: 0.824
- Breach F1-score: 0.727
- Majority baseline accuracy: 0.515
- Confusion matrix (actual rows, predicted columns): `[[17, 15], [6, 28]]`
- Aggregated top original predictors: `SLA_Days` (0.272), `Service_Type` (0.256), `Required_Documents_Complete` (0.180)

These are copied from the provided report and are not proof that this app has loaded or re-evaluated the fitted artifact.

## Run locally (VS Code or terminal)

1. Install Python 3.10 or 3.11.
2. Extract the ZIP and open the `student-service-monitoring` folder in VS Code.
3. Create a virtual environment:

   Windows:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Streamlit prints a local URL, usually `http://localhost:8501`.

Without `models/trained_model.joblib`, the Home, Dataset, Data Analysis, Dashboard, and System Design pages still work; Prediction clearly reports the missing artifact and does not generate fake predictions.

## Add the original fitted model

Copy the original fitted sklearn `Pipeline` object (the one fitted in the original midterm run) to `models/trained_model.joblib`. For example, in the original environment where the fitted variable `pipeline` still exists:

```python
import joblib
joblib.dump(pipeline, "trained_model.joblib")
```

Move that file into this project's `models/` directory. Ensure the pipeline is trusted and was created with a compatible scikit-learn version. Do not load joblib files from unknown sources. Restart the app after adding the file. The app checks feature names when `feature_names_in_` is available.

If the fitted object is no longer available, do not label a newly retrained model as the original artifact. Ask the instructor whether rebuilding from the exact notebook code and dataset is acceptable, then document that change and compare predictions against saved midterm test cases.

## Tests

Run:
```bash
pytest -q
```
Tests cover CSV parsing, empty/malformed files, required columns, missing/duplicate reporting, feature order/types, dashboard calculations, and missing-model errors. Model-consistency tests are conditional on the original model being present; no model-consistency result can be claimed without that artifact.

## GitHub setup

1. Create a new repository on GitHub, for example `student-service-monitoring`.
2. Keep it private while reviewing data and artifacts. Do not upload passwords, tokens, private student records, or credentials.
3. Copy the project files into the repository.
4. In a terminal opened in this folder:
   ```bash
   git init
   git add app.py requirements.txt README.md modules assets data tests source_files reports
   git commit -m "Create Streamlit student service monitoring prototype"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```
   Replace the example remote with the actual URL shown by GitHub. Do not commit model/data artifacts until approved for sharing.

## Deploy to Streamlit Community Cloud

1. Confirm that the repository contains only synthetic or properly authorized data.
2. Visit <https://share.streamlit.io> and sign in with GitHub.
3. Choose **Create app**.
4. Select the repository, branch (`main`, unless you chose another), and main file path `app.py`.
5. Deploy and wait for the build logs to finish.
6. Open the generated app URL and test every page. The URL is created by Streamlit; this README does not claim deployment has happened.
7. If the model is included, verify the model artifact is approved for deployment and uses compatible library versions.

## Troubleshooting

- **ModuleNotFoundError:** activate the correct virtual environment and run `pip install -r requirements.txt`.
- **CSV parser error:** re-export as a standard UTF-8 CSV; check delimiters, quotes, and row widths.
- **Missing columns:** use the same schema as the midterm CSV; the Dataset page lists missing required columns.
- **Missing model:** add the original fitted pipeline to `models/trained_model.joblib`; do not substitute an unrelated model.
- **Model version / sklearn error:** use the version that was used to serialize the artifact or explicitly validate compatibility in a separate environment.
- **Cloud build failure:** open app settings/logs, check Python/library compatibility, file paths, and whether all required files were committed.
- **Slow startup:** keep data small, use Streamlit caching, and avoid retraining in `app.py`.

## Privacy, limitations, and responsible use

This is a classroom prototype. It does not implement authentication, authorization, persistent review decisions, or a production database. Masking Request_ID in the preview is not a substitute for access control. Do not deploy real student data to a public app. A predicted breach is not a confirmed breach, and no model output should automatically deny a student service. Small groups, class imbalance, missing outcomes, and distribution changes can affect reliability.
