from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import plotly.express as px

from modules.data_processing import read_csv_bytes, validate_dataset, prepare_analysis_copy, MODEL_FEATURES
from modules.exploratory_analysis import service_volume_chart, target_chart, office_breach_chart, queue_chart
from modules.dashboard import dashboard_kpis, review_queue
from modules.prediction import load_model, predict_one, MODEL_PATH

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Student Service Request Monitoring System", page_icon="📊", layout="wide")
st.title("Student Service Request Monitoring System")
st.caption("Section B · BS Information Systems · Advisory decision support")

@st.cache_data
def load_example_data(path: str):
    return pd.read_csv(path)

def get_data():
    uploaded = st.session_state.get("uploaded_bytes")
    if uploaded:
        try: return read_csv_bytes(uploaded)
        except ValueError as e:
            st.error(str(e))
            return pd.DataFrame()
    try: return load_example_data(str(BASE / "data" / "sample_data.csv"))
    except Exception: return pd.DataFrame()

menu = st.sidebar.selectbox("Navigate", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption("Use synthetic or authorized data only.")
upload = st.sidebar.file_uploader("Optional CSV for this session", type=["csv"])
if upload is not None:
    st.session_state["uploaded_bytes"] = upload.getvalue()
elif "uploaded_bytes" not in st.session_state:
    st.session_state["uploaded_bytes"] = None
df_raw = get_data()
df = prepare_analysis_copy(df_raw) if not df_raw.empty else df_raw

if menu == "Home":
    st.header("Welcome")
    st.write("This application supports student-services staff by exploring service-request data, estimating potential SLA breach risk using the original fitted model when supplied, and prioritizing human review.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Dataset rows", f"{len(df):,}")
    c2.metric("Dataset columns", f"{len(df.columns):,}")
    c3.metric("Saved model available", "Yes" if MODEL_PATH.exists() else "No")
    st.subheader("Available modules")
    st.markdown("- **Dataset:** upload, validate, and inspect CSV files.\n- **Data Analysis:** explore request volume and observed outcomes.\n- **Prediction:** use the saved fitted pipeline when it is provided.\n- **Dashboard:** review operational KPIs and request queue.\n- **System Design:** requirements, actors, use case, and workflow.")
    st.warning("Predictions are advisory only. Staff must verify the request and make the final decision. The current package does not include a saved fitted model artifact, so model-based predictions remain disabled until the original pipeline is supplied.")
    st.caption("Access limitation: authentication and role-based access control are not implemented in this prototype. Do not expose confidential records publicly.")

elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    st.info("Upload a CSV in the sidebar. Without an upload, the included midterm cleaned dataset is shown.")
    if df_raw.empty:
        st.error("No readable data is available.")
    else:
        validation = validate_dataset(df_raw)
        st.write(f"Rows: **{len(df_raw):,}** · Columns: **{len(df_raw.columns):,}")
        st.subheader("Data preview")
        safe_preview = df_raw.head(20).copy()
        # Mask identifier values in the preview to avoid unnecessary exposure.
        if "Request_ID" in safe_preview: safe_preview["Request_ID"] = safe_preview["Request_ID"].astype(str).map(lambda _: "••••")
        st.dataframe(safe_preview, use_container_width=True)
        st.subheader("Validation")
        if validation["valid_schema"]: st.success("All expected midterm dataset columns are present.")
        else: st.error("Missing expected columns: " + ", ".join(validation["missing_columns"]))
        if validation["numeric_type_issues"]: st.warning("Values that cannot be parsed as numbers: " + str(validation["numeric_type_issues"]))
        c1,c2,c3 = st.columns(3)
        c1.metric("Missing cells", sum(validation["missing_values"].values()))
        c2.metric("Duplicate rows", validation["duplicate_rows"])
        c3.metric("Duplicate Request_ID", validation["duplicate_request_ids"] if validation["duplicate_request_ids"] is not None else "Column absent")
        with st.expander("Missing values by column"):
            st.dataframe(pd.DataFrame({"Column":list(validation["missing_values"]), "Missing":list(validation["missing_values"].values())}), use_container_width=True)
        with st.expander("Descriptive statistics"):
            st.dataframe(df_raw.describe(include="all").transpose(), use_container_width=True)
        st.download_button("Download current dataset CSV", df_raw.to_csv(index=False).encode("utf-8"), "validated_dataset_copy.csv", "text/csv")

elif menu == "Data Analysis":
    st.header("Exploratory Data Analysis")
    if df.empty: st.warning("No data is available to analyze.")
    else:
        st.subheader("Descriptive statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        charts = [
            ("Request volume by service type", service_volume_chart(df)),
            ("Observed target distribution", target_chart(df)),
            ("Observed breach rate by office", office_breach_chart(df)),
            ("Observed breach rate by queue category", queue_chart(df)),
        ]
        available = [(name, fig) for name,fig in charts if fig is not None]
        if available:
            chosen = st.multiselect("Choose charts", [name for name,_ in available], default=[name for name,_ in available[:3]])
            for name,fig in available:
                if name in chosen: st.plotly_chart(fig, use_container_width=True)
        labeled = df["SLA_Breached"].dropna() if "SLA_Breached" in df else pd.Series(dtype=float)
        if len(labeled):
            st.info(f"Observed SLA breach rate among labeled records: {labeled.mean()*100:.1f}% ({int(labeled.sum())} breaches out of {len(labeled)} labeled records). This is a historical outcome, not a prediction for pending requests.")
        else: st.info("No labeled SLA outcomes are available; target-based charts are omitted.")

elif menu == "Prediction":
    st.header("Predictive Modeling")
    st.write("Inputs follow the exact ten submission-time predictors in the midterm notebook. Outcome fields such as Processing_Days, Status, Reopened, and Satisfaction_Score are intentionally excluded.")
    try:
        model = load_model()
        st.success("Original fitted model artifact loaded.")
    except Exception as exc:
        model = None
        st.error(str(exc))
    # Categories are inferred from the supplied dataset; model's OneHotEncoder ignores unseen categories.
    def options(col, fallback):
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            if vals: return vals
        return fallback
    with st.form("prediction_form"):
        a,b = st.columns(2)
        with a:
            service = st.selectbox("Service type", options("Service_Type", ["Records"]))
            requester = st.selectbox("Requester type", options("Requester_Type", ["Student"]))
            channel = st.selectbox("Requested channel", options("Requested_Channel", ["Portal"]))
            priority = st.selectbox("Priority", options("Priority", ["Normal"]))
            office = st.selectbox("Assigned office", options("Assigned_Office", ["Student Affairs"]))
        with b:
            queue_length = st.number_input("Queue length at submission", min_value=0, max_value=100000, value=5, step=1)
            queue_category = st.selectbox("Queue category", ["Low","Medium","High"], index=0)
            docs = st.selectbox("Required documents complete", options("Required_Documents_Complete", ["Yes","No"]))
            sla_days = st.number_input("SLA days", min_value=0.0, max_value=3650.0, value=5.0, step=1.0)
            weekday = st.selectbox("Submission weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        submitted = st.form_submit_button("Predict SLA breach risk", disabled=model is None)
    if submitted and model is not None:
        try:
            result = predict_one(model, {"Service_Type":service,"Requester_Type":requester,"Requested_Channel":channel,
                "Priority":priority,"Assigned_Office":office,"Queue_Length":queue_length,"Queue_Category":queue_category,
                "Required_Documents_Complete":docs,"SLA_Days":sla_days,"Submission_Weekday":weekday})
            st.subheader(result["prediction_label"])
            if "breach_probability" in result: st.metric("Model-estimated breach probability", f"{result['breach_probability']*100:.1f}%")
            st.caption(f"Prediction timestamp (UTC): {result['timestamp_utc']} · Model version: {result['model_version']}")
            st.warning("This is a model estimate, not a confirmed breach or guarantee. Check document completeness, workload, and current case details before acting.")
        except Exception as exc: st.error(f"Prediction failed: {exc}")
    st.subheader("Five fictional sample cases")
    sample_cases = [
        dict(Service_Type="Records",Requester_Type="Student",Requested_Channel="Portal",Priority="Normal",Assigned_Office="Student Affairs",Queue_Length=4,Queue_Category="Low",Required_Documents_Complete="Yes",SLA_Days=5,Submission_Weekday="Monday"),
        dict(Service_Type="Scholarship",Requester_Type="Student",Requested_Channel="Walk-in",Priority="High",Assigned_Office="Scholarship Office",Queue_Length=18,Queue_Category="High",Required_Documents_Complete="No",SLA_Days=3,Submission_Weekday="Tuesday"),
        dict(Service_Type="Enrollment",Requester_Type="Student",Requested_Channel="Portal",Priority="Normal",Assigned_Office="Registrar",Queue_Length=9,Queue_Category="Medium",Required_Documents_Complete="Yes",SLA_Days=5,Submission_Weekday="Wednesday"),
        dict(Service_Type="ID",Requester_Type="Student",Requested_Channel="Email",Priority="Low",Assigned_Office="Student Affairs",Queue_Length=2,Queue_Category="Low",Required_Documents_Complete="Yes",SLA_Days=7,Submission_Weekday="Thursday"),
        dict(Service_Type="Scholarship",Requester_Type="Student",Requested_Channel="Portal",Priority="High",Assigned_Office="Scholarship Office",Queue_Length=12,Queue_Category="Medium",Required_Documents_Complete="No",SLA_Days=3,Submission_Weekday="Friday"),
    ]
    samples = pd.DataFrame(sample_cases, index=[f"Fictional case {i}" for i in range(1,6)])
    st.dataframe(samples, use_container_width=True)
    if model is not None:
        results=[]
        for i,case in enumerate(sample_cases,1):
            try:
                r=predict_one(model,case); results.append({"Case":i,"Prediction":r["prediction_label"],"Breach probability":r.get("breach_probability"),"Timestamp UTC":r["timestamp_utc"]})
            except Exception as e: results.append({"Case":i,"Prediction":f"Error: {e}"})
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Predictions for these five cases are intentionally not fabricated. Add the original fitted pipeline to models/trained_model.joblib to generate them.")

elif menu == "Dashboard":
    st.header("Decision-Support Dashboard")
    if df.empty: st.warning("No data is available.")
    else:
        k = dashboard_kpis(df)
        a,b,c,d = st.columns(4)
        a.metric("Total requests", f"{k['total_requests']:,}")
        b.metric("Completed requests", k["completed_requests"] if k["completed_requests"] is not None else "N/A")
        c.metric("Labeled outcomes", k["labeled_requests"])
        d.metric("Observed SLA breach rate", f"{k['observed_breach_rate_percent']:.1f}%" if k["observed_breach_rate_percent"] is not None else "N/A")
        st.caption("Observed breach metrics use completed/labeled records. Predicted breaches are not counted until a fitted model is available.")
        if "Status" in df.columns:
            st.subheader("Request status")
            st.plotly_chart(px.histogram(df, x="Status", title="Request volume by status"), use_container_width=True)
        if "Queue_Category" in df.columns:
            st.subheader("Queue workload")
            st.plotly_chart(px.histogram(df, x="Queue_Category", title="Requests by queue category"), use_container_width=True)
        st.subheader("Review queue")
        q = review_queue(df)
        if q.empty: st.info("No review queue fields are available.")
        else:
            st.dataframe(q, use_container_width=True)
            st.caption("Historical labels are clearly marked as observed outcomes. They must not be interpreted as predictions for pending requests.")
        st.subheader("Suggested human follow-up")
        st.markdown("- Verify required documents and the request's current status.\n- Check workload and the applicable SLA.\n- Contact the responsible office if the queue is unusually high.\n- Record the staff decision in the authorized operational system; this prototype does not persist review decisions.")

else:
    st.header("System Design")
    st.subheader("A. Functional requirements")
    requirements = [
        ("FR-01","The system shall accept CSV uploads and report missing required columns, missing values, duplicate rows, duplicate Request_ID values, and numeric parsing issues."),
        ("FR-02","The system shall show descriptive statistics and interactive charts for service volume, labeled target distribution, office breach rates, and queue-category breach rates when the needed columns exist."),
        ("FR-03","The system shall accept the ten submission-time features and call the saved fitted pipeline to predict SLA breach risk when the original model artifact is available."),
        ("FR-04","The system shall display request-volume, status, workload, labeled-outcome KPIs, and a review queue while distinguishing observed outcomes from predicted risks."),
        ("FR-05","The system shall present risk as advisory and provide human follow-up guidance without automatically denying or approving a service request.")
    ]
    st.table(pd.DataFrame(requirements,columns=["ID","Requirement"]))
    st.subheader("B. Non-functional requirements (proposed; stakeholder approval needed)")
    st.table(pd.DataFrame([
        ("NFR-01 Privacy/security","Proposed: mask Request_ID values in preview by default; public deployment must use synthetic or explicitly authorized data. Authentication/role-based access is not implemented."),
        ("NFR-02 Performance","Proposed: on a standard laptop, load the 400-row sample dataset and render the main dashboard within 3 seconds after the app is warm; verify on target hosting."),
        ("NFR-03 Reliability/usability","Proposed: invalid CSVs and missing model files show actionable errors instead of crashing; all six navigation pages remain accessible.")
    ],columns=["ID","Requirement"]))
    st.subheader("C. Actors and access")
    st.markdown("- **Student Service Staff / Service Officer:** review request details and advisory flags. Authentication is not implemented.\n- **Office Supervisor:** review office-level queue and SLA patterns; proposed role, pending stakeholder confirmation.\n- **System Administrator:** maintain deployment and approved model/data artifacts. This prototype does not implement account management.\n- **Student/requester:** submits requests in the wider service system; request submission is outside this monitoring prototype.")
    st.subheader("D. UML use case diagram")
    img = BASE / "assets" / "use_case_diagram.png"
    if img.exists(): st.image(str(img), caption="Use case diagram (editable Graphviz source: assets/use_case_diagram.dot)", use_container_width=True)
    st.subheader("E. Analytical process flow")
    st.code("Request Dataset → Data Validation → Preprocessing → Saved Predictive Model → SLA Breach Prediction → Dashboard and Review Queue → Authorized Staff Review → Human Decision and Follow-Up", language="text")
    st.markdown("The prediction step is available only when the supplied fitted model artifact passes schema checks. Human review and final decisions remain outside the model.")
