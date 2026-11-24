from pathlib import Path
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils.data_processing import load_and_clean_data, apply_filters, make_recommendation
from utils.modeling import train_risk_models, run_student_clustering
from utils.reports import build_pdf_report

st.set_page_config(page_title="Student Performance | Academic Support", page_icon="🎓", layout="wide")

st.markdown("""
<style>
:root { --navy:#172554; --blue:#2563eb; --purple:#7c3aed; }
[data-testid="stAppViewContainer"] {background:linear-gradient(135deg,#f7f8ff 0%,#f4f5fb 55%,#f8f7ff 100%);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#172554,#312e81); }
[data-testid="stSidebar"] * {color:#fff !important;}
.block-container {padding-top:1.6rem; padding-bottom:2rem; max-width:1500px;}
.hero {background:linear-gradient(110deg,#1d4ed8,#6d28d9);padding:25px 28px;border-radius:18px;color:white;margin-bottom:18px;box-shadow:0 10px 28px #1e3a8a22;}
.hero h1 {font-size:30px; margin:0 0 5px 0; color:white;}
.hero p {margin:0;color:#e0e7ff;}
.kpi {background:white;border:1px solid #e5e7eb;border-radius:15px;padding:17px 18px;box-shadow:0 4px 15px #1e293b0b;min-height:112px;}
.kpi-label {font-size:12px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.07em;}
.kpi-value {font-size:27px;font-weight:800;color:#172554;margin-top:7px;}
.kpi-note {font-size:12px;color:#64748b;margin-top:3px;}
.section-title {font-size:20px;font-weight:800;color:#172554;margin:18px 0 8px;}
.small-note {font-size:12px;color:#64748b;}
div[data-testid="stMetric"] {background:white;padding:14px;border-radius:14px;border:1px solid #e5e7eb;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_and_clean_data(ROOT / "data" / "student_performance.csv")

df, cleaning = get_data()

with st.sidebar:
    st.markdown("## 🎓 Academic Insight")
    st.caption("Student Performance Monitoring System")
    page = st.radio("NAVIGATION", ["Overview", "Performance Analysis", "Risk & Priority List", "Machine Learning", "Student Clustering", "Reports"])
    st.divider()
    st.markdown("### FILTERS")
    years = st.multiselect("School year", sorted(df.school_year.dropna().unique()), default=sorted(df.school_year.dropna().unique()))
    levels = st.multiselect("Year level", sorted(df.year_level.dropna().unique()), default=sorted(df.year_level.dropna().unique()))
    sections = st.multiselect("Section", sorted(df.section.dropna().unique()), default=sorted(df.section.dropna().unique()))
    risks = st.multiselect("Risk level", ["Low", "Moderate", "High"], default=["Low", "Moderate", "High"])
    st.caption("Subject filter is available on Performance Analysis.")
    st.divider()
    st.markdown("**DEMO DATA NOTICE**")
    st.caption("This application uses synthetic sample records. It is not connected to an actual school database.")

filtered = apply_filters(df, years, levels, sections, risks)

def header(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def kpi(label, value, note=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("No student records match the selected filters. Please change the filters in the sidebar.")
    st.stop()

if page == "Overview":
    header("Student Performance Overview", "A decision-support view of academic performance, attendance, and student support needs.")
    a,b,c,d = st.columns(4)
    with a: kpi("Total students", f"{len(filtered):,}", "Students in current filter")
    with b: kpi("Average general grade", f"{filtered.general_average.mean():.1f}", "Current general average")
    with c: kpi("Average attendance", f"{filtered.attendance_rate.mean():.1f}%", "Current attendance rate")
    with d: kpi("High-risk students", f"{(filtered.risk_level=='High').sum():,}", "Requires priority review")
    st.markdown('<div class="section-title">Academic risk distribution</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1,1.35])
    with c1:
        counts = filtered.risk_level.value_counts().reindex(["Low","Moderate","High"], fill_value=0).rename_axis("Risk level").reset_index(name="Students")
        fig = px.pie(counts, names="Risk level", values="Students", hole=.58, color="Risk level",
                     color_discrete_map={"Low":"#16a34a","Moderate":"#f59e0b","High":"#dc2626"})
        fig.update_layout(margin=dict(t=20,b=20,l=10,r=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        by_level = filtered.groupby("year_level", as_index=False).agg(Students=("student_code","count"), Average_Grade=("general_average","mean"))
        fig = px.bar(by_level, x="year_level", y="Average_Grade", color="year_level", text_auto=".1f",
                     labels={"year_level":"Year level","Average_Grade":"Average grade"}, title="Average grade by year level")
        fig.update_layout(showlegend=False, yaxis_range=[0,100], margin=dict(t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-title">Students needing attention</div>', unsafe_allow_html=True)
    priority = filtered[filtered.risk_level.isin(["High","Moderate"])].copy()
    priority["recommended_action"] = priority.apply(make_recommendation, axis=1)
    st.dataframe(priority[["student_code","year_level","section","general_average","attendance_rate","failed_subjects","risk_level","recommended_action"]].sort_values(["risk_level","general_average"], ascending=[False,True]).head(12), use_container_width=True, hide_index=True)
    st.caption(f"Data cleaning: {cleaning['missing_values_filled']} missing values filled; {cleaning['duplicate_rows_removed']} duplicate rows removed.")

elif page == "Performance Analysis":
    header("Performance Analysis", "Compare subject grades, attendance patterns, and failed-subject counts.")
    subject = st.selectbox("Choose subject", ["All subjects","Math","English","Science"])
    grade_cols = {"Math":"math_grade","English":"english_grade","Science":"science_grade"}
    if subject == "All subjects":
        long = filtered.melt(id_vars=["student_code","year_level","section"], value_vars=list(grade_cols.values()), var_name="Subject", value_name="Grade")
        long["Subject"] = long["Subject"].str.replace("_grade","",regex=False).str.title()
    else:
        long = filtered[["student_code","year_level","section",grade_cols[subject]]].rename(columns={grade_cols[subject]:"Grade"})
        long["Subject"] = subject
    c1,c2 = st.columns(2)
    with c1:
        fig = px.box(long, x="Subject", y="Grade", color="Subject", points="outliers", title="Grade distribution by subject")
        fig.update_layout(yaxis_range=[0,100], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        att = px.histogram(filtered, x="attendance_rate", nbins=18, color="risk_level", barmode="overlay",
                           title="Attendance distribution", labels={"attendance_rate":"Attendance rate (%)","risk_level":"Risk"})
        st.plotly_chart(att, use_container_width=True)
    failed = filtered.groupby("year_level", as_index=False)["failed_subjects"].sum()
    fig = px.bar(failed, x="year_level", y="failed_subjects", color="year_level", title="Total failed subjects by year level",
                 labels={"year_level":"Year level","failed_subjects":"Failed subjects"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Subject summary")
    st.dataframe(long.groupby("Subject")["Grade"].agg(["count","mean","median","min","max"]).round(2), use_container_width=True)

elif page == "Risk & Priority List":
    header("Risk Analysis & Priority Support", "Review students by academic risk and use suggested interventions as a starting point for staff review.")
    risk_counts = filtered.risk_level.value_counts().reindex(["Low","Moderate","High"],fill_value=0)
    c1,c2,c3 = st.columns(3)
    with c1: kpi("Low risk", int(risk_counts["Low"]), "Continue regular monitoring")
    with c2: kpi("Moderate risk", int(risk_counts["Moderate"]), "Follow-up and academic advising")
    with c3: kpi("High risk", int(risk_counts["High"]), "Priority staff assessment")
    priority = filtered.copy()
    priority["recommended_action"] = priority.apply(make_recommendation, axis=1)
    priority = priority[priority.risk_level.isin(["High","Moderate"])].sort_values(["risk_level","general_average"], ascending=[True,True])
    search = st.text_input("Search student code", placeholder="e.g. STU-0001")
    if search:
        priority = priority[priority.student_code.str.contains(search, case=False, na=False)]
    st.dataframe(priority[["student_code","school_year","year_level","section","general_average","attendance_rate","failed_subjects","risk_level","recommended_action"]], use_container_width=True, hide_index=True)
    st.info("Risk categories are demonstration labels generated for this synthetic dataset. Recommendations are decision-support suggestions, not automatic decisions about students.")

elif page == "Machine Learning":
    header("Predictive Modeling", "Train and evaluate Decision Tree and Random Forest models using prior-period information.")
    st.markdown("The target is current `risk_level`. Model inputs are prior-period grades, attendance, and failed subjects. Current grades and current attendance are excluded from predictors to reduce target leakage.")
    with st.spinner("Training and evaluating models..."):
        results = train_risk_models(df)
    if "error" in results:
        st.error(results["error"])
    else:
        c1,c2 = st.columns(2)
        for col, name in [(c1,"Decision Tree"),(c2,"Random Forest")]:
            m = results[name]
            with col:
                st.subheader(name)
                x1,x2,x3 = st.columns(3)
                x1.metric("Accuracy", f"{m['accuracy']:.3f}")
                x2.metric("Precision", f"{m['precision']:.3f}")
                x3.metric("Recall", f"{m['recall']:.3f}")
                st.metric("F1-score (weighted)", f"{m['f1']:.3f}")
                cm = m["confusion_matrix"]
                fig = px.imshow(cm, text_auto=True, aspect="auto", x=m["labels"], y=m["labels"],
                                labels=dict(x="Predicted",y="Actual",color="Count"), title="Confusion matrix")
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Random Forest feature importance")
        imp = results["Random Forest"]["importance"]
        fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                     title="Relative feature importance")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Evaluation uses a stratified hold-out test set. Metrics describe performance on this synthetic demonstration dataset; they are not evidence of real-world school performance.")

elif page == "Student Clustering":
    header("Student Segmentation with K-Means", "Group students with similar academic characteristics for exploratory analysis.")
    k = st.slider("Number of clusters (K)", min_value=2, max_value=6, value=3)
    cluster_result = run_student_clustering(filtered, k=k)
    if "error" in cluster_result:
        st.error(cluster_result["error"])
    else:
        clustered = cluster_result["data"]
        c1,c2 = st.columns(2)
        with c1:
            st.metric("Silhouette score", f"{cluster_result['silhouette']:.3f}")
            st.caption("Closer to 1 generally indicates more separated clusters; values near 0 indicate overlap.")
        with c2:
            st.metric("Clusters", k)
            st.caption("K-Means uses scaled prior-period academic variables.")
        fig = px.scatter(clustered, x="prior_general_average", y="prior_attendance_rate", color="cluster",
                         hover_data=["student_code","year_level","section"], title="Student clusters",
                         labels={"prior_general_average":"Prior general average","prior_attendance_rate":"Prior attendance (%)","cluster":"Cluster"})
        st.plotly_chart(fig, use_container_width=True)
        profile = clustered.groupby("cluster").agg(Students=("student_code","count"), Prior_Average=("prior_general_average","mean"),
                    Prior_Attendance=("prior_attendance_rate","mean"), Prior_Failed=("prior_failed_subjects","mean")).round(2)
        st.subheader("Cluster profiles")
        st.dataframe(profile, use_container_width=True)
        st.caption("Cluster numbers are arbitrary labels. Interpret each group using its profile, not the cluster number alone.")

elif page == "Reports":
    header("Reports & Data Export", "Download filtered student records and a concise PDF summary.")
    export = filtered.copy()
    export["recommended_action"] = export.apply(make_recommendation, axis=1)
    st.metric("Records included in report", len(export))
    st.download_button("⬇ Download filtered CSV", export.to_csv(index=False).encode("utf-8"), "student_performance_filtered_report.csv", "text/csv", use_container_width=True)
    pdf = build_pdf_report(export)
    st.download_button("⬇ Download PDF summary", pdf, "student_performance_summary.pdf", "application/pdf", use_container_width=True)
    st.dataframe(export.head(20), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('<div class="small-note">Academic Insight Dashboard · Demonstration version · Synthetic data only · Human review is required before any student-support action.</div>', unsafe_allow_html=True)
