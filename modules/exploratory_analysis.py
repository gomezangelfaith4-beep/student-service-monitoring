"""Actual-data EDA helpers and Plotly charts."""
import pandas as pd
import plotly.express as px

def target_distribution(frame):
    if "SLA_Breached" not in frame:
        return pd.DataFrame(columns=["SLA_Breached", "Count"])
    labeled = frame.loc[frame["SLA_Breached"].notna(), "SLA_Breached"]
    if labeled.empty:
        return pd.DataFrame(columns=["SLA_Breached", "Count"])
    return labeled.astype(int).value_counts().sort_index().rename_axis("SLA_Breached").reset_index(name="Count")

def service_volume_chart(frame):
    if "Service_Type" not in frame: return None
    counts = frame["Service_Type"].fillna("Missing").value_counts().rename_axis("Service_Type").reset_index(name="Requests")
    return px.bar(counts, x="Service_Type", y="Requests", title="Request volume by service type")

def target_chart(frame):
    dist = target_distribution(frame)
    if dist.empty: return None
    dist["Class"] = dist["SLA_Breached"].map({0:"No observed breach",1:"Observed breach"})
    return px.bar(dist, x="Class", y="Count", title="Observed target-class distribution")

def office_breach_chart(frame):
    needed = {"Assigned_Office", "SLA_Breached"}
    if not needed.issubset(frame.columns): return None
    labeled = frame.dropna(subset=["Assigned_Office", "SLA_Breached"])
    if labeled.empty: return None
    agg = labeled.groupby("Assigned_Office")["SLA_Breached"].agg(["count","mean"]).reset_index()
    agg["Breach rate (%)"] = agg["mean"] * 100
    return px.bar(agg, x="Assigned_Office", y="Breach rate (%)", hover_data=["count"], title="Observed SLA breach rate by office")

def queue_chart(frame):
    if not {"Queue_Category","SLA_Breached"}.issubset(frame.columns): return None
    labeled = frame.dropna(subset=["Queue_Category","SLA_Breached"])
    if labeled.empty: return None
    agg = labeled.groupby("Queue_Category")["SLA_Breached"].agg(["count","mean"]).reset_index()
    agg["Breach rate (%)"] = agg["mean"] * 100
    return px.bar(agg, x="Queue_Category", y="Breach rate (%)", hover_data=["count"], title="Observed SLA breach rate by queue category")
