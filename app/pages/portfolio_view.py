"""Portfolio View — executive dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import (
    load_scored_data, CLUSTER_NAMES, CLUSTER_COLORS,
    TIER_ORDER, MODEL_METRICS,
)

st.header("Portfolio View")
st.caption("Executive summary of customer base health, segmentation, and model performance.")

scored = load_scored_data()

# ── KPI row ────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{len(scored):,}")
k2.metric("Mean Predicted CLV", f"£{scored['predicted_clv_ml'].mean():,.0f}")
k3.metric("Total Predicted Revenue", f"£{scored['predicted_clv_ml'].sum():,.0f}")
k4.metric("Avg Churn Probability", f"{scored['churn_probability_ml'].mean():.1%}")

st.divider()

# ── Segment distribution ───────────────────────────────────
st.subheader("Customer Segments")

seg_stats = (
    scored.groupby("cluster_name")
    .agg(
        count=("customer_id", "size"),
        total_clv=("predicted_clv_ml", "sum"),
        mean_clv=("predicted_clv_ml", "mean"),
        mean_churn=("churn_probability_ml", "mean"),
    )
    .reset_index()
    .sort_values("total_clv", ascending=False)
)

col_donut, col_bar = st.columns(2)

with col_donut:
    fig_donut = px.pie(
        seg_stats, values="count", names="cluster_name",
        color="cluster_name", color_discrete_map=CLUSTER_COLORS,
        hole=0.45, title="Customer Count by Segment",
    )
    fig_donut.update_traces(textinfo="label+percent", textposition="outside")
    fig_donut.update_layout(showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_donut, width="stretch")

with col_bar:
    fig_bar = px.bar(
        seg_stats, x="cluster_name", y="total_clv",
        color="cluster_name", color_discrete_map=CLUSTER_COLORS,
        title="Total Predicted CLV by Segment",
        labels={"total_clv": "Total CLV (£)", "cluster_name": ""},
    )
    fig_bar.update_layout(showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_bar, width="stretch")

# Segment detail table
st.dataframe(
    seg_stats.rename(columns={
        "cluster_name": "Segment",
        "count": "Customers",
        "total_clv": "Total CLV (£)",
        "mean_clv": "Mean CLV (£)",
        "mean_churn": "Avg Churn Prob",
    }).style.format({
        "Total CLV (£)": "£{:,.0f}",
        "Mean CLV (£)": "£{:,.0f}",
        "Avg Churn Prob": "{:.1%}",
    }),
    width="stretch",
    hide_index=True,
)

st.divider()

# ── Revenue concentration (Lorenz curve) ───────────────────
st.subheader("Revenue Concentration")

sorted_clv = np.sort(scored["predicted_clv_ml"].values)
cumulative_customers = np.arange(1, len(sorted_clv) + 1) / len(sorted_clv)
cumulative_revenue = np.cumsum(sorted_clv) / sorted_clv.sum()

fig_lorenz = go.Figure()
fig_lorenz.add_trace(go.Scatter(
    x=cumulative_customers * 100, y=cumulative_revenue * 100,
    mode="lines", name="Actual", line=dict(color="#636EFA", width=2.5),
))
fig_lorenz.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100],
    mode="lines", name="Perfect equality",
    line=dict(color="#CCCCCC", dash="dash"),
))

# Find the 80/20 point
pct_80_rev = np.searchsorted(cumulative_revenue, 0.80)
pct_cust_at_80 = cumulative_customers[pct_80_rev] * 100 if pct_80_rev < len(cumulative_customers) else 100

fig_lorenz.add_annotation(
    x=pct_cust_at_80, y=80,
    text=f"Top {100 - pct_cust_at_80:.0f}% of customers → 80% of revenue",
    showarrow=True, arrowhead=2,
)
fig_lorenz.update_layout(
    title="Revenue Concentration Curve",
    xaxis_title="% of Customers (sorted by CLV ascending)",
    yaxis_title="% of Total Predicted Revenue",
    height=400, margin=dict(t=40, b=10),
    showlegend=True, legend=dict(x=0.05, y=0.95),
)
st.plotly_chart(fig_lorenz, width="stretch")

st.divider()

# ── CLV tier breakdown ─────────────────────────────────────
st.subheader("CLV Tier Breakdown")

tier_stats = (
    scored.groupby("clv_tier")
    .agg(
        count=("customer_id", "size"),
        total_clv=("predicted_clv_ml", "sum"),
        mean_clv=("predicted_clv_ml", "mean"),
        mean_churn=("churn_probability_ml", "mean"),
    )
    .reindex(TIER_ORDER)
    .reset_index()
    .dropna(subset=["count"])
)
tier_stats["pct_customers"] = tier_stats["count"] / tier_stats["count"].sum()
tier_stats["pct_revenue"] = tier_stats["total_clv"] / tier_stats["total_clv"].sum()

st.dataframe(
    tier_stats.rename(columns={
        "clv_tier": "Tier",
        "count": "Customers",
        "pct_customers": "% of Customers",
        "total_clv": "Total CLV (£)",
        "pct_revenue": "% of Revenue",
        "mean_clv": "Mean CLV (£)",
        "mean_churn": "Avg Churn Prob",
    }).style.format({
        "% of Customers": "{:.1%}",
        "Total CLV (£)": "£{:,.0f}",
        "% of Revenue": "{:.1%}",
        "Mean CLV (£)": "£{:,.0f}",
        "Avg Churn Prob": "{:.1%}",
    }),
    width="stretch",
    hide_index=True,
)

st.divider()

# ── Model performance ──────────────────────────────────────
st.subheader("Model Performance")

m = MODEL_METRICS
col_clv, col_churn = st.columns(2)

with col_clv:
    st.markdown("**CLV Regressor (XGBoost, Optuna-tuned)**")
    fig_mae = go.Figure(go.Bar(
        x=["Probabilistic\n(BG/NBD + GG)", "XGBoost\n(untuned)", "XGBoost\n(Optuna-tuned)"],
        y=[m["clv_mae_probabilistic"], m["clv_mae_untuned"], m["clv_mae_tuned"]],
        marker_color=["#CCCCCC", "#93B5FF", "#636EFA"],
        text=[f"£{v:,.0f}" for v in [m["clv_mae_probabilistic"], m["clv_mae_untuned"], m["clv_mae_tuned"]]],
        textposition="outside",
    ))
    fig_mae.update_layout(
        title="MAE Comparison (lower is better)",
        yaxis_title="MAE (£)", height=350,
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_mae, width="stretch")
    st.metric("R²", f"{m['clv_r2']:.3f}")
    st.metric("RMSE", f"£{m['clv_rmse']:,.0f}")

with col_churn:
    st.markdown("**Churn Classifier (XGBoost, Optuna-tuned)**")
    fig_auc = go.Figure(go.Bar(
        x=["P(alive) Baseline", "XGBoost\n(Optuna-tuned)"],
        y=[m["churn_auc_baseline"], m["churn_auc_ml"]],
        marker_color=["#CCCCCC", "#00CC96"],
        text=[f"{v:.3f}" for v in [m["churn_auc_baseline"], m["churn_auc_ml"]]],
        textposition="outside",
    ))
    fig_auc.update_layout(
        title="AUC-ROC Comparison (higher is better)",
        yaxis_title="AUC-ROC", height=350,
        yaxis_range=[0, 1],
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_auc, width="stretch")
    st.metric("F1 Score", f"{m['churn_f1']:.3f}")
    st.metric("AUC Improvement over Baseline", f"{89.1}%")