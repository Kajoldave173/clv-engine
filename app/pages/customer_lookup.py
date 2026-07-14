"""Customer Lookup — individual customer scorecard with SHAP explanation."""

import streamlit as st
import pandas as pd
from data_loader import (
    load_scored_data, load_feature_matrix, load_models,
    compute_shap, CLUSTER_NAMES, FEATURE_DISPLAY,
)
from components.shap_plots import shap_waterfall

st.header("Customer Lookup")
st.caption("View any customer's predicted CLV, churn risk, and the features driving their score.")

scored = load_scored_data()
features = load_feature_matrix()
models = load_models()

# ── Customer selector ──────────────────────────────────────
customer_ids = sorted(scored["customer_id"].tolist())
selected_id = st.selectbox(
    "Select a customer",
    customer_ids,
    format_func=lambda x: f"Customer {x}",
)

row = scored[scored["customer_id"] == selected_id].iloc[0]

# ── Scorecard ──────────────────────────────────────────────
st.divider()
st.subheader("Scorecard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted CLV (ML)", f"£{row['predicted_clv_ml']:,.0f}")
c2.metric("Predicted CLV (Probabilistic)", f"£{row['predicted_clv_probabilistic']:,.0f}")
c3.metric("Churn Probability", f"{row['churn_probability_ml']:.1%}")
c4.metric("P(Alive)", f"{row['p_alive']:.1%}")

c5, c6, c7 = st.columns(3)
cluster_name = CLUSTER_NAMES.get(row["behavioral_cluster"], f"Cluster {row['behavioral_cluster']}")
c5.metric("Segment", cluster_name)
c6.metric("CLV Tier", row["clv_tier"])

# Percentile
pctile = (scored["predicted_clv_ml"] <= row["predicted_clv_ml"]).mean()
c7.metric("CLV Percentile", f"{pctile:.0%}")

# ── SHAP waterfall ─────────────────────────────────────────
st.divider()
st.subheader("What's Driving This Customer's CLV?")

if selected_id in features.index:
    feat_row = features.loc[selected_id]
    shap_vals, feat_names, base_val = compute_shap(feat_row, models)

    fig = shap_waterfall(shap_vals, feat_names, base_val, max_display=10)
    st.plotly_chart(fig, width="stretch")

    # Top drivers in plain English
    drivers = row.get("top_3_shap_drivers", "")
    if drivers:
        st.markdown(f"**Top SHAP drivers:** `{drivers}`")
else:
    st.warning("Feature data not available for this customer.")

# ── Feature values ─────────────────────────────────────────
st.divider()
st.subheader("Feature Values")

if selected_id in features.index:
    feat_row = features.loc[selected_id]
    display_df = pd.DataFrame({
        "Feature": [FEATURE_DISPLAY.get(f, f) for f in feat_row.index],
        "Value": feat_row.values,
        "Raw Name": feat_row.index,
    })
    display_df = display_df[["Feature", "Value"]].copy()

    # Format: round floats, keep ints
    display_df["Value"] = display_df["Value"].apply(
        lambda v: f"{v:,.2f}" if isinstance(v, float) else str(v)
    )
    st.dataframe(display_df, width="stretch", hide_index=True)

# ── Peer comparison ────────────────────────────────────────
st.divider()
st.subheader("Peer Comparison")

if selected_id in features.index:
    cluster_id = row["behavioral_cluster"]
    cluster_mask = scored["behavioral_cluster"] == cluster_id
    cluster_scored = scored[cluster_mask]

    peer_features = features.loc[
        features.index.isin(cluster_scored["customer_id"])
    ]

    if not peer_features.empty:
        feat_row = features.loc[selected_id]
        comparison = []
        for col in ["frequency", "monetary_value", "avg_basket_value",
                     "days_since_last_purchase", "category_breadth"]:
            if col in feat_row.index and col in peer_features.columns:
                cust_val = feat_row[col]
                peer_mean = peer_features[col].mean()
                pct_diff = ((cust_val - peer_mean) / peer_mean * 100) if peer_mean != 0 else 0
                comparison.append({
                    "Feature": FEATURE_DISPLAY.get(col, col),
                    "This Customer": f"{cust_val:,.2f}",
                    "Cluster Avg": f"{peer_mean:,.2f}",
                    "vs Peers": f"{pct_diff:+.0f}%",
                })

        st.dataframe(
            pd.DataFrame(comparison),
            width="stretch",
            hide_index=True,
        )