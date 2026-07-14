"""What-If Simulator — adjust features and see CLV impact in real time."""

import streamlit as st
import pandas as pd
import numpy as np
from data_loader import (
    load_scored_data, load_feature_matrix, load_models,
    predict_clv, predict_churn, compute_shap,
    CLUSTER_NAMES, FEATURE_DISPLAY,
)
from components.shap_plots import shap_waterfall

st.header("What-If Simulator")
st.caption(
    "Start from a real customer, adjust key features with sliders, "
    "and see how predicted CLV changes. Use this to estimate the ROI "
    "of retention campaigns."
)

scored = load_scored_data()
features = load_feature_matrix()
models = load_models()

# ── Customer selector ──────────────────────────────────────
customer_ids = sorted(scored["customer_id"].tolist())
selected_id = st.selectbox(
    "Start from customer",
    customer_ids,
    format_func=lambda x: f"Customer {x}",
    key="whatif_customer",
)

if selected_id not in features.index:
    st.error("Feature data not available for this customer.")
    st.stop()

original = features.loc[selected_id].copy()
row = scored[scored["customer_id"] == selected_id].iloc[0]

# ── Original scorecard ─────────────────────────────────────
st.divider()
original_clv = predict_clv(original, models)
original_churn = predict_churn(original, models)

st.subheader("Current State")
oc1, oc2, oc3 = st.columns(3)
oc1.metric("Predicted CLV", f"£{original_clv:,.0f}")
oc2.metric("Churn Probability", f"{original_churn:.1%}")
oc3.metric("Segment", CLUSTER_NAMES.get(row["behavioral_cluster"], "Unknown"))

# ── Adjustable features ────────────────────────────────────
st.divider()
st.subheader("Adjust Features")
st.caption(
    "Move the sliders to simulate changes — e.g., what happens if a "
    "campaign increases this customer's purchase frequency by 30%?"
)

ADJUSTABLE = [
    ("frequency", 0.0, None),
    ("monetary_value", 0.0, None),
    ("avg_basket_value", 0.0, None),
    ("days_since_last_purchase", 0.0, None),
    ("category_breadth", 1.0, None),
    ("purchase_velocity_recent_vs_early", 0.0, None),
]

adjusted = original.copy()

col_left, col_right = st.columns(2)
sliders = [col_left, col_right]

for i, (feat, floor, ceil) in enumerate(ADJUSTABLE):
    current = float(original[feat])
    display_name = FEATURE_DISPLAY.get(feat, feat)

    # Set range: ±50% of current, clamped to floor
    low = max(floor, current * 0.5) if current >= 0 else current * 1.5
    high = current * 1.5 if current >= 0 else current * 0.5
    if ceil is not None:
        high = min(high, ceil)

    # Handle edge cases
    if abs(current) < 0.01:
        low, high = 0.0, 10.0
    if low >= high:
        low, high = 0.0, max(current * 2, 1.0)

    step = max((high - low) / 100, 0.01)

    with sliders[i % 2]:
        new_val = st.slider(
            display_name,
            min_value=float(low),
            max_value=float(high),
            value=float(current),
            step=float(step),
            key=f"slider_{feat}",
            help=f"Current: {current:,.2f}",
        )
        adjusted[feat] = new_val

# ── Re-predict ─────────────────────────────────────────────
st.divider()
st.subheader("Impact")

new_clv = predict_clv(adjusted, models)
new_churn = predict_churn(adjusted, models)
delta_clv = new_clv - original_clv
delta_churn = new_churn - original_churn

ic1, ic2, ic3 = st.columns(3)
ic1.metric(
    "Adjusted CLV",
    f"£{new_clv:,.0f}",
    delta=f"£{delta_clv:+,.0f}",
    delta_color="normal",
)
ic2.metric(
    "Adjusted Churn Prob",
    f"{new_churn:.1%}",
    delta=f"{delta_churn:+.1%}",
    delta_color="inverse",
)
ic3.metric(
    "CLV Change",
    f"{delta_clv / original_clv * 100:+.1f}%" if original_clv != 0 else "N/A",
)

# Business framing
if abs(delta_clv) > 1:
    direction = "increases" if delta_clv > 0 else "decreases"
    st.info(
        f"If these feature changes are achieved (e.g., through a targeted campaign), "
        f"this customer's predicted CLV {direction} from "
        f"£{original_clv:,.0f} to £{new_clv:,.0f} — "
        f"a **£{abs(delta_clv):,.0f}** {'uplift' if delta_clv > 0 else 'decline'}."
    )

# ── Side-by-side SHAP ──────────────────────────────────────
st.divider()
st.subheader("SHAP Comparison")

shap_left, shap_right = st.columns(2)

with shap_left:
    st.markdown("**Original**")
    sv_orig, fn_orig, bv_orig = compute_shap(original, models)
    fig_orig = shap_waterfall(sv_orig, fn_orig, bv_orig, max_display=8)
    st.plotly_chart(fig_orig, width="stretch", key="shap_original")

with shap_right:
    st.markdown("**Adjusted**")
    sv_adj, fn_adj, bv_adj = compute_shap(adjusted, models)
    fig_adj = shap_waterfall(sv_adj, fn_adj, bv_adj, max_display=8)
    st.plotly_chart(fig_adj, width="stretch", key="shap_adjusted")

# ── Feature change summary ─────────────────────────────────
st.divider()
st.subheader("Changes Applied")
changes = []
for feat, _, _ in ADJUSTABLE:
    old_val = float(original[feat])
    new_val = float(adjusted[feat])
    if abs(new_val - old_val) > 0.001:
        pct = ((new_val - old_val) / old_val * 100) if old_val != 0 else float("inf")
        changes.append({
            "Feature": FEATURE_DISPLAY.get(feat, feat),
            "Original": f"{old_val:,.2f}",
            "Adjusted": f"{new_val:,.2f}",
            "Change": f"{pct:+.1f}%",
        })

if changes:
    st.dataframe(pd.DataFrame(changes), width="stretch", hide_index=True)
else:
    st.caption("No features adjusted yet — move the sliders above.")