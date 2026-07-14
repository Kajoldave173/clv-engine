"""Custom Plotly SHAP waterfall renderer.

Replaces the default Matplotlib-based shap.plots.waterfall with an
interactive Plotly chart that embeds cleanly in Streamlit.
"""

import numpy as np
import plotly.graph_objects as go
from data_loader import FEATURE_DISPLAY


def shap_waterfall(shap_values, feature_names, base_value, max_display=10):
    """Render a horizontal SHAP waterfall chart.

    Args:
        shap_values: 1-D array of SHAP values for one customer.
        feature_names: Feature names matching shap_values order.
        base_value: Model's expected value E[f(x)].
        max_display: Max features to show (rest grouped).

    Returns:
        plotly.graph_objects.Figure
    """
    abs_vals = np.abs(shap_values)
    top_idx = np.argsort(abs_vals)[::-1][:max_display]
    # Waterfall reads bottom-to-top, so reverse
    top_idx = top_idx[::-1]

    top_values = [shap_values[i] for i in top_idx]
    top_names = [
        FEATURE_DISPLAY.get(feature_names[i], feature_names[i])
        for i in top_idx
    ]

    # Sum of features NOT shown
    other_idx = [i for i in range(len(shap_values)) if i not in top_idx]
    other_sum = sum(shap_values[i] for i in other_idx)
    prediction = base_value + float(np.sum(shap_values))

    # Build waterfall data: base → (others) → features → total
    y_labels = [f"E[f(x)] = £{base_value:,.0f}"]
    x_values = [base_value]
    measures = ["absolute"]

    if len(other_idx) > 0 and abs(other_sum) > 0.5:
        y_labels.append(f"{len(other_idx)} other features")
        x_values.append(other_sum)
        measures.append("relative")

    for name, val in zip(top_names, top_values):
        y_labels.append(name)
        x_values.append(val)
        measures.append("relative")

    y_labels.append(f"f(x) = £{prediction:,.0f}")
    x_values.append(0)
    measures.append("total")

    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=y_labels,
        x=x_values,
        measure=measures,
        increasing={"marker": {"color": "#FF4B4B"}},
        decreasing={"marker": {"color": "#636EFA"}},
        totals={"marker": {"color": "#2D2D2D"}},
        connector={"line": {"color": "#CCCCCC", "width": 1, "dash": "dot"}},
        textposition="outside",
        text=[
            f"£{base_value:,.0f}" if m == "absolute"
            else (f"£{prediction:,.0f}" if m == "total" else f"{v:+,.0f}")
            for v, m in zip(x_values, measures)
        ],
    ))

    fig.update_layout(
        height=max(350, 38 * len(y_labels) + 60),
        margin=dict(l=10, r=80, t=30, b=10),
        showlegend=False,
        xaxis_title="CLV Contribution (£)",
        waterfallgap=0.3,
    )

    return fig