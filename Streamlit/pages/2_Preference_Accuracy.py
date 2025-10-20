"""
Preference Accuracy Page
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Preference Accuracy", page_icon="🎯", layout="wide")
st.title("🎯 Preference Prediction Accuracy")

if "metrics_df" not in st.session_state or st.session_state.metrics_df.empty:
    st.warning("⚠️ No experiment selected.")
    st.page_link("Home.py", label="← Back to Home", icon="🏠")
    st.stop()

metrics_df = st.session_state.metrics_df

st.markdown("""
Measures how accurately the SSM predicts time-varying preference weights:
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
""")

# Create subplot
fig = make_subplots(rows=1, cols=2, subplot_titles=("MSE", "MAE"))

# MSE
fig.add_trace(
    go.Scatter(
        x=metrics_df["update"],
        y=metrics_df["preference_mse"],
        name="MSE",
        mode="lines+markers",
        line=dict(color="#1f77b4"),
    ),
    row=1,
    col=1,
)

# MAE
fig.add_trace(
    go.Scatter(
        x=metrics_df["update"],
        y=metrics_df["preference_mae"],
        name="MAE",
        mode="lines+markers",
        line=dict(color="#2ca02c"),
    ),
    row=1,
    col=2,
)

fig.update_xaxes(title_text="Update")
fig.update_yaxes(title_text="MSE", row=1, col=1)
fig.update_yaxes(title_text="MAE", row=1, col=2)
fig.update_layout(height=500, showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Final MSE", f"{metrics_df.iloc[-1]['preference_mse']:.6f}")
with col2:
    st.metric("Best MSE", f"{metrics_df['preference_mse'].min():.6f}")
with col3:
    st.metric("Final MAE", f"{metrics_df.iloc[-1]['preference_mae']:.6f}")
with col4:
    st.metric("Best MAE", f"{metrics_df['preference_mae'].min():.6f}")
