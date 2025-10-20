"""
Training Metrics Page

Visualize training losses and episode rewards over time.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Training Metrics", page_icon="📈", layout="wide")

st.title("📈 Training Metrics")

# Check if experiment is selected
if "metrics_df" not in st.session_state or st.session_state.metrics_df.empty:
    st.warning(
        "⚠️ No experiment selected. Please go to Home page and select an experiment."
    )
    st.page_link("Home.py", label="← Back to Home", icon="🏠")
    st.stop()

metrics_df = st.session_state.metrics_df

# Training Losses Section
st.header("Training Losses")

st.markdown("""
These plots show how different loss components evolve during training:
- **Total Loss**: Combined loss (Soft IQ + Mismatch)
- **Soft IQ Loss**: Inverse Q-learning loss
- **Actor Loss**: Policy gradient loss (SAC-style)
- **Mismatch Loss**: Preference-Q alignment regularization
""")

# Create 2x2 subplot for losses
fig_losses = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("Total Loss", "Soft IQ Loss", "Actor Loss", "Mismatch Loss"),
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
)

# Total Loss
fig_losses.add_trace(
    go.Scatter(
        x=metrics_df["update"],
        y=metrics_df["total_loss"],
        name="Total Loss",
        mode="lines",
        line=dict(color="#1f77b4", width=2),
    ),
    row=1,
    col=1,
)

# Soft IQ Loss
fig_losses.add_trace(
    go.Scatter(
        x=metrics_df["update"],
        y=metrics_df["soft_iq_loss"],
        name="Soft IQ Loss",
        mode="lines",
        line=dict(color="#ff7f0e", width=2),
    ),
    row=1,
    col=2,
)

# Actor Loss
fig_losses.add_trace(
    go.Scatter(
        x=metrics_df["update"],
        y=metrics_df["actor_loss"],
        name="Actor Loss",
        mode="lines",
        line=dict(color="#2ca02c", width=2),
    ),
    row=2,
    col=1,
)

# Mismatch Loss
fig_losses.add_trace(
    go.Scatter(
        x=metrics_df["update"],
        y=metrics_df["mismatch_loss"],
        name="Mismatch Loss",
        mode="lines",
        line=dict(color="#d62728", width=2),
    ),
    row=2,
    col=2,
)

# Update axes
fig_losses.update_xaxes(title_text="Update", row=2, col=1)
fig_losses.update_xaxes(title_text="Update", row=2, col=2)
fig_losses.update_yaxes(title_text="Loss", row=1, col=1)
fig_losses.update_yaxes(title_text="Loss", row=1, col=2)
fig_losses.update_yaxes(title_text="Loss", row=2, col=1)
fig_losses.update_yaxes(title_text="Loss", row=2, col=2)

fig_losses.update_layout(height=700, showlegend=False, hovermode="x unified")

st.plotly_chart(fig_losses, use_container_width=True)

# Loss statistics
st.subheader("Loss Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Final Total Loss",
        f"{metrics_df.iloc[-1]['total_loss']:.4f}",
        delta=f"{metrics_df.iloc[-1]['total_loss'] - metrics_df.iloc[0]['total_loss']:.4f}",
    )

with col2:
    st.metric(
        "Final Soft IQ Loss",
        f"{metrics_df.iloc[-1]['soft_iq_loss']:.4f}",
        delta=f"{metrics_df.iloc[-1]['soft_iq_loss'] - metrics_df.iloc[0]['soft_iq_loss']:.4f}",
    )

with col3:
    st.metric(
        "Final Actor Loss",
        f"{metrics_df.iloc[-1]['actor_loss']:.4f}",
        delta=f"{metrics_df.iloc[-1]['actor_loss'] - metrics_df.iloc[0]['actor_loss']:.4f}",
    )

with col4:
    st.metric(
        "Final Mismatch Loss",
        f"{metrics_df.iloc[-1]['mismatch_loss']:.4f}",
        delta=f"{metrics_df.iloc[-1]['mismatch_loss'] - metrics_df.iloc[0]['mismatch_loss']:.4f}",
    )

# Episode Rewards Section
st.header("Episode Rewards")

st.markdown("""
Episode reward shows how well the learned policy performs in the environment.
The shaded area represents the standard deviation across evaluation episodes.
""")

# Create figure for episode rewards
fig_rewards = go.Figure()

# Mean reward line
fig_rewards.add_trace(
    go.Scatter(
        x=metrics_df["update"],
        y=metrics_df["mean_episode_reward"],
        name="Mean Reward",
        mode="lines+markers",
        line=dict(color="#9467bd", width=2),
        marker=dict(size=4),
    )
)

# Std band
if "std_episode_reward" in metrics_df.columns:
    upper_bound = metrics_df["mean_episode_reward"] + metrics_df["std_episode_reward"]
    lower_bound = metrics_df["mean_episode_reward"] - metrics_df["std_episode_reward"]

    fig_rewards.add_trace(
        go.Scatter(
            x=metrics_df["update"].tolist() + metrics_df["update"].tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(148,103,189,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="± Std Dev",
        )
    )

fig_rewards.update_layout(
    title="Episode Reward Over Training",
    xaxis_title="Update",
    yaxis_title="Episode Reward",
    height=500,
    hovermode="x unified",
)

st.plotly_chart(fig_rewards, use_container_width=True)

# Download data
st.markdown("---")
csv_data = metrics_df.to_csv(index=False)
st.download_button(
    label="📥 Download Metrics CSV",
    data=csv_data,
    file_name="training_metrics.csv",
    mime="text/csv",
)
