"""
IQL Dashboard - Home Page

Main landing page for experiment selection and overview.
"""

import base64
import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List

st.set_page_config(page_title="IQL Dashboard", page_icon="🤖", layout="wide")


def load_experiment_dirs(base_dir: str = "train_results") -> List[Path]:
    """Load all experiment directories from results folder."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Get all subdirectories with timestamps
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    exp_dirs.sort(reverse=True)  # Most recent first
    return exp_dirs


def load_config(exp_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_expert_config(exp_dir: Path) -> Dict[str, Any]:
    """Load expert configuration."""
    expert_config_path = exp_dir / "expert_config.yaml"
    if not expert_config_path.exists():
        return {}

    with open(expert_config_path, "r") as f:
        return yaml.safe_load(f)


def load_metrics(exp_dir: Path) -> pd.DataFrame:
    """Load metrics CSV."""
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()

    return pd.read_csv(metrics_path)


# Main page
st.title("🤖 IQL Training Results Dashboard")

st.markdown("""
Welcome to the IQL (Inverse Q-Learning) Results Dashboard!

This dashboard helps you visualize and analyze training and simulation results from your IQL experiments.
""")

# Display baseline episode GIF if available
baseline_gif_path = Path("dataset/dst/episode.gif")
if baseline_gif_path.exists():
    st.subheader("🎮 Baseline Expert Performance")
    st.markdown("**RL Expert Agent (Deep Sea Treasure)**")

    # Display animated GIF using base64 encoding
    file_ = open(baseline_gif_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="Baseline RL Expert Episode" width="400">',
        unsafe_allow_html=True,
    )

    st.caption(
        "This shows the trained RL expert navigating the Deep Sea Treasure environment."
    )

st.markdown("""
### Navigation
Use the sidebar to navigate between different views:
- **Home** (this page): Experiment selection and overview
- **Training Comparison**: Compare training metrics across multiple models
- **Simulation Returns**: Compare model performance vs baseline (RL expert)

### Getting Started
1. Navigate to **Training Comparison** to analyze training metrics across models
2. Navigate to **Simulation Returns** to compare exploitation performance
3. Download results as CSV for further analysis
""")

# Sidebar for experiment selection
st.sidebar.header("🔍 Experiment Selection")

base_dir = st.sidebar.text_input("Results Directory", value="train_results")
exp_dirs = load_experiment_dirs(base_dir)

if not exp_dirs:
    st.warning(f"⚠️ No experiments found in `{base_dir}`")
    st.info("💡 Run training first: `python src/IQL/train.py`")
    st.stop()

# Select experiment
exp_names = [d.name for d in exp_dirs]
selected_exp = st.sidebar.selectbox(
    "Select Experiment", exp_names, help="Choose an experiment to analyze"
)
selected_exp = Path(selected_exp)

if selected_exp:
    exp_dir = base_dir / selected_exp

    # Store in session state for other pages
    st.session_state.exp_dir = str(exp_dir)
    st.session_state.base_dir = base_dir

    # Load data
    config = load_config(exp_dir)
    expert_config = load_expert_config(exp_dir)
    metrics_df = load_metrics(exp_dir)

    # Store in session state
    st.session_state.config = config
    st.session_state.expert_config = expert_config
    st.session_state.metrics_df = metrics_df

    # Display experiment overview
    st.header(f"📊 Experiment: {selected_exp}")

    # Key metrics
    col1, col2 = st.columns(2)

    with col1:
        env_name = expert_config.get("env_name", "Unknown")
        st.metric("Environment", env_name)

    with col2:
        n_trajectories = config.get("n_trajectories", "N/A")
        st.metric("N Trajectories", n_trajectories)

    # Quick navigation
    st.markdown("---")
    st.subheader("Quick Navigation")

    col1, col2 = st.columns(2)

    with col1:
        st.page_link(
            "pages/1_Training_Comparison.py",
            label="View Training Comparison",
            icon="📈",
        )

    with col2:
        st.page_link(
            "pages/2_Simulation_Returns.py",
            label="View Simulation Returns",
            icon="🎯",
        )

else:
    st.info("👈 Select an experiment from the sidebar to get started")
