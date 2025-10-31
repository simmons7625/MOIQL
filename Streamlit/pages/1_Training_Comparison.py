"""
Training Metrics Comparison Page

Compare training metrics across multiple models with customizable column selection.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Training Comparison", page_icon="📊", layout="wide")

st.title("📊 Training Metrics Comparison")

# Define paths
TRAIN_RESULTS_DIR = Path("train_results")

# Get all available models
if not TRAIN_RESULTS_DIR.exists():
    st.error(f"Training results directory not found: {TRAIN_RESULTS_DIR}")
    st.stop()

model_dirs = sorted([d for d in TRAIN_RESULTS_DIR.iterdir() if d.is_dir()])
model_names = [d.name for d in model_dirs]

if not model_names:
    st.warning("No training results found.")
    st.stop()

st.markdown("""
Select models and metrics to compare training performance across different configurations.
""")

# Sidebar for selections
st.sidebar.header("Selection Options")

# Model selection
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    options=model_names,
    default=model_names,  # Select all by default
    help="Choose which models to include in the comparison",
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()


# Load training metrics for selected models
@st.cache_data
def load_training_metrics(model_name):
    """Load training metrics CSV for a model."""
    csv_path = TRAIN_RESULTS_DIR / model_name / "train_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["model"] = model_name
        return df
    return None


# Load all metrics
all_metrics = []
for model in selected_models:
    df = load_training_metrics(model)
    if df is not None:
        all_metrics.append(df)
    else:
        st.warning(f"⚠️ No train_metrics.csv found for {model}")

if not all_metrics:
    st.error("No valid training metrics found for selected models.")
    st.stop()

# Get available columns (excluding 'update' and 'model')
sample_df = all_metrics[0]
available_columns = [col for col in sample_df.columns if col not in ["update", "model"]]

# Column selection
selected_columns = st.sidebar.multiselect(
    "Select Metrics to Plot",
    options=available_columns,
    default=available_columns[:4] if len(available_columns) >= 4 else available_columns,
    help="Choose which metrics to visualize",
)

if not selected_columns:
    st.warning("Please select at least one metric to plot.")
    st.stop()

# Combine all dataframes
combined_df = pd.concat(all_metrics, ignore_index=True)

# Display summary statistics
st.header("Summary Statistics")

# Create summary table
summary_data = []
for model in selected_models:
    model_df = combined_df[combined_df["model"] == model]
    row = {"Model": model}
    for col in selected_columns:
        if col in model_df.columns:
            final_value = model_df[col].iloc[-1]
            row[f"{col} (final)"] = f"{final_value:.4f}"
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# Plot metrics
st.header("Metrics Over Training")

# Color palette for models
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Create subplot layout based on number of selected columns
n_cols = min(2, len(selected_columns))
n_rows = (len(selected_columns) + n_cols - 1) // n_cols

# Create individual plots for each metric
for idx, metric in enumerate(selected_columns):
    st.subheader(metric.replace("_", " ").title())

    fig = go.Figure()

    for model_idx, model in enumerate(selected_models):
        model_df = combined_df[combined_df["model"] == model]

        if metric in model_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_df["update"],
                    y=model_df[metric],
                    name=model,
                    mode="lines",
                    line=dict(color=colors[model_idx % len(colors)], width=2),
                    hovertemplate=f"<b>{model}</b><br>Update: %{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis_title="Update",
        yaxis_title=metric.replace("_", " ").title(),
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

# Download combined data
st.markdown("---")
st.subheader("Download Data")

csv_data = combined_df.to_csv(index=False)
st.download_button(
    label="📥 Download Combined Metrics CSV",
    data=csv_data,
    file_name="combined_training_metrics.csv",
    mime="text/csv",
)
