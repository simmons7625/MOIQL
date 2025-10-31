"""
Simulation Returns Comparison Page

Compare true returns across all models to evaluate exploitation performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np

st.set_page_config(page_title="Simulation Returns", page_icon="🎯", layout="wide")

st.title("🎯 Simulation Return Comparison")

# Define paths
SIM_RESULTS_DIR = Path("simulation_results")
DATASET_DIR = Path("dataset/dst")


# Load baseline (expert/RL) statistics
@st.cache_data
def load_baseline_stats():
    """Load baseline statistics from dataset."""
    baseline_path = DATASET_DIR / "statistics.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            stats = json.load(f)
        stats["model"] = "Baseline (RL Expert)"
        return stats
    return None


baseline_stats = load_baseline_stats()

# Get all available models
if not SIM_RESULTS_DIR.exists():
    st.error(f"Simulation results directory not found: {SIM_RESULTS_DIR}")
    st.stop()

model_dirs = sorted([d for d in SIM_RESULTS_DIR.iterdir() if d.is_dir()])
model_names = [d.name for d in model_dirs]

if not model_names:
    st.warning("No simulation results found.")
    st.stop()

st.markdown("""
This page shows the **true return** statistics for all models, computed using ground truth preference weights.
The baseline (RL expert) performance is included for comparison.
""")


# Load simulation statistics
@st.cache_data
def load_simulation_stats(model_name):
    """Load simulation statistics JSON for a model."""
    stats_path = SIM_RESULTS_DIR / model_name / "statistics.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
        stats["model"] = model_name
        return stats
    return None


# Load all statistics
all_stats = []

# Add baseline first if available
if baseline_stats is not None:
    all_stats.append(baseline_stats)
else:
    st.info("ℹ️ Baseline statistics not found at dataset/dst/statistics.json")

for model in model_names:
    stats = load_simulation_stats(model)
    if stats is not None:
        all_stats.append(stats)
    else:
        st.warning(f"⚠️ No statistics.json found for {model}")

if not all_stats:
    st.error("No valid simulation statistics found.")
    st.stop()

# Create DataFrame from statistics
stats_df = pd.DataFrame(all_stats)

# Ensure we have return statistics
if "mean_return" not in stats_df.columns:
    st.error(
        "Return statistics not found in simulation results. Please re-run simulations with the updated script."
    )
    st.stop()

# Summary Statistics Table
st.header("Return Statistics Summary")

# Create summary table
summary_cols = [
    "model",
    "mean_return",
    "std_return",
    "min_return",
    "max_return",
    "n_episodes",
]
if all(col in stats_df.columns for col in summary_cols):
    summary_df = stats_df[summary_cols].copy()
    summary_df = summary_df.sort_values("mean_return", ascending=False)

    # Format numeric columns
    summary_df["mean_return"] = summary_df["mean_return"].apply(lambda x: f"{x:.4f}")
    summary_df["std_return"] = summary_df["std_return"].apply(lambda x: f"{x:.4f}")
    summary_df["min_return"] = summary_df["min_return"].apply(lambda x: f"{x:.4f}")
    summary_df["max_return"] = summary_df["max_return"].apply(lambda x: f"{x:.4f}")

    # Rename columns for display
    summary_df.columns = [
        "Model",
        "Mean Return",
        "Std Return",
        "Min Return",
        "Max Return",
        "Episodes",
    ]

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
else:
    st.error("Required columns not found in statistics.")

# Visualizations
st.header("Return Comparison Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📈 Box Plot", "🎯 Detailed Stats"])

with tab1:
    st.subheader("Mean Return by Model")

    # Sort by mean return for better visualization
    plot_df = stats_df.sort_values("mean_return", ascending=True)

    # Create color array - highlight baseline differently
    colors = []
    for model in plot_df["model"]:
        if "Baseline" in model or "Expert" in model:
            colors.append("#FFD700")  # Gold for baseline
        else:
            colors.append("#1f77b4")  # Blue for IQL models

    fig_bar = go.Figure()

    # Add bar chart with error bars
    fig_bar.add_trace(
        go.Bar(
            y=plot_df["model"],
            x=plot_df["mean_return"],
            error_x=dict(type="data", array=plot_df["std_return"], visible=True),
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(
                    color=[
                        "#FF8C00" if "Baseline" in m or "Expert" in m else "#084594"
                        for m in plot_df["model"]
                    ],
                    width=2,
                ),
            ),
            text=plot_df["mean_return"].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Mean Return: %{x:.4f}<br>Std: %{error_x.array:.4f}<extra></extra>",
        )
    )

    # Add a vertical line for baseline if it exists
    if baseline_stats is not None:
        baseline_return = baseline_stats["mean_return"]
        fig_bar.add_vline(
            x=baseline_return,
            line_dash="dash",
            line_color="red",
            annotation_text="Baseline",
            annotation_position="top",
        )

    fig_bar.update_layout(
        xaxis_title="Mean Return",
        yaxis_title="Model",
        height=max(400, len(all_stats) * 50),
        showlegend=False,
        hovermode="y unified",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Return Distribution by Model")

    st.markdown("""
    This box plot shows the distribution of returns. The box represents the interquartile range (IQR),
    with the line inside showing the mean. Whiskers extend to min/max values.
    """)

    # Create box plot data
    fig_box = go.Figure()

    for idx, row in stats_df.iterrows():
        # Approximate box plot using available statistics
        # Since we don't have all individual values, we'll show mean, std, min, max
        model = row["model"]
        mean_val = row["mean_return"]
        std_val = row["std_return"]
        min_val = row["min_return"]
        max_val = row["max_return"]

        # Add a trace showing the range
        fig_box.add_trace(
            go.Box(
                y=[model],
                x=[mean_val],
                name=model,
                boxmean="sd",  # Show mean and std
                orientation="h",
                hovertemplate=f"<b>{model}</b><br>Mean: {mean_val:.4f}<br>Std: {std_val:.4f}<br>Min: {min_val:.4f}<br>Max: {max_val:.4f}<extra></extra>",
                # Use statistics to approximate box plot
                lowerfence=[min_val],
                upperfence=[max_val],
                q1=[mean_val - std_val],
                median=[mean_val],
                q3=[mean_val + std_val],
            )
        )

    fig_box.update_layout(
        xaxis_title="Return",
        yaxis_title="Model",
        height=max(400, len(model_names) * 50),
        showlegend=False,
        hovermode="y unified",
    )

    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("Detailed Statistics Comparison")

    # Create comparison metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Mean Return",
            f"{stats_df['mean_return'].max():.4f}",
            delta=None,
            help="Highest average return across all models",
        )
        best_model = stats_df.loc[stats_df["mean_return"].idxmax(), "model"]
        st.caption(f"Model: **{best_model}**")

    with col2:
        st.metric(
            "Best Max Return",
            f"{stats_df['max_return'].max():.4f}",
            delta=None,
            help="Highest single episode return across all models",
        )
        best_max_model = stats_df.loc[stats_df["max_return"].idxmax(), "model"]
        st.caption(f"Model: **{best_max_model}**")

    with col3:
        st.metric(
            "Most Consistent",
            f"{stats_df['std_return'].min():.4f}",
            delta=None,
            help="Lowest standard deviation (most consistent performance)",
        )
        most_consistent_model = stats_df.loc[stats_df["std_return"].idxmin(), "model"]
        st.caption(f"Model: **{most_consistent_model}**")

    st.markdown("---")

    # Detailed comparison table
    st.subheader("Full Statistics Table")

    # Show all available statistics
    display_df = stats_df.copy()

    # Format numeric columns
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "n_episodes":
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Additional insights
st.header("📌 Key Insights")

# Calculate some insights
best_mean_model = stats_df.loc[stats_df["mean_return"].idxmax()]
worst_mean_model = stats_df.loc[stats_df["mean_return"].idxmin()]
return_range = best_mean_model["mean_return"] - worst_mean_model["mean_return"]

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Best Performing Model**: {best_mean_model['model']}
    - Mean Return: {best_mean_model['mean_return']:.4f} ± {best_mean_model['std_return']:.4f}
    - Max Return: {best_mean_model['max_return']:.4f}
    - Episodes: {int(best_mean_model['n_episodes'])}
    """)

with col2:
    st.info(f"""
    **Performance Range**
    - Best to Worst Difference: {return_range:.4f}
    - Total Models Evaluated: {len(all_stats)}
    - Average Std Across Models: {stats_df['std_return'].mean():.4f}
    """)

# Baseline comparison
if baseline_stats is not None:
    st.subheader("🎯 Comparison to Baseline (RL Expert)")

    # Filter IQL models only (exclude baseline)
    iql_models = stats_df[~stats_df["model"].str.contains("Baseline|Expert", na=False)]

    if not iql_models.empty:
        baseline_return = baseline_stats["mean_return"]

        # Calculate performance relative to baseline
        iql_models_copy = iql_models.copy()
        iql_models_copy["relative_performance"] = (
            (iql_models_copy["mean_return"] - baseline_return)
            / abs(baseline_return)
            * 100
        )

        # Find best and worst relative to baseline
        best_relative = iql_models_copy.loc[
            iql_models_copy["relative_performance"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Baseline Mean Return",
                f"{baseline_return:.4f}",
                help="RL expert performance",
            )

        with col2:
            best_iql_return = iql_models_copy["mean_return"].max()
            delta_vs_baseline = best_iql_return - baseline_return
            st.metric(
                "Best IQL Model Return",
                f"{best_iql_return:.4f}",
                delta=f"{delta_vs_baseline:.4f}",
                delta_color="normal" if delta_vs_baseline >= 0 else "inverse",
                help=f"Best: {best_relative['model']}",
            )

        with col3:
            models_above_baseline = len(
                iql_models_copy[iql_models_copy["mean_return"] >= baseline_return]
            )
            st.metric(
                "Models Above Baseline",
                f"{models_above_baseline}/{len(iql_models_copy)}",
                help="Number of IQL models matching or exceeding baseline performance",
            )

        # Show relative performance table
        st.markdown("**Relative Performance (%)**")
        relative_df = iql_models_copy[
            ["model", "mean_return", "relative_performance"]
        ].copy()
        relative_df = relative_df.sort_values("relative_performance", ascending=False)
        relative_df["mean_return"] = relative_df["mean_return"].apply(
            lambda x: f"{x:.4f}"
        )
        relative_df["relative_performance"] = relative_df["relative_performance"].apply(
            lambda x: f"{x:+.2f}%"
        )
        relative_df.columns = ["Model", "Mean Return", "vs Baseline (%)"]
        st.dataframe(relative_df, use_container_width=True, hide_index=True)

# Download data
st.markdown("---")
st.subheader("Download Data")

csv_data = stats_df.to_csv(index=False)
st.download_button(
    label="📥 Download Simulation Statistics CSV",
    data=csv_data,
    file_name="simulation_statistics.csv",
    mime="text/csv",
)
