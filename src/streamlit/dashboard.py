import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import wandb

st.set_page_config(page_title="MOIQL Dashboard", layout="wide")

st.title("Multi-Objective Inverse Q-Learning Dashboard")


@st.cache_data
def load_wandb_data(project_name: str, run_id: str = None):
    api = wandb.Api()

    if run_id:
        run = api.run(f"{project_name}/{run_id}")
        history = run.history()
        return history, run
    else:
        runs = api.runs(project_name)
        return runs, None


st.sidebar.header("Configuration")
project_name = st.sidebar.text_input("W&B Project Name", value="moiql")
run_selector = st.sidebar.selectbox("Run Selection", ["Latest", "Specific Run ID"])

if run_selector == "Specific Run ID":
    run_id = st.sidebar.text_input("Run ID")
else:
    run_id = None

if st.sidebar.button("Load Data"):
    try:
        if run_selector == "Latest":
            runs, _ = load_wandb_data(project_name)
            if len(runs) > 0:
                latest_run = runs[0]
                history = latest_run.history()
                st.session_state["data"] = history
                st.session_state["run_name"] = latest_run.name
                st.success(f"Loaded latest run: {latest_run.name}")
            else:
                st.error("No runs found")
        else:
            history, run = load_wandb_data(project_name, run_id)
            st.session_state["data"] = history
            st.session_state["run_name"] = run.name
            st.success(f"Loaded run: {run.name}")
    except Exception as e:
        st.error(f"Error loading data: {e}")

if "data" in st.session_state:
    df = st.session_state["data"]

    st.header(f"Run: {st.session_state.get('run_name', 'Unknown')}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Total Reward",
            f"{df['total_reward'].iloc[-1]:.2f}" if "total_reward" in df else "N/A",
        )

    with col2:
        st.metric(
            "Final Total Regret",
            f"{df['total_regret'].iloc[-1]:.2f}" if "total_regret" in df else "N/A",
        )

    with col3:
        st.metric(
            "Final Q-Loss", f"{df['q_loss'].iloc[-1]:.4f}" if "q_loss" in df else "N/A"
        )

    with col4:
        st.metric("Total Epochs", len(df))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Reward Curves", "Regret Curves", "Loss Curves", "Pareto Front"]
    )

    with tab1:
        st.subheader("Reward Progress")

        fig_reward = go.Figure()

        if "reward_obj0" in df:
            fig_reward.add_trace(
                go.Scatter(
                    x=df["epoch"], y=df["reward_obj0"], mode="lines", name="Objective 0"
                )
            )

        if "reward_obj1" in df:
            fig_reward.add_trace(
                go.Scatter(
                    x=df["epoch"], y=df["reward_obj1"], mode="lines", name="Objective 1"
                )
            )

        if "total_reward" in df:
            fig_reward.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["total_reward"],
                    mode="lines",
                    name="Total Reward",
                    line=dict(dash="dash"),
                )
            )

        fig_reward.update_layout(
            xaxis_title="Epoch", yaxis_title="Reward", hovermode="x unified"
        )

        st.plotly_chart(fig_reward, use_container_width=True)

    with tab2:
        st.subheader("Regret Progress")

        fig_regret = go.Figure()

        if "regret_obj0" in df:
            fig_regret.add_trace(
                go.Scatter(
                    x=df["epoch"], y=df["regret_obj0"], mode="lines", name="Objective 0"
                )
            )

        if "regret_obj1" in df:
            fig_regret.add_trace(
                go.Scatter(
                    x=df["epoch"], y=df["regret_obj1"], mode="lines", name="Objective 1"
                )
            )

        if "total_regret" in df:
            fig_regret.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["total_regret"],
                    mode="lines",
                    name="Total Regret",
                    line=dict(dash="dash"),
                )
            )

        fig_regret.update_layout(
            xaxis_title="Epoch", yaxis_title="Regret", hovermode="x unified"
        )

        st.plotly_chart(fig_regret, use_container_width=True)

    with tab3:
        st.subheader("Training Loss")

        fig_loss = go.Figure()

        if "q_loss" in df:
            fig_loss.add_trace(
                go.Scatter(x=df["epoch"], y=df["q_loss"], mode="lines", name="Q-Loss")
            )

        fig_loss.update_layout(
            xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified"
        )

        st.plotly_chart(fig_loss, use_container_width=True)

    with tab4:
        st.subheader("Pareto Front Evolution")

        if "reward_obj0" in df and "reward_obj1" in df:
            window_size = st.slider("Epoch Window", 1, len(df), len(df))

            df_window = df.tail(window_size)

            fig_pareto = px.scatter(
                df_window,
                x="reward_obj0",
                y="reward_obj1",
                color="epoch",
                title="Reward Space (Objective 0 vs Objective 1)",
                labels={
                    "reward_obj0": "Objective 0 Reward",
                    "reward_obj1": "Objective 1 Reward",
                },
                color_continuous_scale="viridis",
            )

            fig_pareto.update_traces(marker=dict(size=8))

            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.info("Pareto front requires reward_obj0 and reward_obj1 data")

    with st.expander("Raw Data"):
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV", csv, "training_data.csv", "text/csv", key="download-csv"
        )

else:
    st.info("Please configure and load a run from the sidebar")
