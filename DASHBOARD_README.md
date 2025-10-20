# IQL Training Results Dashboard

Interactive Streamlit dashboard for visualizing IQL training results and policy rollouts.

## Features

### 📊 Training Metrics
- **Loss Curves**: Total loss, Soft IQ loss, Actor loss, Mismatch loss
- **Episode Rewards**: Mean reward over training with std deviation bands
- Interactive plots with zoom, pan, and hover tooltips

### 🎯 Preference Accuracy
- **MSE & MAE**: Preference prediction error metrics
- **Error Bands**: Visualization of std deviation
- **Summary Statistics**: Final accuracy metrics

### 🎮 Policy Visualization
- **Pygame Rendering**: Real-time policy rollout visualization
- **Episode Info**: Display of reward, action, MO rewards, preferences
- **Interactive**: Click button to play episodes

### ⚙️ Configuration
- View training configuration
- View expert configuration
- Experiment metadata

## Installation

Required packages:
```bash
pip install streamlit plotly pygame
```

Or if using uv:
```bash
uv pip install streamlit plotly pygame
```

## Usage

### Launch Dashboard

**Option 1: Using launch script**
```bash
./run_dashboard.sh
```

**Option 2: Direct streamlit command**
```bash
streamlit run src/IQL/dashboard.py
```

**Option 3: With custom port**
```bash
streamlit run src/IQL/dashboard.py --server.port 8501
```

### Using the Dashboard

1. **Select Results Directory**
   - Default: `moiql_results`
   - Change in sidebar if needed

2. **Choose Experiment**
   - Dropdown shows all experiments (most recent first)
   - Format: `YYYYMMDD_HHMMSS`

3. **Explore Tabs**
   - **Training Metrics**: View loss curves and reward progression
   - **Preference Accuracy**: Analyze preference prediction quality
   - **Policy Visualization**: Watch trained policy in action
   - **Configuration**: Review experiment settings

4. **Play Policy Rollout**
   - Go to "Policy Visualization" tab
   - Click "▶️ Play Episode with Pygame"
   - Watch policy execute in pygame window
   - Close pygame window to return to dashboard

## Dashboard Structure

```
src/IQL/dashboard.py
├── Experiment Selection (Sidebar)
│   ├── Results directory input
│   └── Experiment dropdown
│
└── Main Content (Tabs)
    ├── Tab 1: Training Metrics
    │   ├── Training losses (4 subplots)
    │   └── Episode rewards with std bands
    │
    ├── Tab 2: Preference Accuracy
    │   ├── MSE plot with std bands
    │   ├── MAE plot with std bands
    │   └── Summary statistics
    │
    ├── Tab 3: Policy Visualization
    │   ├── Play button
    │   └── Episode results display
    │
    └── Tab 4: Configuration
        ├── Training config (JSON)
        └── Expert config (JSON)
```

## Metrics Displayed

### Training Losses
- `total_loss`: Combined loss (Soft IQ + Mismatch)
- `soft_iq_loss`: Soft IQ-Learning loss
- `actor_loss`: SAC policy loss
- `mismatch_loss`: Preference-Q mismatch regularization

### Preference Metrics
- `preference_mse`: Mean Squared Error of preference prediction
- `preference_mae`: Mean Absolute Error of preference prediction
- Standard deviations for both metrics

### Episode Metrics
- `mean_episode_reward`: Average total reward per episode
- `std_episode_reward`: Reward standard deviation
- `mean_episode_length`: Average episode length

## Pygame Controls

During policy visualization:
- **Watch**: Policy executes automatically
- **Close Window**: Stop episode and return to dashboard
- **Info Overlay**: Shows step, reward, action, MO rewards, preferences

## Troubleshooting

### No experiments found
- Check results directory path
- Ensure experiments have been run
- Verify directory contains subdirectories with timestamps

### Pygame window not opening
- Check if environment supports rendering
- Verify pygame is installed: `pip list | grep pygame`
- Try running in terminal (not SSH without X forwarding)

### Model loading errors
- Ensure `final_model.pt` exists in experiment directory
- Check config files are present
- Verify model architecture matches config

### Plot not displaying
- Ensure `metrics.csv` exists
- Check CSV file is not corrupted
- Verify pandas can read the file

## Tips

- **Compare Experiments**: Open multiple browser tabs with different experiments
- **Download Plots**: Plotly charts have built-in download buttons
- **Export Data**: Metrics CSV can be opened in Excel/Python for custom analysis
- **Remote Access**: Use `--server.address 0.0.0.0` to access from other machines

## Example Workflow

```bash
# 1. Train IQL agent
python src/IQL/train.py --config configs/iql.yaml

# 2. Launch dashboard
./run_dashboard.sh

# 3. In browser (usually http://localhost:8501)
#    - Select experiment from dropdown
#    - Review training curves
#    - Check preference accuracy
#    - Play policy rollout
#    - Compare with other experiments
```

## Advanced Usage

### Custom Results Directory
```python
# In dashboard sidebar
Results Directory: custom_results/iql_experiments
```

### Multiple Experiments Analysis
1. Train multiple configs
2. Open dashboard
3. Switch between experiments in dropdown
4. Compare metrics visually

### Exporting Plots
- Click camera icon in plot toolbar
- Download as PNG
- Use in papers/presentations
