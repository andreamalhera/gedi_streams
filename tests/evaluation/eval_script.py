# Activate virtual environment if necessary: source .venv/bin/activate

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import inspect
import builtins

# Assuming gedi_streams is importable
from gedi_streams.generator.generator import StreamProcessingManager
from gedi_streams.utils.param_keys.features import FEATURE_PARAMS, FEATURE_SET
from gedi_streams.utils.param_keys import OUTPUT_PATH, INPUT_PATH, PIPELINE_STEP
# Import config or define DEFAULT_CONFIG_SPACE
try:
    import config # Assuming config.py exists and defines DEFAULT_CONFIG_SPACE
except ImportError:
    print("Warning: 'config.py' not found or 'DEFAULT_CONFIG_SPACE' not defined.")
    print("Using a placeholder DEFAULT_CONFIG_SPACE.")
    # Define a placeholder if config is not available
    DEFAULT_CONFIG_SPACE = {
        "mode": (5, 20), "sequence": (0.1, 0.9), "choice": (0.1, 0.9),
        "loop": (0.1, 0.7), "silent": (0.1, 0.5), "lt_dependency": (0.1, 0.9),
        "num_traces": (50, 200), "parallel": 0.0, "duplicate": 0.0, "or": 0.0
    }
    config = type('obj', (object,), {'DEFAULT_CONFIG_SPACE': DEFAULT_CONFIG_SPACE})()


# --- Feature Set Definitions (from user's main.py) ---
stream_features = [
    "activity_appearance_rate", "variant_appearance_rate", "drift_indicator",
    "direct_follows_entropy", "trace_length_variability", "concurrent_activities_ratio",
    "activity_entropy", "unique_paths_ratio", "structured_complexity",
    "long_term_activity_shift", "variant_stability", "throughput_trend",
    "cycle_time_variation"
]

advanced_features = [
    'window_entropy_variability', 'drift_gradualness', 'variant_evolution_rate',
    'recurrence_factor', 'temporal_locality', 'case_overlap_ratio',
    'path_consistency', 'stream_homogeneity', 'loop_structure_stability',
    'reachability_preservation' # Removed duplicate 'stream_homogeneity'
]

baseline_features = [
    'temporal_dependency', 'case_concurrency', 'concept_stability',
    'case_throughput_stability', 'parallel_activity_ratio',
    'activity_duration_stability', 'case_priority_dynamics',
    # 'concept_drift', # This seems to be defined in StructuredStreamFeature, check if intended
    'long_term_dependencies',
]


# --- Configuration (from user's main.py) ---
PRINT_EVENTS = True # Set to False for less console output during simulation
N_WINDOWS = 5       # Number of windows to simulate (increased for better visualization)
WINDOW_SIZE = 50    # Number of traces per window
FEATURE_SET_TO_USE = advanced_features # Choose which feature set to analyze

INPUT_PARAMS: dict = {
    'pipeline_step': 'feature_extraction', # Context for internal logic if any
    'input_path': 'data/test', # Often placeholder for stream simulation
    FEATURE_PARAMS: {        # Use the imported key name 'feature_params'
        FEATURE_SET: FEATURE_SET_TO_USE # Use the imported key name 'feature_set'
    },
    "config_space": config.DEFAULT_CONFIG_SPACE, # From config or placeholder
    'output_path': 'output/stream_analysis_custom_plots', # Base output dir
    # Optional: Path to real event log features for comparison
    'real_eventlog_path': 'data/BaselineED_feat.csv', # Example path
    # --- Plotting hints (used by custom code below) ---
    'plot_type': 'violinplot', # 'violinplot' or 'boxplot' for distributions
    'font_size': 12,         # Font size for plots
    # 'boxplot_width': 10    # Less relevant for multi-plot figures
}

# --- Setup Output ---
output_directory = INPUT_PARAMS['output_path']
plot_output_directory = os.path.join(output_directory, "plots")
stream_features_path = os.path.join(output_directory, "stream_features_over_windows.csv")

os.makedirs(plot_output_directory, exist_ok=True)

# --- Custom Plotting Configuration ---
plot_style = 'seaborn-v0_8-whitegrid'
plt.style.use(plot_style)
plot_font_size = INPUT_PARAMS.get('font_size', 12) # Use font_size from INPUT_PARAMS

# --- Custom Print Function (from user's main.py) ---
# This helps trace where print statements originate
original_print = builtins.print
def custom_print(*args, **kwargs):
    try:
        caller_frame = inspect.currentframe().f_back
        frame_info = inspect.getframeinfo(caller_frame)
        file_name = frame_info.filename
        line_number = frame_info.lineno
        cwd = os.getcwd()
        if file_name.startswith(cwd):
            relative_path = os.path.relpath(file_name, cwd)
            file_name = relative_path
        prefix = f"[{file_name}:{line_number}]"
        original_print(prefix, *args, **kwargs)
    except Exception: # Avoid errors within print itself
        original_print(*args, **kwargs)
builtins.print = custom_print

# --- Custom Plotting Functions ---

def plot_feature_evolution(features_df, save_path):
    """
    Creates time series plots showing the evolution of each feature over windows.
    """
    feature_cols = features_df.columns
    num_features = len(feature_cols)
    if num_features == 0:
        print("Warning: No features to plot for evolution.")
        return

    # Determine subplot layout
    ncols = int(math.ceil(math.sqrt(num_features)))
    nrows = int(math.ceil(num_features / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), squeeze=False, sharex=True)
    axes = axes.flatten()

    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        ax.plot(features_df.index, features_df[feature], marker='o', linestyle='-', markersize=4)
        ax.set_title(feature.replace('_', ' ').title(), fontsize=plot_font_size * 0.9)
        ax.tick_params(axis='y', labelsize=plot_font_size * 0.8)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i >= num_features - ncols: # Add x-label only to bottom row
             ax.set_xlabel("Window Number", fontsize=plot_font_size * 0.9)
        ax.tick_params(axis='x', labelsize=plot_font_size * 0.8)


    # Hide unused subplots
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Feature Evolution Over Simulated Windows', fontsize=plot_font_size * 1.2, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Feature Evolution plot to: {save_path}")


def plot_feature_distribution_across_windows(stream_features_df, real_features_df, plot_kind, save_path):
    """
    Creates plots comparing distributions of features across stream windows vs. real features.
    """
    plot_stream_df_numeric = stream_features_df.select_dtypes(include=np.number)

    if real_features_df is not None:
        plot_real_df_numeric = real_features_df.select_dtypes(include=np.number)
        common_cols = list(set(plot_stream_df_numeric.columns) & set(plot_real_df_numeric.columns))
        if not common_cols:
             print("Warning: Skipping distribution plot. No common numeric feature columns between stream and real features.")
             return

        # Prepare data: Melt stream features, keep real features as is for comparison reference
        df_stream_melt = plot_stream_df_numeric[common_cols].melt(var_name='Feature', value_name='Value')
        df_stream_melt['Source'] = 'Stream Windows'

        df_real = plot_real_df_numeric[common_cols].copy()
        df_real_melt = df_real.melt(var_name='Feature', value_name='Value')
        df_real_melt['Source'] = 'Real Log'

        combined_df = pd.concat([df_stream_melt, df_real_melt], ignore_index=True)
        hue_order = ['Real Log', 'Stream Windows']
        palette = {'Real Log': 'skyblue', 'Stream Windows': 'lightcoral'}
        plot_mode = 'compare'
    else:
        # Only plot distribution across stream windows
        common_cols = list(plot_stream_df_numeric.columns)
        if not common_cols:
             print("Warning: Skipping distribution plot. No numeric feature columns found in stream features.")
             return
        combined_df = plot_stream_df_numeric[common_cols].melt(var_name='Feature', value_name='Value')
        combined_df['Source'] = 'Stream Windows' # Still useful for consistent plotting code
        hue_order=['Stream Windows']
        palette = {'Stream Windows': 'lightcoral'}
        plot_mode = 'stream_only'

    num_features = len(common_cols)
    if num_features == 0: return

    # Determine subplot layout
    ncols = int(math.ceil(math.sqrt(num_features)))
    nrows = int(math.ceil(num_features / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes = axes.flatten()

    plot_func = sns.violinplot if plot_kind == 'violinplot' else sns.boxplot

    for i, feature in enumerate(common_cols):
        ax = axes[i]
        feature_data = combined_df[combined_df['Feature'] == feature]

        plot_func(data=feature_data, x='Source', y='Value', ax=ax, palette=palette, order=hue_order,
                  linewidth=1, inner='quartile' if plot_kind=='violinplot' else None, cut=0 if plot_kind=='violinplot' else None)

        # Overlay strip plot for individual points
        sns.stripplot(data=feature_data, x='Source', y='Value', ax=ax, hue='Source', order=hue_order,
                       palette={'Real Log': '#56B4E9', 'Stream Windows': '#D55E00'}, # Use distinct colors
                       dodge=True, size=3.5, alpha=0.6, legend=False)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(feature.replace('_', ' ').title(), fontsize=plot_font_size)
        ax.tick_params(axis='both', which='major', labelsize=plot_font_size*0.9)

    # Hide unused subplots
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Feature Distributions: Stream Windows vs Real Log' if plot_mode == 'compare' else 'Feature Distributions Across Stream Windows',
                 fontsize=plot_font_size * 1.2, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved Feature Distribution ({plot_kind}) plot to: {save_path}")


# --- Main Execution ---
if __name__=='__main__':
    print(f"\n--- Running Stream Simulation ({N_WINDOWS} windows, {WINDOW_SIZE} traces/window) ---")
    print(f"Using feature set: {FEATURE_SET_TO_USE}")

    # Run the DEFact stream simulation wrapper
    # This returns a list of dicts, where each dict contains the features for one window
    all_features_over_windows = StreamProcessingManager.defact_wrapper(
        N_WINDOWS, INPUT_PARAMS, WINDOW_SIZE, PRINT_EVENTS
    )

    if not all_features_over_windows:
        print("ERROR: Stream simulation did not return any features.")
        exit()

    print("\n--- Stream Simulation Finished ---")
    print(f"Collected features for {len(all_features_over_windows)} windows.")

    # Convert list of feature dicts to DataFrame
    features_df = pd.DataFrame(all_features_over_windows)
    # Add a window index column if desired (optional, index serves this purpose)
    # features_df['window'] = range(len(features_df))

    # Save the features DataFrame
    features_df.to_csv(stream_features_path, index_label="window_index")
    print(f"Saved stream features over windows to: {stream_features_path}")

    # --- Execute Custom Plotting ---
    print("\n--- Generating Custom Plots ---")

    # 1. Plot Feature Evolution over Windows
    evo_plot_path = os.path.join(plot_output_directory, "feature_evolution_over_windows.png")
    plot_feature_evolution(features_df, evo_plot_path)

    # 2. Plot Feature Distributions (across windows, optionally vs real)
    real_features_df = None
    real_log_path = INPUT_PARAMS.get('real_eventlog_path')
    if real_log_path and os.path.exists(real_log_path):
        try:
            real_features_df = pd.read_csv(real_log_path)
            print(f"Loaded real features for comparison from: {real_log_path}")
        except Exception as e:
            print(f"Warning: Could not load real features CSV '{real_log_path}'. Error: {e}")
            real_features_df = None
    else:
        print("No real event log features provided or path invalid.")

    dist_plot_path = os.path.join(plot_output_directory, f"feature_distributions_{INPUT_PARAMS.get('plot_type', 'violinplot')}.png")
    plot_feature_distribution_across_windows(
        stream_features_df=features_df,
        real_features_df=real_features_df,
        plot_kind=INPUT_PARAMS.get('plot_type', 'violinplot'), # Use plot_type from config
        save_path=dist_plot_path
    )

    print("\n--- Analysis Script Finished ---")
    # Restore original print if needed (outside __main__)
    # builtins.print = original_print