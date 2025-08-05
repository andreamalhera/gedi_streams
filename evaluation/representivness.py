from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from os import listdir
from os.path import isfile, join

from pathlib import Path
from typing import List, Mapping, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def temporal_dependency_strength(df: pd.DataFrame) -> float:
    """
    Calculate the temporal dependency strength of the dataset by measuring
    the autocorrelation of inter-event times.

    :param df: pandas DataFrame containing event log with 'timestamp' column.
    :return: strength value between 0 and 1.
    """
    ts = pd.to_datetime(df['time:timestamp']).sort_values().reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds().dropna()
    if len(diffs) < 2:
        return 0.0
    acf1 = diffs.autocorr(lag=1)
    if np.isnan(acf1):
        return 0.0
    return float(min(max(abs(acf1), 0.0), 1.0))


def long_term_dependency_strength(df: pd.DataFrame) -> float:
    """
    Calculate the long-term dependency strength by estimating the Hurst exponent
    of inter-event times.

    :param df: pandas DataFrame containing event log with 'timestamp' column.
    :return: strength value between 0 and 1.
    """
    ts = pd.to_datetime(df['time:timestamp']).sort_values().reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds().dropna().values
    N = len(diffs)
    if N < 20:
        return 0.0
    lags = np.floor(np.logspace(0.5, np.log10(N / 2), num=20)).astype(int)
    tau = []
    for lag in lags:
        d = diffs[lag:] - diffs[:-lag]
        tau.append(np.sqrt(np.std(d)))
    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = float(slope)
    strength = (hurst - 0.5) * 2
    return float(min(max(strength, 0.0), 1.0))


def non_linear_dependency_strength(df: pd.DataFrame) -> float:
    """
    Calculate the non-linear dependency strength by computing the normalized mutual
    information between consecutive inter-event times.

    :param df: pandas DataFrame containing event log with 'timestamp' column.
    :return: strength value between 0 and 1.
    """
    ts = pd.to_datetime(df['time:timestamp']).sort_values().reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds().dropna().values
    if len(diffs) < 2:
        return 0.0
    bins = np.histogram_bin_edges(diffs, bins=10)
    binned = np.digitize(diffs, bins)
    x = binned[:-1]
    y = binned[1:]
    mi = mutual_info_score(x, y)
    h = mutual_info_score(y, y)
    if h == 0:
        return 0.0
    strength = mi / h
    return float(min(max(strength, 0.0), 1.0))


def out_of_order_strength(df: pd.DataFrame) -> float:
    """
    Calculate out-of-order strength by measuring the fraction and magnitude of
    out-of-order timestamps.

    :param df: pandas DataFrame containing event log with 'timestamp' column.
    :return: strength value between 0 and 1.
    """
    ts = pd.to_datetime(df['time:timestamp']).reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 0.0
    rev = diffs < 0
    frac = float(rev.mean())
    if frac == 0:
        return 0.0
    mag = float(np.mean(np.abs(diffs[rev])))
    total_range = (ts.max() - ts.min()).total_seconds()
    mag_norm = mag / total_range if total_range > 0 else 0.0
    strength = (frac + mag_norm) / 2
    return float(min(max(strength, 0.0), 1.0))


def fractal_behavior_strength(df: pd.DataFrame) -> float:
    """
    Calculate fractal behavior strength using the box-counting method on event times.

    :param df: pandas DataFrame containing event log with 'timestamp' column.
    :return: strength value between 0 and 1.
    """
    ts = pd.to_datetime(df['time:timestamp']).sort_values().reset_index(drop=True)
    times = (ts - ts.iloc[0]).dt.total_seconds().values
    if len(times) < 2:
        return 0.0
    duration = times[-1]
    if duration <= 0:
        return 0.0
    scales = np.logspace(np.log10(duration / 100),
                         np.log10(duration / 2), num=10)
    Ns = []
    for r in scales:
        boxes = int(np.ceil(duration / r))
        idx = np.clip((times / r).astype(int), 0, boxes - 1)
        present = np.zeros(boxes)
        present[idx] = 1
        Ns.append(present.sum())
    logs = np.log(scales)
    logNs = np.log(Ns)
    slope, _ = np.polyfit(logs, logNs, 1)
    fractal_dim = -float(slope)
    return float(min(max(fractal_dim, 0.0), 1.0))


def calculate_features_with_window(df: pd.DataFrame, window_size: int = 1000) -> pd.DataFrame:
    """
    Calculate temporal features using a sliding window approach.
    
    :param df: pandas DataFrame containing event log with 'time:timestamp' column
    :param window_size: Size of the sliding window (number of events)
    :return: DataFrame with calculated features for each window
    """
    # Sort by timestamp
    df_sorted = df.sort_values('time:timestamp').reset_index(drop=True)
    
    features_list = []
    step_size = max(1, window_size // 4)  # Overlap windows for more data points
    
    for start_idx in range(0, len(df_sorted) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_df = df_sorted.iloc[start_idx:end_idx].copy()
        
        if len(window_df) < 10:  # Skip if window too small
            continue
            
        features = {
            'temporal_dependency_strength': temporal_dependency_strength(window_df),
            'long_term_dependency_strength': long_term_dependency_strength(window_df),
            'non_linear_dependency_strength': non_linear_dependency_strength(window_df),
            'out_of_order_strength': out_of_order_strength(window_df),
            'fractal_behavior_strength': fractal_behavior_strength(window_df)
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def remove_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove rows from `data` where any feature has an absolute z‑score greater than `threshold`.

    :param data: 2‑D array containing numeric samples.
    :param threshold: Absolute z‑score cut‑off.
    :return: A new array with the outlier rows removed.
    """
    if data.ndim != 2:
        raise ValueError("`data` must be a 2‑D array.")

    mean: np.ndarray = data.mean(axis=0)
    std: np.ndarray = data.std(axis=0, ddof=0)

    z_scores: np.ndarray = (data - mean) / std
    mask: np.ndarray = np.all(np.abs(z_scores) < threshold, axis=1)

    return data[mask]


def compute_hull(points: np.ndarray) -> ConvexHull | None:
    """
    Build a convex hull for 2-D points.

    :param points: Array shaped (n_samples, 2).
    :return: ConvexHull instance or None if the hull is degenerate.
    """
    if points.shape[0] < 3:
        return None
    try:
        return ConvexHull(points)
    except Exception:
        return None

def find_csv_files_with_prefix(directory: Path, prefix: str) -> List[Path]:
    """
    Recursively search for CSV files in the given directory and its subdirectories
    whose filenames start with the specified prefix.

    :param directory: Directory in which to search for CSV files.
    :param prefix: Starting substring of the CSV filenames to match.
    :return : of objects.
    :return: List of Path objects pointing to matching CSV files.
    """
    l = [f for f in listdir(directory) if isfile(join(directory, f))]

    return [
        directory / file for file in l if file.endswith('.csv') and file.startswith(prefix)
    ]

def hull_area(hull: ConvexHull | None) -> float:
    """
    Area of a 2-D convex hull.

    :param hull: ConvexHull returned by compute_hull.
    :return: Area (0.0 if hull is None).
    """
    return float(getattr(hull, "volume", 0.0)) if hull is not None else 0.0


def percentage_increase(orange: float, blue: float) -> float:
    """
    Percentage increase from blue area to orange area.

    :param orange: Area of the generator (orange) hull.
    :param blue: Area of the real-log (blue) hull.
    :return: Percentage by which orange exceeds blue (0.0 if blue is 0).
    """
    if blue == 0.0:
        return 0.0
    return (orange - blue) / blue * 100.0


def calculate_experimental_features(experiments_folder: str) -> pd.DataFrame:
    """
    Load experimental event logs and calculate temporal features for each using sliding window.
    
    :param experiments_folder: Path to folder containing experiment CSV files
    :return: DataFrame with calculated features from all experimental event logs
    """
    exp_path = Path(experiments_folder)
    exp_files = find_csv_files_with_prefix(exp_path, "experiment_")

    if not exp_files:
        print(f"No experiment_ files found in {exp_path}")
        return pd.DataFrame()
    
    all_features = []
    
    for file in exp_files:
        try:
            print(f"Processing experimental file: {file.name}")
            # Load the experimental event log
            event_log_df = load_event_log(file)
            
            # Calculate features using sliding window
            features_df = calculate_features_with_window(
                event_log_df, 
                window_size=min(1000, len(event_log_df) // 4)
            )
            
            if len(features_df) > 0:
                all_features.append(features_df)
                
        except Exception as e:
            print(f"Error processing experimental file {file}: {e}")
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    else:
        return pd.DataFrame()


def load_event_log(file_path: str) -> pd.DataFrame:
    """
    Load an event log CSV file and ensure it has the required timestamp column.
    
    :param file_path: Path to the CSV file
    :return: DataFrame with event log data
    """
    df = pd.read_csv(file_path)
    
    # Try to find timestamp column with different possible names
    timestamp_cols = ['time:timestamp', 'timestamp', 'time', 'Time', 'Timestamp']
    timestamp_col = None
    
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {file_path}. Available columns: {df.columns.tolist()}")
    
    # Standardize column name
    if timestamp_col != 'time:timestamp':
        df = df.rename(columns={timestamp_col: 'time:timestamp'})
    
    return df


# Constants for plot configuration
FONT_SIZE: int = 12
SUBFIGURE_SIZE: tuple[int, int] = (4, 3)

COLUMNS: list[str] = [
    "temporal_dependency_strength",
    "long_term_dependency_strength", 
    "non_linear_dependency_strength",
    "out_of_order_strength",
    "fractal_behavior_strength"
]


def main() -> None:
    """
    Plot PCA clouds for each event log and print hull-overlap statistics.
    """
    # Apply global font size
    plt.rcParams.update({"font.size": FONT_SIZE})

    # Calculate features for experimental data (orange/yellow hull)
    print("Loading and processing experimental data...")
    experimental_df = calculate_experimental_features(EXPERIMENTS_FOLDER)
    
    if experimental_df.empty:
        print("No experimental data found!")
        return
    
    # Prepare experimental data for PCA
    exp_numeric = experimental_df[COLUMNS].select_dtypes(include=[np.number]).dropna()
    
    exp_pca = None
    exp_hull = None
    if len(exp_numeric) >= 3:
        exp_scaled = StandardScaler().fit_transform(exp_numeric)
        exp_pca = PCA(n_components=2).fit_transform(exp_scaled)
        exp_pca = remove_outliers_zscore(exp_pca, threshold=3.0)
        exp_hull = compute_hull(exp_pca)

    orange_area = hull_area(exp_hull)

    # Find event log files
    event_logs_path = Path(EVENT_LOGS_FOLDER)
    csv_files = [f for f in event_logs_path.glob("*.csv") if not f.name.startswith("experiment_")]
    
    if not csv_files:
        print(f"No event log CSV files found in {EVENT_LOGS_FOLDER}")
        return

    # Calculate grid layout
    n_files = len(csv_files)
    n_cols = 3
    n_rows = (n_files + n_cols - 1) // n_cols

    # Create figure
    fig_width = SUBFIGURE_SIZE[0] * n_cols
    fig_height = SUBFIGURE_SIZE[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Print header
    header = (
        f"{'Event-log':<30s}"
        f"{'Area experimental':>15s}"
        f"{'Area real':>12s}"
        f"{'Diff':>12s}"
        f"{'% inc.':>12s}"
    )
    print(header)
    print("-" * len(header))

    for i, csv_file in enumerate(csv_files):
        try:
            print(f"Processing {csv_file.name}...")
            
            # Load event log
            event_log_df = load_event_log(csv_file)
            
            # Calculate features using sliding window
            features_df = calculate_features_with_window(event_log_df, window_size=min(10000, len(event_log_df) // 4))
            
            if len(features_df) < 3:
                axes[i].set_title(f"{csv_file.stem} (insufficient data)")
                axes[i].axis("off")
                continue

            # Prepare data for PCA
            numeric_df = features_df[COLUMNS].select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_df) < 3:
                axes[i].set_title(f"{csv_file.stem} (insufficient data)")
                axes[i].axis("off")
                continue

            # Perform PCA
            scaled = StandardScaler().fit_transform(numeric_df)
            pca_res = PCA(n_components=2).fit_transform(scaled)
            pca_res = remove_outliers_zscore(pca_res, threshold=3.0)

            # Plot blue points (real event log)
            axes[i].scatter(pca_res[:, 0], pca_res[:, 1], 
                          c='blue', s=20, alpha=0.6, label='Real log')

            # Calculate and plot blue hull
            blue_hull = compute_hull(pca_res)
            blue_area = hull_area(blue_hull)

            if blue_hull is not None:
                for simplex in blue_hull.simplices:
                    axes[i].plot(pca_res[simplex, 0], pca_res[simplex, 1], "b-", alpha=0.8)
                axes[i].fill(
                    pca_res[blue_hull.vertices, 0],
                    pca_res[blue_hull.vertices, 1],
                    "blue",
                    alpha=0.1,
                )

            # Plot experimental data (orange/yellow points)
            if exp_pca is not None:
                axes[i].scatter(exp_pca[:, 0], exp_pca[:, 1], 
                              c='orange', s=20, alpha=0.6, label='Experimental')
                
                if exp_hull is not None:
                    for simplex in exp_hull.simplices:
                        axes[i].plot(
                            exp_pca[simplex, 0], exp_pca[simplex, 1], 
                            color="orange", alpha=0.8
                        )
                    axes[i].fill(
                        exp_pca[exp_hull.vertices, 0],
                        exp_pca[exp_hull.vertices, 1],
                        "orange",
                        alpha=0.1,
                    )

            axes[i].set_title(csv_file.stem, fontsize=10)
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].grid(True, alpha=0.3)

            # Calculate statistics
            diff = orange_area - blue_area
            pct_inc = percentage_increase(orange_area, blue_area)

            print(
                f"{csv_file.stem:<30s}"
                f"{orange_area:>15.4f}"
                f"{blue_area:>12.4f}"
                f"{diff:>12.4f}"
                f"{pct_inc:>12.2f}"
            )

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            axes[i].set_title(f"{csv_file.stem} (error)")
            axes[i].axis("off")

    # Hide unused subplots
    for j in range(len(csv_files), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Update these paths to match your directory structure
    EVENT_LOGS_FOLDER = "/Users/christianimenkamp/Documents/Data-Repository/Community/csv_only"
    EXPERIMENTS_FOLDER = os.getcwd() + "/experiment_results"
    
    main()