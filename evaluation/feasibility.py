#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ───────────────────────────────────────── Constants ─────────────────────────────────────────

FONT_SIZE: int = 25
FONT_SIZE_AXIS_LABELS: int = FONT_SIZE
FONT_SIZE_CELL_TEXT: int = FONT_SIZE



# Event Stream Generator parameters
TEST_LABELS: List[str] = [
    "temporal_dependency_strength",
    "long_term_dependency_strength", 
    "non_linear_dependency_strength",
    "out_of_order_strength",
    "fractal_behavior_strength"
]

VALUE_SET: List[float] = [0.0, 0.1, 0.5, 0.7, 1.0]
VALUE_INDEX: Dict[float, int] = {val: idx for idx, val in enumerate(VALUE_SET)}

# Map internal names → display labels for event stream generator
DISPLAY_LABELS: Dict[str, str] = {
    "temporal_dependency_strength": "Temporal\nDependency\nStrength",
    "long_term_dependency_strength": "Long Term\nDependency\nStrength", 
    "non_linear_dependency_strength": "Non Linear\nDependency\nStrength",
    "out_of_order_strength": "Out of Order\nStrength",
    "fractal_behavior_strength": "Fractal\nBehavior\nStrength"
}

OUTPUT_DIR: str = "../../../../Desktop/Stream Enviorment/experiment_results"

N_LABELS: int = len(TEST_LABELS)
MOSAIC_DETAIL: int = len(VALUE_SET)
FIG_SIZE: Tuple[float, float] = (16.0, 12.0)
COLOR_MAP_NAME: str = "viridis"
TEXT_COLOR_THRESHOLD: float = 0.45

# ────────────────────────────────────────── Helpers ──────────────────────────────────────────

def display_label(name: str) -> str:
    """
    Translate an internal parameter name into the presentation label.

    :param name: Internal identifier as used in the CSV files.
    :return: Label to show on the figure (falls back to *name* if not mapped).
    """
    return DISPLAY_LABELS.get(name, name)


def parse_filename(filename: str) -> Optional[Tuple[str, float, str, float]]:
    """
    Extract the two parameter/value pairs from an experiment CSV filename.
    Now handles comma decimal separators (e.g., "0,5" instead of "0.5").

    :param filename: The CSV filename.
    :return: (param₁, value₁, param₂, value₂) or *None* if the pattern does not match.
    """
    # Remove extension and experiment prefix
    if not filename.startswith("experiment_") or not filename.endswith(".csv"):
        return None
    
    # Remove "experiment_" and ".csv"
    core = filename[11:-4]  # Remove "experiment_" (11 chars) and ".csv" (4 chars)
    
    # Find the pattern: NUM_PARAM1_vs_PARAM2_VAL1_VAL2
    # First remove the experiment number
    parts = core.split("_")
    if len(parts) < 5:  # Need at least: num, param1, vs, param2, val1, val2
        return None
    
    # Find the "vs" separator
    vs_index = -1
    for i, part in enumerate(parts):
        if part == "vs":
            vs_index = i
            break
    
    if vs_index == -1 or vs_index < 2:  # Need at least one part before "vs"
        return None
    
    # The last two parts should be the values
    try:
        # Replace comma with dot for proper float conversion
        val2_str = parts[-1].replace(',', '.')
        val1_str = parts[-2].replace(',', '.')
        val2 = float(val2_str)
        val1 = float(val1_str)
    except (ValueError, IndexError):
        return None
    
    # Everything before vs_index (except first part which is the number) is param1
    # Everything after vs_index and before the last two parts is param2
    param1_parts = parts[1:vs_index]
    param2_parts = parts[vs_index+1:-2]
    
    if not param1_parts or not param2_parts:
        return None
    
    param1 = "_".join(param1_parts)
    param2 = "_".join(param2_parts)
    
    return param1, val1, param2, val2


def read_experiment_data(filepath: str) -> Optional[Dict[str, float]]:
    """
    Read experiment CSV and extract metrics from event log data.

    :param filepath: Path of the experiment CSV file.
    :return: Mapping of metrics or *None* if the file is empty/unreadable.
    """
    try:
        df: pd.DataFrame = pd.read_csv(filepath)
        if df.empty:
            return None
        
        # Calculate metrics from the event log data
        metrics = calculate_metrics_from_event_log(df)
        
        if not metrics:
            return None
            
        return metrics
    except (FileNotFoundError, pd.errors.ParserError, Exception) as e:
        print(f"Error reading {filepath}: {e}")
        return None


def calculate_metrics_from_event_log(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics from event log data.
    """
    if df.empty:
        return {}
    
    metrics = {}
    
    try:
        # Basic event counts
        metrics['total_events'] = len(df)
        metrics['unique_cases'] = df['case:concept:name'].nunique() if 'case:concept:name' in df.columns else 0
        metrics['unique_activities'] = df['concept:name'].nunique() if 'concept:name' in df.columns else 0
        
        # Time-based metrics
        if 'time:timestamp' in df.columns:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
            time_diffs = df['time:timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                metrics['avg_time_between_events'] = time_diffs.dt.total_seconds().mean() / 60.0  # minutes
            else:
                metrics['avg_time_between_events'] = 0.0
        
        # Out of order events
        out_of_order_count = 0
        if 'attributes' in df.columns:
            for attr_str in df['attributes'].dropna():
                if 'out_of_order' in str(attr_str) and 'True' in str(attr_str):
                    out_of_order_count += 1
        metrics['out_of_order_percentage'] = (out_of_order_count / len(df)) * 100.0 if len(df) > 0 else 0.0
        
        # Activity diversity (simplified)
        if metrics['unique_cases'] > 0:
            metrics['activity_diversity'] = metrics['unique_activities'] / metrics['unique_cases']
        else:
            metrics['activity_diversity'] = 0.0
            
        # Fractal behavior (simplified - based on repeated patterns)
        fractal_count = 0
        if 'concept:name' in df.columns:
            activity_names = df['concept:name'].tolist()
            # Look for repeated activity patterns
            for i in range(len(activity_names) - 1):
                if i > 0 and activity_names[i] == activity_names[i-1]:
                    fractal_count += 1
        metrics['fractal_percentage'] = (fractal_count / len(df)) * 100.0 if len(df) > 0 else 0.0
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}
    
    return metrics


def calculate_parameter_difference(config_val: float, expected_val: float, 
                                   actual_metrics: Dict[str, float], 
                                   param_name: str) -> float:
    """
    Calculate the difference between expected and actual parameter effectiveness.
    """
    # Map parameters to their key output metrics
    param_to_metric = {
        "temporal_dependency_strength": "avg_time_between_events",
        "long_term_dependency_strength": "activity_diversity", 
        "non_linear_dependency_strength": "unique_activities",
        "out_of_order_strength": "out_of_order_percentage",
        "fractal_behavior_strength": "fractal_percentage"
    }
    
    metric_name = param_to_metric.get(param_name)
    if not metric_name or metric_name not in actual_metrics:
        # Fallback: use a simple difference based on expected value
        # This gives us a baseline measure when metrics aren't available
        difference = abs(expected_val - 0.5)  # Distance from middle value
        return difference
    
    actual_metric = actual_metrics[metric_name]
    
    # For out_of_order and fractal, the relationship is more direct
    if param_name in ["out_of_order_strength", "fractal_behavior_strength"]:
        # Expected percentage roughly correlates with strength * 20
        expected_percentage = expected_val * 20  # Rough scaling factor
        difference = abs(actual_metric - expected_percentage) / 100.0
        return difference
    
    # For temporal dependency, higher values should increase avg time between events
    elif param_name == "temporal_dependency_strength":
        # Expected time increases with strength (normalized)
        expected_time = expected_val * 10  # Scale to reasonable time range
        normalized_actual = min(10.0, actual_metric)  # Cap at reasonable max
        difference = abs(normalized_actual - expected_time) / 10.0
        return difference
    
    # For other parameters, normalize to 0-1 range
    else:
        if param_name == "long_term_dependency_strength":
            # Activity diversity should increase with long-term dependency
            expected_diversity = expected_val * 5  # Rough scaling
            normalized_actual = min(5.0, actual_metric)
            difference = abs(normalized_actual - expected_diversity) / 5.0
        elif param_name == "non_linear_dependency_strength":
            # Unique activities should increase with non-linear dependency
            expected_activities = expected_val * 10  # Rough scaling
            normalized_actual = min(10.0, actual_metric)
            difference = abs(normalized_actual - expected_activities) / 10.0
        else:
            # Generic normalization
            normalized_actual = min(1.0, actual_metric / 10.0)
            difference = abs(normalized_actual - expected_val)
        
        return difference


def format_cell_text(value: float) -> str:
    """
    Convert a float into a compact string while preserving useful precision.

    :param value: Float to be formatted.
    :return: "N/A" for *NaN*; otherwise formatted string.
    """
    if np.isnan(value):
        return "N/A"
    abs_val: float = abs(value)
    if abs_val < 1e-3:
        return f"{0.0}"
    if abs_val < 1e-2:
        return f"{value:.4f}"
    if abs_val < 1e-1:
        return f"{value:.3f}"
    return f"{value:.2f}"


# ───────────────────────────────────── Data aggregation ──────────────────────────────────────

def prepare_heatmap_data() -> Tuple[
    np.ndarray,
    List[List[float]],
    List[List[np.ndarray]],
]:
    """
    Read every experiment CSV and build the heatmap data structures.

    :return: (data_matrix, param_devs, mosaic_grids).
    """
    param_devs: List[List[float]] = [[] for _ in range(N_LABELS)]
    pair_devs: List[List[List[float]]] = [
        [[] for _ in range(N_LABELS)] for _ in range(N_LABELS)
    ]

    raw_mosaics: List[List[List[List[List[float]]]]] = [
        [
            [[[] for _ in VALUE_SET] for _ in VALUE_SET]
            for _ in range(N_LABELS)
        ]
        for _ in range(N_LABELS)
    ]

    if not os.path.isdir(OUTPUT_DIR):
        print(f"Directory {OUTPUT_DIR} not found!")
        nan_matrix: np.ndarray = np.full((N_LABELS, N_LABELS), np.nan)
        empty_mosaics: List[List[np.ndarray]] = [
            [np.full((MOSAIC_DETAIL, MOSAIC_DETAIL), np.nan) for _ in range(N_LABELS)]
            for _ in range(N_LABELS)
        ]
        return nan_matrix, param_devs, empty_mosaics

    files_processed = 0
    files_found = 0
    
    print("Processing files:")
    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".csv") or filename == "all_results.csv":
            continue
        
        files_found += 1
        print(f"  Found: {filename}")
            
        parsed = parse_filename(filename)
        if parsed is None:
            print(f"    ❌ Could not parse filename: {filename}")
            continue
            
        p1, val1, p2, val2 = parsed
        print(f"    ✓ Parsed: {p1}={val1}, {p2}={val2}")
        
        if p1 not in TEST_LABELS or p2 not in TEST_LABELS:
            print(f"    ❌ Unknown parameters: {p1}, {p2}")
            continue

        idx1: int = TEST_LABELS.index(p1)
        idx2: int = TEST_LABELS.index(p2)
        if idx1 == idx2:
            print(f"    ❌ Same parameter: {p1}")
            continue

        data = read_experiment_data(os.path.join(OUTPUT_DIR, filename))
        if not data:
            print(f"    ❌ Could not read data from: {filename}")
            continue

        # Calculate parameter effectiveness differences
        diff1 = calculate_parameter_difference(data.get(f"config_{p1}", val1), val1, data, p1)
        diff2 = calculate_parameter_difference(data.get(f"config_{p2}", val2), val2, data, p2)
        
        if np.isnan(diff1) or np.isnan(diff2):
            print(f"    ❌ NaN differences: diff1={diff1}, diff2={diff2}")
            continue

        combined: float = (diff1 + diff2) / 2.0
        print(f"    ✓ Combined difference: {combined:.4f}")

        param_devs[idx1].append(diff1)
        param_devs[idx2].append(diff2)
        pair_devs[idx1][idx2].append(combined)
        pair_devs[idx2][idx1].append(combined)

        try:
            r_id: int = VALUE_INDEX[val1]
            c_id: int = VALUE_INDEX[val2]
            raw_mosaics[idx1][idx2][r_id][c_id].append(combined)
            raw_mosaics[idx2][idx1][c_id][r_id].append(combined)
        except KeyError:
            print(f"    ⚠️ Values not in VALUE_SET: {val1}, {val2}")
            pass
        
        files_processed += 1

    print(f"\nSummary: Found {files_found} CSV files, successfully processed {files_processed} experiment files")

    # Build final data structures
    data_matrix: np.ndarray = np.full((N_LABELS, N_LABELS), np.nan)
    for i in range(N_LABELS):
        if param_devs[i]:
            data_matrix[i, i] = float(np.nanmean(param_devs[i]))
        for j in range(i + 1, N_LABELS):
            if pair_devs[i][j]:
                mean_val: float = float(np.nanmean(pair_devs[i][j]))
                data_matrix[i, j] = data_matrix[j, i] = mean_val

    mosaic_grids: List[List[np.ndarray]] = [
        [np.full((MOSAIC_DETAIL, MOSAIC_DETAIL), np.nan) for _ in range(N_LABELS)]
        for _ in range(N_LABELS)
    ]
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            grid: np.ndarray = np.full((MOSAIC_DETAIL, MOSAIC_DETAIL), np.nan)
            for r in range(MOSAIC_DETAIL):
                for c in range(MOSAIC_DETAIL):
                    vals: List[float] = raw_mosaics[i][j][r][c]
                    if vals:
                        grid[r, c] = float(np.nanmean(vals))
            mosaic_grids[i][j] = grid

    return data_matrix, param_devs, mosaic_grids


def generate_heatmap_image(
    data_matrix: np.ndarray,
    param_devs: List[List[float]],
    mosaics: List[List[np.ndarray]],
) -> None:
    """
    Render the heat-map with mosaics and grid lines for event stream generator.

    :param data_matrix: Diagonal + upper-triangle averages.
    :param param_devs: Per-parameter deviation lists.
    :param mosaics: N × N grid of 5 × 5 deviation matrices.
    :return: None.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    cmap: matplotlib.colors.Colormap = plt.get_cmap(f"{COLOR_MAP_NAME}_r")
    cmap.set_bad(color="lightgray")

    finite: np.ndarray = data_matrix[~np.isnan(data_matrix)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        vmin, vmax = 0.0, 1.0
        normalise = lambda x: 0.5  # type: ignore
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        normalise = lambda x: (x - vmin) / (vmax - vmin)  # type: ignore

    for r in range(N_LABELS):
        for c in range(N_LABELS):
            diag: bool = r == c
            upper: bool = r < c
            val: float = data_matrix[r, c]

            if diag:
                # Diagonal: show parameter deviation range
                devs = param_devs[r]
                text = (
                    f"[{format_cell_text(float(np.nanmin(devs)))},"
                    f"{format_cell_text(float(np.nanmax(devs)))}]"
                    if devs
                    else "[N/A,N/A]"
                )
                color = cmap(normalise(val)) if not np.isnan(val) else "lightgray"
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
                # flip logic: dark tiles -> white text, light tiles -> black text
                tc: str = (
                    "white"
                    if not np.isnan(val) and normalise(val) > TEXT_COLOR_THRESHOLD
                    else "black"
                )
                ax.text(
                    c,
                    r,
                    text,
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_CELL_TEXT - 2,
                    color=tc,
                    weight="bold",
                )
            elif upper:
                # Upper triangle: average pair deviation
                color = cmap(normalise(val)) if not np.isnan(val) else "lightgray"
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
                # flip logic here as well
                tc = (
                    "white"
                    if not np.isnan(val) and normalise(val) > TEXT_COLOR_THRESHOLD
                    else "black"
                )
                ax.text(
                    c,
                    r,
                    format_cell_text(val),
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_CELL_TEXT,
                    color=tc,
                    weight="bold",
                )
            else:
                # Lower triangle: detailed mosaic
                grid = mosaics[r][c]
                ax.imshow(
                    grid,
                    extent=(c - 0.5, c + 0.5, r + 0.5, r - 0.5),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    aspect="auto",
                )

    # (rest of function unchanged...)
    ax.set_xticks(np.arange(N_LABELS))
    ax.set_xticklabels(
        [display_label(lbl) for lbl in TEST_LABELS],
        fontsize=FONT_SIZE_AXIS_LABELS,
        rotation=45,
        ha="right",
        weight="bold"
    )
    ax.set_yticks(np.arange(N_LABELS))
    ax.set_yticklabels(
        [display_label(lbl) for lbl in TEST_LABELS],
        fontsize=FONT_SIZE_AXIS_LABELS,
        weight="bold"
    )

    ax.set_xlim(-0.5, N_LABELS - 0.5)
    ax.set_ylim(N_LABELS - 0.5, -0.5)

    for pos in np.arange(-0.5, N_LABELS, 1.0):
        ax.axhline(pos, color="black", linewidth=3.0)
        ax.axvline(pos, color="black", linewidth=3.0)

    cbar_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=cbar_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.75, pad=0.03)
    cbar.set_label(
        "Average Difference to Target Value",
        fontsize=FONT_SIZE_AXIS_LABELS,
        weight="bold"
    )
    cbar.ax.tick_params(labelsize=FONT_SIZE_CELL_TEXT)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    output_file = os.path.join(OUTPUT_DIR, "parameter_analysis_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to: {output_file}")
    plt.show()



# ─────────────────────────────────────────── Main ────────────────────────────────────────────

def main() -> None:
    """
    Entry-point of the script to generate the event stream generator heatmap.
    """
    print("Generating Event Stream Generator Parameter Analysis Heatmap...")
    print("=" * 60)
    print("=" * 60)
    
    data_matrix, param_devs, mosaics = prepare_heatmap_data()
    
    # Print some statistics
    finite_values = data_matrix[~np.isnan(data_matrix)]
    if len(finite_values) > 0:
        print(f"Parameter effectiveness deviation range: {finite_values.min():.6f} - {finite_values.max():.6f}")
        print(f"Average deviation: {finite_values.mean():.6f}")
    else:
        print("No valid data found - check that experiment results exist in 'experiment_results' directory")
        return
    
    generate_heatmap_image(data_matrix, param_devs, mosaics)


if __name__ == "__main__":
    main()