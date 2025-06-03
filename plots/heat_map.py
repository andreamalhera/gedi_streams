#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ───────────────────────────────────────── Constants ─────────────────────────────────────────

FONT_SIZE: int = 17
FONT_SIZE_AXIS_LABELS: int = FONT_SIZE
FONT_SIZE_CELL_TEXT: int = FONT_SIZE

TEST_LABELS: List[str] = [
    "activity_count_ratio",
    "trace_count_measure",
    "loop_repetition_ratio",
    "long_range_dependency_measure",
    "choice_branching_factor",
    "trace_length_uniformity",
]

VALUE_SET: List[float] = [0.0, 0.1, 0.5, 0.7, 1.0]
VALUE_INDEX: Dict[float, int] = {val: idx for idx, val in enumerate(VALUE_SET)}

# Map internal names → display labels (edit as needed; fallback is the key itself)
DISPLAY_LABELS: Dict[str, str] = {
    "activity_count_ratio": "Activity\nAppearance\nRate",
    "trace_count_measure": "Unique Paths\nRatio",
    "loop_repetition_ratio": "Loop\nRepetition\nRatio",
    "long_range_dependency_measure": "Long Term\nDependency",
    "choice_branching_factor": "Temporal\nDependency",
    "trace_length_uniformity": "Parallel\nActivity\nRatio",
}

OUTPUT_DIR: str = "../output"

N_LABELS: int = len(TEST_LABELS)
MOSAIC_DETAIL: int = len(VALUE_SET)
FIG_SIZE: Tuple[float, float] = (14.0, 11.0)
COLOR_MAP_NAME: str = "viridis"
TEXT_COLOR_THRESHOLD: float = 0.45

FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"stream_features_\["
    r"([a-zA-Z_]+)=([0-9]+\.[0-9]+)__"
    r"([a-zA-Z_]+)=([0-9]+\.[0-9]+)"
    r"\]_.*\.csv"
)

# ────────────────────────────────────────── Helpers ──────────────────────────────────────────

def display_label(name: str) -> str:
    """
    Translate an internal parameter name into the presentation label.

    :param name: Internal identifier as used in the CSV files.
    :return : of objects.
    :return: Label to show on the figure (falls back to *name* if not mapped).
    """
    return DISPLAY_LABELS.get(name, name)


def parse_filename(filename: str) -> Optional[Tuple[str, float, str, float]]:
    """
    Extract the two parameter/value pairs from a CSV filename.

    :param filename: The CSV filename.
    :return : of objects.
    :return: (param₁, value₁, param₂, value₂) or *None* if the pattern does not
        match.
    """
    match: Optional[re.Match[str]] = FILENAME_PATTERN.match(filename)
    if match:
        p1_name, v1_str, p2_name, v2_str = match.groups()
        try:
            return p1_name, float(v1_str), p2_name, float(v2_str)
        except ValueError:
            return None
    return None


def read_csv_data(filepath: str) -> Optional[Dict[str, float]]:
    """
    Read the first row of a CSV file into a *dict* mapping column → float value.

    :param filepath: Path of the CSV file.
    :return : of objects.
    :return: Mapping or *None* if the file is empty/unreadable.
    """
    try:
        df: pd.DataFrame = pd.read_csv(filepath)
        if df.empty:
            return None
        df.columns = df.columns.astype(str)
        result: Dict[str, float] = {}
        for col, val in df.iloc[0].items():
            try:
                result[col] = float(val)
            except (ValueError, TypeError):
                result[col] = float("nan")
        return result
    except (FileNotFoundError, pd.errors.ParserError):
        return None


def format_cell_text(value: float) -> str:
    """
    Convert a float into a compact string while preserving useful precision.

    :param value: Float to be formatted.
    :return : of objects.
    :return: “N/A” for *NaN*; otherwise:
             • scientific notation for |value| < 1 × 10⁻³
             • four decimals for 1 × 10⁻³ ≤ |value| < 1 × 10⁻²
             • three decimals for 1 × 10⁻² ≤ |value| < 1 × 10⁻¹
             • two decimals for anything larger.
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
    Read every CSV in *OUTPUT_DIR* and build three data structures:

    * ``data_matrix``  – 6 × 6 float matrix for the diagonal and upper-triangle.
    * ``param_devs``   – list of per-parameter deviation lists.
    * ``mosaic_grids`` – 6 × 6 matrix of 5 × 5 arrays (lower-triangle mosaics).

    :return : of objects.
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
        nan_matrix: np.ndarray = np.full((N_LABELS, N_LABELS), np.nan)
        empty_mosaics: List[List[np.ndarray]] = [
            [np.full((MOSAIC_DETAIL, MOSAIC_DETAIL), np.nan) for _ in range(N_LABELS)]
            for _ in range(N_LABELS)
        ]
        return nan_matrix, param_devs, empty_mosaics

    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".csv"):
            continue
        parsed = parse_filename(filename)
        if parsed is None:
            continue
        p1, val1, p2, val2 = parsed
        if p1 not in TEST_LABELS or p2 not in TEST_LABELS:
            continue

        idx1: int = TEST_LABELS.index(p1)
        idx2: int = TEST_LABELS.index(p2)
        if idx1 == idx2:
            continue

        data = read_csv_data(os.path.join(OUTPUT_DIR, filename))
        if not data:
            continue

        act1: float = data.get(p1, float("nan"))
        act2: float = data.get(p2, float("nan"))
        if np.isnan(act1) or np.isnan(act2):
            continue

        diff1: float = abs(act1 - val1)
        diff2: float = abs(act2 - val2)
        combined: float = (diff1 + diff2) / 2.0

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
            pass

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
    Render the heat-map with real mosaics and black grid lines.

    :param data_matrix: Diagonal + upper-triangle averages.
    :param param_devs: Per-parameter deviation lists.
    :param mosaics: 6 × 6 grid of 5 × 5 deviation matrices.
    :return : of objects.
    :return: *None*.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    cmap: matplotlib.colors.Colormap = plt.get_cmap(COLOR_MAP_NAME)
    cmap.set_bad(color="lightgray")

    finite: np.ndarray = data_matrix[~np.isnan(data_matrix)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        vmin, vmax = 0.0, 1.0
        normalise = lambda x: 0.5
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        normalise = lambda x: (x - vmin) / (vmax - vmin)

    for r in range(N_LABELS):
        for c in range(N_LABELS):
            diag: bool = r == c
            upper: bool = r < c
            val: float = data_matrix[r, c]

            if diag:
                devs = param_devs[r]
                text = (
                    f"[{format_cell_text(float(np.nanmin(devs)))},"
                    f"{format_cell_text(float(np.nanmax(devs)))}]"
                    if devs
                    else "[N/A,N/A]"
                )
                color: Union[str, Tuple[float, ...]] = (
                    cmap(normalise(val)) if not np.isnan(val) else "lightgray"
                )
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
                tc: str = (
                    "black" if normalise(val) > TEXT_COLOR_THRESHOLD else "white"
                )
                ax.text(
                    c,
                    r,
                    text,
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_CELL_TEXT,
                    color=tc,
                    weight="bold",
                )
            elif upper:
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
                tc = "black" if normalise(val) > TEXT_COLOR_THRESHOLD else "white"
                ax.text(
                    c,
                    r,
                    format_cell_text(val),
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE_CELL_TEXT,
                    color=tc,
                )
            else:
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

    ax.set_xticks(np.arange(N_LABELS))
    ax.set_xticklabels(
        [display_label(lbl) for lbl in TEST_LABELS],
        fontsize=FONT_SIZE_AXIS_LABELS,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(N_LABELS))
    ax.set_yticklabels(
        [display_label(lbl) for lbl in TEST_LABELS],
        fontsize=FONT_SIZE_AXIS_LABELS,
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
    cbar.set_label("Average Difference", fontsize=FONT_SIZE_AXIS_LABELS)
    cbar.ax.tick_params(labelsize=FONT_SIZE_CELL_TEXT)

    # ─── make outer tick‐labels bold ────────────────────────────────────────────
    # get the Text objects for x/y tick labels
    xticks = ax.get_xticklabels()
    yticks = ax.get_yticklabels()

    for tick in zip(xticks, yticks):
        tick[0].set_fontweight("bold")
        tick[1].set_fontweight("bold")


    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.show()


# ─────────────────────────────────────────── Main ────────────────────────────────────────────


def main() -> None:
    """
    Entry-point of the script.

    :return : of objects.
    :return: *None*.
    """
    data_matrix, param_devs, mosaics = prepare_heatmap_data()
    generate_heatmap_image(data_matrix, param_devs, mosaics)


if __name__ == "__main__":
    main()
