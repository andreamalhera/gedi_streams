"""
Stream-feature experiment runner (single-feature targets).

Six experiments are executed—one per feature in ``TEST``—with the selected
feature fixed to 1.0 and all other features left unset.
"""

from copy import deepcopy
from datetime import datetime as dt
from typing import Any, Dict, List, Tuple

import os
import pandas as pd

import config
from gedi_streams.generator.generator import StreamProcessingManager

# ————————————————————————————————————————————————
# constants copied from main
PRINT_EVENTS: bool = False
N_WINDOWS: int = 1
WINDOW_SIZE: int = 25

TEST: List[str] = [
    "activity_count_ratio",
    "trace_count_measure",
    "loop_repetition_ratio",
    "long_range_dependency_measure",
    "choice_branching_factor",
    "trace_length_uniformity",
]

INPUT_PARAMS: Dict[str, Any] = {
    "pipeline_step": "feature_extraction",
    "input_path": "data/test",
    "feature_params": {"feature_set": TEST},
    "config_space": config.DEFAULT_CONFIG_SPACE,
    "target_features": {},  # filled during runtime
    "n_trials_adaptive": 10,
    "font_size": 24,
    "boxplot_width": 10,
}
# ————————————————————————————————————————————————


def generate_target_configs(
    feature_list: List[str],
    value: float = 0.0,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Create one configuration per feature with the chosen feature fixed to
    ``value`` and all other features left undefined.

    :param feature_list: Features to iterate over.
    :param value: Numerical value assigned to the active feature.
    :return : of objects.
    :return: List of ``(feature_name, {feature_name: value})`` tuples.
    """
    return [(feature, {feature: value}) for feature in feature_list]


def run_experiments(
    n_windows: int,
    window_size: int,
    base_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Execute a separate experiment for every target-feature configuration,
    persist individual CSV files, and concatenate all outcomes.

    :param n_windows: Number of sliding windows.
    :param window_size: Size of each window.
    :param base_params: Template dictionary forwarded to ``defact_wrapper``.
    :return : of objects.
    :return: Combined DataFrame with results from all experiments.
    """
    results: List[Dict[str, Any]] = []
    target_configs: List[Tuple[str, Dict[str, float]]] = generate_target_configs(
        TEST,
        value=1.0,
    )

    for feature, targets in target_configs:
        params: Dict[str, Any] = deepcopy(base_params)
        params["target_features"] = targets

        experiment_output: List[Dict[str, Any]] = StreamProcessingManager.defact_wrapper(
            n_windows,
            params,
            window_size,
            PRINT_EVENTS,
        )

        if experiment_output:
            df_single: pd.DataFrame = pd.DataFrame(experiment_output, copy=False)
            timestamp: str = dt.now().strftime("%Y%m%d_%H%M%S")
            filename: str = f"stream_features_[{feature}=1.0]_{timestamp}.csv"
            output_path: str = os.path.join(os.getcwd(), "output", filename)
            df_single.to_csv(output_path, index=False)

            for record in experiment_output:
                record.update(
                    {
                        "target_feature": feature,
                        "target_value": 1.0,
                    }
                )
            results.extend(experiment_output)

    return pd.DataFrame(results, copy=False)


if __name__ == "__main__":
    experiment_df: pd.DataFrame = run_experiments(
        N_WINDOWS,
        WINDOW_SIZE,
        INPUT_PARAMS,
    )
