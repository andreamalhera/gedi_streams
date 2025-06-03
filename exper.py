from itertools import combinations, product
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import os
from datetime import datetime as dt

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

VALUE_SET: List[float] = [0.0, 0.1, 0.5, 0.7, 1.0]

INPUT_PARAMS: Dict[str, Any] = {
    "pipeline_step": "feature_extraction",
    "input_path": "data/test",
    "feature_params": {"feature_set": TEST},
    "config_space": config.DEFAULT_CONFIG_SPACE,
    "target_features": {},  # filled during runtime
    "n_trials_adaptive": 15,
    "font_size": 24,
    "boxplot_width": 10,
}
# ————————————————————————————————————————————————


def generate_target_configs(
    feature_list: List[str],
    values: List[float],
) -> List[Tuple[str, str, Dict[str, float]]]:
    """
    Create all unordered pairs of features with every ordered pair of values.

    :param feature_list: Features to be combined.
    :param values: Values assigned to every feature in a pair.
    :return : of objects.
    :return: Tuples of (feature_a, feature_b, {feature_a: value_a, feature_b: value_b}).
    """
    configs: List[Tuple[str, str, Dict[str, float]]] = []
    for feature_a, feature_b in combinations(feature_list, 2):
        for value_a, value_b in product(values, repeat=2):
            configs.append(
                (
                    feature_a,
                    feature_b,
                    {feature_a: value_a, feature_b: value_b},
                )
            )
    return configs


def run_experiments(
    n_windows: int,
    window_size: int,
    base_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Execute an experiment for every target-feature configuration, save individual
    CSVs, and concatenate all outcomes.

    :param n_windows: Number of sliding windows.
    :param window_size: Size of a single window.
    :param base_params: Template dictionary passed to defact_wrapper.
    :return : of objects.
    :return: DataFrame with results from all experiments.
    """
    results: List[Dict[str, Any]] = []
    target_configs = generate_target_configs(TEST, VALUE_SET)

    for feature_a, feature_b, targets in target_configs:
        params: Dict[str, Any] = deepcopy(base_params)
        params["target_features"] = targets

        experiment_output = StreamProcessingManager.defact_wrapper(
            n_windows,
            params,
            window_size,
            PRINT_EVENTS,
        )

        if experiment_output:
            df_single: pd.DataFrame = pd.DataFrame(experiment_output, copy=False)
            timestamp: str = dt.now().strftime("%Y%m%d_%H%M%S")
            filename: str = (
                f"stream_features_[{feature_a}={targets[feature_a]}__"
                f"{feature_b}={targets[feature_b]}]_{timestamp}.csv"
            )
            output_path: str = os.path.join(os.getcwd(), "output", filename)
            df_single.to_csv(output_path, index=False)

            for record in experiment_output:
                record.update(
                    {
                        "target_feature_1": feature_a,
                        "target_value_1": targets[feature_a],
                        "target_feature_2": feature_b,
                        "target_value_2": targets[feature_b],
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
