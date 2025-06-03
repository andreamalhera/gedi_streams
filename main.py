import datetime
import os
from typing import List

import pandas as pd
import pm4py
from pm4py import generate_process_tree

import config
from datetime import datetime as dt

import sys
import inspect
import builtins

from gedi_streams.def_configurations.utils.def_utils import  visualize_markov_chain
from gedi_streams.generator.generator import StreamProcessingManager
from gedi_streams.generator.model import create_PTLG

original_print = builtins.print

def custom_print(*args, **kwargs ):
    caller_frame = inspect.currentframe().f_back
    frame_info = inspect.getframeinfo(caller_frame)
    file_name = frame_info.filename
    line_number = frame_info.lineno

    cwd = os.getcwd()
    if file_name.startswith(cwd):
        relative_path = file_name[len(cwd):]
        if relative_path.startswith(os.sep):
            relative_path = relative_path[1:]
        file_name = relative_path

    prefix = f"[{file_name}:{line_number}]"
    original_print(prefix, *args, **kwargs)

builtins.print = custom_print

stream_features = [
    "activity_appearance_rate",
    "unique_paths_ratio",
]

baseline_features = [
    'temporal_dependency',
    'parallel_activity_ratio',
    'long_term_dependencies',
]

optimized_features = [
    "process_complexity",
    "choice_diversity",
    "loop_intensity",
    "sequence_predictability",
    "concurrent_execution_rate",
    "resource_utilization_pattern",
    "trace_diversity",
    "activity_frequency_balance",
    "trace_length_consistency",
    "activity_count_ratio",
    "case_activity_ratio",
    "long_term_dependency_strength"
]
TEST = [
    "activity_count_ratio",
    "trace_count_measure",
    "loop_repetition_ratio",
    "long_range_dependency_measure",
    "choice_branching_factor",
    "trace_length_uniformity"
    ]
if __name__=='__main__':
    PRINT_EVENTS = True
    N_WINDOWS = 1
    WINDOW_SIZE = 100

    features_computation: List[str] = TEST

    #target_feature_values = {feature_name: 0.8 for feature_name in features_computation}

    # target_feature_values = {
    #     "long_term_dependency_strength": 0.8,
    #     "temporal_dependency": 0.8,
    # }
    target_feature_values = {
        "activity_count_ratio": 0.0,  # Based on your 14 activities
        "trace_count_measure": 0.0,  # Based on your 10 traces
        "loop_repetition_ratio": 0.0,  # Based on your loop patterns
        "long_range_dependency_measure": 0.0,
        "choice_branching_factor": 0.0,
        "trace_length_uniformity": 0.0,
    }

    INPUT_PARAMS: dict = {
        'pipeline_step': 'feature_extraction',
        'input_path': 'data/test',
        'feature_params': {
            'feature_set': features_computation,
        },
        "config_space": config.DEFAULT_CONFIG_SPACE,
        "target_features": target_feature_values,
        "n_trials_adaptive": 100,
        'font_size': 24,
        'boxplot_width': 10
    }

    all_features = StreamProcessingManager.defact_wrapper(N_WINDOWS, INPUT_PARAMS, WINDOW_SIZE, PRINT_EVENTS)

    if all_features and len(all_features) > 0:
        pd.DataFrame(all_features).to_csv(f"{os.getcwd()}/output/stream_features[{datetime.datetime.now()}].csv")