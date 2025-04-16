import os

import pm4py

import config
from datetime import datetime as dt

import sys
import inspect
import builtins

from gedi_streams.generator.generator import StreamProcessingManager

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
    "variant_appearance_rate",
    "drift_indicator",
    "direct_follows_entropy",
    "trace_length_variability",
    "concurrent_activities_ratio",
    "activity_entropy",
    "unique_paths_ratio",
    "structured_complexity",
    "long_term_activity_shift",
    "variant_stability",
    "throughput_trend",
    "cycle_time_variation"
]

advanced_features = [
    'window_entropy_variability',
    'drift_gradualness',
    'variant_evolution_rate',
    'recurrence_factor',
    'temporal_locality',
    'case_overlap_ratio',
    'path_consistency',
    'stream_homogeneity',
    'loop_structure_stability',
    'reachability_preservation',
    'stream_homogeneity'
]

baseline_features = [
    'temporal_dependency',
    'case_concurrency',
    'concept_stability',
    'case_throughput_stability',
    'parallel_activity_ratio',
    'activity_duration_stability',
    'case_priority_dynamics',
    # 'concept_drift',
    'long_term_dependencies',
]



if __name__=='__main__':
    PRINT_EVENTS = True
    N_WINDOWS = 3
    WINDOW_SIZE = 50
    INPUT_PARAMS: dict[str, str | int | dict[str, list[str]]] = {
        'pipeline_step': 'feature_extraction',
        'input_path': 'data/test',
        'feature_params': {
            'feature_set': advanced_features
        },
        "config_space": config.DEFAULT_CONFIG_SPACE,
        'output_path': 'output/plots',
        'real_eventlog_path': 'data/BaselineED_feat.csv',
        'plot_type': 'boxplot',
        'font_size': 24,
        'boxplot_width': 10
    }

    all_features = StreamProcessingManager.defact_wrapper(N_WINDOWS, INPUT_PARAMS, WINDOW_SIZE, PRINT_EVENTS)

