import os

import config
from datetime import datetime as dt

import sys
import inspect
import builtins

from gedi_streams.generator.generator import DEFact_wrapper

original_print = builtins.print

def custom_print(*args, **kwargs):
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


if __name__=='__main__':
    PRINT_EVENTS = True
    N_WINDOWS = 10
    WINDOW_SIZE = 30
    INPUT_PARAMS: dict[str, str | int | dict[str, list[str]]] = {
        'pipeline_step': 'feature_extraction',
        'input_path': 'data/test',
        'feature_params': {'feature_set': [
            'n_traces',
            'ratio_unique_traces_per_trace',
            'ratio_most_common_variant',
            'epa_normalized_variant_entropy',
            'epa_normalized_sequence_entropy_exponential_forgetting',
            'n_events',
            'n_windows'
            ]
        },
        'output_path': 'output/plots',
        'real_eventlog_path': 'data/BaselineED_feat.csv',
        'plot_type': 'boxplot',
        'font_size': 24,
        'boxplot_width': 10
    }

    all_features = DEFact_wrapper(N_WINDOWS, INPUT_PARAMS, WINDOW_SIZE, PRINT_EVENTS)