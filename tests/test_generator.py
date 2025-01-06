import pandas as pd
import pytest
from gedi_streams.run import gedi
from gedi_streams.generator.generator import GenerateEventLogs, DEFact_wrapper

def test_GenerateEventLogs():
    INPUT_PARAMS = {'generator_params': {'experiment': {'input_path': 'data/test/grid_feat.csv', 'objectives': ['ratio_top_20_variants', 'epa_normalized_sequence_entropy_linear_forgetting']}, 'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1], 'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 10001], 'duplicate': [0], 'or': [0]}, 'n_trials': 50}}
    VALIDATION_OUTPUT = [0.74, 0.61, 0.74, 1.0]
    genED = GenerateEventLogs(INPUT_PARAMS)
    similarities = [round(experiment['features']['target_similarity'], 2) for experiment in genED.log_config]
    assert similarities == VALIDATION_OUTPUT

def test_abbr_GenerateEventLogs():
    INPUT_PARAMS = {'generator_params': {'experiment': {'input_path': 'data/test/igedi_table_1.csv', 'objectives': ['rmcv', 'ense']}, 'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1], 'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 10001], 'duplicate': [0], 'or': [0]}, 'n_trials': 2}}
    VALIDATION_OUTPUT = [0.93, 0.66, 0.66]
    genED = GenerateEventLogs(INPUT_PARAMS)
    similarities = [round(experiment['features']['target_similarity'], 2) for experiment in genED.log_config]
    assert similarities == VALIDATION_OUTPUT

def test_DEFact_wrapper():
    N_WINDOWS = 10
    WINDOW_SIZE = 30
    INPUT_PARAMS = {'pipeline_step': 'feature_extraction','input_path': 'data/test',
                    'feature_params': {'feature_set':
                                       ['n_traces',
                                        'ratio_unique_traces_per_trace',
                                        'ratio_most_common_variant',
                                        'ratio_top_10_variants',
                                        'epa_normalized_variant_entropy',
                                        'epa_normalized_sequence_entropy',
                                        'epa_normalized_sequence_entropy_linear_forgetting',
                                        'epa_normalized_sequence_entropy_exponential_forgetting',
                                        'n_events']},
                    'output_path': 'output/plots', 'real_eventlog_path': 'data/BaselineED_feat.csv', 'plot_type': 'boxplot', 'font_size': 24, 'boxplot_width': 10}

    all_features = DEFact_wrapper(N_WINDOWS, INPUT_PARAMS, WINDOW_SIZE)
    assert len(all_features) == N_WINDOWS
