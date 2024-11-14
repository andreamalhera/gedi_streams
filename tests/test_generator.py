import pandas as pd
import pytest
from gedi_streams.run import gedi
from gedi_streams.generator.generator import GenerateEventLogs

def test_GenerateEventLogs():
    INPUT_PARAMS = {'generator_params': {'experiment': {'input_path': 'data/test/grid_feat.csv', 'objectives': ['ratio_top_20_variants', 'epa_normalized_sequence_entropy_linear_forgetting']}, 'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1], 'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 10001], 'duplicate': [0], 'or': [0]}, 'n_trials': 50}}
    VALIDATION_OUTPUT = [0.74, 0.61, 0.74, 1.0]
    genED = GenerateEventLogs(INPUT_PARAMS)
    similarities = [round(experiment['metafeatures']['target_similarity'], 2) for experiment in genED.log_config]
    assert similarities == VALIDATION_OUTPUT