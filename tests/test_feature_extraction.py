from gedi_streams.features.feature_extraction import FeatureExtraction, compute_metafeatures
from pm4py.objects.log.obj import EventLog, Trace, Event
from datetime import datetime as dt

import os
import pytest
import time

def test_FeatureExtraction():
    INPUT_PARAMS = {'pipeline_step': 'feature_extraction','input_path': 'data/test', 'feature_params': {'feature_set': ['ratio_unique_traces_per_trace', 'ratio_most_common_variant', 'ratio_top_10_variants', 'epa_normalized_variant_entropy', 'epa_normalized_sequence_entropy', 'epa_normalized_sequence_entropy_linear_forgetting', 'epa_normalized_sequence_entropy_exponential_forgetting']}, 'output_path': 'output/plots', 'real_eventlog_path': 'data/BaselineED_feat.csv', 'plot_type': 'boxplot', 'font_size': 24, 'boxplot_width': 10}
    VALIDATION_OUTPUT = {'log': {0: 'gen_el_168', 1: 'gen_el_169'}, 'ratio_most_common_variant': {0: 0.14, 1: 0.26}, 'ratio_top_10_variants': {0: 0.57, 1: 0.68}, 'epa_normalized_variant_entropy': {0: 0.69, 1: 0.65}, 'epa_normalized_sequence_entropy': {0: 0.62, 1: 0.49}, 'epa_normalized_sequence_entropy_linear_forgetting': {0: 0.06, 1: 0.06}, 'epa_normalized_sequence_entropy_exponential_forgetting': {0: 0.26, 1: 0.21}, 'ratio_variants_per_number_of_traces': {0: 0.44, 1: 0.32}}
    features = FeatureExtraction(ft_params=INPUT_PARAMS)
    result = features.feat.round(2).to_dict()
    assert result == VALIDATION_OUTPUT

def test_compute_metafeatures():
    def prepare_log():
        # Prepare a mock event log for testing
        log = EventLog()

        # Create a trace and add events to it
        trace = Trace()
        trace.attributes['concept:name'] = 'trace_1'

        event1 = Event()
        event1['concept:name'] = 'event_1'
        event1['time:timestamp'] = dt(2024, 12, 11, 11, 41, 58)
        event1['lifecycle:transition'] = 'start'

        event2 = Event()
        event2['concept:name'] = 'event_2'
        event2['time:timestamp'] = dt(2024, 12, 12, 6, 9, 47)
        event2['lifecycle:transition'] = 'complete'

        # Adds events to the trace
        trace.append(event1)
        trace.append(event2)

        # Add trace to the log
        log.append(trace)

        return log

    FEATURE_SET =['n_traces','trace_len_min','ratio_most_common_variant', 'n_unique_activities', 'n_unique_start_activities',
                  'n_unique_end_activities', 'eventropy_trace', 'epa_variant_entropy']

    result = compute_metafeatures(feature_set=FEATURE_SET, log=prepare_log())

    print(result)
    assert result.keys() == set(FEATURE_SET)
    assert result == {'n_traces': 1, 'trace_len_min': 2, 'ratio_most_common_variant': 1.0, 'n_unique_activities': 2, 'n_unique_start_activities': 1, 'n_unique_end_activities': 1, 'eventropy_trace': -0.0, 'epa_variant_entropy': 0.0}