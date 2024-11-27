from gedi_streams.features.feature_extraction import FeatureExtraction, DEF_wrapper
from multiprocessing import Process, Queue

import pandas as pd
import pytest
import time

def test_FeatureExtraction():
    INPUT_PARAMS = {'pipeline_step': 'feature_extraction','input_path': 'data/test', 'feature_params': {'feature_set': ['ratio_unique_traces_per_trace', 'ratio_most_common_variant', 'ratio_top_10_variants', 'epa_normalized_variant_entropy', 'epa_normalized_sequence_entropy', 'epa_normalized_sequence_entropy_linear_forgetting', 'epa_normalized_sequence_entropy_exponential_forgetting']}, 'output_path': 'output/plots', 'real_eventlog_path': 'data/BaselineED_feat.csv', 'plot_type': 'boxplot', 'font_size': 24, 'boxplot_width': 10}
    VALIDATION_OUTPUT = {'log': {0: 'gen_el_168', 1: 'gen_el_169'}, 'ratio_most_common_variant': {0: 0.14, 1: 0.26}, 'ratio_top_10_variants': {0: 0.57, 1: 0.68}, 'epa_normalized_variant_entropy': {0: 0.69, 1: 0.65}, 'epa_normalized_sequence_entropy': {0: 0.62, 1: 0.49}, 'epa_normalized_sequence_entropy_linear_forgetting': {0: 0.06, 1: 0.06}, 'epa_normalized_sequence_entropy_exponential_forgetting': {0: 0.26, 1: 0.21}, 'ratio_variants_per_number_of_traces': {0: 0.44, 1: 0.32}}
    features = FeatureExtraction(ft_params=INPUT_PARAMS)
    result = features.feat.round(2).to_dict()
    assert result == VALIDATION_OUTPUT

def test_DEF_wrapper():
    WINDOW_SIZE = 10
    # Start the two processes
    INPUT_PARAMS = {'pipeline_step': 'feature_extraction','input_path': 'data/test', 'feature_params': {'feature_set': ['ratio_unique_traces_per_trace', 'ratio_most_common_variant', 'ratio_top_10_variants', 'epa_normalized_variant_entropy', 'epa_normalized_sequence_entropy', 'epa_normalized_sequence_entropy_linear_forgetting', 'epa_normalized_sequence_entropy_exponential_forgetting']}, 'output_path': 'output/plots', 'real_eventlog_path': 'data/BaselineED_feat.csv', 'plot_type': 'boxplot', 'font_size': 24, 'boxplot_width': 10}
    output_queue = Queue()

    p1 = Process(target=DEF_wrapper, args=(output_queue,))
    p2 = Process(target=FeatureExtraction, kwargs={'ft_params':INPUT_PARAMS,})

    p1.start()
    p2.start()

    for i in range(WINDOW_SIZE):
        print("CAUGHT: ", i, output_queue.get())

    # Simulate some condition in the main process
    time.sleep(WINDOW_SIZE)  # Let both processes run for a while
    print("Condition met. Stopping the second process.")


    #stop_event.set()  # Signal the second process to stop
    p1.terminate()

    # Wait for both processes to complete
    p1.join()
    p2.join()

    print("All processes complete.")
    assert True