from feeed.feature_extractor import extract_features
from gedi_streams.features.feature_extraction import FeatureExtraction
from gedi_streams.generator.simulation import play_DEFact
from gedi_streams.utils.stream_to_eventlog import convert_to_eventlog
from multiprocessing import Process, Queue

import pandas as pd
import pytest
import time
import os

def test_FeatureExtraction():
    INPUT_PARAMS = {'pipeline_step': 'feature_extraction','input_path': 'data/test', 'feature_params': {'feature_set': ['ratio_unique_traces_per_trace', 'ratio_most_common_variant', 'ratio_top_10_variants', 'epa_normalized_variant_entropy', 'epa_normalized_sequence_entropy', 'epa_normalized_sequence_entropy_linear_forgetting', 'epa_normalized_sequence_entropy_exponential_forgetting']}, 'output_path': 'output/plots', 'real_eventlog_path': 'data/BaselineED_feat.csv', 'plot_type': 'boxplot', 'font_size': 24, 'boxplot_width': 10}
    VALIDATION_OUTPUT = {'log': {0: 'gen_el_168', 1: 'gen_el_169'}, 'ratio_most_common_variant': {0: 0.14, 1: 0.26}, 'ratio_top_10_variants': {0: 0.57, 1: 0.68}, 'epa_normalized_variant_entropy': {0: 0.69, 1: 0.65}, 'epa_normalized_sequence_entropy': {0: 0.62, 1: 0.49}, 'epa_normalized_sequence_entropy_linear_forgetting': {0: 0.06, 1: 0.06}, 'epa_normalized_sequence_entropy_exponential_forgetting': {0: 0.26, 1: 0.21}, 'ratio_variants_per_number_of_traces': {0: 0.44, 1: 0.32}}
    features = FeatureExtraction(ft_params=INPUT_PARAMS)
    result = features.feat.round(2).to_dict()
    assert result == VALIDATION_OUTPUT

def test_DEFact_wrapper():
    N_WINDOWS = 10
    WINDOW_SIZE = 20
    INPUT_PARAMS = {'pipeline_step': 'feature_extraction','input_path': 'data/test', 'feature_params': {'feature_set': ['ratio_unique_traces_per_trace', 'ratio_most_common_variant', 'ratio_top_10_variants', 'epa_normalized_variant_entropy', 'epa_normalized_sequence_entropy', 'epa_normalized_sequence_entropy_linear_forgetting', 'epa_normalized_sequence_entropy_exponential_forgetting']}, 'output_path': 'output/plots', 'real_eventlog_path': 'data/BaselineED_feat.csv', 'plot_type': 'boxplot', 'font_size': 24, 'boxplot_width': 10}
    FEATURE_SET = INPUT_PARAMS.get('feature_params').get('feature_set')
    all_features = pd.DataFrame()

    for window_num in range(1, N_WINDOWS + 1):
        OUTPUT_PATH = os.path.join("data", "test", "stream_windows", f"stream_window{WINDOW_SIZE}_{window_num}.xes")
        print(f"Processing window {window_num}/{N_WINDOWS}...")

        window = []
        output_queue = Queue()
        features_queue = Queue()

        # TODO: Move funtionality to main and feature extraction. This should be a test only.
        p1 = Process(target=play_DEFact, kwargs={'queue': output_queue})
        p1.start()

        while len(window) < WINDOW_SIZE:
            window.append(output_queue.get())

        el = convert_to_eventlog(window, output_path=OUTPUT_PATH)
        print(f"   SUCCESS: Generated eventlog from stream {len(window)}", el)

        INPUT_PARAMS['input_path'] = OUTPUT_PATH
        # TODO: Use directly event log instead of writing into memory
        p2 = Process(target=FeatureExtraction, kwargs = {'ft_params': INPUT_PARAMS, 'queue': features_queue})
        p2.start()

        features = features_queue.get()
        print(f"   SUCCESS: Extracted {len(features)} features from stream window:", features.iloc[0].to_dict())
        all_features = pd.concat([all_features, features], ignore_index=True)

        #stop_event.set()  # Signal the  process to stop
        p1.terminate()
        p2.terminate()

        # Wait for both processes to complete
        p1.join()
        p2.join()

        print(f"Window {window_num}/{N_WINDOWS} processed successfully.\n")

    print("All windows processed. Total features extracted:", len(all_features), all_features)
    assert True