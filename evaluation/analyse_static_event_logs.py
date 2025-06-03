import inspect
import os
from typing import List, Dict
from xml.sax.handler import all_features

import pandas as pd
from pm4py.objects.log.obj import Event

from gedi_streams.features.memory import ComputedFeatureMemory
from gedi_streams.features.simple_stream_stats import SimpleStreamStats
from gedi_streams.features.stream_feature import StructuredStreamFeature

DATA_REPO_PATH = f"/Users/christianimenkamp/Documents/Data-Repository/Community/"

LOGS: List[str] = ["bpi-c-2012", "bpi-c-2013", "bpi-c-2015", "bpi-c-2017", "bpi-c-2019", "daily_living", "hospital-billing", "Road-Traffic-Fine-Management-Process", "sepsis"]

WINDOW_SIZE: int = 100

stream_features = [
    "activity_appearance_rate",
    # "direct_follows_entropy",
    # "trace_length_variability",
    # "concurrent_activities_ratio",
    # "activity_entropy",
    "unique_paths_ratio",
    # "structured_complexity",
]

baseline_features = [
    'temporal_dependency',
    'parallel_activity_ratio',
    'long_term_dependencies',
]

# optimized_features = [
#     "process_complexity",
#     "choice_diversity",
#     "loop_intensity",
#     "sequence_predictability",
#     "parallel_activity_ratio",
#     "concurrent_execution_rate",
#     "resource_utilization_pattern",
#     "trace_diversity",
#     "activity_frequency_balance",
#     "trace_length_consistency",
#     "activity_count_ratio",
#     "case_activity_ratio",
#     "long_term_dependency_strength"
# ]

features: List[str] = baseline_features + stream_features

for log_name in LOGS:
    path = DATA_REPO_PATH + log_name

    log: pd.DataFrame = pd.read_feather(path + "/data.feather")

    window: List[Event] = []
    log_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    window_results: Dict[str, Dict[str, float]] = {}
    print("Processing log:", log_name)
    window_counter = 0
    for start in range(0, len(log), WINDOW_SIZE):
        end = start + WINDOW_SIZE
        batch = log.iloc[start:end]

        ssf = dict(inspect.getmembers(StructuredStreamFeature,
                                predicate=inspect.ismethod))
        simple_stream = dict(inspect.getmembers(SimpleStreamStats,
                                predicate=inspect.ismethod))

        feature_memory = ComputedFeatureMemory()



        print("Processing window", start, "to", end)

        for index, row in batch.iterrows():
            event = Event()
            event["case:concept:name"] = row["case:concept:name"]
            event["concept:name"] = row["concept:name"]
            event["time:timestamp"] = str(row["time:timestamp"])

            if row.get("lifecycle:transition"):
                event["lifecycle:transition"] = row["lifecycle:transition"]

            window.append(event)

        all_features: Dict[str, float] = {}

        for feature in features:
            float_result = 0
            if feature in ssf:
                float_result: float = ssf[feature](events=window, memory=feature_memory)
            elif feature in simple_stream:
                float_result: float = simple_stream[feature](events=window, memory= feature_memory)

            all_features[feature] = float_result

        window.clear()
        window_results[f"{start} - {end}"] = all_features
        window_counter += 1
        if window_counter == 100:
            break

    log_results[log_name] = window_results

    df_output: pd.DataFrame = pd.DataFrame.from_dict(log_results[log_name], orient='index')
    df_output.index.name = "index"

    output_path: str = "static_analysis" + "/" + log_name + ".csv"
    df_output.to_csv(output_path)