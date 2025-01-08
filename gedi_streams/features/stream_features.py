import inspect
import numpy as np
import re

from gedi_streams.features.feature import Feature
from gedi_streams.features.memory import ComputedFeatureMemory
from gedi_streams.utils.io_helpers import list_classes_in_file
from pm4py.objects.log.obj import EventLog
from scipy import stats


def stream_feature_type(feature_name):
    FEATURE_TYPES = list_classes_in_file(__file__)
    print("FEATURE_TYPES in ",__file__, FEATURE_TYPES)
    available_features = []
    for feature_type in FEATURE_TYPES:
        available_features.extend([*eval(feature_type)().available_class_methods])
        available_features.append(str(feature_type))
        available_features.append(re.sub(r'([a-z])([A-Z])', r'\1_\2', str(feature_type)).lower())
        if feature_name in available_features:
            return feature_type
    raise ValueError(f"ERROR: Invalid value for feature_key argument: {feature_name}. See README.md for " +
                     f"supported feature_names or use a sublist of the following: {FEATURE_TYPES} or None")

class StreamFeature(Feature):
    def __init__(self, feature_names='stream_features'):
        self.feature_type='stream_features'
        self.available_class_methods = dict(inspect.getmembers(StreamFeature, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

class SimpleStreamStats(StreamFeature):
    def __init__(self, feature_names='simple_stream_stats', memory=None):
        self.feature_type='simple_stream_stats'
        self.available_class_methods = dict(inspect.getmembers(SimpleStreamStats, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    #NEXTTODO: Add memory into computation through feature.py", line 12, in extract: feature_value = feature_fn(log)
    @classmethod
    def n_events(self, window: EventLog,  memory: ComputedFeatureMemory):
        previous_value = memory.get_feature_value('n_events')
        n_events = sum(len(trace) for trace in window)
        return n_events + previous_value if previous_value is not None else n_events

    @classmethod
    def n_traces(self, window: EventLog, memory: ComputedFeatureMemory):
        return len(window)

    @classmethod
    def n_windows(self, window: EventLog, memory: ComputedFeatureMemory):
        return len([window])

    @classmethod
    def ratio_events_per_window(self, window: EventLog, memory: ComputedFeatureMemory):
        return sum(len(trace) for trace in window)/len([window])

    @classmethod
    def ratio_traces_per_window(self, window: EventLog, memory: ComputedFeatureMemory):
        return len(window)/len([window])



#TODO stats over multiple windows
