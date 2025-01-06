import inspect
import numpy as np

from gedi_streams.features.feature import Feature
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
        if feature_name in available_features:
            return feature_type
    raise ValueError(f"ERROR: Invalid value for feature_key argument: {feature_name}. See README.md for " +
                     f"supported feature_names or use a sublist of the following: {FEATURE_TYPES} or None")

class StreamFeatures(Feature):
    def __init__(self, feature_names='stream_features'):
        self.feature_type='stream_features'
        self.available_class_methods = dict(inspect.getmembers(StreamFeatures, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    @classmethod
    def n_events(self, window: EventLog):
        return sum(len(trace) for trace in window)

    @classmethod
    def n_traces(self, window: EventLog):
        return len(window)

    @classmethod
    def n_windows(self, window: EventLog):
        return len([window])

    @classmethod
    def ratio_events_per_window(self, window: EventLog):
        return sum(len(trace) for trace in window)/len([window])

    @classmethod
    def ratio_traces_per_window(self, window: EventLog):
        return len(window)/len([window])



#TODO stats over multiple windows
