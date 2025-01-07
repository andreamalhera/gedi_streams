import inspect
import numpy as np
import re

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
        available_features.append(re.sub(r'([a-z])([A-Z])', r'\1_\2', str(feature_type)).lower())
        if feature_name in available_features:
            return feature_type
    raise ValueError(f"ERROR: Invalid value for feature_key argument: {feature_name}. See README.md for " +
                     f"supported feature_names or use a sublist of the following: {FEATURE_TYPES} or None")

class StreamFeature(Feature):
    # Class-level memory to store feature values
    feature_memory = {}

    @classmethod
    def store_feature_value(cls, key, value):
        """Stores a value in feature_memory (class-level)"""
        cls.feature_memory[key] = value

    @classmethod
    def get_feature_value(cls, key):
        """Retrieves a value from feature_memory (class-level)"""
        return cls.feature_memory.get(key, None)  # Default to None if not found


class SimpleStreamStats(StreamFeature):
    def __init__(self, feature_names='simple_stream_stats'):
        super().__init__(feature_names=feature_names)
        self.feature_type = 'simple_stream_stats'
        self.available_class_methods = {
                name: method for name, method in inspect.getmembers(SimpleStreamStats, predicate=inspect.ismethod)
                if name not in ['get_feature_value', 'store_feature_value']
                }
        #self.available_class_methods = dict(inspect.getmembers(SimpleStreamStats,
        #                                                       predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    @classmethod
    def n_events(cls, window: 'EventLog'):
        """Calculate the number of events in the given window and store it"""
        result = sum(len(trace) for trace in window)  # Count events in all traces

        # If there is already a stored value for n_events, aggregate it with the current result
        if 'n_events' in cls.feature_memory:
            result += cls.feature_memory['n_events']

        # Store the result in the class-level memory (feature_memory)
        cls.store_feature_value('n_events', result)

        return result

    @classmethod
    def n_traces(cls, window: EventLog):
        return len(window)

    @classmethod
    def n_windows(cls, window: EventLog):
        return len([window])

    @classmethod
    def ratio_events_per_window(cls, window: EventLog):
        return sum(len(trace) for trace in window)/len([window])

    @classmethod
    def ratio_traces_per_window(cls, window: EventLog):
        return len(window)/len([window])


#TODO stats over multiple windows
