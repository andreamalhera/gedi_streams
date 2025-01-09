import inspect
import numpy as np

from gedi_streams.features.stream_feature import StreamFeature
from gedi_streams.features.memory import ComputedFeatureMemory
from pm4py.objects.log.obj import EventLog
from scipy import stats

class SimpleStreamStats(StreamFeature):
    def __init__(self, feature_names='simple_stream_stats', memory=None):
        self.feature_type='simple_stream_stats'
        self.available_class_methods = dict(inspect.getmembers(SimpleStreamStats, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    @classmethod
    def n_events(self, window: EventLog,  memory: ComputedFeatureMemory):
        previous_value = memory.get_feature_value('n_events')
        n_events = sum(len(trace) for trace in window)
        return n_events + previous_value if previous_value is not None else n_events

    @classmethod
    def n_traces(self, window: EventLog, memory: ComputedFeatureMemory):
        previous_value = memory.get_feature_value('n_traces')
        n_traces = len(window)
        return len(window) + previous_value if previous_value is not None else len(window)

    @classmethod
    def n_windows(self, window: EventLog, memory: ComputedFeatureMemory):
        previous_value = memory.get_feature_value('n_windows')
        return 1+previous_value if previous_value is not None else 1
        n_windows = 1
        return n_windows + previous_value if previous_value is not None else n_windows

    @classmethod
    def ratio_events_per_window(self, window: EventLog, memory: ComputedFeatureMemory):
        previous_n_events = memory.get_feature_value('n_events')
        previous_n_windows = memory.get_feature_value('n_windows')

        new_n_events = sum(len(trace) for trace in window) + previous_n_events if previous_n_events is not None else sum(len(trace) for trace in window)
        new_n_windows = 1 + previous_n_windows if previous_n_windows is not None else 1

        return new_n_events / new_n_windows

    @classmethod
    def ratio_traces_per_window(self, window: EventLog, memory: ComputedFeatureMemory):
        previous_n_traces = memory.get_feature_value('n_traces')
        previous_n_windows = memory.get_feature_value('n_windows')

        new_n_traces = len(window) + previous_n_traces if previous_n_traces is not None else len(window)
        new_n_windows = 1 + previous_n_windows if previous_n_windows is not None else 1

        return new_n_traces / new_n_windows