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