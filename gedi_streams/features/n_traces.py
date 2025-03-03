import inspect
from gedi_streams.features.stream_feature import StreamFeature
from gedi_streams.features.memory import ComputedFeatureMemory
from pm4py.objects.log.obj import EventLog

class NTracesPerWindow(StreamFeature):
    def __init__(self, feature_names='n_traces_per_window', memory=None):
        self.feature_type='n_traces_per_window'
        self.available_class_methods = dict(inspect.getmembers(NTracesPerWindow,
                                                               predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    @classmethod
    def n_traces_pw_min(self, window: EventLog, memory: ComputedFeatureMemory):
        previous_value = memory.get_feature_value('n_traces_pw_min')
        if previous_value is not None:
            return min(previous_value, len(window))
        else:
            return len(window)

    @classmethod
    def n_traces_pw_max(self, window: EventLog, memory: ComputedFeatureMemory):
        previous_value = memory.get_feature_value('n_traces_pw_max')
        if previous_value is not None:
            return max(previous_value, len(window))
        else:
            return len(window)

    @classmethod
    def n_traces_pw_mean(self, window: EventLog, memory: ComputedFeatureMemory):
        previous_value = memory.get_feature_value('n_traces_pw_mean')
        if previous_value is not None:
            return (previous_value * memory.get_feature_value('n_windows') + len(window)) / (memory.get_feature_value('n_windows') + 1)
        else:
            return len(window)

    @classmethod
    def n_traces_pw_median(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_mode(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_std(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_variance(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_q1(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_q3(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_iqr(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_geometric_mean(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_geometric_std(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_harmonic_mean(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_skewness(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_kurtosis(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_coefficient_variation(self, window: EventLog, memory: ComputedFeatureMemory):
        pass

    @classmethod
    def n_traces_pw_entropy(self, window: EventLog, memory: ComputedFeatureMemory):
        pass