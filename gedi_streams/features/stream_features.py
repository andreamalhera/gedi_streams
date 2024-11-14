import inspect
import numpy as np

from .feature import Feature
from scipy import stats

class StreamFeatures(Feature):
    def __init__(self, feature_names='stream'):
        self.feature_type='stream'
        self.available_class_methods = dict(inspect.getmembers(StreamFeatures, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    def window(self, stream):
        return # TODO: attributes_filter.get_attribute_values(log, "concept:name")

    @classmethod
    def n_events_per_window(stream):
        return # TODO