import inspect
import os
import re

from feeed.feature import Feature
from gedi_streams.features.memory import ComputedFeatureMemory
from gedi_streams.utils.io_helpers import list_classes_in_file
from itertools import chain

class StreamFeature(Feature):
    def __init__(self, feature_names='stream_features'):
        self.feature_type='stream_features'
        self.available_class_methods = dict(inspect.getmembers(StreamFeature, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    def extract(self, log, memory: ComputedFeatureMemory):
        feature_names=self.feature_names

        output = {}
        for feature_name in feature_names:
            feature_fn = self.available_class_methods[feature_name]
            feature_value = feature_fn(log, memory)
            output[f"{feature_name}"] = feature_value
        return output

