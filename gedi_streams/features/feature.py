from gedi_streams.features.memory import ComputedFeatureMemory

# TODO: Maybe import from FEEED instead
class Feature:
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def extract(self, log, memory: ComputedFeatureMemory):
        feature_names=self.feature_names

        output = {}
        for feature_name in feature_names:
            feature_fn = self.available_class_methods[feature_name]
            feature_value = feature_fn(log, memory)
            output[f"{feature_name}"] = feature_value

        return output