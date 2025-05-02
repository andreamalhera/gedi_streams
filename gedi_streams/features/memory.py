from typing import List

from pm4py.objects.log.obj import EventLog


class SingletonMeta(type):
    """A metaclass for creating Singleton classes."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # If an instance does not exist, create one and store it
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ComputedFeatureMemory(metaclass=SingletonMeta):
    """Singleton memory for computed feature values."""
    def __init__(self):
        self.feature_memory = {}
        self.window_memory: List[EventLog] = list()

    def set_feature_value(self, feature_key, feature_value):
        """
        Stores a computed feature value.

        :param feature_key: Key representing the feature (e.g., 'n_events').
        :param feature_value: The computed value for the feature.
        """
        self.feature_memory[feature_key] = feature_value

    def set_multiple_features(self, feature_dict: dict):
        """
        Stores multiple computed feature values from a dictionary.

        :param feature_dict: Dictionary containing feature keys and their computed values.
        """
        if not isinstance(feature_dict, dict):
            raise ValueError("Input must be a dictionary.")
        self.feature_memory.update(feature_dict)

    def get_feature_value(self, feature_key):
        """
        Retrieves a computed feature value.

        :param feature_key: Key representing the feature.
        :return: The computed value for the feature, or None if not found.
        """
        return self.feature_memory.get(feature_key)

    def get_all_features(self):
        """
        Retrieves all stored feature values.

        :return: Dictionary of all feature keys and their values.
        """
        return self.feature_memory

    def clear_memory(self):
        """Clears all stored feature values."""
        self.feature_memory.clear()

    def add_window(self, window: EventLog):
        """
        Adds a new window to the memory.

        :param window: The new window to be added.
        """
        self.window_memory.append(window)

    def get_latest_window(self):
        """
        Retrieves the latest window from the memory.

        :return: The latest window.
        """
        if self.window_memory:
            return self.window_memory[-1]
        return None

    def get_last_n_windows(self, n: int):
        """
        Retrieves the last n windows from the memory.

        :param n: Number of windows to retrieve.
        :return: List of the last n windows.
        """
        return self.window_memory[-n:] if len(self.window_memory) >= n else self.window_memory