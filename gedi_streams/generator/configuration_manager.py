from typing import Dict, Any, Union

from ConfigSpace import ConfigurationSpace


class ConfigurationManager:
    """Manages configurations for log generation."""

    @staticmethod
    def create_default_config_space() -> ConfigurationSpace:
        """Creates a default configuration space."""
        return ConfigurationSpace({
            "mode": (5, 40),
            "sequence": (0.01, 1),
            "choice": (0.01, 1),
            "parallel": (0.01, 1),
            "loop": (0.01, 1),
            "silent": (0.01, 1),
            "lt_dependency": (0.01, 1),
            "num_traces": (100, 1001),
            "duplicate": (0),
            "or": (0),
            "concurrent_probability": (0.0, 1.0),  # New parameter for concurrency control
            "activity_duration": (15, 50),  # New parameter for activity duration control
        })

    @staticmethod
    def convert_list_params_to_tuples(config_dict: Dict[str, Any]) -> Dict[str, Union[Any, tuple]]:
        """
        Converts list parameters to tuples for ConfigurationSpace, if applicable.

        :param config_dict: Dictionary of configuration parameters.
        :return: Dictionary with list values converted to tuples when appropriate.
        """
        config_tuples: Dict[str, Union[Any, tuple]] = {}
        for k, v in config_dict.items():
            if isinstance(v, list):
                config_tuples[k] = v[0] if len(v) == 1 else tuple(v)
            else:
                config_tuples[k] = v
        return config_tuples