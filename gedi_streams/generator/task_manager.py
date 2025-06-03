import os
from typing import Tuple

import pandas as pd


class TaskManager:
    """Handles task loading and processing."""

    @staticmethod
    def get_tasks(experiment, output_path="", reference_feature=None) -> Tuple[pd.DataFrame, str]:
        """Loads tasks from various sources."""
        # Read tasks from file
        if isinstance(experiment, str) and experiment.endswith(".csv"):
            tasks = pd.read_csv(experiment, index_col=None)
            output_path = os.path.join(output_path, os.path.split(experiment)[-1].split(".")[0])
            if 'task' in tasks.columns:
                tasks.rename(columns={"task": "log"}, inplace=True)
        # Read tasks from a list in config file
        elif isinstance(experiment, list):
            tasks = pd.DataFrame.from_dict(data=experiment)
        # Read single task from config file
        elif isinstance(experiment, dict):
            tasks = pd.DataFrame.from_dict(data=[experiment])
        else:
            raise FileNotFoundError(f"{experiment} not found. Please check path in filesystem.")

        return tasks, output_path