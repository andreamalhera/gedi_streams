import copy
import multiprocessing
import os
import time
import random
import re
import traceback
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple, Union
from xml.dom import minidom
from xml.sax.handler import all_features

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from datetime import datetime as dt
from functools import partial
from multiprocessing import Process, Queue
from pm4py import generate_process_tree, write_xes
from pm4py.objects.log.obj import EventLog
from pm4py.sim import play_out
from smac import HyperparameterOptimizationFacade, Scenario
from tabulate import tabulate

from config import DEFAULT_CONFIG_SPACE
from gedi_streams.features.feature_extraction import FeatureExtraction, compute_features_from_event_data
from gedi_streams.utils.param_keys import OUTPUT_PATH, INPUT_PATH
from gedi_streams.utils.param_keys.generator import GENERATOR_PARAMS, EXPERIMENT, CONFIG_SPACE, N_TRIALS, \
    SIMULATION_METHOD
from gedi_streams.utils.param_keys.features import FEATURE_PARAMS, FEATURE_SET
from gedi_streams.utils.io_helpers import get_output_key_value_location, dump_features_json, compute_similarity
from gedi_streams.utils.io_helpers import read_csvs
from gedi_streams.utils.column_mappings import column_mappings
from gedi_streams.utils.data_conversions import window_to_eventlog
from gedi_streams.generator.model import create_PTLG
from gedi_streams.generator.simulation import play_DEFact

# Constants
RANDOM_SEED = 10


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


class XesFileFormatter:
    """Handles XES file formatting operations."""

    @staticmethod
    def remove_extra_lines(elem):
        """Removes extra lines from XML elements."""
        has_words = re.compile("\\w")
        for element in elem.iter():
            if not re.search(has_words, str(element.tail)):
                element.tail = ""
            if not re.search(has_words, str(element.text)):
                element.text = ""

    @staticmethod
    def add_extension_before_traces(xes_file):
        """Adds standard extensions to XES files."""
        # Register the namespace
        ET.register_namespace('', "http://www.xes-standard.org/")

        # Parse the original XML
        tree = ET.parse(xes_file)
        root = tree.getroot()

        # Add extensions
        extensions = [
            {'name': 'Lifecycle', 'prefix': 'lifecycle', 'uri': 'http://www.xes-standard.org/lifecycle.xesext'},
            {'name': 'Time', 'prefix': 'time', 'uri': 'http://www.xes-standard.org/time.xesext'},
            {'name': 'Concept', 'prefix': 'concept', 'uri': 'http://www.xes-standard.org/concept.xesext'}
        ]

        for ext in extensions:
            extension_elem = ET.Element('extension', ext)
            root.insert(0, extension_elem)

        # Add global variables
        globals_config = [
            {
                'scope': 'event',
                'attributes': [
                    {'key': 'lifecycle:transition', 'value': 'complete'},
                    {'key': 'concept:name', 'value': '__INVALID__'},
                    {'key': 'time:timestamp', 'value': '1970-01-01T01:00:00.000+01:00'}
                ]
            },
            {
                'scope': 'trace',
                'attributes': [
                    {'key': 'concept:name', 'value': '__INVALID__'}
                ]
            }
        ]

        for global_var in globals_config:
            global_elem = ET.Element('global', {'scope': global_var['scope']})
            for attr in global_var['attributes']:
                string_elem = ET.SubElement(global_elem, 'string', {'key': attr['key'], 'value': attr['value']})
            root.insert(len(extensions), global_elem)

        # Pretty print the XES
        XesFileFormatter.remove_extra_lines(root)
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml()
        with open(xes_file, "w") as f:
            f.write(xml_str)


class SimulationEngine:
    """Handles simulation of process models."""

    @staticmethod
    def simulate_ptlg_model(model, num_traces) -> EventLog:
        """Simulates a model using PTLG approach."""
        log = play_out(model, parameters={"num_traces": num_traces})
        for i, trace in enumerate(log):
            trace.attributes['concept:name'] = str(i)
            for j, event in enumerate(trace):
                event['time:timestamp'] = dt.now()
                event['lifecycle:transition'] = "complete"
        return log

    @staticmethod
    def init_defact(queue: Queue, config: ConfigurationSpace, print_events: bool = True) -> Process:
        """Initializes DEFact simulation process."""
        process = Process(
            target=play_DEFact,
            kwargs={'queue': queue, 'print_events': print_events, "config": config}
        )
        process.start()
        return process

    @staticmethod
    def terminate_defact(process: Process):
        """Terminates a DEFact simulation process."""
        process.terminate()
        process.join()

    @staticmethod
    def simulate_defact_model(config: Configuration, window_size: int) -> EventLog:
        """Simulates a model using DEFact approach."""
        output_queue = multiprocessing.Queue()

        config_space = None
        if hasattr(config, 'config_space'):
            config_space = config.config_space
        elif isinstance(config, ConfigurationSpace):
            config_space = config
        else:
            config_space = ConfigurationSpace(config)

        process = SimulationEngine.init_defact(output_queue, config_space, print_events=True)

        window = []
        try:
            while len(window) < window_size:
                window.append(output_queue.get())
            SimulationEngine.terminate_defact(process)

            log = window_to_eventlog(window)
            return log
        except Exception as e:
            SimulationEngine.terminate_defact(process)
            raise e


class GeneratorTask:
    """Represents a single generation task with objectives and optimization."""

    def __init__(self, task_tuple, config_space, n_trials):
        """
        Initialize a generator task.

        Args:
            task_tuple: Tuple containing task index and series
            config_space: Configuration space for optimization
            n_trials: Number of optimization trials
        """
        random.seed(RANDOM_SEED)

        try:
            self.identifier = [x for x in task_tuple[1] if isinstance(x, str)][0]
        except IndexError:
            self.identifier = task_tuple[0] + 1

        self.task_series = task_tuple[1].loc[lambda x, identifier=self.identifier: x != identifier]

        # Extract objectives from task parameters
        self.objectives = {key: 0.0 for key in self.task_series["feature_params"]["feature_set"]}
        self.config_space = config_space
        self.n_trials = n_trials
        self.configs = None
        self.simulation_method = 'PTLG'  # Default

    def set_simulation_method(self, method: str):
        """Sets the simulation method."""
        self.simulation_method = method

    def optimize(self):
        """Optimizes model parameters to match objectives."""
        objectives = list(self.objectives.keys())

        # Scenario object for multi-objective optimization
        scenario = Scenario(
            self.config_space,
            deterministic=True,
            n_trials=4,
            objectives=objectives,
            n_workers=-1
        )

        # Use SMAC for optimization
        random.seed(RANDOM_SEED)
        multi_obj = HyperparameterOptimizationFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[1] * len(self.objectives),
        )

        random.seed(RANDOM_SEED)
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=self.gen_log,
            multi_objective_algorithm=multi_obj,
            logging_level=False,
            overwrite=True,

        )

        random.seed(RANDOM_SEED)
        incumbent = smac.optimize()
        return incumbent

    def gen_log(self, config: Configuration, seed: int = RANDOM_SEED):
        """Generates a log with the given configuration and evaluates it."""
        random.seed(RANDOM_SEED)
        model = create_PTLG(config)
        log = self.simulate_model(model, config)

        random.seed(RANDOM_SEED)
        result = self.eval_log(log)
        return result

    def eval_log(self, log):
        """Evaluates log against objectives."""
        random.seed(RANDOM_SEED)

        features = compute_features_from_event_data(self.objectives.keys(), log)

        log_evaluation = {}
        for key in self.objectives.keys():
            log_evaluation[key] = abs(self.objectives[key] - features[key])
        return log_evaluation

    def generate_optimized_log(self, config):
        """Returns event log from given configuration with features."""
        model = create_PTLG(config)
        log = self.simulate_model(model, config)

        random.seed(RANDOM_SEED)
        features = compute_features_from_event_data(self.objectives.keys(), log)
        return {
            "configuration": config,
            "log": log,
            "features": features,
        }

    def simulate_model(self, model, config):
        """Simulates the model using specified method."""
        random.seed(RANDOM_SEED)

        if self.simulation_method == 'DEFact':
            window_size = config.get("num_traces", 20) if hasattr(config, "get") else 20
            return SimulationEngine.simulate_defact_model(config, window_size)

        elif self.simulation_method == 'PTLG':
            return SimulationEngine.simulate_ptlg_model(model, config["num_traces"])

        else:
            raise NotImplementedError(f"Play out method {self.simulation_method} not implemented.")


class EventLogGenerator:
    """Main class for generating event logs."""

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the event log generator.

        Args:
            params: Dictionary containing generation parameters
        """
        print("=========================== Generator ==========================")

        if params is None or params.get(GENERATOR_PARAMS) is None:
            default_params = {
                'generator_params': {
                    'experiment': {
                        'ratio_top_20_variants': 0.2,
                        'epa_normalized_sequence_entropy_linear_forgetting': 0.4
                    },
                    'config_space': {
                        'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1],
                        'parallel': [0.01, 1], 'loop': [0.01, 1], 'silent': [0.01, 1],
                        'lt_dependency': [0.01, 1], 'num_traces': [10, 101],
                        'duplicate': [0], 'or': [0]
                    },
                    'n_trials': 50
                }
            }
            raise TypeError(
                f"Missing 'params'. Please provide a dictionary with generator parameters as so: {default_params}. "
                f"See https://github.com/lmu-dbs/gedi for more info."
            )

        self.parse_params(params)
        self.log_config = None

        if hasattr(self, 'output_path') and self.output_path.endswith('csv'):
            self.log_config = pd.read_csv(self.output_path)
            return

        self.run_generation()

    def parse_params(self, params: Dict[str, Any]):
        """Parse input parameters."""
        # Set output path
        self.output_path = params.get(OUTPUT_PATH, 'data/generated')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        # Get generator parameters
        generator_params = params.get(GENERATOR_PARAMS, {})

        # Set simulation method
        self.simulation_method = generator_params.get(SIMULATION_METHOD, 'PTLG')

        # Set configuration space
        config_space_dict = generator_params.get(CONFIG_SPACE)
        if config_space_dict is None:
            self.config_space = ConfigurationManager.create_default_config_space()
            print(f"WARNING: No config_space specified in config file. Continuing with {self.config_space}")
        else:
            config_tuples = ConfigurationManager.convert_list_params_to_tuples(config_space_dict)
            self.config_space = ConfigurationSpace(config_tuples)

        # Set number of trials
        self.n_trials = generator_params.get(N_TRIALS, 20)
        if not generator_params.get(N_TRIALS):
            print(f"INFO: Running with n_trials={self.n_trials}")

        # Set experiment
        self.experiment = generator_params.get(EXPERIMENT)
        self.tasks = None

        if self.experiment is not None:
            self.tasks, output_path = TaskManager.get_tasks(self.experiment, self.output_path)
            columns_to_rename = {
                col: column_mappings()[col]
                for col in self.tasks.columns if col in column_mappings()
            }
            self.tasks = self.tasks.rename(columns=columns_to_rename)
            self.output_path = output_path

    def run_generation(self):
        """Run the event log generation process."""
        random.seed(RANDOM_SEED)

        if self.tasks is not None:
            self.feature_keys = sorted([
                feature for feature in self.tasks.columns.tolist() if feature != "log"
            ])

            log_configs = []

            for index, row in self.tasks.iterrows():
                print(f"INFO: Generating log for task {index + 1}/{len(self.tasks)}...")
                log_config = self._process_task((index, row))
                log_configs.append(log_config)

            self.log_config = log_configs

        else:
            # Generate one optimized log if no tasks are specified
            self.configs = self._optimize_single()

            if not isinstance(self.configs, list):
                self.configs = [self.configs]

            temp = self._generate_single_log(self.configs[0])
            self.log_config = [temp]

            # Handle hotfix for specific experiment parameter
            if self.experiment and self.experiment.get('ratio_unique_traces_per_trace'):
                self.experiment['ratio_variants_per_number_of_traces'] = self.experiment.pop(
                    'ratio_unique_traces_per_trace'
                )

            # Save generated log
            save_path = get_output_key_value_location(
                self.experiment, self.output_path, "genEL"
            ) + ".xes"

            write_xes(temp['log'], save_path)
            XesFileFormatter.add_extension_before_traces(save_path)

    def _process_task(self, task_tuple):
        """Process an individual generation task."""
        task = GeneratorTask(task_tuple, self.config_space, self.n_trials)
        task.set_simulation_method(self.simulation_method)

        random.seed(RANDOM_SEED)
        task.configs = task.optimize()

        random.seed(RANDOM_SEED)
        if isinstance(task.configs, list):
            log_config = task.generate_optimized_log(task.configs[0])
        else:
            log_config = task.generate_optimized_log(task.configs)

        identifier = 'genEL' + str(task.identifier)

        features_to_dump = log_config['features']
        features_to_dump['target_similarity'] = compute_similarity(task.objectives, features_to_dump)

        return log_config

    def _optimize_single(self):
        """Optimize parameters for a single log."""
        # Default objectives if none specified
        objectives = self.experiment or {"ratio_top_20_variants": 0.2}

        # Create a temporary task using default objectives
        task_series = pd.Series({"feature_params": {"feature_set": list(objectives.keys())}})
        task = GeneratorTask((0, task_series), self.config_space, self.n_trials)

        random.seed(RANDOM_SEED)
        return task.optimize()

    def _generate_single_log(self, config):
        """Generate a single log using the given configuration."""
        task_series = pd.Series({"feature_params": {"feature_set": list(self.experiment.keys())}})
        task = GeneratorTask((0, task_series), self.config_space, self.n_trials)
        task.set_simulation_method(self.simulation_method)

        random.seed(RANDOM_SEED)
        return task.generate_optimized_log(config)


class StreamProcessingManager:
    """Manages stream processing operations for DEFact."""

    @staticmethod
    def defact_wrapper(
            n_windows: int,
            input_params: Dict[str, Any],
            window_size: int = 20,
            print_events: bool = True
    ) -> List[Dict[str, float]]:

        """Wrapper for DEFact stream processing."""
        output_queue = Queue()
        config_space = ConfigurationSpace(input_params["config_space"])
        initial_process = SimulationEngine.init_defact(output_queue, config_space, print_events)

        all_features = []
        final_process = initial_process

        try:
            all_features, final_process = StreamProcessingManager.process_windows(
                input_params, n_windows, output_queue, window_size, initial_process, print_events
            )
        except ValueError as e:
            traceback.print_exc()
            print(f"ERROR: {e}")
        except KeyboardInterrupt:
            print("INFO: Received KeyboardInterrupt. Terminating process.")
        finally:
            time.sleep(1)
            if final_process.is_alive():
                SimulationEngine.terminate_defact(final_process)

        StreamProcessingManager.print_results(all_features)

        print("SUCCESS: All windows processed. Total features extracted:", len(all_features), all_features)
        return all_features

    @staticmethod
    def abbreviate_key(key: str) -> str:
        """Abbreviate a feature key for display."""
        parts = key.split('_')
        abbreviated = '_'.join([part[0] for part in parts])
        return abbreviated

    @staticmethod
    def print_results(all_features: List[Dict[str, float]]):
        """Print features in tabular format."""
        if not all_features:
            print("No features to display")
            return

        original_headers = list(all_features[0].keys())
        abbreviated_headers = [StreamProcessingManager.abbreviate_key(key) for key in original_headers]

        rows = []
        for item in all_features:
            rows.append([item[header] for header in original_headers])

        print(tabulate(rows, headers=abbreviated_headers))

    @staticmethod
    def process_windows(
            input_params: Dict[str, Any],
            n_windows: int,
            output_queue: Queue,
            window_size: int,
            process: Process,
            print_events: bool = True
    ) -> Tuple[List[Dict[str, float]], Process]:
        """Process multiple windows of events."""
        window = []
        all_features = []
        current_process = process

        feature_set = input_params.get(FEATURE_PARAMS, {}).get(FEATURE_SET, [])

        for window_num in range(1, n_windows + 1):
            print(f"    INFO: Processing window {window_num}/{n_windows}...")

            while len(window) < window_size:
                window.append(output_queue.get())

            SimulationEngine.terminate_defact(current_process)

            el = window_to_eventlog(window)

            input_params['input_path'] = OUTPUT_PATH

            features_per_window = compute_features_from_event_data(feature_set, el)

            params_for_gen = {
                "generator_params": {
                    "simulation_method": "DEFact",
                    "experiment": input_params,
                    "config_space": input_params["config_space"],
                },
                "n_trials": 50
            }

            generator = EventLogGenerator(params_for_gen)

            log_config = generator.log_config[0]

            config = log_config["configuration"]

            # Only create a new process if we're not at the last window
            if window_num < n_windows:
                current_process = SimulationEngine.init_defact(
                    output_queue, config.config_space, print_events
                )

            all_features.append(features_per_window)

            print(
                f"   SUCCESS: Window {window_num}/{n_windows} processed successfully.",
                f"Extracted {len(features_per_window)} features from stream window"
            )

            window.clear()

        return all_features, current_process