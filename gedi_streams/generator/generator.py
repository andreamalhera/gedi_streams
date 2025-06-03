import copy
import csv
import datetime
import multiprocessing
import os
import time
import traceback
from random import shuffle
from typing import Dict, List, Any, Tuple, Union, Optional
from xml.sax.handler import all_features

import pm4py
from pm4py import generate_process_tree

import pandas as pd
import random
import re
import xml.etree.ElementTree as ET

from ConfigSpace import Configuration, ConfigurationSpace
from datetime import datetime as dt
from functools import partial
from pm4py import write_xes
from pm4py.objects.heuristics_net.obj import HeuristicsNet
from pm4py.objects.log.obj import EventLog, Event
from pm4py.sim import play_out
from smac import HyperparameterOptimizationFacade, Scenario
from tabulate import tabulate

from config import DEFAULT_CONFIG_SPACE, RANDOM_SEED
from gedi_streams.features.feature_extraction import FeatureExtraction, compute_features_from_event_data
from gedi_streams.generator.configuration_manager import ConfigurationManager
from gedi_streams.generator.down_stream_task_evaluation import discovery_algorithm, get_models
from gedi_streams.generator.task_manager import TaskManager
from gedi_streams.generator.xes_file_formatter import XesFileFormatter

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
from multiprocessing import Process, Queue
from xml.dom import minidom

RESULTS_FILE: str = "results.csv"

def save_results(results: Dict[str, float], file_path: str = RESULTS_FILE) -> None:
    """Append a row of metrics to *file_path* in CSV format.

    :param results: A mapping from metric names to their numeric values.
    :param file_path: Destination CSV file path. Defaults to ``results.csv`` in
        the current working directory.
    :return : of objects.
    :return: ``None`` – the function performs I/O but returns nothing.
    """
    file_exists: bool = os.path.isfile(file_path)

    # Attach an execution timestamp so rows remain identifiable.
    timestamp: str = datetime.datetime.utcnow().isoformat(timespec="seconds")
    row: Dict[str, Union[str, float]] = {"timestamp": timestamp, **results}

    if file_exists:
        # Read the existing header to keep column order consistent. Any new
        # keys are appended at the end of the header list.
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            try:
                existing_header: List[str] = next(reader)
            except StopIteration:
                existing_header = []
        header: List[str] = existing_header + [
            k for k in row.keys() if k not in existing_header
        ]
    else:
        header = list(row.keys())

    # Ensure all header fields are present in the row dictionary.
    for column in header:
        row.setdefault(column, "")

    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class EventLogGenerator:
    """Main class for generating event logs."""

    def __init__(self, params: Optional[Dict[str, Any]]):
        """
        Initialize the event log generator.

        Args:
            params: Dictionary containing generation parameters. Can be None.
        """
        print("=========================== Generator ==========================")
        # Set random seed once at initialization
        random.seed(RANDOM_SEED)

        self.output_path: str = 'data/generated'
        self.log_config: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]]
        self.config_space: ConfigurationSpace
        self.n_trials: int = 2
        self.experiment: Optional[Dict[str, float]] = None
        self.tasks: Optional[pd.DataFrame] = None
        self.feature_keys: Optional[List[str]] = None
        self.configs: Optional[List[Dict[str, Any]]] = None

        current_params: Dict[str, Any]

        if params is None:
            print(
                f"Warning: Input 'params' is None. Using minimal default structure for parsing."
            )
            current_params = {GENERATOR_PARAMS: {}}
        else:
            current_params = params

        if current_params.get(GENERATOR_PARAMS) is None:
            default_params_for_print: Dict[str, Any] = {
                GENERATOR_PARAMS: {
                    EXPERIMENT: {
                        'temporal_dependency': 0, 'case_concurrency': 0.8,
                        'concept_stability': 0.8, 'case_throughput_stability': 0.8,
                        'parallel_activity_ratio': 0.8, 'activity_duration_stability': 0.8,
                        'case_priority_dynamics': 0.8, 'concept_drift': 0.8,
                        'long_term_dependencies': 0.8
                    },
                    CONFIG_SPACE: {
                        'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1],
                        'parallel': [0.01, 1], 'loop': [0.01, 1], 'silent': [0.01, 1],
                        'lt_dependency': [0.01, 1], 'num_traces': [10, 101],
                        'duplicate': [0], 'or': [0]
                    },
                    N_TRIALS: 50
                }
            }
            print(
                f"Missing '{GENERATOR_PARAMS}' key in 'params'. "
                f"Please provide a dictionary with generator parameters as so: {default_params_for_print}"
            )

            if GENERATOR_PARAMS not in current_params:
                current_params[GENERATOR_PARAMS] = {}

        self.parse_params(current_params)

        if self.output_path.endswith('csv'):
            try:
                self.log_config = pd.read_csv(self.output_path)
                print(f"INFO: Successfully loaded log configuration from CSV: {self.output_path}")
                return  # Skip generation if log_config is loaded from CSV
            except FileNotFoundError:
                print(f"INFO: Specified CSV log_config '{self.output_path}' not found. Proceeding with generation.")
                self.log_config = None  # Ensure it's None to trigger generation
            except Exception as e:
                print(f"ERROR: Could not read CSV '{self.output_path}': {e}. Proceeding with generation.")
                self.log_config = None

        self.run_generation()

    def parse_params(self, params: Dict[str, Any]) -> None:
        """Parse input parameters and set up generator configuration."""

        self.output_path = str(params.get(OUTPUT_PATH, self.output_path))
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path, exist_ok=True)
            except OSError as e:
                print(f"Error creating output directory {self.output_path}: {e}. Using current directory.")
                self.output_path = "."

        generator_params: Dict[str, Any] = params.get(GENERATOR_PARAMS, {})

        config_space_dict: Optional[Dict[str, List[Union[int, float]]]] = generator_params.get(CONFIG_SPACE)
        if config_space_dict is None:
            self.config_space = ConfigurationManager.create_default_config_space()
            print(f"WARNING: No '{CONFIG_SPACE}' specified. Using default: {self.config_space}")
        else:
            try:
                config_tuples: Dict[str, Tuple[Union[int, float], Union[int, float]]] = \
                    ConfigurationManager.convert_list_params_to_tuples(config_space_dict)
                self.config_space = ConfigurationSpace(config_tuples, seed=RANDOM_SEED)
            except ValueError as e:
                print(f"ERROR: Invalid format for '{CONFIG_SPACE}': {e}. Using default config space.")
                self.config_space = ConfigurationManager.create_default_config_space()

        self.n_trials = int(generator_params.get(N_TRIALS, self.n_trials))
        if N_TRIALS not in generator_params:
            print(f"INFO: Using n_trials={self.n_trials} (default or previously set).")

        self.experiment = generator_params.get(EXPERIMENT)  # type: Optional[Dict[str, float]]

        if self.experiment is not None:
            tasks_df: pd.DataFrame
            updated_output_path: str
            tasks_df, updated_output_path = TaskManager.get_tasks(self.experiment, self.output_path)
            self.tasks = tasks_df

            map_for_columns: Dict[str, str] = column_mappings()
            columns_to_rename_dict: Dict[str, str] = {
                col: map_for_columns[col]
                for col in self.tasks.columns if col in map_for_columns
            }
            if columns_to_rename_dict:
                self.tasks = self.tasks.rename(columns=columns_to_rename_dict)
            self.output_path = updated_output_path
        else:
            self.tasks = None

    def run_generation(self) -> None:
        """Run the event log generation process based on parsed parameters."""
        # No need to re-seed random here

        if self.tasks is not None and not self.tasks.empty:

            task_columns_as_str: List[str] = [str(col) for col in self.tasks.columns.tolist()]
            self.feature_keys = sorted([
                feature_name for feature_name in task_columns_as_str if feature_name != "log"
            ])

            generated_log_configs_list: List[Dict[str, Any]] = []
            task_idx: Any
            task_row_data: pd.Series
            for task_idx, task_row_data in self.tasks.iterrows():
                printable_task_idx: str = str(task_idx + 1) if isinstance(task_idx, int) else str(task_idx)
                print(f"INFO: Generating log for task {printable_task_idx}/{len(self.tasks)}...")
                current_task_tuple: Tuple[int, pd.Series] = (
                    task_idx if isinstance(task_idx, int) else hash(task_idx),
                    task_row_data)

                processed_log_config: Dict[str, Any] = self._process_task(current_task_tuple)
                generated_log_configs_list.append(processed_log_config)

            self.log_config = generated_log_configs_list
        else:
            optimized_single_config_or_list: Union[Dict[str, Any], List[Dict[str, Any]]] = self._optimize_single()

            if not isinstance(optimized_single_config_or_list, list):
                self.configs = [optimized_single_config_or_list]
            else:
                self.configs = optimized_single_config_or_list

            if not self.configs:
                print("ERROR: Optimization did not return any configuration. Cannot generate log.")
                self.log_config = []
                return

            temp_log_item: Dict[str, Any] = self._generate_single_log(self.configs[0])
            self.log_config = [temp_log_item]

            if self.experiment and 'ratio_unique_traces_per_trace' in self.experiment:
                popped_value: float = self.experiment.pop('ratio_unique_traces_per_trace')
                self.experiment['ratio_variants_per_number_of_traces'] = popped_value

            output_file_path_prefix: str = get_output_key_value_location(
                self.experiment, self.output_path, "genEL"
            )
            final_save_path: str = output_file_path_prefix + ".xes"

            actual_log_data_to_write: Any = temp_log_item.get('log')
            if actual_log_data_to_write is not None:
                write_xes(actual_log_data_to_write, final_save_path)
                XesFileFormatter.add_extension_before_traces(final_save_path)
                print(f"INFO: Single generated log saved to {final_save_path}")
            else:
                print(f"WARNING: No log data found in generated item to save for {final_save_path}")

    def _process_task(self, task_tuple: Tuple[int, pd.Series]) -> Dict[str, Any]:
        """Process an individual generation task: optimize and generate log."""

        task_obj: GeneratorTask = GeneratorTask(task_tuple, self.config_space, self.n_trials)

        optimized_task_configs: Union[Dict[str, Any], List[Dict[str, Any]]] = task_obj.optimize()
        task_obj.configs = optimized_task_configs

        log_configuration_result: Dict[str, Any]
        config_to_use: Dict[str, Any]

        if isinstance(optimized_task_configs, list):
            if not optimized_task_configs:
                print(f"Warning: Optimization for task ID {task_obj.identifier} returned an empty list of configs.")
                return {'log': None, 'features': {'error': 'No config found from optimization'}}
            config_to_use = optimized_task_configs[0]
        else:
            config_to_use = optimized_task_configs

        log_configuration_result = task_obj.generate_optimized_log(config_to_use)

        features_dict_to_dump: Dict[str, Any] = log_configuration_result.get('features', {})
        if not isinstance(features_dict_to_dump, dict):
            features_dict_to_dump = {}

        similarity: float = compute_similarity(task_obj.objectives, features_dict_to_dump)
        features_dict_to_dump['target_similarity'] = similarity
        log_configuration_result['features'] = features_dict_to_dump

        return log_configuration_result

    def _optimize_single(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Optimize for a single log based on self.experiment as target features."""
        target_objectives: Dict[str, float]
        if self.experiment:
            target_objectives = self.experiment
        else:
            print(f"WARNING: No experiment objectives provided to _optimize_single. Defaulting to basic objective.")
            target_objectives = {"ratio_top_20_variants": 0.2}

        objectives_series: pd.Series = pd.Series(target_objectives)
        single_task_obj: GeneratorTask = GeneratorTask((0, objectives_series), self.config_space, self.n_trials)

        incumbent_config: Union[Dict[str, Any], List[Dict[str, Any]]] = single_task_obj.optimize()
        return incumbent_config

    def _generate_single_log(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single log using the given optimized configuration."""
        feature_set_keys: List[str] = []
        if self.experiment:
            feature_set_keys = list(self.experiment.keys())
        else:
            print("Warning: self.experiment is None in _generate_single_log. Feature set might be empty.")

        task_data_for_gen: Dict[str, Any] = {
            "feature_params": {"feature_set": feature_set_keys}
        }

        task_series_for_gen: pd.Series = pd.Series(task_data_for_gen, name="single_log_generation_task")

        single_gen_task_obj: GeneratorTask = GeneratorTask(
            (0, task_series_for_gen), self.config_space, self.n_trials
        )

        generated_log_item: Dict[str, Any] = single_gen_task_obj.generate_optimized_log(config)
        return generated_log_item


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

        all_features: list[dict[str, float]] = []
        final_process = initial_process

        try:
            features, final_process = StreamProcessingManager.process_windows(
                input_params, n_windows, output_queue, window_size, initial_process, print_events
            )
            all_features.extend(features)
        except ValueError as e:
            traceback.print_exc()
            print(f"ERROR: {e}")
        except KeyboardInterrupt:
            print("INFO: Received KeyboardInterrupt. Terminating process.")
        finally:
            time.sleep(1)
            if final_process.is_alive():
                SimulationEngine.terminate_defact(final_process)

        # add a sum to each dict
        for feature in all_features:
            feature['sum'] = sum(feature.values())

        StreamProcessingManager.print_results(all_features)

        print("SUCCESS: All windows processed. Total features extracted:")
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

        SimulationEngine.feature_values = input_params["target_features"]

        window: List[Any] = []
        all_features_results: List[Dict[str, float]] = []
        current_process: Process = process

        feature_set_to_calculate: List[str] = input_params.get(FEATURE_PARAMS, {}).get(FEATURE_SET, [])

        target_objectives_for_generator: Dict[str, float] = input_params.get("target_features", {})
        if not target_objectives_for_generator:
            print(
                f"[{StreamProcessingManager.__name__}.process_windows] WARNING: No 'target_features' defined in input_params. Optimization targets may be default or incorrect.")

        adaptive_n_trials: int = input_params.get("n_trials_adaptive", 3)

        # Initialize tracking for cumulative errors and moving averages
        cumulative_errors: Dict[str, float] = {key: 0.0 for key in target_objectives_for_generator.keys()}
        window_count: int = 0
        feature_history: List[Dict[str, float]] = []

        for window_num in range(1, n_windows + 1):
            print(f"    INFO: Processing window {window_num}/{n_windows}...")

            current_window_events: List[Any] = []
            try:
                while len(current_window_events) < window_size:
                    event_data: Any = output_queue.get(timeout=60 * 5)
                    current_window_events.append(event_data)
            except multiprocessing.queues.Empty:
                if not current_window_events:
                    print(f"    Skipping window {window_num} due to no events.")
                    if window_num < n_windows and current_process.is_alive():
                        SimulationEngine.terminate_defact(current_process)
                        print(f"    Attempting to re-initialize DEFact with the original config space for next window.")
                        restart_config_space = input_params.get(CONFIG_SPACE)
                        if not isinstance(restart_config_space, ConfigurationSpace):
                            restart_config_space = ConfigurationSpace(restart_config_space)
                        SimulationEngine.feature_values = {}
                        current_process = SimulationEngine.init_defact(output_queue, restart_config_space, print_events)
                    continue



            if current_process.is_alive():
                SimulationEngine.terminate_defact(current_process)

            # Calculate features for current window
            features_this_window: Dict[str, float] = compute_features_from_event_data(feature_set_to_calculate,
                                                                                      current_window_events)

            # Store features for history
            feature_history.append(features_this_window.copy())
            window_count += 1

            # Calculate current errors (target - actual)
            current_errors: Dict[str, float] = {}
            for key in target_objectives_for_generator.keys():
                if key in features_this_window:
                    error = target_objectives_for_generator[key] - features_this_window[key]
                    current_errors[key] = error
                    cumulative_errors[key] += error

            # Calculate corrective target values for next window optimization
            corrective_targets: Dict[str, float] = {}

            # Feedback control: Use proportional control to correct trajectory
            for key in target_objectives_for_generator.keys():
                if key in features_this_window:
                    target_value = target_objectives_for_generator[key]
                    actual_value = features_this_window[key]
                    error = target_value - actual_value

                    # Adaptive gain based on error magnitude
                    if abs(error) > 0.5:
                        kp = 1.0  # Strong correction for large errors
                    elif abs(error) > 0.2:
                        kp = 0.5  # Medium correction
                    else:
                        kp = 0.2  # Gentle correction for small errors

                    corrective_target = target_value + (kp * error)
                    corrective_target = max(0.0, min(1.0, corrective_target))
                    corrective_targets[key] = corrective_target

            for key in target_objectives_for_generator.keys():
                if key in features_this_window:
                    target_value = target_objectives_for_generator[key]
                    actual_value = features_this_window[key]
                    error = target_value - actual_value

                    # Proportional correction: adjust target for next window
                    # If we're below target, aim higher; if above target, aim lower
                    corrective_target = target_value + (kp * error)

                    # Clamp corrective targets to reasonable bounds (0 to 1 for most features)
                    corrective_target = max(0.0, min(1.0, corrective_target))
                    corrective_targets[key] = corrective_target
                else:
                    # If feature not computed, use original target
                    corrective_targets[key] = target_objectives_for_generator[key]

            # Print diagnostic information
            print(f"    Window {window_num} Feature Analysis:")
            print(f"      Target features: {target_objectives_for_generator}")
            print(f"      Actual features: {features_this_window}")
            print(f"      Current errors: {current_errors}")
            print(f"      Corrective targets for next window: {corrective_targets}")

            # Prepare generator parameters with corrective targets
            generator_call_params: Dict[str, Any] = {
                GENERATOR_PARAMS: {
                    SIMULATION_METHOD: "DEFact",
                    EXPERIMENT: corrective_targets,  # Use corrective targets instead of original targets
                    CONFIG_SPACE: input_params.get(CONFIG_SPACE),
                    N_TRIALS: adaptive_n_trials
                },
                OUTPUT_PATH: input_params.get(OUTPUT_PATH, os.path.join("output", "stream_processing_temp"))
            }

            if not input_params.get(CONFIG_SPACE):
                raise ValueError(f"Missing 'CONFIG_SPACE' in input_params for EventLogGenerator.")


            # Clear any previous memory state to avoid interference
            from gedi_streams.features.memory import ComputedFeatureMemory
            memory = ComputedFeatureMemory()
            memory.clear_memory()

            temp_event_log_generator = EventLogGenerator(generator_call_params)
            if not temp_event_log_generator.log_config or not isinstance(temp_event_log_generator.log_config,
                                                                         list) or not \
                    temp_event_log_generator.log_config[0]:
                raise RuntimeError("EventLogGenerator did not produce a valid log_config.")

            optimized_log_config: Dict[str, Any] = temp_event_log_generator.log_config[0]
            config_for_next_run: Configuration = optimized_log_config["configuration"]

            if not isinstance(config_for_next_run, Configuration):
                raise TypeError(
                    f"Optimized configuration is not a Configuration object. Got {type(config_for_next_run)}")

            print(f"    Optimization successful. Next window will use corrective configuration.")



            # Initialize next window process with optimized configuration (if not the last window)
            if window_num < n_windows:
                current_process = SimulationEngine.init_defact(
                    output_queue,
                    config_for_next_run,
                    print_events
                )

            all_features_results.append(features_this_window)

            print(
                f"   SUCCESS: Window {window_num}/{n_windows} processed successfully. ",
                f"Extracted {len(features_this_window)} features."
            )

            current_window_events.clear()

        # Print final convergence analysis
        print(f"\n=== FINAL CONVERGENCE ANALYSIS ===")
        print(f"Target features: {target_objectives_for_generator}")
        if all_features_results:
            final_features = all_features_results[-1]
            final_errors = {key: abs(target_objectives_for_generator[key] - final_features.get(key, 0))
                            for key in target_objectives_for_generator.keys()}
            print(f"Final features: {final_features}")
            print(f"Final absolute errors: {final_errors}")
            avg_error = sum(final_errors.values()) / len(final_errors)
            print(f"Average final error: {avg_error:.4f}")

        return all_features_results, current_process


class SimulationEngine:
    """Handles simulation of process models."""

    feature_values: dict[str, float] = {}

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
    def init_defact(queue: Queue, config_to_run: ConfigurationSpace,
                    print_events: bool = True) -> Process:

        process = Process(
            target=play_DEFact,
            kwargs={'queue': queue, 'print_events': print_events, "config": config_to_run, "visualize": False}
        )
        process.start()
        return process

    @staticmethod
    def terminate_defact(process: Process):
        """Terminates a DEFact simulation process."""
        print("TERMINATING DEFACT SIMULATION PROCESS", process.pid)
        process.terminate()
        process.join()

    @staticmethod
    def simulate_defact_model(config: Configuration, window_size: int, model: pm4py.ProcessTree) -> List[Event]:
        """Simulates a model using DEFact approach."""
        output_queue = multiprocessing.Queue()

        process = SimulationEngine.init_defact(output_queue, config, print_events=True)

        window = []
        try:
            while len(window) < window_size:
                window.append(output_queue.get())

            SimulationEngine.terminate_defact(process)

            return window
        except Exception as e:
            SimulationEngine.terminate_defact(process)
            raise e

    def calculate_process_discover_metrics(
            window: List[Event],
            model: Union[HeuristicsNet, pm4py.ProcessTree],
            features: dict[str, float] = None
    ) -> None:
        """Calculates process discovery metrics for the given log and model."""
        shuffle(window)
        for event in window:
            if event.get("lifecycle:transition") == "start" and "process" not in event.get("lifecycle:transition"):
                discovery_algorithm(event)

        features_config = SimulationEngine.feature_values
        models: Dict[str, HeuristicsNet] = get_models()

        hmlcb_net: HeuristicsNet = models["HeuristicsMinerLossyCountingBudget"]
        hmlc_net: HeuristicsNet = models["HeuristicsMinerLossyCounting"]

        hmlcb_pn = pm4py.convert_to_petri_net(hmlcb_net)
        hmlc_pn = pm4py.convert_to_petri_net(hmlc_net)

        log_eval: EventLog = pm4py.play_out(model, parameters={"num_traces": 100})

        hmlcb_res_fitness: Dict[str, float] = pm4py.fitness_token_based_replay(log_eval, *hmlcb_pn)
        hmlc_res_fitness: Dict[str, float] = pm4py.fitness_token_based_replay(log_eval, *hmlc_pn)
        hmlcb_res_precision: float = pm4py.precision_token_based_replay(log_eval, *hmlcb_pn)
        hmlc_res_precision: float = pm4py.precision_token_based_replay(log_eval, *hmlc_pn)


        conformance_metrics: Dict[str, float] = {
            "fitness_hmlcb": hmlcb_res_fitness.get("log_fitness", -999),
            "fitness_hmlc": hmlc_res_fitness.get("log_fitness", -999),
            "precision_hmlcb": hmlcb_res_precision,
            "precision_hmlc": hmlc_res_precision
        }

        save_results(conformance_metrics, f"down_stream_task[{features_config}].csv")

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
        # Set random seed once at initialization
        random.seed(RANDOM_SEED)

        objectives_series: pd.Series = task_tuple[1]

        self.identifier: List[str] = objectives_series.keys().values

        self.objectives: Dict[str, float] = {}

        for key, value in objectives_series.items():
            if isinstance(value, (int, float)):
                self.objectives[key] = float(value)

        self.config_space = config_space
        self.n_trials = n_trials
        self.configs = None

    def optimize(self) -> Configuration | list[Configuration]:
        """Optimizes model parameters to match objectives."""
        # Set random seed once at the beginning of optimization
        random.seed(RANDOM_SEED)

        objectives = list(self.objectives.keys())

        # Clear memory before optimization to avoid interference
        from gedi_streams.features.memory import ComputedFeatureMemory
        memory = ComputedFeatureMemory()
        memory.clear_memory()

        # Scenario object for multi-objective optimization
        print(
            f"INFO: Starting optimization for task {self.identifier} with objectives: {objectives} and trials: {self.n_trials}")
        scenario = Scenario(
            self.config_space,
            deterministic=True,
            n_trials=self.n_trials,
            objectives=objectives,
            n_workers=-1
        )

        # Use SMAC for optimization
        multi_obj = HyperparameterOptimizationFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[1] * len(self.objectives),
        )

        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=self.gen_log,
            multi_objective_algorithm=multi_obj,
            logging_level=False,
            overwrite=True,
        )

        incumbent: Configuration | list[Configuration] = smac.optimize()
        print("#########################################################")
        print("#########################################################")
        print("#########################################################")
        print(f"INFO: Optimization completed for task {self.identifier}.")
        print(f"INFO: Best configuration found: {incumbent}")
        print("#########################################################")
        print("#########################################################")
        print("#########################################################")
        return incumbent

    def gen_log(self, config: Configuration, seed: int = RANDOM_SEED):
        """Generates a log with the given configuration and evaluates it."""
        # Clear memory for each evaluation to avoid interference
        from gedi_streams.features.memory import ComputedFeatureMemory
        memory = ComputedFeatureMemory()
        memory.clear_memory()

        model = create_PTLG(config)
        log = self.simulate_model(model, config)

        result = self.eval_log(log, model)
        return result

    def eval_log(self, log, model):
        """Evaluates log against objectives."""
        features = compute_features_from_event_data(self.objectives.keys(), log)
        SimulationEngine.calculate_process_discover_metrics(log, model, features)
        log_evaluation = {}
        for key in self.objectives.keys():
            log_evaluation[key] = abs(self.objectives[key] - features[key])
        return log_evaluation

    def generate_optimized_log(self, config):
        """Returns event log from given configuration with features."""
        # Clear memory before generating final log
        from gedi_streams.features.memory import ComputedFeatureMemory
        memory = ComputedFeatureMemory()
        memory.clear_memory()

        model = create_PTLG(config)
        log = self.simulate_model(model, config)



        features = compute_features_from_event_data(self.objectives.keys(), log)
        return {
            "configuration": config,
            "log": log,
            "features": features,
        }

    def simulate_model(self, model, config):
        """Simulates the model using specified method."""
        window_size = 50
        return SimulationEngine.simulate_defact_model(config, window_size, model)