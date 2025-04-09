import copy
import multiprocessing
import os
from typing import Dict, List, Any
from xml.sax.handler import all_features
from pm4py import generate_process_tree

import pandas as pd
import random
import re
import xml.etree.ElementTree as ET

from ConfigSpace import Configuration, ConfigurationSpace
from datetime import datetime as dt
from functools import partial
from pm4py import write_xes
from pm4py.objects.log.obj import EventLog
from pm4py.sim import play_out
from smac import HyperparameterOptimizationFacade, Scenario

from config import DEFAULT_CONFIG_SPACE
from gedi_streams.features.feature_extraction import FeatureExtraction, compute_features_from_event_data
from gedi_streams.utils.param_keys import OUTPUT_PATH, INPUT_PATH
from gedi_streams.utils.param_keys.generator import GENERATOR_PARAMS, EXPERIMENT, CONFIG_SPACE, N_TRIALS, SIMULATION_METHOD
from gedi_streams.utils.param_keys.features import FEATURE_PARAMS, FEATURE_SET
from gedi_streams.utils.io_helpers import get_output_key_value_location, dump_features_json, compute_similarity
from gedi_streams.utils.io_helpers import read_csvs
from gedi_streams.utils.column_mappings import column_mappings
from gedi_streams.utils.data_conversions import window_to_eventlog
from gedi_streams.generator.model import create_PTLG
from gedi_streams.generator.simulation import play_DEFact
from multiprocessing import Process, Queue
from xml.dom import minidom

RANDOM_SEED = 10
random.seed(RANDOM_SEED)

def get_tasks(experiment, output_path="", reference_feature=None):
    #Read tasks from file.
    if isinstance(experiment, str) and experiment.endswith(".csv"):
        tasks = pd.read_csv(experiment, index_col=None)
        output_path=os.path.join(output_path,os.path.split(experiment)[-1].split(".")[0])
        if 'task' in tasks.columns:
            tasks.rename(columns={"task":"log"}, inplace=True)
    # elif isinstance(experiment, str) and os.path.isdir(os.path.join(os.getcwd(), experiment)):
    #     tasks = read_csvs(experiment, reference_feature)
    # #Read tasks from a real log features selection.
    # elif isinstance(experiment, dict) and INPUT_PATH in experiment.keys():
    #     output_path=os.path.join(output_path,os.path.split(experiment.get(INPUT_PATH))[-1].split(".")[0])
    #     tasks = pd.read_csv(experiment.get(INPUT_PATH), index_col=None)
    #     id_col = tasks.select_dtypes(include=['object']).dropna(axis=1).columns[0]
    #     if "objectives" in experiment.keys():
    #         incl_cols = experiment["objectives"]
    #         tasks = tasks[(incl_cols +  [id_col])]
    # TODO: Solve/Catch error for different objective keys.
    #Read tasks from config_file with list of targets
    elif isinstance(experiment, list):
        tasks = pd.DataFrame.from_dict(data=experiment)
    #Read single tasks from config_file
    elif isinstance(experiment, dict):
        tasks = pd.DataFrame.from_dict(data=[experiment])
    else:
        raise FileNotFoundError(f"{experiment} not found. Please check path in filesystem.")
    return tasks, output_path

def removeextralines(elem):
    hasWords = re.compile("\\w")
    for element in elem.iter():
        if not re.search(hasWords,str(element.tail)):
            element.tail=""
        if not re.search(hasWords,str(element.text)):
            element.text = ""

def add_extension_before_traces(xes_file):
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
    globals = [
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

    for global_var in globals:
        global_elem = ET.Element('global', {'scope': global_var['scope']})
        for attr in global_var['attributes']:
            string_elem = ET.SubElement(global_elem, 'string', {'key': attr['key'], 'value': attr['value']})
        root.insert(len(extensions), global_elem)


    # Pretty print the Xes
    removeextralines(root)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml()
    with open(xes_file, "w") as f:
        f.write(xml_str)

class GenerateEventLogs:
    # TODO: Clarify nomenclature: experiment, task, objective as in notebook (https://github.com/lmu-dbs/gedi/blob/main/notebooks/grid_objectives.ipynb)
    def __init__(self, params):
        print("=========================== Generator ==========================")
        if params is None or params.get(GENERATOR_PARAMS) is None:
            default_params = {'generator_params': {'experiment': {'ratio_top_20_variants': 0.2, 'epa_normalized_sequence_entropy_linear_forgetting': 0.4}, 'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1], 'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 101], 'duplicate': [0], 'or': [0]}, 'n_trials': 50}}
            raise TypeError(f"Missing 'params'. Please provide a dictionary with generator parameters as so: {default_params}. See https://github.com/lmu-dbs/gedi for more info.")
        start = dt.now()
        self._parse_params(params)

        if self.output_path.endswith('csv'):
            self.log_config = pd.read_csv(self.output_path)
            return

        if self.tasks is not None:
            self.feature_keys = sorted([feature for feature in self.tasks.columns.tolist() if feature != "log"])

            log_configs: List[Dict[str, EventLog]] = []

            for index, row in self.tasks.iterrows():
                print(f"INFO: Generating log for task {index+1}/{len(self.tasks)}...")
                log_config = self.generator_wrapper((index, row), self.simulation_method)
                log_configs.append(log_config)

            self.log_config = log_configs

        else:
            random.seed(RANDOM_SEED)
            self.configs = self.optimize()
            if type(self.configs) is not list:
                self.configs = [self.configs]
            temp = self.generate_optimized_log(self.configs[0])
            self.log_config = [temp]
            #TODO: Replace hotfix
            if self.experiment.get('ratio_unique_traces_per_trace'):#HOTFIX
                self.experiment['ratio_variants_per_number_of_traces']=self.experiment.pop('ratio_unique_traces_per_trace')

            save_path = get_output_key_value_location(self.experiment,
                                             self.output_path, "genEL")+".xes"
            write_xes(temp['log'], save_path)
            add_extension_before_traces(save_path)

    def _parse_params(self, params):
        if params.get(OUTPUT_PATH) is None:
            self.output_path = 'data/generated'
        else:
            self.output_path = params.get(OUTPUT_PATH)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        if params.get(GENERATOR_PARAMS).get(SIMULATION_METHOD) is None:
            self.simulation_method = 'PTLG'
        else:
            self.simulation_method = params.get(GENERATOR_PARAMS).get(SIMULATION_METHOD)

        self.config_space = params.get(GENERATOR_PARAMS).get(CONFIG_SPACE)
        self.n_trials = params.get(GENERATOR_PARAMS).get(N_TRIALS)
        self.experiment = params.get(GENERATOR_PARAMS).get(EXPERIMENT)
        if self.experiment is not None:
            self.tasks, output_path = get_tasks(self.experiment, self.output_path)
            columns_to_rename = {col: column_mappings()[col] for col in self.tasks.columns if col in column_mappings()}
            self.tasks = self.tasks.rename(columns=columns_to_rename)
            self.output_path = output_path

        if self.config_space is None:
            self.config_space = ConfigurationSpace({
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
            print(f"WARNING: No config_space specified in config file. Continuing with {self.config_space}")
        else:
            configspace_lists = self.config_space
            configspace_tuples = {}
            for k, v in configspace_lists.items():
                if len(v) == 1:
                    configspace_tuples[k] = v[0]
                else:
                    configspace_tuples[k] = tuple(v)
            self.config_space = ConfigurationSpace(configspace_tuples)

        if self.n_trials is None:
            self.n_trials = 20
            print(f"INFO: Running with n_trials={self.n_trials}")
        else:
            self.n_trials = self.n_trials
        return

    def generator_wrapper(self, task_tuple, SIMULATION_METHOD='OTHER'):
        task = self.GeneratorTask(task_tuple, self)
        random.seed(RANDOM_SEED)
        task.configs = task.optimize()

        random.seed(RANDOM_SEED)
        if isinstance(task.configs, list):
            log_config = task.generate_optimized_log(task.configs[0])
        else:
            log_config = task.generate_optimized_log(task.configs)

        identifier = 'genEL'+str(task.identifier)

        features_to_dump = log_config['features']

        features_to_dump['target_similarity'] = compute_similarity(task.objectives, features_to_dump)

        return log_config

    class GeneratorTask:
        def __init__(self, task_tuple, generator):
            try:
                self.identifier = [x for x in task_tuple[1] if isinstance(x, str)][0]
            except IndexError:
                self.identifier = task_tuple[0]+1
            self.task_series = task_tuple[1].loc[lambda x, identifier=self.identifier: x!=identifier]

            # self.objectives = self.task_series.dropna().to_dict()
            self.objectives = {key: 0.0 for key in self.task_series["feature_params"]["feature_set"]}
            self.config_space = generator.config_space
            self.n_trials = generator.n_trials
            self.generator = generator
            self.configs = {}

        def optimize(self):
            objectives = [*self.objectives.keys()]

            # Scenario object specifying the multi-objective optimization environment
            scenario = Scenario(
                self.config_space,
                deterministic=True,
                n_trials=self.n_trials,
                objectives=objectives,
                n_workers=-1
            )

            # Use SMAC to find the best configuration/hyperparameters
            random.seed(RANDOM_SEED)
            multi_obj = HyperparameterOptimizationFacade.get_multi_objective_algorithm(
                    scenario,
                    objective_weights=[1]*len(self.objectives),
                )


            random.seed(RANDOM_SEED)
            smac = HyperparameterOptimizationFacade(
                scenario=scenario,
                target_function=self.gen_log,
                multi_objective_algorithm=multi_obj,
                # logging_level=False,
                overwrite=True,
            )

            random.seed(RANDOM_SEED)
            incumbent = smac.optimize()
            return incumbent

        def gen_log(self, config: Configuration, seed: int = 0):
            random.seed(RANDOM_SEED)
            model = create_PTLG(config)
            log = self.simulate_Model(model, config)
            random.seed(RANDOM_SEED)
            result = self.eval_log(log)
            return result

        def eval_log(self, log):
            random.seed(RANDOM_SEED)

            features = compute_features_from_event_data(self.objectives.keys(), log)

            log_evaluation = {}
            for key in self.objectives.keys():
                log_evaluation[key] = abs(self.objectives[key] - features[key])
            return log_evaluation

        def generate_optimized_log(self, config):
            ''' Returns event log from given configuration'''
            model = create_PTLG(config)
            log = self.simulate_Model(model, config)

            random.seed(RANDOM_SEED)
            features = compute_features_from_event_data( self.objectives.keys(), log)
            return {
                "configuration": config,
                "log": log,
                "features": features,
            }

        def simulate_Model(self, model, config):
            random.seed(RANDOM_SEED)
            if self.generator.simulation_method == 'DEFact':
                log = play_DEFact(model, config)
            elif self.generator.simulation_method == 'PTLG':
                log = play_out(model, parameters={"num_traces": config["num_traces"]})
                for i, trace in enumerate(log):
                    trace.attributes['concept:name'] = str(i)
                    for j, event in enumerate(trace):
                        event['time:timestamp'] = dt.now()
                        event['lifecycle:transition'] = "complete"
            else:
               raise NotImplementedError(f"Play out method {self.generator.simulation_method} not implemented.")
            return log


def init_DEFact(queue: Queue, config: ConfigurationSpace, print_events: bool=True):
    process = Process(target=play_DEFact, kwargs={'queue': queue, 'print_events': print_events, "config": config})
    process.start()
    return process


def terminate_DEFact(process: Process):
    process.terminate()
    process.join()

def DEFact_wrapper(
        n_windows: int,
        input_params: dict[str, str | int | dict[str, list[str]]],
        window_size: int = 20,
        print_events: bool=True
):
    output_queue = Queue()
    config_space: ConfigurationSpace = ConfigurationSpace(input_params["config_space"])
    process: Process = init_DEFact(output_queue, config_space, print_events=print_events)

    all_features = []
    try:
        all_features = main_gedi_event_loop(input_params, n_windows, output_queue, window_size, process, print_events)
    except ValueError as e:
        print(f"ERROR: {e}")
    except KeyboardInterrupt:
        print("INFO: Received KeyboardInterrupt. Terminating process.")
    finally:
        terminate_DEFact(process)

    print("SUCCESS: All windows processed. Total features extracted:", len(all_features), all_features)
    terminate_DEFact(process)
    return all_features


def main_gedi_event_loop(
        input_params:Dict[str, Any],
        n_windows: int,
        output_queue: Queue,
        window_size: int,
        process: Process,
        print_events: bool=True
):
    window = []
    all_features: list[dict[str, float | int]] = []

    feature_set: List[str] = input_params.get(FEATURE_PARAMS).get(FEATURE_SET)

    for window_num in range(1, n_windows + 1):
        print(f"    INFO: Processing window {window_num}/{n_windows}...")

        while len(window) < window_size:
            window.append(output_queue.get())

        terminate_DEFact(process)

        el: EventLog = window_to_eventlog(window)

        input_params['input_path'] = OUTPUT_PATH

        features_per_window: Dict[str, float | int] = compute_features_from_event_data(feature_set, el)

        params_for_gen = {
            "generator_params": {
                "experiment": input_params,
                "config_space": input_params["config_space"],
            },
            "n_trials": 50
        }

        genED = GenerateEventLogs(params_for_gen)

        log_config: Dict[Configuration, EventLog, Dict[str, float]] = genED.log_config[0]

        config: Configuration= log_config["configuration"]

        init_DEFact(output_queue, config.config_space, print_events=print_events)

        all_features.append(features_per_window)

        print(f"   SUCCESS: Window {window_num}/{n_windows} processed successfully.",
              f"Extracted {len(features_per_window)} features from stream window")

        window.clear()


    return all_features