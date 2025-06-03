import os
import sys
import traceback
from pathlib import Path
from queue import Queue
from typing import List, Optional, Union, Dict, Set, Any

import pandas as pd
import pm4py
from ConfigSpace import ConfigurationSpace, Configuration
from pm4py import ProcessTree, play_out
from pm4py.objects.log.obj import EventLog

from distributed_event_factory.core.datasource import GenericDataSource
from gedi_streams.def_configurations.sink.gedi_sink import GEDIAdapter
from DistributedEventFactory.distributed_event_factory.event_factory import EventFactory
from gedi_streams.def_configurations.utils.def_utils import init_and_compile_def_using_markov_chain, \
    visualize_markov_chain, convert_process_tree_to_dataframe_event_log, \
    build_markov_chain
from gedi_streams.generator.model import create_PTLG

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
submodule_path = os.path.join(project_root, "DistributedEventFactory")
sys.path.append(submodule_path)


class QueueOutput:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        # Flush is needed to avoid warnings about buffered output
        pass


def play_DEFact(
        config: Union[Configuration, ConfigurationSpace],
        queue: Queue,
        print_events: bool = True,
        visualize: bool = False
):
    """
    Enhanced DEFact simulation with structure-based parallelism.
    """
    specific_config: Configuration
    if isinstance(config, ConfigurationSpace):
        if print_events:
            print(f"INFO: play_DEFact received ConfigurationSpace. Sampling a configuration.")
        specific_config = config.sample_configuration()
        if specific_config is None:
            print("ERROR: ConfigurationSpace.sample_configuration() returned None. Cannot proceed.")
            return
    elif isinstance(config, Configuration):
        if print_events:
            print(f"INFO: play_DEFact received a specific Configuration.")
        specific_config = config
    elif isinstance(config, dict):
        config_space = ConfigurationSpace()
        specific_config = Configuration(config_space, config)
    else:
        print(f"ERROR: play_DEFact received invalid config type: {type(config)}. Exiting simulation.")
        return

    if print_events:
        print(
            f"INFO: Using configuration for DEFact: {specific_config.get_dictionary() if isinstance(specific_config, Configuration) else 'Sampled from Space'}")

    # Generate process tree
    tree: pm4py.ProcessTree = create_PTLG(specific_config)

    if print_events:
        print(f"INFO: Generated Process Tree for DEFact: {tree}")

    # Convert process tree to event log with parallelism information
    event_log: pd.DataFrame
    start_nodes: List[str]
    end_nodes: List[str]
    parallel_info: Dict[str, Set[str]]

    event_log, start_nodes, end_nodes, parallel_info = convert_process_tree_to_dataframe_event_log(tree)

    if print_events and parallel_info:
        print(f"INFO: Detected parallel relationships: {dict(parallel_info)}")
    elif print_events:
        print("INFO: No parallel relationships detected in process tree.")

    start_nodes = list(set(start_nodes))
    end_nodes = list(set(end_nodes))

    # Build enhanced Markov chain with concurrency information
    markov_chain: Dict[str, Dict[str, Dict[str, Any]]] = build_markov_chain(event_log, parallel_info)

    if print_events:
        print(f"INFO: Built enhanced Markov chain with {len(markov_chain)} states")
        # Print concurrency information in the chain
        concurrent_transitions = 0
        total_transitions = 0
        for state, transitions in markov_chain.items():
            for next_state, transition_info in transitions.items():
                total_transitions += 1
                if isinstance(transition_info, dict) and transition_info.get('concurrent', False):
                    concurrent_transitions += 1
                    if print_events:
                        print(f"    Concurrent transition: {state} -> {next_state}")

        if total_transitions > 0:
            concurrency_ratio = concurrent_transitions / total_transitions
            print(
                f"INFO: Concurrency ratio in Markov chain: {concurrency_ratio:.2%} ({concurrent_transitions}/{total_transitions})")

    if visualize:
        pm4py.view_petri_net(*pm4py.convert_to_petri_net(tree))
        # Convert enhanced markov chain to simple format for visualization
        simple_chain = {}
        for state, transitions in markov_chain.items():
            simple_chain[state] = {}
            for next_state, transition_info in transitions.items():
                if isinstance(transition_info, dict):
                    simple_chain[state][next_state] = transition_info.get('probability', 0)
                else:
                    simple_chain[state][next_state] = transition_info
        visualize_markov_chain(simple_chain)

    if 'submodule_path' not in globals() and 'submodule_path' not in locals():
        print(f"ERROR: 'submodule_path' is not defined. It's required for init_and_compile_def_using_markov_chain.")
        return

    # Extract activity duration from configuration
    activity_duration = specific_config.get("activity_duration", 5)

    # Note: concurrent_probability is now ignored since concurrency comes from structure
    concurrent_probability = specific_config.get("concurrent_probability", 0.3)

    if print_events:
        print(
            f"INFO: Using activity_duration={activity_duration}, ignoring concurrent_probability={concurrent_probability} (using structure-based concurrency)")

    # Initialize DEF with enhanced Markov chain
    def_instance: EventFactory = init_and_compile_def_using_markov_chain(
        markov_chain,
        start_nodes,
        end_nodes,
        submodule_path,
        queue,
        print_events,
        concurrent_probability,  # Passed for compatibility but ignored
        activity_duration
    )

    if print_events:
        print("INFO: Starting DEF simulation with structure-based concurrency...")

    def_instance.run()