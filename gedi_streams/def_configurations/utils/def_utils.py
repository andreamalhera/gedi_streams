import multiprocessing
import random
from collections import defaultdict
from typing import Dict, Tuple, Any, List, Set

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pm4py
from pm4py import ProcessTree
from pm4py.objects.log.obj import EventLog

from typing import Dict
import pandas as pd

from distributed_event_factory.core.abstract_datasource import DataSource
from distributed_event_factory.core.datasource import GenericDataSource
from distributed_event_factory.core.datasource_id import DataSourceId, START_SENSOR_ID
from distributed_event_factory.core.start_datasource import StartDataSource
from distributed_event_factory.event_factory import EventFactory
from distributed_event_factory.provider.activity.activity_provider import ConstantActivityProvider
from distributed_event_factory.provider.event.event_provider import EventDataProvider, CustomEventDataProvider
from distributed_event_factory.provider.eventselection.generic_probability_event_selection_provider import \
    GenericProbabilityEventSelectionProvider
from distributed_event_factory.provider.sink.test.test_sink_parser import TestSinkParser
from distributed_event_factory.provider.transition.duration.constant_duration import ConstantDurationProvider
from distributed_event_factory.provider.transition.transition.constant_transition import ConstantTransitionProvider
from gedi_streams.def_configurations.sink.gedi_sink import GEDIAdapter

LOOP_ACTIVITY_SUFFIX: str = "_self_loop"
DEFAULT_GROUP_ID: str = "markov_chain"
START_STATE_NAME: str = "<start>"
END_STATE_NAME: str = "<end>"

NODE_COLOR: str = 'white'
NODE_BORDER_COLOR: str = 'black'
NODE_BORDER_WIDTH: float = 1.0
NODE_LABEL_COLOR: str = 'black'  # Color for the labels on the nodes

# Edge styles
EDGE_FONT_COLOR: str = 'black'
ARROW_SIZE_DEFAULT: int = 15

# Font styles
FONT_SIZE_DEFAULT: int = 10
FONT_WEIGHT: str = 'bold'

# Node size
NODE_SIZE_DEFAULT: int = 1000

# Layout parameters
LAYOUT_K_FACTOR: float = 0.9  # Optimal distance factor for spring_layout, increased for better separation
LAYOUT_ITERATIONS: int = 75  # Increased iterations for spring_layout convergence
LAYOUT_SEED: int = 42  # Seed for reproducible layouts

# Figure settings
FIGURE_SIZE: Tuple[int, int] = (12, 12)

# Thresholds for dynamic adjustments
NODE_COUNT_THRESHOLD_MEDIUM: int = 20
NODE_COUNT_THRESHOLD_LARGE: int = 50

# Scaling factors for dynamic adjustments
SCALE_FACTOR_MEDIUM_GRAPH: float = 0.8
SCALE_FACTOR_LARGE_GRAPH: float = 0.6
NODE_SIZE_MIN: int = 100
FONT_SIZE_MIN: int = 12
ARROW_SIZE_MIN: int = 8


def visualize_markov_chain(markov_chain: Dict[str, Dict[str, float]]) -> None:
    """
    Visualizes a Markov chain as a directed weighted graph, optimized for larger chains
    with specific node and edge label styling.

    Nodes are white with a black border. Edge labels and node labels are black.
    Layout and element sizes are adjusted to reduce node overlap for larger chains.

    :param markov_chain: Dictionary representation of the Markov chain.
                         Example: {'A': {'B': 0.5, 'C': 0.5}, 'B': {'A': 1.0}}
    :return: None. Displays the plot.
    """
    graph: nx.DiGraph = nx.DiGraph()

    node_names_to_add: Set[str] = set()
    for state_key_outer in markov_chain:
        node_names_to_add.add(str(state_key_outer))
        for target_key_inner in markov_chain[state_key_outer]:
            if isinstance(markov_chain[state_key_outer][target_key_inner], dict):
                # New format with concurrency metadata
                node_names_to_add.add(str(target_key_inner))
            else:
                # Old format - direct probability
                node_names_to_add.add(str(target_key_inner))

    for name in node_names_to_add:
        graph.add_node(name)

    # Add edges
    for state, transitions in markov_chain.items():
        for next_state, transition_data in transitions.items():
            if isinstance(transition_data, dict):
                # New format with concurrency metadata
                prob = transition_data.get('probability', transition_data.get('prob', 0))
            else:
                # Old format - direct probability
                prob = transition_data

            if prob > 0:  # Optionally, only draw edges with non-zero probability
                graph.add_edge(str(state), str(next_state), weight=prob)

    num_nodes: int = graph.number_of_nodes()
    pos: Dict[str, Tuple[float, float]]

    if num_nodes == 0:  # Should be caught by the initial check, but as a safeguard
        plt.figure(figsize=FIGURE_SIZE)
        plt.title("Markov Chain Visualization (No Nodes)")
        plt.axis('off')
        plt.show()
        return
    elif num_nodes == 1:
        single_node_name: str = list(graph.nodes())[0]
        pos = {single_node_name: (0.5, 0.5)}
    else:
        pos = nx.spring_layout(
            graph,
            k=LAYOUT_K_FACTOR,
            iterations=LAYOUT_ITERATIONS,
            seed=LAYOUT_SEED
        )

    edge_labels: Dict[Tuple[str, str], str] = {
        (str(u), str(v)): f'{d.get("weight", 0.0):.2f}'
        for u, v, d in graph.edges(data=True)
    }

    current_node_size: int = NODE_SIZE_DEFAULT
    current_font_size: int = FONT_SIZE_DEFAULT
    current_arrow_size: int = ARROW_SIZE_DEFAULT

    if num_nodes > NODE_COUNT_THRESHOLD_LARGE:
        scale_factor: float = float(NODE_COUNT_THRESHOLD_LARGE) / num_nodes
        current_node_size = max(NODE_SIZE_MIN, int(NODE_SIZE_DEFAULT * scale_factor * SCALE_FACTOR_LARGE_GRAPH))
        current_font_size = max(FONT_SIZE_MIN, int(FONT_SIZE_DEFAULT * scale_factor * SCALE_FACTOR_LARGE_GRAPH))
        current_arrow_size = max(ARROW_SIZE_MIN, int(ARROW_SIZE_DEFAULT * scale_factor * SCALE_FACTOR_LARGE_GRAPH))
    elif num_nodes > NODE_COUNT_THRESHOLD_MEDIUM:
        scale_factor: float = float(NODE_COUNT_THRESHOLD_MEDIUM) / num_nodes
        current_node_size = max(NODE_SIZE_MIN, int(NODE_SIZE_DEFAULT * scale_factor * SCALE_FACTOR_MEDIUM_GRAPH))
        current_font_size = max(FONT_SIZE_MIN, int(FONT_SIZE_DEFAULT * scale_factor * SCALE_FACTOR_MEDIUM_GRAPH))
        current_arrow_size = max(ARROW_SIZE_MIN, int(ARROW_SIZE_DEFAULT * scale_factor * SCALE_FACTOR_MEDIUM_GRAPH))

    plt.figure(figsize=FIGURE_SIZE)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=current_node_size,
        node_color=NODE_COLOR,
        edgecolors=NODE_BORDER_COLOR,
        linewidths=NODE_BORDER_WIDTH,
        arrowsize=current_arrow_size,
        font_size=current_font_size,
        font_weight=FONT_WEIGHT,
        font_color=NODE_LABEL_COLOR
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_color=EDGE_FONT_COLOR,
        font_size=max(FONT_SIZE_MIN, current_font_size - 2),
        font_weight=FONT_WEIGHT
    )

    plt.title("Markov Chain Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def analyze_process_tree_parallelism(tree: ProcessTree) -> Dict[str, Set[str]]:
    """
    Analyze process tree to identify activities that should execute concurrently.

    Returns:
        Dict mapping each activity to set of activities it should run concurrently with
    """
    parallel_groups: Dict[str, Set[str]] = defaultdict(set)

    def traverse_tree(node: ProcessTree, parent_is_parallel: bool = False) -> Set[str]:
        """
        Recursively traverse the process tree to find parallel constructs.
        Returns set of activity names found in this subtree.
        """
        if node is None:
            return set()

        # Get the operator type
        if hasattr(node, 'operator'):
            operator = node.operator
        else:
            operator = None

        # Base case: leaf node (activity)
        if not hasattr(node, 'children') or not node.children:
            if hasattr(node, 'label') and node.label:
                return {node.label}
            else:
                return set()

        # Recursive case: internal node with children
        current_activities = set()

        if operator and str(operator) == 'Operator.PARALLEL':
            # This is a parallel operator - its children should execute concurrently
            child_activity_groups = []

            for child in node.children:
                child_activities = traverse_tree(child, parent_is_parallel=True)
                child_activity_groups.append(child_activities)
                current_activities.update(child_activities)

            # Mark all activities in different parallel branches as concurrent with each other
            for i, group1 in enumerate(child_activity_groups):
                for j, group2 in enumerate(child_activity_groups):
                    if i != j:  # Different parallel branches
                        for activity1 in group1:
                            for activity2 in group2:
                                parallel_groups[activity1].add(activity2)
                                parallel_groups[activity2].add(activity1)
        else:
            # Sequential, choice, loop, or other operator
            for child in node.children:
                child_activities = traverse_tree(child, parent_is_parallel=False)
                current_activities.update(child_activities)

        return current_activities

    # Start traversal from root
    traverse_tree(tree)

    return dict(parallel_groups)


def build_markov_chain(event_log: pd.DataFrame, parallel_info: Dict[str, Set[str]] = None) -> Dict[
    str, Dict[str, Dict[str, Any]]]:
    """
    Build a first-order Markov chain from an event log, including 'START' and 'END' pseudo-states.
    Enhanced to include concurrency information from process tree structure.

    :param event_log: DataFrame with columns 'case:concept:name', 'concept:name'.
    :param parallel_info: Dict mapping activities to sets of concurrent activities
    :return: A dict mapping each state to a dict of next-state information including probability and concurrency
    """
    if parallel_info is None:
        parallel_info = {}

    # sort by case and timestamp
    df_sorted = event_log.sort_values(
        by=['case:concept:name']
    )
    counts: Dict[str, Dict[str, int]] = {}

    for _, group in df_sorted.groupby('case:concept:name'):
        activities = group['concept:name'].tolist()
        if not activities:
            continue
        # transition from START into each case
        counts.setdefault(START_STATE_NAME, {}).setdefault(activities[0], 0)
        counts[START_STATE_NAME][activities[0]] += 1
        # transitions within the case
        for current_act, next_act in zip(activities, activities[1:]):
            counts.setdefault(current_act, {}).setdefault(next_act, 0)
            counts[current_act][next_act] += 1
        # transition from last activity to END
        last_state = activities[-1]
        counts.setdefault(last_state, {}).setdefault(END_STATE_NAME, 0)
        counts[last_state][END_STATE_NAME] += 1

    # convert counts to probabilities and add concurrency information
    chain: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for state, transitions in counts.items():
        total = sum(transitions.values())
        chain[state] = {}
        for next_state, count in transitions.items():
            probability = count / total

            # Determine if this transition should be concurrent
            is_concurrent = False
            if state in parallel_info and next_state in parallel_info[state]:
                is_concurrent = True

            chain[state][next_state] = {
                'probability': probability,
                'concurrent': is_concurrent
            }

    # ensure <end> appears as a state (with no outgoing transitions)
    chain.setdefault(END_STATE_NAME, {})

    return chain


def convert_process_tree_to_dataframe_event_log(tree_: ProcessTree) -> Tuple[
    pd.DataFrame, List[str], List[str], Dict[str, Set[str]]]:
    """
    Convert a process tree to a DataFrame event log.
    Enhanced to also return parallelism information.

    :param tree_: ProcessTree object.
    :return: Tuple of (DataFrame, start_nodes, end_nodes, parallel_info)
    """
    # Analyze the process tree for parallel constructs
    parallel_info = analyze_process_tree_parallelism(tree_)

    # Play out the process tree
    traces: EventLog = pm4py.play_out(tree_, parameters={"num_traces": 100})
    case_id: int = 0

    start_nodes: List[str] = []
    end_nodes: List[str] = []

    for trace in traces:
        if len(trace) > 0 and "concept:name" in trace[0] and "concept:name" in trace[-1]:
            start_nodes.append(trace[0]["concept:name"])
            end_nodes.append(trace[-1]["concept:name"])

        for event in trace:
            event["case:concept:name"] = case_id
        case_id += 1

    return pm4py.convert_to_dataframe(traces), [START_STATE_NAME], [END_STATE_NAME], parallel_info


def create_transition_event_provider(
        state: str,
        next_state: str,
        is_loop: bool,
        is_concurrent: bool = False,  # NEW: structure-derived concurrency
        activity_duration: int = 5
) -> CustomEventDataProvider:
    """
    Create an event data provider for a state transition.
    If is_loop is True, the activity name is modified.
    Enhanced to use structure-based concurrency instead of random probability.

    :param state: Current state name
    :param next_state: Next state name
    :param is_loop: Whether this is a loop transition
    :param is_concurrent: Whether this transition should be concurrent (from process structure)
    :param activity_duration: Duration for the activity
    """
    activity_name: str = state

    if is_loop:
        activity_name = f"{state}{LOOP_ACTIVITY_SUFFIX}"

    return CustomEventDataProvider(
        duration_provider=ConstantDurationProvider(activity_duration),
        activity_provider=ConstantActivityProvider(activity_name),
        transition_provider=ConstantTransitionProvider(next_state),
        concurrent_next=is_concurrent,  # Use structure-derived concurrency
    )


def create_event_selection_provider(
        state: str,
        transitions: Dict[str, Dict[str, Any]],  # Enhanced structure
        activity_duration: int = 5
) -> GenericProbabilityEventSelectionProvider:
    """
    Create an event selection provider based on transition probabilities.
    Enhanced to handle new transition structure with concurrency information.

    :param state: Current state name
    :param transitions: Dictionary of next states and their transition information
    :param activity_duration: Duration for the activity
    """
    potential_events: List[EventDataProvider] = []
    probability_distribution: List[float] = []

    for next_state, transition_info in transitions.items():
        # Extract probability and concurrency from transition info
        if isinstance(transition_info, dict):
            probability = transition_info.get('probability', transition_info.get('prob', 0))
            is_concurrent = transition_info.get('concurrent', False)
        else:
            # Backward compatibility - assume old format
            probability = transition_info
            is_concurrent = False

        is_loop: bool = (state == next_state)
        event_provider: CustomEventDataProvider = create_transition_event_provider(
            state,
            next_state,
            is_loop,
            is_concurrent,  # Use structure-derived concurrency
            activity_duration
        )
        potential_events.append(event_provider)
        probability_distribution.append(probability)

    return GenericProbabilityEventSelectionProvider(
        potential_events=potential_events,
        probability_distribution=probability_distribution
    )


def create_state_data_source(
        state: str,
        transitions: Dict[str, Dict[str, Any]],  # Enhanced structure
        activity_duration: int = 5
) -> GenericDataSource:
    """
    Create a data source for a state in the Markov chain.
    Enhanced to handle new transition structure.

    :param state: State name
    :param transitions: Dictionary of next states and their transition information
    :param activity_duration: Duration for the activity
    """
    event_provider: GenericProbabilityEventSelectionProvider = \
        create_event_selection_provider(state, transitions, activity_duration)

    return GenericDataSource(
        data_source_id=DataSourceId(state),
        group_id=DEFAULT_GROUP_ID,
        event_provider=event_provider
    )


def create_start_data_source(
        markov_chain: Dict[str, Dict[str, Dict[str, Any]]],  # Enhanced structure
        activity_duration: int = 5
) -> GenericDataSource:
    """
    Create a data source for the start state.
    Enhanced to handle new markov chain structure.

    :param markov_chain: The markov chain dictionary with enhanced structure
    :param activity_duration: Duration for the activity
    """
    transitions: Dict[str, Dict[str, Any]] = markov_chain[START_STATE_NAME]

    potential_events: List[EventDataProvider] = []
    probability_distribution: List[float] = []

    for node, transition_info in transitions.items():
        if isinstance(transition_info, dict):
            probability = transition_info.get('probability', transition_info.get('prob', 0))
            is_concurrent = transition_info.get('concurrent', False)
        else:
            # Backward compatibility
            probability = transition_info
            is_concurrent = False

        event_provider = create_transition_event_provider(
            START_STATE_NAME,
            node,
            is_loop=False,
            is_concurrent=is_concurrent,
            activity_duration=activity_duration
        )
        potential_events.append(event_provider)
        probability_distribution.append(probability)

    return GenericDataSource(
        data_source_id=DataSourceId(START_STATE_NAME),
        group_id=DEFAULT_GROUP_ID,
        event_provider=GenericProbabilityEventSelectionProvider(
            potential_events=potential_events,
            probability_distribution=probability_distribution
        )
    )


def update_end_nodes(
        datasource_definitions: Dict[str, DataSource],
        end_nodes: List[str],
        activity_duration: int = 5
) -> None:
    """
    Update data sources for end nodes to transition to a terminal <end> state.
    Simplified to remove unused concurrency parameters.

    :param datasource_definitions: Dictionary of data sources
    :param end_nodes: List of end node names
    :param activity_duration: Duration for the activity
    """
    for key, ds in datasource_definitions.items():
        if key in end_nodes:
            event_provider: CustomEventDataProvider = create_transition_event_provider(
                key,
                END_STATE_NAME,
                is_loop=False,
                is_concurrent=False,  # End transitions are never concurrent
                activity_duration=activity_duration
            )
            ds.event_provider = GenericProbabilityEventSelectionProvider(
                potential_events=[event_provider],
                probability_distribution=[1.0]
            )


def build_event_factory(
        datasource_definitions: Dict[str, DataSource],
        submodule_path: str,
        queue: multiprocessing.Queue,
        print_events: bool
) -> EventFactory:
    """Build and configure the event factory with data sources and sink."""
    event_factory: EventFactory = EventFactory()

    for name, data_source in datasource_definitions.items():
        event_factory.add_datasource(name, data_source)

    datasource_keys: List[str] = list(datasource_definitions.keys())
    gedi_sink: GEDIAdapter = GEDIAdapter(
        "gedi-sink",
        datasource_keys,
        queue,
        disable_console_print=not print_events
    )
    event_factory.add_sink("gedi-sink", gedi_sink)

    event_factory.add_file(f"{submodule_path}/config/simulation/stream.yaml")

    return event_factory


def init_and_compile_def_using_markov_chain(
        markov_chain: Dict[str, Dict[str, Dict[str, Any]]],  # Enhanced structure
        start_nodes: List[str],
        end_nodes: List[str],
        submodule_path: str,
        queue: multiprocessing.Queue,
        print_events: bool = False,
        concurrent_probability: float = 0.3,  # Deprecated but kept for compatibility
        activity_duration: int = 5
) -> EventFactory:
    """
    Initialize and compile event factory definition using a Markov chain.
    Enhanced to use structure-based concurrency instead of random probability.

    :param markov_chain: The markov chain dictionary with enhanced structure
    :param start_nodes: List of start node names
    :param end_nodes: List of end node names
    :param submodule_path: Path to submodule
    :param queue: Multiprocessing queue
    :param print_events: Whether to print events
    :param concurrent_probability: Deprecated - concurrency now comes from structure
    :param activity_duration: Duration for the activity
    """
    print(
        f"INFO: Building DEF with structure-based concurrency. Random concurrent_probability={concurrent_probability} is ignored.")

    datasource_definitions: Dict[str, DataSource] = {
        state: create_state_data_source(state, transitions, activity_duration)
        for state, transitions in markov_chain.items()
        if state != START_STATE_NAME  # Handle start state separately
    }

    datasource_definitions[START_STATE_NAME] = create_start_data_source(markov_chain, activity_duration)

    update_end_nodes(datasource_definitions, end_nodes, activity_duration)

    event_factory: EventFactory = build_event_factory(
        datasource_definitions,
        submodule_path,
        queue,
        print_events
    )

    return event_factory