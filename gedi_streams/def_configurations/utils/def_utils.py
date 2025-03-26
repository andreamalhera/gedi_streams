import multiprocessing
from collections import defaultdict
from typing import Dict, Tuple, Any, List

import networkx as nx
import matplotlib.pyplot as plt
import pm4py
from pm4py import ProcessTree
from pm4py.objects.process_tree.obj import Operator

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

def visualize_markov_chain(markov_chain: dict[str, dict[str, float]]) -> None:
    """
    Visualizes a Markov chain as a directed weighted graph.

    :param markov_chain: Dictionary representation of the Markov chain.
    :return: None.
    """
    G = nx.DiGraph()

    for state, transitions in markov_chain.items():
        for next_state, prob in transitions.items():
            G.add_edge(state, next_state, weight=prob)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1000,
        node_color='lightblue',
        arrowsize=20,
        font_size=12,
        font_weight='bold',
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Markov Chain Visualization")
    plt.show()

def get_traces(process_tree: ProcessTree, loop_limit: int = 1) -> List[List[str]]:
    """
    Recursively extracts all possible traces from a process tree up to a given loop limit.

    :param process_tree: The process tree to extract traces from.
    :param loop_limit: Maximum number of loop iterations allowed.
    :return: A list of traces, where each trace is a list of labels (str).
    """
    if process_tree.operator is None:
        if process_tree.label is not None and process_tree.label != 'tau':
            return [[process_tree.label]]
        else:
            return [[]]
    op = process_tree.operator
    if op == Operator.SEQUENCE:
        traces: List[List[str]] = [[]]
        for child in process_tree.children:
            child_traces: List[List[str]] = get_traces(child, loop_limit)
            new_traces: List[List[str]] = []
            for trace in traces:
                for c_trace in child_traces:
                    new_traces.append(trace + c_trace)
            traces = new_traces
        return traces
    elif op == Operator.XOR or op == Operator.OR:
        traces: List[List[str]] = []
        for child in process_tree.children:
            traces.extend(get_traces(child, loop_limit))
        return traces
    elif op == Operator.LOOP:
        if len(process_tree.children) < 2:
            return get_traces(process_tree.children[0], loop_limit)
        body_traces: List[List[str]] = get_traces(process_tree.children[0], loop_limit)
        redo_traces: List[List[str]] = get_traces(process_tree.children[1], loop_limit)
        combined_traces: List[List[str]] = []
        for body in body_traces:
            combined_traces.append(body)
        for _ in range(loop_limit):
            new_combined: List[List[str]] = []
            for trace in combined_traces:
                for redo in redo_traces:
                    new_combined.append(trace + redo)
            combined_traces.extend(new_combined)
        return combined_traces
    else:
        raise NotImplementedError(f"Operator {op} not supported in trace extraction.")


def process_tree_to_markov_chain(process_tree: ProcessTree, loop_limit: int = 1) -> Dict[str, Dict[str, float]]:
    """
    Converts a process tree into a Markov chain represented as a nested dictionary with transition probabilities.

    :param process_tree: The process tree to convert.
    :param loop_limit: Maximum number of loop iterations to consider.
    :return: A Markov chain as a dictionary where keys are state labels and values are dictionaries
             mapping successor state labels to transition probabilities.
    """
    traces: List[List[str]] = get_traces(process_tree, loop_limit)
    count_transitions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for trace in traces:
        states: List[str] = ['START'] + [s for s in trace if s != ''] + ['END']
        for i in range(len(states) - 1):
            count_transitions[states[i]][states[i + 1]] += 1.0
    markov_chain: Dict[str, Dict[str, float]] = {}
    for state, next_states in count_transitions.items():
        total: float = sum(next_states.values())
        markov_chain[state] = {next_state: count / total for next_state, count in next_states.items()}
    return markov_chain

def create_transition_event_provider(state: str, next_state: str) -> CustomEventDataProvider:
    """Create an event data provider for a state transition."""
    return CustomEventDataProvider(
        duration_provider=ConstantDurationProvider(1),
        activity_provider=ConstantActivityProvider(state),
        transition_provider=ConstantTransitionProvider(next_state)
    )


def create_event_selection_provider(
        state: str,
        transitions: Dict[str, float]
) -> GenericProbabilityEventSelectionProvider:
    """Create an event selection provider based on transition probabilities."""
    potential_events: List[EventDataProvider] = []
    probability_distribution = []

    for next_state, probability in transitions.items():
        event_provider = create_transition_event_provider(state, next_state)
        potential_events.append(event_provider)
        probability_distribution.append(probability)

    return GenericProbabilityEventSelectionProvider(
        potential_events=potential_events,
        probability_distribution=probability_distribution
    )


def create_state_data_source(
        state: str,
        transitions: Dict[str, float]
) -> GenericDataSource:
    """Create a data source for a state in the Markov chain."""
    event_provider = create_event_selection_provider(state, transitions)

    return GenericDataSource(
        data_source_id=DataSourceId(state),
        group_id="markov_chain",
        event_provider=event_provider
    )


def create_start_data_source(start_nodes: List[str]) -> GenericDataSource:
    """Create a data source for the start state."""
    potential_events = [
        create_transition_event_provider("<start>", node)
        for node in start_nodes
    ]
    probability_distribution = [1 / len(start_nodes) for _ in start_nodes]

    return GenericDataSource(
        data_source_id=DataSourceId("<start>"),
        group_id="markov_chain",
        event_provider=GenericProbabilityEventSelectionProvider(
            potential_events=potential_events,
            probability_distribution=probability_distribution
        )
    )


def update_end_nodes(
        datasource_definitions: Dict[str, DataSource],
        end_nodes: List[str]
) -> None:
    """Update data sources for end nodes."""

    for (key, ds) in datasource_definitions.items():
        if key in end_nodes:
            ds.event_provider = GenericProbabilityEventSelectionProvider(
                potential_events=[
                    create_transition_event_provider(key, "<end>")
                ],
                probability_distribution=[1]
            )


def build_event_factory(
        datasource_definitions: Dict[str, DataSource],
        submodule_path: str,
        queue: multiprocessing.Queue,
        print_events: bool
) -> EventFactory:
    """Build and configure the event factory with data sources and sink."""
    event_factory = EventFactory()

    # Add all data sources
    for name, data_source in datasource_definitions.items():
        event_factory.add_datasource(name, data_source)

    # Add GEDI sink
    datasource_keys = list(datasource_definitions.keys())
    event_factory.add_sink(
        "gedi-sink",
        GEDIAdapter(
            "gedi-sink",
            datasource_keys,
            queue,
            disable_console_print=not print_events
        )
    )

    event_factory.add_file(f"{submodule_path}/config/simulation/stream.yaml")

    return event_factory


def init_and_compile_def_using_markov_chain(
        markov_chain: Dict[str, Dict[str, float]],
        start_nodes: List[str],
        end_nodes: List[str],
        submodule_path: str,
        queue: multiprocessing.Queue,
        print_events: bool = False,
) -> EventFactory:
    """Initialize and compile event factory definition using a Markov chain."""

    # Create data sources for each state in the Markov chain
    datasource_definitions = {
        state: create_state_data_source(state, transitions)
        for state, transitions in markov_chain.items()
    }

    # Add start state data source
    datasource_definitions["<start>"] = create_start_data_source(start_nodes)

    # Update end nodes to transition to <end>
    update_end_nodes(datasource_definitions, end_nodes)

    # Build and return the event factory
    event_factory: EventFactory = build_event_factory(
        datasource_definitions,
        submodule_path,
        queue,
        print_events
    )

    return event_factory

