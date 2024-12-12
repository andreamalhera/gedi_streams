from pprint import pprint

import numpy as np
import pm4py.objects.petri_net.saw_net.semantics
from pm4py import generate_process_tree

from distributed_event_factory.core.datasource import GenericDataSource
from distributed_event_factory.core.datasource_id import START_SENSOR_ID, DataSourceId
from distributed_event_factory.event_factory import EventFactory
from distributed_event_factory.provider.activity.activity_provider import ConstantActivityProvider
from distributed_event_factory.provider.event.event_provider import CustomEventDataProvider, EventDataProvider
from distributed_event_factory.provider.eventselection.generic_probability_event_selection_provider import \
    GenericProbabilityEventSelectionProvider
from distributed_event_factory.provider.sink.test.test_sink_parser import TestSinkParser
from distributed_event_factory.provider.transition.duration.constant_duration import ConstantDurationProvider
from distributed_event_factory.provider.transition.transition.constant_transition import ConstantTransitionProvider
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import defaultdict

class MarkovChain:
    """Core Markov chain implementation."""

    def __init__(self, transition_matrix: Dict[str, Dict[str, float]]):
        self.transition_matrix = transition_matrix
        self.validate_transitions()

    def validate_transitions(self) -> None:
        """Validate that probabilities sum to 1.0 for each state."""
        for state, transitions in self.transition_matrix.items():
            total_prob = sum(transitions.values())
            if not (0.99 <= total_prob <= 1.01):  # Allow for minor float imprecision
                raise ValueError(f"Probabilities for state {state} sum to {total_prob}, not 1.0")

    def get_transitions(self, state: str) -> Dict[str, float]:
        """Get all possible transitions for a given state."""
        return self.transition_matrix.get(state, {})

    def get_states(self) -> List[str]:
        """Get all states in the Markov chain."""
        return list(self.transition_matrix.keys())



@dataclass
class DefaultEventConfig:
    """Configuration for event generation."""
    duration: int = 1
    group_id: str = "markov_chain"


class MarkovEventAdapter:
    """Adapter to convert Markov chain states into event factory events."""

    def __init__(self,
                 markov_chain: MarkovChain,
                 event_factory: EventFactory,
                 config: DefaultEventConfig):
        self.markov_chain = markov_chain
        self.event_factory = event_factory
        self.config = config

    def create_event_provider(self, state: str, next_state: str) -> EventDataProvider:
        """Create an event provider for a state transition."""
        return CustomEventDataProvider(
            duration_provider=ConstantDurationProvider(self.config.duration),
            activity_provider=ConstantActivityProvider(state),
            transition_provider=ConstantTransitionProvider(next_state)
        )

    def create_datasource(self, state: str, transitions: Dict[str, float]) -> GenericDataSource:
        """Create a data source for a state and its transitions."""
        potential_events = [
            self.create_event_provider(state, next_state)
            for next_state in transitions.keys()
        ]

        event_selection_provider = GenericProbabilityEventSelectionProvider(
            potential_events=potential_events,
            probability_distribution=list(transitions.values())
        )

        return GenericDataSource(
            data_source_id=DataSourceId(state),
            group_id=self.config.group_id,
            event_provider=event_selection_provider
        )

    def setup_start_state(self, initial_state: str) -> None:
        """Set up the initial state in the event factory."""
        start_event_provider = self.create_event_provider("<start>", initial_state)
        start_datasource = GenericDataSource(
            data_source_id=START_SENSOR_ID,
            group_id=self.config.group_id,
            event_provider=GenericProbabilityEventSelectionProvider(
                potential_events=[start_event_provider],
                probability_distribution=[1.0]
            )
        )
        self.event_factory.add_datasource("<start>", start_datasource)

    def register_all_states(self) -> None:
        """Register all Markov chain states as data sources."""
        for state in self.markov_chain.get_states():
            transitions = self.markov_chain.get_transitions(state)
            datasource = self.create_datasource(state, transitions)
            self.event_factory.add_datasource(state, datasource)




def create_transition_matrix(sequence_data: Dict[List[str], float]) -> Dict[str, Dict[str, float]]:
    """
    Convert sequence data into a Markov chain transition matrix.

    Args:
        sequence_data: Dictionary mapping sequences (as lists) to their probabilities

    Returns:
        Dictionary representing the transition matrix where transitions[state1][state2]
        gives the probability of transitioning from state1 to state2
    """
    # Initialize transition counts
    transitions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    state_counts: Dict[str, float] = defaultdict(float)

    # Process each sequence and its probability
    for sequence, probability in sequence_data.items():
        if len(sequence) <= 1:
            continue

        # Add transitions for each consecutive pair in the sequence
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]

            # Weight the transition by the sequence probability
            transitions[current_state][next_state] += probability
            state_counts[current_state] += probability

    # Normalize transitions to get probabilities
    for state in transitions:
        if state_counts[state] > 0:
            for next_state in transitions[state]:
                transitions[state][next_state] /= state_counts[state]

    return {
        state: dict(state_transitions)
        for state, state_transitions in transitions.items()
    }


def process_transition_matrix(
        transitions: Dict[str, Dict[str, float]],
        threshold: float = 0.1,
        all_states: bool = True
) -> Dict[str, Dict[str, float]]:
    all_state_set = set()
    for state in transitions:
        all_state_set.add(state)
        all_state_set.update(transitions[state].keys())

    processed: Dict[str, Dict[str, float]] = defaultdict(dict)

    for current_state in transitions:
        valid_transitions = {
            next_state: prob
            for next_state, prob in transitions[current_state].items()
            if prob >= threshold
        }

        if not valid_transitions:
            print(f"Warning: State '{current_state}' has no transitions above threshold {threshold}")
            valid_transitions = transitions[current_state]  # Keep original transitions in this case

        total_prob = sum(valid_transitions.values())
        normalized_transitions = {
            state: prob / total_prob
            for state, prob in valid_transitions.items()
        }

        processed[current_state] = normalized_transitions

        if all_states:
            for state in all_state_set:
                if state not in processed[current_state]:
                    processed[current_state][state] = 0.0

    if all_states:
        for state in all_state_set:
            if state not in processed:
                processed[state] = {s: 0.0 for s in all_state_set}

    return dict(processed)


def analyze_states(transition_dict: Dict[str, Dict[str, float]]) -> Tuple[Set[str], Set[str]]:
    """
    Analyze the Markov chain to identify absorbing and transient states.

    Args:
        transition_dict: Dictionary of state transitions

    Returns:
        Tuple of (absorbing states, transient states)
    """
    states = set(transition_dict.keys())
    absorbing = set()

    # First pass: identify states with no outgoing transitions or only self-transitions
    for state, transitions in transition_dict.items():
        non_zero_transitions = {dest for dest, prob in transitions.items() if prob > 0}
        if not non_zero_transitions or non_zero_transitions == {state}:
            absorbing.add(state)

    # Identify strongly connected components that form absorbing classes
    def find_reachable_states(start_state: str, visited: Set[str] = None) -> Set[str]:
        if visited is None:
            visited = set()
        visited.add(start_state)
        for next_state, prob in transition_dict[start_state].items():
            if prob > 0 and next_state not in visited:
                find_reachable_states(next_state, visited)
        return visited

    # For each absorbing state, find all states that can only reach other absorbing states
    potential_absorbing = set()
    for state in states:
        reachable = find_reachable_states(state)
        if all(s in absorbing for s in reachable - {state}):
            potential_absorbing.add(state)

    absorbing = potential_absorbing
    transient = states - absorbing

    return absorbing, transient


def calculate_absorption_probabilities(transition_dict: Dict[str, Dict[str, float]],
                                       start_state: str = None) -> Dict[str, float]:
    """
    Calculate absorption probabilities for a Markov chain with proper handling of absorbing states.

    Args:
        transition_dict: Dictionary of state transitions
        start_state: Initial state (if None, will use first state alphabetically)

    Returns:
        Dictionary mapping absorbing states to their absorption probabilities
    """
    # Identify absorbing and transient states
    absorbing_states, transient_states = analyze_states(transition_dict)

    if not absorbing_states:
        raise ValueError("No absorbing states found in the Markov chain")

    # Create ordered list of states with transient states first, then absorbing states
    ordered_states = sorted(list(transient_states)) + sorted(list(absorbing_states))
    n = len(ordered_states)
    state_to_idx = {state: idx for idx, state in enumerate(ordered_states)}

    # Build the transition matrix P
    P = np.zeros((n, n))
    for from_state, transitions in transition_dict.items():
        from_idx = state_to_idx[from_state]
        row_sum = 0
        for to_state, prob in transitions.items():
            to_idx = state_to_idx[to_state]
            P[from_idx, to_idx] = prob
            row_sum += prob
        # Normalize if row sum is not 1
        if row_sum > 0 and row_sum != 1:
            P[from_idx, :] /= row_sum

    # Get Q (transitions between transient states) and R (transitions to absorbing states)
    n_transient = len(transient_states)
    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]

    # Calculate fundamental matrix N = (I - Q)^(-1)
    I = np.eye(n_transient)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to compute fundamental matrix. Check chain structure.")

    # Calculate absorption probabilities B = NR
    B = N @ R

    # Create initial distribution
    if start_state is None:
        start_state = min(transient_states)
    if start_state not in state_to_idx:
        raise ValueError(f"Start state {start_state} not found in chain")

    start_idx = state_to_idx[start_state]
    if start_idx >= n_transient:
        # If start state is absorbing, probability is 1 for that state
        result = {state: 1.0 if state == start_state else 0.0
                  for state in absorbing_states}
    else:
        # Calculate absorption probabilities from start state
        probs = B[start_idx]
        result = {state: probs[i]
                  for i, state in enumerate(sorted(list(absorbing_states)))}

    return result


def main():

    tree = generate_process_tree(parameters={"mode": 15})
    pn, im, fm = pm4py.convert_to_petri_net(tree)
    pm4py.view_petri_net(pn, im, fm)

    for trans in pn.transitions:
        print(trans)


    sl: dict[list[str], float] = pm4py.get_stochastic_language(pn, im, fm)

    transition_matrix: dict[str, dict[str, float]] = create_transition_matrix(sl)

    new_matrix = process_transition_matrix(transition_matrix)

    print(pprint(new_matrix))

    try:
        A, absorption_probs = calculate_absorption_probabilities(new_matrix)

        print("Normalized transition matrix A:")
        print(A)
        print(f"\nAbsorption probabilities: {absorption_probs}")

    except ValueError as e:
        print(f"Error: {e}")
    exit()

    # Create the Markov chain
    markov_chain = MarkovChain(markov_definition)

    event_factory = EventFactory()

    # Configure the event factory
    event_factory.add_sink_parser("test", TestSinkParser())
    event_factory.add_file("/Users/christianimenkamp/Documents/Git-Repositorys/DistributedEventFactory/config/simulation/stream.yaml")
    event_factory.add_file("/Users/christianimenkamp/Documents/Git-Repositorys/DistributedEventFactory/config/sink/console-sink.yaml")

    # Create the adapter with configuration
    config = DefaultEventConfig(duration=1, group_id="markov_chain")
    adapter = MarkovEventAdapter(markov_chain, event_factory, config)

    # Set up the system
    adapter.setup_start_state("a")
    adapter.register_all_states()

    # Run the event factory
    event_factory.run()


if __name__ == "__main__":
    main()
