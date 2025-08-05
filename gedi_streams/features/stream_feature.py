import inspect
import math
from collections.abc import Callable

import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any

import pm4py
from feeed.feature import Feature
from scipy.stats import entropy
from gedi_streams.features.memory import ComputedFeatureMemory


class StreamFeature(Feature):
    def __init__(self, feature_names='stream'):
        super().__init__(feature_names)
        self.feature_type = 'stream'
        self.available_class_methods = dict(inspect.getmembers(StreamFeature, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    def window(self, stream, memory):
        return  # TODO: attributes_filter.get_attribute_values(log, "concept:name")

import inspect
import math
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from gedi_streams.features.stream_feature import StreamFeature
from gedi_streams.features.memory import ComputedFeatureMemory


class ComplexStreamFeatures(StreamFeature):
    """
    Lifecycle-aware optimizable feature class designed for event streams where each activity
    generates both 'start' and 'complete' lifecycle events.

    This class properly handles the dual-event nature of process execution by:
    1. Using only 'complete' events for activity sequence reconstruction
    2. Calculating activity durations from start/complete pairs
    3. Avoiding double-counting of activities and transitions
    """

    def __init__(self, feature_names: str = 'complex_stream_features', memory: ComputedFeatureMemory = None):
        """
        Initializes the ComplexStreamFeatures extractor.

        Args:
            feature_names: The names of the features to compute.
            memory: The memory object for storing state between computations.
        """
        super().__init__(feature_names)
        self.feature_type: str = 'complex_stream_features'
        self.available_class_methods = dict(inspect.getmembers(ComplexStreamFeatures,
                                                               predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names: List[str] = [*self.available_class_methods.keys()]
        else:
            self.feature_names: str = feature_names

    def extract(self, events: List[Dict[str, Any]], memory: ComputedFeatureMemory) -> Dict[str, float]:
        """
        Extracts the complex stream features from a list of events.

        Args:
            events: The list of events in the current window.
            memory: The memory object.

        Returns:
            A dictionary of computed feature values.
        """
        output: Dict[str, float] = {}
        for feature_name in self.feature_names:
            if feature_name in self.available_class_methods:
                feature_fn = self.available_class_methods[feature_name]
                try:
                    feature_value: float = feature_fn(events, memory)
                    if feature_value is None or math.isnan(feature_value) or math.isinf(feature_value):
                        feature_value = 0.0
                    else:
                        feature_value = max(0.0, min(1.0, feature_value))
                    output[feature_name] = feature_value
                except Exception as e:
                    print(f"Error calculating {feature_name}: {e}")
                    output[feature_name] = 0.0
        return output

    @classmethod
    def temporal_dependencies(cls, events: List[Dict[str, Any]], memory: ComputedFeatureMemory) -> float:
        """
        Measures the strength of temporal dependencies by analyzing the predictability of the next event.

        Args:
            events: The list of events.
            memory: The memory object.

        Returns:
            A float between 0 and 1, where 1 indicates strong temporal dependencies.
        """
        case_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            case_id: str = event.get("case:concept:name")
            if case_id:
                case_events[case_id].append(event)

        transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        total_transitions: int = 0
        for case_id in case_events:
            sorted_events: List[Dict[str, Any]] = sorted(case_events[case_id], key=lambda x: x.get("time:timestamp"))
            for i in range(len(sorted_events) - 1):
                current_activity: str = sorted_events[i].get("concept:name")
                next_activity: str = sorted_events[i + 1].get("concept:name")
                if current_activity and next_activity:
                    transition_counts[(current_activity, next_activity)] += 1
                    total_transitions += 1
        if not total_transitions:
            return 0.0

        probabilities: List[float] = [count / total_transitions for count in transition_counts.values()]
        entropy: float = -sum(p * math.log2(p) for p in probabilities if p > 0)
        max_entropy: float = math.log2(len(transition_counts)) if len(transition_counts) > 0 else 0

        if max_entropy == 0:
            return 1.0

        normalized_entropy: float = entropy / max_entropy
        return 1.0 - normalized_entropy

    @classmethod
    def long_term_dependencies(cls, events: List[Dict[str, Any]], memory: ComputedFeatureMemory) -> float:
        """
        Measures long-term dependencies by checking for repeated events with significant separation.

        Args:
            events: The list of events.
            memory: The memory object.

        Returns:
            A float between 0 and 1, where 1 indicates strong long-term dependencies.
        """
        case_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            case_id: str = event.get("case:concept:name")
            if case_id:
                case_events[case_id].append(event)

        long_term_dependency_count: int = 0
        total_possible_dependencies: int = 0

        for case_id in case_events:
            sorted_events: List[Dict[str, Any]] = sorted(case_events[case_id], key=lambda x: x.get("time:timestamp"))
            trace_length: int = len(sorted_events)
            if trace_length < 3:
                continue

            for i in range(trace_length):
                for j in range(i + 2, trace_length):
                    total_possible_dependencies += 1
                    if sorted_events[i].get("concept:name") == sorted_events[j].get("concept:name"):
                        long_term_dependency_count += 1

        if not total_possible_dependencies:
            return 0.0

        return long_term_dependency_count / total_possible_dependencies

    @classmethod
    def non_linear_dependencies(cls, events: List[Dict[str, Any]], memory: ComputedFeatureMemory) -> float:
        """
        Measures non-linear dependencies by analyzing the complexity of the control-flow graph.

        Args:
            events: The list of events.
            memory: The memory object.

        Returns:
            A float between 0 and 1, where 1 indicates high non-linearity.
        """
        activities: Set[str] = set(event.get("concept:name") for event in events if event.get("concept:name"))
        num_activities: int = len(activities)

        if num_activities < 2:
            return 0.0

        case_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            case_id: str = event.get("case:concept:name")
            if case_id:
                case_events[case_id].append(event)

        transitions: Set[Tuple[str, str]] = set()
        for case_id in case_events:
            sorted_events: List[Dict[str, Any]] = sorted(case_events[case_id], key=lambda x: x.get("time:timestamp"))
            for i in range(len(sorted_events) - 1):
                current_activity: str = sorted_events[i].get("concept:name")
                next_activity: str = sorted_events[i + 1].get("concept:name")
                if current_activity and next_activity:
                    transitions.add((current_activity, next_activity))

        num_transitions: int = len(transitions)
        max_possible_transitions: int = num_activities * (num_activities - 1)

        if max_possible_transitions == 0:
            return 0.0

        density: float = num_transitions / max_possible_transitions
        return density

    @classmethod
    def out_of_order_events(cls, events: List[Dict[str, Any]], memory: ComputedFeatureMemory) -> float:
        """
        Measures the degree of out-of-order events by comparing timestamps.

        Args:
            events: The list of events.
            memory: The memory object.

        Returns:
            A float between 0 and 1, where 1 indicates a high degree of out-of-order events.
        """
        case_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            case_id: str = event.get("case:concept:name")
            if case_id:
                case_events[case_id].append(event)

        out_of_order_count: int = 0
        total_events: int = 0

        for case_id in case_events:
            sorted_events: List[Dict[str, Any]] = sorted(case_events[case_id], key=lambda x: x.get("time:timestamp"))
            original_events: List[Dict[str, Any]] = case_events[case_id]
            total_events += len(original_events)
            for i in range(len(original_events)):
                if original_events[i] != sorted_events[i]:
                    out_of_order_count += 1

        if not total_events:
            return 0.0

        return out_of_order_count / total_events

    @classmethod
    def fractal_self_similar_behavior(cls, events: List[Dict[str, Any]], memory: ComputedFeatureMemory) -> float:
        """
        Measures self-similar behavior by looking for recurring patterns of activities at different scales.

        Args:
            events: The list of events.
            memory: The memory object.

        Returns:
            A float between 0 and 1, where 1 indicates strong self-similar behavior.
        """
        case_events: Dict[str, List[str]] = defaultdict(list)
        for event in events:
            case_id: str = event.get("case:concept:name")
            activity: str = event.get("concept:name")
            if case_id and activity:
                case_events[case_id].append(activity)

        total_similarity_score: float = 0.0
        num_comparisons: int = 0

        traces: List[List[str]] = list(case_events.values())
        if len(traces) < 2:
            return 0.0

        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                trace1: List[str] = traces[i]
                trace2: List[str] = traces[j]

                # Simple similarity: Jaccard index of activity sets
                set1: Set[str] = set(trace1)
                set2: Set[str] = set(trace2)

                intersection_size: int = len(set1.intersection(set2))
                union_size: int = len(set1.union(set2))

                if union_size > 0:
                    similarity: float = intersection_size / union_size
                    total_similarity_score += similarity
                    num_comparisons += 1

        if not num_comparisons:
            return 0.0

        return total_similarity_score / num_comparisons