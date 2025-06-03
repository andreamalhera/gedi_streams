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


class OptimizableProcessFeatures(StreamFeature):
    """
    Lifecycle-aware optimizable feature class designed for event streams where each activity
    generates both 'start' and 'complete' lifecycle events.

    This class properly handles the dual-event nature of process execution by:
    1. Using only 'complete' events for activity sequence reconstruction
    2. Calculating activity durations from start/complete pairs
    3. Avoiding double-counting of activities and transitions
    """

    def __init__(self, feature_names='optimizable_features', memory=None):
        """
        Initialize lifecycle-aware optimizable feature extractor.

        Features designed to correlate strongly with specific config parameters:
        - activity_count_ratio → 'mode' parameter
        - trace_count_measure → 'num_traces' parameter
        - parallel_execution_indicator → 'parallel' + 'concurrent_probability'
        - choice_branching_factor → 'choice' parameter
        - loop_repetition_ratio → 'loop' parameter
        - sequence_determinism → 'sequence' parameter
        - long_range_dependency_measure → 'lt_dependency' parameter
        - activity_duration_consistency → 'activity_duration' parameter
        """
        super().__init__(feature_names)
        self.feature_type = 'optimizable_features'
        self.available_class_methods = dict(inspect.getmembers(OptimizableProcessFeatures,
                                                               predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    def extract(self, events: List, memory: ComputedFeatureMemory):
        """Extract optimizable features from lifecycle-aware events."""
        feature_names = self.feature_names
        output = {}

        for feature_name in feature_names:
            if feature_name in self.available_class_methods:
                feature_fn = self.available_class_methods[feature_name]
                try:
                    feature_value = feature_fn(events, memory)
                    # Ensure value is in [0,1] range and valid
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
    def _filter_to_complete_events(cls, events: List) -> List:
        """
        Filter events to only include 'complete' lifecycle events.
        This gives us the actual activity execution sequence without duplication.
        """
        complete_events = []
        for event in events:
            lifecycle = event.get("lifecycle:transition", "complete").lower()
            if lifecycle == "complete":
                complete_events.append(event)
        return complete_events

    @classmethod
    def _get_activity_sequences(cls, events: List) -> Dict[str, List[str]]:
        """
        Extract activity sequences for each case using only complete events.
        Returns dict mapping case_id -> list of activities in execution order.
        """
        # Filter to complete events only
        complete_events = cls._filter_to_complete_events(events)

        # Group by case and sort by timestamp
        case_events = defaultdict(list)
        for event in complete_events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp and extract activity sequences
        case_sequences = {}
        for case_id, case_event_list in case_events.items():
            case_event_list.sort(key=lambda e: e.get("time:timestamp", ""))
            activities = []
            for event in case_event_list:
                activity = event.get("concept:name", "")
                if activity and activity not in ['<start>', '<end>', 'process_start', 'process_end']:
                    # Clean up self_loop suffixes for consistency
                    clean_activity = activity.replace("_self_loop", "")
                    activities.append(clean_activity)

            if activities:  # Only include non-empty sequences
                case_sequences[case_id] = activities

        return case_sequences

    @classmethod
    def _get_unique_activities(cls, events: List) -> Set[str]:
        """Get set of unique activities from complete events only."""
        complete_events = cls._filter_to_complete_events(events)
        activities = set()

        for event in complete_events:
            activity = event.get("concept:name", "")
            if activity and activity not in ['<start>', '<end>', 'process_start', 'process_end']:
                clean_activity = activity.replace("_self_loop", "")
                activities.add(clean_activity)

        return activities

    @classmethod
    def activity_count_ratio(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the ratio of unique activities to a normalized maximum.
        Uses only complete events to avoid double-counting activities.

        DIRECTLY OPTIMIZABLE with 'mode' parameter:
        - mode=5 should yield ~0.25 (5/20)
        - mode=10 should yield ~0.50 (10/20)
        - mode=20 should yield ~1.00 (20/20)

        Strong correlation expected: r > 0.8

        Returns:
            float: Normalized activity count (0-1)
        """
        unique_activities = cls._get_unique_activities(events)
        n_activities = len(unique_activities)

        # Normalize by maximum expected (mode parameter max is 20)
        normalized_ratio = min(n_activities / 20.0, 1.0)

        return normalized_ratio

    @classmethod
    def trace_count_measure(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the ratio of traces to a normalized maximum.
        Uses activity sequences to count actual process instances.

        DIRECTLY OPTIMIZABLE with 'num_traces' parameter:
        - num_traces=10 should yield ~0.10 (10/100)
        - num_traces=50 should yield ~0.50 (50/100)
        - num_traces=100 should yield ~1.00 (100/100)

        Strong correlation expected: r > 0.9

        Returns:
            float: Normalized trace count (0-1)
        """
        case_sequences = cls._get_activity_sequences(events)
        n_traces = len(case_sequences)

        # Normalize by maximum expected (num_traces parameter max is 101)
        normalized_ratio = min(n_traces / 101.0, 1.0)

        return normalized_ratio

    @classmethod
    def parallel_execution_indicator(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures actual parallel execution patterns by analyzing start/complete event overlaps.

        DIRECTLY OPTIMIZABLE with 'parallel' and 'concurrent_probability' parameters:
        - Higher parallel + concurrent_probability should yield higher values
        - Detects activities with overlapping execution periods

        Expected correlation: r > 0.6

        Returns:
            float: Parallel execution intensity (0-1)
        """
        if len(events) < 4:  # Need at least 2 complete activities (4 events)
            return 0.0

        # Group all events by case and sort by timestamp
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", ""))

        parallel_indicators = 0
        total_activity_pairs = 0

        for case_event_list in case_events.values():
            # Track activity execution periods (start to complete)
            activity_periods = {}

            for event in case_event_list:
                activity = event.get("concept:name", "")
                lifecycle = event.get("lifecycle:transition", "complete").lower()
                timestamp_str = event.get("time:timestamp", "")

                if not activity or not timestamp_str:
                    continue

                clean_activity = activity.replace("_self_loop", "")

                try:
                    timestamp = datetime.fromisoformat(timestamp_str)

                    if lifecycle == "start":
                        if clean_activity not in activity_periods:
                            activity_periods[clean_activity] = []
                        activity_periods[clean_activity].append({"start": timestamp, "complete": None})

                    elif lifecycle == "complete":
                        if clean_activity in activity_periods:
                            # Find the most recent uncompleted start for this activity
                            for period in reversed(activity_periods[clean_activity]):
                                if period["complete"] is None:
                                    period["complete"] = timestamp
                                    break

                except (ValueError, TypeError):
                    continue

            # Check for overlapping activity periods (parallelism)
            completed_periods = []
            for activity, periods in activity_periods.items():
                for period in periods:
                    if period.get("start") and period.get("complete"):
                        completed_periods.append((activity, period["start"], period["complete"]))

            # Compare all pairs of activity periods for overlap
            for i in range(len(completed_periods)):
                for j in range(i + 1, len(completed_periods)):
                    total_activity_pairs += 1

                    _, start1, end1 = completed_periods[i]
                    _, start2, end2 = completed_periods[j]

                    # Check if periods overlap
                    if start1 < end2 and start2 < end1:
                        parallel_indicators += 1

        if total_activity_pairs == 0:
            return 0.0

        parallel_ratio = parallel_indicators / total_activity_pairs
        return min(parallel_ratio * 2.0, 1.0)  # Scale up for better sensitivity

    @classmethod
    def choice_branching_factor(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the average number of different next activities per activity.
        Uses activity sequences from complete events to avoid double-counting.

        DIRECTLY OPTIMIZABLE with 'choice' parameter:
        - Higher choice probability should create more branching points
        - Measures variety in activity transitions

        Expected correlation: r > 0.7

        Returns:
            float: Average branching factor normalized (0-1)
        """
        case_sequences = cls._get_activity_sequences(events)

        if not case_sequences:
            return 0.0

        # Count possible next activities for each activity
        activity_followers = defaultdict(set)

        for case_sequence in case_sequences.values():
            for i in range(len(case_sequence) - 1):
                current_activity = case_sequence[i]
                next_activity = case_sequence[i + 1]

                if current_activity and next_activity:
                    activity_followers[current_activity].add(next_activity)

        if not activity_followers:
            return 0.0

        # Calculate average branching factor
        total_branches = sum(len(followers) for followers in activity_followers.values())
        avg_branches = total_branches / len(activity_followers)

        # Normalize by reasonable maximum (5 choices per activity is high branching)
        normalized_branching = min(avg_branches / 5.0, 1.0)

        return normalized_branching

    @classmethod
    def loop_repetition_ratio(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures how frequently activities repeat within traces.
        Uses activity sequences from complete events to avoid double-counting.

        DIRECTLY OPTIMIZABLE with 'loop' parameter:
        - Higher loop probability should create more repeated activities
        - Counts activities appearing multiple times in same trace

        Expected correlation: r > 0.8

        Returns:
            float: Ratio of repeated activities (0-1)
        """
        case_sequences = cls._get_activity_sequences(events)

        if not case_sequences:
            return 0.0

        total_activity_instances = 0
        repeated_activity_instances = 0

        for case_sequence in case_sequences.values():
            # Count activity occurrences in this case
            activity_counts = Counter(case_sequence)

            for activity, count in activity_counts.items():
                total_activity_instances += count
                if count > 1:
                    repeated_activity_instances += (count - 1)

        if total_activity_instances == 0:
            return 0.0

        repetition_ratio = repeated_activity_instances / total_activity_instances
        return min(repetition_ratio, 1.0)

    @classmethod
    def sequence_determinism(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures how deterministic the activity sequences are.
        Uses activity sequences from complete events.

        DIRECTLY OPTIMIZABLE with 'sequence' parameter:
        - Higher sequence probability should create more deterministic patterns
        - Low entropy in transitions indicates high determinism

        Expected correlation: r > 0.6

        Returns:
            float: Sequence determinism (0-1, where 1 = highly deterministic)
        """
        case_sequences = cls._get_activity_sequences(events)

        if not case_sequences:
            return 0.0

        # Count transitions for entropy calculation
        transition_counts = defaultdict(lambda: defaultdict(int))
        total_transitions = 0

        for case_sequence in case_sequences.values():
            for i in range(len(case_sequence) - 1):
                current = case_sequence[i]
                next_activity = case_sequence[i + 1]

                if current and next_activity:
                    transition_counts[current][next_activity] += 1
                    total_transitions += 1

        if total_transitions == 0:
            return 0.0

        # Calculate average determinism across all activities
        determinism_scores = []

        for current_activity, next_activities in transition_counts.items():
            total_from_current = sum(next_activities.values())

            if total_from_current <= 1:
                determinism_scores.append(1.0)  # Perfectly deterministic
                continue

            # Calculate entropy of transitions from this activity
            probabilities = [count / total_from_current for count in next_activities.values()]
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

            # Convert entropy to determinism (lower entropy = higher determinism)
            max_entropy = math.log2(len(next_activities))
            if max_entropy > 0:
                determinism = 1.0 - (entropy / max_entropy)
            else:
                determinism = 1.0

            # Weight by frequency of this activity
            weight = total_from_current / total_transitions
            determinism_scores.append(determinism * weight)

        return sum(determinism_scores) if determinism_scores else 0.0

    @classmethod
    def long_range_dependency_measure(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures long-range dependencies using activity sequences.
        Uses complete events to avoid double-counting.

        DIRECTLY OPTIMIZABLE with 'lt_dependency' parameter:
        - Counts activities that appear at positions separated by distance >= 2
        - More focused on actual parameter influence

        Expected correlation: r > 0.7

        Returns:
            float: Long-range dependency strength (0-1)
        """
        case_sequences = cls._get_activity_sequences(events)

        if not case_sequences:
            return 0.0

        total_long_range_pairs = 0
        total_possible_pairs = 0

        for case_sequence in case_sequences.values():
            # Look for same activities separated by distance >= 2
            for i in range(len(case_sequence)):
                for j in range(i + 2, len(case_sequence)):  # Skip adjacent (distance >= 2)
                    total_possible_pairs += 1

                    if case_sequence[i] and case_sequence[j] and case_sequence[i] == case_sequence[j]:
                        total_long_range_pairs += 1

        if total_possible_pairs == 0:
            return 0.0

        dependency_ratio = total_long_range_pairs / total_possible_pairs

        # Scale for better sensitivity (multiply by factor)
        return min(dependency_ratio * 3.0, 1.0)

    @classmethod
    def activity_duration_consistency(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures consistency of activity durations using start/complete event pairs.

        DIRECTLY OPTIMIZABLE with 'activity_duration' parameter:
        - Should reflect the configured duration settings
        - Lower variance indicates more consistent durations

        Expected correlation: Depends on implementation, but should be stable

        Returns:
            float: Duration consistency (0-1, where 1 = very consistent)
        """
        if len(events) < 2:
            return 1.0

        # Group events by case and activity to calculate durations
        case_activity_events = defaultdict(lambda: defaultdict(list))

        for event in events:
            case_id = event.get("case:concept:name", "")
            activity = event.get("concept:name", "")
            timestamp_str = event.get("time:timestamp", "")
            lifecycle = event.get("lifecycle:transition", "complete").lower()

            if case_id and activity and timestamp_str:
                clean_activity = activity.replace("_self_loop", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    case_activity_events[case_id][clean_activity].append({
                        'timestamp': timestamp,
                        'lifecycle': lifecycle
                    })
                except (ValueError, TypeError):
                    continue

        # Calculate durations for each activity instance
        durations = []

        for case_id, activities in case_activity_events.items():
            for activity, events_list in activities.items():
                # Sort events by timestamp
                events_list.sort(key=lambda x: x['timestamp'])

                # Match start/complete pairs
                starts = [e for e in events_list if e['lifecycle'] == 'start']
                completes = [e for e in events_list if e['lifecycle'] == 'complete']

                # Simple pairing: match each start with the next complete
                for start_event in starts:
                    # Find the next complete after this start
                    matching_completes = [e for e in completes if e['timestamp'] > start_event['timestamp']]
                    if matching_completes:
                        complete_event = min(matching_completes, key=lambda x: x['timestamp'])
                        duration = (complete_event['timestamp'] - start_event['timestamp']).total_seconds()
                        if duration > 0:
                            durations.append(duration)

        if len(durations) < 2:
            return 1.0

        # Calculate coefficient of variation (inverse consistency)
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        if mean_duration == 0:
            return 1.0

        cv = std_duration / mean_duration

        # Convert to consistency score (lower CV = higher consistency)
        # CV of 0.5 or higher is considered inconsistent
        consistency = max(0.0, 1.0 - min(cv / 0.5, 1.0))

        return consistency

    @classmethod
    def trace_length_uniformity(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures how uniform trace lengths are across the dataset.
        Uses activity sequences from complete events.

        OPTIMIZABLE with multiple parameters:
        - Influenced by 'loop', 'choice', and process structure
        - Uniform traces indicate structured process execution

        Expected correlation: Moderate with structural parameters

        Returns:
            float: Trace length uniformity (0-1, where 1 = very uniform)
        """
        case_sequences = cls._get_activity_sequences(events)

        if len(case_sequences) < 2:
            return 1.0

        # Calculate trace lengths based on activity sequences
        trace_lengths = [len(sequence) for sequence in case_sequences.values()]

        # Calculate coefficient of variation
        mean_length = np.mean(trace_lengths)
        std_length = np.std(trace_lengths)

        if mean_length == 0:
            return 1.0

        cv = std_length / mean_length

        # Convert to uniformity score (lower CV = higher uniformity)
        uniformity = max(0.0, 1.0 - min(cv, 1.0))

        return uniformity

    @classmethod
    def silent_activity_indicator(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Detects potential silent activities through timing gaps between activity completions.

        DIRECTLY OPTIMIZABLE with 'silent' parameter:
        - Silent activities create timing gaps in execution
        - Higher silent probability should create more gaps

        Expected correlation: r > 0.5

        Returns:
            float: Silent activity indicator (0-1)
        """
        case_sequences = cls._get_activity_sequences(events)

        if len(case_sequences) < 2:
            return 0.0

        # Get completion times for activities in each case
        case_completion_times = defaultdict(list)
        complete_events = cls._filter_to_complete_events(events)

        for event in complete_events:
            case_id = event.get("case:concept:name", "")
            timestamp_str = event.get("time:timestamp", "")

            if case_id and timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    case_completion_times[case_id].append(timestamp)
                except (ValueError, TypeError):
                    continue

        # Sort completion times within each case
        for case_id in case_completion_times:
            case_completion_times[case_id].sort()

        total_gaps = 0
        total_intervals = 0

        for case_id, completion_times in case_completion_times.items():
            if len(completion_times) < 2:
                continue

            # Calculate time intervals between consecutive completions
            time_diffs = []
            for i in range(len(completion_times) - 1):
                time_diff = (completion_times[i + 1] - completion_times[i]).total_seconds()
                if time_diff > 0:
                    time_diffs.append(time_diff)

            if len(time_diffs) < 2:
                continue

            # Detect unusually large gaps (potential silent activities)
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)

            if std_diff > 0:
                threshold = mean_diff + 2 * std_diff  # 2 standard deviations

                for diff in time_diffs:
                    total_intervals += 1
                    if diff > threshold:
                        total_gaps += 1

        if total_intervals == 0:
            return 0.0

        gap_ratio = total_gaps / total_intervals

        return min(gap_ratio * 2.0, 1.0)  # Scale for better sensitivity



class StructuredStreamFeature(StreamFeature):
    MEMORY_KEY_ABS_AVG_DIST_ACT = 'distinct_average_cases_absolut'
    INCREASE_SCALE_FACTOR = 10.0

    def __init__(self, feature_names='stream_features', memory=None):
        """
        Initialize the StreamFeature feature extractor.

        This class computes process mining features from event streams, focusing on
        metrics relevant for process analysis and optimization.

        Args:
            feature_names: List of feature names to compute, or 'stream_features'
                           to compute all available features
            memory: Optional memory object to store state between computations for incremental features
        """
        super().__init__(feature_names)
        self.feature_type = 'stream_features'
        self.available_class_methods = dict(inspect.getmembers(StructuredStreamFeature,
                                                               predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    def __getitem__(self, item: str) -> Callable[[List, ComputedFeatureMemory], float]:
        return self.available_class_methods[item]

    def extract(self, events: List, memory: ComputedFeatureMemory):
        feature_names = self.feature_names

        output = {}
        for feature_name in feature_names:
            feature_fn = self.available_class_methods[feature_name]
            feature_value = feature_fn(events, memory)
            output[f"{feature_name}"] = feature_value
        return output

    @classmethod
    def _calculate_absolute_avg_distinct_activities(cls, events: List) -> float:
        """
        Internal helper method to calculate the absolute average number of distinct activities.
        """
        # Group events by case ID
        case_activities: Dict[str, Set[str]] = defaultdict(set)

        for event in events:
            case_id = event.get("case:concept:name", "")
            activity_label = event.get("concept:name")

            if case_id and activity_label is not None:
                case_activities[case_id].add(activity_label)

        if not case_activities:
            return 0.0

        # Calculate distinct activities per case
        distinct_activities_counts = [len(activities) for activities in case_activities.values()]
        average_distinct_activities = sum(distinct_activities_counts) / len(distinct_activities_counts)
        return average_distinct_activities

    @classmethod
    def case_throughput_stability(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the consistency of case completion rates over time.

        This feature calculates a stability score by comparing the current
        case completion rate with historical rates. Higher values indicate
        more consistent throughput, which suggests a more predictable process
        flow and resource allocation.

        Returns:
            float: Throughput stability score (0-1)
        """
        # Count completed cases in this window
        completed_cases: int = cls._count_completed_cases(events)

        # Get historical completion rates
        completion_history: List[int] = memory.get_feature_value('completion_history')
        if completion_history is None:
            completion_history = []

        # If we have history, calculate stability
        stability: float = 1.0
        if completion_history:
            # Calculate mean and standard deviation of historical rates
            mean_rate: float = np.mean(completion_history)
            std_rate: float = np.std(completion_history) if len(completion_history) > 1 else 1.0

            # Calculate z-score of current rate
            z_score: float = abs(completed_cases - mean_rate) / max(std_rate, 1)

            # Convert to stability score (1 = perfectly stable, declining with higher z-scores)
            stability = math.exp(-z_score)

        # Update history (limited to last 10 windows)
        completion_history.append(completed_cases)
        if len(completion_history) > 10:
            completion_history = completion_history[-10:]

        memory.set_feature_value('completion_history', completion_history)

        return stability

    @classmethod
    def activity_duration_stability(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the consistency of activity execution times.

        This feature analyzes how consistently activities take to execute across
        different cases. Higher values indicate more predictable activity durations,
        suggesting a more standardized process with reliable execution times.

        Returns:
            float: Activity duration stability score (0-1)
        """
        # Calculate activity durations across cases
        activity_durations: Dict[str, List[float]] = defaultdict(list)

        # Keep track of start times for each activity per case
        case_activity_starts: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for event in events:
            case_id = event.get("case:concept:name", "")
            activity: str = event.get("concept:name", "")
            timestamp = event.get("time:timestamp", None)
            lifecycle: str = event.get("lifecycle:transition", "").lower()

            if not case_id or not activity or timestamp is None:
                continue

            if lifecycle == "start":
                # Record start time
                case_activity_starts[case_id][activity] = timestamp
            elif lifecycle == "complete":
                # Calculate duration if we have a start time
                if activity in case_activity_starts[case_id]:
                    start_time = case_activity_starts[case_id][activity]
                    try:
                        # Try to calculate duration
                        if isinstance(timestamp, (int, float)) and isinstance(start_time, (int, float)):
                            duration = float(timestamp - start_time)
                        elif isinstance(timestamp, datetime.datetime) and isinstance(start_time, datetime.datetime):
                            duration = float((timestamp - start_time).total_seconds())
                        else:
                            # Skip if types don't match or aren't supported
                            continue

                        activity_durations[activity].append(duration)
                    except (TypeError, ValueError):
                        # Skip on any calculation error
                        continue

        # Calculate coefficient of variation for each activity
        stability_scores: List[float] = []

        for activity, durations in activity_durations.items():
            if len(durations) < 2:
                continue

            mean_duration: float = float(np.mean(durations))
            std_duration: float = float(np.std(durations))

            # Coefficient of variation (lower is more stable)
            cv: float = std_duration / mean_duration if mean_duration > 0 else float('inf')

            # Convert to stability score (1 = perfectly stable, declining with higher CV)
            stability: float = math.exp(-cv)
            stability_scores.append(stability)

        # Return average stability across activities
        return float(np.mean(stability_scores)) if stability_scores else 1.0

    @classmethod
    def case_priority_dynamics(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the entropy of case priority changes.

        This feature analyzes how case priorities change over time within the events.
        Higher values indicate more unpredictable/chaotic priority changes, which can
        suggest a less structured prioritization system or frequent reprioritization.

        Note: Requires 'priority' attribute in events

        Returns:
            float: Case priority dynamics score (0-1)
        """
        # Group events by case ID and sort by timestamp
        case_events: Dict[str, List] = defaultdict(list)

        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        # Track priority changes for each case
        case_priority_changes: Dict[str, List[Tuple[Any, Any]]] = defaultdict(list)

        for case_id, case_event_list in case_events.items():
            # Extract priorities from events
            priorities: List[Any] = []
            for event in case_event_list:
                priority = event.get("priority", None)
                if priority is not None:
                    priorities.append(priority)

            # Record priority changes
            for i in range(1, len(priorities)):
                if priorities[i] != priorities[i - 1]:
                    case_priority_changes[case_id].append((priorities[i - 1], priorities[i]))

        # Count frequency of different priority change patterns
        priority_change_counts: Counter = Counter()
        total_changes: int = 0

        for case_id, changes in case_priority_changes.items():
            for change in changes:
                priority_change_counts[change] += 1
                total_changes += 1

        # Calculate entropy of priority changes
        dynamics_score: float = 0.0
        if total_changes > 0 and len(priority_change_counts) > 0:
            probabilities: List[float] = [count / total_changes for count in priority_change_counts.values()]
            max_entropy: float = math.log(len(priority_change_counts)) if len(priority_change_counts) > 1 else 0.0
            if max_entropy > 0:
                entropy_val: float = entropy(probabilities)
                dynamics_score = entropy_val / max_entropy

        return dynamics_score


    @classmethod
    def _count_completed_cases(cls, events: List) -> int:
        """
        Helper method to count completed cases from events.

        Returns:
            int: Number of completed cases
        """
        # Group events by case ID and find completion indicators
        case_events: Dict[str, List] = defaultdict(list)

        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp to find last event
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        completed_count: int = 0

        for case_event_list in case_events.values():
            if case_event_list:
                # Check last event for completion marker
                last_event = case_event_list[-1]
                event_name: str = last_event.get("concept:name", "").lower()
                lifecycle: str = last_event.get("lifecycle:transition", "").lower()

                if "complete" in event_name or "end" in event_name or \
                        lifecycle == "complete" or lifecycle == "terminated":
                    completed_count += 1

        return completed_count

    @classmethod
    def _get_activity_distribution(cls, events: List) -> Dict[str, float]:
        """
        Helper method to get activity distribution from events.

        Returns:
            Dict[str, float]: Mapping of activity names to their relative frequencies
        """
        activity_counts: Dict[str, int] = {}
        total: int = 0

        for event in events:
            activity: str = event.get("concept:name", "")
            if activity:
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
                total += 1

        # Normalize to get probabilities
        if total > 0:
            return {k: v / total for k, v in activity_counts.items()}
        else:
            return {}

    @classmethod
    def _compare_distributions(cls, dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
        """
        Compare two probability distributions.

        Returns:
            float: A similarity score (1 = identical, 0 = completely different)
        """
        # Get all keys
        all_keys: Set[str] = set(dist1) | set(dist2)

        if not all_keys:
            return 1.0  # Empty distributions are considered identical

        # Create arrays with zeros for missing keys
        p: np.ndarray = np.array([dist1.get(k, 0) for k in all_keys])
        q: np.ndarray = np.array([dist2.get(k, 0) for k in all_keys])

        # Ensure non-zero probabilities (avoid log(0))
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate JS divergence
        m: np.ndarray = 0.5 * (p + q)

        # Since we're comparing distributions, entropy should not fail,
        # but just in case, let's wrap it in a try-except
        try:
            js_div: float = 0.5 * (entropy(p, m) + entropy(q, m))
            # Convert to a similarity score (1 - normalized_divergence)
            return 1.0 - min(1.0, js_div)
        except (ValueError, ZeroDivisionError):
            # If entropy calculation fails, fall back to a simpler metric: L1 distance
            l1_dist: float = np.sum(np.abs(p - q)) / 2  # Normalized L1 (between 0-1)
            return 1.0 - l1_dist

    @classmethod
    def _get_direct_follow_relations(cls, events: List) -> Set[Tuple[str, str]]:
        """
        Helper method to extract direct-follow relations from events.

        Returns:
            Set[Tuple[str, str]]: Set of direct-follow relations as (activity1, activity2) pairs
        """
        # Group events by case ID and sort by timestamp
        case_events: Dict[str, List] = defaultdict(list)

        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        dfr: Set[Tuple[str, str]] = set()

        for case_event_list in case_events.values():
            activities: List[str] = [event.get("concept:name", "") for event in case_event_list]

            for i in range(len(activities) - 1):
                if activities[i] and activities[i + 1]:  # Check for non-empty
                    dfr.add((activities[i], activities[i + 1]))

        return dfr

    @classmethod
    def _compare_relation_sets(cls, set1: Set[Tuple[str, str]], set2: Set[Tuple[str, str]]) -> float:
        """
        Compare two sets of relations using Jaccard similarity.

        Returns:
            float: Similarity score (1 = identical, 0 = no overlap)
        """
        if not set1 and not set2:
            return 1.0  # Both empty = identical

        intersection: int = len(set1.intersection(set2))
        union: int = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    @classmethod
    def _get_variant_distribution(cls, events: List) -> Dict[Tuple[str, ...], float]:
        """
        Helper method to get process variant distribution from events.

        Returns:
            Dict[Tuple[str, ...], float]: Mapping of variants to their relative frequencies
        """
        # Group events by case ID and sort by timestamp
        case_events: Dict[str, List] = defaultdict(list)

        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        variant_counts: Dict[Tuple[str, ...], int] = {}

        for case_event_list in case_events.values():
            variant: Tuple[str, ...] = tuple(event.get("concept:name", "") for event in case_event_list)
            variant_counts[variant] = variant_counts.get(variant, 0) + 1

        # Normalize to get probabilities
        total: int = len(case_events)
        if total > 0:
            return {k: v / total for k, v in variant_counts.items()}
        else:
            return {}

    @classmethod
    def _calculate_correlation_at_distance(cls, trace_activities: List[List[str]], distance: int) -> float:
        """
        Calculate correlation between activities separated by specified distance.

        Args:
            trace_activities: List of activity sequences from traces
            distance: Distance between activities to analyze

        Returns:
            float: Correlation score between 0-1
        """
        activity_pairs: List[Tuple[str, str]] = []

        for trace in trace_activities:
            for i in range(len(trace) - distance):
                activity_pairs.append((trace[i], trace[i + distance]))

        if not activity_pairs:
            return 0.0

        # Count co-occurrences
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        act1_counts: Dict[str, int] = defaultdict(int)
        act2_counts: Dict[str, int] = defaultdict(int)
        total_pairs: int = len(activity_pairs)

        for a1, a2 in activity_pairs:
            pair_counts[(a1, a2)] += 1
            act1_counts[a1] += 1
            act2_counts[a2] += 1

        # Calculate correlation coefficient
        correlation_sum: float = 0.0
        norm_factor: float = 0.0

        for (a1, a2), count in pair_counts.items():
            # Expected probability under independence
            expected_prob: float = (act1_counts[a1] / total_pairs) * (act2_counts[a2] / total_pairs)
            # Actual probability
            actual_prob: float = count / total_pairs
            # Contribution to correlation
            correlation_sum += (actual_prob - expected_prob) ** 2
            norm_factor += expected_prob

        # Normalize correlation
        return correlation_sum / norm_factor if norm_factor > 0 else 0.0


    @classmethod
    def process_complexity(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures structural complexity based on activity count and connections.

        OPTIMIZABLE: Influenced by 'mode', 'sequence', 'choice', 'parallel' parameters.

        Returns: 0-1, normalized complexity score
        """
        # Count unique activities
        activities = set()
        for event in events:
            activity = event.get("concept:name", "")
            if activity:
                activities.add(activity)

        n_activities = len(activities)

        # Count direct-follow relations
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        df_relations = set()
        for case_event_list in case_events.values():
            for i in range(len(case_event_list) - 1):
                a = case_event_list[i].get("concept:name", "")
                b = case_event_list[i + 1].get("concept:name", "")
                if a and b:
                    df_relations.add((a, b))

        n_relations = len(df_relations)

        # Normalize complexity: combines activity count and relation density
        max_possible_relations = n_activities * n_activities if n_activities > 0 else 1
        relation_density = n_relations / max_possible_relations

        # Combine activity count (normalized by max possible 20) and relation density
        activity_complexity = min(n_activities / 20.0, 1.0)  # Max 20 from config

        return (activity_complexity + relation_density) / 2

    @classmethod
    def choice_diversity(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures how many different choices are available at each decision point.

        OPTIMIZABLE: Influenced by 'choice' parameter.

        Returns: 0-1, where 1 means maximum choice diversity
        """
        # Group events by case and sort by timestamp
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        # Count choices available after each activity
        activity_followers = defaultdict(set)

        for case_event_list in case_events.values():
            for i in range(len(case_event_list) - 1):
                current = case_event_list[i].get("concept:name", "")
                next_act = case_event_list[i + 1].get("concept:name", "")
                if current and next_act:
                    activity_followers[current].add(next_act)

        if not activity_followers:
            return 0.0

        # Calculate average branching factor
        total_choices = sum(len(followers) for followers in activity_followers.values())
        avg_choices = total_choices / len(activity_followers)

        # Normalize (assume max 5 choices per activity is high diversity)
        return min(avg_choices / 5.0, 1.0)

    @classmethod
    def loop_intensity(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures how frequently activities repeat within traces.

        OPTIMIZABLE: Influenced by 'loop' parameter.

        Returns: 0-1, where 1 means high loop activity
        """
        # Group events by case
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        total_loops = 0
        total_activities = 0

        for case_event_list in case_events.values():
            activities = [event.get("concept:name", "") for event in case_event_list]
            activity_counts = Counter(activities)

            # Count loops (activities appearing more than once)
            for activity, count in activity_counts.items():
                if activity and count > 1:
                    total_loops += count - 1  # Extra occurrences beyond first
                total_activities += count

        return total_loops / max(total_activities, 1)

    @classmethod
    def trace_diversity(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures diversity of execution paths (variant uniqueness).

        OPTIMIZABLE: Influenced by 'choice', 'parallel', 'loop', 'num_traces' parameters.

        Returns: 0-1, where 1 means every trace is unique
        """
        # Group events by case and sort by timestamp
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        # Create variants (unique execution paths)
        variants = set()
        for case_event_list in case_events.values():
            variant = tuple(event.get("concept:name", "") for event in case_event_list)
            variants.add(variant)

        total_traces = len(case_events)
        unique_variants = len(variants)

        return unique_variants / max(total_traces, 1)

    @classmethod
    def sequence_predictability(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures how predictable the next activity is given the current one.

        OPTIMIZABLE: Influenced by 'sequence', 'choice' parameters.

        Returns: 0-1, where 1 means completely predictable sequences
        """
        # Group events by case and sort by timestamp
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))
        total_transitions = 0

        for case_event_list in case_events.values():
            for i in range(len(case_event_list) - 1):
                current = case_event_list[i].get("concept:name", "")
                next_act = case_event_list[i + 1].get("concept:name", "")
                if current and next_act:
                    transitions[current][next_act] += 1
                    total_transitions += 1

        if total_transitions == 0:
            return 0.0

        # Calculate predictability (1 - entropy)
        total_entropy = 0.0
        for current_activity, next_activities in transitions.items():
            total_from_current = sum(next_activities.values())
            if total_from_current > 1:
                # Calculate entropy for this activity's transitions
                probabilities = [count / total_from_current for count in next_activities.values()]
                entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
                max_entropy = math.log2(len(next_activities))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                total_entropy += normalized_entropy * (total_from_current / total_transitions)

        return 1.0 - total_entropy  # Convert entropy to predictability

    @classmethod
    def activity_frequency_balance(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures how evenly distributed activity frequencies are.

        OPTIMIZABLE: Influenced by 'mode', 'choice', 'sequence' parameters.

        Returns: 0-1, where 1 means perfectly balanced activity frequencies
        """
        # Count activity frequencies
        activity_counts = Counter()
        for event in events:
            activity = event.get("concept:name", "")
            if activity:
                activity_counts[activity] += 1

        if len(activity_counts) <= 1:
            return 1.0  # Perfect balance with 0 or 1 activities

        # Calculate coefficient of variation (lower = more balanced)
        frequencies = list(activity_counts.values())
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)

        if mean_freq == 0:
            return 1.0

        cv = std_freq / mean_freq

        # Convert CV to balance score (1 = perfect balance, 0 = very imbalanced)
        # Assume CV > 2 is very imbalanced
        return max(0.0, 1.0 - min(cv / 2.0, 1.0))

    @classmethod
    def trace_length_consistency(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures consistency of trace lengths.

        OPTIMIZABLE: Influenced by 'loop', 'choice', 'parallel' parameters.

        Returns: 0-1, where 1 means very consistent trace lengths
        """
        # Group events by case
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        if len(case_events) <= 1:
            return 1.0  # Perfect consistency with 0 or 1 traces

        # Calculate trace lengths
        trace_lengths = [len(events_list) for events_list in case_events.values()]

        mean_length = np.mean(trace_lengths)
        std_length = np.std(trace_lengths)

        if mean_length == 0:
            return 1.0

        # Coefficient of variation
        cv = std_length / mean_length

        # Convert to consistency score (1 = perfect consistency, 0 = very inconsistent)
        return max(0.0, 1.0 - min(cv, 1.0))

    @classmethod
    def concurrent_execution_rate(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the rate of concurrent activity execution based on timestamp clustering.

        OPTIMIZABLE: Influenced by 'concurrent_probability', 'parallel' parameters.

        Returns: 0-1, where 1 means high concurrency
        """
        if len(events) < 2:
            return 0.0

        # Sort all events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get("time:timestamp", 0))

        # Count events that happen within a small time window (concurrent)
        concurrent_groups = 0
        total_events = len(sorted_events)
        time_window = 5  # Events within 5 time units are considered concurrent

        i = 0
        while i < len(sorted_events):
            current_time = sorted_events[i].get("time:timestamp", 0)
            group_size = 1

            # Count events in the same time window
            j = i + 1
            while j < len(sorted_events):
                event_time = sorted_events[j].get("time:timestamp", 0)

                if (datetime.fromisoformat(event_time) - datetime.fromisoformat(current_time)).total_seconds() <= time_window:
                    group_size += 1
                    j += 1
                else:
                    break

            if group_size > 1:
                concurrent_groups += group_size

            i = j if j > i + 1 else i + 1

        return concurrent_groups / total_events

    @classmethod
    def resource_utilization_pattern(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures consistency of resource usage over time.

        OPTIMIZABLE: Influenced by 'activity_duration', 'concurrent_probability' parameters.

        Returns: 0-1, where 1 means consistent resource utilization
        """
        if len(events) < 2:
            return 1.0

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get("time:timestamp", 0))

        # Create time windows and count activity load
        time_windows = {}
        window_size = 10  # 10-minute windows

        for event in sorted_events:
            timestamp = event.get("time:timestamp", 0)
            date = datetime.fromisoformat(timestamp)

            # Calculate total minutes since epoch
            epoch_minutes = date.timestamp() / 60
            window = int(epoch_minutes // window_size)

            if window not in time_windows:
                time_windows[window] = 0
            time_windows[window] += 1

        if len(time_windows) <= 1:
            return 1.0

        # Calculate consistency of load across time windows
        loads = list(time_windows.values())
        mean_load = np.mean(loads)
        std_load = np.std(loads)

        if mean_load == 0:
            return 1.0

        cv = std_load / mean_load

        # Convert to utilization consistency (1 = consistent, 0 = very variable)
        return max(0.0, 1.0 - min(cv, 1.0))

    # =================== SIMPLE VOLUME METRICS ===================

    @classmethod
    def parallel_activity_ratio(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures actual parallel execution based on overlapping activities.

        OPTIMIZABLE: Influenced by 'parallel' and 'concurrent_probability' parameters.

        Returns: 0-1, where 1 means maximum parallelism
        """
        # Group events by case and sort by timestamp
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        total_overlaps = 0
        total_possible_overlaps = 0

        for case_event_list in case_events.values():
            # Find overlapping activities (concurrent execution)
            for i in range(len(case_event_list)):
                for j in range(i + 1, len(case_event_list)):
                    event1 = case_event_list[i]
                    event2 = case_event_list[j]

                    time1: str = event1.get("time:timestamp", 0)
                    time2: str = event2.get("time:timestamp", 0)



                    # Check if events overlap in time (concurrent)
                    if abs(datetime.fromisoformat(time1) - datetime.fromisoformat(time2)).total_seconds() < 10:  # Within 10 time units = overlapping
                        total_overlaps += 1

                    total_possible_overlaps += 1

        return total_overlaps / max(total_possible_overlaps, 1)

    @classmethod
    def activity_count_ratio_1(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the ratio of unique activities to total events.

        OPTIMIZABLE: Influenced by 'mode', 'num_traces' parameters.

        Returns: 0-1, where higher values mean more activity diversity per event
        """
        if not events:
            return 0.0

        unique_activities = set()
        for event in events:
            activity = event.get("concept:name", "")
            if activity:
                unique_activities.add(activity)

        return len(unique_activities) / len(events)

    @classmethod
    def case_activity_ratio(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the ratio of cases to unique activities.

        OPTIMIZABLE: Influenced by 'num_traces', 'mode' parameters.

        Returns: 0-1, normalized ratio
        """
        case_ids = set()
        activities = set()

        for event in events:
            case_id = event.get("case:concept:name", "")
            activity = event.get("concept:name", "")

            if case_id:
                case_ids.add(case_id)
            if activity:
                activities.add(activity)

        if len(activities) == 0:
            return 0.0

        ratio = len(case_ids) / len(activities)

        # Normalize (assume ratio of 10 cases per activity is high)
        return min(ratio / 10.0, 1.0)

    @classmethod
    def long_term_dependency_strength(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures strength of long-term dependencies in the process.

        OPTIMIZABLE: Influenced by 'lt_dependency' parameter.

        Returns: 0-1, where 1 means strong long-term dependencies
        """
        # Group events by case and sort by timestamp
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        total_long_deps = 0
        total_possible_deps = 0

        for case_event_list in case_events.values():
            activities = [event.get("concept:name", "") for event in case_event_list]

            # Look for dependencies with distance > 1
            for i in range(len(activities)):
                for j in range(i + 2, len(activities)):  # Skip adjacent activities
                    if activities[i] and activities[j]:
                        total_possible_deps += 1
                        # Check if this pattern repeats (indicating dependency)
                        pattern = (activities[i], activities[j])

                        # Count occurrences of this pattern across all traces
                        pattern_count = 0
                        for other_case in case_events.values():
                            other_activities = [e.get("concept:name", "") for e in other_case]
                            for k in range(len(other_activities)):
                                for l in range(k + 2, len(other_activities)):
                                    if (other_activities[k], other_activities[l]) == pattern:
                                        pattern_count += 1

                        if pattern_count > 1:  # Pattern repeats = dependency
                            total_long_deps += 1

        return total_long_deps / max(total_possible_deps, 1)

    @classmethod
    def temporal_dependency(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the consistency of control flow sequences (sequence predictability).

        This feature analyzes how predictable the next activity is based on the current activity,
        regardless of timing. Higher values indicate more consistent/predictable control flow patterns.

        Returns:
            float: Sequence predictability score (0-1)
        """
        # Group events by case ID to reconstruct traces
        case_events: Dict[str, List] = defaultdict(list)

        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        # Extract direct-follow relations and their frequencies
        activity_transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_transitions = 0

        for case_event_list in case_events.values():
            activities = [event.get("concept:name", "") for event in case_event_list if
                          event.get("concept:name", "")]

            # Count direct-follow relations
            for i in range(len(activities) - 1):
                current_act = activities[i]
                next_act = activities[i + 1]

                activity_transitions[current_act][next_act] += 1
                total_transitions += 1

        if total_transitions == 0:
            return 0.0

        # Calculate predictability for each activity
        predictability_scores = []

        for current_act, next_acts in activity_transitions.items():
            total_from_current = sum(next_acts.values())

            # Calculate entropy of next activities
            probabilities = [count / total_from_current for count in next_acts.values()]
            entropy_val = entropy(probabilities)

            # Convert entropy to predictability (lower entropy = higher predictability)
            max_entropy = math.log(len(next_acts)) if len(next_acts) > 1 else 0.0
            if max_entropy > 0:
                predictability = 1.0 - (entropy_val / max_entropy)
            else:
                predictability = 1.0  # Only one choice = perfectly predictable

            # Weight by frequency of this activity
            weight = total_from_current / total_transitions
            predictability_scores.append(predictability * weight)

        # Overall temporal dependency is weighted average predictability
        return sum(predictability_scores) if predictability_scores else 0.0

    @classmethod
    def long_term_dependencies(cls, events: List, memory: ComputedFeatureMemory) -> float:
        """
        Measures the span between lifecycle events (start/complete patterns).

        This feature analyzes how much activity happens between the start and completion
        of activities. Higher values indicate more complex lifecycle patterns with
        many intervening activities.

        Returns:
            float: Lifecycle span score (0-1)
        """
        # Group events by case ID and sort by timestamp
        case_events: Dict[str, List] = defaultdict(list)

        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        lifecycle_spans = []

        for case_event_list in case_events.values():
            # Group events by activity name and lifecycle state
            activity_events = defaultdict(list)

            for i, event in enumerate(case_event_list):
                activity_name = event.get("concept:name", "")
                lifecycle = event.get("lifecycle:transition", "complete").lower()

                if activity_name:
                    activity_events[activity_name].append((i, lifecycle))

            # Calculate spans for activities with both start and complete
            for activity, events_list in activity_events.items():
                starts = [pos for pos, lifecycle in events_list if lifecycle == "start"]
                completes = [pos for pos, lifecycle in events_list if lifecycle == "complete"]

                # Match starts with completes (simple pairing)
                for start_pos in starts:
                    # Find the next complete after this start
                    matching_completes = [pos for pos in completes if pos > start_pos]
                    if matching_completes:
                        complete_pos = min(matching_completes)
                        span = complete_pos - start_pos

                        # Normalize by trace length
                        normalized_span = span / len(case_event_list) if len(case_event_list) > 0 else 0.0
                        lifecycle_spans.append(normalized_span)

        if not lifecycle_spans:
            # If no lifecycle events found, analyze activity repetition patterns
            # as a proxy for long-term dependencies
            return cls._analyze_activity_repetition_spans(events)

        # Return average normalized span
        return np.mean(lifecycle_spans) if lifecycle_spans else 0.0

    @classmethod
    def _analyze_activity_repetition_spans(cls, events: List) -> float:
        """
        Fallback method when no lifecycle events are present.
        Analyzes spans between repeated activities as a proxy for long-term dependencies.
        """
        # Group events by case ID and sort by timestamp
        case_events: Dict[str, List] = defaultdict(list)

        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        repetition_spans = []

        for case_event_list in case_events.values():
            activities = [event.get("concept:name", "") for event in case_event_list if
                          event.get("concept:name", "")]

            # Find spans between repeated activities
            activity_positions = defaultdict(list)
            for i, activity in enumerate(activities):
                activity_positions[activity].append(i)

            # Calculate spans for activities that appear multiple times
            for activity, positions in activity_positions.items():
                if len(positions) > 1:
                    # Calculate spans between first and last occurrence
                    span = positions[-1] - positions[0]
                    normalized_span = span / len(activities) if len(activities) > 0 else 0.0
                    repetition_spans.append(normalized_span)

        return np.mean(repetition_spans) if repetition_spans else 0.0

    @classmethod
    def unique_paths_ratio(cls, events: list, memory: ComputedFeatureMemory):
        """
        Measures the diversity of execution paths as the ratio of unique variants to total traces.

        This feature calculates how many different execution paths (variants) exist relative
        to the total number of traces, indicating process variability.

        Interpretation: Values close to 0 indicate a standardized process with few variants.
        Values approaching 1 indicate high variability where almost every case follows a
        unique path. Higher values may suggest an unstructured or ad-hoc process.

        Returns:
            float: Ratio between 0-1 of unique execution paths to total traces
        """
        # Group events by case and sort by timestamp
        case_events = defaultdict(list)
        for event in events:
            case_id = event.get("case:concept:name", "")
            if case_id:
                case_events[case_id].append(event)

        if not case_events:
            return 0.0

        # Sort events within each case by timestamp
        for case_id in case_events:
            case_events[case_id].sort(key=lambda e: e.get("time:timestamp", 0))

        unique_paths = set()
        for case_event_list in case_events.values():
            path = tuple(event.get("concept:name", "") for event in case_event_list)
            unique_paths.add(path)

        return len(unique_paths) / len(case_events)

# Additional utility function to validate feature ranges
def validate_feature_ranges(features: Dict[str, float]) -> Dict[str, float]:
    """
    Validates that all features are in expected 0-1 range and fixes obvious errors.
    """
    validated = {}

    for feature_name, value in features.items():
        if math.isnan(value) or math.isinf(value):
            validated[feature_name] = 0.0
        elif value < 0:
            validated[feature_name] = 0.0
        elif value > 1:
            # For features that should be 0-1, cap at 1
            if any(keyword in feature_name.lower() for keyword in
                   ['ratio', 'rate', 'probability', 'consistency', 'balance', 'predictability']):
                validated[feature_name] = 1.0
            else:
                # For other features, normalize by dividing by a reasonable maximum
                if 'entropy' in feature_name.lower():
                    validated[feature_name] = min(value / 5.0, 1.0)  # Max entropy ~5
                elif 'complexity' in feature_name.lower():
                    validated[feature_name] = min(value / 10.0, 1.0)  # Max complexity ~10
                else:
                    validated[feature_name] = min(value, 1.0)
        else:
            validated[feature_name] = value

    return validated