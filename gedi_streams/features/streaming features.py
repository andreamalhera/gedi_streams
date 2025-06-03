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



