import inspect
import math
import numpy as np
import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any

from feeed.feature import Feature
from scipy.stats import entropy
from gedi_streams.features.memory import ComputedFeatureMemory
from pm4py.objects.log.obj import EventLog

class StreamFeature(Feature):
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
        self.available_class_methods = dict(inspect.getmembers(StreamFeature,
                                                               predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    def extract(self, log: EventLog, memory: ComputedFeatureMemory):
        feature_names = self.feature_names

        output = {}
        for feature_name in feature_names:
            feature_fn = self.available_class_methods[feature_name]
            feature_value = feature_fn(log, memory)
            output[f"{feature_name}"] = feature_value
        return output

    @classmethod
    def case_concurrency(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the proportion of concurrent cases in the window.

        Concurrent cases are those with overlapping execution periods. This feature
        calculates a normalized measure of how many cases execute in parallel during
        the window timeframe. Higher values indicate more parallel case execution,
        which can suggest resource contention or a more interleaved process.

        Returns:
            float: Proportion of concurrent cases (0-1)
        """
        concurrent_pairs: int = 0
        total_pairs: int = 0

        # Get timeframe for each case (min and max timestamp)
        case_timeframes: Dict[str, Tuple[Any, Any]] = {}

        for trace in window:
            case_id: str = trace.attributes.get("concept:name", "")
            if not case_id:
                continue

            timestamps: List[Any] = []
            for event in trace:
                ts = event.get("time:timestamp", None)
                if ts:
                    timestamps.append(ts)

            if timestamps:
                case_timeframes[case_id] = (min(timestamps), max(timestamps))

        # Check each pair of cases for overlap
        case_ids: List[str] = list(case_timeframes.keys())
        for i in range(len(case_ids)):
            for j in range(i + 1, len(case_ids)):
                total_pairs += 1

                # Cases i and j
                start_i, end_i = case_timeframes[case_ids[i]]
                start_j, end_j = case_timeframes[case_ids[j]]

                # Check for overlap
                if (start_i <= end_j and start_j <= end_i):
                    concurrent_pairs += 1

        # Return concurrency ratio
        return concurrent_pairs / total_pairs if total_pairs > 0 else 0.0

    @classmethod
    def concept_stability(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the stability of process behavior compared to previous windows.

        This feature compares activity distributions and control-flow patterns
        between the current window and previous windows. Higher values indicate
        more stable process behavior, suggesting consistent execution patterns.

        Returns:
            float: Stability score (0-1)
        """
        # Create activity distribution for current window
        current_activity_dist: Dict[str, float] = cls._get_activity_distribution(window)

        # Get previous activity distribution from memory
        prev_activity_dist: Optional[Dict[str, float]] = memory.get_feature_value('prev_activity_dist')

        if prev_activity_dist is None:
            # First window, no comparison possible
            memory.set_feature_value('prev_activity_dist', current_activity_dist)
            return 1.0  # Assume fully stable initially

        # Compare distributions (similarity score)
        stability: float = cls._compare_distributions(current_activity_dist, prev_activity_dist)

        # Update memory with current distribution (with some smoothing)
        alpha: float = 0.2  # Smoothing factor
        updated_dist: Dict[str, float] = {
            k: alpha * current_activity_dist.get(k, 0) + (1 - alpha) * prev_activity_dist.get(k, 0)
            for k in set(current_activity_dist) | set(prev_activity_dist)
        }

        memory.set_feature_value('prev_activity_dist', updated_dist)

        return stability

    @classmethod
    def case_throughput_stability(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
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
        completed_cases: int = cls._count_completed_cases(window)

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
    def parallel_activity_ratio(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the degree of parallel activity execution within processes.

        This feature analyzes the control flow to identify potentially parallel activities
        (those that can occur in different orders across traces). Higher values indicate
        more parallel activity execution patterns in the process, suggesting a more
        flexible or concurrent workflow.

        Returns:
            float: Parallel activity ratio (0-1)
        """
        # Extract direct-follow relations
        df_relations: Set[Tuple[str, str]] = set()
        all_activities: Set[str] = set()

        for trace in window:
            activities_in_trace: List[str] = [event.get("concept:name", "") for event in trace]

            # Track all activities
            all_activities.update([a for a in activities_in_trace if a])

            # Extract direct-follow relations
            for i in range(len(activities_in_trace) - 1):
                a: str = activities_in_trace[i]
                b: str = activities_in_trace[i + 1]
                if a and b:  # Ensure non-empty activity names
                    df_relations.add((a, b))

        # Count potential parallel activities (a>b and b>a both exist)
        parallel_pairs: int = 0
        total_pairs: int = 0

        for a in all_activities:
            for b in all_activities:
                if a < b:  # Check each pair once
                    total_pairs += 1
                    if (a, b) in df_relations and (b, a) in df_relations:
                        parallel_pairs += 1

        # Return ratio of parallel pairs to total pairs
        return parallel_pairs / total_pairs if total_pairs > 0 else 0.0

    @classmethod
    def activity_duration_stability(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
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

        for trace in window:
            # Keep track of start times for each activity
            activity_starts: Dict[str, Any] = {}

            for event in trace:
                activity: str = event.get("concept:name", "")
                timestamp = event.get("time:timestamp", None)
                lifecycle: str = event.get("lifecycle:transition", "").lower()

                if not activity or timestamp is None:
                    continue

                if lifecycle == "start":
                    # Record start time
                    activity_starts[activity] = timestamp
                elif lifecycle == "complete":
                    # Calculate duration if we have a start time
                    if activity in activity_starts:
                        start_time = activity_starts[activity]
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
    def case_priority_dynamics(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the entropy of case priority changes.

        This feature analyzes how case priorities change over time within the window.
        Higher values indicate more unpredictable/chaotic priority changes, which can
        suggest a less structured prioritization system or frequent reprioritization.

        Note: Requires 'priority' attribute in events or traces

        Returns:
            float: Case priority dynamics score (0-1)
        """
        # Track priority changes for each case
        case_priority_changes: Dict[str, List[Tuple[Any, Any]]] = defaultdict(list)

        for trace in window:
            case_id: str = trace.attributes.get("concept:name", "")
            if not case_id:
                continue

            # Extract priority from trace or events
            priorities: List[Any] = []
            for event in trace:
                # Try to get priority from event
                priority = event.get("priority", None)
                if priority is not None:
                    priorities.append(priority)

            # If no priorities found in events, check trace attributes
            if not priorities and "priority" in trace.attributes:
                priorities = [trace.attributes["priority"]]

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
    def concept_drift(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Detects concept drift in the process over time.

        This feature measures how much the process behavior has changed compared
        to a reference model of the process. It combines multiple indicators of
        drift including control flow changes, performance changes, and data changes.
        Higher values indicate significant process evolution or change.

        Returns:
            float: Concept drift score (0-1), where higher values indicate more drift
        """
        # Extract multiple process characteristics
        current_characteristics: Dict[str, Any] = {
            'activity_dist': cls._get_activity_distribution(window),
            'dfr': cls._get_direct_follow_relations(window),
            'variants': cls._get_variant_distribution(window)
        }

        # Get reference characteristics from memory
        ref_characteristics: Optional[Dict[str, Any]] = memory.get_feature_value('ref_process_characteristics')

        if ref_characteristics is None:
            # First window, establish reference
            memory.set_feature_value('ref_process_characteristics', current_characteristics)
            return 0.0  # No drift from reference (since this is the reference)

        # Calculate drift indicators
        drift_indicators: List[float] = []

        # Compare activity distributions
        act_similarity = cls._compare_distributions(current_characteristics['activity_dist'],
                                                    ref_characteristics['activity_dist'])
        if act_similarity is not None:
            drift_indicators.append(act_similarity)

        # Compare direct-follow relations
        dfr_similarity = cls._compare_relation_sets(current_characteristics['dfr'],
                                                    ref_characteristics['dfr'])
        if dfr_similarity is not None:
            drift_indicators.append(dfr_similarity)

        # Compare variant distributions
        var_similarity = cls._compare_distributions(current_characteristics['variants'],
                                                    ref_characteristics['variants'])
        if var_similarity is not None:
            drift_indicators.append(var_similarity)

        # Average drift indicators (convert from similarity to distance)
        if drift_indicators:
            avg_drift = 1.0 - np.mean(drift_indicators)
        else:
            avg_drift = 0.0

        # Update reference model (slow adaptation)
        if avg_drift > 0.3:  # If significant drift, update reference
            memory.set_feature_value('ref_process_characteristics', current_characteristics)

        return avg_drift

    @classmethod
    def long_term_dependencies(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the presence of long-term dependencies in the process.

        This feature analyzes how activities far apart in traces might be related,
        indicating complex dependencies that span across large portions of the process.
        Higher values indicate more long-term dependencies, suggesting a more
        complex process with strong relationships between distant activities.

        Returns:
            float: Long-term dependency score (0-1)
        """
        # Extract activities from traces
        trace_activities: List[List[str]] = []
        for trace in window:
            activities = [event.get("concept:name", "") for event in trace if event.get("concept:name", "")]
            if activities:
                trace_activities.append(activities)

        if not trace_activities:
            return 0.0

        # Calculate correlations at different distances
        max_distance: int = max(len(trace) for trace in trace_activities) // 2
        max_distance = min(max_distance, 10)  # Limit to reasonable distances

        correlation_scores: List[float] = []

        for distance in range(1, max_distance + 1):
            correlation = cls._calculate_correlation_at_distance(trace_activities, distance)
            correlation_scores.append(correlation)

        # If we have no correlation scores, return 0
        if len(correlation_scores) == 0:
            return 0.0

        # Weight longer distances more heavily
        weights = np.linspace(1.0, 2.0, len(correlation_scores))
        weighted_scores = [w * s for w, s in zip(weights, correlation_scores)]

        # Long-term dependency score is the weighted average of correlation scores
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            ltd_score = sum(weighted_scores) / weight_sum
        else:
            ltd_score = 0.0

        # Normalize to 0-1 range
        normalized_score = min(1.0, ltd_score)

        return normalized_score

    @classmethod
    def temporal_dependency(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the significance of temporal dependencies in the process.

        This feature analyzes how the timing of activities affects subsequent
        process behavior. Higher values indicate that timing has a stronger
        influence on the process flow, suggesting a process where execution
        patterns are significantly influenced by time factors.

        Returns:
            float: Temporal dependency score (0-1)
        """
        # Categorize events by timing patterns
        timing_patterns: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_patterns: int = 0

        for trace in window:
            if len(trace) < 2:
                continue

            events = [(event.get("concept:name", ""), event.get("time:timestamp", None))
                      for event in trace]

            # Filter out events without timestamps
            events = [e for e in events if e[1] is not None]

            if len(events) < 2:
                continue

            # Categorize time gaps
            for i in range(len(events) - 1):
                current_act: str = events[i][0]
                next_act: str = events[i + 1][0]
                current_time = events[i][1]
                next_time = events[i + 1][1]

                if not current_act or not next_act:
                    continue

                try:
                    # Calculate time gap based on timestamp type
                    time_gap: float = 0.0

                    if isinstance(current_time, (int, float)) and isinstance(next_time, (int, float)):
                        time_gap = float(next_time - current_time)
                    elif isinstance(current_time, datetime.datetime) and isinstance(next_time, datetime.datetime):
                        time_gap = float((next_time - current_time).total_seconds())
                    elif isinstance(current_time, str) and isinstance(next_time, str):
                        # Try to parse strings to datetime
                        try:
                            # Try common formats
                            format_attempts = [
                                "%Y-%m-%d %H:%M:%S.%f",
                                "%Y-%m-%d %H:%M:%S",
                                "%Y/%m/%d %H:%M:%S",
                                "%d.%m.%Y %H:%M:%S"
                            ]

                            for fmt in format_attempts:
                                try:
                                    dt1 = datetime.datetime.strptime(current_time, fmt)
                                    dt2 = datetime.datetime.strptime(next_time, fmt)
                                    time_gap = float((dt2 - dt1).total_seconds())
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            # Skip if we can't parse timestamps
                            continue
                    else:
                        # Skip if timestamps are incompatible types
                        continue

                    # Categorize time gap (simplified)
                    gap_category: str = ""
                    if time_gap < 60:  # seconds
                        gap_category = "very_short"
                    elif time_gap < 3600:  # 1 hour
                        gap_category = "short"
                    elif time_gap < 86400:  # 1 day
                        gap_category = "medium"
                    else:
                        gap_category = "long"

                    # Record that current_act followed by gap_category leads to next_act
                    timing_patterns[(current_act, gap_category)][next_act] += 1
                    total_patterns += 1

                except (TypeError, ValueError) as e:
                    # Skip this pair if we can't calculate a time gap
                    continue

        # Calculate how predictive timing is of next activities
        predictivity_scores: List[float] = []

        for (act, gap), next_acts in timing_patterns.items():
            # Calculate entropy of next activities given current activity and gap
            total: int = sum(next_acts.values())
            probabilities: List[float] = [count / total for count in next_acts.values()]
            entropy_val: float = entropy(probabilities)

            # Lower entropy = more predictive
            # Normalize by maximum possible entropy
            max_entropy: float = math.log(len(next_acts))
            predictivity: float = 1.0 - (entropy_val / max_entropy) if max_entropy > 0 else 1.0

            # Weight by frequency
            weighted_predictivity: float = predictivity * (total / total_patterns)
            predictivity_scores.append(weighted_predictivity)

        # Overall temporal dependency score
        return sum(predictivity_scores) if predictivity_scores else 0.0

    # Helper methods

    @classmethod
    def _count_completed_cases(cls, window: EventLog) -> int:
        """
        Helper method to count completed cases in a window.

        Returns:
            int: Number of completed cases
        """
        completed_count: int = 0

        for trace in window:
            # Check if trace has events
            if len(trace) > 0:
                # Check last event for completion marker
                last_event = trace[-1]
                event_name: str = last_event.get("concept:name", "").lower()
                lifecycle: str = last_event.get("lifecycle:transition", "").lower()

                if "complete" in event_name or "end" in event_name or \
                        lifecycle == "complete" or lifecycle == "terminated":
                    completed_count += 1

        return completed_count

    @classmethod
    def _get_activity_distribution(cls, window: EventLog) -> Dict[str, float]:
        """
        Helper method to get activity distribution from window.

        Returns:
            Dict[str, float]: Mapping of activity names to their relative frequencies
        """
        activity_counts: Dict[str, int] = {}
        total: int = 0

        for trace in window:
            for event in trace:
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
    def _get_direct_follow_relations(cls, window: EventLog) -> Set[Tuple[str, str]]:
        """
        Helper method to extract direct-follow relations from window.

        Returns:
            Set[Tuple[str, str]]: Set of direct-follow relations as (activity1, activity2) pairs
        """
        dfr: Set[Tuple[str, str]] = set()

        for trace in window:
            activities: List[str] = [event.get("concept:name", "") for event in trace]

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
    def _get_variant_distribution(cls, window: EventLog) -> Dict[Tuple[str, ...], float]:
        """
        Helper method to get process variant distribution from window.

        Returns:
            Dict[Tuple[str, ...], float]: Mapping of variants to their relative frequencies
        """
        variant_counts: Dict[Tuple[str, ...], int] = {}

        for trace in window:
            variant: Tuple[str, ...] = tuple(event.get("concept:name", "") for event in trace)
            variant_counts[variant] = variant_counts.get(variant, 0) + 1

        # Normalize to get probabilities
        total: int = len(window)
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

