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

from gedi_streams.features.stream_feature import StreamFeature


class AdvancedStreamFeatures(StreamFeature):
    def __init__(self, feature_names='advanced_stream_features', memory=None):
        """
        Initialize the AdvancedStreamFeatures feature extractor.

        This class computes advanced process mining features from event streams,
        focusing on stream-specific control flow metrics normalized between 0 and 1.

        Args:
            feature_names: List of feature names to compute, or 'advanced_stream_features'
                           to compute all available features
            memory: Optional memory object to store state between computations for incremental features
        """
        super().__init__(feature_names)
        self.feature_type = 'advanced_stream_features'
        self.available_class_methods = dict(inspect.getmembers(AdvancedStreamFeatures,
                                                               predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    def extract(self, log: EventLog, memory: ComputedFeatureMemory):
        feature_names = self.feature_names

        output = {}
        for feature_name in feature_names:
            if feature_name in self.available_class_methods:
                feature_fn = self.available_class_methods[feature_name]
                feature_value = feature_fn(log, memory)
                output[f"{feature_name}"] = feature_value
        return output

    @classmethod
    def window_entropy_variability(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures how normalized entropy varies across sliding sub-windows.

        Higher values indicate a more dynamic stream with changing entropy patterns.

        Returns:
            float: Normalized variability score (0-1)
        """
        # Define sub-window size (adaptive to window size)
        sub_window_size = max(3, len(window) // 5)

        # Create sliding sub-windows
        sub_windows = []
        for i in range(0, len(window) - sub_window_size + 1, max(1, sub_window_size // 2)):
            sub_windows.append(window[i:i + sub_window_size])

        # If we can't create meaningful sub-windows, return 0
        if len(sub_windows) < 2:
            return 0.0

        # Calculate normalized entropy for each sub-window
        entropy_values = []
        for sub_window in sub_windows:
            # Get variant distribution
            variant_dist = cls._get_variant_distribution(sub_window)
            if not variant_dist:
                continue

            # Calculate normalized entropy
            probabilities = list(variant_dist.values())
            entropy_val = entropy(probabilities)
            max_entropy = math.log(len(variant_dist))
            norm_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
            entropy_values.append(norm_entropy)

        # Calculate coefficient of variation (CV) of entropy values
        if len(entropy_values) < 2:
            return 0.0

        mean_entropy = np.mean(entropy_values)
        std_entropy = np.std(entropy_values)

        # Normalize CV to [0,1] using a sigmoid-like function
        cv = std_entropy / mean_entropy if mean_entropy > 0 else 0.0
        variability = 1.0 - math.exp(-cv)

        return min(1.0, max(0.0, variability))

    @classmethod
    def drift_gradualness(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures how gradually control flow patterns change across the stream.

        Lower values indicate sudden changes (abrupt drifts), higher values indicate
        gradual evolution of the process.

        Returns:
            float: Drift gradualness score (0-1)
        """
        # Create sub-windows to analyze change patterns
        num_segments = 4  # Divide the window into this many segments
        segment_size = len(window) // num_segments

        if segment_size < 2 or num_segments < 2:
            return 1.0  # Not enough data to detect drift patterns

        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(window)
            segments.append(window[start_idx:end_idx])

        # Calculate activity distributions for each segment
        distributions = []
        for segment in segments:
            dist = cls._get_activity_distribution(segment)
            if dist:  # Only add non-empty distributions
                distributions.append(dist)

        if len(distributions) < 2:
            return 1.0  # Not enough data

        # Calculate changes between consecutive distributions
        changes = []
        for i in range(len(distributions) - 1):
            similarity = cls._compare_distributions(distributions[i], distributions[i + 1])
            change = 1.0 - similarity  # Convert similarity to distance
            changes.append(change)

        # Measure the "smoothness" of changes
        if not changes:
            return 1.0

        # Calculate statistics of changes
        mean_change = np.mean(changes)
        max_change = np.max(changes)

        # If mean_change is very small, process is stable
        if mean_change < 0.05:
            return 1.0

        # Ratio of max to mean (high ratio = less gradual)
        smoothness_ratio = max_change / mean_change if mean_change > 0 else 1.0

        # Convert to gradualness score (higher = more gradual)
        gradualness = 1.0 - min(1.0, (smoothness_ratio - 1.0) / 5.0)

        return gradualness

    @classmethod
    def variant_evolution_rate(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures how quickly new variants appear in the stream over time.

        Higher values indicate continuous introduction of new variants throughout the stream.
        Lower values indicate that most variants appear early in the stream.

        Returns:
            float: Variant evolution rate (0-1)
        """
        # Get all trace variants in the window
        all_variants = set()
        variant_first_seen = {}

        # Track when each variant is first seen
        for i, trace in enumerate(window):
            variant = tuple(event.get("concept:name", "") for event in trace)
            normalized_position = i / len(window) if len(window) > 1 else 0.5

            if variant not in all_variants:
                all_variants.add(variant)
                variant_first_seen[variant] = normalized_position

        if not variant_first_seen:
            return 0.0

        # Calculate distribution of first appearances
        appearance_positions = list(variant_first_seen.values())

        # Calculate statistics
        mean_position = np.mean(appearance_positions)

        # Calculate variance in positions (modified to prioritize later appearances)
        weights = np.array([1.0 + 2.0 * pos for pos in appearance_positions])
        weighted_mean = np.sum(weights * np.array(appearance_positions)) / np.sum(weights)

        # Evolution rate is higher when mean position is higher
        # and when variants appear throughout the stream
        evolution_rate = 0.7 * weighted_mean + 0.3 * mean_position

        return min(1.0, max(0.0, evolution_rate))

    @classmethod
    def recurrence_factor(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the extent to which control flow patterns reappear after disappearing.

        Higher values indicate that the stream exhibits recurring patterns over time.

        Returns:
            float: Recurrence factor (0-1)
        """
        # Define number of segments for analysis
        num_segments = min(6, len(window) // 10)
        if num_segments < 3:
            return 0.0  # Not enough data

        segment_size = len(window) // num_segments
        segments = []

        # Create segments
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(window)
            segments.append(window[start_idx:end_idx])

        # Calculate activity distributions for each segment
        distributions = []
        for segment in segments:
            dist = cls._get_activity_distribution(segment)
            distributions.append(dist)

        # Calculate direct follow relations for each segment
        dfrs = []
        for segment in segments:
            dfr = cls._get_direct_follow_relations(segment)
            dfrs.append(dfr)

        # Calculate similarities between non-adjacent segments
        recurrence_scores = []

        # Skip adjacent segments (i+1) and look at i+2 and beyond
        for i in range(len(segments) - 2):
            for j in range(i + 2, len(segments)):
                # Calculate similarity between segments i and j
                activity_sim = cls._compare_distributions(distributions[i], distributions[j])
                dfr_sim = cls._compare_relation_sets(dfrs[i], dfrs[j])

                # Combined similarity (weighted)
                combined_sim = 0.4 * activity_sim + 0.6 * dfr_sim

                # Weight by distance between segments (farther = more significant)
                distance_weight = (j - i) / (len(segments) - 1)
                weighted_sim = combined_sim * (0.5 + 0.5 * distance_weight)

                recurrence_scores.append(weighted_sim)

        # Overall recurrence factor is the maximum weighted similarity
        recurrence_factor = max(recurrence_scores) if recurrence_scores else 0.0

        return recurrence_factor

    @classmethod
    def temporal_locality(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the degree to which similar traces appear close in time.

        Higher values indicate that similar traces tend to cluster together in time.

        Returns:
            float: Temporal locality score (0-1)
        """
        if len(window) < 4:
            return 0.5  # Not enough data

        # Calculate similarity between each trace and the next N traces
        look_ahead = min(10, len(window) // 3)
        similarity_scores = []

        for i in range(len(window) - 1):
            current_trace_activities = [event.get("concept:name", "") for event in window[i]]

            # Look at the next N traces
            for j in range(i + 1, min(i + look_ahead + 1, len(window))):
                next_trace_activities = [event.get("concept:name", "") for event in window[j]]

                # Calculate similarity (Jaccard)
                current_set = set(current_trace_activities)
                next_set = set(next_trace_activities)

                if not current_set or not next_set:
                    continue

                intersection = len(current_set.intersection(next_set))
                union = len(current_set.union(next_set))

                similarity = intersection / union if union > 0 else 0.0

                # Weight by proximity (closer = higher weight)
                proximity_weight = 1.0 - ((j - i - 1) / look_ahead)
                weighted_similarity = similarity * proximity_weight

                similarity_scores.append(weighted_similarity)

        # No valid comparisons
        if not similarity_scores:
            return 0.5

        # Calculate average weighted similarity
        avg_weighted_similarity = np.mean(similarity_scores)

        # Compare with random expectation (0.5)
        # Scale to [0,1] where 0.5 is random, 1.0 is perfect locality
        locality = 0.5 + avg_weighted_similarity

        # Ensure in [0,1] range
        return min(1.0, max(0.0, locality))

    @classmethod
    def case_overlap_ratio(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the proportion of time where events from multiple cases appear interleaved.

        Higher values indicate that the process has many parallel cases with interleaved events.

        Returns:
            float: Case overlap ratio (0-1)
        """
        # Extract timestamp information
        case_events = defaultdict(list)

        for trace in window:
            case_id = trace.attributes.get("concept:name", "")
            if not case_id:
                continue

            for event in trace:
                timestamp = event.get("time:timestamp", None)
                if timestamp:
                    case_events[case_id].append(timestamp)

        # Sort timestamps for each case
        for case_id in case_events:
            case_events[case_id].sort()

        # Count overlapping segments
        total_time_span = 0
        overlapping_time_span = 0

        # Get all cases with at least 2 events (to have a timespan)
        valid_cases = [case_id for case_id, events in case_events.items() if len(events) >= 2]

        if len(valid_cases) < 2:
            return 0.0  # Not enough data

        # Calculate case timespans
        case_timespans = []
        for case_id in valid_cases:
            events = case_events[case_id]
            first_event = events[0]
            last_event = events[-1]

            # Create timespan (start, end, case_id)
            case_timespans.append((first_event, last_event, case_id))

        # Sort timespans by start time
        case_timespans.sort(key=lambda x: x[0])

        # Merge overlapping intervals
        merged_spans = []
        active_cases = []

        for start, end, case_id in case_timespans:
            # Remove cases that have ended
            while active_cases and active_cases[0][0] <= start:
                old_end, old_case = active_cases.pop(0)

            # Add current case
            active_cases.append((end, case_id))
            active_cases.sort()  # Sort by end time

            # If more than one active case, we have overlap
            if len(active_cases) > 1:
                merged_spans.append((start, min(end, active_cases[1][0]), len(active_cases)))

        # Calculate total timespan
        if not case_timespans:
            return 0.0

        overall_start = case_timespans[0][0]
        overall_end = max(span[1] for span in case_timespans)

        try:
            # Calculate total and overlapping time
            if isinstance(overall_start, (int, float)) and isinstance(overall_end, (int, float)):
                total_time = float(overall_end - overall_start)
            elif isinstance(overall_start, datetime.datetime) and isinstance(overall_end, datetime.datetime):
                total_time = float((overall_end - overall_start).total_seconds())
            else:
                return 0.5  # Can't calculate

            # Calculate overlapping time and weight by number of cases
            overlap_time = 0.0
            for o_start, o_end, num_cases in merged_spans:
                if isinstance(o_start, (int, float)) and isinstance(o_end, (int, float)):
                    segment_time = float(o_end - o_start)
                elif isinstance(o_start, datetime.datetime) and isinstance(o_end, datetime.datetime):
                    segment_time = float((o_end - o_start).total_seconds())
                else:
                    continue

                # Weight by number of cases (more cases = higher weight)
                weight = min(1.0, (num_cases - 1) / 5.0)  # Cap at 6 concurrent cases
                overlap_time += segment_time * weight

            # Calculate ratio
            overlap_ratio = overlap_time / total_time if total_time > 0 else 0.0

            return min(1.0, overlap_ratio)

        except (TypeError, ValueError):
            return 0.5  # Error in calculation

    @classmethod
    def path_consistency(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the consistency of successor relationships in the stream.

        Higher values indicate predictable control flow with consistent successor relationships.

        Returns:
            float: Path consistency score (0-1)
        """
        # Extract activity successor relationships
        successor_counts = defaultdict(Counter)
        total_successors = 0

        for trace in window:
            activities = [event.get("concept:name", "") for event in trace if event.get("concept:name", "")]

            if len(activities) < 2:
                continue

            for i in range(len(activities) - 1):
                current = activities[i]
                next_act = activities[i + 1]

                successor_counts[current][next_act] += 1
                total_successors += 1

        if not successor_counts:
            return 1.0  # Default to fully consistent

        # Calculate entropy of successor distributions
        entropies = []
        weights = []

        for activity, successors in successor_counts.items():
            total = sum(successors.values())
            probabilities = [count / total for count in successors.values()]

            act_entropy = entropy(probabilities)
            max_entropy = math.log(len(successors))
            normalized_entropy = act_entropy / max_entropy if max_entropy > 0 else 0.0

            # Weight by activity frequency
            activity_weight = total / total_successors

            entropies.append(normalized_entropy)
            weights.append(activity_weight)

        # Calculate weighted average entropy
        if not entropies:
            return 1.0

        weighted_entropy = np.average(entropies, weights=weights)

        # Convert to consistency score (higher entropy = lower consistency)
        consistency = 1.0 - weighted_entropy

        return consistency

    @classmethod
    def stream_homogeneity(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the uniformity of variant distribution across the entire stream.

        Higher values indicate more uniform distribution of variants.

        Returns:
            float: Stream homogeneity score (0-1)
        """
        # Get variant distribution
        variant_dist = cls._get_variant_distribution(window)

        if not variant_dist:
            return 1.0  # Empty log is considered homogeneous

        # Calculate entropy
        probabilities = list(variant_dist.values())
        entropy_val = entropy(probabilities)

        # Calculate maximum possible entropy
        max_entropy = math.log(len(variant_dist))

        # Normalized entropy as homogeneity measure
        homogeneity = entropy_val / max_entropy if max_entropy > 0 else 1.0

        return homogeneity

    @classmethod
    def loop_structure_stability(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures the consistency of loop structures in the process.

        Higher values indicate more consistent loop behavior across cases.

        Returns:
            float: Loop structure stability score (0-1)
        """
        # Identify potential loops in traces
        loop_counts = defaultdict(list)

        for trace in window:
            # Extract activities
            activities = [event.get("concept:name", "") for event in trace if event.get("concept:name", "")]

            # Identify sequences that might be loops
            activity_indices = defaultdict(list)

            for i, activity in enumerate(activities):
                activity_indices[activity].append(i)

            # Analyze potential loops (activities that appear multiple times)
            for activity, indices in activity_indices.items():
                if len(indices) > 1:
                    # Calculate distances between consecutive occurrences
                    distances = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]

                    # Store loop distances for this activity
                    if distances:
                        loop_counts[activity].extend(distances)

        if not loop_counts:
            return 0.5  # No loops detected

        # Calculate coefficient of variation for each loop
        loop_stability_scores = []

        for activity, distances in loop_counts.items():
            if len(distances) < 2:
                continue

            mean_distance = np.mean(distances)
            std_distance = np.std(distances)

            # Coefficient of variation (lower is more stable)
            cv = std_distance / mean_distance if mean_distance > 0 else float('inf')

            # Convert to stability score (higher = more stable)
            stability = math.exp(-min(5.0, cv))
            loop_stability_scores.append(stability)

        # Overall stability is the average of individual loop stabilities
        overall_stability = np.mean(loop_stability_scores) if loop_stability_scores else 0.5

        return overall_stability

    @classmethod
    def reachability_preservation(cls, window: EventLog, memory: ComputedFeatureMemory) -> float:
        """
        Measures how well reachability relationships are preserved over time.

        Higher values indicate that activity reachability relationships remain consistent
        throughout the stream.

        Returns:
            float: Reachability preservation score (0-1)
        """
        # Divide the window into segments for comparison
        num_segments = min(5, max(2, len(window) // 20))
        segment_size = len(window) // num_segments

        if segment_size < 3 or num_segments < 2:
            return 0.5  # Not enough data

        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(window)
            segments.append(window[start_idx:end_idx])

        # Calculate reachability matrices for each segment
        reachability_matrices = []

        for segment in segments:
            # Get all activities
            all_activities = set()
            for trace in segment:
                for event in trace:
                    activity = event.get("concept:name", "")
                    if activity:
                        all_activities.add(activity)

            # Create mapping from activity to index
            activity_to_index = {activity: i for i, activity in enumerate(sorted(all_activities))}

            # Initialize reachability matrix
            matrix_size = len(all_activities)
            reachability = np.zeros((matrix_size, matrix_size))

            # Fill reachability matrix
            for trace in segment:
                activities = [event.get("concept:name", "") for event in trace if event.get("concept:name", "")]

                for i in range(len(activities)):
                    for j in range(i + 1, len(activities)):
                        # Mark that activities[j] is reachable from activities[i]
                        src_idx = activity_to_index.get(activities[i])
                        dst_idx = activity_to_index.get(activities[j])

                        if src_idx is not None and dst_idx is not None:
                            reachability[src_idx, dst_idx] = 1.0

            reachability_matrices.append((reachability, activity_to_index))

        # Calculate preservation scores between consecutive segments
        preservation_scores = []

        for i in range(len(reachability_matrices) - 1):
            current_matrix, current_mapping = reachability_matrices[i]
            next_matrix, next_mapping = reachability_matrices[i + 1]

            # Get common activities
            common_activities = set(current_mapping.keys()) & set(next_mapping.keys())

            if len(common_activities) < 2:
                continue

            # Calculate preservation for common activities
            preserved_relations = 0
            total_relations = 0

            for src_act in common_activities:
                for dst_act in common_activities:
                    if src_act != dst_act:
                        src_idx_curr = current_mapping[src_act]
                        dst_idx_curr = current_mapping[dst_act]

                        src_idx_next = next_mapping[src_act]
                        dst_idx_next = next_mapping[dst_act]

                        # Check if reachability is preserved
                        if current_matrix[src_idx_curr, dst_idx_curr] == next_matrix[src_idx_next, dst_idx_next]:
                            preserved_relations += 1

                        total_relations += 1

            # Calculate preservation ratio
            preservation_ratio = preserved_relations / total_relations if total_relations > 0 else 0.0
            preservation_scores.append(preservation_ratio)

        # Overall preservation score is the average across all segment comparisons
        overall_preservation = np.mean(preservation_scores) if preservation_scores else 0.5

        return overall_preservation

    # Helper methods adapted from StreamFeature class

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
        all_keys = set(dist1) | set(dist2)

        if not all_keys:
            return 1.0  # Empty distributions are considered identical

        # Create arrays with zeros for missing keys
        p = np.array([dist1.get(k, 0) for k in all_keys])
        q = np.array([dist2.get(k, 0) for k in all_keys])

        # Ensure non-zero probabilities (avoid log(0))
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate JS divergence
        m = 0.5 * (p + q)

        # Since we're comparing distributions, entropy should not fail,
        # but just in case, let's wrap it in a try-except
        try:
            js_div = 0.5 * (entropy(p, m) + entropy(q, m))
            # Convert to a similarity score (1 - normalized_divergence)
            return 1.0 - min(1.0, js_div)
        except (ValueError, ZeroDivisionError):
            # If entropy calculation fails, fall back to a simpler metric: L1 distance
            l1_dist = np.sum(np.abs(p - q)) / 2  # Normalized L1 (between 0-1)
            return 1.0 - l1_dist

    @classmethod
    def _get_direct_follow_relations(cls, window: EventLog) -> Set[Tuple[str, str]]:
        """
        Helper method to extract direct-follow relations from window.

        Returns:
            Set[Tuple[str, str]]: Set of direct-follow relations as (activity1, activity2) pairs
        """
        dfr = set()

        for trace in window:
            activities = [event.get("concept:name", "") for event in trace]

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

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    @classmethod
    def _get_variant_distribution(cls, window: EventLog) -> Dict[Tuple[str, ...], float]:
        """
        Helper method to get process variant distribution from window.

        Returns:
            Dict[Tuple[str, ...], float]: Mapping of variants to their relative frequencies
        """
        variant_counts = {}

        for trace in window:
            variant = tuple(event.get("concept:name", "") for event in trace)
            variant_counts[variant] = variant_counts.get(variant, 0) + 1

        # Normalize to get probabilities
        total = len(window)
        if total > 0:
            return {k: v / total for k, v in variant_counts.items()}
        else:
            return {}