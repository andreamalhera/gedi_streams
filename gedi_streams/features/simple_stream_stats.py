import inspect
import math

import numpy as np

from gedi_streams.features.stream_feature import StreamFeature
from gedi_streams.features.memory import ComputedFeatureMemory
from pm4py.objects.log.obj import EventLog
from scipy import stats


class SimpleStreamStats(StreamFeature):
    def __init__(self, feature_names='simple_stream_stats', memory=None):
        """
        Initialize the SimpleStreamStats feature extractor.

        This class computes various statistical features from event streams, tracking metrics
        related to process volume, variety, structure, and evolution over time.

        Args:
            feature_names: List of feature names to compute, or 'simple_stream_stats' to compute all available features
            memory: Optional memory object to store state between computations for incremental features
        """
        super().__init__(feature_names)
        self.feature_type = 'simple_stream_stats'
        self.available_class_methods = dict(inspect.getmembers(SimpleStreamStats, predicate=inspect.ismethod))
        if self.feature_type in feature_names:
            self.feature_names = [*self.available_class_methods.keys()]
        else:
            self.feature_names = feature_names

    @classmethod
    def n_events(self, window: EventLog, memory: ComputedFeatureMemory):
        """
        Calculates the cumulative number of events observed in the stream so far.

        This feature counts the total number of events across all traces in the current window
        and adds it to the running total from previous windows if available.

        Interpretation: Higher values indicate more events being processed. The growth rate
        can indicate process volume trends over time.

        Returns:
            int: The cumulative count of all events observed so far
        """
        previous_value = memory.get_feature_value('n_events')
        n_events = sum(len(trace) for trace in window)
        return n_events + previous_value if previous_value is not None else n_events

    @classmethod
    def n_traces(self, window: EventLog, memory: ComputedFeatureMemory):
        """
        Calculates the cumulative number of process traces observed in the stream so far.

        This feature counts the number of complete process traces in the current window
        and adds it to the running total from previous windows if available.

        Interpretation: This metric tracks case volume over time. Increasing values indicate
        more process instances being executed.

        Returns:
            int: The cumulative count of all traces observed so far
        """
        previous_value = memory.get_feature_value('n_traces')
        n_traces = len(window)
        return len(window) + previous_value if previous_value is not None else len(window)

    @classmethod
    def n_windows(self, window: EventLog, memory: ComputedFeatureMemory):
        """
        Tracks the number of windows processed so far in the event stream.

        This feature increments a counter for each new window of data processed,
        maintaining state between calls to track the total.

        Interpretation: Used primarily for normalization of other metrics and to
        track the progression of stream processing.

        Returns:
            int: The total number of windows processed so far
        """
        previous_value = memory.get_feature_value('n_windows')
        return 1 + previous_value if previous_value is not None else 1

    @classmethod
    def ratio_events_per_window(self, window: EventLog, memory: ComputedFeatureMemory):
        """
        Calculates the average number of events per window across all processed windows.

        This feature divides the cumulative number of events by the cumulative number
        of windows to show the average window size in terms of events.

        Interpretation: Provides insight into event volume density over time. Changes in this
        metric may indicate shifts in process execution patterns or data collection strategies.

        Returns:
            float: Average number of events per window
        """
        previous_n_events = memory.get_feature_value('n_events')
        previous_n_windows = memory.get_feature_value('n_windows')

        new_n_events = sum(
            len(trace) for trace in window) + previous_n_events if previous_n_events is not None else sum(
            len(trace) for trace in window)
        new_n_windows = 1 + previous_n_windows if previous_n_windows is not None else 1

        return new_n_events / new_n_windows

    @classmethod
    def ratio_traces_per_window(self, window: EventLog, memory: ComputedFeatureMemory):
        """
        Calculates the average number of traces per window across all processed windows.

        This feature divides the cumulative number of traces by the cumulative number
        of windows to show the average number of process instances per window.

        Interpretation: Provides insight into case volume density over time. Changes may
        indicate shifts in process throughput or window sizing strategy.

        Returns:
            float: Average number of traces per window
        """
        previous_n_traces = memory.get_feature_value('n_traces')
        previous_n_windows = memory.get_feature_value('n_windows')

        new_n_traces = len(window) + previous_n_traces if previous_n_traces is not None else len(window)
        new_n_windows = 1 + previous_n_windows if previous_n_windows is not None else 1

        return new_n_traces / new_n_windows

    @classmethod
    def activity_appearance_rate(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the rate of new activity types appearing in the event stream.

        This feature identifies activities that haven't been seen before in the stream
        and calculates the ratio of new activities to the total number of traces in the window.

        Interpretation: High values indicate process evolution with new activities being introduced.
        Values close to zero suggest a stable process with established activities. Spikes may
        indicate process changes or the introduction of new variants.

        Returns:
            float: Ratio of newly observed activities to the number of traces in the window
        """
        current_activities = set()
        for trace in window:
            for event in trace:
                current_activities.add(event["concept:name"])

        previous_activities = memory.get_feature_value('known_activities')
        if previous_activities is None:
            previous_activities = set()

        new_activities = current_activities - previous_activities
        memory.set_feature_value('known_activities', previous_activities.union(current_activities))

        return len(new_activities) / (len(window) or 1)  # Avoid division by zero

    @classmethod
    def variant_appearance_rate(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the rate of new process variants appearing in the event stream.

        A process variant is a unique sequence of activities representing a specific
        execution path through the process. This feature identifies variants that
        haven't been seen before and calculates their ratio to total traces.

        Interpretation: High values indicate process evolution with new execution paths.
        Values close to zero suggest a stable process with established pathways.
        Sustained high values may indicate a flexible or unstructured process.

        Returns:
            float: Ratio of newly observed variants to the number of traces in the window
        """
        current_variants = set()
        for trace in window:
            variant = tuple(event["concept:name"] for event in trace)
            current_variants.add(variant)

        previous_variants = memory.get_feature_value('known_variants')
        if previous_variants is None:
            previous_variants = set()

        new_variants = current_variants - previous_variants
        memory.set_feature_value('known_variants', previous_variants.union(current_variants))

        return len(new_variants) / (len(window) or 1)

    @classmethod
    def drift_indicator(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Detects distribution changes in process variants that may indicate concept drift.

        This feature compares the current distribution of process variants with the
        previously observed distribution using a simplified Hellinger distance measure.
        The distance metric ranges from 0 (identical distributions) to 1 (completely different).

        Interpretation: Values close to 0 indicate stable process behavior, while
        values approaching 1 indicate significant drift in the process execution patterns.
        Sudden spikes may indicate abrupt process changes or disruptions.

        Returns:
            float: Normalized distance measure between 0-1 indicating degree of drift
        """
        # Get current variant distribution
        current_variants = {}
        for trace in window:
            variant = tuple(event["concept:name"] for event in trace)
            current_variants[variant] = current_variants.get(variant, 0) + 1

        # Normalize current distribution
        total_traces = len(window)
        current_dist = {v: count / total_traces for v, count in current_variants.items()} if total_traces > 0 else {}

        # Get previous distribution
        previous_dist = memory.get_feature_value('variant_distribution')
        if previous_dist is None:
            memory.set_feature_value('variant_distribution', current_dist)
            return 0.0

        # Calculate distribution distance (simplified Hellinger distance)
        all_variants = set(list(current_dist.keys()) + list(previous_dist.keys()))
        distance = 0.0
        for variant in all_variants:
            p1 = current_dist.get(variant, 0)
            p2 = previous_dist.get(variant, 0)
            distance += (math.sqrt(p1) - math.sqrt(p2)) ** 2

        memory.set_feature_value('variant_distribution', current_dist)
        return math.sqrt(distance) / math.sqrt(2)

    # === Window-Based Control Flow Features ===

    @classmethod
    def direct_follows_entropy(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the entropy (uncertainty) in the direct-follows relations within the process.

        This feature calculates the Shannon entropy of direct-follows relations (where one
        activity is immediately followed by another) within the current window. Higher entropy
        indicates more variety and uncertainty in the transitions between activities.

        Interpretation: Low values suggest a structured process with predictable flows.
        High values indicate a flexible process with many possible paths. Changes may
        indicate shifts in process structure or execution patterns.

        Returns:
            float: Entropy value of direct-follows relations, where higher values indicate more complexity
        """
        df_counts = {}
        total_relations = 0

        for trace in window:
            for i in range(len(trace) - 1):
                a = trace[i]["concept:name"]
                b = trace[i + 1]["concept:name"]
                df_counts[(a, b)] = df_counts.get((a, b), 0) + 1
                total_relations += 1

        if total_relations == 0:
            return 0.0

        entropy = 0.0
        for count in df_counts.values():
            prob = count / total_relations
            entropy -= prob * math.log2(prob)

        return entropy

    @classmethod
    def trace_length_variability(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the variability in process trace lengths using coefficient of variation.

        This feature calculates how consistent or variable the lengths of process traces are
        within the current window. It uses the coefficient of variation (standard deviation
        divided by mean) to quantify the relative dispersion.

        Interpretation: Low values indicate consistent trace lengths, suggesting a standardized
        process. High values indicate wide variation in process execution, possibly indicating
        different case types or exception handling. Changes may indicate process restructuring
        or shifts in case mix.

        Returns:
            float: Coefficient of variation of trace lengths (standard deviation / mean)
        """
        if not window:
            return 0.0

        lengths = [len(trace) for trace in window]
        mean_length = sum(lengths) / len(lengths)

        if mean_length == 0:
            return 0.0

        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)

        return std_dev / mean_length

    @classmethod
    def concurrent_activities_ratio(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Estimates the degree of concurrency in the process by analyzing activity relationships.

        This feature identifies potentially concurrent activities by finding pairs where
        activity A can follow B and B can follow A, suggesting they might be executed in
        parallel branches. The ratio is calculated as the number of such pairs divided by
        the total possible pairs.

        Interpretation: Values close to 0 indicate sequential processes with little parallelism.
        Higher values suggest more concurrent execution paths. Changes may indicate shifts in
        process architecture or execution strategies.

        Returns:
            float: Ratio between 0-1 representing the estimated degree of concurrency
        """
        # Build direct follow relations
        df_relations = set()
        for trace in window:
            for i in range(len(trace) - 1):
                a = trace[i]["concept:name"]
                b = trace[i + 1]["concept:name"]
                df_relations.add((a, b))

        # Find potential concurrency by checking if a>b and b>a
        activities = set()
        for a, b in df_relations:
            activities.add(a)
            activities.add(b)

        concurrent_pairs = 0
        total_possible_pairs = len(activities) * (len(activities) - 1) / 2

        if total_possible_pairs == 0:
            return 0.0

        for a in activities:
            for b in activities:
                if a < b:  # To avoid counting twice
                    if (a, b) in df_relations and (b, a) in df_relations:
                        concurrent_pairs += 1

        return concurrent_pairs / total_possible_pairs

    # === Control Flow Complexity Metrics ===

    @classmethod
    def activity_entropy(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the entropy (diversity) of activities within the process.

        This feature calculates the Shannon entropy of the activity distribution,
        providing insight into how evenly or unevenly activities occur within the process.
        Higher entropy indicates more uniform activity distribution.

        Interpretation: Lower values indicate a process dominated by few activities,
        while higher values suggest a more balanced distribution of activities.
        Changes may indicate shifts in process focus or structure.

        Returns:
            float: Entropy value of activity distribution, higher values indicate more diversity
        """
        activity_counts = {}
        total_events = 0

        for trace in window:
            for event in trace:
                act = event["concept:name"]
                activity_counts[act] = activity_counts.get(act, 0) + 1
                total_events += 1

        if total_events == 0:
            return 0.0

        entropy = 0.0
        for count in activity_counts.values():
            prob = count / total_events
            entropy -= prob * math.log2(prob)

        return entropy

    @classmethod
    def unique_paths_ratio(cls, window: EventLog, memory: ComputedFeatureMemory):
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
        if not window:
            return 0.0

        unique_paths = set()
        for trace in window:
            path = tuple(event["concept:name"] for event in trace)
            unique_paths.add(path)

        return len(unique_paths) / len(window)

    @classmethod
    def structured_complexity(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Provides a heuristic measure of process complexity based on control flow patterns.

        This feature combines the number of activities, the number of direct-follow relations,
        and the density of these relations to estimate overall process structural complexity.

        Interpretation: Higher values indicate more complex processes with many activities
        and dense relationships between them. Changes may indicate process refinement,
        extension, or simplification efforts. The metric increases with more activities
        and denser connections between them.

        Returns:
            float: A complexity score where higher values indicate more complex process structure
        """
        # Count activity occurrences
        act_counts = {}
        for trace in window:
            for event in trace:
                act = event["concept:name"]
                act_counts[act] = act_counts.get(act, 0) + 1

        # Count direct follow relations
        df_relations = {}
        total_relations = 0

        for trace in window:
            for i in range(len(trace) - 1):
                a = trace[i]["concept:name"]
                b = trace[i + 1]["concept:name"]
                df_relations[(a, b)] = df_relations.get((a, b), 0) + 1
                total_relations += 1

        n_activities = len(act_counts)
        n_relations = len(df_relations)

        if n_activities <= 1:
            return 0.0

        # Complexity increases with more relations between activities
        max_possible_relations = n_activities * n_activities
        density = n_relations / max_possible_relations if max_possible_relations > 0 else 0

        return density * math.log(n_activities + 1)

    # === Stream Evolution Metrics ===

    @classmethod
    def long_term_activity_shift(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures how the activity distribution shifts over the long term using exponential moving averages.

        This feature compares the current activity distribution with a long-term exponential
        moving average (EMA) to detect gradual shifts in activity patterns. The distance is
        calculated using the L1 norm (total variation distance).

        Interpretation: Values close to 0 indicate stable activity distributions over time.
        Higher values indicate significant shifts in which activities are being executed.
        This metric is sensitive to gradual changes that might not trigger immediate drift detection.

        Returns:
            float: Distance measure between 0-1 indicating the degree of activity distribution shift
        """
        # Current activity distribution
        current_act_counts = {}
        total_events = 0

        for trace in window:
            for event in trace:
                act = event["concept:name"]
                current_act_counts[act] = current_act_counts.get(act, 0) + 1
                total_events += 1

        # Normalize current distribution
        if total_events > 0:
            current_dist = {act: count / total_events for act, count in current_act_counts.items()}
        else:
            current_dist = {}

        # Initialize or update exponential moving average
        ema_dist = memory.get_feature_value('activity_ema')
        alpha = 0.1  # EMA smoothing factor

        if ema_dist is None:
            ema_dist = current_dist
        else:
            # Update EMA for all activities
            all_activities = set(list(current_dist.keys()) + list(ema_dist.keys()))
            new_ema = {}
            for act in all_activities:
                curr_val = current_dist.get(act, 0)
                prev_val = ema_dist.get(act, 0)
                new_ema[act] = alpha * curr_val + (1 - alpha) * prev_val
            ema_dist = new_ema

        memory.set_feature_value('activity_ema', ema_dist)

        # Calculate distance between current and EMA distributions
        distance = 0.0
        all_activities = set(list(current_dist.keys()) + list(ema_dist.keys()))

        for act in all_activities:
            p1 = current_dist.get(act, 0)
            p2 = ema_dist.get(act, 0)
            distance += abs(p1 - p2)

        return distance / 2  # Normalize to [0,1]

    @classmethod
    def variant_stability(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the stability of dominant process variants over time.

        This feature tracks how consistent the top process variants remain across windows.
        It maintains a record of previously dominant variants and calculates how prominent
        they remain in subsequent windows, with higher weights given to previously higher-ranked variants.

        Interpretation: Values close to 1 indicate stable processes where the same variants
        dominate over time. Lower values indicate process instability with changing dominant patterns.
        This metric is useful for detecting shifts in the most common execution paths.

        Returns:
            float: Stability score between 0-1, where higher values indicate more stable dominant variants
        """
        current_variants = {}
        for trace in window:
            variant = tuple(event["concept:name"] for event in trace)
            current_variants[variant] = current_variants.get(variant, 0) + 1

        # Calculate stability based on previous dominant variants
        previous_dominant = memory.get_feature_value('dominant_variants')
        if previous_dominant is None:
            # Initialize with top variants from first window
            sorted_variants = sorted(current_variants.items(), key=lambda x: x[1], reverse=True)
            top_n = min(3, len(sorted_variants))  # Top 3 or all if fewer
            previous_dominant = {v[0]: i + 1 for i, v in enumerate(sorted_variants[:top_n])}
            memory.set_feature_value('dominant_variants', previous_dominant)
            return 1.0  # Default stability

        # Check if previous dominant variants are still prominent
        total_traces = len(window)
        stability = 0.0

        if total_traces > 0:
            for variant, rank in previous_dominant.items():
                current_count = current_variants.get(variant, 0)
                current_ratio = current_count / total_traces
                # Higher weight for previously higher-ranked variants
                stability += current_ratio * (1.0 / rank)

        # Update dominant variants
        sorted_variants = sorted(current_variants.items(), key=lambda x: x[1], reverse=True)
        top_n = min(3, len(sorted_variants))
        new_dominant = {v[0]: i + 1 for i, v in enumerate(sorted_variants[:top_n])}
        memory.set_feature_value('dominant_variants', new_dominant)

        return min(1.0, stability)  # Cap at 1.0

    # === Performance Metrics ===

    @classmethod
    def throughput_trend(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the trend in process throughput by comparing current and previous event counts.

        This feature calculates the relative change in the number of events between the current
        window and the previous window, indicating whether throughput is increasing or decreasing.

        Interpretation: Positive values indicate increasing throughput (more events processed),
        while negative values indicate decreasing throughput. The magnitude represents the
        relative change rate. This metric helps identify capacity trends or seasonal patterns.

        Returns:
            float: Relative change in throughput, where positive values indicate increasing throughput
        """
        current_events = sum(len(trace) for trace in window)
        previous_events = memory.get_feature_value('previous_events')

        if previous_events is None:
            memory.set_feature_value('previous_events', current_events)
            return 0.0  # No trend yet

        trend = (current_events - previous_events) / (previous_events or 1)  # Avoid division by zero
        memory.set_feature_value('previous_events', current_events)

        return trend

    @classmethod
    def cycle_time_variation(cls, window: EventLog, memory: ComputedFeatureMemory):
        """
        Measures the consistency of process cycle times using coefficient of variation.

        This feature calculates how variable the end-to-end cycle times are across different
        process instances by computing the coefficient of variation (standard deviation
        divided by mean). Cycle time is measured as the time difference between the first
        and last event in each trace.

        Interpretation: Low values indicate consistent cycle times across process instances,
        suggesting a predictable process. High values indicate wide variation in completion times,
        which may signal different case types, resource constraints, or handling procedures.

        Returns:
            float: Coefficient of variation of cycle times, where higher values indicate more variation
        """
        cycle_times = []

        for trace in window:
            if len(trace) >= 2:  # Need at least start and end events
                start_time = trace[0].get("time:timestamp")
                end_time = trace[-1].get("time:timestamp")

                if start_time and end_time and isinstance(start_time, (int, float)) and isinstance(end_time,
                                                                                                   (int, float)):
                    cycle_time = end_time - start_time
                    cycle_times.append(cycle_time)

        if not cycle_times:
            return 0.0

        mean_time = sum(cycle_times) / len(cycle_times)
        if mean_time == 0:
            return 0.0

        variance = sum((t - mean_time) ** 2 for t in cycle_times) / len(cycle_times)
        std_dev = math.sqrt(variance)

        return std_dev / mean_time