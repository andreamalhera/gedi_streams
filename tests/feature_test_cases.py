# Sequential cases with no overlap
import datetime
from typing import Tuple, List, Optional

from pm4py.objects.log.obj import Event

from gedi_streams.features.memory import ComputedFeatureMemory

LOW_CASE_CONCURRENCY = [
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 0), "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 30), "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 0), "case:concept:name": "C1"}),

    # Next case starts after first case completes
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 30), "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 12, 0), "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": datetime.datetime(2023, 1, 1, 12, 30), "case:concept:name": "C2"}),

    # Next case starts after second case completes
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 0), "case:concept:name": "C3"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 30), "case:concept:name": "C3"}),
    Event({"concept:name": "C", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 0), "case:concept:name": "C3"}),
]

# Highly concurrent cases (all overlap)
HIGH_CASE_CONCURRENCY = [
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 0), "case:concept:name": "C1"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 5), "case:concept:name": "C2"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 10), "case:concept:name": "C3"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 15), "case:concept:name": "C4"}),

    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 30), "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 35), "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 40), "case:concept:name": "C3"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 45), "case:concept:name": "C4"}),

    Event({"concept:name": "C", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 0), "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 5), "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 10), "case:concept:name": "C3"}),
    Event({"concept:name": "C", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 15), "case:concept:name": "C4"}),
]

# Current window with very different activity distribution from memory
LOW_CONCEPT_STABILITY = [
    Event({"concept:name": "X", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "Y", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "Z", "time:timestamp": 3, "case:concept:name": "C1"}),

    Event({"concept:name": "X", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "Y", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "Z", "time:timestamp": 6, "case:concept:name": "C2"}),

    # Memory would contain activities A, B, C from previous window
]

# For LOW_CONCEPT_STABILITY
low_concept_stability_memory = ComputedFeatureMemory()
# Previous window had activities A, B, C (very different from current X, Y, Z)
low_concept_stability_memory.set_feature_value('prev_activity_dist', {
    'A': 0.33,
    'B': 0.33,
    'C': 0.34
})

# Current window with identical activity distribution to memory
HIGH_CONCEPT_STABILITY = [
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": 6, "case:concept:name": "C2"}),

    # Memory would contain activities A, B, C from previous window in similar proportions
]

high_concept_stability_memory = ComputedFeatureMemory()
# Previous window had the same activities in similar distribution
high_concept_stability_memory.set_feature_value('prev_activity_dist', {
    'A': 0.33,
    'B': 0.33,
    'C': 0.34
})

# Current completion rate very different from history
LOW_CASE_THROUGHPUT_STABILITY = [
    # Only 1 completed case (previous windows had many more)
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1", "lifecycle:transition": "complete"}),

    # Incomplete cases
    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2"}),

    Event({"concept:name": "A", "time:timestamp": 6, "case:concept:name": "C3"}),

    # Assume memory contains [10, 10, 9, 11, 10] completed cases history
]

low_concept_drift_memory = ComputedFeatureMemory()
# Reference process characteristics similar to current window (A→B→C)
low_concept_drift_memory.set_feature_value('ref_process_characteristics', {
    'activity_dist': {'A': 0.33, 'B': 0.33, 'C': 0.34},
    'dfr': {('A', 'B'), ('B', 'C')},
    'variants': {('A', 'B', 'C'): 1.0}
})

HIGH_CASE_THROUGHPUT_STABILITY = [
    # 10 completed cases, matching history
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1", "lifecycle:transition": "complete"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": 6, "case:concept:name": "C2", "lifecycle:transition": "complete"}),

    # ... 8 more similar completed cases ...
    # Assume memory contains [10, 9, 10, 11, 10] completed cases history
]

high_concept_drift_memory = ComputedFeatureMemory()

high_concept_drift_memory.set_feature_value('ref_process_characteristics', {
    'activity_dist': {'A': 0.33, 'B': 0.33, 'C': 0.34},
    'dfr': {('A', 'B'), ('B', 'C')},
    'variants': {('A', 'B', 'C'): 0.8, ('A', 'C', 'B'): 0.2}
})

# Strict sequence - no activities appear in different orders
LOW_PARALLEL_ACTIVITY_RATIO = [
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": 6, "case:concept:name": "C2"}),

    Event({"concept:name": "A", "time:timestamp": 7, "case:concept:name": "C3"}),
    Event({"concept:name": "B", "time:timestamp": 8, "case:concept:name": "C3"}),
    Event({"concept:name": "C", "time:timestamp": 9, "case:concept:name": "C3"}),
]

# Many activities appear in different orders across traces
HIGH_PARALLEL_ACTIVITY_RATIO = [
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": 6, "case:concept:name": "C2"}),

    Event({"concept:name": "B", "time:timestamp": 7, "case:concept:name": "C3"}),
    Event({"concept:name": "A", "time:timestamp": 8, "case:concept:name": "C3"}),
    Event({"concept:name": "C", "time:timestamp": 9, "case:concept:name": "C3"}),

    Event({"concept:name": "C", "time:timestamp": 10, "case:concept:name": "C4"}),
    Event({"concept:name": "B", "time:timestamp": 11, "case:concept:name": "C4"}),
    Event({"concept:name": "A", "time:timestamp": 12, "case:concept:name": "C4"}),
]

# Highly variable activity durations
LOW_ACTIVITY_DURATION_STABILITY = [
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 0), "case:concept:name": "C1",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 5), "case:concept:name": "C1",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 0), "case:concept:name": "C2",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 12, 0), "case:concept:name": "C2",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 0), "case:concept:name": "C3",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 2), "case:concept:name": "C3",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 0), "case:concept:name": "C1",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 10), "case:concept:name": "C1",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 15, 0), "case:concept:name": "C2",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 16, 30), "case:concept:name": "C2",
           "lifecycle:transition": "complete"}),
]

# Consistent activity durations
HIGH_ACTIVITY_DURATION_STABILITY = [
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 0), "case:concept:name": "C1",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 10), "case:concept:name": "C1",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 0), "case:concept:name": "C2",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 10), "case:concept:name": "C2",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 12, 0), "case:concept:name": "C3",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 12, 10), "case:concept:name": "C3",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 0), "case:concept:name": "C1",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 30), "case:concept:name": "C1",
           "lifecycle:transition": "complete"}),

    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 0), "case:concept:name": "C2",
           "lifecycle:transition": "start"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 30), "case:concept:name": "C2",
           "lifecycle:transition": "complete"}),
]

# Consistent priorities, few changes
LOW_CASE_PRIORITY_DYNAMICS = [
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1", "priority": "high"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1", "priority": "high"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1", "priority": "high"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2", "priority": "medium"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2", "priority": "medium"}),
    Event({"concept:name": "C", "time:timestamp": 6, "case:concept:name": "C2", "priority": "medium"}),

    Event({"concept:name": "A", "time:timestamp": 7, "case:concept:name": "C3", "priority": "low"}),
    Event({"concept:name": "B", "time:timestamp": 8, "case:concept:name": "C3", "priority": "low"}),
    Event({"concept:name": "C", "time:timestamp": 9, "case:concept:name": "C3", "priority": "low"}),
]

# Frequent and varied priority changes
HIGH_CASE_PRIORITY_DYNAMICS = [
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1", "priority": "low"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1", "priority": "medium"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1", "priority": "high"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2", "priority": "high"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2", "priority": "low"}),
    Event({"concept:name": "C", "time:timestamp": 6, "case:concept:name": "C2", "priority": "medium"}),

    Event({"concept:name": "A", "time:timestamp": 7, "case:concept:name": "C3", "priority": "medium"}),
    Event({"concept:name": "B", "time:timestamp": 8, "case:concept:name": "C3", "priority": "high"}),
    Event({"concept:name": "C", "time:timestamp": 9, "case:concept:name": "C3", "priority": "low"}),

    Event({"concept:name": "A", "time:timestamp": 10, "case:concept:name": "C4", "priority": "high"}),
    Event({"concept:name": "B", "time:timestamp": 11, "case:concept:name": "C4", "priority": "medium"}),
    Event({"concept:name": "C", "time:timestamp": 12, "case:concept:name": "C4", "priority": "high"}),
]

# Current window matches reference model in memory
LOW_CONCEPT_DRIFT = [
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": 6, "case:concept:name": "C2"}),

    # Assume memory contains reference with similar distribution and sequences
]

# Current window significantly different from reference in memory
HIGH_CONCEPT_DRIFT = [
    Event({"concept:name": "X", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "Y", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "X", "time:timestamp": 3, "case:concept:name": "C1"}),

    Event({"concept:name": "Y", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "Z", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "W", "time:timestamp": 6, "case:concept:name": "C2"}),

    # Assume memory contains reference with activities A, B, C in different sequences
]

# Each activity only relates to the immediately preceding one
LOW_LONG_TERM_DEPENDENCIES = [
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": 4, "case:concept:name": "C2"}),
    Event({"concept:name": "B", "time:timestamp": 5, "case:concept:name": "C2"}),
    Event({"concept:name": "C", "time:timestamp": 6, "case:concept:name": "C2"}),

    Event({"concept:name": "A", "time:timestamp": 7, "case:concept:name": "C3"}),
    Event({"concept:name": "B", "time:timestamp": 8, "case:concept:name": "C3"}),
    Event({"concept:name": "C", "time:timestamp": 9, "case:concept:name": "C3"}),
]

# Strong patterns between distant activities
HIGH_LONG_TERM_DEPENDENCIES = [
    # Activity X always follows A after several other activities
    Event({"concept:name": "A", "time:timestamp": 1, "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": 2, "case:concept:name": "C1"}),
    Event({"concept:name": "C", "time:timestamp": 3, "case:concept:name": "C1"}),
    Event({"concept:name": "D", "time:timestamp": 4, "case:concept:name": "C1"}),
    Event({"concept:name": "X", "time:timestamp": 5, "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": 6, "case:concept:name": "C2"}),
    Event({"concept:name": "F", "time:timestamp": 7, "case:concept:name": "C2"}),
    Event({"concept:name": "G", "time:timestamp": 8, "case:concept:name": "C2"}),
    Event({"concept:name": "H", "time:timestamp": 9, "case:concept:name": "C2"}),
    Event({"concept:name": "X", "time:timestamp": 10, "case:concept:name": "C2"}),

    Event({"concept:name": "A", "time:timestamp": 11, "case:concept:name": "C3"}),
    Event({"concept:name": "J", "time:timestamp": 12, "case:concept:name": "C3"}),
    Event({"concept:name": "K", "time:timestamp": 13, "case:concept:name": "C3"}),
    Event({"concept:name": "L", "time:timestamp": 14, "case:concept:name": "C3"}),
    Event({"concept:name": "X", "time:timestamp": 15, "case:concept:name": "C3"}),
]

# Timing doesn't affect next activities
LOW_TEMPORAL_DEPENDENCY = [
    # Random timing, consistent next activities
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 0), "case:concept:name": "C1"}),
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 5), "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 0), "case:concept:name": "C2"}),
    Event({"concept:name": "B1", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 0), "case:concept:name": "C2"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 0), "case:concept:name": "C3"}),
    Event({"concept:name": "B2", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 1), "case:concept:name": "C3"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 15, 0), "case:concept:name": "C4"}),
    Event({"concept:name": "B3", "time:timestamp": datetime.datetime(2023, 1, 1, 17, 0), "case:concept:name": "C4"}),
]

# Timing strongly predicts next activities
HIGH_TEMPORAL_DEPENDENCY = [
    # Short gap always followed by activity X
    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 0), "case:concept:name": "C1"}),
    Event({"concept:name": "X", "time:timestamp": datetime.datetime(2023, 1, 1, 10, 5), "case:concept:name": "C1"}),

    Event({"concept:name": "A", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 0), "case:concept:name": "C2"}),
    Event({"concept:name": "X", "time:timestamp": datetime.datetime(2023, 1, 1, 11, 3), "case:concept:name": "C2"}),

    # Medium gap always followed by activity Y
    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 12, 0), "case:concept:name": "C1"}),
    Event({"concept:name": "X", "time:timestamp": datetime.datetime(2023, 1, 1, 13, 0), "case:concept:name": "C1"}),

    Event({"concept:name": "B", "time:timestamp": datetime.datetime(2023, 1, 1, 14, 0), "case:concept:name": "C2"}),
    Event({"concept:name": "X", "time:timestamp": datetime.datetime(2023, 1, 1, 15, 0), "case:concept:name": "C2"}),
]


def create_mock_previous_window():
    """Creates a simple event log that could be used as a previous window"""
    from pm4py.objects.log.obj import EventLog, Trace, Event

    mock_log = EventLog()

    # Create a few traces with A, B, C activities
    for i in range(5):
        trace = Trace()
        trace.attributes["concept:name"] = f"case_{i}"

        # Add A, B, C events
        event_a = Event()
        event_a["concept:name"] = "A"
        event_a["time:timestamp"] = datetime.datetime(2023, 1, 1, 9, i * 5)
        trace.append(event_a)

        event_b = Event()
        event_b["concept:name"] = "B"
        event_b["time:timestamp"] = datetime.datetime(2023, 1, 1, 9, i * 5 + 10)
        trace.append(event_b)

        event_c = Event()
        event_c["concept:name"] = "C"
        event_c["time:timestamp"] = datetime.datetime(2023, 1, 1, 9, i * 5 + 20)
        event_c["lifecycle:transition"] = "complete"
        trace.append(event_c)

        mock_log.append(trace)

    return mock_log

STRUCTURED_STREAM_FEATURES = [
    'temporal_dependency',
    'case_concurrency',
    'concept_stability',
    'case_throughput_stability',
    'parallel_activity_ratio',
    'activity_duration_stability',
    'case_priority_dynamics',
    'concept_drift',
    'long_term_dependencies',
]

COMBINED_TESTS: List[Tuple[str, List[Event], Optional[ComputedFeatureMemory], str]] = [
    ("Low Case Concurrency", LOW_CASE_CONCURRENCY, None, "case_concurrency"),
    ("High Case Concurrency", HIGH_CASE_CONCURRENCY, None,"case_concurrency"),
    ("Low Concept Stability", LOW_CONCEPT_STABILITY, low_concept_stability_memory, "concept_stability"),
    ("High Concept Stability", HIGH_CONCEPT_STABILITY, high_concept_stability_memory, "concept_stability"),
    ("Low Case Throughput Stability", LOW_CASE_THROUGHPUT_STABILITY, low_concept_drift_memory, "case_throughput_stability"),
    ("High Case Throughput Stability", HIGH_CASE_THROUGHPUT_STABILITY, high_concept_drift_memory, "case_throughput_stability"),
    ("Low Parallel Activity Ratio", LOW_PARALLEL_ACTIVITY_RATIO, None,"parallel_activity_ratio"),
    ("High Parallel Activity Ratio", HIGH_PARALLEL_ACTIVITY_RATIO, None,"parallel_activity_ratio"),
    ("Low Activity Duration Stability", LOW_ACTIVITY_DURATION_STABILITY, None,"activity_duration_stability"),
    ("High Activity Duration Stability", HIGH_ACTIVITY_DURATION_STABILITY, None,"activity_duration_stability"),
    ("Low Case Priority Dynamics", LOW_CASE_PRIORITY_DYNAMICS, None,"case_priority_dynamics"),
    ("High Case Priority Dynamics", HIGH_CASE_PRIORITY_DYNAMICS, None,"case_priority_dynamics"),
    ("Low Concept Drift", LOW_CONCEPT_DRIFT, low_concept_drift_memory,"concept_drift"),
    ("High Concept Drift", HIGH_CONCEPT_DRIFT, high_concept_drift_memory,"concept_drift"),
    ("Low Long Term Dependencies", LOW_LONG_TERM_DEPENDENCIES, None,"long_term_dependencies"),
    ("High Long Term Dependencies", HIGH_LONG_TERM_DEPENDENCIES, None,"long_term_dependencies"),
    ("Low Temporal Dependency", LOW_TEMPORAL_DEPENDENCY, None,"temporal_dependency"),
    ("High Temporal Dependency", HIGH_TEMPORAL_DEPENDENCY, None,"temporal_dependency")
]
