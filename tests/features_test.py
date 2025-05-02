from typing import List, Optional

import pm4py
from graphviz import Digraph
from matplotlib import image as mpimg, pyplot as plt
from matplotlib.table import Table
from pm4py.objects.log.obj import Event, EventLog, Trace
from gedi_streams.features.memory import ComputedFeatureMemory
from gedi_streams.features.stream_feature import StructuredStreamFeature
from tests.feature_test_cases import COMBINED_TESTS


def create_event_log(events_list: List[Event]) -> EventLog:
    log = EventLog()
    case_traces = {}

    for event_dict in events_list:
        case_id = event_dict["case:concept:name"]

        if case_id not in case_traces:
            trace = Trace()
            trace.attributes["concept:name"] = case_id
            case_traces[case_id] = trace
            log.append(trace)

        event = Event()
        for key, value in event_dict.items():
            if key != "case:concept:name":  # Don't add case ID to event attributes
                event[key] = value

        case_traces[case_id].append(event)

    return log


def visualize_process_model(
    event_log: EventLog,
    feature_results: dict[str, float],
    label: str,
    mark_cell: Optional[str] = None
) -> None:
    """
    Visualizes a Petri net and displays the feature results as a table below the image.

    :param event_log: The event log to visualize.
    :param feature_results: A dictionary of computed feature values.
    :param label: A label to annotate the Petri net.
    :param mark_cell: The feature name to highlight in the table.
    :return: None.
    """
    net, im, fm = pm4py.discover_petri_net_inductive(event_log)
    digraph: Digraph = pm4py.view_petri_net(net, im, fm)
    digraph.render('petri_net', format='png', cleanup=True)

    img = mpimg.imread('petri_net.png')

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})

    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title(f"Process Model for {label}", fontsize=14)

    axs[1].axis('off')
    rows = [[k, f"{v:.4f}"] for k, v in feature_results.items()]
    table: Table = axs[1].table(
        cellText=rows,
        colLabels=["Feature", "Value"],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    if mark_cell is not None:
        for row_idx, (feature, _) in enumerate(rows, start=1):  # +1 to skip header row
            if feature == mark_cell:
                for col_idx in range(2):
                    table[(row_idx, col_idx)].set_facecolor('#ffcccb')  # light red

    fig.tight_layout()
    plt.show()



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

for (label, el, memory, key) in COMBINED_TESTS:

    event_log: EventLog = create_event_log(el)

    if memory is None:
        computed_memory =  ComputedFeatureMemory()
    else:
        computed_memory = memory

    structured_stream_features = StructuredStreamFeature()
    feature_results: dict[str, float] = {}

    for feature in STRUCTURED_STREAM_FEATURES:
        result = structured_stream_features[feature](event_log, computed_memory)
        feature_results[feature] = result

    visualize_process_model(event_log, feature_results, label, key)
