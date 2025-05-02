import pm4py.objects.log.obj
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py import write_xes
from toolz import groupby


def window_to_eventlog(input_data: list[Event]) -> EventLog:
    # Step 1: Parse the strings into dictionaries
    event_list: list[Event] = input_data

    event_log = EventLog()
    ev: Event = event_list[0]
    event_list_trace = groupby(lambda x: x["case:concept:name"], event_list)

    for key, value in event_list_trace.items():
        trace = Trace()
        trace.attributes["concept:name"] = key
        for ev in value:
            trace.append(ev)
        event_log.append(trace)

    return event_log

