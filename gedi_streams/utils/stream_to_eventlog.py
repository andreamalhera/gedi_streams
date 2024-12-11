from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py import write_xes
import ast

def convert_to_eventlog(input_data: list, output_path: str="") -> EventLog:
    # Step 1: Parse the strings into dictionaries
    event_list = []
    for line in input_data:
        if line.strip():  # Ignore empty lines
            # Extract the dictionary part from the string
            sensor_prefix, dict_part = line.split(": ", 1)
            # Convert the string representation of the dictionary into an actual dictionary
            event_data = ast.literal_eval(dict_part.strip())
             # Replace 'activity' with 'concept:name' to align with PM4Py
            event_data["concept:name"] = event_data.pop("activity")
            # Replace 'timestamp' with 'time:timestamp' to align with PM4Py
            event_data["time:timestamp"] = event_data.pop("timestamp")
            # Append to the list of events
            event_list.append(event_data)

    # Step 2: Organize events by `caseid`
    caseid_to_events = {}
    for event in event_list:
        caseid = event["caseid"]
        if caseid not in caseid_to_events:
            caseid_to_events[caseid] = []
        caseid_to_events[caseid].append(event)

    # Step 3: Convert to PM4Py EventLog
    event_log = EventLog()

    for caseid, events in caseid_to_events.items():
        trace = Trace()
        trace.attributes["concept:name"] = caseid  # Assign trace ID
        for event_data in events:
            event = Event(event_data)  # Create Event object
            trace.append(event)        # Add event to trace
        event_log.append(trace)        # Add trace to EventLog

    if output_path:
        write_xes(event_log, output_path)
        print(f"      SUCCESS: Saved event log to {output_path}")
    return event_log

