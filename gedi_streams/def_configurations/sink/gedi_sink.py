from multiprocessing import Queue
from typing import List

from process_mining_core.datastructure.core.event import Event as PMCEvent
from pm4py.objects.log.obj import Event as PM4PYEvent

from distributed_event_factory.provider.sink.sink_provider import Sink, SinkProvider


def translate_event_to_event(event: PMCEvent) -> PM4PYEvent:
    event_ref: PM4PYEvent = PM4PYEvent()
    event_ref["concept:name"] = event.get_activity()
    event_ref["time:timestamp"] = event.get_timestamp()
    event_ref["case:concept:name"] = event.get_case()

    event_ref["attr:node"] = event.node
    event_ref["attr:group"] = event.group
    return event_ref

class GEDIAdapter(Sink):

    def __init__(self, id: str, data_source_ref: List[str], queue: Queue, disable_console_print: bool = True):
        super().__init__(data_source_ref)
        self.id: str = id
        self.queue: Queue = queue
        self.disable_console_print: bool = disable_console_print

    def send(self, event: PMCEvent) -> None:
        # Write the event to the GEDI system (e.g. via Queue)
        pm4py_event: PM4PYEvent = translate_event_to_event(event)
        self.queue.put(pm4py_event)

        if not self.disable_console_print:
            print(f"[{event.node}] -> ({event.get_activity()} : {event.get_case()} : {event.get_timestamp()})")

    def start_timeframe(self):

        pass

    def end_timeframe(self):
        pass

# class GEDISinkProvider(SinkProvider):
#     def get_sender(self, id) -> Sink:
#         return GEDIAdapter(id)
