from multiprocessing import Queue
from typing import List

from process_mining_core.datastructure.core.event import Event

from distributed_event_factory.provider.sink.sink_provider import Sink, SinkProvider


class GEDIAdapter(Sink):

    def __init__(self, id: str, data_source_ref: List[str], queue: Queue, disable_console_print: bool = True):
        super().__init__(data_source_ref)
        self.id: str = id
        self.queue: Queue = queue
        self.disable_console_print: bool = disable_console_print

    def send(self, event: Event) -> None:
        # Write the event to the GEDI system (e.g. via Queue)
        self.queue.put(event)

        if not self.disable_console_print:
            print(f"Sensor ({event.node}): " + ": " + str(event))

    def start_timeframe(self):

        pass

    def end_timeframe(self):
        pass

# class GEDISinkProvider(SinkProvider):
#     def get_sender(self, id) -> Sink:
#         return GEDIAdapter(id)
