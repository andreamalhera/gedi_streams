import os
import sys
from pathlib import Path
from queue import Queue
from typing import List, Optional

from distributed_event_factory.core.datasource import GenericDataSource
from gedi_streams.def_configurations.sink.gedi_sink import GEDIAdapter
from DistributedEventFactory.distributed_event_factory.event_factory import EventFactory

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
submodule_path = os.path.join(project_root, "DistributedEventFactory")
sys.path.append(submodule_path)


class QueueOutput:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        # Flush is needed to avoid warnings about buffered output
        pass



def play_DEFact(model=None, config=None, queue: Optional[Queue]=None, print_events: bool=True):
    if queue is None:
        print("Queue is None")
        return None
    event_factory = EventFactory()

    data_sources: List[GenericDataSource] = []
    # data_source_ref: List[str] = list(map(lambda x: x.name, data_sources))
    data_source_ref = ["GoodsDelivery", "MaterialPreparation", "AssemblyLineSetup", "Assembling", "QualityControl",
                       "Packaging", "Shipping"]

    event_factory.add_sink("gedi-sink", GEDIAdapter("gedi-sink", data_source_ref, queue, disable_console_print=not print_events))
    (event_factory
     .add_directory(f"{submodule_path}/config/datasource/assemblyline")
     .add_file(f"{submodule_path}/config/simulation/stream.yaml")
     ).run()
