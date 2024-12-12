import multiprocessing
import os
from pathlib import Path
from typing import List, Optional
from multiprocessing import Process, Queue
from distributed_event_factory.core.datasource import GenericDataSource
from distributed_event_factory.event_factory import EventFactory
from gedi_streams.def_configurations.sink.gedi_sink import GEDIAdapter
from process_mining_core.datastructure.core.event import Event

CWD: str = str(Path(__file__).parent.parent)

def play_DEFact_temp(queue: Optional[Queue]=None) -> None:
    if queue is None:
        print("Queue is None")
        return None
    event_factory = EventFactory()

    data_sources: List[GenericDataSource] = []
    # data_source_ref: List[str] = list(map(lambda x: x.name, data_sources))
    data_source_ref = ["GoodsDelivery", "MaterialPreparation", "AssemblyLineSetup", "Assembling", "QualityControl",
                       "Packaging", "Shipping"]

    print("added sink")

    event_factory.add_sink("gedi-sink", GEDIAdapter("gedi-sink", data_source_ref, queue, disable_console_print=True))

    (event_factory
     .add_directory(f"{CWD}/DistributedEventFactory/config/datasource/assemblyline")
     .add_file(f"{CWD}/DistributedEventFactory/config/simulation/stream.yaml")
     ).run()

if __name__ == "__main__":
    output_queue: Queue = Queue(maxsize=1000)

    p1: Process = Process(target=play_DEFact_temp, kwargs={'queue': output_queue})
    p1.start()

    try:
        while True: # Track the windows and max windows to create an end condition

            data = output_queue.get()
            print("Received data:", data)

    except KeyboardInterrupt:
        # Handle graceful shutdown
        print("\nShutting down...")
        p1.terminate()  # Stop the simulation process
        p1.join()
        print("Terminated Gracefully")