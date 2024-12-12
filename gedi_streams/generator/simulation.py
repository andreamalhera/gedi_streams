import os
import sys

# Add the submodule to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
submodule_path = os.path.join(project_root, "DistributedEventFactory")
sys.path.append(submodule_path)
from DistributedEventFactory.distributed_event_factory.event_factory import EventFactory

class QueueOutput:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        # Flush is needed to avoid warnings about buffered output
        pass

def play_DEFact(model=None, config=None, queue=None):
    if queue is None:
        ##TODO: Play out a DEFact stream here.
        pass
    else:
        sys.stdout = QueueOutput(queue)
        event_factory = EventFactory()
        (event_factory
        .add_directory("DistributedEventFactory/config/datasource/assemblyline/")
        .add_file("DistributedEventFactory/config/simulation/stream.yaml")
        .add_file("DistributedEventFactory/config/sink/console-sink.yaml")
        ).run()
