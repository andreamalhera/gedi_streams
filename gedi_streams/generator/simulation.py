import os
import sys
from pathlib import Path
from queue import Queue
from typing import List, Optional

import pm4py
from ConfigSpace import ConfigurationSpace
from pm4py import ProcessTree, play_out
from pm4py.objects.log.obj import EventLog

from distributed_event_factory.core.datasource import GenericDataSource
from gedi_streams.def_configurations.sink.gedi_sink import GEDIAdapter
from DistributedEventFactory.distributed_event_factory.event_factory import EventFactory
from gedi_streams.def_configurations.utils.def_utils import init_and_compile_def_using_markov_chain, \
     visualize_markov_chain, process_tree_to_markov_chain
from gedi_streams.generator.model import create_PTLG

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



def play_DEFact(
        config: ConfigurationSpace,
        queue: Queue,
        print_events: bool=True,
        visualize: bool=True
):

    tree: pm4py.ProcessTree = create_PTLG(config.sample_configuration())
    print(tree)
    event_log: EventLog = play_out(tree)
    start_nodes: List[str] = list(set([trace[0]["concept:name"] for trace in event_log]))
    end_nodes: List[str] = list(set([trace[-1]["concept:name"] for trace in event_log]))

    # stoc: dict[list[str], float] = pm4py.get_stochastic_language(tree)
    # markov_chain = stochastic_language_to_markov_chain(stoc)
    markov_chain = process_tree_to_markov_chain(tree, 3)

    if visualize:
        # pm4py.view_petri_net(*pm4py.convert_to_petri_net(tree))
        visualize_markov_chain(markov_chain)

    def_instance: EventFactory = init_and_compile_def_using_markov_chain(
        markov_chain,
        start_nodes,
        end_nodes,
        submodule_path,
        queue,
        print_events
    )

    def_instance.run()
