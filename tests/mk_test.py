import pandas as pd
import pm4py
from pm4py import generate_process_tree, ProcessTree
from typing import Dict, Tuple, List

from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import Operator

from gedi_streams.def_configurations.utils.def_utils import visualize_markov_chain, build_markov_chain

from typing import Dict, Tuple, List

tree: pm4py.ProcessTree = generate_process_tree(parameters={
    "min": 5,
    "max": 10,
    "mode": 10 ,
    "sequence": 0.5,
    "choice": 0.0,
    "parallel": 0,
    "loop" :0.0 ,
    "silent" :0.0 ,
    "lt_dependency" :0.0 ,
    "duplicate": 0.0,
    "or": 0,
    "no_models": 1
})

pm4py.convert_to_petri_net(tree)
temp = pm4py.play_out(tree, parameters={"num_traces": 100})
case_id: int = 0

for trace in temp:
    for event in trace:
        event["case:concept:name"] = case_id
    case_id += 1


el: pd.DataFrame = pm4py.convert_to_dataframe(temp)

print(f"INFO: Generated Event Log: Distinct Activities: {el["concept:name"].nunique()}, Num Events: {len(el)}, Num Cases: {el["case:concept:name"].nunique()}")

markov_chain = build_markov_chain(el)

pm4py.view_process_tree(tree)
visualize_markov_chain(markov_chain)