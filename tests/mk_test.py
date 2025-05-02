from collections import defaultdict
from typing import Dict, Tuple

import pm4py
from pm4py import generate_process_tree, ProcessTree
from pm4py.objects.process_tree.obj import Operator

from gedi_streams.def_configurations.utils.def_utils import stochastic_language_to_markov_chain


# tree: pm4py.ProcessTree = generate_process_tree()

def simple_sequence_tree() -> ProcessTree:
    """
    a -> b -> c
    """

    tree_ref = ProcessTree(operator=Operator.SEQUENCE)

    node_a = ProcessTree(label="a", parent=tree_ref)
    node_b = ProcessTree(label="b", parent=tree_ref)
    node_c = ProcessTree(label="c", parent=tree_ref)

    tree_ref.children.append(node_a)
    tree_ref.children.append(node_b)
    tree_ref.children.append(node_c)

    return tree_ref

tree = simple_sequence_tree()

pm4py.view_process_tree(tree)

stoc: dict[list[str], float] = pm4py.get_stochastic_language(tree)

markov_chain = stochastic_language_to_markov_chain(stoc)

print(markov_chain)