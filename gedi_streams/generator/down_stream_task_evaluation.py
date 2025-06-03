import datetime
from typing import Optional

from pm4py.objects.log.obj import Event
from pybeamline.algorithms.discovery.heuristics_miner_lossy_counting import HeuristicsMinerLossyCounting
from pybeamline.algorithms.discovery.heuristics_miner_lossy_counting_budget import HeuristicsMinerLossyCountingBudget
from pybeamline.bevent import BEvent

hmlc = HeuristicsMinerLossyCounting()
hmlcb = HeuristicsMinerLossyCountingBudget()


def pm4py_event_to_bevent(event: Event) -> Optional[BEvent]:
    """
    Convert a pm4py event to a BEVENT event
    :param event: pm4py event
    :return: BEVENT event
    """

    bevent = BEvent(
        event["concept:name"],
        event["case:concept:name"],
        "",
        datetime.datetime.fromisoformat(event["time:timestamp"]),
    )
    return bevent

def discovery_algorithm(event: Event):
    """
    Apply the Heuristics Miner algorithm to the event
    :param event: pm4py event
    :return: BEVENT event
    """
    bevent = pm4py_event_to_bevent(event)
    hmlc.ingest_event(bevent)
    hmlcb.ingest_event(bevent)


def get_models():
    """
    Get the models from the Heuristics Miner algorithm
    :return: models
    """
    hmlc_model = hmlc.get_model()
    hmlcb_model = hmlcb.get_model()
    return {"HeuristicsMinerLossyCounting":hmlc_model, "HeuristicsMinerLossyCountingBudget": hmlcb_model}
