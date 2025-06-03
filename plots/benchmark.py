from __future__ import annotations

import pandas as pd
import pm4py
from typing import Any, Dict, List

from pm4py.objects.log.obj import Event
from process_mining_core.datastructure.core.model.heuristics_net import HeuristicsNet
from gedi_streams.generator.down_stream_task_evaluation import (
    discovery_algorithm,
    get_models,
)

DATA_REPO_PATH: str = "/Users/christianimenkamp/Documents/Data-Repository/Community/"
LOGS: List[str] = [
    "bpi-c-2012",
    "bpi-c-2013",
    "bpi-c-2015",
    "bpi-c-2017",
    "bpi-c-2019",
    "daily_living",
    "hospital-billing",
    "Road-Traffic-Fine-Management-Process",
    "sepsis",
]
WINDOW_SIZE: int = 100
MAX_WINDOWS: int = 10


def _evaluate_window(log_slice: pd.DataFrame, log_eval: pd.DataFrame) -> Dict[str, float]:
    """
    Compute fitness values for a single window.

    :param log_slice: Window of length ``WINDOW_SIZE`` taken from the full log.
    :param log_eval: Trimmed sample (first 2 000 events) used for fitness replay.
    :return: Flattened fitness metrics for the two discovery algorithms.
    """
    for _, row in log_slice.iterrows():
        event = Event()
        event["case:concept:name"] = row["case:concept:name"]
        event["concept:name"] = row["concept:name"]
        event["time:timestamp"] = str(row["time:timestamp"])
        if "lifecycle:transition" in row:
            event["lifecycle:transition"] = row["lifecycle:transition"]
        discovery_algorithm(event)

    models: Dict[str, HeuristicsNet] = get_models()
    hmlcb_net: HeuristicsNet = models["HeuristicsMinerLossyCountingBudget"]
    hmlc_net: HeuristicsNet = models["HeuristicsMinerLossyCounting"]

    hmlcb_pn = pm4py.convert_to_petri_net(hmlcb_net)
    hmlc_pn = pm4py.convert_to_petri_net(hmlc_net)

    hmlcb_res: Dict[str, float] = pm4py.fitness_token_based_replay(log_eval, *hmlcb_pn)
    hmlc_res: Dict[str, float] = pm4py.fitness_token_based_replay(log_eval, *hmlc_pn)

    flat: Dict[str, float] = {f"hmlcb_{k}": v for k, v in hmlcb_res.items()}
    flat.update({f"hmlc_{k}": v for k, v in hmlc_res.items()})
    return flat


def evaluate_log(log_name: str) -> pd.DataFrame:
    """
    Split one log into non-overlapping windows and compute fitness
    for the first ``MAX_WINDOWS`` windows.

    :param log_name: Name of the log folder inside ``DATA_REPO_PATH``.
    :return: DataFrame with exactly ``MAX_WINDOWS`` rows (one per window).
    """
    log: pd.DataFrame = pd.read_feather(f"{DATA_REPO_PATH}{log_name}/data.feather")
    log_eval: pd.DataFrame = log.iloc[:2000]

    rows: List[Dict[str, Any]] = []
    for window_idx, start in enumerate(range(0, len(log), WINDOW_SIZE)):
        if window_idx >= MAX_WINDOWS:
            break
        end: int = start + WINDOW_SIZE
        batch: pd.DataFrame = log.iloc[start:end]

        metrics: Dict[str, float] = _evaluate_window(batch, log_eval)
        rows.append(
            {
                "log": log_name,
                "window": window_idx,
                "start": start,
                "end": end,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """
    Produce a CSV with ten rows per dataset in ``static_analysis_benchmark/``.
    """
    for log_name in LOGS:
        print(f"Processing log: {log_name}")
        df: pd.DataFrame = evaluate_log(log_name)
        df.to_csv(f"static_analysis_benchmark/{log_name}.csv", index=False)


if __name__ == "__main__":
    main()
