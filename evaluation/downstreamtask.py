from typing import Dict, List
import pm4py
import pandas as pd

def discover_petri_net():
    df = pd.read_csv('result_downstream_task_dataset.csv', parse_dates=['time:timestamp'])

    print(df.head())

    pn, im,fm = pm4py.discover_petri_net_heuristics(df, dependency_threshold=0.9, and_threshold = 0.85, loop_two_threshold = 0.7)

    pm4py.view_petri_net(pn, im, fm)


def generate_data(): 
    def sort_by_case_and_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy["time:timestamp"]):
            df_copy["time:timestamp"] = pd.to_datetime(df_copy["time:timestamp"])
        df_sorted: pd.DataFrame = df_copy.sort_values(
            by=["case:concept:name", "time:timestamp"]
        ).reset_index(drop=True)
        return df_sorted

    def replace_concept_name_by_dict(
            df: pd.DataFrame, mapping: Dict[str, str]
    ) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        def _replace(value: str) -> str:
            for key, new_val in mapping.items():
                if key in value:
                    return new_val
            return value

        df_copy["concept:name"] = df_copy["concept:name"].apply(_replace)
        return df_copy

    df = pd.read_csv('result_downstream_task_dataset.csv', parse_dates=['time:timestamp'])
    df = df[df["livecycle:type"] != "start"].reset_index(drop=True)

    df = sort_by_case_and_timestamp(df)

    mask = ~df["case:concept:name"].str.contains("fractal", case=False, na=False)
    df =  df[mask].reset_index(drop=True)


    mapping = {
        "Process": "P",
        "Review": "R",
        "Analyze": "A",
        "Check": "C",
        "Validate": "V",
        "Submit": "S",
        "Approve": "Ap",
        "Execute": "E",
        "Monitor": "M",
        "Report": "Re",
        "Update": "U",
        "Create": "Cr",
        "Verify": "Ve",
        "Send": "Se",
        "Receive": "Re",
        "Transform": "T",
        "Calculate": "Ca"
    }

    df = replace_concept_name_by_dict(df, mapping)

    df.to_csv('result_downstream_task_dataset clean.csv', index=False)

if __name__ == "__main__":
    # generate_data()
    discover_petri_net()