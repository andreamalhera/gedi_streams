import os
import glob
import argparse
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

FONT_SIZE: int = 18

def load_csv_files(folder_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the specified folder.

    :param folder_path: Path to the folder containing CSV files.
    :return: Dictionary mapping each filename (without extension) to its DataFrame.
    """
    pattern = os.path.join(folder_path, '*.csv')
    files = glob.glob(pattern)
    result: Dict[str, pd.DataFrame] = {}
    for filepath in files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        result[name] = pd.read_csv(filepath)
    return result


def calculate_f1_average(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the per-row F1 score using average fitness and average precision.

    :param df: DataFrame containing columns whose names start with 'fitness' and 'precision'.
    :return: Series of F1 scores.
    """
    fitness_cols = [c for c in df.columns if c.startswith('fitness')]
    precision_cols = [c for c in df.columns if c.startswith('precision')]
    avg_fitness = df[fitness_cols].mean(axis=1)
    avg_precision = df[precision_cols].mean(axis=1)
    numerator = 2 * avg_precision * avg_fitness
    denominator = avg_precision + avg_fitness
    f1 = numerator.div(denominator).fillna(0)
    return f1


def plot_f1_scores(f1_scores: Dict[str, pd.Series], max_x: int) -> None:
    """
    Plot F1 score timelines for multiple CSV runs.

    :param f1_scores: Mapping from run label to its F1 score Series.
    :param max_x: Maximum X-axis value (number of points to plot).
    :return: None.
    """
    x = list(range(1, max_x + 1))
    plt.figure(figsize=(12, 4))
    for label, series in f1_scores.items():
        y = series.tolist()[:max_x]
        plt.plot(x, y, label=label)
    plt.xlabel('Window Index', fontsize=FONT_SIZE)
    plt.ylabel('F1 Score', fontsize=FONT_SIZE)

    # change the font size of the labels
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    # change the font size of the legend

    plt.xlim(1, max_x)
    plt.legend( prop={'size': FONT_SIZE})
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Parse command-line arguments, load CSVs, compute F1 scores, and plot them.

    :return: None.
    """
    data_frames = load_csv_files("concurrency")
    f1_scores = {
        name: calculate_f1_average(df)
        for name, df in data_frames.items()
    }

    first_10_avg = pd.concat([scores.iloc[:10] for scores in f1_scores.values()]).mean()
    next_10_avg = pd.concat([scores.iloc[10:20] for scores in f1_scores.values()]).mean()

    print(f"Average F1 Score for the first 10 rows: {first_10_avg:.4f}")
    print(f"Average F1 Score for the next 10 rows: {next_10_avg:.4f}")

    plot_f1_scores(f1_scores, 20)


if __name__ == '__main__':
    main()
