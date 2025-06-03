#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"stream_features_\["
    r"([a-zA-Z_]+)=([0-9]+\.[0-9]+)__"
    r"([a-zA-Z_]+)=([0-9]+\.[0-9]+)"
    r"\]_(\d{8}_\d{6})\.csv"
)


def list_csv_files(directory: Path) -> List[Path]:
    """
    List all CSV files in a given directory.

    :param parma directory: Path to the directory containing CSV files.
    :return : of objects.
    :return: A list of Paths for each CSV file in the directory.
    """
    return [p for p in directory.iterdir() if p.is_file() and p.suffix == ".csv"]


def parse_filename(path: Path) -> Tuple[Tuple[str, float, str, float], str]:
    """
    Parse a filename into its feature tuple and timestamp.

    :param parma path: Path object representing the CSV file.
    :return : of objects.
    :return: A tuple where the first element is a 4-tuple
             (feature1, value1, feature2, value2) and the
             second element is the timestamp string.
    """
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match pattern: {path.name}")
    feat1, val1, feat2, val2, ts = m.groups()
    return (feat1, float(val1), feat2, float(val2)), ts


def find_duplicate_newer_files(
    files: List[Path]
) -> List[Path]:
    """
    Identify newer duplicates among a list of CSV files
    based on matching feature sets.

    :param parma files: List of CSV file Paths to analyze.
    :return : of objects.
    :return: List of Paths corresponding to the newer
             duplicate files.
    """
    groups: Dict[Tuple[str, float, str, float], List[Tuple[str, Path]]] = {}

    for f in files:
        key, ts = parse_filename(f)
        groups.setdefault(key, []).append((ts, f))

    duplicates: List[Path] = []
    for entries in groups.values():
        if len(entries) <= 1:
            continue

        # sort by timestamp ascending
        entries.sort(key=lambda x: x[0])
        # all but the earliest are duplicates
        dupes = [f for _, f in entries[1:]]
        duplicates.extend(dupes)

    return duplicates


def main() -> None:
    """
    Entry point: parse arguments, scan directory,
    and print duplicate newer files.

    :return : of objects.
    :return: None.
    """

    directory = Path(os.getcwd()+"/output/")

    csv_files = list_csv_files(directory)
    duplicates = find_duplicate_newer_files(csv_files)
    print(len(duplicates))
    if duplicates:
        print("Newer duplicate files:")
        for dup in duplicates:
            print(f"  {dup.name}")
            os.remove(dup)
    else:
        print("No newer duplicates found.")


if __name__ == "__main__":
    main()
