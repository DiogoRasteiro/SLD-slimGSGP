"""Cleaned, binary-labelled per-task data for the classification experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.core.schema import (
    BINARY_POSITIVE_LABEL,
    BINARY_TARGET_COLUMN,
    KINEMATIC_FEATURES,
    TARGET_COLUMN,
)
from src.data.loader import load_task
from src.eda.io import measurement_masks


@dataclass(frozen=True)
class CleaningReport:
    cohort: str
    task: int
    raw_rows: int
    absent_rows: int
    all_zero_rows: int
    kept_rows: int
    positive_rows: int

    @property
    def positive_rate(self) -> float:
        return self.positive_rows / self.kept_rows if self.kept_rows else float("nan")


def load_binary_task(
    cohort: str,
    task: int,
    data_dir: Optional[Union[str, Path]] = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop unlabelled and all-zero rows, then add the binary target column.

    All-zero rows are failed recordings; in 1year some of them carry a valid
    label, so filtering ``Label == -1`` alone does not remove them.
    """
    raw = load_task(cohort, task, data_dir=data_dir, filter_absent=False)
    absent = raw[TARGET_COLUMN].eq(-1)
    all_zero = measurement_masks(raw[KINEMATIC_FEATURES])["all_zero_measurements"]
    df = raw[~absent & ~all_zero].reset_index(drop=True)
    if df[KINEMATIC_FEATURES].isna().any().any():
        raise ValueError(
            f"{cohort} Task {task}: missing feature values after cleaning; "
            "the pipeline does not impute."
        )
    df[BINARY_TARGET_COLUMN] = df[TARGET_COLUMN].eq(BINARY_POSITIVE_LABEL).astype(int)
    report = CleaningReport(
        cohort=cohort,
        task=task,
        raw_rows=len(raw),
        absent_rows=int(absent.sum()),
        all_zero_rows=int((all_zero & ~absent).sum()),
        kept_rows=len(df),
        positive_rows=int(df[BINARY_TARGET_COLUMN].sum()),
    )
    return df, report


def features_and_target(df: pd.DataFrame):
    """Only the 30 kinematic features are model inputs; demographics are excluded."""
    return (
        df[KINEMATIC_FEATURES].to_numpy(dtype=float),
        df[BINARY_TARGET_COLUMN].to_numpy(dtype=int),
    )
