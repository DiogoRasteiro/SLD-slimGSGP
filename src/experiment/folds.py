"""Subject-level outer CV folds, fixed in advance and shared by the chosen tasks."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from src.core.schema import (
    BINARY_TARGET_COLUMN,
    CV_FOLD_SEED_CANDIDATES,
    CV_OUTER_FOLDS,
    FOLDS_DIR_NAME,
)
from src.eda.splits import candidate_assignment, split_diagnostics
from src.experiment.dataset import load_binary_task

FOLD_COLUMN = "fold"


@dataclass
class FoldAssignment:
    cohort: str
    tasks: tuple[int, ...]
    n_folds: int
    seed: int
    folds: pd.Series  # index: subject Id (cohort-local), values: 0..n_folds-1
    diagnostics: pd.DataFrame

    @property
    def run_stem(self) -> str:
        """Fold identity for results folders, so runs on different folds never share one."""
        return f"{fold_stem(self.cohort, self.tasks, self.n_folds)}_fs{self.seed}"

    def check_matches(self, cohort: str, tasks: Sequence[int]) -> None:
        if cohort != self.cohort or tuple(sorted(tasks)) != self.tasks:
            raise ValueError(
                f"Fold file was built for {self.cohort} tasks {list(self.tasks)}, "
                f"not {cohort} tasks {sorted(tasks)}. Rebuild the folds."
            )


def binary_label_matrix(
    cohort: str, tasks: Sequence[int], data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Subjects x tasks with values 0/1, NaN where a child has no usable row."""
    columns = {}
    for task in tasks:
        df, _ = load_binary_task(cohort, task, data_dir=data_dir)
        columns[f"Task_{task}"] = df.set_index("Id")[BINARY_TARGET_COLUMN]
    matrix = pd.DataFrame(columns).sort_index()
    matrix.index.name = "Id"
    return matrix


def _worst_positive_gap(diagnostics: pd.DataFrame) -> float:
    positive = diagnostics[diagnostics["label"].eq(1)]
    return float(positive["test_proportion_difference"].abs().max())


def build_folds(
    cohort: str,
    tasks: Sequence[int],
    n_folds: int = CV_OUTER_FOLDS,
    seeds: Sequence[int] = CV_FOLD_SEED_CANDIDATES,
    data_dir: Optional[Union[str, Path]] = None,
) -> FoldAssignment:
    """Pick, among candidate seeds, the assignment with the best positive-rate balance.

    Reuses the EDA allocator, which stratifies on every task's class indicators at
    once. With binary values the label-2 indicators are empty and simply ignored.
    Selection uses balance only, never model results.
    """
    matrix = binary_label_matrix(cohort, tasks, data_dir=data_dir)
    best = None
    for seed in seeds:
        assignment = candidate_assignment(matrix, n_folds, seed)
        diagnostics = split_diagnostics(matrix, assignment, n_folds)
        diagnostics = diagnostics[diagnostics["label"].isin([0, 1])]
        if diagnostics["test_class_missing"].any():
            continue
        gap = _worst_positive_gap(diagnostics)
        if best is None or gap < best[0]:
            best = (gap, seed, assignment, diagnostics)
    if best is None:
        raise ValueError("No candidate seed gives both classes in every test fold.")
    _, seed, assignment, diagnostics = best
    return FoldAssignment(
        cohort=cohort,
        tasks=tuple(sorted(tasks)),
        n_folds=n_folds,
        seed=seed,
        folds=assignment.rename(FOLD_COLUMN),
        diagnostics=diagnostics.reset_index(drop=True),
    )


def fold_balance(
    assignment: FoldAssignment, data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Per task and fold: test/train sizes and positive rates, recomputed from the data.

    Works on any assignment, including one loaded from disk, and does not reuse the
    allocator's diagnostics, so it is an independent check of what the folds contain.
    Reports only; it applies no limits.
    """
    rows = []
    for task in assignment.tasks:
        df, _ = load_binary_task(assignment.cohort, task, data_dir=data_dir)
        fold = df["Id"].map(assignment.folds)
        if fold.isna().any():
            raise ValueError(f"Task {task}: some children have no fold. Rebuild the folds.")
        y = df[BINARY_TARGET_COLUMN]
        task_rate = y.mean()
        for k in range(assignment.n_folds):
            test = fold.eq(k)
            test_rate = y[test].mean()
            rows.append(dict(
                task=task,
                fold=k,
                children=int(test.sum()),
                positives=int(y[test].sum()),
                test_rate=test_rate,
                train_rate=y[~test].mean(),
                task_rate=task_rate,
                gap_pts=100 * (test_rate - task_rate),
            ))
    return pd.DataFrame(rows)


def save_folds(assignment: FoldAssignment, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = assignment.folds.reset_index()
    frame.insert(0, "Cohort", assignment.cohort)
    frame.to_csv(path, index=False)
    meta = dict(
        cohort=assignment.cohort,
        tasks=list(assignment.tasks),
        n_folds=assignment.n_folds,
        seed=assignment.seed,
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_folds(path: Union[str, Path]) -> FoldAssignment:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    frame = pd.read_csv(path, dtype={"Cohort": "str", "Id": "int64", FOLD_COLUMN: "int64"})
    if frame["Id"].duplicated().any() or set(frame["Cohort"]) != {meta["cohort"]}:
        raise ValueError(f"Malformed fold file: {path}")
    return FoldAssignment(
        cohort=meta["cohort"],
        tasks=tuple(meta["tasks"]),
        n_folds=meta["n_folds"],
        seed=meta["seed"],
        folds=frame.set_index("Id")[FOLD_COLUMN],
        diagnostics=pd.DataFrame(),
    )


def fold_stem(cohort: str, tasks: Sequence[int], n_folds: int) -> str:
    """One naming scheme for fold files and the results folders built on them."""
    return f"{cohort}_tasks-{'-'.join(str(t) for t in sorted(tasks))}_k{n_folds}"


def default_folds_path(results_dir: Union[str, Path], cohort: str, tasks: Sequence[int], n_folds: int) -> Path:
    return Path(results_dir) / FOLDS_DIR_NAME / f"{fold_stem(cohort, tasks, n_folds)}.csv"
