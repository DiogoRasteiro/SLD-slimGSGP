"""Outer-CV loop over tasks, folds, models and seeds, with resumable CSV output."""

import json
import os
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import pandas as pd

from src.core.schema import (
    DEFAULT_SEED,
    LIBRARY_LOG_DIR_NAME,
    METRICS_FILE_NAME,
    PREDICTIONS_FILE_NAME,
    RUN_CONFIG_FILE_NAME,
)
from src.experiment.dataset import features_and_target, load_binary_task
from src.experiment.folds import FoldAssignment
from src.experiment.metrics import binary_metrics
from src.experiment.models import configured_estimator, effective_spec, fit_predict
from src.experiment.provenance import data_fingerprint, environment, folds_fingerprint

RUN_KEY = ["task", "model", "fold", "seed"]


def _done_keys(path: Path) -> set:
    if not path.is_file():
        return set()
    done = pd.read_csv(path, usecols=RUN_KEY)
    return set(done[RUN_KEY].itertuples(index=False, name=None))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Replace a file in one step, so an interruption can't leave it half-written."""
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _append(path: Path, frame: pd.DataFrame) -> None:
    """Append rows, keeping one header even when models report different extra columns."""
    if not path.is_file():
        frame.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    new = [c for c in frame.columns if c not in header]
    if new:
        _write_csv(path, pd.concat([pd.read_csv(path), frame], ignore_index=True))
    else:
        frame.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)


def _drop_orphan_predictions(path: Path, done: set) -> None:
    """Remove predictions of a run interrupted before its metrics row was written."""
    if not path.is_file():
        return
    preds = pd.read_csv(path)
    keys = preds[RUN_KEY].itertuples(index=False, name=None)
    keep = [k in done for k in keys]
    if not all(keep):
        _write_csv(path, preds[keep])


def _normalized(obj):
    return json.loads(json.dumps(obj, default=str))


def _merge_config(path: Path, current: dict, invocation: dict) -> dict:
    """Refuse to resume into a folder made with other folds, data or model settings.

    Resume skips finished (task, model, fold, seed) rows, so silently mixing rows
    trained under different settings, folds or data would corrupt the results.
    Each call is also logged under ``invocations`` (code and package versions, seeds).
    """
    current = _normalized(current)
    invocation = _normalized(invocation)
    if not path.is_file():
        return dict(current, invocations=[invocation])
    saved = json.loads(path.read_text())
    refuse = lambda what: ValueError(f"{path.parent.name} was run with {what}. Use a new --run-name.")
    for key in ("cohort", "fold_tasks", "n_folds", "fold_seed", "folds_fingerprint"):
        if saved.get(key) != current[key]:
            raise refuse(f"{key}={saved.get(key)}, not {current[key]}")
    for task, digest in current["data_fingerprints"].items():
        if task in saved["data_fingerprints"] and saved["data_fingerprints"][task] != digest:
            raise refuse(f"different data for Task {task}")
    for section in ("models", "resolved_params"):
        for model, value in current[section].items():
            if model in saved[section] and saved[section][model] != value:
                raise refuse(f"different settings for '{model}'")
    return dict(
        current,
        tasks=sorted(set(saved["tasks"]) | set(current["tasks"])),
        data_fingerprints={**saved["data_fingerprints"], **current["data_fingerprints"]},
        models={**saved["models"], **current["models"]},
        resolved_params={**saved["resolved_params"], **current["resolved_params"]},
        invocations=[*saved["invocations"], invocation],
    )


def run_cv(
    assignment: FoldAssignment,
    tasks: Sequence[int],
    models: Sequence[str],
    seeds: Sequence[int],
    out_dir: Union[str, Path],
    folds: Optional[Sequence[int]] = None,
    spec_overrides: Optional[Mapping[str, Mapping]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    on_result: Optional[Callable[[dict], None]] = None,
) -> pd.DataFrame:
    """Run every (task, outer fold, model, seed) not yet in ``out_dir``; return all metrics.

    Models whose spec is not stochastic run once per fold, always under ``DEFAULT_SEED``.
    ``spec_overrides`` maps a model name to changes of its spec (see ``effective_spec``).
    """
    cohort = assignment.cohort
    unknown = set(tasks) - set(assignment.tasks)
    if unknown:
        raise ValueError(f"Tasks {sorted(unknown)} are not in the fold file ({list(assignment.tasks)}).")
    fold_ids = list(folds) if folds is not None else list(range(assignment.n_folds))
    bad_folds = [f for f in fold_ids if f not in range(assignment.n_folds)]
    if bad_folds:
        raise ValueError(f"Folds {bad_folds} don't exist; folds are 0..{assignment.n_folds - 1}.")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError(f"Seeds must be non-empty and distinct, got {list(seeds)}.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, preds_path = out / METRICS_FILE_NAME, out / PREDICTIONS_FILE_NAME
    overrides = spec_overrides or {}
    specs = {m: effective_spec(m, overrides.get(m)) for m in models}
    # Building each estimator fails on bad settings before any training.
    estimators = {m: configured_estimator(m, spec, DEFAULT_SEED) for m, spec in specs.items()}
    data = {task: load_binary_task(cohort, task, data_dir=data_dir)[0] for task in sorted(tasks)}
    config_path = out / RUN_CONFIG_FILE_NAME
    config = _merge_config(
        config_path,
        dict(
            cohort=cohort,
            fold_tasks=list(assignment.tasks),
            n_folds=assignment.n_folds,
            fold_seed=assignment.seed,
            folds_fingerprint=folds_fingerprint(assignment.folds),
            tasks=sorted(tasks),
            data_fingerprints={str(t): data_fingerprint(df) for t, df in data.items()},
            models=specs,
            # Every setting each model actually runs with, library defaults included (seed aside).
            resolved_params={
                m: {k: v for k, v in getattr(e, "resolved_params", e.get_params)().items() if k != "random_state"}
                for m, e in estimators.items()
            },
        ),
        dict(environment(), seeds=list(seeds), folds=fold_ids),
    )
    config_path.write_text(json.dumps(config, indent=2))
    done = _done_keys(metrics_path)
    _drop_orphan_predictions(preds_path, done)

    for task, df in data.items():
        fold_of = df["Id"].map(assignment.folds)
        if fold_of.isna().any():
            raise ValueError(f"Task {task}: some children have no fold; rebuild the folds.")
        X, y = features_and_target(df)
        for fold in fold_ids:
            test = fold_of.eq(fold).to_numpy()
            for model in models:
                model_seeds = seeds if specs[model]["stochastic"] else [DEFAULT_SEED]
                for seed in model_seeds:
                    if (task, model, fold, seed) in done:
                        continue
                    y_pred, y_score, info = fit_predict(
                        model, specs[model], X[~test], y[~test], X[test], seed,
                        log_dir=out / LIBRARY_LOG_DIR_NAME,
                    )
                    if "best_params" in info:
                        info["best_params"] = json.dumps(info["best_params"], default=str)
                    # Whatever the model reports (tree size, inner score, ...) becomes a column;
                    # models that report nothing leave those cells empty.
                    row = dict(
                        cohort=cohort, task=task, model=model, fold=fold, seed=seed,
                        n_train=int((~test).sum()), n_train_positive=int(y[~test].sum()),
                        **binary_metrics(y[test], y_pred, y_score),
                        **info,
                    )
                    _append(
                        preds_path,
                        pd.DataFrame(
                            dict(
                                Cohort=cohort, Id=df.loc[test, "Id"].to_numpy(),
                                task=task, fold=fold, model=model, seed=seed,
                                y_true=y[test], y_pred=y_pred,
                                y_score=y_score if y_score is not None else float("nan"),
                            )
                        ),
                    )
                    # Metrics last: a metrics row is what marks the run as done on resume.
                    _append(metrics_path, pd.DataFrame([row]))
                    if on_result:
                        on_result(row)
    return pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()


# auc last: a backup for when thresholded metrics collapse, not a main result.
SCORE_COLUMNS = ["balanced_accuracy", "mcc", "sensitivity", "specificity", "auc"]
# Tree sizes are skewed (one bloated run can dominate the mean), so the median is reported too.
SIZE_COLUMNS = ["nodes", "depth"]


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per task and model, over all fold x seed runs: mean/std of scores, mean/std/median of size.

    Size is empty for models that have no tree.
    """
    groups = metrics.groupby(["task", "model"])
    parts = [groups[SCORE_COLUMNS].agg(["mean", "std"])]
    sizes = [c for c in SIZE_COLUMNS if c in metrics]
    if sizes:
        parts.append(groups[sizes].agg(["mean", "std", "median"]))
    agg = pd.concat(parts, axis=1)
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["runs"] = groups.size()
    return agg.reset_index()
