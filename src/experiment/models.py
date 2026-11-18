"""The single per-fold step every model goes through: preprocess, optionally tune, fit, predict."""

import copy
import time
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.core.schema import (
    CV_INNER_FOLDS,
    EXPERIMENT_MODELS,
    MODEL_SPECS,
    TUNING_N_JOBS,
    TUNING_SCORING,
)
from src.experiment.estimators import build_estimator
from src.experiment.preprocessing import make_preprocessor

MODEL_STEP = "model"


def effective_spec(name: str, overrides: Optional[Mapping] = None) -> dict:
    """The model's spec with overrides applied: ``fixed`` is merged, ``grid``/``stochastic`` replaced."""
    if name not in EXPERIMENT_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {EXPERIMENT_MODELS}.")
    spec = copy.deepcopy(MODEL_SPECS[name])
    overrides = overrides or {}
    spec["fixed"] = {**spec["fixed"], **overrides.get("fixed", {})}
    for key in ("grid", "stochastic"):
        if key in overrides:
            spec[key] = overrides[key]
    return spec


def configured_estimator(name: str, spec: Mapping, seed: int, log_dir=None):
    """Estimator with the spec's fixed settings applied; raises early on unknown or reserved names."""
    estimator = build_estimator(name)
    params = {**spec["fixed"], "random_state": seed}
    if log_dir is not None and "log_dir" in estimator.get_params():
        params["log_dir"] = log_dir
    estimator.set_params(**params)
    if spec["grid"]:
        estimator.set_params(**{k: v[0] for k, v in spec["grid"].items()})
    return estimator


def fit_predict(
    name: str,
    spec: Mapping,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    log_dir: Optional[Union[str, Path]] = None,
):
    """Train on one outer training fold; return 0/1 predictions, raw scores and run info.

    If the spec's grid is non-empty, settings are chosen by inner CV on the
    training fold (preprocessing refitted inside each inner split), then the best
    setting is refitted on the whole training fold. Raw scores come from the
    model's ``decision_function`` (None if it has none); they are kept for the
    per-child ensemble.
    """
    pipe = Pipeline([
        ("prep", make_preprocessor()),
        (MODEL_STEP, configured_estimator(name, spec, seed, log_dir)),
    ])
    start = time.perf_counter()
    info = {}
    if spec["grid"]:
        search = GridSearchCV(
            pipe,
            {f"{MODEL_STEP}__{k}": v for k, v in spec["grid"].items()},
            cv=StratifiedKFold(CV_INNER_FOLDS, shuffle=True, random_state=seed),
            scoring=TUNING_SCORING,
            n_jobs=TUNING_N_JOBS,
            # A setting that crashes should stop the run, not be silently scored as NaN.
            error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        info["best_params"] = {k.removeprefix(f"{MODEL_STEP}__"): v for k, v in search.best_params_.items()}
        info["inner_score"] = float(search.best_score_)
    else:
        fitted = pipe.fit(X_train, y_train)
    y_pred = np.asarray(fitted.predict(X_test), dtype=int)
    y_score = (
        np.asarray(fitted.decision_function(X_test), dtype=float)
        if hasattr(fitted, "decision_function") else None
    )
    info.update(getattr(fitted.named_steps[MODEL_STEP], "summary_", {}))
    info["elapsed_seconds"] = time.perf_counter() - start
    return y_pred, y_score, info
