"""Every model as a scikit-learn classifier, so the pipeline can treat them alike."""

import functools
import importlib
import inspect
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from src.core.schema import DEFAULT_SEED, PREDICT_RECURSION_LIMIT

# algorithm -> (module, function, attribute holding the elite's node count)
_LIBRARY = {
    "gp": ("slim_gsgp.main_gp", "gp", "node_count"),
    "gsgp": ("slim_gsgp.main_gsgp", "gsgp", "nodes"),
    "slim": ("slim_gsgp.main_slim", "slim", "nodes_count"),
}
_OWN_PARAMS = ("algorithm", "random_state", "log_dir")
# Library arguments the wrapper sets itself; a spec must not set them.
_RESERVED_SETTINGS = frozenset(
    {"X_train", "y_train", "X_test", "y_test", "dataset_name", "seed", "log_level", "log_path", "verbose"}
)


@functools.lru_cache(maxsize=None)
def _library_function(algorithm: str):
    if algorithm not in _LIBRARY:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {sorted(_LIBRARY)}.")
    module, function, _ = _LIBRARY[algorithm]
    return getattr(importlib.import_module(module), function)


def _library_defaults(algorithm: str) -> dict:
    parameters = inspect.signature(_library_function(algorithm)).parameters
    return {k: p.default for k, p in parameters.items() if k not in _RESERVED_SETTINGS}


def _check_settings(algorithm: str, settings) -> None:
    reserved = _RESERVED_SETTINGS & set(settings)
    if reserved:
        raise ValueError(
            f"Settings {sorted(reserved)} are set by the pipeline, not the model spec "
            "(the seed comes from random_state)."
        )
    # Checked here so a typo fails when the spec is built, not at the first fit hours later.
    unknown = set(settings) - set(_library_defaults(algorithm))
    if unknown:
        raise ValueError(f"slim_v2's {algorithm} has no setting(s) {sorted(unknown)}.")


class EvolutionaryClassifier(ClassifierMixin, BaseEstimator):
    """slim_v2's gp / gsgp / slim behind fit/predict.

    Any library argument (pop_size, ms_upper, p_inflate, ...) is an ordinary
    estimator parameter, so ``clone`` and ``GridSearchCV`` work on it. Labels are
    encoded and outputs decoded by slim_v2's strategy for the chosen
    ``fitness_function`` (for sigmoid_rmse: {0, 1} targets, raw output >= 0 -> 1).
    """

    def __init__(
        self,
        algorithm: str = "slim",
        random_state: int = DEFAULT_SEED,
        log_dir: Optional[Union[str, Path]] = None,
        **settings,
    ):
        _check_settings(algorithm, settings)
        self.algorithm = algorithm
        self.random_state = random_state
        self.log_dir = log_dir
        self.settings = dict(settings)

    # BaseEstimator only sees named __init__ arguments; library settings are
    # passed through **settings, so they are exposed here explicitly.
    def get_params(self, deep: bool = True) -> dict:
        return {**{p: getattr(self, p) for p in _OWN_PARAMS}, **self.settings}

    def set_params(self, **params):
        _check_settings(params.get("algorithm", self.algorithm), set(params) - set(_OWN_PARAMS))
        for key, value in params.items():
            if key in _OWN_PARAMS:
                setattr(self, key, value)
            else:
                self.settings[key] = value
        return self

    def resolved_params(self) -> dict:
        """Every setting the library will use, its own defaults included, for the run record."""
        return {**{p: getattr(self, p) for p in _OWN_PARAMS if p != "log_dir"},
                **_library_defaults(self.algorithm), **self.settings}

    def _strategy(self):
        from slim_gsgp.classification.strategies import get_strategy

        return get_strategy(self.settings["fitness_function"])

    def fit(self, X, y):
        import torch

        fn = _library_function(self.algorithm)
        nodes_attr = _LIBRARY[self.algorithm][2]
        log_path = str(Path(self.log_dir) / f"{self.algorithm}.csv") if self.log_dir is not None else None
        # X_test=None keeps held-out children out of evolution entirely. The library
        # writes a settings CSV even with log_level=0, so log_path redirects it.
        self.elite_ = fn(
            X_train=torch.tensor(np.asarray(X), dtype=torch.float32),
            y_train=self._strategy().encode(torch.tensor(np.asarray(y), dtype=torch.float32)),
            X_test=None,
            y_test=None,
            dataset_name=f"sld_{self.algorithm}",
            seed=int(self.random_state),
            log_level=0,
            log_path=log_path,
            verbose=0,
            **self.settings,
        )
        self.classes_ = np.array([0, 1])
        self.summary_ = dict(
            nodes=int(getattr(self.elite_, nodes_attr)),
            depth=int(self.elite_.depth),
            train_fitness=float(self.elite_.fitness),
        )
        return self

    def _raw(self, X):
        import torch

        previous = sys.getrecursionlimit()
        sys.setrecursionlimit(max(previous, PREDICT_RECURSION_LIMIT))
        try:
            return self.elite_.predict(torch.tensor(np.asarray(X), dtype=torch.float32))
        finally:
            sys.setrecursionlimit(previous)

    def decision_function(self, X) -> np.ndarray:
        """The tree's raw output, before thresholding (as SVC's decision_function)."""
        return self._raw(X).detach().numpy().astype(float).ravel()

    def predict(self, X):
        import torch

        strategy = self._strategy()
        positive_code = strategy.encode(torch.tensor([0.0, 1.0]))[1]
        return (strategy.decode(self._raw(X)) == positive_code).to(torch.int64).numpy()


def build_estimator(name: str) -> BaseEstimator:
    """The only place that maps a model name to an estimator."""
    if name == "svm":
        return SVC()
    if name in _LIBRARY:
        return EvolutionaryClassifier(algorithm=name)
    raise ValueError(f"Unknown model '{name}'.")
