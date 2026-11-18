"""
Tests for the per-task binary CV pipeline (src/experiment), on the real CSVs.
"""

import dataclasses
import json
import sys

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from src.core.schema import (
    BINARY_TARGET_COLUMN,
    DEFAULT_SEED,
    KINEMATIC_FEATURES,
    METRICS_FILE_NAME as METRICS_FILE,
    PREDICTIONS_FILE_NAME as PREDICTIONS_FILE,
    RUN_CONFIG_FILE_NAME as CONFIG_FILE,
)
from src.data.loader import get_available_tasks
from src.experiment.cv import run_cv, summarize
from src.experiment.dataset import features_and_target, load_binary_task
from src.experiment.folds import build_folds, fold_balance, load_folds, save_folds
from src.experiment.estimators import EvolutionaryClassifier
from src.experiment.metrics import binary_metrics
from src.experiment.models import effective_spec, fit_predict
from src.experiment.preprocessing import make_preprocessor

COHORT = "1year"
TASKS = (3, 6, 9)
TINY = dict(pop_size=10, n_iter=2)
# Small settings so every model runs in seconds; the pipeline itself is unchanged.
FAST = {
    "svm": {"grid": {"C": [1.0, 10.0]}},
    "gp": {"fixed": TINY},
    "gsgp": {"fixed": TINY},
    "slim": {"fixed": TINY},
}


@pytest.fixture(scope="module")
def assignment():
    return build_folds(COHORT, TASKS, seeds=(DEFAULT_SEED, DEFAULT_SEED + 1))


@pytest.fixture(scope="module")
def small_split():
    df, _ = load_binary_task(COHORT, 3)
    X, y = features_and_target(df)
    return X[:100], y[:100], X[100:]


def test_all_zero_rows_removed_in_1year():
    total = 0
    for task in get_available_tasks(COHORT):
        df, report = load_binary_task(COHORT, task)
        total += report.all_zero_rows
        assert not df[KINEMATIC_FEATURES].eq(0).all(axis=1).any()
    # 21 all-zero rows in 1year: 20 labelled (dropped by the all-zero filter), 1 already -1.
    assert total == 20


def test_binary_target_and_inputs():
    df, report = load_binary_task(COHORT, 3)
    assert set(df[BINARY_TARGET_COLUMN].unique()) <= {0, 1}
    assert (df[BINARY_TARGET_COLUMN].eq(1) == df["Label"].eq(2)).all()
    assert report.kept_rows == len(df)
    X, y = features_and_target(df)
    assert X.shape == (len(df), len(KINEMATIC_FEATURES)) == (len(df), 30)
    assert y.shape == (len(df),)


def test_folds_partition_children(assignment):
    folds = assignment.folds
    assert not folds.index.duplicated().any()
    assert set(folds.unique()) == set(range(assignment.n_folds))
    sizes = folds.value_counts()
    assert sizes.max() - sizes.min() <= 1
    for task in TASKS:
        df, _ = load_binary_task(COHORT, task)
        assert df["Id"].isin(folds.index).all()


def test_every_test_fold_has_both_classes(assignment):
    d = assignment.diagnostics
    assert not d["test_class_missing"].any()
    assert not d["train_class_missing"].any()


def test_folds_deterministic(assignment):
    again = build_folds(COHORT, TASKS, seeds=(DEFAULT_SEED, DEFAULT_SEED + 1))
    assert again.seed == assignment.seed
    assert again.folds.equals(assignment.folds)


def test_fold_file_roundtrip_and_task_check(assignment, tmp_path):
    path = tmp_path / "folds.csv"
    save_folds(assignment, path)
    loaded = load_folds(path)
    assert loaded.folds.sort_index().equals(assignment.folds.sort_index())
    loaded.check_matches(COHORT, list(reversed(TASKS)))
    with pytest.raises(ValueError):
        loaded.check_matches(COHORT, (3, 6, 10))
    with pytest.raises(ValueError):
        loaded.check_matches("2year", TASKS)


def test_fold_balance_matches_allocator_and_saved_file(assignment, tmp_path):
    path = tmp_path / "folds.csv"
    save_folds(assignment, path)
    balance = fold_balance(load_folds(path))
    assert len(balance) == len(TASKS) * assignment.n_folds
    for task in TASKS:
        df, report = load_binary_task(COHORT, task)
        rows = balance[balance["task"].eq(task)]
        assert rows["children"].sum() == report.kept_rows
        assert rows["positives"].sum() == report.positive_rows
    d = assignment.diagnostics
    pos = d[d["label"].eq(1)].assign(task=lambda x: x["task"].str.removeprefix("Task_").astype(int))
    merged = balance.merge(pos, on=["task", "fold"])
    assert len(merged) == len(balance)
    assert (merged["children"] == merged["eligible_test"]).all()
    assert (merged["positives"] == merged["test_count"]).all()
    assert np.allclose(merged["test_rate"], merged["test_proportion"])
    assert np.allclose(merged["task_rate"], merged["full_eligible_proportion"])


def test_preprocessor_depends_on_its_fit_data():
    df, _ = load_binary_task(COHORT, 3)
    X, _ = features_and_target(df)
    train, test = X[:100], X[100:]
    prep = make_preprocessor().fit(train)
    assert np.allclose(prep.transform(train).mean(axis=0), 0.0, atol=1e-6)
    refit = make_preprocessor().fit(test)
    assert not np.allclose(prep.transform(test), refit.transform(test))


def test_fit_predict_fits_preprocessing_on_training_children_only(small_split):
    # The same model refitted by hand on training rows only must give identical scores;
    # preprocessing fitted on any test rows would shift them.
    X_tr, y_tr, X_te = small_split
    _, y_score, _ = fit_predict("svm", effective_spec("svm", {"grid": {}}), X_tr, y_tr, X_te, DEFAULT_SEED)
    by_hand = make_pipeline(make_preprocessor(), SVC(random_state=DEFAULT_SEED)).fit(X_tr, y_tr)
    assert np.allclose(y_score, by_hand.decision_function(X_te))
    leaky_prep = make_preprocessor().fit(np.vstack([X_tr, X_te]))
    leaky = SVC(random_state=DEFAULT_SEED).fit(leaky_prep.transform(X_tr), y_tr)
    assert not np.allclose(y_score, leaky.decision_function(leaky_prep.transform(X_te)))


def test_binary_metrics_toy():
    m = binary_metrics([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1])
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 5)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 5) / 2)
    assert m["n_test"] == 8 and m["n_test_positive"] == 3 and m["n_pred_positive"] == 4
    assert binary_metrics([0, 1, 0, 1], [0, 1, 0, 1])["mcc"] == pytest.approx(1.0)
    assert np.isnan(m["auc"])
    # Every child predicted 0, yet the scores still rank all positives first: BA 0.5, AUC 1.
    collapsed = binary_metrics([1, 0, 0, 0], [0, 0, 0, 0], y_score=[-0.1, -0.5, -0.7, -0.9])
    assert collapsed["balanced_accuracy"] == pytest.approx(0.5)
    assert collapsed["auc"] == pytest.approx(1.0)


def test_run_cv_smoke_and_resume(assignment, tmp_path):
    overrides = FAST
    models = list(FAST)
    metrics = run_cv(assignment, TASKS, models, [DEFAULT_SEED], tmp_path,
                     folds=[0], spec_overrides=overrides)
    assert len(metrics) == len(TASKS) * len(models)
    # Tree size is an ordinary column: filled for tree models, empty for the SVM.
    trees = metrics[metrics["model"].ne("svm")]
    assert (trees["nodes"] > 0).all() and (trees["depth"] > 0).all()
    assert metrics.loc[metrics["model"].eq("svm"), ["nodes", "depth"]].isna().all().all()
    assert metrics.loc[metrics["model"].eq("svm"), "best_params"].notna().all()
    summary = summarize(metrics)
    assert {"nodes_mean", "nodes_std", "nodes_median", "balanced_accuracy_mean"} <= set(summary.columns)
    preds = pd.read_csv(tmp_path / PREDICTIONS_FILE)
    assert set(preds["y_pred"].unique()) <= {0, 1}
    # Every model saves its raw score; positive score <-> class 1 (exact zeros are tie-broken per model).
    assert preds["y_score"].notna().all()
    assert metrics["auc"].between(0, 1).all()
    assert "auc_mean" in summarize(metrics)
    nonzero = preds["y_score"].ne(0)
    assert (preds.loc[nonzero, "y_score"].gt(0) == preds.loc[nonzero, "y_pred"].eq(1)).all()
    test_ids = set(assignment.folds[assignment.folds.eq(0)].index)
    for (task, model), group in preds.groupby(["task", "model"]):
        assert not group["Id"].duplicated().any()
        assert set(group["Id"]) <= test_ids
        df, _ = load_binary_task(COHORT, task)
        assert len(group) == df["Id"].isin(test_ids).sum()
    again = run_cv(assignment, TASKS, models, [DEFAULT_SEED], tmp_path,
                   folds=[0], spec_overrides=overrides)
    assert len(again) == len(metrics)
    assert len(pd.read_csv(tmp_path / METRICS_FILE)) == len(metrics)
    with pytest.raises(ValueError):
        run_cv(assignment, (3, 10), models, [DEFAULT_SEED], tmp_path, folds=[0])


def test_evolutionary_classifier_is_a_sklearn_estimator(small_split, tmp_path):
    model = EvolutionaryClassifier(algorithm="slim", random_state=1, ms_upper=0.5, **TINY)
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    copy.set_params(ms_upper=0.1, random_state=2)
    assert copy.get_params()["ms_upper"] == 0.1 and model.get_params()["ms_upper"] == 0.5
    X_tr, y_tr, X_te = small_split
    pred = EvolutionaryClassifier(algorithm="gp", random_state=1, log_dir=tmp_path,
                                  fitness_function="sigmoid_rmse", **TINY).fit(X_tr, y_tr).predict(X_te)
    assert pred.shape == (len(X_te),) and set(pred.tolist()) <= {0, 1}


def test_any_model_can_be_tuned(small_split, tmp_path):
    X_tr, y_tr, X_te = small_split
    overrides = {"fixed": TINY, "grid": {"init_depth": [2, 3]}}
    pred, _, info = fit_predict("gp", effective_spec("gp", overrides), X_tr, y_tr, X_te, DEFAULT_SEED,
                                log_dir=tmp_path)
    assert info["best_params"]["init_depth"] in (2, 3)
    assert "nodes" in info and set(pred.tolist()) <= {0, 1}


def test_any_model_can_skip_tuning(small_split):
    X_tr, y_tr, X_te = small_split
    pred, _, info = fit_predict("svm", effective_spec("svm", {"grid": {}}), X_tr, y_tr, X_te, DEFAULT_SEED)
    assert "best_params" not in info
    assert pred.shape == (len(X_te),)


def test_stochastic_flag_drives_seed_repeats(assignment, tmp_path):
    overrides = {
        "svm": {"grid": {}, "stochastic": False},
        "gp": {"fixed": TINY, "stochastic": True},
    }
    seeds = [DEFAULT_SEED, DEFAULT_SEED + 1]
    metrics = run_cv(assignment, (3,), ["svm", "gp"], seeds, tmp_path, folds=[0],
                     spec_overrides=overrides)
    counts = metrics.groupby("model").size()
    assert counts["svm"] == 1 and counts["gp"] == 2


def test_resume_refuses_changed_settings(assignment, tmp_path):
    base = {"gp": {"fixed": TINY}}
    run_cv(assignment, (3,), ["gp"], [DEFAULT_SEED], tmp_path, folds=[0], spec_overrides=base)
    changed = {"gp": {"fixed": {**TINY, "n_iter": 3}}}
    with pytest.raises(ValueError, match="different settings"):
        run_cv(assignment, (3,), ["gp"], [DEFAULT_SEED], tmp_path, folds=[0], spec_overrides=changed)
    # A model not run before can still be added to the same folder.
    run_cv(assignment, (3,), ["svm"], [DEFAULT_SEED], tmp_path, folds=[0],
           spec_overrides={"svm": {"grid": {}}})
    config = json.loads((tmp_path / CONFIG_FILE).read_text())
    assert len(config["invocations"]) == 2
    assert config["invocations"][0]["slim_v2"]["commit"]
    assert config["resolved_params"]["gp"]["prob_const"] is not None


def test_resume_refuses_moved_children_under_same_fold_seed(assignment, tmp_path):
    svm = {"svm": {"grid": {}}}
    run_cv(assignment, (3,), ["svm"], [DEFAULT_SEED], tmp_path, folds=[0], spec_overrides=svm)
    folds = assignment.folds.copy()
    folds.iloc[0] = (folds.iloc[0] + 1) % assignment.n_folds
    moved = dataclasses.replace(assignment, folds=folds)
    assert moved.seed == assignment.seed
    with pytest.raises(ValueError, match="folds_fingerprint"):
        run_cv(moved, (3,), ["svm"], [DEFAULT_SEED], tmp_path, folds=[0], spec_overrides=svm)


def test_non_stochastic_models_ignore_seed_list(assignment, tmp_path):
    metrics = run_cv(assignment, (3,), ["svm"], [7, 8], tmp_path, folds=[0],
                     spec_overrides={"svm": {"grid": {}}})
    assert metrics["seed"].tolist() == [DEFAULT_SEED]


def test_bad_folds_and_seeds_fail_before_training(assignment, tmp_path):
    with pytest.raises(ValueError, match="don't exist"):
        run_cv(assignment, (3,), ["svm"], [DEFAULT_SEED], tmp_path, folds=[assignment.n_folds])
    with pytest.raises(ValueError, match="distinct"):
        run_cv(assignment, (3,), ["gp"], [1, 1], tmp_path, folds=[0])
    assert not (tmp_path / METRICS_FILE).exists()


def test_orphan_predictions_dropped_on_resume(assignment, tmp_path):
    overrides = {"svm": {"grid": {}}}
    run_cv(assignment, (3,), ["svm"], [DEFAULT_SEED], tmp_path, folds=[0], spec_overrides=overrides)
    preds = pd.read_csv(tmp_path / PREDICTIONS_FILE)
    # Simulate an interrupt after predictions were written but before the metrics row.
    orphan = preds.assign(fold=1)
    pd.concat([preds, orphan]).to_csv(tmp_path / PREDICTIONS_FILE, index=False)
    run_cv(assignment, (3,), ["svm"], [DEFAULT_SEED], tmp_path, folds=[0], spec_overrides=overrides)
    assert set(pd.read_csv(tmp_path / PREDICTIONS_FILE)["fold"]) == {0}


def test_reserved_settings_rejected(small_split):
    X_tr, y_tr, _ = small_split
    with pytest.raises(ValueError, match="set by the pipeline"):
        EvolutionaryClassifier(algorithm="gp", verbose=1, **TINY)
    with pytest.raises(ValueError, match="set by the pipeline"):
        EvolutionaryClassifier(algorithm="gp", **TINY).set_params(seed=3)


@pytest.mark.parametrize("bad", [{"verbose": 1}, {"p_inflat": 0.5}, {"p_inflate": 0.5}])
def test_bad_spec_fails_before_any_training(assignment, tmp_path, bad):
    # A reserved name, a typo, and a SLIM-only setting given to GP.
    with pytest.raises(ValueError):
        run_cv(assignment, (3,), ["svm", "gp"], [DEFAULT_SEED], tmp_path, folds=[0],
               spec_overrides={"svm": {"grid": {}}, "gp": {"fixed": {**TINY, **bad}}})
    assert not (tmp_path / METRICS_FILE).exists()


@pytest.mark.parametrize("model", ["gp", "gsgp", "slim"])
def test_same_seed_gives_identical_run(model, small_split, tmp_path):
    # Every repeated-seed result assumes "seed = run"; this checks slim_v2 is deterministic given the seed.
    X_tr, y_tr, X_te = small_split
    spec = effective_spec(model, {"fixed": TINY})
    first = fit_predict(model, spec, X_tr, y_tr, X_te, seed=7, log_dir=tmp_path)
    second = fit_predict(model, spec, X_tr, y_tr, X_te, seed=7, log_dir=tmp_path)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2]["nodes"] == second[2]["nodes"]


def test_gsgp_predicts_after_2000_generations(small_split, tmp_path):
    # GSGP's predict() recurses once per generation; Python's default limit fails near here.
    X_tr, y_tr, X_te = small_split
    spec = effective_spec("gsgp", {"fixed": {"pop_size": 10, "n_iter": 2000}})
    before = sys.getrecursionlimit()
    y_pred, _, _ = fit_predict("gsgp", spec, X_tr, y_tr, X_te, DEFAULT_SEED, log_dir=tmp_path)
    assert len(y_pred) == len(X_te)
    assert sys.getrecursionlimit() == before


def test_shuffled_labels_score_at_chance(assignment):
    # Leakage check: with labels shuffled across children, nothing real is left to learn, so
    # balanced accuracy over the real outer folds (with tuning) must sit around 0.5.
    df, _ = load_binary_task(COHORT, 3)
    X, y = features_and_target(df)
    fold_of = df["Id"].map(assignment.folds).to_numpy()
    # A small grid keeps tuning in the path (where leakage could hide) without the full grid's cost.
    spec = effective_spec("svm", {"grid": {"C": [0.1, 1.0]}})
    rng = np.random.default_rng(DEFAULT_SEED)
    scores = []
    for _ in range(3):
        y_shuffled = rng.permutation(y)
        for fold in range(assignment.n_folds):
            test = fold_of == fold
            pred, _, _ = fit_predict("svm", spec, X[~test], y_shuffled[~test], X[test], DEFAULT_SEED)
            scores.append(binary_metrics(y_shuffled[test], pred)["balanced_accuracy"])
    # 15 runs on ~28 test children each: chance-level noise of the mean is about +-0.03.
    assert abs(np.mean(scores) - 0.5) < 0.08
