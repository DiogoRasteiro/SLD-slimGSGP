# Commands

## Setup

```bash
uv sync              # installs deps, including the slim-gsgp submodule (slim_v2/)
pytest tests          # 24 tests, run against real data/ CSVs
```

## CLI (`main.py`)

```bash
python main.py        # no flags: full dashboard (overview, label distribution,
                       # feature breakdown, tensor prep demo on 1year/Task 1)
```

### Selecting cohorts and tasks

`--cohort`/`-c` and `--task`/`-t` both accept one value, several space-separated values, or
the literal `all`. **Omitting either means "all"** — this is symmetric and deliberate.

```bash
python main.py --cohort 1year --task 1              # single cohort, single task
python main.py --cohort 1year 2year --task 1 5 9     # 2 cohorts x 3 tasks
python main.py --cohort all --task 1                 # every cohort, task 1 only
python main.py --task all                            # every cohort (default), every task
python main.py --tensor-demo                         # no --cohort/--task: ALL data by default
```

Task IDs are resolved per cohort (`3year` only goes up to 18, not 21), so `--task all` with
multiple cohorts correctly loads a different task range per cohort.

### Flags

| Flag | Effect |
|---|---|
| `--overview`, `-o` | Dataset summary table (files, records, per-cohort label distribution) |
| `--cohort NAME...`, `-c` | Select cohort(s): one, several, or `all` (default: all) |
| `--task N...`, `-t` | Select task(s): one, several, or `all` (default: all) |
| `--features`, `-f` | 30 kinematic features, grouped by category, with stats over the selection |
| `--subject N`, `-s` | Longitudinal trajectory for subject `N`, rendered once per selected cohort (subject IDs are cohort-local — see [DATA.md](DATA.md)) |
| `--tensor-demo` | Build PyTorch tensors from the selection, show shape/dtype, run a train/val/test split |
| `--run-slim` | Run a fast SLIM-GSGP demo on the selection (small pop/iterations, for a quick check — not a real experiment) |
| `--data-dir PATH`, `-d` | Override the data directory (default: `./data`) |

Combining `--cohort`/`--task` with a task-summary view (no other action flag) prints one
table per resolved `(cohort, task)` pair:

```bash
python main.py --cohort 1year 2year --task 1 2   # prints 4 task-summary tables
```

Bad selections (unknown cohort, missing task file) print `Error: ...` and exit 1, instead of
a raw traceback.

## Python API

Import from the real module — there are no `src`-level re-exports (see [CLAUDE.md](../CLAUDE.md)).

### Load data

```python
from src.data.loader import load_task, load_cohort, load_all

df_task   = load_task("1year", 1, filter_absent=True)   # one cohort, one task
df_cohort = load_cohort("2year")                         # one cohort, all its tasks
df_all    = load_all()                                   # every cohort, every task
```

All three accept `data_dir=` (override) and `filter_absent=` (drop `Label == -1` rows,
default `False`). `load_task`/`load_cohort` also accept a `Cohort` enum member instead of a
string.

```python
from src.data.loader import get_available_cohorts, get_available_tasks

get_available_cohorts()          # ['1year', '2year', '3year']
get_available_tasks("3year")     # [1, 2, ..., 18]
```

### Clean and prepare tensors

```python
from src.data.preprocessor import clean_and_impute_demographics, prepare_tensors, train_val_test_split

df_clean = clean_and_impute_demographics(df_task)   # fixes Age/Gender/Dominant_Hand anomalies

tensors = prepare_tensors(
    df_task,
    feature_set="kinematic",   # "kinematic" (30 features) | "all" (+ Age, encoded demographics) | list[str]
    filter_absent=True,
    standardize=True,           # z-score normalize X
)
# tensors.X: (N, 30) float32, tensors.y: (N,) float32

(X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(tensors)
# ratios/seed default to DEFAULT_TRAIN_RATIO / DEFAULT_VAL_RATIO / DEFAULT_TEST_RATIO / DEFAULT_SEED
```

### Stats and rendering

```python
from src.analysis.stats import compute_feature_stats, get_subject_trajectory
from src.visualization.viewer import render_task_summary, render_subject_profile

stats = compute_feature_stats(df_task)          # mean/std/min/median/max/iqr per feature

traj = get_subject_trajectory(df_all, subject_id=1, cohort="1year")   # pass cohort explicitly
render_subject_profile(df_cohort, subject_id=1)                        # rich table
```

### Run SLIM-GSGP

```python
from src.genetic.runner import SlimConfig, run_slim_experiment

config = SlimConfig(pop_size=50, n_iter=20, slim_version="SLIM+SIG2")
result = run_slim_experiment(tensors, config=config)
print(result.test_rmse, result.test_mae)
```

`run_slim_experiment` imports the `slim_gsgp` package lazily and raises `ImportError` if the
`slim_v2/` submodule isn't installed — everything else in `src/` works without it.

## Per-task CV experiments (`run_experiment.py`)

One binary classifier per task (labels {0, 1} → 0, 2 → 1), evaluated with 5-fold stratified CV
by child. The folds are fixed in advance and shared by the chosen tasks. Only the 30 kinematic
features are used. Yeo–Johnson + standardization is fitted on each training fold.

Every model (`svm`, `gp`, `gsgp`, `slim`) goes through the same per-fold step. Each one is an
entry in `MODEL_SPECS` (`src/core/schema.py`) with three fields:

- `fixed`: settings always used.
- `grid`: settings tuned by inner CV on the training fold. `{}` means no tuning.
- `stochastic`: `True` runs once per seed, `False` once per fold.

Deciding which model is tuned, or which settings it gets, is a config change only. The slim_v2
models are wrapped as scikit-learn classifiers (`src/experiment/estimators.py`), trained with
`sigmoid_rmse`, and their raw output ≥ 0 → class 1.

```bash
python run_experiment.py --task 3 6 9 --build-folds      # build + save folds for exactly these tasks
python run_experiment.py --task 3 6 9 --check-folds      # re-check the saved folds' positive rates
python run_experiment.py --task 3 6 9 --timing           # one fold x one seed per model, 100 generations, to budget runs
python run_experiment.py --task 3 6 9                    # full run: all folds, models, seeds
python run_experiment.py --task 3 6 9 --models svm gp --seeds 42 43 --folds 0 1
```

`--cohort` defaults to `1year`. `--task` is required and doesn't accept `all`, unlike `main.py`: the
folds are balanced on exactly the tasks given, so a fold file is only valid for that task set. Outputs go to `results/`
(gitignored, since they list subject IDs):

- `results/folds/<cohort>_tasks-<ids>_k5.csv` (+ `.json`): one fold per child.

- `results/<run>/`: by default `<cohort>_tasks-<ids>_k<folds>_fs<fold seed>`, so runs on different
  folds never share a folder.
- `results/<run>/metrics.csv`: one row per (task, model, fold, seed). Besides the four scores, and `auc`
  (ROC AUC on `y_score`, a backup for when rare positives make the thresholded scores collapse), it
  has whatever the model reports as its own column: `nodes`, `depth`, `train_fitness` for tree
  models (empty for the SVM), and `best_params` (JSON) and `inner_score` for tuned models.
  `summarize` reports mean/std of the scores (AUC included) and mean/std/median of `nodes` and `depth`.
- `results/<run>/predictions.csv`: held-out predictions per child, for later ensembling: `y_pred`
  (0/1) and `y_score`, the model's raw output from `decision_function` (positive → class 1;
  the scale differs between models).
- `results/<run>/config.json`: what the folder was computed from: fold seed and a fingerprint of
  the fold assignment, a fingerprint of each task's cleaned data, each model's spec and every
  setting it ran with (library defaults included), and one entry per invocation with the seeds,
  folds, code and slim_v2 commits, and package versions.
- `results/timing/<stem>_task<t>_<timestamp>/`: one new folder per `--timing` call. Timing runs
  use `TIMING_N_ITER` (100) generations instead of the full setting, so they finish in minutes.

The fold balance table (per task and fold: test children, positives, test/train/task positive
rate, gap in points) is recomputed from the saved fold file and the cleaned data, independently
of the allocator. `--build-folds` prints it after saving, and `--check-folds` prints it again at
any time. It reports only; no limits are applied.

Runs are resumable: combinations already in `metrics.csv` are skipped. Resuming into a folder made
with other folds (even under the same fold seed), other data, or different settings for a model
already in it, is refused (use a new
`--run-name`). Adding models to an existing folder is fine. Models whose spec isn't `stochastic` run
once per fold under `DEFAULT_SEED`, whatever `--seeds` says.
