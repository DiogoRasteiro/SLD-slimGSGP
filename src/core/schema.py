"""
Schema definitions, feature taxonomy, and data models for SLD (Specific Learning Disorder) online handwriting dataset.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch


class Cohort(str, Enum):
    """Primary school grade cohorts."""
    YEAR_1 = "1year"
    YEAR_2 = "2year"
    YEAR_3 = "3year"


class FeatureCategory(str, Enum):
    """Kinematic and dynamic feature categories."""
    PRESSURE = "pressure"
    TILT = "tilt"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    KINEMATICS = "kinematics"
    EVENTS = "events"


# --- Feature Definitions ---
PRESSURE_FEATURES: List[str] = [
    "mean:pressure",
    "std:pressure",
    "iqr:pressure",
    "number_of_changes_in_pressure_profile",
]

TILT_FEATURES: List[str] = [
    "mean:tilt(on-surface)",
    "std:tilt(on-surface)",
    "iqr:tilt(on-surface)",
]

TEMPORAL_FEATURES: List[str] = [
    "writing_duration(on-surface)",
    "writing_duration_overall",
    "mean:stroke_duration(on-surface)",
    "in_air_movement_time",
    "in_air_on_paper_ratio",
]

SPATIAL_FEATURES: List[str] = [
    "writing_width(on-surface)",
    "writing_height(on-surface)",
    "mean:stroke_width(on-surface)",
    "std:stroke_width(on-surface)",
    "iqr:stroke_width(on-surface)",
    "mean:stroke_height(on-surface)",
    "std:stroke_height(on-surface)",
    "iqr:stroke_height(on-surface)",
    "mean:stroke_length(on-surface)",
    "number_of_changes_in_x_profile",
    "number_of_changes_in_y_profile",
]

KINEMATICS_FEATURES: List[str] = [
    "mean:velocity:axis-xy(on-surface)",
    "std:velocity:axis-xy(on-surface)",
    "mean:acceleration:axis-xy(on-surface)",
    "std:acceleration:axis-xy(on-surface)",
    "mean:jerk:axis-xy(on-surface)",
    "average_normalized_jerk",
]

EVENT_FEATURES: List[str] = [
    "number_of_interruptions",
]

# All 30 pure numerical kinematic features extracted from stylus
KINEMATIC_FEATURES: List[str] = (
    PRESSURE_FEATURES
    + TILT_FEATURES
    + TEMPORAL_FEATURES
    + SPATIAL_FEATURES
    + KINEMATICS_FEATURES
    + EVENT_FEATURES
)

DEMOGRAPHIC_COLUMNS: List[str] = [
    "Gender",
    "Age",
    "Dominant_Hand",
]

# Valid categorical demographic values as recorded by the acquisition device.
# Any other value (e.g. the literal string '0') marks an unrecorded field.
GENDER_VALUES: List[str] = ["Femminile", "Maschile"]
DOMINANT_HAND_VALUES: List[str] = ["Destra", "Sinistra"]

# One-hot / binary encodings used by encode_demographics(), keyed to the
# same value lists above so the two stay in sync by construction.
GENDER_ENCODING: Dict[str, float] = {v: float(i) for i, v in enumerate(GENDER_VALUES)}
DOMINANT_HAND_ENCODING: Dict[str, float] = {v: float(i) for i, v in enumerate(DOMINANT_HAND_VALUES)}

# Plausible age range (years) for primary-school subjects (grades 1-3).
# Values outside this range (e.g. -4992.0, 0.0) are acquisition artifacts.
AGE_MIN_VALID: float = 3.0
AGE_MAX_VALID: float = 20.0
# Fallback age when no valid value exists anywhere for a subject/cohort
# (approximate median age for a first-grade child).
AGE_DEFAULT_FALLBACK: float = 7.0

# Default split ratios for train_val_test_split(); must sum to 1.0.
DEFAULT_TRAIN_RATIO: float = 0.7
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

# Default RNG seed for reproducible splits and SLIM-GSGP runs.
DEFAULT_SEED: int = 42

TARGET_COLUMN: str = "Label"

FEATURE_CATEGORIES_MAP: Dict[FeatureCategory, List[str]] = {
    FeatureCategory.PRESSURE: PRESSURE_FEATURES,
    FeatureCategory.TILT: TILT_FEATURES,
    FeatureCategory.TEMPORAL: TEMPORAL_FEATURES,
    FeatureCategory.SPATIAL: SPATIAL_FEATURES,
    FeatureCategory.KINEMATICS: KINEMATICS_FEATURES,
    FeatureCategory.EVENTS: EVENT_FEATURES,
}

# Explicit dtypes for pd.read_csv(), derived from the feature taxonomy above.
# Avoids per-file dtype sniffing (60 CSVs) and guards against silent dtype
# drift between files. 'Age' is float64 because raw files contain float
# anomalies (e.g. -4992.0) that must survive parsing for cleaning downstream.
CSV_DTYPES: Dict[str, str] = {
    "Id": "int64",
    **{f: "float64" for f in KINEMATIC_FEATURES},
    "Gender": "str",
    "Age": "float64",
    "Dominant_Hand": "str",
    TARGET_COLUMN: "int64",
    "Task": "int64",
}

# Audit ingestion preserves malformed tokens and missing identifiers for inspection.
EDA_CSV_DTYPES: Dict[str, str] = {column: "string" for column in CSV_DTYPES}
EDA_VALID_LABELS: Tuple[int, ...] = (0, 1, 2)
EDA_UNKNOWN_LABEL: int = -1
EDA_EXPECTED_TASKS: Dict[str, int] = {"1year": 21, "2year": 21, "3year": 18}
EDA_FOLD_COUNTS: Tuple[int, ...] = (2, 3, 4, 5)
EDA_CANDIDATE_SEEDS: Tuple[int, ...] = (DEFAULT_SEED, DEFAULT_SEED + 1)
EDA_RARE_CLASS_THRESHOLD: int = 5
EDA_MIN_FOLD_SUPPORT: int = 5
EDA_NEAR_CONSTANT_FRACTION: float = 0.95
EDA_EXTREME_IQR_MULTIPLIER: float = 3.0
EDA_PERCENTILES: Tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
EDA_SCALE_RATIO_WARNING: float = 1000.0
EDA_DURATION_RELATIONSHIPS: Tuple[Tuple[str, str], ...] = (
    ("writing_duration_overall", "writing_duration(on-surface)"),
)

# --- Per-task binary CV experiments (agreed with the team, Nov 2026) ---
# Labels {0, 1} -> 0 and {2} -> 1 ("struggled").
BINARY_POSITIVE_LABEL: int = 2
BINARY_TARGET_COLUMN: str = "y"
DEFAULT_EXPERIMENT_COHORT: str = "1year"
CV_OUTER_FOLDS: int = 5
CV_INNER_FOLDS: int = 3
# Fold assignment is chosen among these seeds by class balance only, never by model results.
CV_FOLD_SEED_CANDIDATES: Tuple[int, ...] = tuple(range(DEFAULT_SEED, DEFAULT_SEED + 20))
# Seeds for every stochastic model (6 for now; 6 or 10 to be decided after timing).
EXPERIMENT_SEEDS: Tuple[int, ...] = tuple(range(DEFAULT_SEED, DEFAULT_SEED + 6))
# Every model goes through the same per-fold pipeline. Each entry only decides:
#   fixed      - settings always used
#   grid       - settings tuned by inner CV on the training fold ({} = no tuning)
#   stochastic - True: one run per seed; False: one run per fold
# Evolutionary settings are provisional, to be confirmed after timing.
_EVOLUTION_COMMON: Dict[str, object] = {
    "pop_size": 500,
    "n_iter": 2000,
    "init_depth": 6,
    "tournament_size": 4,
    "fitness_function": "sigmoid_rmse",
}
MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "svm": {
        "fixed": {},
        "grid": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", 0.01, 0.1],
            # No class_weight: the evolutionary models have no class weighting either,
            # so every model sees the labels as they are.
        },
        "stochastic": False,
    },
    "gp": {
        "fixed": {**_EVOLUTION_COMMON},
        "grid": {},
        "stochastic": True,
    },
    "gsgp": {
        "fixed": {**_EVOLUTION_COMMON, "p_xo": 0.0, "ms_lower": 0.0, "ms_upper": 1.0, "reconstruct": True},
        "grid": {},
        "stochastic": True,
    },
    "slim": {
        "fixed": {
            **_EVOLUTION_COMMON,
            "slim_version": "SLIM+SIG2",
            "p_inflate": 0.7,
            "ms_lower": 0.0,
            "ms_upper": 1.0,
            # slim_v2 requires max_depth >= init_depth + 6.
            "max_depth": 15,
            "reconstruct": True,
        },
        "grid": {},
        "stochastic": True,
    },
}
EXPERIMENT_MODELS: Tuple[str, ...] = tuple(MODEL_SPECS)
TUNING_SCORING: str = "balanced_accuracy"
# GSGP's predict() recurses once per mutation in a tree's history; Python's default
# limit (1000) fails between 1,500 and 2,000 generations. Raised only while predicting.
PREDICT_RECURSION_LIMIT: int = 10_000
# Sequential: tuning in parallel ran the machine short of memory.
TUNING_N_JOBS: int = 1
# Output layout under results/ (gitignored: fold files and predictions list subject IDs).
RESULTS_DIR_NAME: str = "results"
FOLDS_DIR_NAME: str = "folds"
TIMING_DIR_NAME: str = "timing"
# Generations in a --timing run: enough to measure cost per generation, not hours.
TIMING_N_ITER: int = 100
LIBRARY_LOG_DIR_NAME: str = "library_log"
METRICS_FILE_NAME: str = "metrics.csv"
PREDICTIONS_FILE_NAME: str = "predictions.csv"
RUN_CONFIG_FILE_NAME: str = "config.json"

# Task Performance Labels: 0 best, 2 worst; -1 indicates subject not present
LABEL_DESCRIPTION: Dict[int, str] = {
    0: "Best Performance (Score 0)",
    1: "Intermediate Performance (Score 1)",
    2: "Worst Performance (Score 2)",
    -1: "Absent / Not Present",
}


@dataclass
class CohortInfo:
    """Metadata summary of a single grade cohort."""
    cohort: Cohort
    task_count: int
    subject_count: int
    total_records: int
    tasks: List[int] = field(default_factory=list)
    label_distribution: Dict[int, int] = field(default_factory=dict)


@dataclass
class DatasetSummary:
    """High-level summary of the entire dataset directory."""
    data_dir: Path
    cohorts: Dict[str, CohortInfo]
    total_files: int
    total_records: int
    num_kinematic_features: int
    unique_total_subjects: int


@dataclass
class TaskData:
    """Container for a loaded task dataset."""
    cohort: str
    task_id: int
    df: pd.DataFrame
    num_samples: int
    num_features: int
    label_counts: Dict[int, int]

    def __repr__(self) -> str:
        return f"<TaskData cohort={self.cohort} task={self.task_id} samples={self.num_samples} features={self.num_features}>"


@dataclass
class PreparedTensors:
    """Container for model-ready PyTorch tensors."""
    X: torch.Tensor
    y: torch.Tensor
    feature_names: List[str]
    subject_ids: torch.Tensor
    task_ids: torch.Tensor
    cohorts: List[str]

    @property
    def shape(self) -> Tuple[torch.Size, torch.Size]:
        """Return (X.shape, y.shape)."""
        return self.X.shape, self.y.shape

    def __repr__(self) -> str:
        return f"<PreparedTensors X={tuple(self.X.shape)} y={tuple(self.y.shape)} dtype={self.X.dtype}>"
