"""What a results folder was computed from: fingerprints of folds and data, code and package versions."""

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.core.schema import BINARY_TARGET_COLUMN, KINEMATIC_FEATURES

REPO_DIR = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_DIR / "slim_v2"
PACKAGES = ("slim-gsgp", "torch", "scikit-learn", "numpy", "pandas")


def _digest(obj) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(obj, index=False).to_numpy().tobytes()).hexdigest()[:16]


def folds_fingerprint(folds: pd.Series) -> str:
    """Changes whenever any child moves fold, even if the fold seed is the same."""
    return _digest(folds.sort_index().reset_index())


def data_fingerprint(df: pd.DataFrame) -> str:
    """Changes whenever a cleaned task's children, features or binary labels change."""
    return _digest(df[["Id", *KINEMATIC_FEATURES, BINARY_TARGET_COLUMN]].sort_values("Id"))


def _git(path: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        # egg-info is rewritten by every editable install, so it doesn't count as a code change.
        changes = run("status", "--porcelain", "--", ".", ":(exclude)*.egg-info")
        return dict(commit=run("rev-parse", "HEAD"), uncommitted_changes=bool(changes))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, uncommitted_changes=None)


def _version(package: str):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    return dict(
        started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        code=_git(REPO_DIR),
        slim_v2=_git(LIBRARY_DIR),
        python=platform.python_version(),
        packages={p: _version(p) for p in PACKAGES},
    )
